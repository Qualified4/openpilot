import time
import math
import copy
import itertools
import numpy as np
from collections import deque
from opendbc.car import CanBusBase, structs
from opendbc.car.crc import CRC16_XMODEM
from opendbc.car.hyundai.values import HyundaiFlags, HyundaiExtFlags
from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV
from openpilot.cereal import log

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = log.Desire

ACC_CONTROL_DT = 1.0 / 50.0

# ══════════════════════════════════════════════════════════════════════════════
# [차량 모델 선택 (ccNC DBC 규격)]
# 0: 순정(1), 1: 승용차(3), 2: 트럭(5), 3: 보행자(7), 4: 자전거(9), 5: 오토바이(11), 6: 라바콘(13)
# ══════════════════════════════════════════════════════════════════════════════
CAR_MODEL_TYPE = 1

_MODEL_ID_MAP = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}
CAR_MODEL_ID = _MODEL_ID_MAP.get(CAR_MODEL_TYPE, 1)


def _ccnc_valid_boundary(x, y):
  # Confidence can fluctuate while the boundary geometry remains useful for display.
  return (len(x) >= 2 and len(x) == len(y)
          and all(math.isfinite(v) for v in x)
          and all(math.isfinite(v) for v in y)
          and all(x[i - 1] < x[i] for i in range(1, len(x))))


class _CcncRadarDisplayTracker:
  """Observation continuity and lane-free fallback for the CCNC display only."""
  def __init__(self):
    self.live = None
    self.last_frame = -1
    self.tracks = {}
    self.selected = (None, None, None)
    self.positions = {}
    self.lane_ready = True

  def observe(self, live, frame):
    if frame < self.last_frame or frame - self.last_frame > 15:
      self.live = None
      self.lane_ready = True
    if frame < self.last_frame or frame - self.last_frame > 15 or live is None:
      self.tracks.clear()
      self.selected = (None, None, None)
      self.positions.clear()
    self.last_frame = frame
    if live is self.live:
      return
    self.live = live  # card keeps the same RadarData object until the next radar update.
    current = {}
    if live is not None:
      for p in live.points:
        if (str(p.radarSource) != "frontRadar" or p.dRel < 1
            or not all(math.isfinite(v) for v in (p.dRel, p.yRel, p.vRel, p.vLead))):
          continue
        start, count = frame, 1
        previous = self.tracks.get(p.trackId)
        if previous is not None:
          first, last, n, _, (old_d, old_y, old_v) = previous
          dt = (frame - last) * 0.01
          if (0 < frame - last <= 15
              and abs(p.dRel - old_d - old_v * dt) <= 1.0 + 2.0 * dt
              and abs(p.yRel - old_y) <= 0.5 + 5.0 * dt):
            start, count = first, n + 1
        current[p.trackId] = (start, frame, count, p, (p.dRel, p.yRel, p.vRel))
    self.tracks = current
    self.positions = {key: value for key, value in self.positions.items()
                      if key in current and value[0] == current[key][0]}

  def stable(self, track_id, frames=15):
    entry = self.tracks.get(track_id)
    return entry is not None and entry[2] >= 3 and entry[1] - entry[0] >= frames

  def lane_available(self, probability, frame):
    if probability < 0.1:
      self.lane_ready = False
    elif probability >= 0.3:
      # Reliable geometry decides the slot immediately; smooth only the position.
      self.lane_ready = True
    return self.lane_ready

  def path_lead(self, md, minimum_speed):
    if md is None:
      return None, 0.0
    xs = np.asarray(md.position.x, dtype=np.float64)
    ys = np.asarray(md.position.y, dtype=np.float64)
    if len(xs) != len(ys):
      return None, 0.0
    # A turning path can double back in x further ahead; use its forward-only prefix.
    stop = next((i for i in range(1, len(xs)) if xs[i] <= xs[i - 1]), len(xs))
    xs, ys = xs[:stop], ys[:stop]
    if not _ccnc_valid_boundary(xs, ys):
      return None, 0.0
    best, best_y, score = None, 0.0, math.inf
    for track_id, (_, _, _, p, _) in self.tracks.items():
      if not self.stable(track_id) or p.vLead * CV.MS_TO_KPH <= minimum_speed:
        continue
      # Do not extrapolate a short/invalid model path to distant radar reflections.
      if not max(1.0, xs[0]) <= p.dRel <= min(80.0, xs[-1]):
        continue
      aligned = p.yRel + float(np.interp(p.dRel, xs, ys) - ys[0])
      retained = track_id in self.selected
      # Vision is model-frame (right positive); radar is left positive.
      # The camera/radar longitudinal origin offset is 1.52m.
      vision_match = any(
        lead.prob >= (0.3 if retained else 0.5) and len(lead.x) and len(lead.y) and len(lead.v)
        and all(math.isfinite(v) for v in (lead.x[0], lead.y[0], lead.v[0]))
        and abs(p.dRel - (lead.x[0] - 1.52)) <= max(8.0 if retained else 6.0, p.dRel * 0.2)
        and abs(p.yRel + lead.y[0]) <= 1.5
        and abs(p.vLead - lead.v[0]) <= 5.0
        for lead in (md.leadsV3[i] for i in range(min(1, len(md.leadsV3)))))
      corridor = 1.8 if retained else 1.2
      if not ((abs(aligned) <= corridor and (retained or vision_match))
              or (retained and vision_match and abs(aligned) <= 4.5)):
        continue
      distance = p.dRel * p.dRel + p.yRel * p.yRel
      if distance < score:
        best, best_y, score = p, aligned, distance
    return best, best_y

  def filter_position(self, point, aligned_y, reference, frame):
    """Keep a physical track's filters across slots, before CAN sign/deadband mapping."""
    track_id = point.trackId
    observation = self.tracks.get(track_id)
    birth = observation[0] if observation is not None else frame
    previous = self.positions.get(track_id)
    if previous is None or previous[0] != birth or not 0 <= frame - previous[1] <= 15:
      distance = NoiseFilter(3, point.dRel, [0.3, 0.9], [1.0, 4.0])
      lateral = NoiseFilter(3, aligned_y, 0.3, 0.6)
      bias = 0.0
    else:
      _, last, old_reference, old_raw, old_input, bias, distance, lateral = previous
      if reference != old_reference:
        # Cancel the reference change, while retaining the radar's actual lateral motion.
        bias = old_input + (point.yRel - old_raw) - aligned_y
      step = (frame - last) * 0.03
      bias = math.copysign(max(0.0, abs(bias) - step), bias)
    filtered_input = aligned_y + bias
    self.positions[track_id] = (birth, frame, reference, point.yRel, filtered_input, bias, distance, lateral)
    return distance.apply(point.dRel), lateral.apply(filtered_input)

  def finish(self, ff, lf, rf, ff_y, lane_mode, frame):
    ids = tuple(p.trackId if p is not None else None for p in (ff, lf, rf))
    changed = tuple(a != b for a, b in zip(ids, self.selected))
    self.selected = ids
    return ff_y, changed


class _CcncRadarLaneSelector:
  """Keep a usable reference lane until the other is clearly better for 0.3s."""
  def __init__(self):
    self.is_left = None
    self.challenger_since = None
    self.last_frame = None

  def update(self, left_prob, right_prob, frame):
    # frame is the 100Hz CarController counter, not the model frameId.
    if self.last_frame is not None and (frame < self.last_frame or frame - self.last_frame > 100):
      self.is_left = None
      self.challenger_since = None
    self.last_frame = frame
    left_prob = left_prob if math.isfinite(left_prob) else 0.0
    right_prob = right_prob if math.isfinite(right_prob) else 0.0
    if max(left_prob, right_prob) < 0.1:
      self.is_left = None
      self.challenger_since = None
      return False
    preferred_left = left_prob > right_prob
    current_prob = left_prob if self.is_left else right_prob
    if self.is_left is None or current_prob < 0.1:
      self.is_left = preferred_left
      self.challenger_since = None
    elif preferred_left != self.is_left and abs(left_prob - right_prob) >= 0.1:
      if self.challenger_since is None:
        self.challenger_since = frame
      elif frame - self.challenger_since >= 30:
        self.is_left = preferred_left
        self.challenger_since = None
    else:
      self.challenger_since = None
    return self.is_left


def longitudinal_interlock_active(CS) -> bool:
  return CS.out.brakeHoldActive or CS.out.parkingBrake


def apply_accel_jerk_limit(a_raw: float, a_value_last: float, jerk_u: float, jerk_l: float,
                           dt: float = ACC_CONTROL_DT) -> float:
  """Ramp aReqValue toward aReqRaw using the asymmetric stock SCC jerk limits."""
  upper_step = max(0.0, float(jerk_u)) * dt
  lower_step = max(0.0, float(jerk_l)) * dt
  return float(np.clip(a_raw, a_value_last - lower_step, a_value_last + upper_step))

class LaneHighlightStateMachine:
  def __init__(self):
    self.state = 0  # 현재 하이라이트 상태

  def update(self, accel_kph, drive_mode, v_ego_kph):
    # 상태별 전이 로직
    if self.state == 4:
      if accel_kph < -1.0 and v_ego_kph > 1.0:
        return self.state # 상태 유지
    elif self.state == 5:
      if drive_mode != 4 and accel_kph > 5.0:
        return self.state # 상태 유지

    # 진입 로직 (우선순위 순서)
    if accel_kph < -10.0:
      self.state = 4  # 급제동 (최우선)
    elif drive_mode == 4 or accel_kph > 10.0:
      self.state = 5  # 급가속/고속
    elif drive_mode < 3:
      self.state = 3  # 연비/안전
    else:
      self.state = 0  # 기본 상태 (회색 혹은 꺼짐)

    return self.state

class ThresholdTracker:
  def __init__(self, bounds, states):
    """
    :param bounds: (상한선, 하한선) 튜플
    :param states: (상한 이탈 시 상태, 하한 이탈 시 상태) 튜플
    :param initial_state: 객체 생성 시점의 초기 상태
    """
    self._upper_bound, self._lower_bound = bounds
    self._state_high, self._state_low = states
    self._current_state = self._state_high

  def apply(self, value):
    """
    입력값에 따라 상태를 업데이트하고 반환
    """
    if value > self._upper_bound:
        self._current_state = self._state_high
    elif value < self._lower_bound:
        self._current_state = self._state_low

    # 두 기준값 사이(박스권)에 있을 때는 기존 상태를 유지
    return self._current_state

class NoiseFilter:
  """
  고정/스텝/가변 알파를 지원하는 Median + LPF 통합 필터.
  - alpha_range: 필수 입력 (None 허용 안 함).
  - error_range: 선택 입력 (None일 경우 고정 알파 모드).
  """
  def __init__(self, median_buffer_size, lowpass_default, alpha_range, error_range=None):
    self._default_value = lowpass_default
    self._filtered_value = lowpass_default
    self._buffer = deque([lowpass_default] * median_buffer_size, maxlen=median_buffer_size)

    def normalize_range(r, default_val):
      # 리스트/튜플에서 첫 번째와 마지막 값 추출 (1개일 때도 r[0]==r[-1]로 안전)
      if isinstance(r, (int, float)): return [float(r), float(r)]
      if isinstance(r, (list, tuple)) and len(r) >= 1:
        return [float(r[0]), float(r[-1])]
      return [default_val, default_val]

    # 1. 알파 설정 (필수 입력값 정문화)
    norm_alpha = normalize_range(alpha_range, 1.0)
    self._a_min, self._a_max = [np.clip(v, 0.001, 1.0) for v in norm_alpha]

    # 2. 에러 범위 및 전략 할당
    if error_range is not None:
      self._err_min, self._err_max = normalize_range(error_range, 0.0)

      # 전략 선택: 경계값이 같으면 Step(리셋), 다르면 Adaptive(보간)
      if self._err_min == self._err_max:
        self.apply = self._apply_step
      else:
        self.apply = self._apply_adaptive
    else:
      # error_range가 None이면 리셋 로직 없이 고정 알파 필터로 동작
      self._alpha = self._a_min
      self.apply = self._apply_fixed

  def reset(self, new_value=None):
    """값을 초기화하고 버퍼를 완전히 비웁니다."""
    self._filtered_value = new_value if new_value is not None else self._default_value
    self._buffer.clear()
    return self._filtered_value

  def fill(self, value):
    """
    현재 필터의 버퍼를 특정 값으로 가득 채우고 필터 출력값도 동기화합니다.
    reset과 달리 내부 설정값(default_value 등)은 유지하며 데이터 흐름만 강제 수정합니다.
    """
    self._filtered_value = value
    self._buffer.extend([self._filtered_value] * self._buffer.maxlen)

    return self._filtered_value

  def update_alpha(self, new_alpha):
    """
    고정 알파 모드에서 알파 값을 업데이트합니다.
    """
    if self.apply == self._apply_fixed:
      self._alpha = np.clip(new_alpha, 0.001, 1.0)

  def reset_alpha(self):
    """
    고정 알파 모드에서 알파 값을 초기값으로 되돌립니다.
    """
    if self.apply == self._apply_fixed:
      self._alpha = self._a_min

  def _get_median(self):
    buf_len = len(self._buffer)
    if buf_len == 0:
        return self._filtered_value # 버퍼가 비어있으면 현재 필터값 반환
    # 꽉 차지 않은 상태(초기 진입 시)에도 현재 데이터 개수 기준으로 중간값 산출
    return sorted(self._buffer)[buf_len // 2]

  def _check_hard_reset(self, target):
    step_err = abs(target - self._filtered_value)
    if step_err > self._err_max:
      self.reset(target)  # 버퍼 비우고 target으로 즉시 리셋
      return True

    return False

  def _apply_fixed(self, target):
    self._buffer.append(target)
    med = self._get_median()
    self._filtered_value = (self._alpha * med) + ((1.0 - self._alpha) * self._filtered_value)
    return self._filtered_value

  def _apply_step(self, target):
    if self._check_hard_reset(target):
      return self._filtered_value

    self._buffer.append(target)
    med = self._get_median()

    self._filtered_value = (self._a_min * med) + ((1.0 - self._a_min) * self._filtered_value)
    return self._filtered_value

  def _apply_adaptive(self, target):
    if self._check_hard_reset(target):
      return self._filtered_value

    self._buffer.append(target)
    med = self._get_median()

    track_err = abs(med - self._filtered_value)

    # 오차 정도에 따라 알파값 보간
    alpha = float(np.interp(track_err, [self._err_min, self._err_max], [self._a_min, self._a_max]))
    self._filtered_value = (alpha * med) + ((1.0 - alpha) * self._filtered_value)

    return self._filtered_value

  @property
  def value(self):
    return self._filtered_value

def ease_in_interp(x, x_range, y_range, power=2):
  # x를 0~1 사이 비율로 변환
  t = (x - x_range[0]) / (x_range[1] - x_range[0])
  t = max(0, min(1, t)) # 범위 제한

  # Ease-in 적용
  eased_t = t ** power

  # 결과값 매핑
  return y_range[0] + (y_range[1] - y_range[0]) * eased_t

def apply_curved_deadband(value, center, radius, degree=2):
    # 1. 입력의 부호(+1 또는 -1) 추출
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)

    # 2. 중심(center)으로부터의 거리 계산
    diff = abs_val - center

    # 3. 데드밴드 영향 범위를 벗어나면 원본 값 그대로 반환
    # (center 기준 radius 이상 멀어지거나, center 안쪽으로 radius 이상 들어간 경우)
    if diff >= radius or diff <= -radius:
        return value

    # 4. 곡선 보간 비율 t 계산 (0 ~ 1 범위로 정규화)
    # diff = 0 (center 위치)일 때 t = 0 (완전 감쇠)
    # diff = radius 또는 -radius 일 때 t = 1 (원본 유지)
    t = abs(diff) / radius

    # 5. 곡선 가중치 적용
    weight = t ** degree

    # 6. center 위치로 당겨주는 보간 수행 후, 원본 부호 복원
    smoothed_abs = (1.0 - weight) * center + (weight * abs_val)

    return sign * smoothed_abs

def hyundai_crc8(data: bytes) -> int:
  poly = 0x2F
  crc = 0xFF

  for byte in data:
    crc ^= byte
    for _ in range(8):
      if crc & 0x80:
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc = (crc << 1) & 0xFF

  return crc ^ 0xFF

class CanBus(CanBusBase):
  def __init__(self, CP, fingerprint=None, lka_steering=None) -> None:
    super().__init__(CP, fingerprint)

    if lka_steering is None:
      lka_steering = CP.flags & HyundaiFlags.CANFD_HDA2.value if CP is not None else False

    # On the CAN-FD platforms, the LKAS camera is on both A-CAN and E-CAN. LKA steering cars
    # have a different harness than the LFA steering variants in order to split
    # a different bus, since the steering is done by different ECUs.
    self._a, self._e = 1, 0
    if lka_steering and Params().get_int("HyundaiCameraSCC") == 0:  #배선개조는 무조건 Bus0가 ECAN임.
      self._a, self._e = 0, 1

    self._a += self.offset
    self._e += self.offset
    self._cam = 2 + self.offset

  @property
  def ECAN(self):
    return self._e

  @property
  def ACAN(self):
    return self._a

  @property
  def CAM(self):
    return self._cam

# CAN LIST (CAM)  - 롱컨개조시... ADAS + CAM
# 160: ADRV_0x160
# 1da: ADRV_0x1da
# 1ea: ADRV_0x1ea
# 200: ADRV_0x200
# 345: ADRV_0x345
# 1fa: CLUSTER_SPEED_LIMIT
# 12a: LFA
# 1e0: LFAHDA_CLUSTER
# 11a:
# 1b5:
# 1a0: SCC_CONTROL

# CAN LIST (ACAN)
# 160: ADRV_0x160
# 51: ADRV_0x51
# 180: CAM_0x180
# ...
# 185: CAM_0x185
# 1b6: CAM_0x1b6
# ...
# 1b9: CAM_0x1b9
# 1fb: CAM_0x1fb
# 2a2 - 2a4
# 2bb - 2be
# LKAS
# 201 - 2a0



def create_steering_messages_camera_scc(frame, packer, CP, CAN, CC, lat_active, apply_steer, CS, apply_angle, max_torque, angle_control):

  emergency_steering = False
  if CS.adrv_0x161 is not None:
    values = CS.adrv_0x161
    emergency_steering = values["ALERTS_1"] in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26]


  ret = []
  if CS.mdps is not None:
    values = CS.mdps.copy()
    #rx_counter = values.pop("COUNTER", None)
    if angle_control:
      if CS.lfa_alt is not None:
        values["LFA2_ACTIVE"] = CS.lfa_alt["LKAS_ANGLE_ACTIVE"]
    else:
      if CS.lfa is not None:
        values["LKA_ACTIVE"] = 1 if CS.lfa["STEER_REQ"] == 1 else 0

    if frame % 1000 < 40:
      values["STEERING_COL_TORQUE"] += 220
    #ret.append(packer.make_can_msg("MDPS", CAN.CAM, values, rx_counter = rx_counter))
    ret.append(packer.make_can_msg("MDPS", CAN.CAM, values))

  if frame % 10 == 0:
    if CS.steer_touch_2af is not None:
      values = CS.steer_touch_2af.copy()
      if frame % 1000 < 40:
        values["TOUCH_DETECT"] = 3
        values["TOUCH1"] = 50
        values["TOUCH2"] = 50
        values["CHECKSUM_"] = 0
        dat = packer.make_can_msg("STEER_TOUCH_2AF", 0, values)[1]
        values["CHECKSUM_"] = hyundai_crc8(dat[1:8])

      ret.append(packer.make_can_msg("STEER_TOUCH_2AF", CAN.CAM, values))

  if angle_control:
    if CS.lfa_alt is not None:
      values = CS.lfa_alt.copy()
      rx_counter = values.pop("COUNTER", None)
      if emergency_steering:
        pass
      else:
        #values = {} #CS.lfa_alt
        values["LKAS_ANGLE_ACTIVE"] = 2 if CC.latActive else 1
        values["LKAS_ANGLE_CMD"] = -apply_angle
        values["LKAS_ANGLE_MAX_TORQUE"] = max_torque if CC.latActive else 0
      ret.append(packer.make_can_msg("LFA_ALT", CAN.ECAN, values, rx_counter = rx_counter))

    if CS.lfa is not None:
      values = CS.lfa.copy()
      rx_counter = values.pop("COUNTER", None)
      if not emergency_steering:
        values["LKA_MODE"] = 0
        values["LKA_ICON"] = 2 if CC.latActive else 1
        values["TORQUE_REQUEST"] = -1024  # apply_steer,
        values["VALUE63"] = 0 # LKA_ASSIST
        values["STEER_REQ"] = 0  # 1 if lat_active else 0,
        values["HAS_LANE_SAFETY"] = 0  # hide LKAS settings
        values["LKA_ACTIVE"] = 3 if CC.latActive else 0  # this changes sometimes, 3 seems to indicate engaged
        values["VALUE64"] = 0  #STEER_MODE, NEW_SIGNAL_2
        values["LKAS_ANGLE_CMD"] = -25.6 #-apply_angle,
        values["LKAS_ANGLE_ACTIVE"] = 0 #2 if lat_active else 1,
        values["LKAS_ANGLE_MAX_TORQUE"] = 0 #max_torque if lat_active else 0,
        values["NEW_SIGNAL_1"] = 10
      ret.append(packer.make_can_msg("LFA", CAN.ECAN, values, rx_counter = rx_counter))

  elif CS.lfa is not None:
    values = {}
    values["LKA_MODE"] = 2
    values["LKA_ICON"] = 2 if lat_active else 1
    values["TORQUE_REQUEST"] = apply_steer
    values["STEER_REQ"] = 1 if lat_active else 0
    values["VALUE64"] = 0  # STEER_MODE, NEW_SIGNAL_2
    values["HAS_LANE_SAFETY"] = 0
    values["LKA_ACTIVE"] = 0 # NEW_SIGNAL_1

    values["DampingGain"] = 0 if lat_active else 100
    #values["VALUE63"] = 0

    #values["VALUE82_SET256"] = 0

    ret.append(packer.make_can_msg("LFA", CAN.ECAN, values))

  return ret

def create_steering_messages(packer, CP, CAN, enabled, lat_active, apply_steer, apply_angle, max_torque, angle_control):

  ret = []
  if angle_control:
    values = {
      "LKA_MODE": 0,
      "LKA_ICON": 2 if enabled else 1,
      "TORQUE_REQUEST": 0,  # apply_steer,
      "VALUE63": 0, # LKA_ASSIST
      "STEER_REQ": 0,  # 1 if lat_active else 0,
      "HAS_LANE_SAFETY": 0,  # hide LKAS settings
      "LKA_ACTIVE": 3 if lat_active else 0,  # this changes sometimes, 3 seems to indicate engaged
      "VALUE64": 0,  #STEER_MODE, NEW_SIGNAL_2
      "LKAS_ANGLE_CMD": -apply_angle,
      "LKAS_ANGLE_ACTIVE": 2 if lat_active else 1,
      "LKAS_ANGLE_MAX_TORQUE": max_torque if lat_active else 0,

      # test for EV6PE
      "NEW_SIGNAL_1": 10, #2,
      "DampingGain": 9,
      "VALUE231": 146,
      "VALUE239": 1,
      "VALUE247": 255,
      "VALUE255": 255,
    }
  else:
    values = {
      "LKA_MODE": 2,
      "LKA_ICON": 2 if enabled else 1,
      "TORQUE_REQUEST": apply_steer,
      "DampingGain": 100, #3 if enabled else 100,
      "STEER_REQ": 1 if lat_active else 0,
      #"STEER_MODE": 0,
      "HAS_LANE_SAFETY": 0,  # hide LKAS settings
      "VALUE63": 0,
      "VALUE64": 100,
    }

  if CP.flags & HyundaiFlags.CANFD_HDA2:
    lkas_msg = "LKAS_ALT" if CP.flags & HyundaiFlags.CANFD_HDA2_ALT_STEERING else "LKAS"
    if CP.openpilotLongitudinalControl:
      ret.append(packer.make_can_msg("LFA", CAN.ECAN, values))
    if not (CP.flags & HyundaiFlags.CAMERA_SCC.value):
      ret.append(packer.make_can_msg(lkas_msg, CAN.ACAN, values))
  else:
    ret.append(packer.make_can_msg("LFA", CAN.ECAN, values))

  return ret

def create_suppress_lfa(packer, CAN, CS):
  if CS.cam_0x362 is not None:
    suppress_msg = "CAM_0x362"
    lfa_block_msg = CS.cam_0x362
  elif CS.cam_0x2a4 is not None:
    suppress_msg = "CAM_0x2a4"
    lfa_block_msg = CS.cam_0x2a4
  else:
    return []

  #values = {f"BYTE{i}": lfa_block_msg[f"BYTE{i}"] for i in range(3, msg_bytes) if i != 7}
  values = copy.copy(lfa_block_msg)
  values["COUNTER"] = lfa_block_msg["COUNTER"]
  values["SET_ME_0"] = 0
  values["SET_ME_0_2"] = 0
  values["LEFT_LANE_LINE"] = 0
  values["RIGHT_LANE_LINE"] = 0
  return [packer.make_can_msg(suppress_msg, CAN.ACAN, values)]

def create_buttons(packer, CP, CAN, cnt, btn):
  values = {
    "COUNTER": cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }

  #bus = CAN.ECAN if CP.flags & HyundaiFlags.CANFD_HDA2 else CAN.CAM
  bus = CAN.ECAN
  return packer.make_can_msg("CRUISE_BUTTONS", bus, values)

def create_acc_cancel(packer, CP, CAN, cruise_info_copy):
  # TODO: why do we copy different values here?
  if CP.flags & HyundaiFlags.CANFD_CAMERA_SCC.value:
    values = {s: cruise_info_copy[s] for s in [
      "COUNTER",
      "CHECKSUM",
      "NEW_SIGNAL_1",
      "MainMode_ACC",
      "ACCMode",
      "ZEROS_9",
      "CRUISE_STANDSTILL",
      "ZEROS_5",
      "DISTANCE_SETTING",
      "VSetDis",
    ]}
  else:
    values = {s: cruise_info_copy[s] for s in [
      "COUNTER",
      "CHECKSUM",
      "ACCMode",
      "VSetDis",
      "CRUISE_STANDSTILL",
    ]}
  values.update({
    "ACCMode": 4,
    "aReqRaw": 0.0,
    "aReqValue": 0.0,
  })
  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)

def create_lfahda_cluster(packer, CS, CAN, long_active, lat_active):


  if CS.lfahda_cluster is not None:
    values = CS.lfahda_cluster.copy()
    rx_counter = values.pop("COUNTER", None)
  else:
    return []
    values = {}
    rx_counter = None
    values["LFA_OptUsmSta"] = 2
    values["HDA_OptUsmSta"] = 2
  values["HDA_CntrlModSta"] = 2 if long_active else 0
  values["HDA_LFA_SymSta"] = 2 if lat_active else 0
  return [packer.make_can_msg("LFAHDA_CLUSTER", CAN.ECAN, values, rx_counter=rx_counter)]

def create_lfa_icon_non_camera_scc(packer, CS, CAN, CC):
  ret = []
  if CS.adrv_0x161 is not None:
    values = CS.adrv_0x161.copy()
    rx_counter = values.pop("COUNTER", None)

    lat_active = CC.latActive
    lat_enabled = CS.out.latEnabled

    values["LFA_ICON"] = 2 if lat_active else 1 if lat_enabled else 0
    values["LKA_ICON"] = 4 if lat_active else 3 if lat_enabled else 0

    if values["ALERTS_2"] in [1, 2, 5, 6, 10, 21, 22]:
      values["ALERTS_2"] = 0
      values["DAW_ICON"] = 0

    if values["ALERTS_1"] == 0:
      values["SOUNDS_1"] = 0
      values["SOUNDS_2"] = 0
      values["SOUNDS_4"] = 0

    if values["ALERTS_3"] in [3, 4, 11, 12, 13, 14, 17, 19, 26, 7, 8, 9, 10]:
      values["ALERTS_3"] = 0
      values["SOUNDS_3"] = 0

    if values["ALERTS_5"] in [1, 2, 3, 4, 5]:
      values["ALERTS_5"] = 0

    ret.append(packer.make_can_msg("ADRV_0x161", CAN.ECAN, values, rx_counter=rx_counter))
  return ret

def create_acc_control_scc2(packer, CAN, enabled, accel_value_last, accel, stopping, gas_override, set_speed, hud_control, hyundai_jerk, CS):

  if CS.scc_control is None:
    return None, accel_value_last
  interlock_active = longitudinal_interlock_active(CS)
  soft_hold_active = CS.softHoldActive > 0 and CS.out.cruiseState.available
  acc_control_enabled = (enabled or soft_hold_active) and CS.out.cruiseState.available and CS.paddle_button_prev == 0 and not interlock_active
  enabled = acc_control_enabled

  acc_mode = 0 if not enabled else (2 if gas_override else 1)

  if hyundai_jerk.carrot_cruise == 1:
    acc_mode = 4 if enabled else 0
    enabled = False
    accel = accel_value_last = 0.5

  elif hyundai_jerk.carrot_cruise == 2:
    accel = accel_value_last = hyundai_jerk.carrot_cruise_accel

  jerk_u = 2.0 if stopping or soft_hold_active else hyundai_jerk.jerk_u
  jerk_l = hyundai_jerk.jerk_l
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel
    a_val = apply_accel_jerk_limit(a_raw, accel_value_last, jerk_u, jerk_l)

  values = copy.copy(CS.scc_control)
  rx_counter = values.pop("COUNTER", None)
  values["ACCMode"] = acc_mode
  values["MainMode_ACC"] = 1
  values["StopReq"] = 1 if acc_control_enabled and (stopping or soft_hold_active) else 0  # 1: Stop control is required, 2: Not used, 3: Error Indicator
  values["aReqValue"] = a_val
  values["aReqRaw"] = a_raw
  values["VSetDis"] = set_speed
  #values["JerkLowerLimit"] = jerk if enabled else 1
  #values["JerkUpperLimit"] = 3.0
  values["JerkLowerLimit"] = jerk_l if enabled else 1
  values["JerkUpperLimit"] = jerk_u
  values["DISTANCE_SETTING"] = hud_control.leadDistanceBars # + 5
  #values["DISTANCE_SETTING"] = hud_control.leadDistanceBars  + 5

  #values["ACC_ObjDist"] = 1
  #values["ObjValid"] = 0
  #values["OBJ_STATUS"] =  2
  #values["NSCCOper"] = 1 if enabled else 0 # 0: off, 1: Ready, 2: Act, 3: Error Indicator
  #values["NSCCOnOff"] = 2  # 0: Default, 1: Off, 2: On, 3: Invalid
  #values["SET_ME_3"] = 0x3  # objRelsped와 충돌
  #values["ACC_ObjLatPos"] = - hud_control.leadDPath
  values["DriveMode"] = 0 # 0: Default, 1: Comfort Mode, 2:Normal mode, 3:Dynamic mode, reserved

  # Preserve the received HUD_LEAD_INFO and TARGET_DISTANCE for cluster flicker testing.

  values["DriverAlert"] = 0   # 1: SCC Disengaged, 2: No SCC Engage condition, 3: SCC Disenganed when the vehicle stops


  soft_hold_info = 1 if soft_hold_active and CS.softHoldActive > 1 and enabled else 0

  # 이거안하면 정지중 뒤로 밀리는 현상 발생하는듯.. (신호정지중에 뒤로 밀리는 경험함.. 시험해봐야)
  if values["InfoDisplay"] != 5: #5: Front Car Departure Notice
    values["InfoDisplay"] = 4 if not interlock_active and stopping and CS.out.aEgo > -0.3 else 0  # 1: SCC Mode, 2: Convention Cruise Mode, 3: Object disappered at low speed, 4: Available to resume acceleration control, 5: Front vehicle departure notice, 6: Reserved, 7: Invalid

  values["TakeOverReq"] = 0    # 1: Takeover request, 2: Not used, 3: Error indicator , 이것이 켜지면 가속을 안하는듯함.
  #values["NEW_SIGNAL_4"] = 9 if hud_control.leadVisible else 0
  # AccelLimitBandUpper, Lower
  values["SysFailState"] = 0    # 1: Performance degredation, 2: system temporairy unavailble, 3: SCC Service required , 눈이 묻어 레이더오류시... 2가 됨. 이때 가속을 안함...

  values["AccelLimitBandUpper"] = 0.0   # 이값이 1.26일때 가속을 안하는 증상이 보임..
  values["AccelLimitBandLower"] = 0.0

  values["ZEROS_7"] = 1

  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values), a_val

def create_acc_control(packer, CAN, enabled, accel_last, accel, stopping, gas_override, set_speed, hud_control, jerk_u, jerk_l, CS):

  interlock_active = longitudinal_interlock_active(CS)
  soft_hold_active = CS.softHoldActive > 0 and CS.out.cruiseState.available
  acc_control_enabled = (enabled or soft_hold_active) and CS.out.cruiseState.available and not interlock_active
  enabled = acc_control_enabled
  jerk = 5
  jn = jerk / 50
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel
    a_val = np.clip(accel, accel_last - jn, accel_last + jn)

  values = {
    "ACCMode": 0 if not enabled else (2 if gas_override else 1),
    "MainMode_ACC": 1,
    "StopReq": 1 if acc_control_enabled and (stopping or soft_hold_active) else 0,
    "aReqValue": a_val,
    "aReqRaw": a_raw,
    "VSetDis": set_speed,
    #"JerkLowerLimit": jerk if enabled else 1,
    #"JerkUpperLimit": 3.0,
    "JerkLowerLimit": jerk_l if enabled else 1,
    "JerkUpperLimit": jerk_u,

    "ACC_ObjDist": 1,
    #"ObjValid": 0,
    #"OBJ_STATUS": 2,
    "NSCCOper": 0,
    "NSCCOnOff": 2,
    "DriveMode": 0,
    #"SET_ME_3": 0x3,
    "ACC_ObjLatPos": 0x64,
    "DISTANCE_SETTING": hud_control.leadDistanceBars, # + 5,
    "InfoDisplay": 4 if not interlock_active and stopping and CS.out.cruiseState.standstill else 0,
  }

  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)


def create_spas_messages(packer, CAN, frame, left_blink, right_blink):
  ret = []

  values = {
  }
  ret.append(packer.make_can_msg("SPAS1", CAN.ECAN, values))

  blink = 0
  if left_blink:
    blink = 3
  elif right_blink:
    blink = 4
  values = {
    "BLINKER_CONTROL": blink,
  }
  ret.append(packer.make_can_msg("SPAS2", CAN.ECAN, values))

  return ret


def create_fca_warning_light(CP, packer, CAN, frame):
  ret = []
  if CP.flags & HyundaiFlags.CAMERA_SCC.value:
    return ret

  if frame % 2 == 0:
    values = {
      'AEB_SETTING': 0x1,  # show AEB disabled icon
      'SET_ME_2': 0x2,
      'SET_ME_FF': 0xff,
      'SET_ME_FC': 0xfc,
      'SET_ME_9': 0x9,
      #'DATA102': 1,
    }
    ret.append(packer.make_can_msg("ADRV_0x160", CAN.ECAN, values))
  return ret

def create_tcs_messages(packer, CAN, CS):
  ret = []
  if CS.tcs is not None:
    values = copy.copy(CS.tcs)
    #rx_counter = values.pop("COUNTER", None)
    values["DriverBraking"] = 0
    values["NEW_SIGNAL_20"] = 0
    values["NEW_SIGNAL_11"] = 0
    values["DriverBrakingLowSens"] = 0
    #values["NEW_SIGNAL_1"] = 0 # accel과 관련..  옆두부 꺼지는것과 관련? 확인필요
    #values["ACC_REQ"] = 1 # 옆두부 꺼지는것과 관련? 확인필요.. 항상 켜지게함..
    values["NEW_SIGNAL_1"] = 0 if values["ACC_REQ"] == 1 else 1 # 옆두부..
    #ret.append(packer.make_can_msg("TCS", CAN.CAM, values, rx_counter = rx_counter))
    ret.append(packer.make_can_msg("TCS", CAN.CAM, values))
  return ret

def forward_button_message(packer, CAN, frame, CS, cruise_button, MainMode_ACC_trigger, LFA_trigger):
  ret = []
  if frame % 2 == 0:
    if CS.cruise_buttons_msg is not None:
      values = copy.copy(CS.cruise_buttons_msg)
      # A held MAIN is reported on this bit and switches some clusters to LIMIT mode.
      values["NORMAL_CRUISE_MAIN_BTN"] = 0
      #rx_counter = values.pop("COUNTER", None)
      cruise_button_driver = values["CRUISE_BUTTONS"]
      if cruise_button_driver == 0:
        values["CRUISE_BUTTONS"] = cruise_button
      if MainMode_ACC_trigger > 0:
        #values["ADAPTIVE_CRUISE_MAIN_BTN"] = 1
        pass
      elif LFA_trigger > 0:
        values["LFA_BTN"] = 1

      #ret.append(packer.make_can_msg(CS.cruise_btns_msg_canfd, CAN.CAM, values, rx_counter = rx_counter))
      ret.append(packer.make_can_msg(CS.cruise_btns_msg_canfd, CAN.CAM, values))
  return ret

def create_adrv_messages(CP, packer, CAN, frame):
  # messages needed to car happy after disabling
  # the ADAS Driving ECU to do longitudinal control

  ret = []

  if not CP.flags & HyundaiFlags.CAMERA_SCC.value:
    values = {}

    ret.extend(create_fca_warning_light(CP, packer, CAN, frame))
    if frame % 5 == 0:
      values = {
        #'HDA_MODE1': 0x8,
        'HDA_MODE2': 0x1,
        #'SET_ME_1C': 0x1c,
        'SET_ME_FF': 0xff,
        #'SET_ME_TMP_F': 0xf,
        #'SET_ME_TMP_F_2': 0xf,
        #'DATA26': 1,  #1
        #'DATA32': 5,  #5
      }
      ret.append(packer.make_can_msg("ADRV_0x1ea", CAN.ECAN, values))

      values = {
        'SET_ME_E1': 0xe1,
        #'SET_ME_3A': 0x3a,
        'TauGapSet' : 1,
        'NEW_SIGNAL_2': 3,
      }
      ret.append(packer.make_can_msg("ADRV_0x200", CAN.ECAN, values))

    if frame % 20 == 0:
      values = {
        'SET_ME_15': 0x15,
      }
      ret.append(packer.make_can_msg("ADRV_0x345", CAN.ECAN, values))

    if frame % 100 == 0:
      values = {
        'SET_ME_22': 0x22,
        'SET_ME_41': 0x41,
      }
      ret.append(packer.make_can_msg("ADRV_0x1da", CAN.ECAN, values))

  return ret

## carrot
def alt_cruise_buttons(packer, CP, CAN, buttons, cruise_btns_msg, cnt):
  cruise_btns_msg["CRUISE_BUTTONS"] = buttons
  cruise_btns_msg["COUNTER"] = (cruise_btns_msg["COUNTER"] + 1 + cnt) % 256
  bus = CAN.ECAN if CP.flags & HyundaiFlags.CANFD_HDA2 else CAN.CAM
  return packer.make_can_msg("CRUISE_BUTTONS_ALT", bus, cruise_btns_msg)

def hkg_can_fd_checksum(address: int, sig, d: bytearray) -> int:
  crc = 0
  for i in range(2, len(d)):
    crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ d[i]]) & 0xFFFF
  crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ ((address >> 0) & 0xFF)]) & 0xFFFF
  crc = ((crc << 8) ^ CRC16_XMODEM[(crc >> 8) ^ ((address >> 8) & 0xFF)]) & 0xFFFF
  if len(d) == 8:
    crc ^= 0x5F29
  elif len(d) == 16:
    crc ^= 0x041D
  elif len(d) == 24:
    crc ^= 0x819D
  elif len(d) == 32:
    crc ^= 0x9F5B
  return crc




def _clip_int(x, lo, hi):
  return lo if x < lo else hi if x > hi else int(x)

def _get_desire_and_lane_changing(md):
  desire = 0
  lane_changing = 0
  if md is not None:
    desire = md.meta.desire.raw
    ds = md.meta.desireState
    if len(ds) > 4:
      if ds[1] > 0.9: lane_changing = 1
      if ds[2] > 0.9: lane_changing = 2
      if ds[3] > 0.9: lane_changing = 3
      if ds[4] > 0.9: lane_changing = 4
  return desire, lane_changing

def _apply_lane_desire(values, desire):
  #values['LANE_CHANGING'] = 0

  if desire == 1:  # 좌회전
    values['LANE_CHANGING'] = 1
    values["LANELINE_CURVATURE"] = 15
    values["LANELINE_CURVATURE_DIRECTION"] = 0

  elif desire == 2:  # 우회전
    values['LANE_CHANGING'] = 2
    values["LANELINE_CURVATURE"] = 15
    values["LANELINE_CURVATURE_DIRECTION"] = 1

  elif desire == 3:  # 좌차선변경
    values['LANE_CHANGING'] = 3

  elif desire == 4:  # 우차선변경
    values['LANE_CHANGING'] = 4

def _apply_radar_blink(values, radar_pairs, frame, *,
                      disp_dist=30.0, min_dist=14.0,
                      max_interval=100, t=1.0):
  """
  거리 > min_dist 일 때만 깜빡임.
  거리 멀수록 interval 커짐(느리게).
  """
  for det_key, dist_key in radar_pairs:
    dist = values[dist_key]
    if dist <= min_dist:
      continue

    d = min(dist, disp_dist)
    interval = int((1 + (max_interval - 1) * (d / disp_dist)) * t)
    interval = _clip_int(interval, 1, max_interval)

    blink = (frame // interval) & 1
    if CAR_MODEL_ID > 1:
      values[det_key] = (CAR_MODEL_ID + 1) if blink else CAR_MODEL_ID
    else:
      values[det_key] = 2 - blink
    values[dist_key] = min_dist

def _suppress_trailer_mode_warning(values, CS):
  # Logs from IONIQ 9 show ALERTS_5=6 is the periodic
  # "driver assistance limited in trailer mode" popup.
  if CS.trailer_connected and values.get("ALERTS_5") == 6:
    values["ALERTS_5"] = 0


def _hide_replaced_adas_service_warning(values):
  # Openpilot replaces the stock lane-change/highway-driving control path, so
  # the camera can latch their service-required flags during low-speed turns.
  # Preserve blocked-sensor warnings and unrelated DAS faults. The original
  # camera-side message remains available in logcan for diagnosis.
  service_warning_hidden = False
  for fault in ("FAULT_LCA", "FAULT_HDA"):
    if values.get(fault) == 1:
      values[fault] = 0
      service_warning_hidden = True

  if service_warning_hidden and values.get("FAULT_DAS") == 1:
    values["FAULT_DAS"] = 0


def _select_cluster_background(cruise_enabled, lat_active, paddle_pressed, paddle_mode):
  if paddle_mode > 0 and paddle_pressed:
    return 6
  return 1 if cruise_enabled else 3 if lat_active else 7


def _make_ccnc_values(values, CS, lat_active, frame, hud_control,
                     lane_line=True, corner_radar=True,
                     desire=0,
                     blink_pairs=None,
                     blink_t=1.0):
  if lane_line:
    curvature = round(CS.out.steeringAngleDeg / 3)
    mag = min(abs(curvature), 15)
    curv = mag + (-1 if curvature < 0 else 0)
    direction = 1 if curvature < 0 else 0
    values["LANELINE_CURVATURE"] = curv if lat_active else 0
    values["LANELINE_CURVATURE_DIRECTION"] = direction if lat_active else 0
    if desire:
      _apply_lane_desire(values, desire)

  if corner_radar:
    radar_all = [
      ('LF_DETECT', 'LF_DETECT_DISTANCE'),
      ('RF_DETECT', 'RF_DETECT_DISTANCE'),
      ('LR_DETECT', 'LR_DETECT_DISTANCE'),
      ('RR_DETECT', 'RR_DETECT_DISTANCE'),
    ]
    for det_key, dist_key in radar_all:
      if values[det_key] > 0 and values[dist_key] != 0:
        values[det_key] = CAR_MODEL_ID

    if blink_pairs:
      _apply_radar_blink(values, blink_pairs, frame, t=blink_t)

def create_ccnc_messages(CP, packer, CAN, frame, CC, CS, hud_control,
                         disp_angle, left_lane_warning, right_lane_warning,
                         enable_corner_radar, stopping, canfd_debug, paddle_mode):
  ret = []
  interlock_active = longitudinal_interlock_active(CS)

  if not hasattr(create_ccnc_messages, '_lane_line_check') or frame % 100 == 0:
    create_ccnc_messages._lane_line_check = Params().get_int("LaneLineCheck")
  lane_line_check = create_ccnc_messages._lane_line_check

  if CP.flags & HyundaiFlags.CAMERA_SCC.value:
    md = CS.modelV2
    v_ego_kph = CS.out.vEgo * CV.MS_TO_KPH
    a_ego_kph = CS.out.aEgo * CV.MS_TO_KPH
    desire, lane_changing = _get_desire_and_lane_changing(md)
    HDA_CntrlModSta = 0
    HDA_LFA_SymSta = 0
    if CS.lfahda_cluster is not None:
      HDA_CntrlModSta = CS.lfahda_cluster["HDA_CntrlModSta"]
      HDA_LFA_SymSta = CS.lfahda_cluster["HDA_LFA_SymSta"]

    if frame % 2 == 0:
      #if CS.adrv_0x160 is not None:
      #  values = copy.copy(CS.adrv_0x160)
      #  ret.append(packer.make_can_msg("ADRV_0x160", CAN.ECAN, values))

      if CS.cruise_buttons_msg is not None:
        values = copy.copy(CS.cruise_buttons_msg)
        # Keep the physical long press on ECAN for CarState, but don't forward it to CAM.
        values["NORMAL_CRUISE_MAIN_BTN"] = 0

        if  HDA_LFA_SymSta == 0 and 0 < frame % 200 < 12:
          values["LFA_BTN"] = 1

        if CC.enabled and not interlock_active:
          if not CS.MainMode_ACC:
            if 10 < frame % 200 <= 16 and v_ego_kph > 10.:
              values["ADAPTIVE_CRUISE_MAIN_BTN"] = 1
          elif CS.ACCMode in [0, 4]:
            if 10 < frame % 200 <= 16 and v_ego_kph > 10.:
              values["CRUISE_BUTTONS"] = 2
          elif CS.scc_control is not None and CS.scc_control["InfoDisplay"] == 4:
            if 10 < frame % 30 <= 16 and not stopping:
              values["CRUISE_BUTTONS"] = 2
          else:
            if CS.adrv_0x1ea is not None and CS.adrv_0x1ea["HDA_MODE2"] == 0: # if corner radar is disabled, send main btn
              if 10 < frame % 1000 <= 16 and v_ego_kph > 10.:
                values["ADAPTIVE_CRUISE_MAIN_BTN"] = 1

        ret.append(packer.make_can_msg(CS.cruise_btns_msg_canfd, CAN.CAM, values))

    # --- 0x161/0x200/0x1ea/0x162 (frame%5) ---
    if frame % 5 == 0:
      lat_active = CC.latActive

      if CS.adrv_0x161 is not None:
        main_enabled = CS.out.cruiseState.available
        cruise_enabled = CC.enabled
        lat_enabled = CS.out.latEnabled
        nav_active = hud_control.activeCarrot > 1
        vehicle_navi_available = CS.out.vehicleNaviAvailable
        nav_icon_available = nav_active or vehicle_navi_available

        # hdpuse carrot
        hdp_use = Params().get_int("HDPuse")
        hdp_active = False
        if hdp_use == 1:
          hdp_active = cruise_enabled and nav_active
        elif hdp_use == 2:
          hdp_active = cruise_enabled
        # hdpuse carrot

        values = CS.adrv_0x161.copy()
        rx_counter = values.pop("COUNTER", None)
        values["SETSPEED"] = (6 if hdp_active else 3 if cruise_enabled else 1) if main_enabled else 0
        values["SETSPEED_HUD"] = (5 if hdp_active else 3 if cruise_enabled else 1) if main_enabled else 0

        set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)
        values["vSetDis"] = int(set_speed_in_units + 0.5)

        if cruise_enabled:
          if CS.out.vCruiseCluster > values["vSetDis"]:
            if create_ccnc_messages.sla_active_time < 1:
              create_ccnc_messages.sla_active_time = time.monotonic()
            values["SETSPEED"] = 2
            values["SETSPEED_HUD"] = 2
            elapsed = time.monotonic() - create_ccnc_messages.sla_active_time
            values["SLA_ICON"] = 2 if (elapsed % 3.5) < 2.0 else 0
          else:
            create_ccnc_messages.sla_active_time = 0
            if CS.ccnc_0x162 is not None and values["SLA_ICON"] > 0:
              if CS.ccnc_0x162["SPEEDLIMIT"] > CS.out.vCruiseCluster:
                values["SLA_ICON"] = 3
              elif CS.ccnc_0x162["SPEEDLIMIT"] < CS.out.vCruiseCluster:
                values["SLA_ICON"] = 4
              else:
                values["SLA_ICON"] = 0
        else:
          create_ccnc_messages.sla_active_time = 0

        values["DISTANCE"] = 4 if hdp_active else hud_control.leadDistanceBars
        values["DISTANCE_LEAD"] = 2 if cruise_enabled and hud_control.leadVisible else 1 if main_enabled and hud_control.leadVisible else 0
        values["DISTANCE_CAR"] = 3 if hdp_active else 2 if cruise_enabled else 1 if main_enabled else 0
        values["DISTANCE_SPACING"] = 5 if hdp_active else 1 if cruise_enabled else 0

        # Preserve the received TARGET and TARGET_DISTANCE for cluster flicker testing.

        values["BACKGROUND"] = _select_cluster_background(
          cruise_enabled, lat_active, CS.paddle_button_prev > 0, paddle_mode,
        )
        values["CENTERLINE"] = 1 if HDA_CntrlModSta > 0 else 0
        values["CAR_CIRCLE"] = 2 if hdp_active else 1 if cruise_enabled else 0

        values["NAV_ICON"] = 2 if nav_icon_available and cruise_enabled else 1 if main_enabled and nav_icon_available else 0
        values["HDA_ICON"] = 5 if hdp_active else 2 if cruise_enabled else 1 if main_enabled else 0
        # ==============================================================================
        # 계기판 LFA 아이콘 상태 제어
        # [LFA_ICON] 0: HIDDEN, 1: GRAY, 2: GREEN, 3: WHITE, 5: CYAN
        # [LKA_ICON] 할당 생략 (순정 값 유지)
        # ==============================================================================
        if lat_enabled:
          if lat_active:
            if CS.out.steeringPressed:
              # 횡컨 활성 중 운전자 조향 개입
              values["LFA_ICON"] = 3  # WHITE
            else:
              # 능동 제어 중 (HDP: 청색, 일반: 녹색)
              values["LFA_ICON"] = 5 if hdp_active else 2
          else:
            # 횡컨 켜짐 + 모델 미제어 (대기 상태)
            values["LFA_ICON"] = 1  # GRAY
        else:
          # 횡컨 OFF -> 아이콘 숨김
          values["LFA_ICON"] = 0
        values["FCA_ALT_ICON"] = 0

        if values["ALERTS_2"] in [1, 2, 5, 6, 10, 21, 22]:
          values["ALERTS_2"] = 0
          values["DAW_ICON"] = 0

        if values["ALERTS_1"] == 0: # alerts가 있으면 사운드도 같이 나옴
          values["SOUNDS_1"] = 0
          values["SOUNDS_2"] = 0
          values["SOUNDS_4"] = 0

        if values["ALERTS_3"] in [3, 4, 11, 12, 13, 14, 17, 19, 20, 26, 27, 28, 7, 8, 9, 10]: # hide gap distance msg.(11,12,13,14), lanechange(19,20,27, 28)
          values["ALERTS_3"] = 0
          values["SOUNDS_3"] = 0

        if values["ALERTS_5"] in [1, 2, 3, 4, 5]:
          values["ALERTS_5"] = 0

        # if values["ALERTS_5"] in [11] and CS.softHoldActive == 0:
        #   values["ALERTS_5"] = 0

        # curvature 표시(0x161쪽 기존 로직 유지)
        _suppress_trailer_mode_warning(values, CS)

        curvature = round(CS.out.steeringAngleDeg / 3)
        values["LANELINE_CURVATURE"] = (min(abs(curvature), 15) + (-1 if curvature < 0 else 0)) if lat_active else 0
        values["LANELINE_CURVATURE_DIRECTION"] = 1 if curvature < 0 and lat_active else 0

        trailer_lane_change_blocked = CS.trailer_connected
        if trailer_lane_change_blocked:
          values["LANELINE_LEFT"] = 2 if hud_control.leftLaneVisible else 0
          values["LANELINE_RIGHT"] = 2 if hud_control.rightLaneVisible else 0
        else:
          if lat_active:
            lane_color = 6 if md is not None and md.meta.laneChangeAvailableLeft else 2
          else:
            lane_color = 0
          if lane_line_check >= 1:
            lane_line_warn_left = CS.out.leftLaneLine % 10 not in (0, 5)
          else:
            lane_line_warn_left = CS.out.leftLaneLine // 10 == 2
          lane_color = 4 if lane_line_warn_left or CS.out.leftBlindspot else lane_color
          if hud_control.leftLaneDepart:
            values["LANELINE_LEFT"] = 4 if (frame // 50) % 2 == 0 else 1
          else:
            values["LANELINE_LEFT"] = lane_color if hud_control.leftLaneVisible else 0

          if lat_active:
            lane_color = 6 if md is not None and md.meta.laneChangeAvailableRight else 2
          else:
            lane_color = 0
          if lane_line_check >= 1:
            lane_line_warn_right = CS.out.rightLaneLine % 10 not in (0, 5)
          else:
            lane_line_warn_right = CS.out.rightLaneLine // 10 == 2
          lane_color = 4 if lane_line_warn_right or CS.out.rightBlindspot else lane_color
          if hud_control.rightLaneDepart:
            values["LANELINE_RIGHT"] = 4 if (frame // 50) % 2 == 0 else 1
          else:
            values["LANELINE_RIGHT"] = lane_color if hud_control.rightLaneVisible else 0

        values["LCA_LEFT_ARROW"] = 2 if CS.out.leftBlinker else 0
        values["LCA_RIGHT_ARROW"] = 2 if CS.out.rightBlinker else 0

        # 기어 상태에 따른 차로 색 변경
        if CS.out.gearShifter == structs.CarState.GearShifter.drive:
          try:
            # Carrot의 드라이브 모드 파라미터를 가져옵니다 (1: Eco, 2: Safe, 3: Normal, 4: High Speed)
            drive_mode = Params().get_int("MyDrivingMode")
          except Exception:
            drive_mode = 3  # 기본값 (Normal)

          # 속도에 비례해 하이라이트 길이 동적으로 조절
          values["LANE_HIGHLIGHT_DISTANCE"] = int(ease_in_interp(v_ego_kph, [0, 80], [3, 60], power=1.5))
          values["LANE_HIGHLIGHT"] = create_ccnc_messages.drive_lane_color.update(a_ego_kph, drive_mode, v_ego_kph)
        elif CS.out.gearShifter == structs.CarState.GearShifter.reverse:
          values["LANE_HIGHLIGHT"] = 5
        elif CS.out.gearShifter == structs.CarState.GearShifter.neutral:
          values["LANE_HIGHLIGHT"] = 4
        elif CS.out.gearShifter == structs.CarState.GearShifter.park:
          if not CS.out.parkingBrake:
            values["LANE_HIGHLIGHT"] = 2

        # 차선 변경 판단
        is_auto_lane_changing = desire in (3, 4)
        is_blinking = CS.out.leftBlinker != CS.out.rightBlinker
        is_currently_lane_changing = is_auto_lane_changing or (is_blinking and v_ego_kph > 20.0)

        try:
          if lat_enabled:
            # 스칼라 np.interp 오버헤드 제거 (선형 보간 수식 직접 계산: 20~100 kph -> 30~80 m)
            max_lookahead_x = 30.0 + min(max((v_ego_kph - 20.0) / 80.0, 0.0), 1.0) * 50.0

            # 객체 속성 접근 오버헤드 캐싱 (루프 내 다중 점근 방지)
            pos = md.position
            pos_x, pos_y, pos_y_std = pos.x, pos.y, pos.yStd

            trust_threshold = 0.8
            max_y_abs = 0.0
            peak_idx = 0
            start_search_idx = 0
            start_found = not is_currently_lane_changing
            min_calc_dist = 20.0 if is_currently_lane_changing else 0.0

            for i in range(1, len(pos_x)):
              x = pos_x[i]

              if not start_found and x >= min_calc_dist:
                start_search_idx = i
                start_found = True

              if pos_y_std[i] > trust_threshold or x > max_lookahead_x:
                break

              y_abs = abs(pos_y[i])
              if y_abs > max_y_abs:
                max_y_abs = y_abs
                peak_idx = i

            if start_search_idx != peak_idx and pos_x[peak_idx] >= (20.0 + min_calc_dist):
              x_dist = pos_x[peak_idx]
              y_diff = pos_y[peak_idx] - pos_y[start_search_idx]
              # 곡률 공식: (2y / x^2) * 1800 -> (3600 * y) / x^2
              max_curve_val = (3600.0 * y_diff) / (x_dist * x_dist)
            else:
              max_curve_val = 0.0

            curvature = round(create_ccnc_messages.lane_curv.apply(-max_curve_val))
          else:
            curvature = round(CS.out.steeringAngleDeg / 3)

        except:
          # 모델 데이터 예외 발생 시 핸들 각도 기반 백업
          curvature = round(CS.out.steeringAngleDeg / 3)
          values["LFA_ICON"] = 5

        values["LANELINE_CURVATURE"] = min(abs(curvature), 15) + (-1 if curvature < 0 else 0)
        values["LANELINE_CURVATURE_DIRECTION"] = 1 if curvature < 0 else 0

        try:
          # 차선 위치 갱신: 항시 적용
          l_prob = md.laneLineProbs[1]
          r_prob = md.laneLineProbs[2]

          # --- 차선 변경 상태 관리 및 알파값 조정 ---
          if is_currently_lane_changing != create_ccnc_messages._is_lane_change_active:
            if is_currently_lane_changing:
              create_ccnc_messages.l_lane_f.update_alpha(0.6)
              create_ccnc_messages.r_lane_f.update_alpha(0.6)
            else:
              create_ccnc_messages.l_lane_f.reset_alpha()
              create_ccnc_messages.r_lane_f.reset_alpha()
            create_ccnc_messages._is_lane_change_active = is_currently_lane_changing

          leftlaneraw = abs(md.laneLines[1].y[0])
          rightlaneraw = abs(md.laneLines[2].y[0])

          l_valid = l_prob > 0.3 or is_auto_lane_changing or is_blinking
          r_valid = r_prob > 0.3 or is_auto_lane_changing or is_blinking

          if not l_valid and not r_valid:
            leftlaneraw = rightlaneraw = 1.5
          elif not l_valid:
            leftlaneraw = create_ccnc_messages.last_known_lane_width - rightlaneraw
          elif not r_valid:
            rightlaneraw = create_ccnc_messages.last_known_lane_width - leftlaneraw

          if is_currently_lane_changing:
            is_moving_left = CS.out.leftBlinker or desire == 3
            # 위상 변화 시 차선 강조 변경
            if not create_ccnc_messages.draw_center:

              lane_raw = leftlaneraw if is_moving_left else rightlaneraw

              is_phase_shifted = lane_raw < 0.1 or (lane_raw - create_ccnc_messages.lane_phase_min) > 0.3
              create_ccnc_messages.lane_phase_min = min(create_ccnc_messages.lane_phase_min, lane_raw)

              if is_phase_shifted:
                create_ccnc_messages.draw_center = create_ccnc_messages.hold_lane = True
                create_ccnc_messages.lane_phase_min = 6.0
                create_ccnc_messages.hold_lane_escape_count = 0

                prev_l_val = create_ccnc_messages.l_lane_f.value
                prev_r_val = create_ccnc_messages.r_lane_f.value
                if is_moving_left:
                  create_ccnc_messages.r_lane_f.reset(prev_l_val)
                else:
                  create_ccnc_messages.l_lane_f.reset(prev_r_val)

            # RNN 보간 방지
            if create_ccnc_messages.hold_lane:
              swapped_lane_position = rightlaneraw if is_moving_left else leftlaneraw

              # 줄어드는 최솟값을 지속적으로 갱신
              create_ccnc_messages.lane_phase_min = min(create_ccnc_messages.lane_phase_min, swapped_lane_position)

              # 최솟값 대비 0.1m 이상 반등하면 작아지다 커지는 위상으로 판단
              if swapped_lane_position - create_ccnc_messages.lane_phase_min > 0.1:
                create_ccnc_messages.hold_lane_escape_count += 1
                if create_ccnc_messages.hold_lane_escape_count >= 2:
                  create_ccnc_messages.hold_lane = False
              else:
                create_ccnc_messages.hold_lane_escape_count = 0

              holding_factor = create_ccnc_messages.hold_lane_escape_count * 0.1
              if is_moving_left:
                current_l_target = create_ccnc_messages.l_lane_f.reset(create_ccnc_messages.last_known_lane_width - holding_factor)
                current_r_target = create_ccnc_messages.r_lane_f.reset(holding_factor)
              else:
                current_l_target = create_ccnc_messages.l_lane_f.reset(holding_factor)
                current_r_target = create_ccnc_messages.r_lane_f.reset(create_ccnc_messages.last_known_lane_width - holding_factor)
            elif create_ccnc_messages.draw_center:
              MAX_STEP = 0.15  # 한 루프(프레임)당 최대 허용 변화량 (m단위, 부드러움 조절용)
              prev_l = create_ccnc_messages.l_lane_f.value
              prev_r = create_ccnc_messages.r_lane_f.value
              # 실제 값과 이전 값의 차이를 MAX_STEP 이내로 제한 (클리핑)
              bounded_l = prev_l + np.clip(leftlaneraw - prev_l, -MAX_STEP, MAX_STEP)
              bounded_r = prev_r + np.clip(rightlaneraw - prev_r, -MAX_STEP, MAX_STEP)
              current_l_target = create_ccnc_messages.l_lane_f.apply(bounded_l)
              current_r_target = create_ccnc_messages.r_lane_f.apply(bounded_r)
            else:
              current_l_target = create_ccnc_messages.l_lane_f.apply(leftlaneraw)
              current_r_target = create_ccnc_messages.r_lane_f.apply(rightlaneraw)

            # LCA 중에는 차로 강조
            if is_auto_lane_changing:
              if create_ccnc_messages.draw_center:
                values["LANE_HIGHLIGHT"] = 1
                values["LANE_HIGHLIGHT_DISTANCE"] = 60
              else:
                values["LANE_LEFT" if desire == 3 else "LANE_RIGHT"] = 1
            elif abs(current_l_target - current_r_target) < create_ccnc_messages.last_known_lane_width / 5:
              create_ccnc_messages.draw_center = create_ccnc_messages.hold_lane = False
              create_ccnc_messages.hold_lane_escape_count = 0
              create_ccnc_messages.lane_phase_min = 10.0
          else:
            create_ccnc_messages.draw_center = create_ccnc_messages.hold_lane = False
            create_ccnc_messages.hold_lane_escape_count = 0
            create_ccnc_messages.lane_phase_min = 10.0
            current_l_target = create_ccnc_messages.l_lane_f.apply(leftlaneraw)
            current_r_target = create_ccnc_messages.r_lane_f.apply(rightlaneraw)

            lane_width = current_l_target + current_r_target
            if 2 < lane_width < 4.6:
              create_ccnc_messages.last_known_lane_width = lane_width # 마지막 차선 폭을 기억해둠

          values["LANELINE_LEFT_POSITION"] = int(round(np.interp(current_l_target, [0.0, 3.0], [0, 30])))
          values["LANELINE_RIGHT_POSITION"] = int(round(np.interp(current_r_target, [0.0, 3.0], [0, 30])))

          # 차선 변경 아이콘
          if lat_enabled:
            values["LCA_LEFT_ICON"] = 1 if CS.out.leftBlindspot else 4 if CS.out.rightBlinker or not md.meta.laneChangeAvailableLeft else 2
            values["LCA_RIGHT_ICON"] = 1 if CS.out.rightBlindspot else 4 if CS.out.leftBlinker or not md.meta.laneChangeAvailableRight else 2
        except:
          # Only show startup status while the required model lane data is absent.
          # Calculation errors with populated lane data keep the original alert.
          if (md is None or len(md.laneLineProbs) < 3 or len(md.laneLines) < 3 or
              len(md.laneLines[1].y) == 0 or len(md.laneLines[2].y) == 0):
            values["ALERTS_5"] = 19  # ACTIVATING_HIGHWAY_DRIVING_PILOT_SYSTEM
          else:
            values["LANELINE_LEFT_POSITION"] = 30
            values["LANELINE_RIGHT_POSITION"] = 30
            values["LANE_HIGHLIGHT"] = 3
            values["LANE_HIGHLIGHT_DISTANCE"] = 60
            values["LANE_LEFT"] = 1
            values["LANE_RIGHT"] = 1
            values["LKA_ICON"] = 1

        ret.append(packer.make_can_msg("ADRV_0x161", CAN.ECAN, values, rx_counter = rx_counter))

      if CS.adrv_0x200 is not None:
        values = CS.adrv_0x200.copy()
        rx_counter = values.pop("COUNTER", None)
        values["TauGapSet"] = hud_control.leadDistanceBars
        ret.append(packer.make_can_msg("ADRV_0x200", CAN.ECAN, values, rx_counter = rx_counter))

      if CS.adrv_0x1ea is not None:
        values = CS.adrv_0x1ea.copy()
        rx_counter = values.pop("COUNTER", None)
        # blinker hold
        values['LEFT_BLINK_HOLD'] = 1 if lane_changing == 3 else 0
        values['RIGHT_BLINK_HOLD'] = 1 if lane_changing == 4 else 0

        _make_ccnc_values(
          values, CS, lat_active, frame, hud_control,
          lane_line=True,
          corner_radar=True,
          desire=desire,
          # 기존대로 LR/RR만 깜빡임
          blink_pairs=[('LR_DETECT', 'LR_DETECT_DISTANCE'),
                       ('RR_DETECT', 'RR_DETECT_DISTANCE')],
          blink_t=1.0
        )

        ret.append(packer.make_can_msg("ADRV_0x1ea", CAN.ECAN, values, rx_counter = rx_counter))

      if CS.ccnc_0x162 is not None:
        values = CS.ccnc_0x162.copy()

        # --- liveTracks 원본 레이더를 이용한 전방 차량 감지 ---
        try:
          ff_lead = lf_lead = rf_lead = None
          ff_yRel = lf_yRel = rf_yRel = 0

          display_tracker = create_ccnc_messages.radar_display_tracker
          display_tracker.observe(CS.live_tracks, frame)
          left_prob = right_prob = 0.0
          if md is not None and len(md.laneLineProbs) >= 3 and len(md.laneLines) >= 3:
            if _ccnc_valid_boundary(md.laneLines[1].x, md.laneLines[1].y):
              left_prob = md.laneLineProbs[1]
            if _ccnc_valid_boundary(md.laneLines[2].x, md.laneLines[2].y):
              right_prob = md.laneLineProbs[2]
          selected_lane_is_left = create_ccnc_messages.radar_lane_selector.update(left_prob, right_prob, frame)
          selected_lane_prob = left_prob if selected_lane_is_left else right_prob

          # 차선이 유효하면 기존 분류를 사용하고, FF는 차선 소실 시 경로/영상으로 보완합니다.
          lane_mode = selected_lane_prob >= 0.1
          ff_lane_mode = display_tracker.lane_available(selected_lane_prob, frame)
          if CS.live_tracks is not None and lane_mode:
            lane_lines = md.laneLines
            road_edges = md.roadEdges
            # Cap’n Proto 목록을 보간 호출마다 변환하지 않도록 한 번만 배열로 만듭니다.
            left_inner_x = np.asarray(lane_lines[1].x, dtype=np.float64)
            left_inner_y = np.asarray(lane_lines[1].y, dtype=np.float64)
            right_inner_x = np.asarray(lane_lines[2].x, dtype=np.float64)
            right_inner_y = np.asarray(lane_lines[2].y, dtype=np.float64)
            left_outer_x, left_outer_y = lane_lines[0].x, lane_lines[0].y
            right_outer_x, right_outer_y = lane_lines[3].x, lane_lines[3].y
            left_road_edge_x, left_road_edge_y = road_edges[0].x, road_edges[0].y
            right_road_edge_x, right_road_edge_y = road_edges[1].x, road_edges[1].y
            selected_lane_x, selected_lane_y = (
              (left_inner_x, left_inner_y) if selected_lane_is_left else (right_inner_x, right_inner_y)
            )
            selected_lane_y0 = selected_lane_y[0]

            interp = np.interp
            ms_to_kph = CV.MS_TO_KPH

            has_left_outer = md.laneLineProbs[0] > 0.1 and _ccnc_valid_boundary(left_outer_x, left_outer_y)
            has_right_outer = md.laneLineProbs[3] > 0.1 and _ccnc_valid_boundary(right_outer_x, right_outer_y)
            # HUD와 같이 roadEdgeStds로 경계를 버리지 않고 실제 좌표를 사용합니다.
            has_left_edge = _ccnc_valid_boundary(left_road_edge_x, left_road_edge_y)
            has_right_edge = _ccnc_valid_boundary(right_road_edge_x, right_road_edge_y)

            if has_left_outer:
              left_outer_x = np.asarray(left_outer_x, dtype=np.float64)
              left_outer_y = np.asarray(left_outer_y, dtype=np.float64)
            if has_right_outer:
              right_outer_x = np.asarray(right_outer_x, dtype=np.float64)
              right_outer_y = np.asarray(right_outer_y, dtype=np.float64)
            if has_left_edge:
              left_road_edge_x = np.asarray(left_road_edge_x, dtype=np.float64)
              left_road_edge_y = np.asarray(left_road_edge_y, dtype=np.float64)
            if has_right_edge:
              right_road_edge_x = np.asarray(right_road_edge_x, dtype=np.float64)
              right_road_edge_y = np.asarray(right_road_edge_y, dtype=np.float64)

            # 여러 차로의 후보를 허용하되, 보정 후 횡거리 5.4m 밖의 측면 점은 선택하지 않습니다.
            max_side_lateral = 5.4
            ff_min_dist = lf_min_dist = rf_min_dist = math.inf
            min_front_lead_speed = -100 if a_ego_kph < -3 else interp(v_ego_kph, [30, 40, 100], [-100, 0, 20])
            min_side_lead_speed = interp(v_ego_kph, [0, 30, 100], [2, 10, 20])
            lowspeed_side_lead_speed = interp(v_ego_kph, [10, 40], [-1, 10])

            for lead in CS.live_tracks.points:
              dRel = lead.dRel
              yRel, vRel, vLead = lead.yRel, lead.vRel, lead.vLead
              # 순정 SCC 선행차는 횡위치가 없으므로 개별 전방 레이더 점만 표시합니다.
              if (str(lead.radarSource) != "frontRadar" or dRel < 1
                  or not all(math.isfinite(v) for v in (dRel, yRel, vRel, vLead))):
                continue

              velocity = vLead * ms_to_kph
              # FF와 저속 측면 예외까지 모두 탈락하는 점은 보간 전에 제외합니다.
              if not (velocity > min_front_lead_speed or velocity > min_side_lead_speed
                      or (dRel < 30 and velocity > lowspeed_side_lead_speed)):
                continue

              # Only stable moving tracks may use the additional outer-lane allowance.
              allow_lane_margin = (velocity > min_side_lead_speed
                                   and display_tracker.stable(lead.trackId, 30))

              # 차선 보간값과 시작점의 차이로 도로의 휘어짐만 보정합니다.
              lane_y_at_drel = interp(dRel, selected_lane_x, selected_lane_y)
              road_aligned_yRel = yRel + (lane_y_at_drel - selected_lane_y0)

              # 분류는 원본 레이더 좌표(좌측+)와 같은 좌표계의 차선 경계로 판단합니다.
              if selected_lane_is_left:
                left_inner_bound = -lane_y_at_drel
                right_inner_bound = -interp(dRel, right_inner_x, right_inner_y)
              else:
                left_inner_bound = -interp(dRel, left_inner_x, left_inner_y)
                right_inner_bound = -lane_y_at_drel
              if (not math.isfinite(left_inner_bound) or not math.isfinite(right_inner_bound)
                  or right_inner_bound >= left_inner_bound):
                continue

              # 거리 순위만 필요하므로 원본 좌표의 제곱거리로 비교합니다.
              dist_score = dRel * dRel + yRel * yRel

              # 2. [전방 주행 차선] - 외곽선/도로경계선 interp 4회 전부 생략
              if right_inner_bound <= yRel <= left_inner_bound:
                if dist_score < ff_min_dist:
                  if velocity > min_front_lead_speed:
                    ff_min_dist, ff_lead, ff_yRel = dist_score, lead, road_aligned_yRel

              # 3. [왼쪽 차선 차량] - 좌측 외곽/도로경계선만 지연 계산 (우측 2회 interp 생략)
              elif left_inner_bound < yRel:
                if (display_tracker.stable(lead.trackId)
                    and abs(road_aligned_yRel) <= max_side_lateral and dist_score < lf_min_dist):

                  # 속도 조건을 통과한 모든 후보에 외곽 차선/도로 경계 검사를 적용합니다.
                  if (velocity > min_side_lead_speed
                      or (dRel < 30 and velocity > lowspeed_side_lead_speed)):
                    # Expand the outer lane for moving candidates, keeping road-edge clearance.
                    lane_margin = 0.75 if allow_lane_margin else -0.25
                    if lead.trackId == display_tracker.selected[1]:
                      lane_margin += 0.3
                    valid_left_bounds = []
                    left_effective_bound = math.inf
                    if has_left_outer and left_outer_x[0] <= dRel <= left_outer_x[-1]:
                      outer_bound = -interp(dRel, left_outer_x, left_outer_y)
                      valid_left_bounds.append(outer_bound)
                      left_effective_bound = outer_bound + lane_margin
                    if has_left_edge and left_road_edge_x[0] <= dRel <= left_road_edge_x[-1]:
                      edge_bound = -interp(dRel, left_road_edge_x, left_road_edge_y)
                      valid_left_bounds.append(edge_bound)
                      left_effective_bound = min(left_effective_bound, edge_bound - 0.25)

                    if valid_left_bounds:
                      # Preserve the original width check before expanding candidate acceptance.
                      left_width_bound = min(valid_left_bounds) - 0.25
                      if yRel < left_effective_bound and (left_width_bound - left_inner_bound > 1.8):
                        lf_min_dist, lf_lead, lf_yRel = dist_score, lead, road_aligned_yRel

              # 4. [오른쪽 차선 차량] - 우측 외곽/도로경계선만 지연 계산 (좌측 2회 interp 생략)
              elif yRel < right_inner_bound:
                if (display_tracker.stable(lead.trackId)
                    and abs(road_aligned_yRel) <= max_side_lateral and dist_score < rf_min_dist):

                  # 속도 조건을 통과한 모든 후보에 외곽 차선/도로 경계 검사를 적용합니다.
                  if (velocity > min_side_lead_speed
                      or (dRel < 30 and velocity > lowspeed_side_lead_speed)):
                    # Expand the outer lane for moving candidates, keeping road-edge clearance.
                    lane_margin = 0.75 if allow_lane_margin else -0.25
                    if lead.trackId == display_tracker.selected[2]:
                      lane_margin += 0.3
                    valid_right_bounds = []
                    right_effective_bound = -math.inf
                    if has_right_outer and right_outer_x[0] <= dRel <= right_outer_x[-1]:
                      outer_bound = -interp(dRel, right_outer_x, right_outer_y)
                      valid_right_bounds.append(outer_bound)
                      right_effective_bound = outer_bound - lane_margin
                    if has_right_edge and right_road_edge_x[0] <= dRel <= right_road_edge_x[-1]:
                      edge_bound = -interp(dRel, right_road_edge_x, right_road_edge_y)
                      valid_right_bounds.append(edge_bound)
                      right_effective_bound = max(right_effective_bound, edge_bound + 0.25)

                    if valid_right_bounds:
                      # Preserve the original width check before expanding candidate acceptance.
                      right_width_bound = max(valid_right_bounds) + 0.25
                      if yRel > right_effective_bound and (right_inner_bound - right_width_bound > 1.8):
                        rf_min_dist, rf_lead, rf_yRel = dist_score, lead, road_aligned_yRel

          if CS.live_tracks is not None and not ff_lane_mode:
            minimum_speed = -100 if a_ego_kph < -3 else np.interp(v_ego_kph, [30, 40, 100], [-100, 0, 20])
            ff_lead, ff_yRel = display_tracker.path_lead(md, minimum_speed)

          # A retained FF must not also occupy a side slot during lane recovery.
          if ff_lead is not None:
            if lf_lead is not None and lf_lead.trackId == ff_lead.trackId:
              lf_lead = None
            if rf_lead is not None and rf_lead.trackId == ff_lead.trackId:
              rf_lead = None
          # Shared physical coordinates and filter history follow the ID across LF/FF/RF.
          filtered_positions = []
          for point, aligned, is_lane in ((ff_lead, ff_yRel, ff_lane_mode),
                                          (lf_lead, lf_yRel, True), (rf_lead, rf_yRel, True)):
            reference = ("lane", selected_lane_is_left) if is_lane else ("path", False)
            filtered_positions.append(display_tracker.filter_position(point, aligned, reference, frame)
                                      if point is not None else (0.0, 0.0))
          ff_yRel = filtered_positions[0][1]
          lf_yRel = float(np.clip(filtered_positions[1][1], -5.4, 5.4))
          rf_yRel = float(np.clip(filtered_positions[2][1], -5.4, 5.4))
          ff_yRel, changed_tracks = display_tracker.finish(ff_lead, lf_lead, rf_lead, ff_yRel, ff_lane_mode, frame)

          # 전방(FF) 차량 정보 업데이트
          if ff_lead:
            values["FF_DISTANCE"] = filtered_positions[0][0] * 0.8
            values["FF_LATERAL"] = apply_curved_deadband(-ff_yRel, 0, 0.7, 1)
            values["FF_DETECT"] = CAR_MODEL_ID + 1 if ff_lead.vLead < 3 else create_ccnc_messages.ff_detect.apply(ff_lead.vRel)
          else:
            values["FF_DETECT"] = 0 # 순정 디텍션 제거
          # LF/RF 횡거리는 4m로 압축하지 않고 CAN 신호 범위(7-bit unsigned, 0.1m)만 제한합니다.
          # 전방 좌측(LF) 차량 정보 업데이트
          if lf_lead:
            values["LF_DETECT_DISTANCE"] = filtered_positions[1][0] * 0.8
            values["LF_DETECT_LATERAL"] = float(np.clip(apply_curved_deadband(lf_yRel, 3, 0.9, 2), 0.0, 12.7))
            values["LF_DETECT"] = create_ccnc_messages.lf_detect.apply(lf_lead.vRel)
          else:
            values["LF_DETECT"] = 0
          # 전방 우측(RF) 차량 정보 업데이트
          if rf_lead:
            values["RF_DETECT_DISTANCE"] = filtered_positions[2][0] * 0.8
            values["RF_DETECT_LATERAL"] = float(np.clip(apply_curved_deadband(-rf_yRel, 3, 0.9, 2), 0.0, 12.7))
            values["RF_DETECT"] = create_ccnc_messages.rf_detect.apply(rf_lead.vRel)
          else:
            values["RF_DETECT"] = 0

          center_lane_offset = (create_ccnc_messages.r_lane_f.value - create_ccnc_messages.l_lane_f.value) / 2

          # --- 후측방은 BSD 경고 시 고정 위치에 두부 출력. HDA1은 후측방 레이더 정보가 안채워져서 옴 ---
          BSD_LATERAL_FIXED = 2.8
          if CS.out.leftBlindspot:
            values["LR_DETECT_DISTANCE"] = create_ccnc_messages.lr_distance.apply(8)
            values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
            values["LR_DETECT"] = CAR_MODEL_ID + 1
          elif create_ccnc_messages.lr_distance.value < 15:
            values["LR_DETECT_DISTANCE"] = create_ccnc_messages.lr_distance.apply(16)
            values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
            values["LR_DETECT"] = CAR_MODEL_ID

          if CS.out.rightBlindspot:
            values["RR_DETECT_DISTANCE"] = create_ccnc_messages.rr_distance.apply(8) # 8m
            values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
            values["RR_DETECT"] = CAR_MODEL_ID + 1
          elif create_ccnc_messages.rr_distance.value < 15:
            values["RR_DETECT_DISTANCE"] = create_ccnc_messages.rr_distance.apply(16)
            values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
            values["RR_DETECT"] = CAR_MODEL_ID

        except:
          values = CS.ccnc_0x162.copy()
          values["FF_DISTANCE"] = 24
          values["FF_DETECT"] = 7
          values["LF_DETECT_DISTANCE"] = 12
          values["LF_DETECT_LATERAL"] = 1.5
          values["LF_DETECT"] = 4
          values["RF_DETECT_DISTANCE"] = 12
          values["RF_DETECT_LATERAL"] = 1.5
          values["RF_DETECT"] = 9
          values["LR_DETECT_DISTANCE"] = 1
          values["LR_DETECT_LATERAL"] = 3
          values["LR_DETECT"] = 6
          values["RR_DETECT_DISTANCE"] = 1
          values["RR_DETECT_LATERAL"] = 3
          values["RR_DETECT"] = 11

        if (left_lane_warning and not CS.out.leftBlinker) or (right_lane_warning and not CS.out.rightBlinker):
          values["VIBRATE"] = 1

        _hide_replaced_adas_service_warning(values)

        if canfd_debug > 0:
          values["FAULT_LSS"] = 0
          values["FAULT_DAS"] = 0

        ret.append(packer.make_can_msg("CCNC_0x162", CAN.ECAN, values))

    # --- NEW_MSG_4B9 (corner radar keep-alive?) ---
    if enable_corner_radar > 0:
      if HDA_CntrlModSta == 0:
        if frame % 500 in [10, 20, 30]:
          values = {
            'BYTE_1': 0,
            'BYTE_2': 0,
            'BYTE_3': 0x80,
            'BYTE_4': 0x8A,
            'BYTE_5': 0x32,
            'BYTE_6': 0x30,
            'BYTE_7': 0x01,
            'BYTE_8': 0x00,
          }
          ret.append(packer.make_can_msg("NEW_MSG_4B9", CAN.CAM, values))
        elif frame % 500 in [40, 50, 60]:
          values = {
            'BYTE_1': 0xff,
            'BYTE_2': 0xff,
            'BYTE_3': 0xff,
            'BYTE_4': 0xff,
            'BYTE_5': 0xff,
            'BYTE_6': 0xff,
            'BYTE_7': 0xff,
            'BYTE_8': 0xff,
          }
          ret.append(packer.make_can_msg("NEW_MSG_4B9", CAN.CAM, values))

      if False:  # canfd_debug > 1 and frame % 20 == 0:
        if CS.hda_info_4a3 is not None:
          values = CS.hda_info_4a3.copy()
          values["LinkClass"] = 1
          values["SPEED_LIMIT"] = 100
          ret.append(packer.make_can_msg("HDA_INFO_4A3", CAN.CAM, values))

  return ret

# 곡률 노이즈 필터
create_ccnc_messages.lane_curv = NoiseFilter(3, 0, alpha_range=0.5) # 3-frame median, 0 initial, 0.5 alpha

# 차선 변경 상태 플래그
create_ccnc_messages._is_lane_change_active = False

# 차선 넘어감 감지
create_ccnc_messages.draw_center = False
create_ccnc_messages.hold_lane_escape_count = 0
create_ccnc_messages.lane_phase_min = 10.0

# 레이더 곡률 보정용 기준 차선 선택 상태
create_ccnc_messages.radar_lane_selector = _CcncRadarLaneSelector()
create_ccnc_messages.radar_display_tracker = _CcncRadarDisplayTracker()

# 차선 노이즈 필터
create_ccnc_messages.last_known_lane_width = 3.0 # Default lane width
create_ccnc_messages.l_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2) # 3-frame median, 1.5 initial, 0.2 alpha
create_ccnc_messages.r_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2) # 3-frame median, 1.5 initial, 0.2 alpha

# 차량 거리 필터
create_ccnc_messages.ff_detect = ThresholdTracker(bounds=(2, -1), states=(CAR_MODEL_ID, CAR_MODEL_ID + 1))
create_ccnc_messages.lf_detect = ThresholdTracker(bounds=(2, -1), states=(CAR_MODEL_ID, CAR_MODEL_ID + 1))
create_ccnc_messages.rf_detect = ThresholdTracker(bounds=(2, -1), states=(CAR_MODEL_ID, CAR_MODEL_ID + 1))

create_ccnc_messages.lr_distance = NoiseFilter(1, 15, alpha_range=0.05)
create_ccnc_messages.rr_distance = NoiseFilter(1, 15, alpha_range=0.05)

create_ccnc_messages.drive_lane_color = LaneHighlightStateMachine()

create_ccnc_messages.sla_active_time = 0
