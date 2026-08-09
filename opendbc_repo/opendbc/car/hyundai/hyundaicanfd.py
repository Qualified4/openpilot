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

class LaneHighlightStateMachine:
  def __init__(self):
    self.state = 0  # 현재 하이라이트 상태

  def update(self, accel, drive_mode, v_ego):
    # 상태별 전이 로직
    if self.state == 4:
      if accel < -0.3 and v_ego > 0.2:
        return self.state # 상태 유지
    elif self.state == 5:
      if drive_mode != 4 and accel > 1:
        return self.state # 상태 유지

    # 진입 로직 (우선순위 순서)
    if accel < -2.7:
      self.state = 4  # 급제동 (최우선)
    elif drive_mode == 4 or accel > 2.5:
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
    values = copy.copy(CS.mdps)
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
      values = copy.copy(CS.steer_touch_2af)
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
      values = copy.copy(CS.lfa_alt)
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
      values = copy.copy(CS.lfa)
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
    values = copy.copy(CS.lfahda_cluster)
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
    values = copy.copy(CS.adrv_0x161)
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

def create_acc_control_scc2(packer, CAN, enabled, accel_last, accel, stopping, gas_override, set_speed, hud_control, hyundai_jerk, CS):

  if CS.scc_control is None:
    return None
  enabled = (enabled or CS.softHoldActive > 0) and CS.paddle_button_prev == 0

  acc_mode = 0 if not enabled else (2 if gas_override else 1)

  if hyundai_jerk.carrot_cruise == 1:
    acc_mode = 4 if enabled else 0
    enabled = False
    accel = accel_last = 0.5

  elif hyundai_jerk.carrot_cruise == 2:
    accel = accel_last = hyundai_jerk.carrot_cruise_accel

  jerk_u = hyundai_jerk.jerk_u
  jerk_l = hyundai_jerk.jerk_l
  jerk = 5
  jn = jerk / 50
  if not enabled or gas_override:
    a_val, a_raw = 0, 0
  else:
    a_raw = accel
    a_val = accel #np.clip(accel, accel_last - jn, accel_last + jn)

  values = copy.copy(CS.scc_control)
  rx_counter = values.pop("COUNTER", None)
  values["ACCMode"] = acc_mode
  values["MainMode_ACC"] = 1
  values["StopReq"] = 1 if stopping or CS.softHoldActive > 0 else 0  # 1: Stop control is required, 2: Not used, 3: Error Indicator
  values["aReqValue"] = a_val
  values["aReqRaw"] = a_raw
  values["VSetDis"] = set_speed
  #values["JerkLowerLimit"] = jerk if enabled else 1
  #values["JerkUpperLimit"] = 3.0
  values["JerkLowerLimit"] = jerk_l if enabled else 1
  values["JerkUpperLimit"] = 2.0 if stopping or CS.softHoldActive else jerk_u
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

  hud_lead_info = 0
  if hud_control.leadVisible:
    hud_lead_info = 1 if values["ACC_ObjRelSpd"] > 0 else 2
  values["HUD_LEAD_INFO"] = hud_lead_info  #1: in-path object detected(uncontrollable), 2: controllable long, 3: controllable long & lat, ... reserved

  values["DriverAlert"] = 0   # 1: SCC Disengaged, 2: No SCC Engage condition, 3: SCC Disenganed when the vehicle stops

  values["TARGET_DISTANCE"] = CS.out.vEgo * 1.0 + 4.0

  soft_hold_info = 1 if CS.softHoldActive > 1 and enabled else 0

  # 이거안하면 정지중 뒤로 밀리는 현상 발생하는듯.. (신호정지중에 뒤로 밀리는 경험함.. 시험해봐야)
  if values["InfoDisplay"] != 5: #5: Front Car Departure Notice
    values["InfoDisplay"] = 4 if stopping and CS.out.aEgo > -0.3 else 0  # 1: SCC Mode, 2: Convention Cruise Mode, 3: Object disappered at low speed, 4: Available to resume acceleration control, 5: Front vehicle departure notice, 6: Reserved, 7: Invalid

  values["TakeOverReq"] = 0    # 1: Takeover request, 2: Not used, 3: Error indicator , 이것이 켜지면 가속을 안하는듯함.
  #values["NEW_SIGNAL_4"] = 9 if hud_control.leadVisible else 0
  # AccelLimitBandUpper, Lower
  values["SysFailState"] = 0    # 1: Performance degredation, 2: system temporairy unavailble, 3: SCC Service required , 눈이 묻어 레이더오류시... 2가 됨. 이때 가속을 안함...

  values["AccelLimitBandUpper"] = 0.0   # 이값이 1.26일때 가속을 안하는 증상이 보임..
  values["AccelLimitBandLower"] = 0.0

  values["ZEROS_7"] = 1

  return packer.make_can_msg("SCC_CONTROL", CAN.ECAN, values)

def create_acc_control(packer, CAN, enabled, accel_last, accel, stopping, gas_override, set_speed, hud_control, jerk_u, jerk_l, CS):

  enabled = enabled or CS.softHoldActive > 0
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
    "StopReq": 1 if stopping or CS.softHoldActive > 0 else 0,
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
    "InfoDisplay": 4 if stopping and CS.out.cruiseState.standstill else 0,
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
    values[det_key] = 2 - blink
    values[dist_key] = min_dist

def _suppress_trailer_mode_warning(values, CS):
  # Logs from IONIQ 9 show ALERTS_5=6 is the periodic
  # "driver assistance limited in trailer mode" popup.
  if CS.trailer_connected and values.get("ALERTS_5") == 6:
    values["ALERTS_5"] = 0

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
      if values[det_key] >= 4 and values[dist_key] != 0:
        values[det_key] = 1

    if blink_pairs:
      _apply_radar_blink(values, blink_pairs, frame, t=blink_t)

def create_ccnc_messages(CP, packer, CAN, frame, CC, CS, hud_control,
                         disp_angle, left_lane_warning, right_lane_warning,
                         enable_corner_radar, stopping, canfd_debug):
  ret = []

  md = CS.modelV2
  if not hasattr(create_ccnc_messages, '_lane_line_check') or frame % 100 == 0:
    create_ccnc_messages._lane_line_check = Params().get_int("LaneLineCheck")
  lane_line_check = create_ccnc_messages._lane_line_check
  desire, lane_changing = _get_desire_and_lane_changing(md)

  if CP.flags & HyundaiFlags.CAMERA_SCC.value:
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

        if CC.enabled:
          if not CS.MainMode_ACC:
            if 10 < frame % 200 <= 16 and CS.out.vEgo > 3.:
              values["ADAPTIVE_CRUISE_MAIN_BTN"] = 1
          elif CS.ACCMode in [0, 4]:
            if 10 < frame % 200 <= 16 and CS.out.vEgo > 3.:
              values["CRUISE_BUTTONS"] = 2
          elif CS.scc_control is not None and CS.scc_control["InfoDisplay"] == 4:
            if 10 < frame % 30 <= 16 and not stopping:
              values["CRUISE_BUTTONS"] = 2
          else:
            if CS.adrv_0x1ea is not None and CS.adrv_0x1ea["HDA_MODE2"] == 0: # if corner radar is disabled, send main btn
              if 10 < frame % 1000 <= 16 and CS.out.vEgo > 3:
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

        # hdpuse carrot
        hdp_use = int(Params().get("HDPuse"))
        hdp_active = False
        if hdp_use == 1:
          hdp_active = cruise_enabled and nav_active
        elif hdp_use == 2:
          hdp_active = cruise_enabled
        # hdpuse carrot

        values = copy.copy(CS.adrv_0x161)
        rx_counter = values.pop("COUNTER", None)
        values["SETSPEED"] = (6 if hdp_active else 3 if cruise_enabled else 1) if main_enabled else 0
        values["SETSPEED_HUD"] = (5 if hdp_active else 3 if cruise_enabled else 1) if main_enabled else 0

        set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)
        values["vSetDis"] = int(set_speed_in_units + 0.5)
        try:
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
        except:
          values["SLA_ICON"] = 1 if (frame % 40) < 20 else 4

        values["DISTANCE"] = 4 if hdp_active else hud_control.leadDistanceBars
        values["DISTANCE_LEAD"] = 2 if cruise_enabled and hud_control.leadVisible else 1 if main_enabled and hud_control.leadVisible else 0
        values["DISTANCE_CAR"] = 3 if hdp_active else 2 if cruise_enabled else 1 if main_enabled else 0
        values["DISTANCE_SPACING"] = 5 if hdp_active else 1 if cruise_enabled else 0

        values["TARGET"] = 1 if hud_control.leadVisible and cruise_enabled else 0
        values["TARGET_DISTANCE"] = int(hud_control.leadDistance)

        values["BACKGROUND"] = 1 if cruise_enabled else 3 if lat_active else 7
        if (left_lane_warning and not CS.out.leftBlinker) or (right_lane_warning and not CS.out.rightBlinker):
          values["BACKGROUND"] = 4
        values["CENTERLINE"] = 1 if HDA_CntrlModSta > 0 or lat_enabled else 0
        values["CAR_CIRCLE"] = 2 if hdp_active or CS.softHoldActive else 1 if cruise_enabled else 0

        # values["NAV_ICON"] = 2 if nav_active and cruise_enabled else 1 if main_enabled and nav_active else 0
        values["HDA_ICON"] = 5 if hdp_active else 2 if cruise_enabled else 1 if main_enabled else 0
        values["LFA_ICON"] = 5 if hdp_active else 2 if lat_active else 1 if lat_enabled else 0
        values["LKA_ICON"] = 4 if lat_active else 3 if lat_enabled else 0
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

        if values["ALERTS_5"] in [11] and CS.softHoldActive == 0:
          values["ALERTS_5"] = 0

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
          lane_color = 6 if md is not None and md.meta.laneChangeAvailableLeft else 2
          if lane_line_check >= 1:
            lane_line_warn_left = CS.out.leftLaneLine % 10 not in (0, 5)
          else:
            lane_line_warn_left = CS.out.leftLaneLine // 10 == 2
          lane_color = 4 if lane_line_warn_left or CS.out.leftBlindspot else lane_color
          if hud_control.leftLaneDepart:
            values["LANELINE_LEFT"] = 4 if (frame // 50) % 2 == 0 else 1
          else:
            values["LANELINE_LEFT"] = lane_color if hud_control.leftLaneVisible else 0

          lane_color = 6 if md is not None and md.meta.laneChangeAvailableRight else 2
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
          values["LANE_HIGHLIGHT_DISTANCE"] = int(ease_in_interp(CS.out.vEgo * CV.MS_TO_KPH, [0, 80], [3, 60], power=1.5))
          values["LANE_HIGHLIGHT"] = create_ccnc_messages.drive_lane_color.update(CS.out.aEgo, drive_mode, CS.out.vEgo)
        elif CS.out.gearShifter == structs.CarState.GearShifter.reverse:
          values["LANE_HIGHLIGHT"] = 5
        elif CS.out.gearShifter == structs.CarState.GearShifter.neutral:
          values["LANE_HIGHLIGHT"] = 4
        elif CS.out.gearShifter == structs.CarState.GearShifter.park:
          if not CS.out.parkingBrake:
            values["LANE_HIGHLIGHT"] = 2

        is_auto_lane_changing = desire in (3, 4)
        is_blinking = CS.out.leftBlinker != CS.out.rightBlinker
        is_currently_lane_changing = is_auto_lane_changing or (is_blinking and CS.out.vEgo > 6)

        # 차선 곡률 표시 (주행 경로의 시작과 끝 y 좌표 차이 이용)
        try:
          if lat_enabled:
            max_lookahead_x = np.interp(CS.out.vEgo * CV.MS_TO_KPH, [20, 100], [30, 80])

            # --- 차량 주행 경로 기반 ---
            # Peak Search: 경로 중 횡방향 변위(절대값)가 가장 큰 지점을 탐색
            trust_threshold = 0.8
            max_y_abs = 0.0
            peak_idx = start_search_idx = 0
            start_found = not is_currently_lane_changing

            min_curvature_calc_distance = 30 if is_currently_lane_changing else 0

            for i in range(1, len(md.position.x)):
              x = md.position.x[i]

              if not start_found and x >= min_curvature_calc_distance:
                start_search_idx = i
                start_found = True

              if md.position.yStd[i] > trust_threshold or x > max_lookahead_x:
                break

              y_abs = abs(md.position.y[i])
              if y_abs > max_y_abs:
                max_y_abs = y_abs
                peak_idx = i

            if start_search_idx != peak_idx and md.position.x[peak_idx] >= (20.0 + min_curvature_calc_distance):
              x_dist = md.position.x[peak_idx]
              y_diff = md.position.y[peak_idx] - md.position.y[start_search_idx]

              # 물리 곡률 공식 (2y / x^2) 적용
              # 상수 2000.0 설명:
              # - 물리적 곡률(Kappa = 1/R)은 보통 0.0001~0.01 사이의 아주 작은 값임.
              # - 이를 ccNC 계기판 표시 범위인 0~15 사이의 직관적인 수치로 증폭하는 Gain 역할.
              # - 시뮬레이션 결과: R=500m(일반코너)에서 약 8단계, R=150m(급코너)에서 약 15단계 수준임.
              # - 튜닝 팁: 계기판 게이지가 너무 민감하게 차오르면 1500으로 낮추고, 너무 둔하면 2500으로 높여 조절.
              max_curve_val = (2.0 * y_diff) / (x_dist ** 2) * 1800
            else:
              max_curve_val = 0.0

            curvature = round(create_ccnc_messages.lane_curv.apply(-max_curve_val)) # 디스플레이 곡률은 음수 반전해야 정상 방향이 나옴
          else:
            # 횡컨 아니면 핸들 각도 기반 조향
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
            if 2 < lane_width < 4.5:
              create_ccnc_messages.last_known_lane_width = lane_width # 마지막 차선 폭을 기억해둠

          values["LANELINE_LEFT_POSITION"] = int(round(np.interp(current_l_target, [0.0, 3.0], [0, 30])))
          values["LANELINE_RIGHT_POSITION"] = int(round(np.interp(current_r_target, [0.0, 3.0], [0, 30])))

          # 차선 변경 아이콘
          if lat_enabled:
            values["LCA_LEFT_ICON"] = 1 if CS.out.leftBlindspot else 4 if CS.out.rightBlinker or not md.meta.laneChangeAvailableLeft else 2
            values["LCA_RIGHT_ICON"] = 1 if CS.out.rightBlindspot else 4 if CS.out.leftBlinker or not md.meta.laneChangeAvailableRight else 2
        except:
          values["LANELINE_LEFT_POSITION"] = 30
          values["LANELINE_RIGHT_POSITION"] = 30
          values["LANE_HIGHLIGHT"] = 1
          values["LANE_HIGHLIGHT_DISTANCE"] = 60
          values["LANE_LEFT"] = 1
          values["LANE_RIGHT"] = 1
          values["LKA_ICON"] = 1

        ret.append(packer.make_can_msg("ADRV_0x161", CAN.ECAN, values, rx_counter = rx_counter))

      if CS.adrv_0x200 is not None:
        values = copy.copy(CS.adrv_0x200)
        rx_counter = values.pop("COUNTER", None)
        values["TauGapSet"] = hud_control.leadDistanceBars
        ret.append(packer.make_can_msg("ADRV_0x200", CAN.ECAN, values, rx_counter = rx_counter))

      if CS.adrv_0x1ea is not None:
        values = copy.copy(CS.adrv_0x1ea)
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
        values = copy.copy(CS.ccnc_0x162)

        # --- radarState를 이용한 전방 차량 감지 ---
        # 2024 쏘나타는 차량 인식 두부(1, 2)만 출력 가능
        try:
          ff_lead = lf_lead = rf_lead = None
          ff_yRel = lf_yRel = rf_yRel = 0
          ff_min_dist = lf_min_dist = rf_min_dist = 1000.0
          min_front_lead_speed = -100 if CS.out.aEgo < -1 else np.interp(CS.out.vEgo * CV.MS_TO_KPH, [30, 40, 100], [-100, 0, 20])
          min_side_lead_speed = np.interp(CS.out.vEgo * CV.MS_TO_KPH, [0, 30, 100], [2, 10, 20])

          # 레이더 정보 갱신
          if CS.radar_state:

            # 상단에서 계산된 계기판 표시용 curvature 변수 활용 (UnboundLocalError 방지)
            current_curvature = create_ccnc_messages.lane_curv.value

            # lane_bound = np.interp(abs(current_curvature), [0, 15], [1.5, 2.5])

            valid_leads = (
              l for l in itertools.chain(CS.radar_state.leadsLeft,
                                        CS.radar_state.leadsRight,
                                        CS.radar_state.leadsCenter)
              if l.dRel > 1 and l.radar
            )

            _selected_lane_line = md.laneLines[1] if md.laneLineProbs[1] > md.laneLineProbs[2] else md.laneLines[2]

            _left_outer_line_prob = md.laneLineProbs[0]
            _right_outer_line_prob = md.laneLineProbs[3]
            _left_outer_line = md.laneLines[0].y[0]
            _right_outer_line = md.laneLines[3].y[0]
            _left_adjacent_lane_exists = _left_outer_line_prob > 0.1 and _left_outer_line - md.laneLines[1].y[0] >= 2
            _right_adjacent_lane_exists = _right_outer_line_prob > 0.1 and _right_outer_line - md.laneLines[2].y[0] <= -2

            _left_detection_bound = _left_outer_line if _left_outer_line_prob > 0.2 else 4.5
            _right_detection_bound = _right_outer_line if _right_outer_line_prob > 0.2 else -4.5

            for lead in valid_leads:
              dRel = lead.dRel
              yRel = lead.yRel

              # 직선 물리 좌표 yRel에서 곡선 오프셋을 빼주어 현재 차선 중앙 기준의 횡방향 거리 산출
              road_aligned_yRel = yRel - (np.interp(dRel, _selected_lane_line.x, _selected_lane_line.y) - _selected_lane_line.y[0]) * 1.5
              dist_score = dRel + abs(road_aligned_yRel)

              # # 1. 상단에서 계산한 curvature(계기판 표시용 곡률)을 횡방향 물리 오프셋으로 역산
              # # 곡률(kappa) = -curvature / 1800.0 (curvature가 음수일 때 좌측 커브)
              # # 오프셋 = 0.5 * kappa * dRel^2 = -curvature * (dRel ** 2) / 3600.0
              # curve_offset_y = -current_curvature * (dRel ** 2) / 4000.0

              # # 2. 직선 물리 좌표 yRel에서 곡률 오프셋을 빼주어 현재 차선 중앙 기준의 횡방향 거리 산출
              # road_aligned_yRel = yRel + curve_offset_y
              # dist_score = dRel + abs(road_aligned_yRel)

              # 전방 차량
              if -1.5 <= road_aligned_yRel <= 1.5: # 전방 좁은 영역
                if dist_score < ff_min_dist and lead.vLead * CV.MS_TO_KPH > min_front_lead_speed:
                  ff_min_dist, ff_lead, ff_yRel = dist_score, lead, road_aligned_yRel * np.interp(dRel, [70, 100], [1.0, 0.6]) # yRel 보간

              # 왼쪽 차선 차량
              elif 1.5 < road_aligned_yRel < _left_detection_bound:
                if dist_score < lf_min_dist and (lead.vLead * CV.MS_TO_KPH > min_side_lead_speed or (_left_adjacent_lane_exists and lead.vLead > -1 and road_aligned_yRel < _left_outer_line)):
                  lf_min_dist, lf_lead, lf_yRel = dist_score, lead, road_aligned_yRel * np.interp(dRel, [70, 100], [1.0, 1.1])

              # 오른쪽 차선 차량
              elif _right_detection_bound < road_aligned_yRel < -1.5:
                if dist_score < rf_min_dist and (lead.vLead * CV.MS_TO_KPH > min_side_lead_speed or (_right_adjacent_lane_exists and lead.vLead > -1 and road_aligned_yRel > _right_outer_line)):
                  rf_min_dist, rf_lead, rf_yRel = dist_score, lead, road_aligned_yRel * np.interp(dRel, [70, 100], [1.0, 1.1])

          # 전방(FF) 차량 정보 업데이트
          if ff_lead:
            values["FF_DISTANCE"] = create_ccnc_messages.ff_distance.apply(ff_lead.dRel) * 0.8
            values["FF_LATERAL"] = create_ccnc_messages.ff_lateral.apply(apply_curved_deadband(-ff_yRel, 0, 0.7, 1))
            values["FF_DETECT"] = 2 if ff_lead.vLead < 3 else create_ccnc_messages.ff_detect.apply(ff_lead.vRel)
          else:
            values["FF_DETECT"] = 0 # 순정 디텍션 제거
          # 전방 좌측(LF) 차량 정보 업데이트
          if lf_lead:
            if lf_lead.dRel < 5.0:
              lf_yRel = max(lf_yRel, 2.5)
            values["LF_DETECT_DISTANCE"] = create_ccnc_messages.lf_distance.apply(lf_lead.dRel) * 0.8
            values["LF_DETECT_LATERAL"] = create_ccnc_messages.lf_lateral.apply(apply_curved_deadband(min(4, lf_yRel), 3, 0.9, 2))
            values["LF_DETECT"] = create_ccnc_messages.lf_detect.apply(lf_lead.vRel)
          # 전방 우측(RF) 차량 정보 업데이트
          if rf_lead:
            if rf_lead.dRel < 5.0:
              rf_yRel = min(rf_yRel, -2.5)
            values["RF_DETECT_DISTANCE"] = create_ccnc_messages.rf_distance.apply(rf_lead.dRel) * 0.8
            values["RF_DETECT_LATERAL"] = create_ccnc_messages.rf_lateral.apply(apply_curved_deadband(max(-4, -rf_yRel), 3, 0.9, 2))
            values["RF_DETECT"] = create_ccnc_messages.rf_detect.apply(rf_lead.vRel)

          center_lane_offset = (create_ccnc_messages.r_lane_f.value - create_ccnc_messages.l_lane_f.value) / 2

          # --- 후측방은 BSD 경고 시 고정 위치에 두부 출력. HDA1은 후측방 레이더 정보가 안채워져서 옴 ---
          BSD_LATERAL_FIXED = 2.8
          if CS.out.leftBlindspot:
            values["LR_DETECT_DISTANCE"] = create_ccnc_messages.lr_distance.apply(8)
            values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
            values["LR_DETECT"] = 2
          elif create_ccnc_messages.lr_distance.value < 15:
            values["LR_DETECT_DISTANCE"] = create_ccnc_messages.lr_distance.apply(16)
            values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
            values["LR_DETECT"] = 1

          if CS.out.rightBlindspot:
            values["RR_DETECT_DISTANCE"] = create_ccnc_messages.rr_distance.apply(8)
            values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
            values["RR_DETECT"] = 2
          elif create_ccnc_messages.rr_distance.value < 15:
            values["RR_DETECT_DISTANCE"] = create_ccnc_messages.rr_distance.apply(16)
            values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
            values["RR_DETECT"] = 1

        except:
          values = copy.copy(CS.ccnc_0x162)
          values["FF_DISTANCE"] = 24
          values["FF_DETECT"] = 2
          values["LF_DETECT_DISTANCE"] = 12
          values["LF_DETECT_LATERAL"] = 1.5
          values["LF_DETECT"] = 1
          values["RF_DETECT_DISTANCE"] = 12
          values["RF_DETECT_LATERAL"] = 1.5
          values["RF_DETECT"] = 1
          values["LR_DETECT_DISTANCE"] = 1
          values["LR_DETECT_LATERAL"] = 3
          values["LR_DETECT"] = 2
          values["RR_DETECT_DISTANCE"] = 1
          values["RR_DETECT_LATERAL"] = 3
          values["RR_DETECT"] = 2

        if (left_lane_warning and not CS.out.leftBlinker) or (right_lane_warning and not CS.out.rightBlinker):
          values["VIBRATE"] = 1

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
          values = copy.copy(CS.hda_info_4a3)
          values["LinkClass"] = 1
          values["SPEED_LIMIT"] = 100
          ret.append(packer.make_can_msg("HDA_INFO_4A3", CAN.CAM, values))

  return ret

# 곡률 노이즈 필터
create_ccnc_messages.lane_curv = NoiseFilter(3, 0, alpha_range=0.5)

# 차선 변경 상태 플래그
create_ccnc_messages._is_lane_change_active = False

# 차선 넘어감 감지
create_ccnc_messages.draw_center = False
create_ccnc_messages.hold_lane_escape_count = 0
create_ccnc_messages.lane_phase_min = 10.0

# 차선 노이즈 필터
create_ccnc_messages.last_known_lane_width = 3.0
create_ccnc_messages.l_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2)
create_ccnc_messages.r_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2)

# 차량 거리 필터
create_ccnc_messages.ff_distance = NoiseFilter(3, 0, alpha_range=[0.3, 0.9], error_range=[1.0, 4.0])
create_ccnc_messages.lf_distance = NoiseFilter(3, 0, alpha_range=[0.3, 0.9], error_range=[1.0, 4.0])
create_ccnc_messages.rf_distance = NoiseFilter(3, 0, alpha_range=[0.3, 0.9], error_range=[1.0, 4.0])
create_ccnc_messages.ff_lateral = NoiseFilter(3, 0, alpha_range=0.3, error_range=0.6)
create_ccnc_messages.lf_lateral = NoiseFilter(3, 3, alpha_range=0.3, error_range=0.6)
create_ccnc_messages.rf_lateral = NoiseFilter(3, 3, alpha_range=0.3, error_range=0.6)
create_ccnc_messages.ff_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))
create_ccnc_messages.lf_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))
create_ccnc_messages.rf_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))

create_ccnc_messages.lr_distance = NoiseFilter(1, 15, alpha_range=0.05)
create_ccnc_messages.rr_distance = NoiseFilter(1, 15, alpha_range=0.05)

create_ccnc_messages.drive_lane_color = LaneHighlightStateMachine()

create_ccnc_messages.sla_active_time = 0
