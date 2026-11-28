import math
import numpy as np
from openpilot.cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params
from opendbc.car.hyundai.values import is_ray_pedal_control
from openpilot.selfdrive.controls.lib.cruise_coasting import (
  CruiseCoastingControl, MAX_PLAN_AGE, coasting_relief, no_coasting_lead,
)
from openpilot.selfdrive.controls.lib.unified_long import (
  DEFAULT_STOP_DISTANCE,
  MIN_SAFE_STOP_DISTANCE,
  UnifiedLongController,
  UnifiedLongInput,
)

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
E2E_STOP_XSTATES = (3, 5)  # XState.e2eStop, XState.e2eStopped
STOP_DISTANCE_MAX = 10.0


def valid_road_pitch(calibrated_pose, calib_valid: bool, live_pose_valid: bool) -> float:
  """Return finite calibrated road pitch (rad) only when calibration and livePose are valid."""
  if calibrated_pose is None or not calib_valid or not live_pose_valid:
    return 0.0
  pitch = float(calibrated_pose.orientation.xyz[1])
  return pitch if math.isfinite(pitch) else 0.0


def estimate_plan_stop_distance(speeds, accels=None, t_idxs=CONTROL_N_T_IDX) -> float | None:
  """Integrate planned speed trajectory to estimate remaining distance to a solo stop line."""
  if speeds is None or len(speeds) < 2:
    return None
  n = min(len(speeds), len(t_idxs))
  dist = 0.0
  reached_stop = False
  for i in range(n - 1):
    v0 = float(speeds[i])
    v1 = float(speeds[i + 1])
    if not (math.isfinite(v0) and math.isfinite(v1)):
      return None
    dt = float(t_idxs[i + 1] - t_idxs[i])
    if v0 <= 0.05:
      reached_stop = True
      break
    if v1 <= 0.05:
      frac = max(0.0, min(1.0, (v0 - 0.05) / max(1e-6, v0 - v1)))
      dist += 0.5 * (v0 + 0.05) * dt * frac
      reached_stop = True
      break
    dist += 0.5 * (v0 + v1) * dt
  if not reached_stop:
    v_end = float(speeds[n - 1])
    if not math.isfinite(v_end):
      return None
    if v_end <= 0.05:
      reached_stop = True
    elif accels is not None and len(accels) >= n:
      a_end = float(accels[n - 1])
      if math.isfinite(a_end) and a_end < -0.1:
        dist += (v_end * v_end) / (2.0 * abs(a_end))
        reached_stop = True
  return dist if (reached_stop and math.isfinite(dist)) else None

HYUNDAI_LONGITUDINAL_KP = 1.0
HYUNDAI_LONGITUDINAL_KI = 0.0
HYUNDAI_LONGITUDINAL_KF = 1.0
STOPPING_ACCEL_DEFAULT = -50  # Params use hundredths of m/s^2
STOPPING_ACCEL_MIN = -100
STOPPING_ACCEL_MAX = -50

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState):
  stopping_condition = should_stop
  stopping_accel = stopping_accel if stopping_accel < 0.0 else -0.5
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        leadOne = radarState.leadOne
        fcw_stop = leadOne.status and leadOne.dRel < 4.0
        if a_ego > stopping_accel or fcw_stop:
          long_control_state = LongCtrlState.stopping
        elif long_control_state == LongCtrlState.starting:
          long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.last_shaped_jerk_u = None
    self.coasting = CruiseCoastingControl()
    self.unified = UnifiedLongController()
    self.road_pitch = 0.0


    self.params = Params()
    self.readParamCount = 0
    self._refresh_stopping_accel()
    self._refresh_stop_distance()
    self.j_lead = 0.0

    ray_pedal_control = is_ray_pedal_control(CP)
    self.hyundai_fixed_longitudinal_tuning = CP.brand == "hyundai" and not ray_pedal_control
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()
    elif ray_pedal_control:
      self._refresh_longitudinal_tuning()
    elif CP.brand == "toyota":
      self._apply_toyota_longitudinal_tuning()

    self.use_accel_pid = False
    if CP.brand == "toyota":
      self.use_accel_pid = True

  def _refresh_stopping_accel(self):
    try:
      value = float(self.params.get_float("StoppingAccel"))
    except (TypeError, ValueError):
      value = STOPPING_ACCEL_DEFAULT
    if not math.isfinite(value):
      value = STOPPING_ACCEL_DEFAULT
    # Enforce the menu bounds even for stale Params or direct writes.
    self.stopping_accel = min(STOPPING_ACCEL_MAX, max(STOPPING_ACCEL_MIN, value)) * 0.01

  def _refresh_stop_distance(self):
    try:
      value = float(self.params.get_float("StopDistanceCarrot")) * 0.01
    except (TypeError, ValueError):
      value = DEFAULT_STOP_DISTANCE
    if not math.isfinite(value) or value <= 0.0:
      value = DEFAULT_STOP_DISTANCE
    self.stop_distance = min(STOP_DISTANCE_MAX, max(MIN_SAFE_STOP_DISTANCE, value))

  def _apply_hyundai_longitudinal_tuning(self):
    # Hyundai, Kia, and Genesis all use the opendbc "hyundai" brand. Keep the
    # complete acceleration/deceleration feedforward path intact instead of
    # allowing a stale or unsafe persistent tuning value to override it.
    self.pid._k_p = ([0.0], [HYUNDAI_LONGITUDINAL_KP])
    self.pid._k_i = ([0.0], [HYUNDAI_LONGITUDINAL_KI])
    self.pid.k_f = HYUNDAI_LONGITUDINAL_KF

  def _refresh_longitudinal_tuning(self):
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()
    elif self.CP.brand == "toyota":
      self._apply_toyota_longitudinal_tuning()
    elif len(self.CP.longitudinalTuning.kpBP) == 1 and len(self.CP.longitudinalTuning.kiBP) == 1:
      longitudinalTuningKpV = self.params.get_float("LongTuningKpV") * 0.01
      longitudinalTuningKiV = self.params.get_float("LongTuningKiV") * 0.001
      self.pid._k_p = (self.CP.longitudinalTuning.kpBP, [longitudinalTuningKpV])
      self.pid._k_i = (self.CP.longitudinalTuning.kiBP, [longitudinalTuningKiV])
      self.pid.k_f = self.params.get_float("LongTuningKf") * 0.01

  def _apply_toyota_longitudinal_tuning(self):
    # Toyota's CarController closes the acceleration loop at the PCM send rate.
    # Pass the planner target through; a second acceleration PID here feeds
    # measured-acceleration noise and duplicate correction into that loop.
    self.pid._k_p = ([0.0], [0.0])
    self.pid._k_i = ([0.0], [0.0])
    self.pid.k_f = 1.0

  def reset(self):
    self.pid.reset()
    self.coasting.reset()
    self.unified.reset()
    self.last_shaped_jerk_u = None

  def update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):
    soft_hold_active = CS.softHoldActive > 0
    a_target_ff = long_plan.aTarget
    v_target_now = long_plan.vTargetNow
    j_target_now = long_plan.jTargetNow
    should_stop = long_plan.shouldStop

    self.readParamCount += 1
    if self.readParamCount >= 100:
      self.readParamCount = 0
      self._refresh_stopping_accel()
      self._refresh_stop_distance()
    elif self.readParamCount == 10:
      self._refresh_longitudinal_tuning()

    """Update longitudinal control via the unified continuous controller."""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    if not active:
      self.reset()
      self.long_control_state = LongCtrlState.off
      self.last_output_accel = 0.0
      return 0.0, a_target_ff, j_target_now

    # Sync external state if a caller/test seeded last_output_accel or long_control_state directly.
    if (not self.unified.state.initialized
        or abs(float(self.last_output_accel) - self.unified.state.a_ref) > 1e-6
        or (self.long_control_state == LongCtrlState.stopping and not self.unified.state.hold_latched)):
      self.unified.sync_external_state(
        a_ref=float(self.last_output_accel),
        in_stopping=(self.long_control_state == LongCtrlState.stopping),
      )

    fcw = bool(getattr(long_plan, 'fcw', False))
    brake_pressed = bool(getattr(CS, 'brakePressed', False))
    gas_pressed = bool(getattr(CS, 'gasPressed', False))
    cruise_standstill = bool(CS.cruiseState.standstill)

    # 1. Continuous closed-loop desired acceleration (PID + CruiseCoasting)
    if soft_hold_active:
      a_des = float(self.CP.stopAccel)
      self.pid.reset()
      self.coasting.reset()
    elif self.unified.state.hold_latched and should_stop:
      # In stopping hold, track planner feedforward directly without speed-error P windup.
      a_des = float(a_target_ff) if math.isfinite(a_target_ff) else float(self.stopping_accel)
      self.pid.reset()
      self.coasting.reset()
    else:
      target = getattr(long_plan, 'cruiseCoastingTarget', 0.0)
      percent = getattr(long_plan, 'cruiseCoastingPercent', 0)
      relief = 0.0
      if (target > 0.0 and percent > 0 and self.CP.openpilotLongitudinalControl and
          0.0 <= t_since_plan <= MAX_PLAN_AGE and
          long_plan.longitudinalPlanSource == 'cruise' and not fcw and not should_stop and
          not brake_pressed and not gas_pressed and not CS.carrotCruise and
          abs(long_plan.cruiseTarget - CS.vCruise) < 0.001 and
          not cruise_standstill and no_coasting_lead(radarState) and
          math.isfinite(CS.aEgo) and math.isfinite(a_target_ff) and math.isfinite(v_target_now) and
          accel_limits[1] >= 0.0):
        relief = coasting_relief(CS.vEgo, target, percent)
      if relief == 0.0:
        self.coasting.reset()
      if self.use_accel_pid:
        error = a_target_ff - CS.aEgo
      else:
        error = v_target_now - CS.vEgo
      previous_integral = self.pid.i
      a_des = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff,
                              freeze_integrator=self.unified.state.envelope_limited)
      if relief > 0.0 and a_des < 0.0:
        # Preserve the ordinary PID exactly for positive commands and at 0%.
        # Do not integrate a speed error whose braking output we are suppressing.
        self.pid.i = previous_integral
        a_des = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff, freeze_integrator=True)
        a_des = self.coasting.apply(a_des, relief, DT_CTRL)
      else:
        self.coasting.reset()

    lead_one = getattr(radarState, 'leadOne', None) if radarState is not None else None
    has_lead = bool(lead_one is not None and lead_one.status)
    is_e2e_stop = int(getattr(long_plan, 'xState', 0)) in E2E_STOP_XSTATES
    plan_stop_dist = (
      None if has_lead
      else estimate_plan_stop_distance(getattr(long_plan, 'speeds', None), getattr(long_plan, 'accels', None))
    )

    out = self.unified.update(UnifiedLongInput(
      active=bool(active),
      dt=DT_CTRL,
      v_ego=float(CS.vEgo),
      a_ego=float(CS.aEgo),
      a_des=float(a_des),
      a_target=float(a_target_ff),
      v_target_now=float(v_target_now),
      j_target_now=float(j_target_now),
      accel_min=float(accel_limits[0]),
      accel_max=float(accel_limits[1]),
      should_stop=bool(should_stop),
      soft_hold=bool(soft_hold_active),
      brake_pressed=brake_pressed,
      gas_pressed=gas_pressed,
      cruise_standstill=cruise_standstill,
      fcw=fcw,
      has_lead=has_lead,
      d_rel=float(getattr(lead_one, 'dRel', 100.0)) if has_lead else 100.0,
      v_lead=float(getattr(lead_one, 'vLead', 0.0)) if has_lead else 0.0,
      pitch_rad=float(self.road_pitch),
      is_e2e_stop=is_e2e_stop,
      plan_stop_dist=plan_stop_dist,
      stopping_accel_cfg=float(self.stopping_accel),
      cp_stop_accel=float(self.CP.stopAccel),
      cp_stopping_decel_rate=float(self.CP.stoppingDecelRate),
      cp_starting_state=bool(self.CP.startingState),
      cp_v_starting=float(self.CP.vEgoStarting),
      cp_start_accel=float(getattr(self.CP, 'startAccel', 0.0)),
      d_stop_cfg=float(self.stop_distance),
    ))

    if out.mode == "stopping":
      self.long_control_state = LongCtrlState.stopping
      self.pid.reset()
      self.coasting.reset()
    elif out.mode == "starting":
      self.long_control_state = LongCtrlState.starting
      self.pid.reset()
      self.coasting.reset()
    elif out.mode == "off":
      self.long_control_state = LongCtrlState.off
      self.pid.reset()
      self.coasting.reset()
    else:
      self.long_control_state = LongCtrlState.pid

    self.last_shaped_jerk_u = out.j_u_shaped
    self.last_output_accel = float(np.clip(out.a_cmd, accel_limits[0], accel_limits[1]))
    return self.last_output_accel, a_target_ff, out.j_cmd
