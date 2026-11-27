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
from openpilot.selfdrive.controls.lib.limousine_launch import LimousineLaunchController
from openpilot.selfdrive.controls.lib.limousine_stop import (
  EMERGENCY_DECEL_THRESHOLD,
  V_MIN_ACTIVE,
  LimousineStopController,
)

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
E2E_STOP_XSTATES = (3, 5)  # XState.e2eStop, XState.e2eStopped


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
    self._raw_stopping_accel = 0.0
    self._was_in_stopping = False
    self.coasting = CruiseCoastingControl()
    self.limousine_stop = LimousineStopController()
    self.limousine_launch = LimousineLaunchController()
    self.road_pitch = 0.0


    self.params = Params()
    self.readParamCount = 0
    self._refresh_stopping_accel()
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
    self.limousine_stop.reset()
    self.limousine_launch.reset()
    self._raw_stopping_accel = 0.0
    self._was_in_stopping = False

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
    elif self.readParamCount == 10:
      self._refresh_longitudinal_tuning()


    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill, CS.aEgo, self.stopping_accel, radarState)
    if active and soft_hold_active:
      self.long_control_state = LongCtrlState.stopping

    if self.long_control_state != LongCtrlState.stopping:
      self._was_in_stopping = False

    fcw = bool(getattr(long_plan, 'fcw', False))
    a_target_finite = math.isfinite(a_target_ff)

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      if not self._was_in_stopping:
        self._raw_stopping_accel = self.last_output_accel
        self._was_in_stopping = True

      if soft_hold_active:
        self._raw_stopping_accel = self.CP.stopAccel
      # Preserve unsoftened one-way stopping ramp separately from limousine_stop output
      # so a sudden safety override or emergency command restores full braking on tick 0.
      elif self._raw_stopping_accel > self.stopping_accel:
        self._raw_stopping_accel = min(self._raw_stopping_accel, 0.0) - self.CP.stoppingDecelRate * DT_CTRL

      emergency_stop_req = fcw or (a_target_finite and a_target_ff < EMERGENCY_DECEL_THRESHOLD)
      if CS.vEgo <= V_MIN_ACTIVE and not emergency_stop_req and not soft_hold_active and self.limousine_stop.active:
        # Normal low-speed handover into standstill: avoid snapping back to pre-relief decel.
        self._raw_stopping_accel = min(self.stopping_accel, self.last_output_accel)

      output_accel = self._raw_stopping_accel
      if emergency_stop_req and a_target_finite:
        output_accel = min(output_accel, float(a_target_ff))

      self.pid.reset()
      self.coasting.reset()
      self.limousine_launch.notify_stopping()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.pid.reset()
      self.coasting.reset()
      self.limousine_stop.reset()

    else:  # LongCtrlState.pid
      target = getattr(long_plan, 'cruiseCoastingTarget', 0.0)
      percent = getattr(long_plan, 'cruiseCoastingPercent', 0)
      relief = 0.0
      if (target > 0.0 and percent > 0 and self.CP.openpilotLongitudinalControl and
          0.0 <= t_since_plan <= MAX_PLAN_AGE and
          long_plan.longitudinalPlanSource == 'cruise' and not long_plan.fcw and not should_stop and
          not CS.brakePressed and not CS.gasPressed and not CS.carrotCruise and
          abs(long_plan.cruiseTarget - CS.vCruise) < 0.001 and
          not CS.cruiseState.standstill and no_coasting_lead(radarState) and
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
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff)
      if relief > 0.0 and output_accel < 0.0:
        # Preserve the ordinary PID exactly for positive commands and at 0%.
        # Do not integrate a speed error whose braking output we are suppressing.
        self.pid.i = previous_integral
        output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff, freeze_integrator=True)
        output_accel = self.coasting.apply(output_accel, relief, DT_CTRL)
      else:
        self.coasting.reset()

    if self.long_control_state in (LongCtrlState.pid, LongCtrlState.stopping):
      lead_one = getattr(radarState, 'leadOne', None) if radarState is not None else None
      has_lead = bool(lead_one is not None and lead_one.status)
      is_e2e_stop = int(getattr(long_plan, 'xState', 0)) in E2E_STOP_XSTATES
      plan_stop_dist = (
        None if has_lead
        else estimate_plan_stop_distance(getattr(long_plan, 'speeds', None), getattr(long_plan, 'accels', None))
      )
      output_accel = self.limousine_stop.update(
        a_cmd=float(output_accel),
        previous_accel=float(self.last_output_accel),
        v_ego=float(CS.vEgo),
        dt=DT_CTRL,
        enabled=(active and not soft_hold_active and not getattr(CS, 'brakePressed', False)
                 and not getattr(CS, 'gasPressed', False) and not CS.cruiseState.standstill),
        should_stop=bool(should_stop),
        in_stopping_state=(self.long_control_state == LongCtrlState.stopping),
        fcw=fcw,
        v_target_now=float(v_target_now),
        has_lead=has_lead,
        d_rel=float(getattr(lead_one, 'dRel', 100.0)) if has_lead else 100.0,
        v_lead=float(getattr(lead_one, 'vLead', 0.0)) if has_lead else 0.0,
        pitch_rad=float(self.road_pitch),
        is_e2e_stop=is_e2e_stop,
        stop_dist=plan_stop_dist,
      )
      if self.long_control_state == LongCtrlState.stopping and self.limousine_stop.safety_override and a_target_finite:
        output_accel = min(output_accel, float(a_target_ff))
    else:
      self.limousine_stop.reset()

    if self.long_control_state in (LongCtrlState.starting, LongCtrlState.pid):
      output_accel = self.limousine_launch.update(
        a_cmd=float(output_accel),
        previous_accel=float(self.last_output_accel),
        v_ego=float(CS.vEgo),
        dt=DT_CTRL,
        enabled=(active and not soft_hold_active and not getattr(CS, 'brakePressed', False)
                 and not getattr(CS, 'gasPressed', False)),
        should_stop=bool(should_stop),
        fcw=fcw,
        cruise_standstill=bool(CS.cruiseState.standstill),
        in_starting_state=(self.long_control_state == LongCtrlState.starting),
        pitch_rad=float(self.road_pitch),
      )
      if self.limousine_launch.active and self.limousine_launch.shaped_jerk is not None:
        aligned_jerk = self.limousine_launch.shaped_jerk * 0.5
        if self.limousine_launch.tracking_upstream:
          j_target_now = aligned_jerk
        elif j_target_now > 0.0:
          j_target_now = min(float(j_target_now), aligned_jerk)
    elif self.long_control_state != LongCtrlState.stopping:
      self.limousine_launch.reset()

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel, a_target_ff, j_target_now
