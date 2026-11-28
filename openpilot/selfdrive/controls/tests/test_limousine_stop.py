"""Unit tests for LimousineStopController and its integration in LongControl."""
from types import SimpleNamespace

import pytest

from openpilot.cereal import car, log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.limousine_stop import (
  BRAKE_LAMP_HOLD_ACCEL,
  DEFAULT_STOP_DISTANCE,
  EMERGENCY_DECEL_THRESHOLD,
  GRAVITY,
  MAX_APPROACH_DECEL,
  MAX_RELEASE_JERK,
  MAX_UPHILL_SLOPE_COMP,
  MIN_SAFE_STOP_DISTANCE,
  SAFETY_OVERRIDE_MARGIN,
  STOP_DISTANCE_TOLERANCE,
  V_EASE_END,
  V_EASE_START,
  LimousineStopController,
  brake_lamp_hold_floor,
  limousine_decel_envelope,
)
from openpilot.selfdrive.controls.lib.longcontrol import (
  CONTROL_N_T_IDX,
  LongControl,
  LongCtrlState,
  estimate_plan_stop_distance,
  valid_road_pitch,
)


def test_envelope_is_monotonic_and_bounded():
  speeds = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7, 0.3, 0.0]
  vals = [limousine_decel_envelope(v) for v in speeds]
  assert vals[0] == pytest.approx(-MAX_APPROACH_DECEL)
  assert vals[-1] == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)
  assert limousine_decel_envelope(V_EASE_END) == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)
  # As speed decreases, envelope becomes monotonically less negative (closer to BRAKE_LAMP_HOLD_ACCEL).
  for prev, nxt in zip(vals[:-1], vals[1:]):
    assert nxt >= prev - 1e-9
    assert nxt <= BRAKE_LAMP_HOLD_ACCEL + 1e-9


def test_curve_smoothness_respects_release_jerk_cap():
  ctrl = LimousineStopController()
  prev = -1.6
  outputs = []
  # Simulate decelerating from 2.5 m/s down to 0.7 m/s with constant strong upstream braking.
  v = 2.5
  for _ in range(250):
    out = ctrl.update(
      a_cmd=-1.6,
      previous_accel=prev,
      v_ego=v,
      dt=DT_CTRL,
      enabled=True,
      should_stop=True,
      in_stopping_state=True,
      has_lead=True,
      d_rel=8.0,
      v_lead=0.0,
    )
    assert out <= BRAKE_LAMP_HOLD_ACCEL + 1e-9
    assert out >= -1.6 - 1e-9
    jerk = (out - prev) / DT_CTRL
    assert jerk <= MAX_RELEASE_JERK + 1e-6
    outputs.append(out)
    prev = out
    v = max(V_EASE_END, v + out * DT_CTRL)

  # Should smoothly reach the brake-lamp hold floor by V_EASE_END.
  assert ctrl.active is True
  assert outputs[-1] == pytest.approx(BRAKE_LAMP_HOLD_ACCEL, abs=0.02)


def test_brake_lamp_floor_never_violated():
  ctrl = LimousineStopController()
  # Even at very low speed (0.7 m/s) with abundant distance, never relax weaker than -0.42 m/s^2.
  prev = -1.2
  for _ in range(200):
    prev = ctrl.update(
      a_cmd=-1.2,
      previous_accel=prev,
      v_ego=0.7,
      dt=DT_CTRL,
      enabled=True,
      should_stop=True,
      in_stopping_state=True,
      has_lead=True,
      d_rel=12.0,
      v_lead=0.0,
    )
    assert prev <= BRAKE_LAMP_HOLD_ACCEL + 1e-9
  assert prev == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)


def test_non_increasing_peak_principle():
  ctrl = LimousineStopController()
  # Already gentle braking (-0.25) must NOT be strengthened to -0.42.
  out = ctrl.update(
    a_cmd=-0.25,
    previous_accel=-0.25,
    v_ego=1.2,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
  )
  assert out == pytest.approx(-0.25)
  assert ctrl.active is False

  # Positive acceleration (departure) must NEVER be altered.
  out_pos = ctrl.update(
    a_cmd=0.8,
    previous_accel=0.0,
    v_ego=1.0,
    dt=DT_CTRL,
    enabled=True,
    should_stop=False,
    in_stopping_state=False,
  )
  assert out_pos == pytest.approx(0.8)


def test_stop_distance_tolerance_allows_gentle_stop_within_margin():
  ctrl = LimousineStopController()
  # Nominal stop_distance=4.0m, tolerance=1.0m -> tol_stop_dist=3.0m.
  # At v=1.0 m/s and d_rel=4.0m (already at nominal stop_distance, 1.0m ahead of tol_stop_dist):
  # a_req_tol = -1.0^2 / (2 * 1.0) = -0.50 m/s^2, so strong -1.5 m/s^2 should relax toward -0.50 m/s^2!
  prev = -1.5
  for _ in range(150):
    prev = ctrl.update(
      a_cmd=-1.5,
      previous_accel=prev,
      v_ego=1.0,
      dt=DT_CTRL,
      enabled=True,
      should_stop=True,
      in_stopping_state=True,
      has_lead=True,
      d_rel=DEFAULT_STOP_DISTANCE,
      v_lead=0.0,
      stop_distance=DEFAULT_STOP_DISTANCE,
    )
  assert ctrl.active is True
  assert ctrl.safety_override is False
  assert prev == pytest.approx(-0.50, abs=0.02)


def test_collision_safety_floor_forces_full_braking_immediately():
  ctrl = LimousineStopController()
  # First enter limousine relief safely.
  out = ctrl.update(
    a_cmd=-1.4,
    previous_accel=-1.4,
    v_ego=1.5,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    has_lead=True,
    d_rel=7.0,
    v_lead=0.0,
  )
  assert ctrl.active is True
  assert out > -1.4

  # Case 1: d_rel breaches MIN_SAFE_STOP_DISTANCE + SAFETY_OVERRIDE_MARGIN -> immediate full braking!
  out_close = ctrl.update(
    a_cmd=-1.8,
    previous_accel=out,
    v_ego=1.2,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    has_lead=True,
    d_rel=MIN_SAFE_STOP_DISTANCE + SAFETY_OVERRIDE_MARGIN - 0.05,
    v_lead=0.0,
  )
  assert out_close == pytest.approx(-1.8)
  assert ctrl.safety_override is True
  assert ctrl.active is False

  # Case 2: high closing speed where a_req_hard <= a_cmd -> immediate full braking!
  # d_rel = 3.8m -> d_hard = 1.3m; v=2.3 m/s -> a_req_hard = -2.03 <= -1.8
  out_fast = ctrl.update(
    a_cmd=-1.8,
    previous_accel=-0.8,
    v_ego=2.3,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    has_lead=True,
    d_rel=3.8,
    v_lead=0.0,
  )
  assert out_fast == pytest.approx(-1.8)
  assert ctrl.safety_override is True

  # Case 3: FCW active -> immediate full braking!
  out_fcw = ctrl.update(
    a_cmd=-1.5,
    previous_accel=-0.8,
    v_ego=1.5,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    fcw=True,
    has_lead=True,
    d_rel=8.0,
    v_lead=0.0,
  )
  assert out_fcw == pytest.approx(-1.5)
  assert ctrl.active is False

  # Case 4: Emergency braking (< EMERGENCY_DECEL_THRESHOLD) -> untouched!
  out_emg = ctrl.update(
    a_cmd=EMERGENCY_DECEL_THRESHOLD - 0.2,
    previous_accel=-1.0,
    v_ego=2.0,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    has_lead=True,
    d_rel=10.0,
    v_lead=0.0,
  )
  assert out_emg == pytest.approx(EMERGENCY_DECEL_THRESHOLD - 0.2)
  assert ctrl.safety_override is True


def test_longcontrol_integration_unwinds_stopping_accel_safely():
  CP = car.CarParams.new_message(
    brand="hyundai",
    openpilotLongitudinalControl=True,
    stoppingDecelRate=0.8,
    stopAccel=-2.0,
    startAccel=0.0,
    vEgoStarting=0.5,
  )
  loc = LongControl(CP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -1.4

  cs = SimpleNamespace(
    vEgo=1.2,
    aEgo=-1.2,
    brakePressed=False,
    gasPressed=False,
    softHoldActive=0,
    cruiseState=SimpleNamespace(standstill=False),
  )
  plan = SimpleNamespace(
    aTarget=-1.4,
    vTargetNow=0.2,
    jTargetNow=0.0,
    shouldStop=True,
    fcw=False,
    longitudinalPlanSource="lead0",
    cruiseTarget=30.0,
    cruiseCoastingTarget=0.0,
    cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=6.5, vLead=0.0),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )

  # Over 1 second (100 ticks), strong stopping accel (-1.4) should smoothly unwind toward -0.54..-0.42.
  prev = -1.4
  for _ in range(100):
    accel, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
    assert accel >= prev - 1e-6
    assert accel <= BRAKE_LAMP_HOLD_ACCEL + 1e-9
    prev = accel
  assert prev > -0.75
  assert loc.unified.state.stop_relief_active is True


def test_uphill_slope_compensation_preserves_hydraulic_brake_floor():
  import math
  # Flat ground and downhill keep the flat brake-lamp floor (-0.42 m/s^2).
  assert brake_lamp_hold_floor(0.0) == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)
  assert brake_lamp_hold_floor(-0.05) == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)
  assert brake_lamp_hold_floor(float("nan")) == pytest.approx(BRAKE_LAMP_HOLD_ACCEL)

  # +4.6% uphill (480-7 observed grade): g*sin(pitch) ~= 0.45 m/s^2 -> floor ~= -0.87 m/s^2.
  pitch_480 = math.atan(0.046)
  expected_floor = BRAKE_LAMP_HOLD_ACCEL - GRAVITY * math.sin(pitch_480)
  assert brake_lamp_hold_floor(pitch_480) == pytest.approx(expected_floor, abs=1e-4)

  # Steep uphill caps compensation at MAX_UPHILL_SLOPE_COMP.
  assert brake_lamp_hold_floor(math.atan(0.20)) == pytest.approx(BRAKE_LAMP_HOLD_ACCEL - MAX_UPHILL_SLOPE_COMP)

  # On +4.6% uphill with gentle upstream braking (-0.77 m/s^2, hydraulic ~-0.32 m/s^2):
  # must NOT relax hydraulic pressure to zero; returns a_cmd untouched.
  ctrl = LimousineStopController()
  out_gentle = ctrl.update(
    a_cmd=-0.77,
    previous_accel=-0.77,
    v_ego=1.0,
    dt=DT_CTRL,
    enabled=True,
    should_stop=True,
    in_stopping_state=True,
    has_lead=True,
    d_rel=5.0,
    v_lead=0.0,
    pitch_rad=pitch_480,
  )
  assert out_gentle == pytest.approx(-0.77)
  assert ctrl.active is False

  # On +4.6% uphill with heavy braking (-1.6 m/s^2, hydraulic ~-1.15 m/s^2):
  # smoothly eases toward slope-compensated floor (-0.87 m/s^2), keeping hydraulic <= -0.42 m/s^2.
  prev = -1.6
  for _ in range(150):
    prev = ctrl.update(
      a_cmd=-1.6,
      previous_accel=prev,
      v_ego=0.8,
      dt=DT_CTRL,
      enabled=True,
      should_stop=True,
      in_stopping_state=True,
      has_lead=True,
      d_rel=8.0,
      v_lead=0.0,
      pitch_rad=pitch_480,
    )
    assert prev <= expected_floor + 1e-9
    assert prev >= -1.6 - 1e-9
  assert prev == pytest.approx(expected_floor, abs=0.02)
  assert ctrl.active is True


def test_raw_stopping_accel_preserved_for_emergency_and_safety_override():
  CP = car.CarParams.new_message(
    brand="hyundai",
    openpilotLongitudinalControl=True,
    stoppingDecelRate=0.8,
    stopAccel=-2.0,
    startAccel=0.0,
    vEgoStarting=0.5,
  )
  loc = LongControl(CP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -1.4

  cs = SimpleNamespace(
    vEgo=1.5, aEgo=-1.2, brakePressed=False, gasPressed=False,
    softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
  )
  plan = SimpleNamespace(
    aTarget=-1.4, vTargetNow=0.2, jTargetNow=0.0, shouldStop=True, fcw=False,
    longitudinalPlanSource="lead0", cruiseTarget=30.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=7.0, vLead=0.0),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )

  # Unwind for 60 ticks: output relaxes smoothly via unified controller.
  for _ in range(60):
    accel, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert loc.unified.state.stop_relief_active is True
  assert accel > -0.80

  # Sudden close lead (d_rel <= 2.8m) with stronger planner demand (-2.2): 0-tick full restoration!
  radar.leadOne.dRel = 2.6
  plan.aTarget = -2.2
  accel_override, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert loc.unified.state.safety_override is True
  assert accel_override == pytest.approx(-2.2)

  # Emergency FCW / < EMERGENCY_DECEL_THRESHOLD: 0-tick restoration to -3.0.
  radar.leadOne.dRel = 6.0
  plan.fcw = True
  plan.aTarget = -3.0
  accel_emg, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert accel_emg == pytest.approx(-3.0)

  # Normal low-speed handover into V_MIN_ACTIVE without override: no snap-back to -1.4.
  loc.reset()
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -1.4
  plan.fcw = False
  plan.aTarget = -1.4
  radar.leadOne.dRel = 7.0
  cs.vEgo = 0.75
  for _ in range(100):
    accel, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert accel > -0.50
  cs.vEgo = 0.45
  accel_handover, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert accel_handover == pytest.approx(-0.50, abs=0.05)


def test_solo_traffic_light_stop_unified_d_stop_and_lead_stop_equivalence():
  ctrl = LimousineStopController()

  # Case 1: Solo traffic-light stop (has_lead=False, is_e2e_stop=True, should_stop=False) with stop_dist=6.0m.
  prev = -1.4
  for _ in range(80):
    prev = ctrl.update(
      a_cmd=-1.4, previous_accel=prev, v_ego=2.0, dt=DT_CTRL,
      enabled=True, should_stop=False, in_stopping_state=False,
      has_lead=False, is_e2e_stop=True, stop_dist=6.0,
    )
  assert ctrl.active is True
  assert ctrl.safety_override is False
  assert -1.4 < prev <= BRAKE_LAMP_HOLD_ACCEL

  # Case 2: Tight stop line (v=2.5 m/s, stop_dist=1.5m -> a_req=-2.08 <= -1.8) -> immediate safety override.
  out_tight = ctrl.update(
    a_cmd=-1.8, previous_accel=prev, v_ego=2.5, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=False, is_e2e_stop=True, stop_dist=1.5,
  )
  assert out_tight == pytest.approx(-1.8)
  assert ctrl.safety_override is True
  assert ctrl.active is False

  # Case 3: Moving lead emergency stop equivalence (v_ego=3.0, v_lead=1.4, d_rel=4.5 -> d_eq=4.745, a_req=-2.00 <= -1.8).
  out_lead_eq = ctrl.update(
    a_cmd=-1.8, previous_accel=-1.0, v_ego=3.0, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=True, d_rel=4.5, v_lead=1.4,
  )
  assert out_lead_eq == pytest.approx(-1.8)
  assert ctrl.safety_override is True

  # Case 4: estimate_plan_stop_distance + LongControl solo e2eStop integration.
  t_idxs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
  # Constant -0.75 m/s^2 from 3.0 m/s -> exact stopping distance = 3.0^2 / (2 * 0.75) = 6.0 m.
  speeds = [max(0.0, 3.0 - 0.75 * t) for t in t_idxs]
  accels = [-0.75] * len(t_idxs)
  d_est = estimate_plan_stop_distance(speeds, accels, t_idxs=t_idxs)
  assert d_est == pytest.approx(6.0, abs=0.05)

  CP = car.CarParams.new_message(
    brand="hyundai", openpilotLongitudinalControl=True,
    stoppingDecelRate=0.8, stopAccel=-2.0, startAccel=0.0, vEgoStarting=0.5,
  )
  loc = LongControl(CP)
  loc.long_control_state = LongCtrlState.pid
  loc.last_output_accel = -1.4
  cs = SimpleNamespace(
    vEgo=2.0, aEgo=-1.2, brakePressed=False, gasPressed=False,
    softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
  )
  plan = SimpleNamespace(
    aTarget=-1.4, vTargetNow=2.0, jTargetNow=0.0, shouldStop=False, fcw=False,
    xState=3, speeds=[max(0.0, 2.0 - 0.5 * float(t)) for t in CONTROL_N_T_IDX], accels=[-0.5] * len(CONTROL_N_T_IDX),
    longitudinalPlanSource="cruise", cruiseTarget=30.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )
  out_e2e, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert loc.unified.state.stop_relief_active is True
  assert out_e2e > -1.4


def test_controlsd_valid_road_pitch_gate():
  import math
  pose_ok = SimpleNamespace(orientation=SimpleNamespace(xyz=[0.0, 0.045, 0.0]))
  pose_nan = SimpleNamespace(orientation=SimpleNamespace(xyz=[0.0, float("nan"), 0.0]))
  pose_inf = SimpleNamespace(orientation=SimpleNamespace(xyz=[0.0, float("inf"), 0.0]))

  assert valid_road_pitch(None, True, True) == 0.0
  assert valid_road_pitch(pose_ok, False, True) == 0.0
  assert valid_road_pitch(pose_ok, True, False) == 0.0
  assert valid_road_pitch(pose_nan, True, True) == 0.0
  assert valid_road_pitch(pose_inf, True, True) == 0.0
  assert valid_road_pitch(pose_ok, True, True) == pytest.approx(0.045)
  assert math.isfinite(valid_road_pitch(pose_ok, True, True))


def test_lamp_floor_hysteresis_hold_prevents_uphill_brake_lamp_drop_488_seg69():
  """Regression test for 488 seg 69: uphill low-speed relaxation must hold lamp_floor until stop or clear departure."""
  import math
  assert BRAKE_LAMP_HOLD_ACCEL == pytest.approx(-0.45)

  pitch_rad = math.radians(0.58)  # 488 seg 69 observed uphill pitch
  floor = brake_lamp_hold_floor(pitch_rad)
  assert floor < -0.54

  ctrl = LimousineStopController()
  # Step 1: Active braking (a_cmd <= floor) arms hysteresis hold.
  out1 = ctrl.update(
    a_cmd=-0.65, previous_accel=-0.65, v_ego=1.4, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=True, d_rel=5.8, v_lead=0.19, pitch_rad=pitch_rad,
  )
  assert out1 <= floor + 1e-9
  assert ctrl.lamp_hold_active is False

  # Step 2: Upstream planner relaxes to -0.20 m/s^2 while lead creeps slightly (+0.38 m/s) at d_rel=5.3 m.
  # Must hold lamp_floor instead of dropping hydraulic pressure & turning off 0x412 brake lamp.
  out2 = ctrl.update(
    a_cmd=-0.20, previous_accel=out1, v_ego=1.1, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=True, d_rel=5.3, v_lead=0.38, pitch_rad=pitch_rad,
  )
  assert out2 == pytest.approx(floor)
  assert ctrl.active is True
  assert ctrl.lamp_hold_active is True

  # Step 3: Clear lead departure (v_lead > 0.8 m/s) -> 0-tick immediate release!
  out_depart = ctrl.update(
    a_cmd=-0.10, previous_accel=out2, v_ego=1.0, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=True, d_rel=5.5, v_lead=0.95, pitch_rad=pitch_rad,
  )
  assert out_depart == pytest.approx(-0.10)
  assert ctrl.active is False
  assert ctrl.lamp_hold_active is False

  # Step 4: Re-arm, then positive acceleration command (a_cmd > 0) -> 0-tick immediate release!
  ctrl.update(
    a_cmd=-0.70, previous_accel=-0.70, v_ego=1.2, dt=DT_CTRL,
    enabled=True, should_stop=True, in_stopping_state=False,
    has_lead=False, is_e2e_stop=True, stop_dist=5.0, pitch_rad=pitch_rad,
  )
  out_solo_hold = ctrl.update(
    a_cmd=-0.25, previous_accel=floor, v_ego=1.0, dt=DT_CTRL,
    enabled=True, should_stop=True, in_stopping_state=False,
    has_lead=False, is_e2e_stop=True, stop_dist=4.0, pitch_rad=pitch_rad,
  )
  assert out_solo_hold == pytest.approx(floor)
  assert ctrl.lamp_hold_active is True

  out_pos = ctrl.update(
    a_cmd=0.15, previous_accel=floor, v_ego=1.0, dt=DT_CTRL,
    enabled=True, should_stop=False, in_stopping_state=False,
    has_lead=False, is_e2e_stop=False, pitch_rad=pitch_rad,
  )
  assert out_pos == pytest.approx(0.15)
  assert ctrl.lamp_hold_active is False

  # Step 5: Emergency braking during hold -> 0-tick immediate bypass to full a_cmd!
  ctrl.update(
    a_cmd=-0.70, previous_accel=-0.70, v_ego=1.2, dt=DT_CTRL,
    enabled=True, should_stop=True, in_stopping_state=True,
    has_lead=True, d_rel=5.0, v_lead=0.0, pitch_rad=pitch_rad,
  )
  out_emg = ctrl.update(
    a_cmd=EMERGENCY_DECEL_THRESHOLD - 0.5, previous_accel=floor, v_ego=1.1, dt=DT_CTRL,
    enabled=True, should_stop=True, in_stopping_state=True,
    has_lead=True, d_rel=4.5, v_lead=0.0, pitch_rad=pitch_rad,
  )
  assert out_emg == pytest.approx(EMERGENCY_DECEL_THRESHOLD - 0.5)
  assert ctrl.safety_override is True
  assert ctrl.lamp_hold_active is False
