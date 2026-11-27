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
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


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
  assert loc.limousine_stop.active is True


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
