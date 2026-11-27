"""Unit tests for LimousineLaunchController and its integration with LongControl and LimousineStop."""
import math
from types import SimpleNamespace

import pytest

from openpilot.cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.limousine_launch import (
  BREAKAWAY_ACCEL_FLAT,
  BREAKAWAY_RAMP_TIME,
  LAUNCH_JERK_MIN,
  LAUNCH_JERK_PEAK,
  LAUNCH_RAMP_TIME,
  REACCEL_BRAKE_RELEASE_JERK,
  REACCEL_JERK_MIN,
  REACCEL_JERK_PEAK,
  V_BREAKAWAY,
  V_REACCEL_MAX,
  LimousineLaunchController,
  breakaway_accel_ceiling,
  launch_jerk_limit,
  uphill_gravity_comp,
)
from opendbc.car.hyundai.carcontroller import calculate_canfd_jerk_limits
from openpilot.selfdrive.controls.lib.limousine_stop import BRAKE_LAMP_HOLD_ACCEL, GRAVITY, MAX_UPHILL_SLOPE_COMP
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


def _make_cp(*, starting_state: bool = False, start_accel: float = 1.0):
  return car.CarParams.new_message(
    brand="hyundai",
    openpilotLongitudinalControl=True,
    stoppingDecelRate=0.8,
    stopAccel=-2.0,
    startAccel=start_accel,
    startingState=starting_state,
    vEgoStarting=0.1,
  )


def test_flat_standstill_launch_prevents_step_jump_and_slingshot():
  ctrl = LimousineLaunchController()
  ctrl.notify_stopping()

  # Phase 1: stationary brake-hold release deadtime (0.75 s, v=0.0) while upstream demands +1.5 m/s^2.
  prev = 0.0
  for step in range(75):
    out = ctrl.update(
      a_cmd=1.5,
      previous_accel=-0.5 if step == 0 else prev,
      v_ego=0.0,
      dt=DT_CTRL,
      enabled=True,
      pitch_rad=0.0,
    )
    assert ctrl.active is True
    assert ctrl.mode == "standstill"
    # Never exceed upstream a_cmd or flat breakaway ceiling while stationary.
    assert 0.0 <= out <= BREAKAWAY_ACCEL_FLAT + 1e-9
    # First tick must not jump from -0.5 to +0.5+.
    if step == 0:
      assert out < 0.05
    prev = out

  # After BREAKAWAY_RAMP_TIME (0.45 s), output settles cleanly at BREAKAWAY_ACCEL_FLAT (0.42 m/s^2).
  assert prev == pytest.approx(BREAKAWAY_ACCEL_FLAT, abs=1e-3)

  # Phase 2: vehicle breaks away (v > V_BREAKAWAY); S-curve ramps smoothly up to 1.5 m/s^2.
  v = V_BREAKAWAY + 0.05
  max_jerk = 0.0
  for _ in range(180):
    out = ctrl.update(
      a_cmd=1.5,
      previous_accel=prev,
      v_ego=v,
      dt=DT_CTRL,
      enabled=True,
      pitch_rad=0.0,
    )
    assert out <= 1.5 + 1e-9
    jerk = (out - prev) / DT_CTRL
    max_jerk = max(max_jerk, jerk)
    prev = out
    v += max(0.0, out) * DT_CTRL

  assert max_jerk <= LAUNCH_JERK_PEAK + 1e-5
  assert prev == pytest.approx(1.5, abs=1e-3)
  assert ctrl.active is False


def test_uphill_launch_adds_gravity_compensation_without_rollback():
  # Flat, downhill, and NaN receive zero uphill compensation.
  assert uphill_gravity_comp(0.0) == 0.0
  assert uphill_gravity_comp(-0.05) == 0.0
  assert uphill_gravity_comp(float("nan")) == 0.0
  assert breakaway_accel_ceiling(0.0) == pytest.approx(BREAKAWAY_ACCEL_FLAT)

  # +4.75% uphill (480-7 observed grade): g*sin(pitch) ~= 0.465 m/s^2.
  pitch_480 = math.atan(0.0475)
  g_up = GRAVITY * math.sin(pitch_480)
  expected_ceil = BREAKAWAY_ACCEL_FLAT + g_up
  assert breakaway_accel_ceiling(pitch_480) == pytest.approx(expected_ceil, abs=1e-4)

  # Steep uphill caps compensation at MAX_UPHILL_SLOPE_COMP.
  assert uphill_gravity_comp(math.atan(0.25)) == pytest.approx(MAX_UPHILL_SLOPE_COMP)

  # Simulate stationary hold release on +4.75% uphill with upstream a_cmd=1.74 m/s^2.
  ctrl = LimousineLaunchController()
  ctrl.notify_stopping()
  prev = 0.0
  for _ in range(int(BREAKAWAY_RAMP_TIME / DT_CTRL) + 10):
    prev = ctrl.update(
      a_cmd=1.74,
      previous_accel=prev,
      v_ego=0.0,
      dt=DT_CTRL,
      enabled=True,
      pitch_rad=pitch_480,
    )
  # Breakaway ceiling exceeds g*sin(pitch) by BREAKAWAY_ACCEL_FLAT -> zero rollback, no 1.74 slingshot.
  assert prev == pytest.approx(expected_ceil, abs=1e-3)
  assert prev > g_up + 0.35
  assert prev < 1.25

  # Uphill post-breakaway jerk limit gets grade bonus so uphill launch stays responsive.
  assert launch_jerk_limit(0.25, mode="standstill", pitch_rad=pitch_480) > launch_jerk_limit(0.25, mode="standstill", pitch_rad=0.0)


def test_short_ramp_duration_preserves_target_accel_responsiveness():
  # Owner requirement: short S-curve ramp (0.35-0.55s) so arrival-time loss is minimal.
  assert 0.35 <= LAUNCH_RAMP_TIME <= 0.55
  ctrl = LimousineLaunchController()
  ctrl.notify_stopping()
  # Break away immediately at v=0.2 m/s with target 1.5 m/s^2; must reach >= 90% of target within 0.60s.
  prev = 0.0
  t_reach = None
  for step in range(1, 100):
    out = ctrl.update(a_cmd=1.5, previous_accel=prev, v_ego=0.2 + 0.01 * step, dt=DT_CTRL, enabled=True)
    prev = out
    if t_reach is None and out >= 1.35:
      t_reach = step * DT_CTRL
  assert t_reach is not None
  assert t_reach <= 0.60


def test_low_speed_congestion_decel_to_reaccel_s_curve():
  ctrl = LimousineLaunchController()
  # Rolling braking at 3.3 m/s (like 486-6): braking must pass through untouched.
  for _ in range(20):
    out = ctrl.update(a_cmd=-0.45, previous_accel=-0.45, v_ego=3.3, dt=DT_CTRL, enabled=True)
    assert out == pytest.approx(-0.45)
    assert ctrl.active is False

  # Suddenly switch to strong +1.6 m/s^2 acceleration demand.
  prev = -0.45
  v = 3.3
  jerks = []
  for step in range(150):
    out = ctrl.update(a_cmd=1.6, previous_accel=prev, v_ego=v, dt=DT_CTRL, enabled=True)
    assert out <= 1.6 + 1e-9
    assert out >= 0.0
    if step > 0:
      jerks.append((out - prev) / DT_CTRL)
    prev = out
    v += out * DT_CTRL

  # First positive tick starts smoothly near 0, not jumping to 1.6.
  assert jerks[0] <= REACCEL_JERK_MIN + 0.05
  assert max(jerks) <= REACCEL_JERK_PEAK + 1e-5
  assert prev == pytest.approx(1.6, abs=1e-3)
  assert ctrl.active is False

  # Above V_REACCEL_MAX (highway speed), decel->accel passes straight through untouched.
  ctrl.reset()
  ctrl.update(a_cmd=-0.3, previous_accel=-0.3, v_ego=V_REACCEL_MAX + 3.0, dt=DT_CTRL, enabled=True)
  out_hw = ctrl.update(a_cmd=1.2, previous_accel=-0.3, v_ego=V_REACCEL_MAX + 3.0, dt=DT_CTRL, enabled=True)
  assert out_hw == pytest.approx(1.2)
  assert ctrl.active is False


def test_emergency_braking_fcw_and_should_stop_bypass_immediately():
  ctrl = LimousineLaunchController()
  ctrl.notify_stopping()
  # Enter active launch shaping.
  out = ctrl.update(a_cmd=1.4, previous_accel=-0.5, v_ego=0.2, dt=DT_CTRL, enabled=True)
  assert ctrl.active is True
  assert out < 1.4

  # Case 1: sudden hard braking (-2.5 m/s^2) -> 0-tick bypass.
  out_brk = ctrl.update(a_cmd=-2.5, previous_accel=out, v_ego=0.5, dt=DT_CTRL, enabled=True)
  assert out_brk == pytest.approx(-2.5)
  assert ctrl.active is False

  # Re-enter active launch.
  ctrl.notify_stopping()
  out = ctrl.update(a_cmd=1.4, previous_accel=-0.5, v_ego=0.2, dt=DT_CTRL, enabled=True)
  assert ctrl.active is True

  # Case 2: FCW active -> immediate bypass.
  out_fcw = ctrl.update(a_cmd=0.8, previous_accel=out, v_ego=0.5, dt=DT_CTRL, enabled=True, fcw=True)
  assert out_fcw == pytest.approx(0.8)
  assert ctrl.active is False

  # Case 3: should_stop -> immediate bypass.
  ctrl.notify_stopping()
  out = ctrl.update(a_cmd=1.4, previous_accel=-0.5, v_ego=0.2, dt=DT_CTRL, enabled=True)
  assert ctrl.active is True
  out_stop = ctrl.update(a_cmd=0.5, previous_accel=out, v_ego=0.5, dt=DT_CTRL, enabled=True, should_stop=True)
  assert out_stop == pytest.approx(0.5)
  assert ctrl.active is False


def test_coexistence_limousine_stop_then_launch_in_longcontrol():
  loc = LongControl(_make_cp(starting_state=False))
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -1.4

  cs = SimpleNamespace(
    vEgo=1.2, aEgo=-1.2, brakePressed=False, gasPressed=False,
    softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
  )
  plan = SimpleNamespace(
    aTarget=-1.4, vTargetNow=0.2, jTargetNow=0.0, shouldStop=True, fcw=False,
    longitudinalPlanSource="lead0", cruiseTarget=30.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=6.5, vLead=0.0),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )

  # Stage 1: Limousine Stop smoothly unwinds strong braking toward -0.42 m/s^2.
  for _ in range(80):
    accel, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
    assert accel <= BRAKE_LAMP_HOLD_ACCEL + 1e-9
  assert loc.limousine_stop.active is True
  assert loc.limousine_launch.active is False

  # Stage 2: Come to standstill, then lead departs -> shouldStop flips 1 -> 0.
  cs.vEgo = 0.0
  cs.aEgo = 0.0
  loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  plan.shouldStop = False
  plan.aTarget = 0.8
  plan.vTargetNow = 0.5
  plan.jTargetNow = 2.0

  accel_first, _, j_first = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert loc.long_control_state == LongCtrlState.pid
  assert loc.limousine_stop.active is False
  assert loc.limousine_launch.active is True
  # No single-tick +1.0 m/s^2 step jump; shaped jerk also caps j_target_now.
  assert 0.0 <= accel_first < 0.05
  assert j_first <= LAUNCH_JERK_MIN + 1e-6


def test_starting_state_start_accel_is_shaped_smoothly():
  loc = LongControl(_make_cp(starting_state=True, start_accel=1.0))
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.5

  cs = SimpleNamespace(
    vEgo=0.0, aEgo=0.0, brakePressed=False, gasPressed=False,
    softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
  )
  plan = SimpleNamespace(
    aTarget=0.8, vTargetNow=0.4, jTargetNow=1.5, shouldStop=False, fcw=False,
    longitudinalPlanSource="lead0", cruiseTarget=30.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=5.0, vLead=0.5),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )

  # Even with startingState=True and startAccel=1.0, launch ramps smoothly instead of jumping to 1.0.
  accel_0, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  assert loc.long_control_state == LongCtrlState.starting
  assert loc.limousine_launch.active is True
  assert 0.0 <= accel_0 < 0.05

  # Non-increasing peak: stays <= startAccel (1.0) and <= BREAKAWAY_ACCEL_FLAT while stationary.
  for _ in range(50):
    accel, _, _ = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
    assert accel <= BREAKAWAY_ACCEL_FLAT + 1e-9


def test_breakaway_tracking_aligns_j_target_now_when_planner_jerk_saturates():
  loc = LongControl(_make_cp(starting_state=False))
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.5

  cs = SimpleNamespace(
    vEgo=0.0, aEgo=0.0, brakePressed=False, gasPressed=False,
    softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
  )
  # Planner jTargetNow saturated to 0.0 during stationary brake-hold release deadtime.
  plan = SimpleNamespace(
    aTarget=1.7, vTargetNow=0.6, jTargetNow=0.0, shouldStop=False, fcw=False,
    longitudinalPlanSource="lead0", cruiseTarget=30.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
  )
  radar = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=6.0, vLead=1.0),
    leadTwo=SimpleNamespace(status=False, dRel=0.0, vLead=0.0),
  )

  # Phase 1: stationary hold deadtime (0.75 s) -> j_out stays 0.0 (no premature inflation).
  for _ in range(75):
    accel, _, j_out = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
    assert accel <= BREAKAWAY_ACCEL_FLAT + 1e-9
    assert j_out == pytest.approx(0.0)

  # Phase 2: vehicle breaks away (vEgo > V_BREAKAWAY) with planner jTargetNow still 0.0.
  cs.vEgo = V_BREAKAWAY + 0.06
  seen_jerk_u = []
  for _ in range(45):
    accel, _, j_out = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
    if loc.limousine_launch.active and loc.limousine_launch.tracking_upstream:
      ju, _ = calculate_canfd_jerk_limits(accel, j_out, 0.0)
      assert ju == pytest.approx(loc.limousine_launch.shaped_jerk, abs=1e-4)
      seen_jerk_u.append(ju)
    cs.vEgo += max(0.0, accel) * DT_CTRL

  # Bottleneck at 1.0 m/s^3 is broken: jerk_u ramps smoothly from LAUNCH_JERK_MIN toward LAUNCH_JERK_PEAK.
  assert len(seen_jerk_u) >= 20
  assert seen_jerk_u[0] == pytest.approx(LAUNCH_JERK_MIN, abs=0.05)
  assert max(seen_jerk_u) > 3.0
  assert max(seen_jerk_u) <= LAUNCH_JERK_PEAK + 1e-4

  # Phase 3: when planner jTargetNow is huge (e.g. 5.0), CAN-FD jerk_u still matches shaped_jerk (not 2x).
  loc.reset()
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.5
  cs.vEgo = V_BREAKAWAY + 0.06
  plan.jTargetNow = 5.0
  accel, _, j_out = loc.update(True, cs, plan, [-3.5, 2.0], 0.0, radar)
  ju, _ = calculate_canfd_jerk_limits(accel, j_out, 0.0)
  assert ju == pytest.approx(loc.limousine_launch.shaped_jerk, abs=1e-4)


def test_reaccel_zero_crossing_sync_preserves_soft_s_curve_after_brake_release():
  ctrl = LimousineLaunchController()
  for _ in range(10):
    ctrl.update(a_cmd=-0.50, previous_accel=-0.50, v_ego=4.0, dt=DT_CTRL, enabled=True)

  # Enter reaccel from -0.50 m/s^2: release takes 0.50 / 2.50 = 0.20 s (20 ticks).
  prev = -0.50
  release_ticks = 0
  post_zero_jerks = []
  for step in range(120):
    out = ctrl.update(a_cmd=1.6, previous_accel=prev, v_ego=4.0, dt=DT_CTRL, enabled=True)
    assert 0.0 <= out <= 1.6 + 1e-9
    if out == 0.0:
      release_ticks += 1
      assert ctrl._ramp_t == pytest.approx(0.0)
      assert ctrl.shaped_jerk == pytest.approx(REACCEL_BRAKE_RELEASE_JERK)
    else:
      post_zero_jerks.append((out - prev) / DT_CTRL)
    prev = out

  # Brake-release sync held _ramp_t at 0 for ~0.20 s, then S-curve started cleanly at REACCEL_JERK_MIN.
  assert 18 <= release_ticks <= 21
  assert len(post_zero_jerks) >= 20
  assert post_zero_jerks[0] == pytest.approx(REACCEL_JERK_MIN, abs=0.05)
  assert max(post_zero_jerks) <= REACCEL_JERK_PEAK + 1e-5
  assert prev == pytest.approx(1.6, abs=1e-3)

  # If already released to 0.0 (via decel memory), S-curve starts on tick 0 with zero holdback.
  ctrl.reset()
  ctrl.update(a_cmd=-0.30, previous_accel=-0.30, v_ego=4.0, dt=DT_CTRL, enabled=True)
  ctrl.update(a_cmd=0.0, previous_accel=0.0, v_ego=4.0, dt=DT_CTRL, enabled=True)
  out0 = ctrl.update(a_cmd=1.2, previous_accel=0.0, v_ego=4.0, dt=DT_CTRL, enabled=True)
  assert out0 > 0.0
  assert out0 / DT_CTRL == pytest.approx(REACCEL_JERK_MIN, abs=0.05)
