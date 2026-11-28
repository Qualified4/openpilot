"""Unit and regression tests for UnifiedLongController (Stage 1 of #253)."""
import math
from types import SimpleNamespace

import pytest

from openpilot.common.realtime import DT_CTRL
from opendbc.car.hyundai.carcontroller import HyundaiJerk, LongCtrlState, calculate_canfd_jerk_limits
from opendbc.car.hyundai.hyundaicanfd import apply_accel_jerk_limit
from opendbc.car.hyundai.stopping import (
  APPROACH_ACCEL,
  CanfdStopping,
  PRECHARGE_MAX_TIME,
  StopPhase,
  DT as SCC_DT,
)
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  lead_departure_confidence,
  rolling_approach_weight,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  STANDSTILL_ANCHOR_A_MARGIN,
  STANDSTILL_ANCHOR_V_MARGIN,
  anchor_standstill_planner_state,
  apply_stopping_queue_settle_cap,
)
from openpilot.selfdrive.controls.lib.unified_long import (
  A_HYD_LAMP_FLOOR,
  BREAKAWAY_ACCEL_FLAT,
  DEFAULT_STOP_DISTANCE,
  EMERGENCY_DECEL_THRESHOLD,
  JERK_DECEL_RELIEF_LOW,
  JERK_REACCEL_MIN,
  JERK_UNWIND_TO_POS,
  MIN_SAFE_STOP_DISTANCE,
  UnifiedLongController,
  UnifiedLongInput,
  V_BREAKAWAY,
  brake_lamp_floor,
  uphill_gravity_comp,
)


def _make_inp(**overrides) -> UnifiedLongInput:
  defaults = dict(
    active=True,
    dt=DT_CTRL,
    v_ego=1.5,
    a_ego=-0.6,
    a_des=-0.6,
    a_target=-0.6,
    v_target_now=1.0,
    j_target_now=0.0,
    accel_min=-3.5,
    accel_max=2.0,
    should_stop=False,
    soft_hold=False,
    brake_pressed=False,
    gas_pressed=False,
    cruise_standstill=False,
    fcw=False,
    has_lead=True,
    d_rel=5.5,
    v_lead=0.0,
    pitch_rad=0.0,
    is_e2e_stop=False,
    plan_stop_dist=None,
    stopping_accel_cfg=-0.50,
    cp_stop_accel=-2.0,
    cp_stopping_decel_rate=0.8,
    cp_starting_state=False,
    cp_v_starting=0.5,
    cp_start_accel=0.0,
    d_stop_cfg=4.0,
    prev_scc_areq_val=0.0,
  )
  defaults.update(overrides)
  return UnifiedLongInput(**defaults)


def test_four_quadrant_continuity_zero_step_jumps():
  """Continuous decel -> relief -> brake unwind -> zero-crossing -> S-curve has zero mode step jumps."""
  ctrl = UnifiedLongController()
  prev_a = -1.6
  ctrl.sync_external_state(a_ref=prev_a, in_stopping=False)

  # Phase 1: Strong braking -> low-speed stop ease-out (1.2 m/s^3)
  for _ in range(80):
    out = ctrl.update(_make_inp(v_ego=1.2, a_ego=prev_a, a_des=-1.6, a_target=-1.6, d_rel=7.0, should_stop=True))
    assert out.a_cmd - prev_a <= (JERK_DECEL_RELIEF_LOW * DT_CTRL + 1e-6)
    assert out.a_cmd <= A_HYD_LAMP_FLOOR + 1e-9
    prev_a = out.a_cmd

  assert prev_a < -0.44
  # Phase 2: Lead accelerates -> brake unwind at 2.5 m/s^3 to a=0, then soft S-curve from 0.85 m/s^3
  crossed_zero = False
  post_zero_jerks = []
  for _ in range(100):
    out = ctrl.update(_make_inp(
      v_ego=2.0, a_ego=prev_a, a_des=1.4, a_target=1.4, v_target_now=3.0,
      should_stop=False, has_lead=True, d_rel=8.0, v_lead=3.0,
    ))
    da = out.a_cmd - prev_a
    if prev_a < -1e-9:
      assert 0.0 <= da <= JERK_UNWIND_TO_POS * DT_CTRL + 1e-6
      if out.a_cmd == 0.0:
        crossed_zero = True
    else:
      if da > 1e-9:
        post_zero_jerks.append(da / DT_CTRL)
    prev_a = out.a_cmd

  assert crossed_zero is True
  assert len(post_zero_jerks) >= 20
  assert post_zero_jerks[0] == pytest.approx(JERK_REACCEL_MIN, abs=0.08)
  assert max(post_zero_jerks) <= 2.50


def test_kinematic_stop_pocket_precharge_prevents_488_seg69_lamp_drop():
  """In uphill stop pocket (488 seg 69), pre-charge holds a_hyd <= -0.45 until StopReq or departure."""
  ctrl = UnifiedLongController()
  pitch_rad = math.radians(0.58)
  floor = brake_lamp_floor(pitch_rad)
  assert floor == pytest.approx(-0.45 - uphill_gravity_comp(pitch_rad))

  # Step 1: Braking in stop pocket (v=1.22 m/s, d_rel=5.5 m -> d_rem=1.5 m <= dx_lamp+1.2)
  out1 = ctrl.update(_make_inp(
    v_ego=1.22, a_ego=-0.60, a_des=-0.65, a_target=-0.65, d_rel=5.5, v_lead=0.15, pitch_rad=pitch_rad,
  ))
  assert out1.precharge_latched is True
  assert out1.a_cmd <= floor + 1e-9

  # Step 2: Lead creeps slightly (0.38 m/s) and planner relaxes to -0.20 -> must hold floor!
  out2 = ctrl.update(_make_inp(
    v_ego=1.10, a_ego=-0.45, a_des=-0.20, a_target=-0.20, d_rel=5.3, v_lead=0.38, pitch_rad=pitch_rad,
  ))
  assert out2.precharge_latched is True
  assert out2.a_cmd <= floor + 1e-9
  assert out2.a_cmd + uphill_gravity_comp(pitch_rad) <= A_HYD_LAMP_FLOOR + 1e-9

  # Step 3: Far from stop pocket (d_rel=12.0 m) -> no premature early stop; smoothly relaxes above floor!
  ctrl_far = UnifiedLongController()
  out_far1 = ctrl_far.update(_make_inp(
    v_ego=1.50, a_ego=-0.60, a_des=-0.60, a_target=-0.60, d_rel=12.0, v_lead=0.0, pitch_rad=pitch_rad,
  ))
  assert out_far1.precharge_latched is False
  for _ in range(40):
    out_far2 = ctrl_far.update(_make_inp(
      v_ego=1.40, a_ego=-0.30, a_des=-0.20, a_target=-0.20, d_rel=11.5, v_lead=0.0, pitch_rad=pitch_rad,
    ))
  assert out_far2.precharge_latched is False
  assert out_far2.a_cmd == pytest.approx(-0.20, abs=1e-3)
  assert out_far2.a_cmd > floor

  # Step 4: Clear lead departure (v_lead > 0.85 m/s) -> 0-tick release of precharge latch and smooth unwind!
  out_dep = ctrl.update(_make_inp(
    v_ego=1.00, a_ego=-0.40, a_des=-0.10, a_target=-0.10, d_rel=5.6, v_lead=0.95, pitch_rad=pitch_rad,
  ))
  assert out_dep.precharge_latched is False
  for _ in range(50):
    out_dep = ctrl.update(_make_inp(
      v_ego=1.00, a_ego=-0.40, a_des=-0.10, a_target=-0.10, d_rel=5.6, v_lead=0.95, pitch_rad=pitch_rad,
    ))
  assert out_dep.a_cmd == pytest.approx(-0.10, abs=1e-3)
  assert out_dep.a_cmd > floor


@pytest.mark.parametrize("trigger", ["fcw", "hard_decel", "close_lead", "brake_pedal", "gas_pedal"])
def test_zero_tick_safety_and_pedal_bypass(trigger):
  """FCW, <-2.5 m/s^2, d_rel <= 2.8 m, and driver pedals bypass all shaping on tick 0."""
  ctrl = UnifiedLongController()
  ctrl.sync_external_state(a_ref=-0.50, in_stopping=False)
  # Arm precharge first
  ctrl.update(_make_inp(v_ego=1.2, a_ego=-0.50, a_des=-0.60, a_target=-0.60, d_rel=5.2, v_lead=0.0))
  assert ctrl.state.precharge_latched is True

  kwargs = dict(v_ego=1.2, a_ego=-0.50, a_des=-1.8, a_target=-1.8, d_rel=5.2, v_lead=0.0)
  if trigger == "fcw":
    kwargs["fcw"] = True
  elif trigger == "hard_decel":
    kwargs["a_des"] = EMERGENCY_DECEL_THRESHOLD - 0.5
    kwargs["a_target"] = EMERGENCY_DECEL_THRESHOLD - 0.5
  elif trigger == "close_lead":
    kwargs["d_rel"] = 2.7
  elif trigger == "brake_pedal":
    kwargs["brake_pressed"] = True
  elif trigger == "gas_pedal":
    kwargs["gas_pressed"] = True
    kwargs["a_des"] = 0.5
    kwargs["a_target"] = 0.5

  out = ctrl.update(_make_inp(**kwargs))
  assert out.precharge_latched is False
  if trigger in ("fcw", "hard_decel", "close_lead"):
    assert out.safety_override is True
    assert out.a_cmd == pytest.approx(kwargs["a_des"])
  else:
    assert out.a_cmd == pytest.approx(kwargs["a_des"])


def test_canfd_stopping_precharge_continuity_guard_and_1to1_jerk_contract():
  """Late StopReq with unpressurized brakes pre-charges up to 0.16s before asserting StopReq=1."""
  pitch_rad = math.radians(0.58)
  stopper = CanfdStopping()

  # Unpressurized late entry at v=0.62 m/s (488 seg 69 scenario): a_hyd_prev = -0.24 + 0.10 = -0.14 > -0.38
  prev_v = -0.24
  approach_frames = 0
  stop_req_entry_hyd = None
  for _ in range(15):
    cmd = stopper.update(
      active=True, requested=True, speed=0.60, held=False,
      accel=-0.55, previous_value=prev_v, jerk_u=1.2, jerk_l=2.5,
      pitch_rad=pitch_rad, soft_hold=False, close_lead=False,
    )
    if cmd.stop_req == 0:
      approach_frames += 1
      prev_v = cmd.value
    else:
      stop_req_entry_hyd = prev_v + uphill_gravity_comp(pitch_rad)
      break

  assert 1 <= approach_frames <= int(round(PRECHARGE_MAX_TIME / SCC_DT)) + 1
  assert stop_req_entry_hyd is not None
  assert stop_req_entry_hyd <= -0.38

  # Close-lead emergency bypasses pre-charge dwell immediately on frame 0!
  stopper_emg = CanfdStopping()
  cmd_emg = stopper_emg.update(
    active=True, requested=True, speed=0.60, held=False,
    accel=-0.55, previous_value=-0.20, jerk_u=1.2, jerk_l=2.5,
    pitch_rad=pitch_rad, soft_hold=False, close_lead=True,
  )
  assert cmd_emg.stop_req == 1

  # 1:1 CAN-FD jerk contract: aReqValue matches aReqRaw with zero lag during S-curve launch
  hj = HyundaiJerk.__new__(HyundaiJerk)
  hj.jerk = hj.jerk_u = hj.jerk_l = hj.cb_upper = hj.cb_lower = 0.0
  hj.jerk_u_min = 0.5
  from collections import deque
  from openpilot.common.filter_simple import MyMovingAverage
  hj.accel_request_history = deque(maxlen=50)
  hj.jerk_error_filter = MyMovingAverage(40, 0.0)
  cp = SimpleNamespace(flags=HyundaiFlags.CANFD)
  cs = SimpleNamespace(out=SimpleNamespace(aEgo=0.0, brakePressed=False, gasPressed=False))

  ctrl = UnifiedLongController()
  ctrl.sync_external_state(a_ref=0.0, in_stopping=True)
  aval = 0.0
  for tick in range(50):
    out = ctrl.update(_make_inp(
      v_ego=V_BREAKAWAY + 0.06 + tick * 0.01, a_ego=aval, a_des=1.5, a_target=1.5,
      v_target_now=2.0, j_target_now=0.0, should_stop=False, has_lead=True, d_rel=8.0, v_lead=3.0,
    ))
    act = SimpleNamespace(longControlState=LongCtrlState.pid, jerk=out.j_cmd)
    hj.make_jerk(cp, cs, out.a_cmd, act, None)
    if out.j_u_shaped is not None:
      assert hj.jerk_u == pytest.approx(max(1.0, min(5.0, out.j_u_shaped)), abs=1e-4)
    if tick % 2 == 0:
      aval = apply_accel_jerk_limit(out.a_cmd, aval, hj.jerk_u, hj.jerk_l)
      assert aval == pytest.approx(out.a_cmd, abs=0.015)


def test_stage2_standstill_planner_state_anchoring():
  """Planner (v_desired, a_desired) anchors to stopped car during hold/breakaway and releases once rolling."""
  # During IBS hold (v_ego=0.0, a_act=-0.5): v_desired clamped to 0.12, positive a_desired clamped to 0.15.
  v0, a0 = anchor_standstill_planner_state(0.80, 1.30, v_ego=0.0, a_act=-0.50)
  assert v0 == pytest.approx(STANDSTILL_ANCHOR_V_MARGIN)
  assert a0 == pytest.approx(STANDSTILL_ANCHOR_A_MARGIN)

  # During breakaway ramp (v_ego=0.02, a_act=0.65): tracks right ahead of actual command.
  v1, a1 = anchor_standstill_planner_state(0.60, 1.40, v_ego=0.02, a_act=0.65)
  assert v1 == pytest.approx(0.02 + STANDSTILL_ANCHOR_V_MARGIN)
  assert a1 == pytest.approx(0.65 + STANDSTILL_ANCHOR_A_MARGIN)

  # Negative braking a_desired is never lifted by anchoring.
  v_brk, a_brk = anchor_standstill_planner_state(0.05, -0.80, v_ego=0.0, a_act=-0.50)
  assert v_brk == pytest.approx(0.05)
  assert a_brk == pytest.approx(-0.80)

  # Once wheels roll (v_ego > 0.05 m/s) or on reset/gas_pressed, anchoring releases immediately.
  assert anchor_standstill_planner_state(0.80, 1.30, v_ego=0.06, a_act=0.65) == (0.80, 1.30)
  assert anchor_standstill_planner_state(0.80, 1.30, v_ego=0.0, a_act=0.0, reset_state=True) == (0.80, 1.30)
  assert anchor_standstill_planner_state(0.80, 1.30, v_ego=0.0, a_act=0.0, gas_pressed=True) == (0.80, 1.30)


def test_stage2_alpha_depart_and_stopping_queue_settle_cap():
  """Continuous alpha_depart blocks 488 seg 80 creep re-accel while preserving 0.00 s standstill launch."""
  # 488 seg 80 snapshot: v_ego=1.09, d_rel=5.70, d_stop_cfg=3.50, v_lead=1.42, a_desired=-0.34
  assert lead_departure_confidence(1.09, 5.70, 1.42, d_stop_cfg=3.50) == pytest.approx(0.0)
  assert rolling_approach_weight(1.09) == pytest.approx(1.0)
  assert apply_stopping_queue_settle_cap(
    1.6, a_desired=-0.34, v_ego=1.09, has_lead=True, d_rel=5.70, v_lead=1.42, d_stop_cfg=3.50,
  ) == pytest.approx(0.0)

  # Standstill launch (v_ego <= 0.35 m/s): 0.00 s delay (full a_max_turns preserved).
  assert rolling_approach_weight(0.0) == 0.0
  assert rolling_approach_weight(0.35) == 0.0
  assert apply_stopping_queue_settle_cap(
    1.6, a_desired=-0.20, v_ego=0.0, has_lead=True, d_rel=4.0, v_lead=0.5, d_stop_cfg=3.50,
  ) == pytest.approx(1.6)
  assert apply_stopping_queue_settle_cap(
    1.6, a_desired=-0.20, v_ego=0.35, has_lead=True, d_rel=4.0, v_lead=0.5, d_stop_cfg=3.50,
  ) == pytest.approx(1.6)

  # Smooth intermediate transition as lead accelerates (1.50 -> 2.40 m/s) or gap widens (2.40 -> 3.60 m excess).
  a_mid_v = lead_departure_confidence(1.1, 5.5, 1.95, d_stop_cfg=3.5)
  a_mid_d = lead_departure_confidence(1.1, 6.5, 1.20, d_stop_cfg=3.5)
  assert 0.0 < a_mid_v < 1.0
  assert 0.0 < a_mid_d < 1.0
  assert lead_departure_confidence(1.1, 5.5, 2.40, d_stop_cfg=3.5) == pytest.approx(1.0)
  assert lead_departure_confidence(1.1, 7.10, 1.00, d_stop_cfg=3.5) == pytest.approx(1.0)
  assert lead_departure_confidence(2.60, 5.0, 1.00, d_stop_cfg=3.5) == pytest.approx(1.0)

  # Already accelerating normally (a_desired > 0.05) -> settle cap does not interfere.
  assert apply_stopping_queue_settle_cap(
    1.6, a_desired=0.40, v_ego=1.09, has_lead=True, d_rel=5.70, v_lead=1.42, d_stop_cfg=3.50,
  ) == pytest.approx(1.6)


def test_stage2_longcontrol_stop_distance_carrot_sync():
  """LongControl refreshes actual StopDistanceCarrot and clamps safely to [2.5, 10.0] m."""
  from openpilot.cereal import car
  cp = car.CarParams.new_message()
  cp.brand = "hyundai"
  cp.openpilotLongitudinalControl = True
  loc = LongControl(cp)

  class FakeParams:
    def __init__(self, val):
      self.val = val
    def get_float(self, key):
      assert key == "StopDistanceCarrot"
      return self.val

  loc.params = FakeParams(350.0)
  loc._refresh_stop_distance()
  assert loc.stop_distance == pytest.approx(3.5)

  loc.params = FakeParams(550.0)
  loc._refresh_stop_distance()
  assert loc.stop_distance == pytest.approx(5.5)

  # Unset (0) falls back to DEFAULT_STOP_DISTANCE (4.0 m); unsafe (< 2.5 m) clamps to 2.5 m.
  loc.params = FakeParams(0.0)
  loc._refresh_stop_distance()
  assert loc.stop_distance == pytest.approx(DEFAULT_STOP_DISTANCE)

  loc.params = FakeParams(180.0)
  loc._refresh_stop_distance()
  assert loc.stop_distance == pytest.approx(MIN_SAFE_STOP_DISTANCE)
