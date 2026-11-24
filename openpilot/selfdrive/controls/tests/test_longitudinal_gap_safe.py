"""GapSafeDistance: gap levels assume how hard the lead brakes; d_min replaces the TF distance at lead nodes."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib import longitudinal_gap_safe as gs

DT = 0.05
STOP = 3.7
V80 = 80 / 3.6


def full_gap(lead_decel=6.0, reaction=0.6, extra=0.0, t_follow=1.0, comfort_brake=2.4):
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=lead_decel, snap=True, dt=DT)
  return s.lead_gap(reaction=reaction, extra_time=extra, t_follow=t_follow, comfort_brake=comfort_brake,
                    danger_factor=0.8, inv_base_lead_decel=1 / 2.5)


def test_formula_at_full_weight():
  g = full_gap()
  assert g.t_follow == pytest.approx(0.6) and g.comfort_brake == pytest.approx(gs.EGO_MAX_DECEL)
  assert gs.EGO_MAX_DECEL == 4.0 and g.danger_factor == 1.0 and g.comfort_time == pytest.approx(gs.COMFORT_TIME)
  v, vl = V80, 15.0
  d_min = 0.6 * v + v ** 2 / 8.0 - vl ** 2 / 12.0 + STOP
  assert gs.min_distance(v, vl, g, STOP) == pytest.approx(d_min)
  assert gs.desired_distance(v, vl, g, STOP) == pytest.approx(d_min + 0.15 * v)
  # Steady following (same speed): equivalent TF = (target - stop) / v.
  assert gs.equivalent_t_follow(v, g) * v == pytest.approx(gs.desired_distance(v, v, g, STOP) - STOP)


def test_drive_mode_and_braking_extra_only_add_comfort_time():
  g = full_gap(extra=0.2)
  assert g.comfort_time == pytest.approx(gs.COMFORT_TIME + 0.2)
  assert gs.min_distance(V80, V80, g, STOP) == pytest.approx(gs.min_distance(V80, V80, full_gap(), STOP))


def test_blend_in_over_enable_time_and_off_at_once():
  s = gs.GapSafeState()
  steps = round(gs.ENABLE_TIME / DT)
  for i in range(steps):
    s.update(enabled=True, lead_decel=6.0, snap=False, dt=DT)
    if i == 0:
      assert s.weight == pytest.approx(DT / gs.ENABLE_TIME)
  assert s.weight == pytest.approx(1.0)
  half = gs.GapSafeState()
  half.weight, half.inv_lead_decel = 0.5, 1 / 6.0
  g = half.lead_gap(reaction=0.6, extra_time=0.0, t_follow=1.0, comfort_brake=2.4, danger_factor=0.8, inv_base_lead_decel=0.4)
  # Linear in tF, 1/comfort_brake and 1/lead_decel, so the target distance is the mix of the two.
  old = 1.0 * V80 + V80 ** 2 / 4.8 + STOP - 15.0 ** 2 / 5.0
  new = gs.desired_distance(V80, 15.0, full_gap(t_follow=1.0), STOP)
  assert gs.desired_distance(V80, 15.0, g, STOP) == pytest.approx(0.5 * old + 0.5 * new)
  s.update(enabled=False, lead_decel=6.0, snap=False, dt=DT)
  assert s.weight == 0.0 and s.inv_lead_decel is None


def test_snap_while_not_engaged():
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=8.0, snap=True, dt=DT)
  assert s.weight == 1.0 and s.inv_lead_decel == pytest.approx(1 / 8.0)


def test_level_change_lengthens_gradually_and_shortens_at_once():
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=5.0, snap=True, dt=DT)
  s.update(enabled=True, lead_decel=8.0, snap=False, dt=DT)   # 1 -> 4: longer gap, ramped
  assert s.inv_lead_decel == pytest.approx(1 / 5.0 - gs.LEAD_DECEL_INV_RATE * DT)
  for _ in range(round(3.0 / DT)):
    s.update(enabled=True, lead_decel=8.0, snap=False, dt=DT)
  assert s.inv_lead_decel == pytest.approx(1 / 8.0)
  s.update(enabled=True, lead_decel=5.0, snap=False, dt=DT)   # 4 -> 1: shorter gap, at once
  assert s.inv_lead_decel == pytest.approx(1 / 5.0)


@pytest.mark.parametrize("kw", [{"lead_decel": float("nan")}, {"dt": 0.0}, {"enabled": False}])
def test_invalid_input_resets(kw):
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=6.0, snap=True, dt=DT)
  s.update(**{"enabled": True, "lead_decel": 6.0, "snap": False, "dt": DT, **kw})
  assert s.weight == 0.0 and s.inv_lead_decel is None


def test_limits():
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=50.0, snap=True, dt=DT)
  assert s.inv_lead_decel == pytest.approx(1 / gs.LEAD_DECEL_LIMITS[1])
  g = s.lead_gap(reaction=5.0, extra_time=-1.0, t_follow=1.0, comfort_brake=2.4, danger_factor=0.8, inv_base_lead_decel=0.4)
  assert g.t_follow == gs.REACTION_LIMITS[1] and g.comfort_time == pytest.approx(gs.COMFORT_TIME)


def test_carrot_reads_and_clips_settings():
  from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner, DrivingMode, DrivingModeDetector
  carrot = CarrotPlanner.__new__(CarrotPlanner)
  values = {"GapSafeDistance": 3, "GapSafeLeadDecel1": 30, "GapSafeLeadDecel2": 60, "GapSafeLeadDecel3": 75,
            "GapSafeLeadDecel4": 200, "GapSafeReaction": 10, "MyDrivingMode": DrivingMode.Normal.value, "SpeedTFFactor": 10}
  carrot.params = NS(get_int=lambda k: int(values.get(k, 0)), get_float=lambda k: float(values.get(k, 0)))
  carrot.frame = carrot.params_count = 19
  carrot.myDrivingMode_last = carrot.myDrivingMode = DrivingMode.Normal
  carrot.myDrivingMode_disable_auto = False
  carrot.drivingModeDetector = DrivingModeDetector()
  carrot._params_update()
  assert carrot.gapSafeDistance == 1
  assert carrot.gapSafeLeadDecel == [4.0, 6.0, 7.5, 10.0]
  assert carrot.gapSafeReaction == pytest.approx(0.3)


# ---------------------------------------------------------------- native MPC

@pytest.fixture
def native():
  return pytest.importorskip("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc")


def carrot(**kw):
  c = NS(mode="acc", comfort_brake=2.4, stop_distance=STOP, v_cruise=V80 + 3.0, stop_dist=1000.,
         trafficStopDistanceAdjust=100., trafficStopModelLeadOffset=2.,
         get_T_FOLLOW=lambda *a: 0.62, jerk_factor=1., leadAccelResponse=0,
         autoNaviSpeedDecelRate=1., lane_change_active=False, myDrivingMode=3,
         gapSafeLeadDecel=[5.0, 6.0, 7.0, 8.0], gapSafeReaction=0.6, t_follow_gap_safe_extra=0.0)
  c.__dict__.update(kw)
  return c


def solve(native, *, enabled, cycles=20, reset=False, distance=30.0, lead_speed=V80, flag=True,
          personality=log.LongitudinalPersonality.standard, mpc=None, mode="acc"):
  mpc = mpc or native.LongitudinalMpc()
  radar = log.RadarState.new_message()
  radar.leadOne = {"status": True, "radar": True, "radarTrackId": 1, "dRel": distance, "vLead": lead_speed,
                   "vRel": lead_speed - V80, "aLeadTau": 1.5, "modelProb": 1.0}
  n = native.N + 1
  c = carrot(mode=mode)
  for _ in range(cycles):
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(V80, 0.)
    kw = {"gap_safe_enabled": enabled} if flag else {}
    mpc.update(c, reset, radar, V80 + 3.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
               personality=personality, **kw)
    assert mpc.solution_status == 0
  return mpc


FIELDS = ("params", "yref", "x_sol", "a_solution", "lead_gap_margins", "base_desired_distances")


def test_off_is_the_original_path(native):
  base = solve(native, enabled=False, flag=False)
  off = solve(native, enabled=False)
  for name in FIELDS:
    np.testing.assert_array_equal(getattr(off, name), getattr(base, name))
  assert off.desired_distance == base.desired_distance and off.t_follow == base.t_follow
  assert off.gap_safe_state.weight == 0.0 and off.lead_gap is None


def test_on_sets_lead_nodes_to_the_safety_distance(native):
  off = solve(native, enabled=False)
  mpc = native.LongitudinalMpc()
  mpc.gap_safe_state.update(enabled=True, lead_decel=6.0, snap=True, dt=DT)
  on = solve(native, enabled=True, mpc=mpc)
  assert on.gap_safe_state.weight == 1.0
  assert on.source == "lead0"
  lead_nodes = np.ones(native.N + 1, dtype=bool)   # 30 m behind a same-speed lead: leadOne binds every node
  np.testing.assert_allclose(on.params[lead_nodes, 4], 0.6)
  np.testing.assert_allclose(on.params[lead_nodes, 5], 1.0)
  np.testing.assert_allclose(on.params[lead_nodes, 6], 4.0)
  # Accel limits and stop distance are unchanged; the lead obstacle uses v_lead^2 / (2 * 6).
  np.testing.assert_array_equal(on.params[:, [0, 1, 7]], off.params[:, [0, 1, 7]])
  assert np.all(on.params[:, 2] < off.params[:, 2])
  g = on.lead_gap
  expected = 0.6 * V80 + V80 ** 2 / 8.0 - V80 ** 2 / 12.0 + STOP + 0.15 * V80
  assert on.base_desired_distances[0] == pytest.approx(expected)
  assert on.desired_distance == pytest.approx(expected)
  assert on.t_follow == pytest.approx(gs.equivalent_t_follow(V80, g))
  # Comfort time rides on the reference (yref) only.
  assert np.all(on.lead_gap_margins[:, 0] > 0.0)
  # 30 m at 80 km/h is inside the new 41 m target: the plan brakes harder than TF 0.62 s.
  assert on.a_solution[1] < off.a_solution[1]


def test_cruise_nodes_keep_tf_values(native):
  mpc = native.LongitudinalMpc()
  mpc.gap_safe_state.update(enabled=True, lead_decel=6.0, snap=True, dt=DT)
  far = solve(native, enabled=True, mpc=mpc, distance=150.0, lead_speed=V80 + 8.0, cycles=5)
  off = solve(native, enabled=False, distance=150.0, lead_speed=V80 + 8.0, cycles=5)
  assert far.source == "cruise"
  cruise_nodes = ~np.isclose(far.params[:, 4], 0.6)
  assert cruise_nodes[0]
  np.testing.assert_array_equal(far.params[cruise_nodes][:, [4, 5, 6]], off.params[cruise_nodes][:, [4, 5, 6]])


@pytest.mark.parametrize("personality,decel", [(log.LongitudinalPersonality.aggressive, 5.0),
                                               (log.LongitudinalPersonality.moreRelaxed, 8.0)])
def test_gap_level_selects_the_lead_decel(native, personality, decel):
  on = solve(native, enabled=True, reset=True, personality=personality, cycles=2)
  assert on.gap_safe_state.inv_lead_decel == pytest.approx(1 / decel)
  assert on.gap_safe_state.weight == 1.0   # longitudinal off: no blend needed


def test_blended_mode_uses_the_original_path(native):
  base = solve(native, enabled=False, mode="blended", cycles=5)
  on = solve(native, enabled=True, mode="blended", cycles=5)
  assert on.lead_gap is None
  for name in FIELDS:
    np.testing.assert_array_equal(getattr(on, name), getattr(base, name))


def test_gap_level_order_matches_tfollowgap(native):
  """Cluster gap level n uses TFollowGap<n> and GapSafeLeadDecel<n> for every personality."""
  from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
  tf = CarrotPlanner.__new__(CarrotPlanner)
  tf.tFollowGap1, tf.tFollowGap2, tf.tFollowGap3, tf.tFollowGap4 = 1.0, 2.0, 3.0, 4.0
  decels = [4.5, 6.5, 8.5, 9.5]
  for name, personality in log.LongitudinalPersonality.schema.enumerants.items():
    level = int(tf._get_base_t_follow(personality, 0.0))
    mpc = native.LongitudinalMpc()
    mpc.set_accel_limits(-3.5, 1.6)
    radar = log.RadarState.new_message()
    n = native.N + 1
    mpc.update(carrot(gapSafeLeadDecel=decels), True, radar, V80, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
               personality=personality, gap_safe_enabled=True)
    assert mpc.gap_safe_state.inv_lead_decel == pytest.approx(1 / decels[level - 1]), name


def test_unknown_gap_level_turns_it_off(native):
  mpc = native.LongitudinalMpc()
  mpc.set_accel_limits(-3.5, 1.6)
  radar = log.RadarState.new_message()
  n = native.N + 1
  mpc.update(carrot(), True, radar, V80, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), personality=7, gap_safe_enabled=True)
  assert mpc.gap_safe_state.weight == 0.0 and mpc.lead_gap is None
