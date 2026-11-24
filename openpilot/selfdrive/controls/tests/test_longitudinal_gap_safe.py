"""Safety-distance gap: gap levels assume how hard the lead brakes; d_min replaces the TF distance at lead nodes."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib import longitudinal_gap_safe as gs

DT = 0.05
STOP = 3.7
V80 = 80 / 3.6


def full_gap(lead_decel=6.0, extra=0.0, t_follow=1.0, comfort_brake=2.4):
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=lead_decel, snap=True, dt=DT)
  return s.lead_gap(extra_time=extra, t_follow=t_follow, comfort_brake=comfort_brake,
                    danger_factor=0.8, inv_base_lead_decel=1 / 2.5)


def test_owner_constants():
  assert gs.LEAD_DECELS == (5.0, 6.0, 7.0, 8.0) and gs.REACTION == 0.6 and gs.GAP_LEVELS == 4
  assert gs.COMFORT_TIME == 0.35


def test_formula_at_full_weight():
  g = full_gap()
  assert g.t_follow == pytest.approx(0.6) and g.comfort_brake == pytest.approx(gs.EGO_MAX_DECEL)
  assert gs.EGO_MAX_DECEL == 4.0 and g.danger_factor == 1.0 and g.comfort_time == pytest.approx(gs.COMFORT_TIME)
  v, vl = V80, 15.0
  d_min = 0.6 * v + v ** 2 / 8.0 - vl ** 2 / 12.0 + STOP
  assert gs.min_distance(v, vl, g, STOP) == pytest.approx(d_min)
  closing = gs.CLOSING_TIME * (v - vl) + (v ** 2 - vl ** 2) * (1 / 4.8 - 1 / 8.0)
  assert gs.desired_distance(v, vl, g, STOP) == pytest.approx(d_min + gs.COMFORT_TIME * v + closing)
  # Steady following (same speed): no closing terms, equivalent TF = (target - stop) / v.
  assert gs.desired_distance(v, v, g, STOP) == pytest.approx(gs.min_distance(v, v, g, STOP) + gs.COMFORT_TIME * v)
  assert gs.equivalent_t_follow(v, g) * v == pytest.approx(gs.desired_distance(v, v, g, STOP) - STOP)


def test_closing_terms_only_count_a_slower_lead():
  g = full_gap()
  assert gs.comfort_margin(V80, V80 + 3.0, g) == pytest.approx(gs.COMFORT_TIME * V80)
  stopped = gs.comfort_margin(V80, 0.0, g)
  assert stopped == pytest.approx(gs.COMFORT_TIME * V80 + gs.CLOSING_TIME * V80 + V80 ** 2 * (1 / 4.8 - 1 / 8.0))
  # A comfort brake at or above the ego limit adds no v^2 closing term.
  assert full_gap(comfort_brake=5.0).closing_factor == 0.0


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
  g = half.lead_gap(extra_time=0.0, t_follow=1.0, comfort_brake=2.4, danger_factor=0.8, inv_base_lead_decel=0.4)
  # Linear in tF, 1/comfort_brake, 1/lead_decel and the comfort terms, so the target is the mix of the two.
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


def test_negative_extra_time_is_ignored():
  g = full_gap(extra=-1.0)
  assert g.comfort_time == pytest.approx(gs.COMFORT_TIME)


def cutin(c, **kw):
  args = {"enabled": True, "track_id": 2, "distance": 20.0, "desired": 40.0, "closing_speed": 1.0, "lead_accel": 0.0,
          "cushion": 3.0, "dt": DT}
  return c.update(**{**args, **kw})


def test_cutin_credit_keeps_the_cushion_and_shrinks():
  c = gs.CutinCredit()
  assert cutin(c) == pytest.approx(40.0 - 20.0 - 3.0)
  assert cutin(c) == pytest.approx(17.0 - gs.CUTIN_RECOVER_RATE * DT)   # same track: only shrinks
  for _ in range(round(17.0 / gs.CUTIN_RECOVER_RATE / DT) + 2):
    cutin(c, distance=5.0)
  assert c.credit == 0.0


def test_cutin_credit_only_for_a_non_braking_lead_at_about_ego_speed():
  assert cutin(gs.CutinCredit(), closing_speed=gs.CUTIN_CLOSING_MAX + 0.1) == 0.0
  assert cutin(gs.CutinCredit(), lead_accel=gs.CUTIN_LEAD_ACCEL_MIN - 0.1) == 0.0
  assert cutin(gs.CutinCredit(), lead_accel=float("nan")) == 0.0
  assert cutin(gs.CutinCredit(), distance=45.0) == 0.0
  assert cutin(gs.CutinCredit(), closing_speed=-2.0) > 0.0   # pulling away


def test_cutin_credit_ends_when_that_lead_brakes():
  c = gs.CutinCredit()
  assert cutin(c) > 0.0
  assert cutin(c, lead_accel=-1.0) == 0.0
  assert cutin(c) == 0.0   # same track: no new credit


@pytest.mark.parametrize("kw", [{"enabled": False}, {"distance": float("nan")}])
def test_cutin_credit_resets(kw):
  c = gs.CutinCredit()
  cutin(c)
  assert cutin(c, **kw) == 0.0 and c.track_id is None


# ---------------------------------------------------------------- native MPC

@pytest.fixture
def native():
  return pytest.importorskip("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc")


def carrot(**kw):
  c = NS(mode="acc", comfort_brake=2.4, stop_distance=STOP, v_cruise=V80 + 3.0, stop_dist=1000.,
         trafficStopDistanceAdjust=100., trafficStopModelLeadOffset=2.,
         get_T_FOLLOW=lambda *a: 0.62, jerk_factor=1., leadAccelResponse=0,
         autoNaviSpeedDecelRate=1., lane_change_active=False, myDrivingMode=3,
         t_follow_gap_safe_extra=0.0)
  c.__dict__.update(kw)
  return c


def radar_state(distance, lead_speed, track_id=1):
  radar = log.RadarState.new_message()
  radar.leadOne = {"status": True, "radar": True, "radarTrackId": track_id, "dRel": distance, "vLead": lead_speed,
                   "vRel": lead_speed - V80, "aLeadTau": 1.5, "modelProb": 1.0}
  return radar


def solve(native, *, cycles=20, reset=False, distance=30.0, lead_speed=V80, track_id=1,
          personality=log.LongitudinalPersonality.standard, mpc=None, mode="acc"):
  mpc = mpc or native.LongitudinalMpc()
  radar = radar_state(distance, lead_speed, track_id)
  n = native.N + 1
  c = carrot(mode=mode)
  for _ in range(cycles):
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(V80, 0.)
    mpc.update(c, reset, radar, V80 + 3.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), personality=personality)
    assert mpc.solution_status == 0
  return mpc


def test_lead_nodes_use_the_safety_distance(native):
  mpc = native.LongitudinalMpc()
  mpc.gap_safe_state.update(enabled=True, lead_decel=6.0, snap=True, dt=DT)
  on = solve(native, mpc=mpc)
  assert on.gap_safe_state.weight == 1.0 and on.source == "lead0"
  lead_nodes = np.ones(native.N + 1, dtype=bool)   # 30 m behind a same-speed lead: leadOne binds every node
  np.testing.assert_allclose(on.params[lead_nodes, 4], 0.6)
  np.testing.assert_allclose(on.params[lead_nodes, 5], 1.0)
  np.testing.assert_allclose(on.params[lead_nodes, 6], 4.0)
  g = on.lead_gap
  expected = 0.6 * V80 + V80 ** 2 / 8.0 - V80 ** 2 / 12.0 + STOP + gs.COMFORT_TIME * V80
  assert on.base_desired_distances[0] == pytest.approx(expected)
  assert on.t_follow == pytest.approx(gs.equivalent_t_follow(V80, g))
  # The lead obstacle uses v_lead^2 / (2 * 6), moved out only by what is left of the first-sight credit.
  assert on.params[0, 2] - on.cutin_credit.credit == pytest.approx(30.0 + V80 ** 2 / 12.0, abs=0.5)
  # Comfort time rides on the reference (yref) only.
  assert np.all(on.lead_gap_margins[:, 0] > 0.0)


def test_cruise_nodes_keep_tf_values(native):
  far = solve(native, distance=150.0, lead_speed=V80 + 8.0, cycles=5)
  assert far.source == "cruise"
  cruise_nodes = ~np.isclose(far.params[:, 4], 0.6)
  assert cruise_nodes[0]
  np.testing.assert_allclose(far.params[cruise_nodes][:, [4, 5, 6]], [[0.62, native.LEAD_DANGER_FACTOR, 2.4]] * int(cruise_nodes.sum()))


@pytest.mark.parametrize("personality,decel", [(log.LongitudinalPersonality.aggressive, 5.0),
                                               (log.LongitudinalPersonality.moreRelaxed, 8.0)])
def test_gap_level_selects_the_lead_decel(native, personality, decel):
  on = solve(native, reset=True, personality=personality, cycles=2)
  assert on.gap_safe_state.inv_lead_decel == pytest.approx(1 / decel)
  assert on.gap_safe_state.weight == 1.0   # longitudinal off: no blend needed


def test_blended_mode_keeps_the_tf_lead_path(native):
  on = solve(native, mode="blended", cycles=5)
  assert on.lead_gap is None
  np.testing.assert_allclose(on.params[:, 4], 0.62)


def test_gap_level_order_matches_tfollowgap(native):
  """Cluster gap level n uses TFollowGap<n> for cruise and LEAD_DECELS[n - 1] for leads, for every personality."""
  from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
  tf = CarrotPlanner.__new__(CarrotPlanner)
  tf.tFollowGap1, tf.tFollowGap2, tf.tFollowGap3, tf.tFollowGap4 = 1.0, 2.0, 3.0, 4.0
  for name, personality in log.LongitudinalPersonality.schema.enumerants.items():
    level = int(tf._get_base_t_follow(personality, 0.0))
    mpc = native.LongitudinalMpc()
    mpc.set_accel_limits(-3.5, 1.6)
    n = native.N + 1
    mpc.update(carrot(), True, log.RadarState.new_message(), V80, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
               personality=personality)
    assert mpc.gap_safe_state.inv_lead_decel == pytest.approx(1 / gs.LEAD_DECELS[level - 1]), name


def test_unknown_gap_level_keeps_the_tf_lead_path(native):
  mpc = native.LongitudinalMpc()
  mpc.set_accel_limits(-3.5, 1.6)
  n = native.N + 1
  mpc.update(carrot(), True, log.RadarState.new_message(), V80, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), personality=7)
  assert mpc.gap_safe_state.weight == 0.0 and mpc.lead_gap is None


@pytest.mark.parametrize("closing,credited", [(1.4, True), (3.0, False)])
def test_cutin_moves_the_lead_obstacle_by_the_credit(native, closing, credited):
  mpc = solve(native, distance=45.0, cycles=80)        # settled behind track 1, first-sight credit spent
  assert mpc.cutin_credit.credit == 0.0
  cut = solve(native, mpc=mpc, distance=20.0, lead_speed=V80 - closing, track_id=2, cycles=1)
  assert (cut.cutin_credit.credit > 10.0) == credited
  lead_obstacle = 20.0 + (V80 - closing) ** 2 / 12.0
  assert cut.params[0, 2] == pytest.approx(lead_obstacle + cut.cutin_credit.credit, abs=0.5)
