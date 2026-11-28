"""Safety-distance gap: gap levels assume how hard the lead brakes; d_min replaces the TF distance at lead nodes."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib import longitudinal_gap_safe as gs

DT = 0.05
STOP = 3.7
V80 = 80 / 3.6


def full_gap(lead_decel=6.0, comfort_brake=2.4, v_ego=V80, a_ego=0.0):
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=lead_decel, snap=True, dt=DT)
  return s.lead_gap(comfort_brake=comfort_brake, v_ego=v_ego, a_ego=a_ego)


def ct(v, g):
  return float(gs.comfort_times(v, g))


def test_owner_constants():
  assert gs.LEAD_DECELS == (4.25, 5.0, 6.5, 8.0) and gs.REACTION == 0.6 and gs.GAP_LEVELS == 4
  assert gs.LEAD_DECEL_MIN == 3.0 and gs.LEAD_DECEL_MAX == 10.0
  assert gs.COMFORT_TIMES == (0.05, 0.20, 0.30, 0.35)


@pytest.mark.parametrize("kph,gaps", [(30, (9.6, 12.1, 14.5, 16.0)), (50, (14.1, 19.6, 25.5, 29.0)), (80, (21.8, 33.8, 47.4, 55.7)),
                                      (100, (27.4, 45.2, 65.8, 78.3))])
def test_steady_gap_per_level_and_speed(kph, gaps):
  """Equal-speed comfort targets of the owner's choice C (level 1 about the old TF level 1, level 4 as 0a704cf3)."""
  v = kph / 3.6
  for decel, expected in zip(gs.LEAD_DECELS, gaps, strict=True):
    g = full_gap(lead_decel=decel, v_ego=v)
    assert gs.desired_distance(v, v, g, STOP) == pytest.approx(expected, abs=0.1)


def test_level_ramp_blends_the_comfort_time():
  assert gs.level_comfort_time(1 / 4.25) == pytest.approx(0.05) and gs.level_comfort_time(1 / 8.0) == pytest.approx(0.35)
  # halfway in lead decel from level 3 (6.5) to level 4 (8.0), and outside the table clamped
  assert gs.level_comfort_time(1 / 7.25) == pytest.approx(0.325)
  assert gs.level_comfort_time(1 / 3.0) == pytest.approx(0.05) and gs.level_comfort_time(1 / 9.0) == pytest.approx(0.35)
  assert full_gap(lead_decel=5.0, v_ego=10.0).comfort_time == full_gap(lead_decel=5.0, v_ego=30.0).comfort_time == pytest.approx(0.20)


def test_formula_at_full_weight():
  g = full_gap()
  assert g.t_follow == pytest.approx(0.6) and g.comfort_brake == pytest.approx(gs.EGO_MAX_DECEL)
  assert gs.EGO_MAX_DECEL == 4.0 and g.danger_factor == 1.0 and g.comfort_time == pytest.approx(ct(V80, g))
  v, vl = V80, 15.0
  d_min = 0.6 * v + v ** 2 / 8.0 - vl ** 2 / 12.0 + STOP
  assert gs.min_distance(v, vl, g, STOP) == pytest.approx(d_min)
  closing = gs.CLOSING_TIME * (v - vl) + (v ** 2 - vl ** 2) * (1 / 4.8 - 1 / 8.0)
  assert gs.desired_distance(v, vl, g, STOP) == pytest.approx(d_min + ct(v, g) * v + closing)
  # Steady following (same speed): no closing terms, equivalent TF = (target - stop) / v.
  assert gs.desired_distance(v, v, g, STOP) == pytest.approx(gs.min_distance(v, v, g, STOP) + ct(v, g) * v)
  assert gs.steady_distance(v, v, g, STOP) == pytest.approx(gs.desired_distance(v, v, g, STOP))
  assert gs.equivalent_t_follow(v, g) * v == pytest.approx(gs.desired_distance(v, v, g, STOP) - STOP)


def test_closing_terms_only_count_a_slower_lead():
  g = full_gap()
  assert gs.comfort_margin(V80, V80 + 3.0, g) == pytest.approx(ct(V80, g) * V80)
  stopped = gs.comfort_margin(V80, 0.0, g)
  assert stopped == pytest.approx(ct(V80, g) * V80 + gs.CLOSING_TIME * V80 + V80 ** 2 * (1 / 4.8 - 1 / 8.0))
  # A comfort brake at or above the ego limit adds no v^2 closing term.
  assert full_gap(comfort_brake=5.0).closing_factor == 0.0


def test_comfort_margin_fades_out_near_standstill():
  g = full_gap()
  lo, hi = gs.COMFORT_FADE_SPEEDS

  def full(v, vl):
    return ct(v, g) * v + gs.CLOSING_TIME * max(0.0, v - vl) + max(0.0, v ** 2 - vl ** 2) * (1 / 4.8 - 1 / 8.0)
  # Above the fade the margin is the full formula; at or below its low end only d_min is left.
  assert gs.comfort_margin(hi, 0.0, g) == pytest.approx(full(hi, 0.0))
  assert gs.comfort_margin(lo, 0.0, g) == 0.0
  assert gs.desired_distance(lo, 0.0, g, STOP) == pytest.approx(gs.min_distance(lo, 0.0, g, STOP))
  mid = (lo + hi) / 2
  assert gs.comfort_margin(mid, 0.0, g) == pytest.approx(0.5 * full(mid, 0.0))
  # Per node on an array of plan speeds, never more than the full margin.
  v = np.linspace(0.0, 6.0, 13)
  m = gs.comfort_margin(v, np.zeros_like(v), g)
  assert np.all(m <= np.array([full(x, 0.0) for x in v]) + 1e-12) and np.all(np.diff(m) >= 0.0)
  # While ego brakes hard the full margin stays; the fade comes back over COMFORT_FADE_ACCELS.
  hard_lo, hard_hi = gs.COMFORT_FADE_ACCELS
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=6.0, snap=True, dt=DT)
  hard = s.lead_gap(comfort_brake=2.4, a_ego=hard_lo - 0.5)
  assert hard.fade_weight == 0.0 and gs.comfort_margin(lo, 0.0, hard) == pytest.approx(full(lo, 0.0))
  half = s.lead_gap(comfort_brake=2.4, a_ego=(hard_lo + hard_hi) / 2)
  assert gs.comfort_margin(lo, 0.0, half) == pytest.approx(0.5 * full(lo, 0.0))
  assert s.lead_gap(comfort_brake=2.4, a_ego=hard_hi).fade_weight == 1.0


def test_on_and_off_at_once():
  s = gs.GapSafeState()
  assert not s.active
  s.update(enabled=True, lead_decel=6.0, snap=False, dt=DT)
  assert s.active and s.inv_lead_decel == pytest.approx(1 / 6.0)
  s.update(enabled=False, lead_decel=6.0, snap=False, dt=DT)
  assert not s.active and s.inv_lead_decel is None


def test_snap_while_not_engaged():
  s = gs.GapSafeState()
  s.update(enabled=True, lead_decel=8.0, snap=True, dt=DT)
  assert s.active and s.inv_lead_decel == pytest.approx(1 / 8.0)


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
  assert not s.active and s.inv_lead_decel is None


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


def test_cutin_credit_not_for_the_same_car_under_a_new_track_id():
  c = gs.CutinCredit()
  cutin(c, track_id=1, distance=30.0, desired=20.0)   # settled, no shortfall
  assert cutin(c, track_id=5, distance=30.0 + gs.SAME_LEAD_DISTANCE - 0.1) == 0.0
  assert cutin(c, track_id=6, distance=20.0) > 0.0     # a different car closer in


def test_cutin_credit_not_for_a_lead_seen_again():
  c = gs.CutinCredit()
  cutin(c, distance=30.0, desired=20.0)
  assert c.hold(DT) == 0.0                               # dropout: the last lead is kept
  assert cutin(c, distance=20.0) == 0.0                  # same track back: no credit
  # After CUTIN_HOLD_EXPIRE without a lead, the old lead is forgotten.
  c = gs.CutinCredit()
  cutin(c, distance=30.0, desired=20.0)
  for _ in range(round(gs.CUTIN_HOLD_EXPIRE / DT) + 2):
    c.hold(DT)
  assert cutin(c, distance=20.0) > 0.0
  # A lane change (or not engaged) follows the lead without crediting it, so its end gives none.
  c = gs.CutinCredit()
  assert cutin(c, track_id=3, grant=False) == 0.0
  assert cutin(c, track_id=3) == 0.0


def test_cutin_credit_vision_leads_are_not_one_car():
  c = gs.CutinCredit()
  cutin(c, track_id=-1, distance=30.0, desired=20.0)
  assert cutin(c, track_id=-1, distance=20.0) > 0.0      # another vision-only lead, 10 m closer


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
         autoNaviSpeedDecelRate=1., lane_change_active=False, myDrivingMode=3)
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
  decel = gs.LEAD_DECELS[1]   # the default personality's level
  mpc.gap_safe_state.update(enabled=True, lead_decel=decel, snap=True, dt=DT)
  on = solve(native, mpc=mpc)
  assert on.gap_safe_state.active and on.source == "lead0"
  lead_nodes = np.ones(native.N + 1, dtype=bool)   # 30 m behind a same-speed lead: leadOne binds every node
  np.testing.assert_allclose(on.params[lead_nodes, 4], 0.6)
  np.testing.assert_allclose(on.params[lead_nodes, 5], 1.0)
  np.testing.assert_allclose(on.params[lead_nodes, 6], 4.0)
  g = on.lead_gap
  expected = 0.6 * V80 + V80 ** 2 / 8.0 - V80 ** 2 / (2 * decel) + STOP + ct(V80, g) * V80
  assert on.base_desired_distances[0] == pytest.approx(expected)
  assert on.t_follow == pytest.approx(gs.equivalent_t_follow(V80, g))
  # The lead obstacle uses v_lead^2 / (2 * decel), moved out by at most what is left of the first-sight credit.
  obstacle = 30.0 + V80 ** 2 / (2 * decel)
  assert obstacle - 0.5 <= on.params[0, 2] <= obstacle + on.cutin_credit.credit + 0.5
  # Comfort time rides on the reference (yref) only.
  assert np.all(on.lead_gap_margins[:, 0] > 0.0)


def test_cruise_nodes_keep_tf_values(native):
  far = solve(native, distance=150.0, lead_speed=V80 + 8.0, cycles=5)
  assert far.source == "cruise"
  cruise_nodes = ~np.isclose(far.params[:, 4], 0.6)
  assert cruise_nodes[0]
  np.testing.assert_allclose(far.params[cruise_nodes][:, [4, 5, 6]], [[0.62, native.LEAD_DANGER_FACTOR, 2.4]] * int(cruise_nodes.sum()))


@pytest.mark.parametrize("personality,decel", [(log.LongitudinalPersonality.aggressive, gs.LEAD_DECELS[0]),
                                               (log.LongitudinalPersonality.moreRelaxed, gs.LEAD_DECELS[3])])
def test_gap_level_selects_the_lead_decel(native, personality, decel):
  on = solve(native, reset=True, personality=personality, cycles=2)
  assert on.gap_safe_state.inv_lead_decel == pytest.approx(1 / decel)
  assert on.gap_safe_state.active


def test_blended_mode_constrains_at_the_comfort_target(native):
  """Blended keeps the lead as a constraint only; with the safety-distance gap it sits at the ACC comfort target."""
  on = solve(native, mode="blended", cycles=5, mpc=native.LongitudinalMpc(mode="blended"))
  g = on.lead_gap
  assert g is not None and on.source == "lead0"
  np.testing.assert_allclose(on.params[:, 4], 0.6)
  np.testing.assert_allclose(on.params[:, 5], 1.0)
  np.testing.assert_allclose(on.params[:, 6], 4.0)
  # obstacle - safe distance = lead position - constraint distance. A lead first seen inside the steady comfort
  # target (d_min + comfort time, no closing terms) gets the cut-in credit, never looser than the TF target.
  steady = gs.desired_distance(V80, V80, g, STOP)
  tf_target = native.desired_follow_distance(V80, V80, 2.4, STOP, 0.62)
  floor = on.params[0, 2] - gs.safe_distance(V80, g, STOP)
  assert 30.0 - steady - 0.5 <= floor <= 30.0 - tf_target + 0.5
  # Once the credit has run out, the constraint sits at the steady comfort target.
  on = solve(native, mode="blended", cycles=400, mpc=on)
  floor = on.params[0, 2] - gs.safe_distance(V80, g, STOP)
  assert on.cutin_credit.credit == 0.0 and floor == pytest.approx(30.0 - steady, abs=0.5)


def test_lead_inside_the_comfort_target_binds(native):
  """Between d_min and the comfort target the lead, not cruise, binds the first node (source lead0)."""
  g = full_gap(lead_decel=gs.LEAD_DECELS[1])
  d = float(0.5 * (gs.min_distance(V80, V80, g, STOP) + gs.desired_distance(V80, V80, g, STOP)) + 1.0)
  on = solve(native, distance=d, cycles=3)
  assert on.source == "lead0"
  np.testing.assert_allclose(on.params[0, 4:7], [0.6, 1.0, 4.0])


def test_faster_lead_beyond_the_target_never_binds(native):
  """D1: a lead looser than cruise even with its comfort margins leaves cruise in charge."""
  on = solve(native, distance=80.0, lead_speed=V80 + 6.0, cycles=3)
  assert on.source == "cruise"


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
  assert not mpc.gap_safe_state.active and mpc.lead_gap is None


@pytest.mark.parametrize("closing,credited", [(1.4, True), (3.0, False)])
def test_cutin_credit_never_lowers_the_danger_floor_below_tf(native, closing, credited):
  mpc = solve(native, distance=45.0, cycles=80)        # settled behind track 1, first-sight credit spent
  assert mpc.cutin_credit.credit == 0.0
  cut = solve(native, mpc=mpc, distance=20.0, lead_speed=V80 - closing, track_id=2, cycles=1)
  assert (cut.cutin_credit.credit > 10.0) == credited
  vl = V80 - closing
  shift = np.broadcast_to(cut.cutin_shift, (native.N + 1,))[0]   # obstacle move at node 0 (ego at V80)
  d_min = 0.6 * V80 + V80 ** 2 / 8.0 - vl ** 2 / 12.0 + STOP
  tf_floor = 0.8 * (V80 ** 2 / 4.8 + 0.62 * V80 + STOP) - vl ** 2 / 5.0
  assert shift == pytest.approx(min(cut.cutin_credit.credit, max(0.0, d_min - tf_floor)), abs=0.5)
  assert d_min - shift >= tf_floor - 0.5               # danger floor: d_min - shift, never below TF
  lead_nodes = np.isclose(cut.params[:, 4], 0.6)
  np.testing.assert_allclose(cut.params[lead_nodes, 5], 1.0)
  assert np.all(cut.lead_gap_margins[:, 0] >= -1e-9)   # the reference never goes below that floor
  tf_target = V80 ** 2 / 4.8 + 0.62 * V80 + STOP - vl ** 2 / 5.0
  assert d_min - shift + cut.lead_gap_margins[0, 0] >= min(tf_target, gs.desired_distance(V80, vl, cut.lead_gap, STOP)) - 0.5


def nolead_mpc(native, v, v_cruise, personality, cycles=40, lead_speed=None):
  mpc = native.LongitudinalMpc()
  n = native.N + 1
  radar = log.RadarState.new_message() if lead_speed is None else radar_state(60.0, lead_speed)
  c = carrot(v_cruise=v_cruise)
  for _ in range(cycles):
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(v, 0.)
    mpc.update(c, False, radar, v_cruise, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), personality=personality)
    assert mpc.solution_status == 0
  return mpc


@pytest.mark.parametrize("personality", list(log.LongitudinalPersonality.schema.enumerants.values()))
def test_no_lead_follows_cruise_with_tf_values(native, personality):
  """The fake far lead keeps the TF computation, so cruise binds as before at every gap level."""
  v = 110 / 3.6
  mpc = nolead_mpc(native, v, v, personality)
  assert mpc.source == "cruise"
  np.testing.assert_allclose(mpc.params[:, 4], 0.62)
  np.testing.assert_allclose(mpc.params[:, 5], native.LEAD_DANGER_FACTOR)
  assert np.all(mpc.lead_gap_margins == 0.0)
  assert mpc.a_solution[0] < 0.05


def test_lead_faster_than_cruise_does_not_bind(native):
  """A real lead faster than the set speed is compared on its own target, not its raw obstacle."""
  v = 100 / 3.6
  mpc = nolead_mpc(native, v, v, log.LongitudinalPersonality.aggressive, lead_speed=v + 4.0)
  assert mpc.source == "cruise"
  assert mpc.a_solution[0] < 0.05


def test_comfort_binding_only_takes_over_from_cruise(native):
  """N1: a lead's comfort target replaces only a cruise binding; leadTwo / traffic / Tesla stops keep theirs."""
  # columns: leadOne, leadTwo, cruise, traffic stop
  targets = np.array([[50., 90., 40., 90.],    # cruise closest, leadOne preferred -> leadOne
                      [50., 90., 60., 45.],    # traffic stop closest -> stays the traffic stop
                      [50., 48., 90., 90.],    # leadTwo closest -> stays leadTwo
                      [50., 90., 40., 90.]])   # leadOne looser than cruise even with its margin -> cruise
  margins = np.array([[15., 0.], [15., 0.], [15., 0.], [5., 0.]])
  preferred = targets.copy()
  preferred[:, :2] -= margins
  np.testing.assert_array_equal(native.bind_obstacles(targets, preferred), [0, 3, 1, 2])


def test_custom_lead_decel_updates_mpc_desired_distance_clamps_and_ramps(native):
  """LeadDecelGap changes propagate into MPC desired_distance, clamp to [3.0, 10.0], and ramp when lengthening."""
  n = native.N + 1
  radar = radar_state(40.0, V80)

  def run_step(mpc, lead_decel, *, reset=False, personality=log.LongitudinalPersonality.standard):
    c = carrot(get_lead_decel=lambda lvl: lead_decel)
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(V80, 0.)
    mpc.update(c, reset, radar, V80 + 3.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), personality=personality)
    assert mpc.solution_status == 0
    return mpc

  # Lower lead_decel -> shorter desired_distance; higher lead_decel -> wider desired_distance
  low = run_step(native.LongitudinalMpc(), 4.0, reset=True)
  high = run_step(native.LongitudinalMpc(), 8.0, reset=True)
  assert high.base_desired_distances[0] > low.base_desired_distances[0]
  assert low.gap_safe_state.inv_lead_decel == pytest.approx(1 / 4.0)
  assert high.gap_safe_state.inv_lead_decel == pytest.approx(1 / 8.0)

  # Out-of-range values clamp to [3.0, 10.0], non-finite falls back to level default
  clamped_lo = run_step(native.LongitudinalMpc(), 1.2, reset=True)
  clamped_hi = run_step(native.LongitudinalMpc(), 15.0, reset=True)
  fallback_nan = run_step(native.LongitudinalMpc(), float("nan"), reset=True)
  assert clamped_lo.gap_safe_state.inv_lead_decel == pytest.approx(1 / gs.LEAD_DECEL_MIN)
  assert clamped_hi.gap_safe_state.inv_lead_decel == pytest.approx(1 / gs.LEAD_DECEL_MAX)
  assert fallback_nan.gap_safe_state.inv_lead_decel == pytest.approx(1 / gs.LEAD_DECELS[1])

  # Live increase while driving ramps smoothly via LEAD_DECEL_INV_RATE; decrease applies at once
  mpc = run_step(native.LongitudinalMpc(), 4.5, reset=True)
  before_dist = float(mpc.base_desired_distances[0])
  run_step(mpc, 8.0, reset=False)
  assert mpc.gap_safe_state.inv_lead_decel == pytest.approx(1 / 4.5 - gs.LEAD_DECEL_INV_RATE * DT)
  assert before_dist < mpc.base_desired_distances[0] < high.base_desired_distances[0]
  run_step(mpc, 4.5, reset=False)
  assert mpc.gap_safe_state.inv_lead_decel == pytest.approx(1 / 4.5)


def test_carrot_planner_reads_lead_decel_gap_and_preserves_tf_fallbacks():
  """CarrotPlanner._params_update reads LeadDecelGap1~4 with clamping and no longer reads TFollowGap1~4 or SpeedTFFactor."""
  from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner, DrivingMode, DrivingModeDetector

  requested_keys = []
  float_values = {
    "LeadDecelGap1": 250.0,   # below 3.0 -> clamp to 3.0
    "LeadDecelGap2": 525.0,   # 5.25
    "LeadDecelGap3": 650.0,   # 6.50
    "LeadDecelGap4": 1200.0,  # above 10.0 -> clamp to 10.0
    "DynamicTFollowLC": 100.0,
    "TFollowDecelBoost": 0.0,
  }
  int_values = {
    "MyDrivingMode": DrivingMode.Normal.value,
    "MyDrivingModeAuto": 0,
    "LeadAccelResponse": 2,
    "LeadAccelResponseTF1": -1,
    "LeadAccelResponseTF2": 3,
    "LeadAccelResponseTF3": -1,
    "LeadAccelResponseTF4": 1,
  }

  class FakeParams:
    def get_float(self, key):
      requested_keys.append(key)
      return float_values[key]
    def get_int(self, key):
      requested_keys.append(key)
      return int_values[key]

  planner = CarrotPlanner.__new__(CarrotPlanner)
  planner.params = FakeParams()
  planner.frame = 19
  planner.params_count = 19
  planner.myDrivingMode = planner.myDrivingMode_last = DrivingMode.Normal
  planner.myDrivingModeAuto = 0
  planner.myDrivingMode_disable_auto = False
  planner.drivingModeDetector = DrivingModeDetector()
  planner.tFollowGap1, planner.tFollowGap2, planner.tFollowGap3, planner.tFollowGap4 = 1.1, 1.3, 1.45, 1.6
  planner.speedTFFactor = 10
  planner.leadDecelGap = list(gs.LEAD_DECELS)

  planner._params_update()

  assert planner.leadDecelGap == pytest.approx([3.0, 5.25, 6.5, 10.0])
  assert [planner.get_lead_decel(i) for i in range(4)] == pytest.approx([3.0, 5.25, 6.5, 10.0])
  assert planner.get_lead_decel(99) == 0.0
  # Removed params must not be read; internal fallback constants stay intact
  assert not any(k in requested_keys for k in ("TFollowGap1", "TFollowGap2", "TFollowGap3", "TFollowGap4", "SpeedTFFactor"))
  assert (planner.tFollowGap1, planner.tFollowGap2, planner.tFollowGap3, planner.tFollowGap4) == (1.1, 1.3, 1.45, 1.6)
  assert planner.speedTFFactor == 10
  # LeadAccelResponseTF1~4 remain intact
  assert planner.leadAccelResponseBase == 2
  assert planner.leadAccelResponseTF == [-1, 3, -1, 1]


def test_p1_parabolic_landing_and_reaction_attenuation(native):
  """P1: broadband comfort fade + active-braking reaction attenuation without touching cruise safety."""
  assert gs.COMFORT_FADE_SPEEDS == (0.8, 5.2)
  assert gs.COMFORT_FADE_ACCELS == (-2.9, -2.4)

  # Cruising / sudden-brake onset (a_ego >= -0.15): REACTION 0.60 s is 100% preserved
  assert gs.stopping_reaction_time(1.0, 0.0, a_ego=0.0) == pytest.approx(gs.REACTION)
  assert gs.stopping_reaction_time(1.0, 0.0, a_ego=-0.10) == pytest.approx(gs.REACTION)
  assert gs.parabolic_landing_margin(1.0, 0.0, a_ego=0.0) == pytest.approx(0.0)

  # Active braking (a_ego = -0.60) behind stopped lead at low speed: REACTION attenuates to 0.048 s
  t_eff = gs.stopping_reaction_time(0.3, 0.0, a_ego=-0.60)
  assert t_eff == pytest.approx(gs.REACTION * 0.08, abs=1e-6)
  d_parab = gs.parabolic_landing_margin(1.0, 0.0, a_ego=-0.60, a_lamp=-0.45)
  assert d_parab > 0.0

  # Moving lead (v_lead >= 1.5 m/s) keeps full REACTION 0.60 s even while braking
  assert gs.stopping_reaction_time(1.0, 1.6, a_ego=-0.80) == pytest.approx(gs.REACTION)

  # Native MPC binds attenuated params[:, 4] when actively braking behind a stopped lead
  mpc = native.LongitudinalMpc()
  n = native.N + 1
  radar = log.RadarState.new_message()
  radar.leadOne = {"status": True, "radar": True, "radarTrackId": 1, "dRel": 4.2, "vLead": 0.0,
                   "vRel": -1.2, "aLeadK": 0.0, "aLeadTau": 1.5, "modelProb": 1.0}
  c = carrot(v_cruise=15.0)
  for _ in range(5):
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(1.2, -0.6)
    mpc.update(c, False, radar, 15.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
               personality=log.LongitudinalPersonality.standard)
    assert mpc.solution_status == 0
  assert mpc.source == "lead0"
  assert mpc.params[0, 4] < 0.30
  assert np.min(mpc.params[:, 4]) < 0.15
