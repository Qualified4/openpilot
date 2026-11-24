"""ApproachSoftStop: early, soft braking for a fast approach to a slow lead seen from far."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_approach import (
  APPROACH_CLOSING_MIN,
  APPROACH_DECEL_GAIN,
  APPROACH_DECEL_MAX,
  APPROACH_DECEL_MIN,
  APPROACH_HOLD_LEAD_SPEED_MARGIN,
  APPROACH_LEAD_ACCEL_MAX,
  APPROACH_LEAD_SPEED_MAX,
  APPROACH_OBSTACLE_COST_GAIN,
  APPROACH_RAMP_TIME,
  APPROACH_RELEASE_TIME,
  APPROACH_TIME_GAP_MIN,
  ApproachState,
  approach_decel,
)

DT = 0.05
STOP = 3.7
V60 = 60 / 3.6


def args(**kw):
  a = {"enabled": True, "lead_status": True, "track_id": 1, "distance": 90.0, "lead_speed": 0.0, "lead_end_speed": 0.0,
       "lead_accel": 0.0, "v_ego": V60, "stop_distance": STOP, "dt": DT}
  a.update(kw)
  return a


def ramp(state, steps=None, **kw):
  w = 0.0
  for _ in range(steps or round(APPROACH_RAMP_TIME / DT)):
    w = state.update(**args(**kw))
  return w


def test_entry_freezes_decel_at_detection_and_ramps_in_one_second():
  s = ApproachState()
  assert s.update(**args()) == pytest.approx(DT / APPROACH_RAMP_TIME)
  expected = APPROACH_DECEL_GAIN * V60 ** 2 / (2 * (90.0 - STOP))
  assert s.decel == pytest.approx(expected)
  # Closer on the same track: the decel stays the one chosen at detection.
  assert ramp(s, distance=40.0) == pytest.approx(1.0)
  assert s.decel == pytest.approx(expected)
  assert s.obstacle_cost_factor() == pytest.approx(1.0 + APPROACH_OBSTACLE_COST_GAIN)


def test_decel_is_clipped():
  assert approach_decel(distance=400.0, lead_speed=0.0, v_ego=V60, stop_distance=STOP) == APPROACH_DECEL_MIN
  assert approach_decel(distance=30.0, lead_speed=0.0, v_ego=V60, stop_distance=STOP) == APPROACH_DECEL_MAX
  # A moving lead adds its stopping distance (2.5 m/s^2), like the MPC lead obstacle.
  moving = approach_decel(distance=100.0, lead_speed=4.0, v_ego=V60, stop_distance=STOP)
  assert moving == pytest.approx(APPROACH_DECEL_GAIN * V60 ** 2 / (2 * (100.0 + 16.0 / 5.0 - STOP)))


def test_lead_that_stops_within_the_horizon_counts_as_slow():
  s = ApproachState()
  assert s.update(**args(lead_speed=8.0, lead_end_speed=0.5, v_ego=8.0 + APPROACH_CLOSING_MIN)) > 0.0


@pytest.mark.parametrize("kw", [
  {"enabled": False},
  {"lead_status": False},
  {"lead_speed": APPROACH_LEAD_SPEED_MAX + 0.1, "lead_end_speed": 5.0},   # lead not slow
  {"lead_speed": 3.0, "v_ego": 3.0 + APPROACH_CLOSING_MIN - 0.1},         # not closing fast enough
  {"distance": (APPROACH_TIME_GAP_MIN - 0.05) * V60},                     # already close: B-type stop, untouched
  {"distance": float("nan")},
  {"lead_speed": float("inf")},
  {"v_ego": float("nan")},
  {"distance": 0.0},
  {"lead_accel": float("nan")},
  {"lead_speed": 1.0, "lead_end_speed": 6.0, "lead_accel": APPROACH_LEAD_ACCEL_MAX + 0.1},  # pulling away
  {"distance": 30.0},                                                     # needs > 2.0 m/s^2: not a soft stop
])
def test_no_entry(kw):
  s = ApproachState()
  assert s.update(**args(**kw)) == 0.0
  assert not s.active
  np.testing.assert_array_equal(s.reference_margins(ego_speeds=np.full(3, V60), following_distances=np.zeros(3),
                                                    stop_distance=STOP), np.zeros(3))
  assert s.obstacle_cost_factor() == 1.0


def test_hold_and_exit_when_the_lead_speeds_up_again():
  s = ApproachState()
  ramp(s)
  # Hysteresis: a lead creeping slightly above 15 km/h keeps the approach.
  assert s.update(**args(lead_speed=APPROACH_LEAD_SPEED_MAX + APPROACH_HOLD_LEAD_SPEED_MARGIN - 0.1,
                         lead_end_speed=6.0)) == pytest.approx(1.0)
  # Lead faster than the hold limit: released over APPROACH_RELEASE_TIME.
  assert s.update(**args(lead_speed=8.0, lead_end_speed=9.0)) == pytest.approx(1.0 - DT / APPROACH_RELEASE_TIME)
  assert ramp(s, steps=round(APPROACH_RELEASE_TIME / DT), lead_speed=8.0, lead_end_speed=9.0) == 0.0
  assert s.obstacle_cost_factor() == 1.0
  # Ego no longer closing: released the same way.
  ramp(s)
  assert ramp(s, steps=round(APPROACH_RELEASE_TIME / DT), lead_speed=2.0, v_ego=2.5) == 0.0


@pytest.mark.parametrize("kw", [
  {"enabled": False},          # setting off, longitudinal off, driver pedal
  {"lead_status": False},
  {"distance": float("nan")},
])
def test_reset_clears_state(kw):
  s = ApproachState()
  ramp(s)
  assert s.update(**args(**kw)) == 0.0
  assert s.track_id is None and not s.active and s.decel == APPROACH_DECEL_MAX


def test_track_change_reevaluates_entry():
  s = ApproachState()
  ramp(s)
  old = s.decel
  # A new track already close is not an approach.
  assert s.update(**args(track_id=2, distance=20.0)) == 0.0
  # A new far track enters again, ramping from zero with its own decel.
  assert s.update(**args(track_id=3, distance=150.0)) == pytest.approx(DT / APPROACH_RAMP_TIME)
  assert s.decel < old


def test_reference_margin_follows_the_constant_decel_curve():
  s = ApproachState()
  w = ramp(s, steps=10)
  v = np.array([V60, 12.0, 6.0, 0.0])
  base = np.array([30.0, 25.0, 30.0, 3.7])
  target = STOP + v ** 2 / (2 * s.decel)
  np.testing.assert_allclose(s.reference_margins(ego_speeds=v, following_distances=base, stop_distance=STOP),
                             w * np.maximum(0.0, target - base))


# ---------------------------------------------------------------- native MPC

@pytest.fixture
def native():
  return pytest.importorskip("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc")


CARROT = NS(mode="acc", comfort_brake=2.5, stop_distance=STOP, v_cruise=V60 + 15.0, stop_dist=1000.,
            trafficStopDistanceAdjust=100., trafficStopModelLeadOffset=2.,
            get_T_FOLLOW=lambda *a: 1.2, jerk_factor=1., leadAccelResponse=0,
            autoNaviSpeedDecelRate=1., lane_change_active=False, myDrivingMode=3)


def solve(native, *, enabled, cycles=30, reset=False, distance=90.0, lead_speed=0.0, flag=True):
  mpc = native.LongitudinalMpc()
  radar = log.RadarState.new_message()
  radar.leadOne = {"status": True, "radar": True, "radarTrackId": 1, "dRel": distance, "vLead": lead_speed,
                   "vRel": lead_speed - V60, "aLeadTau": 1.5, "modelProb": 1.0}
  n = native.N + 1
  for _ in range(cycles):
    mpc.set_accel_limits(-3.5, 1.6)
    mpc.set_cur_state(V60, 0.)
    kw = {"approach_soft_stop_enabled": enabled} if flag else {}
    mpc.update(CARROT, reset, radar, V60 + 15.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), **kw)
    assert mpc.solution_status == 0
  return mpc


def test_off_is_the_original_path(native):
  base = solve(native, enabled=False, flag=False)
  off = solve(native, enabled=False)
  for name in ("params", "yref", "x_sol", "a_solution", "lead_gap_margins"):
    np.testing.assert_array_equal(getattr(off, name), getattr(base, name))
  assert off.obstacle_cost_factor == base.obstacle_cost_factor
  assert off.desired_distance == base.desired_distance


def test_on_brakes_earlier_and_softer_without_touching_constraints(native):
  off = solve(native, enabled=False)
  on = solve(native, enabled=True)
  assert on.approach_state.weight == pytest.approx(1.0)
  # Obstacles, accel limits, TF, comfort brake, stop distance and the danger factor are unchanged.
  np.testing.assert_array_equal(on.params[:, [0, 1, 2, 4, 5, 6, 7]], off.params[:, [0, 1, 2, 4, 5, 6, 7]])
  # Reference only, at nodes where the lead binds (the cruise obstacle still binds the first ones at 90 m).
  assert np.all(on.yref[:, 0] >= off.yref[:, 0]) and np.any(on.yref[:, 0] > off.yref[:, 0])
  assert np.all(on.lead_gap_margins[:, 1] == off.lead_gap_margins[:, 1])
  assert on.obstacle_cost_factor == pytest.approx(off.obstacle_cost_factor * (1 + APPROACH_OBSTACLE_COST_GAIN))
  # The plan brakes earlier in the first second (the danger limit is not what moved it).
  assert np.all(on.a_solution[1:5] < off.a_solution[1:5])


@pytest.mark.parametrize("kw", [
  {"reset": True},                              # longitudinal off
  {"distance": 18.0},                           # close (time gap < 1.5 s): B-type stop
  {"lead_speed": 12.0},                         # moving lead, no fast closing to a slow lead
])
def test_on_matches_off_outside_the_approach(native, kw):
  off = solve(native, enabled=False, cycles=5, **kw)
  on = solve(native, enabled=True, cycles=5, **kw)
  assert on.approach_state.weight == 0.0
  for name in ("params", "yref", "x_sol", "lead_gap_margins"):
    np.testing.assert_array_equal(getattr(on, name), getattr(off, name))


def test_lead_pulling_away_releases_over_half_a_second():
  """47c 10:44: a lead starting from standstill must not be held up to 22 km/h."""
  s = ApproachState()
  ramp(s, lead_speed=0.2, lead_end_speed=0.0)
  decel = s.decel
  away = {"lead_speed": 2.0, "lead_end_speed": 8.0, "lead_accel": APPROACH_LEAD_ACCEL_MAX + 0.2}
  weights = [s.update(**args(**away)) for _ in range(round(APPROACH_RELEASE_TIME / DT))]
  assert not s.active
  np.testing.assert_allclose(weights, 1.0 - DT / APPROACH_RELEASE_TIME * np.arange(1, len(weights) + 1), atol=1e-9)
  assert s.decel == decel  # the curve keeps its decel while it fades
  # A lead that brakes back toward a stop may be approached again.
  assert s.update(**args(lead_speed=2.0, lead_end_speed=0.0, lead_accel=-0.5)) > 0.0
  # A braking/stopping prediction keeps the hold even with a positive aLeadK sample.
  ramp(s)
  assert s.update(**args(lead_speed=1.0, lead_end_speed=0.0, lead_accel=1.0)) == pytest.approx(1.0)


def test_no_entry_when_the_soft_curve_would_need_more_than_the_maximum():
  """446 5:38: entry at 26 m with a capped 2.0 curve only moved the stop closer."""
  v = 42 / 3.6
  limit = APPROACH_DECEL_GAIN * v ** 2 / (2 * APPROACH_DECEL_MAX) + STOP   # distance where the request is exactly 2.0
  assert ApproachState().update(**args(v_ego=v, distance=limit - 0.5)) == 0.0
  s = ApproachState()
  assert s.update(**args(v_ego=v, distance=limit + 0.5)) > 0.0
  assert s.decel <= APPROACH_DECEL_MAX
