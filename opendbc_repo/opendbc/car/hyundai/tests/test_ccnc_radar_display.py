"""CCNC display-only regressions, isolated from platform-specific CAN/Params imports."""
import ast
import math
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pytest


def load_helpers():
  path = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[1] / 'hyundaicanfd.py'))
  names = {'_ccnc_valid_boundary', '_CcncRadarDisplayTracker', '_CcncRadarPositionFilter', 'NoiseFilter'}
  nodes = [n for n in ast.parse(path.read_text(encoding='utf8')).body
           if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
  env = dict(math=math, np=np, deque=deque, CV=N(MS_TO_KPH=3.6))
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
  return env


H = load_helpers()
Tracker = H['_CcncRadarDisplayTracker']


def point(track_id=1, x=10., y=0., speed=8.):
  return N(trackId=track_id, radarSource='frontRadar', dRel=x, yRel=y, vRel=0., vLead=speed)


def observe(tracker, points, start=0):
  for frame in range(start, start + 21, 5):
    tracker.observe(N(points=points), frame)


def model(y=0., x=11.52, probability=.99, path_y=(0., 0., 0.)):
  return N(position=N(x=[0., 10., 30.], y=list(path_y)),
           leadsV3=[N(prob=probability, x=[x], y=[-y], v=[8.])])


def test_same_radar_publication_does_not_mature():
  t = Tracker()
  data = N(points=[point()])
  for frame in range(50):
    t.observe(data, frame)
  assert not t.stable(1)


def test_missing_publication_or_discontinuity_resets_maturity():
  t = Tracker()
  observe(t, [point()])
  assert t.stable(1)
  t.observe(N(points=[point(x=50.)]), 25)
  assert not t.stable(1)
  t.observe(None, 26)
  assert not t.tracks
  assert t.selected == (None, None, None)


def test_disappeared_track_cannot_be_held_by_id():
  t = Tracker()
  observe(t, [point()])
  t.finish(point(), None, None, 0., True, 20)
  t.observe(N(points=[]), 25)
  assert t.path_lead(model(), -100)[0] is None


def test_followed_vehicle_survives_turn_outside_model_path_with_vision_support():
  t = Tracker()
  p = point(x=7., y=4.)
  observe(t, [p])
  t.finish(p, None, None, 0., True, 20)
  chosen, aligned = t.path_lead(model(y=4., x=14., path_y=(0., -.3, -2.)), 0.)
  assert chosen.trackId == 1
  assert aligned == pytest.approx(3.79)


def test_departing_vehicle_loses_front_slot_without_vision_support():
  t = Tracker()
  p = point(y=4.)
  observe(t, [p])
  t.finish(p, None, None, 0., True, 20)
  assert t.path_lead(model(y=0.), 0.)[0] is None


def test_new_lane_free_candidate_requires_vision_and_nearest_candidate_wins():
  t = Tracker()
  observe(t, [point(1, 10.), point(2, 12.)])
  assert t.path_lead(model(probability=0.), 0.)[0] is None
  assert t.path_lead(model(), 0.)[0].trackId == 1


def test_turning_path_uses_forward_prefix_without_extrapolating():
  t = Tracker()
  observe(t, [point()])
  md = model()
  md.position = N(x=[0., 15., 12.], y=[0., 0., -10.])
  assert t.path_lead(md, 0.)[0].trackId == 1
  md.position = N(x=[0., 5., 3.], y=[0., 0., -10.])
  assert t.path_lead(md, 0.)[0] is None


@pytest.mark.parametrize('xs,ys', [([0., 10.], [0.]), ([0., float('nan')], [0., 0.]), ([], [])])
def test_invalid_model_geometry_is_not_used(xs, ys):
  t = Tracker()
  observe(t, [point()])
  md = model()
  md.position = N(x=xs, y=ys)
  assert t.path_lead(md, 0.)[0] is None


def test_reliable_lane_recovers_slot_without_waiting():
  t = Tracker()
  assert not t.lane_available(.05, 0)
  assert not t.lane_available(.15, 20)
  assert t.lane_available(.5, 21)


def test_reference_switch_blends_without_losing_raw_lateral_motion():
  t = Tracker()
  p = point(y=1.)
  observe(t, [p])
  assert t.filter_position(p, 1., ('lane', True), 20)[1] == 1.
  p.yRel = 1.1
  y = t.filter_position(p, 5.1, ('path', False), 21)[1]
  assert 1. <= y < 1.15
  assert 1. < t.filter_position(p, 5.1, ('path', False), 22)[1] < 1.2
  assert t.filter_position(point(2, y=5.1), 5.1, ('path', False), 22)[1] == 5.1


@pytest.mark.parametrize('slots', [(1, 0, 2), (2, 0, 1)])
def test_same_track_keeps_observation_and_filters_across_slots(slots):
  t = Tracker()
  p = point(y=1.)
  observe(t, [p])
  t.filter_position(p, 1., ('lane', True), 20)
  distance_filter, lateral_filter = t.positions[p.trackId][-2:]
  for frame, slot in enumerate(slots, 21):
    points = [None, None, None]
    points[slot] = p
    t.finish(*points, 1., True, frame)
    t.filter_position(p, 1., ('lane', True), frame)
    assert t.stable(p.trackId)
    assert t.selected[slot] == p.trackId
    assert t.positions[p.trackId][-2] is distance_filter
    assert t.positions[p.trackId][-1] is lateral_filter


def test_brief_unselected_interval_preserves_track_filters_but_disappearance_drops_them():
  t = Tracker()
  p = point()
  observe(t, [p])
  t.filter_position(p, 1., ('lane', True), 20)
  filt = t.positions[1][-1]
  t.finish(None, None, None, 0., True, 21)
  t.observe(N(points=[p]), 25)
  t.filter_position(p, 1.1, ('lane', True), 25)
  assert t.positions[1][-1] is filt
  t.observe(N(points=[]), 30)
  assert not t.positions


def test_low_speed_crossing_has_no_forced_two_point_five_meter_position():
  t = Tracker()
  p = point(y=1.6, speed=.5)
  observe(t, [p])
  assert t.filter_position(p, 1.6, ('lane', True), 20)[1] == pytest.approx(1.6)


def test_reused_id_with_discontinuous_position_does_not_inherit_filter():
  t = Tracker()
  p = point()
  observe(t, [p])
  t.filter_position(p, 1., ('lane', True), 20)
  p.dRel = 50.
  t.observe(N(points=[p]), 25)
  assert t.filter_position(p, 3., ('lane', True), 25) == pytest.approx((50., 3.))


def test_distance_filter_can_reset_on_target_change():
  f = H['NoiseFilter'](3, 100., [.3, .9], [1., 4.])
  f.apply(100.)
  f.reset(98.)
  assert f.apply(98.) == 98.


def test_mutated_radar_point_does_not_overwrite_previous_observation():
  t = Tracker()
  p = point()
  observe(t, [p])
  assert t.stable(1)
  p.dRel = 50.
  t.observe(N(points=[p]), 25)
  assert not t.stable(1)


def test_recent_identity_survives_unselected_frame_without_holding_old_point():
  t = Tracker()
  p = point(y=1.6)
  observe(t, [p])
  t.finish(p, None, None, 1.6, False, 20)
  t.finish(None, None, None, 0., False, 21)
  updated = point(y=1.7)
  t.observe(N(points=[updated]), 25)
  chosen, y = t.path_lead(model(probability=0.), 0.)
  assert chosen is updated
  assert y == pytest.approx(1.7)
  for frame in range(30, 56, 5):
    t.observe(N(points=[updated]), frame)
  assert not t.recently_selected(1)
  assert t.path_lead(model(probability=0.), 0.)[0] is None


def test_missing_or_reused_id_drops_recent_front_identity():
  t = Tracker()
  p = point()
  observe(t, [p])
  t.finish(p, None, None, 0., True, 20)
  assert t.recently_selected(1, front_only=True)
  t.observe(N(points=[point(x=50.)]), 25)
  assert not t.recently_selected(1, front_only=True)
  t.observe(None, 26)
  assert not t.recent_front and not t.recent_selected


def test_lane_recovery_override_needs_recent_front_and_vision():
  t = Tracker()
  p = point(y=1.6)
  observe(t, [p])
  t.finish(None, p, None, 0., True, 20)
  assert t.path_lead(model(y=1.6), 0., require_vision=True, front_only=True)[0] is None
  t.observe(N(points=[p]), 21)
  t.finish(p, None, None, 1.6, False, 21)
  assert t.path_lead(model(y=1.6, probability=0.), 0., require_vision=True, front_only=True)[0] is None
  assert t.path_lead(model(y=1.6), 0., require_vision=True, front_only=True)[0] is p


def test_recent_identity_does_not_accept_departing_vehicle():
  t = Tracker()
  p = point(y=1.6)
  observe(t, [p])
  t.finish(p, None, None, 1.6, False, 20)
  t.finish(None, None, None, 0., False, 21)
  t.observe(N(points=[point(y=2.0)]), 25)
  assert t.path_lead(model(probability=0.), 0.)[0] is None



def test_strong_vision_reacquires_mature_track_without_recent_identity():
  t = Tracker()
  p = point(y=1.7)
  observe(t, [p])
  assert t.path_lead(model(y=1.7, probability=.7), 0.)[0] is None
  assert t.path_lead(model(y=1.7, probability=.9), 0.)[0] is p
  t.observe(N(points=[point(y=3.)]), 25)
  assert t.path_lead(model(y=3., probability=.99), 0.)[0] is None


@pytest.mark.parametrize('distance,dy,dv,prob', [(4.,1.7,0.,.99), (0.,2.1,0.,.99),
                                                (0.,1.7,3.1,.99), (0.,1.7,0.,.79)])
def test_wider_turn_match_requires_strict_range_speed_and_probability(distance, dy, dv, prob):
  t = Tracker()
  p = point(y=1.9)
  observe(t, [p])
  md = model(y=p.yRel-dy, x=11.52+distance, probability=prob)
  md.leadsV3[0].v = [8.+dv]
  assert t.path_lead(md, 0.)[0] is None


def test_fallback_failure_keeps_lane_selected_front():
  source = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[1] / 'hyundaicanfd.py')).read_text(encoding='utf8')
  start = source.index('          if CS.live_tracks is not None and (not ff_lane_mode')
  end = source.index('          # A retained FF', start)
  import textwrap
  candidate = point()
  tracker = N(path_lead=lambda *args, **kwargs: (None, 0.), recently_selected=lambda *args, **kwargs: True)
  env = dict(CS=N(live_tracks=object()), ff_lane_mode=False, selected_lane_prob=.15,
             a_ego_kph=0., v_ego_kph=20., min_front_lead_speed=-100., np=np, display_tracker=tracker, md=model(),
             ff_lead=candidate, ff_yRel=1.)
  exec(textwrap.dedent(source[start:end]), env)
  assert env['ff_lead'] is candidate
  assert env['ff_yRel'] == 1.


def test_failed_fallback_does_not_introduce_new_target_from_weak_lanes():
  source = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[1] / 'hyundaicanfd.py')).read_text(encoding='utf8')
  start = source.index('          if CS.live_tracks is not None and (not ff_lane_mode')
  end = source.index('          # A retained FF', start)
  import textwrap
  env = dict(CS=N(live_tracks=object()), ff_lane_mode=False, selected_lane_prob=.15,
             a_ego_kph=0., v_ego_kph=20., min_front_lead_speed=-100., np=np, md=model(), ff_lead=point(), ff_yRel=1.,
             display_tracker=N(path_lead=lambda *a, **kw: (None, 0.),
                               recently_selected=lambda *a, **kw: False))
  exec(textwrap.dedent(source[start:end]), env)
  assert env['ff_lead'] is None


@pytest.mark.parametrize('distance', [True, False])
def test_specialized_position_filter_matches_general_filter(distance):
  fast = H['_CcncRadarPositionFilter'](1., distance)
  reference = H['NoiseFilter'](3, 1., [.3, .9] if distance else .3,
                               [1., 4.] if distance else .6)
  rng = np.random.default_rng(423)
  for value in np.cumsum(rng.normal(0., .4, 3000)):
    assert fast.apply(float(value)) == reference.apply(float(value))
  for value in [100., 0., 1., 1.6, 1.6000001, -50., 2., 3., 4.]:
    assert fast.apply(value) == reference.apply(value)


def test_scalar_speed_thresholds_match_original_interpolation():
  speeds = list(np.linspace(-10., 150., 2001)) + [float('inf'), float('-inf')]
  for boundary in [0., 10., 30., 40., 100.]:
    speeds.extend([boundary, np.nextafter(boundary, -np.inf), np.nextafter(boundary, np.inf)])
  for acceleration in [-4., -3., 0.]:
    for v in speeds:
      expected = (-100 if acceleration < -3 else np.interp(v, [30,40,100], [-100,0,20]),
                  np.interp(v, [0,30,100], [2,10,20]), np.interp(v,[10,40],[-1,10]))
      assert Tracker.speed_thresholds(v, acceleration) == pytest.approx(expected, abs=1e-12)
  assert all(math.isnan(v) for v in Tracker.speed_thresholds(float('nan'), 0.))


def lane_model(stamp=1, offset=0.):
  curve = lambda y: N(x=[0., 10., 30.], y=[y, y+offset, y+offset*2])
  return N(timestampEof=stamp, laneLineProbs=[.9]*4,
           laneLines=[curve(-5.), curve(-1.5), curve(1.5), curve(5.)],
           roadEdges=[curve(-7.), curve(7.)])


def test_projection_cache_refreshes_for_new_model_and_reused_point_objects():
  t = Tracker()
  p = point(x=10.)
  live = N(points=[p])
  t.observe(live, 0)
  md = lane_model()
  t.lane_probabilities(md)
  original = t.lane_projection(live)
  assert t.lane_projection(live) is original
  md2 = lane_model(2, 1.)
  t.lane_probabilities(md2)
  updated = t.lane_projection(live)
  assert updated is not original
  assert updated[2][0][0] == -.5
  p.dRel = 20.
  next_live = N(points=[p])
  t.observe(next_live, 5)
  assert t.lane_projection(next_live)[2][0][0] == 0.
  p.dRel = 10.
  t.observe(next_live, 30)  # A control gap must also invalidate same-object caches.
  assert t.lane_projection(next_live)[2][0][0] == -.5


def test_cached_path_cannot_reuse_maturity_after_control_gap():
  t = Tracker()
  p = point()
  observe(t, [p])
  md = model(); md.timestampEof = 1
  assert t.path_lead(md, 0.)[0] is p
  same_live = t.live
  t.observe(same_live, 50)
  assert t.path_lead(md, 0.)[0] is None
  t.observe(None, 51)
  assert not t._points
  assert t.path_lead(md, 0.)[0] is None
