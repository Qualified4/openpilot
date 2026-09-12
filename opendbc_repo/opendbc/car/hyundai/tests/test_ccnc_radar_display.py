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
  names = {'_ccnc_valid_boundary', '_CcncRadarDisplayTracker', 'NoiseFilter'}
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
