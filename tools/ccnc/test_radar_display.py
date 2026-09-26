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
  path = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py'))
  names = {'_ccnc_valid_boundary', '_ccnc_side_lane_center', '_CcncRadarDisplayTracker', '_CcncRadarPositionFilter', 'NoiseFilter', 'apply_curved_deadband'}
  nodes = [n for n in ast.parse(path.read_text(encoding='utf8')).body
           if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
  env = dict(math=math, np=np, deque=deque, CV=N(MS_TO_KPH=3.6))
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
  return env


H = load_helpers()
Tracker = H['_CcncRadarDisplayTracker']


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('invalid', [None, 'inner_prob', 'outer_prob', 'nan_prob', 'nan_origin',
                                    'inf_coordinate', 'mismatched_xy', 'reversed_x', 'reversed_boundaries', 'no_20m'])
def test_side_lane_center_rejects_unreliable_geometry(side, invalid):
  md = N(laneLines=[N(x=[0., 10., 20., 30.], y=[y, y + .1, y + .4, y + .9])
                    for y in (-5.4, -1.8, 1.8, 5.4)], laneLineProbs=[.6] * 4)
  inner, outer = ((1, 0) if side == 0 else (2, 3))
  line = md.laneLines[inner]
  if invalid == 'inner_prob':
    md.laneLineProbs[inner] = .59
  elif invalid == 'outer_prob':
    md.laneLineProbs[outer] = .59
  elif invalid == 'nan_prob':
    md.laneLineProbs[inner] = math.nan
  elif invalid == 'nan_origin':
    line.y[0] = math.nan
  elif invalid == 'inf_coordinate':
    md.laneLines[outer].y[-1] = math.inf
  elif invalid == 'mismatched_xy':
    line.y.pop()
  elif invalid == 'reversed_x':
    line.x.reverse()
  elif invalid == 'reversed_boundaries':
    md.laneLines[inner], md.laneLines[outer] = md.laneLines[outer], md.laneLines[inner]
  elif invalid == 'no_20m':
    line.x = [0., 5., 10., 15.]
  center = H['_ccnc_side_lane_center'](md, side)
  if invalid is None:
    assert center == pytest.approx(3.6)
  else:
    assert center is None
  # Rejected input must preserve the last good center, not poison its filter.
  filtered = H['NoiseFilter'](3, 3.6, .03)
  if center is not None:
    filtered.apply(center)
  assert filtered.value == pytest.approx(3.6)


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


@pytest.mark.parametrize('reference', [('lane', True), ('path', False)])
def test_curve_spike_blends_and_recovers_without_a_lingering_offset(reference):
  t = Tracker()
  p = point(y=0.)
  observe(t, [p])
  t.filter_position(p, 0., reference, 20)
  t.filter_position(p, 2., reference, 25)
  assert t.positions[p.trackId][4] == pytest.approx(.15)
  t.filter_position(p, 0., reference, 30)
  assert t.positions[p.trackId][4] == pytest.approx(0.)
  for frame in range(35, 111, 5):
    previous = t.positions[p.trackId][4]
    t.filter_position(p, 2., reference, frame)
    assert 0 <= t.positions[p.trackId][4] - previous <= .150001
  assert t.positions[p.trackId][4] == pytest.approx(2.)


def test_curve_blend_preserves_raw_radar_motion_and_new_target_position():
  t = Tracker()
  p = point(y=0.)
  observe(t, [p])
  t.filter_position(p, 0., ('lane', True), 20)
  t.filter_position(p, 2., ('lane', True), 25)
  p.yRel = 1.
  t.filter_position(p, 3., ('lane', True), 30)
  assert t.positions[p.trackId][4] == pytest.approx(1.3)
  assert t.filter_position(point(2, y=4.), 6., ('lane', True), 30)[1] == pytest.approx(6.)


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
  source = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py')).read_text(encoding='utf8')
  start = source.index('    if CS.live_tracks is not None and (not ff_lane_mode')
  end = source.index('    # Only bridge an empty slot', start)
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
  source = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py')).read_text(encoding='utf8')
  start = source.index('    if CS.live_tracks is not None and (not ff_lane_mode')
  end = source.index('    # Only bridge an empty slot', start)
  import textwrap
  env = dict(CS=N(live_tracks=object()), ff_lane_mode=False, selected_lane_prob=.15,
             a_ego_kph=0., v_ego_kph=20., min_front_lead_speed=-100., np=np, md=model(), ff_lead=point(), ff_yRel=1.,
             display_tracker=N(path_lead=lambda *a, **kw: (None, 0.),
                               recently_selected=lambda *a, **kw: False))
  exec(textwrap.dedent(source[start:end]), env)
  assert env['ff_lead'] is None


@pytest.mark.parametrize('distance', [True])
def test_specialized_position_filter_matches_general_filter(distance):
  fast = H['_CcncRadarPositionFilter'](1., distance)
  reference = H['NoiseFilter'](3, 1., [.3, .9] if distance else .3,
                               [1., 4.] if distance else .6)
  rng = np.random.default_rng(423)
  for value in np.cumsum(rng.normal(0., .4, 3000)):
    assert fast.apply(float(value)) == reference.apply(float(value))
  for value in [100., 0., 1., 1.6, 1.6000001, -50., 2., 3., 4.]:
    assert fast.apply(value) == reference.apply(value)


@pytest.mark.parametrize('step', [-1.2, 1.2])
def test_lateral_median_rejects_single_spike_but_follows_sustained_step(step):
  lateral = H['_CcncRadarPositionFilter'](0., False)
  assert lateral.apply(step) == 0.
  assert lateral.apply(0.) == 0.
  assert lateral.apply(0.) == 0.
  assert lateral.apply(step) == 0.
  previous = 0.
  for _ in range(6):
    value = lateral.apply(step)
    assert 0 <= (value - previous) * math.copysign(1., step) <= .250001
    previous = value
  assert lateral.apply(step) == pytest.approx(step)
  assert H['_CcncRadarPositionFilter'](step, False).apply(step) == pytest.approx(step)


def test_lateral_fast_follow_uses_elapsed_time_and_recovers_from_reversal():
  outputs = []
  for dt in (.01, .05):
    f = H['_CcncRadarPositionFilter'](0., False)
    f.dt = dt
    f.apply(1.2)  # First sample is deliberately rejected by the median.
    for _ in range(round(.2 / dt)):
      f.apply(1.2)
    outputs.append(f.value)
    for _ in range(round(.4 / dt)):
      f.apply(0.)
    assert f.value == pytest.approx(0.)
  assert outputs == pytest.approx([1., 1.])


def test_far_lane_small_geometry_steps_are_limited_but_raw_motion_is_not():
  t = Tracker()
  p = point(x=140.)
  observe(t, [p])
  t.filter_position(p, 0., ('lane', True), 20)
  for frame in range(25, 51, 5):
    old = t.positions[1][4]
    t.filter_position(p, (frame - 20) * .04, ('lane', True), frame)
    assert t.positions[1][4] - old == pytest.approx(.075)
  old = t.positions[1][4]
  p.yRel = .5
  t.filter_position(p, 1.7, ('lane', True), 55)
  assert t.positions[1][4] - old == pytest.approx(.575)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_far_curve_preserves_cancelled_radar_motion(sign):
  t = Tracker()
  p = point(x=80., y=-6. * sign)
  observe(t, [p])
  t.filter_position(p, -1.4 * sign, ('lane', True), 20)
  for frame in range(25, 86, 5):
    p.yRel += .4 * sign
    t.filter_position(p, -1.4 * sign, ('lane', True), frame)
    assert t.positions[1][4] == pytest.approx(-1.4 * sign)
    assert t.positions[1][-1].value == pytest.approx(-1.4 * sign)


def test_far_curve_cancellation_still_limits_unmatched_geometry_and_recovers_bias():
  t = Tracker()
  p = point(x=140.)
  observe(t, [p])
  t.filter_position(p, 0., ('lane', True), 20)
  p.yRel = -.2
  t.filter_position(p, 1., ('lane', True), 25)
  assert t.positions[1][4] == pytest.approx(.075)
  # Keep a fixed road-aligned target while radar and road geometry rotate.
  for frame in range(30, 101, 5):
    p.yRel -= .2
    t.filter_position(p, 1., ('lane', True), frame)
  assert t.positions[1][4] == pytest.approx(1.)
  assert t.positions[1][5] == pytest.approx(0.)


def test_recent_side_needs_path_and_vision_but_true_front_survives_lane_loss():
  t = Tracker()
  p = point(y=1.7)
  observe(t, [p])
  t.finish(None, p, None, 0., True, 20)
  assert t.path_lead(model(probability=0.), 0.)[0] is None
  assert t.path_lead(model(y=1.7), 0.)[0] is None
  p.yRel = .5
  t.observe(N(points=[p]), 25)
  observe(t, [p], 30)
  assert t.path_lead(model(y=.5), 0.)[0] is p
  t.finish(p, None, None, .5, True, 50)
  assert t.path_lead(model(probability=0.), 0.)[0] is p


def test_reliable_side_history_expires_and_does_not_follow_reused_id():
  t = Tracker()
  p = point(y=-2.)
  observe(t, [p])
  md = lane_model()
  t.lane_probabilities(md)
  t.lane_projection(t.live)
  assert t.lane_sides[1][-1] == -1
  t.finish(p, None, None, -2., False, 20)
  assert t.path_lead(model(y=-2., path_y=(0., .3, .3)), 0.)[0] is None
  assert t.path_lead(model(y=-2., path_y=(0., 1., 1.)), 0.)[0] is p
  t.observe(N(points=[point(x=50., y=-2.)]), 25)
  assert not t.lane_sides
  t.lane_probabilities(lane_model())
  t.lane_projection(t.live)
  assert not t.lane_sides  # Never extrapolate trusted lane evidence beyond its range.


def test_reliable_side_history_is_bounded_by_time_and_travel():
  for expired_by in ('time', 'travel', 'missing'):
    t = Tracker()
    p = point(y=-2.)
    observe(t, [p])
    t.lane_probabilities(lane_model())
    t.lane_projection(t.live)
    if expired_by == 'travel':
      t.stop_distance = 16.
    if expired_by == 'time':
      for frame in range(25, 126, 5):
        t.observe(N(points=[p]), frame)
    else:
      t.observe(None if expired_by == 'missing' else N(points=[p]), 25)
    assert not t.lane_sides


def test_expired_side_evidence_cannot_block_a_current_vision_match():
  t = Tracker()
  p = point(y=-1.7)
  observe(t, [p])
  t.lane_sides[1] = (t.tracks[1][0], -100, 0., -1)
  assert t.path_lead(model(y=-1.7), 0.)[0] is p


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


def boundary_step(tracker, p, frame, md=None, ego_kph=0.):
  tracker.observe(N(points=[p]), frame)
  tracker.update_stop(ego_kph, frame)
  tracker.update_boundary_admission(lane_model() if md is None else md)


@pytest.mark.parametrize('y,blocked', [(1.21, True), (1.81, False), (-1.79, True), (5.1, True), (0., False)])
def test_boundary_stationary_start_is_latched_despite_lane_jitter(y, blocked):
  t = Tracker()
  p = point(y=y, speed=0.)
  for frame in range(0, 36, 5):
    boundary_step(t, p, frame)
  assert (1 in t.boundary_rejected) == blocked
  # Mature detections and a disappearing/moving model boundary cannot grant admission.
  for frame in range(40, 141, 5):
    md = lane_model(frame, offset=2.)
    md.laneLineProbs = [0.]*4
    boundary_step(t, p, frame, md)
  assert (1 in t.boundary_rejected) == blocked


def test_boundary_start_uses_initial_window_not_single_frame_probability():
  t = Tracker()
  p = point(y=-1.6, speed=0.)
  for frame in range(0, 36, 5):
    md = lane_model(frame)
    if frame < 10:
      md.laneLineProbs = [0.]*4
    boundary_step(t, p, frame, md)
  assert 1 in t.boundary_rejected
  # A lone near-boundary sample does not permanently reject a vehicle.
  t = Tracker()
  for frame in range(0, 36, 5):
    boundary_step(t, p, frame, lane_model(frame, offset=0. if frame == 0 else 2.))
  assert not t.boundary_rejected


@pytest.mark.parametrize('initial_speed,initial_y', [(8., 1.5), (0., 0.)])
def test_previously_admitted_vehicle_can_stop_at_boundary(initial_speed, initial_y):
  t = Tracker()
  p = point(y=initial_y, speed=initial_speed)
  for frame in range(0, 61, 5):
    p.dRel = 10. + initial_speed * frame * .01
    boundary_step(t, p, frame)
  p.vLead = 0.
  for frame in range(65, 126, 5):
    p.yRel = min(1.5, p.yRel + .15)
    boundary_step(t, p, frame)
  assert not t.boundary_rejected


def test_boundary_check_uses_target_distance_and_does_not_extrapolate():
  md = lane_model(offset=1.)
  for x, expected in ((10., True), (35., False)):
    t = Tracker()
    for frame in range(0, 36, 5):
      boundary_step(t, point(x=x, y=.5, speed=0.), frame, md)
    assert bool(t.boundary_rejected) == expected


def test_same_radar_publication_cannot_confirm_boundary_start():
  t = Tracker()
  live = N(points=[point(y=1.5, speed=0.)])
  for frame in range(100):
    t.observe(live, frame)
    t.update_stop(0., frame)
    t.update_boundary_admission(lane_model(frame))
  assert t.boundary_admission[1]['samples'] == 1
  assert t.boundary_admission[1]['status'] == 'pending'


@pytest.mark.parametrize('ego_kph', [0., 3.6])
def test_rejected_point_can_start_moving_with_ego_compensated_displacement(ego_kph):
  t = Tracker()
  for frame in range(0, 61, 5):
    p = point(x=10. - ego_kph / 3.6 * frame * .01, y=1.5, speed=0.)
    p.vRel = -ego_kph / 3.6
    boundary_step(t, p, frame, ego_kph=ego_kph)
  assert 1 in t.boundary_rejected
  for frame in range(65, 126, 5):
    p = point(x=10. - ego_kph / 3.6 * frame * .01 + 2. * (frame-60) * .01, y=1.5, speed=2.)
    p.vRel = 2. - ego_kph / 3.6
    boundary_step(t, p, frame, ego_kph=ego_kph)
  assert not t.boundary_rejected


@pytest.mark.parametrize('reason', ['speed_only', 'position_only'])
def test_boundary_release_requires_consistent_motion_not_spikes(reason):
  t = Tracker()
  for frame in range(0, 61, 5):
    boundary_step(t, point(y=1.5, speed=0.), frame)
  for frame in range(65, 141, 5):
    x = 10. if reason == 'speed_only' else 10. + (frame-60) * .01
    speed = 0. if reason == 'position_only' else 2.
    boundary_step(t, point(x=x, y=1.5, speed=speed), frame)
  assert 1 in t.boundary_rejected


@pytest.mark.parametrize('initially_stopped', [False, True])
def test_lateral_motion_is_not_a_stationary_boundary_object(initially_stopped):
  t = Tracker()
  for frame in range(0, 141, 5):
    moving = not initially_stopped or frame > 60
    p = point(y=1.5 + (max(0, frame-60) if initially_stopped else frame) * .01, speed=0.)
    p.yvRel = 1. if moving else 0.
    boundary_step(t, p, frame)
    if frame == 60:
      assert bool(t.boundary_rejected) == initially_stopped
  assert not t.boundary_rejected


@pytest.mark.parametrize('reset', ['missing', 'jump', 'gap'])
def test_new_identity_does_not_inherit_boundary_rejection(reset):
  t = Tracker()
  for frame in range(0, 36, 5):
    boundary_step(t, point(y=1.5, speed=0.), frame)
  assert 1 in t.boundary_rejected
  if reset == 'missing':
    t.observe(N(points=[]), 40)
    t.update_boundary_admission(lane_model())
    assert not t.boundary_admission
  boundary_step(t, point(x=20. if reset == 'jump' else 10., y=1.5, speed=8.), 70 if reset == 'gap' else 45)
  assert not t.boundary_rejected


def test_boundary_rejection_applies_to_direct_and_vision_fallback_selection():
  step = display_step_for_test()
  md = model(y=1.5)
  geom = lane_model()
  md.laneLines, md.laneLineProbs, md.roadEdges = geom.laneLines, geom.laneLineProbs, geom.roadEdges
  p = point(y=1.5, speed=0.)
  for frame in range(0, 61, 5):
    assert step(md, N(live_tracks=N(points=[p])), frame, 0., 0.) == (None, None, None)
  md.laneLineProbs = [0.]*4
  md.leadsV3[0].v = [0.]
  for frame in range(65, 126, 5):
    assert step(md, N(live_tracks=N(points=[p])), frame, 0., 0.) == (None, None, None)


def test_first_lateral_speed_spike_does_not_exempt_stationary_boundary_start():
  t = Tracker()
  for frame in range(0, 81, 5):
    p = point(y=1.5, speed=0.)
    p.yvRel = -4. if frame == 0 else 0.
    boundary_step(t, p, frame)
  assert t.boundary_admission[1]['status'] == 'blocked'


def test_weak_nearest_boundary_waits_despite_confident_opposite_lane():
  t = Tracker()
  for frame in range(0, 101, 5):
    md = lane_model(frame)
    md.laneLineProbs[1] = .05  # The right side cannot establish left geometry.
    boundary_step(t, point(y=1.5, speed=0.), frame, md)
  assert t.boundary_admission[1]['samples'] == 0
  assert t.boundary_admission[1]['status'] == 'pending'
  assert not t.boundary_rejected  # Unknown geometry does not exclude front or side vehicles.
  for frame in range(105, 146, 5):
    boundary_step(t, point(y=1.5, speed=0.), frame)
  assert t.boundary_admission[1]['status'] == 'blocked'


def test_unknown_geometry_recovers_for_stationary_interior():
  t = Tracker()
  for frame in range(0, 101, 5):
    md = lane_model(frame)
    md.laneLineProbs = [0.] * 4
    boundary_step(t, point(y=0., speed=0.), frame, md)
  assert t.boundary_admission[1]['status'] == 'pending'
  assert not t.boundary_rejected
  for frame in range(105, 146, 5):
    boundary_step(t, point(y=0., speed=0.), frame)
  assert t.boundary_admission[1]['status'] == 'allowed'
  assert not t.boundary_rejected


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


@pytest.mark.parametrize('selected', [False, True])
def test_small_lateral_jump_grace_only_for_selected_side(selected):
  t = Tracker()
  observe(t, [point(y=3.3)])
  if selected:
    t.finish(None, point(y=3.3), None, 0., True, 20)
  t.observe(N(points=[point(y=2.45)]), 26)
  assert t.stable(1) == selected
  if selected:
    # A second discontinuity inside the cooldown must still reset maturity.
    t.observe(N(points=[point(y=3.3)]), 32)
    assert not t.stable(1)


@pytest.mark.parametrize('x,y,vr', [(15., 2.45, 0.), (10., 1.8, 0.), (10., 2.45, 2.)])
def test_lateral_grace_does_not_hide_other_discontinuities(x, y, vr):
  t = Tracker()
  observe(t, [point(y=3.3)])
  t.finish(None, point(y=3.3), None, 0., True, 20)
  p = point(x=x, y=y)
  p.vRel = vr
  t.observe(N(points=[p]), 26)
  assert not t.stable(1)


def display_step_for_test():
  import textwrap
  path = Path(os.environ.get('CCNC_TEST_SOURCE', Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py'))
  source = path.read_text(encoding='utf8')
  env = dict(H, CAR_MODEL_ID=3)
  selector = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == '_CcncRadarLaneSelector')
  exec(compile(ast.Module(body=[selector], type_ignores=[]), str(path), 'exec'), env)
  control = N(radar_display_tracker=Tracker(), radar_lane_selector=env['_CcncRadarLaneSelector']())
  for side in ('ff', 'lf', 'rf'):
    setattr(control, side + '_detect', N(apply=lambda v: 1))
  for side in ('lf', 'rf'):
    setattr(control, side + '_center', H['NoiseFilter'](3, 3., .03))
  env['state'] = control
  start = source.index('    ff_lead = lf_lead = rf_lead = None')
  end = source.index('    center_lane_offset =', start)
  body = 'def step(md, CS, frame, v_ego_kph, a_ego_kph):\n  values = {}\n'
  body += textwrap.indent(textwrap.dedent(source[start:end]), '  ')
  body += '\n  return ff_lead, lf_lead, rf_lead\n'
  exec(body, env)
  return env['step']


@pytest.mark.parametrize('edge,other,expected', [(3.4, False, (1, None, None)),
                                                (6., False, (None, None, 1)),
                                                (3.4, True, (2, None, None))])
def test_front_boundary_bridge_yields_to_side_or_new_front(edge, other, expected):
  step = display_step_for_test()
  curve = lambda y: N(x=[0., 100.], y=[y, y])
  md = model()
  md.timestampEof = 1
  md.laneLineProbs = [0., .9, .9, 0.]
  md.laneLines = [curve(-5.), curve(-1.8), curve(1.8), curve(5.)]
  md.roadEdges = [curve(-6.), curve(edge)]
  for frame in range(0, 26, 5):
    step(md, N(live_tracks=N(points=[point(y=-1.7)])), frame, 0., 0.)
  points = [point(y=-2.)]
  if other:
    points.append(point(2, x=8., y=0.))
  selected = step(md, N(live_tracks=N(points=points)), 30, 0., 0.)
  assert tuple(p.trackId if p else None for p in selected) == expected
  # Crossing farther than the bridge or losing the observation must not hold FF.
  selected = step(md, N(live_tracks=N(points=[point(y=-2.3)])), 35, 0., 0.)
  if edge == 3.4:
    assert selected[0] is None
  assert step(md, N(live_tracks=N(points=[])), 40, 0., 0.) == (None, None, None)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('near,far,limit,expected', [(4., 4., 100., True),
                                                  (3., 4., 100., False),
                                                  (4., 3., 100., False),
                                                  (4., 4., 40., False)])
def test_stationary_side_entry_checks_two_extra_widths(side, near, far, limit, expected):
  t = Tracker()
  sign = 1. if side else -1.
  curve = lambda xs, ys: N(x=xs, y=[sign*y for y in ys])
  inner = curve([0., 100.], [1.5, 1.5])
  edge = curve([0., 3., 23., 43., 100.], [near, near, 4., far, far])
  if limit == 40.:
    edge.x[-2:] = [35., 40.]
  md = model()
  md.timestampEof = 1
  md.laneLineProbs = [0., .9, .9, 0.]
  md.laneLines = [inner, inner, inner, inner]
  md.roadEdges = [edge, edge]
  q = point(x=23., speed=0.)
  live = N(points=[q])
  t.observe(live, 0)
  t.lane_probabilities(md)
  t.lane_projection(live)
  assert t.side_entry_width(q, side) == expected
  assert t.side_entry_width(q, side) == expected  # Cached geometry.
  t.finish(None, q, None, 0., True, 0)
  assert t.side_entry_width(q, side)  # Continuous selected target can stop.
  t.recent_selected.clear()
  q.vLead = 1.
  assert t.side_entry_width(q, side)  # Moving candidate keeps existing policy.
  q.vLead = 0.
  md.laneLineProbs[3 if side else 0] = .5
  assert t.side_entry_width(q, side)


@pytest.mark.parametrize('exit_kind', ['departure', 'motion', 'sensor_loss', 'other_slot'])
def test_stationary_display_hold_and_release(exit_kind):
  t = Tracker()
  q = point(x=8., y=3., speed=0.)
  values = {'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 6.4, 'LF_DETECT_LATERAL': 3.}
  for frame in range(0, 101, 5):
    t.observe(N(points=[q]), frame)
    t.update_stop(0., frame)
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display(values, (q, None), frame)
  assert 0 in t.stop_holds
  for frame in range(105, 151, 5):
    t.observe(N(points=[]), frame)
    t.update_stop(0., frame)
    t.finish(None, None, None, 0., True, frame)
    out = {'LF_DETECT': 0}
    t.stopped_display(out, (None, None), frame)
    assert out['LF_DETECT'] == 1
    assert out['LF_DETECT_DISTANCE'] == 6.4
    assert t.selected == (None, None, None)
    assert not t.recent_selected
  moving = point(track_id=9, x=8., y=3., speed=2.)
  t.observe(None if exit_kind == 'sensor_loss' else N(points=[moving] if exit_kind == 'motion' else []), 155)
  t.update_stop(1. if exit_kind == 'departure' else 0., 155)
  if exit_kind == 'other_slot':
    t.observe(N(points=[q]), 156)
    t.finish(q, None, None, 0., True, 156)
  out = {'LF_DETECT': 0}
  t.stopped_display(out, (None, None), 156)
  assert out['LF_DETECT'] == 0
  assert not t.stop_holds


def test_saved_width_fades_and_expires_after_travel():
  t = Tracker()
  curve = lambda y: N(x=[0., 100.], y=[y, y])
  def md(prob):
    m = model()
    m.timestampEof = prob
    m.laneLineProbs = [prob, .9, .9, prob]
    m.laneLines = [curve(-5.), curve(-1.5), curve(1.5), curve(5.)]
    m.roadEdges = [curve(-6.), curve(6.)]
    return m
  q = point(x=23., y=-3., speed=0.)
  live = N(points=[q])
  t.observe(live, 0)
  t.update_stop(0., 0)
  t.lane_probabilities(md(.8));t.lane_projection(live)
  assert t.saved_widths[1] is not None
  weak = md(0.)
  weak.roadEdges = [curve(-2.), curve(2.)]
  t.lane_probabilities(weak)
  t.lane_projection(live)
  projection = t.side_projection(1)[0]
  assert projection[0] == pytest.approx(5.)
  assert projection[1] == pytest.approx(6.)
  assert t.side_entry_width(q, 1)
  for frame in range(5, 110, 5):
    t.update_stop(36., frame)
  assert t.saved_widths == [None, None]
  t.update_stop(0., 110)
  t.lane_probabilities(weak)
  t.lane_projection(live)
  assert t.side_projection(1)[0][1] == pytest.approx(2.)


def test_held_motion_clears_even_when_another_candidate_is_selected():
  t = Tracker()
  held_point = point(1, x=8., y=3., speed=0.)
  other = point(2, x=30., y=3., speed=2.)
  for frame in range(0, 101, 5):
    t.observe(N(points=[held_point]), frame)
    t.update_stop(0., frame)
    t.finish(None, held_point, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 6.4, 'LF_DETECT_LATERAL': 3.}, (held_point, None), frame)
  assert 0 in t.stop_holds
  moving = point(9, x=8., y=3., speed=2.)
  t.observe(N(points=[moving, other]), 105)
  t.update_stop(0., 105)
  t.finish(None, other, None, 0., True, 105)
  out = {'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 24., 'LF_DETECT_LATERAL': 3.}
  t.stopped_display(out, (other, None), 105)
  assert not t.stop_holds
  assert out['LF_DETECT_DISTANCE'] == 24.


@pytest.mark.parametrize('approaching', [False, True])
def test_unreported_lateral_motion_cannot_be_saved_as_stopped(approaching):
  t = Tracker()
  for frame in range(0, 201, 5):
    q = point(x=10. - (frame*.01 if approaching else 0.), y=2. + frame*.01, speed=0.)
    q.vRel = -1. if approaching else 0.
    q.yvRel = 0.  # Crosswalk case: position moves despite zero reported lateral speed.
    t.observe(N(points=[q]), frame)
    t.update_stop(3.6 if approaching else 0., frame, -1.)
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 8., 'LF_DETECT_LATERAL': q.yRel}, (q, None), frame)
    if not approaching and frame >= 100:
      assert not t.stop_holds and t.stop_motion[1]['blocked']
    if approaching:
      assert not t.stop_motion


@pytest.mark.parametrize('continuation', ['missing', 'unselected', 'lateral_departure', 'other_candidate'])
def test_stopped_hold_requires_fresh_stationary_identity(continuation):
  t = Tracker()
  q = point(x=8., y=3., speed=0.)
  for frame in range(0, 101, 5):
    q.yRel = 3. + (.05 if frame % 10 else 0.)
    t.observe(N(points=[q]), frame)
    t.update_stop(0., frame)
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 6.4, 'LF_DETECT_LATERAL': 3.}, (q, None), frame)
  assert 0 in t.stop_holds
  for frame in range(105, 251, 5):
    q.yRel = 6. if continuation in ('lateral_departure', 'other_candidate') else 3.
    other = point(2, x=30., y=3., speed=3.) if continuation == 'other_candidate' else None
    t.observe(N(points=[] if continuation == 'missing' else [q] + ([other] if other else [])), frame)
    t.update_stop(0., frame)
    t.finish(None, other, None, 0., True, frame)
    out = {'LF_DETECT': int(other is not None), 'LF_DETECT_DISTANCE': 24., 'LF_DETECT_LATERAL': 3.}
    t.stopped_display(out, (other, None), frame)
    if continuation == 'missing':
      assert out['LF_DETECT'] == 1
    elif continuation == 'unselected':
      assert out['LF_DETECT'] == 1 and out['LF_DETECT_DISTANCE'] == 6.4
    else:
      # A one-frame discontinuity is a new identity, not proof of a crossing.
      assert t.stop_holds
      assert out['LF_DETECT'] == 1


@pytest.mark.parametrize('confirmation,expected', [('none', False), ('vision', True), ('lane', True),
                                                 ('moving', True), ('wrong_distance', False), ('wrong_speed', False)])
def test_stationary_front_admission_is_shared_by_direct_and_path_selection(confirmation, expected):
  step = display_step_for_test()
  md = model(x=11.52, probability=.9 if confirmation != 'none' else .05)
  geom = lane_model()
  md.laneLines, md.roadEdges = geom.laneLines, geom.roadEdges
  md.laneLineProbs = [.9 if confirmation == 'lane' else .2] * 4
  md.leadsV3[0].v = [0.]
  if confirmation == 'wrong_distance': md.leadsV3[0].x = [60.]
  if confirmation == 'wrong_speed': md.leadsV3[0].v = [15.]
  q = point(speed=3. if confirmation == 'moving' else 0.)
  for frame in range(0, 61, 5):
    selected = step(md, N(live_tracks=N(points=[q])), frame, 0., 0.)
  assert bool(selected[0]) == expected


def test_moving_front_vehicle_can_brake_to_rest_with_weak_vision_and_lanes():
  step = display_step_for_test()
  md = model(probability=.01)
  geom = lane_model()
  md.laneLines, md.roadEdges = geom.laneLines, geom.roadEdges
  md.laneLineProbs = [.2] * 4
  for frame in range(0, 101, 5):
    q = point(x=10. + min(frame, 50)*.03, speed=3. if frame < 50 else 0.)
    selected = step(md, N(live_tracks=N(points=[q])), frame, 0., 0.)
    assert selected[0].trackId == 1


@pytest.mark.parametrize('initially_stationary', [False, True])
def test_confirmed_moving_car_is_held_until_ego_departure(initially_stationary):
  t = Tracker()
  start = 60 if initially_stationary else 0
  for frame in range(0, 161, 5):
    distance = max(0, min(frame-start, 50)) * .02
    speed = 2. if start <= frame < start+50 else 0.
    q = point(x=8. + distance, y=3., speed=speed)
    boundary_step(t, q, frame)
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': q.dRel*.8, 'LF_DETECT_LATERAL': 3.}, (q, None), frame)
  assert 0 in t.stop_holds
  for frame in range(165, 501, 5):
    t.observe(N(points=[]), frame)
    t.update_stop(0., frame)
    t.finish(None, None, None, 0., True, frame)
    out = {'LF_DETECT': 0}
    t.stopped_display(out, (None, None), frame)
    assert out['LF_DETECT'] == 1
  t.update_stop(1., 505)
  assert not t.stop_holds


def test_approach_confirmed_car_survives_coordinate_step_loss_and_reused_id():
  t = Tracker()
  for frame in range(0, 101, 5):
    q = point(44, x=10.-frame*.01, y=2.4, speed=0.)
    q.vRel = -1.
    t.observe(N(points=[q]), frame)
    t.update_stop(3.6, frame, -1.)
    t.update_boundary_admission(lane_model(frame))
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': q.dRel*.8, 'LF_DETECT_LATERAL': 2.4}, (q, None), frame)
  q.vRel = 0.
  for frame in range(105, 141, 5):
    q.yRel = 2.15 if frame >= 125 else 2.4
    boundary_step(t, q, frame)
    t.finish(None, q, None, 0., True, frame)
    t.stopped_display({'LF_DETECT': 1, 'LF_DETECT_DISTANCE': q.dRel*.8, 'LF_DETECT_LATERAL': q.yRel}, (q, None), frame)
  assert 0 in t.stop_holds
  for frame in range(145, 351, 5):
    new = point(44, x=26., y=1.5, speed=0.)  # Reused slot is a rejected new boundary point.
    boundary_step(t, new, frame)
    t.finish(None, None, None, 0., True, frame)
    out = {'LF_DETECT': 0}
    t.stopped_display(out, (None, None), frame)
    assert out['LF_DETECT'] == 1
    assert out['LF_DETECT_DISTANCE'] < 10.
  assert 44 in t.boundary_rejected


def test_established_hold_cancels_real_continuous_lateral_departure():
  t = Tracker()
  q = point(y=3., speed=0.)
  observe(t, [q])
  t.update_stop(0., 20)
  t.stop_holds[0] = (1, 10., 3., (1, 8., 3.), 0)
  for frame in range(25, 91, 5):
    q.yRel += .1
    t.observe(N(points=[q]), frame)
    t.update_stop(0., frame)
    out = {'LF_DETECT': 0}
    t.stopped_display(out, (None, None), frame)
  assert not t.stop_holds and out['LF_DETECT'] == 0


@pytest.mark.parametrize('kind', ['single_step', 'jitter', 'duplicate', 'moving_ego', 'reused_id'])
def test_crossing_history_does_not_infer_motion_from_discontinuities(kind):
  t = Tracker()
  live = N(points=[point(y=3., speed=0.)])
  for frame in range(0, 151, 5):
    if kind == 'single_step':
      y = 3. + (0.7 if frame >= 50 else 0.) + (0.05 if frame >= 70 else 0.) + (0.05 if frame >= 90 else 0.)
    elif kind == 'jitter':
      y = 3. + (0.25 if frame % 10 else 0.)
    else:
      y = 3. + frame*.02
    q = point(x=10. + (30. if kind == 'reused_id' and frame >= 30 else 0.), y=y, speed=0.)
    if kind == 'reused_id':
      q.yRel = 3. if frame < 30 else 6.
    if kind != 'duplicate': live = N(points=[q])
    t.observe(live, frame)
    t.update_stop(3.6 if kind == 'moving_ego' else 0., frame)
    assert not any(h['blocked'] for h in t.stop_motion.values())


def test_crossing_can_be_saved_again_after_two_seconds_of_real_stop():
  t = Tracker()
  for frame in range(0, 401, 5):
    q = point(x=10., y=2. + min(frame, 100)*.01, speed=0.)
    q.yvRel = 0.
    t.observe(N(points=[q]), frame)
    t.update_stop(0., frame)
    t.finish(None, q, None, 0., True, frame)
    out = {'LF_DETECT': 1, 'LF_DETECT_DISTANCE': 8., 'LF_DETECT_LATERAL': q.yRel}
    t.stopped_display(out, (q, None), frame)
    # Suppression affects stored fallback only, never the selected live candidate.
    assert out['LF_DETECT'] == 1
    if frame == 200:
      assert t.stop_motion[1]['blocked'] and not t.stop_holds
  assert not t.stop_motion[1]['blocked']
  assert 0 in t.stop_holds


@pytest.mark.parametrize('x,visible', [(-1.01, False), (-1., True), (-.2, True), (0., True), (1., True)])
def test_approach_hold_clamps_output_only(x, visible):
  t = Tracker()
  t.live = N(points=[])
  t.approaching = True
  t.stop_distance = 1.5 - x
  t.approach_holds[0] = (33, 1.5, 2., (1, 1.2, 2.), 0, 0., 0)
  out = {'LF_DETECT': 0}
  t.stopped_display(out, (None, None), 10)
  assert bool(out['LF_DETECT']) == visible
  if visible:
    assert out['LF_DETECT_DISTANCE'] == pytest.approx(max(0., x)*.8)
    assert t.approach_holds[0][1] - t.stop_distance == pytest.approx(x)


@pytest.mark.parametrize('reason', ['timeout', 'travel', 'acceleration', 'sensor', 'motion'])
def test_approach_hold_release(reason):
  t = Tracker()
  t.observe(N(points=[]), 0)
  t.update_stop(5., 0, -1.)
  t.approach_holds[0] = (33, 10., 2., (1, 8., 2.), 0, 0., 0)
  if reason == 'travel':
    t.stop_distance = 5.01
  elif reason == 'acceleration':
    t.update_stop(5., 5, 1.)
  elif reason == 'sensor':
    t.observe(None, 5)
    t.update_stop(5., 5, -1.)
  elif reason == 'motion':
    t.live = N(points=[point(99, x=10., y=2., speed=2.)])
  out = {'LF_DETECT': 0}
  t.stopped_display(out, (None, None), 301 if reason == 'timeout' else 10)
  assert out['LF_DETECT'] == 0
  assert not t.approach_holds


@pytest.mark.parametrize('reused', [False, True])
def test_approach_transfers_negative_distance_to_stop_and_departure_clears(reused):
  t = Tracker()
  t.observe(N(points=[]), 0)
  t.update_stop(5., 0, -1.)
  t.approach_holds[0] = (33, 1.5, 2., (1, 1.2, 2.), 0, 0., 0)
  t.stop_distance = 1.7
  # Same ID reused far away must not become the remembered vehicle.
  t.observe(N(points=[point(33 if reused else 99, x=50., y=-12.)]), 5)
  t.update_stop(0., 5, -1.)
  out = {'LF_DETECT': 0}
  t.stopped_display(out, (None, None), 5)
  assert out['LF_DETECT'] == 1 and out['LF_DETECT_DISTANCE'] == 0.
  assert t.stop_holds[0][1] == pytest.approx(-.2)
  assert not t.approach_holds
  t.update_stop(1., 10, 1.)
  assert not t.stop_holds


@pytest.mark.parametrize('side', [0, 1])
def test_approaching_stationary_target_is_qualified_before_loss(side):
  t = Tracker()
  sign = 1. if side == 0 else -1.
  prefix = 'LF' if side == 0 else 'RF'
  for frame in range(0, 101, 5):
    q = point(33, x=5. - frame*.01, y=sign*2.5, speed=0.)
    q.vRel = -1.
    t.observe(N(points=[q]), frame)
    t.update_stop(3.6, frame, -1.)
    md = lane_model(frame)
    md.laneLineProbs = [0.] * 4
    t.update_boundary_admission(md)
    leads = (q, None) if side == 0 else (None, q)
    t.finish(None, *leads, 0., True, frame)
    t.stopped_display({prefix+'_DETECT': 1, prefix+'_DETECT_DISTANCE': q.dRel*.8,
                       prefix+'_DETECT_LATERAL': 2.5}, leads, frame)
  assert side in t.approach_holds
  t.observe(N(points=[]), 105)
  t.update_stop(3.6, 105, -1.)
  t.finish(None, None, None, 0., True, 105)
  out = {prefix+'_DETECT': 0}
  t.stopped_display(out, (None, None), 105)
  assert out[prefix+'_DETECT'] == 1
  assert out[prefix+'_DETECT_DISTANCE'] == pytest.approx(3.95*.8)


def test_side_projection_is_lazy_cached_and_refreshed(monkeypatch):
  t = Tracker()
  live = N(points=[point(x=10.)])
  t.observe(live, 0)
  t.lane_probabilities(lane_model())
  calls = []
  interp = np.interp

  def counted(*args, **kwargs):
    calls.append(1)
    return interp(*args, **kwargs)

  monkeypatch.setattr(np, "interp", counted)
  t.lane_projection(live)
  assert len(calls) == 2  # Inner lanes only.
  left = t.side_projection(0)
  assert left == ((-5., -7.),)
  assert len(calls) == 4
  assert t.side_projection(0) is left
  assert t._side_projection[1] is None
  assert len(calls) == 4
  t.lane_probabilities(lane_model(2, 1.))
  t.lane_projection(live)
  assert t._side_projection == [None, None]
  assert t.side_projection(0) == ((-4., -6.),)
  live2 = N(points=[point(x=20.)])
  t.observe(live2, 5)
  t.lane_projection(live2)
  assert t.side_projection(0) == ((-3.5, -5.5),)


def test_empty_tracks_still_save_width_without_side_projection():
  t = Tracker()
  live = N(points=[])
  t.observe(live, 0)
  t.update_stop(0., 0)
  t.lane_probabilities(lane_model())
  t.lane_projection(live)
  assert all(width is not None for width in t.saved_widths)
  assert t._side_projection == [None, None]


def test_side_projection_masks_out_of_range_and_invalid_boundaries():
  t = Tracker()
  live = N(points=[point(x=10.), point(2, x=40.)])
  t.observe(live, 0)
  md = lane_model()
  md.laneLineProbs[0] = 0.
  t.lane_probabilities(md)
  t.lane_projection(live)
  left = t.side_projection(0)
  assert math.isnan(left[0][0]) and left[0][1] == -7.
  assert math.isnan(left[1][0]) and math.isnan(left[1][1])


@pytest.mark.parametrize("side", [1, 2])
def test_crossing_tracks_latest_radar_and_changes_slot(side):
  t = Tracker()
  sign = 1 if side == 1 else -1
  p = point(y=2.5 * sign)
  observe(t, [p])
  curve = lambda y: (np.array([0., 80.]), np.array([y, y]))
  t._lane_data = ([curve(y) for y in (-1.8, 1.8, -5., 5., -6., 6.)], (True,) * 4)
  leads = [None] * 3
  leads[side] = p
  t.bridge_crossing(leads, [0.] * 3, .9, True, 20, (0., 0., 0.))
  # A weak but still accepted lane must not erase the reliable snapshot.
  t.bridge_crossing(leads[:], [0.] * 3, .15, True, 25, (0., 0., 0.))
  for frame, y, slot in [(25, 2.2 * sign, side), (30, 1.6 * sign, 0)]:
    p = point(y=y)
    t.observe(N(points=[p]), frame)
    got, lateral = t.bridge_crossing([None] * 3, [0.] * 3, .05, True, frame, (0., 0., 0.))
    assert got[slot] is p and sum(x is not None for x in got) == 1
    assert lateral[slot] == pytest.approx(y)


@pytest.mark.parametrize("reason", ["timeout", "distance", "missing", "jump", "occupied", "edge", "slow", "recovery"])
def test_crossing_cancels_stale_or_invalid_target(reason):
  t = Tracker()
  p = point(y=2.5)
  observe(t, [p])
  curve = lambda y: (np.array([0., 80.]), np.array([y, y]))
  t._lane_data = ([curve(y) for y in (-1.8, 1.8, -5., 5., -6., 6.)], (True,) * 4)
  t.bridge_crossing([None, p, None], [0.] * 3, .9, True, 20, (0., 0., 0.))
  frame, probability, leads = 25, .05, [None] * 3
  if reason == "timeout": frame = 225
  if reason == "distance": t.stop_distance = 30.1
  if reason == "missing": t.observe(None, frame)
  if reason == "jump": t.observe(N(points=[point(x=50., y=2.5)]), frame)
  if reason == "occupied": leads[1] = point(track_id=2, y=2.5)
  if reason == "edge": p.yRel = 5.1
  if reason == "slow": p.vLead = -1.
  if reason == "recovery": probability = .9
  got, _ = t.bridge_crossing(leads, [0.] * 3, probability, True, frame, (0., 0., 0.))
  assert all(x is None or x.trackId != 1 for x in got)


def test_unknown_nearest_boundary_does_not_hide_new_stationary_side_vehicle():
  step = display_step_for_test()
  md = model()
  geom = lane_model()
  md.laneLines, md.roadEdges = geom.laneLines, geom.roadEdges
  md.laneLineProbs = [.9, .9, .05, .2]
  q = point(y=-2.5, speed=0.)
  for frame in range(0, 101, 5):
    md.timestampEof = frame
    selected = step(md, N(live_tracks=N(points=[q])), frame, 0., 0.)
  assert selected[2] is q
