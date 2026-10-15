"""Independent map invariants; does not import the old LF/FF/RF selector."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest


PATH = Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_radar_map.py'
spec = importlib.util.spec_from_file_location('ccnc_map_test_module', PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
RadarMap = module.CcncRadarMap


def point(key, x, y=0., speed=0., **extra):
  extra.setdefault('trackState', 2)  # A measured track, unless the test says otherwise.
  return N(trackId=key, dRel=x, yRel=y, vLead=speed, vRel=speed, radarSource='frontRadar', measured=True, **extra)


def model(probability=.95):
  return N(laneLines=[N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-5.4, -1.8, 1.8, 5.4)],
           laneLineProbs=[probability] * 4, orientationRate=N(z=[0.]), leadsV3=[])


def run(points, count=25, speed=0., radar=None):
  radar = RadarMap() if radar is None else radar
  for i in range(count):
    now = round(i * .05, 6)
    radar.update(N(points=points(now)), model(), now, speed)
  return radar


@pytest.mark.parametrize('y,slot', [(0., 0), (3.5, 1), (-3.5, 2)])
def test_stationary_lane_points_remain_unknown_without_vehicle_evidence(y, slot):
  # Ego is driving: a stopped ego admits a resting adjacent-lane return as a waiting vehicle.
  radar = run(lambda t: [point(1, 25. - 5. * t, y)], speed=5.)
  assert radar.selected[slot] is None
  assert next(iter(radar.objects.values())).vehicle < .6
  assert not radar.walls


def test_radar_slot_reuse_does_not_teleport_or_inherit_vehicle_identity():
  radar = run(lambda t: [point(1, 10. + t, 3.5, 1.)])
  old = radar.selected[1]
  radar.update(N(points=[point(1, RadarMap.MAX_X - 2., -2.1)]), model(), 1.25, 0.)
  new = max(radar.objects)
  assert old != new
  assert not radar.objects[new].motion_confirmed
  assert radar.objects[new].hits == 1
  assert radar.objects[old].x < 15.


def test_id_change_near_predicted_position_reconnects_without_duplicate():
  radar = run(lambda t: [point(1 if t < .5 else 2, 10. + t, 3.5, 1.)])
  assert len(radar.objects) == 1
  assert next(iter(radar.objects.values())).source == 2
  assert radar.selected[1] is not None


def test_two_nearby_measurements_cannot_update_one_track_twice():
  radar = run(lambda t: [point(1, 10., 3.5)], count=5)
  old = next(iter(radar.objects.values()))
  hits = old.hits
  radar.update(N(points=[point(1, 10.01, 3.5), point(2, 10.1, 3.5)]), model(), .25, 0.)
  assert old.hits == hits + 1
  assert len(radar.objects) == 2


@pytest.mark.parametrize('sign', [-1, 1])
def test_boundary_stationary_row_becomes_wall_but_interior_row_does_not(sign):
  radar = run(lambda t: [point(i, x, sign * 2.1) for i, x in enumerate((5., 7., 9., 11.))])
  assert radar.walls
  assert all(o.structure for o in radar.objects.values())
  assert all(k is None for k in radar.selected)
  interior = run(lambda t: [point(i, x, sign * 3.5, trackState=2) for i, x in enumerate((5., 7., 9., 11.))])
  assert not interior.walls
  # Ego is stopped here, so the nearest resting return a car body's distance into the lane is a waiting vehicle.
  assert interior.objects[interior.selected[1 if sign == 1 else 2]].admitted


@pytest.mark.parametrize('velocity', [.5, 1., 2., 6.])
def test_moving_periodic_array_never_becomes_wall(velocity):
  # Reported speed matches the motion, as in every recorded log. Position-only
  # motion (zero reported speed) was a synthetic input and is no longer evidence.
  radar = RadarMap()
  for i in range(81):
    t = round(i * .05, 6)
    radar.update(N(points=[point(k, x + velocity * t, 2.1, velocity) for k, x in enumerate((5., 7., 9., 11.))]), model(), t, 0.)
    assert not radar.walls
  # .5m/s is below the Doppler evidence limit: not a wall, but not a confirmed vehicle either.
  assert any(o.motion_confirmed for o in radar.objects.values()) == (velocity > RadarMap.DOPPLER_SPEED)


def test_stationary_row_with_oscillating_returns_can_form_wall():
  radar = run(lambda t: [point(k, x + (.12 if round(t * 20) % 2 else -.12), 2.1)
                         for k, x in enumerate((5., 7., 9., 11.))])
  assert radar.walls


def test_moving_vehicle_that_stops_keeps_vehicle_history():
  radar = run(lambda t: [point(1, 10. + min(t, .5) * 2., 3.5, 2. if t < .5 else 0.)], count=45)
  obj = radar.objects[radar.selected[1]]
  assert obj.motion_confirmed and obj.stationary and not obj.structure


def test_missing_stationary_object_follows_ego_braking_not_old_relative_velocity():
  radar = run(lambda t: [point(1, 12. + min(t, .3) - t * 2., 3.5, 1. if t < .3 else 0.)], count=21, speed=2.)
  key = radar.selected[1]
  assert key is not None
  radar.update(N(points=[]), model(), 1.05, 0.)
  before = radar.to_ego(radar.objects[key].x, radar.objects[key].y)
  for i in range(22, 41):
    radar.update(N(points=[]), model(), round(i * .05, 6), 0.)
  after = radar.to_ego(radar.objects[key].x, radar.objects[key].y)
  assert after == pytest.approx(before, abs=.01)
  assert radar.selected[1] == key


def test_prediction_and_cached_input_never_add_evidence():
  radar = RadarMap()
  live = N(points=[point(1, 10., 3.5)])
  md = model()
  for i in range(20):
    radar.update(live, md, i * .05, 0.)
  obj = next(iter(radar.objects.values()))
  assert obj.hits == 1 and len(obj.history) == 1
  assert obj.vehicle < .6 and not radar.walls


def test_explicit_repeated_timestamp_cannot_add_evidence_with_new_wrapper():
  radar = RadarMap()
  for i in range(4):
    radar.update(N(points=[point(1, 10., 3.5)]), model(), i * .05, 0., radar_time=0.)
  assert next(iter(radar.objects.values())).hits == 1


def test_distance_retention_is_separate_from_display_uncertainty():
  radar = run(lambda t: [point(1, 10. + t, 3.5, 1.)])
  key = radar.selected[1]
  for i in range(25, 201):
    radar.update(N(points=[]), model(), round(i * .05, 6), 0.)
  assert key in radar.objects
  assert radar.objects[key].uncertainty > radar.MAX_DISPLAY_UNCERTAINTY
  assert key not in radar.selected


def test_points_are_removed_only_after_passing_rear_distance_or_other_bounds():
  radar = run(lambda t: [point(1, 2., 3.5)], count=21)
  key = next(iter(radar.objects))
  for i in range(21, 181):
    radar.update(N(points=[]), model(), round(i * .05, 6), 1.)
  assert key not in radar.objects


@pytest.mark.parametrize('kind', ['nan', 'reverse', 'gap', 'time_reversal', 'missing_yaw'])
def test_unknown_pose_clears_map(kind):
  radar = run(lambda t: [point(1, 10., 3.5)])
  md, now, speed = model(), 1.25, 0.
  if kind == 'nan':speed = math.nan
  if kind == 'gap':now = 2.
  if kind == 'time_reversal':now = .5
  if kind == 'missing_yaw':md.orientationRate.z = []
  radar.update(N(points=[]), md, now, speed, reverse=kind == 'reverse')
  assert not radar.objects and not radar.lanes


def test_capacity_and_sensor_source_are_bounded():
  points = [point(i, .2 + (i % 100), -10. + (i // 100) * .8) for i in range(3000)]
  points.append(N(trackId=9999, dRel=10., yRel=0., vLead=0., vRel=0., measured=True, radarSource='scc'))
  radar = RadarMap()
  radar.update(N(points=points), model(), 0., 0.)
  assert len(radar.objects) <= radar.MAX_OBJECTS
  assert all(o.source != 9999 for o in radar.objects.values())


def test_center_preference_does_not_merge_two_points():
  radar = run(lambda t: [point(1, 10. + t, 2.3, 1.), point(2, 10.3 + t, 3.5, 1.)])
  assert len(radar.objects) == 2
  assert radar.objects[radar.selected[1]].source == 2


def test_display_adapter_does_not_mutate_inputs_or_unrelated_can_fields():
  radar = run(lambda t: [point(1, 10. + t, -3.5, 1.)])
  values = dict(untouched=123)
  radar.display_values(values)
  assert values['untouched'] == 123
  assert values['RF_DETECT'] != 0 and values['LF_DETECT'] == 0
  assert 0. <= values['RF_DETECT_DISTANCE'] <= 204.7
  assert 0. <= values['RF_DETECT_LATERAL'] <= 12.7


def test_wall_can_be_revoked_by_observed_motion():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))])
  assert radar.walls
  for i in range(25, 70):
    t = round(i * .05, 6)
    radar.update(N(points=[point(k, x + (t - 1.2), 2.1, 1.) for k, x in enumerate((5., 7., 9., 11.))]), model(), t, 0.)
  assert not radar.walls


def test_measurement_time_uses_historical_ego_pose():
  radar = RadarMap()
  for i in range(30):
    now = round(i * .05, 6)
    stamp = round(now - .1, 6)
    live = N(points=[point(1, 20. - 2. * stamp, 3.5)])
    radar.update(live, model(), now, 2., radar_time=stamp)
  obj = next(iter(radar.objects.values()))
  assert not obj.motion_confirmed
  assert obj.stationary
  assert obj.x == pytest.approx(20., abs=.05)


def test_one_position_step_is_not_confirmed_motion():
  radar = run(lambda t: [point(1, 10. + (.3 if t >= .1 else 0.), 2.1)])
  assert not any(o.motion_confirmed for o in radar.objects.values())


def test_wall_limits_later_lane_expansion_without_editing_input():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))])
  assert radar.walls
  for i in range(25, 31):
    md = model()
    md.laneLines[1].y = [-3.] * 4
    radar.update(N(points=[]), md, i * .05, 0.)
    assert md.laneLines[1].y == [-3.] * 4
  assert radar._lane(7., 2.6)[0] == 0
  assert radar._lane(7., 2.6, respect_walls=True)[0] is None


@pytest.mark.parametrize('yaw', [-.12, .12])
def test_ego_turn_does_not_turn_fixed_world_point_into_moving_vehicle(yaw):
  radar = RadarMap()
  wx, wy = 20., 3.5
  px = py = angle = 0.
  for i in range(31):
    now = round(i * .05, 6)
    if i:
      px += .1 * math.cos(angle + yaw * .025)
      py += .1 * math.sin(angle + yaw * .025)
      angle += yaw * .05
    x = math.cos(angle) * (wx - px) + math.sin(angle) * (wy - py)
    y = -math.sin(angle) * (wx - px) + math.cos(angle) * (wy - py)
    radar.update(N(points=[point(1, x, y)]), model(), now, 2., yaw_rate=yaw)
  obj = next(iter(radar.objects.values()))
  assert obj.stationary and not obj.motion_confirmed
  assert (obj.x, obj.y) == pytest.approx((wx, wy), abs=.02)


@pytest.mark.parametrize('field', ['dRel', 'yRel', 'vLead', 'vRel'])
def test_nonfinite_measurement_cannot_poison_map(field):
  p = point(1, 10., 3.5)
  setattr(p, field, math.nan)
  radar = run(lambda t: [p])
  assert not radar.objects


def test_ambiguous_new_id_does_not_inherit_either_nearby_identity():
  radar = run(lambda t: [point(1, 10., 3.3), point(2, 10., 3.7)])
  previous = set(radar.objects)
  radar.update(N(points=[point(3, 10., 3.5)]), model(), 1.25, 0.)
  created = set(radar.objects) - previous
  assert len(created) == 1
  assert radar.objects[created.pop()].hits == 1


def test_weak_current_lanes_stop_admitting_new_lane_based_outputs():
  radar = run(lambda t: [point(1, 10. + t, 3.5, 1.)])
  for i in range(25, 65):
    t = round(i * .05, 6)
    radar.update(N(points=[point(1, 10. + t, 3.5, 1.)]), model(.05), t, 0.)  # below the .1 lane-use limit
  assert radar.objects and radar.lanes
  assert all(k is None for k in radar.selected)


def test_collapsed_projected_wall_segment_is_unknown_not_division_by_zero():
  assert module.interpolate([(1., 2.), (1., 3.)], 1.) is None


@pytest.mark.parametrize('y,slot', [(0., 0), (3.5, 1), (-3.5, 2)])
def test_strong_vision_can_corroborate_a_stationary_vehicle(y, slot):
  radar = RadarMap()
  for i in range(25):
    md = model()
    md.leadsV3 = [N(prob=.9, x=[11.52], y=[-y])]
    radar.update(N(points=[point(1, 10., y)]), md, round(i * .05, 6), 0.)
  assert radar.selected[slot] is not None
  assert not radar.walls


def test_capacity_eviction_cannot_remove_a_track_assigned_later_in_same_scan():
  radar = RadarMap()
  radar.MAX_OBJECTS = 3
  radar.update(N(points=[point(1, 10.), point(2, RadarMap.MAX_X - 3.), point(3, RadarMap.MAX_X)]), model(), 0., 0.)
  for obj in radar.objects.values():
    obj.existence = .1 if obj.source == 3 else 1.
  radar.update(N(points=[point(4, RadarMap.MAX_X - 1.5), point(3, RadarMap.MAX_X)]), model(), .05, 0.)
  assert len(radar.objects) == 3
  assert {3, 4}.issubset({obj.source for obj in radar.objects.values()})
  assert next(obj for obj in radar.objects.values() if obj.source == 3).hits == 2


@pytest.mark.parametrize('y', [-3.5, 3.5])
def test_opposing_vehicle_is_remembered_but_not_selected_for_side_slot(y):
  radar = run(lambda t: [point(1, 20. - 3. * t, y, -3.)])
  assert any(o.motion_confirmed for o in radar.objects.values())
  assert all(key is None for key in radar.selected)


def test_near_same_id_reflection_shift_keeps_identity_but_other_id_does_not():
  for source, count in ((1, 1), (2, 2)):
    radar = run(lambda t: [point(1, 10., 2.1)])
    radar.update(N(points=[point(source, 10.8, 2.1)]), model(), 1.25, 0.)
    assert len(radar.objects) == count
    assert next(iter(radar.objects.values())).hits == (26 if source == 1 else 25)


def test_admission_edge_has_retention_margin_and_near_zero_is_observable():
  radar = RadarMap()
  radar.update(N(points=[point(1, .05), point(2, RadarMap.MAX_X), point(3, RadarMap.MAX_X + .1)]), model(), 0., 0.)
  assert {o.source for o in radar.objects.values()} == {1, 2}
  far = next(o for o in radar.objects.values() if o.source == 2)
  far.x = RadarMap.KEEP_X - 1.
  radar.update(N(points=[]), model(), .05, 0.)
  assert far.key in radar.objects
  far.x = RadarMap.KEEP_X + .1
  radar.update(N(points=[]), model(), .1, 0.)
  assert far.key not in radar.objects


def test_capacity_preserves_near_unknown_points_against_new_distant_returns():
  radar = RadarMap()
  radar.MAX_OBJECTS = 3
  radar.update(N(points=[point(1, 2.), point(2, 5.), point(3, 10.)]), model(), 0., 0.)
  for i in range(1, 10):
    radar.update(N(points=[point(10 + i, 28.)]), model(), i * .05, 0.)
  assert {o.source for o in radar.objects.values()} == {1, 2, 3}
  assert radar.evicted == 0


def test_single_member_departure_preserves_remaining_wall_and_fixed_anchors():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))])
  before = {k: p for w in radar.walls for k, p in zip(w.members, w.points)}
  for i in range(25, 85):
    t = round(i * .05, 6)
    radar.update(N(points=[point(k, x + (t - 1.2 if k == 0 else 0.), 2.1, 1. if k == 0 else 0.)
                           for k, x in enumerate((5., 7., 9., 11.))]), model(), t, 0.)
  moving = next(o for o in radar.objects.values() if o.source == 0)
  assert moving.motion_confirmed and not moving.structure
  assert radar.walls
  assert all(p == before[k] for w in radar.walls for k, p in zip(w.members, w.points))
  assert all(o.structure for o in radar.objects.values() if o.source != 0)


def test_established_wall_survives_bounded_reflection_motion():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))])
  for i in range(25, 45):
    t = round(i * .05, 6)
    radar.update(N(points=[point(k, x + (min(1., t - 1.2) if k == 0 else 0.), 2.1)
                           for k, x in enumerate((5., 7., 9., 11.))]), model(), t, 0.)
  assert all(o.structure for o in radar.objects.values())
  assert radar.selected[1] is None


@pytest.mark.parametrize('witnesses,expected', [(1, False), (2, True)])
def test_wall_can_use_remembered_neighbor_but_needs_two_fresh_witnesses(witnesses, expected):
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9.))])
  radar.walls.clear()
  radar.update(N(points=[point(k, x, 2.1) for k, x in enumerate((5., 7., 9.)) if k < witnesses]), model(), 1.25, 0.)
  assert bool(radar.walls) is expected


def test_hybrid_keeps_unknown_lane_vehicle_and_distant_front_but_hides_wall_member():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9.))] + [point(9, 8., -3.5)])
  points = [point(20, 100.), point(0, 5., 2.1), point(9, 8., -3.5)]
  values = {'unrelated': 123}
  for p, (d, y, flag) in zip(points, radar.SLOT_FIELDS):
    values.update({d: p.dRel * .8, y: abs(p.yRel), flag: 2})
  before = dict(values)
  radar.refine_existing(values, points)
  assert values == dict(before, LF_DETECT=0)
  assert radar.selected[2] is not None


def test_hybrid_does_not_associate_a_reused_id_or_mismatched_stop_memory_output():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9.))])
  for points, x in (([None, point(0, 20., 2.1), None], 20.), ([None, point(0, 5., 2.1), None], 9.)):
    values = dict(LF_DETECT=2, LF_DETECT_DISTANCE=x * .8)
    assert radar.refine_existing(values, points)['LF_DETECT'] == 2

@pytest.mark.parametrize('middle_y,expected', [(2.1, True), (3.2, False)])
def test_two_walls_use_aligned_intermediate_point_beyond_fixed_three_meter_gap(middle_y, expected):
  points = [point(k, x, 2.1) for k, x in enumerate((2., 4., 6., 14., 16., 18.))]
  radar = run(lambda t: points)
  assert len(radar.walls) == 2
  for i in range(25, 45):
    radar.update(N(points=points + [point(20, 10., middle_y)]), model(), round(i * .05, 6), 0.)
  joined = [w for w in radar.walls if len(w.members) == 7]
  assert bool(joined) is expected
  if expected:
    assert next(o for o in radar.objects.values() if o.source == 20).structure


def test_wall_gap_stays_open_without_stationary_intermediate_evidence():
  points = [point(k, x, 2.1) for k, x in enumerate((2., 4., 6., 14., 16., 18.))]
  radar = run(lambda t: points)
  for i in range(25, 45):
    t = round(i * .05, 6)
    radar.update(N(points=points + [point(20, 9. + (t - 1.2), 2.1, 1.)]), model(), t, 0.)
  assert len(radar.walls) == 2
  assert not next(o for o in radar.objects.values() if o.source == 20).structure


def test_different_wall_directions_are_not_joined_by_a_middle_point(monkeypatch):
  # The straight-row joining does not cross a change of direction. (A bend of up to WALL_BEND is followed
  # post by post since 2026-10-14; this row bends 19 degrees, so that is switched off here.)
  monkeypatch.setattr(RadarMap, 'WALL_BEND', 0.)
  radar = run(lambda t: [point(k, x, 2.1 + (max(0., x - 14.) * .35))
                         for k, x in enumerate((2., 4., 6., 10., 14., 16., 18.))])
  assert not any(len(w.members) == 7 for w in radar.walls)


def passing(target_speed, ego=8., y=3.2, start=14.):
  """Reported values of an off-axis return while ego passes: speed and distance both
  carry the unprojected range-rate remainder ego*(1-cos(azimuth))."""
  state = dict(world=start, last=0.)
  def points(now):
    dt = now - state['last']
    state['last'] = now
    x = state['world'] - ego * now
    cosine = x / math.hypot(x, y)
    reported = ego * (1. - cosine) + target_speed * cosine
    state['world'] += reported * dt
    x = state['world'] - ego * now
    return [point(1, x, y, reported)] if x >= 1. else []
  return points


def test_fixed_off_axis_return_is_not_motion_confirmed_by_range_rate_geometry():
  radar = run(passing(0.), count=30, speed=8.)
  obj = next(iter(radar.objects.values()))
  assert obj.drift > .3
  assert not obj.motion_confirmed
  assert radar.selected == (None, None, None)
  assert max(a[1] for a in obj.history) - min(a[1] for a in obj.history) < .1


def test_slow_adjacent_vehicle_remains_motion_confirmed_after_geometry_compensation():
  radar = run(passing(2.), count=30, speed=8.)
  assert any(o.motion_confirmed for o in radar.objects.values())

def stopping_beside(radar, until, ego_speed=lambda t: 2., a_ego=-.1, blindspot=(False, False)):
  """Ego slows past a slower left-lane vehicle that leaves radar coverage at 1.5m."""
  state = radar.__dict__.setdefault('_test_state', dict(t=0., ego=0.))
  while state['t'] < until - 1e-9:
    now = state['t']
    relative = 6. + min(now, 4.8) - state['ego']
    cosine = relative / math.hypot(relative, 3.5)
    points = [point(1, relative, 3.5, ego_speed(now) * (1. - cosine) + cosine)] if relative >= 1.5 else []
    radar.update(N(points=points), model(), now, ego_speed(now), a_ego=a_ego, blindspot=blindspot)
    state['ego'] += ego_speed(now) * .05
    state['t'] = round(now + .05, 6)
  return radar


def test_stop_memory_draws_unseen_side_vehicle_at_zero_distance_until_four_metres_behind():
  radar = stopping_beside(RadarMap(), 6.)
  key = radar.selected[1]
  assert key is not None and radar.remembered[1]
  assert -4. <= radar.local[key][0] < 0.
  values = radar.display_values({})
  assert values['LF_DETECT'] and values['LF_DETECT_DISTANCE'] == 0.
  stopping_beside(radar, 8., ego_speed=lambda t: 0., a_ego=0.)
  assert radar.selected[1] == key
  stopping_beside(radar, 10., ego_speed=lambda t: 2.)
  assert radar.selected[1] is None


def test_stop_memory_is_not_drawn_while_driving_or_with_same_side_blindspot_display():
  driving = stopping_beside(RadarMap(), 6., a_ego=0.)
  assert not driving.remembered[1]  # Only the ordinary short prediction may remain.
  assert stopping_beside(driving, 7., a_ego=0.).selected[1] is None
  assert stopping_beside(RadarMap(), 6., blindspot=(True, False)).selected[1] is None
  assert stopping_beside(RadarMap(), 6., blindspot=(False, True)).selected[1] is not None

@pytest.mark.parametrize('y,slot,missing,edge', [(3.3, 1, 0, 6.), (-3.3, 2, 3, 6.), (3.3, None, 0, 3.5), (-3.3, None, 3, None)])
def test_moving_vehicle_beside_unmarked_outer_boundary_needs_road_edge_room(y, slot, missing, edge):
  radar = RadarMap()
  for i in range(30):
    now = round(i * .05, 6)
    md = model()
    md.laneLineProbs[missing] = 0.
    if edge is not None:
      md.roadEdges = [N(x=[0., 10., 30., 100.], y=[side * edge] * 4) for side in (-1., 1.)]
    radar.update(N(points=[point(1, 10. + 2. * now, y, 2.)]), md, now, 0.)
  if slot is None:
    assert radar.selected == (None, None, None)
    return
  obj = radar.objects[radar.selected[slot]]
  assert obj.motion_confirmed and not obj.interior

def test_radar_moving_class_is_vehicle_evidence_even_after_the_target_stops():
  radar = RadarMap()
  for i in range(12):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 10., 3.5)]), model(), now, 0., evidence={1: (2 if i < 2 else 1, 10.)})
  obj = radar.objects[radar.selected[1]]
  assert obj.radar_moving and obj.motion_confirmed and obj.doppler == 0


def test_radar_class_for_another_distance_or_stationary_class_is_not_evidence():
  for report in ((2, 20.), (1, 10.), (0, 10.)):
    radar = run(lambda t: [point(1, 10., 3.5)], count=12)
    radar = RadarMap()
    for i in range(12):
      radar.update(N(points=[point(1, 10., 3.5)]), model(), round(i * .05, 6), 0., evidence={1: report})
    assert not any(o.motion_confirmed or o.radar_moving for o in radar.objects.values())


def test_doppler_evidence_needs_repeated_unexplained_speed_within_45_degrees():
  radar = RadarMap()
  for i in range(4):
    radar.update(N(points=[point(1, 10., 3.5, 1.)]), model(), round(i * .05, 6), 0.)
  assert not any(o.motion_confirmed for o in radar.objects.values())
  radar.update(N(points=[point(1, 10., 3.5, 1.)]), model(), .2, 0.)
  assert next(iter(radar.objects.values())).motion_confirmed
  wide = RadarMap()
  for i in range(8):
    wide.update(N(points=[point(1, 2., 3.5, 1.)]), model(), round(i * .05, 6), 0.)
  assert not any(o.motion_confirmed for o in wide.objects.values())
  # Recorded 2026-10-12: returns 0.3-2m straight ahead reported 1-2m/s that no object had.
  near = RadarMap()
  for i in range(8):
    near.update(N(points=[point(1, 2., .5, 1.5)]), model(), round(i * .05, 6), 0.)
  assert not any(o.doppler or o.motion_confirmed for o in near.objects.values())
  # Recorded on the 130-segment set: a bus and cars alongside at 1.8-3.4m ahead, 2.0-2.6m to the side.
  beside = RadarMap()
  for i in range(8):
    now = round(i * .05, 6)
    beside.update(N(points=[point(1, 2.4 + 2. * now, -2.3, 2.)]), model(), now, 0.)
  assert next(iter(beside.objects.values())).doppler_confirmed


def test_fixed_return_reporting_less_than_the_geometric_speed_is_not_doppler_evidence():
  # Recorded 2026-10-12 at 70km/h: bollards and a wall 2-3m beside ego reported .1-.5m/s where
  # vEgo*(1-cos azimuth) was 1-5m/s, and five such samples made them vehicles.
  radar = RadarMap()
  for i in range(10):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 12. - 19. * now, 2.7, .3)]), model(), now, 19.)
  obj = next(iter(radar.objects.values()))
  assert len(radar.objects) == 1 and obj.hits == 10 and obj.doppler == 0 and not obj.motion_confirmed
  # Speed towards ego is outside the fixed-return band and stays evidence.
  oncoming = RadarMap()
  for i in range(6):
    now = round(i * .05, 6)
    oncoming.update(N(points=[point(1, 15. - 2. * now, 3.5, -2.)]), model(), now, 0.)
  assert next(iter(oncoming.objects.values())).doppler_confirmed


def test_confirmed_vehicle_keeps_its_lane_when_the_outer_line_is_too_close_but_the_road_edge_leaves_room():
  # Recorded 2026-10-12: the model's right outer line was 1.99m from the inner one, the road edge 3.3m,
  # and a radar class-2 vehicle 3.5m to the right lost its slot.
  md = N(laneLines=[N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-5.4, -1.8, 1.8, 3.79)], laneLineProbs=[.95] * 4,
         roadEdges=[N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-9., 5.1)], orientationRate=N(z=[0.]), leadsV3=[])
  radar = RadarMap()
  radar.update(N(points=[]), md, 0., 0.)
  assert radar._lane(10., -3.5, moving=True)[0] == 2
  assert radar._lane(10., -3.5)[0] is None  # An unconfirmed return there has no lane.

@pytest.mark.parametrize('y', [2.0, 2.6, 4.5, -2.0, -2.6])
def test_resting_return_too_close_to_ego_lane_line_or_past_the_lane_is_not_a_waiting_vehicle(y):
  # Reviewed video: posts, a bollard and the lead truck's corner were 1.5-2.1m from ego's axis.
  radar = run(lambda t: [point(1, 10., y)], count=40)
  assert radar.selected == (None, None, None) and not any(o.admitted for o in radar.objects.values())


def test_waiting_admission_only_while_ego_is_stopped():
  stopped = run(lambda t: [point(1, 10., 3.5, trackState=2)], count=8)
  assert stopped.selected[1] is None  # Not yet .5s at rest.
  # Recorded 2026-10-12: a return the radar only coasted (13 reports, none measured) stood beside a stopped ego.
  coasted = run(lambda t: [point(1, 10., 3.5, trackState=3)], count=40)
  assert coasted.selected[1] is None and not any(o.admitted for o in coasted.objects.values())
  stopped = run(lambda t: [point(1, 10., 3.5, trackState=2)], count=20)
  obj = stopped.objects[stopped.selected[1]]
  assert obj.admitted and not obj.motion_confirmed and obj.vehicle < .6
  for i in range(20, 60):  # A measured track for 1s only: shown while predicted, not remembered.
    stopped.update(N(points=[]), model(), round(i * .05, 6), 0.)
  assert stopped.selected[1] is None and not stopped.remembered[1]
  stopped = run(lambda t: [point(1, 10., 3.5, trackState=2)], count=45)
  obj = stopped.objects[stopped.selected[1]]
  for i in range(45, 445):  # Tracked for 2s, then dropped by the radar: remembered while ego waits.
    stopped.update(N(points=[]), model(), round(i * .05, 6), 0.)
    assert stopped.selected[1] == obj.key
  assert stopped.remembered[1] and not obj.motion_confirmed
  near_line = run(lambda t: [point(1, 10., 2.0, trackState=2)], count=20)
  assert near_line.selected[1] is None  # On the lane line, not a lane-centre vehicle.
  driving = run(lambda t: [point(1, 30. - 5. * t, 3.5, trackState=2)], count=20, speed=5.)
  assert driving.selected[1] is None and not any(o.admitted for o in driving.objects.values())


@pytest.mark.parametrize('side,released', [(.4, True), (-1.7, False)])
def test_remembered_vehicle_is_released_when_it_reappears_just_ahead_and_pulls_away(side, released):
  # Recorded 2026-10-12: a waiting car beside a stopped ego left radar coverage, drove off 10s later and stayed
  # displayed for 30s. The radar first measured it again 3.7m ahead of the remembered position, .4m to the side,
  # at 1.4m/s. Cars passing another remembered car in the next lane first appeared 1.5-1.9m to the side.
  radar = run(lambda t: [point(1, 1.8, -2.9, trackState=2)], count=45)
  waiting = radar.objects[radar.selected[2]]
  frame = 45
  def step(points):
    nonlocal frame
    radar.update(N(points=points), model(), round(frame * .05, 6), 0.)
    frame += 1
  for _ in range(100):
    step([])
  assert radar.selected[2] == waiting.key and radar.remembered[2]
  x, speed = 1.8 + 3.7, 1.4
  while x < 22.:
    step([point(2, x, -2.9 + side, speed, trackState=2, motionState=2)])
    x, speed = x + speed * .05, speed + .15
  for _ in range(40):
    step([])
  assert waiting.departed == released
  assert radar.selected[2] == (None if released else waiting.key)


@pytest.mark.parametrize('speed,remembered', [(-1.6, False), (1.6, True)])
def test_object_confirmed_only_by_speed_towards_ego_is_not_remembered_beside_ego(speed, remembered):
  # Recorded 2026-10-12: a return 16-20m ahead, 3.4m to the right, reported -0.9..-1.9m/s (towards ego) and was
  # confirmed. 14s later two coasted reports of the same radar id (+0.57m/s) re-linked it and it was displayed
  # as a remembered vehicle for 47s on an empty crosswalk. A vehicle travelling ego's way never has such speed.
  radar, frame = RadarMap(), 0
  def step(points):
    nonlocal frame
    radar.update(N(points=points), model(), round(frame * .05, 6), 0.)
    frame += 1
  start = 18. if speed < 0. else 10.
  for i in range(50):
    step([point(37, start + speed * i * .05, -3.4, speed, trackState=2 if i % 3 else 3)])
  obj = next(iter(radar.objects.values()))
  assert obj.doppler_confirmed and bool(obj.forward_samples) == remembered
  for _ in range(280):
    step([])
  x, y = radar.local[obj.key]
  for _ in range(2):
    step([point(37, x, y, .57, trackState=3)])
  assert len(radar.objects) == 1 and obj.hits == 52
  for i in range(100):
    step([])
    if i >= 8:  # Past the short prediction of the last measurement.
      assert (radar.selected[2] == obj.key and radar.remembered[2]) == remembered


@pytest.mark.parametrize('moving_class,remembered', [(2, False), (1, True)])
def test_position_left_behind_by_a_moving_radar_track_is_not_remembered(moving_class, remembered):
  # Recorded 2026-10-12: a car pulling away beside a stopped ego was reported at 2.5-4m/s for .5s while its
  # reported position stayed within .5m, then the same radar id jumped 1.17m (beyond the association gate)
  # with the radar's moving class and drove off. The object left behind was displayed as a remembered
  # vehicle for 1.75s. A resting track that jumps (another reflection on the same body) keeps its memory.
  radar, frame = RadarMap(), 0
  def step(points):
    nonlocal frame
    radar.update(N(points=points), model(), round(frame * .05, 6), 0.)
    frame += 1
  for i, x in enumerate((3.4, 3.55, 3.15, 3.15, 3.3, 3.35, 3.4, 3.45, 3.6, 3.3, 3.7)):
    step([point(39, x, -2.9, 2.6 + .12 * i, trackState=2 if i >= 2 else 3)])
  stale = next(iter(radar.objects.values()))
  assert stale.doppler_confirmed and len(radar.objects) == 1
  x, speed = 3.58 + 1.17, 4.2
  while x < 22.:
    step([point(39, x, -2.9, speed if moving_class == 2 else 0., trackState=2, motionState=moving_class)])
    if moving_class == 2:
      x, speed = x + speed * .05, speed + .15
    elif frame > 40:
      break
  assert len(radar.objects) == 2 and stale.superseded == (not remembered)
  for i in range(40):
    step([])
    if i >= 8 and moving_class == 2:
      assert radar.selected[2] is None


def test_stop_memory_ignores_a_vehicle_that_was_crossing_when_it_left_coverage():
  radar = RadarMap()
  for i in range(20):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 4., 5.2 - 3. * now, 3., yvRel=-3.)]), model(), now, 0., evidence={1: (2, 4.)})
  for i in range(20, 40):
    radar.update(N(points=[]), model(), round(i * .05, 6), 0., evidence={})
  assert not radar.remembered[1] and not radar.remembered[2]

def test_object_that_walked_sideways_is_neither_a_waiting_vehicle_nor_remembered():
  # 395/7: pedestrians crossing in front paused in the adjacent lane and were remembered for 24s.
  radar = RadarMap()
  for i in range(80):
    now = round(i * .05, 6)
    y = max(3.5, 6.5 - 1.2 * now)  # Walks 3m sideways at 10m, then stands in the left lane.
    radar.update(N(points=[point(1, 10., y, trackState=2)]), model(), now, 0.)
  assert not any(o.admitted for o in radar.objects.values())
  for i in range(80, 140):
    radar.update(N(points=[]), model(), round(i * .05, 6), 0.)
  assert radar.selected[1] is None and not radar.remembered[1]


def test_undetected_opposing_vehicle_does_not_reappear_as_a_side_display():
  # Review 2026-10-12: freezing an unseen object's position must not erase its last real direction.
  radar = RadarMap()
  for i in range(24):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 12. - now, 3.5, -1.)]), model(), now, 0., evidence={1: (2, 12. - now)})
    assert radar.selected[1] is None
  for i in range(24, 124):
    radar.update(N(points=[]), model(), round(i * .05, 6), 0., evidence={})
    assert radar.selected[1] is None, i

def test_radar_vehicle_evidence_releases_one_wall_member_without_removing_the_wall():
  # Review 2026-10-12: a wall member later classified as moving by the radar must not stay hidden.
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))])
  assert radar.walls and all(o.structure for o in radar.objects.values())
  for i in range(25, 35):
    radar.update(N(points=[point(k, x, 2.1) for k, x in enumerate((5., 7., 9., 11.))]), model(), round(i * .05, 6), 0.,
                 evidence={0: (2, 5.)})
  first = next(o for o in radar.objects.values() if o.source == 0)
  assert first.motion_confirmed and not first.structure
  assert radar.walls and first.key not in radar.walls[0].members
  assert all(o.structure for o in radar.objects.values() if o.source != 0)
  shown = radar.refine_existing({'FF_DISTANCE': 0., 'FF_DETECT': 0, 'LF_DETECT_DISTANCE': 4., 'LF_DETECT': 2, 'RF_DETECT_DISTANCE': 0., 'RF_DETECT': 0},
                                [None, point(0, 5., 2.1), None])
  assert shown['LF_DETECT'] == 2

def test_sporadic_doppler_samples_do_not_accumulate_into_vehicle_evidence():
  # Review 2026-10-12: one unexplained-speed sample per second on a resting return is noise, not travel.
  radar = RadarMap()
  for i in range(160):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 10., 3.5, 1. if i % 20 == 0 else 0.)]), model(), now, 5.)
  obj = next(iter(radar.objects.values()))
  assert obj.doppler >= 5 and not obj.doppler_confirmed and not obj.motion_confirmed


def test_confirmed_doppler_motion_stays_vehicle_evidence_after_the_vehicle_stops():
  radar = RadarMap()
  for i in range(60):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 10. + min(now, .5), 3.5, 1. if now < .5 else 0.)]), model(), now, 0.)
  obj = radar.objects[radar.selected[1]]
  assert obj.doppler_confirmed and obj.motion_confirmed and obj.stationary

def test_unseen_object_is_shown_only_as_a_short_prediction_or_as_stop_memory():
  # Review 2026-10-12: an object unseen for seconds must not stay an ordinary candidate
  # and bypass the stop-memory conditions (ego stopped/slowing, no same-side blind-spot display).
  def scene(speed, blindspot):
    radar = RadarMap()
    for i in range(20):
      now = round(i * .05, 6)
      radar.update(N(points=[point(1, 3., 3.5)]), model(), now, speed, evidence={1: (2, 3.)}, blindspot=blindspot)
    assert radar.selected[1] is not None
    shown = []
    for i in range(20, 70):
      now = round(i * .05, 6)
      radar.update(N(points=[]), model(), now, speed, evidence={}, blindspot=blindspot)
      shown.append((round(now - .95, 2), radar.selected[1] is not None, radar.remembered[1]))
    return shown
  stopped = scene(0., (False, False))
  assert all(on for _, on, _ in stopped)                                   # resting vehicle: prediction, then memory
  assert all(memory == (age > RadarMap.PREDICTION_HOLD) for age, _, memory in stopped)
  blind = scene(0., (True, False))
  assert all(on == (age <= RadarMap.PREDICTION_HOLD) for age, on, _ in blind)   # memory suppressed by the blind-spot display
  driving = scene(0., (False, False))[:1] and RadarMap()
  for i in range(20):
    driving.update(N(points=[point(1, 20. - 3. * i * .05, 3.5, 2.)]), model(), round(i * .05, 6), 5., evidence={1: (2, 20. - 3. * i * .05)})
  assert driving.selected[1] is not None
  for i in range(20, 50):
    driving.update(N(points=[]), model(), round(i * .05, 6), 5., evidence={})
    assert (driving.selected[1] is not None) == (round(i * .05 - .95, 2) <= RadarMap.PREDICTION_HOLD)

@pytest.mark.parametrize('target,resting', [(0., True), (10.6, False)])
def test_resting_is_judged_by_net_movement_against_ego_travel_not_by_report_scatter(target, resting):
  # Recorded 2026-10-12 at 67-77km/h: reported positions of fixed returns moved a median .5-.7m between
  # reports, so "all reports within .4m" classed 0-3 of 10-18 roadside returns as resting and no wall
  # formed. Real vehicles there moved a median 10.6m/s; none moved less than .3m in the window.
  radar = run(lambda t: [point(1, 19.5 + (target - 19.) * t + (.25 if round(t * 20) % 2 else -.25), 3.2, target)], count=16, speed=19.)
  obj = next(iter(radar.objects.values()))
  assert obj.hits == 16 and obj.stationary == resting


@pytest.mark.parametrize('target,resting', [(2.5, False), (0., True)])
def test_vehicle_the_radar_reports_driving_is_not_classed_as_resting_by_the_travel_bound(target, resting):
  # Recorded on 130 segments: the travel bound alone classed 525 object-sends of radar class-2 vehicles
  # reporting over 2m/s as resting (slower than 15% of ego's speed), which zeroes their map velocity.
  # A class-2 vehicle that has stopped still is (30,301 object-sends while ego stands).
  def points(t):
    x = 19.5 + (target - 19.) * t + (.25 if round(t * 20) % 2 else -.25)
    return [point(1, x, 3.2, target, motionState=2)]
  radar = run(points, count=16, speed=19.)
  obj = next(iter(radar.objects.values()))
  assert obj.hits == 16 and obj.radar_moving and obj.stationary == resting


@pytest.mark.parametrize('speed,shown', [(-1.6, False), (1.6, True)])
def test_object_confirmed_only_by_speed_towards_ego_is_not_a_side_display_once_it_rests(speed, shown):
  # Recorded on 130 segments once resting returns were recognised at speed: roadside returns of radar class 1
  # with five samples of speed towards ego had been kept out of the side slots only by the map's noisy
  # negative velocity. Classed as resting they hold no velocity, and 154 sends showed them beside ego.
  # Ego drives at 6m/s: no waiting admission. The object moves for 1.5s, then rests.
  radar = RadarMap()
  for i in range(60):
    x = (24. if speed < 0. else 20.) + speed * min(i, 30) * .05 - 6. * i * .05
    radar.update(N(points=[point(37, x, -3.4, speed if i < 30 else 0., trackState=2)]), model(), round(i * .05, 6), 6.)
  obj = next(iter(radar.objects.values()))
  assert obj.doppler_confirmed and obj.stationary and obj.last_motion[0] > -1. / 3.6
  assert (radar.selected[2] == obj.key) == shown


def curved_model(bend=.012):
  xs = [0., 4., 8., 12., 16., 20., 30., 100.]
  return N(laneLines=[N(x=xs, y=[y - bend * x * x for x in xs]) for y in (-5.4, -1.8, 1.8, 5.4)],
           laneLineProbs=[.95] * 4, orientationRate=N(z=[0.]), leadsV3=[])


@pytest.mark.parametrize('follows_road', [True, False])
def test_posts_at_a_constant_offset_from_the_lane_line_form_a_band_on_a_curve(follows_road):
  # The model's y is right-positive: these lines bend left. Posts .3m beyond ego's left line do not lie
  # on a straight row (a row takes three of the five), but share one offset from the line.
  posts = [(i, x, 2.1 + (.012 * x * x if follows_road else .45 * (i % 2))) for i, x in enumerate((4., 7., 10., 13., 16.))]
  radar = RadarMap()
  for i in range(25):
    radar.update(N(points=[point(k, x, y) for k, x, y in posts]), curved_model(), round(i * .05, 6), 0.)
  assert len(radar.bands) == follows_road
  assert all(o.structure for o in radar.objects.values()) == follows_road
  if follows_road:
    band = radar.bands[0]
    assert len(band.members) == 5 and abs(band.offset - .3) < .05 and abs(band.spacing - 3.) < .1


@pytest.mark.parametrize('offset,evidence,joins', [(.3, {}, True), (1.1, {}, False), (.3, {9: (2, 19.)}, False)])
def test_band_takes_the_next_post_at_its_first_report_unless_it_is_elsewhere_or_a_vehicle(offset, evidence, joins):
  # Recorded on 130 segments: 8,152 returns joined this way, none later had radar class 2 or a vision match.
  posts = [(i, x, 2.1 + .012 * x * x) for i, x in enumerate((4., 7., 10., 13., 16.))]
  radar = RadarMap()
  for i in range(25):
    radar.update(N(points=[point(k, x, y) for k, x, y in posts]), curved_model(), round(i * .05, 6), 0.)
  new = point(9, 19., 1.8 + offset + .012 * 19. * 19., trackState=3)
  radar.update(N(points=[point(k, x, y) for k, x, y in posts] + [new]), curved_model(), 1.25, 0., evidence=evidence)
  obj = max(radar.objects.values(), key=lambda o: o.key)
  assert obj.source == 9 and obj.hits == 1 and not obj.stationary
  assert obj.structure == joins


@pytest.mark.parametrize('outer,final,stays', [(5.4, 5.9, False), (5.4, 5.2, True), (4.9, 5.2, True), (3.8, 4.3, True)])
def test_vehicle_settled_beyond_the_outer_line_is_in_the_lane_beyond_but_is_followed_while_it_moves_there(outer, final, stays):
  # Recorded 2026-10-13: cars that had moved on to the second lane over stayed LF for seconds at .3-.8m past
  # the outer line (the radar reports the near corner), one of them instead of the car in the adjacent lane.
  # Not so inside the line, within a lane width of the inner line (recorded: a car in a last lane whose line
  # was drawn .4m too close), or where the outer line is drawn under 2.3m from the inner one.
  md = N(laneLines=[N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-outer, -1.8, 1.8, 5.4)], laneLineProbs=[.95] * 4,
         roadEdges=[N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-9., 9.)], orientationRate=N(z=[0.]), leadsV3=[])
  radar, shown = RadarMap(), []
  for i in range(100):
    t = round(i * .05, 6)
    y = 3.5 + (final - 3.5) * min(1., max(0., t - 1.))
    radar.update(N(points=[point(1, 8., y, 15., motionState=2, trackState=2)]), md, t, 15.)
    shown.append(radar.selected[1] is not None)
  obj = next(iter(radar.objects.values()))
  crossed = 1. + (outer - 3.5) / (final - 3.5) if final > outer else 9.
  assert all(shown[10:int(min(crossed + 1., 4.9) * 20)])  # Never dropped while changing lane, nor in the second after.
  assert shown[-1] == stays and obj.settled == (final > outer)


@pytest.mark.parametrize('state,confirmed', [(2, True), (0, False)])
def test_unexplained_speed_confirms_a_vehicle_only_on_a_track_with_measured_reports(state, confirmed):
  # Recorded 2026-10-13: tracks that never had a measured report (track state 2) carried five samples of
  # speed and were shown 4.7m and 9.6m from ego with nothing there. The samples are still collected.
  radar = RadarMap()
  for i in range(8):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 15. - 2. * now, 3.5, -2., trackState=state)]), model(), now, 0.)
  obj = next(iter(radar.objects.values()))
  assert obj.doppler == 8 and obj.tracked == (8 if state == 2 else 0)
  assert obj.doppler_confirmed == confirmed and obj.motion_confirmed == confirmed
  # Once measured reports arrive, the samples already collected confirm it.
  for i in range(8, 11):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 15. - 2. * now, 3.5, -2., trackState=2)]), model(), now, 0.)
  assert obj.doppler_confirmed


@pytest.mark.parametrize('yaw,samples', [(.6, False), (.1, True)])
def test_speed_reported_by_a_fixed_return_while_ego_turns_is_not_vehicle_evidence(yaw, samples):
  # Recorded 2026-10-13 at .5-.75rad/s and 10-18km/h: a wall corner 30deg to the side reported 2-3.5m/s
  # (the radar moves sideways with the turn; the report allows for forward speed only) and became a vehicle.
  radar = RadarMap()
  for i in range(10):
    now = round(i * .05, 6)
    radar.update(N(points=[point(1, 6., -3., 2.5, trackState=2)]), model(), now, 4., yaw_rate=yaw)
  objects = list(radar.objects.values())
  assert sum(o.hits for o in objects) == 10
  assert any(o.doppler for o in objects) == samples and any(o.motion_confirmed for o in objects) == samples


@pytest.mark.parametrize('row,y,waiting', [(True, 3.4, False), (False, 3.4, True), (True, 4.3, True)])
def test_lone_resting_return_on_the_line_of_a_post_row_is_not_a_waiting_vehicle(row, y, waiting):
  # Recorded 2026-10-13: bollards on the centre line, 1.6m beyond ego's left line where the model drew a
  # lane. The row had an 11.5m gap; the post beyond it was not taken into the band and, once ego stopped,
  # was shown for 50s as a vehicle waiting in that lane. A return off that line, or without a row, still is one.
  posts = [(i, .5 + 1.2 * i, 3.4) for i in range(6)] if row else []  # To 6.5m; the lone return is 8m beyond.
  radar = RadarMap()
  for i in range(40):
    radar.update(N(points=[point(k, x, py) for k, x, py in posts] + [point(20, 14.5, y)]), model(), round(i * .05, 6), 0.)
  lone = next(o for o in radar.objects.values() if o.source == 20)
  assert len(radar.bands) == row and not lone.structure
  assert lone.admitted == waiting and (radar.selected[1] == lone.key) == waiting


def bent_model(bend):
  xs = [float(x) for x in range(0, 101, 5)]
  return N(laneLines=[N(x=xs, y=[y - bend * x * x for x in xs]) for y in (-5.4, -1.8, 1.8, 5.4)],
           laneLineProbs=[.95] * 4, orientationRate=N(z=[0.]), leadsV3=[])


@pytest.mark.parametrize('bend,lead,beyond,side', [(.002, True, .8, False), (.002, True, 2., True), (.002, False, .8, True), (0., True, .8, True)])
def test_vehicle_beyond_the_lead_just_past_egos_lane_line_on_a_curve_is_not_a_side_display(bend, lead, beyond, side):
  # Recorded 2026-10-13: the second vehicle ahead in ego's lane, 45-75m away on a curve, was shown LF/RF (here
  # 44m: beyond 50m the later far limit hides it anyway). The
  # model draws far curves too straight, so it lay .1-1.4m past ego's lane line. On a straight road a real
  # neighbour's near corner is reported .9-1.3m past the line, and without a lead nothing hides it either way.
  radar = RadarMap()
  centre = bend * 44. * 44.  # The model's y is right-positive: this lane bends left.
  for i in range(30):
    points = [point(2, 44., centre + 1.8 + beyond, 15., motionState=2)] + ([point(1, 15., bend * 225., 15., motionState=2)] if lead else [])
    radar.update(N(points=points), bent_model(bend), round(i * .05, 6), 15.)
  assert (radar.selected[0] is not None) == lead
  assert (radar.selected[1] is not None) == side


@pytest.mark.parametrize('bends_at,lane,side', [(60., 0., False), (45., 0., True), (60., 3., True)])
def test_vehicle_beyond_the_lead_that_drove_through_egos_lane_is_not_a_side_display(bends_at, lane, side):
  # The road bends left and the model still draws it straight: the vehicle 45m ahead, beyond the lead, appears
  # 2.5-4m to the left. Bending 15m after it was first seen, its recorded positions 5-25m ahead of ego lie in
  # ego's lane, so it is the second lead. Bending where it was first seen it has no such record and stays, as
  # does one in the next lane.
  radar, shown = RadarMap(), []
  for i in range(int((bends_at + 19. - 45.) / .75) + 3):
    t = round(i * .05, 6)
    bend = .01 * max(0., 45. + 15. * t - bends_at) ** 2
    y = lane if lane else bend  # The one in the next lane is reported on a straight line, 1.2m past ego's lane line.
    radar.update(N(points=[point(1, 15., 0., 15., motionState=2), point(2, 45., y, 15., motionState=2)]), model(), t, 15.)
    if 2.4 <= bend <= 3.6:
      shown.append(radar.selected[1] is not None)
  assert len(shown) >= 4 and all(s == side for s in shown[1:])


@pytest.mark.parametrize('lead,x,side', [(15., 48., True), (15., 55., True), (15., 65., False), (None, 65., True), (40., 65., True)])
def test_side_slot_far_behind_the_displayed_lead_is_not_given(lead, x, side):
  # Recorded 2026-10-13: with a lead displayed, vehicles 58-79m away in ego's own lane were shown RF/LF on a
  # straight road, 1.7-2.5m past ego's lane line. Far lane geometry is off by up to a lane width there. Beyond
  # 60m a side slot is given only without a lead, or within 30m of the lead. 50-60m is given (2026-10-14: the
  # side vehicles the existing selector showed there were real cars on video).
  radar = RadarMap()
  for i in range(30):
    points = [point(2, x, 3.5, 15., motionState=2)] + ([point(1, lead, 0., 15., motionState=2)] if lead else [])
    radar.update(N(points=points), model(), round(i * .05, 6), 15.)
  assert (radar.selected[0] is not None) == (lead is not None)
  assert (radar.selected[1] is not None) == side


def test_far_side_candidate_behind_the_lead_does_not_displace_stop_memory_while_holding():
  # Recorded 2026-10-14 (356/21, 3c1/21, 3c1/23, 3dd/28-29): while ego was stopped a 50-60m side vehicle behind
  # the lead replaced the remembered car right beside ego. While holding the side limit stays at 50m.
  radar = stopping_beside(RadarMap(), 6.)
  key = radar.selected[1]
  assert key is not None and radar.remembered[1]
  now = radar._test_state['t']
  for i in range(30):
    points = [point(7, 15., 0., 3., motionState=2), point(8, 55., 3.5, 3., motionState=2)]
    radar.update(N(points=points), model(), round(now + i * .05, 6), 0., a_ego=0.)
  assert radar.holding and radar.selected[0] is not None
  assert radar.selected[1] == key and radar.remembered[1]
def bridge_scene(lead, steps=60, vision=lambda i: True, lane_change=lambda i: False, lead_x=12., vision_y=lambda i: 0., lead_y=lambda i: 0.):
  """Ego at 15m/s; a lead ``lead_x`` ahead in ego's lane until step 30, then whatever ``lead(i)`` reports; a second
  vehicle beyond it in the lane (40m, or 70m behind a far lead). Returns the map and, per step, the FF object's key and source."""
  radar, shown = RadarMap(), []
  for i in range(steps):
    m = model()
    m.leadsV3 = [N(prob=.9, x=[lead_x + 1.52], y=[-vision_y(i)])] if vision(i) else []
    far = 70. if lead_x > 40. else 40.
    points = [point(2, far, -.5, 15., motionState=2)] + ([point(1, lead_x, lead_y(i), 15., motionState=2)] if i < 30 else lead(i))
    radar.update(N(points=points), m, round(i * .05, 6), 15., lane_change=lane_change(i))
    ff = radar.objects.get(radar.selected[0])
    shown.append(None if ff is None else (ff.key, ff.source, radar.local[ff.key]))
  return radar, shown


@pytest.mark.parametrize('vision,lane_change,kept', [(True, False, True), (False, False, False), (True, True, False)])
def test_lead_track_jumping_beside_ego_lane_stays_ff_while_vision_sees_the_lead(vision, lane_change, kept):
  # Recorded 2026-10-14: next to a trailer, the radar track of the car 10m ahead jumped 1.2-1.7m sideways across
  # the lane line for 1.5s (the radar merged it with the trailer) while vision kept the car in ego's lane, and FF
  # moved to a car 42m ahead. The lead stays FF, drawn where it was last reported in the lane. Without vision,
  # or while ego changes lanes, the ordinary selection decides.
  radar, shown = bridge_scene(lambda i: [point(1, 12., -2.6, 15., motionState=2)], vision=lambda i: vision or i < 30,
                              lane_change=lambda i: lane_change and i >= 30)
  lead = shown[29][0]
  after = shown[31:]
  assert all(s is not None for s in after)
  assert all(s[0] == lead for s in after) == kept
  if kept:
    assert all(abs(s[2][1]) < .3 for s in after)
  else:
    assert after[-1][2][0] > 30.


@pytest.mark.parametrize('x,kept', [(12., True), (55., False)])
def test_lead_track_pushed_slowly_across_the_line_stays_ff_while_vision_sees_the_lead(x, kept):
  # Recorded 2026-10-14 (3e1/19 16.9s): an overtaking SUV pushed the track of the car 8m ahead 2m to the right
  # over a second, with no jump, while vision kept the car in the lane; FF jumped to a car 21m ahead. Beyond
  # 50m the vision gate and the lane lines are too loose for this (there the far FF hold still keeps it for .3s).
  radar, shown = bridge_scene(lambda i: [point(1, x, -.1 * (i - 29), 15., motionState=2)] if i < 50 else [point(1, x, -2., 15., motionState=2)],
                              steps=60, lead_x=x)
  lead = shown[29][0]
  assert all(s[0] == lead for s in shown[30:]) == kept
  if kept:
    assert radar.bridge['cause'] == 'drift'


def test_lead_leaving_the_lane_in_vision_too_is_not_held():
  # A lead really changing lanes leaves the lane in vision as well (recorded: a median 1s before the map).
  radar, shown = bridge_scene(lambda i: [point(1, 12., -.1 * (i - 29), 15., motionState=2)], vision_y=lambda i: -.1 * max(0, i - 29))
  assert shown[-1][2][0] > 30. and radar.bridge is None


def test_lost_lead_continues_under_a_new_radar_id_near_its_bridged_position():
  # The lead's report stops; the radar reports it again under another id .5s later, 1m from where it was drawn
  # (beyond the ordinary association gate). It is the same object,
  # not a new one that must wait for its own reports before it is shown.
  radar, shown = bridge_scene(lambda i: [] if i < 40 else [point(3, 12.8, -.6, 15., motionState=2)])
  lead = shown[29][0]
  assert all(s[0] == lead for s in shown[30:])
  assert shown[-1][1] == 3 and radar.bridge is None


def test_bridge_ends_after_its_hold_time():
  radar, shown = bridge_scene(lambda i: [point(1, 12., -2.6, 15., motionState=2)], steps=80)
  lead = shown[29][0]
  start = 30
  hold = round(RadarMap.BRIDGE_HOLD / .05)
  assert all(s[0] == lead for s in shown[start:start + hold - 1])
  assert shown[-1][0] != lead and shown[-1][2][0] > 30.


@pytest.mark.parametrize('x,takes', [(7., True), (40., False)])
def test_nearer_vehicle_takes_ff_from_a_bridged_lead_but_a_farther_one_cannot(x, takes):
  radar, shown = bridge_scene(lambda i: [point(1, 12., -2.6, 15., motionState=2)] + ([point(4, x, 0., 15., motionState=2)] if i >= 33 else []))
  lead = shown[29][0]
  assert (shown[-1][0] != lead) == takes
  if takes:
    assert shown[-1][1] == 4


@pytest.mark.parametrize('weaving', [False, True])
def test_bridged_lead_is_drawn_where_its_reports_last_stood_still(weaving):
  # Recorded 2026-10-14: a slow merge pushes the last in-lane report to the line (3e1/19 drawn 1.6m right of a
  # car vision saw centred). The bridged lead is drawn at the mean of its reports from before the push, once
  # they had stood still for .5s; without such a stretch, at its last in-lane report as before.
  pre = (lambda i: .3 if i % 2 else -.3) if weaving else (lambda i: .2)
  radar, shown = bridge_scene(lambda i: [point(1, 12., .2 - .12 * (i - 29), 15., motionState=2)], lead_y=pre)
  bridged = [s for s in shown[40:] if radar.bridge is not None]
  lateral = radar.local[radar.bridge['key']][1]
  assert bridged and radar.bridge['cause'] == 'drift'
  if weaving:
    assert lateral < -1.  # The last in-lane report, already pushed towards the line.
  else:
    assert abs(lateral - .2) < .1


@pytest.mark.parametrize('angle,moving,joins', [(20., False, True), (40., False, False), (20., True, False)])
def test_wall_follows_a_row_bending_into_a_turn_pocket(angle, moving, joins):
  # Recorded 2026-10-14 (3e3/15): posts beside ego 1.6m left, then bending 20 degrees outwards where a turn pocket
  # opens. The bent part joins the wall post by post, so its first post is not admitted as a waiting vehicle.
  # A sharper bend does not, nor does a vehicle at the wall's end that the radar classes as moving.
  slope = math.tan(math.radians(angle))
  posts = [(i, -4. + 2. * i, 1.6) for i in range(5)]  # To 4m.
  radar = RadarMap()
  for i in range(30):
    points = [point(k, x, y) for k, x, y in posts]
    if i >= 10:
      points.append(point(20, 6.5, 1.6 + 2.5 * slope, motionState=2 if moving else 0))
    radar.update(N(points=points), model(), round(i * .05, 6), 0.)
  bent = next(o for o in radar.objects.values() if o.source == 20)
  assert bent.structure == joins
  assert not (joins and bent.admitted)



def far_lead(left, x=70., lane_change=lambda i: False, near=lambda i: None):
  """Ego at 15m/s and a vehicle ``x`` ahead at the same speed, reported 1.6m left; ego's left lane line is drawn
  ``left[i]`` from the centre (1.8 keeps the vehicle in ego's lane, 1.4 puts it in the left lane), as far lane
  lines wander. Optionally a nearer in-lane car ``near(i)`` metres ahead. Per step, the FF object's source."""
  radar, shown = RadarMap(), []
  for i, line in enumerate(left):
    m = N(laneLines=[N(x=[0., 10., 30., 100., 200.], y=[y] * 5) for y in (-5.4, -line, 1.8, 5.4)],
          laneLineProbs=[.95] * 4, orientationRate=N(z=[0.]), leadsV3=[])
    points = [point(1, x, 1.6, 15., motionState=2)] + ([point(2, near(i), 0., 15., motionState=2)] if near(i) else [])
    radar.update(N(points=points), m, round(i * .05, 6), 15., lane_change=lane_change(i))
    ff = radar.objects.get(radar.selected[0])
    shown.append(None if ff is None else ff.source)
  return shown


@pytest.mark.parametrize('x,kept', [(70., True), (40., False)])
def test_far_lead_stays_ff_while_its_lane_slot_flickers(x, kept):
  # Recorded 2026-10-14: 339 of 356 FF flickers beyond 50m were the lead's lane slot leaving ego's lane for a
  # cycle or two (243 of them .1s) while it was measured throughout. Beyond 50m the displayed FF object stays
  # FF through such a flicker; nearer, the lane decides at once as before.
  shown = far_lead([1.8] * 30 + [1.4, 1.4, 1.8, 1.4, 1.8, 1.8, 1.4, 1.4, 1.8, 1.8], x)
  assert all(s == 1 for s in shown[20:30])
  assert all(s == 1 for s in shown[30:]) == kept


@pytest.mark.parametrize('case', ['leaves', 'lane change', 'nearer car'])
def test_far_lead_hold_is_short_and_yields(case):
  # The hold lasts FF_FAR_HOLD: a lead that stays out of ego's lane leaves FF after .3s. It is not applied
  # while ego changes lanes, and a nearer car in ego's lane takes FF at once.
  if case == 'nearer car':  # The lane keeps flickering, so without yielding the far lead would stay FF.
    shown = far_lead([1.8] * 30 + [1.4, 1.8] * 10, near=lambda i: 30. if i >= 32 else None)
    assert all(s == 1 for s in shown[30:32]) and all(s == 2 for s in shown[38:])
    return
  shown = far_lead([1.8] * 30 + [1.4] * 15, lane_change=lambda i: case == 'lane change' and i >= 30)
  assert shown[29] == 1
  if case == 'leaves':
    assert all(s == 1 for s in shown[30:36]) and all(s is None for s in shown[38:])
  else:
    assert all(s is None for s in shown[31:])


@pytest.mark.parametrize('mode', ['prefer_adjacent', 'alone', 'outer_valid', 'ego_change', 'stopped'])
def test_missing_outer_line_prefers_an_adjacent_candidate_without_removing_other_policies(mode):
  radar = RadarMap()
  for n in range(30):
    now = n * .05
    md = model()
    md.laneLineProbs[3] = .95 if mode == 'outer_valid' else 0.
    if mode == 'outer_valid':
      md.laneLines[3].y = [6.4] * 4
    md.roadEdges = [N(x=[0., 10., 30., 100.], y=[v] * 4) for v in (-9., 9.)]
    points = [point(42, 6., -5.7, 10., motionState=2)]
    if mode != 'alone':
      points.append(point(34, 12., -3., 10., motionState=2))
    radar.update(N(points=points), md, now, 0. if mode == 'stopped' else 10., lane_change=mode == 'ego_change')
  assert radar.objects[radar.selected[2]].source == (34 if mode == 'prefer_adjacent' else 42)

