"""Original selector plus map classification, without changing the live module."""
from types import SimpleNamespace as N

import pytest

from opendbc.can import CANPacker
from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as existing
from tools.ccnc.test_radar_map import RadarMap, model, module, point, run


def geometry(stamp=0.):
  md = model()
  md.timestampEof = int(stamp * 1e9)
  md.roadEdges = [N(x=[0., 10., 30., 100.], y=[y] * 4) for y in (-7., 7.)]
  md.position = N(x=[0., 10., 30., 100.], y=[0.] * 4)
  md.meta = N(laneChangeState='off')
  return md


@pytest.fixture
def setup(monkeypatch):
  monkeypatch.setattr(existing, 'state', N())
  monkeypatch.setattr(existing, '_options', None)
  existing.reset()
  existing.configure(True, True, True, True)
  radar = RadarMap()
  display = module.integrated_tracker(existing.state.radar_display_tracker, radar)
  existing.state.radar_display_tracker = display
  packer = CANPacker('hyundai_canfd_generated')
  fields = packer.dbc.name_to_msg['CCNC_0x162'].sigs
  cs = N(out=N(leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False, gearShifter='drive'),
         live_tracks=None, ccnc_0x162=dict.fromkeys(fields, 0))

  def step(frame, points):
    now = frame * .01
    md = geometry(now)
    cs.live_tracks = N(points=points)
    radar.update(cs.live_tracks, md, now, 0., select=False)
    values = existing.update_vehicles(dict(cs.ccnc_0x162), cs, md, frame, 0., 0.)
    assert values['FF_DETECT'] != 7, 'Existing exception fallback was entered'
    packer.make_can_msg('CCNC_0x162', 0, values)
    return values

  return radar, display, step


def test_preselection_finds_next_vehicle_and_does_not_remember_wall(setup):
  _, display, step = setup
  for frame in range(0, 151, 5):
    values = step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)]
                  + [point(99, 12. + frame * .01, 3.4, 1.)])
  assert display.selected[1] == 99
  assert values['LF_DETECT']
  assert not any(saved[0] in (1, 2, 3) for holds in (display.stop_holds, display.approach_holds) for saved in holds.values())
  assert display.temporal is not None and display.position_correction is not None


def test_removing_wall_does_not_seed_unknown_static_replacement_memory(setup):
  _, display, step = setup
  for frame in range(0, 151, 5):
    values = step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)] + [point(99, 12., 3.4)])
  assert display.selected[1] is None
  assert not values['LF_DETECT']
  assert not display.stop_holds


def test_ordinary_stationary_vehicle_admission_and_stop_memory_remain(setup):
  _, display, step = setup
  for frame in range(0, 151, 5):
    values = step(frame, [point(99, 12., 3.4)])
  assert values['LF_DETECT']
  assert display.selected[1] == 99
  assert display.stop_holds[0][0] == 99


@pytest.mark.parametrize('protection', ['motion', 'vision', 'stop', 'approach'])
def test_existing_vehicle_guards_protect_wall_member(setup, protection):
  radar, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)])
  key = 1
  birth = display.tracks[key][0]
  if protection == 'motion':
    display.boundary_admission[key].update(status='allowed', moving=(0, (3., 2.1), (1., 0.)))
  elif protection in ('stop', 'approach'):
    holds = display.stop_holds if protection == 'stop' else display.approach_holds
    holds[0] = (key, 3., 2.1, (1, 2.4, 2.1), birth) if protection == 'stop' else (key, 3., 2.1, (1, 2.4, 2.1), 100, 0., birth)
  md = geometry(1.)
  if protection == 'vision':
    md.leadsV3 = [N(prob=.99, x=[4.52], y=[-2.1])]
  display.update_boundary_admission(md)
  assert key not in display.map_blocked
  assert radar.walls


def test_source_id_reconnection_and_prediction_cannot_extend_wall(setup):
  radar, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)])
  old = display.tracks.pop(1)
  alias = 1 << 32
  old[3].trackId = alias
  old[3].ccnc_source_id = 1
  display.tracks[alias] = old
  display.update_boundary_admission(geometry(1.))
  assert alias in display.map_blocked
  old[3].ccnc_fresh = old[3].measured = False
  count = sum(len(w.members) for w in radar.walls)
  display.last_frame = 110
  display.update_boundary_admission(geometry(1.1))
  assert alias in display.map_blocked
  assert sum(len(w.members) for w in radar.walls) == count
  display.last_frame = 135
  display.update_boundary_admission(geometry(1.35))
  assert alias not in display.map_blocked


def test_far_row_crossing_thirty_metres_keeps_original_suppression(setup):
  _, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((27., 29., 31., 33.), 1)])
  display.selected = (None, 3, None)
  assert display.structure_suppresses(0, geometry(1.), 100)
  assert display.structure_regions


def test_fresh_stationary_extension_is_limited_by_distance_not_by_missing_positions():
  # Was "crosses one missing position only". Recorded 356/2 shows a slowing radar reporting a
  # 1.5m post row with 4-9m between fresh returns, so the limit is WALL_CONTINUE metres ahead of
  # the wall end, however many posts were skipped. A return farther than that still does not join.
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((3., 5., 7.), 1)])
  # A later lane expansion must not prevent a point aligned with an established wall.
  for frame in range(25, 42):
    now = frame * .05
    radar.update(N(points=[point(k, x, 2.1) for k, x in enumerate((3., 5., 7., 11., 19.5), 1)]), model(), now, 0.)
  by_source = {o.source: o for o in radar.objects.values()}
  by_source[4].interior = True
  radar.extend_observed_walls({by_source[4].key, by_source[5].key})
  assert by_source[4].structure
  assert not by_source[5].structure


def test_wall_extension_uses_observed_spacing_variation():
  radar = run(lambda t: [point(k, x, 2.1) for k, x in enumerate((3., 4.65, 7.44, 9.14), 1)])
  objects = {o.source: o for o in radar.objects.values()}
  radar.walls = [module.Wall(tuple(objects[k].key for k in (1, 2, 3)), [(objects[k].x, objects[k].y) for k in (1, 2, 3)])]
  objects[4].structure = False
  radar.extend_observed_walls({objects[4].key})
  assert objects[4].structure


def test_structure_update_runs_once_even_when_selector_calls_it_again(setup, monkeypatch):
  _, display, step = setup
  calls = []
  original = existing._CcncRadarDisplayTracker.update_structure_regions
  def count(*args, **kwargs):
    calls.append(args[2])
    return original(*args, **kwargs)
  monkeypatch.setattr(existing._CcncRadarDisplayTracker, 'update_structure_regions', count)
  step(0, [])
  step(5, [])
  assert calls == [0, 5]


def test_shared_lane_validation_matches_independent_geometry():
  independent, shared = RadarMap(), RadarMap()
  display = existing._CcncRadarDisplayTracker()
  for frame in range(30):
    now = frame * .05
    md = geometry(now)
    if frame in (8, 9):
      md.laneLines[0].x = [0., 10., 10., 100.]
    if frame == 12:
      md.laneLineProbs[1] = .1
    live = N(points=[point(1, 12., 2.1)])
    display._update_model(md)
    independent.update(live, md, now, 4., yaw_rate=.03, select=False)
    shared.update(live, md, now, 4., yaw_rate=.03, select=False, lane_curves=display._curve)
    assert independent.snapshot() == shared.snapshot()


def test_preselection_reuses_the_original_raw_projection_cache(setup):
  _, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(1, 10., 2.1, .7)])
  display.update_boundary_admission(geometry(1.))
  projected = display._projection
  assert projected is not None
  assert display._projection_live is display.temporal.raw
  assert display.lane_projection(display.temporal.raw) is projected


def test_row_classification_coasts_without_renewing_measurement_deadline(setup):
  _, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(1, 10., 2.1, .7)])
  birth, stamp, _, p, _ = display.tracks[1]
  p.ccnc_fresh = p.measured = False
  display.map_blocked[1] = birth, None, stamp
  for frame in (110, 120, 130):
    display.last_frame = frame
    display.update_boundary_admission(geometry(frame * .01))
    assert display.map_blocked[1] == (birth, None, stamp)
  display.last_frame = 135
  display.update_boundary_admission(geometry(1.35))
  assert 1 not in display.map_blocked


def test_unmeasured_packets_with_new_timestamps_cannot_renew_map_rejection(setup):
  _, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)])
  saved = display.map_blocked[1]
  birth, stamp, count, p, coordinates = display.tracks[1]
  p.measured = False
  assert p.ccnc_fresh
  for frame in (110, 120, 130):
    display.tracks[1] = birth, frame, count, p, coordinates
    display.last_frame = frame
    display.structure_regions.clear()
    display._map_rows.clear()
    display.structure_frame = frame
    display.update_boundary_admission(geometry(frame * .01))
    assert display.map_blocked[1] == saved
  display.tracks[1] = birth, 135, count, p, coordinates
  display.last_frame = display.structure_frame = 135
  display.update_boundary_admission(geometry(1.35))
  assert 1 not in display.map_blocked


def test_new_near_points_do_not_duplicate_legacy_stationary_memory(setup):
  radar, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1) for k, x in enumerate((3., 5., 7., 27., 29., 31., 33.), 1)])
  assert radar.objects
  assert display.stationary_map
  assert all(p['x'] >= 15. for p in display.stationary_map)


def test_unusable_map_does_not_bypass_original_row_guard(setup):
  radar, display, step = setup
  for frame in range(0, 101, 5):
    step(frame, [point(k, x, 2.1, .7) for k, x in enumerate((3., 5., 7.), 1)])
  display.selected = (None, 1, None)
  radar.usable = False
  display.map_blocked.clear()
  assert display.structure_suppresses(0, geometry(1.), 100)
