import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type.native import _runs
from openpilot.selfdrive.carrot.lane_type.native import native_observation
from openpilot.selfdrive.carrot.lane_type.dual_roi import project
from openpilot.selfdrive.carrot.lane_type.temporal import DistanceEvidence
from tools.lane_type.admit_labels import wilson  # noqa: TID251 - offline repository tooling


def profile(travel=0., solid=False):
  rows = []
  for d in np.arange(4.5, 20.01, .25):
    ink = int(solid or any(a <= d+travel < b for a, b in [(5, 8), (13, 16), (21, 24)]))
    rows.append({'distance': float(d), 'ink': ink, 'color': 'yellow' if ink else None, 'confidence': .99})
  return rows


def test_stationary_repetition_does_not_create_observed_distance_or_a_solid():
  accumulator = DistanceEvidence()
  results = [accumulator.update(profile(solid=True), [-1.7, -1.7], 0., i*200_000_000) for i in range(20)]
  assert all(r['type'] == 'unknown' for r in results)
  assert results[-1]['observedDistinctM'] == results[0]['observedDistinctM']
  assert results[-1]['accumulatedTravelM'] == 0.


def test_moving_pattern_uses_distinct_physical_paint_and_gaps_not_frame_votes():
  accumulator = DistanceEvidence()
  results = [accumulator.update(profile(i), [-1.7, -1.7], 5., i*200_000_000) for i in range(8)]
  assert any(r['type'] == 'dashed' for r in results)
  assert results[0]['type'] == 'unknown'
  assert max(r['observedDistinctM'] for r in results) <= 23.
  assert results[-1]['color'] == 'yellow'


@pytest.mark.parametrize('reason', ['lane_change', 'lost', 'clock', 'boundary'])
def test_tracking_discontinuities_clear_accumulated_evidence(reason):
  a = DistanceEvidence()
  for i in range(8):
    a.update(profile(i), [-1.7, -1.7], 5., i*200_000_000)
  kw = {'lane_change': reason == 'lane_change', 'tracked': reason != 'lost'}
  r = a.update(profile(), [-.5, -.5] if reason == 'boundary' else [-1.7, -1.7], 5.,
               0 if reason == 'clock' else 1_600_000_000, **kw)
  assert r['trackingReset'] and r['type'] == 'unknown'


def test_occluded_and_unknown_pixels_are_not_paint_gaps():
  rows = profile()
  for r in rows:
    if r['ink'] == 0:
      r['ink'] = None
  assert not _runs(rows)['repeatingPatternObserved']
  a = DistanceEvidence()
  for i in range(8):
    a.update(profile(i), [-1.7, -1.7], 5., i*200_000_000)
  r = a.update([], [-1.7, -1.7], 5., 1_600_000_000)
  assert r['type'] == r['color'] == 'unknown'


def test_zero_errors_in_small_audit_is_not_accuracy_certification():
  assert wilson(0, 20) > .1
  assert wilson(0, 100) > .02
  assert wilson(0, 200) < .02


def test_native_inputs_retain_source_colours_and_never_pad_missing_pixels():
  rgb = np.full((760,1344,3),80,np.uint8)
  line = {'x':[0.,10.,25.,40.], 'y':[-1.7]*4, 'z':[1.3]*4}
  k = np.array([[1141.5,0,672],[0,1141.5,380],[0,0,1]])
  maps = project(line,[0,0,0],k,np.linspace(4.5,25,1000),half_width=.07,columns=5)
  for y,x in zip(np.rint(maps[1][:,2]).astype(int),np.rint(maps[0][:,2]).astype(int),strict=True):
    if 0 <= y < 760 and 3 <= x < 1341:
      rgb[y,x-3:x+4] = 220
  result = native_observation(rgb,line,.99,[0,0,0],k)
  assert set(result['heads']) == {'type','color'}
  for head, size in [('type',64),('color',32)]:
    h=result['heads'][head]
    assert h['image'].shape == (size,16,3)
    assert h['uniqueSourceRows'] == size and not h['paddingUsed']
    assert set(np.unique(h['image'])) <= {80,220}  # no invented interpolation levels
    assert h['far'] <= (25.01 if head == 'type' else 15.01)
  hidden = native_observation(rgb,line,.1,[0,0,0],k)
  assert hidden['heads'] == {} and hidden['profile'] == []
