"""Exact geometry equivalence for the display map's cached/vectorized path."""
import math
import struct
from types import SimpleNamespace as N

import numpy as np
import pytest

from tools.ccnc.test_radar_display import H, Tracker  # noqa: TID251
from tools.ccnc.test_radar_map import RadarMap, model  # noqa: TID251


def bits(value):
  if isinstance(value, float):
    return struct.pack('!d', value)
  if isinstance(value, (list, tuple)):
    return tuple(bits(v) for v in value)
  if isinstance(value, dict):
    return {k: bits(v) for k, v in value.items()}
  return value


@pytest.mark.parametrize('invalid', [None, 'nan', 'length', 'reversed', 'short'])
def test_cached_lane_transforms_match_scalar_bits_through_pose_changes(invalid):
  rng = np.random.default_rng(1024)
  scalar, cached, curves = RadarMap(), RadarMap(), Tracker()
  for frame in range(12):
    md = model()
    md.timestampEof = frame
    md.roadEdges = [N(x=[], y=[]), N(x=[], y=[])]
    for index, line in enumerate(md.laneLines + md.roadEdges):
      line.x = np.linspace(-8., 210., 33).tolist()
      line.y = (rng.normal(0., .2, 33) + (index - 1.5) * 3.6).tolist()
    md.laneLineProbs = [.05 if frame % 3 == 0 else .95] * 4
    if invalid == 'nan':
      md.laneLines[1].y[4] = math.nan
    elif invalid == 'length':
      md.laneLines[1].y.pop()
    elif invalid == 'reversed':
      md.laneLines[1].x.reverse()
    elif invalid == 'short':
      md.laneLines[1].x, md.laneLines[1].y = [0.], [1.]
    curves._update_model(md)
    for radar, callback in ((scalar, None), (cached, curves._curve)):
      radar.update(N(points=[]), md, frame * .05, 12., yaw_rate=.17, lane_change=True, lane_curves=callback)
    assert bits(cached.geometry) == bits(scalar.geometry)
    assert bits({k: (v.points, v.probability, v.consistent) for k, v in cached.lanes.items()}) == bits(
      {k: (v.points, v.probability, v.consistent) for k, v in scalar.lanes.items()})
    for side in (0, 1):
      assert bits(H['_ccnc_side_lane_center'](md, side, curves._curve)) == bits(H['_ccnc_side_lane_center'](md, side))


@pytest.mark.parametrize('offset', [-2, -1, 0, 1, 2])
@pytest.mark.parametrize('probability', [0., .1, .6, math.nan])
def test_ego_lines_match_full_lane_view(offset, probability):
  radar = RadarMap()
  radar.lane_offset = offset
  radar.lanes = {i: N(probability=probability, consistent=3) for i in range(6)}
  radar.geometry = {i: [(0., float(i)), (10., i + .2)] for i in range(6)}
  for x in (-1., 0., 3.75, 10., 11.):
    bounds, usable, _ = radar._lane_view(x)
    expected = tuple(bounds.get(i) if i in usable else None for i in (1, 2))
    assert bits(radar._ego_lines(x)) == bits(expected)


def test_curve_cache_ends_with_model_reader_or_timestamp():
  tracker = Tracker()
  first = model()
  first.timestampEof = 1
  tracker._update_model(first)
  curve = tracker._curve(first, 1)
  tracker._update_model(first)
  assert tracker._curve(first, 1) is curve
  second = model()
  second.timestampEof = 1
  tracker._update_model(second)
  assert tracker._curve(second, 1) is not curve
  curve = tracker._curve(second, 1)
  second.timestampEof = 2
  tracker._update_model(second)
  assert tracker._curve(second, 1) is not curve


@pytest.mark.parametrize('angle', [0., -.7, .3, math.pi])
def test_vector_geometry_matches_scalar_bits_including_empty_and_duplicate_points(angle):
  radar = RadarMap()
  radar.pose = (31.9, -12.7, angle)
  radar.cos, radar.sin = math.cos(angle), math.sin(angle)
  radar.lanes = {i: N(points=points) for i, points in enumerate(([], [(2., -0.), (2., -0.), (2.0005, 1.)],
                     [(x, math.sin(x) * 4.) for x in np.linspace(-8., 200., 33).tolist()]))}
  expected = {}
  for i, lane in radar.lanes.items():
    points = sorted(radar.to_ego(x, y) for x, y in lane.points)
    expected[i] = [p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]
  radar._geometry()
  assert bits(radar.geometry) == bits(expected)
