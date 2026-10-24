"""Exact arithmetic/state checks for the optional CCNC options loops."""
from collections import deque
import math
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace as N

import numpy as np
import pytest

from opendbc.car.hyundai import ccnc_map_native
from tools.ccnc.test_radar_map import model  # noqa: TID251


def bits(value):
  if isinstance(value, float):
    return struct.pack('!d', value)
  if isinstance(value, np.ndarray):
    return value.dtype.str, value.shape, value.tobytes()
  if isinstance(value, dict):
    return {k: bits(v) for k, v in value.items()}
  if isinstance(value, (list, tuple, deque)):
    return tuple(bits(v) for v in value)
  if hasattr(value, '__dict__'):
    return {k: bits(v) for k, v in vars(value).items() if k != 'vision_model'}
  return value


@pytest.fixture
def native():
  if ccnc_map_native.options is None:
    pytest.skip('CCNC options extension is not built')
  return ccnc_map_native.options


@pytest.mark.parametrize('size', [1, 2, 3, 4, 33, 64])
def test_interp_matches_numpy_bits(native, size):
  rng = np.random.default_rng(1024 + size)
  xs = np.cumsum(rng.uniform(.01, 4., size))
  ys = rng.normal(0., 4., size)
  ys[0] = -0.
  queries = [*xs, *rng.uniform(xs[0] - 1., xs[-1] + 1., 1000), -math.inf, math.inf, math.nan]
  for q in queries:
    assert bits(native.interp_np(q, xs, ys)) == bits(float(np.interp(q, xs, ys)))
  # A strided array must retain its values without relying on C-contiguous input.
  for q in queries:
    assert bits(native.interp_np(q, xs[::2], ys[::2])) == bits(float(np.interp(q, xs[::2], ys[::2])))
  with pytest.raises(ValueError):
    native.interp_np(0., np.array([]), np.array([]))
  with pytest.raises(ValueError):
    native.interp_np(0., xs, np.array([]))


@pytest.mark.parametrize('x,y', [([], []), ([0.], [1.]), ([0., 1.], [0., 1.]), ([0., 1.], [0.]),
                              ([0., 0.], [1., 2.]), ([1., 0.], [1., 2.]), ([0., math.nan], [1., 2.]),
                              ([0., 1.], [1., math.inf]), ([[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]])])
def test_boundary_validation_matches_reference(native, x, y):
  if len(x) < 2 or len(x) != len(y):
    expected = False
  else:
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    expected = bool(np.isfinite(xs).all() and np.isfinite(ys).all() and (xs[1:] > xs[:-1]).all())
  assert native.valid_boundary(x, y) == expected


def test_nearest_line_ties_and_domain_match(native):
  curves = [(index, np.array([0., 10., 30.]), np.array([y, y + .1, y + .5])) for index, y in ((1, -1.8), (2, 1.8))]
  for x in (-1., 0., 2.5, 10., 30., 31.):
    for y in (-3.5, -0., 0., 1.8, 3.5):
      expected = min([(abs(y + float(np.interp(x, xs, ys))), i) for i, xs, ys in curves if xs[0] <= x <= xs[-1]],
                     default=(math.inf, None))
      assert bits(native.nearest_boundary(x, y, curves)) == bits(expected)


@pytest.fixture
def reference():
  path = Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py'
  mod = ModuleType('options_native_reference')
  sys.modules[mod.__name__] = mod
  source = path.read_text(encoding='utf8').replace('from opendbc.car.hyundai.ccnc_map_native import options as _native_options',
                                                '_native_options = None')
  exec(compile(source, str(path), 'exec'), mod.__dict__)
  return mod


@pytest.mark.parametrize('invalid', [None, 'nan', 'short', 'prob', 'reverse'])
def test_centers_and_ego_follow_state_bits(native, reference, invalid):
  py, cy = reference._CcncVehiclePositionCorrection(), reference._CcncVehiclePositionCorrection()
  for frame in range(100):
    md = model()
    md.timestampEof = frame
    md.meta = N(laneChangeState='laneChangeStarting' if 20 <= frame < 50 else 'off')
    for line in md.laneLines:
      line.y = [y + .2 * math.sin(frame * .1) + .02 * x for x, y in zip(line.x, line.y, strict=True)]
    if invalid == 'nan':
      md.laneLines[1].y[1] = math.nan
    elif invalid == 'short':
      md.laneLines[1].x.pop()
    elif invalid == 'prob':
      md.laneLineProbs[1] = .09
    elif invalid == 'reverse':
      md.laneLines[1].x.reverse()
    for side in (0, 1):
      assert bits(native.side_lane_center(md, side)) == bits(reference._ccnc_side_lane_center(md, side))
    args = md, 12., 3 if 20 <= frame < 50 else 0, frame >= 20, False, frame * .05
    py.update_ego_follow(*args)
    native.update_ego_follow(cy, *args)
    assert bits(py) == bits(cy)


@pytest.mark.parametrize('map_on,options_on', [(True, True), (False, True), (True, False), (False, False)])
def test_options_and_map_switches_are_independent(native, monkeypatch, map_on, options_on):
  monkeypatch.setenv('CCNC_RADAR_MAP_NATIVE', '1' if map_on else '0')
  monkeypatch.setenv('CCNC_OPTIONS_NATIVE', '1' if options_on else '0')
  path = Path(ccnc_map_native.__file__)
  env = {}
  exec(compile(path.read_text(), str(path), 'exec'), env)
  assert (env['loops'] is not None) == map_on
  assert (env['options'] is not None) == options_on
