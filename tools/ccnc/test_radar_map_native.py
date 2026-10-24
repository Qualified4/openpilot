"""Compare optional native map loops against their retained Python reference."""
from collections import deque
import math
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace as N

import numpy as np
import pytest

from opendbc.car.hyundai import ccnc_map_native
from tools.ccnc.test_radar_map import model, point  # noqa: TID251


PATH = Path(__file__).resolve().parents[2] / 'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_radar_map.py'


def load(native):
  mod = ModuleType('map_native_test_' + str(native))
  sys.modules[mod.__name__] = mod
  source = PATH.read_text()
  if not native:
    source = source.replace('from opendbc.car.hyundai.ccnc_map_native import loops as _native', '_native = None')
  exec(compile(source, str(PATH), 'exec'), mod.__dict__)
  return mod


def bits(value):
  if isinstance(value, (float, np.floating)):
    return struct.pack('!d', value)
  if isinstance(value, dict):
    return {k: bits(v) for k, v in value.items()}
  if isinstance(value, (tuple, list, deque)):
    return tuple(bits(v) for v in value)
  if isinstance(value, set):
    return value
  if hasattr(value, '__dict__'):
    return {k: bits(v) for k, v in vars(value).items() if k not in ('generation', 'raw', 'model')}
  return value


@pytest.fixture
def pair():
  if ccnc_map_native.loops is None:
    pytest.skip('optional CCNC map extension is not built')
  return load(False), load(True)


@pytest.mark.parametrize('disabled,broken', [(True, None), (False, ImportError('missing')), (False, OSError('ABI')),
                                          (False, RuntimeError('initialization')), (False, 'incomplete')])
def test_loader_explicit_off_and_missing_or_broken_extension(monkeypatch, disabled, broken):
  path = Path(ccnc_map_native.__file__)
  mod = ModuleType('map_loader_test')
  monkeypatch.setenv('CCNC_RADAR_MAP_NATIVE', '0' if disabled else '1')
  if broken:
    import builtins
    original = builtins.__import__

    def imp(name, *args, **kwargs):
      if name == 'opendbc.car.hyundai':
        if broken == 'incomplete':
          return N(_ccnc_map_native=N())
        raise broken
      return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', imp)
  exec(compile(path.read_text(), str(path), 'exec'), mod.__dict__)
  assert mod.loops is None and mod.BACKEND == 'python'


@pytest.mark.parametrize('offset', [-2, -1, 0, 1, 2])
def test_geometry_transform_interpolation_and_lane_view_bits(pair, offset):
  py, cy = pair
  rng = np.random.default_rng(1024)
  a, b = py.CcncRadarMap(), cy.CcncRadarMap()
  for angle in (-math.pi, -.7, -0., 0., .3, math.pi):
    points = [[], [(0., -0.), (0., 0.), (.001, 1.), (.001000001, 2.)]]
    points += [list(zip(np.linspace(-8., 210., 64).tolist(), rng.normal(0., 4., 64).tolist(), strict=True)) for _ in range(4)]
    for radar in (a, b):
      radar.pose = (31.9, -12.7, angle)
      radar.cos, radar.sin = math.cos(angle), math.sin(angle)
      radar.lane_offset = offset
      radar.lanes = {i: N(points=p, probability=(.05, .1, .6, .95)[i % 4], consistent=i) for i, p in enumerate(points)}
    assert bits(a._geometry()) == bits(b._geometry())
    assert bits(a.geometry) == bits(b.geometry)
    for x in (-100., -0., 0., .001, 10., 30., 200., math.inf, -math.inf, math.nan):
      assert bits(a._lane_view(x)) == bits(b._lane_view(x))
      for pts in a.geometry.values():
        assert bits(py.interpolate(pts, x)) == bits(cy.interpolate(pts, x))
    for x, y in points[-1] + [(0., -0.), (-0., 0.)]:
      assert bits(a.to_ego(x, y)) == bits(b.to_ego(x, y))


@pytest.mark.parametrize('cached', [False, True])
def test_full_map_state_bits_with_invalid_models_memory_and_lane_changes(pair, cached):
  from tools.ccnc.test_radar_display import Tracker  # noqa: TID251
  py, cy = pair
  a, b, curves = py.CcncRadarMap(), cy.CcncRadarMap(), Tracker()
  rng = np.random.default_rng(414)
  for frame in range(180):
    md = model()
    md.timestampEof = frame
    md.meta = N(desire='laneChangeLeft' if 40 <= frame < 90 else 'none', laneChangeState='off')
    md.roadEdges = [N(x=[-8., 0., 30., 210.], y=[y] * 4) for y in (-9., 9.)]
    md.laneLineProbs = [.05 if frame % 11 == 0 else .95] * 4
    for line in md.laneLines:
      line.y = [y + .1 * math.sin(frame * .1) for y in line.y]
    if frame % 13 == 0:
      md.laneLines[1].y[1] = math.nan
    elif frame % 17 == 0:
      md.laneLines[2].x.reverse()
    elif frame % 19 == 0:
      md.laneLines[0].y.pop()
    curves._update_model(md)
    pts = [point(i, 3. + i * 3. + frame * .005, (i % 3 - 1) * 3.5 + rng.normal(0., .05), 1.2, motionState=2)
           for i in range(1, 8)] if frame < 130 else []
    live = N(points=pts)
    for radar in (a, b):
      radar.update(live, md, frame * .05, 0. if frame > 100 else 5., yaw_rate=.03,
                   lane_change=40 <= frame < 90, lane_curves=curves._curve if cached else None)
    assert bits(a) == bits(b), frame
