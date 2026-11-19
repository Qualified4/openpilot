"""The Rust _measurements/_associate loops give the same bits as the Python reference, on the real map class."""
from collections import deque
import importlib.util
import math
import pickle
import random
import struct
from types import SimpleNamespace as N

import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.hyundai import ccnc_core_loader
from opendbc.car.hyundai import hyundaicanfd_ccnc_radar_map as radar_map
from opendbc.car.hyundai.hyundaicanfd_ccnc_radar_map import CcncRadarMap, MapObject

SCENARIOS = 24
FRAMES = 600
DT = .05
LANES = (5.25, 1.75, -1.75, -5.25)


def bits(value):
  """Type-strict and bit-exact form of a value: 1, 1.0 and True all differ, as do 0.0 and -0.0."""
  if isinstance(value, float):
    return 'f', struct.pack('<d', value)
  if isinstance(value, (bool, int, str, type(None))):
    return type(value).__name__, value
  if isinstance(value, np.generic):
    return bits(value.item())
  if isinstance(value, dict):
    return 'd', tuple((bits(k), bits(v)) for k, v in value.items())
  if isinstance(value, (tuple, list, deque)):
    return type(value).__name__, getattr(value, 'maxlen', None), tuple(bits(v) for v in value)
  if isinstance(value, (set, frozenset)):
    return 's', frozenset(bits(v) for v in value)
  if hasattr(value, '__dict__'):
    return 'o', bits({k: v for k, v in vars(value).items() if k not in ('generation', 'raw', 'model', 'log')})
  raise TypeError(type(value))


def same(a, b):
  """bits(a) == bits(b); equal pickles (bit-exact floats, types, order) decide quickly, bits() decides the rest."""
  try:
    if pickle.dumps(a) == pickle.dumps(b):
      return True
  except pickle.PicklingError:  # Classes of a module loaded from its file only.
    pass
  return bits(a) == bits(b)


def state(m):
  return {k: v for k, v in vars(m).items() if k not in ('generation', 'raw', 'model', 'log')}


class Recorder:
  """Records each _measurements/_associate result and the id state they leave; the computation is the real class's."""

  def __init__(self):
    super().__init__()
    self.log = []
    self.associating = False
    self.lane_checks = 0

  def reset(self):
    log, lane_checks = self.log, self.lane_checks
    super().reset()
    self.log, self.lane_checks = log, lane_checks

  def _measurements(self, live, stamp):
    out = super()._measurements(live, stamp)
    self.log.append(('measurements', out, pickle.dumps(self.reports), pickle.dumps(self.jumps)))
    return out

  def _associate(self, measurements, stamp):
    self.associating = True
    try:
      out = super()._associate(measurements, stamp)
    finally:
      self.associating = False
    self.log.append(('associate', out))
    return out

  def _lane(self, *args, **kwargs):
    self.lane_checks += self.associating  # Continuation candidates: the lane check left in Python.
    return super()._lane(*args, **kwargs)


class RustMap(Recorder, CcncRadarMap):
  pass


class PythonMap(Recorder, CcncRadarMap):
  RUST = None


@pytest.fixture(scope='module', autouse=True)
def rust_core():
  core = ccnc_core_loader.load()
  assert core is not None, 'the x86_64 test build must be in prebuilt/linux-x86_64'
  assert CcncRadarMap.RUST is core and core.abi_version() == 2
  return core


def model(rng, frame, *, desire='none', lead=None, bend=0.):
  xs = [-8., 0., 10., 20., 30., 50., 80., 120., 210.]
  wobble = .03 * math.sin(frame * .07)
  lines = [N(x=xs, y=[-(y + bend * x * x + wobble) for x in xs]) for y in LANES]
  probs = [.9, .95, .95, .9]
  if frame % 37 == 0:
    probs[0] = .05
  leads = []
  if lead is not None:
    leads.append(N(x=[lead[0] + 1.52], y=[-lead[1]], v=[lead[2]], prob=.95))
  if frame % 29 == 0:
    leads.append(N(x=[rng.uniform(25., 120.)], y=[rng.uniform(-6., 6.)], v=[], prob=.85))
  return N(laneLines=lines, laneLineProbs=probs, roadEdges=[N(x=xs, y=[-8.75] * len(xs)), N(x=xs, y=[8.75] * len(xs))],
           leadsV3=leads, meta=N(desire=desire, laneChangeState='off'), orientationRate=N(z=[0.]))


def speed_profile(kind, frame):
  t = frame * DT
  if kind == 'still':
    return 0.
  if kind == 'stop':  # Slowing into a stop (holding), waiting, then pulling away.
    return max(0., 8. - 1. * t) if t < 15. else min(6., .8 * (t - 22.)) if t > 22. else 0.
  if kind == 'cruise':
    return 15. + 2. * math.sin(t * .3)
  return 3. + math.sin(t * .5)  # 'slow'


class Scenario:
  """Synthetic radar frames for one seed: targets plus the corner cases the loops must treat like Python."""

  def __init__(self, seed):
    self.rng = rng = random.Random(seed)
    self.seed = seed
    self.kind = ('still', 'stop', 'cruise', 'slow')[seed % 4]
    self.yaw = (0., .02, -.05, .4)[(seed // 4) % 4]  # .4 rad/s blocks Doppler samples.
    self.crowded = seed % 6 == 5  # Enough new returns to reach MAX_OBJECTS.
    self.evidence = seed % 3 == 1
    big = seed % 5 == 0
    self.targets = []
    for i in range(rng.randint(2, 12)):
      moving = rng.random() < .5
      lane = rng.choice((0., 3.5, -3.5, 7., -7.))
      self.targets.append(dict(id=(2 ** 63 + i if big and i % 2 else 100 * seed + i), x=rng.uniform(-3., 60.),
                               y=lane + rng.uniform(-.6, .6), v=rng.uniform(.5, 20.) if moving else 0., moving=moving))
    # A slowly creeping neighbour beside a stopped ego: the stop-continuation candidate.
    self.targets.append(dict(id=900 + seed, x=4., y=3.5, v=1.6, moving=True, creep=True))
    self.posts = [(2. * rng.randint(-2, 9) + rng.choice((0., -0., 1e-12, -1e-12)), rng.choice((-1., 1.)) * (2. * rng.randint(0, 5)))
                  for _ in range(rng.randint(0, 6))]

  def frame(self, frame, v_ego):
    rng, points = self.rng, []
    for t in self.targets:
      if t.get('creep') and not (180 <= frame < 420 and v_ego < .5):
        continue
      t['x'] += (t['v'] - v_ego) * DT
      if not -6. <= t['x'] <= 150.:
        t['x'] = rng.uniform(5., 60.) if t['v'] >= v_ego else rng.uniform(70., 140.)
      y = t['y'] + rng.gauss(0., .05)
      # The same id leaving far away: a jump beyond the association gate, sideways and forward.
      if rng.random() < .03:
        y += rng.choice((-1., 1.)) * rng.uniform(.8, 3.)
      x = t['x'] + (rng.uniform(5., 30.) if rng.random() < .01 else 0.)
      points.append(dict(trackId=t['id'], dRel=x, yRel=y, vRel=t['v'] - v_ego, vLead=t['v'] + rng.gauss(0., .2),
                         measured=rng.random() > .05, radarSource='frontRadar',
                         motionState=2 if t['moving'] and rng.random() < .9 else 1, trackState=rng.choice((2, 2, 2, 1, 3))))
    for j, (x, y) in enumerate(self.posts):  # Exact 2m grid boundaries (ego still: map frame == ego frame).
      points.append(dict(trackId=5000 + j, dRel=x, yRel=y, vRel=-v_ego, vLead=0., measured=True,
                         radarSource='frontRadar', motionState=1, trackState=2))
    if self.crowded:
      for j in range(rng.randint(10, 20)):
        points.append(dict(trackId=10000 + frame * 32 + j, dRel=rng.uniform(-4., 21.), yRel=rng.uniform(-12., 12.),
                           vRel=-v_ego, vLead=rng.uniform(-1., 1.), measured=True, radarSource='frontRadar',
                           motionState=rng.choice((0, 1, 2)), trackState=2))
    if points and rng.random() < .3:  # Duplicated return under a second id: equal association scores.
      twin = dict(points[rng.randrange(len(points))])
      twin['trackId'] = 7000 + rng.randrange(4)
      points.append(twin)
    special = rng.random()
    if special < .1:
      bad = dict(points[0]) if points else dict(trackId=1, dRel=5., yRel=0., vRel=0., vLead=0., measured=True,
                                                radarSource='frontRadar', motionState=0, trackState=0)
      bad[rng.choice(('dRel', 'yRel', 'vLead', 'vRel'))] = rng.choice((math.nan, math.inf, -math.inf))
      points.append(bad)
    elif special < .2:  # Out of range: behind, too far, too wide, implausible speed.
      points.append(dict(trackId=8000, dRel=rng.choice((-1., 205., 30.)), yRel=rng.choice((0., 13., 41.)),
                         vRel=0., vLead=rng.choice((0., 150.)), measured=True, radarSource='frontRadar', motionState=2, trackState=2))
    elif special < .3:
      points.append(dict(trackId=8100, dRel=10., yRel=4., vRel=0., vLead=0., measured=True,
                         radarSource=rng.choice(('corner235', 'corner180', 'scc')), motionState=2, trackState=2))
    rng.shuffle(points)
    return points[:32]

  def evidence_for(self, points):
    if not self.evidence:
      return None
    out = {}
    for p in points:
      if self.rng.random() < .5:
        motion = self.rng.choice((1, 2))
        distance = p['dRel'] + self.rng.choice((0., .4, 3.))  # Linked, linked, not linked.
        out[p['trackId']] = (motion, distance) if self.rng.random() < .5 else (motion, distance, self.rng.choice((1, 2)))
    return out


def capnp_live(points):
  msg = structs.RadarData.new_message()
  out = msg.init('points', len(points))
  for dst, p in zip(out, points, strict=True):
    for name, value in p.items():
      setattr(dst, name, value)
  return msg


def namespace_live(points):
  return N(points=[N(**p) for p in points])


def run(seed, maps, frames=FRAMES):
  scenario = Scenario(seed)
  rng = random.Random(seed + 1000)
  stats = dict(calls=0, measurements=0, assignments=0, jumps=0, max_objects=0, evicted=0, holding=0)
  for frame in range(frames):
    now = 1000. + frame * DT
    v_ego = speed_profile(scenario.kind, frame)
    points = scenario.frame(frame, v_ego)
    live = capnp_live(points) if frame % 2 == 0 else namespace_live(points)
    lead = next(((p['dRel'], p['yRel'], p['vLead']) for p in points if abs(p['yRel']) < 1.5 and 0. < p['dRel'] < 80.
                 and all(math.isfinite(p[k]) for k in ('dRel', 'yRel', 'vLead'))), None)
    md = model(rng, frame, desire='laneChangeLeft' if 300 <= frame < 330 else 'none', lead=lead, bend=.001 if seed % 2 else 0.)
    evidence = scenario.evidence_for(points)
    radar_time = now - (.02 if frame % 7 else .2)  # Some frames are too old to observe.
    for m in maps:
      m.update(live, md, now, v_ego, radar_time=radar_time, yaw_rate=scenario.yaw, evidence=evidence,
               lane_change=300 <= frame < 330)
    first = maps[0]
    for other in maps[1:]:
      assert len(other.log) == len(first.log), (seed, frame)
      for mine, theirs in zip(other.log, first.log, strict=True):
        assert mine[0] == theirs[0] and same(mine[1], theirs[1]), (seed, frame, mine[0])
        assert mine[2:] == theirs[2:] or same(other.reports, first.reports) and same(other.jumps, first.jumps), (seed, frame)
      assert same(other.snapshot(), first.snapshot()), (seed, frame)
      assert same(state(other), state(first)), (seed, frame)
    stats['calls'] += len(first.log)
    stats['measurements'] += sum(len(entry[1]) for entry in first.log if entry[0] == 'measurements')
    stats['assignments'] += sum(len(entry[1]) for entry in first.log if entry[0] == 'associate')
    for m in maps:
      m.log.clear()
    stats['max_objects'] = max(stats['max_objects'], len(first.objects))
    stats['holding'] += first.holding
  stats['jumps'] = len(first.jumps)
  stats['evicted'] = first.evicted
  stats['lane_checks'] = first.lane_checks
  assert all(m.lane_checks == first.lane_checks for m in maps)
  return stats


TOTALS = {}


@pytest.mark.parametrize('seed', range(SCENARIOS))
def test_rust_and_python_paths_give_the_same_bits(seed):
  TOTALS[seed] = run(seed, [PythonMap(), RustMap()])
  assert TOTALS[seed]['calls'] > 0


def test_scenarios_reach_the_corner_cases():
  if len(TOTALS) < SCENARIOS:
    pytest.skip('needs the scenario runs first')
  total = {k: sum(s[k] for s in TOTALS.values()) for k in next(iter(TOTALS.values()))}
  print('\nparity totals', SCENARIOS, 'scenarios x', FRAMES, 'frames:', total)
  assert max(s['max_objects'] for s in TOTALS.values()) == CcncRadarMap.MAX_OBJECTS
  assert total['evicted'] > 0 and total['jumps'] > 0 and total['holding'] > 0
  assert total['lane_checks'] > 0  # The stop-continuation lane check ran from the Rust loop.


def test_missing_module_runs_the_python_loops(monkeypatch):
  monkeypatch.setattr(ccnc_core_loader, 'load', lambda: None)
  spec = importlib.util.spec_from_file_location('ccnc_radar_map_without_rust', radar_map.__file__)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert module.CcncRadarMap.RUST is None

  class Fallback(Recorder, module.CcncRadarMap):
    pass

  stats = run(1, [Fallback(), RustMap()], frames=200)
  assert stats['measurements'] > 0 and stats['assignments'] > 0


def objects_at(m, positions):
  m.now, m.cos, m.sin = 10., 1., 0.
  for key, (x, y, source) in enumerate(positions, 1):
    m.objects[key] = MapObject(key, source, x, y, 0., 0., 9., 9.95, report=(9.95, x, y), motion_confirmed=key % 2 == 0,
                               drift=.01 * (key % 3))
    m.local[key] = (x, y)


@pytest.mark.parametrize('sign', (1., -1.))
def test_grid_edges_and_equal_scores(sign):
  # Objects on exact 2m cell edges (also -0.0 and negative cells) and twins at one position: equal scores
  # are ordered by measurement index, then object key, as the Python tuples sort.
  positions = [(sign * x, sign * y, source) for x, y, source in
               ((0., 0., 1), (-0., -0., 2), (2., 2., 3), (2., 2., 4), (1.999999, -2., 5), (-2., 1.5, 6), (4., -4., 3),
                (3.9, -4.1, 7), (16., 3., 8), (15.5, 3.2, 9))]
  measurements = [(source, sign * x, sign * y, 0., 0., 0., 1., 0., False, 0, True) for source, x, y in
                  ((3, 2., 2.), (4, 2., 2.), (11, 1., 1.), (12, 1., 1.), (5, 2., -2.), (7, 4., -4.), (8, 15.8, 3.1),
                   (13, -1., .75), (2, 0., 0.))]
  maps = [PythonMap(), RustMap()]
  for m in maps:
    objects_at(m, positions)
  for stamp in (10., 9.99, 9.9):
    results = [m._associate(measurements, stamp) for m in maps]
    assert bits(results[0]) == bits(results[1]) and results[0], stamp
