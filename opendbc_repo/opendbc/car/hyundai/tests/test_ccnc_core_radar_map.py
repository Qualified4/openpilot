"""The Rust radar map loops (_measurements, _associate, _observe, _advance's objects, _walls, _bands) give the same bits as the
Python reference, on the real map class. The Rust path keeps its objects in the Rust MapObject class, the Python
path in the dataclass; both are compared as their fields (dataclass vars(), Rust __getstate__())."""
from collections import deque
import functools
import importlib.util
import math
import operator
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
  if type(value).__name__ == 'MapObject':  # The Rust class: its fields in the dataclass's order and form.
    return 'o', bits(value.__getstate__())
  raise TypeError(type(value))


def same(a, b):
  """bits(a) == bits(b); equal pickles (bit-exact floats, types, order) decide quickly, bits() decides the rest."""
  try:
    if pickle.dumps(a) == pickle.dumps(b):
      return True
  except (pickle.PicklingError, TypeError):  # Classes of a module loaded from its file only.
    pass
  return bits(a) == bits(b)


def fields(obj):
  """A map object's fields: dataclass vars() or the Rust class's __getstate__(), the same dict either way."""
  return vars(obj) if hasattr(obj, '__dict__') else obj.__getstate__()


def state(m):
  # The recorder's counts of Python calls, which the Rust path partly makes in Rust.
  out = {k: v for k, v in vars(m).items() if k not in ('generation', 'raw', 'model', 'log', 'lane_checks', 'reconnects',
                                                       'meta_checks', 'meta_changing', 'bridged_walls')}
  out['objects'] = {k: fields(o) for k, o in m.objects.items()}
  return out


class Recorder:
  """Records each _measurements/_associate result and the id state they leave; the computation is the real class's."""

  def __init__(self):
    super().__init__()
    self.log = []
    self.associating = False
    self.lane_checks = self.reconnects = self.meta_checks = self.meta_changing = self.bridged_walls = 0

  def reset(self):
    counts = self.log, self.lane_checks, self.reconnects, self.meta_checks, self.meta_changing, self.bridged_walls
    super().reset()
    self.log, self.lane_checks, self.reconnects, self.meta_checks, self.meta_changing, self.bridged_walls = counts

  def _bridge_walls(self, candidates, measured):  # Python path only.
    before = len(self.walls)
    super()._bridge_walls(candidates, measured)
    self.bridged_walls += before - len(self.walls)

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

  def _reconnect(self, measurements, assignments):  # Python path only.
    before = len(assignments)
    super()._reconnect(measurements, assignments)
    self.reconnects += len(assignments) > before

  def _model_changing(self):
    changing = super()._model_changing()
    self.meta_checks += 1
    self.meta_changing += changing
    return changing

  def _lane(self, *args, **kwargs):
    self.lane_checks += self.associating  # Continuation candidates (Python path only: Rust checks the lane itself).
    return super()._lane(*args, **kwargs)


class RustMap(Recorder, CcncRadarMap):
  pass


class PythonMap(Recorder, CcncRadarMap):
  RUST = None


@pytest.fixture(scope='module', autouse=True)
def rust_core():
  core = ccnc_core_loader.load()
  assert core is not None, 'the x86_64 test build must be in prebuilt/linux-x86_64'
  assert CcncRadarMap.RUST is core and core.abi_version() == 4
  return core


def model(rng, frame, *, desire='none', state='off', lead=None, bend=0., shift=0., weak=False):
  xs = [-8., 0., 10., 20., 30., 50., 80., 120., 210.]
  wobble = .03 * math.sin(frame * .07)
  lines = [N(x=xs, y=[-(y - shift + bend * x * x + wobble) for x in xs]) for y in LANES]
  probs = [.9, .95, .95, .9]
  if frame % 37 == 0:
    probs[0] = .05
  if weak:  # Ego's own lines unusable: _lane's corridor fallback, which reads the model's lane change state.
    probs[1] = probs[2] = .05
  leads = []
  if lead is not None:
    leads.append(N(x=[lead[0] + 1.52], y=[-lead[1]], v=[lead[2]], prob=.95))
  if frame % 29 == 0:
    leads.append(N(x=[rng.uniform(25., 120.)], y=[rng.uniform(-6., 6.)], v=[], prob=.85))
  return N(laneLines=lines, laneLineProbs=probs, roadEdges=[N(x=xs, y=[-8.75] * len(xs)), N(x=xs, y=[8.75] * len(xs))],
           leadsV3=leads, meta=N(desire=desire, laneChangeState=state), orientationRate=N(z=[0.]))


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
    # Cases added for the Rust _observe; their own generator keeps the cases above as they were.
    self.extra = random.Random(seed + 5000)
    # A lead in ego's lane, lost (frames 102-108), back under a new id .9m off the bridged position (reconnect),
    # then reported 2.2m sideways (a jump, 250-254), lost and back under a third id (257-). A loss shows .3s
    # later; a frame whose radar is too old (every 7th) clears the selection and ends a bridge, so each loss
    # and return fall between two of them.
    self.lead = self.kind in ('cruise', 'slow') and seed % 8 in (2, 3)
    if self.lead:
      for t in self.targets:  # No nearer vehicle in ego's lane takes FF from it.
        if abs(t['y']) < 1.:
          t['y'] += 3.5
      self.targets.append(dict(id=950 + seed, x=9., y=.1, v=0., moving=True, lead=True))
    self.vision = None
    self.lane_shift = seed % 3 == 0  # Ego crosses its left line during the lane change (lane_offset).
    self.weak_lines = seed % 3 == 1  # Frames 420-499: lines 1 and 2 below .1, with a model lane change state.

  def shift(self, frame):
    """Lateral move of the model's lines: ego crossing to the left lane, then the model renumbering them."""
    if not self.lane_shift or not 300 <= frame < 370:
      return 0.
    return 2.5 * min(1., (frame - 300) / 20) if frame < 340 else -1. * (370 - frame) / 30

  def meta(self, frame):
    if 300 <= frame < 330:
      return 'laneChangeLeft', 'off'
    if self.weak_lines and 460 <= frame < 480:
      return 'laneChangeRight', 'off'
    if self.weak_lines and 485 <= frame < 490:
      return 'none', 'laneChangeStarting'
    return 'none', 'off'

  def frame(self, frame, v_ego):
    rng, points = self.rng, []
    for t in self.targets:
      if t.get('lead'):
        t['v'] = v_ego + .3 * math.sin(frame * .05)
        t['x'] += (t['v'] - v_ego) * DT
        self.vision = (t['x'], t['y'], t['v'])
        if 102 <= frame < 109 or 255 <= frame < 257:
          continue
        track, x, y = t['id'], t['x'], t['y']
        if frame >= 109:
          track, x = track + 10, x + .9
        if 250 <= frame < 255:
          y += 2.2
        if frame >= 257:
          track, x = track + 10, x + .9
        points.append(dict(trackId=track, dRel=x, yRel=y + rng.gauss(0., .03), vRel=t['v'] - v_ego, vLead=t['v'],
                           measured=True, radarSource='frontRadar', motionState=2, trackState=2))
        continue
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
    if self.extra.random() < .05:  # Exactly -0.0 ahead, or beside on ego's axis.
      points.append(dict(trackId=5100, dRel=-0., yRel=self.extra.choice((-0., 3.5, -2.)), vRel=-v_ego, vLead=0.,
                         measured=True, radarSource='frontRadar', motionState=1, trackState=2))
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


class Structures:
  """Fixed posts for _walls/_bands, from their own generator (the scenario's draws stay as they were): rows 1-3m apart
  beside the road, gaps a later post fills (_bridge_walls), a row bending outwards (_bend), twins at one position
  (equal sort keys), posts reported first far ahead (band_early), dense rows inside a lane, and on some seeds enough
  short rows for MAX_WALLS. Positions are fixed in the map's frame, reported from the map's pose of the last frame."""

  def __init__(self, seed):
    self.rng = random.Random(seed + 7000)
    spacing = (2., 2.5, 3., 1.)[(seed // 2) % 4]
    self.posts = []  # (track id, world x, world y, reported only nearer than this)
    for k in range(int(420. / spacing)):
      x = -10. + k * spacing
      # Right row: five posts, a gap of four spacings, the next five posts .6m further out, and so on. A post in the
      # middle of each gap, on the line between the two rows (.3m off both), joins them only through _bridge_walls.
      j, i = divmod(k, 8)
      if i < 5:
        self.posts.append((20000 + k, x, -6.5 - .6 * (j % 2), 20.))
      elif i == 6:
        self.posts.append((20000 + k, x, -6.8, 20.))
      u = x % 80.  # Left row, bending 20 degrees outwards after 40m of each 80m.
      y = 6. + (math.tan(math.radians(20.)) * (u - 40.) if u > 40. else 0.)
      if y <= 11.:
        self.posts.append((30000 + k, x, y, 20.))
      if spacing == 1. and x % 60. < 15.:  # Dense row inside the left lane.
        self.posts.append((40000 + k, x, 3.1, 20.))
    if seed >= 16:  # Short rows (three posts, then a gap): more walls and bands than MAX_WALLS allows.
      for r, y in enumerate((-7.6, -8.7, -9.8, -10.9, 7.4, 8.5, 9.6, 10.7)):
        for k in range(int(420. / 2.)):
          if k % 5 < 3:
            self.posts.append((50000 + 1000 * r + k, -10. + 2. * k + .3 * r, y, 20.))
    self.twins = {track for track, *_ in self.posts if track % 7 == 3}

  def points(self, pose):
    px, py, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for track, wx, wy, near in self.posts:
      x, y = c * (wx - px) + s * (wy - py), -s * (wx - px) + c * (wy - py)
      if not -3. <= x <= near or self.rng.random() < .08:
        continue
      x, y = x + self.rng.gauss(0., .02), y + self.rng.gauss(0., .02)
      for twin in ((track, track ^ 1 << 17) if track in self.twins else (track,)):
        out.append(dict(trackId=twin, dRel=x, yRel=y, vRel=0., vLead=0., measured=True, radarSource='frontRadar',
                        motionState=1, trackState=2))
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


def wall_keys(m):
  return [frozenset(w.members) for w in m.walls]


def run(seed, maps, frames=FRAMES):
  scenario = Scenario(seed)
  structures = Structures(seed)
  rng = random.Random(seed + 1000)
  stats = dict(calls=0, measurements=0, assignments=0, jumps=0, max_objects=0, evicted=0, holding=0, bridge=0, stable_y=0,
               bridge_stable_y=0, ahead_of=0, departed=0, settled=0, lane_offset=0, negative_zero=0,
               walls=0, wall_created=0, wall_kept=0, wall_cancelled=0, wall_16=0, wall_cap=0, structure_cap=0, bends=0,
               band_created=0, band_extended=0, band_vanished=0, band_early=0, early_twins=0)
  walls, bands, early, sizes = [], [], set(), {}
  for frame in range(frames):
    now = 1000. + frame * DT
    v_ego = speed_profile(scenario.kind, frame)
    points = scenario.frame(frame, v_ego)
    posts = structures.points(maps[0].pose)
    live = capnp_live(points + posts) if frame % 2 == 0 else namespace_live(points + posts)
    lead = scenario.vision or next(((p['dRel'], p['yRel'], p['vLead']) for p in points if abs(p['yRel']) < 1.5 and 0. < p['dRel'] < 80.
                                    and all(math.isfinite(p[k]) for k in ('dRel', 'yRel', 'vLead'))), None)
    desire, change_state = scenario.meta(frame)
    md = model(rng, frame, desire=desire, state=change_state, lead=lead, bend=.001 if seed % 2 else 0., shift=scenario.shift(frame),
               weak=scenario.weak_lines and 420 <= frame < 500)
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
    stats['bridge'] += first.bridge is not None
    stats['bridge_stable_y'] += first.bridge is not None and first.objects[first.bridge['key']].stable_y is not None
    stats['lane_offset'] += first.lane_offset != 0
    stats['negative_zero'] += sum(math.copysign(1., p[k]) < 0. and p[k] == 0. for p in points for k in ('dRel', 'yRel'))
    for name, test in (('stable_y', lambda o: o.stable_y is not None), ('ahead_of', lambda o: o.ahead_of),
                       ('departed', lambda o: o.departed), ('settled', lambda o: o.settled)):
      stats[name] += any(test(o) for o in first.objects.values())
    # Walls made, kept as they were, or cancelled (no successor shares two members); bands by identity.
    now_walls = wall_keys(first)
    stats['walls'] += bool(now_walls)
    stats['wall_created'] += sum(w not in walls for w in now_walls)
    stats['wall_kept'] += sum(w in walls for w in now_walls)
    stats['wall_cancelled'] += sum(all(len(w & v) < 2 for v in now_walls) for w in walls)
    stats['wall_16'] += any(len(w.members) == 16 for w in first.walls)
    stats['wall_cap'] += len(first.walls) == first.MAX_WALLS
    stats['structure_cap'] += len(first.walls) + len(first.bands) >= first.MAX_WALLS
    stats['bends'] += len(first.bends)
    stats['band_created'] += sum(all(b is not old for old in bands) for b in first.bands)
    stats['band_extended'] += sum(len(b.members) > sizes[id(b)] for b in first.bands if id(b) in sizes)
    stats['band_vanished'] += sum(all(b is not new for new in first.bands) for b in bands)
    added = first.band_early - early
    stats['band_early'] += len(added)
    # A twin joining by prediction while its twin, at the same x, did not (one joins, then the other is no longer
    # a spacing ahead): the stable sort of measured.difference(...) kept the set's order between the two.
    by_source = {o.source: o.key for o in first.objects.values()}
    stats['early_twins'] += sum(first.local[k] == first.local.get(by_source.get(first.objects[k].source ^ 1 << 17), ())
                                for k in added if k in first.objects)
    walls, bands, early = now_walls, list(first.bands), set(first.band_early)
    sizes = {id(b): len(b.members) for b in bands}
  stats['jumps'] = len(first.jumps)
  stats['evicted'] = first.evicted
  for name in ('lane_checks', 'reconnects', 'meta_checks', 'meta_changing', 'bridged_walls'):
    stats[name] = sum(getattr(m, name) for m in maps if m.RUST is None)
  assert all(m.lane_checks == 0 for m in maps if m.RUST is not None)
  return stats


TOTALS = {}


@pytest.mark.parametrize('seed', range(SCENARIOS))
def test_rust_and_python_paths_give_the_same_bits(seed):
  TOTALS[seed] = run(seed, [PythonMap(), RustMap()])
  assert TOTALS[seed]['calls'] > 0


def test_scenarios_reach_the_corner_cases():
  # The counts come from the Python path. Seeds whose parity run happened elsewhere (another pytest-xdist
  # worker) are run here again on that path alone.
  for seed in range(SCENARIOS):
    if seed not in TOTALS:
      TOTALS[seed] = run(seed, [PythonMap()])
  total = {k: sum(s[k] for s in TOTALS.values()) for k in next(iter(TOTALS.values()))}
  print('\nparity totals', SCENARIOS, 'scenarios x', FRAMES, 'frames:', total)
  assert max(s['max_objects'] for s in TOTALS.values()) == CcncRadarMap.MAX_OBJECTS
  assert total['evicted'] > 0 and total['jumps'] > 0 and total['holding'] > 0
  assert total['lane_checks'] > 0  # The stop-continuation lane check ran (counted on the Python path).
  # Bridged FF objects, drawn at their stable report y, continued by a report under a new id.
  assert total['bridge'] > 0 and total['bridge_stable_y'] > 0
  # Crowded scenes keep 32 of their points, so there the lead may be missing on the frame it returns.
  assert all(TOTALS[seed]['reconnects'] > 0 for seed in range(SCENARIOS) if Scenario(seed).lead and not Scenario(seed).crowded)
  assert total['ahead_of'] > 0 and total['departed'] > 0  # Remembered vehicles that drove off.
  assert total['settled'] > 0 and total['lane_offset'] > 0 and total['negative_zero'] > 0
  assert total['meta_checks'] > 0 and 0 < total['meta_changing'] < total['meta_checks']  # _lane's model fallback.
  for name in ('walls', 'wall_created', 'wall_kept', 'wall_cancelled', 'wall_16', 'wall_cap', 'structure_cap', 'bends',
               'bridged_walls', 'band_created', 'band_extended', 'band_vanished', 'band_early', 'early_twins'):
    assert total[name] > 0, name


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
  cls = MapObject if m.RUST is None else m.RUST.MapObject
  for key, (x, y, source) in enumerate(positions, 1):
    m.objects[key] = cls(key, source, x, y, 0., 0., 9., 9.95, report=(9.95, x, y), motion_confirmed=key % 2 == 0,
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


def float_lists(rng, count):
  """Float lists that stress sum(): cancellation, mixed magnitudes, -0.0, infinities and NaN."""
  special = (0., -0., 1e308, -1e308, 1e-308, 5e-324, math.inf, -math.inf, math.nan, .1, .7, 1e16, -1e16, 1.)
  for _ in range(count):
    values = []
    for _ in range(rng.randint(0, 17)):
      kind = rng.random()
      if kind < .05:
        values.append(rng.choice(special))
      elif kind < .25 and values:
        values.append(-rng.choice(values) * rng.choice((1., 1. + 2 ** -52, 1. - 2 ** -53)))
      elif kind < .6:
        values.append(math.hypot(rng.gauss(0., .3), rng.gauss(0., .3)))  # Step lengths, as in _observe.
      else:
        values.append(rng.uniform(-1., 1.) * 10. ** rng.randint(-20, 20))
    yield values


def test_sum_is_python_3_12_sum():
  core = ccnc_core_loader.load()
  lists = list(float_lists(random.Random(3), 20000))
  for values, got in zip(lists, core._sum_for_test(lists), strict=True):
    want = sum(values)
    if got is not None and math.isnan(got) and math.isnan(want):
      continue  # A NaN's sign differs by platform (aarch64 CPython gave -nan where Rust gave nan); NaN is NaN to the map.
    assert (None if values == [] else bits(want)) == (got if got is None else bits(got)), values
  # A plain left-to-right sum differs on these lists, so the comparison above can tell the two apart.
  assert sum(bits(sum(v)) != bits(functools.reduce(operator.add, v, 0)) for v in lists if v) > 1000


def same_floats(got, want):
  """Bit-equal, except that any two NaNs count as equal (their sign differs by platform, as for sum())."""
  got, want = np.array(got, dtype=np.float64), np.array(want, dtype=np.float64)
  return (got.view(np.uint64) == want.view(np.uint64)) | (np.isnan(got) & np.isnan(want))


def test_math_is_cpython_math():
  # A million values each: the ranges the map uses, wide ranges, and special values.
  core = ccnc_core_loader.load()
  rng = np.random.default_rng(4)
  special = [0., -0., 1., -1., 180., -180., 30., math.pi, 1e-300, 5e-324, 1e308, math.inf, -math.inf, math.nan]
  wide = np.concatenate([rng.uniform(-400., 400., 500000), rng.standard_normal(486000) * 10. ** rng.integers(-30, 30, 486000),
                         special * 1000])
  unit = np.concatenate([rng.uniform(-1., 1., 600000), 1. - rng.uniform(0., 1e-6, 200000), np.cos(rng.uniform(0., .6, 199986)),
                         [0., -0., 1., -1., math.nan] * 2, [1. - 2 ** -53, -1. + 2 ** -53, 2 ** -1074, -2 ** -1074]])
  assert len(wide) == len(unit) == 1000000
  for name, values in (('degrees', wide), ('radians', wide), ('acos', unit),
                       ('cos', np.radians(rng.uniform(-360., 360., 1000000)))):
    values = values.tolist()
    want = [getattr(math, name)(v) for v in values]
    assert same_floats(core._math_for_test(name, values), want).all(), name
  for name, value in (('acos', 1. + 2 ** -52), ('acos', -1.5), ('cos', math.inf)):  # "math domain error", as in Python
    with pytest.raises(ValueError):
      getattr(math, name)(value)
    with pytest.raises(ValueError):
      core._math_for_test(name, [value])


class LaneMap(PythonMap):
  """Compares the Rust _lane with the Python one on the lanes of every fifth frame."""

  def __init__(self):
    super().__init__()
    self.checked = self.corridor = 0

  def _walls(self, measured):
    super()._walls(measured)
    core = ccnc_core_loader.load()
    if int(self.now * 20 + .5) % 5:
      return
    for x in (-6., -0., 0., 3., 10., 25.5, 49.9, 80., 80.5, 150., 205.):
      bounds = [radar_map.interpolate(self.geometry.get(i, []), x) for i in range(6)]
      ys = [b + d for b in bounds if b is not None for d in (-.35, -1e-9, 0., 1e-9, .35, 1.8, -1.8)]
      for y in ys + [-0., 0., 1.2, -1.2, 4.2, -4.2, 9.]:
        for moving in (False, True):
          for settled in (False, True):
            want = self._lane(x, y, moving=moving, settled=settled)
            got = core._lane_for_test(self, x, y, moving, settled)
            assert bits(got) == bits(want), (self.now, x, y, moving, settled, got, want)
            self.checked += 1
            self.corridor += want[0] == 0 and not ({1, 2} & self._lane_view(x)[1])


@pytest.mark.parametrize('seed', (0, 1, 2))  # Ego's lane change, ego's lines below .1 with a lane change state, a lead.
def test_rust_lane_is_python_lane(seed):
  m = LaneMap()
  run(seed, [m])
  assert m.checked > 50000 and (m.corridor > 0 or seed != 1)
