"""Experimental, display-only CCNC radar map. Live only with the CcncRadarMap option (default off).

Positions/velocities are stored in a local odometry frame (left-positive).
Input timestamps must share the monotonic clock used by ``update``. Radar IDs
are association hints, never physical identities. No predicted point is a new
measurement. Parameters below are trial heuristics, not calibrated sensor noise.
"""
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
import math
from types import SimpleNamespace

from opendbc.car.carlog import carlog


@dataclass
class MapObject:
  key: int
  source: int
  x: float
  y: float
  vx: float
  vy: float
  born: float
  seen: float
  hits: int = 1
  existence: float = .2
  vehicle: float = 0.
  uncertainty: float = .25
  motion_confirmed: bool = False
  stationary: bool = False
  vision: bool = False
  structure: bool = False
  slot: int | None = None
  interior: bool = False
  drift: float = 0.
  explained: float = 0.
  radar_moving: bool = False
  doppler: int = 0  # All unexplained-speed samples, for inspection only.
  doppler_confirmed: bool = False  # Latched: enough of them close together in time.
  doppler_recent: deque = field(default_factory=lambda: deque(maxlen=5))
  admitted: bool = False  # Stationary neighbour accepted while ego is stopped or stopping.
  last_motion: tuple = (0., 0.)  # Forward/lateral ground speed at the last measurement.
  origin: tuple | None = None  # Where it was first measured.
  tracked: int = 0  # Observations the radar reported as a measured (not coasted) track.
  history: deque = field(default_factory=lambda: deque(maxlen=16))


@dataclass
class LaneBoundary:
  points: list
  probability: float
  consistent: int


@dataclass
class Wall:
  members: tuple
  points: list


def interpolate(points, x):
  if len(points) < 2 or not points[0][0] <= x <= points[-1][0]:
    return None
  i = max(1, bisect_left(points, (x, -math.inf)))
  a, b = points[i - 1], points[i]
  if b[0] <= a[0]:
    return None
  return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])


class CcncRadarMap:
  # Structure mapping range. 95% of unproven stationary side displays are within 20m; 20/30/40m gave
  # the same hybrid result on 130 segments, with fewer objects at 20m.
  MIN_X, MAX_X, MAX_Y = -5., 20., 12.
  KEEP_X = 22.  # Hysteresis: existing tracks can cross the admission edge briefly.
  MAX_SIDE_X = 80.
  FAR_Y = 40.  # Far vehicles on a curve are well off the ego axis; lanes decide their slot.
  FAR_X = 200.  # Returns with vehicle evidence, to the CCNC distance field limit; no structure mapping there.
  MAX_OBJECTS, MAX_WALLS = 256, 24
  MATCH_DISTANCE = .5
  MAX_DISPLAY_UNCERTAINTY = 1.
  # Stop memory: an unseen side vehicle from 4m behind (shown at distance 0) to 15m ahead (existing hold distance).
  MEMORY_MIN_X, MEMORY_MAX_X = -4., 15.
  # Reviewed video: admitting any resting return near an assumed lane centre showed a traffic-island
  # post, a bollard, the lead truck's corner and an empty lane, all 1.5-2.1m from ego's axis, while
  # real waiting vehicles were 2.5-3.9m away. The return must be a car body's distance past ego's
  # own lane line (trial values, from those scenes only).
  WAITING_ADMISSION = True
  MEMORY_TRACKED = 40  # A never-moved neighbour is remembered only after 2s as a measured radar track.
  CROSSING_TRAVEL = 1.5  # Sideways travel since first seen, more than a lane change needs without forward progress.
  BODY_CLEARANCE, BODY_FAR_SIDE = .9, 2.6
  PREDICTION_HOLD = .3  # s; same hold as the existing selector's temporal prediction.
  MEMORY_MOVING_HOLD = 5.  # s; trial value for a vehicle still rolling alongside when last seen.
  REPORT_DELAY_BOUND = .25  # s; observed .07..22 on two segments, not a calibrated constant.
  DOPPLER_SPEED, DOPPLER_SAMPLES = .7, 5  # Fixed returns stay within about .4m/s inside 45deg.
  DOPPLER_WINDOW = 1.  # s; the samples must fall within this span (trial value, checked on c1/23).
  # Position-only motion evidence is off: on 130 recorded segments it only admitted fixed returns
  # (map-only reviewed residual 17 with it, 3 without; known vehicle kept either way).
  DISPLACEMENT_EVIDENCE = False
  DENSE_ROW_POINTS, DENSE_ROW_SPAN = 5, 6.  # A row this long and dense is a structure even inside lane lines.
  WALL_MAX_GAP = 3.5
  WALL_CONTINUE = 7.  # A single fresh return may continue an established wall this far ahead (two gaps).  # Posts near 3m apart measure up to ~3.1m; stopped vehicles are farther apart.
  SLOT_FIELDS = (('FF_DISTANCE', 'FF_LATERAL', 'FF_DETECT'),
                 ('LF_DETECT_DISTANCE', 'LF_DETECT_LATERAL', 'LF_DETECT'),
                 ('RF_DETECT_DISTANCE', 'RF_DETECT_LATERAL', 'RF_DETECT'))

  def __init__(self):
    self.objects = {}
    self.walls = []
    self.wall_geometry = []
    self.lanes = {}
    self.geometry = {}
    self.pose = (0., 0., 0.)
    self.cos, self.sin = 1., 0.
    self.local = {}
    self.created = self.evicted = self.range_removed = 0
    self.poses = deque(maxlen=64)
    self.now = None
    self.speed = 0.
    self.radar_stamp = self.model_stamp = None
    self.raw = self.model = None
    self.next_id = 1
    self.selected = (None, None, None)
    self.usable = False
    self.holding = self.stopped = False
    self.a_ego = 0.
    self.evidence = {}
    self.leads = []
    self.blindspot = (False, False)
    self.remembered = [False, False, False]

  def reset(self):
    self.__init__()

  @staticmethod
  def to_world(x, y, pose):
    px, py, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    return px + c * x - s * y, py + s * x + c * y

  def to_ego(self, x, y):
    px, py, _ = self.pose
    c, s = self.cos, self.sin
    return c * (x - px) + s * (y - py), -s * (x - px) + c * (y - py)

  def pose_at(self, stamp):
    if not self.poses or stamp < self.poses[0][0] - 1e-6:
      return None
    previous = self.poses[0]
    for current in self.poses:
      if current[0] >= stamp:
        dt = current[0] - previous[0]
        f = (stamp - previous[0]) / dt if dt > 0 else 0.
        return tuple(a + f * (b - a) for a, b in zip(previous[1], current[1]))
      previous = current
    return self.pose if stamp <= self.now + 1e-6 else None

  def _advance(self, now, speed, yaw):
    dt = 0. if self.now is None else now - self.now
    x, y, angle = self.pose
    distance = .5 * (speed + self.speed) * dt
    middle = angle + .5 * yaw * dt
    self.pose = x + distance * math.cos(middle), y + distance * math.sin(middle), angle + yaw * dt
    self.cos, self.sin = math.cos(self.pose[2]), math.sin(self.pose[2])
    self.now, self.speed = now, speed
    self.poses.append((now, self.pose))
    for obj in self.objects.values():
      if not self.holding:
        obj.admitted = False
      if self.holding and now - obj.seen > .3:
        obj.vx = obj.vy = 0.  # An unseen neighbor slowing with ego is not extrapolated at its last speed.
      obj.x += obj.vx * dt
      obj.y += obj.vy * dt
      # A stationary remembered vehicle can still depart outside radar coverage.
      obj.uncertainty += (.08 if obj.stationary else .35 + .3 * (now - obj.seen)) * dt + .002 * abs(distance)
      obj.vision = False
    self.local = {key: self.to_ego(obj.x, obj.y) for key, obj in self.objects.items()}
    expired = [key for key, xy in self.local.items() if not self._in_range(*xy) and not self._far_vehicle(self.objects[key], *xy)]
    for key in expired:
      del self.objects[key]
      del self.local[key]
    self.range_removed += len(expired)

  def _in_range(self, x, y):
    return self.MIN_X <= x <= self.KEEP_X and abs(y) <= self.MAX_Y

  def _far_vehicle(self, obj, x, y):
    # A far object exists only while measured with vehicle evidence; it is never remembered.
    return (self.KEEP_X < x <= self.FAR_X + 2. and abs(y) <= self.FAR_Y and self.now - obj.seen <= 1.
            and (obj.motion_confirmed or obj.vehicle >= .6))

  def _geometry(self):
    self.geometry = {}
    for i, lane in self.lanes.items():
      points = sorted(self.to_ego(x, y) for x, y in lane.points)
      points = [p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]
      self.geometry[i] = points

  def _update_lanes(self, model, stamp, lane_curves=None):
    pose = self.pose_at(stamp)
    if pose is None:
      return
    px, py = pose[0], pose[1]
    pc, ps = math.cos(pose[2]), math.sin(pose[2])
    # Indices 4/5 are the left/right road edges, as in the existing selector.
    lines = list(zip(model.laneLines, model.laneLineProbs))[:4]
    lines += [None] * (4 - len(lines)) + [(edge, 1.) for edge in list(getattr(model, 'roadEdges', ()))[:2]]
    for i, entry in enumerate(lines):
      if entry is None:
        continue
      line, probability = entry
      if lane_curves is None:
        coords = list(zip(line.x, line.y))
        valid = (len(coords) >= 2 and len(line.x) == len(line.y)
                 and all(math.isfinite(x) and math.isfinite(y) for x, y in coords)
                 and all(b[0] > a[0] for a, b in zip(coords, coords[1:])))
      else:
        # The existing selector validates/caches the same immutable model arrays.
        xs, ys, valid = lane_curves(model, i)
        coords = list(zip(xs.tolist(), ys.tolist())) if valid else []
      if not valid or not math.isfinite(probability):
        continue
      old = self.lanes.get(i)
      if probability < .1:
        if old is not None:
          old.probability *= .9
        continue
      # Keep one bracketing point at each edge for interpolation, not the full 190m curve.
      xs = [p[0] for p in coords]
      first = max(0, bisect_left(xs, self.MIN_X) - 1)
      last = min(len(coords), bisect_left(xs, self.FAR_X) + 1)
      world = [(px + pc * x + ps * y, py + ps * x - pc * y) for x, y in coords[first:last]]  # to_world(x, -y, pose)
      placed = [self.to_ego(x, y) for x, y in world]
      local = sorted(placed)
      previous_geometry = self.geometry.get(i, [])
      errors = [abs(y - previous) for x, y in local if 0. <= x <= 30.
                for previous in (interpolate(previous_geometry, x),) if previous is not None]
      consistent = old.consistent + 1 if old is not None and errors and max(errors) <= .35 else 1
      # Retain actual previously seen geometry behind the current camera domain.
      limit = local[0][0] - .1
      behind = [(point, ego) for point in old.points for ego in (self.to_ego(*point),) if -5. <= ego[0] < limit][-24:] if old else []
      self.lanes[i] = LaneBoundary([point for point, _ in behind] + world[:40], float(probability), min(consistent, 20))
      # Same result as _geometry() for this line, reusing the transforms above.
      points = sorted([ego for _, ego in behind] + placed[:40])
      self.geometry[i] = [p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]

  def _lane(self, x, y, respect_walls=False, moving=False):
    """Lane slot of an ego-frame position, with the existing selector's lane policy:
    inner lines from probability .1, an adjacent lane bounded by its outer line and/or
    the road edge with 1.8m of room, a wider outer margin for moving vehicles."""
    bounds = {i: interpolate(points, x) for i, points in self.geometry.items()}
    # Indices: [outer left, inner left, inner right, outer right, left road edge, right road edge].
    usable = {i for i, lane in self.lanes.items() if lane.probability >= .1 and bounds.get(i) is not None}
    strong = {i for i in usable if self.lanes[i].probability >= .6 and self.lanes[i].consistent >= 3}
    walls = [wy for wy in (interpolate(points, x) for points in self.wall_geometry) if wy is not None] if respect_walls else []
    if {1, 2} <= usable:
      a, b = bounds[1], bounds[2]
      for wy in walls:
        if wy > .2:
          a = min(a, wy)
        elif wy < -.2:
          b = max(b, wy)
      if 1.8 <= a - b <= 5. and b <= y <= a:
        return 0, {1, 2} <= strong and min(a - y, y - b) >= .35, abs(y - .5 * (a + b))
    elif not usable.intersection((1, 2)) and abs(y) <= 1.2:
      return 0, False, abs(y)  # No lane lines at all: straight-ahead corridor, never trusted.
    for slot, inner, line, edge, sign in ((1, 1, 0, 4, 1.), (2, 2, 3, 5, -1.)):
      if inner not in usable:
        continue
      base = bounds[inner]
      offset = sign * (y - base)
      widths = [sign * (bounds[i] - base) for i in (line, edge) if i in usable]
      if offset <= 0. or not widths or min(widths) - .25 <= 1.8:
        continue
      limit = min(([sign * (bounds[line] - base) + (.75 if moving else -.25)] if line in usable else [])
                  + ([sign * (bounds[edge] - base) - .95] if edge in usable else [])
                  + [sign * (wy - base) for wy in walls if sign * wy > .2])
      if offset <= limit:
        width = min(widths)
        return slot, {inner, line} <= strong and min(offset, width - offset) >= .35, abs(offset - .5 * width)
    return None, False, math.inf
  def _measurements(self, live, stamp):
    pose = self.pose_at(stamp)
    if pose is None:
      return []
    measurements = []
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for p in live.points:
      if str(p.radarSource) != 'frontRadar' or not p.measured:
        continue
      x, y, speed = p.dRel, p.yRel, p.vLead
      if not all(math.isfinite(v) for v in (x, y, speed, p.vRel)):
        continue
      if not 0. <= x <= self.FAR_X or abs(y) > (self.MAX_Y if x <= self.MAX_X else self.FAR_Y) or abs(speed) > 100.:
        continue
      wx, wy = self.to_world(x, y, pose)
      # A fixed off-axis return closes at vEgo*cos(azimuth): the radar reports the
      # remainder as forward speed, and its reported distance advances with it.
      distance = math.hypot(x, y)
      cosine = x / distance if distance > 0. else 1.
      explained = min(max(speed, 0.), self.speed * (1. - cosine))
      # Vehicle evidence that does not depend on map positions: the radar's own
      # latched moving class, or a speed the fixed-return geometry cannot explain.
      # Beyond 45deg reported speeds saturate, so no Doppler evidence is taken there.
      report = self.evidence.get(int(p.trackId))
      linked = report is not None and abs(report[1] - x) <= 1.5
      motion = report[0] if linked else getattr(p, 'motionState', None)
      # Track state 2 is a measured track; 0/1/3 are new or coasted ones (liveTracks trackState).
      tracked = (report[2] if linked and len(report) > 2 else getattr(p, 'trackState', 0)) == 2
      doppler = (cosine >= .707 and abs(speed - self.speed * (1. - cosine))
                 > self.DOPPLER_SPEED + self.REPORT_DELAY_BOUND * abs(self.a_ego))
      # Beyond the wall-mapping range only returns with vehicle evidence are kept:
      # they need no structure reasoning, so no distance limit applies to them.
      # A single Doppler sample is not an admission reason: admitting a far fixed
      # return only on its noisy samples would count noise as repeated evidence.
      if x > self.MAX_X and not (motion == 2 or any(
          abs(x - lx) <= max(3., .2 * lx) and abs(y - ly) <= 1.2 for lx, ly in self.leads)):
        continue
      measurements.append((int(p.trackId), wx, wy, speed * c, speed * s, explained, c, s, motion == 2, doppler, tracked))
    return measurements

  def _associate(self, measurements, stamp):
    # Bounded sparse neighborhood search; do not compare every remembered point.
    grid = {}
    age = self.now - stamp
    for key, obj in self.objects.items():
      # Unconfirmed objects are stored without the explained advance; compare in reported coordinates.
      offset = 0. if obj.motion_confirmed else obj.drift
      x, y = obj.x - obj.vx * age + offset * self.cos, obj.y - obj.vy * age + offset * self.sin
      grid.setdefault((math.floor(x / 2.), math.floor(y / 2.)), []).append((key, x, y))
    pairs = []
    for i, (source, x, y, vx, vy, *_) in enumerate(measurements):
      matches = []
      cell_x, cell_y = math.floor(x / 2.), math.floor(y / 2.)
      for cx in range(cell_x - 1, cell_x + 2):
        for cy in range(cell_y - 1, cell_y + 2):
          for key, px, py in grid.get((cx, cy), ()):
            obj = self.objects[key]
            distance = math.hypot(x - px, y - py)
            same = source == obj.source
            # Near-field reflection motion needs room; a reused ID still must pass
            # spatial/velocity gates. Different IDs retain the tight association.
            near = abs(self.local[key][0]) <= 15.
            gate = min(1.2, (.85 if near else .5) + obj.uncertainty * .25) if same else self.MATCH_DISTANCE
            speed_error = math.hypot(vx - obj.vx, vy - obj.vy)
            if distance <= gate and speed_error <= (6. if same else 3.):
              matches.append((distance + .04 * min(speed_error, 3.) - (.1 if same else 0.), i, key))
      matches.sort()
      if (len(matches) > 1 and matches[1][0] - matches[0][0] < .1
          and self.objects[matches[0][2]].source != source):
        continue  # A new ambiguous return cannot inherit another object's history.
      pairs.extend(matches)
    assigned, used = {}, set()
    for _, i, key in sorted(pairs):
      if i not in assigned and key not in used:
        assigned[i] = key
        used.add(key)
    return assigned

  def _observe(self, live, stamp):
    measurements = self._measurements(live, stamp)
    assignments = self._associate(measurements, stamp)
    reserved = set(assignments.values())
    measured_keys = set()
    for i, (source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked) in enumerate(measurements):
      key = assignments.get(i)
      if key is None:
        if len(self.objects) >= self.MAX_OBJECTS:
          # Retain confirmed nearby vehicles before unconfirmed distant returns.
          candidates = [o for o in self.objects.values() if o.key not in measured_keys and o.key not in reserved]
          if not candidates:
            continue
          def priority(o):
            return (abs(self.local[o.key][0]) <= 15., o.structure or o.motion_confirmed or o.vehicle >= .6, o.existence, o.seen)
          victim = min(candidates, key=priority)
          if abs(self.to_ego(x, y)[0]) > 15. and abs(self.local[victim.key][0]) <= 15.:
            continue  # A new distant return cannot evict a remembered near object.
          del self.objects[victim.key]
          del self.local[victim.key]
          self.evicted += 1
        key = self.next_id
        self.next_id += 1
        obj = MapObject(key, source, x, y, vx, vy, stamp, stamp, origin=(x, y))
        self.objects[key] = obj
        self.created += 1
      else:
        obj = self.objects[key]
        if 0. < stamp - obj.seen <= .15 and not obj.motion_confirmed:
          obj.drift += .5 * (obj.explained + explained) * (stamp - obj.seen)
        obj.source, obj.seen = source, stamp
        obj.hits += 1
        obj.existence = min(1., obj.existence + .15)
      obj.explained = explained
      obj.tracked += tracked
      obj.radar_moving = obj.radar_moving or radar_moving
      if doppler:
        # Isolated noisy samples must not add up over a long observation; once the
        # samples are close enough together the vehicle stays one after it stops.
        obj.doppler += 1
        obj.doppler_recent.append(stamp)
        if len(obj.doppler_recent) == self.DOPPLER_SAMPLES and stamp - obj.doppler_recent[0] <= self.DOPPLER_WINDOW:
          obj.doppler_confirmed = True
      if obj.radar_moving or obj.doppler_confirmed:
        obj.motion_confirmed = True
      cx, cy = x - obj.drift * hc, y - obj.drift * hs
      obj.history.append((stamp, cx, cy))
      measured_keys.add(key)
      history = obj.history
      span = stamp - history[0][0]
      if len(history) >= 4 and span >= .15:
        first, last = history[0], history[-1]
        dx, dy = last[1] - first[1], last[2] - first[2]
        net = math.hypot(dx, dy)
        steps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(history, list(history)[1:])]
        path = sum(steps)
        coherent = net > .10 and net >= .8 * path and max(steps, default=0.) <= .7 * path
        # Third evidence, for returns whose reported speed is unusable: compensated
        # travel beyond the 5cm report resolution and beyond what a delayed report
        # can show while ego accelerates or brakes (apparent speed = aEgo * delay).
        if (self.DISPLACEMENT_EVIDENCE and coherent and span >= .4 and net >= .3
            and net / span > .35 + self.REPORT_DELAY_BOUND * abs(self.a_ego)):
          obj.motion_confirmed = True
        spread = max(math.hypot(a[1] - cx, a[2] - cy) for a in history)
        obj.stationary = span >= .3 and spread <= .4 and not coherent
        if obj.stationary:
          obj.vx = obj.vy = 0.
        elif coherent:
          obj.vx, obj.vy = dx / span, dy / span
        else:
          obj.vx, obj.vy = .8 * obj.vx + .2 * (vx - explained * hc), .8 * obj.vy + .2 * (vy - explained * hs)
      if not obj.motion_confirmed:
        x, y = cx, cy  # A confirmed vehicle keeps its reported position.
      latency = self.now - stamp
      mx, my = x + obj.vx * latency, y + obj.vy * latency
      if obj.hits == 1:
        obj.x, obj.y = mx, my
      else:
        obj.x, obj.y = .4 * obj.x + .6 * mx, .4 * obj.y + .6 * my
      obj.uncertainty = max(.15, min(.4, obj.uncertainty * .6))
      ex, ey = self.to_ego(obj.x, obj.y)
      self.local[key] = ex, ey
      obj.slot, obj.interior, center = self._lane(ex, ey, moving=obj.motion_confirmed)
      obj.last_motion = (hc * obj.vx + hs * obj.vy, hc * obj.vy - hs * obj.vx)
      # As in the existing selector: while ego is stopped or stopping, a return that has
      # rested for .5s near the middle of an adjacent lane is a waiting vehicle, even if
      # it never moved in view. Walls and wall-side positions are excluded at selection.
      if (self.WAITING_ADMISSION and self.holding and obj.stationary and obj.slot in (1, 2) and not obj.structure and 0. < ex <= self.MEMORY_MAX_X
          and stamp - obj.born >= .5 and obj.hits >= 7 and self._body_clear(obj.slot, ex, ey) and not self._crossing(obj)):
        obj.admitted = True
      if obj.motion_confirmed:
        obj.vehicle = max(.8, obj.vehicle)
      elif obj.interior:
        # Lane occupancy alone cannot distinguish a stopped vehicle from an
        # obstacle. It protects against wall inference, but stays below admission.
        obj.vehicle = min(.45, obj.vehicle + .12) if obj.vehicle < .6 else obj.vehicle
      elif obj.stationary and self.geometry:
        obj.vehicle = max(0., obj.vehicle - .3)
    return measured_keys

  def _vision(self, model):
    if model is None:
      return
    for lead in model.leadsV3:
      if lead.prob < .8 or not len(lead.x) or not len(lead.y):
        continue
      x, y = lead.x[0] - 1.52, -lead.y[0]
      if not math.isfinite(x) or not math.isfinite(y):
        continue
      matches = [(math.hypot(px - x, py - y), obj) for obj in self.objects.values() if self.now - obj.seen <= .15
                 for px, py in (self.local[obj.key],)
                 if abs(px - x) <= max(3., .2 * x) and abs(py - y) <= 1.2]
      if matches:
        obj = min(matches, key=lambda v: v[0])[1]
        obj.vision, obj.vehicle = True, max(.8, obj.vehicle)

  def _walls(self, measured):
    for obj in self.objects.values():
      obj.structure = False
    # Anchors stay at their measured world positions. A moving member can leave
    # without destroying the stationary remainder. Small reflection drift alone
    # does not instantly revoke an established structure hypothesis.
    retained = []
    for wall in self.walls:
      # The radar's own vehicle evidence, linked to this object by a fresh matching
      # measurement, releases that member only; the rest of the wall stays. Motion
      # inferred from map positions has no such priority and never releases a member.
      support = [(key, anchor) for key, anchor in zip(wall.members, wall.points)
                 if key in self.objects and not self.objects[key].vision and not self._radar_vehicle(self.objects[key])
                 and self._in_range(*self.to_ego(*anchor))
                 and math.hypot(self.objects[key].x - anchor[0], self.objects[key].y - anchor[1]) <= 1.5]
      if len(support) >= 2:
        retained.append(Wall(tuple(k for k, _ in support), [p for _, p in support]))
    self.walls = retained
    # Remembered stationary neighbors can bridge intermittent returns. Require
    # fresh witnesses below so prediction alone can never create a new wall.
    candidates = sorted(((*self.local[obj.key], obj.key) for obj in self.objects.values()
                         if measured and obj.stationary and not obj.motion_confirmed
                         and obj.uncertainty <= .6 and not obj.vision and self.now - obj.born >= .3))
    # A trusted lane interior protects a few resting returns (waiting vehicles) from wall
    # inference, not a long dense row: 356/2 had 8-10 posts 1.5m apart inside lane lines
    # the model drew around them. Vehicles in a queue are farther apart than WALL_MAX_GAP.
    inside = {obj.key for obj in self.objects.values() if obj.interior}
    for sign in (-1, 1):
      points = [p for p in candidates if p[1] * sign > .2]
      used = set()
      for start in range(len(points) - 2):
        if points[start][2] in used:
          continue
        row = [points[start]]
        for point in points[start + 1:]:
          gap = point[0] - row[-1][0]
          if gap > self.WALL_MAX_GAP:
            break
          if gap < .6 or point[2] in used:
            continue
          slope = (point[1] - row[0][1]) / (point[0] - row[0][0])
          if abs(slope) > .4 or any(abs(y - row[0][1] - slope * (x - row[0][0])) > .25 for x, y, _ in row[1:]):
            continue
          row.append(point)
        if (any(p[2] in inside for p in row)
            and (len(row) < self.DENSE_ROW_POINTS or row[-1][0] - row[0][0] < self.DENSE_ROW_SPAN)):
          row = [p for p in row if p[2] not in inside]
        if len(row) < 3 or row[-1][0] - row[0][0] < 4. - 1e-5 or sum(p[2] in measured for p in row) < 2:
          continue
        members = tuple(p[2] for p in row)
        used.update(members)
        anchors = {k: (self.objects[k].x, self.objects[k].y) for k in members}
        owner = next((j for j, w in enumerate(self.walls) if len(set(w.members).intersection(members)) >= 2), None)
        if owner is not None:
          anchors.update(zip(self.walls[owner].members, self.walls[owner].points))
        keys = sorted(anchors, key=lambda k: self.local[k][0])[:16]
        wall = Wall(tuple(keys), [anchors[k] for k in keys])
        if owner is not None:
          self.walls[owner] = wall
        elif len(self.walls) < self.MAX_WALLS:
          self.walls.append(wall)
    # A wall is a line, not only the returns that first formed it. When ego slows the radar
    # reports the next posts one at a time (356/2: gaps of 4-9m between fresh returns), so a
    # fresh resting return on the forward continuation of an established wall joins it.
    # Without this the wall ended as its first members passed behind ego.
    for wall in self.walls:
      line = sorted((*self.to_ego(*anchor), key) for key, anchor in zip(wall.members, wall.points))
      if len(line) < 3:
        continue
      (x0, y0, _), (x1, y1, _) = line[0], line[-1]
      slope = (y1 - y0) / (x1 - x0) if x1 - x0 >= 1. else 0.
      for x, y, key in candidates:
        if (key in measured and key not in wall.members and .6 <= x - x1 <= self.WALL_CONTINUE
            and abs(y - y1 - slope * (x - x1)) <= .25):
          wall.members += (key,)
          wall.points.append((self.objects[key].x, self.objects[key].y))
          x1, y1 = x, y
      if len(wall.members) > 16:
        keep = sorted(range(len(wall.members)), key=lambda i: -self.to_ego(*wall.points[i])[0])[:16]
        wall.members, wall.points = tuple(wall.members[i] for i in keep), [wall.points[i] for i in keep]
    if measured:
      self._bridge_walls([p for p in candidates if p[2] not in inside], measured)
    for wall in self.walls:
      for key in wall.members:
        if key in self.objects:
          self.objects[key].structure = True
    self.wall_geometry = [sorted(self.to_ego(x, y) for x, y in w.points) for w in self.walls]

  def _bridge_walls(self, candidates, measured):
    """Join adjacent established rows through observed, aligned stationary returns.

    No empty-space extrapolation: each intermediate step needs a point. The
    allowable gap follows the two rows' spacing, bounded to 3..6m, rather than
    rejecting every >3m step without considering the established row geometry.
    """
    result = []
    for sign in (-1, 1):
      rows = []
      for wall in self.walls:
        local = sorted(self.to_ego(*p) for p in wall.points)
        if local[0][1] * sign > 0.:
          rows.append((local[0][0], wall, local))
      rows.sort(key=lambda r: r[0])
      for _, wall, local in rows:
        if not result or result[-1][1][0][1] * sign <= 0.:
          result.append((wall, local))
          continue
        prior, left = result[-1]
        a, b = left[-1], local[0]
        gap = b[0] - a[0]
        if gap <= .1:
          result.append((wall, local))
          continue
        left_dx, right_dx = a[0] - left[-2][0], local[1][0] - b[0]
        if min(left_dx, right_dx) <= .1:
          result.append((wall, local))
          continue
        slope = (b[1] - a[1]) / gap
        if (abs(slope) > .4 or abs(slope - (a[1] - left[-2][1]) / left_dx) > .25
            or abs(slope - (local[1][1] - b[1]) / right_dx) > .25):
          result.append((wall, local))
          continue
        middle = [p for p in candidates if a[0] + .1 < p[0] < b[0] - .1
                  and abs(p[1] - a[1] - slope * (p[0] - a[0])) <= .25]
        gap_limit = min(6., max(3., 2. * max(left_dx, right_dx)))
        chain = [a, *middle, b]
        if not any(p[2] in measured for p in middle) or any(q[0] - p[0] > gap_limit for p, q in zip(chain, chain[1:])):
          result.append((wall, local))
          continue
        anchors = {k: (self.objects[k].x, self.objects[k].y) for _, _, k in middle}
        anchors.update(zip(wall.members, wall.points))
        anchors.update(zip(prior.members, prior.points))
        keys = sorted(anchors, key=lambda k: self.cos * anchors[k][0] + self.sin * anchors[k][1])[:16]
        joined = Wall(tuple(keys), [anchors[k] for k in keys])
        result[-1] = joined, sorted(self.to_ego(*p) for p in joined.points)
    self.walls = [w for w, _ in result]

  def reported_local(self, key):
    """Ego position as the radar reports it, for matching raw points to map objects."""
    obj = self.objects[key]
    x, y = self.local[key]
    return (x, y) if obj.motion_confirmed else (x + obj.drift, y)

  def _body_clear(self, slot, x, y):
    """A vehicle's rear centre in the adjacent lane is at least half a car width beyond
    ego's own lane line and not past the far side of that lane. Posts and bollards on the
    line, and the corner of the vehicle ahead, are closer than that."""
    inner = 1 if slot == 1 else 2
    lane = self.lanes.get(inner)
    base = interpolate(self.geometry.get(inner, []), x)
    if lane is None or lane.probability < .1 or base is None:
      return False
    return self.BODY_CLEARANCE <= (y - base if slot == 1 else base - y) <= self.BODY_FAR_SIDE

  def _crossing(self, obj):
    """Reviewed video: pedestrians and cyclists on a crosswalk travel sideways with little
    forward progress. A vehicle in an adjacent lane does not; such an object is neither a
    waiting neighbour nor remembered once it leaves coverage."""
    if obj.origin is None:
      return False
    dx, dy = obj.x - obj.origin[0], obj.y - obj.origin[1]
    lateral = abs(self.cos * dy - self.sin * dx)
    return lateral >= self.CROSSING_TRAVEL and lateral > abs(self.cos * dx + self.sin * dy)

  def _radar_vehicle(self, obj):
    return obj.radar_moving or obj.doppler_confirmed

  def _wall_outside(self, x, y):
    for points in self.wall_geometry:
      wy = interpolate(points, x)
      if wy is not None and wy * y > 0. and abs(y) >= abs(wy) - .25:
        return True
    return False

  def _select(self):
    groups = [[], [], []]
    for obj in self.objects.values():
      x, y = self.local[obj.key]
      # Ordinary candidates are fresh measurements or a short prediction of one. Anything
      # unseen for longer is shown only through _stop_memory and its conditions.
      if self.now - obj.seen > self.PREDICTION_HOLD + 1e-6:
        continue
      if (not 0. <= x <= (self.FAR_X if obj.motion_confirmed or obj.vision else self.MAX_X) or obj.hits < 3 or obj.seen - obj.born < .15
          or obj.existence < .5 or (obj.vehicle < .6 and not obj.admitted)
          or obj.uncertainty > self.MAX_DISPLAY_UNCERTAINTY or obj.structure):
        continue
      slot, _, center_distance = self._lane(x, y, respect_walls=True, moving=obj.motion_confirmed)
      if slot is not None and slot != 0 and x > self.MAX_SIDE_X:
        continue  # Same side-slot distance limit as the existing selector.
      if slot is None and obj.vision and abs(y) <= 2. and not self._wall_outside(x, y):
        slot, center_distance = 0, abs(y)
      if slot is None:
        continue
      # The frozen position of an unseen object says nothing about its direction:
      # use the direction of its last real measurement, never the zeroed velocity.
      if slot != 0 and obj.last_motion[0] < -1. / 3.6:
        continue  # Keep opposing traffic in the map, outside same-direction side slots.
      if obj.structure or (not obj.motion_confirmed and not obj.vision and self._wall_outside(x, y)):
        continue
      # Similar longitudinal distance: prefer an interior representative, without
      # merging the underlying tracks or claiming that they are the same body.
      groups[slot].append((x, center_distance, obj.key))
    selected = []
    for slot, candidates in enumerate(groups):
      if not candidates:
        selected.append(None)
        continue
      nearest = min(c[0] for c in candidates)
      close = [c for c in candidates if c[0] <= nearest + 1.]
      old = next((c for c in close if c[2] == self.selected[slot]), None)
      best = min(close, key=lambda c: (c[1], c[0], c[2]))
      selected.append(old[2] if old and old[1] <= best[1] + .25 else best[2])
    self.remembered = [False, False, False]
    if self.holding:
      for slot in (1, 2):
        if selected[slot] is None and not self.blindspot[slot - 1]:
          selected[slot] = self._stop_memory(slot)
          self.remembered[slot] = selected[slot] is not None
    self.selected = tuple(selected)

  def explain(self, source, x, y, slot):
    """Offline diagnostic only: why a reported radar point is not this slot's display."""
    matches = [o for o in self.objects.values() if o.source == source and self.now - o.seen <= .15
               and math.dist(self.reported_local(o.key), (x, y)) <= 1.5]
    if not matches:
      return 'no map object (beyond range without evidence)' if x > self.MAX_X else 'no map object'
    obj = min(matches, key=lambda o: math.dist(self.reported_local(o.key), (x, y)))
    if obj.key == self.selected[slot]:
      return 'shown'
    lx, ly = self.local[obj.key]
    if obj.structure:
      return 'wall member'
    if obj.vehicle < .6:
      return 'no vehicle evidence'
    if obj.hits < 3 or obj.seen - obj.born < .15 or obj.existence < .5:
      return 'too new'
    if obj.uncertainty > self.MAX_DISPLAY_UNCERTAINTY:
      return 'uncertain position'
    lane, _, _ = self._lane(lx, ly, respect_walls=True, moving=obj.motion_confirmed)
    if lane is not None and lane != 0 and lx > self.MAX_SIDE_X:
      return 'beyond side distance limit'
    if lane is None and obj.vision and abs(ly) <= 2. and not self._wall_outside(lx, ly):
      lane = 0
    if lane is None:
      return 'no lane slot'
    if lane != slot:
      return f"lane slot {('FF', 'LF', 'RF')[lane]} instead"
    if slot != 0 and obj.last_motion[0] < -1. / 3.6:
      return 'opposing direction'
    if not obj.motion_confirmed and not obj.vision and self._wall_outside(lx, ly):
      return 'beyond wall'
    return 'another object preferred'

  def _stop_memory(self, slot):
    """A side vehicle that left radar coverage while ego stops; display memory only."""
    candidates = []
    for obj in self.objects.values():
      x, _ = self.local[obj.key]
      if (obj.slot != slot or obj.structure or (obj.vehicle < .6 and not obj.admitted) or obj.hits < 3 or obj.existence < .5
          or self.now - obj.seen <= self.PREDICTION_HOLD + 1e-6 or not self.MEMORY_MIN_X <= x <= self.MEMORY_MAX_X):
        continue
      # Reviewed scenes: a vehicle that was crossing, turning or oncoming when it left
      # coverage is gone, not waiting beside ego. Remember a vehicle that was at rest
      # (as the existing selector does), or briefly one slowing alongside in its lane.
      forward, lateral = obj.last_motion
      resting = obj.stationary or math.hypot(forward, lateral) <= .6
      alongside = abs(lateral) <= .6 and 0. <= forward <= 3. and self.now - obj.seen <= self.MEMORY_MOVING_HOLD
      if not (resting or alongside) or self._crossing(obj) or (not obj.motion_confirmed and obj.tracked < self.MEMORY_TRACKED):
        continue
      candidates.append((obj.key != self.selected[slot], abs(x), obj.key))
    return min(candidates)[2] if candidates else None

  def update(self, live, model, now, v_ego, *, radar_time=None, model_time=None, yaw_rate=None, reverse=False,
             select=True, lane_curves=None, a_ego=None, blindspot=(False, False), evidence=None):
    """Return FF/LF/RF objects; retained state is never fed back to raw radar.

    ``yaw_rate`` is left-positive in rad/s. If omitted, use the current model's
    opposite-sign orientationRate. Missing/stale pose or stream discontinuities
    invalidate the local frame; ordinary point loss only grows uncertainty.
    """
    if yaw_rate is None:
      rates = getattr(getattr(model, 'orientationRate', None), 'z', ())
      yaw_rate = -float(rates[0]) if len(rates) else math.nan
    if (not all(math.isfinite(v) for v in (now, v_ego, yaw_rate)) or reverse
        or v_ego < -.5 or abs(yaw_rate) > 1. or (self.now is not None and not 0. <= now - self.now <= .25)):
      self.reset()
      return (None, None, None)
    v_ego = max(0., v_ego)  # The filtered speed reads slightly negative at standstill; that is not reversing.
    if a_ego is None:
      a_ego = (v_ego - self.speed) / (now - self.now) if self.now is not None and now > self.now else 0.
    # Same condition as the existing stop/approach display memory: stopped, or <=10km/h while slowing.
    stopped = v_ego <= .3 / 3.6
    self.a_ego = a_ego if math.isfinite(a_ego) else 0.
    self.holding = stopped or (not self.stopped and v_ego <= 10. / 3.6 and math.isfinite(a_ego) and a_ego < -.1 / 3.6)
    self.stopped = stopped
    self.blindspot = tuple(blindspot)
    # Offline: {radar track id: (radar motion class, reported distance)} read from the raw
    # track message. A live build would carry the class on each RadarPoint as motionState.
    self.evidence = evidence or {}
    self._advance(now, v_ego, yaw_rate)
    self._geometry()
    radar_time = now if radar_time is None else radar_time
    model_time = now if model_time is None else model_time
    if (model is not None and 0. <= now - model_time <= .25
        and ((model_time != self.model_stamp) if self.model_stamp is not None else True)
        and model is not self.model):
      self._update_lanes(model, model_time, lane_curves)
      self.model, self.model_stamp = model, model_time
    self.usable = live is not None and 0. <= now - radar_time <= .15
    self.leads = [(lead.x[0] - 1.52, -lead.y[0]) for lead in getattr(model, 'leadsV3', ())
                  if lead.prob >= .8 and len(lead.x) and len(lead.y)] if model is not None and 0. <= now - model_time <= .25 else []
    measured = set()
    if self.usable and live is not self.raw and (self.radar_stamp is None or radar_time > self.radar_stamp):
      measured = self._observe(live, radar_time)
      self.raw, self.radar_stamp = live, radar_time
    if model is not None and 0. <= now - model_time <= .25:
      self._vision(model)
    self._walls(measured)
    if select:
      self._select()
    else:
      self.selected = (None, None, None)
    if not self.usable:
      self.selected = (None, None, None)
    return tuple(self.objects.get(k) for k in self.selected)

  def display_values(self, values):
    """Write only the nine vehicle fields in a caller-owned CCNC_0x162 copy."""
    for slot, (distance, lateral, detect) in enumerate(self.SLOT_FIELDS):
      obj = self.objects.get(self.selected[slot]) if self.usable else None
      if obj is None:
        values[detect] = 0
        continue
      x, y = self.to_ego(obj.x, obj.y)
      values[distance] = min(204.7, max(0., x * .8))
      values[lateral] = min(12.7, max(0., y if slot == 1 else -y)) if slot else min(12.7, max(-12.8, -y))
      values[detect] = 2 if obj.stationary else 1
    return values

  def refine_existing(self, values, points):
    """Offline hybrid: preserve existing candidates unless matched to a known wall.

    ``points`` must contain the original selected FF/LF/RF radar observations,
    not coordinates reconstructed from filtered CAN fields. Unknown identity,
    stop-memory-only output and targets beyond 30m keep the existing decision.
    """
    if not self.usable:
      return values
    selected = []
    for point, (distance, _, detect) in zip(points, self.SLOT_FIELDS):
      selected.append(None)
      if point is None or not values.get(detect) or not 0. <= point.dRel <= self.MAX_X:
        continue
      if abs(values[distance] / .8 - point.dRel) > 2.:
        continue
      matches = [o for o in self.objects.values() if o.source == point.trackId and self.now - o.seen <= .15
                 and math.dist(self.reported_local(o.key), (point.dRel, point.yRel)) <= 1.]
      if len(matches) == 1 and matches[0].structure and not matches[0].vision:
        values[detect] = 0
      elif len(matches) == 1:
        selected[-1] = matches[0].key
    self.selected = tuple(selected)
    return values

  def extend_observed_walls(self, eligible):
    """Attach a fresh stationary return across one missing row position.

    Existing wall geometry may overrule a later lane expansion. Eligibility is
    supplied by the existing vehicle admission/vision/stop-memory protections.
    No predicted observation extends a wall, and anchors never chase a point.
    """
    for wall in self.walls:
      geometry = sorted((self.to_ego(*anchor), key, anchor) for key, anchor in zip(wall.members, wall.points))
      if len(geometry) < 3:
        continue
      gaps = sorted(b[0][0] - a[0][0] for a, b in zip(geometry, geometry[1:]) if .6 <= b[0][0] - a[0][0] <= 3.)
      if not gaps:
        continue
      (x0, y0), (x1, y1) = geometry[0][0], geometry[-1][0]
      if x1 - x0 < 1.:
        continue
      slope = (y1 - y0) / (x1 - x0)
      for key in eligible:
        obj = self.objects[key]
        if key in wall.members or not obj.stationary or self.now - obj.born < .3 or self.now - obj.seen > .15:
          continue
        x, y = self.local[key]
        gap = x0 - x if x < x0 else x - x1 if x > x1 else 0.
        if (gap < .6 or gap > 2. * gaps[-1] + .35
            or not any(min(abs(gap - spacing), abs(gap - 2. * spacing)) <= .35 for spacing in gaps)
            or abs(y - y0 - slope * (x - x0)) > .25 or len(wall.members) >= 16):
          continue
        wall.members += (key,)
        wall.points.append((obj.x, obj.y))
        obj.structure = True
        # Only the original endpoints establish this frame's extension bound.
    self.wall_geometry = [sorted(self.to_ego(x, y) for x, y in wall.points) for wall in self.walls]

  def snapshot(self):
    """Small, JSON-ready map for offline replay inspection; no vehicle transport."""
    return dict(pose=self.pose, selected=self.selected,
                counters=dict(created=self.created, evicted=self.evicted, range_removed=self.range_removed),
                objects=[dict(key=o.key, source=o.source, xy=self.to_ego(o.x, o.y), velocity=(o.vx, o.vy),
                              existence=o.existence, vehicle=o.vehicle, uncertainty=o.uncertainty,
                              moving=o.motion_confirmed, radar_moving=o.radar_moving, doppler=o.doppler, doppler_confirmed=o.doppler_confirmed, admitted=o.admitted, stationary=o.stationary, structure=o.structure,
                              slot=o.slot, interior=o.interior, vision=o.vision,
                              born=o.born, seen=o.seen, hits=o.hits) for o in self.objects.values()],
                lanes=self.geometry, lane_probability={i: (v.probability, v.consistent) for i, v in self.lanes.items()},
                walls=[[self.to_ego(x, y) for x, y in w.points] for w in self.walls])


_live = SimpleNamespace(tracker=None, raw=None, radar_time=0., model=None, model_time=0., failed=False, filters={})


def reset_live():
  _live.tracker, _live.raw, _live.model, _live.filters = CcncRadarMap(), None, None, {}


def _road_alignment(tracker, slot, x, y):
  """Road-relative lateral position, its reference, and the lane bounds of one displayed map
  object, in the form the existing position correction takes (left-positive metres)."""
  lines = {}
  for i in range(4):
    lane, points = tracker.lanes.get(i), tracker.geometry.get(i)
    if lane is None or lane.probability < .1 or not points:
      continue
    here, near = interpolate(points, x), interpolate(points, min(max(0., points[0][0]), points[-1][0]))
    if here is not None and near is not None:
      lines[i] = here, near
  inner = [lines[i] for i in (1, 2) if i in lines]
  if not inner or (len(inner) == 2 and not all(2.3 <= left - right <= 4.8 for left, right in zip(*inner))):
    return y, ('map', None), None
  shift = sum(here - near for here, near in inner) / len(inner)
  left, right = ((1, 2), (0, 1), (2, 3))[slot]
  bounds = (lines[left][0] - shift, lines[right][0] - shift) if left in lines and right in lines else None
  return y - shift, ('lane', 'center' if len(inner) == 2 else 1 in lines), bounds


def _correct_positions(tracker, correction, values, frame, md, ext):
  """CcncVehiclePositionCorrection on the map's selection: the existing centre lock, lateral
  following and slot debounce, keyed by map object instead of radar ID. The map already
  confirms and predicts objects itself, so the existing temporal track checks are not used."""
  centers = [ext._ccnc_side_lane_center(md, side, None, correction.MIN_LANE_PROBABILITY) for side in (0, 1)]
  observations = {o.key: (o.born, round(o.seen * 100.)) for o in tracker.objects.values()}
  correction.update_centers(centers[0], centers[1], frame, observations)
  filters = {}
  for slot, (distance, lateral, detect) in enumerate(tracker.SLOT_FIELDS):
    obj = tracker.objects.get(tracker.selected[slot])
    if obj is None or not values[detect]:
      continue
    x, y = tracker.local[obj.key]
    saved = correction.tracks.get(obj.key)
    if tracker.remembered[slot]:
      # A remembered vehicle is not measured: keep where it was last drawn, learn nothing.
      output = saved['output'] if saved is not None and saved['birth'] == obj.born else y
    else:
      aligned, reference, bounds = _road_alignment(tracker, slot, x, y)
      previous = _live.filters.get(obj.key)
      if previous is None or previous[0] != obj.born or not 0 <= frame - previous[1] <= 15:
        previous = (obj.born, frame, ext._CcncRadarPositionFilter(x, True), ext._CcncRadarPositionFilter(aligned, False))
      _, last, far, side = previous
      side.dt = (frame - last) * .01
      filters[obj.key] = (obj.born, frame, far, side)
      values[distance] = min(204.7, max(0., far.apply(x) * .8))
      point = SimpleNamespace(trackId=obj.key, dRel=x, yRel=y, ccnc_fresh=obj.seen == tracker.radar_stamp)
      reliable = bounds is not None and (slot == 0 or correction.center_valid[slot - 1])
      output = correction.apply(point, aligned, side.apply(aligned), slot, reference, bounds, observations[obj.key], frame, reliable)
    if slot == 0:
      values[lateral] = min(12.7, max(-12.8, -output))
    else:
      output = min(max(output, -5.4), 5.4)
      values[lateral] = min(max(float(output if slot == 1 else -output), 0.), 12.7)
  _live.filters = filters
  correction.stabilize_slots(values, tracker.selected, observations, frame, bypass=tracker.holding)


def update_vehicles(values, CS, md, frame, v_ego_kph, a_ego_kph, model_lanes, ext):
  """Live FF/LF/RF display from the map alone (CcncRadarMap option); ``ext`` is the existing
  extension module, used for its display conventions and the rear blind-spot icons. Any
  failure returns to the existing selector for that send."""
  try:
    if _live.tracker is None:
      reset_live()
    tracker, state, out = _live.tracker, ext.state, CS.out
    now = frame * .01
    # The controller receives each message only while fresh, without its timestamp: a new
    # reader is stamped on arrival (replay: display differences of a few sends per segment).
    live = CS.live_tracks
    if live is not None and live is not _live.raw:
      _live.raw, _live.radar_time = live, now
    if md is not None and md is not _live.model:
      _live.model, _live.model_time = md, now
    tracker.update(live, md, now, out.vEgo, radar_time=_live.radar_time, model_time=_live.model_time,
                   reverse=str(out.gearShifter) == 'reverse', a_ego=out.aEgo,
                   blindspot=(out.leftBlindspot, out.rightBlindspot))
    for side, center in enumerate((state.lf_center, state.rf_center)):
      lane_center = ext._ccnc_side_lane_center(md, side)
      if lane_center is not None:
        center.apply(lane_center)
    tracker.display_values(values)
    for slot, (_, lateral, detect) in enumerate(tracker.SLOT_FIELDS):
      obj = tracker.objects.get(tracker.selected[slot])
      if obj is None or not values[detect]:
        continue
      y = tracker.local[obj.key][1]
      forward = tracker.cos * obj.vx + tracker.sin * obj.vy
      closing = forward - tracker.speed
      if slot == 0:
        values[lateral] = ext.apply_curved_deadband(-y, 0, .7, 1)
        values[detect] = 2 if forward < 3. else state.ff_detect.apply(closing)
      else:
        center = (state.lf_center if slot == 1 else state.rf_center).value
        values[lateral] = min(max(float(ext.apply_curved_deadband(y if slot == 1 else -y, center, .9, 2)), 0.), 12.7)
        values[detect] = (state.lf_detect if slot == 1 else state.rf_detect).apply(closing)
    correction = state.radar_display_tracker.position_correction
    if correction is not None:
      _correct_positions(tracker, correction, values, frame, md, ext)
    # Rear blind-spot icons, as in the existing display.
    offset = (state.r_lane_f.value - state.l_lane_f.value) / 2 if model_lanes else 0.
    for prefix, active, distance, lateral in (('LR', out.leftBlindspot, state.lr_distance, 2.8 - offset),
                                              ('RR', out.rightBlindspot, state.rr_distance, 2.8 + offset)):
      if active or distance.value < 15:
        values[prefix + '_DETECT_DISTANCE'] = distance.apply(8 if active else 16)
        values[prefix + '_DETECT_LATERAL'] = lateral
        values[prefix + '_DETECT'] = 2 if active else 1
    return values
  except Exception:
    if not _live.failed:
      _live.failed = True
      carlog.exception('CCNC radar map display failed; using the existing selector')
    _live.tracker = None
    return ext.update_vehicles(values, CS, md, frame, v_ego_kph, a_ego_kph, model_lanes)

def integrated_tracker(previous, radar_map):
  """Offline integration with an isolated legacy selector; no live-module patch.

  Keep the caller's observation/position/stop policies. Near wall decisions enter
  the existing rejected-ID path before all candidate and retention selections.
  Retain simultaneous-row evidence; the old stationary-point memory keeps a
  15m overlap for the 30..50m region instead of duplicating the nearest points.
  """
  class MapDisplayTracker(type(previous)):
    def __init__(self):
      self.__dict__.update(previous.__dict__)
      self.map_blocked = {}
      self.far_raw_source = self.far_raw = None
      self.structure_frame = None
      self.map_reverse = False
      self.map_vision = []

    def map_protected(self, key, entry, vision):
      birth, _, _, point, _ = entry
      admission = self.boundary_admission.get(key)
      if admission is not None and admission['status'] == 'allowed' and admission['moving'] is not None:
        return True
      if any(saved[0] == key and saved[-1] == birth
             for holds in (self.stop_holds, self.approach_holds) for saved in holds.values()):
        return True
      return any(abs(point.dRel - x) <= max(3., point.dRel * .2) and abs(point.yRel - y) <= 1.2 for x, y in vision)

    def update_boundary_admission(self, md):
      super().update_boundary_admission(md)
      self.update_structure_regions(md, self.last_frame, self.map_reverse)
      blocked = {}
      self.map_vision = []
      if radar_map.usable:
        vision = [(lead.x[0] - 1.52, -lead.y[0]) for lead in getattr(md, 'leadsV3', ())
                  if lead.prob >= .8 and len(lead.x) and len(lead.y)]
        self.map_vision = vision
        by_source = {}
        for obj in radar_map.objects.values():
          if radar_map.now - obj.seen <= .15:
            by_source.setdefault(obj.source, []).append(obj)
        candidates = {}
        for key, entry in self.tracks.items():
          birth, stamp, _, point, _ = entry
          if (not 0. <= point.dRel <= radar_map.MAX_X or key in self.boundary_rejected
              or self.map_protected(key, entry, vision)):
            continue
          fresh = (self.temporal is None or point.ccnc_fresh) and getattr(point, 'measured', True)
          if fresh:
            source = point.ccnc_source_id if self.temporal is not None else point.trackId
            matches = [obj for obj in by_source.get(source, ())
                       if math.dist(radar_map.reported_local(obj.key), (point.dRel, point.yRel)) <= 1.]
            obj = matches[0] if len(matches) == 1 else None
          else:
            saved = self.map_blocked.get(key)
            hold = self.temporal.HOLD if self.temporal is not None else 15
            retained = saved and saved[0] == birth and 0 <= self.last_frame - saved[2] <= hold
            if retained and saved[1] is None:
              blocked[key] = saved  # A measured row classification may cover the existing coast, without renewal.
              continue
            obj = radar_map.objects.get(saved[1]) if retained else None
            if retained:
              stamp = saved[2]  # Even an accepted but unmeasured radar packet cannot renew this deadline.
          candidates[key] = birth, stamp, point, obj, fresh
        radar_map.extend_observed_walls({obj.key for _, _, _, obj, fresh in candidates.values() if obj is not None and fresh})
        bounds = {}
        if candidates and max(self.lane_probabilities(md)) >= .1 and self.live is not None:
          raw = self.temporal.raw if self.temporal is not None else self.live
          _, _, projected = self.lane_projection(raw)
          bounds = {p[0].trackId: (-left, -right) for p, (left, right) in zip(self._points, projected)}
        for key, (birth, stamp, point, obj, _) in candidates.items():
          if obj is not None and obj.structure and not obj.vision:
            blocked[key] = birth, obj.key, stamp
            continue
          old_row = False
          left, right = bounds.get(key, (math.inf, -math.inf))
          if point.yRel > left or point.yRel < right:
            side = 0 if point.yRel > left else 1
            selected = self.selected
            try:
              slots = list(selected)
              slots[side + 1] = key
              self.selected = tuple(slots)
              old_row = super().structure_suppresses(side, md, self.last_frame)
            finally:
              self.selected = selected
          if old_row:
            blocked[key] = birth, obj.key if obj is not None else None, stamp
      self.map_blocked = blocked

    def front_admitted(self, point, md):
      return point.trackId not in self.map_blocked and super().front_admitted(point, md)

    def side_representative(self, candidates):
      # Keep the original eligible list until here. Rejecting before admission
      # would let a non-candidate wall influence another vehicle's selection.
      original, lateral = super().side_representative(candidates)
      self.boundary_rejected.update(self.map_blocked)
      if original is None or original.trackId not in self.map_blocked:
        return original, lateral
      # Removing one bollard must not promote the next unknown static return and
      # seed a long stop memory. Prefer a replacement with existing vehicle proof;
      # an ordinary stationary candidate without a rejected predecessor is unchanged.
      eligible = [item for item in candidates if item[0].trackId not in self.map_blocked
                  and self.map_protected(item[0].trackId, self.tracks[item[0].trackId], self.map_vision)]
      return super().side_representative(eligible)

    def path_lead(self, *args, **kwargs):
      self.boundary_rejected.update(self.map_blocked)
      return super().path_lead(*args, **kwargs)

    def bridge_crossing(self, *args, **kwargs):
      self.boundary_rejected.update(self.map_blocked)
      return super().bridge_crossing(*args, **kwargs)

    def update_structure_regions(self, md, frame, reverse=False):
      if self.structure_frame != frame:
        super().update_structure_regions(md, frame, reverse)
        self.structure_frame = frame

    def update_stationary_map(self, md, frame, reverse=False):
      live, temporal = self.live, self.temporal
      raw = temporal.raw if temporal is not None else live
      if raw is not self.far_raw_source:
        self.far_raw_source = raw
        self.far_raw = SimpleNamespace(points=[p for p in raw.points if p.dRel >= 15.]) if raw is not None else None
      try:
        if temporal is not None:
          self.temporal = SimpleNamespace(raw=self.far_raw, HOLD=temporal.HOLD)
        else:
          self.live = self.far_raw
        super().update_stationary_map(md, frame, reverse)
      finally:
        self.live, self.temporal = live, temporal

    def structure_suppresses(self, side, md, frame):
      key = self.selected[side + 1]
      entry = self.tracks.get(key)
      if radar_map.usable and entry is not None and 0. <= entry[3].dRel <= radar_map.MAX_X:
        return key in self.map_blocked
      return super().structure_suppresses(side, md, frame)

  return MapDisplayTracker()
