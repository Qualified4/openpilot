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
  forward_samples: int = 0  # Those above the fixed-return band; the rest were speed towards ego.
  vision_seen: bool = False  # Latched vision match.
  doppler_recent: deque = field(default_factory=lambda: deque(maxlen=5))
  admitted: bool = False  # Stationary neighbour accepted while ego is stopped or stopping.
  last_motion: tuple = (0., 0.)  # Forward/lateral ground speed at the last measurement.
  origin: tuple | None = None  # Where it was first measured.
  departed: bool = False  # A remembered vehicle that was seen driving off.
  superseded: bool = False  # Its radar id was reported elsewhere in the next cycle: the same track moved on.
  ahead_of: tuple = ()  # (key, gap) of remembered vehicles this object first appeared just ahead of.
  tracked: int = 0  # Observations the radar reported as a measured (not coasted) track.
  beyond: float | None = None  # Since when it has been measured outside its side's outer lane line.
  settled: bool = False  # Outside that line for a while and no longer moving sideways: the lane beyond.
  history: deque = field(default_factory=lambda: deque(maxlen=16))
  trail: deque = field(default_factory=lambda: deque(maxlen=20))  # Where a moving vehicle has driven, one point per TRAIL_STEP.
  report: tuple | None = None  # (stamp, x, y): its last radar report, ego frame, before smoothing.
  lateral_reports: deque = field(default_factory=lambda: deque(maxlen=16))  # (stamp, report y from ego's lane centre).
  calm_since: float | None = None  # Since when the reports have not moved sideways.
  stable_y: float | None = None  # Report y from ego's lane centre, averaged over its last calm stretch: drawn while bridged.


@dataclass
class LaneBoundary:
  points: list
  probability: float
  consistent: int


@dataclass
class Wall:
  members: tuple
  points: list


@dataclass
class Band:
  sign: int  # 1 left, -1 right.
  reference: int  # Lane index the offset is measured from.
  offset: float
  members: dict  # Object key -> distance along the road of each supporting return.
  supported: float  # Odometer when a member was last measured.
  spacing: float = 0.


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
  WAITING_TRACKED = 5  # A return the radar only ever coasted (recorded: 13 reports, none measured) is not a waiting vehicle.
  MEMORY_TRACKED = 40  # A never-moved neighbour is remembered only after 2s as a measured radar track.
  CROSSING_TRAVEL = 1.5  # Sideways travel since first seen, more than a lane change needs without forward progress.
  BODY_CLEARANCE, BODY_FAR_SIDE = .9, 2.6
  PREDICTION_HOLD = .3  # s; same hold as the existing selector's temporal prediction.
  MEMORY_MOVING_HOLD = 5.  # s; trial value for a vehicle still rolling alongside when last seen.
  REPORT_DELAY_BOUND = .25  # s; observed .07..22 on two segments, not a calibrated constant.
  DOPPLER_SPEED, DOPPLER_SAMPLES = .7, 5  # Fixed returns stay within about .4m/s inside 45deg.
  # Returns nearer than this in ego's own corridor reported 1-2m/s that no object had (recorded while
  # turning, |y| up to 1.2m). Vehicles alongside at that range (recorded 2.0-2.6m to the side) still count.
  DOPPLER_MIN_X, DOPPLER_NEAR_Y = 3., 1.5
  # Recorded 2026-10-12: a remembered waiting car drove off and stayed displayed for 30s. It reappeared
  # 3.7m ahead of its remembered position, .5m to the side, at 1.4m/s, then pulled away. Cars passing
  # another remembered car in the next lane first appeared 1.5-1.9m to the side (trial values).
  RELEASE_GAP, RELEASE_LATERAL, RELEASE_SPEED, RELEASE_ADVANCE = (2., 6.), 1., 3., 1.
  ID_CYCLE = .15  # s; the radar reports every 50ms, the same freshness bound as elsewhere.
  DOPPLER_WINDOW = 1.  # s; the samples must fall within this span (trial value, checked on c1/23).
  # Position-only motion evidence is off: on 130 recorded segments it only admitted fixed returns
  # (map-only reviewed residual 17 with it, 3 without; known vehicle kept either way).
  DISPLACEMENT_EVIDENCE = False
  RESTING_SPEED = 1.  # m/s; a vehicle reporting more is not at rest.
  # A moving vehicle is followed .75m past the outer line while it changes lane, or where the model draws
  # that line too close. Once it has settled beyond a normally placed line it is in the lane beyond:
  # the radar reports the near corner of its body, .3-.8m past the line (recorded 2026-10-13).
  OUTER_NARROW = 2.3  # m from the inner line; an outer line nearer than this is one drawn too close.
  OUTER_LATERAL, OUTER_TIME = .3, 1.  # m/s, s
  # Lines drawn .3-.5m too close at a normal width also occur (recorded: a car in the last lane, 3.3m from
  # the inner line): only a return a full lane width from the inner line is taken as the lane beyond.
  OUTER_FAR = 3.5
  DOPPLER_TRACKED = 3  # Measured reports before unexplained speed confirms a vehicle; coasted tracks carry none.
  # The reported speed is compensated for ego's forward speed only. While ego turns, the radar ahead of the
  # rear axle also moves sideways, and fixed returns to the side report 1-3m/s (recorded at .5-.75rad/s).
  DOPPLER_YAW = .3  # rad/s; no unexplained-speed samples above it.
  RESTING_TRAVEL, RESTING_NET = .15, .5  # Net movement of a resting return: this share of ego's travel plus this many metres.
  DENSE_ROW_POINTS, DENSE_ROW_SPAN = 5, 6.  # A row this long and dense is a structure even inside lane lines.
  WALL_MAX_GAP = 3.5
  # A wall follows the road. Besides straight rows of returns seen together, resting returns are
  # compared by their offset from ego's lane line of that side, collected over the last BAND_MEMORY
  # metres driven, so that posts reported one at a time or on a curve form one band. Offsets are
  # counted in BAND_BIN cells; a band is two adjacent cells (+-.25m, as the straight-row test).
  # Recorded on 130 segments: members of established rows keep their offset from ego's lane line
  # within .5m in 95% of sends above 60km/h (road edge 90%), but only in 63% at rest (road edge
  # 36%), where the model's lines move; hence the rows are kept and the lane line is the reference.
  BAND_MEMORY, BAND_BIN, BAND_FOLD = 40., .25, 10.
  BAND_PREDICTION = .5  # A new return this near the band's next expected post joins before it has a history.
  BAND_GAP = 3.5  # As WALL_MAX_GAP.
  # The model draws curves beyond 30m too straight, so the second vehicle ahead in ego's own lane lands just
  # past ego's lane line and was shown beside ego (recorded 2026-10-13, 45-75m). With a lead displayed, a
  # vehicle more than LEAD_BEYOND past it gets a side slot on a curve (lane centre over LEAD_CURVE off ego's
  # axis there) only LEAD_MARGIN beyond the line; on a straight road a real neighbour's near corner is
  # reported .9-1.3m beyond it and nothing changes. Nor does it get one if it drove through ego's lane:
  # its recorded positions TRAIL_NEAR-TRAIL_FAR ahead of ego all lie within TRAIL_ROOM of ego's lane lines.
  LEAD_BEYOND, LEAD_MARGIN, LEAD_CURVE = 5., 1.2, 2.
  TRAIL_STEP, TRAIL_NEAR, TRAIL_FAR, TRAIL_ROOM = 3., 5., 25., .3
  # Beyond LEAD_SIDE_X a side slot behind a displayed lead is not given at all, unless the vehicle is within
  # LEAD_SIDE_PAST of that lead: the far lane geometry is off by up to a lane width there and video cannot
  # tell the lanes apart (recorded 2026-10-13: second leads at 58-79m shown RF/LF on a straight road). Raised from
  # 50 (2026-10-14): the side vehicles the existing selector showed at 50-80m were real cars on video. While
  # holding it stays at LEAD_SIDE_HOLD_X: a 50-60m candidate there displaced the stop memory of a car right
  # beside ego (15 windows, 169 sends: 356/21, 3c1/21, 3c1/23, 3dd/28-29), when the display is watched most.
  LEAD_SIDE_X, LEAD_SIDE_HOLD_X, LEAD_SIDE_PAST = 60., 50., 30.
  # The radar can merge the lead's track with a large vehicle beside it, or lose it, while vision keeps seeing
  # the lead in ego's lane (recorded 2026-10-14: the same id 1.2-1.7m sideways across the line for 1.5s next to
  # a trailer, FF jumping to a car 42m ahead). The displayed FF object is then kept for up to BRIDGE_HOLD while
  # a vision lead (probability BRIDGE_PROB) is at its distance and within BRIDGE_LANE of the lane centre.
  # Vision only continues an object that was FF: it creates no object, fills no side slot and gives no evidence.
  BRIDGE_HOLD = 2.  # s; trial value
  BRIDGE_PROB, BRIDGE_LANE = .5, 1.2
  # A reported lateral step this large within this time is not a vehicle moving (5m/s sideways).
  BRIDGE_JUMP, BRIDGE_JUMP_TIME = .8, .15
  BRIDGE_RECONNECT = 1.5  # m; a new report this near the bridged position, in ego's lane, continues the object.
  # The merge can also push the lead's track across the line over several cycles, without a jump (recorded
  # 2026-10-14: 3e1/19 16.9s, the car 8m ahead pushed 2m right by an overtaking SUV, FF jumping to 21m).
  # Vision then still sees the lead in the lane, while a lead really leaving the lane leaves it in vision
  # first (recorded: a median 1s before the map, 62 of 64 departures).
  # Not beyond BRIDGE_DRIFT_X: there the vision gate spans 20m and the lane lines are off by up to a lane width.
  BRIDGE_DRIFT_X = 50.
  # A merge has already pushed the last in-lane report towards the line (recorded: 3e1/19 drawn 1.6m right of a car
  # vision saw centred). A bridged object is drawn at its report y from before the push: the mean over STABLE_TIME
  # of reports moving sideways slower than STABLE_SPEED (over STABLE_WINDOW), kept while they move faster.
  # Reports come in .05m steps: one step over the window is resolution, not movement (.33m/s over .15s).
  STABLE_SPEED, STABLE_WINDOW, STABLE_TIME, REPORT_STEP = .3, .15, .5, .05
  # Beyond FF_FAR_X the lane lines are off by up to a lane width and the lead's slot flickers from ego's lane to
  # none or a side lane for a cycle or two (recorded 2026-10-14: 339 of 356 FF flickers beyond 50m, 243 of them
  # .1s, the object measured throughout). The displayed FF object, still measured and otherwise eligible, stays FF
  # for up to FF_FAR_HOLD after its slot leaves ego's lane; a nearer FF candidate takes FF at once.
  FF_FAR_X, FF_FAR_HOLD = 50., .3
  # A row bends where a turn pocket opens (recorded 2026-10-14: 3e3/15, posts beside ego at 1.6m, then 20 degrees
  # outwards; the bent part became a wall only once three of its posts were seen, after one was admitted as a
  # waiting vehicle). A fresh resting return within WALL_MAX_GAP of a wall end, at most WALL_BEND off the
  # direction of the end segment, continues the wall; the next one is compared with the new segment.
  WALL_BEND = 30.  # degrees
  WALL_CONTINUE = 7.  # A single fresh return may continue an established wall this far ahead (two gaps).  # Posts near 3m apart measure up to ~3.1m; stopped vehicles are farther apart.
  SLOT_FIELDS = (('FF_DISTANCE', 'FF_LATERAL', 'FF_DETECT'),
                 ('LF_DETECT_DISTANCE', 'LF_DETECT_LATERAL', 'LF_DETECT'),
                 ('RF_DETECT_DISTANCE', 'RF_DETECT_LATERAL', 'RF_DETECT'))

  def __init__(self):
    self.objects = {}
    self.walls = []
    self.wall_geometry = []
    self.bends = []
    self.bands = []
    self.band_points = {}  # Object key -> (sign, reference, road distance, offset, odometer when measured, inside a lane).
    self.band_early = set()  # Keys taken by prediction, for inspection.
    self.lanes = {}
    self.geometry = {}
    self.pose = (0., 0., 0.)
    self.cos, self.sin = 1., 0.
    self.local = {}
    self.created = self.evicted = self.range_removed = 0
    self.poses = deque(maxlen=64)
    self.now = None
    self.speed = 0.
    self.odometer = 0.  # Distance ego has travelled, for conditions that follow the road instead of the clock.
    self.radar_stamp = self.model_stamp = None
    self.raw = self.model = None
    self.next_id = 1
    self.selected = (None, None, None)
    self.usable = False
    self.holding = self.stopped = False
    self.a_ego = 0.
    self.yaw_rate = 0.
    self.evidence = {}
    self.leads = []
    self.blindspot = (False, False)
    self.remembered = [False, False, False]
    self.vision_leads = []  # Vision leads at BRIDGE_PROB or more: (x, y, speed or None), map frame.
    self.lane_change = False  # Ego changing lanes (model desire or a turn signal).
    self.reports = {}  # Radar id -> recent (stamp, x, y) reports, ego frame.
    self.jumps = {}  # Radar id -> stamp of its last lateral jump.
    self.hold = None  # (key, lateral, centred, speed): the FF object at its last report inside ego's lane.
    self.bridge = None  # The FF object kept on vision: dict(key, start, x, y, dx, speed, cause, kind), x/y as drawn.
    self.far_since = None  # (key, time): the displayed far FF object left ego's lane slot then.

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
    self.odometer += abs(distance)
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
    bridged = self.bridge['key'] if self.bridge else None
    expired = [key for key, xy in self.local.items()
               if key != bridged and not self._in_range(*xy) and not self._far_vehicle(self.objects[key], *xy)]
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

  def _lane(self, x, y, respect_walls=False, moving=False, settled=False):
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
      # A confirmed moving vehicle is not denied its lane by a model outer line drawn too close
      # (recorded: 1.99m from the inner line) while the road edge leaves the room.
      if offset <= 0. or not widths or (max(widths) if moving and edge in usable else min(widths)) - .25 <= 1.8:
        continue
      margin = -.25 if not moving else .75
      if settled and line in usable and sign * (bounds[line] - base) >= self.OUTER_NARROW and offset >= self.OUTER_FAR:
        margin = 0.
      limit = min(([sign * (bounds[line] - base) + margin] if line in usable else [])
                  + ([sign * (bounds[edge] - base) - .95] if edge in usable else [])
                  + [sign * (wy - base) for wy in walls if sign * wy > .2])
      if offset <= limit:
        width = min(widths)
        return slot, {inner, line} <= strong and min(offset, width - offset) >= .35, abs(offset - .5 * width)
    return None, False, math.inf

  def _settle(self, obj, x, y, stamp):
    """Whether a moving vehicle has come to stay outside the outer lane line of its side. It stays so until
    it is measured inside that line again: its sideways speed is too noisy to decide this report by report."""
    line = 0 if y > 0. else 3
    if line not in self.geometry or self.lanes[line].probability < .1 or (outer := interpolate(self.geometry[line], x)) is None:
      return
    if not obj.motion_confirmed or (y - outer if y > 0. else outer - y) <= 0.:
      obj.beyond, obj.settled = None, False
      return
    if obj.beyond is None:
      obj.beyond = stamp
    obj.settled = obj.settled or (stamp - obj.beyond > self.OUTER_TIME and abs(obj.last_motion[1]) < self.OUTER_LATERAL)

  def _band_line(self, x, y):
    """Whether an ego-frame position lies on the line of a band, within BAND_MEMORY metres of its members
    along the road. A row of posts has gaps the band does not span (recorded 2026-10-13: 11.5m to the next
    post), and the lone post beyond one rests where a waiting neighbour would."""
    for band in self.bands:
      lane = self.lanes.get(band.reference)
      if y * band.sign <= .2 or not band.members or lane is None or lane.probability < .1:
        continue
      base = interpolate(self.geometry.get(band.reference, []), x)
      if (base is not None and abs(y - base - band.offset) <= self.BAND_BIN
          and min(band.members.values()) - self.BAND_MEMORY <= self.odometer + x <= max(band.members.values()) + self.BAND_MEMORY):
        return True
    return False

  def _trail(self, obj):
    """Where the vehicle drove through the stretch TRAIL_NEAR-TRAIL_FAR ahead of ego: for each recorded point, how
    far it lies outside ego's own lane lines (negative inside). The lines are trusted at that range; the
    model draws curves beyond 30m too straight (recorded 2026-10-13)."""
    lines = [self.geometry.get(i) for i in (1, 2) if i in self.lanes and self.lanes[i].probability >= .1]
    if len(lines) < 2:
      return []
    offsets = []
    for point in obj.trail:
      x, y = self.to_ego(*point)
      if self.TRAIL_NEAR <= x <= self.TRAIL_FAR:
        left, right = interpolate(lines[0], x), interpolate(lines[1], x)
        if left is not None and right is not None:
          offsets.append(max(y - left, right - y))
    return offsets

  def _behind_lead(self, key, slot, lead_x):
    """A vehicle beyond the displayed lead whose side slot rests on far lane geometry only, or is too far to tell."""
    obj = self.objects[key]
    x, y = self.local[key]
    if x <= lead_x + self.LEAD_BEYOND:
      return False
    if x > (self.LEAD_SIDE_HOLD_X if self.holding else self.LEAD_SIDE_X) and x > lead_x + self.LEAD_SIDE_PAST:
      return True
    left, right = interpolate(self.geometry.get(1, []), x), interpolate(self.geometry.get(2, []), x)
    if left is not None and right is not None and abs(.5 * (left + right)) > self.LEAD_CURVE:
      if (y - left if slot == 1 else right - y) < self.LEAD_MARGIN:
        return True
    offsets = self._trail(obj)
    return len(offsets) >= 2 and max(offsets) <= self.TRAIL_ROOM

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
      recent = self.reports.setdefault(int(p.trackId), deque(maxlen=4))
      if any(0. < stamp - t <= self.BRIDGE_JUMP_TIME + 1e-6 and abs(x - rx) <= self.BRIDGE_RECONNECT and abs(y - ry) >= self.BRIDGE_JUMP
             for t, rx, ry in recent):
        self.jumps[int(p.trackId)] = stamp
      recent.append((stamp, x, y))
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
      # A fixed return reports anything from zero to the geometric value (recorded: .1-.5m/s
      # where 2-5m/s was expected at 30-45deg). Only speed outside that band is unexplained:
      # above it, or towards ego.
      residual = speed - explained
      doppler = (cosine >= .707 and (x >= self.DOPPLER_MIN_X or abs(y) >= self.DOPPLER_NEAR_Y) and abs(residual)
                 > self.DOPPLER_SPEED + self.REPORT_DELAY_BOUND * abs(self.a_ego)
                 and abs(self.yaw_rate) <= self.DOPPLER_YAW)
      doppler = (1 if residual > 0. else -1) if doppler else 0  # Above the band, or towards ego.
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
    # A selected car's reports can move along its body while ego waits. Keep a
    # recent same-id interior track; stale reflections must not steal the report.
    recent = {}
    for key, obj in self.objects.items():
      if 0. < stamp - obj.seen <= self.ID_CYCLE and obj.report is not None and abs(self.local[key][0]) <= 15.:
        if obj.source not in recent or obj.seen > self.objects[recent[obj.source]].seen:
          recent[obj.source] = key
    continuations = {}
    for i, (source, x, y, vx, vy, *_) in enumerate(measurements):
      key = recent.get(source)
      if key is None:
        continue
      obj = self.objects[key]
      age = self.now - stamp
      dx, dy = x - (obj.x - obj.vx * age), y - (obj.y - obj.vy * age)
      forward, lateral = dx * self.cos + dy * self.sin, dy * self.cos - dx * self.sin
      if (self.holding and key in self.selected[1:] and self.cos * vx + self.sin * vy > self.RESTING_SPEED
          and obj.motion_confirmed and obj.interior and obj.slot in (1, 2) and not self.lane_change
          and key not in continuations.values()
          and abs(forward) <= 2. and abs(lateral) <= .35 and math.hypot(vx - obj.vx, vy - obj.vy) <= 6.
          and self._lane(*self.to_ego(x, y), moving=True)[1]):
        continuations[i] = key
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
    assigned, used = dict(continuations), set(continuations.values())
    for _, i, key in sorted(pairs):
      if i not in assigned and key not in used:
        assigned[i] = key
        used.add(key)
    return assigned

  def _reconnect(self, measurements, assignments):
    """A bridged FF object continues with a new report near its bridged position in ego's lane, whatever its
    radar id (recorded: the lead returned under a new id after its old one moved onto the trailer)."""
    if self.bridge is None or self.bridge['key'] in assignments.values():
      return
    key = self.bridge['key']
    bx, by = self.bridge['x'], self.bridge['y']
    best = None
    for i, (_, x, y, *_) in enumerate(measurements):
      if i in assignments:
        continue
      ex, ey = self.to_ego(x, y)
      distance = math.hypot(ex - bx, ey - by)
      if distance <= self.BRIDGE_RECONNECT and self._lane(ex, ey, moving=True)[0] == 0 and (best is None or distance < best[0]):
        best = distance, i
    if best is not None:
      obj = self.objects[key]
      obj.x, obj.y = self.to_world(bx, by, self.pose)  # Continue from where it was drawn, not from its prediction.
      self._forget_lateral(obj)
      obj.vx, obj.vy = measurements[best[1]][3:5]  # Its own reported speed, not the merged track's.
      assignments[best[1]] = key

  def _stable(self, obj):
    """Update the object's stable report y (from ego's lane centre) with its latest report."""
    stamp, x, y = obj.report
    centre = self._centre(x)
    reports = obj.lateral_reports
    # While bridged its reports are the merged track's: they would make the merged position look stable.
    if centre is None or (self.bridge is not None and self.bridge['key'] == obj.key):
      reports.clear()
      obj.calm_since = None
      return
    reports.append((stamp, y - centre))
    earlier = [r for r in reports if .1 - 1e-6 <= stamp - r[0] <= self.STABLE_WINDOW + 1e-6]
    if not earlier:
      obj.calm_since = None
      return
    t0, y0 = earlier[0]
    if max(0., abs(reports[-1][1] - y0) - self.REPORT_STEP) / (stamp - t0) >= self.STABLE_SPEED:
      obj.calm_since = None
      return
    obj.calm_since = t0 if obj.calm_since is None else obj.calm_since
    if stamp - obj.calm_since >= self.STABLE_TIME - 1e-6:
      recent = [r[1] for r in reports if stamp - r[0] <= self.STABLE_TIME + 1e-6]
      obj.stable_y = sum(recent) / len(recent)

  def _forget_lateral(self, obj):
    """Drop the reports and sideways speed a bridged object took from the vehicle its track merged with."""
    obj.history.clear()
    forward = self.cos * obj.vx + self.sin * obj.vy
    obj.vx, obj.vy = forward * self.cos, forward * self.sin

  def _observe(self, live, stamp):
    measurements = self._measurements(live, stamp)
    assignments = self._associate(measurements, stamp)
    self._reconnect(measurements, assignments)
    reserved = set(assignments.values())
    measured_keys = set()
    for i, (source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked) in enumerate(measurements):
      key = assignments.get(i)
      if key is None:
        if len(self.objects) >= self.MAX_OBJECTS:
          # Retain confirmed nearby vehicles before unconfirmed distant returns.
          candidates = [o for o in self.objects.values() if o.key not in measured_keys and o.key not in reserved
                        and not (self.bridge and o.key == self.bridge['key'])]
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
        # One radar id is in one place: an object measured with this id until the previous cycle is the
        # same track, which jumped beyond the association gate. If the radar classes it as moving, the
        # vehicle has left the old position, which is not remembered. A resting track that jumps has
        # only changed its reflection point on the same body, and the old position stays valid.
        if radar_moving:
          for other in self.objects.values():
            if other.source == source and 0. < stamp - other.seen <= self.ID_CYCLE:
              other.superseded = True
        obj = MapObject(key, source, x, y, vx, vy, stamp, stamp, origin=(x, y))
        self.objects[key] = obj
        self.created += 1
      else:
        obj = self.objects[key]
        if 0. < stamp - obj.seen <= .15 and not obj.motion_confirmed:
          obj.drift += .5 * (obj.explained + explained) * (stamp - obj.seen)
        obj.source, obj.seen = source, stamp
        obj.departed = obj.superseded = False
        obj.hits += 1
        obj.existence = min(1., obj.existence + .15)
      obj.explained = explained
      obj.report = (stamp, *self.to_ego(x, y))
      self._stable(obj)
      obj.tracked += tracked
      obj.radar_moving = obj.radar_moving or radar_moving
      if doppler:
        # Isolated noisy samples must not add up over a long observation; once the
        # samples are close enough together the vehicle stays one after it stops.
        obj.doppler += 1
        obj.forward_samples += doppler > 0
        obj.doppler_recent.append(stamp)
      # Confirmation waits for measured reports; the samples themselves are kept as before.
      if (len(obj.doppler_recent) == self.DOPPLER_SAMPLES and obj.doppler_recent[-1] - obj.doppler_recent[0] <= self.DOPPLER_WINDOW
          and obj.tracked >= self.DOPPLER_TRACKED):
        obj.doppler_confirmed = True
      if obj.radar_moving or obj.doppler_confirmed:
        obj.motion_confirmed = True
      cx, cy = x - obj.drift * hc, y - obj.drift * hs
      obj.history.append((stamp, cx, cy, self.odometer))
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
        # Reported positions scatter with ego speed (recorded: a median .5-.7m between reports above
        # 30km/h, .15-.2m at rest), so the scatter of single reports says little. What a resting
        # return cannot do is travel: net movement over the window stays small against ego's own.
        obj.stationary = span >= .3 and net <= self.RESTING_TRAVEL * (last[3] - first[3]) + self.RESTING_NET and not coherent
        if obj.motion_confirmed and abs(hc * vx + hs * vy) > self.RESTING_SPEED:
          obj.stationary = False  # Slower than the travel bound, but the radar reports it driving.
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
      obj.last_motion = (hc * obj.vx + hs * obj.vy, hc * obj.vy - hs * obj.vx)
      self._settle(obj, ex, ey, stamp)
      obj.slot, obj.interior, center = self._lane(ex, ey, moving=obj.motion_confirmed, settled=obj.settled)
      # A remembered vehicle that drives off reappears just ahead of where it stood, slowly, with the
      # radar's moving class or its old radar id, and the gap then grows. It is no longer remembered.
      if obj.hits == 1:
        if self.holding and obj.slot in (1, 2) and hc * vx + hs * vy <= self.RELEASE_SPEED:
          obj.ahead_of = tuple(
            (m.key, ex - self.local[m.key][0]) for m in self.objects.values()
            if m is not obj and m.slot == obj.slot and not m.departed and self.now - m.seen > self.PREDICTION_HOLD
            and (m.vehicle >= .6 or m.admitted) and self.MEMORY_MIN_X <= self.local[m.key][0] <= self.MEMORY_MAX_X
            and self.RELEASE_GAP[0] <= ex - self.local[m.key][0] <= self.RELEASE_GAP[1]
            and abs(ey - self.local[m.key][1]) <= self.RELEASE_LATERAL)
      else:
        for remembered, gap in obj.ahead_of:
          m = self.objects.get(remembered)
          if (m is not None and (obj.radar_moving or source == m.source)
              and ex - self.local[remembered][0] >= gap + self.RELEASE_ADVANCE):
            m.departed = True
      # As in the existing selector: while ego is stopped or stopping, a return that has
      # rested for .5s near the middle of an adjacent lane is a waiting vehicle, even if
      # it never moved in view. Walls and wall-side positions are excluded at selection.
      if (self.WAITING_ADMISSION and self.holding and obj.stationary and obj.slot in (1, 2) and not obj.structure and 0. < ex <= self.MEMORY_MAX_X
          and stamp - obj.born >= .5 and obj.hits >= 7 and obj.tracked >= self.WAITING_TRACKED
          and self._body_clear(obj.slot, ex, ey) and not self._crossing(obj) and not self._band_line(ex, ey)):
        obj.admitted = True
      if obj.motion_confirmed:
        obj.vehicle = max(.8, obj.vehicle)
        if not obj.trail or math.hypot(obj.x - obj.trail[-1][0], obj.y - obj.trail[-1][1]) >= self.TRAIL_STEP:
          obj.trail.append((obj.x, obj.y))
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
        obj.vision_seen = True

  def _walls(self, measured):
    self.bends = []  # (key, degrees, gap) joined this cycle through a bend, for inspection.
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
      if self.WALL_BEND > 0.:
        self._bend(wall, candidates, measured)
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
    self._bands(candidates, measured, inside)

  def _bend(self, wall, candidates, measured):
    """Continue a wall from either end through returns that follow its end segment within WALL_BEND."""
    limit = math.cos(math.radians(self.WALL_BEND))
    for forward in (True, False):
      line = sorted(self.to_ego(*anchor) for anchor in wall.points)
      if len(line) < 2:
        return
      a, b = (line[-2], line[-1]) if forward else (line[1], line[0])
      while len(wall.members) < 24:
        dx, dy = b[0] - a[0], b[1] - a[1]
        segment = math.hypot(dx, dy)
        if segment < .3:
          break
        best = None
        for x, y, key in candidates:
          # Not the lane-interior protection of new rows: the posts of an opening pocket stand inside the lane
          # the model draws there (3e3/15). A vehicle that ever moved, or that vision saw, is never a candidate.
          if key in wall.members or key not in measured:
            continue
          ex, ey = x - b[0], y - b[1]
          gap = math.hypot(ex, ey)
          if .6 <= gap <= self.WALL_MAX_GAP and (ex * dx + ey * dy) / (gap * segment) >= limit and (best is None or gap < best[0]):
            best = gap, x, y, key
        if best is None:
          break
        gap, x, y, key = best
        angle = math.degrees(math.acos(min(1., ((x - b[0]) * dx + (y - b[1]) * dy) / (gap * segment))))
        self.bends.append((key, round(angle, 1), round(gap, 2)))
        wall.members += (key,)
        wall.points.append((self.objects[key].x, self.objects[key].y))
        a, b = b, (x, y)

  def _band_row(self, points):
    """Returns of one offset as (road distance, offset, key, inside a lane); those inside a trusted
    lane count only in a long dense row, as for straight rows."""
    if any(p[3] for p in points) and (len(points) < self.DENSE_ROW_POINTS
                                      or max(p[0] for p in points) - min(p[0] for p in points) < self.DENSE_ROW_SPAN):
      points = [p for p in points if not p[3]]
    return sorted(points)

  def _bands(self, candidates, measured, inside):
    lines = {}
    for sign, order in ((1, (1, 0)), (-1, (2, 3))):
      for i in order:
        lane = self.lanes.get(i)
        if lane is not None and lane.probability >= .1 and len(self.geometry.get(i, ())) >= 2:
          lines[sign] = i, self.geometry[i]
          break
    if measured:
      self.band_early.intersection_update(self.objects)
      early = [(*self.local[key], key) for key in self.band_early if key in measured and key in self.band_points]
      for x, y, key in candidates + early:
        sign = 1 if y > 0. else -1
        base = interpolate(lines[sign][1], x) if abs(y) > .2 and sign in lines else None
        if base is None:
          continue
        old = self.band_points.get(key)
        self.band_points[key] = (sign, lines[sign][0], self.odometer + x, y - base,
                                 self.odometer if key in measured or old is None else old[4], key in inside)
      # A return is forgotten after BAND_MEMORY metres without a measurement, or at once with vehicle evidence.
      for key in [k for k, p in self.band_points.items() if self.odometer - p[4] > self.BAND_MEMORY
                  or ((obj := self.objects.get(k)) is not None and (obj.vision or self._radar_vehicle(obj)))]:
        del self.band_points[key]
      claimed = set()
      for band in self.bands:
        points = {key: p for key, p in self.band_points.items() if p[0] == band.sign and p[1] == band.reference and key not in claimed}
        # The model's line moves between frames and all offsets move with it: follow the members.
        current = sorted(points[key][3] for key in band.members if key in points)
        if not current:
          band.members = {}
          continue
        band.offset = current[len(current) // 2]
        row = self._band_row([(p[2], p[3], key, p[5]) for key, p in points.items() if abs(p[3] - band.offset) <= self.BAND_BIN])
        positions = [p[0] for p in row if p[2] in band.members]
        if not positions:
          band.members = {}
          continue
        # The band continues through returns within WALL_CONTINUE of its ends, as a straight row does.
        low, high = min(positions), max(positions)
        for p in row:
          if high < p[0] <= high + self.WALL_CONTINUE:
            high = p[0]
        for p in reversed(row):
          if low - self.WALL_CONTINUE <= p[0] < low:
            low = p[0]
        band.members = {p[2]: p[0] for p in row if low <= p[0] <= high}
        claimed.update(band.members)
        if any(key in measured for key in band.members):
          band.supported = self.odometer
        positions = sorted(band.members.values())
        gaps = sorted(b - a for a, b in zip(positions, positions[1:]) if b - a >= .6)
        band.spacing = gaps[len(gaps) // 2] if gaps else 0.
        # The next post is expected one spacing beyond the last (or two, with one post missed), at the
        # band's offset. A fresh return there without any vehicle evidence is that post: it joins now,
        # whatever its age or radar track state, instead of after it has shown that it rests.
        line = lines.get(band.sign)
        if band.spacing <= 0. or line is None or line[0] != band.reference:
          continue
        for key in sorted(measured.difference(self.band_points), key=lambda k: self.local[k][0]):
          obj = self.objects[key]
          x, y = self.local[key]
          base = interpolate(line[1], x) if y * band.sign > .2 else None
          if base is None or obj.motion_confirmed or obj.doppler or obj.vision or obj.vision_seen or obj.interior:
            continue
          ahead = self.odometer + x - positions[-1]
          if (abs(y - base - band.offset) <= self.BAND_PREDICTION
              and min(abs(ahead - band.spacing), abs(ahead - 2. * band.spacing)) <= self.BAND_PREDICTION):
            self.band_points[key] = (band.sign, band.reference, self.odometer + x, y - base, self.odometer, False)
            self.band_early.add(key)
            band.members[key] = self.odometer + x
            claimed.add(key)
            positions.append(self.odometer + x)
      for sign, (reference, _) in lines.items():
        cells = {}
        for key, p in self.band_points.items():
          if p[0] == sign and p[1] == reference and key not in claimed:
            cells.setdefault(math.floor(p[3] / self.BAND_BIN), []).append((p[2], p[3], key, p[5]))
        for window in sorted((cells[c] + cells.get(c + 1, []) for c in cells), key=len, reverse=True):
          run = []
          for p in self._band_row([p for p in window if p[2] not in claimed]) + [None]:
            if p is not None and (not run or p[0] - run[-1][0] <= self.BAND_GAP):
              run.append(p)
              continue
            # Three posts (returns nearer than .6m are one post) no farther apart than in a straight row, over 4m.
            posts = 1 + sum(b[0] - a[0] >= .6 for a, b in zip(run, run[1:])) if run else 0
            if (posts >= 3 and run[-1][0] - run[0][0] >= 4. - 1e-5 and any(q[2] in measured for q in run)
                and len(self.bands) + len(self.walls) < self.MAX_WALLS):
              claimed.update(q[2] for q in run)
              self.bands.append(Band(sign, reference, sum(q[1] for q in run) / len(run), {q[2]: q[0] for q in run}, self.odometer))
            run = [p] if p is not None else []
    # A band ages with the road, not the clock: it is folded after driving three of its spacings
    # (at least BAND_FOLD metres) without a measured member, and never while ego stands.
    self.bands = [band for band in self.bands
                  if band.members and self.odometer - band.supported <= max(3. * band.spacing, self.BAND_FOLD)]
    for band in self.bands:
      reference, line = lines.get(band.sign, (None, None))
      if reference != band.reference:
        continue
      first, last = min(band.members.values()) - self.odometer, max(band.members.values()) - self.odometer
      ends = [(x, y + band.offset) for x in (first, last) for y in (interpolate(line, x),) if y is not None]
      geometry = sorted(ends + [(x, y + band.offset) for x, y in line if first < x < last])
      if len(geometry) >= 2 and geometry[-1][0] > geometry[0][0]:
        self.wall_geometry.append(geometry)
      for key in band.members:
        obj = self.objects.get(key)
        if obj is not None and not obj.vision and not self._radar_vehicle(obj):
          obj.structure = True

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

  @staticmethod
  def _approach_only(obj):
    """A same-direction vehicle never has speed towards ego: an object confirmed by approach speed
    alone was oncoming, crossing or noise. It is neither beside ego nor waiting there, whatever
    velocity the map holds for it (a resting one holds none)."""
    return not (obj.radar_moving or obj.forward_samples or obj.vision_seen or obj.admitted)

  def _wall_outside(self, x, y):
    for points in self.wall_geometry:
      wy = interpolate(points, x)
      if wy is not None and wy * y > 0. and abs(y) >= abs(wy) - .25:
        return True
    return False

  def _select(self):
    groups = [[], [], []]
    far = None  # (x, key): the displayed FF object beyond FF_FAR_X, eligible but not in ego's lane slot.
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
      slot, _, center_distance = self._lane(x, y, respect_walls=True, moving=obj.motion_confirmed, settled=obj.settled)
      if (slot != 0 and obj.key == self.selected[0] and x > self.FF_FAR_X
          and (obj.motion_confirmed or obj.vision or not self._wall_outside(x, y))):
        far = (x, obj.key)
      if slot is not None and slot != 0 and x > self.MAX_SIDE_X:
        continue  # Same side-slot distance limit as the existing selector.
      if slot is None and obj.vision and abs(y) <= 2. and not self._wall_outside(x, y):
        slot, center_distance = 0, abs(y)
      if slot is None:
        continue
      # The frozen position of an unseen object says nothing about its direction:
      # use the direction of its last real measurement, never the zeroed velocity.
      if slot != 0 and (obj.last_motion[0] < -1. / 3.6 or self._approach_only(obj)):
        continue  # Keep opposing traffic in the map, outside same-direction side slots.
      if obj.structure or (not obj.motion_confirmed and not obj.vision and self._wall_outside(x, y)):
        continue
      # Similar longitudinal distance: prefer an interior representative, without
      # merging the underlying tracks or claiming that they are the same body.
      groups[slot].append((x, center_distance, obj.key))
    selected = []
    for slot, candidates in enumerate(groups):
      if slot and selected[0] is not None:
        lead_x = self.local[selected[0]][0]
        # A bridged FF object measured beside ego's lane stays FF, not a side vehicle as well.
        candidates = [c for c in candidates if c[2] != selected[0] and not self._behind_lead(c[2], slot, lead_x)]
      if slot and candidates and not self.holding and not self.lane_change:
        # An absent outer line must not let the lane beyond displace a measured adjacent candidate.
        inner, outer, sign = (1, 0, 1.) if slot == 1 else (2, 3, -1.)
        line = self.lanes.get(outer)
        line_usable = line is not None and line.probability >= .1
        adjacent = []
        for candidate in candidates:
          x, y = self.local[candidate[2]]
          outer_valid = line_usable and interpolate(self.geometry.get(outer, []), x) is not None
          base = interpolate(self.geometry.get(inner, []), x)
          if outer_valid or base is None or sign * (y - base) <= self.OUTER_FAR:
            adjacent.append(candidate)
        if adjacent:
          candidates = adjacent
      if not candidates:
        selected.append(None)
      else:
        nearest = min(c[0] for c in candidates)
        close = [c for c in candidates if c[0] <= nearest + 1.]
        old = next((c for c in close if c[2] == self.selected[slot]), None)
        best = min(close, key=lambda c: (c[1], c[0], c[2]))
        selected.append(old[2] if old and old[1] <= best[1] + .25 else best[2])
      if not slot:
        selected[0] = self._bridge(self._far_hold(selected[0], far))
    self.remembered = [False, False, False]
    if self.holding:
      for slot in (1, 2):
        if selected[slot] is None and not self.blindspot[slot - 1]:
          selected[slot] = self._stop_memory(slot)
          if selected[slot] is not None and selected[slot] == selected[0]:
            selected[slot] = None
          self.remembered[slot] = selected[slot] is not None
    self.selected = tuple(selected)

  def _ego_lines(self, x):
    """Ego's left and right lane lines at x (None where unusable)."""
    return tuple(interpolate(self.geometry.get(i, []), x) if i in self.lanes and self.lanes[i].probability >= .1 else None for i in (1, 2))

  def _centre(self, x):
    """Middle of ego's lane lines at x, or None."""
    left, right = self._ego_lines(x)
    return None if left is None or right is None else .5 * (left + right)

  def _far_hold(self, normal, far):
    """The displayed far FF object while its lane slot flickers away, for up to FF_FAR_HOLD; else ``normal``."""
    if far is None or self.lane_change or self.bridge is not None or (normal is not None and self.local[normal][0] <= far[0]):
      self.far_since = None
      return normal
    if self.far_since is None or self.far_since[0] != far[1]:
      self.far_since = (far[1], self.now)
    return far[1] if self.now - self.far_since[1] <= self.FF_FAR_HOLD + 1e-6 else normal

  def _bridge(self, normal):
    """The FF object, kept on vision while its radar report is lost or has jumped out of ego's lane; else ``normal``."""
    kept = self._bridged(normal)
    if kept != normal or normal is None:
      return kept
    # Record where the FF object was last reported inside ego's lane, unless that report is part of a jump.
    obj = self.objects[normal]
    if obj.report is not None and obj.report[0] == self.radar_stamp and self.now - self.jumps.get(obj.source, -math.inf) > self.BRIDGE_JUMP_TIME + 1e-6:
      _, x, y = obj.report
      left, right = self._ego_lines(x)
      speed = self.cos * obj.vx + self.sin * obj.vy
      if left is not None and right is not None:
        if right <= y <= left:
          self.hold = (normal, y - .5 * (left + right), True, speed)  # Relative to the lane centre.
      elif abs(y) <= self.BRIDGE_LANE:
        self.hold = (normal, y, False, speed)
    if self.hold is not None and self.hold[0] != normal:
      self.hold = None
    return normal

  def _bridged(self, normal):
    bridge = self.bridge
    key = bridge['key'] if bridge else self.selected[0]
    obj = self.objects.get(key) if key is not None else None
    self.bridge = None
    if obj is not None and bridge is not None and normal == key:
      self._forget_lateral(obj)  # Back in ego's lane on its own reports.
    if obj is None or normal == key or self.lane_change or not self.usable:
      return normal
    if bridge is None:
      # A lost report, a report that jumped sideways, or one that crossed the lane line slowly (vision decides
      # below whether the vehicle really left the lane).
      unseen = obj.seen != self.radar_stamp
      jumped = self.radar_stamp - self.jumps.get(obj.source, -math.inf) <= self.PREDICTION_HOLD + 1e-6
      # A slow crossing is one measured into a side lane; an object left out of FF for another reason
      # (recorded: a pillar in a car park, once a structure) is not bridged.
      crossed = obj.slot in (1, 2) and not obj.structure and self.local[key][0] <= self.BRIDGE_DRIFT_X
      if not (unseen or jumped or crossed):
        return normal
      cause = 'lost' if unseen else 'jump' if jumped else 'drift'
      start, x0, dx = self.now, self.local[key][0], None
      held = self.hold is not None and self.hold[0] == key
      speed = self.hold[3] if held else self.cos * obj.vx + self.sin * obj.vy
    else:
      start, x0, dx, speed, cause = bridge['start'], bridge['x'], bridge['dx'], bridge['speed'], bridge['cause']
      if self.now - start > self.BRIDGE_HOLD:
        return normal
    matches = []
    for x, y, v in self.vision_leads:
      centre = self._centre(x)
      if abs(x - x0) <= max(3., .2 * x0) and abs(y - (centre if centre is not None else 0.)) <= self.BRIDGE_LANE:
        matches.append((abs(x - x0), x, v))
    if not matches:
      return normal
    _, vision_x, vision_speed = min(matches, key=lambda m: m[:2])
    # The displayed speed (approach indication) is vision's, else the last one before: the merged
    # track reports the speed of the vehicle it merged with.
    speed = vision_speed if vision_speed is not None else speed
    # Follow the vision distance from where the object was drawn: the radar's reflection point and the
    # vision lead differ by 1-2m, and the display should not jump by that.
    dx = x0 - vision_x if dx is None else dx
    x = vision_x + dx
    if normal is not None and self.local[normal][0] <= x + max(3., .2 * x):
      # A nearer measured vehicle, such as one cutting in, takes FF; so does one at the bridged distance,
      # which is the lead measured again as a new object. Only a farther one cannot.
      return normal
    hold = self.hold if self.hold is not None and self.hold[0] == key else None
    centre = self._centre(x)
    centre = 0. if centre is None else centre
    if obj.stable_y is not None:
      y = centre + obj.stable_y
    elif hold is None:
      y = centre
    else:
      _, lateral, centred, _ = hold
      y = centre + lateral if centred else lateral
    self.local[key] = x, y
    self.bridge = dict(key=key, start=start, x=x, y=y, dx=dx, speed=speed, cause=cause, kind='lost' if obj.seen != self.radar_stamp else 'jump')
    return key

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
    lane, _, _ = self._lane(lx, ly, respect_walls=True, moving=obj.motion_confirmed, settled=obj.settled)
    if lane is not None and lane != 0 and lx > self.MAX_SIDE_X:
      return 'beyond side distance limit'
    if lane is None and obj.vision and abs(ly) <= 2. and not self._wall_outside(lx, ly):
      lane = 0
    if lane is None:
      return 'no lane slot'
    if lane != slot:
      return f"lane slot {('FF', 'LF', 'RF')[lane]} instead"
    if slot != 0 and (obj.last_motion[0] < -1. / 3.6 or self._approach_only(obj)):
      return 'opposing direction'
    if not obj.motion_confirmed and not obj.vision and self._wall_outside(lx, ly):
      return 'beyond wall'
    return 'another object preferred'

  def _stop_memory(self, slot):
    """A side vehicle that left radar coverage while ego stops; display memory only."""
    candidates = []
    for obj in self.objects.values():
      x, _ = self.local[obj.key]
      if (obj.slot != slot or obj.departed or obj.superseded or obj.structure or (obj.vehicle < .6 and not obj.admitted) or obj.hits < 3 or obj.existence < .5
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
      if self._approach_only(obj):
        continue
      candidates.append((obj.key != self.selected[slot], abs(x), obj.key))
    return min(candidates)[2] if candidates else None

  def update(self, live, model, now, v_ego, *, radar_time=None, model_time=None, yaw_rate=None, reverse=False,
             select=True, lane_curves=None, a_ego=None, blindspot=(False, False), evidence=None, lane_change=False):
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
    self.lane_change = bool(lane_change)
    # Offline: {radar track id: (radar motion class, reported distance)} read from the raw
    # track message. A live build would carry the class on each RadarPoint as motionState.
    self.evidence = evidence or {}
    self.yaw_rate = yaw_rate
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
    self.vision_leads = [(lead.x[0] - 1.52, -lead.y[0], float(lead.v[0]) if len(getattr(lead, 'v', ())) and math.isfinite(lead.v[0]) else None)
                         for lead in getattr(model, 'leadsV3', ())
                         if lead.prob >= self.BRIDGE_PROB and len(lead.x) and len(lead.y) and math.isfinite(lead.x[0]) and math.isfinite(lead.y[0])
                         ] if model is not None and 0. <= now - model_time <= .25 else []
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
      x, y = self.local[obj.key]  # A bridged FF object is drawn where the bridge holds it.
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
                walls=self.wall_geometry)


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
    desire = str(getattr(getattr(md, 'meta', None), 'desire', ''))
    tracker.update(live, md, now, out.vEgo, radar_time=_live.radar_time, model_time=_live.model_time,
                   reverse=str(out.gearShifter) == 'reverse', a_ego=out.aEgo,
                   blindspot=(out.leftBlindspot, out.rightBlindspot),
                   lane_change=desire in ('laneChangeLeft', 'laneChangeRight') or bool(getattr(out, 'leftBlinker', False) or getattr(out, 'rightBlinker', False)))
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
      bridged = slot == 0 and tracker.bridge is not None and tracker.bridge['key'] == obj.key
      forward = tracker.bridge['speed'] if bridged else tracker.cos * obj.vx + tracker.sin * obj.vy
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
