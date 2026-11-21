"""Python reference of the redesigned CCNC front-slot display (ccnc_core src/pos.rs, pos/common.rs, display_a.rs,
display_b.rs): CcncPosDisplayMode 1 draws display A, 2 display B. The device runs the Rust core; this repeats it
operation for operation on the same plain frames (the Rust trace or ``_pos_run_for_test``), so the two compare
exactly. Keep both in step.

A frame is (now, yaw, selected, lead, objs): yaw the livePose yaw rate (left +) or None, selected the existing
selector's FF/LF/RF map keys, lead radarState leadOne (dRel, yRel) or None, objs (vid, x, y, seen, report, lines)
with report the map object's last (stamp, x, y) in the ego frame and lines the ego lane view's left outer, left
inner, right inner and right outer line at x (left +) or None. An output is (assigned keys, drawn lateral).
"""
import math

JUMP = 0.3
ABSORB_TAU = 1.75
NOISE_BASE = 0.06
NOISE_SLOPE = 0.004
GAIN_REF = 0.05
GAIN_MIN = 0.15
GAIN_MAX = 0.6
LANE_WIDTH = 3.5
WIDTH_MIN = 1.8
WIDTH_MAX = 5.0
STRADDLE_BAND = 0.4
LEAD_GATE_X = 3.0
LEAD_GATE_Y = 1.5
PINGPONG_HOLD = 1.0
OWNER_FRESH = 0.3

CROSS_HOLD = 2.0
BAND_HOLD = 1.0
ANIM_SPEED = 1.2
REMAP = 1.2


def _clamp(v, lo, hi):
  return lo if v < lo else hi if v > hi else v


class Track:
  def __init__(self, o, now):
    vid, x, y, seen, report, lines = o
    stamp, x, m = report if report is not None else (-math.inf, x, y)
    self.x, self.p, self.m, self.b, self.y, self.stamp, self.now = x, m, m, 0., m, stamp, now

  @staticmethod
  def gain(x):
    noise = NOISE_BASE + NOISE_SLOPE * max(x, 0.)
    return _clamp(GAIN_REF / noise, GAIN_MIN, GAIN_MAX)

  def step(self, o, now, yaw):
    dt = max(now - self.now, 0.)
    if yaw is not None:
      turn = self.x * yaw * dt
      self.m -= turn
      self.p -= turn
    report = o[4]
    if report is not None:
      stamp, rx, ry = report
      if stamp > self.stamp:
        gap = stamp - self.stamp if math.isfinite(self.stamp) else 0.05
        limit = JUMP * max(gap / 0.05, 1.)
        jump = ry - self.p
        if abs(jump) > limit:
          self.b += jump
          self.m += jump
        else:
          self.m += Track.gain(rx) * (ry - self.m)
        self.p = ry
        self.stamp = stamp
        self.x = rx
    self.b -= self.b * min(dt / ABSORB_TAU, 1.)
    self.y = self.m - self.b
    self.now = now


class LaneView:
  def __init__(self, lines):
    l, r = lines[1], lines[2]
    if l is not None and r is not None:
      left, right = l, r
    elif l is not None:
      left, right = l, l - LANE_WIDTH
    elif r is not None:
      left, right = r + LANE_WIDTH, r
    else:
      left, right = 0.5 * LANE_WIDTH, -0.5 * LANE_WIDTH
    width = left - right
    if not (WIDTH_MIN <= width <= WIDTH_MAX):
      mid = 0.5 * (left + right)
      left = mid + 0.5 * LANE_WIDTH
      right = mid - 0.5 * LANE_WIDTH
    width = left - right

    def outer(line, inner, sign):
      if line is not None and WIDTH_MIN <= sign * (line - inner) <= WIDTH_MAX:
        return line
      return inner + sign * width
    self.boundaries = (outer(lines[0], left, 1.), left, right, outer(lines[3], right, -1.))

  def lane(self, y):
    b = self.boundaries
    if y > b[0]:
      return -2
    if y > b[1]:
      return -1
    if y >= b[2]:
      return 0
    if y >= b[3]:
      return 1
    return 2

  def center(self, lane):
    b = self.boundaries
    width = b[1] - b[2]
    if lane == -2:
      return b[0] + 0.5 * width
    if lane == -1:
      return 0.5 * (b[0] + b[1])
    if lane == 0:
      return 0.5 * (b[1] + b[2])
    if lane == 1:
      return 0.5 * (b[2] + b[3])
    return b[3] - 0.5 * width

  def straddled(self, y):
    best = None
    for i, line in enumerate(self.boundaries):
      d = abs(y - line)
      if d <= STRADDLE_BAND and (best is None or d < best[1]):
        best = (i, d)
    return None if best is None else best[0]


def side_of(view, y):
  i = view.straddled(y)
  if i is not None:
    return 1 if i in (0, 1) else 2
  lane = view.lane(y)
  return 0 if lane == 0 else 1 if lane < 0 else 2


class Common:
  def __init__(self):
    self.tracks = {}
    self.slots = [[None, None, 0.] for _ in range(3)]  # owner, previous, since

  def update(self, f):
    now, yaw, selected, lead_one, objs = f
    objs = {o[0]: o for o in objs}
    self.tracks = {vid: t for vid, t in self.tracks.items() if vid in objs}
    for vid in sorted(objs):
      if vid in self.tracks:
        self.tracks[vid].step(objs[vid], now, yaw)
      else:
        self.tracks[vid] = Track(objs[vid], now)
    lead = self._lead(lead_one, selected, objs)
    return self._assign(now, selected, objs, lead), objs

  def _lead(self, lead_one, selected, objs):
    if lead_one is None:
      return None
    d, y = lead_one
    best = None
    for vid in selected:
      if vid is None or vid not in objs:
        continue
      dx = abs(objs[vid][1] - d)
      if dx <= LEAD_GATE_X and abs(self.tracks[vid].y - y) <= LEAD_GATE_Y and (best is None or dx < best[0]):
        best = (dx, vid)
    return None if best is None else best[1]

  def _side(self, objs, vid):
    return side_of(LaneView(objs[vid][5]), self.tracks[vid].y)

  def _assign(self, now, selected, objs, lead):
    slots = [None] * 3
    x = lambda vid: objs[vid][1]
    nearer = lambda a, b: b is None or x(a) < x(b)
    for s in (1, 2):
      if selected[s] is not None and selected[s] in objs:
        slots[s] = selected[s]
    vid = selected[0]
    if vid is not None and vid in objs:
      side = self._side(objs, vid)
      if side == 0 or vid == lead:
        slots[0] = vid
      elif nearer(vid, slots[side]):
        slots[side] = vid
      else:
        slots[0] = vid
    for s in (1, 2):
      vid = slots[s]
      if vid is None:
        continue
      side = self._side(objs, vid)
      if (side == 0 and slots[0] is None) or vid == lead:
        old = slots[0] if vid == lead else None
        if old is not None:
          slots[s] = old if self._side(objs, old) == s else None
        else:
          slots[s] = None
        slots[0] = vid
    for s in range(3):
      new = slots[s]
      owner, previous, since = self.slots[s]
      if owner is not None:
        fresh = owner in objs and now - objs[owner][3] <= OWNER_FRESH
        elsewhere = any(t != s and v == owner for t, v in enumerate(slots))
        lead_in = s == 0 and new is not None and new == lead
        if (new is not None and new != owner and new == previous and now - since < PINGPONG_HOLD and fresh
            and not elsewhere and not lead_in):
          slots[s] = owner
      memory = self.slots[s]
      if slots[s] != memory[0]:
        if slots[s] is not None and memory[0] is not None:
          memory[1] = memory[0]
          memory[2] = now
        memory[0] = slots[s]
    return slots


def _target(view, place):
  kind, i = place
  return view.center(i) if kind == 'center' else view.boundaries[i]


def _since(memory, value, now):
  return memory if memory is not None and memory[0] == value else (value, now)


class Vehicle:
  def __init__(self, view, y, now):
    i = view.straddled(y)
    self.place = ('straddle', i) if i is not None else ('center', view.lane(y))
    self.value = _target(view, self.place)
    self.moving = False
    self.away = self.band = self.out = None
    self.now = now
    self.goal = self.value

  def step(self, view, y, now):
    dt = max(now - self.now, 0.)
    self.now = now
    if abs(_target(view, self.place) - self.goal) > REMAP:
      last = self.goal
      def nearest(n, at):
        best = 0
        for i in range(1, n):
          if abs(at(i) - last) < abs(at(best) - last):
            best = i
        return best
      if self.place[0] == 'center':
        self.place = ('center', nearest(5, lambda i: view.center(i - 2)) - 2)
      else:
        self.place = ('straddle', nearest(4, lambda i: view.boundaries[i]))
      self.away = self.band = self.out = None
    lane = view.lane(y)
    band = view.straddled(y)
    nxt = None
    kind, current = self.place
    if kind == 'center':
      self.out = None
      self.away = _since(self.away, lane, now) if lane != current else None
      self.band = _since(self.band, band, now) if band is not None else None
      if self.away is not None and now - self.away[1] >= CROSS_HOLD - 1e-9:
        nxt = ('center', self.away[0])
      if nxt is None and self.band is not None and now - self.band[1] >= BAND_HOLD - 1e-9:
        nxt = ('straddle', self.band[0])
    else:
      self.away = None
      self.band = _since(self.band, band, now) if band is not None and band != current else None
      self.out = (self.out if self.out is not None else now) if band is None else None
      if self.out is not None and now - self.out >= BAND_HOLD - 1e-9:
        nxt = ('center', lane)
      if self.band is not None and now - self.band[1] >= BAND_HOLD - 1e-9:
        nxt = ('straddle', self.band[0])
    if nxt is not None:
      self.place = nxt
      self.moving = True
      self.away = self.band = self.out = None
    goal = _target(view, self.place)
    if self.moving:
      step = ANIM_SPEED * dt
      gap = goal - self.value
      if abs(gap) <= step:
        self.value = goal
        self.moving = False
      else:
        self.value += step if gap > 0. else -step
    else:
      self.value = goal
    self.goal = goal


class PosState:
  def __init__(self):
    self.mode = 0
    self.common = Common()
    self.vehicles = {}

  def run(self, mode, f):
    if mode != self.mode:
      self.__init__()
      self.mode = mode
    assigned, objs = self.common.update(f)
    tracks = self.common.tracks
    if mode == 2:
      now = f[0]
      self.vehicles = {vid: v for vid, v in self.vehicles.items() if vid in tracks}
      for vid in sorted(tracks):
        view = LaneView(objs[vid][5])
        if vid in self.vehicles:
          self.vehicles[vid].step(view, tracks[vid].y, now)
        else:
          self.vehicles[vid] = Vehicle(view, tracks[vid].y, now)
      drawn = [None if vid is None else self.vehicles[vid].value for vid in assigned]
    else:
      drawn = [None if vid is None else tracks[vid].y for vid in assigned]
    return assigned, drawn


def run(mode, frames):
  state = PosState()
  return [state.run(mode, f) for f in frames]
