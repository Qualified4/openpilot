# cython: language_level=3, infer_types=False
"""Exact float64 loops for the optional display-only CCNC map backend.

Build with -fno-fast-math -ffp-contract=off. Keep Python sorting, bisect,
validation and state transitions; only elementary arithmetic is unboxed.
"""
from bisect import bisect_left
import math


cdef list _placed(points, double px, double py, double c, double s):
  cdef double x, y, dx, dy
  cdef list result = []
  for x, y in points:
    dx, dy = x - px, y - py
    result.append((c * dx + s * dy, -s * dx + c * dy))
  return result


cdef list _world(points, double px, double py, double c, double s):
  cdef double x, y
  cdef list result = []
  for x, y in points:
    result.append((px + c * x + s * y, py + s * x - c * y))
  return result


def to_ego(self, double x, double y):
  cdef double px = self.pose[0], py = self.pose[1]
  cdef double c = self.cos, s = self.sin
  return c * (x - px) + s * (y - py), -s * (x - px) + c * (y - py)


def geometry(self):
  self.geometry = {}
  placed = {}
  for i, lane in self.lanes.items():
    placed[i] = _placed(lane.points, self.pose[0], self.pose[1], self.cos, self.sin)
    points = sorted(placed[i])
    self.geometry[i] = [p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]
  return placed


def interpolate(points, x):
  if len(points) < 2 or not points[0][0] <= x <= points[-1][0]:
    return None
  i = max(1, bisect_left(points, (x, -math.inf)))
  a, b = points[i - 1], points[i]
  if b[0] <= a[0]:
    return None
  cdef double ax = a[0], ay = a[1], bx = b[0], by = b[1], xx = x
  return ay + (by - ay) * (xx - ax) / (bx - ax)


def update_lanes(self, model, stamp, lane_curves, previous_placed, boundary_type):
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
      if not self.ego_change or i > 3:
        if old is not None:
          old.probability *= .9
        continue
      # While ego changes lanes the model's line confidence collapses, but its current positions stay
      # continuous (the lane display ignores the confidence then too): use them, never as strong.
      probability = .1
      self.weak.add(i)
    else:
      # Only a confident observation of the line ends its position-only standing; an invalid or skipped
      # model keeps the stored line (and its standing) as it was.
      self.weak.discard(i)
    # Keep one bracketing point at each edge for interpolation, not the full 190m curve.
    distances = [p[0] for p in coords]
    first = max(0, bisect_left(distances, self.MIN_X) - 1)
    last = min(len(coords), bisect_left(distances, self.FAR_X) + 1)
    world = _world(coords[first:last], px, py, pc, ps)  # to_world(x, -y, pose)
    placed = _placed(world, self.pose[0], self.pose[1], self.cos, self.sin)
    local = sorted(placed)
    previous_geometry = self.geometry.get(i, [])
    errors = [abs(y - previous) for x, y in local if 0. <= x <= 30.
              for previous in (interpolate(previous_geometry, x),) if previous is not None]
    consistent = old.consistent + 1 if old is not None and errors and max(errors) <= .35 else 1
    # Retain actual previously seen geometry behind the current camera domain.
    limit = local[0][0] - .1
    behind = []
    if old is not None:
      old_ego = previous_placed[i] if previous_placed is not None else (self.to_ego(*point) for point in old.points)
      behind = [(point, ego) for point, ego in zip(old.points, old_ego, strict=False) if -5. <= ego[0] < limit][-24:]
    self.lanes[i] = boundary_type([point for point, _ in behind] + world[:40], float(probability), min(consistent, 20))
    # Same result as _geometry() for this line, reusing the transforms above.
    points = sorted([ego for _, ego in behind] + placed[:40])
    self.geometry[i] = [p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]

def lane_view(self, x):
  # Keep the stored physical lines; only their roles follow ego across a late renumber.
  indices = {i: i + self.lane_offset for i in range(4) if 0 <= i + self.lane_offset < 4}
  indices.update({4: 4, 5: 5})
  bounds = {i: interpolate(self.geometry.get(j, []), x) for i, j in indices.items()}
  usable = {i for i, j in indices.items() if j in self.lanes and self.lanes[j].probability >= .1 and bounds[i] is not None}
  strong = {i for i in usable if self.lanes[indices[i]].probability >= .6 and self.lanes[indices[i]].consistent >= 3}
  return bounds, usable, strong


# Scalar interpolation keeps numpy.interp's slope-first order and exact-knot handling.
# Arrays reach these loops only after the existing finite/strictly-increasing validation.
import numpy as np
from libc.math cimport isfinite, isnan, fabs


cpdef double interp_np(double x, const double[:] xs, const double[:] ys):
  cdef Py_ssize_t n = xs.shape[0], lo = 0, hi = n, mid, j
  cdef double slope, value
  if n == 0 or ys.shape[0] != n:
    raise ValueError('interpolation array size mismatch')
  if n == 1:
    return ys[0]
  if isnan(x):
    return x
  if x < xs[0]:
    return ys[0]
  if x > xs[n - 1]:
    return ys[n - 1]
  while lo < hi:
    mid = lo + (hi - lo) // 2
    if x >= xs[mid]:
      lo = mid + 1
    else:
      hi = mid
  j = lo - 1
  if j == n - 1 or xs[j] == x:
    return ys[j]
  slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
  value = slope * (x - xs[j]) + ys[j]
  if isnan(value):
    value = slope * (x - xs[j + 1]) + ys[j + 1]
    if isnan(value) and ys[j] == ys[j + 1]:
      value = ys[j]
  return value


def valid_boundary(x, y):
  if len(x) < 2 or len(x) != len(y):
    return False
  xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
  # Retain the reference's behavior for alternate multidimensional test samplers.
  if xs.ndim != 1 or ys.ndim != 1:
    return bool(np.isfinite(xs).all() and np.isfinite(ys).all() and (xs[1:] > xs[:-1]).all())
  return _valid_arrays(xs, ys)


cdef bint _valid_arrays(const double[:] xs, const double[:] ys):
  cdef Py_ssize_t i, n = xs.shape[0]
  if n < 2 or ys.shape[0] != n:
    return False
  for i in range(n):
    if not isfinite(xs[i]) or not isfinite(ys[i]) or (i > 0 and xs[i] <= xs[i - 1]):
      return False
  return True


def nearest_boundary(double x, double y, curves):
  nearest = []
  for index, xs, ys in curves:
    if xs[0] <= x <= xs[-1]:
      nearest.append((fabs(y + interp_np(x, xs, ys)), index))
  return min(nearest, default=(math.inf, None))


def side_lane_center(md, side, curve=None, min_probability=.6):
  inner_idx, outer_idx = (1, 0) if side == 0 else (2, 3)
  if md is None or len(md.laneLines) < 4 or len(md.laneLineProbs) < 4:
    return None
  if not all(math.isfinite(md.laneLineProbs[i]) and md.laneLineProbs[i] >= min_probability for i in (inner_idx, outer_idx)):
    return None
  if curve is None:
    inner, outer = md.laneLines[inner_idx], md.laneLines[outer_idx]
    ix, iy = np.asarray(inner.x, dtype=np.float64), np.asarray(inner.y, dtype=np.float64)
    ox, oy = np.asarray(outer.x, dtype=np.float64), np.asarray(outer.y, dtype=np.float64)
    valid = valid_boundary(ix, iy) and valid_boundary(ox, oy)
  else:
    (ix, iy, iv), (ox, oy, ov) = curve(md, inner_idx), curve(md, outer_idx)
    valid = iv and ov
  if not valid or not (ix[0] <= 20. <= ix[-1] and ox[0] <= 20. <= ox[-1]):
    return None
  cdef double inner_y = interp_np(20., ix, iy), outer_y = interp_np(20., ox, oy)
  cdef double width = inner_y - outer_y if side == 0 else outer_y - inner_y
  cdef double center
  if not 2.3 <= width <= 4.8:
    return None
  center = abs(float(iy[0])) + width * .5
  return center if isfinite(center) else None


_ccnc_valid_boundary = valid_boundary


def update_ego_follow(self, md, speed, desire, left, right, now):
  was_active = self.ego_follow_active
  if self.ego_follow_time is not None and not 0 <= now - self.ego_follow_time <= .15:
    self.ego_follow_until = -math.inf
    self.ego_follow_motion.clear()
    self.ego_change_motion.clear()
  self.ego_follow_time = now
  self.ego_targets.clear()
  motion = self.ego_follow_motion
  stamp = getattr(md, 'timestampEof', None) or id(md)
  if md is not None and stamp != self.ego_follow_model:
    self.ego_follow_model = stamp
    velocities = []
    lines, probabilities = getattr(md, 'laneLines', ()), getattr(md, 'laneLineProbs', ())
    if len(lines) >= 3 and len(probabilities) >= 3:
      for index in (1, 2):
        line = lines[index]
        if (probabilities[index] < .1 or not math.isfinite(probabilities[index])
            or not _ccnc_valid_boundary(line.x, line.y) or line.x[0] > 0 or line.x[-1] < 5):
          continue
        y0, ym, y5 = np.interp((0., 2.5, 5.), line.x, line.y)
        velocities.append(float(speed * (-3 * y0 + 4 * ym - y5) / 5.))
    if len(velocities) == 2 and abs(velocities[0] - velocities[1]) <= .5:
      motion.append((now, sum(velocities) * .5))
    if len(velocities) == 1 or (len(velocities) == 2 and abs(velocities[0] - velocities[1]) <= .5):
      self.ego_change_motion.append((now, sum(velocities) / len(velocities)))
  while motion and now - motion[0][0] > .3:
    motion.popleft()
  moving = (left != right and len(motion) >= 4 and motion[-1][0] - motion[0][0] >= .15
            and (1 if left else -1) * self._median([v for _, v in motion]) >= .3)
  state = str(getattr(getattr(md, 'meta', None), 'laneChangeState', 'off'))
  confirmed = self.ego_change_motion
  while confirmed and now - confirmed[0][0] > .3:
    confirmed.popleft()
  changing = (left != right and len(confirmed) >= 4 and confirmed[-1][0] - confirmed[0][0] >= .15
              and (1 if left else -1) * self._median([v for _, v in confirmed]) >= .3)
  self.ego_change_active = (bool(desire in (3, 4) or left or right)
                            and (desire in (3, 4) or state in ('laneChangeStarting', 'laneChangeFinishing') or changing))
  if desire in (3, 4) or state in ('laneChangeStarting', 'laneChangeFinishing') or moving:
    self.ego_follow_until = now + 1.
  self.ego_follow_active = now <= self.ego_follow_until
  if was_active and not self.ego_follow_active:
    self.ego_resume, self.ego_drawn = self.ego_drawn, {}
  elif not was_active and self.ego_follow_active:
    self.ego_resume.clear()
  self.ego_resume = {key: value for key, value in self.ego_resume.items() if 0 <= now - value[1] * .01 <= .15}
