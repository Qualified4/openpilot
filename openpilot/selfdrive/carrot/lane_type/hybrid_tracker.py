"""Offline 5Hz selection + camera-rate narrow tracking experiment. No CAN.

Image gates qualify observations, not calibrated confidence or physical truth.
The optional continuous-paint alternative never changes the default type output.
"""
import math

import cv2
import numpy as np

from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import (
  PaintTracker, candidates, nearest_s, road_points, project_road, project_static_points, to_local, to_world, world_road_points,
)


class RegisteredPaintTracker(PaintTracker):
  """Require image correspondence before admitting new spatial evidence."""
  require_physical_registration = True

  def reset(self):
    super().reset()
    self.last_registered_stamp = None
    self.continuous = None

  def update(self, *args, continuous_alternative=False, **kwargs):
    stamp = args[5] if len(args) > 5 else kwargs['stamp']
    chain_reset = (self.last_registered_stamp is not None and isinstance(stamp, (int, np.integer)) and
                   stamp-self.last_registered_stamp > 300_000_000)
    if chain_reset:
      self.reset()
    self.registration = {'qualified': False, 'reason': 'no_previous_view'}
    generation = self.generation
    result = super().update(*args, **kwargs)
    result['registration_chain_reset'] = bool(chain_reset)
    if result.get('registration_qualified'):
      self.last_registered_stamp = stamp
    if continuous_alternative:
      self._continuous_alternative(result, generation, stamp)
    return result

  def _static_road_points(self, prev, gray, pose, line, matrix, road_pose=None):
    near_paint = cv2.dilate(prev['mask'].astype(np.uint8), np.ones((15, 15), np.uint8))
    mask = cv2.dilate(prev['gap_mask'] | (prev['mask'].astype(np.uint8)*255), np.ones((1, 21), np.uint8))
    stats = {'qualified': False, 'reason': 'insufficient_unique_static_features', 'source_corners': 0,
             'accepted': 0, 'paint_endpoint_corners': 0, 'road_corners': 0, 'ambiguous_patches': 0}
    self.registration = stats
    yy, xx = np.nonzero(mask)
    if not len(xx):
      return np.empty((0, 2))
    a, b, c, d = max(0, xx.min()-4), max(0, yy.min()-4), min(gray.shape[1], xx.max()+5), min(gray.shape[0], yy.max()+5)
    source = prev['gray'][b:d, a:c]
    points = cv2.goodFeaturesToTrack(source, maxCorners=200, qualityLevel=.01, minDistance=7, mask=mask[b:d, a:c], blockSize=5)
    if points is None:
      return np.empty((0, 2))
    xy = np.rint(points[:, 0]).astype(int)
    eigen = cv2.cornerEigenValsAndVecs(source, 5, 3)[xy[:, 1], xy[:, 0]]
    points += np.array([a, b], np.float32)
    keep = eigen[:, 1] >= .08*eigen[:, 0]  # exclude long-line aperture features
    points = points[keep]
    if not len(points):
      return np.empty((0, 2))
    xyz = road_points(points[:, 0], prev['line'], prev['matrix'])
    if road_pose is not None and prev.get('road_pose') is not None:
      world3d = xyz @ prev['road_pose'][:3, :3].T+prev['road_pose'][:3, 3]
      predicted, now = project_static_points(world3d, road_pose, matrix)
      world = world3d[:, :2]
    else:
      world = to_world(xyz[:, :2], prev['pose'])
      now = to_local(world, pose)
      predicted = project_road(now, line, matrix)
    _, lateral = nearest_s(world, prev['world_path'])
    valid = np.isfinite(now).all(1) & np.isfinite(predicted).all(1) & (lateral <= .18)
    valid &= (xyz[:, 0] >= 4.5) & (xyz[:, 0] <= 20) & (now[:, 0] >= 4.5) & (now[:, 0] <= 20)
    valid &= (predicted[:, 0] >= 16) & (predicted[:, 0] < gray.shape[1]-16)
    valid &= (predicted[:, 1] >= 16) & (predicted[:, 1] < gray.shape[0]-16)
    indices, counts = [], {}
    for i in np.flatnonzero(valid):
      key = math.floor(xyz[i, 0]/.5)
      counts[key] = counts.get(key, 0)+1
      if counts[key] <= 2 and len(indices) < 60:
        indices.append(i)
    if not indices:
      return np.empty((0, 2))
    points, world, predicted, now = points[indices], world[indices], predicted[indices], now[indices]
    stats['source_corners'] = len(points)
    combined = np.vstack((points[:, 0], predicted))
    low = np.maximum(0, np.floor(combined.min(0)).astype(int)-32)
    high = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(combined.max(0)).astype(int)+33)
    source = prev['gray'][low[1]:high[1], low[0]:high[0]]
    target = gray[low[1]:high[1], low[0]:high[0]]
    local_points = points-low.astype(np.float32)
    initial = (predicted-low).astype(np.float32).reshape(-1, 1, 2)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(source, target, local_points,
      initial, winSize=(15, 15), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(target, source, forward,
      local_points.copy(), winSize=(15, 15), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    forward += low.astype(np.float32)
    back += low.astype(np.float32)
    residual = np.linalg.norm(forward[:, 0]-predicted, axis=1)
    good = status[:, 0].astype(bool) & backward_status[:, 0].astype(bool)
    good &= np.linalg.norm(back[:, 0]-points[:, 0], axis=1) <= .8
    good &= residual <= 2.
    measured = road_points(forward[:, 0], line, matrix)[:, :2]
    good &= np.linalg.norm(measured-now[:, :2], axis=1) <= .18
    # Reject a visually interchangeable neighbour; ego proximity alone does
    # not make repeated grooves or another dash the same physical landmark.
    for i in np.flatnonzero(good):
      x, y = np.rint(points[i, 0]).astype(int)
      u, v = np.rint(forward[i, 0]).astype(int)
      if min(x, y, u, v) < 15 or x+15 >= gray.shape[1] or u+15 >= gray.shape[1] or y+15 >= gray.shape[0] or v+15 >= gray.shape[0]:
        good[i] = False
        continue
      patch = prev['gray'][y-5:y+6, x-5:x+6]
      search = gray[v-14:v+15, u-14:u+15]
      scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
      _, peak, _, location = cv2.minMaxLoc(scores)
      a, b = location
      other = scores.copy()
      other[max(0, b-3):b+4, max(0, a-3):a+4] = -1
      if peak < .85 or peak-float(other.max()) < .05 or np.linalg.norm(np.array(location)-[9, 9]) > 2:
        good[i] = False
        stats['ambiguous_patches'] += 1
    accepted = np.flatnonzero(good)
    xy = np.rint(points[:, 0]).astype(int)
    road = (prev['gap_mask'][xy[:, 1], xy[:, 0]] != 0) & (near_paint[xy[:, 1], xy[:, 0]] == 0)
    paint = prev['mask'][xy[:, 1], xy[:, 0]]
    stats.update(accepted=len(accepted), road_corners=int((good & road).sum()),
      paint_endpoint_corners=int((good & paint).sum()), adjacent_road_corners=int((good & ~road & ~paint).sum()),
      residual_median_px=float(np.median(residual[good])) if good.any() else None,
      residual_p95_px=float(np.percentile(residual[good], 95)) if good.any() else None)
    if len(accepted) >= 3:
      s, _ = nearest_s(world[good], prev['world_path'])
      stats['qualified'] = bool(len(np.unique(np.floor(s/.5))) >= 3 and np.ptp(s) >= 2.)
    if stats['qualified']:
      stats['reason'] = 'image_gates_passed_not_independent_ground_truth'
    return world[good & road] if stats['qualified'] else np.empty((0, 2))

  def _continuous_alternative(self, result, generation, stamp):
    result['alternative_type'] = result['type']
    result['continuous_paint_m'] = 0.
    start, longest, interval = None, 0., None
    if result.get('registration_qualified') and self.reference is not None:
      for row in result['profile']:
        if row['ink'] != 1 or 'local_xy' not in row:
          start = None
          continue
        d = row['distance']
        if start is None:
          start = row
        length = abs(d-start['distance'])
        if length > longest:
          longest = length
          points = np.array([start['local_xy'], row['local_xy']])
          prev = self.previous
          world = world_road_points(points, prev['line'], prev['road_pose'])[:, :2] if prev['road_pose'] is not None else to_world(points, prev['pose'])
          ss, _ = nearest_s(world, self.reference, self.origin)
          interval = sorted(ss)
    result['continuous_paint_m'] = longest
    old = getattr(self, 'continuous', None)
    fully_observed = bool(result['profile']) and all(r['ink'] == 1 for r in result['profile'] if r['geometry_visible'])
    result['continuous_full_visible_paint'] = fully_observed
    if interval is None or longest < 10 or not fully_observed or result['gap_m'] > 0 or generation != self.generation:
      self.continuous = None
      return
    overlap = [max(interval[0], old[0][0]), min(interval[1], old[0][1])] if old else interval
    if old and overlap[1]-overlap[0] >= 10:
      self.continuous = (overlap, old[1])
      if stamp-old[1] >= 500_000_000 and result['type'] == 'unknown':
        result['alternative_type'] = 'solid'
    else:
      self.continuous = (interval, stamp)


class HybridPaintTracker(RegisteredPaintTracker):
  """Precise keyframes, bounded original-pixel examination in between."""
  def update(self, *args, precise=True, **kwargs):
    self.precise = precise
    self.current_road_pose = kwargs.get('road_pose')
    if self.current_road_pose is not None:
      self.current_road_pose = np.asarray(self.current_road_pose, float)
    self.current_pose = kwargs.get('pose')
    result = super().update(*args, **kwargs)
    if precise:
      self.key_tracks = result['tracks']
    result['selection_mode'] = 'precise' if precise else 'narrow_tracking'
    return result

  def _observations(self, rgb, line, probability, rpy, k, matrix, gray):
    if self.precise:
      return super()._observations(rgb, line, probability, rpy, k, matrix, gray)
    prev = self.previous
    if prev is None or not getattr(self, 'key_tracks', None):
      return gray, [], []
    if self.current_road_pose is not None and prev['world_path_3d'] is not None:
      predicted = (prev['world_path_3d']-self.current_road_pose[:3, 3]) @ self.current_road_pose[:3, :3]
    else:
      predicted = to_local(prev['world_path'], self.current_pose)
    if not np.isfinite(predicted).all() or np.any(np.diff(predicted[:, 0]) <= 0):
      return gray, [], []
    hint = dict(line, y=np.interp(line['x'], predicted[:, 0], predicted[:, 1]).tolist())
    ribbon = collect_pixels(rgb, hint, probability, rpy, k, end_m=20., half_width_m=.22, include_pixels=False)
    for row in ribbon['rows']:
      row['geometry_visible'] &= bool(predicted[0, 0] <= row['distance_m'] <= predicted[-1, 0])
    visible = [r for r in ribbon['rows'] if r['geometry_visible']]
    if not visible:
      return gray, [], []
    bounds = (min(r['left_x'] for r in visible), min(r['y'] for r in visible),
              max(r['right_x'] for r in visible)+1, max(r['y'] for r in visible)+1)
    return candidates(rgb, ribbon, line, matrix, gray=gray, bounds=bounds, batched=True)

  def _paths(self, found, prior, bootstrap, probability):
    if self.precise:
      return super()._paths(found, prior, bootstrap, probability)
    if bootstrap or prior is None or not getattr(self, 'key_tracks', None):
      return [], 'tracking_without_precise_boundary'
    slope, offset = np.polyfit(np.array([8., 12., 16.])-12, prior, 1)
    if abs(slope) > .075:
      return [], 'tracking_direction_inconsistent'
    primary = self.key_tracks[0]
    paths = []
    for old in self.key_tracks:
      path = dict(old, offset=float(offset+old['offset']-primary['offset']), slope=float(slope+old['slope']-primary['slope']))
      matched = [c for c in found if c['color'] == path['color'] and abs(c['width']-path['width']) <= .10 and
                 abs(c['offset']-path['offset']-path['slope']*(c['x']-12)) <= .075]
      support = len({math.floor(c['x']/.25) for c in matched})*.25
      if len(matched) >= 8 and support >= 2 and np.ptp([c['x'] for c in matched]) >= 3:
        paths.append(dict(path, support_m=support))
      elif not paths:
        return [], 'tracked_primary_not_observed'
    return paths, 'tracked_original_pixels'
