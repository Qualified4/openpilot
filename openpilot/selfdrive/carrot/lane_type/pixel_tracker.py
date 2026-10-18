"""Offline target-paint tracking on unchanged source pixels. No runtime/CAN call.

modelV2 bounds the search, not the selected paint. Width/direction/longitudinal
support select paths; previous road-aligned paths constrain association. UNKNOWN
is distinct from a gap. A gap needs exposed-road appearance, static-road optical
flow and a second translated view. These are experimental visibility heuristics,
not semantic segmentation ground truth.
"""
import math

import cv2
import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels

BIN_M = .25


def rotation(yaw):
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s], [s, c]])


def to_world(points, pose):
  return np.asarray(points) @ rotation(pose[2]).T+pose[:2]


def to_local(points, pose):
  return (np.asarray(points)-pose[:2]) @ rotation(pose[2])


def road_points(xy, line, matrix):
  """Intersect source rays with the model's camera-relative road height.

  This is a local road-surface approximation, not measured per-pixel depth.
  Keeping the original pixel coordinates allows independent geometry review.
  """
  rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(matrix).T
  d = np.full(len(xy), 10.)
  for _ in range(4):
    z = np.interp(d, line['x'], line['z'])
    scale = np.divide(z, rays[:, 2], out=np.full(len(xy), np.nan), where=rays[:, 2] > .001)
    points = rays*scale[:, None]
    d = points[:, 0]
  return points


def project_road(points, line, matrix):
  xyz = np.column_stack((points, np.interp(points[:, 0], line['x'], line['z'])))
  p = xyz @ matrix.T
  return p[:, :2]/p[:, 2, None]


def world_road_points(points, line, transform):
  """Preserve a road observation's height while applying measured attitude."""
  xyz = np.column_stack((points, np.interp(points[:, 0], line['x'], line['z'])))
  return xyz @ transform[:3, :3].T+transform[:3, 3]


def project_static_points(world, transform, matrix):
  """Do not replace a static endpoint's height with a new model prediction."""
  local = (world-transform[:3, 3]) @ transform[:3, :3]
  p = local @ matrix.T
  return p[:, :2]/p[:, 2, None], local


def nearest_s(points, reference, origin=0.):
  """Project onto a fixed world boundary polyline, including curved roads."""
  if len(reference) < 2:
    raise ValueError('missing reference')
  v = np.diff(reference, axis=0)
  lengths = np.linalg.norm(v, axis=1)
  t = np.clip(np.einsum('nsi,si->ns', points[:, None]-reference[:-1], v)/np.maximum(lengths**2, 1e-8), 0, 1)
  delta = points[:, None]-(reference[:-1]+t[..., None]*v)
  distances = np.linalg.norm(delta, axis=2)
  j = distances.argmin(1)
  s = np.r_[0., np.cumsum(lengths)]
  return origin+s[j]+t[np.arange(len(points)), j]*lengths[j], distances[np.arange(len(points)), j]


def candidates(rgb, ribbon, line, matrix, *, gray=None, bounds=None, batched=False):
  origin_x, origin_y, end_x, end_y = bounds if bounds is not None else (0, 0, rgb.shape[1], rgb.shape[0])
  hsv = cv2.cvtColor(rgb[origin_y:end_y, origin_x:end_x], cv2.COLOR_RGB2HSV)
  if gray is None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  elif gray.shape != rgb.shape[:2] or gray.dtype != np.uint8:
    raise ValueError('gray must be an aligned uint8 source image')
  rows, found, pending = [], [], []
  for row in ribbon['rows']:
    row = dict(row, candidates=[], broad=[])
    rows.append(row)
    if not row['geometry_visible']:
      continue
    y, a, b = row['y'], row['left_x'], row['right_x']
    values = hsv[y-origin_y, a-origin_x:b-origin_x+1]
    luma = gray[y, a:b+1].astype(float)
    if batched and len(luma) > 1:
      q = .3*(len(luma)-1)
      j = int(q)
      partition = np.partition(luma, (j, j+1))
      road = float(partition[j]+(q-j)*(partition[j+1]-partition[j]))
    else:
      road = float(np.percentile(luma, 30))
    yellow = (values[:, 0] >= 14) & (values[:, 0] <= 43) & (values[:, 1] >= 40) & (values[:, 2] >= 60) & (luma-road >= 8)
    # White paint in blue skylight/shadow is not necessarily low-saturation.
    # This is only a candidate; metric width, direction and boundary association
    # still apply. Never extend the yellow mask into this cool-colour branch.
    neutral = values[:, 1] <= 75
    cool_shadow = (values[:, 0] >= 85) & (values[:, 0] <= 120) & (values[:, 1] <= 150)
    white = (neutral | cool_shadow) & (values[:, 2] >= 70) & (luma-road >= 18) & ~yellow
    # An asphalt texture peak alone can pass the 18-level pixel cutoff. Require
    # a stronger original-pixel ridge somewhere in each run, retaining its dim
    # edges unchanged. No morphology-filled pixel becomes paint evidence.
    white_bounds = np.flatnonzero(np.diff(np.r_[False, white, False])).reshape(-1, 2)
    for lo, hi in white_bounds:
      if np.max(luma[lo:hi]-road) < 30:
        white[lo:hi] = False
    # Faded transverse paint can alternate between colour masks. Test its
    # combined width before splitting colours. Closing only rejects ambiguity;
    # these filled pixels are NEVER selected paint or a verified road gap.
    # Cool shadow candidates must not join the dark background to a yellow
    # stripe and turn that stripe into a falsely broad marking. A broad cool
    # run still fails its own metric-width check below.
    combined = cv2.morphologyEx((yellow | (white & neutral)).astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]
    broad_bounds = np.flatnonzero(np.diff(np.r_[False, combined.astype(bool), False])).reshape(-1, 2)
    for lo, hi in broad_bounds:
      if batched:
        pending.append((len(rows)-1, a+lo, a+hi, y, None))
        continue
      p = road_points(np.array([[a+lo, y], [a+hi, y]]), line, matrix)
      if abs(p[1, 1]-p[0, 1]) > .34:
        row['broad'].append((int(a+lo), int(a+hi-1)))
    # Separate colours: a white crosswalk must not pull a yellow ridge sideways.
    for color, mask in [('yellow', yellow), ('white', white)]:
      bounds = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
      for lo, hi in bounds:
        x0, x1 = a+lo, a+hi-1
        if batched:
          pending.append((len(rows)-1, x0, x1+1, y, color))
          continue
        p = road_points(np.array([[x0, y], [x1+1, y], [(x0+x1)/2, y]]), line, matrix)
        width = abs(p[1, 1]-p[0, 1])
        x, lateral = p[2, :2]
        if width > .34:
          row['broad'].append((int(x0), int(x1)))
        if hi-lo < 2 or not .04 <= width <= .30 or not 4.5 <= x <= 20:
          continue
        c = {'row': len(rows)-1, 'y': y, 'lo': int(x0), 'hi': int(x1), 'x': float(x), 'lateral': float(lateral),
             'offset': float(lateral-np.interp(x, line['x'], line['y'])), 'width': float(width), 'color': color}
        row['candidates'].append(len(found))
        found.append(c)
  if pending:
    pixels = np.array([[[a, y], [b, y], [(a+b-1)/2, y]] for _, a, b, y, _ in pending])
    points = road_points(pixels.reshape(-1, 2), line, matrix).reshape(-1, 3, 3)
    for (index, a, b, y, color), p in zip(pending, points, strict=True):
      row = rows[index]
      width = abs(p[1, 1]-p[0, 1])
      if width > .34:
        row['broad'].append((int(a), int(b-1)))
      if color is None:
        continue
      x, lateral = p[2, :2]
      if b-a < 2 or not .04 <= width <= .30 or not 4.5 <= x <= 20:
        continue
      c = {'row': index, 'y': y, 'lo': int(a), 'hi': int(b-1), 'x': float(x), 'lateral': float(lateral),
           'offset': float(lateral-np.interp(x, line['x'], line['y'])), 'width': float(width), 'color': color}
      row['candidates'].append(len(found))
      found.append(c)
  return gray, rows, found


def fit_path(found, prior=None):
  """Robust, distance-balanced offset fit; never independently recenter a row."""
  if len(found) < 8:
    return None
  if len({c['color'] for c in found}) > 1:
    fits = []
    for color in ('white', 'yellow'):
      indices = [i for i, c in enumerate(found) if c['color'] == color]
      fitted = fit_path([found[i] for i in indices], prior)
      if fitted:
        selected = np.zeros(len(found), bool)
        selected[np.asarray(indices)[fitted['selected']]] = True
        fits.append(dict(fitted, selected=selected))
    return max(fits, key=lambda f: f['support_m']-.15*abs(f['offset'])) if fits else None
  x, y, width = np.array([[c['x'], c['offset'], c['width']] for c in found]).T
  bins = np.floor(x/.25).astype(int)
  rng = np.random.default_rng(42)
  hypotheses = [(float(a), 0.) for a in np.quantile(y, np.linspace(0, 1, 25))]
  for _ in range(120):
    a, b = rng.choice(len(found), 2, replace=False)
    if abs(x[a]-x[b]) < 3.:
      continue
    slope = (y[b]-y[a])/(x[b]-x[a])
    if abs(slope) <= .075:
      hypotheses.append((float(y[a]-slope*(x[a]-12)), float(slope)))
  best = None
  for intercept, slope in hypotheses:
    residual = abs(y-intercept-slope*(x-12))
    selected = residual <= .075
    if selected.sum() < 8:
      continue
    support = len(np.unique(bins[selected]))*.25
    span = np.ptp(x[selected])
    if support < 2. or span < 3.:
      continue
    score = support-.5*float(np.median(residual[selected]))-.15*abs(intercept)
    if prior is not None:
      error = np.median(abs(intercept+slope*(np.array([8., 12., 16.])-12)-prior))
      if error > .22:
        continue
      score -= 4*error
    if best is None or score > best[0]:
      best = (score, selected)
  if best is None:
    return None
  selected = best[1]
  # Each quarter metre has equal fit weight, irrespective of source row density.
  counts = {b: int((bins[selected] == b).sum()) for b in np.unique(bins[selected])}
  weights = np.array([1/math.sqrt(counts[b]) for b in bins[selected]])
  slope, intercept = np.polyfit(x[selected]-12, y[selected], 1, w=weights)
  if abs(slope) > .075:
    return None
  selected = abs(y-intercept-slope*(x-12)) <= .075
  selected &= abs(width-np.median(width[selected])) <= .10
  if selected.sum() < 8 or len(np.unique(bins[selected]))*.25 < 2. or np.ptp(x[selected]) < 3.:
    return None
  colors = [c['color'] for c, ok in zip(found, selected, strict=True) if ok]
  color = max(('yellow', 'white'), key=colors.count) if colors else 'unknown'
  purity = colors.count(color)/max(1, len(colors))
  if purity < .9:
    return None
  return {'offset': float(intercept), 'slope': float(slope), 'width': float(np.median(width[selected])),
          'color': color, 'support_m': len(np.unique(bins[selected]))*.25, 'selected': selected}


def path_y(path, line, x):
  return np.interp(x, line['x'], line['y'])+path['offset']+path['slope']*(x-12)


class PaintTracker:
  """One actual left/right boundary, three seconds of spatial evidence."""
  def __init__(self, side):
    if side not in ('left', 'right'):
      raise ValueError('invalid side')
    self.side = side
    self.reset()

  def reset(self):
    self.previous = None
    self.reference = None
    self.origin = 0.
    self.bins = {}
    self.generation = getattr(self, 'generation', 0)+1

  def _observations(self, rgb, line, probability, rpy, k, matrix, gray):
    ribbon = collect_pixels(rgb, line, probability, rpy, k, end_m=20., half_width_m=.9, include_pixels=False)
    return candidates(rgb, ribbon, line, matrix, gray=gray)

  def _paths(self, found, prior, bootstrap, probability):
    first = fit_path(found, prior)
    if bootstrap and first and probability < .8 and first['support_m'] < 4.5:
      first = None
    if first is None:
      return [], 'no_consistent_target_path'
    remaining = [c for c, ok in zip(found, first['selected'], strict=True) if not ok]
    second = fit_path(remaining)
    paths = [first]
    if second and second['support_m'] >= min(3., .55*first['support_m']):
      separation = abs(second['offset']-first['offset'])
      if .12 <= separation <= .40 and abs(second['slope']-first['slope']) <= .02 and second['color'] == first['color']:
        paths.append(second)
      elif separation > .15 and second['support_m'] >= .85*first['support_m'] and bootstrap:
        return [], 'competing_boundaries'
    return paths, 'target_pixels_observed'

  def update(self, rgb, line, probability, rpy, k, stamp, *, pose=None, pose_error_m=None, road_pose=None, leads=(), lane_change=False, gray=None):
    empty = {'type': 'unknown', 'color': 'unknown', 'reason': 'insufficient_target_evidence', 'tracks': [], 'profile': [],
             'target_mask': np.zeros(rgb.shape[:2], bool), 'generation': self.generation, 'tracking_error_m': None,
             'distinct_m': 0., 'gap_m': 0., 'held': False, 'confidence_kind': 'uncalibrated_evidence', 'pose_qualified': False}
    motion_valid = (pose is not None and np.shape(pose) == (3,) and np.isfinite(pose).all() and
                    pose_error_m is not None and np.isfinite(pose_error_m) and 0 <= pose_error_m <= .18)
    if road_pose is not None:
      road_pose = np.asarray(road_pose, float)
      valid_road_pose = (road_pose.shape == (4, 4) and np.isfinite(road_pose).all() and
        np.allclose(road_pose[3], [0, 0, 0, 1], atol=1e-6) and
        np.allclose(road_pose[:3, :3].T @ road_pose[:3, :3], np.eye(3), atol=1e-6) and
        abs(np.linalg.det(road_pose[:3, :3])-1) < 1e-6)
      motion_valid = bool(motion_valid and valid_road_pose)
    if not isinstance(stamp, (int, np.integer)) or isinstance(stamp, bool) or stamp < 0:
      self.reset()
      return dict(empty, reason='invalid_camera_time', generation=self.generation)
    prev = self.previous
    if lane_change or not np.isfinite(probability) or probability < .5:
      self.reset()
      return dict(empty, reason='lane_change_or_boundary_lost', generation=self.generation)
    if prev and (not 0 < stamp-prev['stamp'] <= 300_000_000 or not motion_valid):
      self.reset()
      prev = None
    if prev and ((road_pose is None) != (prev.get('road_pose') is None)):
      self.reset()
      prev = None
    try:
      matrix = np.asarray(k) @ view_frame_from_device_frame @ rot_from_euler(rpy)
      gray, rows, found = self._observations(rgb, line, probability, rpy, k, matrix, gray)
    except (ValueError, KeyError, np.linalg.LinAlgError):
      self.reset()
      return dict(empty, reason='invalid_projection_or_geometry', generation=self.generation)
    prior = None
    if prev and motion_valid:
      if road_pose is not None and prev.get('world_path_3d') is not None:
        predicted = (prev['world_path_3d']-road_pose[:3, 3]) @ road_pose[:3, :3]
      else:
        predicted = to_local(prev['world_path'], pose)
      if np.any(np.diff(predicted[:, 0]) <= 0):
        self.reset()
        prev = None
      else:
        probes = np.array([8., 12., 16.])
        prior = np.interp(probes, predicted[:, 0], predicted[:, 1])-np.interp(probes, line['x'], line['y'])
    paths, selection_reason = self._paths(found, prior, prev is None, probability)
    if not paths:
      self.reset()
      return dict(empty, reason=selection_reason, generation=self.generation)
    # An ego-adjacent stripe is the primary member of a double boundary. Keep
    # the other stripe separate; its gaps must never fill the primary's gaps.
    paths.sort(key=lambda p: p['offset'], reverse=self.side == 'left')
    primary = paths[0]
    branch_distances = [c['x'] for c in found if c['color'] == primary['color'] and
      abs(c['lateral']-path_y(primary, line, c['x'])) < .20 and c['width'] > primary['width']*1.45]
    xs = np.arange(4.5, 20.001, .25)
    current_path = np.column_stack((xs, path_y(primary, line, xs)))
    world3d = world_road_points(current_path, line, road_pose) if motion_valid and road_pose is not None else None
    world = world3d[:, :2] if world3d is not None else to_world(current_path, pose) if motion_valid else None
    error = None
    if prev and motion_valid:
      _, distances = nearest_s(world, prev['world_path'])
      overlap = distances < .5
      error = float(np.median(distances[overlap])) if overlap.sum() >= 12 else None
      if error is None or error > .18:
        self.reset()
        prev = None
    mask = empty['target_mask']
    profile = []
    for row in rows:
      d, y = row['distance_m'], row['y']
      item = {'distance': d, 'ink': None, 'color': None, 'confidence': 0., 'reason': 'not_observed', 'y': y}
      item['geometry_visible'] = row['geometry_visible']
      profile.append(item)
      if not row['geometry_visible']:
        continue
      lateral = float(path_y(primary, line, d))
      point = np.array([[d, lateral]])
      px = project_road(point, line, matrix)[0]
      if self._occluded(point[0], leads):
        item['reason'] = 'model_lead_occlusion'
        continue
      if any(abs(d-x) < .5 for x in branch_distances):
        item['reason'] = 'branch_or_transverse_intersection'
        continue
      broad_here = any(lo-2 <= px[0] <= hi+2 for lo, hi in row['broad'])
      selected_primary = False
      for pi, path in enumerate(paths):
        expected = float(path_y(path, line, d))
        options = [found[j] for j in row['candidates'] if abs(found[j]['lateral']-expected) <= .075 and
                   found[j]['color'] == path['color'] and abs(found[j]['width']-path['width']) <= .10]
        if not options:
          continue
        c = min(options, key=lambda c: abs(c['lateral']-expected))
        # A candidate overlapping broad paint has unresolved target identity.
        if any(lo <= (c['lo']+c['hi'])/2 <= hi for lo, hi in row['broad']):
          continue
        mask[y, c['lo']:c['hi']+1] = True
        if pi == 0:
          selected_primary = True
          item.update(ink=1, color=c['color'], confidence=.95, reason='target_paint', distance=c['x'],
                      paint_x0=c['lo'], paint_x1=c['hi'])
          point = np.array([[c['x'], c['lateral']]])
      if not selected_primary:
        x = int(round(px[0]))
        width_px = max(2, int(round(np.linalg.norm(matrix[0])*primary['width']/max(d, 1)/2)))
        a, b = max(1, x-width_px-2), min(rgb.shape[1]-2, x+width_px+2)
        region = gray[y, a:b+1]
        exterior = np.r_[gray[y, max(0, a-5):a], gray[y, b+1:min(rgb.shape[1], b+6)]]
        if (not broad_here and len(region) >= 4 and len(exterior) >= 4 and 50 <= np.median(region) <= 165 and
            np.std(region) <= 14 and abs(float(np.median(region))-float(np.median(exterior))) <= 8):
          item.update(reason='unverified_exposed_road', gap_candidate=True)
        else:
          item['reason'] = 'transverse_shadow_or_unresolved'
      item['local_xy'] = point[0].tolist()
    # A tiny bright island inside a crosswalk is not a longitudinal marking.
    # Closing is used only to group adjacent original pixels for rejection;
    # none of its filled pixels enter the selected mask or paint/gap evidence.
    connected = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    for j in range(1, count):
      x, y, width, height, _ = stats[j]
      p = road_points(np.array([[x+width/2, y], [x+width/2, y+height-1]]), line, matrix)
      if height < 3 or abs(p[0, 0]-p[1, 0]) < .35:
        mask[labels == j] = False
    for item in profile:
      if item['ink'] == 1 and not mask[item['y'], item['paint_x0']:item['paint_x1']+1].any():
        item.update(ink=None, color=None, confidence=0., reason='isolated_marking_fragment')
    result = dict(empty, tracks=[{k: v for k, v in p.items() if k != 'selected'} for p in paths], profile=profile,
                  target_mask=mask, generation=self.generation, tracking_error_m=error, pose_qualified=motion_valid,
                  reason='target_pixels_observed')
    if not motion_valid:
      self.reset()
      return dict(result, generation=self.generation, reason='motion_unqualified_no_temporal_confirmation')
    if self.reference is None:
      self.reference = world.copy()
    else:
      # Extend only a locally connected world path; never change previous s bins.
      for point in world:
        tangent = self.reference[-1]-self.reference[-2]
        delta = point-self.reference[-1]
        if np.dot(delta, tangent) > 0 and .1 < np.linalg.norm(delta) <= .75:
          self.reference = np.vstack((self.reference, point))
    static_points = self._static_road_points(prev, gray, pose, line, matrix, road_pose) if prev else np.empty((0, 2))
    qualified = not getattr(self, 'require_physical_registration', False) or getattr(self, 'registration', {}).get('qualified', False)
    for key, item in list(self.bins.items()):
      if stamp-item['stamp'] > 3_000_000_000:
        del self.bins[key]
    observed = [r for r in profile if 'local_xy' in r] if qualified else []
    if observed:
      local_points = np.asarray([r['local_xy'] for r in observed])
      points = world_road_points(local_points, line, road_pose)[:, :2] if road_pose is not None else to_world(local_points, pose)
      ss, dd = nearest_s(points, self.reference, self.origin)
      flow_s, flow_d = nearest_s(static_points, self.reference, self.origin) if len(static_points) else (np.empty(0), np.empty(0))
      frame_bins = {}
      for row, s, delta in zip(observed, ss, dd, strict=True):
        if delta > .18:
          continue
        key = math.floor(s/BIN_M)
        static = bool(np.any((abs(flow_s-s) < .5) & (flow_d < .30)))
        entry = frame_bins.setdefault(key, {'ink': set(), 'color': set(), 'gap': True, 'static': False, 'near': False})
        if row['ink'] == 1:
          entry['ink'].add(1)
          entry['color'].add(row['color'])
          entry['near'] |= row['distance'] <= 12.
        entry['gap'] &= bool(row.get('gap_candidate', False))
        entry['static'] |= static
      for key, value in frame_bins.items():
        old = self.bins.get(key)
        ink = 1 if value['ink'] == {1} else None
        gap = value['gap'] and value['static']
        first_pose = old['first_pose'] if old else np.asarray(pose[:2]).copy()
        translated = np.linalg.norm(np.asarray(pose[:2])-first_pose) >= .5
        if ink is None and gap and old and old['gap_candidate'] and translated:
          ink = 0
        conflict_until = old.get('conflict_until', 0) if old else 0
        if old and old['ink'] is not None and ink is not None and old['ink'] != ink:
          conflict_until = stamp+3_000_000_000
        conflict = stamp < conflict_until
        if conflict:
          ink = None
        color = next(iter(value['color'])) if len(value['color']) == 1 else None
        if ink is None and old and old['ink'] is not None and not conflict:
          # UNKNOWN is neither a contradiction nor a fresh observation. Keep
          # its original evidence timestamp; do not erase/rejuvenate a gap.
          self.bins[key] = dict(old, gap_candidate=gap)
          continue
        self.bins[key] = {'ink': ink, 'color': color, 'stamp': stamp, 'gap_candidate': gap,
                          'first_pose': first_pose, 'near': value['near'], 'conflict_until': conflict_until}
      # Expose only *verified* physical gap states in this frame's profile.
      for row, s in zip(observed, ss, strict=True):
        if row.get('gap_candidate') and self.bins.get(math.floor(s/BIN_M), {}).get('ink') == 0:
          row.update(ink=0, confidence=.85, reason='translated_static_road_gap')
    result.update(self._classification(stamp))
    if not qualified:
      result.update(type='unknown', color='unknown', reason='physical_correspondence_unknown')
    if getattr(self, 'require_physical_registration', False):
      result.update(registration_qualified=qualified, registration=getattr(self, 'registration', {}))
    if prev is None:
      result['color'] = 'unknown'  # independent head still needs a second view
    if self.bins:
      ss = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.reference, axis=0), axis=1))]+self.origin
      keep = max(0, int(np.searchsorted(ss, min(self.bins)*BIN_M-1.))-1)
      if keep:
        self.origin = float(ss[keep])
        self.reference = self.reference[keep:]
    self.previous = {'stamp': stamp, 'pose': np.asarray(pose), 'world_path': world, 'gray': gray,
                     'road_pose': road_pose, 'world_path_3d': world3d,
                     'tracks': result['tracks'],
                     'line': line, 'matrix': matrix, 'mask': mask, 'gap_mask': self._gap_mask(profile, rgb.shape[:2], line, matrix)}
    return result

  @staticmethod
  def _occluded(point, leads):
    x, y = point
    for lead in leads:
      if lead['prob'] >= .5 and lead['x'] and lead['y']:
        if x >= lead['x'][0]-2.5 and abs(y-lead['y'][0]) < 1.25:
          return True
    return False

  @staticmethod
  def _gap_mask(profile, shape, line, matrix):
    mask = np.zeros(shape, np.uint8)
    for row in profile:
      if row.get('gap_candidate') and 'local_xy' in row:
        p = project_road(np.array([row['local_xy']]), line, matrix)[0]
        cv2.circle(mask, tuple(np.rint(p).astype(int)), 5, 255, -1)
    return mask

  @staticmethod
  def _static_road_points(prev, gray, pose, line, matrix, road_pose=None):
    # A bright paint endpoint at the edge of a gap is not an exposed-road
    # texture corner. It can also set the relative corner threshold so high
    # that white paint suppresses all asphalt features (yellow did not).
    road_mask = prev['gap_mask'].copy()
    near_paint = cv2.dilate(prev['mask'].astype(np.uint8), np.ones((15, 15), np.uint8))
    road_mask[near_paint != 0] = 0
    points = cv2.goodFeaturesToTrack(prev['gray'], maxCorners=600, qualityLevel=.01, minDistance=3, mask=road_mask)
    if points is None or not len(points):
      return np.empty((0, 2))
    local = road_points(points[:, 0], prev['line'], prev['matrix'])[:, :2]
    # Balance by physical distance. Near rows must not consume every corner.
    counts, selected = {}, []
    for i, point in enumerate(local):
      if not np.isfinite(point).all() or not 4.5 <= point[0] <= 20.:
        continue
      key = math.floor(point[0]/.75)
      counts[key] = counts.get(key, 0)+1
      if counts[key] <= 8:
        selected.append(i)
    if not selected:
      return np.empty((0, 2))
    points, local = points[selected], local[selected]
    if road_pose is not None and prev.get('road_pose') is not None:
      xyz = road_points(points[:, 0], prev['line'], prev['matrix'])
      world3d = xyz @ prev['road_pose'][:3, :3].T+prev['road_pose'][:3, 3]
      predicted, now = project_static_points(world3d, road_pose, matrix)
      world = world3d[:, :2]
    else:
      world = to_world(local, prev['pose'])
      now = to_local(world, pose)
      predicted = project_road(now, line, matrix)
    valid = np.isfinite(predicted).all(1) & (now[:, 0] >= 4.5) & (now[:, 0] <= 20.)
    valid &= (predicted[:, 0] >= 8) & (predicted[:, 0] < gray.shape[1]-8)
    valid &= (predicted[:, 1] >= 8) & (predicted[:, 1] < gray.shape[0]-8)
    points, world, predicted = points[valid], world[valid], predicted[valid]
    if not len(points):
      return np.empty((0, 2))
    # At 5Hz a near-road corner may move >100px. Starting LK at its old pixel
    # can converge on another asphalt patch even with a forward/backward check.
    # Initialising the search does not change the independent .8px/2px gates.
    forward, status, _ = cv2.calcOpticalFlowPyrLK(prev['gray'], gray, points,
      predicted.astype(np.float32).reshape(-1, 1, 2), winSize=(15, 15), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev['gray'], forward,
      points.copy(), winSize=(15, 15), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    good = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
    good &= np.linalg.norm(backward[:, 0]-points[:, 0], axis=1) <= .8
    good &= np.linalg.norm(forward[:, 0]-predicted, axis=1) <= 2.
    good &= np.isfinite(world).all(1)
    return world[good]

  def _classification(self, stamp):
    keys = sorted(k for k, v in self.bins.items() if v['ink'] is not None)
    result = {'type': 'unknown', 'color': 'unknown', 'distinct_m': len(keys)*BIN_M,
              'gap_m': sum(v['ink'] == 0 for v in self.bins.values())*BIN_M}
    if not keys:
      return result
    runs = []
    for key in range(keys[0], keys[-1]+1):
      value = self.bins.get(key, {}).get('ink')
      if runs and runs[-1][0] == value:
        runs[-1][2] = key
      else:
        runs.append([value, key, key])
    paints = [r for r in runs if r[0] == 1 and (r[2]-r[1]+1)*BIN_M >= .75]
    # UNKNOWN edge bins do not destroy already verified road evidence. They
    # also do not contribute to gap length. Fragmented paint with no verified
    # intervening road belongs to the same group, never a new dashed stroke.
    groups, gaps = [], []
    for paint in paints:
      if not groups:
        groups.append(paint.copy())
        continue
      a, b = groups[-1][2]+1, paint[1]
      gap_length = sum(self.bins.get(j, {}).get('ink') == 0 for j in range(a, b))*BIN_M
      if gap_length >= 1.25 and (b-a)*BIN_M <= 10:
        gaps.append(gap_length)
        groups.append(paint.copy())
      else:
        groups[-1][2] = paint[2]
    fresh = any(v['ink'] is not None and stamp-v['stamp'] <= 500_000_000 for v in self.bins.values())
    if fresh and len(groups) >= 3 and len(gaps) >= 2 and result['distinct_m'] >= 18:
      result['type'] = 'dashed'
    if fresh and result['distinct_m'] >= 22 and result['gap_m'] == 0 and len(keys)/(keys[-1]-keys[0]+1) >= .98:
      result['type'] = 'solid'
    near = [v['color'] for v in self.bins.values() if v['ink'] == 1 and v['near'] and v['color'] and stamp-v['stamp'] <= 500_000_000]
    if len(near) >= 3:
      dominance = max(near.count('yellow'), near.count('white'))/len(near)
      if dominance >= .97:
        result['color'] = 'yellow' if near.count('yellow') > near.count('white') else 'white'
    result.update(paint_runs=len(groups), verified_gap_runs=len(gaps), verified_gap_lengths_m=gaps)
    return result
