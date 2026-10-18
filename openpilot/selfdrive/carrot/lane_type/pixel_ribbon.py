"""Offline lossless boundary pixel collection; deliberately no classifier/warp.

Preserve every integer source pixel in a projected ribbon, including its source
coordinate and distance. A ragged array is not a rectangular neural input.
Missing, occluded and ambiguous observations never establish a paint gap.
Not connected to the driving daemon, display or CAN.
"""
import numpy as np

from openpilot.selfdrive.carrot.lane_type.dual_roi import project, visible_rows


def collect_pixels(rgb, line, probability, rpy, intrinsics, leads=(), end_m=20., half_width_m=.45, include_pixels=True):
  if not 4.5 < end_m <= 25.:
    raise ValueError('range must begin at 4.5m and end no farther than 25m')
  if not .1 <= half_width_m <= 1.:
    raise ValueError('invalid boundary search width')
  empty = {'rgb': np.empty((0, 3), np.uint8), 'xy': np.empty((0, 2), np.int32),
           'distance_m': np.empty(0), 'row_offsets': np.array([0]), 'rows': [],
           'classification': 'unknown', 'gap_evidence': False, 'padding_used': False}
  if not np.isfinite(probability) or probability < .5:
    return empty
  distances = np.arange(4.5, end_m, .025)
  distances = np.r_[distances, end_m]
  maps = project(line, rpy, intrinsics, distances, columns=65, half_width=half_width_m)
  center_y = maps[1][:, 32]
  if np.any(np.diff(center_y) >= 0):
    raise ValueError('cannot invert a nonmonotonic boundary projection')
  clear = visible_rows(maps, rgb.shape, leads, line, rpy, intrinsics)
  # Invert geometry only. Never interpolate image intensities or shift each row
  # to a newly detected bright spot. Keep every pixel, with its original x/y.
  ys = np.arange(max(0, int(np.ceil(center_y[-1]))),
                 min(rgb.shape[0]-1, int(np.floor(center_y[0])))+1)
  ds = np.interp(ys, center_y[::-1], distances[::-1])
  left = np.interp(ds, distances, maps[0][:, 0])
  right = np.interp(ds, distances, maps[0][:, -1])
  valid = np.interp(ds, distances, clear.astype(float)) > .999
  pixels, coords, depths, rows, offsets = [], [], [], [], [0]
  for y, d, lo, hi, ok in zip(ys, ds, left, right, valid, strict=True):
    a, b = int(np.ceil(lo)), int(np.floor(hi))
    observable = bool(ok and a >= 0 and b < rgb.shape[1] and b >= a)
    rows.append({'y': int(y), 'distance_m': float(d), 'left_x': a, 'right_x': b,
                 'geometry_visible': observable, 'paint_state': 'unknown',
                 'visibility_semantically_verified': False})
    if observable and include_pixels:
      xs = np.arange(a, b+1)
      pixels.append(rgb[y, xs].copy())
      coords.append(np.column_stack((xs, np.full(len(xs), y))))
      depths.extend([d]*len(xs))
      offsets.append(offsets[-1]+len(xs))
    else:
      offsets.append(offsets[-1])
  if not pixels:
    return dict(empty, rows=rows, row_offsets=np.asarray(offsets))
  return dict(empty, rgb=np.concatenate(pixels), xy=np.concatenate(coords).astype(np.int32),
              distance_m=np.asarray(depths), row_offsets=np.asarray(offsets), rows=rows)


def source_box(rgb, ribbon):
  """Unresized context plus a separate inclusion mask; no black replacement."""
  xy = ribbon['xy']
  if not len(xy):
    return None
  a, b = xy.min(0), xy.max(0)+1
  image = rgb[a[1]:b[1], a[0]:b[0]].copy()
  mask = np.zeros(image.shape[:2], bool)
  mask[xy[:, 1]-a[1], xy[:, 0]-a[0]] = True
  return {'image': image, 'ribbon_mask': mask, 'origin_xy': a}
