"""No detector or learned weights: paint masks and longitudinal continuity."""
import numpy as np

from openpilot.selfdrive.carrot.lane_type.result import unknown
from openpilot.selfdrive.carrot.lane_type.roi import NEAR_M, FAR_M


def classify_strip(rgb, near=NEAR_M, far=FAR_M):
  import cv2
  if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16 or rgb.dtype != np.uint8:
    raise ValueError('expected an RGB lane strip')
  if not np.isfinite(near+far) or not 0 < near < far:
    raise ValueError('invalid observed distance range')
  hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  h, w = gray.shape
  # Road brightness varies by distance and shadow: compare paint to row edges.
  road = np.median(np.concatenate((gray[:, :w//6], gray[:, -w//6:]), axis=1), axis=1)
  contrast = gray.astype(np.float32)-road[:, None]
  yellow = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 65) & (hsv[..., 2] >= 60))
  white = ((hsv[..., 1] <= 80) & (hsv[..., 2] >= 130) & (contrast >= 20))
  # Ignore neighbouring stripes at the edges of the 0.9m strip.
  center = slice(w//3, 2*w//3)
  yellow_count, white_count = int(yellow[:, center].sum()), int(white[:, center].sum())
  total = yellow_count+white_count
  if total < h*2:
    return unknown('no_paint_evidence')
  color = 'yellow' if yellow_count > white_count else 'white'
  color_score = max(yellow_count, white_count)/total
  paint = yellow if color == 'yellow' else white
  occupied = (paint[:, center].sum(axis=1) >= 2).astype(np.uint8)
  occupied = cv2.morphologyEx(occupied[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).ravel().astype(bool)
  changes = np.r_[0, np.flatnonzero(occupied[1:] != occupied[:-1])+1, h]
  runs = [(bool(occupied[a]), (b-a)*(far-near)/h) for a, b in zip(changes[:-1], changes[1:], strict=True)]
  gaps = [length for ink, length in runs if not ink]
  segments = sum(ink and length >= .6 for ink, length in runs)
  coverage = float(occupied.mean())
  longest_gap = max(gaps, default=0)
  internal_gaps = [length for i, (ink, length) in enumerate(runs) if not ink and 0 < i < len(runs)-1 and
                   runs[i-1][0] and runs[i+1][0] and length >= 1.5]
  if coverage >= .80 and longest_gap <= 2.0:
    line_type, type_score = 'solid', min(.99, .85+.14*(coverage-.8)/.2)
  elif segments >= 2 and longest_gap >= 1.5 and .12 <= coverage <= .78:
    line_type, type_score = 'dashed', .90
  else:
    line_type, type_score = 'unknown', 0.0
  return {'type': line_type, 'color': color if color_score >= .85 else 'unknown',
              'typeConfidence': type_score, 'colorConfidence': color_score,
              'reason': 'mask_continuity', 'coverage': coverage, 'longestGapM': longest_gap,
              'paintSegments': int(segments), 'internalGapCount': len(internal_gaps),
              'repeatingPatternObserved': bool(line_type == 'dashed' and segments >= 3 and len(internal_gaps) >= 2),
              'confidenceKind': 'heuristic_evidence_not_calibrated_probability'}
