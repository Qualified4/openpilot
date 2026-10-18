"""Conservative visibility gating and bounded, lane-local display memory."""
import math

import numpy as np

from openpilot.selfdrive.carrot.lane_type.result import unknown
from openpilot.selfdrive.carrot.lane_type.roi import FAR_M, NEAR_M, ROI_HEIGHT

MIN_OBSERVED_M = 32.0
MAX_OCCLUDED_FRACTION = 0.15
HOLD_NS = 500_000_000


def visibility(maps, model, line, rpy, intrinsics):
  """Mask conservative lead silhouettes using *existing* model leads only.

  These are geometric occlusion bounds, not a new detector or ground-truth
  visibility segmentation. Unknown/unmodelled occluders remain a limitation.
  """
  from openpilot.common.transformations.camera import view_frame_from_device_frame
  from openpilot.common.transformations.orientation import rot_from_euler
  mx, my = maps
  clear = np.ones(mx.shape, dtype=bool)
  projection = np.asarray(intrinsics) @ view_frame_from_device_frame @ rot_from_euler(rpy)
  for lead in getattr(model, 'leadsV3', ()):
    if lead.prob < .5 or not len(lead.x) or not len(lead.y):
      continue
    x, y = float(lead.x[0]), float(lead.y[0])
    if not math.isfinite(x+y) or not 2 < x < 80:
      continue
    z = float(np.interp(x, line.x, line.z))
    corners = np.array([[x, y+dy, z+dz] for dy in (-1.3, 1.3) for dz in (-2.0, .2)]) @ projection.T
    if (corners[:, 2] <= 0).any():
      continue
    pixels = corners[:, :2]/corners[:, 2, None]
    low, high = pixels.min(0), pixels.max(0)
    clear &= ~((mx >= low[0]) & (mx <= high[0]) & (my >= low[1]) & (my <= high[1]))
  # A line's centre must be visible, not merely some road at the strip edge.
  row_clear = clear[:, mx.shape[1]//3:2*mx.shape[1]//3].mean(1) >= .9
  length = float(row_clear.sum()*(FAR_M-NEAR_M)/ROI_HEIGHT)
  occluded = float(1-row_clear.mean())
  return clear, {'observedLengthM': length, 'occludedFraction': occluded,
                    'sufficient': length >= MIN_OBSERVED_M and occluded <= MAX_OCCLUDED_FRACTION,
                    'visibilitySource': 'model_lead_silhouette_geometric_bound'}


class ObservationGuard:
  """No distance accumulation from repeated stationary views or lane indices.

  Image stitching was deliberately not adopted: sparse model lane identities
  cannot prove that longitudinal paint from different frames is the same line.
  Only fresh, sufficiently long observations can create/refresh a result.
  """
  # ponytail: no spatial stitching until physical lane identity and odometry alignment are validated.
  def __init__(self):
    self.previous = {}

  def reset(self):
    self.previous.clear()

  def update(self, side, result, observation, boundary, *, now_ns, camera_ns, lane_change=False, tracked=True):
    if (lane_change or not tracked or boundary is None or len(boundary) != 2 or
        not all(math.isfinite(x) for x in boundary) or not 0 <= now_ns-camera_ns <= HOLD_NS):
      self.previous.pop(side, None)
      return dict(unknown('lane_change' if lane_change else 'tracking_lost_or_stale'), trackingReset=True)
    previous = self.previous.get(side)
    tracking_reset = False
    if previous and (now_ns < previous['stamp'] or max(abs(a-b) for a, b in zip(boundary, previous['boundary'], strict=True)) > .35):
      self.previous.pop(side, None)
      previous = None
      tracking_reset = True
    sufficient = (observation.get('sufficient') is True and
                  observation.get('observedLengthM', 0) >= MIN_OBSERVED_M and
                  0 <= observation.get('occludedFraction', 1) <= MAX_OCCLUDED_FRACTION)
    previous = previous or {'stamp': camera_ns, 'boundary': tuple(boundary), 'heads': {}}
    output = unknown('insufficient_observation' if not sufficient else result['reason'])
    held = False
    for head, classes in [('type', ('dashed', 'solid')), ('color', ('white', 'yellow'))]:
      if sufficient and result[head] in classes and result[head+'Confidence'] >= .85:
        previous['heads'][head] = (result[head], result[head+'Confidence'], camera_ns)
        previous['stamp'], previous['boundary'] = camera_ns, tuple(boundary)
      old = previous['heads'].get(head)
      if old and 0 <= now_ns-old[2] <= HOLD_NS:
        output[head], output[head+'Confidence'] = old[:2]
        output[head+'MonoTimeNanos'] = old[2]
        held |= old[2] != camera_ns
      else:
        previous['heads'].pop(head, None)
    if previous['heads']:
      self.previous[side] = previous
    else:
      self.previous.pop(side, None)
    if held:
      output['reason'] = 'insufficient_observation_hold'
    elif sufficient:
      output['reason'] = result['reason']
    return dict(output, held=held, trackingReset=tracking_reset, **observation)
