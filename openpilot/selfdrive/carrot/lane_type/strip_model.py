"""Frozen StripNet on the 4.5-20 m row-aligned YCbCr strip (display-only lane marking recogniser).

The strip is built exactly as in training (tools/lane_profile/build_dataset.py): profile_scanner geometry
and sampling, uint8, rows area-resized to 128. Inference is OpenCV DNN on the CPU (one thread), the same
backend as the existing lane classifier. Below the confidence threshold a head is UNKNOWN; dashed+yellow is
not a learned class and is UNKNOWN for both heads.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.carrot.lane_type.profile_scanner import sample, strip_geometry

ASSETS = Path(__file__).with_name('assets')
ROWS, COLUMNS = 128, 96
TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')


def strip_input(y_plane, uv_plane, line, rpy, k):
  """3 x 128 x 96 float32 raw YCbCr, or None when the boundary geometry is unavailable."""
  import cv2
  geo = strip_geometry(line, rpy, k, y_plane.shape[1], y_plane.shape[0])
  if geo is None:
    return None
  lum, cb, cr = sample(y_plane, uv_plane, geo)
  ycc = np.clip(np.stack((lum, cb, cr), -1), 0, 255).astype(np.uint8)
  ycc = cv2.resize(ycc, (COLUMNS, ROWS), interpolation=cv2.INTER_AREA if len(ycc) > ROWS else cv2.INTER_LINEAR)
  return ycc.transpose(2, 0, 1).astype(np.float32)


class StripModel:
  def __init__(self, directory=ASSETS, threshold=None):
    import cv2
    directory = Path(directory)
    manifest = json.loads((directory/'lane_strip.json').read_text(encoding='utf-8'))
    data = (directory/'lane_strip.onnx').read_bytes()
    if (hashlib.sha256(data).hexdigest() != manifest['model_sha256'] or manifest['input']['shape'] != [3, ROWS, COLUMNS] or
        manifest['outputs'] != {'line_type': list(TYPES), 'color': list(COLORS)}):
      raise ValueError('lane strip model artifact contract mismatch')
    self.threshold = float(manifest['threshold'] if threshold is None else threshold)
    self.model_hash = manifest['model_sha256']
    self.net = cv2.dnn.readNetFromONNX(np.frombuffer(data, dtype=np.uint8))
    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

  def infer(self, strips):
    """strips: list of 3x128x96 float32 -> list of result dicts (same keys as result.unknown())."""
    if not strips:
      return []
    self.net.setInput(np.ascontiguousarray(np.stack(strips)))
    types, colors = self.net.forward(['line_type', 'color'])
    return [self.decide(t, c) for t, c in zip(types, colors, strict=True)]

  def decide(self, type_p, color_p):
    out = {'reason': 'strip_model'}
    for head, p, classes in (('type', type_p, TYPES), ('color', color_p, COLORS)):
      i = int(np.argmax(p))
      confident = float(p[i]) >= self.threshold
      out[head] = classes[i] if confident else 'unknown'
      out[head + 'Confidence'] = float(p[i])
      out[head + 'Candidate'] = classes[i]
    if out['type'] == 'dashed' and out['color'] == 'yellow':
      out.update(type='unknown', color='unknown', reason='yellow_dashed_not_learned')
    return out
