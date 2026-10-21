"""Frozen StripNet on the 4.5-20 m row-aligned YCbCr strip (display-only lane marking recogniser).

The strip is built exactly as in training (tools/lane_profile/build_dataset.py): profile_scanner geometry
and sampling, uint8, rows area-resized to 128. Its lateral width comes from the manifest's optional ``strip``
({half_m, cols, rows}; absent = +-0.75 m, 96 columns, 128 rows = v1-v3); the ONNX input shape must match it.
Inference is OpenCV DNN on the CPU (one thread), the same backend as the existing lane classifier.
Below the confidence threshold a head is UNKNOWN; dashed+yellow is not a learned class and is UNKNOWN for both heads.

Manifest v4 (``outputs`` == OUTPUTS_V4, ONNX outputs ``color`` (N, 3) and ``type`` (N, 5) logits) is mapped onto the
same two-class result the strip filter reads (StripModel.decide4); v1-v3 manifests (``outputs`` {line_type, color},
two softmax heads) keep StripModel.decide unchanged. ``mirror_right`` (v4): marking_daemon flips the right strip's
columns before inference so the ego side is on the same column side as on the left strip (see MIRROR_AXIS).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.carrot.lane_type.profile_scanner import HALF_M, sample, strip_geometry

ASSETS = Path(__file__).with_name('assets')
ROWS, COLUMNS = 128, 96
DEFAULT_STRIP = {'half_m': HALF_M, 'cols': COLUMNS, 'rows': ROWS}  # v1-v3 (manifest without 'strip')
TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')
OUTPUTS_V3 = {'line_type': list(TYPES), 'color': list(COLORS)}
# v4 contract (shared with tools/lane_profile training): near = the ego-side line of a double marking.
TYPES_V4, COLORS_V4 = ('dashed', 'solid', 'dbl_near', 'dbl_far', 'unknown'), ('white', 'yellow', 'blue')
OUTPUTS_V4 = {'color': list(COLORS_V4), 'type': list(TYPES_V4)}
# profile_scanner.strip_geometry: column 0 = line y - half_m, the last column = y + half_m (image left -> right), so the
# ego side is the HIGH columns of the left strip and the LOW columns of the right strip. With mirror_right the right
# strip is flipped on its column axis (the last one of 3 x rows x cols), putting its ego side on the high columns too.
MIRROR_AXIS = -1


def strip_of(manifest):
  """Strip geometry {half_m, cols, rows} of a manifest; a missing 'strip' (or key) keeps the v1-v3 default."""
  given = manifest.get('strip') or {}
  if not isinstance(given, dict) or set(given) - set(DEFAULT_STRIP):
    raise ValueError(f'lane strip manifest: invalid strip {given!r}')
  out = {**DEFAULT_STRIP, **given}
  half_m, cols, rows = out['half_m'], out['cols'], out['rows']
  if (isinstance(half_m, bool) or not isinstance(half_m, (int, float)) or not 0 < half_m <= 3 or
      any(isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 1024 for v in (cols, rows))):
    raise ValueError(f'lane strip manifest: invalid strip {given!r}')
  return {'half_m': float(half_m), 'cols': cols, 'rows': rows}


def _fields(buf):
  """(field number, wire type, value) of one protobuf message (value: int or bytes)."""
  i, n = 0, len(buf)
  def varint():
    nonlocal i
    v = shift = 0
    while True:
      b = buf[i]
      i += 1
      v |= (b & 0x7f) << shift
      shift += 7
      if b < 0x80:
        return v
  while i < n:
    key = varint()
    field, wire = key >> 3, key & 7
    if wire == 0:
      yield field, wire, varint()
    elif wire == 2:
      size = varint()
      yield field, wire, buf[i:i + size]
      i += size
    elif wire in (1, 5):
      i += 8 if wire == 1 else 4
    else:
      raise ValueError('lane strip model: unreadable ONNX')


def onnx_input_shape(data, name, field=11):
  """Declared dims of graph input ``name`` (None for a symbolic dim), or None when the input is absent.
  field=12: a graph output instead (GraphProto.output)."""
  data = memoryview(data)
  for f, w, graph in _fields(data):
    if f != 7 or w != 2:  # ModelProto.graph
      continue
    for g, gw, vi in _fields(graph):
      if g != field or gw != 2:  # GraphProto.input (11) / output (12)
        continue
      fields = list(_fields(vi))
      if bytes(next((v for k, kw, v in fields if k == 1 and kw == 2), b'')).decode() != name:
        continue
      dims = []
      for k, kw, tp in fields:
        if k == 2 and kw == 2:  # ValueInfoProto.type -> TypeProto.tensor_type -> shape -> dim
          for t, tw, tensor in _fields(tp):
            for s, sw, shape in (_fields(tensor) if t == 1 and tw == 2 else ()):
              for d, dw, dim in (_fields(shape) if s == 2 and sw == 2 else ()):
                if d == 1 and dw == 2:
                  dims.append(next((int(v) for x, xw, v in _fields(dim) if x == 1 and xw == 0), None))
      return dims
  return None


def onnx_output_shape(data, name):
  """Declared dims of graph output ``name`` (None for a symbolic dim), or None when the output is absent."""
  return onnx_input_shape(data, name, field=12)


def softmax(x):
  e = np.exp(np.asarray(x, np.float64) - np.max(x))
  return e/e.sum()


def strip_input(y_plane, uv_plane, line, rpy, k, half_m=HALF_M, cols=COLUMNS, rows=ROWS):
  """3 x rows x cols float32 raw YCbCr (default 3 x 128 x 96 over +-0.75 m), or None when the boundary geometry is
  unavailable. half_m / cols / rows: the model manifest's strip (StripModel.strip)."""
  import cv2
  geo = strip_geometry(line, rpy, k, y_plane.shape[1], y_plane.shape[0], half_m=half_m, cols=cols)
  if geo is None:
    return None
  lum, cb, cr = sample(y_plane, uv_plane, geo)
  ycc = np.clip(np.stack((lum, cb, cr), -1), 0, 255).astype(np.uint8)
  ycc = cv2.resize(ycc, (cols, rows), interpolation=cv2.INTER_AREA if len(ycc) > rows else cv2.INTER_LINEAR)
  return ycc.transpose(2, 0, 1).astype(np.float32)


class StripModel:
  def __init__(self, directory=ASSETS, threshold=None):
    import cv2
    directory = Path(directory)
    manifest = json.loads((directory/'lane_strip.json').read_text(encoding='utf-8'))
    data = (directory/'lane_strip.onnx').read_bytes()
    outputs = manifest.get('outputs')
    if hashlib.sha256(data).hexdigest() != manifest['model_sha256'] or outputs not in (OUTPUTS_V3, OUTPUTS_V4):
      raise ValueError('lane strip model artifact contract mismatch')
    self.v4 = outputs == OUTPUTS_V4
    self.mirror_right = manifest.get('mirror_right', False)
    if not isinstance(self.mirror_right, bool) or (self.mirror_right and not self.v4):
      raise ValueError(f"lane strip manifest: mirror_right {self.mirror_right!r} needs a bool and the v4 outputs")
    if self.v4:
      for head, classes in outputs.items():
        dims = onnx_output_shape(data, head)
        if dims is None or len(dims) != 2 or dims[1] != len(classes):
          expects = f'[N, {len(classes)}] for {classes}'
          raise ValueError(f"lane strip ONNX output {head!r} shape {dims} does not match the manifest v4 outputs (expects {expects})")
    self.heads = ['type', 'color'] if self.v4 else ['line_type', 'color']
    self.strip = strip_of(manifest)
    want = [3, self.strip['rows'], self.strip['cols']]
    geometry = f"strip +-{self.strip['half_m']:g} m / {self.strip['cols']} cols / {self.strip['rows']} rows"
    if manifest['input']['shape'] != want:
      raise ValueError(f"lane strip manifest input shape {manifest['input']['shape']} does not match its {geometry} (expects {want})")
    dims = onnx_input_shape(data, manifest['input']['name'])
    if dims is None or len(dims) != 4 or dims[1:] != want:
      expects = ', '.join(map(str, want))
      raise ValueError(f"lane strip ONNX input {manifest['input']['name']!r} shape {dims} does not match the manifest {geometry} (expects [N, {expects}])")
    self.threshold = float(manifest['threshold'] if threshold is None else threshold)
    self.model_hash = manifest['model_sha256']
    self.net = cv2.dnn.readNetFromONNX(np.frombuffer(data, dtype=np.uint8))
    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

  def infer(self, strips):
    """strips: list of 3 x rows x cols float32 (self.strip) -> list of result dicts (same keys as result.unknown())."""
    if not strips:
      return []
    self.net.setInput(np.ascontiguousarray(np.stack(strips)))
    types, colors = self.net.forward(self.heads)
    decide = self.decide4 if self.v4 else self.decide
    return [decide(t, c) for t, c in zip(types, colors, strict=True)]

  def decide(self, type_p, color_p, yellow_dashed_learned=False):
    """yellow_dashed_learned (v4): dashed+yellow is a legal Korean marking (centre-line passing zone, roadside parking line)
    and is passed through; v1-v3 (default) report it as unknown."""
    out = {'reason': 'strip_model'}
    for head, p, classes in (('type', type_p, TYPES), ('color', color_p, COLORS)):
      i = int(np.argmax(p))
      confident = float(p[i]) >= self.threshold
      out[head] = classes[i] if confident else 'unknown'
      out[head + 'Confidence'] = float(p[i])
      out[head + 'Candidate'] = classes[i]
    if out['type'] == 'dashed' and out['color'] == 'yellow' and not yellow_dashed_learned:
      out.update(type='unknown', color='unknown', reason='yellow_dashed_not_learned')
    return out

  def decide4(self, type_logits, color_logits):
    """v4 logits -> the two-class result decide() makes, plus the v4 extras (typeClass, colorClass, double, blue, pUnknown).

    Type: an 'unknown' argmax abstains (reason 'unknown': no probabilities, both heads unknown; marking_daemon passes it like
    'lane_prob_dip'); otherwise P(dashed) = (P(dashed) + P(dbl_near)) / (1 - P(unknown)) (near = ego side -> dashed display,
    far -> solid). Colour: P(yellow) of the 3-way softmax; blue is shown as white (P(white) = 1 - P(yellow)). Yellow + dashed
    is passed through (type dashed, colour yellow, reason strip_model), unlike v1-v3."""
    pt, pc = softmax(type_logits), softmax(color_logits)
    t, c = int(np.argmax(pt)), int(np.argmax(pc))
    u = float(pt[TYPES_V4.index('unknown')])
    extras = {'typeClass': TYPES_V4[t], 'colorClass': COLORS_V4[c], 'double': TYPES_V4[t] in ('dbl_near', 'dbl_far'),
              'blue': COLORS_V4[c] == 'blue', 'pUnknown': u}
    if TYPES_V4[t] == 'unknown':
      return {'reason': 'unknown', 'type': 'unknown', 'typeConfidence': 0., 'typeCandidate': 'unknown',
              'color': 'unknown', 'colorConfidence': 0., 'colorCandidate': 'unknown', **extras}
    dashed = min(1., float(pt[0] + pt[2])/(1. - u))
    yellow = float(pc[1])
    out = self.decide(np.array([dashed, 1. - dashed]), np.array([1. - yellow, yellow]), yellow_dashed_learned=True)
    out.update(extras)
    return out
