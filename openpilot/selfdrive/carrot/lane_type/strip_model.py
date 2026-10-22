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

Model slots (SLOTS, selected by the LaneStripModel param in marking_daemon, see load_slot): 0 = assets/lane_strip.{onnx,json}
(the deployed model), 1 = assets/lane_strip_v4.{onnx,json} (the new model), HYBRID_SLOT 2 = both: HybridStripModel takes the
TYPE from slot 1 (v4) and the COLOUR from slot 0 (v3). Slots 1 and 2 fall back to slot 0 when their files are missing or invalid.
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
SLOTS = {0: 'lane_strip', 1: 'lane_strip_v4'}  # file stem in ASSETS per LaneStripModel value
HYBRID_SLOT = 2  # LaneStripModel 2: type from SLOTS[1] (v4), colour from SLOTS[0] (v3) (HybridStripModel)


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


_DEFAULT_STRIP_INPUT = strip_input


def batch_strip_inputs(y_plane, uv_plane, lines, rpy, k, geometries, extractor):
  """Width-major strips. Preserve the scalar sampler hook used by cached replays and alternate samplers."""
  if strip_input is not _DEFAULT_STRIP_INPUT:
    groups = []
    for line in lines:
      group = []
      for geometry in geometries:
        value = strip_input(y_plane, uv_plane, line, rpy, k, **geometry)
        if value is None:
          group = [None]*len(geometries)
          break
        group.append(value)
      groups.append(group)
    return [group[j] for j in range(len(geometries)) for group in groups]
  widths = sorted({(g.get('half_m', HALF_M), g.get('cols', COLUMNS), g.get('rows', ROWS)) for g in geometries})
  strips = extractor.extract(y_plane, uv_plane, lines, rpy, k, widths)
  return [strips[widths.index((g.get('half_m', HALF_M), g.get('cols', COLUMNS), g.get('rows', ROWS)))*len(lines) + i]
          for g in geometries for i in range(len(lines))]


class StripModel:
  def __init__(self, directory=ASSETS, threshold=None, name='lane_strip'):
    """``name``: the file stem of the manifest / ONNX pair in ``directory`` (SLOTS)."""
    import cv2
    directory = Path(directory)
    manifest = json.loads((directory/f'{name}.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):  # valid JSON but not an object ([], null, 3): a contract error, so load_slot falls back
      raise ValueError(f'lane strip manifest {name}.json: top level is {type(manifest).__name__}, expected an object')
    data = (directory/f'{name}.onnx').read_bytes()
    self.name, self.version, self.slot = name, manifest.get('version'), next((s for s, n in SLOTS.items() if n == name), None)
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

  def info(self):
    """What the daemon logs and publishes for the active model (packet 'model')."""
    return {'slot': self.slot, 'name': self.name, 'version': self.version, 'sha256': self.model_hash[:12], 'v4': self.v4}

  def forward(self, strips):
    """The network's (type, colour) output rows for a non-empty list of strips (v4: logits, v1-v3: softmax probabilities)."""
    self.net.setInput(np.ascontiguousarray(np.stack(strips)))
    return self.net.forward(self.heads)

  def infer(self, strips):
    """strips: list of 3 x rows x cols float32 (self.strip) -> list of result dicts (same keys as result.unknown())."""
    if not strips:
      return []
    types, colors = self.forward(strips)
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

  def decide4(self, type_logits, color_logits, yellow=None):
    """v4 logits -> the two-class result decide() makes, plus the v4 extras (typeClass, colorClass, double, blue, pUnknown).
    ``yellow`` (HybridStripModel): P(yellow) from another model, used instead of this model's colour head.

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
    yellow = float(pc[1]) if yellow is None else float(yellow)
    out = self.decide(np.array([dashed, 1. - dashed]), np.array([1. - yellow, yellow]), yellow_dashed_learned=True)
    out.update(extras)
    return out


class HybridStripModel:
  """LaneStripModel 2: TYPE from a v4 model (StripNet2, its own strip and mirror_right), COLOUR from a v1-v3 model (its own
  strip, never mirrored). Both run on every inferred side (marking_daemon.Recogniser builds one strip per entry of
  ``strip_inputs``; a side without a valid strip in either geometry is not inferred: invalid_lane_geometry).

  Per side the v4 decision comes first (StripModel.decide4): a type argmax 'unknown' abstains on both heads (reason 'unknown',
  paused like lane_prob_dip). Otherwise P(dashed) = (dashed + dbl_near)/(1 - unknown) from v4 (dbl_near -> dashed, dbl_far ->
  solid), P(yellow) = the v3 colour softmax, both through decide4's two-class rule at the v4 threshold (yellow + dashed passed
  through). The v4 colour head is ignored: colorClass = the v3 argmax (white / yellow), blue always False; typeClass, double,
  pUnknown from v4. Same mapping as the replay reference .analysis/lane-hybrid-20261022/tools/hybrid.py."""
  slot, name, v4 = HYBRID_SLOT, 'hybrid', True

  def __init__(self, type_model, color_model):
    if not type_model.v4 or color_model.v4:
      raise ValueError('lane strip hybrid: needs a v4 (StripNet2) type model and a v1-v3 colour model')
    self.type_model, self.color_model = type_model, color_model
    self.threshold = type_model.threshold
    # (strip geometry, mirror the right strip) per model, in the order of the strips infer() is given per side
    self.strip_inputs = ((type_model.strip, type_model.mirror_right), (color_model.strip, False))

  def info(self):
    t, c = self.type_model.info(), self.color_model.info()
    return {'slot': self.slot, 'name': self.name, 'version': f"{t['version']}+{c['version']}", 'sha256': f"{t['sha256']}+{c['sha256']}",
            'v4': True, 'typeModel': {k: t[k] for k in ('name', 'version', 'sha256')},
            'colorModel': {k: c[k] for k in ('name', 'version', 'sha256')}}

  def infer(self, strips):
    """strips: list of (v4 strip, v3 strip) per side -> list of result dicts (StripModel.decide4 keys)."""
    if not strips:
      return []
    types, colors4 = self.type_model.forward([s[0] for s in strips])
    _, colors3 = self.color_model.forward([s[1] for s in strips])
    return [self.decide(t, c4, c3) for t, c4, c3 in zip(types, colors4, colors3, strict=True)]

  def decide(self, type_logits, color4_logits, color3_p):
    yellow = float(color3_p[COLORS.index('yellow')])
    out = self.type_model.decide4(type_logits, color4_logits, yellow=yellow)
    out.update(colorClass='yellow' if yellow > .5 else 'white', blue=False)
    return out


def load_slot(slot, directory=None, warn=None):
  """StripModel of LaneStripModel slot ``slot`` (SLOTS; HYBRID_SLOT: a HybridStripModel of slots 1 and 0). A slot other than 0
  whose files are missing, unreadable or fail the manifest / ONNX contract falls back to slot 0 (the deployed model), calling
  ``warn(message)`` first; an unknown slot is 0. The returned model's ``slot`` is the slot actually loaded. ``directory`` None:
  ASSETS (looked up at call time)."""
  import cv2
  directory = ASSETS if directory is None else directory
  if slot == HYBRID_SLOT:
    try:
      return HybridStripModel(StripModel(directory, name=SLOTS[1]), StripModel(directory, name=SLOTS[0]))
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
      if warn is not None:
        what = f'slot {slot} (hybrid: {SLOTS[1]} type + {SLOTS[0]} colour)'
        warn(f'lane strip model {what} not loaded, using {SLOTS[0]}: {type(error).__name__}: {error}')
  elif slot in SLOTS and slot != 0:
    try:
      return StripModel(directory, name=SLOTS[slot])
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
      if warn is not None:
        warn(f'lane strip model slot {slot} ({SLOTS[slot]}) not loaded, using {SLOTS[0]}: {type(error).__name__}: {error}')
  return StripModel(directory, name=SLOTS[0])
