"""Lane strip width as a model-manifest parameter (lane_strip_v4.json 'strip': {half_m, cols, rows}; absent keys = the v4 1.0 / 128 / 128)."""
import hashlib
import json
import shutil

import numpy as np
import pytest

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.carrot.lane_type import strip_model as SM
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser
from openpilot.selfdrive.carrot.lane_type.profile_scanner import sample, strip_geometry

# strip_input / sample of the frame below, computed with the code before the width parameter existed (2026-10-21)
V3_STRIP_SHA = 'b1f14cb67be70222dd10a18eb86edde21df861d2a11dded23dc33ff98848bb93'
V3_SAMPLE_SHA = 'e876a29b61ef4f43fb474a3c6dae45a18614da74e2f62a61cc14677b1dcea7a1'


def frame():
  rng = np.random.default_rng(20261021)
  h, w = 1208, 1928
  y = rng.integers(0, 256, (h, w), dtype=np.uint8)
  uv = rng.integers(0, 256, (h//2, w//2, 2), dtype=np.uint8)
  k = DEVICE_CAMERAS[('tici', 'ar0231')].fcam.intrinsics.copy()
  x = np.linspace(0., 60., 33)
  return y, uv, {'x': list(x), 'y': list(1.7 + .002*x**2), 'z': list(1.25 + 0*x)}, [0., .03, .01], k


def sha(a):
  return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_v3_width_strip_is_bit_identical_and_the_default_is_v4():
  y, uv, line, rpy, k = frame()
  s = SM.strip_input(y, uv, line, rpy, k, half_m=.75, cols=96, rows=128)  # the sampler itself is unchanged at the v3 width
  assert s.shape == (3, 128, 96) and sha(s) == V3_STRIP_SHA
  default = SM.strip_input(y, uv, line, rpy, k)
  assert default.shape == (3, 128, 128) and np.array_equal(default, SM.strip_input(y, uv, line, rpy, k, **SM.strip_of({})))
  assert np.array_equal(default, SM.strip_input(y, uv, line, rpy, k, half_m=1., cols=128, rows=128))
  geo = strip_geometry(line, rpy, k, y.shape[1], y.shape[0])
  assert sha(np.stack(sample(y, uv, geo))) == V3_SAMPLE_SHA


def test_wide_strip_has_128_columns_at_the_same_pitch():
  y, uv, line, rpy, k = frame()
  narrow, wide = SM.strip_input(y, uv, line, rpy, k, half_m=.75, cols=96), SM.strip_input(y, uv, line, rpy, k, half_m=1., cols=128)
  assert wide.shape == (3, 128, 128)
  assert 2*1./128 == 2*.75/96  # 1.5625 cm per column
  g0 = strip_geometry(line, rpy, k, y.shape[1], y.shape[0])
  g1 = strip_geometry(line, rpy, k, y.shape[1], y.shape[0], half_m=1., cols=128)
  assert g1['mx'].shape == (len(g0['ys']), 128*4) and (g1['ys'] == g0['ys']).all()
  # wide column 16 + i covers narrow column i: same metric offsets, the source u differs only by the perspective of the row
  # (u is interpolated linearly between the strip edges; < 0.3 px here, ~0.6 column)
  assert np.abs(g1['mx'][:, 64:64 + 96*4] - g0['mx']).max() < .3
  assert np.isfinite(wide).all() and wide.min() >= 0 and wide.max() <= 255 and narrow.shape == (3, 128, 96)


def test_manifest_strip_defaults_and_validation():
  assert SM.strip_of({}) == SM.strip_of({'strip': None}) == {'half_m': 1., 'cols': 128, 'rows': 128}
  assert SM.strip_of({'strip': {'half_m': .75, 'cols': 96}}) == {'half_m': .75, 'cols': 96, 'rows': 128}
  for bad in ({'cols': 0}, {'half_m': -1}, {'cols': 12.5}, {'width': 2}, {'half_m': True}, [1, 2]):
    with pytest.raises(ValueError):
      SM.strip_of({'strip': bad})


def model_dir(tmp_path, strip=None, shape=None):
  d = tmp_path/'m'
  d.mkdir()
  shutil.copy(SM.ASSETS/f'{SM.MODEL_NAME}.onnx', d/f'{SM.MODEL_NAME}.onnx')
  m = json.loads((SM.ASSETS/f'{SM.MODEL_NAME}.json').read_text(encoding='utf-8'))
  if strip is not None:
    m['strip'] = strip
  if shape is not None:
    m['input']['shape'] = shape
  (d/f'{SM.MODEL_NAME}.json').write_text(json.dumps(m), encoding='utf-8')
  return d


def test_shipped_model_keeps_v4_geometry(tmp_path):
  assert SM.StripModel().strip == {'half_m': 1., 'cols': 128, 'rows': 128}
  assert SM.onnx_input_shape((SM.ASSETS/f'{SM.MODEL_NAME}.onnx').read_bytes(), 'strip_ycbcr') == [None, 3, 128, 128]
  assert SM.StripModel(model_dir(tmp_path, {'half_m': 1., 'cols': 128, 'rows': 128})).strip['cols'] == 128


def test_shape_mismatch_is_a_clear_error(tmp_path):
  with pytest.raises(ValueError, match=r'ONNX input .* \[None, 3, 128, 128\] does not match the manifest strip \+-0.75 m / 96 cols'):
    SM.StripModel(model_dir(tmp_path, {'half_m': .75, 'cols': 96, 'rows': 128}, [3, 128, 96]))
  with pytest.raises(ValueError, match=r'manifest input shape \[3, 128, 128\] does not match its strip \+-0.75 m / 96 cols'):
    (tmp_path/'b').mkdir()
    SM.StripModel(model_dir(tmp_path/'b', {'half_m': .75, 'cols': 96, 'rows': 128}))


def test_onnx_input_shape_reader():
  def v(n):
    out = b''
    while True:
      out += bytes([(n & 0x7f) | (0x80 if n > 0x7f else 0)])
      n >>= 7
      if not n:
        return out
  def ld(field, payload):
    return v(field << 3 | 2) + v(len(payload)) + payload
  dims = ld(1, ld(2, b'n')) + ld(1, v(1 << 3) + v(3)) + ld(1, v(1 << 3) + v(128)) + ld(1, v(1 << 3) + v(128))
  vi = ld(1, b'strip_ycbcr') + ld(2, ld(1, v(1 << 3) + v(1) + ld(2, dims)))
  model = v(1 << 3) + v(8) + ld(7, ld(5, b'\x00'*300) + ld(11, ld(1, b'other')) + ld(11, vi))
  assert SM.onnx_input_shape(model, 'strip_ycbcr') == [None, 3, 128, 128]
  assert SM.onnx_input_shape(model, 'missing') is None


def test_recogniser_passes_manifest_geometry(monkeypatch):
  seen = []

  class Fake:
    strip = {'half_m': 1., 'cols': 128, 'rows': 128}

    def infer(self, strips):
      return [{'type': 'solid', 'color': 'white', 'typeConfidence': .99, 'colorConfidence': .99, 'reason': 'strip_model'} for _ in strips]

  import importlib
  sm = importlib.import_module(SM.__name__)  # the module marking_daemon imports now (another test may have reloaded it)
  monkeypatch.setattr(sm, 'strip_input', lambda y, uv, line, rpy, k, **kw: seen.append(kw) or np.zeros((3, 128, 128), np.float32))
  rec = Recogniser(2, strip_filter=False)
  rec.model = Fake()
  f = {'lines': {'left': {'y': [1.8]}, 'right': {'y': [-1.8]}}, 'probs': {'left': .9, 'right': .9}, 'rpy': None, 'k': None,
       'stamp': 1, 'odometer': 0., 'leads': [], 'lane_change': False}
  out = rec.process(None, None, f)
  assert seen == [Fake.strip]*2 and out['left']['type'] == 'solid'
  rec.model = type('Old', (), {'infer': Fake.infer})()  # a replay wrapper without .strip: default geometry
  seen.clear()
  rec.process(None, None, f)
  assert seen == [{}, {}]
