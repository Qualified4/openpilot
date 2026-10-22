"""One wide strip: manifest-derived crop bounds and explicit side metadata through the real recogniser."""
from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import strip_model as sm
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, packet
from tools.lane_type.test_lane_strip_toggle import assets  # noqa: TID251 - repository fixture
from tools.lane_type.test_strip_model_v4 import frame  # noqa: TID251 - repository fixture


def models(wide=None, color=None):
  return (N(v4=True, threshold=.9, mirror_right=True, strip=wide or {'half_m': 1., 'cols': 128, 'rows': 128}),
          N(v4=False, strip=color or {'half_m': .75, 'cols': 96, 'rows': 128}))


@pytest.mark.parametrize('wide,color,expected', [
  ({'half_m': 1., 'cols': 128, 'rows': 128}, {'half_m': .75, 'cols': 96, 'rows': 128}, (16, 112)),
  ({'half_m': .3, 'cols': 60, 'rows': 128}, {'half_m': .1, 'cols': 20, 'rows': 128}, (20, 40)),
  ({'half_m': .75, 'cols': 96, 'rows': 128}, {'half_m': .75, 'cols': 96, 'rows': 128}, (0, 96)),
])
def test_crop_bounds_come_from_both_manifests(wide, color, expected):
  m = sm.HybridStripModel(*models(wide, color))
  assert m.crop_columns == expected and m.strip_inputs == ((wide, False),)


@pytest.mark.parametrize('wide,color', [
  ({'half_m': 1., 'cols': 96, 'rows': 128}, None),  # unequal pitch
  ({'half_m': 1.0078125, 'cols': 129, 'rows': 128}, None),  # equal pitch, half-column offset
  ({'half_m': .5, 'cols': 64, 'rows': 128}, None),  # colour extends beyond wide
  (None, {'half_m': .75, 'cols': 96, 'rows': 96}),
])
def test_incompatible_crop_contract_is_rejected(wide, color):
  with pytest.raises(ValueError, match='colour crop'):
    sm.HybridStripModel(*models(wide, color))


def test_bad_crop_manifest_uses_existing_slot0_fallback(tmp_path):
  warned = []
  # Valid 96-column probe ONNX, valid manifest, incompatible with v3's metric spacing.
  m = sm.load_slot(2, assets(tmp_path, strip={'half_m': 1.}), warn=warned.append)
  assert m.slot == 0 and len(warned) == 1 and 'colour crop' in warned[0]


@pytest.mark.parametrize('valid', [('left', 'right'), ('left',), ('right',), ()])
def test_recogniser_extracts_once_and_only_type_mirrors_right(valid, monkeypatch):
  m = sm.load_slot(2, warn=pytest.fail)
  original = np.arange(3*128*128, dtype=np.float32).reshape(3, 128, 128) % 251
  sources = {s: original.copy() + i for i, s in enumerate(('left', 'right'))}
  calls, type_inputs, color_inputs = [], [], []
  def sample(y, uv, line, rpy, k, **geometry):
    calls.append((line['side'], geometry))
    return sources[line['side']]
  def type_forward(inputs):
    type_inputs.extend(inputs)
    return np.tile([0., 20., 0., 0., 0.], (len(inputs), 1)), np.tile([20., 0., 0.], (len(inputs), 1))
  def color_forward(inputs):
    color_inputs.extend(inputs)
    return np.tile([0., 1.], (len(inputs), 1)), np.tile([1., 0.], (len(inputs), 1))
  monkeypatch.setattr(sm, 'strip_input', sample)
  monkeypatch.setattr(m.type_model, 'forward', type_forward)
  monkeypatch.setattr(m.color_model, 'forward', color_forward)
  rec = Recogniser(2, model=m, strip_filter=False)
  f = frame(0)
  f['probs'] = {s: 1. if s in valid else 0. for s in ('left', 'right')}
  rec.process(None, None, f)
  assert [s for s, g in calls] == list(valid)
  assert all(g == m.type_model.strip for s, g in calls)
  for s, t, c in zip(valid, type_inputs, color_inputs, strict=True):
    assert np.array_equal(t, sources[s][..., ::-1] if s == 'right' else sources[s])
    assert np.array_equal(c, sources[s][..., 16:112])
  assert np.array_equal(sources['left'], original)  # preparation never changes pixels in place


def test_missing_side_metadata_is_not_guessed():
  m = sm.HybridStripModel(*models())
  s = np.zeros((3, 128, 128), np.float32)
  for kwargs in ({}, {'sides': ['unknown']}, {'sides': ['left', 'right']}):
    with pytest.raises(ValueError, match='side name'):
      m.infer([(s,)], **kwargs)
  assert m.infer([]) == []


def test_packet_marks_shipped_crop_without_changing_model_identity():
  m = sm.load_slot(2, warn=pytest.fail)
  info = m.info()
  assert info['slot'] == 2 and info['version'] == '4+3' and info['colorInput'] == 'wide_crop_16_112'
  assert info['typeModel']['sha256'] == sm.load_slot(1).info()['sha256']
  assert info['colorModel']['sha256'] == sm.load_slot(0).info()['sha256']
  assert packet(2, 1, 0, {}, {}, 0, {}, {}, info)['model']['colorInput'] == 'wide_crop_16_112'
