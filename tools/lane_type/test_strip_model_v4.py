"""StripModel manifest v4 (5-class or 4-class type / 3-class colour logits, mirror_right), the only manifest (10-26: the v1-v3
lane_strip model was removed). A weightless probe ONNX (fixtures/make_strip_v4.py) reads its logits
from the strip itself (type = channel 0 row 0 last 5 columns, colour = channel 1 row 0 last 3 columns), so each test
writes the logits it wants into the strip."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking, strip_model
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, packet

FIX = Path(__file__).with_name('fixtures')
FRAME_NS = 50_000_000
STRONG = 20.  # a logit this far above the others is a ~1.0 softmax


def manifest_dir(tmp_path, onnx=FIX/'strip_v4_probe.onnx', **over):
  data = Path(onnx).read_bytes()
  mf = {'version': 4, 'model_sha256': hashlib.sha256(data).hexdigest(), 'threshold': .9, 'mirror_right': True,
        'strip': {'half_m': .75, 'cols': 96, 'rows': 128},  # the probe's 96 columns (the manifest default is the v4 128-column strip)
        'input': {'name': 'strip_ycbcr', 'shape': [3, 128, 96]}, 'outputs': strip_model.OUTPUTS_V4, **over}
  mf = {k: v for k, v in mf.items() if v is not None}
  tmp_path.mkdir(parents=True, exist_ok=True)
  (tmp_path/f'{strip_model.MODEL_NAME}.onnx').write_bytes(data)
  (tmp_path/f'{strip_model.MODEL_NAME}.json').write_text(json.dumps(mf), encoding='utf-8')
  return tmp_path


def strip(type_logits=None, color_logits=None, at='ego'):
  """A strip whose probe logits are ``type_logits`` / ``color_logits``; at='ego' writes them where the probe reads them,
  at='mirrored' where they land after a column flip (a right strip, whose ego side is the low columns)."""
  s = np.zeros((3, 128, 96), np.float32)
  for ch, v in ((0, type_logits), (1, color_logits)):
    if v is not None:
      v = np.asarray(v, np.float32)
      if at == 'ego':
        s[ch, 0, 96 - len(v):] = v
      else:
        s[ch, 0, :len(v)] = v[::-1]
  return s


def onehot(n, i, big=STRONG):
  v = np.zeros(n, np.float32)
  v[i] = big
  return v


T = {name: onehot(5, i) for i, name in enumerate(strip_model.TYPES_V4)}
C = {name: onehot(3, i) for i, name in enumerate(strip_model.COLORS_V4)}


def frame(i, metres=1.):
  return {'lines': {'left': {'side': 'left'}, 'right': {'side': 'right'}}, 'probs': {'left': 1., 'right': 1.}, 'rpy': None,
          'k': None, 'stamp': i*FRAME_NS, 'odometer': i*metres, 'leads': [], 'lane_change': False}


@pytest.fixture
def strips(monkeypatch):
  """CUR[side] -> the strip strip_input returns for that side (as eval_cached's cached-strip replay)."""
  cur = {}
  monkeypatch.setattr(strip_model, 'strip_input', lambda y, uv, line, rpy, k, **kw: cur.get(line['side']))
  return cur


# ---- v4: manifest / ONNX contract

def test_v4_manifest_loads(tmp_path):
  m = strip_model.StripModel(manifest_dir(tmp_path))
  assert m.v4 and m.mirror_right is True and m.heads == ['type', 'color']
  assert strip_model.StripModel(manifest_dir(tmp_path/'nm', mirror_right=None)).mirror_right is False


@pytest.mark.parametrize('kind', ['v3_outputs', 'type4', 'outputs', 'mirror_type'])
def test_v4_contract_mismatch_is_a_clear_value_error(tmp_path, kind):
  case = {'v3_outputs': ({'outputs': {'line_type': ['dashed', 'solid'], 'color': ['white', 'yellow']}}, 'contract mismatch'),  # v1-v3: no longer loaded
          'type4': ({'onnx': FIX/'strip_v4_probe_type4.onnx'}, r"output 'type' shape \[None, 4\].*expects \[N, 5\]"),
          'outputs': ({'outputs': {'color': ['white', 'yellow', 'blue'], 'type': ['dashed', 'solid', 'unknown']}}, 'contract mismatch'),
          'mirror_type': ({'mirror_right': 1}, 'mirror_right')}[kind]
  with pytest.raises(ValueError, match=case[1]):
    strip_model.StripModel(manifest_dir(tmp_path, **case[0]))


# ---- v4: mapping

def test_v4_mapping_double_lines_blue_and_formula(tmp_path):
  m = strip_model.StripModel(manifest_dir(tmp_path))
  near = m.decide4(T['dbl_near'], C['white'])
  far = m.decide4(T['dbl_far'], C['blue'])
  assert (near['type'], near['color'], near['double'], near['blue'], near['typeClass']) == ('dashed', 'white', True, False, 'dbl_near')
  assert (far['type'], far['color'], far['double'], far['blue'], far['colorClass']) == ('solid', 'white', True, True, 'blue')
  yellow = m.decide4(T['solid'], C['yellow'])
  assert (yellow['type'], yellow['color'], yellow['double'], yellow['blue']) == ('solid', 'yellow', False, False)
  # P(dashed) = (P_dashed + P_dbl_near) / (1 - P_unknown), P(yellow) = P_yellow of the 3-way softmax (blue -> white)
  tl, cl = np.log([.30, .10, .25, .05, .30]), np.log([.03, .02, .95])
  r = m.decide4(tl.copy(), cl.copy())
  assert r['typeClass'] == 'dashed'  # argmax ties go to the first class, as np.argmax
  assert r['pUnknown'] == pytest.approx(.30)
  from openpilot.selfdrive.carrot.lane_type.strip_filter import p_dashed, p_yellow
  assert p_dashed(r) == pytest.approx(.55/.70) and p_yellow(r) == pytest.approx(.02)
  assert (r['type'], r['color'], r['reason']) == ('unknown', 'white', 'strip_model')  # .786 < 0.90; white .98


def test_v4_unknown_argmax_abstains(tmp_path):
  m = strip_model.StripModel(manifest_dir(tmp_path))
  r = m.decide4(T['unknown'], C['yellow'])
  assert (r['reason'], r['type'], r['color'], r['typeCandidate'], r['colorCandidate']) == ('unknown',)*5
  assert r['pUnknown'] == pytest.approx(1.) and r['colorClass'] == 'yellow' and not r['double']


def test_v4_mirror_applies_to_the_right_strip_only(tmp_path, strips):
  rec = Recogniser(2, strip_filter=False)
  rec.model = strip_model.StripModel(manifest_dir(tmp_path))
  # left: logits on its ego (high) columns; right: on its ego (low) columns -> the flip brings them to the probe
  strips['left'] = strip(T['solid'], C['yellow'], at='ego')
  strips['right'] = strip(T['dbl_near'], C['white'], at='mirrored')
  out = rec.process(None, None, frame(0))
  assert (out['left']['type'], out['left']['color'], out['left']['typeClass']) == ('solid', 'yellow', 'solid')
  assert (out['right']['type'], out['right']['color'], out['right']['typeClass']) == ('dashed', 'white', 'dbl_near')
  assert out['right']['double'] and not out['left']['double']
  # without mirror_right the right strip's logits are not where the probe reads: all-zero logits (uniform softmax)
  rec.model = strip_model.StripModel(manifest_dir(tmp_path/'nm', mirror_right=False))
  out = rec.process(None, None, frame(1))
  assert out['left']['typeClass'] == 'solid' and out['right']['pUnknown'] == pytest.approx(.2) and out['right']['type'] == 'unknown'


def test_v4_unknown_pauses_like_lane_prob_dip_and_keeps_the_latch(tmp_path, strips):
  from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
  rec, policy = Recogniser(2), DisplayPolicy()
  rec.model = strip_model.StripModel(manifest_dir(tmp_path))
  # the far line of a blue double on both sides (the right strip mirrored): white solid
  strips['left'], strips['right'] = strip(T['dbl_far'], C['blue']), strip(T['dbl_far'], C['blue'], at='mirrored')
  for i in range(12):
    out = rec.process(None, None, frame(i))
    confirmed = policy.confirm({'cameraMonoTimeNanos': i*FRAME_NS, **out}, i*FRAME_NS)
  assert all((out[s]['type'], out[s]['color'], out[s]['blue']) == ('solid', 'white', True) for s in ('left', 'right'))
  assert confirmed['left']['type']['value'] == 'solid' and confirmed['left']['color']['value'] == 'white'
  sides = rec.strip_filter.sides
  latched, items = dict(sides['left'].latched), list(sides['left'].items)
  strips['left'] = strip(T['unknown'], C['yellow'])  # the right side keeps its line
  for i in range(12, 20):
    out = rec.process(None, None, frame(i))
    confirmed = policy.confirm({'cameraMonoTimeNanos': i*FRAME_NS, **out}, i*FRAME_NS)
    assert out['left']['reason'] == 'unknown' and (out['left']['type'], out['left']['color']) == ('unknown', 'unknown')
    assert out['left']['typePDashed'] is None and out['left']['colorPYellow'] is None
    assert sides['left'].latched == latched and sides['left'].state == 'solid'
    assert list(sides['left'].items) == [x for x in items if i - x[1] <= 8.]  # nothing new entered, nothing emptied
    assert confirmed['left']['type']['value'] == 'solid'  # DisplayPolicy keeps its heads (0.5 s expiry)
    assert out['right']['type'] == 'solid'
  pk = packet(2, 1, 19*FRAME_NS, out, confirmed, 19*FRAME_NS, {}, {})
  assert pk['left']['raw']['reason'] == 'unknown' and pk['left']['raw']['typeClass'] == 'unknown'
  assert pk['right']['raw']['double'] is True and pk['right']['raw']['blue'] is True
  parsed = marking.parse(json.dumps(pk).encode(), 19*FRAME_NS)  # the card parser ignores the extra raw fields
  assert parsed['left']['type'][0] == 'solid' and parsed['right']['color'][0] == 'white'
  strips['left'] = strip(np.zeros(5), np.zeros(3))  # back, uncertain: the latch still shows white solid
  out = rec.process(None, None, frame(20))
  assert (out['left']['type'], out['left']['color']) == ('solid', 'white')


def test_v4_yellow_dashed_passes_through_but_v3_stays_unknown(tmp_path, strips):
  """Yellow dashed is legal in Korea (centre-line passing zones, roadside parking lines): v4 passes it to the filter."""
  from openpilot.selfdrive.carrot.lane_type.strip_filter import p_dashed, p_yellow
  m = strip_model.StripModel(manifest_dir(tmp_path))
  for t in ('dashed', 'dbl_near'):
    r = m.decide4(T[t], C['yellow'])
    assert (r['type'], r['color'], r['reason']) == ('dashed', 'yellow', 'strip_model')
    assert p_dashed(r) > .99 and p_yellow(r) > .99
  rec = Recogniser(2)
  rec.model = m
  strips['left'], strips['right'] = strip(T['dashed'], C['yellow']), strip(T['dashed'], C['yellow'], at='mirrored')
  out = rec.process(None, None, frame(0))
  for side in ('left', 'right'):
    assert (out[side]['type'], out[side]['color'], out[side]['reason']) == ('dashed', 'yellow', 'strip_model')
    assert out[side]['typePDashed'] > .99 and out[side]['colorPYellow'] > .99  # it enters the filter windows
  v3 = m.decide(np.array([.99, .01]), np.array([.02, .98]))  # decide() itself (the two-class rule) keeps the v1-v3 behaviour
  assert (v3['type'], v3['color'], v3['reason']) == ('unknown', 'unknown', 'yellow_dashed_not_learned')


# ---- v4 with the 4-type head (OUTPUTS_V4_4: no 'unknown' class, abstention by the largest type probability)

T4 = {name: onehot(4, i) for i, name in enumerate(strip_model.TYPES_V4_4)}


def manifest4_dir(tmp_path, **over):
  return manifest_dir(tmp_path, onnx=FIX/'strip_v4_probe_type4.onnx', outputs=strip_model.OUTPUTS_V4_4, **over)


def test_v4_4type_manifest_loads_and_contract_is_checked(tmp_path):
  m = strip_model.StripModel(manifest4_dir(tmp_path))
  assert m.v4 and m.mirror_right is True and m.heads == ['type', 'color']
  with pytest.raises(ValueError, match=r"output 'type' shape \[None, 5\].*expects \[N, 4\]"):  # 5-type ONNX, 4-type manifest
    strip_model.StripModel(manifest_dir(tmp_path/'x', outputs=strip_model.OUTPUTS_V4_4))


def test_v4_4type_mapping_and_abstention(tmp_path):
  m = strip_model.StripModel(manifest4_dir(tmp_path))
  near, far = m.decide4(T4['dbl_near'], C['white']), m.decide4(T4['dbl_far'], C['blue'])
  assert (near['type'], near['color'], near['double'], near['typeClass'], near['pUnknown']) == ('dashed', 'white', True, 'dbl_near', 0.)
  assert (far['type'], far['color'], far['double'], far['blue']) == ('solid', 'white', True, True)
  # confident: largest 4-type probability .92 >= .90; P(dashed) = dashed + dbl_near (no 1 - P(unknown) division)
  from openpilot.selfdrive.carrot.lane_type.strip_filter import p_dashed, p_yellow
  r = m.decide4(np.log([.92, .03, .03, .02]), np.log([.97, .02, .01]))
  assert (r['type'], r['color'], r['reason']) == ('dashed', 'white', 'strip_model') and p_dashed(r) == pytest.approx(.95)
  assert p_yellow(r) == pytest.approx(.02)
  # the default (HOLD_BY_FAMILY, DaBin 10-26 F4) abstains when neither display family reaches .90 (unknown is a confidence state,
  # not a class): families .55 / .45 and .55 / .45
  for probs in ([.45, .40, .10, .05], [.5, .4, .05, .05]):
    r = m.decide4(np.log(probs), np.log([.03, .96, .01]))
    assert (r['reason'], r['type'], r['color'], r['typeCandidate'], r['colorCandidate']) == ('unknown',)*5
    assert r['typeClass'] == 'dashed' and r['colorClass'] == 'yellow' and r['pUnknown'] == 0.
  # largest class .85 but dashed family .95: confident by default; the single-class rule (HOLD_BY_FAMILY False) abstains
  r = m.decide4(np.log([.85, .05, .10, .00001]), np.log([.03, .96, .01]))
  assert (r['reason'], r['type'], r['color']) == ('strip_model', 'dashed', 'yellow')
  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(strip_model, 'HOLD_BY_FAMILY', False)
    assert m.decide4(np.log([.85, .05, .10, .00001]), np.log([.03, .96, .01]))['reason'] == 'unknown'


def test_shipped_product_manifest_is_the_5type_r5_small_s0_model():
  """DaBin 10-27: ship R5 small s0 (unknown argmax abstains); HOLD_BY_FAMILY only acts on 4-type manifests."""
  m = strip_model.StripModel()
  assert strip_model.HOLD_BY_FAMILY is True and strip_model.HOLD_COLOR is False
  assert (m.v4, m.mirror_right, m.threshold, m.heads) == (True, True, .9, ['type', 'color'])
  manifest = json.loads((strip_model.ASSETS/f'{strip_model.MODEL_NAME}.json').read_text(encoding='utf-8'))
  assert manifest['outputs'] == strip_model.OUTPUTS_V4 and manifest['params'] == 34079
  assert manifest['training']['seed'] == 0 and manifest['training']['capacity'] == 'small'
  assert manifest['deviceValidated'] is False  # R5 has not been validated on the device.
  assert strip_model.onnx_output_shape((strip_model.ASSETS/f'{strip_model.MODEL_NAME}.onnx').read_bytes(), 'type') == [None, 5]
  types, colors = m.forward([np.full((3, 128, 128), 100., np.float32)])
  assert types.shape == (1, 5) and colors.shape == (1, 3)
  assert m.decide4(T['unknown'], C['white'])['reason'] == 'unknown'  # the 5-type abstention path


def test_v4_4type_abstention_pauses_like_the_5type_unknown(tmp_path, strips):
  rec = Recogniser(2)
  rec.model = strip_model.StripModel(manifest4_dir(tmp_path))
  strips['left'], strips['right'] = strip(T4['solid'], C['yellow']), strip(T4['dbl_near'], C['white'], at='mirrored')
  for i in range(3):
    out = rec.process(None, None, frame(i))
  assert (out['left']['type'], out['left']['color']) == ('solid', 'yellow')
  assert (out['right']['type'], out['right']['typeClass']) == ('dashed', 'dbl_near')
  latched = dict(rec.strip_filter.sides['left'].latched)
  strips['left'] = strip(np.zeros(4), C['white'])  # uniform 4-type softmax (.25): abstains, the latch is kept
  out = rec.process(None, None, frame(3))
  assert out['left']['reason'] == 'unknown' and out['left']['typePDashed'] is None and out['left']['colorPYellow'] is None
  assert rec.strip_filter.sides['left'].latched == latched
