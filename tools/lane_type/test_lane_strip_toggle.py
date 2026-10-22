"""LaneStripModel: runtime choice between the deployed StripNet (slot 0, assets/lane_strip.*, v3), the new-model slot
(slot 1, assets/lane_strip_v4.*) and the hybrid (slot 2: type from slot 1, colour from slot 0), with fallback to slot 0 and a hot
swap in marking_daemon that starts from clean state. Unless a test says otherwise the v4 files are the synthetic probe fixture
(fixtures/make_strip_v4.py), copied under the slot-1 name; the real shipped pair is checked separately."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import types
from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking, strip_model
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser
from tools.lane_type.test_strip_model_v4 import FIX, FRAME_NS, C, T, frame, strip, strips  # noqa: F401,TID251 - fixture

ROOT = Path(__file__).resolve().parents[2]
V3 = 'e6c5f34fb84d1b184d90bda5e245ab4fc32c01b4a3f55665a7b332debb87053f'
V4_SHIPPED = '0e22e71364b60bdb8761590d77a8b08880fa6571aef75184e76882405729e745'  # nolat-v3init-c seed 4 (round 3)
HALF_V4 = 1.  # the probe's strip half width in the hybrid tests: tells its strip_input calls from the v3 ones (0.75 m)


def assets(tmp_path, v4=FIX/'strip_v4_probe.onnx', strip=None):
  """An assets directory with the deployed v3 pair and, unless v4 is None, a v4 manifest + ONNX in slot 1 (``strip``: its
  manifest strip, e.g. {'half_m': 1.0}; the probe keeps 96 columns)."""
  for ext in ('onnx', 'json'):
    shutil.copy(strip_model.ASSETS/f'lane_strip.{ext}', tmp_path/f'lane_strip.{ext}')
  if v4 is not None:
    data = v4.read_bytes()
    mf = {'version': 4, 'model_sha256': hashlib.sha256(data).hexdigest(), 'threshold': .9, 'mirror_right': True,
          'input': {'name': 'strip_ycbcr', 'shape': [3, 128, 96]}, 'outputs': strip_model.OUTPUTS_V4}
    if strip is not None:
      mf['strip'] = strip
    (tmp_path/'lane_strip_v4.onnx').write_bytes(data)
    (tmp_path/'lane_strip_v4.json').write_text(json.dumps(mf), encoding='utf-8')
  return tmp_path


def test_params_and_defaults():
  for slot in (0, 1, 2):
    assert marking.model_of(N(get_int=lambda key, s=slot: s)) == slot
  for bad in (3, -1):
    assert marking.model_of(N(get_int=lambda key, b=bad: b)) == 0
  def broken(key):
    raise ValueError(key)
  assert marking.model_of(N(get_int=broken)) == 0 and marking.mode_of(N(get_int=broken)) == 0
  text = (ROOT/'openpilot/selfdrive/carrot_settings.json').read_text(encoding='utf-8')
  item = next(p for p in json.loads(text)['params'] if p['name'] == marking.MODEL_PARAM)
  assert (item['min'], item['max'], item['default'], item['control']) == (0, 2, 0, 'select')
  assert item['options']['ko'] == ['현재', '새 모델', '하이브리드(타입 새 모델 + 색 현재)']
  assert all(len(v) == 3 for v in item['options'].values())
  assert set(marking.MODEL_SLOTS) == {0, 1, 2} and strip_model.HYBRID_SLOT == 2 and set(strip_model.SLOTS) == {0, 1}
  assert re.search(r'"CcncLaneTypeDetection",\s+"LaneStripModel",', text)  # listed right after it in CANFD·HDA
  keys = (ROOT/'openpilot/common/params_keys.h').read_text(encoding='utf-8')
  assert '{"LaneStripModel", {PERSISTENT, INT, "0"}}' in keys


def test_slot0_is_the_deployed_v3():
  m = strip_model.load_slot(0)
  assert (m.slot, m.name, m.version, m.model_hash, m.v4) == (0, 'lane_strip', 3, V3, False)
  assert m.info() == {'slot': 0, 'name': 'lane_strip', 'version': 3, 'sha256': V3[:12], 'v4': False}


def test_slot1_with_v4_files_loads_v4(tmp_path):
  warned = []
  m = strip_model.load_slot(1, assets(tmp_path), warn=warned.append)
  assert (m.slot, m.name, m.version, m.v4, m.mirror_right) == (1, 'lane_strip_v4', 4, True, True) and not warned


@pytest.mark.parametrize('broken', ['missing', 'onnx_missing', 'bad_json', 'contract', 'hash'])
def test_slot1_falls_back_to_v3_with_a_warning(tmp_path, broken):
  d = assets(tmp_path, v4=FIX/('strip_v4_probe_type4.onnx' if broken == 'contract' else 'strip_v4_probe.onnx'))
  if broken == 'missing':
    for ext in ('onnx', 'json'):
      (d/f'lane_strip_v4.{ext}').unlink()
  elif broken == 'onnx_missing':
    (d/'lane_strip_v4.onnx').unlink()
  elif broken == 'bad_json':
    (d/'lane_strip_v4.json').write_text('{', encoding='utf-8')
  elif broken == 'hash':
    (d/'lane_strip_v4.onnx').write_bytes((d/'lane_strip_v4.onnx').read_bytes() + b'\0')
  warned = []
  m = strip_model.load_slot(1, d, warn=warned.append)
  assert (m.slot, m.version, m.model_hash) == (0, 3, V3)
  assert len(warned) == 1 and 'slot 1 (lane_strip_v4) not loaded, using lane_strip' in warned[0]


def test_hot_swap_resets_filter_and_dip_state(tmp_path, strips):  # noqa: F811
  rec = Recogniser(2, model=strip_model.load_slot(0))
  v4 = strip_model.load_slot(1, assets(tmp_path))
  z = np.load(FIX/'v3_golden_407.npz')
  strips['left'], strips['right'] = z['strips'][0].astype(np.float32), z['strips'][6].astype(np.float32)
  for i in range(10):
    rec.process(None, None, frame(i))
  f = frame(10)
  f['probs']['left'] = .4  # a dip timer is running on the left
  rec.process(None, None, f)
  sides = rec.strip_filter.sides
  assert rec.dip_since and any(len(s.items) for s in sides.values())
  filt = rec.strip_filter
  rec.swap_model(v4)
  assert rec.model is v4 and rec.strip_filter is filt and not rec.dip_since
  assert all(not s.items and not s.latched for s in sides.values())
  # the v4 model's own results from here on (probe logits: left solid yellow, right mirrored dbl_near white)
  strips['left'], strips['right'] = strip(T['solid'], C['yellow']), strip(T['dbl_near'], C['white'], at='mirrored')
  out = rec.process(None, None, frame(11))
  assert (out['left']['typeClass'], out['right']['typeClass']) == ('solid', 'dbl_near')
  with pytest.raises(ValueError):
    Recogniser(1).swap_model(v4)


def run_daemon(monkeypatch, d, slots, remove_at=None):
  """marking_daemon.main with fake IPC and the assets in ``d``: LaneStripModel follows ``slots`` (one entry per param poll,
  PARAM_CHECK_S 0: once per loop; CcncLaneTypeDetection on for len(slots) polls, then off); at poll ``remove_at`` the v4
  files disappear. -> (recognisers built, cloudlog lines, polls at which DisplayPolicy was reset, packets sent)."""
  from openpilot.selfdrive.carrot.lane_type import marking_daemon as md
  real_load = strip_model.load_slot
  monkeypatch.setattr(strip_model, 'load_slot', lambda slot, directory=None, warn=None: real_load(slot, d, warn))
  state = {'poll': 0, 'i': 0}

  def get_int(key):
    if key == marking.PARAM:
      state['poll'] += 1
      return 1 if state['poll'] <= len(slots) else 0
    if state['poll'] == remove_at:
      for ext in ('onnx', 'json'):
        (d/f'lane_strip_v4.{ext}').unlink(missing_ok=True)
    return slots[min(state['poll'], len(slots)) - 1] if state['poll'] else slots[0]
  monkeypatch.setitem(sys.modules, 'openpilot.common.params', N(Params=lambda: N(get_int=get_int)))
  monkeypatch.setattr(md, 'PARAM_CHECK_S', 0.)
  monkeypatch.setattr(md.os, 'nice', lambda n: None, raising=False)
  built, resets, logs = [], [], {'warning': [], 'info': []}
  monkeypatch.setattr(md, 'Recogniser', lambda *a, **kw: built.append(a) or Recogniser(*a, **kw))
  from openpilot.selfdrive.carrot.lane_type import display
  real_reset = display.DisplayPolicy.reset
  monkeypatch.setattr(display.DisplayPolicy, 'reset', lambda self: resets.append(state['poll']) or real_reset(self))
  w, h = 96, 64
  nv12 = np.full(w*h*3//2, 128, np.uint8)
  eofs = {}
  line = N(x=[float(x) for x in range(5, 25)], y=[1.8]*20, z=[0.]*20)

  class SM:
    def __init__(self, services, poll=None):
      self.updated, self.valid, self.alive = {}, dict.fromkeys(services, True), dict.fromkeys(services, True)
      self.seen, self.logMonoTime = dict.fromkeys(services, False), dict.fromkeys(services, 0)
    def update(self, timeout):
      state['i'] += 1
      self.updated = {'modelV2': True}
    def __getitem__(self, key):
      i = state['i']
      return {'modelV2': N(frameId=i, timestampEof=eofs.setdefault(i, time.monotonic_ns()), laneLines=[line]*4,
                           laneLineProbs=[1.]*4, leadsV3=[], meta=N(laneChangeState='off')),
              'liveCalibration': N(calStatus='calibrated', rpyCalib=[0., 0., 0.]),
              'carState': N(vEgo=10., leftBlinker=False, rightBlinker=False),
              'deviceState': N(started=True, deviceType='mici'), 'roadCameraState': N(sensor='os04c10'),
              'radarState': N(leadOne=N(dRel=30., status=False))}[key]

  class Client:
    def __init__(self, name, stream, conflate):
      self.frame_id = self.timestamp_eof = None
      self.last = 0
    def is_connected(self):
      return True
    def recv(self, timeout_ms=0):
      if self.last >= state['i']:
        return None
      self.last = i = state['i']
      self.frame_id, self.timestamp_eof = i, eofs.setdefault(i, time.monotonic_ns())
      return N(width=w, height=h, stride=w, uv_offset=w*h, data=nv12.tobytes(), frame_id=i)
  sent = []
  messaging = types.SimpleNamespace(SubMaster=SM, PubMaster=lambda s: N(send=lambda name, msg: sent.append(json.loads(getattr(msg, name)))),
                                    new_message=lambda name, size, valid: N())
  monkeypatch.setitem(sys.modules, 'openpilot.cereal', types.SimpleNamespace(messaging=messaging, __path__=[]))
  monkeypatch.setitem(sys.modules, 'openpilot.cereal.messaging', messaging)
  monkeypatch.setitem(sys.modules, 'openpilot.common.swaglog',
                      N(cloudlog=N(warning=lambda *a: logs['warning'].append(a[1] if len(a) > 1 else a[0]),
                                   info=lambda *a: logs['info'].append(a[1:]))))
  monkeypatch.setitem(sys.modules, 'msgq.visionipc', N(VisionIpcClient=Client, VisionStreamType=N(VISION_STREAM_ROAD=0)))
  import openpilot.common.transformations.camera as cam
  k = np.array([[100., 0., w/2], [0., 100., h/2], [0., 0., 1.]])
  monkeypatch.setattr(cam, 'DEVICE_CAMERAS', {('mici', 'os04c10'): N(fcam=N(intrinsics=k, width=w, height=h))})
  md.main()
  return built, logs, resets, sent


def test_daemon_polls_the_param_and_hot_swaps(monkeypatch, tmp_path):
  """LaneStripModel 0 -> 1 (v4 present) -> 0 -> 1 (v4 removed: v3 + warning), with the packet 'model' entry following and the
  display filter reset at each swap; no recogniser rebuild."""
  built, logs, resets, sent = run_daemon(monkeypatch, assets(tmp_path), [0]*4 + [1]*4 + [0]*4 + [1]*4, remove_at=13)
  assert built == [(2,)]  # one recogniser for the whole drive: the model is swapped in place
  assert [x[0] for x in logs['info']] == ['lane_strip', 'lane_strip_v4', 'lane_strip', 'lane_strip']
  assert [(x[1], x[2]) for x in logs['info']] == [(0, 0), (1, 1), (0, 0), (1, 0)]  # (LaneStripModel, slot loaded)
  assert len(logs['warning']) == 1 and 'not loaded, using lane_strip' in logs['warning'][0]
  assert {5, 9, 13} <= set(resets)  # the display filter restarts at every swap
  seen = [(p['model']['requested'], p['model']['slot'], p['model']['version']) for p in sent]
  assert seen and seen[0] == (0, 0, 3) and seen[-1] == (1, 0, 3)
  assert [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]] == [(0, 0, 3), (1, 1, 4), (0, 0, 3), (1, 0, 3)]


# ---- slot 2: hybrid (type from the v4 slot, colour from the v3 slot)

@pytest.fixture
def pair(monkeypatch):
  """(side, half_m) -> the strip strip_input returns: the v4 probe asks with half_m HALF_V4, the v3 model with 0.75 m."""
  cur = {}
  monkeypatch.setattr(strip_model, 'strip_input', lambda y, uv, line, rpy, k, **kw: cur.get((line['side'], kw['half_m'])))
  return cur


def hybrid(tmp_path, warned=None):
  return strip_model.load_slot(2, assets(tmp_path, strip={'half_m': HALF_V4}), warn=None if warned is None else warned.append)


def reference(m, t4, c4, c3):
  """One side of .analysis/lane-hybrid-20261022/tools/hybrid.py HybridModel.infer (the replay the slot reproduces)."""
  r = m.type_model.decide4(t4, c4)
  if r['reason'] == 'unknown':
    return r
  pt = strip_model.softmax(t4)
  dashed = min(1., float(pt[0] + pt[2])/(1. - float(pt[4])))
  yellow = float(c3[1])
  h = strip_model.StripModel.decide(N(threshold=m.threshold), np.array([dashed, 1. - dashed]), np.array([1. - yellow, yellow]),
                                    yellow_dashed_learned=True)
  h.update(typeClass=r['typeClass'], double=r['double'], pUnknown=r['pUnknown'], colorClass='yellow' if yellow > .5 else 'white', blue=False)
  return h


def test_slot2_loads_both_models(tmp_path):
  warned = []
  m = hybrid(tmp_path, warned)
  assert isinstance(m, strip_model.HybridStripModel) and not warned
  assert (m.slot, m.v4, m.threshold) == (2, True, .9)
  assert (m.type_model.name, m.type_model.v4, m.color_model.name, m.color_model.model_hash) == ('lane_strip_v4', True, 'lane_strip', V3)
  assert m.strip_inputs == ((m.type_model.strip, True), (m.color_model.strip, False))
  info = m.info()
  probe = m.type_model.model_hash[:12]
  assert info == {'slot': 2, 'name': 'hybrid', 'version': '4+3', 'sha256': f'{probe}+{V3[:12]}', 'v4': True,
                  'typeModel': {'name': 'lane_strip_v4', 'version': 4, 'sha256': probe},
                  'colorModel': {'name': 'lane_strip', 'version': 3, 'sha256': V3[:12]}}


def test_slot2_type_from_v4_colour_from_v3(tmp_path, pair):
  m = hybrid(tmp_path)
  rec = Recogniser(2, model=m, strip_filter=False)
  z = np.load(FIX/'v3_golden_407.npz')
  yellow_solid, white_dashed = z['strips'][0].astype(np.float32), z['strips'][6].astype(np.float32)
  # left: v4 dashed + blue, v3 yellow -> dashed yellow (passed through as in v4); right (mirrored v4 strip): v4 dbl_far + yellow,
  # v3 white -> solid white (dbl_far -> solid; the v4 colour head is ignored, blue never set)
  pair[('left', HALF_V4)], pair[('left', .75)] = strip(T['dashed'], C['blue']), yellow_solid
  pair[('right', HALF_V4)], pair[('right', .75)] = strip(T['dbl_far'], C['yellow'], at='mirrored'), white_dashed
  out = rec.process(None, None, frame(0))
  left, right = out['left'], out['right']
  assert (left['type'], left['color'], left['reason'], left['typeClass'], left['colorClass'], left['blue']) == \
         ('dashed', 'yellow', 'strip_model', 'dashed', 'yellow', False)
  assert (right['type'], right['color'], right['typeClass'], right['double'], right['colorClass'], right['blue']) == \
         ('solid', 'white', 'dbl_far', True, 'white', False)
  # the v3 strip is never mirrored: its colour confidence is v3's own on the unflipped right strip
  p_white = float(m.color_model.forward([white_dashed])[1][0][0])
  assert right['colorConfidence'] == pytest.approx(p_white, abs=1e-6)
  assert p_white != pytest.approx(float(m.color_model.forward([np.ascontiguousarray(white_dashed[..., ::-1])])[1][0][0]), abs=1e-6)
  # v4 'unknown' abstains on both heads whatever v3 says (no probabilities: paused like lane_prob_dip in the strip filter)
  pair[('left', HALF_V4)] = strip(T['unknown'], C['white'])
  left = rec.process(None, None, frame(1))['left']
  assert (left['type'], left['color'], left['reason'], left['blue']) == ('unknown', 'unknown', 'unknown', False)


@pytest.mark.parametrize('missing', [HALF_V4, .75])
def test_slot2_side_without_both_strips_is_not_inferred(tmp_path, pair, missing):
  rec = Recogniser(2, model=hybrid(tmp_path), strip_filter=False)
  z = np.load(FIX/'v3_golden_407.npz')
  for half in (HALF_V4, .75):
    pair[('left', half)] = strip(T['solid'], C['white']) if half == HALF_V4 else z['strips'][0].astype(np.float32)
    pair[('right', half)] = strip(T['solid'], C['white'], at='mirrored') if half == HALF_V4 else z['strips'][6].astype(np.float32)
  pair[('right', missing)] = None
  out = rec.process(None, None, frame(0))
  assert out['right']['reason'] == 'invalid_lane_geometry' and out['right']['type'] == 'unknown'
  assert out['left']['reason'] == 'strip_model' and out['left']['type'] == 'solid'


def test_slot2_mapping_equals_the_replay_reference(tmp_path):
  m = hybrid(tmp_path)
  rng = np.random.default_rng(7)
  seen = set()
  for _ in range(3000):
    t4 = rng.normal(0, 4, 5).astype(np.float32)
    c4 = rng.normal(0, 4, 3).astype(np.float32)
    y = float(rng.choice([rng.uniform(), rng.uniform(.9, 1.), rng.uniform(0., .1)]))
    c3 = np.array([1. - y, y], np.float32)
    got, want = m.decide(t4, c4, c3), reference(m, t4, c4, c3)
    if want['reason'] == 'unknown':  # the product sets the colour extras from v3 / False there too (v4 colour ignored)
      want = dict(want, colorClass='yellow' if c3[1] > .5 else 'white', blue=False)
    assert got == want
    seen.add((got['reason'], got['type'], got['color']))
  assert {('unknown', 'unknown', 'unknown'), ('strip_model', 'dashed', 'yellow'), ('strip_model', 'solid', 'white')} <= seen


@pytest.mark.parametrize('broken', ['missing', 'contract', 'hash', 'v3_in_v4_slot'])
def test_slot2_falls_back_to_v3_with_a_warning(tmp_path, broken):
  d = assets(tmp_path, v4=FIX/('strip_v4_probe_type4.onnx' if broken == 'contract' else 'strip_v4_probe.onnx'))
  if broken == 'missing':
    (d/'lane_strip_v4.json').unlink()
  elif broken == 'hash':
    (d/'lane_strip_v4.onnx').write_bytes((d/'lane_strip_v4.onnx').read_bytes() + b'\0')
  elif broken == 'v3_in_v4_slot':  # a valid model, but not a v4 one: no type model for the hybrid
    for ext in ('onnx', 'json'):
      shutil.copy(d/f'lane_strip.{ext}', d/f'lane_strip_v4.{ext}')
  warned = []
  m = strip_model.load_slot(2, d, warn=warned.append)
  assert isinstance(m, strip_model.StripModel) and (m.slot, m.version, m.model_hash) == (0, 3, V3)
  assert len(warned) == 1 and 'slot 2 (hybrid: lane_strip_v4 type + lane_strip colour) not loaded, using lane_strip' in warned[0]


def test_shipped_v4_pair_and_real_hybrid():
  """The shipped assets: slot 1 = nolat-v3init-c seed 4 (manifest v4, w100 strip, mirror_right); slot 2 runs it with the v3
  colour, and its result per side equals v4's type decision with v3's P(yellow)."""
  data = (strip_model.ASSETS/'lane_strip_v4.onnx').read_bytes()
  assert hashlib.sha256(data).hexdigest() == V4_SHIPPED and len(data) < 200_000
  v4 = strip_model.load_slot(1, warn=pytest.fail)
  assert (v4.slot, v4.version, v4.v4, v4.mirror_right, v4.threshold) == (1, 4, True, True, .9)
  assert v4.strip == {'half_m': 1., 'cols': 128, 'rows': 128}
  m = strip_model.load_slot(2, warn=pytest.fail)
  assert isinstance(m, strip_model.HybridStripModel)
  assert m.info()['sha256'] == f'{V4_SHIPPED[:12]}+{V3[:12]}' and m.info()['version'] == '4+3'
  z = np.load(FIX/'v3_golden_407.npz')
  s3 = list(z['strips'].astype(np.float32))
  s4 = [np.pad(s, ((0, 0), (0, 0), (16, 16)), mode='edge') for s in s3]  # 96 -> 128 columns, centred (+-1.0 m at 1.5625 cm/col)
  got = m.infer(list(zip(s4, s3, strict=True)))
  alone4, alone3 = v4.infer(s4), m.color_model.infer(s3)
  assert len(got) == len(s3)
  for r, r4, r3 in zip(got, alone4, alone3, strict=True):
    assert {k: r[k] for k in ('typeClass', 'double', 'pUnknown')} == {k: r4[k] for k in ('typeClass', 'double', 'pUnknown')}
    assert r['blue'] is False
    if r4['reason'] == 'unknown':
      assert (r['reason'], r['type'], r['color']) == ('unknown', 'unknown', 'unknown')
      continue
    assert (r['type'], r['typeConfidence'], r['typeCandidate']) == (r4['type'], r4['typeConfidence'], r4['typeCandidate'])
    p3 = r3['colorConfidence'] if r3['colorCandidate'] == 'yellow' else 1. - r3['colorConfidence']
    pr = r['colorConfidence'] if r['colorCandidate'] == 'yellow' else 1. - r['colorConfidence']
    assert pr == pytest.approx(p3, abs=1e-6)  # v3 float32 softmax, 1 - p in float64


def test_packet_carries_both_models(tmp_path):
  from openpilot.selfdrive.carrot.lane_type.marking_daemon import packet
  info = {**hybrid(tmp_path).info(), 'requested': 2}
  data = json.loads(json.dumps(packet(2, 1, 0, {}, {}, 0, {}, {}, info)))
  assert data['model']['slot'] == 2 and data['model']['requested'] == 2
  assert (data['model']['typeModel']['version'], data['model']['colorModel']['version']) == (4, 3)
  assert data['model']['colorModel']['sha256'] == V3[:12] and len(data['model']['typeModel']['sha256']) == 12
  assert marking.parse(json.dumps(data), 0)['left'] == {'type': None, 'color': None}  # card ignores 'model'


def test_daemon_hot_swaps_0_2_1_0(monkeypatch, tmp_path):
  """LaneStripModel 0 -> 2 (hybrid) -> 1 (v4) -> 0: each swap loads the slot, resets the display filter, and the packets'
  'model' entry follows (the hybrid one with both versions / sha256 prefixes)."""
  built, logs, resets, sent = run_daemon(monkeypatch, assets(tmp_path, strip={'half_m': HALF_V4}), [0]*4 + [2]*4 + [1]*4 + [0]*4)
  assert built == [(2,)] and not logs['warning']
  assert [x[0] for x in logs['info']] == ['lane_strip', 'hybrid', 'lane_strip_v4', 'lane_strip']
  assert [(x[1], x[2]) for x in logs['info']] == [(0, 0), (2, 2), (1, 1), (0, 0)]
  assert {5, 9, 13} <= set(resets)
  seen = [(p['model']['requested'], p['model']['slot'], p['model']['version']) for p in sent]
  assert [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]] == [(0, 0, 3), (2, 2, '4+3'), (1, 1, 4), (0, 0, 3)]
  hyb = [p['model'] for p in sent if p['model']['slot'] == 2]
  assert hyb and all(x['colorModel']['sha256'] == V3[:12] and x['typeModel']['name'] == 'lane_strip_v4' for x in hyb)


def test_non_object_manifest_falls_back_to_slot0(tmp_path):
  """Codex review F1 (10-22): valid JSON that is not an object ([], null, 3) must fall back to slot 0, not crash."""
  import shutil
  from openpilot.selfdrive.carrot.lane_type import strip_model as sm
  for f in ('lane_strip.onnx', 'lane_strip.json', 'lane_strip_v4.onnx'):
    shutil.copy(sm.ASSETS/f, tmp_path/f)
  for bad in ('[]', 'null', '3'):
    (tmp_path/'lane_strip_v4.json').write_text(bad, encoding='utf-8')
    for slot in (1, sm.HYBRID_SLOT):
      warned = []
      m = sm.load_slot(slot, directory=tmp_path, warn=warned.append)
      assert m.slot == 0 and len(warned) == 1 and 'ValueError' in warned[0], (bad, slot, warned)


def test_sample_rejects_wrong_map_width():
  """Codex review F3 (10-22): maps not shaped (rows, cols*SUB) raise instead of being silently resampled."""
  import numpy as np
  import pytest
  from openpilot.selfdrive.carrot.lane_type import profile_scanner as ps
  y, uv = np.zeros((8, 8), np.uint8), np.zeros((4, 4, 2), np.uint8)
  geo = {'ys': np.arange(1), 'cols': 2, 'mx': np.zeros((1, 9), np.float32), 'my': np.zeros((1, 9), np.float32)}
  with pytest.raises(ValueError):
    ps.sample(y, uv, geo)
  geo['mx'], geo['my'] = np.zeros((1, 2*ps.SUB), np.float32), np.zeros((1, 2*ps.SUB), np.float32)
  ps.sample(y, uv, geo)
