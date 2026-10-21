"""LaneStripModel: runtime choice between the deployed StripNet (slot 0, assets/lane_strip.*, v3) and the new-model slot
(slot 1, assets/lane_strip_v4.*), with fallback to slot 0 and a hot swap in marking_daemon that starts from clean state.
The v4 files are the synthetic probe fixture (fixtures/make_strip_v4.py), copied under the slot-1 name."""
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


def assets(tmp_path, v4=FIX/'strip_v4_probe.onnx'):
  """An assets directory with the deployed v3 pair and, unless v4 is None, a v4 manifest + ONNX in slot 1."""
  for ext in ('onnx', 'json'):
    shutil.copy(strip_model.ASSETS/f'lane_strip.{ext}', tmp_path/f'lane_strip.{ext}')
  if v4 is not None:
    data = v4.read_bytes()
    mf = {'version': 4, 'model_sha256': hashlib.sha256(data).hexdigest(), 'threshold': .9, 'mirror_right': True,
          'input': {'name': 'strip_ycbcr', 'shape': [3, 128, 96]}, 'outputs': strip_model.OUTPUTS_V4}
    (tmp_path/'lane_strip_v4.onnx').write_bytes(data)
    (tmp_path/'lane_strip_v4.json').write_text(json.dumps(mf), encoding='utf-8')
  return tmp_path


def test_params_and_defaults():
  assert marking.model_of(N(get_int=lambda key: 0)) == 0 and marking.model_of(N(get_int=lambda key: 1)) == 1
  for bad in (2, -1):
    assert marking.model_of(N(get_int=lambda key, b=bad: b)) == 0
  def broken(key):
    raise ValueError(key)
  assert marking.model_of(N(get_int=broken)) == 0 and marking.mode_of(N(get_int=broken)) == 0
  text = (ROOT/'openpilot/selfdrive/carrot_settings.json').read_text(encoding='utf-8')
  item = next(p for p in json.loads(text)['params'] if p['name'] == marking.MODEL_PARAM)
  assert (item['min'], item['max'], item['default'], item['control']) == (0, 1, 0, 'select')
  assert item['options']['ko'] == ['현재', '새 모델']
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


def test_daemon_polls_the_param_and_hot_swaps(monkeypatch, tmp_path):
  """marking_daemon.main with fake IPC: LaneStripModel 0 -> 1 (v4 present) -> 0 -> 1 (v4 removed: v3 + warning), with the
  packet 'model' entry following and the display filter reset at each swap; no recogniser rebuild."""
  from openpilot.selfdrive.carrot.lane_type import marking_daemon as md
  d = assets(tmp_path)
  real_load = strip_model.load_slot
  monkeypatch.setattr(strip_model, 'load_slot', lambda slot, directory=None, warn=None: real_load(slot, d, warn))
  # one entry per param poll (PARAM_CHECK_S 0: once per loop); CcncLaneTypeDetection on for 16 polls, then off
  slots = [0]*4 + [1]*4 + [0]*4 + [1]*4
  state = {'poll': 0, 'i': 0}

  def get_int(key):
    if key == marking.PARAM:
      state['poll'] += 1
      return 1 if state['poll'] <= len(slots) else 0
    if state['poll'] == 13:  # before the 4th swap the v4 files disappear
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
  assert built == [(2,)]  # one recogniser for the whole drive: the model is swapped in place
  assert [x[0] for x in logs['info']] == ['lane_strip', 'lane_strip_v4', 'lane_strip', 'lane_strip']
  assert [(x[1], x[2]) for x in logs['info']] == [(0, 0), (1, 1), (0, 0), (1, 0)]  # (LaneStripModel, slot loaded)
  assert len(logs['warning']) == 1 and 'not loaded, using lane_strip' in logs['warning'][0]
  assert {5, 9, 13} <= set(resets)  # the display filter restarts at every swap
  seen = [(p['model']['requested'], p['model']['slot'], p['model']['version']) for p in sent]
  assert seen and seen[0] == (0, 0, 3) and seen[-1] == (1, 0, 3)
  assert [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]] == [(0, 0, 3), (1, 1, 4), (0, 0, 3), (1, 0, 3)]
