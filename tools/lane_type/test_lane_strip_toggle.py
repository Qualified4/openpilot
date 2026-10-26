"""The StripNet model is assets/lane_strip_v4.* only (DaBin 10-26: the v1-v3 model, the model choice setting, its slots, fallback and
hot swap were removed). A model that is missing or fails its contract checks is not replaced: strip_model.load returns None with a
warning and marking_daemon publishes nothing (the existing lane display stays) until CcncLaneTypeDetection is turned off. Unless a
test says otherwise the model files are the synthetic probe fixture (fixtures/make_strip_v4.py) under the model name; the real
shipped pair is checked separately."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import types
from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking, strip_model
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser
from tools.lane_type.test_strip_model_v4 import FIX  # noqa: TID251

ROOT = Path(__file__).resolve().parents[2]
V4_SHIPPED = '23ff1bc749421977e400977e9d14e70dd2e30ab59391a0a59e890f7051f9afb7'  # 10-24 rain1024b D1 s2 (5 types); 10-26 4-type model rolled back


def assets(tmp_path, v4=FIX/'strip_v4_probe.onnx', strip={'half_m': .75, 'cols': 96, 'rows': 128}):
  """An assets directory with a v4 manifest + ONNX under strip_model.MODEL_NAME (``strip``: its manifest strip; the probe keeps
  96 columns)."""
  data = v4.read_bytes()
  mf = {'version': 4, 'model_sha256': hashlib.sha256(data).hexdigest(), 'threshold': .9, 'mirror_right': True,
        'input': {'name': 'strip_ycbcr', 'shape': [3, 128, 96]}, 'outputs': strip_model.OUTPUTS_V4}
  if strip is not None:
    mf['strip'] = strip
  (tmp_path/f'{strip_model.MODEL_NAME}.onnx').write_bytes(data)
  (tmp_path/f'{strip_model.MODEL_NAME}.json').write_text(json.dumps(mf), encoding='utf-8')
  return tmp_path


def test_single_model_api_and_setting_removed():
  assert strip_model.MODEL_NAME == 'lane_strip_v4'
  assert not hasattr(strip_model, 'SLOTS') and not hasattr(strip_model, 'load_slot') and not hasattr(strip_model, 'OUTPUTS_V3')
  assert not hasattr(marking, 'model_of') and not hasattr(marking, 'MODEL_PARAM')
  assert not hasattr(Recogniser, 'swap_model')
  assert sorted(p.name for p in strip_model.ASSETS.glob('lane_strip*')) == ['lane_strip_v4.json', 'lane_strip_v4.onnx']
  text = (ROOT/'openpilot/selfdrive/carrot_settings.json').read_text(encoding='utf-8')
  assert re.search(r'"CcncLaneTypeDetection",\s+"CcncRadarVehicles",', text)  # nothing listed between them in CANFD·HDA any more
  keys = (ROOT/'openpilot/common/params_keys.h').read_text(encoding='utf-8')
  assert re.search(r'\{"CcncLaneTypeDetection", \{PERSISTENT, INT, "0"\}\},\s+\{"CcncRadarMap"', keys)


def test_load_v4(tmp_path):
  warned = []
  m = strip_model.load(assets(tmp_path), warn=warned.append)
  assert (m.name, m.version, m.v4, m.mirror_right) == ('lane_strip_v4', 4, True, True) and not warned
  assert set(m.info()) == {'name', 'version', 'sha256', 'v4'}


@pytest.mark.parametrize('broken', ['missing', 'onnx_missing', 'bad_json', 'contract', 'hash', '[]', 'null', '3'])
def test_load_failure_returns_none_with_a_warning(tmp_path, broken):
  """No fallback model: a missing or invalid model (incl. Codex review F1 10-22: valid JSON that is not an object) is None."""
  d = assets(tmp_path, v4=FIX/('strip_v4_probe_type4.onnx' if broken == 'contract' else 'strip_v4_probe.onnx'))
  name = strip_model.MODEL_NAME
  if broken == 'missing':
    for ext in ('onnx', 'json'):
      (d/f'{name}.{ext}').unlink()
  elif broken == 'onnx_missing':
    (d/f'{name}.onnx').unlink()
  elif broken == 'bad_json':
    (d/f'{name}.json').write_text('{', encoding='utf-8')
  elif broken == 'hash':
    (d/f'{name}.onnx').write_bytes((d/f'{name}.onnx').read_bytes() + b'\0')
  elif broken in ('[]', 'null', '3'):
    (d/f'{name}.json').write_text(broken, encoding='utf-8')
  warned = []
  assert strip_model.load(d, warn=warned.append) is None
  assert len(warned) == 1 and 'lane_strip_v4 not loaded, lane types are not shown' in warned[0]


def run_daemon(monkeypatch, d, polls):
  """marking_daemon.main with fake IPC and the model files in ``d`` (CcncLaneTypeDetection on for ``polls`` param polls, then
  off; PARAM_CHECK_S 0). -> (recognisers built, cloudlog lines, polls at which DisplayPolicy was reset, packets sent)."""
  from openpilot.selfdrive.carrot.lane_type import marking_daemon as md
  real_load = strip_model.load
  monkeypatch.setattr(strip_model, 'load', lambda directory=None, warn=None: real_load(d, warn))
  state = {'poll': 0, 'i': 0}

  def get_int(key):
    assert key == marking.PARAM  # the only param the daemon reads
    state['poll'] += 1
    return 1 if state['poll'] <= polls else 0
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
  import openpilot
  cereal = types.SimpleNamespace(messaging=messaging, __path__=[])
  monkeypatch.setattr(openpilot, 'cereal', cereal)
  monkeypatch.setitem(sys.modules, 'openpilot.cereal', cereal)
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


def test_daemon_runs_the_v4_model(monkeypatch, tmp_path):
  built, logs, resets, sent = run_daemon(monkeypatch, assets(tmp_path), 8)
  assert built == [(2,)] and not logs['warning']
  assert [x[0] for x in logs['info']] == ['lane_strip_v4']
  assert sent and all(p['model'] == {'name': 'lane_strip_v4', 'version': 4, 'sha256': p['model']['sha256'], 'v4': True} for p in sent)


def test_daemon_without_a_model_publishes_nothing(monkeypatch, tmp_path):
  """Missing model files: one warning, no recogniser, no packet (the existing lane display stays); the process returns once the
  option is turned off."""
  for ext in ('onnx', 'json'):
    (assets(tmp_path)/f'{strip_model.MODEL_NAME}.{ext}').unlink()
  built, logs, resets, sent = run_daemon(monkeypatch, tmp_path, 5)
  assert built == [] and sent == [] and not logs['info']
  assert len(logs['warning']) == 1 and 'not loaded, lane types are not shown' in logs['warning'][0]


def test_shipped_v4_pair():
  """The shipped model: 10-24 rain1024b D1 s2 (manifest v4, 5 types, w100 strip, mirror_right; restored by the 10-26 rollback)."""
  data = (strip_model.ASSETS/f'{strip_model.MODEL_NAME}.onnx').read_bytes()
  assert hashlib.sha256(data).hexdigest() == V4_SHIPPED and len(data) < 500_000
  v4 = strip_model.load(warn=pytest.fail)
  assert (v4.version, v4.v4, v4.mirror_right, v4.threshold) == (4, True, True, .9)
  assert v4.strip == {'half_m': 1., 'cols': 128, 'rows': 128}
  assert Recogniser(2).model.model_hash == V4_SHIPPED  # the default Recogniser model


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
