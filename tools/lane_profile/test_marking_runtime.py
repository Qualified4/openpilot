import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, filter_input, packet, planes, requested

MS = 1_000_000


def confirmed(policy, now, left, right=('unknown', 'unknown'), **kw):
  raw = {'left': filter_input(*left, .99, .99, now, dashed_observed=left[0] == 'dashed', **kw),
         'right': filter_input(*right, .99, .99, now, dashed_observed=right[0] == 'dashed')}
  return raw, policy.confirm({'cameraMonoTimeNanos': now, **raw}, now)


def test_display_filter_needs_two_results_within_window_and_expires():
  policy, t = DisplayPolicy(), 10_000*MS
  _, c = confirmed(policy, t, ('solid', 'yellow'))
  assert c['left'] == {}  # one result is only pending
  _, c = confirmed(policy, t + 50*MS, ('solid', 'yellow'))
  assert c['left']['type']['value'] == 'solid' and c['left']['color']['value'] == 'yellow'
  _, c = confirmed(policy, t + 450*MS, ('unknown', 'unknown'))  # unknown keeps the last evidence until expiry
  assert c['left']['color']['value'] == 'yellow'
  _, c = confirmed(policy, t + 600*MS, ('unknown', 'unknown'))
  assert c['left'] == {}  # 0.5 s after the last evidence
  policy2 = DisplayPolicy()
  confirmed(policy2, t, ('dashed', 'white'))
  _, c = confirmed(policy2, t + 500*MS, ('dashed', 'white'))  # second result outside 0.4 s: still pending
  assert 'type' not in c['left']


def test_lane_change_resets_and_sides_are_independent():
  policy, t = DisplayPolicy(), 10_000*MS
  for i in range(2):
    confirmed(policy, t + i*50*MS, ('solid', 'yellow'), ('dashed', 'white'))
  _, c = confirmed(policy, t + 100*MS, ('unknown', 'unknown'), ('dashed', 'white'), reason='lane_change')
  assert c['left'] == {} and c['right']['type']['value'] == 'dashed'


def test_packet_parse_and_expiry_are_anchored_on_receive_time():
  policy, t = DisplayPolicy(), 10_000*MS
  for i in range(2):
    raw, c = confirmed(policy, t + i*50*MS, ('solid', 'yellow'), ('dashed', 'white'))
  data = packet(1, 7, t + 50*MS, raw, c, t + 80*MS, {'processingMs': 1.}, {'processed': 2})
  assert data['left']['typeAgeNs'] == 30*MS and data['right']['color'] == 'white'
  parsed = marking.parse(json.dumps(data).encode(), received_ns=5_000*MS)  # receiver clock differs
  assert marking.current(parsed, 'left', 5_000*MS) == ('solid', 'yellow')
  assert marking.current(parsed, 'left', 5_000*MS + 470*MS) == ('solid', 'yellow')
  assert marking.current(parsed, 'left', 5_000*MS + 471*MS) == (None, None)  # 0.5 s after the evidence
  with pytest.raises(ValueError):
    marking.parse(b'{"type":"other","version":1}', 0)


def test_modes_and_selected_engine_only(monkeypatch):
  params = N(get_int=lambda key: 0)
  assert not requested(True, params)
  for mode in (1, 2):
    assert requested(True, N(get_int=lambda key, m=mode: m)) and not requested(False, N(get_int=lambda key, m=mode: m))
  assert marking.mode_of(N(get_int=lambda key: 7)) == 0
  sys.modules.pop('openpilot.selfdrive.carrot.lane_type.strip_model', None)
  scanner = Recogniser(1)
  assert scanner.model is None and 'openpilot.selfdrive.carrot.lane_type.strip_model' not in sys.modules
  strip = Recogniser(2)
  assert strip.scanners is None and strip.model.threshold == pytest.approx(.90)
  with pytest.raises(ValueError):
    Recogniser(0)


def test_strip_model_threshold_and_unlearned_yellow_dashed():
  model = Recogniser(2).model
  assert model.decide(np.array([.95, .05]), np.array([.97, .03]))['type'] == 'dashed'
  low = model.decide(np.array([.85, .15]), np.array([.97, .03]))
  assert low['type'] == 'unknown' and low['color'] == 'white'
  yd = model.decide(np.array([.99, .01]), np.array([.02, .98]))
  assert yd['type'] == yd['color'] == 'unknown'
  out = model.infer([np.full((3, 128, 96), 100., np.float32)]*2)
  assert len(out) == 2 and {'type', 'color', 'typeConfidence', 'colorConfidence'} <= set(out[0])


def test_nv12_planes_respect_stride_without_copy():
  w, h, stride = 8, 4, 16
  raw = np.arange(stride*h*3//2, dtype=np.uint8)
  buf = N(width=w, height=h, stride=stride, uv_offset=stride*h, data=raw.tobytes())
  y, uv = planes(buf)
  assert y.shape == (4, 8) and uv.shape == (2, 4, 2)
  assert y[1, 0] == stride and uv[1, 0, 0] == (stride*h + stride) % 256


def test_off_mode_entrypoint_imports_and_loads_nothing(monkeypatch):
  import builtins
  from openpilot.selfdrive.carrot.lane_type import marking_daemon
  monkeypatch.setitem(sys.modules, 'openpilot.common.params', N(Params=lambda: N(get_int=lambda key: 0)))
  monkeypatch.setattr(marking_daemon, 'Recogniser', lambda *a: pytest.fail('OFF built a recogniser'))
  original = builtins.__import__

  def guarded(name, *args, **kwargs):
    if name in ('cv2', 'numpy', 'msgq.visionipc', 'openpilot.cereal.messaging'):
      raise AssertionError('OFF imported inference or IPC: ' + name)
    return original(name, *args, **kwargs)
  with monkeypatch.context() as m:
    m.setattr(builtins, '__import__', guarded)
    marking_daemon.main()


def test_daemon_main_loop_with_fake_ipc(monkeypatch):
  """Drive marking_daemon.main() end to end with fake VisionIPC/SubMaster/PubMaster on a synthetic yellow
  solid boundary: exact frame sync, display confirmation, publication, mode switch and OFF exit."""
  import types
  from openpilot.selfdrive.carrot.lane_type import marking_daemon as md
  from tools.lane_profile.synthetic import to_nv12_planes  # noqa: TID251
  from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251
  rgb, line, k, rpy, *_ = synthetic_frame(np.zeros(3))
  y, uv = to_nv12_planes(rgb)
  h, w = y.shape
  nv12 = np.concatenate((y, uv.reshape(h//2, w)))
  modes = iter([1]*6 + [2]*4 + [0]*10)
  monkeypatch.setitem(sys.modules, 'openpilot.common.params', N(Params=lambda: N(get_int=lambda key: next(modes))))
  monkeypatch.setattr(md, 'PARAM_CHECK_S', 0.)
  monkeypatch.setattr(md.os, 'nice', lambda n: None, raising=False)
  import time
  state = {'i': 0, 'eof': {}, 'queue': [], 'sent': set()}

  def eof(i):  # a camera EOF is always in the (recent) past on the device's monotonic time base
    return state['eof'].setdefault(i, time.monotonic_ns())

  def model_msg(i):
    lane = N(x=line['x'], y=line['y'], z=line['z'])
    return N(frameId=i, timestampEof=eof(i), laneLines=[lane, lane, lane, lane], laneLineProbs=[1., 1., 1., 1.],
             leadsV3=[], meta=N(laneChangeState='off'))

  class SM:
    def __init__(self, services, poll=None):
      self.updated, self.valid, self.alive = {}, dict.fromkeys(services, True), dict.fromkeys(services, True)
    def update(self, timeout):
      state['i'] += 1
      # camerad is one frame ahead of modeld, as in the recorded timing: frames i and i+1 are queued
      state['queue'] += [i for i in (state['i'], state['i'] + 1) if i not in state['sent']]
      state['sent'].update(state['queue'])
      self.updated = {'modelV2': True}
    def __getitem__(self, key):
      i = state['i']
      return {'modelV2': model_msg(i), 'liveCalibration': N(calStatus='calibrated', rpyCalib=list(rpy)),
              'carState': N(vEgo=10., leftBlinker=False, rightBlinker=False),
              'deviceState': N(started=True, deviceType='mici'), 'roadCameraState': N(sensor='os04c10')}[key]
  sent = []

  class Client:
    def __init__(self, name, stream, conflate):
      assert conflate is False  # every frame is received; the daemon's own buffer keeps the newest three
      self.frame_id = self.timestamp_eof = None
    def is_connected(self):
      return True
    def recv(self, timeout_ms=0):
      if not state['queue']:
        return None
      i = state['queue'].pop(0)
      self.frame_id, self.timestamp_eof = i, eof(i)
      return N(width=w, height=h, stride=w, uv_offset=w*h, data=nv12.tobytes(), frame_id=i)  # camerad stamps the slot
  messaging = types.SimpleNamespace(SubMaster=SM, PubMaster=lambda s: N(send=lambda name, msg: sent.append(json.loads(getattr(msg, name)))),
                                    new_message=lambda name, size, valid: N())
  # Fake the cereal package itself: the real one needs capnp schemas (and msgq) that are unavailable here.
  monkeypatch.setitem(sys.modules, 'openpilot.cereal', types.SimpleNamespace(messaging=messaging, __path__=[]))
  monkeypatch.setitem(sys.modules, 'openpilot.cereal.messaging', messaging)
  monkeypatch.setitem(sys.modules, 'openpilot.common.swaglog', N(cloudlog=N(warning=lambda *a: None)))
  monkeypatch.setitem(sys.modules, 'msgq.visionipc', N(VisionIpcClient=Client, VisionStreamType=N(VISION_STREAM_ROAD=0)))
  camera = N(intrinsics=np.array(k, float), width=w, height=h)
  import openpilot.common.transformations.camera as cam
  monkeypatch.setattr(cam, 'DEVICE_CAMERAS', {('mici', 'os04c10'): N(fcam=camera)})
  md.main()
  assert len(sent) >= 6 and sent[0]['mode'] == 'scanner' and sent[-1]['mode'] == 'stripnet'
  assert sent[0]['left']['color'] == 'unknown'  # first result is only pending in the display filter
  scanner = [p for p in sent if p['mode'] == 'scanner']
  assert any(p["left"]["color"] == "yellow" for p in scanner)
  assert all(p['timing']['processingMs'] >= 0 and p['counters']['frame_mismatch'] == 0 for p in sent)
  assert all(p['timing']['frameMatch'] == 'exact' and p['timing']['frameOffsetMs'] == 0 for p in sent)  # from the buffer
  assert all(v == 0 for p in sent for c, v in p['counters'].items() if c.startswith('drop_'))


def test_frame_buffer_exact_first_then_nearest_within_one_frame():
  from openpilot.selfdrive.carrot.lane_type.marking_daemon import FrameBuffer, line_dict
  ms = 1_000_000
  frames = [(10, 1000*ms), (11, 1050*ms), (12, 1100*ms), (13, 1150*ms)]

  class Client:
    def __init__(self):
      self.items, self.frame_id, self.timestamp_eof = list(frames), None, None
    def recv(self, timeout_ms=0):
      if not self.items:
        return None
      self.frame_id, self.timestamp_eof = self.items.pop(0)
      return object()
  fb = FrameBuffer()
  assert fb.drain(Client()) == 4 and [f[0] for f in fb.frames] == [11, 12, 13]  # oldest dropped
  assert fb.select(12, 1100*ms)[3] == 'exact'
  assert fb.select(12, 1101*ms)[:2] == (12, 1100*ms) and fb.select(12, 1101*ms)[3] == 'nearest'  # EOF differs: not exact
  assert fb.select(10, 1000*ms)[:2] == (11, 1050*ms)          # 50 ms: the neighbouring frame
  assert fb.select(9, 999*ms) is None                         # 51 ms: rejected
  assert fb.select(14, 1201*ms) is None
  fb.clear()                                                  # on VisionIPC reconnect
  assert fb.select(12, 1100*ms) is None
  assert line_dict(N(x=[5., 20.], y=[1., 1.], z=[0., 0.]), 1.5)['x'] == [3.5, 18.5]


def test_buffer_drop_reason_hold_time_and_slot_frame_id():
  from openpilot.selfdrive.carrot.lane_type.marking_daemon import MAX_HOLD_NS, buffer_drop_reason
  buf = N(frame_id=12)
  assert buffer_drop_reason(12, buf, 0, MAX_HOLD_NS - 1) is None
  assert buffer_drop_reason(12, buf, 0, MAX_HOLD_NS) == 'hold'      # held 0.5 s: dropped even with the same id
  assert buffer_drop_reason(11, buf, 0, 1) == 'frame_id'            # slot already republished with another frame


def test_lane_marking_channel_exists_in_the_real_cereal_schema(tmp_path):
  """card subscribes to and marking_daemon publishes customReservedRawData2: it must be a real Event member and service
  (a missing member made card crash at start on the device). Schema copied to an ASCII temp path for capnp."""
  import re
  import shutil
  capnp = pytest.importorskip('capnp')
  root = Path(__file__).resolve().parents[2]
  for directory in ('openpilot/cereal', 'opendbc_repo/opendbc/car'):
    for source in (root/directory).rglob('*.capnp'):
      target = tmp_path/source.relative_to(root)
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(source, target)
  c, car = tmp_path/'openpilot/cereal', tmp_path/'opendbc_repo/opendbc/car'
  log = capnp.SchemaParser().load(str(c/'log.capnp'), imports=[str(c), str(c/'include'), str(car), str(car/'include')])
  services = (root/'openpilot/cereal/services.py').read_text(encoding='utf-8')
  card = (root/'openpilot/selfdrive/car/card.py').read_text(encoding='utf-8')
  subscribed = re.findall(r"'(\w+)'", re.search(r'self\.sm = messaging\.SubMaster\(\[(.*?)\]', card, re.S).group(1))
  assert 'customReservedRawData2' in subscribed
  fields = log.Event.schema.fields
  for name in subscribed:
    assert f'"{name}":' in services and name in fields, name
  event = log.Event.new_message()
  event.init('customReservedRawData2', 26)  # what messaging.new_message(name, size=...) does
  event.customReservedRawData2 = b'{"type":"ccncLaneMarking"}'
  with log.Event.from_bytes(event.to_bytes()) as read:
    assert bytes(read.customReservedRawData2) == b'{"type":"ccncLaneMarking"}'
