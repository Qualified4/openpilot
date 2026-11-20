"""card's SubMaster keeps the bytes of the liveTracks and modelV2 messages it holds (for the CCNC display's Rust
readers): over real messaging sockets, the bytes are those of the messages sm['liveTracks'] and sm['modelV2'] are, and
every service's updated/valid/logMonoTime/seen are what the plain SubMaster gives for the same messages. The liveTracks
bytes then reach the display's copy through CarInterfaceBase.apply and create_ccnc_messages; the modelV2 bytes reach
CS.model_v2_bytes next to CS.modelV2."""
import random
import struct
from types import SimpleNamespace

import pytest

import openpilot.cereal.messaging as messaging
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.card import CarSubMaster
from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.hyundai import ccnc_core_loader, hyundaicanfd
from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ccnc_extension
from opendbc.car.hyundai import hyundaicanfd_ccnc_points as ccnc_points
from opendbc.car.hyundai.values import HyundaiFlags

SERVICES = ['carControl', 'radarState', 'modelV2', 'liveTracks', 'livePose']
NAMES = ('trackId', 'dRel', 'yRel', 'vRel', 'aRel', 'yvRel', 'measured', 'vLead', 'aLead', 'jLead', 'radarSource',
         'trackState', 'motionState')
SOURCES = ('frontRadar', 'scc', 'corner235', 'corner180', 'corner430')


class Captured(Exception):
  pass


def bits(value):
  return ('f', struct.pack('<d', value)) if isinstance(value, float) else (type(value).__name__, value)


def points_of(live):
  return [tuple(bits(str(p.radarSource) if n == 'radarSource' else getattr(p, n)) for n in NAMES) for p in live.points]


def live_tracks_message(rng):
  msg = messaging.new_message('liveTracks')
  msg.valid = rng.random() < .8
  msg.liveTracks = {'points': [{'trackId': rng.getrandbits(64), 'dRel': rng.uniform(0., 200.), 'yRel': rng.uniform(-10., 10.),
                                 'vRel': rng.uniform(-30., 30.), 'vLead': rng.uniform(0., 40.), 'measured': rng.random() < .7,
                                 'radarSource': rng.choice(SOURCES), 'trackState': rng.randrange(4), 'motionState': rng.randrange(3)}
                                for _ in range(rng.randrange(40))]}
  return msg


def model_v2_message(rng):
  msg = messaging.new_message('modelV2')
  msg.valid = rng.random() < .8
  xs = [float(x) for x in range(0, 99, 3)]

  def line():
    return {'x': xs, 'y': [rng.uniform(-6., 6.) + rng.gauss(0., .01) * x for x in xs], 'z': [0.] * len(xs)}

  msg.modelV2 = {'frameId': rng.getrandbits(32), 'laneLines': [line() for _ in range(4)],
                 'laneLineProbs': [rng.random() for _ in range(4)], 'roadEdges': [line() for _ in range(rng.choice((0, 2)))]}
  return msg


def model_lines(model):
  def lines(ls):
    return [(list(map(bits, line.x)), list(map(bits, line.y))) for line in ls]

  return lines(model.laneLines), list(map(bits, model.laneLineProbs)), lines(model.roadEdges)


def rust_model_lines(parsed):
  def lines(ls):
    return [(list(map(bits, x)), list(map(bits, y))) for x, y in ls]

  return lines(parsed[0]), list(map(bits, parsed[1])), lines(parsed[2])


def test_kept_bytes_are_the_message_the_submaster_holds(monkeypatch):
  update_msgs = messaging.SubMaster.update_msgs

  def update_msgs_at_frame_time(submaster, _now, msgs):
    # Use the same time for both receivers so scheduling cannot change their liveness or frequency comparison.
    update_msgs(submaster, 1.0 + (submaster.frame + 1) * 0.02, msgs)

  monkeypatch.setattr(messaging.SubMaster, 'update_msgs', update_msgs_at_frame_time)

  core = ccnc_core_loader.load()
  assert core is not None
  rng = random.Random(3)
  with OpenpilotPrefix(create_dirs_on_enter=True):
    pubs = {s: messaging.pub_sock(s) for s in SERVICES}
    sm, reference = CarSubMaster(SERVICES), messaging.SubMaster(SERVICES)
    sent = received = 0
    model_sent = model_received = 0
    for i in range(120):
      expected = model_expected = None
      published = False
      for s in SERVICES:
        if rng.random() < .5:
          continue
        # Sometimes two liveTracks in a row: the conflating socket gives the last one.
        for _ in range(2 if s in ('liveTracks', 'modelV2') and rng.random() < .2 else 1):
          msg = (live_tracks_message(rng) if s == 'liveTracks' else model_v2_message(rng) if s == 'modelV2'
                 else messaging.new_message(s, valid=rng.random() < .8))
          raw = msg.to_bytes()
          pubs[s].send(raw)
          published = True
          if s == 'liveTracks':
            expected, sent = raw, sent + 1
          elif s == 'modelV2':
            model_expected, model_sent = raw, model_sent + 1
      before, model_before = sm['liveTracks'], sm['modelV2']
      for m in (sm, reference):
        m.update(100 if published else 0)
      for name in ('updated', 'valid', 'logMonoTime', 'seen', 'alive', 'freq_ok', 'recv_frame'):
        assert getattr(sm, name) == getattr(reference, name), (i, name)
      assert sm.frame == reference.frame
      for s in SERVICES:
        assert str(sm[s]) == str(reference[s]), (i, s)
      kept = sm.live_tracks_bytes(sm['liveTracks'])
      if sm.updated['liveTracks']:
        assert kept == expected, i  # The last one sent, as the message the SubMaster holds.
        received += 1
      else:
        assert sm['liveTracks'] is before
      if kept is not None:
        assert points_of(messaging.log_from_bytes(kept).liveTracks) == points_of(sm['liveTracks'])
        rust = core._parse_for_test(kept)
        assert [tuple(bits(v) for v in p) for p in rust] == points_of(sm['liveTracks'])
      # Another object than the one held gets no bytes.
      assert sm.live_tracks_bytes(reference['liveTracks']) is None
      kept = sm.model_v2_bytes(sm['modelV2'])
      if sm.updated['modelV2']:
        assert kept == model_expected, i
        model_received += 1
      else:
        assert sm['modelV2'] is model_before
      if kept is not None:
        assert model_lines(messaging.log_from_bytes(kept).modelV2) == model_lines(sm['modelV2'])
        assert rust_model_lines(core._parse_model_for_test(kept)) == model_lines(sm['modelV2'])
      assert sm.model_v2_bytes(reference['modelV2']) is None and sm.model_v2_bytes(None) is None
  assert received > 30 and sent >= received
  assert model_received > 30 and model_sent >= model_received


def test_bytes_reach_the_display_copy(monkeypatch):
  """CarInterfaceBase.apply stores the bytes next to live_tracks; create_ccnc_messages hands the copy it makes from
  them to the display: the same points, values and types as the copy made from the objects."""
  monkeypatch.setattr(hyundaicanfd, "Params", lambda: SimpleNamespace(get_int=lambda key: 0, get=lambda key: "0",
                                                                      get_bool=lambda key: False))
  packer = CANPacker("hyundai_canfd_generated")
  rng = random.Random(9)
  seen = []

  def capture(values, CS, *args, **kwargs):
    seen.append(CS.live_tracks)
    raise Captured  # Only the copy matters here.

  monkeypatch.setattr(ccnc_extension, 'update_vehicles', capture)
  for _ in range(20):
    msg = live_tracks_message(rng)
    raw = msg.to_bytes()
    live = messaging.log_from_bytes(raw).liveTracks
    copies = []
    for kept in (raw, None):
      monkeypatch.setattr(ccnc_points, '_last', (None, None))
      cs = SimpleNamespace(radarState=None, scc_control=None, softHoldActive=0, paddle_button_prev=0, modelV2=None,
                           lfahda_cluster=None, cruise_buttons_msg=None, adrv_0x161=None, adrv_0x200=None, adrv_0x1ea=None,
                           ccnc_0x162=dict.fromkeys(packer.dbc.name_to_msg["CCNC_0x162"].sigs, 0),
                           out=SimpleNamespace(aEgo=0., vEgo=15., steeringAngleDeg=0., leftBlinker=False, rightBlinker=False,
                                               brakeHoldActive=False, parkingBrake=False,
                                               cruiseState=SimpleNamespace(available=True)))

      def update(c, CS, now_nanos):
        return hyundaicanfd.create_ccnc_messages(
          SimpleNamespace(flags=HyundaiFlags.CAMERA_SCC), packer, SimpleNamespace(ECAN=0, CAM=2), 5,
          SimpleNamespace(enabled=True, latActive=True), CS, structs.CarControl().hudControl, 0, False, False, 0, False, 0, 0,
          extended_ccnc=True)

      stub = SimpleNamespace(CS=cs, CC=SimpleNamespace(update=update))
      with pytest.raises(Captured):
        CarInterfaceBase.apply(stub, structs.CarControl(), 1, None, None, live, kept)
      assert cs.live_tracks_bytes is kept and cs.model_v2_bytes is None
      copies.append(seen.pop())
    from_bytes, from_objects = copies
    assert type(from_bytes) is type(from_objects) is ccnc_points.LiveTracks and from_bytes is not from_objects
    assert len(from_bytes.points) == len(from_objects.points) == len(live.points)
    assert points_of(from_bytes) == points_of(from_objects) == points_of(live)


@pytest.mark.parametrize('mismatch,passed', [(None, True), ('RadarPoint.dRel: schema 3, Rust reader 2', False), ('no Rust core', False)])
def test_no_bytes_are_passed_when_the_schema_differs(monkeypatch, mismatch, passed):
  """card asks once, when it starts, whether the Rust reader's offsets are this schema's; if not, the display gets no
  bytes and reads the objects."""
  monkeypatch.setattr(ccnc_points, 'schema_mismatch', lambda log, car: mismatch)
  with OpenpilotPrefix():
    sm = CarSubMaster(['liveTracks'])
    held = object()
    sm._live_tracks = (held, b'bytes')
    assert sm.live_tracks_bytes(held) == (b'bytes' if passed else None)
    assert sm.live_tracks_bytes(object()) is None


def test_a_failing_schema_check_passes_no_bytes(monkeypatch):
  def fail(log, car):
    raise RuntimeError('schema introspection failed')
  monkeypatch.setattr(ccnc_points, 'schema_mismatch', fail)
  with OpenpilotPrefix():
    sm = CarSubMaster(['liveTracks'])
    held = object()
    sm._live_tracks = (held, b'bytes')
    assert sm.live_tracks_bytes(held) is None


def test_model_v2_bytes_reach_the_car_state():
  """CarInterfaceBase.apply stores the modelV2 bytes next to CS.modelV2, where the map's update_vehicles reads them."""
  raw = model_v2_message(random.Random(2)).to_bytes()
  model = messaging.log_from_bytes(raw).modelV2
  cs = SimpleNamespace()
  stub = SimpleNamespace(CS=cs, CC=SimpleNamespace(update=lambda c, CS, now_nanos: (None, [])))
  CarInterfaceBase.apply(stub, structs.CarControl(), 1, model, None, None, None, raw)
  assert cs.modelV2 is model and cs.model_v2_bytes is raw
  CarInterfaceBase.apply(stub, structs.CarControl(), 1, model)
  assert cs.model_v2_bytes is None and cs.live_tracks_bytes is None


@pytest.mark.parametrize('mismatch,passed', [(None, True), ("XYZTData.x: schema (0, 'float32'), Rust reader (1, 'float32')", False),
                                             ('no Rust core', False), (RuntimeError('schema introspection failed'), False)])
def test_no_model_bytes_are_passed_when_the_schema_differs(monkeypatch, mismatch, passed):
  """As for liveTracks, separately: a modelV2 mismatch keeps the liveTracks bytes, and the other way round."""
  def check(log):
    if isinstance(mismatch, Exception):
      raise mismatch
    return mismatch
  monkeypatch.setattr(ccnc_points, 'model_schema_mismatch', check)
  monkeypatch.setattr(ccnc_points, 'schema_mismatch', lambda log, car: None)
  with OpenpilotPrefix():
    sm = CarSubMaster(['liveTracks', 'modelV2'])
    held, live = object(), object()
    sm._model_v2, sm._live_tracks = (held, b'model'), (live, b'live')
    assert sm.model_v2_bytes(held) == (b'model' if passed else None)
    assert sm.model_v2_bytes(object()) is None
    assert sm.live_tracks_bytes(live) == b'live'
