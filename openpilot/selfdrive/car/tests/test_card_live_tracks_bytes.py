"""card's SubMaster keeps the bytes of the liveTracks message it holds (for the CCNC display's Rust reader): over real
messaging sockets, the bytes are those of the message sm['liveTracks'] is, and every service's updated/valid/
logMonoTime/seen are what the plain SubMaster gives for the same messages. The bytes then reach the display's copy
through CarInterfaceBase.apply and create_ccnc_messages."""
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


def test_kept_bytes_are_the_message_the_submaster_holds():
  core = ccnc_core_loader.load()
  assert core is not None
  rng = random.Random(3)
  with OpenpilotPrefix(create_dirs_on_enter=True):
    pubs = {s: messaging.pub_sock(s) for s in SERVICES}
    sm, reference = CarSubMaster(SERVICES), messaging.SubMaster(SERVICES)
    sent = received = 0
    for i in range(120):
      expected = None
      published = False
      for s in SERVICES:
        if rng.random() < .5:
          continue
        # Sometimes two liveTracks in a row: the conflating socket gives the last one.
        for _ in range(2 if s == 'liveTracks' and rng.random() < .2 else 1):
          msg = live_tracks_message(rng) if s == 'liveTracks' else messaging.new_message(s, valid=rng.random() < .8)
          raw = msg.to_bytes()
          pubs[s].send(raw)
          published = True
          if s == 'liveTracks':
            expected, sent = raw, sent + 1
      before = sm['liveTracks']
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
  assert received > 30 and sent >= received


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
      assert cs.live_tracks_bytes is kept
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
