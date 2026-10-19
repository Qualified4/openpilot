"""Native CAN packing through the live entry point; no socket or control writes."""
import binascii
import copy
from types import SimpleNamespace as N

import pytest

from opendbc.can import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car import structs
from opendbc.car.hyundai import hyundaicanfd as main
from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ext
from opendbc.car.hyundai import hyundaicanfd_ccnc_radar_map as radar_map
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401
from opendbc.car.hyundai.values import HyundaiFlags
from tools.ccnc.test_radar_map_integration import geometry  # noqa: TID251


@pytest.mark.parametrize('use_map', [False, True])
@pytest.mark.parametrize('bsd_interrupt', [False, True])
@pytest.mark.parametrize('enhanced', [False, True])
def test_native_live_entry_radar_option_enables_handover_and_display_bit_allowlist(display, use_map, bsd_interrupt, enhanced):  # noqa: F811
  params, cs, _ = display
  params.update(CcncRadarVehicles=True, CcncModelLanes=2 if enhanced else 0,
                CcncVehiclePositionCorrection=enhanced, CcncRadarMap=use_map)
  msg = CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162']
  cs.ccnc_0x162 = dict.fromkeys(msg.sigs, 0)
  for key, value in {'COUNTER': 7, 'ALERTS_2': 3, 'VIBRATE': 1, 'BCA_LEFT': 1, 'BCA_RIGHT': 1}.items():
    if key in msg.sigs:
      cs.ccnc_0x162[key] = value
  cs.out.aEgo = 0.
  cs.out.vEgo = 10.  # the track's vLead 8 = vEgo + vRel -2
  cs.out.canValid = True
  raw_source = copy.deepcopy(cs.ccnc_0x162)
  hud = structs.CarControl().hudControl
  hud.leftLaneVisible = hud.rightLaneVisible = True

  def replay(enabled):
    ext.reset()
    radar_map.reset_live()
    for name in ('_display_options', '_radar_map'):
      if hasattr(main.create_ccnc_messages, name):
        delattr(main.create_ccnc_messages, name)
    packer = CANPacker('hyundai_canfd_generated')
    results, handoffs, events = [], 0, []
    for i in range(110):
      frame = 5*i
      cs.out.leftBlindspot = bsd_interrupt and 40 <= i < 50
      cs.modelV2 = geometry(frame*.01)
      cs.modelV2.meta.desire = N(raw=0)
      cs.modelV2.meta.desireState = []
      cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = True
      cs.modelV2.position.yStd = [0.]*4
      cs.live_tracks = N(points=[N(trackId=21, dRel=3.1-.1*i, yRel=3.5, vRel=-2., vLead=8., yvRel=0.,
        radarSource='frontRadar', measured=True, trackState=2, motionState=2)] if i <= 20 else [])
      values = main.create_ccnc_messages(N(flags=HyundaiFlags.CAMERA_SCC.value), packer, N(ECAN=0, CAM=2), frame,
        N(enabled=True, latActive=True), cs, hud, 0., False, False, 0, False, 0, 0,
        **({} if enabled else {'rear_handover': False}))
      results.append(values)
      if enabled:
        handoffs += sum(e['event'] == 'handoff' for e in ext.state.rear_handover.events)
        events.extend(ext.state.rear_handover.events)
        if cs.out.leftBlindspot:
          assert ext.state.rear_handover.fronts == ext.state.rear_handover.rears == [None, None]
      else:
        assert ext.state.rear_handover is None
      assert cs.ccnc_0x162 == raw_source
    return results, handoffs, events

  before, _, _ = replay(False)
  after, handoffs, events = replay(True)
  allowed = {f'{slot}_{suffix}' for slot in ('LF_DETECT', 'RF_DETECT', 'LR_DETECT', 'RR_DETECT') for suffix in ('DISTANCE', 'LATERAL')}
  allowed |= {'LF_DETECT', 'RF_DETECT', 'LR_DETECT', 'RR_DETECT', 'CHECKSUM'}
  changed = 0
  previous_front = None
  previous_counter = None
  previous_rear = 0
  for i, (old_messages, new_messages) in enumerate(zip(before, after, strict=True)):
    assert [(a, b) for a, _, b in old_messages] == [(a, b) for a, _, b in new_messages]
    for (address, old, _), (_, new, _) in zip(old_messages, new_messages, strict=True):
      if address != 0x162:
        assert old == new
        continue
      changed += old != new
      for data in (old, new):
        crc = binascii.crc_hqx(data[2:] + address.to_bytes(2, 'little'), 0) ^ 0x9F5B
        assert int.from_bytes(data[:2], 'little') == crc
      if previous_counter is not None:
        assert new[2] == (previous_counter+1) % 256
      previous_counter = new[2]
      front_detect = get_raw_value(new, msg.sigs['LF_DETECT'])
      rear_detect = get_raw_value(new, msg.sigs['LR_DETECT'])
      if rear_detect and not previous_rear and not (bsd_interrupt and 40 <= i < 50):
        # No gap: LR appears in the very send in which the car leaves LF (DaBin 10-19), at 0.
        assert previous_front is not None and not front_detect, 'the handed-over car vanished from every slot'
        assert get_raw_value(new, msg.sigs['LR_DETECT_DISTANCE']) == 0
      previous_front = get_raw_value(new, msg.sigs['LF_DETECT_DISTANCE'])*.1 if front_detect else None
      previous_rear = rear_detect
      for key, signal in msg.sigs.items():
        if key not in allowed:
          assert get_raw_value(old, signal) == get_raw_value(new, signal), key
  assert handoffs == 1 and changed > 0, events


@pytest.mark.parametrize('use_map', [False, True])
@pytest.mark.parametrize('radar_enabled,hda2', [(False, False), (True, False), (True, True)])
def test_radar_option_and_hda2_gate_clear_handover_memory(display, use_map, radar_enabled, hda2):  # noqa: F811
  params, _, send = display
  params.update(CcncRadarVehicles=radar_enabled, CcncRadarMap=use_map)
  flags = HyundaiFlags.CAMERA_SCC.value | (HyundaiFlags.CANFD_HDA2.value if hda2 else 0)
  send(0, flags=flags)
  policy = ext.state.rear_handover
  assert (policy is not None) == (radar_enabled and not hda2)
  if policy is not None:
    policy.fronts[0] = {'key': (1, 0)}
    policy.rears[1] = {'key': (2, 0)}
    params['CcncRadarVehicles'] = False
    send(100, flags=flags)
    assert ext.state.rear_handover is None
    params['CcncRadarVehicles'] = True
    send(200, flags=flags)
    assert ext.state.rear_handover is not policy
    assert ext.state.rear_handover.fronts == ext.state.rear_handover.rears == [None, None]


@pytest.mark.parametrize('use_map', [False, True])
def test_rear_exception_skips_only_the_rear_memory(display, monkeypatch, use_map):  # noqa: F811
  """A rear handover exception must not replace the vehicle display (map reset / fixed fallback values)."""
  params, cs, _ = display
  params.update(CcncRadarVehicles=True, CcncModelLanes=2, CcncVehiclePositionCorrection=True, CcncRadarMap=use_map)
  msg = CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162']
  cs.ccnc_0x162 = dict.fromkeys(msg.sigs, 0)
  cs.out.aEgo = 0.
  cs.out.vEgo = 10.
  cs.out.canValid = True
  hud = structs.CarControl().hudControl
  hud.leftLaneVisible = hud.rightLaneVisible = True
  logged = []
  for module in (ext, radar_map):
    monkeypatch.setattr(module, 'carlog', N(exception=lambda *a, **k: logged.append(a)), raising=False)

  def replay(rear):
    ext.reset()
    radar_map.reset_live()
    for name in ('_display_options', '_radar_map'):
      if hasattr(main.create_ccnc_messages, name):
        delattr(main.create_ccnc_messages, name)
    packer = CANPacker('hyundai_canfd_generated')
    out = []
    for i in range(40):
      frame = 5*i
      cs.modelV2 = geometry(frame*.01)
      cs.modelV2.meta.desire = N(raw=0)
      cs.modelV2.meta.desireState = []
      cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = True
      cs.modelV2.position.yStd = [0.]*4
      cs.live_tracks = N(points=[N(trackId=21, dRel=20.-.2*i, yRel=3.5, vRel=-2., vLead=8., yvRel=0.,
        radarSource='frontRadar', measured=True, trackState=2, motionState=2)])
      messages = main.create_ccnc_messages(N(flags=HyundaiFlags.CAMERA_SCC.value), packer, N(ECAN=0, CAM=2), frame,
        N(enabled=True, latActive=True), cs, hud, 0., False, False, 0, False, 0, 0, rear_handover=rear)
      out.append([data for address, data, _ in messages if address == 0x162])
      if use_map:
        assert radar_map._live.tracker is not None
    return out

  baseline = replay(False)
  monkeypatch.setattr(ext.RearHandover, 'update', lambda self, *a, **k: (_ for _ in ()).throw(TypeError('an integer is required')))
  assert replay(True) == baseline
  assert len(logged) == 1
