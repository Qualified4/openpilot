"""Native CAN packing through the live entry point; no socket or control writes."""
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
def test_native_live_entry_default_off_and_display_bit_allowlist(display, use_map):  # noqa: F811
  params, cs, _ = display
  params.update(CcncRadarVehicles=True, CcncModelLanes=2, CcncVehiclePositionCorrection=True, CcncRadarMap=use_map)
  msg = CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162']
  cs.ccnc_0x162 = dict.fromkeys(msg.sigs, 0)
  for key, value in {'COUNTER': 7, 'ALERTS_2': 3, 'VIBRATE': 1, 'BCA_LEFT': 1, 'BCA_RIGHT': 1}.items():
    if key in msg.sigs:
      cs.ccnc_0x162[key] = value
  cs.out.aEgo = 0.
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
      cs.modelV2 = geometry(frame*.01)
      cs.modelV2.meta.desire = N(raw=0)
      cs.modelV2.meta.desireState = []
      cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = True
      cs.modelV2.position.yStd = [0.]*4
      cs.live_tracks = N(points=[N(trackId=21, dRel=3.1-.1*i, yRel=3.5, vRel=-2., vLead=8., yvRel=0.,
        radarSource='frontRadar', measured=True, trackState=2, motionState=2)] if i <= 20 else [])
      values = main.create_ccnc_messages(N(flags=HyundaiFlags.CAMERA_SCC.value), packer, N(ECAN=0, CAM=2), frame,
        N(enabled=True, latActive=True), cs, hud, 0., False, False, 0, False, 0, 0, rear_handover=enabled)
      results.append(values)
      if enabled:
        handoffs += sum(e['event'] == 'handoff' for e in ext.state.rear_handover.events)
        events.extend(ext.state.rear_handover.events)
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
  for old_messages, new_messages in zip(before, after, strict=True):
    assert [(a, b) for a, _, b in old_messages] == [(a, b) for a, _, b in new_messages]
    for (address, old, _), (_, new, _) in zip(old_messages, new_messages, strict=True):
      if address != 0x162:
        assert old == new
        continue
      changed += old != new
      front_detect = get_raw_value(new, msg.sigs['LF_DETECT'])
      rear_detect = get_raw_value(new, msg.sigs['LR_DETECT'])
      if rear_detect and not front_detect and previous_front is not None:
        assert previous_front <= .2, 'handoff skipped the last drawn front distance'
      previous_front = get_raw_value(new, msg.sigs['LF_DETECT_DISTANCE'])*.1 if front_detect else None
      for key, signal in msg.sigs.items():
        if key not in allowed:
          assert get_raw_value(old, signal) == get_raw_value(new, signal), key
  assert handoffs == 1 and changed > 0, events
