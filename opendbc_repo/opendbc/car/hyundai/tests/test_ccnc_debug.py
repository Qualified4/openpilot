import copy
import math
from types import SimpleNamespace as NS

import pytest

from opendbc.can import CANPacker, native
from opendbc.can.parser import get_raw_value
from opendbc.car import structs
from opendbc.car.hyundai import ccnc_debug as debug, hyundaicanfd
from opendbc.car.hyundai.values import HyundaiFlags


def selected():
  config = debug.defaults()
  config.update(transmit=True, vehicles_enabled=True, lanes_enabled=True, curve_enabled=True, curvature=15, direction=1)
  for entry in config['vehicles'].values():
    entry['enabled'] = True
  for entry in config['lanes'].values():
    entry.update(enabled=True, display=14, position=63)
  return config


def vehicle(now=100.):
  out = structs.CarState(canValid=True, canTimeout=False, standstill=True, gearShifter=structs.CarState.GearShifter.park)
  return NS(flags=HyundaiFlags.CAMERA_SCC), structs.CarControl(), NS(
    out=out, cp_cam=NS(ts_nanos={name: {'COUNTER': int(now * 1e9)} for name in debug.MESSAGES.values()}),
    trailer_connected=False)


def lease(tmp_path, config=None, now=100., session='arm1'):
  live = debug.DebugOverride(tmp_path / 'command', tmp_path / 'status')
  debug.write_json(live.command_path, {'session': session, 'updated_at': now, 'config': config or selected()})
  return live


@pytest.mark.parametrize('slot', debug.SLOTS)
@pytest.mark.parametrize('kind', ('min', 'max'))
def test_slot_raw_limits_and_motorola_rear_distance(slot, kind):
  config = selected()
  distance_max = 25.5 if slot in ('LR', 'RR') else 204.7
  entry = config['vehicles'][slot]
  entry.update(distance=0 if kind == 'min' else distance_max,
               lateral=(-6.4 if kind == 'min' else 6.3) if slot == 'FF' else (0 if kind == 'min' else 12.7))
  result = debug.preview(config)['CCNC_0x162']['fields']
  _, distance, lateral = debug.vehicle_fields(slot)
  assert result[distance] == {'raw': 0 if kind == 'min' else (255 if slot in ('LR', 'RR') else 2047), 'physical': entry['distance']}
  assert result[lateral] == {'raw': (64 if kind == 'min' else 63) if slot == 'FF' else (0 if kind == 'min' else 127), 'physical': entry['lateral']}
  assert debug.schema()['signals']['CCNC_0x162']['LR_DETECT_DISTANCE']['little_endian'] is False


@pytest.mark.parametrize('value', [-6.4, -6.3, -3.1, -.1, 0., .1, 6.3])
def test_ff_wrap_is_explicit_while_dbc_is_unsigned(value):
  config = selected()
  config['vehicles']['FF']['lateral'] = value
  result = debug.preview(config)['CCNC_0x162']['fields']['FF_LATERAL']
  assert result == {'raw': round(value * 10) % 128, 'physical': value}
  assert not debug.schema()['signals']['CCNC_0x162']['FF_LATERAL']['signed']


@pytest.mark.parametrize('slot', debug.SLOTS)
@pytest.mark.parametrize('field,value', [('detect', 32), ('detect', True), ('distance', -.1), ('distance', 10.05),
                                        ('lateral', 12.8), ('lateral', math.nan), ('lateral', math.inf)])
def test_invalid_vehicle_input_rejected(slot, field, value):
  config = selected()
  config['vehicles'][slot][field] = value
  with pytest.raises(ValueError):
    debug.preview(config)


@pytest.mark.parametrize('slot,field,value', [('FF', 'lateral', 6.4), ('FF', 'lateral', -6.5),
                                            ('LF', 'lateral', -.1), ('RF', 'lateral', -.1),
                                            ('LR', 'distance', 25.6), ('RR', 'distance', 25.6)])
def test_slot_specific_overflow_rejected(slot, field, value):
  config = selected()
  config['vehicles'][slot][field] = value
  with pytest.raises(ValueError):
    debug.preview(config)


@pytest.mark.parametrize('code', debug.LANE_CODES)
def test_lane_codes_independent_positions_and_shared_curve(code):
  config = selected()
  config['lanes']['left'].update(display=code, position=63)
  config['lanes']['right'].update(display=2, position=0)
  result = debug.preview(config)['ADRV_0x161']['fields']
  assert result['LANELINE_LEFT']['raw'] == code
  assert result['LANELINE_RIGHT']['raw'] == 2
  assert result['LANELINE_LEFT_POSITION']['raw'] == 63
  assert result['LANELINE_RIGHT_POSITION']['raw'] == 0
  assert result['LANELINE_CURVATURE']['raw'] == 15
  assert result['LANELINE_CURVATURE_DIRECTION']['raw'] == 1
  codes = {entry['value']: entry for entry in debug.schema()['lane_codes']}
  assert codes[code]['documented'] is (code in (1, 2, 4, 6))


@pytest.mark.parametrize('path,value', [('curvature', 16), ('direction', 2), ('transmit', 1), ('curve_enabled', 'true')])
def test_invalid_global_values(path, value):
  config = selected()
  config[path] = value
  with pytest.raises(ValueError):
    debug.preview(config)


def test_allowlist_groups_and_slots_are_independent():
  config = selected()
  all_fields = debug.overrides(config)
  for slot in debug.SLOTS:
    config['vehicles'][slot]['enabled'] = False
    assert debug.overrides(config)['CCNC_0x162'].keys() == all_fields['CCNC_0x162'].keys() - set(debug.vehicle_fields(slot))
    config['vehicles'][slot]['enabled'] = True
  config['vehicles_enabled'] = False
  assert debug.overrides(config) == {'ADRV_0x161': all_fields['ADRV_0x161'], 'CCNC_0x162': {}}
  config['vehicles_enabled'], config['lanes_enabled'] = True, False
  assert debug.overrides(config) == {'ADRV_0x161': {}, 'CCNC_0x162': all_fields['CCNC_0x162']}
  config['COUNTER'] = 0
  with pytest.raises(ValueError):
    debug.validate(config)


@pytest.mark.parametrize('failure', ['canValid', 'canTimeout', 'standstill', 'vEgo', 'vEgoRaw', 'wheel', 'gear',
                                    'gas', 'control', 'lat', 'long', 'rx', 'rx_future', 'warning', 'hda2', 'direct', 'blindspot'])
def test_unsafe_state_reverts_and_requires_new_arm(tmp_path, failure):
  live = lease(tmp_path)
  cp, cc, cs = vehicle()
  assert live.poll(cp, cc, cs, now=100.) == debug.overrides(selected())
  if failure in ('canValid', 'standstill'):
    setattr(cs.out, failure, False)
  elif failure == 'canTimeout':
    cs.out.canTimeout = True
  elif failure in ('vEgo', 'vEgoRaw'):
    setattr(cs.out, failure, .01)
  elif failure == 'wheel':
    cs.out.wheelSpeeds.fl = .01
  elif failure == 'gear':
    cs.out.gearShifter = structs.CarState.GearShifter.drive
  elif failure == 'gas':
    cs.out.gasPressed = True
  elif failure == 'control':
    cc.enabled = True
  elif failure == 'lat':
    cc.latActive = True
  elif failure == 'long':
    cc.longActive = True
  elif failure == 'blindspot':
    cs.out.leftBlindspot = True
  elif failure == 'hda2':
    cp.flags |= HyundaiFlags.CANFD_HDA2
  elif failure == 'direct':
    cp.flags |= HyundaiFlags.CANFD_CLUSTER_DIRECT_TX
  elif failure.startswith('rx'):
    cs.cp_cam.ts_nanos['ADRV_0x161']['COUNTER'] = int((99.8 if failure == 'rx' else 101.) * 1e9)
  assert live.poll(cp, cc, cs, warning=failure == 'warning', now=100.) == {}
  cp, cc, cs = vehicle()
  assert live.poll(cp, cc, cs, now=100.) == {}  # Returning to P/valid CAN does not rearm.
  assert debug.read_json(live.status_path)['disarmed']
  debug.write_json(live.command_path, {'session': 'arm2', 'updated_at': 100., 'config': selected()})
  assert live.poll(cp, cc, cs, now=100.)


def test_neutral_requires_brake_and_preview_never_applies(tmp_path):
  live = lease(tmp_path)
  cp, cc, cs = vehicle()
  cs.out.gearShifter = structs.CarState.GearShifter.neutral
  cs.out.brakePressed = True
  assert live.poll(cp, cc, cs, now=100.)
  cs.out.brakePressed = False
  assert live.poll(cp, cc, cs, now=100.) == {}
  config = selected()
  config['transmit'] = False
  debug.write_json(live.command_path, {'session': 'preview', 'updated_at': 100., 'config': config})
  assert live.poll(cp, cc, cs, now=100.) == {}


@pytest.mark.parametrize('mode', ['expiry', 'future', 'off', 'missing', 'malformed', 'nan'])
def test_lease_failures_revert(tmp_path, mode):
  live = lease(tmp_path)
  cp, cc, cs = vehicle()
  assert live.poll(cp, cc, cs, now=100.)
  if mode == 'missing':
    live.command_path.unlink()
  elif mode == 'malformed':
    live.command_path.write_text('{')
  elif mode == 'nan':
    live.command_path.write_text('{"updated_at":NaN}')
  else:
    config = selected()
    config['transmit'] = mode != 'off'
    debug.write_json(live.command_path, {'session': 'arm1', 'updated_at': 99. if mode == 'expiry' else 101. if mode == 'future' else 100., 'config': config})
  assert live.poll(cp, cc, cs, now=100.) == {}


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_existing_generator_only_changes_allowlisted_bits_and_crc(monkeypatch, tmp_path, backend):
  if backend == 'native' and native.pack is None:
    pytest.skip('native CAN packer not built')
  if backend == 'python':
    monkeypatch.setattr(native, 'pack', None)
  monkeypatch.setattr(hyundaicanfd, 'Params', lambda: NS(get_int=lambda key: 0))
  cp, cc, cs = vehicle()
  packer = CANPacker('hyundai_canfd_generated')
  for name, attr in [('ADRV_0x161', 'adrv_0x161'), ('CCNC_0x162', 'ccnc_0x162'), ('ADRV_0x200', 'adrv_0x200'), ('ADRV_0x1ea', 'adrv_0x1ea')]:
    setattr(cs, attr, {key: 0 for key in packer.dbc.name_to_msg[name].sigs})
    getattr(cs, attr)['COUNTER'] = 254
  cs.ccnc_0x162.update(FAULT_FCA=5, SPEEDLIMIT=80, FF_DETECT_ALT=2, FF_DISTANCE_ALT=21.3)
  cs.adrv_0x161.update(FCA_ICON=3, LANE_HIGHLIGHT=5)
  cs.__dict__.update(modelV2=None, radarState=None, lfahda_cluster=None, cruise_buttons_msg=None,
                     is_metric=True, paddle_button_prev=0, softHoldActive=0)
  original = copy.deepcopy((cs.adrv_0x161, cs.ccnc_0x162))
  live = lease(tmp_path)
  monkeypatch.setattr(debug, 'live', live)
  monkeypatch.setattr(debug.time, 'monotonic', lambda: 100.)

  def send(frame, warning=False):
    return hyundaicanfd.create_ccnc_messages(cp, CANPacker('hyundai_canfd_generated'), NS(ECAN=0, CAM=2), frame,
                                           cc, cs, cc.hudControl, 0, warning, False, 0, False, 0, 0, extended_ccnc=False)
  live.command_path.unlink()
  normal = send(5)
  debug.write_json(live.command_path, {'session': 'arm1', 'updated_at': 100., 'config': selected()})
  active = send(5)
  assert len(active) == len(normal) == 4
  changed = debug.overrides(selected())
  for actual, before in zip(active, normal, strict=False):
    addr, data, bus = actual
    assert (addr, bus) == (before[0], before[2])
    msg = packer.dbc.addr_to_msg[addr]
    if msg.name not in changed:
      assert actual == before
      continue
    mask = bytearray(32)
    from opendbc.can.packer import set_value
    for key in changed[msg.name]:
      set_value(mask, msg.sigs[key], (1 << msg.sigs[key].size) - 1)
    mask[0:2] = b'\xff\xff'
    assert all(((a ^ b) & ~m) == 0 for a, b, m in zip(data, before[1], mask, strict=False))
    assert data[2] == before[1][2] == 255
    assert int.from_bytes(data[:2], 'little') == hyundaicanfd.hkg_can_fd_checksum(addr, None, bytearray(data))
    for key, value in changed[msg.name].items():
      sig = msg.sigs[key]
      assert get_raw_value(data, sig) == round(value / sig.factor) % (1 << sig.size)
  assert send(6) == []  # Never adds a new schedule.
  warning_output = send(5, warning=True)
  assert send(5) == normal
  live.command_path.unlink()
  assert warning_output == send(5, warning=True)
  assert (cs.adrv_0x161, cs.ccnc_0x162) == original
