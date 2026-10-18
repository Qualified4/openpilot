"""Ephemeral CCNC display overrides. No CAN socket, Params, or control messages.

The web process owns a short lease in tmpfs; the existing CCNC generator checks
it against current vehicle/RX state before changing its normal display values.
"""
import json
import math
import os
from pathlib import Path
import tempfile
import time

from opendbc.car.hyundai.values import HyundaiFlags

COMMAND_PATH = Path('/dev/shm/carrot-ccnc-debug.json')
STATUS_PATH = Path('/dev/shm/carrot-ccnc-debug-status.json')
LEASE_SECONDS = 0.6
# 20 Hz messages; a few late card cycles are not a lost camera.
RX_SECONDS = 0.5
# Output stops at once for any unsafe state. Only a change of the vehicle's own
# state that lasts this long ends the session (manual re-enable); a flickering
# signal, a late frame or a slow browser only pauses it.
LATCH_REASONS = frozenset(('vehicle moving', 'select P, or N with brake held', 'controls active'))
LATCH_SECONDS = 0.5
MESSAGES = {'lanes': 'ADRV_0x161', 'vehicles': 'CCNC_0x162'}
SLOTS = ('LF', 'FF', 'RF', 'LR', 'RR')
LANE_CODES = (1, 2, 4, 6, 8, 10, 14)


def vehicle_fields(slot):
  prefix = slot if slot == 'FF' else slot + '_DETECT'
  return slot + '_DETECT', prefix + '_DISTANCE', prefix + '_LATERAL'


def fields():
  return {
    'vehicles': {key for slot in SLOTS for key in vehicle_fields(slot)},
    'lanes': {'LANELINE_LEFT', 'LANELINE_RIGHT', 'LANELINE_LEFT_POSITION',
              'LANELINE_RIGHT_POSITION', 'LANELINE_CURVATURE', 'LANELINE_CURVATURE_DIRECTION'},
  }


def defaults():
  return {'transmit': False, 'vehicles_enabled': False, 'lanes_enabled': False,
          'vehicles': {slot: {'enabled': False, 'detect': 6, 'distance': 10., 'lateral': 0. if slot == 'FF' else 3.}
                       for slot in SLOTS},
          'lanes': {side: {'enabled': False, 'display': 2, 'position': 15} for side in ('left', 'right')},
          'curve_enabled': False, 'curvature': 0, 'direction': 0}


def number(value, low, high, step):
  if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
    raise ValueError(f'expected finite number in [{low}, {high}]')
  quantized = round(value / step)
  if abs(value / step - quantized) > 1e-7:
    raise ValueError(f'expected step {step}')
  return round(quantized * step, 8)


def validate(config):
  """Reject unknown keys, non-booleans, wrapping and off-grid input."""
  def structure(value, template):
    if not isinstance(value, dict) or value.keys() != template.keys():
      raise ValueError('unexpected or missing fields')
    for key, default in template.items():
      if isinstance(default, dict):
        structure(value[key], default)
      elif isinstance(default, bool) and type(value[key]) is not bool:
        raise ValueError(f'{key} must be boolean')
  structure(config, defaults())
  for slot, entry in config['vehicles'].items():
    number(entry['detect'], 0, 31, 1)
    number(entry['distance'], 0, 25.5 if slot in ('LR', 'RR') else 204.7, .1)
    number(entry['lateral'], -6.4 if slot == 'FF' else 0, 6.3 if slot == 'FF' else 12.7, .1)
  for entry in config['lanes'].values():
    number(entry['display'], 0, 15, 1)
    if entry['display'] not in LANE_CODES:
      raise ValueError('unsupported lane display code')
    number(entry['position'], 0, 63, 1)
  number(config['curvature'], 0, 15, 1)
  number(config['direction'], 0, 1, 1)
  return config


def overrides(config):
  validate(config)
  result = {name: {} for name in MESSAGES.values()}
  if config['vehicles_enabled']:
    for slot, entry in config['vehicles'].items():
      if entry['enabled']:
        result[MESSAGES['vehicles']].update(zip(vehicle_fields(slot), (entry['detect'], entry['distance'], entry['lateral']), strict=False))
  if config['lanes_enabled']:
    for side, entry in config['lanes'].items():
      if entry['enabled']:
        result[MESSAGES['lanes']].update({f'LANELINE_{side.upper()}': entry['display'],
                                        f'LANELINE_{side.upper()}_POSITION': entry['position']})
    if config['curve_enabled']:
      result[MESSAGES['lanes']].update(LANELINE_CURVATURE=config['curvature'], LANELINE_CURVATURE_DIRECTION=config['direction'])
  return result


def preview(config):
  """Use the production packer; round trip every selected field before leasing."""
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value
  packer = CANPacker('hyundai_canfd_generated')
  result = {}
  for name, values in overrides(config).items():
    if not values:
      continue
    msg = packer.dbc.name_to_msg[name]
    _, data, _ = packer.make_can_msg(name, 0, values)
    decoded = {}
    for key, value in values.items():
      sig = msg.sigs[key]
      raw = get_raw_value(data, sig)
      # DBC declares FF unsigned; the existing generator encodes negative FF
      # positions by wrapping a 7-bit two's-complement value. Side slots do not.
      physical_raw = raw - 128 if key == 'FF_LATERAL' and raw >= 64 else raw
      physical = round(physical_raw * sig.factor + sig.offset, 8)
      if abs(physical - value) > 1e-7:
        raise ValueError(f'DBC round trip failed: {key}')
      decoded[key] = {'raw': raw, 'physical': physical}
    result[name] = {'address': msg.address, 'fields': decoded, 'example_hex': data.hex()}
  return result


def schema():
  from opendbc.can import CANDefine, CANPacker
  packer = CANPacker('hyundai_canfd_generated')
  enums = CANDefine('hyundai_canfd_generated').dv
  signals = {}
  for group, name in MESSAGES.items():
    msg = packer.dbc.name_to_msg[name]
    if msg.size != 32 or msg.address != (0x161 if group == 'lanes' else 0x162):
      raise ValueError('unexpected CCNC message layout')
    signals[name] = {key: {'start_bit': sig.start_bit, 'bits': sig.size, 'factor': sig.factor,
                           'offset': sig.offset, 'signed': sig.is_signed, 'little_endian': sig.is_little_endian}
                     for key in sorted(fields()[group]) for sig in [msg.sigs[key]]}
  # Fail closed if a future DBC changes the conversion contract of the UI.
  starts = {'LF': (112, 117, 128), 'FF': (64, 69, 80), 'RF': (136, 141, 152),
            'LR': (163, 175, 176), 'RR': (192, 197, 205)}
  for slot in SLOTS:
    for key, bits, factor, start in zip(vehicle_fields(slot), (5, 8 if slot in ('LR', 'RR') else 11, 7),
                                         (1, .1, .1), starts[slot], strict=True):
      sig = packer.dbc.name_to_msg[MESSAGES['vehicles']].sigs[key]
      little = key != 'LR_DETECT_DISTANCE'
      if (sig.start_bit, sig.size, sig.factor, sig.offset, sig.is_signed, sig.is_little_endian) != (start, bits, factor, 0, False, little):
        raise ValueError(f'unsupported DBC layout: {key}')
  starts = {'LANELINE_LEFT': 80, 'LANELINE_LEFT_POSITION': 84, 'LANELINE_RIGHT': 90,
            'LANELINE_RIGHT_POSITION': 94, 'LANELINE_CURVATURE': 100, 'LANELINE_CURVATURE_DIRECTION': 104}
  for key in fields()['lanes']:
    sig = packer.dbc.name_to_msg[MESSAGES['lanes']].sigs[key]
    bits = 6 if key.endswith('POSITION') else 1 if key.endswith('DIRECTION') else 4
    little = not key.endswith('DIRECTION')
    if (sig.start_bit, sig.size, sig.factor, sig.offset, sig.is_signed, sig.is_little_endian) != (starts[key], bits, 1, 0, False, little):
      raise ValueError(f'unsupported DBC layout: {key}')
  return {'defaults': defaults(), 'lease_seconds': LEASE_SECONDS, 'signals': signals,
          'lane_codes': [{'value': n, 'label': enums['ADRV_0x161']['LANELINE_LEFT'].get(n, 'UNDOCUMENTED'),
                          'documented': n in enums['ADRV_0x161']['LANELINE_LEFT']} for n in LANE_CODES],
          'detect_codes': enums['CCNC_0x162']['FF_DETECT']}


def write_json(path, value):
  """Atomic, private, bounded local IPC; never persist debug values across boots."""
  path = Path(path)
  fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
  try:
    with os.fdopen(fd, 'w') as stream:
      json.dump(value, stream, allow_nan=False, separators=(',', ':'))
    os.replace(temporary, path)
  finally:
    Path(temporary).unlink(missing_ok=True)


def read_json(path):
  with Path(path).open() as stream:
    raw = stream.read(16385)
  if len(raw) > 16384:
    raise ValueError('debug message too large')
  return json.loads(raw)


def safety_reason(CP, CC, CS, now):
  if not CP.flags & HyundaiFlags.CAMERA_SCC or CP.flags & HyundaiFlags.CANFD_HDA2:
    return 'HDA1 CAMERA_SCC required'
  if CP.flags & HyundaiFlags.CANFD_CLUSTER_DIRECT_TX:
    return 'RX-based cluster forwarding required'
  out = CS.out
  if not getattr(out, 'canValid', False) or getattr(out, 'canTimeout', True):
    return 'CAN unavailable'
  if CC.enabled or CC.latActive or getattr(CC, 'longActive', False):
    return 'controls active'
  speeds = [getattr(out, key, math.nan) for key in ('vEgo', 'vEgoRaw')]
  wheels = getattr(out, 'wheelSpeeds', None)
  speeds += [getattr(wheels, key, math.nan) for key in ('fl', 'fr', 'rl', 'rr')]
  if not getattr(out, 'standstill', False) or any(not math.isfinite(v) or abs(v) > .001 for v in speeds):
    return 'vehicle moving'
  gear = str(getattr(out, 'gearShifter', 'unknown'))
  if gear != 'park' and not (gear == 'neutral' and getattr(out, 'brakePressed', False)):
    return 'select P, or N with brake held'
  if (getattr(out, 'gasPressed', True) or getattr(out, 'leftBlindspot', False) or
      getattr(out, 'rightBlindspot', False) or getattr(CS, 'trailer_connected', False)):
    return 'pedal, blind spot or trailer warning'
  parser = getattr(CS, 'cp_cam', None)
  for name in MESSAGES.values():
    stamps = getattr(parser, 'ts_nanos', {}).get(name, {})
    stamp = max(stamps.values(), default=0) * 1e-9
    if stamp <= 0 or not 0 <= now - stamp < RX_SECONDS:
      return 'CCNC RX stale'
  return None


class DebugOverride:
  def __init__(self, command_path=COMMAND_PATH, status_path=STATUS_PATH):
    self.command_path, self.status_path = Path(command_path), Path(status_path)
    self.session = None
    self.blocked = None
    self.unsafe_since = None
    self.checked = False

  def poll(self, CP, CC, CS, warning=False, now=None):
    now = time.monotonic() if now is None else now
    try:
      command = read_json(self.command_path)
    except FileNotFoundError:
      return {}
    except (OSError, ValueError, TypeError):
      # A malformed command must not leave the previous arm reusable.
      self.blocked = 'invalid command'
      return {}
    try:
      session, config = command['session'], validate(command['config'])
      if not isinstance(session, str) or not session:
        raise ValueError('session required')
      if session != self.session:
        self.session, self.blocked, self.unsafe_since = session, None, None
      updated = command['updated_at']
      if type(updated) not in (int, float) or not math.isfinite(updated):
        raise ValueError('invalid lease timestamp')
      expired = not 0 <= now - updated < LEASE_SECONDS
      if not self.checked:
        schema()
        self.checked = True
      reason = safety_reason(CP, CC, CS, now)
      if config['transmit'] and reason in LATCH_REASONS:
        self.unsafe_since = now if self.unsafe_since is None else self.unsafe_since
        if now - self.unsafe_since >= LATCH_SECONDS:
          self.blocked = reason
      else:
        self.unsafe_since = None
      pause = reason or ('warning active' if warning else None) or ('settings expired' if expired else None)
      active = config['transmit'] and self.blocked is None and pause is None
      selected = overrides(config) if active else {}
      active = active and any(selected.values())
      idle = 'no field selected' if config['transmit'] and self.blocked is None and pause is None else 'preview'
      write_json(self.status_path, {'session': session, 'updated_at': now, 'active': active,
                                   'reason': self.blocked or pause or ('transmitting' if active else idle),
                                   'disarmed': self.blocked is not None,
                                   'paused': self.blocked is None and pause is not None and config['transmit'],
                                   'fields': selected})
      return selected
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
      self.blocked = 'invalid debug configuration or vehicle state'
      return {}


live = DebugOverride()
