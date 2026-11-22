"""Light helpers shared by the lane marking process and card (no OpenCV/numpy import).

CcncExtendedDisplay enables the StripNet recogniser on supported HDA1 vehicles.
The StripNet model is assets/lane_strip_v4.* (strip_model.MODEL_NAME); the model choice setting was removed (DaBin 10-26).
The recogniser publishes the display-filtered (two identical results within 0.4 s, 0.5 s expiry) left/right type and
colour on customReservedRawData2; card keeps only the latest packet and the CCNC sender applies it without waiting.
"""
import json

from opendbc.car.hyundai.ccnc_capability import hda1_cluster_supported, read_hda1_cluster_support

PARAM = 'CcncExtendedDisplay'
MODES = {1: 'scanner', 2: 'stripnet'}  # Recogniser engines; the daemon only runs STRIPNET
STRIPNET = 2
PACKET_TYPE, PACKET_VERSION = 'ccncLaneMarking', 1
MAX_AGE_NS = 500_000_000  # the display policy's expiry (result.MAX_AGE_NS)
TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')


def mode_of(params, cp=None):
  if not params.get_bool(PARAM):
    return 0
  supported = read_hda1_cluster_support(params) if cp is None else hda1_cluster_supported(cp)
  return STRIPNET if supported else 0


def parse(payload, received_ns):
  """Packet -> {'mode', 'frameId', 'left': {'type': (value, expires_ns)|None, 'color': ...}, 'right': ...}.

  Expiry is anchored on card's receive time minus the head's age at publish, so it never depends on the
  recogniser's clock or on how late card reads the packet."""
  data = json.loads(payload)
  if not isinstance(data, dict) or data.get('type') != PACKET_TYPE or data.get('version') != PACKET_VERSION:
    raise ValueError('expected a ccncLaneMarking v1 packet')
  out = {'mode': data.get('mode'), 'frameId': data.get('frameId')}
  for side in ('left', 'right'):
    entry = data.get(side)
    if entry is None:
      entry = {}
    elif not isinstance(entry, dict):
      raise ValueError(f'invalid lane marking object for {side}')
    heads = {}
    for head, classes in (('type', TYPES), ('color', COLORS)):
      value, age = entry.get(head), entry.get(head + 'AgeNs')
      if (value is not None and not isinstance(value, str)) or (age is not None and type(age) is not int):
        raise ValueError(f'invalid lane marking head for {side}.{head}')
      if value in classes and isinstance(age, int) and not isinstance(age, bool) and 0 <= age <= MAX_AGE_NS:
        heads[head] = (value, received_ns - age + MAX_AGE_NS)
      else:
        heads[head] = None
    reset_ns = entry.get('displayResetNs')
    camera_ns = data.get('cameraMonoTimeNanos')
    if (isinstance(reset_ns, int) and not isinstance(reset_ns, bool) and isinstance(camera_ns, int)
        and not isinstance(camera_ns, bool) and 0 < reset_ns <= camera_ns):
      heads['displayResetNs'] = reset_ns
    out[side] = heads
  return out


def current(marking, side, now_ns):
  """(type, colour) still within the display expiry, each None when unknown or expired."""
  if not marking:
    return None, None
  heads = marking.get(side) or {}
  return tuple(h[0] if h is not None and now_ns <= h[1] else None for h in (heads.get('type'), heads.get('color')))
