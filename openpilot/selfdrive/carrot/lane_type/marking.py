"""Light helpers shared by the lane marking process and card (no OpenCV/numpy import).

CcncLaneTypeDetection selects the display-only recogniser: 0 off (existing CCNC lane display unchanged),
1 Scanner (1-D profile scanner), 2 StripNet (frozen strip model). The recogniser publishes the display-filtered
(two identical results within 0.4 s, 0.5 s expiry) left/right type and colour on customReservedRawData2;
card keeps only the latest packet and the CCNC sender applies it without waiting.
"""
import json

PARAM = 'CcncLaneTypeDetection'
MODES = {1: 'scanner', 2: 'stripnet'}
PACKET_TYPE, PACKET_VERSION = 'ccncLaneMarking', 1
MAX_AGE_NS = 500_000_000  # the display policy's expiry (result.MAX_AGE_NS)
TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')


def mode_of(params):
  try:
    mode = params.get_int(PARAM)
  except (TypeError, ValueError):
    return 0
  return mode if mode in MODES else 0


def parse(payload, received_ns):
  """Packet -> {'mode', 'frameId', 'left': {'type': (value, expires_ns)|None, 'color': ...}, 'right': ...}.

  Expiry is anchored on card's receive time minus the head's age at publish, so it never depends on the
  recogniser's clock or on how late card reads the packet."""
  data = json.loads(payload)
  if not isinstance(data, dict) or data.get('type') != PACKET_TYPE or data.get('version') != PACKET_VERSION:
    raise ValueError('expected a ccncLaneMarking v1 packet')
  out = {'mode': data.get('mode'), 'frameId': data.get('frameId')}
  for side in ('left', 'right'):
    entry = data.get(side) or {}
    heads = {}
    for head, classes in (('type', TYPES), ('color', COLORS)):
      value, age = entry.get(head), entry.get(head + 'AgeNs')
      if value in classes and isinstance(age, int) and not isinstance(age, bool) and 0 <= age <= MAX_AGE_NS:
        heads[head] = (value, received_ns - age + MAX_AGE_NS)
      else:
        heads[head] = None
    out[side] = heads
  return out


def current(marking, side, now_ns):
  """(type, colour) still within the display expiry, each None when unknown or expired."""
  if not marking:
    return None, None
  heads = marking.get(side) or {}
  return tuple(h[0] if h is not None and now_ns <= h[1] else None for h in (heads.get('type'), heads.get('color')))
