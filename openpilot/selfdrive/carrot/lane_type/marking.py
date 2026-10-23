"""Light helpers shared by the lane marking process and card (no OpenCV/numpy import).

CcncLaneTypeDetection turns the display-only recogniser on: 0 off (existing CCNC lane display unchanged), 1 StripNet
(frozen strip model). The value 2 (StripNet in the former 3-way selector, whose 1 was the Scanner) is still read as
StripNet, so a stored 1 or 2 both run StripNet: the profile Scanner is no longer run on the car (2026-10-21); MODES keeps its
name for offline replays (tools/lane_profile/replay_marking.py builds marking_daemon.Recogniser(1) itself).
LaneStripModel picks the StripNet model: 0 the deployed one (assets/lane_strip.*, v3), 1 the new-model slot
(assets/lane_strip_v4.*); 1 runs the deployed one when its files are missing or invalid (strip_model.load_slot). A stored 2
(the removed hybrid option, type from lane_strip_v4 + colour from lane_strip) runs slot 1, the new model (model_of).
The recogniser publishes the display-filtered (two identical results within 0.4 s, 0.5 s expiry) left/right type and
colour on customReservedRawData2; card keeps only the latest packet and the CCNC sender applies it without waiting.
"""
import json

PARAM = 'CcncLaneTypeDetection'
MODES = {1: 'scanner', 2: 'stripnet'}  # Recogniser engines; the daemon only runs STRIPNET
STRIPNET = 2
MODEL_PARAM = 'LaneStripModel'
MODEL_SLOTS = {0: 'current', 1: 'new'}  # strip_model.SLOTS 0 lane_strip.* (v3), 1 lane_strip_v4.*
LEGACY_HYBRID_SLOT = 2  # the removed hybrid option (2026-10-23): a stored 2 runs slot 1 (the new model)
PACKET_TYPE, PACKET_VERSION = 'ccncLaneMarking', 1
MAX_AGE_NS = 500_000_000  # the display policy's expiry (result.MAX_AGE_NS)
TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')


def mode_of(params):
  """STRIPNET when CcncLaneTypeDetection is on (1, or the former selector's 2), else 0 (off)."""
  try:
    mode = params.get_int(PARAM)
  except (TypeError, ValueError):
    return 0
  return STRIPNET if mode in (1, 2) else 0


def model_of(params):
  """LaneStripModel slot (0 current, 1 new; a stored 2, the removed hybrid, is 1); anything else or unreadable: 0."""
  try:
    slot = params.get_int(MODEL_PARAM)
  except (TypeError, ValueError):
    return 0
  if slot == LEGACY_HYBRID_SLOT:
    return 1
  return slot if slot in MODEL_SLOTS else 0


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
