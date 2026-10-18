"""CCNC lane line colour/shape from the display-only lane marking recogniser (Scanner/StripNet).

Applied per side to the existing lane code only where the existing policy shows a normal line (white 2 or
lane-change-available green 6). Priority: recognised yellow -> orange solid 4 (also over green);
otherwise the existing green condition keeps green with the recognised shape (solid 6 / dashed 10);
otherwise white with the recognised shape (solid 2 / dashed 8). UNKNOWN, expired or disabled leaves the
existing code untouched, as do hidden lines, departure blinking and the existing warning/BSD orange,
which are decided outside this function. Display only: lane-change permission is not read or changed.

``marking`` is card's latest parsed packet: {'left': {'type': (value, expires_ns)|None, 'color': ...}, ...}.
"""
WHITE_SOLID, ORANGE_SOLID, GREEN_SOLID, WHITE_DASHED, GREEN_DASHED = 2, 4, 6, 8, 10


def _head(marking, side, head, now_ns):
  entry = (marking.get(side) or {}).get(head) if marking else None
  return entry[0] if entry is not None and now_ns <= entry[1] else None


def lane_code(existing, marking, side, now_ns):
  if existing not in (WHITE_SOLID, GREEN_SOLID) or not marking:
    return existing
  kind, color = _head(marking, side, 'type', now_ns), _head(marking, side, 'color', now_ns)
  if color == 'yellow':
    return ORANGE_SOLID
  if color != 'white' or kind not in ('dashed', 'solid'):
    return existing
  if existing == GREEN_SOLID:
    return GREEN_DASHED if kind == 'dashed' else GREEN_SOLID
  return WHITE_DASHED if kind == 'dashed' else WHITE_SOLID
