"""CCNC lane line colour/shape from the display-only lane marking recogniser (Scanner/StripNet).

Applied per side to the existing lane code only where the existing policy shows a normal line (white 2 or
lane-change-available green 6). Priority: recognised yellow -> orange solid 4 (also over green);
otherwise the existing green condition keeps green with the recognised shape (solid 6 / dashed 10);
otherwise white with the recognised shape (solid 2 / dashed 8). UNKNOWN, expired or disabled leaves the
existing code untouched, as do grey 0 (low confidence), hidden lines, departure blinking and the existing
warning orange, which are decided outside this function. Display only: lane-change permission is not read or changed.

``marking`` is card's latest parsed packet: {'left': {'type': (value, expires_ns)|None, 'color': ...}, ...}.
"""
WHITE_SOLID, ORANGE_SOLID, GREEN_SOLID, WHITE_DASHED, GREEN_DASHED = 2, 4, 6, 8, 10
HOLD_S, GREEN_ON_S, STOP_SPEED = 2., 1., .5
LOW_PROB_ON, LOW_PROB_OFF, LOW_PROB_S, MAX_DT = .3, .5, 1., .2


def _head(marking, side, head, now_ns):
  entry = (marking.get(side) or {}).get(head) if marking else None
  return entry[0] if entry is not None and now_ns <= entry[1] else None


def heads(marking, side, now_ns):
  """Fresh (type, colour) of one side; None for a missing or expired head."""
  return _head(marking, side, 'type', now_ns), _head(marking, side, 'color', now_ns)


def code(existing, kind, color):
  if existing not in (WHITE_SOLID, GREEN_SOLID):
    return existing
  if color == 'yellow':
    return ORANGE_SOLID
  if color != 'white' or kind not in ('dashed', 'solid'):
    return existing
  if existing == GREEN_SOLID:
    return GREEN_DASHED if kind == 'dashed' else GREEN_SOLID
  return WHITE_DASHED if kind == 'dashed' else WHITE_SOLID


def lane_code(existing, marking, side, now_ns):
  return code(existing, *heads(marking, side, now_ns)) if marking else existing


class SideDisplay:
  """Display memory of one side (display only; no control input is read or written).

  - hold: the last fresh type and the last fresh colour (each head separately) fill a missing head for HOLD_S while
    driving and without a time limit while stopped (v < STOP_SPEED); a new fresh value replaces it at once;
    a blinker or lane change clears it.
  - green (lane change available) turns on only after GREEN_ON_S of continuous availability, off at once.
  - low: lane prob < LOW_PROB_ON for LOW_PROB_S of driving -> grey 0; released at >= LOW_PROB_OFF; frozen while stopped.
  """
  def __init__(self):
    self.hold = {'type': None, 'color': None}  # head -> (value, last fresh time)
    self.green, self.green_s, self.low, self.low_s, self.last_t = None, 0., False, 0., None

  def update(self, t, v_ego, lane_change_available, prob, changing, kind, color):
    """Returns (green, low, kind, colour) for this frame."""
    dt = 0. if self.last_t is None else min(max(t - self.last_t, 0.), MAX_DT)
    self.last_t = t
    stopped = v_ego < STOP_SPEED

    available = bool(lane_change_available)
    self.green_s = self.green_s + dt if available else 0.
    if self.green is None or not available or self.green_s >= GREEN_ON_S:
      self.green = available

    if prob is not None and not stopped:
      self.low_s = self.low_s + dt if prob < LOW_PROB_ON else 0.
      if self.low_s >= LOW_PROB_S:
        self.low = True
      elif prob >= LOW_PROB_OFF:
        self.low = False

    shown = {}
    for head, value, classes in (('type', kind, ('dashed', 'solid')), ('color', color, ('white', 'yellow'))):
      held = None if changing else self.hold[head]
      if value in classes:
        held = (value, t)
      elif held is not None and (stopped or t - held[1] <= HOLD_S):
        value = held[0]
        if stopped:
          held = (value, t)
      else:
        held = None
      self.hold[head], shown[head] = held, value
    return self.green, self.low, shown['type'], shown['color']
