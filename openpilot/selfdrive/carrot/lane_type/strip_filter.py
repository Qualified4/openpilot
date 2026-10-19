"""Temporal filter for StripNet type results.

The StripNet path always uses it: ``marking_daemon.Recogniser(2)`` builds the asymmetric filter with LONG_M / SHORT_M
(A8/4) by default; the ``strip_filter`` argument (another StripFilter, or False for none) is for tests and replays.
It runs inside ``Recogniser.process`` (StripNet branch), so it sits between the model and
``display.DisplayPolicy.confirm``:
per side it takes the raw ``filter_input`` result plus the model's P(dashed) and returns a result dict of the
same form. Only the type head is changed; colour (value, confidence, stamp) passes through untouched, so the
yellow-first display is unaffected. ``DisplayPolicy`` and the Scanner path are not touched.

Variants (``StripFilter(kind, **params)``):
  none            raw result unchanged (baseline)
  median n        majority of the last n confident type labels (unknown frames are not entered)
  distance m      certainty-weighted mean P(dashed) over the last m metres of travel, 0.5 split
  asym long/short dashed->solid decided on the short window, solid->dashed only when the long window's mean
                  P(dashed) exceeds 0.5; when the state turns solid the long window is emptied down to the short
                  one, so dashed frames still in the long window cannot turn it straight back to dashed
The other variants and the ``make`` name table are kept for the replay harness.
Windows are emptied on the same reset reasons DisplayPolicy uses, on trackingReset, and on a camera stamp gap
or step back (DistanceEvidence's 0.3 s rule). Distance windows are also capped at DistanceEvidence's 3 s, so a
stopped car does not hold old frames indefinitely. The model's 0.90 head threshold is applied to the filtered
confidence (|mean - 0.5| * 2 for distance windows), as StripModel.decide applies it to the raw one.
"""
from collections import deque

from openpilot.selfdrive.carrot.lane_type.temporal import WINDOW_NS

RESET_REASONS = ('lane_change', 'tracking_lost', 'tracking_lost_or_stale', 'invalid_lane_geometry', 'not_detected')
MAX_GAP_NS = 300_000_000
THRESHOLD = 0.90
LONG_M, SHORT_M = 8., 4.  # the StripNet path's default asymmetric filter (A8/4)


def p_dashed(model_result):
  """P(dashed) from StripModel.decide's output (two-class softmax), or None without a model output."""
  if not isinstance(model_result, dict) or model_result.get('typeCandidate') not in ('dashed', 'solid'):
    return None
  c = float(model_result['typeConfidence'])
  return c if model_result['typeCandidate'] == 'dashed' else 1. - c


class _Side:
  def __init__(self):
    self.items = deque()  # (stamp, odometer, label_or_p, weight)
    self.state = None     # asym: last decided type
    self.last_stamp = None

  def clear(self):
    self.items.clear()
    self.state = None


class StripFilter:
  def __init__(self, kind='none', n=5, m=12., long_m=12., short_m=4., threshold=THRESHOLD):
    if kind not in ('none', 'median', 'distance', 'asym'):
      raise ValueError(kind)
    self.kind, self.n, self.m, self.long_m, self.short_m, self.threshold = kind, n, m, long_m, short_m, threshold
    self.sides = {}

  def reset(self):
    self.sides.clear()

  def _typed(self, result, kind, confidence, stamp):
    out = dict(result)
    confident = kind in ('dashed', 'solid') and confidence >= self.threshold
    out['type'] = kind if confident else 'unknown'
    out['typeConfidence'] = float(confidence) if confident else 0.
    out['typeMonoTimeNanos'] = stamp
    out['dashedPatternObserved'] = out['type'] == 'dashed'
    out['dashedPatternMonoTimeNanos'] = stamp
    return out

  def update(self, side, result, p, stamp, odometer):
    """result: filter_input dict for this side; p: P(dashed) of the model output used for it (None if no
    inference ran, or the frame must not enter the window); stamp: camera EOF ns; odometer: travel m."""
    if self.kind == 'none':
      return result
    s = self.sides.setdefault(side, _Side())
    if (result.get('trackingReset') or result.get('reason') in RESET_REASONS or
        (s.last_stamp is not None and not 0 < stamp - s.last_stamp <= MAX_GAP_NS)):
      s.clear()
    s.last_stamp = stamp
    if result.get('reason') in RESET_REASONS:
      return result
    if self.kind == 'median':
      if result.get('type') in ('dashed', 'solid'):
        s.items.append((stamp, odometer, result['type'], float(result['typeConfidence'])))
        while len(s.items) > self.n:
          s.items.popleft()
      if p is None or not s.items:
        return result if p is None else self._typed(result, 'unknown', 0., stamp)
      votes = {k: [w for _, _, v, w in s.items if v == k] for k in ('dashed', 'solid')}
      kind = max(votes, key=lambda k: len(votes[k]))
      if 2*len(votes[kind]) <= len(s.items):
        return self._typed(result, 'unknown', 0., stamp)
      return self._typed(result, kind, sum(votes[kind])/len(votes[kind]), stamp)
    if p is not None:
      s.items.append((stamp, odometer, p, abs(p - .5)*2))
    span = self.m if self.kind == 'distance' else max(self.long_m, self.short_m)
    while s.items and (odometer - s.items[0][1] > span or stamp - s.items[0][0] > WINDOW_NS):
      s.items.popleft()
    if p is None:
      return result

    def mean(metres):
      xs = [(q, w) for _, o, q, w in s.items if odometer - o <= metres]
      total = sum(w for _, w in xs)
      return sum(q*w for q, w in xs)/total if total > 0 else .5
    if self.kind == 'distance':
      m = mean(self.m)
      return self._typed(result, 'dashed' if m > .5 else 'solid' if m < .5 else 'unknown', abs(m - .5)*2, stamp)
    short, long = mean(self.short_m), mean(self.long_m)
    if s.state == 'dashed':
      kind, m = ('solid', short) if short < .5 else ('dashed', short)
    else:
      if long > .5:
        kind, m = 'dashed', long
      elif short < .5:
        kind, m = 'solid', short
      elif s.state == 'solid':
        kind, m = 'solid', long
      else:
        kind, m = None, .5
    if kind == 'solid' and s.state != 'solid':
      # empty the long window (keep the short one): the dashed frames that made it dashed must not reverse it
      while s.items and odometer - s.items[0][1] > self.short_m:
        s.items.popleft()
    s.state = kind
    return self._typed(result, kind or 'unknown', abs(m - .5)*2, stamp)


VARIANTS = {
  'none': {'kind': 'none'},
  'M3': {'kind': 'median', 'n': 3}, 'M5': {'kind': 'median', 'n': 5}, 'M7': {'kind': 'median', 'n': 7},
  'D8': {'kind': 'distance', 'm': 8.}, 'D12': {'kind': 'distance', 'm': 12.}, 'D15': {'kind': 'distance', 'm': 15.},
  'A12/4': {'kind': 'asym', 'long_m': 12., 'short_m': 4.}, 'A15/4': {'kind': 'asym', 'long_m': 15., 'short_m': 4.},
  'A8/4': {'kind': 'asym', 'long_m': LONG_M, 'short_m': SHORT_M},
}


def make(name):
  return StripFilter(**VARIANTS[name])
