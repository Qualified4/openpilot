"""StripNet temporal filter in marking_daemon.Recogniser (A8/4 by default on the StripNet path), with a fake
StripModel that returns queued probabilities in order, so no model, OpenCV or camera frame is needed."""
import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import strip_filter, strip_model
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, filter_input

FRAME_NS = 50_000_000
STRIP = np.zeros((3, 128, 96), np.float32)


class FakeStripModel:
  """infer() pops one (P(dashed), P(white)) per strip and decides it exactly as StripModel does (0.90)."""
  threshold = .90
  decide = strip_model.StripModel.decide

  def __init__(self):
    self.queue = []

  def infer(self, strips):
    return [self.decide(np.array([p, 1. - p]), np.array([c, 1. - c])) for p, c in (self.queue.pop(0) for _ in strips)]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
  # by module path: another test drops strip_model from sys.modules, and Recogniser imports it afresh
  monkeypatch.setattr(f'{strip_model.__name__}.StripModel', FakeStripModel)
  monkeypatch.setattr(f'{strip_model.__name__}.strip_input', lambda y, uv, line, rpy, k: None if line == 'bad' else STRIP)


def frame(i, *, lane_change=False, probs=(1., 1.), lines=('L', 'R'), metres=1.):
  return {'lines': {'left': lines[0], 'right': lines[1]}, 'probs': {'left': probs[0], 'right': probs[1]}, 'rpy': None,
          'k': None, 'stamp': i*FRAME_NS, 'odometer': i*metres, 'leads': [], 'lane_change': lane_change}


def step(rec, i, p, color=.99, **kw):
  """Process frame i with P(dashed) p on every side that reaches the model."""
  f = frame(i, **kw)
  rec.model.queue += [(p, color)]*sum(not f['lane_change'] and f['probs'][s] >= .5 and f['lines'][s] != 'bad'
                                       for s in ('left', 'right'))
  out = rec.process(None, None, f)
  assert not rec.model.queue
  return out


def expected_raw(f, p, color):
  """The pre-filter Recogniser output (filter_input of the model decision or of the skip reason)."""
  out = {}
  for side in ('left', 'right'):
    if f['lane_change'] or f['probs'][side] < .5:
      out[side] = filter_input('unknown', 'unknown', 0., 0., f['stamp'], reason='lane_change' if f['lane_change'] else 'not_detected')
    elif f['lines'][side] == 'bad':
      out[side] = filter_input('unknown', 'unknown', 0., 0., f['stamp'], reason='invalid_lane_geometry')
    else:
      r = FakeStripModel().decide(np.array([p, 1. - p]), np.array([color, 1. - color]))
      out[side] = filter_input(r['type'], r['color'], r['typeConfidence'], r['colorConfidence'], f['stamp'],
                               dashed_observed=r['type'] == 'dashed', reason=r['reason'])
  return out


CASES = [(.98, .99, {}), (.02, .99, {}), (.85, .99, {}), (.99, .02, {}), (.98, .99, {'lane_change': True}),
         (.98, .99, {'probs': (.3, 1.)}), (.02, .99, {'lines': ('bad', 'R')}), (.02, .01, {})]


def test_stripnet_default_is_a8_4_and_false_turns_it_off():
  default, explicit_none, off = Recogniser(2), Recogniser(2, strip_filter=None), Recogniser(2, strip_filter=False)
  for rec in (default, explicit_none):
    f = rec.strip_filter
    assert (f.kind, f.long_m, f.short_m) == ('asym', strip_filter.LONG_M, strip_filter.SHORT_M) == ('asym', 8., 4.)
  assert default.strip_filter is not explicit_none.strip_filter  # one filter (one set of windows) per Recogniser
  assert off.strip_filter is None and Recogniser(1).strip_filter is None
  reference = strip_filter.make('A8/4')
  for i, (p, color, kw) in enumerate(CASES):
    raw = expected_raw(frame(i, **kw), p, color)
    # off: the raw filter_input results, untouched
    assert step(off, i, p, color, **kw) == raw
    # default: those results through A8/4, with the P(dashed) handed to it
    got = step(default, i, p, color, **kw)
    for side in ('left', 'right'):
      pd = got[side]['typePDashed']
      assert (pd is None) if raw[side]['reason'] != 'strip_model' else pd == pytest.approx(p)
      assert got[side] == dict(reference.update(side, raw[side], pd, i*FRAME_NS, float(i)), typePDashed=pd)
  assert all('typePDashed' not in side for side in raw.values())


def types(outputs, side='left'):
  return [o[side]['type'] for o in outputs]


def test_a8_4_holds_solid_to_dashed_until_long_window_and_releases_dashed_on_short_window():
  rec = Recogniser(2)
  i = 0
  for _ in range(12):  # 1 m per frame: the 8 m window holds 9 frames, the 4 m window 5
    assert step(rec, i, .02)['left']['type'] == 'solid'
    i += 1
  to_dashed = []
  for _ in range(8):
    to_dashed.append(step(rec, i, .98))
    i += 1
  # the raw model is a confident dashed from the first frame; the filter keeps it off until the long (8 m)
  # window has a dashed majority (5 of 9 frames) and shows it once that window is certain enough (6th frame)
  assert types(to_dashed)[:5] == ['unknown']*5 and 'dashed' not in types(to_dashed)[:5]
  assert rec.strip_filter.sides['left'].state == 'dashed' and types(to_dashed)[5:] == ['dashed']*3
  assert to_dashed[5]['left']['dashedPatternObserved'] is True and to_dashed[5]['left']['typeConfidence'] >= .90
  assert all(o['left']['typePDashed'] == pytest.approx(.98) for o in to_dashed)
  to_solid = []
  for _ in range(6):
    to_solid.append(step(rec, i, .02))
    i += 1
  # dashed->solid: the dashed display ends on the first solid frame (the short 4 m window's certainty drops
  # below 0.90) and a confident solid shows once that window is all solid (5th frame), one frame before the
  # opposite change showed dashed; no dashed is shown again while the long window still holds dashed frames
  assert types(to_solid)[:4] == ['unknown']*4 and types(to_solid)[4:] == ['solid']*2
  assert types(to_solid).index('solid') < types(to_dashed).index('dashed')
  # colour passes through untouched
  assert all(o['left']['color'] == 'white' and o['left']['colorConfidence'] == pytest.approx(.99) for o in to_dashed + to_solid)


def test_a8_4_empties_the_long_window_when_it_turns_solid():
  """dashed->solid: the state does not go back to dashed while the long window still holds the dashed frames."""
  rec = Recogniser(2)
  side = rec.strip_filter.sides
  i = 0
  for p in [.02]*12 + [.98]*8:
    step(rec, i, p)
    i += 1
  assert side['left'].state == 'dashed'
  states, sizes = [], []
  for _ in range(6):
    step(rec, i, .02)
    states.append(side['left'].state)
    sizes.append(len(side['left'].items))
    i += 1
  # the short (4 m) window turns solid on the 3rd solid frame; the 8 m window still held 6 dashed frames of 9
  # then, which used to put the state back to dashed on the next frame. Now it starts again from the short window
  assert states == ['dashed', 'dashed', 'solid', 'solid', 'solid', 'solid']
  assert sizes[2] == 5 and sizes[3:] == [6, 7, 8]


@pytest.mark.parametrize('reset, side, reason', [
  ({'lane_change': True}, 'left', 'lane_change'),
  ({'probs': (.2, 1.)}, 'left', 'not_detected'),
  ({'lines': ('bad', 'R')}, 'left', 'invalid_lane_geometry'),
])
def test_reset_reasons_empty_the_window(reset, side, reason):
  rec = Recogniser(2)
  for i in range(12):
    step(rec, i, .02)
  assert len(rec.strip_filter.sides[side].items) == 9 and rec.strip_filter.sides[side].state == 'solid'
  out = step(rec, 12, .98, **reset)
  raw = expected_raw(frame(12, **reset), .98, .99)[side]
  assert out[side] == dict(raw, typePDashed=None) and out[side]['reason'] == reason
  assert len(rec.strip_filter.sides[side].items) == 0 and rec.strip_filter.sides[side].state is None
  # with the window emptied, the next dashed frames are judged on their own instead of waiting for 5 of 9
  assert step(rec, 13, .98)[side]['type'] == 'dashed'
  if reason != 'lane_change':  # the other side kept its solid window and still holds the change back
    other = 'right'
    assert len(rec.strip_filter.sides[other].items) > 1 and step(rec, 14, .98)[other]['type'] != 'dashed'


def test_unlearned_yellow_dashed_passes_through_without_entering_the_window():
  rec = Recogniser(2)
  for i in range(5):
    step(rec, i, .02)
  n = len(rec.strip_filter.sides['left'].items)
  out = step(rec, 5, .99, color=.02)
  assert out['left']['reason'] == 'yellow_dashed_not_learned' and out['left']['type'] == 'unknown'
  assert out['left']['typePDashed'] is None and len(rec.strip_filter.sides['left'].items) == n


def test_recogniser_reset_empties_the_filter():
  rec = Recogniser(2)
  for i in range(3):
    step(rec, i, .02)
  assert rec.strip_filter.sides
  rec.reset()
  assert rec.strip_filter.sides == {}
