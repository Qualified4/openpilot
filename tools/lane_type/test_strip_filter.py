"""StripNet temporal filter in marking_daemon.Recogniser (A8/4 type + 8 m colour window + latch by default on the StripNet path), with a fake
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
  """The pre-filter Recogniser output (filter_input of the model decision or of the skip reason); a prob in
  [0.3, 0.5) is a single-frame dip here (the frames before it are at 1.0)."""
  out = {}
  for side in ('left', 'right'):
    if f['lane_change'] or f['probs'][side] < .5:
      reason = 'lane_change' if f['lane_change'] else 'lane_prob_dip' if f['probs'][side] >= .3 else 'not_detected'
      out[side] = filter_input('unknown', 'unknown', 0., 0., f['stamp'], reason=reason)
    elif f['lines'][side] == 'bad':
      out[side] = filter_input('unknown', 'unknown', 0., 0., f['stamp'], reason='invalid_lane_geometry')
    else:
      r = FakeStripModel().decide(np.array([p, 1. - p]), np.array([color, 1. - color]))
      out[side] = filter_input(r['type'], r['color'], r['typeConfidence'], r['colorConfidence'], f['stamp'],
                               dashed_observed=r['type'] == 'dashed', reason=r['reason'])
  return out


def unlatched():
  """The default filter without the latch (A8/4 + 8 m colour window): the window tests check the windows themselves."""
  return strip_filter.StripFilter('asym', long_m=strip_filter.LONG_M, short_m=strip_filter.SHORT_M, color_m=strip_filter.COLOR_M)


CASES = [(.98, .99, {}), (.02, .99, {}), (.85, .99, {}), (.99, .02, {}), (.98, .99, {'lane_change': True}),
         (.98, .99, {'probs': (.3, 1.)}), (.02, .99, {'lines': ('bad', 'R')}), (.02, .01, {})]


def test_stripnet_default_is_a8_4_latch_and_false_turns_it_off():
  default, explicit_none, off = Recogniser(2), Recogniser(2, strip_filter=None), Recogniser(2, strip_filter=False)
  for rec in (default, explicit_none):
    f = rec.strip_filter
    assert (f.kind, f.long_m, f.short_m) == ('asym', strip_filter.LONG_M, strip_filter.SHORT_M) == ('asym', 8., 4.)
    assert f.color_m == strip_filter.COLOR_M == 8.
    assert f.latch is strip_filter.LATCH is True  # DaBin 10-19: the latch is the default
  assert default.strip_filter is not explicit_none.strip_filter  # one filter (one set of windows) per Recogniser
  assert off.strip_filter is None and Recogniser(1).strip_filter is None
  assert strip_filter.make('A8/4').color_m is None and not strip_filter.make('A8/4').latch  # comparison: type only, no latch
  assert vars(strip_filter.make('A8/4L')) == vars(default.strip_filter)  # 'A8/4L' is the default
  reference = strip_filter.StripFilter('asym', long_m=8., short_m=4., color_m=8., latch=True)
  for i, (p, color, kw) in enumerate(CASES):
    raw = expected_raw(frame(i, **kw), p, color)
    # off: the raw filter_input results, untouched
    assert step(off, i, p, color, **kw) == raw
    # default: those results through A8/4 + latch, with the P(dashed) handed to it
    got = step(default, i, p, color, **kw)
    for side in ('left', 'right'):
      pd, py = got[side]['typePDashed'], got[side]['colorPYellow']
      assert (pd is None) if raw[side]['reason'] != 'strip_model' else pd == pytest.approx(p)
      assert (py is None) if raw[side]['reason'] != 'strip_model' else py == pytest.approx(1. - color)
      assert got[side] == dict(reference.update(side, raw[side], pd, i*FRAME_NS, float(i), py), typePDashed=pd, colorPYellow=py)
  assert all('typePDashed' not in side for side in raw.values())


def types(outputs, side='left'):
  return [o[side]['type'] for o in outputs]


def test_a8_4_holds_solid_to_dashed_until_long_window_and_releases_dashed_on_short_window():
  rec = Recogniser(2, unlatched())
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
  # a steady colour stays: weighted mean P(yellow) 0.01 -> white, confidence 1 - 0.01 (was |0.01 - 0.5| * 2 = 0.98)
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
  assert out[side] == dict(raw, typePDashed=None, colorPYellow=None) and out[side]['reason'] == reason
  assert len(rec.strip_filter.sides[side].items) == 0 and rec.strip_filter.sides[side].state is None
  assert len(rec.strip_filter.sides[side].colors) == 0
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
  assert out['left']['color'] == 'unknown' and out['left']['colorPYellow'] is None
  assert len(rec.strip_filter.sides['left'].colors) == n


def test_recogniser_reset_empties_the_filter():
  rec = Recogniser(2)
  for i in range(3):
    step(rec, i, .02)
  assert rec.strip_filter.sides
  rec.reset()
  assert rec.strip_filter.sides == {}


def colors(outputs, side='left'):
  return [o[side]['color'] for o in outputs]


def test_colour_window_holds_short_flips_and_follows_a_sustained_change():
  """1 m per frame: the 8 m colour window holds 9 frames; certainty-weighted mean P(yellow), 0.90 on the winning
  side's mean (mean P(yellow) or 1 - mean, as for a raw decision)."""
  rec = Recogniser(2, unlatched())
  i = 0
  for _ in range(12):
    assert step(rec, i, .02, color=.01)['left']['color'] == 'yellow'
    i += 1
  # two confident white frames inside a yellow line: the raw model flips, the window keeps yellow-or-unknown
  flip = []
  for _ in range(2):
    flip.append(step(rec, i, .02, color=.99))
    i += 1
  assert 'white' not in colors(flip)
  # a sustained white line: white only once the window's weighted mean P(yellow) is 0.10 or less
  to_white = []
  for _ in range(10):
    to_white.append(step(rec, i, .02, color=.99))
    i += 1
  c = colors(to_white)
  assert 'yellow' not in c and c[-1] == 'white' and c.index('white') > 0
  assert to_white[-1]['left']['colorConfidence'] == pytest.approx(.99) and to_white[-1]['left']['colorMonoTimeNanos'] == (i-1)*FRAME_NS
  # with confident frames (weight 0.98) the mean reaches 0.10 only when all 9 frames in the window are white (8 of 9
  # give 0.119): the 9th white frame (2 flip frames + 7) is the first white, as with the old 0.05 bound
  assert c.index('white') == 6
  # symmetric: the change back to yellow also shows on its 9th frame
  to_yellow = []
  for _ in range(10):
    to_yellow.append(step(rec, i, .02, color=.01))
    i += 1
  assert colors(to_yellow).index('yellow') == 8 and 'white' not in colors(to_yellow)


@pytest.mark.parametrize('p_yellow, shown', [(.92, 'yellow'), (.95, 'yellow'), (.88, 'unknown'), (.08, 'white')])
def test_colour_window_shows_a_steady_colour_the_raw_model_would_show(p_yellow, shown):
  """DaBin 10-19 (fe/2 R 19.06): a steady P(yellow) 0.87-0.95 is yellow raw (>= 0.90) but was unknown in the window,
  whose |mean - 0.5| * 2 >= 0.90 asked for mean >= 0.95. The window's confidence is now the winning side's mean."""
  rec = Recogniser(2)
  out = [step(rec, i, .02, color=1.-p_yellow) for i in range(12)]
  assert colors(out) == [shown]*12
  confidence = max(p_yellow, 1.-p_yellow)
  assert [o['left']['colorConfidence'] for o in out] == pytest.approx([confidence if shown != 'unknown' else 0.]*12)


def test_colour_window_is_bounded_by_distance():
  rec = Recogniser(2)
  for i in range(30):
    step(rec, i, .02, color=.01)
  assert len(rec.strip_filter.sides['left'].colors) == 9  # frames within 8 m (0..8 m back)


@pytest.mark.parametrize('reset, side', [({'lane_change': True}, 'left'), ({'probs': (.2, 1.)}, 'left'),
                                         ({'lines': ('bad', 'R')}, 'left')])
def test_reset_reasons_empty_the_colour_window(reset, side):
  rec = Recogniser(2)
  for i in range(12):
    step(rec, i, .02, color=.01)
  assert len(rec.strip_filter.sides[side].colors) == 9
  step(rec, 12, .02, color=.01, **reset)
  assert len(rec.strip_filter.sides[side].colors) == 0
  # with the window emptied, a white line is shown at once instead of waiting out the yellow frames
  assert step(rec, 13, .02, color=.99)[side]['color'] == 'white'


def test_stamp_gap_empties_both_windows():
  rec = Recogniser(2)
  for i in range(12):
    step(rec, i, .02, color=.01)
  # 0.3 s+ gap between camera frames (the next frame is 7 frame periods later)
  assert step(rec, 18, .02, color=.99)['left']['color'] == 'white'
  assert len(rec.strip_filter.sides['left'].colors) == 1 and len(rec.strip_filter.sides['left'].items) == 1


def test_colour_window_leaves_the_type_result_unchanged():
  """Same frames through A8/4 without and with the colour window: every type field is identical."""
  import random
  rng = random.Random(0)
  plain = strip_filter.StripFilter('asym', long_m=8., short_m=4.)
  coloured = strip_filter.StripFilter('asym', long_m=8., short_m=4., color_m=8.)
  type_keys = ('type', 'typeConfidence', 'typeMonoTimeNanos', 'dashedPatternObserved', 'dashedPatternMonoTimeNanos', 'reason')
  for i in range(400):
    p, c = rng.random(), rng.random()
    r = FakeStripModel().decide(np.array([p, 1. - p]), np.array([1. - c, c]))
    reason = rng.choice([r['reason']]*12 + ['not_detected'])
    raw = filter_input(r['type'], r['color'], r['typeConfidence'], r['colorConfidence'], i*FRAME_NS,
                       dashed_observed=r['type'] == 'dashed', reason=reason)
    learned = reason == 'strip_model'
    pd, py = (strip_filter.p_dashed(r), strip_filter.p_yellow(r)) if learned else (None, None)
    a = plain.update('left', raw, pd, i*FRAME_NS, i*.8)
    b = coloured.update('left', raw, pd, i*FRAME_NS, i*.8, py)
    assert {k: a[k] for k in type_keys} == {k: b[k] for k in type_keys}


def test_off_passes_colour_through():
  rec = Recogniser(2, strip_filter=False)
  out = [step(rec, i, .02, color=c) for i, c in enumerate([.01, .99, .01])]
  assert colors(out) == ['yellow', 'white', 'yellow']


def test_short_lane_prob_dip_skips_the_frame_and_keeps_the_windows():
  from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
  rec, policy = Recogniser(2), DisplayPolicy()
  side = rec.strip_filter.sides
  for i in range(12):
    out = step(rec, i, .02)
    confirmed = policy.confirm({'cameraMonoTimeNanos': i*FRAME_NS, **out}, i*FRAME_NS)
  assert out['left']['type'] == 'solid' and confirmed['left']['type']['value'] == 'solid'
  items, colors = list(side['left'].items), list(side['left'].colors)
  for i in range(12, 15):  # 0.45 for three frames: skipped, unknown, nothing entered or emptied
    out = step(rec, i, .02, probs=(.45, 1.))
    confirmed = policy.confirm({'cameraMonoTimeNanos': i*FRAME_NS, **out}, i*FRAME_NS)
    assert out['left']['reason'] == 'lane_prob_dip' and (out['left']['type'], out['left']['color']) == ('unknown', 'unknown')
    assert out['left']['typePDashed'] is None and out['left']['colorPYellow'] is None
    # nothing new enters; only frames now more than 8 m behind age out, as on any frame
    assert list(side['left'].items) == [x for x in items if i - x[1] <= 8.] and side['left'].state == 'solid'
    assert list(side['left'].colors) == [x for x in colors if i - x[1] <= 8.] and len(side['left'].colors) >= 6
    assert confirmed['left']['type']['value'] == 'solid'  # DisplayPolicy keeps its confirmed heads (0.5 s expiry)
  out = step(rec, 15, .02)  # recovered: the kept window confirms at once
  assert out['left']['type'] == 'solid' and out['left']['color'] == 'white'
  assert out['right']['type'] == 'solid'  # the other side was never touched


@pytest.mark.parametrize('probs', [[.25], [.45]*25])  # one frame below 0.3; 0.45 for 1.25 s (> 1 s)
def test_deep_or_long_lane_prob_dip_resets(probs):
  rec = Recogniser(2)
  for i in range(12):
    step(rec, i, .02)
  reasons = [step(rec, 12 + j, .02, probs=(p, 1.))['left']['reason'] for j, p in enumerate(probs)]
  assert reasons[-1] == 'not_detected' and len(rec.strip_filter.sides['left'].items) == 0
  if len(probs) > 1:  # 0.45 stays a dip for 1 s after the first frame below 0.5, then resets
    assert reasons[:21] == ['lane_prob_dip']*21 and set(reasons[21:]) == {'not_detected'}
  i = 12 + len(probs)
  assert step(rec, i, .98)['left']['type'] == 'dashed'  # emptied: judged on its own
  # back above 0.5 the dip clock restarts: a new short dip is skipped again
  assert step(rec, i + 1, .98, probs=(.45, 1.))['left']['reason'] == 'lane_prob_dip'


def test_default_latch_keeps_the_last_confident_label_and_resets():
  """DaBin 10-19: the StripNet default latches type and colour (A8/4L); 'A8/4' without it stays for comparison."""
  rec = Recogniser(2)
  assert rec.strip_filter.latch and strip_filter.make('A8/4L').latch and not strip_filter.make('A8/4').latch
  i, outs = 0, []
  for p in [.02]*12 + [.98]*8:
    outs.append(step(rec, i, p))
    i += 1
  kinds = types(outs)
  # A8/4 shows unknown for 5 frames at solid->dashed; the latch keeps solid (with its confidence) until dashed is
  # certain, on the same frame A8/4 shows it
  assert kinds[12:17] == ['solid']*5 and kinds[17:] == ['dashed']*3 and 'unknown' not in kinds
  assert outs[14]['left']['typeConfidence'] == outs[11]['left']['typeConfidence']
  # a lane prob dip is not a reset: dip frames pass through unknown, the latch stays and shows again at once
  dips = [step(rec, i + j, .98, probs=(.45, 1.)) for j in range(3)]
  assert types(dips) == ['unknown']*3 and rec.strip_filter.sides['left'].latched['type'][0] == 'dashed'
  i += 3
  assert step(rec, i, .5)['left']['type'] == 'dashed'  # an uncertain frame: the latched dashed
  i += 1
  out = step(rec, i, .98, probs=(.2, 1.))  # reset: the latch is emptied with the windows
  assert out['left']['type'] == 'unknown' and rec.strip_filter.sides['left'].latched == {}


def test_default_latch_keeps_the_colour_through_a_flip():
  """The colour window test's frames through the default: yellow stays through the flip (no unknown) and white shows
  COLOR_LATCH_FRAMES - 1 frames after the frame it shows on without the latch."""
  rec = Recogniser(2)
  c = colors([step(rec, i, .02, color=.01) for i in range(12)] + [step(rec, i, .02, color=.99) for i in range(12, 32)])
  # without the latch white shows on frame 20 (12 yellow + 2 flip + 6); the latch needs 10 confident white frames
  assert c[:29] == ['yellow']*29 and c[29:] == ['white']*3


Y, W, UNSURE = .01, .99, .5  # step(color=...): P(white) of the fake model; 0.5 is no colour decision


def single_frame_windows():
  """The default filter (latch, COLOR_LATCH_FRAMES) with windows of one frame (1 m per frame), so the window result is
  the frame's own decision and only the latch rules are left."""
  rec = Recogniser(2)
  f = rec.strip_filter
  assert f.latch and f.color_latch_frames == strip_filter.COLOR_LATCH_FRAMES == 10
  f.color_m = f.long_m = f.short_m = .5
  return rec


def run_colours(rec, colour_seq, start=0, **kw):
  return [step(rec, start + i, .02, color=c, **kw) for i, c in enumerate(colour_seq)]


def test_colour_latch_ignores_a_flip_shorter_than_ten_frames():
  rec = single_frame_windows()
  out = run_colours(rec, [Y]*12 + [W]*9 + [Y]*3)
  assert colors(out) == ['yellow']*24  # the first confident colour is shown at once, the 9 white frames are not
  assert [o['left']['colorConfidence'] for o in out] == pytest.approx([.99]*24)  # the latched yellow's confidence
  assert rec.strip_filter.sides['left'].flips == 0


def test_colour_latch_follows_ten_consecutive_frames_of_the_other_colour():
  rec = single_frame_windows()
  c = colors(run_colours(rec, [Y]*12 + [W]*10 + [Y]))
  assert c[:21] == ['yellow']*21 and c[21] == 'white'  # the 10th white frame
  assert c[22] == 'white'  # and back the same way: one yellow frame does not change it


def test_colour_latch_count_restarts_when_the_flip_is_broken():
  rec = single_frame_windows()
  out = run_colours(rec, [Y]*12 + [W]*9 + [UNSURE] + [W]*9 + [Y] + [W]*9)  # uncertain, then confident yellow
  assert colors(out) == ['yellow']*len(out)
  # a frame without probabilities (a lane prob dip) neither counts nor breaks the run
  i = len(out)
  out = run_colours(rec, [Y] + [W]*5, i) + run_colours(rec, [W], i + 6, probs=(.45, 1.)) + run_colours(rec, [W]*5, i + 7)
  assert colors(out) == ['yellow']*6 + ['unknown'] + ['yellow']*4 + ['white']  # the 10th white frame across the dip

def test_colour_latch_takes_the_first_confident_colour_at_once_after_a_reset():
  rec = single_frame_windows()
  run_colours(rec, [Y]*12 + [W]*5)
  assert rec.strip_filter.sides['left'].flips == 5
  out = step(rec, 17, .02, color=W, probs=(.2, 1.))  # not_detected: windows, latch and count emptied
  assert out['left']['color'] == 'unknown' and rec.strip_filter.sides['left'].latched == {}
  assert rec.strip_filter.sides['left'].flips == 0
  assert colors(run_colours(rec, [UNSURE, W], 18)) == ['unknown', 'white']  # no warm-up wait


def test_type_latch_still_changes_on_one_confident_frame():
  rec = single_frame_windows()
  kinds = types([step(rec, i, p, color=W) for i, p in enumerate([.02]*12 + [.98] + [.02] + [.5])])  # white: yellow dashed is unlearned
  assert kinds == ['solid']*12 + ['dashed', 'solid', 'solid']


@pytest.mark.parametrize('blinker, lcs, prob, y, v, reason', [
  (True, False, .8, 1.5, 9., None),             # blinker alone at lane-change speed, confident line beside the car
  (True, False, .8, -1.5, 9., None),
  (True, False, .8, 1.5, 8.3, None),
  (True, False, .8, 1.5, 5., 'lane_change'),    # below 30 km/h (turns): reset as before
  (True, False, .8, 1.5, None, 'lane_change'),  # no speed in the frame: reset
  (True, False, .8, .6, 9., 'lane_change'),     # the line being crossed
  (True, False, .6, 1.5, 9., 'lane_change'),    # not confident enough
  (True, True, .8, 1.5, 9., 'lane_change'),     # laneChangeState on: reset as before
  (False, True, .9, 1.8, 9., 'lane_change'),
  (False, False, .4, 1.8, 5., 'lane_prob_dip'),
])
def test_blinker_alone_infers_a_confident_line_beside_the_car(blinker, lcs, prob, y, v, reason):
  rec = Recogniser(2)
  f = dict(frame(0, lane_change=blinker or lcs, probs=(prob, 1.)), blinker=blinker, lcs_change=lcs,
           lines={'left': {'y': [y, y]}, 'right': {'y': [1.8, 1.8]}})
  if v is not None:
    f['v_ego'] = v
  assert rec._skip_reason('left', f) == reason
  f.pop('blinker'), f.pop('lcs_change'), f.pop('v_ego', None)  # a frame without the split flags: any lane change resets, as before
  assert rec._skip_reason('left', f) == ('lane_change' if blinker or lcs else reason)
