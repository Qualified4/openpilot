"""filterab1026 A/B switches of strip_filter / strip_model / marking_daemon. Every default must be the unchanged behaviour."""
import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import strip_filter as sf, strip_model
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, filter_input
from tools.lane_type.test_strip_model_v4 import C, T, T4, frame, manifest4_dir, manifest_dir, strip, strips  # noqa: F401

NS = 50_000_000


def feed(f, p, py, n, start=0, metres=1., side='left', reason='strip_model'):
  out = None
  for i in range(start, start + n):
    r = filter_input('dashed' if p is not None and p > .5 else 'solid', 'yellow' if py is not None and py > .5 else 'white', .99, .99, i*NS,
                     reason=reason)
    out = f.update(side, r, p, i*NS, i*metres, py)
  return out


def test_default_variant_and_recogniser_filter_unchanged():
  base = sf.make('A8/4L')  # DaBin 10-26: the default is F1d1; the colour latch and hold switches are unchanged
  assert (base.type_conf, base.type_latch_m, base.color_latch_frames, base.color_latch_m, base.hold_latch) == \
         ('max', 1., sf.COLOR_LATCH_FRAMES, None, False)
  rec = Recogniser(2).strip_filter
  keys = ('kind', 'long_m', 'short_m', 'color_m', 'latch', 'type_conf', 'type_latch_frames', 'type_latch_m', 'color_latch_frames',
          'color_latch_m', 'hold_latch')
  assert {k: getattr(rec, k) for k in keys} == {k: getattr(base, k) for k in keys}
  assert sf.VARIANTS['A8/4L-F1+F2+F3b+F5'] == dict(sf.VARIANTS['A8/4L-legacy'], type_conf='max', color_latch_frames=3, hold_latch=True, color_m=4.)
  assert not strip_model.HOLD_COLOR and strip_model.HOLD_BY_FAMILY  # DaBin 10-26: F4 is the product default, F3 stays off


def test_f1_type_confidence_matches_the_colour_rule():
  """Window mean P(dashed) 0.92 after a solid latch: |m - .5| * 2 = .84 keeps solid; max(m, 1 - m) = .92 turns dashed (repro B)."""
  for name, want in (('A8/4L-legacy', 'solid'), ('A8/4L-F1', 'dashed'), ('A8/4L', 'dashed')):
    f = sf.make(name)
    feed(f, .01, .01, 10)
    assert feed(f, .92, .01, 12, start=10)['type'] == want


@pytest.mark.parametrize('name,flip_after', [('A8/4L', 10), ('A8/4L-legacy', 10), ('A8/4L-F2', 3)])
def test_f2_colour_latch_frames(name, flip_after):
  f = sf.make(name)
  feed(f, .01, .001, 20)
  f.color_m = .5  # isolate the latch from the window: the window holds only the current frame
  colors = [feed(f, .01, .999, 1, start=20 + k)['color'] for k in range(12)]
  assert colors.index('yellow') == flip_after - 1  # the flip_after-th consecutive confident yellow frame


def test_f2b_colour_latch_by_distance():
  f = sf.make('A8/4L-F2b')
  feed(f, .01, .001, 20)
  f.color_m = .5  # isolate the latch from the window: the window holds only the last half metre
  got = [feed(f, .01, .999, 1, start=20 + k, metres=1.)['color'] for k in range(6)]  # 1 m per frame from frame 20
  assert got[:2] == ['white', 'white'] and got[2] == 'yellow'  # needs >= 2 m since the first confident yellow frame


def test_f3b_latched_heads_shown_while_the_model_abstains():
  for hold, want in ((False, ('unknown', 'unknown')), (True, ('solid', 'white'))):
    f = sf.StripFilter(**dict(sf.VARIANTS['A8/4L'], hold_latch=hold))
    feed(f, .01, .01, 10)
    out = f.update('left', filter_input('unknown', 'unknown', 0., 0., 10*NS, reason='unknown'), None, 10*NS, 10., None)
    assert (out['type'], out['color']) == want and out['reason'] == 'unknown'
    if hold:
      assert out['typeMonoTimeNanos'] == out['colorMonoTimeNanos'] == 10*NS and out['typeConfidence'] >= .9


def test_f3_hold_colour_switch(tmp_path, strips, monkeypatch):  # noqa: F811
  m = strip_model.StripModel(manifest_dir(tmp_path))
  assert 'pYellowHeld' not in m.decide4(T['unknown'], C['yellow'])
  monkeypatch.setattr(strip_model, 'HOLD_COLOR', True)
  assert m.decide4(T['unknown'], C['yellow'])['pYellowHeld'] > .99
  rec = Recogniser(2)
  rec.model = m
  strips['left'] = strip(T['unknown'], C['yellow'])
  strips['right'] = strip(T['solid'], C['white'], at='mirrored')
  out = rec.process(None, None, frame(0))
  assert out['left']['reason'] == 'unknown' and out['left']['typePDashed'] is None and out['left']['colorPYellow'] > .99
  assert out['left']['type'] == 'unknown' and out['left']['color'] == 'yellow'  # first confident colour is stored at once


def test_f4_family_abstention(tmp_path, monkeypatch):
  m = strip_model.StripModel(manifest4_dir(tmp_path))
  probs = np.log([.50, .05, .42, .03])  # largest class .50, dashed family .92
  monkeypatch.setattr(strip_model, 'HOLD_BY_FAMILY', False)  # the single-class rule (before 10-26)
  assert m.decide4(probs, C['white'])['reason'] == 'unknown'
  monkeypatch.setattr(strip_model, 'HOLD_BY_FAMILY', True)  # the default
  r = m.decide4(probs, C['white'])
  assert (r['reason'], r['type'], r['typeClass']) == ('strip_model', 'dashed', 'dashed')
  assert m.decide4(np.log([.45, .40, .10, .05]), C['white'])['reason'] == 'unknown'  # families .55 / .45


@pytest.mark.parametrize('name,after', [('A8/4L-legacy', 1), ('A8/4L-F1', 1), ('A8/4L-F1+h2', 2), ('A8/4L-F1+h3', 3)])
def test_f1h_type_latch_needs_consecutive_confident_frames(name, after):
  f = sf.make(name)
  feed(f, .01, .01, 10)  # solid latched at once (first latch immediate)
  assert f.sides['left'].latched['type'][0] == 'solid'
  f.long_m = f.short_m = .5  # windows hold only the current frame: every frame is a confident dashed decision
  got = [feed(f, .99, .01, 1, start=10 + k)['type'] for k in range(5)]
  assert got.index('dashed') == after - 1


def test_f1h_unconfident_frame_restarts_the_count():
  f = sf.make('A8/4L-F1+h2')
  feed(f, .01, .01, 10)
  f.long_m = f.short_m = .5
  assert feed(f, .99, .01, 1, start=10)['type'] == 'solid'
  assert feed(f, .55, .01, 1, start=11)['type'] == 'solid'  # not confident: count restarts
  assert feed(f, .99, .01, 1, start=12)['type'] == 'solid' and feed(f, .99, .01, 1, start=13)['type'] == 'dashed'


def test_f1d1_type_latch_by_distance():
  f = sf.make('A8/4L-F1+d1')
  feed(f, .01, .01, 10, metres=.5)
  f.long_m = f.short_m = .2
  got = [feed(f, .99, .01, 1, start=10 + k, metres=.5)['type'] for k in range(4)]  # odometers 5, 5.5, 6, 6.5
  assert got == ['solid', 'solid', 'dashed', 'dashed']


def test_type_latch_legacy_is_off():
  f = sf.make('A8/4L-legacy')
  assert f.type_latch_frames is None and f.type_latch_m is None and f.type_conf == 'abs'


def test_new_default_is_f1d1():
  """DaBin 10-26: the product filter (Recogniser default and 'A8/4L') equals the filterab1026 A/B variant F1d1."""
  assert sf.VARIANTS['A8/4L'] == sf.VARIANTS['A8/4L-F1+d1']
  assert vars(sf.make('A8/4L-F1+d1')) == vars(Recogniser(2).strip_filter)


def test_f1d1_travel_without_probabilities_does_not_count():
  """codex review1026 M1: frames without P(dashed) (model abstention, lane_prob_dip, lead_close) pause the 1 m type confirmation;
  their travel is not counted, so 1.5 m of held frames plus one confident frame cannot flip the latch."""
  f = sf.make('A8/4L')  # the product default (F1d1)
  feed(f, .01, .01, 10, metres=.5)  # solid latched, odometers 0 .. 4.5
  f.long_m = f.short_m = .2
  assert feed(f, .99, .01, 1, start=10, metres=.5)['type'] == 'solid'  # odometer 5.0: the run starts
  feed(f, None, .01, 3, start=11, metres=.5, reason='unknown')  # 5.5 .. 6.5 without P(dashed): paused
  assert f.sides['left'].latched['type'][0] == 'solid'
  assert feed(f, .99, .01, 1, start=14, metres=.5)['type'] == 'solid'  # 7.0: 0.5 m counted (2.0 m without the pause)
  assert feed(f, .99, .01, 1, start=15, metres=.5)['type'] == 'dashed'  # 7.5: 1.0 m of confident travel
