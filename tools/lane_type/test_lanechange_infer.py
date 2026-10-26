"""Opt-in lane-change inference; deployed A, per-head pauses and line-identity resets."""

from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking_daemon as md, strip_model
from openpilot.selfdrive.carrot.lane_type import marking
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from opendbc.car.hyundai.ccnc_lane_marking import SideDisplay
import json

PRODUCT_GATE = md.GATE_RELAX  # read before the pre_g1_gate fixture pins the replay value


@pytest.fixture(autouse=True)
def pre_g1_gate(monkeypatch):
  # These tests are written around the pre-10-26 lane prob floors (.5 / .3, GATE_RELAX None); the product default G1 (.25 / .15,
  # DaBin 10-26, rollback1026) is checked by test_product_gate_is_g1. By module path: a test may re-import marking_daemon.
  monkeypatch.setattr('openpilot.selfdrive.carrot.lane_type.marking_daemon.GATE_RELAX', None)


def frame(i, state='laneChangeStarting', prob=0.1, speed=10.0, ys=(-1.8, 1.8), blinker=True):
  return {
    'lines': {s: {'y': [y]} for s, y in zip(('left', 'right'), ys, strict=True)},
    'probs': {'left': prob, 'right': prob},
    'stamp': (i + 1) * 50_000_000,
    'odometer': i * 0.5,
    'rpy': None,
    'k': None,
    'leads': [],
    'lane_change': state != 'off' or blinker,
    'blinker': blinker,
    'lcs_change': state != 'off',
    'lcs': state,
    'v_ego': speed,
  }


@pytest.fixture
def recogniser(monkeypatch):
  monkeypatch.setattr(strip_model, 'strip_input', lambda y, uv, line, rpy, k, **kw: None if line.get('bad') else np.zeros((3, 1, 1), np.float32))

  def make(variant='B'):
    model = N(strip={}, mirror_right=False, tc=0.99, cc=0.99, calls=0)

    def infer(strips):
      model.calls += len(strips)
      return [
        {
          'type': 'dashed' if model.tc >= 0.9 else 'unknown',
          'color': 'white' if model.cc >= 0.9 else 'unknown',
          'typeConfidence': model.tc,
          'colorConfidence': model.cc,
          'typeCandidate': 'dashed',
          'colorCandidate': 'white',
          'reason': 'strip_model',
        }
        for _ in strips
      ]

    model.infer = infer
    return md.Recogniser(2, model=model, lanechange=variant)

  return make


@pytest.mark.parametrize('state,blinker', [('off', True), ('preLaneChange', True), ('laneChangeStarting', False), ('laneChangeFinishing', False)])
@pytest.mark.parametrize('prob', [0.0, 0.299, 0.499, 0.699, 0.9])
def test_active_states_ignore_lane_probability(recogniser, state, blinker, prob):
  rec = recogniser()
  f = frame(0, state, prob, blinker=blinker)
  assert rec._skip_reason('left', f) is None
  assert rec.process(None, None, f)['left']['type'] == 'dashed'


@pytest.mark.parametrize('variant', ['B', 'B-cross', 'B-cross-0.95', 'B-p0.2', 'B-cross-p0.2', 'C1', 'C2', 'C1-0.5'])
@pytest.mark.parametrize('speed', [0.0, 5.0, 6.0, 24.0 / 3.6 - 1e-6])
def test_under_24kmh_resets(recogniser, variant, speed):
  rec = recogniser(variant)
  rec.process(None, None, frame(0))
  out = rec.process(None, None, frame(1, prob=0.9, speed=speed))
  assert out['left']['reason'] == 'lane_change'
  assert not rec.strip_filter.sides['left'].items


@pytest.mark.parametrize('variant', ['B', 'C2'])
@pytest.mark.parametrize('speed', [24.0 / 3.6, 8.0, 30.0 / 3.6])
def test_24kmh_and_confidence_boundary(recogniser, variant, speed):
  rec = recogniser(variant)
  rec.model.tc = rec.model.cc = 0.9
  assert rec.process(None, None, frame(0, speed=speed))['left']['typePDashed'] == 0.9


@pytest.mark.parametrize('low_head', ['type', 'color'])
def test_low_head_pauses_without_refresh_or_window_reset(recogniser, low_head):
  rec = recogniser()
  policy = DisplayPolicy()
  for i in range(2):
    f = frame(i)
    out = rec.process(None, None, f)
    policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **out), f['stamp'])
  state = rec.strip_filter.sides['left']
  before = list(state.items if low_head == 'type' else state.colors)
  old_stamp = policy.sides['left']['heads'][low_head]['stamp']
  setattr(rec.model, 'tc' if low_head == 'type' else 'cc', 0.9 - 1e-9)
  f = frame(2)
  out = rec.process(None, None, f)
  heads = policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **out), f['stamp'])
  assert out['left'][low_head] == 'unknown'
  assert list(state.items if low_head == 'type' else state.colors) == before
  assert heads['left'][low_head]['stamp'] == old_stamp
  assert out['left']['color' if low_head == 'type' else 'type'] != 'unknown'
  for i in range(3, 14):
    f = frame(i)
    out = rec.process(None, None, f)
    heads = policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **out), f['stamp'])
  assert low_head not in heads['left']  # the existing 0.5s expiry is still in force


def test_both_low_pause_and_geometry_lead_guards(recogniser):
  rec = recogniser()
  rec.model.tc = rec.model.cc = 0.89
  out = rec.process(None, None, frame(0))
  assert out['left']['reason'] == 'lane_prob_dip' and not rec.strip_filter.sides['left'].items
  f = frame(1)
  f['lines']['left']['bad'] = True
  assert rec.process(None, None, f)['left']['reason'] == 'invalid_lane_geometry'
  f = frame(2)
  f['lead_d_rel'] = 3.9
  calls = rec.model.calls
  assert rec.process(None, None, f)['left']['reason'] == 'lead_close' and rec.model.calls == calls


@pytest.mark.parametrize('last,current', [((-0.1, 3.5), (-3.5, 0.1)), ((-0.1, 3.5), (0.1, 3.6)), ((-0.1, 3.5), (0.0, 3.6))])
def test_crossing_resets_both_changed_identities(recogniser, last, current):
  rec = recogniser('B-cross')
  policy = DisplayPolicy()
  for i in range(2):
    f = frame(i, ys=last)
    out = rec.process(None, None, f)
    policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **out), f['stamp'])
  f = frame(2, ys=current)
  out = rec.process(None, None, f)
  if current[0] == 0.0:  # exact zero is retained until a sign or renumber actually changes
    assert not out['left']['trackingReset']
    f = frame(3, ys=(0.1, 3.6))
    out = rec.process(None, None, f)
  assert all(out[s]['trackingReset'] for s in ('left', 'right'))
  heads = policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **out), f['stamp'])
  assert heads == {'left': {}, 'right': {}}
  assert len(rec.strip_filter.sides['left'].items) == 1


def test_crossing_rearms_and_gap_forgets_geometry(recogniser):
  rec = recogniser('B-cross')
  seq = [(-0.1, 3.5), (-3.5, 0.1), (-0.1, 3.5), (-3.5, 0.1)]
  resets = [rec.process(None, None, frame(i, ys=ys))['left']['trackingReset'] for i, ys in enumerate(seq)]
  assert resets == [False, True, True, True]
  rec.reset()
  rec.process(None, None, frame(0, ys=(-0.1, 3.5)))
  assert not rec.process(None, None, frame(20, ys=(-3.5, 0.1)))['left']['trackingReset']
  rec.reset()
  assert not rec.crossing_y and rec.crossing_stamp is None


@pytest.mark.parametrize('prob,confidence,enters', [(0.299, 0.949, False), (0.299, 0.95, True), (0.3, 0.90, True)])
def test_optional_threshold_is_per_head_and_only_below_point3(recogniser, prob, confidence, enters):
  rec = recogniser('B-cross-0.95')
  rec.model.tc = confidence
  out = rec.process(None, None, frame(0, prob=prob))
  assert (out['left']['typePDashed'] is not None) == enters
  assert out['left']['colorPYellow'] is not None


def test_default_c2_and_replay_variants(recogniser, monkeypatch):
  monkeypatch.delenv('CCNC_LANECHANGE_INFER', raising=False)
  assert md.lanechange_of() == 'C2' and md.Recogniser(2, model=N()).lanechange == 'C2'
  rec = recogniser('A')  # the earlier rule, kept for replays
  for state in ('preLaneChange', 'laneChangeStarting', 'laneChangeFinishing'):
    assert rec._skip_reason('left', frame(0, state, prob=0.99)) == 'lane_change'
  rec = recogniser('B')
  assert rec._skip_reason('left', frame(0, 'off', prob=0.1, blinker=False)) == 'not_detected'
  monkeypatch.setenv('CCNC_LANECHANGE_INFER', 'B-cross')
  assert md.Recogniser(2, model=N()).lanechange == 'B-cross'
  with pytest.raises(ValueError):
    recogniser('bad')


@pytest.mark.parametrize('variant', ['B-p0.2', 'B-cross-p0.2'])
@pytest.mark.parametrize('state,blinker', [('off', True), ('preLaneChange', True), ('laneChangeStarting', False), ('laneChangeFinishing', False)])
@pytest.mark.parametrize('prob,enters', [(float('nan'), False), (0.0, False), (0.2 - 1e-9, False), (0.2, True), (0.7, True)])
def test_lane_probability_floor_is_inclusive_and_per_side(recogniser, variant, state, blinker, prob, enters):
  rec = recogniser(variant)
  f = frame(0, state, blinker=blinker)
  f['probs'] = {'left': prob, 'right': 0.9}
  out = rec.process(None, None, f)
  assert (out['left']['typePDashed'] is not None) == enters
  assert out['right']['typePDashed'] is not None
  assert rec.model.calls == 1 + int(enters)
  assert not out['left']['trackingReset']


@pytest.mark.parametrize('variant', ['B-p0.2', 'B-cross-p0.2'])
def test_below_floor_pauses_without_refresh_and_expires(recogniser, variant):
  rec = recogniser(variant)
  policy = DisplayPolicy()
  for i in range(2):
    f = frame(i, prob=0.2)
    raw = rec.process(None, None, f)
    policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **raw), f['stamp'])
  before = list(rec.strip_filter.sides['left'].items)
  calls = rec.model.calls
  for i in range(2, 14):
    f = frame(i, prob=0.19)
    raw = rec.process(None, None, f)
    heads = policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **raw), f['stamp'])
    assert raw['left']['reason'] == 'lane_prob_dip'
    assert not raw['left']['trackingReset']
    assert list(rec.strip_filter.sides['left'].items) == before
  assert rec.model.calls == calls
  assert heads == {'left': {}, 'right': {}}
  assert rec.process(None, None, frame(14, prob=0.2))['left']['typePDashed'] is not None


def test_floor_keeps_crossing_reset_even_when_inference_pauses(recogniser):
  rec = recogniser('B-cross-p0.2')
  rec.process(None, None, frame(0, prob=0.9, ys=(-0.1, 3.5)))
  out = rec.process(None, None, frame(1, prob=0.19, ys=(-3.5, 0.1)))
  assert all(out[s]['trackingReset'] for s in ('left', 'right'))
  assert not rec.strip_filter.sides['left'].items


@pytest.mark.parametrize('variant,floor', [('C1', 0.3), ('C2', 0.3), ('C1-0.5', 0.5)])
def test_c_probability_floor_applies_only_to_blinker_off_state(recogniser, variant, floor):
  rec = recogniser(variant)
  for state in ('preLaneChange', 'laneChangeStarting', 'laneChangeFinishing'):
    assert rec._skip_reason('left', frame(0, state, prob=0.01)) is None
  assert rec._skip_reason('left', frame(0, 'off', prob=floor)) is None
  assert rec._skip_reason('left', frame(1, 'off', prob=floor - 1e-9)) == 'lane_prob_dip'
  assert rec._skip_reason('left', frame(2, 'off', prob=floor, speed=6.0)) == 'lane_change'
  assert rec._skip_reason('left', frame(3, 'off', prob=0.2, blinker=False)) == 'not_detected'


def lifecycle_step(rec, policy, memories, i, **kw):
  right_prob = kw.pop('right_prob', None)
  f = frame(i, state='off', prob=0.9, **kw)
  if right_prob is not None:
    f['probs']['right'] = right_prob
  raw = rec.process(None, None, f)
  confirmed = policy.confirm(dict(cameraMonoTimeNanos=f['stamp'], **raw), f['stamp'])
  rec.observe_confirmation(confirmed, f)
  data = md.packet(2, i, f['stamp'], raw, confirmed, f['stamp'], {}, {})
  parsed = marking.parse(json.dumps(data), f['stamp'])
  displays = {}
  for side in ('left', 'right'):
    displays[side] = memories[side].update(f['stamp']/1e9, f['v_ego'], False, f['probs'][side], f['blinker'],
                                         *marking.current(parsed, side, f['stamp']), display_reset_ns=parsed[side].get('displayResetNs'))
  return raw, parsed, displays


@pytest.mark.parametrize('variant', ['C1', 'C2'])
@pytest.mark.parametrize('speed', [0., 6., 24. / 3.6 - 1e-6, 24. / 3.6, 8.])
def test_slowdown_clears_recognition_without_erasing_card_paint(recogniser, variant, speed):
  rec = recogniser(variant)
  policy = DisplayPolicy()
  memories = {side: SideDisplay() for side in ('left', 'right')}
  for i in range(4):
    lifecycle_step(rec, policy, memories, i)
  raw, parsed, displays = lifecycle_step(rec, policy, memories, 4, speed=speed)
  assert not displays['left'][-1] and displays['left'][2:4] == ('dashed', 'white')
  assert 'displayResetNs' not in parsed['left']
  if speed < 24. / 3.6:
    assert raw['left']['reason'] == 'lane_change' and raw['left']['type'] == raw['left']['color'] == 'unknown'
    assert not rec.strip_filter.sides['left'].items and not policy.sides['left']['heads']
  if variant == 'C2' and speed < 24. / 3.6:
    assert raw['left']['trackingReset']


def test_long_slow_blinker_hold_accepts_new_paint_as_soon_as_it_is_confirmed(recogniser):
  rec = recogniser('C2')
  policy = DisplayPolicy()
  memories = {side: SideDisplay() for side in ('left', 'right')}
  for i in range(4):
    lifecycle_step(rec, policy, memories, i)
  for i in range(4, 2004):  # 100 s low-speed turn/queue: inference remains blocked, last paint stays visible.
    raw, parsed, displays = lifecycle_step(rec, policy, memories, i, speed=0.)
    assert raw['left']['reason'] == 'lane_change'
    assert displays['left'][2:4] == ('dashed', 'white') and not displays['left'][-1]
    assert 'displayResetNs' not in parsed['left']
  rec.model.infer = lambda strips: [{'type': 'solid', 'color': 'yellow', 'typeConfidence': .99, 'colorConfidence': .99,
    'typeCandidate': 'solid', 'colorCandidate': 'yellow', 'reason': 'strip_model'} for _ in strips]
  for i in range(2004, 2006):
    _, _, displays = lifecycle_step(rec, policy, memories, i, speed=6., blinker=False)
  assert displays['left'][2:4] == ('solid', 'yellow') and not displays['left'][-1]


def test_c2_has_no_extra_post_blinker_deadline(recogniser):
  rec = recogniser('C2')
  policy = DisplayPolicy()
  memories = {side: SideDisplay() for side in ('left', 'right')}
  for i in range(4):
    lifecycle_step(rec, policy, memories, i)
  rec.model.tc = rec.model.cc = 0.89  # filtered latch alone must not count as fresh model confirmation
  for i in range(4, 24):
    _, _, displays = lifecycle_step(rec, policy, memories, i, blinker=False)
    assert not displays['left'][-1]
  _, parsed, displays = lifecycle_step(rec, policy, memories, 24, blinker=False)
  assert not displays['left'][-1]
  assert 'displayResetNs' not in parsed['left']


def test_c2_does_not_clear_the_other_side_without_blinker_evidence(recogniser):
  rec = recogniser('C2')
  policy = DisplayPolicy()
  memories = {side: SideDisplay() for side in ('left', 'right')}
  memories['right'].update(0., 10., False, 0.9, True, 'solid', 'white')
  for i in range(4):
    lifecycle_step(rec, policy, memories, i, right_prob=0.1)
  _, parsed, displays = lifecycle_step(rec, policy, memories, 4, speed=6.0, right_prob=0.1)
  assert not displays['left'][-1] and 'displayResetNs' not in parsed['left']
  assert displays['left'][2:4] == ('dashed', 'white')
  assert not rec.strip_filter.sides['left'].items  # recognition still resets on the side with fresh blinker evidence
  assert not displays['right'][-1] and 'displayResetNs' not in parsed['right']
  with pytest.raises(ValueError):
    recogniser('C3')  # rejected experiment is retained only in analysis artifacts


@pytest.mark.parametrize('token', [True, 0, -1, 101, 1.0, '1'])
def test_optional_paint_reset_token_is_strict(token):
  data = {'type': marking.PACKET_TYPE, 'version': marking.PACKET_VERSION, 'cameraMonoTimeNanos': 100, 'left': {'displayResetNs': token}}
  assert 'displayResetNs' not in marking.parse(json.dumps(data), 100)['left']
  data['left']['displayResetNs'] = 100
  assert marking.parse(json.dumps(data), 100)['left']['displayResetNs'] == 100


@pytest.mark.parametrize('relax,prob,want', [(None, .4, 'lane_prob_dip'), (None, .1, 'not_detected'), ('G1', .4, None),
                                             ('G1', .2, 'lane_prob_dip'), ('G1', .1, 'not_detected'), ('G2', .01, None)])
def test_gate_relax_switch_replays_only(recogniser, monkeypatch, relax, prob, want):
  """Gate switch: G1 (the product default since 10-26) halves the pre-10-26 lane prob floors, None restores them, G2 ignores the
  lane prob; the lane change gate is unchanged."""
  assert PRODUCT_GATE == 'G1' and (md.LANE_PROB_MIN, md.LANE_PROB_LOST) == (.25, .15)
  monkeypatch.setattr(md, 'GATE_RELAX', relax)
  rec = recogniser('C2')
  assert rec._skip_reason('left', frame(0, 'off', prob=prob, blinker=False)) == want
  assert rec._skip_reason('left', frame(1, 'laneChangeStarting', prob=prob, speed=6.0)) == 'lane_change'
