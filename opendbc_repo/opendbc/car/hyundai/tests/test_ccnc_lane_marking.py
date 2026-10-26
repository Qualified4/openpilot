from itertools import product
import time

import pytest

from opendbc.car.hyundai import ccnc_lane_marking as lm
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401  (fixture)

FAR = 10**18


def marking(left=(None, None), right=(None, None), expires=FAR):
  def heads(kind, color):
    return {'type': (kind, expires) if kind else None, 'color': (color, expires) if color else None}
  return {'mode': 'scanner', 'left': heads(*left), 'right': heads(*right)}


@pytest.mark.parametrize("existing,kind,color", list(product((2, 6), ('dashed', 'solid', None), ('white', 'yellow', None))))
def test_priority_yellow_then_green_then_white(existing, kind, color):
  code = lm.lane_code(existing, marking(left=(kind, color)), 'left', 0)
  if color == 'yellow':
    assert code == 4  # yellow beats lane-change-available green
  elif color == 'white' and kind == 'dashed':
    assert code == (10 if existing == 6 else 8)
  elif color == 'white' and kind == 'solid':
    assert code == existing
  else:
    assert code == existing  # unknown head: existing display policy


@pytest.mark.parametrize("existing", [0, 1, 4, 14])
def test_hidden_warning_and_other_codes_are_never_changed(existing):
  for m in (marking(left=('solid', 'yellow')), marking(left=('dashed', 'white'))):
    assert lm.lane_code(existing, m, 'left', 0) == existing


def test_expiry_none_and_sides_are_independent():
  m = marking(left=('solid', 'yellow'), right=('dashed', 'white'), expires=100)
  assert lm.lane_code(6, m, 'left', 100) == 4 and lm.lane_code(6, m, 'right', 100) == 10
  assert lm.lane_code(6, m, 'left', 101) == 6 and lm.lane_code(2, m, 'right', 101) == 2  # expired
  assert lm.lane_code(2, None, 'left', 0) == 2
  only_left = marking(left=('solid', 'yellow'))
  assert lm.lane_code(6, only_left, 'right', 0) == 6


def lanes(send, frame=0, **kw):
  values = dict(send(frame, **kw))["ADRV_0x161"]
  return values["LANELINE_LEFT"], values["LANELINE_RIGHT"]


@pytest.mark.parametrize("mode", [1, 2])
def test_cluster_codes_follow_priority_per_side(display, mode):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = mode
  cs.ccnc_lane_marking = marking(left=('solid', 'yellow'), right=('dashed', 'white'))
  assert lanes(send) == (4, 10)  # both sides lane-change-available green in the fixture
  cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = False
  assert lanes(send) == (4, 8)
  cs.ccnc_lane_marking = marking(left=('dashed', 'white'), right=('solid', 'yellow'))
  assert lanes(send) == (8, 4)


def test_bsd_no_orange_lat_off_white_and_lane_warning_still_wins(display):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = 1
  cs.ccnc_lane_marking = marking(left=('dashed', 'white'), right=('dashed', 'white'))
  cs.out.leftBlindspot = True
  assert lanes(send) == (10, 10)  # BSD no longer paints the line orange (BSD clears laneChangeAvailable on the car)
  cs.out.leftBlindspot = False
  params["CcncModelLanes"] = 1
  assert lanes(send, lat_active=False) == (8, 8)  # lat off: white base with the recognised shape, never green
  cs.out.leftLaneLine = 20  # existing OEM yellow lane warning still wins
  assert lanes(send) == (4, 10)


def test_cluster_low_prob_grey_and_stop_hold(display):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = 1
  params["CcncModelLanes"] = 1
  cs.ccnc_lane_marking = marking(left=('dashed', 'white'), right=('dashed', 'white'))
  assert lanes(send, 0) == (10, 10)
  cs.modelV2.laneLineProbs = [.9, .2, .9, .9]
  for frame in range(5, 150, 5):
    assert lanes(send, frame) == (10, 10)  # < 1.5 s of low prob: unchanged
  assert lanes(send, 150) == (0, 10)  # 1.5 s below 0.5 while driving: grey, recognised type not applied
  cs.modelV2.laneLineProbs = [.9, .45, .9, .9]
  assert lanes(send, 155) == (0, 10)  # released only at >= 0.5
  cs.modelV2.laneLineProbs = [.9]*4
  assert lanes(send, 160) == (10, 10)
  cs.out.vEgo = .2  # stopped: lat off (white base), the last recognised type is kept without a time limit
  cs.ccnc_lane_marking = None
  assert lanes(send, 165, lat_active=False) == (8, 8)
  assert lanes(send, 2000, lat_active=False) == (8, 8)
  cs.out.leftBlinker = True
  assert lanes(send, 2005, lat_active=False) == (8, 8)  # a blinker keeps the held type through the lane change
  cs.out.leftBlinker = False
  cs.out.vEgo = 10.
  assert lanes(send, 2010, lat_active=False) == (8, 8)
  assert lanes(send, 2500, lat_active=False) == (8, 8)  # 5 s hold after the change's last frame (2005)
  assert lanes(send, 2510, lat_active=False) == (0, 0)  # hold ran out unconfirmed: grey, not the solid-looking white
  cs.ccnc_lane_marking = marking(left=('solid', 'white'), right=('dashed', 'white'))
  assert lanes(send, 2520, lat_active=False) == (2, 8)  # confirmed again


def test_cluster_lapsed_grey_only_with_the_type_option_on(display, monkeypatch):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = 2
  cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = False
  cs.ccnc_lane_marking = marking(left=('dashed', 'white'), right=('solid', 'yellow'))
  assert lanes(send, 0) == (8, 4)
  cs.ccnc_lane_marking = None
  assert lanes(send, 500) == (8, 4)
  assert lanes(send, 505) == (0, 0)
  from opendbc.car.hyundai import hyundaicanfd as main
  monkeypatch.delattr(main.create_ccnc_messages, "_lane_display", raising=False)
  monkeypatch.delattr(main.create_ccnc_messages, "_display_options", raising=False)
  params["CcncLaneTypeDetection"] = 0
  assert lanes(send, 600) == (2, 2)
  assert lanes(send, 1200) == (2, 2)  # OFF: the existing base, never the lapsed grey


class TestSideDisplay:
  def test_hold_five_seconds_while_driving_then_lapsed_grey_until_confirmed(self):
    d = lm.SideDisplay()
    assert d.update(0., 10., False, .9, False, 'dashed', 'white')[2:5] == ('dashed', 'white', False)
    assert d.update(4.9, 10., False, .9, False, None, None)[2:5] == ('dashed', 'white', False)
    assert d.update(5.1, 10., False, .9, False, None, None)[2:5] == (None, None, True)
    assert d.update(60., 10., False, .9, False, None, None)[2:5] == (None, None, True)  # until the recogniser confirms
    assert d.update(60.1, .1, False, .9, False, None, None)[2:5] == (None, None, True)  # stopping does not bring it back
    assert d.update(60.2, 10., False, .9, False, 'solid', 'white')[2:5] == ('solid', 'white', False)
    off = lm.SideDisplay()
    off.update(0., 10., False, .9, False, 'dashed', 'white', False)
    assert off.update(5.1, 10., False, .9, False, None, None, False)[2:5] == (None, None, False)  # option OFF: existing base
    assert lm.SideDisplay().update(9., 10., False, .9, False, None, None)[4:] == (False, True)  # never confirmed: hidden 14
    assert lm.SideDisplay().update(9., 10., False, .9, False, None, None, False)[4:] == (False, False)  # OFF: existing base

  def test_lapsed_grey_counts_only_the_heads_the_code_needs(self):
    d = lm.SideDisplay()
    d.update(0., 10., False, .9, False, 'dashed', 'white')
    assert d.update(4., 10., False, .9, False, None, 'yellow')[2:5] == ('dashed', 'yellow', False)
    assert d.update(5.5, 10., False, .9, False, None, 'yellow')[2:5] == (None, 'yellow', False)  # type lapsed: yellow is orange anyway
    assert d.update(5.6, 10., False, .9, False, None, 'white')[2:5] == (None, 'white', True)  # white needs the type
    assert d.update(5.7, 10., False, .9, False, 'solid', None)[2:5] == ('solid', 'white', False)
    e = lm.SideDisplay()
    e.update(0., 10., False, .9, False, None, 'white')  # type never confirmed, colour lapses
    assert e.update(5.1, 10., False, .9, False, None, None)[4] is True
    assert e.update(5.2, 10., False, .9, False, None, 'white')[4] is False  # white, type never confirmed: existing base

  def test_new_fresh_value_replaces_hold_and_missing_head_is_filled(self):
    d = lm.SideDisplay()
    d.update(0., 10., False, .9, False, 'dashed', 'white')
    assert d.update(.1, 10., False, .9, False, 'solid', 'white')[2:4] == ('solid', 'white')
    assert d.update(.2, 10., False, .9, False, None, 'yellow')[2:4] == ('solid', 'yellow')  # fresh colour wins
    assert d.update(.3, 10., False, .9, False, None, None)[2:4] == ('solid', 'yellow')  # each head holds its last fresh value
    assert d.update(5.15, 10., False, .9, False, None, None)[2:4] == (None, 'yellow')  # type last fresh at 0.1 s, colour at 0.2 s
    assert d.update(5.25, 10., False, .9, False, None, None)[2:4] == (None, None)

  def test_stop_and_lane_change_hold_have_no_time_limit(self):
    d = lm.SideDisplay()
    d.update(0., 3., False, .9, False, 'dashed', 'white')
    for t in (1., 5., 30., 120.):
      assert d.update(t, .1, False, .9, False, None, None)[2:5] == ('dashed', 'white', False)
    assert d.update(124.9, 2., False, .9, False, None, None)[2:5] == ('dashed', 'white', False)  # moving again: 5 s from the last stop
    d.update(130., 10., False, .9, False, 'solid', 'yellow')
    for t in (130.1, 133., 137., 140.):  # blinker / desire / laneChangeState: the last confirmed line stays
      assert d.update(t, 20., False, .9, True, None, None)[2:5] == ('solid', 'yellow', False)
    assert d.update(144.9, 20., False, .9, False, None, None)[2:5] == ('solid', 'yellow', False)  # 5 s from the change's end
    assert d.update(145.1, 20., False, .9, False, None, None)[2:5] == (None, None, True)
    d.update(150., 20., False, .9, True, 'dashed', 'white')
    assert d.update(150.1, 20., False, .9, True, 'solid', None)[2:5] == ('solid', 'white', False)  # new lane confirmed

  def test_creeping_below_two_metres_per_second_holds_without_a_time_limit(self):
    """DaBin 10-19 (fe/5 R 34.9 s: 5.9 s grey in a slow queue): the no-limit hold covers creeping (< 2 m/s), not just a
    stop; the low-prob grey still builds above 0.5 m/s."""
    assert lm.HOLD_STOP_SPEED == 2. and lm.STOP_SPEED == .5
    d = lm.SideDisplay()
    d.update(0., 5., False, .9, False, 'dashed', 'white')
    for t in (1., 5.1, 30., 60.):
      assert d.update(t, 1.9, False, .9, False, None, None)[2:5] == ('dashed', 'white', False)
    assert d.update(64.9, 2., False, .9, False, None, None)[2:5] == ('dashed', 'white', False)  # 2 m/s: 5 s from the last slow frame
    assert d.update(65.1, 2., False, .9, False, None, None)[2:5] == (None, None, True)
    low = lm.SideDisplay()
    t = 0.
    low.update(t, 1., False, .9, False, 'dashed', 'white')
    for _ in range(31):  # creeping at 1 m/s: the hold stays, the low-prob clock runs (frozen only below 0.5 m/s)
      t += .05
      out = low.update(t, 1., False, .1, False, None, None)
    assert out[1] is True and out[2:5] == ('dashed', 'white', False)

  def test_low_prob_grey_comes_before_the_hold(self):
    d = lm.SideDisplay()
    t = 0.
    d.update(t, 10., False, .9, False, 'dashed', 'white')
    for _ in range(31):
      t += .05
      out = d.update(t, 10., False, .1, False, None, None)
    assert out[1] is True and out[2:5] == ('dashed', 'white', False)  # low (grey 0) is applied by the caller first

  def test_green_on_after_one_second_off_at_once(self):
    d = lm.SideDisplay()
    assert d.update(0., 10., False, .9, False, None, None)[0] is False
    t = 0.
    for _ in range(19):
      t += .05
      assert d.update(t, 10., True, .9, False, None, None)[0] is False
    assert d.update(t + .05, 10., True, .9, False, None, None)[0] is True
    assert d.update(t + .1, 10., False, .9, False, None, None)[0] is False
    assert lm.SideDisplay().update(0., 10., True, .9, False, None, None)[0] is True  # first frame: no delay

  def test_low_prob_one_and_a_half_seconds_below_half_and_frozen_while_stopped(self):
    d = lm.SideDisplay()
    t = 0.
    for _ in range(29):
      t += .05
      assert d.update(t, 10., False, .1, False, None, None)[1] is False
    for _ in range(40):  # stopped: neither builds nor releases
      t += .05
      assert d.update(t, .2, False, .1, False, None, None)[1] is False
    states = []
    for _ in range(3):  # 1.4 s built before the stop; ~0.1 s more of driving completes it
      t += .05
      states.append(d.update(t, 10., False, .1, False, None, None)[1])
    assert states[0] is False and states[-1] is True
    t += .05
    assert d.update(t, .2, False, .9, False, None, None)[1] is True  # frozen while stopped
    assert d.update(t + .05, 10., False, .45, False, None, None)[1] is True
    assert d.update(t + .1, 10., False, .5, False, None, None)[1] is False
    d.update(t + .15, 10., False, .1, False, None, None)
    assert d.update(t + .2, 10., False, .55, False, None, None)[1] is False  # >= 0.5 restarts the 1.5 s

  def test_wandering_prob_below_half_turns_grey_after_one_and_a_half_seconds(self):
    d = lm.SideDisplay()
    t, states = 0., []
    for i in range(40):  # an intersection: 0.2 / 0.4 alternating, never 0.3 continuously for long
      t += .05
      states.append(d.update(t, 12., False, .2 if i % 2 else .4, False, None, None)[1])
    first = states.index(True)
    assert 28 <= first <= 30 and all(states[first:])  # ~1.5 s

  def test_changing_lanes_keeps_the_last_confirmed_line_instead_of_grey(self):
    for v in (3., 15.):  # no speed condition
      d = lm.SideDisplay()
      d.update(0., v, False, .9, False, 'dashed', 'white')
      t = 0.
      for _ in range(60):  # 3 s of 0.05 prob with the blinker on
        t += .05
        out = d.update(t, v, False, .05, True, None, None)
        assert out[1] is False and out[2:6] == ('dashed', 'white', False, False)
    d = lm.SideDisplay()
    d.update(0., 12., False, .9, False, 'solid', 'yellow')
    t = 0.
    for _ in range(40):
      t += .05
      out = d.update(t, 12., False, .1, False, None, None)
    assert out[1] is True  # grey before the blinker
    out = d.update(t + .05, 12., False, .1, True, None, None)
    assert out[1] is False and out[2:5] == ('solid', 'yellow', False)  # blinker on: the last confirmed line again
    lapsed = lm.SideDisplay()
    lapsed.update(0., 12., False, .9, False, 'dashed', 'white')
    assert lapsed.update(5.1, 12., False, .9, False, None, None)[2:5] == (None, None, True)
    assert lapsed.update(5.2, 12., False, .9, True, None, None)[2:5] == ('dashed', 'white', False)  # also after lapsing
    never = lm.SideDisplay()
    t = 0.
    for _ in range(31):
      t += .05
      out = never.update(t, 12., False, .1, True, None, None)
    assert out[1] is True  # never confirmed: low applies as before (shown as 14 while unseen)

  def test_hidden_until_the_first_colour_is_confirmed(self):
    d = lm.SideDisplay()
    assert d.update(0., 10., False, .9, False, None, None)[5] is True
    assert d.update(.05, 10., False, .9, False, 'solid', None)[5] is True  # a type alone does not end it
    assert d.update(.1, 10., False, .9, False, 'solid', 'yellow')[5] is False
    assert d.update(60., 10., False, .9, False, None, None)[4:] == (True, False)  # lapsed grey later, never hidden again


def test_off_and_expired_results_match_existing_display(display, monkeypatch):  # noqa: F811
  params, cs, send = display
  baseline = dict(send())["ADRV_0x161"]
  cs.ccnc_lane_marking = marking(left=('solid', 'yellow'), right=('dashed', 'white'))
  params["CcncLaneTypeDetection"] = 0
  assert dict(send())["ADRV_0x161"] == baseline  # OFF: the packet is ignored entirely
  params["CcncLaneTypeDetection"] = 1
  from opendbc.car.hyundai import hyundaicanfd as main
  monkeypatch.delattr(main.create_ccnc_messages, "_display_options", raising=False)
  cs.ccnc_lane_marking = marking(left=('solid', 'yellow'), right=('dashed', 'white'), expires=time.monotonic_ns() - 1)
  hidden = dict(baseline, LANELINE_LEFT=14, LANELINE_RIGHT=14)
  assert dict(send())["ADRV_0x161"] == hidden  # expired, nothing confirmed since start: hidden
  cs.ccnc_lane_marking = None
  assert dict(send())["ADRV_0x161"] == hidden  # no recogniser output yet


def test_cluster_start_hidden_until_confirmed_and_priorities(display):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = 2
  assert lanes(send, 0) == (14, 14)
  cs.out.leftLaneLine = 20  # OEM lane warning wins over the start-up hide
  assert lanes(send, 5) == (4, 14)
  cs.out.leftLaneLine = 0
  cs.modelV2.laneLineProbs = [.9, .1, .1, .9]
  for frame in range(10, 200, 5):
    assert lanes(send, frame) == (14, 14)  # hide wins over the low-prob grey
  cs.modelV2.laneLineProbs = [.9]*4
  cs.ccnc_lane_marking = marking(left=('solid', 'yellow'), right=('dashed', 'white'))
  assert lanes(send, 205) == (4, 10)  # first confirmation: drawn at once
  cs.ccnc_lane_marking = None
  assert lanes(send, 210) == (4, 10)


def test_lane_change_hold_limit_switch_replays_only(monkeypatch):
  """rollback1026 H (default off): a lane-change hold with the lane prob < 0.5 for LC_HOLD_LIMIT_S ends in lapsed grey; a fresh
  value is shown at once again."""
  assert lm.LC_HOLD_LIMIT_S is None
  for limit, grey in ((None, False), (1.5, True)):
    monkeypatch.setattr(lm, 'LC_HOLD_LIMIT_S', limit)
    d = lm.SideDisplay()
    d.update(0., 20., False, .9, False, 'solid', 'yellow')
    for k in range(1, 41):  # 2 s changing lanes at 20 m/s, lane prob 0.2, nothing fresh
      out = d.update(k*.05, 20., False, .2, True, None, None)
    assert out[4] is grey and (out[3] is None) is grey and (out[3] == 'yellow') is not grey
    assert d.update(2.1, 20., False, .2, True, 'dashed', 'white')[2:5] == ('dashed', 'white', False)
