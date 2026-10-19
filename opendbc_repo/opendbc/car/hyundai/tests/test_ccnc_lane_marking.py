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
  for frame in range(5, 100, 5):
    assert lanes(send, frame) == (10, 10)  # < 1 s of low prob: unchanged
  assert lanes(send, 100) == (0, 10)  # 1 s below 0.3 while driving: grey, recognised type not applied
  cs.modelV2.laneLineProbs = [.9, .4, .9, .9]
  assert lanes(send, 105) == (0, 10)  # hysteresis: released only at >= 0.5
  cs.modelV2.laneLineProbs = [.9]*4
  assert lanes(send, 110) == (10, 10)
  cs.out.vEgo = .2  # stopped: lat off (white base), the last recognised type is kept without a time limit
  cs.ccnc_lane_marking = None
  assert lanes(send, 115, lat_active=False) == (8, 8)
  assert lanes(send, 2000, lat_active=False) == (8, 8)
  cs.out.leftBlinker = True
  assert lanes(send, 2005, lat_active=False) == (2, 2)  # a blinker releases the held type


class TestSideDisplay:
  def test_hold_two_seconds_while_driving_then_release(self):
    d = lm.SideDisplay()
    assert d.update(0., 10., False, .9, False, 'dashed', 'white')[2:] == ('dashed', 'white')
    assert d.update(1.9, 10., False, .9, False, None, None)[2:] == ('dashed', 'white')
    assert d.update(2.1, 10., False, .9, False, None, None)[2:] == (None, None)
    assert d.update(2.2, 10., False, .9, False, None, None)[2:] == (None, None)

  def test_new_fresh_value_replaces_hold_and_missing_head_is_filled(self):
    d = lm.SideDisplay()
    d.update(0., 10., False, .9, False, 'dashed', 'white')
    assert d.update(.1, 10., False, .9, False, 'solid', 'white')[2:] == ('solid', 'white')
    assert d.update(.2, 10., False, .9, False, None, 'yellow')[2:] == ('solid', 'yellow')  # fresh colour wins
    assert d.update(.3, 10., False, .9, False, None, None)[2:] == ('solid', 'yellow')  # each head holds its last fresh value
    assert d.update(2.15, 10., False, .9, False, None, None)[2:] == (None, 'yellow')  # type last fresh at 0.1 s, colour at 0.2 s
    assert d.update(2.25, 10., False, .9, False, None, None)[2:] == (None, None)

  def test_stop_hold_has_no_time_limit_and_blinker_or_lane_change_releases(self):
    d = lm.SideDisplay()
    d.update(0., 3., False, .9, False, 'dashed', 'white')
    for t in (1., 5., 30., 120.):
      assert d.update(t, .1, False, .9, False, None, None)[2:] == ('dashed', 'white')
    assert d.update(120.1, 2., False, .9, False, None, None)[2:] == ('dashed', 'white')  # moving again: 2 s from the last stop
    assert d.update(121., .1, False, .9, True, None, None)[2:] == (None, None)
    d.update(122., 10., False, .9, False, 'dashed', 'white')
    assert d.update(122.1, 10., False, .9, True, None, None)[2:] == (None, None)

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

  def test_low_prob_one_second_release_at_half_and_frozen_while_stopped(self):
    d = lm.SideDisplay()
    t = 0.
    for _ in range(19):
      t += .05
      assert d.update(t, 10., False, .1, False, None, None)[1] is False
    for _ in range(40):  # stopped: neither builds nor releases
      t += .05
      assert d.update(t, .2, False, .1, False, None, None)[1] is False
    states = []
    for _ in range(3):  # 0.9 s built before the stop; ~0.1 s more of driving completes it
      t += .05
      states.append(d.update(t, 10., False, .1, False, None, None)[1])
    assert states[0] is False and states[-1] is True
    t += .05
    assert d.update(t, .2, False, .9, False, None, None)[1] is True  # frozen while stopped
    assert d.update(t + .05, 10., False, .45, False, None, None)[1] is True
    assert d.update(t + .1, 10., False, .5, False, None, None)[1] is False
    d.update(t + .15, 10., False, .1, False, None, None)
    assert d.update(t + .2, 10., False, .35, False, None, None)[1] is False  # a short dip restarts the 1 s


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
  assert dict(send())["ADRV_0x161"] == baseline  # expired: existing display
  cs.ccnc_lane_marking = None
  assert dict(send())["ADRV_0x161"] == baseline  # no recogniser output yet
