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


def lanes(send, **kw):
  values = dict(send(**kw))["ADRV_0x161"]
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


def test_existing_warnings_departure_and_inactive_still_win(display):  # noqa: F811
  params, cs, send = display
  params["CcncLaneTypeDetection"] = 1
  cs.ccnc_lane_marking = marking(left=('dashed', 'white'), right=('dashed', 'white'))
  cs.out.leftBlindspot = True
  assert lanes(send) == (4, 10)  # BSD orange unchanged on the left, recognised green dashed on the right
  cs.out.leftBlindspot = False
  params["CcncModelLanes"] = 1
  assert lanes(send, lat_active=False) == (0, 0)  # model lanes hide inactive lines exactly as before


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
