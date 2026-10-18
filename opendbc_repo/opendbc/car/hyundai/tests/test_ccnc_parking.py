from types import SimpleNamespace as N

import pytest

from opendbc.car.hyundai import ccnc_parking as cp
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401  (fixture)


def model(p=.02, std=3., state='off'):
  return N(laneLineProbs=[p, p, p, p], roadEdgeStds=[std, std], meta=N(laneChangeState=state))


def drive(hide, t0, seconds, v, md, left=False, right=False, desire=0, dt=.05):
  t = t0
  out = None
  for _ in range(int(round(seconds/dt))):
    t += dt
    out = hide.update(t, v, md, left, right, desire)
  return t, out


def test_enters_after_slow_lane_free_travel_not_while_standing():
  hide = cp.ParkingLaneHide()
  t, hidden = drive(hide, 0., 600., 0., model())  # ten minutes stopped (red light, queue): never enters
  assert not hidden
  t, hidden = drive(hide, t, 2.4, 2., model())    # 4.8 m
  assert not hidden
  t, hidden = drive(hide, t, .2, 2., model())     # 5.2 m
  assert hidden
  t, hidden = drive(hide, t, 300., 0., model(p=.9, std=.3))  # standing still keeps the state
  assert hidden


@pytest.mark.parametrize('md,v', [(model(p=.6), 2.), (model(std=.5), 2.), (model(), 6.)])
def test_lanes_edges_or_speed_reset_entry_evidence(md, v):
  hide = cp.ParkingLaneHide()
  t, _ = drive(hide, 0., 2., 2., model())
  t, _ = drive(hide, t, .1, v, md)
  t, hidden = drive(hide, t, 2., 2., model())     # only 4 m since the reset
  assert not hidden


def test_lane_change_guard_and_grace_but_hazards_are_not_a_lane_change():
  hide = cp.ParkingLaneHide()
  t, hidden = drive(hide, 0., 10., 2., model(), left=True)
  assert not hidden                                # one blinker: no new entry
  t, hidden = drive(hide, t, 2.9, 2., model())     # still inside the 3 s grace
  assert not hidden and hide.evidence_m == 0.
  t, hidden = drive(hide, t, 2.8, 2., model())
  assert hidden
  for kw in ({'desire': 3}, {}):
    hide2 = cp.ParkingLaneHide()
    md = model(state='laneChangeStarting') if not kw else model()
    _, hidden2 = drive(hide2, 0., 3., 2., md, **kw)
    assert not hidden2
  hazards = cp.ParkingLaneHide()
  _, hidden3 = drive(hazards, 0., 3., 2., model(), left=True, right=True)
  assert hidden3


def test_blinker_does_not_release_and_exit_hysteresis():
  hide = cp.ParkingLaneHide()
  t, _ = drive(hide, 0., 3., 2., model())
  t, hidden = drive(hide, t, 5., 2., model(), right=True)
  assert hidden                                    # a blinker alone never releases
  t, hidden = drive(hide, t, 4.5, 2., model(p=.9, std=.5))  # 9 m with lanes visible
  assert hidden
  t, hidden = drive(hide, t, .1, 2., model())      # lanes lost again: exit evidence restarts
  t, hidden = drive(hide, t, 4.5, 2., model(p=.9, std=.5))
  assert hidden
  t, hidden = drive(hide, t, .6, 2., model(p=.9, std=.5))  # 10.2 m continuous
  assert not hidden
  hide2 = cp.ParkingLaneHide()
  t, _ = drive(hide2, 0., 3., 2., model())
  t, hidden = drive(hide2, t, .9, 9., model())
  assert hidden
  t, hidden = drive(hide2, t, .2, 9., model())     # >= 30 km/h for 1 s
  assert not hidden


def test_no_model_releases_after_one_second():
  hide = cp.ParkingLaneHide()
  t, _ = drive(hide, 0., 3., 2., model())
  t, hidden = drive(hide, t, .9, 2., None)
  assert hidden
  t, hidden = drive(hide, t, .2, 2., None)
  assert not hidden


def test_apply_hides_both_lines_only_while_hidden():
  values = {"LANELINE_LEFT": 4, "LANELINE_RIGHT": 10}
  assert not hasattr(cp, 'LEFT_HIDE_CODE_VERIFIED')
  assert cp.apply(dict(values), True) == {"LANELINE_LEFT": 14, "LANELINE_RIGHT": 14}
  assert cp.apply(dict(values), False) == values


def parking_cs(cs):
  cs.modelV2.laneLineProbs = [.02]*4
  cs.modelV2.roadEdgeStds = [3., 3.]
  cs.modelV2.meta.laneChangeState = 'off'
  cs.out.vEgo = 2.


def test_cluster_parking_hide_overrides_lane_marking_and_returns(display, monkeypatch):  # noqa: F811
  from opendbc.car.hyundai import hyundaicanfd as main
  params, cs, send = display
  monkeypatch.delattr(main.create_ccnc_messages, '_parking', raising=False)
  params["CcncLaneTypeDetection"] = 1
  far = 10**18
  cs.ccnc_lane_marking = {'left': {'type': ('solid', far), 'color': ('yellow', far)}, 'right': {'type': ('dashed', far), 'color': ('white', far)}}
  parking_cs(cs)

  def lanes(frame):
    v = dict(send(frame))["ADRV_0x161"]
    return v["LANELINE_LEFT"], v["LANELINE_RIGHT"]
  assert lanes(0) == (4, 10)                       # yellow > green dashed before the lot is confirmed
  frame = 0
  for frame in range(5, 400, 5):                   # 2 m/s for ~4 s
    codes = lanes(frame)
  assert codes == (14, 14)                         # both lines hidden on the cluster
  cs.modelV2.laneLineProbs = [.95]*4
  cs.modelV2.roadEdgeStds = [.3, .3]
  start = frame
  for frame in range(start + 5, start + 700, 5):   # lanes visible for > 10 m: back to the normal display
    codes = lanes(frame)
  assert codes == (4, 10)
