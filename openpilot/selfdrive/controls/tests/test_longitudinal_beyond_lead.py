"""A slow car beyond leadOne in ego's path counts as leadTwo only while leadOne moves aside."""
from types import SimpleNamespace as NS

import pytest

from openpilot.selfdrive.controls.lib import longitudinal_beyond_lead as bl

DT = 0.05
PATH = tuple((float(x), 0.0) for x in range(0, 101, 5))
V_EGO = 14.0


def lead(t, aside=True):
  # leadOne 25 m ahead at ego speed; moving aside: dPath grows 0.5 -> 1.5 m over 2 s
  return NS(status=True, radarTrackId=7, dRel=25.0, vLead=V_EGO, dPath=(0.5 + 0.5 * t) if aside else 0.1)


def track(track_id=9, d=45.0, y=0.0, v_abs=4.0):
  return NS(trackId=track_id, dRel=d, yRel=y, vRel=v_abs - V_EGO, vLead=v_abs)


def run(seconds=2.0, aside=True, tracks=lambda t: [track()], history=()):
  s = bl.BeyondLeadState()
  out = None
  for pts in history:
    s.update(live_tracks=NS(points=pts), lead_one=lead(0.0, aside=False), v_ego=V_EGO, path=PATH, dt=DT)
  for i in range(round(seconds / DT)):
    t = i * DT
    out = s.update(live_tracks=NS(points=tracks(t)), lead_one=lead(t, aside), v_ego=V_EGO, path=PATH, dt=DT)
  return out


def test_slow_car_beyond_a_lead_moving_aside_becomes_lead_two():
  out = run()
  assert out is not None and out.status and out.radarTrackId == 9
  assert out.dRel == pytest.approx(45.0) and out.vLead == pytest.approx(4.0) and out.aLeadK == 0.0


def test_not_before_it_persisted():
  assert run(seconds=bl.BEYOND_PERSIST_S - 0.2) is None


def test_not_while_the_lead_stays_in_the_path():
  assert run(aside=False) is None


@pytest.mark.parametrize("pts", [
  [track(y=bl.BEYOND_PATH_HALF_WIDTH + 0.3)],          # beside the path
  [track(d=28.0)],                                     # not beyond the lead
  [track(v_abs=bl.BEYOND_SPEED_MAX + 1.0)],            # not slow
  [track(v_abs=0.0)],                                  # stopped and never seen moving: roadside clutter
])
def test_clutter_never_counts(pts):
  assert run(tracks=lambda t: pts) is None


def test_stopped_car_seen_moving_before_counts():
  moving = [[track(v_abs=5.0)]] * 10
  assert run(tracks=lambda t: [track(v_abs=0.0)], history=moving) is not None


def test_nearest_qualifying_track_and_reset_without_lead():
  out = run(tracks=lambda t: [track(track_id=9, d=60.0), track(track_id=11, d=40.0)])
  assert out.radarTrackId == 11
  s = bl.BeyondLeadState()
  assert s.update(live_tracks=NS(points=[track()]), lead_one=NS(status=False), v_ego=V_EGO, path=PATH, dt=DT) is None
  assert s.update(live_tracks=None, lead_one=lead(0.0), v_ego=V_EGO, path=PATH, dt=DT) is None


@pytest.mark.parametrize("history", [
  [[track(d=70.0, v_abs=5.0)]] * 10,                       # the slot's previous car was 25 m farther
  [[track(y=3.0, v_abs=5.0)]] * 10,                        # ... or in another lane
  [[track(v_abs=5.0)]] * 10 + [[]] * 12,                   # ... or the id was unseen for 0.6 s
])
def test_reused_track_id_does_not_pass_its_moving_history_to_a_stopped_object(history):
  # judge-rc2 B1: the owner's radar reuses track ids (slots), a stopped object must not inherit v_max
  assert run(tracks=lambda t: [track(v_abs=0.0)], history=history) is None
