from itertools import product
from types import SimpleNamespace as N

import pytest

from opendbc.car import structs
from opendbc.car.hyundai import hyundaicanfd as main, hyundaicanfd_ccnc as custom
from opendbc.car.hyundai.values import HyundaiFlags

KEYS = ("CcncLaneColor", "CcncModelLanes", "CcncRadarVehicles")


@pytest.fixture
def display(monkeypatch):
  params = dict.fromkeys(KEYS, False)
  monkeypatch.setattr(main, "Params", lambda: N(get_bool=lambda key: params[key], get_int=lambda key: 0, get=lambda key: "0"))
  monkeypatch.setattr(custom, "Params", lambda: N(get_int=lambda key: 3))
  monkeypatch.delattr(main.create_ccnc_messages, "_display_options", raising=False)
  monkeypatch.setattr(custom, "state", N())
  monkeypatch.setattr(custom, "_options", None)
  custom.reset()
  md = N(meta=N(desire=N(raw=0), desireState=[], laneChangeAvailableLeft=True, laneChangeAvailableRight=True),
         position=N(x=[0., 10., 30.], y=[0., 0.5, 2.], yStd=[0., 0., 0.]),
         laneLineProbs=[.9]*4, laneLines=[N(y=[y]) for y in (-5., -1., 2., 6.)])
  out = N(brakeHoldActive=False, parkingBrake=False, vEgo=10., aEgo=.3, steeringAngleDeg=6.,
          latEnabled=True, steeringPressed=False, vehicleNaviAvailable=False, vCruiseCluster=0.,
          cruiseState=N(available=True), gearShifter=structs.CarState.GearShifter.drive,
          leftLaneLine=0, rightLaneLine=0, leftBlindspot=False, rightBlindspot=False,
          leftBlinker=False, rightBlinker=False)
  source = dict(ALERTS_1=0, ALERTS_2=0, ALERTS_3=0, ALERTS_5=0, SLA_ICON=0, LKA_ICON=0,
                LANE_HIGHLIGHT=0, LANE_HIGHLIGHT_DISTANCE=0, LANE_LEFT=0, LANE_RIGHT=0,
                LANELINE_LEFT_POSITION=15, LANELINE_RIGHT_POSITION=15)
  cs = N(out=out, modelV2=md, radarState=None, lfahda_cluster=None, cruise_buttons_msg=None,
         adrv_0x161=source, adrv_0x200=None, adrv_0x1ea=None, ccnc_0x162=None,
         paddle_button_prev=0, softHoldActive=0, trailer_connected=False, is_metric=True)
  hud = structs.CarControl().hudControl
  hud.leftLaneVisible = hud.rightLaneVisible = True
  packer = N(make_can_msg=lambda name, bus, values, **kwargs: (name, dict(values)))

  def send(frame=0, flags=HyundaiFlags.CAMERA_SCC.value, lat_active=True, custom_ccnc=None):
    return main.create_ccnc_messages(N(flags=flags), packer, N(ECAN=0, CAM=2), frame,
                                    N(enabled=True, latActive=lat_active), cs, hud, 0, False, False, 0, False, 0, 0,
                                    custom_ccnc=custom_ccnc)
  return params, cs, send


@pytest.mark.parametrize("options", list(product((False, True), repeat=3)))
def test_three_independent_options(display, monkeypatch, options):
  params, cs, send = display
  params.update(zip(KEYS, options))
  calls = []
  monkeypatch.setattr(custom, "update_vehicles", lambda values, *args: calls.append(args[-1]) or values)
  monkeypatch.setattr(main, "_apply_ccnc_lead", lambda *args: None)
  cs.ccnc_0x162 = dict(SPEEDLIMIT=0, FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
  color, geometry, radar = options
  for frame in (0, 5, 10, 15):
    values = dict(send(frame))["ADRV_0x161"]
  assert (values["LANE_HIGHLIGHT_DISTANCE"] > 0) == color
  assert (values["LANELINE_LEFT_POSITION"] != 15) == geometry
  assert (values["LANELINE_CURVATURE"] != 2) == geometry
  assert calls == ([geometry] * 4 if radar else [])
  # Model geometry and animation switch together, independently of driving-mode colors.
  cs.out.leftBlinker = True
  cs.modelV2.meta.desire.raw = 3
  cs.modelV2.meta.laneChangeAvailableRight = False
  values = dict(send(20))["ADRV_0x161"]
  assert values["LCA_RIGHT_ICON"] == (4 if geometry else 2)
  if not geometry:
    assert values["LANELINE_LEFT_POSITION"] == 15
    assert values["LANELINE_CURVATURE"] == 2
  if not color:
    assert values["LANE_HIGHLIGHT_DISTANCE"] == 0


def test_hda2_never_calls_front_radar_display(display, monkeypatch):
  params, cs, send = display
  params.update(dict.fromkeys(KEYS, True))
  cs.ccnc_0x162 = dict(SPEEDLIMIT=0, FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
  monkeypatch.setattr(main, "_apply_ccnc_lead", lambda *args: None)
  monkeypatch.setattr(custom, "update_vehicles", lambda *args: pytest.fail("HDA1 display called on HDA2"))
  send(flags=HyundaiFlags.CAMERA_SCC.value | HyundaiFlags.CANFD_HDA2.value)
  assert custom._options[2] is False


def test_option_refresh_only_resets_related_state(display):
  params, cs, send = display
  params.update(dict.fromkeys(KEYS, True))
  send(1)
  tracker = custom.state.radar_display_tracker
  tracker.approach_holds[37] = {"dRel": 2.0}
  params[KEYS[0]] = False
  send(99)
  assert custom._options[0] is True
  send(100)
  assert custom._options[0] is False
  assert custom.state.radar_display_tracker is tracker
  params[KEYS[1]] = False
  send(200)
  assert custom.state.radar_display_tracker is tracker
  assert 37 in tracker.approach_holds
  assert custom.state.l_lane_f.value == 1.5
  params[KEYS[2]] = False
  send(300)
  assert custom.state.radar_display_tracker is not tracker
  assert not custom.state.radar_display_tracker.approach_holds


def test_color_only_does_not_need_model(display):
  params, cs, send = display
  params[KEYS[0]] = True
  cs.modelV2 = None
  values = dict(send())["ADRV_0x161"]
  assert values["LANE_HIGHLIGHT_DISTANCE"] > 0
  assert values["ALERTS_5"] == 0
  assert values["LANELINE_LEFT_POSITION"] == 15


@pytest.mark.parametrize("animation", [False, True])
def test_trailer_block_precedes_custom_lane_display(display, monkeypatch, animation):
  params, cs, send = display
  params.update(dict.fromkeys(KEYS, True))
  params["CcncModelLanes"] = animation
  cs.trailer_connected = True
  cs.modelV2.meta.desire.raw = 3
  cs.adrv_0x161.update(LANE_LEFT=1, LANE_RIGHT=1)
  monkeypatch.setattr(custom, "update_lanes", lambda *args: pytest.fail("custom lanes override trailer block"))
  values = dict(send())["ADRV_0x161"]
  assert values["LCA_LEFT_ICON"] == values["LCA_RIGHT_ICON"] == 1
  assert values["LANE_LEFT"] == values["LANE_RIGHT"] == 0


@pytest.mark.parametrize("model_lanes", [False, True])
@pytest.mark.parametrize("other_options", [False, True])
def test_model_option_controls_sla_lfa_lka_and_inactive_lane_display(display, model_lanes, other_options):
  params, cs, send = display
  params.update(CcncLaneColor=other_options, CcncRadarVehicles=other_options, CcncModelLanes=model_lanes)
  cs.out.vCruiseCluster = 80
  cs.out.steeringPressed = True
  active = dict(send())["ADRV_0x161"]
  assert active["SETSPEED"] == active["SETSPEED_HUD"] == (2 if model_lanes else 3)
  assert active["SLA_ICON"] == (2 if model_lanes else 0)
  assert active["LFA_ICON"] == (3 if model_lanes else 2)
  assert active["LKA_ICON"] == (0 if model_lanes else 4)
  inactive = dict(send(5, lat_active=False))["ADRV_0x161"]
  assert inactive["LANELINE_LEFT"] == inactive["LANELINE_RIGHT"] == (0 if model_lanes else 6)
  assert inactive["LKA_ICON"] == (0 if model_lanes else 3)


def test_all_params_off_uses_original_display_path(display):
  params, cs, send = display
  cs.out.vCruiseCluster = 80
  cs.out.steeringPressed = True
  cs.adrv_0x161["ALERTS_5"] = 11
  for lat_active in (False, True):
    assert send(lat_active=lat_active) == send(lat_active=lat_active, custom_ccnc=False)


def test_model_toggle_resets_sla_timer(display):
  params, cs, send = display
  params["CcncModelLanes"] = True
  cs.out.vCruiseCluster = 80
  send()
  assert custom.state.sla_active_time > 0
  params["CcncModelLanes"] = False
  send(100)
  assert custom.state.sla_active_time == 0
