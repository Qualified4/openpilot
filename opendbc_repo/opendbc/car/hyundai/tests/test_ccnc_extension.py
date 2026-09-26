from itertools import product
from types import SimpleNamespace as N

import pytest

from opendbc.can import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car import structs
from opendbc.car.hyundai import hyundaicanfd as main, hyundaicanfd_ccnc_extension as ccnc_extension
from opendbc.car.hyundai.values import HyundaiFlags

KEYS = ("CcncLaneColor", "CcncModelLanes", "CcncRadarVehicles")


@pytest.fixture
def display(monkeypatch):
  params = dict.fromkeys(KEYS, False)
  monkeypatch.setattr(main, "Params", lambda: N(get_bool=lambda key: params[key], get_int=lambda key: 0, get=lambda key: "0"))
  monkeypatch.setattr(ccnc_extension, "Params", lambda: N(get_int=lambda key: 3))
  monkeypatch.delattr(main.create_ccnc_messages, "_display_options", raising=False)
  monkeypatch.setattr(ccnc_extension, "state", N())
  monkeypatch.setattr(ccnc_extension, "_options", None)
  ccnc_extension.reset()
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

  def send(frame=0, flags=HyundaiFlags.CAMERA_SCC.value, lat_active=True, extended_ccnc=None):
    return main.create_ccnc_messages(N(flags=flags), packer, N(ECAN=0, CAM=2), frame,
                                    N(enabled=True, latActive=lat_active), cs, hud, 0, False, False, 0, False, 0, 0,
                                    extended_ccnc=extended_ccnc)
  return params, cs, send


@pytest.mark.parametrize("options", list(product((False, True), repeat=3)))
def test_three_independent_options(display, monkeypatch, options):
  params, cs, send = display
  params.update(zip(KEYS, options))
  calls = []
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda values, *args: calls.append(args[-1]) or values)
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
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda *args: pytest.fail("HDA1 display called on HDA2"))
  for frame in (0, 5, 10, 15):
    values = dict(send(frame, flags=HyundaiFlags.CAMERA_SCC.value | HyundaiFlags.CANFD_HDA2.value))["ADRV_0x161"]
  assert ccnc_extension._options[2] is False
  # The extension name must not disable the existing HDA2 lane/color effects.
  assert values["LANE_HIGHLIGHT_DISTANCE"] > 0
  assert values["LANELINE_LEFT_POSITION"] != 15
  assert values["LANELINE_CURVATURE"] != 2


def test_option_refresh_only_resets_related_state(display):
  params, cs, send = display
  params.update(dict.fromkeys(KEYS, True))
  send(1)
  tracker = ccnc_extension.state.radar_display_tracker
  tracker.approach_holds[37] = {"dRel": 2.0}
  params[KEYS[0]] = False
  send(99)
  assert ccnc_extension._options[0] is True
  send(100)
  assert ccnc_extension._options[0] is False
  assert ccnc_extension.state.radar_display_tracker is tracker
  params[KEYS[1]] = False
  send(200)
  assert ccnc_extension.state.radar_display_tracker is tracker
  assert 37 in tracker.approach_holds
  assert ccnc_extension.state.l_lane_f.value == 1.5
  params[KEYS[2]] = False
  send(300)
  assert ccnc_extension.state.radar_display_tracker is not tracker
  assert not ccnc_extension.state.radar_display_tracker.approach_holds


def test_color_only_does_not_need_model(display):
  params, cs, send = display
  params[KEYS[0]] = True
  cs.modelV2 = None
  values = dict(send())["ADRV_0x161"]
  assert values["LANE_HIGHLIGHT_DISTANCE"] > 0
  assert values["ALERTS_5"] == 0
  assert values["LANELINE_LEFT_POSITION"] == 15


@pytest.mark.parametrize("animation", [False, True])
def test_trailer_block_precedes_extended_lane_display(display, monkeypatch, animation):
  params, cs, send = display
  params.update(dict.fromkeys(KEYS, True))
  params["CcncModelLanes"] = animation
  cs.trailer_connected = True
  cs.modelV2.meta.desire.raw = 3
  cs.adrv_0x161.update(LANE_LEFT=1, LANE_RIGHT=1)
  monkeypatch.setattr(ccnc_extension, "update_lanes", lambda *args: pytest.fail("ccnc_extension lanes override trailer block"))
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
    assert send(lat_active=lat_active) == send(lat_active=lat_active, extended_ccnc=False)


def test_model_toggle_resets_sla_timer(display):
  params, cs, send = display
  params["CcncModelLanes"] = True
  cs.out.vCruiseCluster = 80
  send()
  assert ccnc_extension.state.sla_active_time > 0
  params["CcncModelLanes"] = False
  send(100)
  assert ccnc_extension.state.sla_active_time == 0



def test_driving_mode_cache_refresh_and_reenable(display, monkeypatch):
  params, cs, send = display
  params["CcncLaneColor"] = True
  clock = [10.0]
  mode = [3]
  reads = []

  def read_mode(key):
    reads.append(key)
    return mode[0]

  monkeypatch.setattr(ccnc_extension, "time", N(monotonic=lambda: clock[0]))
  monkeypatch.setattr(ccnc_extension, "Params", lambda: N(get_int=read_mode))
  send()
  mode[0] = 1
  clock[0] = 10.99
  send(5)
  assert ccnc_extension.state.drive_mode == 3 and len(reads) == 1
  clock[0] = 11.0
  send(10)
  assert ccnc_extension.state.drive_mode == 1 and len(reads) == 2
  params["CcncLaneColor"] = False
  send(100)
  params["CcncLaneColor"] = True
  mode[0] = 4
  send(200)
  assert ccnc_extension.state.drive_mode == 4 and len(reads) == 3


def test_scalar_lane_math_matches_numpy_at_rounding_boundaries():
  import ast
  import inspect
  import math
  import numpy as np

  tree = ast.parse(inspect.getsource(ccnc_extension.update_lanes))
  assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)}
  position_expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and ast.unparse(n.targets[0]) == "values['LANELINE_LEFT_POSITION']"
                       and "current_l_target" in ast.unparse(n.value))
  position = compile(ast.Expression(position_expr), "lane_position", "eval")
  step = compile(ast.Expression(assignments["bounded_l"]), "lane_step", "eval")
  boundaries = [0., 3., -0.15, 0.15] + [(i + .5) / 10 for i in range(30)]
  samples = [-math.inf, math.inf, math.nan, -10., 10.] + [x for b in boundaries for x in (np.nextafter(b, -math.inf), b, np.nextafter(b, math.inf))]
  for x in samples:
    expected = np.interp(x, [0., 3.], [0., 30.])
    if math.isnan(expected):
      with pytest.raises(ValueError):
        eval(position, {"current_l_target": x})
    else:
      assert eval(position, {"current_l_target": x}) == int(round(expected))
    for previous in (0., 1.5, 3.):
      expected = previous + np.clip(x - previous, -.15, .15)
      actual = eval(step, {"leftlaneraw": x, "prev_l": previous, "MAX_STEP": .15})
      assert math.isnan(actual) if math.isnan(expected) else actual == expected


@pytest.mark.parametrize("radar", [False, True])
def test_extension_radar_is_final_vehicle_display_authority(display, monkeypatch, radar):
  params, cs, send = display
  params["CcncRadarVehicles"] = radar
  cs.ccnc_0x162 = dict(SPEEDLIMIT=0, FF_DISTANCE=204.6, FF_LATERAL=0., FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
  cs.radarState = N(leadOne=N(status=True, dRel=6., yRel=0., vRel=0., radar=False), leadTwo=None)
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda values, *args: values)
  values = dict(send())["CCNC_0x162"]
  assert values["FF_DETECT"] == (0 if radar else 4)


@pytest.mark.parametrize("detect", range(15))
def test_extension_vehicle_icons_pack_without_changing_geometry(display, monkeypatch, detect):
  params, cs, _ = display
  params["CcncRadarVehicles"] = True
  packer = CANPacker("hyundai_canfd_generated")
  definition = packer.dbc.name_to_msg["CCNC_0x162"]
  source = dict.fromkeys(definition.sigs, 0)
  source.update(FF_DETECT=detect, FF_DISTANCE=81.2, FF_LATERAL=1.3,
                FF_DETECT_ALT=2, FF_DISTANCE_ALT=42.1, FF_LATERAL_ALT=0.7)
  for side in ("LF", "RF", "LR", "RR"):
    source.update({f"{side}_DETECT": detect, f"{side}_DETECT_DISTANCE": 20., f"{side}_DETECT_LATERAL": 2.9})
  cs.ccnc_0x162 = source.copy()
  cs.adrv_0x161 = None
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda values, *args: values)

  messages = main.create_ccnc_messages(N(flags=HyundaiFlags.CAMERA_SCC.value), packer, N(ECAN=0, CAM=2), 0,
                                     N(enabled=False, latActive=False), cs, N(), 0, False, False, 0, False, 0, 0)
  assert len(messages) == 1
  address, data, bus = messages[0]
  assert (address, bus) == (definition.address, 0)
  decoded = {key: get_raw_value(data, sig) * sig.factor + sig.offset for key, sig in definition.sigs.items()}
  expected = source.copy()
  for key in ("FF_DETECT", "LF_DETECT", "RF_DETECT", "LR_DETECT", "RR_DETECT"):
    expected[key] = detect + 2 if detect in (1, 2) else detect
  for key in expected.keys() - {"CHECKSUM", "COUNTER"}:
    assert decoded[key] == pytest.approx(expected[key])
  assert decoded["CHECKSUM"] == main.hkg_can_fd_checksum(address, None, bytearray(data))
  assert cs.ccnc_0x162 == source
