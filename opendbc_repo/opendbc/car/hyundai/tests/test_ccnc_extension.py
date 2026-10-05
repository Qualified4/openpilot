from itertools import product
import math
from types import SimpleNamespace as N

import pytest
import numpy as np

from opendbc.can import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car import structs
from opendbc.car.hyundai import hyundaicanfd as main, hyundaicanfd_ccnc_extension as ccnc_extension
from opendbc.car.hyundai.values import HyundaiFlags

KEYS = ("CcncLaneColor", "CcncModelLanes", "CcncRadarVehicles")
POSITION_KEY = "CcncVehiclePositionCorrection"


@pytest.fixture
def display(monkeypatch):
  params = dict.fromkeys(KEYS, False)
  params[POSITION_KEY] = False
  monkeypatch.setattr(main, "Params", lambda: N(get_bool=lambda key: bool(params[key]), get_int=lambda key: int(params.get(key, 0)), get=lambda key: "0"))
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


@pytest.mark.parametrize("options", list(product((False, True), (0, 1, 2), (False, True))))
def test_three_independent_options(display, monkeypatch, options):
  params, cs, send = display
  params.update(zip(KEYS, options))
  calls = []
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda values, *args: calls.append(args[-1]) or values)
  monkeypatch.setattr(main, "_apply_ccnc_lead", lambda *args: None)
  cs.ccnc_0x162 = dict.fromkeys(CANPacker("hyundai_canfd_generated").dbc.name_to_msg["CCNC_0x162"].sigs, 0)
  cs.ccnc_0x162.update(SPEEDLIMIT=0, FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
  color, geometry, radar = options
  geometry = bool(geometry)
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
  cs.ccnc_0x162 = dict.fromkeys(CANPacker("hyundai_canfd_generated").dbc.name_to_msg["CCNC_0x162"].sigs, 0)
  cs.ccnc_0x162.update(SPEEDLIMIT=0, FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
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


@pytest.mark.parametrize('radar,correction,hda2', list(product((False, True), repeat=3)))
def test_position_correction_requires_hda1_radar_display(display, radar, correction, hda2):
  params, _, send = display
  params[KEYS[2]] = radar
  params[POSITION_KEY] = correction
  send(1, flags=HyundaiFlags.CAMERA_SCC.value | (HyundaiFlags.CANFD_HDA2.value if hda2 else 0))
  enabled = ccnc_extension.state.radar_display_tracker.position_correction is not None
  assert enabled == (radar and correction and not hda2)


def test_live_position_option_preserves_selection_and_existing_filters(display):
  params, _, send = display
  params[KEYS[2]] = True
  send(1)
  tracker = ccnc_extension.state.radar_display_tracker
  tracker.selected = (42, None, None)
  tracker.positions[42] = 'existing filter history'
  tracker.approach_holds[37] = 'existing stop history'
  params[POSITION_KEY] = True
  send(99)
  assert tracker.position_correction is None
  send(100)
  correction = tracker.position_correction
  assert correction is not None and ccnc_extension.state.radar_display_tracker is tracker
  assert tracker.selected == (42, None, None)
  assert tracker.positions[42] == 'existing filter history'
  assert tracker.approach_holds[37] == 'existing stop history'
  send(200)
  assert tracker.position_correction is correction
  params[POSITION_KEY] = False
  send(300)
  assert tracker.position_correction is None and ccnc_extension.state.radar_display_tracker is tracker
  assert tracker.selected == (42, None, None)


@pytest.mark.parametrize('correction', [False, True])
def test_position_correction_changes_lateral_display_only(display, correction):
  params, cs, send = display
  params[KEYS[2]], params[POSITION_KEY] = True, correction
  cs.out.vEgo = 15.
  md = cs.modelV2
  md.laneLines = [N(x=[0., 100.], y=[y, y]) for y in (-4.5, -1.5, 1.5, 4.5)]
  md.roadEdges = [N(x=[0., 100.], y=[y, y]) for y in (-6., 6.)]
  cs.ccnc_0x162 = dict.fromkeys(CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162'].sigs, 0)
  points = [N(trackId=i, radarSource='frontRadar', dRel=20., yRel=y, vRel=0., vLead=15.)
            for i, y in enumerate((.5, 3.5, -3.5), 1)]
  for frame in range(0, 251, 5):
    cs.live_tracks = N(points=points)
    md.timestampEof = frame
    values = dict(send(frame))['CCNC_0x162']
  assert ccnc_extension.state.radar_display_tracker.selected == (1, 2, 3)
  assert [p.yRel for p in points] == [.5, 3.5, -3.5]
  assert values['FF_DISTANCE'] == values['LF_DETECT_DISTANCE'] == values['RF_DETECT_DISTANCE'] == 16.
  assert (values['FF_DETECT'], values['LF_DETECT'], values['RF_DETECT']) == (3, 3, 3)
  if correction:
    assert (values['FF_LATERAL'], values['LF_DETECT_LATERAL'], values['RF_DETECT_LATERAL']) == pytest.approx((0., 3., 3.))
  else:
    assert values['FF_LATERAL'] == pytest.approx(ccnc_extension.apply_curved_deadband(-.5, 0., .7, 1))
    assert values['LF_DETECT_LATERAL'] == values['RF_DETECT_LATERAL'] == pytest.approx(
      ccnc_extension.apply_curved_deadband(3.5, 3., .9, 2))


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


def test_model_mode_polling_switches_and_resets_filter_history(display):
  params, cs, send = display
  params['CcncModelLanes'] = 1
  send(0)
  basic = ccnc_extension.state.lane_geometry
  assert ccnc_extension.state.lane_curve_mode == 1
  params['CcncModelLanes'] = 2
  send(5)
  assert ccnc_extension.state.lane_geometry is basic
  send(100)
  precise = ccnc_extension.state.lane_geometry
  assert precise is not basic and ccnc_extension.state.lane_curve_mode == 2
  send(105)
  assert ccnc_extension.state.lane_geometry is precise
  params['CcncModelLanes'] = 0
  send(200)
  assert ccnc_extension.state.lane_curve_mode == 0
  assert ccnc_extension.state.lane_curv.value == 0


@pytest.mark.parametrize('mode,expected', [(1, [0, -3, -5]), (2, [0, 0, 0])])
def test_basic_uses_original_heading_sensitive_curve_precise_removes_heading(display, monkeypatch, mode, expected):
  _, cs, _ = display
  ccnc_extension.configure(False, True, False, model_lane_mode=mode)
  cs.modelV2 = lane_geometry_model(heading=.1)
  if mode == 1:
    monkeypatch.setattr(ccnc_extension._CcncLaneGeometry, 'road_curve', lambda *args: pytest.fail('Basic ran precise curve'))
  curves = []
  for frame in (0, 5, 10):
    values = {}
    ccnc_extension.update_lanes(values, cs, cs.modelV2, 60., 0., 0, True, False, True, frame)
    curves.append(-(values['LANELINE_CURVATURE'] + 1) if values['LANELINE_CURVATURE_DIRECTION'] else values['LANELINE_CURVATURE'])
  assert curves == expected



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
  cs.ccnc_0x162 = dict.fromkeys(CANPacker("hyundai_canfd_generated").dbc.name_to_msg["CCNC_0x162"].sigs, 0)
  cs.ccnc_0x162.update(SPEEDLIMIT=0, FF_DISTANCE=204.6, FF_LATERAL=0., FF_DETECT=0, LF_DETECT=0, RF_DETECT=0, LR_DETECT=0, RR_DETECT=0)
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


@pytest.mark.parametrize("radar,hda2", list(product((False, True), repeat=2)))
def test_stock_corner_visibility_does_not_override_extension_hiding(display, monkeypatch, radar, hda2):
  params, cs, send = display
  params["CcncRadarVehicles"] = radar
  cs.adrv_0x161 = None
  cs.ccnc_0x162 = dict.fromkeys(CANPacker("hyundai_canfd_generated").dbc.name_to_msg["CCNC_0x162"].sigs, 0)
  for side in ("LF", "RF", "LR", "RR"):
    cs.ccnc_0x162[f"{side}_DETECT_DISTANCE"] = 20.
  monkeypatch.setattr(ccnc_extension, "update_vehicles", lambda values, *args: values)
  flags = HyundaiFlags.CAMERA_SCC.value | (HyundaiFlags.CANFD_HDA2.value if hda2 else 0)
  values = dict(send(flags=flags))["CCNC_0x162"]
  for side in ("LF", "RF", "LR", "RR"):
    assert values[f"{side}_DETECT"] == (0 if radar and not hda2 else 3)
    assert values[f"{side}_DETECT_DISTANCE"] == 20.
    assert cs.ccnc_0x162[f"{side}_DETECT"] == 0


def lane_geometry_model(stamp=1, offset=0., heading=0., bend=0.):
  xs = [float(x) for x in range(0, 101)]
  def line(y):
    return N(x=xs, y=[y + offset + heading*x + .5*bend*x*x for x in xs])
  return N(timestampEof=stamp, laneLines=[line(y) for y in (-5.4, -1.8, 1.8, 5.4)],
           laneLineProbs=[.9]*4, roadEdges=[line(-8.), line(8.)], roadEdgeStds=[.1, .1],
           position=N(x=xs, y=[offset + heading*x + .5*bend*x*x for x in xs], yStd=[0.]*len(xs)),
           meta=N(laneChangeAvailableLeft=True, laneChangeAvailableRight=True))


@pytest.mark.parametrize('heading', [-.15, 0., .15])
@pytest.mark.parametrize('offset', [-3.6, 0., 3.6])
def test_lane_curvature_rejects_translation_and_heading(heading, offset):
  md = lane_geometry_model(offset=offset, heading=heading)
  assert ccnc_extension._CcncLaneGeometry.road_curve(md, 90.) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('bend', [-.004, .004])
@pytest.mark.parametrize('changing', [False, True])
def test_position_curvature_keeps_bend_without_lane_confidence(bend, changing):
  md = lane_geometry_model(bend=bend)
  md.laneLineProbs = [0.]*4
  md.roadEdgeStds = [2., 2.]
  # Distant lane polynomials can be straight while the path still bends.
  md.laneLines = lane_geometry_model().laneLines
  value = ccnc_extension._CcncLaneGeometry.road_curve(md, 90., changing)
  length = 30 + (90-20)/80*50
  length = min(length + (50 if changing else 0), 100)
  midpoint = length * (.75 if changing else .5)
  expected = -1800*bend / (1+(bend*midpoint)**2)**1.5
  assert value == pytest.approx(expected, rel=.005)  # Sparse samples require linear interpolation.


@pytest.mark.parametrize('bad', ['nan', 'short', 'reverse', 'std_length', 'confidence'])
def test_lane_curvature_rejects_bad_geometry(bad):
  md = lane_geometry_model()
  if bad == 'nan': md.position.y[3] = float('nan')
  if bad == 'short': md.position.y.pop()
  if bad == 'reverse': md.position.x = md.position.x[::-1]
  if bad == 'confidence': md.position.yStd = [1.]*len(md.position.x)
  if bad == 'std_length': md.position.yStd.pop()
  assert ccnc_extension._CcncLaneGeometry.road_curve(md, 90.) is None


@pytest.mark.parametrize('changing', [False, True])
@pytest.mark.parametrize('tail', ['fold', 'nan'])
def test_curvature_uses_valid_trusted_prefix_before_rejecting_distant_tail(changing, tail):
  md = lane_geometry_model(bend=.004)
  md.position.yStd[61:] = [1.]*40
  prefix = lane_geometry_model(bend=.004)
  prefix.position = N(x=md.position.x[:61], y=md.position.y[:61], yStd=[0.]*61)
  if tail == 'fold':
    md.position.x[61:] = md.position.x[61:][::-1]
  else:
    md.position.y[61] = float('nan')
  curve = ccnc_extension._CcncLaneGeometry.road_curve
  assert curve(md, 90., changing) == pytest.approx(curve(prefix, 90., changing))
  assert abs(curve(md, 90., changing)) > 1.
  md.position.x[30] = md.position.x[29]
  assert curve(md, 90., changing) is None


def test_stale_model_does_not_renew_curvature_and_recovers():
  g = ccnc_extension._CcncLaneGeometry()
  md = lane_geometry_model(bend=.004)
  for i in range(20): g.update(lane_geometry_model(stamp=i, bend=.004), 90, i*.05)
  before = g.curvature
  g.update(md, 90, 1.)
  for i in range(21, 61): g.update(md, 90, i*.05)
  assert round(g.curvature) == 0 and g.target is None and abs(before) > 5.
  g.update(lane_geometry_model(stamp=100, bend=.004), 90, 3.05)
  assert g.fresh and g.curvature < 0.


@pytest.mark.parametrize('reader', [False, True])
def test_actual_cereal_lane_change_computes_curve_and_motion(display, reader):
  from openpilot.cereal import log
  _, cs, _ = display
  model = lane_geometry_model(heading=.03, bend=.001)
  # Exercise the nonquadratic maneuver branch, not its early bypass.
  model.position.y = [y + .4*math.sin(x/20) for x, y in zip(model.position.x, model.position.y)]
  builder = log.ModelDataV2.new_message(
    timestampEof=1, position=vars(model.position), laneLines=[vars(line) for line in model.laneLines],
    laneLineProbs=model.laneLineProbs,
    meta=dict(laneChangeAvailableLeft=True, laneChangeAvailableRight=True))
  if reader:
    # Use a serialized reader as well as a live builder.
    with log.ModelDataV2.from_bytes(builder.to_bytes()) as md:
      _check_actual_lane_change(cs, md)
  else:
    _check_actual_lane_change(cs, builder)


def _check_actual_lane_change(cs, md):
  g = ccnc_extension.state.lane_geometry
  assert g.road_curve(md, 85., True) is not None
  g.fresh = True
  g.observe_motion(md, True, True, False, 0., 85.)
  assert len(g.motion) == 1
  cs.out.leftBlinker = True
  values = {}
  ccnc_extension.update_lanes(values, cs, md, 85., 0., 3, True, False, True, 5)
  assert ccnc_extension.state.lane_geometry.target is not None
  assert values.get('LFA_ICON') != 5
  assert (values['LANELINE_LEFT_POSITION'], values['LANELINE_RIGHT_POSITION']) != (30, 30)
  assert values.get('LANE_HIGHLIGHT') != 3


def test_actual_reader_shares_lane_arrays_and_replaces_them_on_new_model(display):
  from openpilot.cereal import log
  _, cs, _ = display
  cs.out.leftBlinker = True
  previous = None
  for stamp, heading in ((1, .03), (2, .06)):
    model = lane_geometry_model(stamp=stamp, heading=heading, bend=.001)
    model.position.y = [y + .4 * math.sin(x / 20) for x, y in zip(model.position.x, model.position.y)]
    builder = log.ModelDataV2.new_message(
      timestampEof=stamp, position=vars(model.position), laneLines=[vars(line) for line in model.laneLines],
      laneLineProbs=model.laneLineProbs, meta=dict(laneChangeAvailableLeft=True, laneChangeAvailableRight=True))
    with log.ModelDataV2.from_bytes(builder.to_bytes()) as md:
      ccnc_extension.update_lanes({}, cs, md, 85., 0., 3, True, False, True, stamp * 5)
      tracker = ccnc_extension.state.radar_display_tracker
      assert set(tracker._curves) == {1, 2}
      inner = tracker._curves[1]
      assert inner is not previous
      assert inner[1][10] == pytest.approx(model.laneLines[1].y[10])
      tracker.lane_probabilities(md)
      assert tracker._curve(md, 1) is inner
      previous = inner


def test_failed_fresh_curve_does_not_keep_a_stale_target(monkeypatch):
  g = ccnc_extension._CcncLaneGeometry()
  for i in range(20):
    g.update(lane_geometry_model(stamp=i+1, bend=.004), 90., i*.05)
  before, valid_time = g.curvature, g.valid_time
  def fail(*args):
    raise TypeError('malformed model')
  monkeypatch.setattr(g, 'road_curve', fail)
  failed = lane_geometry_model(stamp=21)
  g.update(failed, 90., 1.)
  assert g.fresh and g.target is None and g.valid_time == valid_time
  assert g.curvature == before
  g.update(failed, 90., 1.05)
  assert not g.fresh and g.target is None
  for i in range(22, 51):
    g.update(failed, 90., i*.05)
  assert round(g.curvature) == 0
  monkeypatch.setattr(g, 'road_curve', ccnc_extension._CcncLaneGeometry.road_curve)
  g.update(lane_geometry_model(stamp=22, bend=.004), 90., 2.55)
  assert g.fresh and g.target is not None and g.curvature < 0


@pytest.mark.parametrize('duration', [0., .2, .4, 1.])
@pytest.mark.parametrize('start', [-.3, 0., .3])
@pytest.mark.parametrize('left', [False, True])
def test_straight_lane_change_sweep_has_no_false_curve(display, duration, start, left):
  _, cs, _ = display
  sign = 1 if left else -1
  cs.out.leftBlinker, cs.out.rightBlinker = left, not left
  for frame in range(0, 801, 5):
    t = frame*.01
    offset = sign * 3.6 * min(max(t/6, 0.), 1.)
    sweep_start = 3. + start
    fraction = float(t >= sweep_start) if duration == 0 else min(max((t-sweep_start)/duration, 0.), 1.)
    md = lane_geometry_model(stamp=frame+1, offset=offset-sign*3.6*fraction, heading=sign*.05)
    values = {}
    ccnc_extension.update_lanes(values, cs, md, 90., 0., 3 if left else 4, True, False, True, frame)
    assert values['LANELINE_CURVATURE'] == 0
    assert 0 <= ccnc_extension.state.lane_geometry.progress <= .45
    assert all(0 <= values['LANELINE_'+side+'_POSITION'] <= 30 for side in ('LEFT','RIGHT'))


@pytest.mark.parametrize('bad', ['none', 'reverse', 'single_jump', 'disagree', 'stale'])
def test_hold_prediction_requires_bounded_coherent_prior_motion(bad):
  g = ccnc_extension._CcncLaneGeometry()
  samples = [(i*.05, .8, .8) for i in range(8)]
  if bad == 'reverse': samples = [(t,-x,-y) for t,x,y in samples]
  if bad == 'single_jump': samples = [(t, .8+(i==7), .8+(i==7)) for i,(t,x,y) in enumerate(samples)]
  if bad == 'disagree': samples = [(t,x,1.6) for t,x,y in samples]
  g.motion.extend(samples)
  g.start_hold(1. if bad == 'stale' else .4)
  if bad == 'none':
    assert g.hold_speed == pytest.approx(.8)
    assert g.hold_progress(.5) > 0
    assert g.hold_progress(1.) == pytest.approx(.24)
    assert g.hold_progress(100.) == pytest.approx(.24)
  else: assert g.hold_speed == 0.


def test_prediction_resets_on_cancel_and_direction_change():
  g = ccnc_extension._CcncLaneGeometry()
  g.direction = True
  g.hold_speed, g.progress, g.hold_start = 1., .1, 0.
  g.observe_motion(None, False, False, False, .1)
  assert g.progress == 0 and g.hold_speed == 0 and g.hold_start is None


@pytest.mark.parametrize('left', [False, True])
def test_parallel_lane_index_sweep_does_not_change_prior_velocity(left):
  g = ccnc_extension._CcncLaneGeometry()
  sign = 1 if left else -1
  for i in range(8):
    md = lane_geometry_model(stamp=i+1, offset=3.6*i/7, heading=sign*.05)
    g.fresh = True
    g.observe_motion(md, left, True, False, i*.05, 60.)
  g.start_hold(.4)
  assert g.hold_speed == pytest.approx(60/3.6*.05)
  assert g.hold_progress(.8) > .2


@pytest.mark.parametrize('speed,duration,limit', [(90., 6., 1.01), (60., 4., 1.01), (50., 3., .2)])
def test_curved_lane_change_path_on_straight_road_has_bounded_residual(speed, duration, limit):
  # A lateral maneuver is curved even after translation/heading removal.
  # Keep its measured residual explicit rather than testing only affine paths.
  v = speed/3.6
  for elapsed in np.linspace(0., duration, 61):
    xs = np.linspace(0., 250., 501)
    u = np.clip((elapsed + xs/v)/duration, 0., 1.)
    y = 3.6*(10*u**3 - 15*u**4 + 6*u**5)
    md = lane_geometry_model()
    md.position = N(x=xs, y=y, yStd=[0.]*len(xs))
    assert abs(ccnc_extension._CcncLaneGeometry.road_curve(md, speed, True)) <= limit


def test_near_straight_lanes_do_not_hide_a_distant_bend_without_maneuver_disagreement():
  md = lane_geometry_model()
  xs = np.linspace(0., 250., 501)
  md.position = N(x=xs, y=.002*np.maximum(xs-40, 0)**2, yStd=[0.]*len(xs))
  md.laneLineProbs = [0.]*4
  with_lanes = ccnc_extension._CcncLaneGeometry.road_curve(md, 90., True)
  md.laneLines = []
  path_only = ccnc_extension._CcncLaneGeometry.road_curve(md, 90., True)
  assert with_lanes == pytest.approx(path_only)
  assert with_lanes < -5.


@pytest.mark.parametrize('bend', [-.002, .002])
def test_maneuver_guard_keeps_real_near_road_bend_at_zero_lane_confidence(bend):
  md = lane_geometry_model(bend=bend)
  xs = np.linspace(0., 250., 501)
  md.laneLineProbs = [0.]*4
  for elapsed in np.linspace(0., 4., 21):
    u = np.clip((elapsed + xs/(60/3.6))/4, 0., 1.)
    md.position = N(x=xs, y=.5*bend*xs**2+3.6*(10*u**3-15*u**4+6*u**5), yStd=[0.]*len(xs))
    value = ccnc_extension._CcncLaneGeometry.road_curve(md, 60., True)
    expected = -1800*bend / (1+(bend*15)**2)**1.5
    assert abs(value-expected) < 1.1


@pytest.mark.parametrize('bend', [-.002, .002])
def test_disagreeing_maneuver_uses_same_interval_road_curve(bend):
  md = lane_geometry_model(bend=bend)
  xs = np.linspace(0., 250., 501)
  u = np.clip(xs/(60/3.6)/4, 0., 1.)
  md.position = N(x=xs, y=.5*bend*xs**2+3.6*(10*u**3-15*u**4+6*u**5), yStd=[0.]*len(xs))
  md.laneLineProbs = [0.]*4
  g = ccnc_extension._CcncLaneGeometry
  local = sum(g.fit_curve(np.asarray(line.x), np.asarray(line.y), 0, 30)[0]
              for line in md.laneLines[1:3])*.5
  near_position = g.fit_curve(md.position.x, md.position.y, 0, 30)[0]
  assert abs(near_position-local) > .5
  assert g.road_curve(md, 60., True) == pytest.approx(local)


def test_unqualified_motion_keeps_original_two_release_steps(display):
  _, cs, _ = display
  cs.out.leftBlinker = True
  ccnc_extension.update_lanes({}, cs, lane_geometry_model(offset=1.75), 90, 0, 3, True, False, True, 0)
  assert ccnc_extension.state.lane_geometry.hold_speed == 0
  # No usable pre-trigger motion: release evidence must still move 0.1/0.2 m.
  for frame, offset, expected in [(5, -1.7, 0), (10, -1.5, 1), (15, -1.3, 2)]:
    values = {}
    ccnc_extension.update_lanes(values, cs, lane_geometry_model(stamp=frame+1, offset=offset),
                                90, 0, 3, True, False, True, frame)
    assert values['LANELINE_RIGHT_POSITION'] == expected


def test_lane_hold_releases_with_fresh_model_while_lateral_control_is_off(display):
  _, cs, _ = display
  cs.out.leftBlinker = True
  ccnc_extension.update_lanes({}, cs, lane_geometry_model(offset=1.75), 90, 0, 3, False, False, True, 0)
  assert ccnc_extension.state.hold_lane
  # After relabelling, the new right boundary moves away in two distinct models.
  for frame, offset in [(5, -1.7), (10, -1.5), (15, -1.3)]:
    values = {}
    ccnc_extension.update_lanes(values, cs, lane_geometry_model(stamp=frame+1, offset=offset),
                                90, 0, 3, False, False, True, frame)
    assert values['LANELINE_CURVATURE'] == round(cs.out.steeringAngleDeg/3)
  assert not ccnc_extension.state.hold_lane


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('lat_enabled', [False, True])
def test_lane_recognition_rebound_before_crossing_does_not_enter_hold(display, auto, lat_enabled):
  _, cs, _ = display
  cs.out.leftBlinker = True
  # A boundary remains at least 0.9 m away despite a large recognition rebound.
  for i, distance in enumerate([1.4, 1.1, .9, 1.3, 1.5]):
    md = lane_geometry_model(stamp=i+1, offset=1.8-distance)
    ccnc_extension.update_lanes({}, cs, md, 90, 0, 3 if auto else 0,
                                lat_enabled, False, True, i*5)
    assert not ccnc_extension.state.draw_center
    assert not ccnc_extension.state.hold_lane
  # A subsequent actual boundary approach still arms the original phase transition.
  for i, distance in enumerate([.4, .2, .04], 5):
    md = lane_geometry_model(stamp=i+1, offset=1.8-distance)
    ccnc_extension.update_lanes({}, cs, md, 90, 0, 3 if auto else 0,
                                lat_enabled, False, True, i*5)
  assert ccnc_extension.state.hold_lane


@pytest.mark.parametrize('new_reader', [False, True])
def test_duplicate_model_cannot_release_hold(display, new_reader):
  _, cs, _ = display
  cs.out.leftBlinker = True
  md = lane_geometry_model(offset=1.75)
  ccnc_extension.update_lanes({}, cs, md, 90, 0, 3, True, False, True, 0)
  assert ccnc_extension.state.hold_lane
  md2 = lane_geometry_model(stamp=2, offset=-1.6)
  for frame in (5,10,15):
    if new_reader:
      md2 = lane_geometry_model(stamp=2, offset=-1.6)
    ccnc_extension.update_lanes({}, cs, md2, 90, 0, 3, True, False, True, frame)
  assert ccnc_extension.state.hold_lane
  assert ccnc_extension.state.hold_lane_escape_count < 2


def test_lane_filter_gain_ramps_and_preserves_output_on_cancel(display):
  _, cs, _ = display
  cs.out.leftBlinker = True
  alphas=[]
  for frame in range(0, 26, 5):
    ccnc_extension.update_lanes({}, cs, lane_geometry_model(stamp=frame+1), 90, 0, 3, True, False, True, frame)
    alphas.append(ccnc_extension.state.lane_geometry.lane_alpha)
  assert alphas[-1] == pytest.approx(.6)
  assert all(0 <= b-a <= .100001 for a,b in zip(alphas,alphas[1:]))
  before=ccnc_extension.state.l_lane_f.value
  cs.out.leftBlinker=False
  ccnc_extension.update_lanes({}, cs, lane_geometry_model(stamp=31), 90, 0, 0, True, False, True, 30)
  assert ccnc_extension.state.lane_geometry.lane_alpha == pytest.approx(.5)
  assert abs(ccnc_extension.state.l_lane_f.value-before) < .1


def test_control_gap_clears_pending_lane_transition(display):
  _, cs, _ = display
  cs.out.leftBlinker=True
  ccnc_extension.update_lanes({},cs,lane_geometry_model(offset=1.75),90,0,3,True,False,True,0)
  assert ccnc_extension.state.hold_lane
  ccnc_extension.update_lanes({},cs,lane_geometry_model(stamp=2),90,0,3,True,False,True,100)
  assert not ccnc_extension.state.hold_lane
  assert ccnc_extension.state.lane_geometry.hold_start is None


def test_new_array_point_cannot_create_stop_or_approach_memory(display, monkeypatch):
  ccnc_extension.reset();ccnc_extension.configure(True,True,True,True,model_lane_mode=2)
  curve=lambda y:N(x=[0.,10.,30.],y=[y]*3)
  md=N(timestampEof=0,laneLines=[curve(y) for y in (-5.,-1.5,1.5,5.)],laneLineProbs=[.9]*4,
       roadEdges=[curve(-7.),curve(7.)],orientationRate=N(z=[0.]),leadsV3=[],meta=N(laneChangeState='off'))
  fields=dict.fromkeys(CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162'].sigs,0)
  cs=N(out=N(leftBlinker=False,rightBlinker=False,leftBlindspot=False,rightBlindspot=False),
       radarState=None,ccnc_0x162=fields)
  for frame in range(0,251,5):
    monkeypatch.setattr(ccnc_extension,'time',N(monotonic=lambda:frame*.01))
    cs.live_tracks=N(points=[N(trackId=i,dRel=x,yRel=2.1,vLead=0.,vRel=0.,yvRel=0.,
                              measured=True,radarSource='frontRadar') for i,x in enumerate((3.,5.5,8.))])
    values=ccnc_extension.update_vehicles(dict(fields),cs,md,frame,0.,0.,True)
    assert values['FF_DETECT']!=7
    if frame>=50:
      assert values['LF_DETECT']==0
      tr=ccnc_extension.state.radar_display_tracker
      assert not tr.stop_pending and not tr.stop_holds
      assert not tr.approach_pending and not tr.approach_holds
  # A different/reused source must not blank an older valid physical memory.
  tr.stop_holds[0]=(99,2.9,3.8,(2,2.32,3.8),-1)
  monkeypatch.setattr(ccnc_extension,'time',N(monotonic=lambda:2.55))
  cs.live_tracks=N(points=[N(trackId=i,dRel=x,yRel=2.1,vLead=0.,vRel=0.,yvRel=0.,
                            measured=True,radarSource='frontRadar') for i,x in enumerate((3.,5.5,8.))])
  values=ccnc_extension.update_vehicles(dict(fields),cs,md,255,0.,0.,True)
  assert values['LF_DETECT']==2
  assert values['LF_DETECT_DISTANCE']==2.32
  assert tr.stop_holds[0][0]==99
