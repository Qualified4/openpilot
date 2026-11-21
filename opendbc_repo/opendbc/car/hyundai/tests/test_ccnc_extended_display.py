from itertools import product

import pytest

from opendbc.can import CANPacker
from opendbc.car.hyundai import hyundaicanfd as main
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401
from opendbc.car.hyundai.values import HyundaiFlags

BASE_FLAGS = HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC
CASES = [
  *[pytest.param(on, cluster, traffic, BASE_FLAGS, True, True, on, id=f"toggle-{int(on)}{int(cluster)}{int(traffic)}")
    for on, cluster, traffic in product((False, True), repeat=3)],
  *[pytest.param(on, True, True, BASE_FLAGS | HyundaiFlags.CANFD_HDA2, True, True, False, id=f"hda2-{int(on)}")
    for on in (False, True)],
  pytest.param(True, False, False, HyundaiFlags.CAMERA_SCC, True, True, False, id="not-canfd"),
  pytest.param(True, False, False, HyundaiFlags.CANFD, True, True, False, id="not-camera-scc"),
  pytest.param(True, False, False, BASE_FLAGS, False, True, False, id="no-openpilot-long"),
  pytest.param(True, False, False, BASE_FLAGS, True, False, False, id="cluster-unconfirmed"),
]


def templates(cs):
  dbc = CANPacker("hyundai_canfd_generated").dbc
  cs.adrv_0x161 = dict.fromkeys(dbc.name_to_msg["ADRV_0x161"].sigs, 0)
  cs.ccnc_0x162 = dict.fromkeys(dbc.name_to_msg["CCNC_0x162"].sigs, 0)
  cs.live_tracks = None
  cs.out.canValid = True
  for line in cs.modelV2.laneLines:
    line.x, line.y = [0., 30.], line.y * 2


@pytest.mark.parametrize("on,cluster,traffic,flags,longitudinal,confirmed,native", CASES)
def test_display_matrix(display, monkeypatch, on, cluster, traffic, flags, longitudinal, confirmed, native):  # noqa: F811
  params, cs, send = display
  params.update(CcncExtendedDisplay=on, CcncClusterDisplay=cluster, CcncTrafficBackground=traffic)
  templates(cs)
  core = main._CCNC_DISPLAY_RUST
  assert core is not None
  calls = []
  entered = core.update_display

  def traced(cs, model, frame, inputs, *args):
    calls.append((frame, inputs is not None))
    return entered(cs, model, frame, inputs, *args)

  monkeypatch.setattr(core, "update_display", traced)
  for frame in range(10):
    send(frame, flags=flags, longitudinal=longitudinal, confirmed=confirmed)
  assert calls == ([(n, n % 5 == 0) for n in range(10)] if native else [])


@pytest.mark.parametrize("xs,ys,stds", [
  ([], [], []),
  ([0., 30.], [], [0., 0.]),
  ([0., 30.], [0., float("nan")], [0., 0.]),
  ([0., 30.], [0., 1.], [0., float("nan")]),
])
def test_invalid_precise_curve_matches_python(display, monkeypatch, xs, ys, stds):  # noqa: F811
  params, cs, send = display
  params["CcncExtendedDisplay"] = True
  templates(cs)
  cs.modelV2.position.x, cs.modelV2.position.y, cs.modelV2.position.yStd = xs, ys, stds
  cs.out.steeringAngleDeg = -12.
  with monkeypatch.context() as patch:
    patch.setattr(main, "_CCNC_DISPLAY_RUST", None)
    expected = dict(send())["ADRV_0x161"]
  main.ccnc_extension.reset()
  actual = dict(send())["ADRV_0x161"]
  keys = ("LANELINE_CURVATURE", "LANELINE_CURVATURE_DIRECTION")
  assert tuple(actual[key] for key in keys) == tuple(expected[key] for key in keys) == (3, 1)


def test_failed_rust_call_falls_back_and_filters_lead_once(display, monkeypatch):  # noqa: F811
  params, cs, send = display
  params["CcncExtendedDisplay"] = True
  templates(cs)
  core = main._CCNC_DISPLAY_RUST
  calls = []
  logged = []

  def fail(*args):
    calls.append(args[2])
    return core._panic_for_test()

  monkeypatch.setattr(core, "update_display", fail)
  monkeypatch.setattr(main.ccnc_radar_map._live, "failed", False)
  monkeypatch.setattr(main.ccnc_radar_map.carlog, "exception", logged.append)
  lead = main.DisplayLeadLateralFilter()
  update = lead.update
  updates = []

  def count(*args):
    updates.append(True)
    return update(*args)

  monkeypatch.setattr(lead, "update", count)
  for frame in (0, 1, 5):
    messages = dict(send(frame, lead_filter=lead))
    if frame % 5 == 0:
      assert {"ADRV_0x161", "CCNC_0x162"} <= messages.keys()
  assert calls == [0, 1, 5]
  assert len(updates) == 3
  assert len(logged) == 1 and "CCNC Rust map display failed" in logged[0]


def test_stop_go_dwell_survives_switching_renderers(display, monkeypatch):  # noqa: F811
  params, cs, send = display
  params.update(CcncExtendedDisplay=False, CcncTrafficBackground=True)
  templates(cs)
  monkeypatch.setattr(main, "_stop_go_background", main.StopGoBackground())
  send.hud.trafficState = 1
  cs.out.standstill = True
  cs.out.vEgo = 0.
  assert dict(send(0))["ADRV_0x161"]["BACKGROUND"] == 1
  assert dict(send(50))["ADRV_0x161"]["BACKGROUND"] == 5
  params["CcncExtendedDisplay"] = True
  assert dict(send(100))["ADRV_0x161"]["BACKGROUND"] == 5
  send.hud.trafficState = 0
  cs.out.standstill = False
  cs.out.vEgo = 3.
  assert dict(send(200))["ADRV_0x161"]["BACKGROUND"] == 5
  assert dict(send(220))["ADRV_0x161"]["BACKGROUND"] == 8
  params["CcncExtendedDisplay"] = False
  assert dict(send(300))["ADRV_0x161"]["BACKGROUND"] == 8


def test_first_map_frame_accepts_parsed_float_detects(display, monkeypatch):  # noqa: F811
  # The car's CAN parser hands 0x162 over as floats. With canValid false (the first display
  # frame of a drive) the rear pass leaves LR/RR_DETECT at that 0.0, and the Rust call failed
  # with "'float' object cannot be interpreted as an integer".
  params, cs, send = display
  params["CcncExtendedDisplay"] = True
  templates(cs)
  cs.ccnc_0x162 = dict.fromkeys(cs.ccnc_0x162, 0.)
  cs.out.canValid = False
  assert main._CCNC_DISPLAY_RUST is not None
  logged = []
  monkeypatch.setattr(main.ccnc_radar_map._live, "failed", False)
  monkeypatch.setattr(main.ccnc_radar_map.carlog, "exception", logged.append)
  with monkeypatch.context() as patch:
    patch.setattr(main, "_CCNC_DISPLAY_RUST", None)
    expected = [dict(send(frame))["CCNC_0x162"] for frame in (0, 5)]
  main.ccnc_extension.reset()
  main.ccnc_radar_map.reset_live()
  actual = [dict(send(frame))["CCNC_0x162"] for frame in (0, 5)]
  assert logged == []
  assert actual == expected
