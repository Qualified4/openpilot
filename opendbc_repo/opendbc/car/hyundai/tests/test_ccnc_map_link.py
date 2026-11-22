from pathlib import Path
import tempfile
from types import SimpleNamespace

from openpilot.cereal import messaging
from openpilot.common.params import Params
from opendbc.can import CANPacker
from opendbc.car.hyundai import ccnc_core_loader, hyundaicanfd as canfd
from opendbc.car.hyundai.tests.test_ccnc_extended_display import templates
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401

NS = 1_000_000_000
TICK = NS // 20


def packet(stamp, *, generation=1, valid=True, occupied=True):
  event = messaging.new_message("customReserved3", logMonoTime=stamp, valid=valid)
  payload = event.customReserved3
  payload.radarLogMonoTime = stamp
  payload.generation = generation
  payload.valid = valid
  slots = payload.init("slots", 5)
  if occupied:
    slots[0].present, slots[0].vid, slots[0].detect = True, 9, 2
    slots[0].distance, slots[0].lateral = 20., 1.25
  return event.to_bytes()


def test_fresh_message_uses_process():
  core = ccnc_core_loader.load()
  assert core.MapLink().use_process(NS, packet(NS), NS)
  values = {}
  core.apply_cmap(values, packet(NS), SimpleNamespace())
  assert (values["FF_DETECT"], values["FF_DISTANCE"], values["FF_LATERAL"]) == (4, 16., -1.25)


def test_stale_after_0_25s_uses_mode0():
  link = ccnc_core_loader.load().MapLink()
  msg = packet(NS)
  assert link.use_process(NS, msg, NS)
  assert link.use_process(NS + NS // 4, msg, NS)
  assert not link.use_process(NS + NS // 4 + 1, msg, NS)
  assert not link.use_process(NS + NS // 2, packet(NS + NS // 2), NS + NS // 2)


def test_invalid_uses_mode0():
  link = ccnc_core_loader.load().MapLink()
  assert link.use_process(NS, packet(NS), NS)
  assert not link.use_process(NS + TICK, packet(NS + TICK, valid=False), NS + TICK)


def test_recover_only_after_5s_stable():
  link = ccnc_core_loader.load().MapLink()
  assert not link.use_process(NS, None, None)
  for i in range(100):
    stamp = 2 * NS + i * TICK
    assert not link.use_process(stamp, packet(stamp), stamp)
  assert link.use_process(7 * NS, packet(7 * NS), 7 * NS)
  # A gap that has already ended when card next checks still breaks continuity.
  assert not link.use_process(8 * NS, packet(8 * NS), 8 * NS)


def test_generation_change_resets_animation():
  core = ccnc_core_loader.load()
  panel = object()
  state = SimpleNamespace()
  values = {}
  core.apply_cmap(values, packet(NS), state)
  state._rust_panel = panel
  core.apply_cmap(values, packet(NS + TICK), state)
  assert state._rust_panel is panel
  core.apply_cmap(values, packet(NS + 2 * TICK, generation=2, occupied=False), state)
  assert not hasattr(state, "_rust_panel")
  assert all(values[name] == 0 for name in ("FF_DETECT", "FF_DISTANCE", "FF_LATERAL"))


def test_extended_display_off_byte_identical(display):  # noqa: F811
  params, cs, send = display
  params.update(CcncExtendedDisplay=False, CcncPosDisplayMode=1, CcncMapProcess=True)
  templates(cs)
  expected = dict(send())
  cs.ccnc_map_bytes = packet(NS), NS
  actual = dict(send())
  for name in ("ADRV_0x161", "CCNC_0x162"):
    baseline, candidate = CANPacker("hyundai_canfd_generated"), CANPacker("hyundai_canfd_generated")
    assert baseline.make_can_msg(name, 0, expected[name]) == candidate.make_can_msg(name, 0, actual[name])


def test_corrupt_message_falls_back_without_exception(display, monkeypatch):  # noqa: F811
  from openpilot.selfdrive.car.card import CcncMapSubMaster

  queue_dir = Path(tempfile.mkdtemp(prefix="msgq_map_link_", dir="/dev/shm"))
  monkeypatch.setenv("OPENPILOT_PREFIX", queue_dir.name.removeprefix("msgq_"))
  pub = messaging.pub_sock("customReserved3")
  sm = CcncMapSubMaster()
  params, cs, send = display
  params.update(CcncExtendedDisplay=True, CcncPosDisplayMode=1, CcncMapProcess=True)
  templates(cs)
  clock = SimpleNamespace(now=NS)
  monkeypatch.setattr(canfd, "time", SimpleNamespace(monotonic_ns=lambda: clock.now, monotonic=lambda: clock.now / NS))
  for frame, raw, want_process in (
    (0, packet(NS), True),
    (5, b"broken", False),
    (10, messaging.new_message("customReserved3", logMonoTime=NS + 2 * TICK, valid=True).to_bytes(), False),
  ):
    clock.now = NS + frame * TICK // 5
    pub.send(raw)
    sm.update(0)
    cs.ccnc_map_bytes = sm.latest
    values = dict(send(frame))["CCNC_0x162"]
    assert canfd.ccnc_extension.state._rust_map_link.using_process == want_process
    assert values["FF_DETECT"] == (4 if want_process else 0)


def test_bad_param_value_is_off(display, monkeypatch, tmp_path):  # noqa: F811
  options, cs, send = display
  options["CcncExtendedDisplay"] = True  # Route the fixture through the real display options.
  templates(cs)
  params = Params(str(tmp_path / "map-link-params"))
  params.put_bool("CcncExtendedDisplay", True)
  params.put("CcncPosDisplayMode", 1)
  for key in ("LaneLineCheck", "CcncClusterDisplay", "CcncTrafficBackground", "MyDrivingMode", "HDPuse"):
    params.put(key, params.get_default_value(key))
  monkeypatch.setattr(canfd, "Params", lambda: params)
  cs.ccnc_map_bytes = packet(NS), NS
  assert "CCNC_0x162" in dict(send(0))
  assert not canfd.ccnc_extension.state.ccnc_map_process
  with open(params.get_param_path("CcncMapProcess"), "wb") as out:
    out.write(b"invalid")
  assert "CCNC_0x162" in dict(send(0))
  assert not canfd.ccnc_extension.state.ccnc_map_process
  assert canfd.ccnc_extension.state.pos_display_mode == 1
  with open(params.get_param_path("CcncPosDisplayMode"), "wb") as out:
    out.write(b"invalid")
  params.put_bool("CcncMapProcess", True)
  assert "CCNC_0x162" in dict(send(100))
  assert not canfd.ccnc_extension.state.ccnc_map_process
  assert canfd.ccnc_extension.state.pos_display_mode == 0
