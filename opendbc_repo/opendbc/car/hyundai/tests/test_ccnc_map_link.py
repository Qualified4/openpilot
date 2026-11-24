from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from opendbc.can import CANPacker
from opendbc.car.hyundai import ccnc_core_loader, hyundaicanfd as canfd
from opendbc.car.hyundai.tests.test_ccnc_extended_display import templates
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401
from opendbc.car.hyundai.tests.test_ccnc_lane_marking import marking

NS = 1_000_000_000
TICK = NS // 20
STALE = 250_000_000


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
  core.apply_cmap(values, packet(NS))
  assert (values["FF_DETECT"], values["FF_DISTANCE"], values["FF_LATERAL"]) == (4, 16., -1.25)


def test_stale_after_0_25s_uses_mode0():
  link = ccnc_core_loader.load().MapLink()
  msg = packet(NS)
  assert link.use_process(NS, msg, NS)
  assert link.use_process(NS + NS // 4, msg, NS)
  assert not link.use_process(NS + NS // 4 + 1, msg, NS)
  assert not link.use_process(NS + NS // 2, packet(NS + NS // 2), NS + NS // 2)


def test_invalid_keeps_the_map_for_0_25s_then_uses_mode0():
  link = ccnc_core_loader.load().MapLink()
  assert link.use_process(NS, packet(NS), NS)
  for i in range(1, 6):
    stamp = NS + i * TICK
    assert link.use_process(stamp, packet(stamp, valid=False), stamp) == (i * TICK <= STALE)


def test_recover_only_after_2s_stable():
  link = ccnc_core_loader.load().MapLink()
  assert not link.use_process(NS, None, None)
  for i in range(40):
    stamp = 2 * NS + i * TICK
    assert not link.use_process(stamp, packet(stamp), stamp)
  assert link.use_process(4 * NS, packet(4 * NS), 4 * NS)
  # A gap that has already ended when card next checks still breaks continuity.
  assert not link.use_process(5 * NS, packet(5 * NS), 5 * NS)


def test_generation_change_drops_the_old_slots():
  core = ccnc_core_loader.load()
  values = {}
  core.apply_cmap(values, packet(NS))
  assert values["FF_DETECT"] == 4
  core.apply_cmap(values, packet(NS + TICK, generation=2, occupied=False))
  assert all(values[name] == 0 for name in ("FF_DETECT", "FF_DISTANCE", "FF_LATERAL"))


@pytest.mark.parametrize("map_process", (False, True))
def test_lane_lines_do_not_depend_on_the_map(display, monkeypatch, map_process):  # noqa: F811
  """The recognised lane lines stay as with the map off while the map resets, goes missing, is late or invalid, and returns."""
  params, cs, send = display
  params.update(CcncExtendedDisplay=True, CcncPosDisplayMode=1, CcncMapProcess=map_process)
  templates(cs)
  cs.modelV2.meta.laneChangeAvailableLeft = cs.modelV2.meta.laneChangeAvailableRight = False
  clock = SimpleNamespace(now=NS)
  monkeypatch.setattr(canfd, "time", SimpleNamespace(monotonic_ns=lambda: clock.now, monotonic=lambda: clock.now / NS))
  recognised = marking(left=("solid", "yellow"), right=("dashed", "white"))

  def map_input(t, now):
    if t < .5:
      return packet(now), now
    if t < 1.:
      return packet(now, generation=2), now  # the map reset itself
    if t < 1.3:
      return None, None  # nothing received
    if t < 1.6:
      return packet(now - NS // 2, generation=2), now - NS // 2  # late
    if t < 1.9:
      return packet(now, generation=2, valid=False), now
    return packet(now, generation=3), now  # back after the 5 s recovery, reset again meanwhile

  using = {}
  for step in range(150):
    t = step / 20
    clock.now = NS + step * TICK
    # The recogniser has no fresh result on the frames where the map changes; the display keeps its last one.
    quiet = t == .5 or 1. <= t < 1.1 or 6.8 <= t < 7.2
    cs.ccnc_lane_marking = None if quiet else recognised
    cs.ccnc_map_bytes = map_input(t, clock.now)
    values = dict(send(step * 5))["ADRV_0x161"]
    assert (values["LANELINE_LEFT"], values["LANELINE_RIGHT"]) == (4, 8), t
    link = getattr(canfd.ccnc_extension.state, "_rust_map_link", None)
    using[t] = link is not None and link.using_process
  if map_process:
    assert [using[t] for t in (.45, .95, 1.2, 1.5, 1.8, 3., 6.8, 7.4)] == [True, True, False, False, False, False, False, True]
  else:
    assert not any(using.values())


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


def test_latest_holds_the_bytes_as_received(monkeypatch):
  from openpilot.selfdrive.car.card import CcncMapSubMaster

  queue_dir = Path(tempfile.mkdtemp(prefix="msgq_map_bytes_", dir="/dev/shm"))
  monkeypatch.setenv("OPENPILOT_PREFIX", queue_dir.name.removeprefix("msgq_"))
  pub = messaging.pub_sock("customReserved3")
  sm = CcncMapSubMaster()
  # A message with far pointers: serializing it again would give other bytes.
  event = log.Event.new_message(num_first_segment_words=1, logMonoTime=NS, valid=True)
  payload = event.init("customReserved3")
  payload.radarLogMonoTime, payload.generation, payload.valid = NS, 1, True
  payload.init("slots", 5)
  raw = event.to_bytes()
  assert messaging.log_from_bytes(raw).as_builder().to_bytes() != raw
  pub.send(raw)
  sm.update(0)
  assert sm.latest == (raw, NS)


def test_same_message_is_decoded_once():
  core = ccnc_core_loader.load()
  link = core.MapLink()
  raw = packet(NS)
  # The link keeps the selection rules every tick while the message stays the same.
  assert link.use_process(NS, raw, NS)
  assert link.use_process(NS + STALE, raw, NS)
  assert not link.use_process(NS + STALE + 1, raw, NS)
  assert not link.use_process(NS + STALE + 1, b"broken", NS + STALE + 1)
  assert not link.use_process(NS + STALE + 2, raw, NS + STALE + 2), "recovering after the gap"


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
