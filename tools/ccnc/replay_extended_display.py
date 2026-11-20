"""Capture or compare CCNC CAN payloads offline using the shared rlog event reader.

Use --baseline-hyundai with the Hyundai package and x86_64 ABI 10 extension
built from a6403c08. Supply the same logs, --extended and --traffic to both runs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace as NS

import replay_full


def capture(args):
  params = replay_full.PARAMS
  params.clear()
  params.update(CcncExtendedDisplay=args.extended, CcncTrafficBackground=args.traffic,
                CcncClusterDisplay=args.cluster, MyDrivingMode=3, HDPuse=0, LaneLineCheck=0)
  if args.baseline_hyundai:
    params.update(CcncLaneColor=args.extended, CcncModelLanes=2 if args.extended else 0,
                  CcncLaneCurvatureSource=3 if args.extended else 0, CcncLaneTypeDetection=int(args.extended),
                  CcncRadarVehicles=args.extended, CcncVehiclePositionCorrection=args.extended, CcncRadarMap=args.extended)
  replay_full._params_stub()

  from opendbc.car import hyundai
  if args.baseline_hyundai:
    hyundai.__path__.insert(0, str(args.baseline_hyundai.resolve()))

  from openpilot.cereal import log
  from openpilot.selfdrive.carrot.lane_type.marking import parse
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value
  from opendbc.car.hyundai import ccnc_core_loader, hyundaicanfd as display
  from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ext, hyundaicanfd_ccnc_radar_map as radar_map
  from opendbc.car.hyundai.values import CAR, HyundaiFlags

  flags = int(HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC)
  cp = NS(brand="hyundai", flags=flags, openpilotLongitudinalControl=True,
          carFingerprint=CAR.HYUNDAI_SONATA_2024, extFlags=1 << 20)
  if not args.baseline_hyundai:
    from opendbc.car.hyundai.values import HyundaiExtFlags
    cp.extFlags = int(HyundaiExtFlags.CCNC_CLUSTER_RX)
  bus = NS(ECAN=0, CAM=2)
  core = ccnc_core_loader.load()
  if core is None:
    raise RuntimeError("CCNC Rust extension required for the replay")

  counters = Counter()

  def count(fn, name):
    def wrapped(*args, **kwargs):
      counters[name] += 1
      return fn(*args, **kwargs)
    return wrapped

  core.update_display = count(core.update_display, "rust")
  if args.extended and not args.baseline_hyundai:
    for owner, names in (
      (ext, ("update_lanes", "update_speed_limit", "update_lfa_icon", "update_vehicles")),
      (display, ("_display_lead", "_apply_driver_monitoring_alert", "_apply_cluster_lane_lines",
                 "_hide_replaced_adas_service_warning", "_select_cluster_background")),
      (display.ccnc_lane_marking.SideDisplay, ("update",)),
      (display.ccnc_parking.ParkingLaneHide, ("update",)),
      (display.DisplayLeadLateralFilter, ("update",)),
    ):
      for name in names:
        setattr(owner, name, count(getattr(owner, name), "python_display"))

  # This replay also needs lane marking packets. Preserve event caches made for replay_full.
  replay_full.WANTED += ("customReservedRawData2",)
  replay_full.CACHE /= "extended-display"
  rx_messages = {
    0x161: ("ADRV_0x161", "adrv_0x161"), 0x162: ("CCNC_0x162", "ccnc_0x162"),
    0x200: ("ADRV_0x200", "adrv_0x200"), 0x1EA: ("ADRV_0x1ea", "adrv_0x1ea"),
    0x1E0: ("LFAHDA_CLUSTER", "lfahda_cluster"),
  }
  clock = NS(ns=0)
  display.time = ext.time = NS(monotonic=lambda: clock.ns / 1e9, monotonic_ns=lambda: clock.ns)
  previous = None
  summary = []
  for path in sorted(args.logs, key=replay_full.segkey):
    route, segment = replay_full.segkey(path)
    if previous != (route, segment - 1):
      packer = CANPacker("hyundai_canfd_generated")
      lead_filter = display.DisplayLeadLateralFilter()
      latest, readers, raw, rx = {}, {}, {}, {}
      marking, origin, last_frame = None, None, None
      display.create_ccnc_messages.__dict__.clear()
      display._stop_go_background.reset()
      ext.reset()
      radar_map.reset_live()
      radar_map._live.failed = False
    previous = route, segment
    counters.clear()
    times, rows = [], []
    for event in replay_full.events(log, path):
      clock.ns = now = event.logMonoTime
      kind = event.which()
      if kind == "can":
        for packet in event.can:
          if packet.src == bus.CAM and packet.address in rx_messages:
            name, attr = rx_messages[packet.address]
            rx[attr] = {key: get_raw_value(packet.dat, sig) * sig.factor + sig.offset
                        for key, sig in packer.dbc.name_to_msg[name].sigs.items()}
        continue
      if kind == "customReservedRawData2":
        try:
          marking = parse(bytes(event.customReservedRawData2), now)
        except (TypeError, ValueError):
          pass
        continue
      elif kind != "sendcan":
        latest[kind] = event
        readers[kind] = getattr(event, kind)
        if kind in ("modelV2", "liveTracks"):
          raw[kind] = event.as_builder().to_bytes()
        continue
      if not any(packet.src == bus.ECAN and packet.address in (0x161, 0x162) for packet in event.sendcan):
        continue
      if not {"carState", "carControl", "modelV2", "liveTracks"} <= latest.keys() or not {"adrv_0x161", "ccnc_0x162"} <= rx.keys():
        continue

      def fresh(key, age):
        entry = latest.get(key)
        return readers[key] if entry is not None and entry.valid and now - entry.logMonoTime < age else None

      if origin is None:
        origin = now
      frame = 5 * round((now - origin) / 50_000_000)
      if last_frame is not None:
        frame = max(frame, last_frame + 5)
      last_frame = frame
      cs, cc = readers["carState"], readers["carControl"]
      model, tracks = fresh("modelV2", 250_000_000), fresh("liveTracks", 150_000_000)
      state = NS(out=cs, modelV2=model, radarState=fresh("radarState", 1_000_000_000), live_tracks=tracks,
                 live_tracks_bytes=raw["liveTracks"] if tracks is not None else None,
                 model_v2_bytes=raw["modelV2"] if model is not None else None, ccnc_lane_marking=marking,
                 cruise_buttons_msg=None, scc_control=None, is_metric=True, trailer_connected=False,
                 softHoldActive=0, paddle_button_prev=0,
                 **{attr: rx.get(attr) for _, attr in rx_messages.values()})
      start = time.perf_counter_ns()
      messages = display.create_ccnc_messages(cp, packer, bus, frame, cc, state, cc.hudControl,
                                              0., False, False, 0, False, 0, 0, lead_filter=lead_filter)
      times.append(time.perf_counter_ns() - start)
      rows.append([frame, *[(addr, dat.hex(), src) for addr, dat, src in messages if addr in (0x161, 0x162, 0x1EA)]])
    summary.append(dict(segment=f"{route}--{segment}", ticks=len(rows), counters=dict(counters), extended=args.extended,
                        mean_ms=mean(times) / 1e6 if times else None, fallback_logged=radar_map._live.failed, rows=rows))
  return summary


def compare(actual, expected):
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value

  dbc = CANPacker("hyundai_canfd_generated").dbc
  equal = True
  for got, want in zip(actual, expected, strict=True):
    assert got["segment"] == want["segment"]
    counts, different, fields = Counter(), Counter(), Counter()
    for (frame, *a), (reference_frame, *b) in zip(got["rows"], want["rows"], strict=True):
      assert frame == reference_frame
      ref = {addr: (data, bus) for addr, data, bus in b}
      assert set(ref) == {addr for addr, _, _ in a}
      for addr, data, bus in a:
        counts[hex(addr)] += 1
        old, old_bus = ref[addr]
        if (data, bus) == (old, old_bus):
          continue
        equal = False
        different[hex(addr)] += 1
        if bus != old_bus:
          fields[f"{addr:#x}/bus"] += 1
        for sig in dbc.addr_to_msg[addr].sigs.values():
          if get_raw_value(bytes.fromhex(data), sig) != get_raw_value(bytes.fromhex(old), sig):
            fields[f"{addr:#x}/{sig.name}"] += 1
    print(json.dumps(dict(segment=got["segment"], frames=dict(counts), mismatches=dict(different),
                          candidate_ms=got["mean_ms"], baseline_ms=want["mean_ms"], counters=got["counters"],
                          fallback_logged=got["fallback_logged"], differing_fields=fields.most_common()), ensure_ascii=False))
    equal &= not got["fallback_logged"] and not want["fallback_logged"]
    if "extended" in got:
      count = got["ticks"] if got["extended"] else 0
      equal &= bool(got["ticks"]) and got["counters"].get("rust", 0) == count
      if got["extended"]:
        equal &= got["counters"].get("python_display", 0) == 0
  return equal


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("logs", type=Path, nargs="*")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--baseline-hyundai", type=Path)
  parser.add_argument("--expected", type=Path)
  parser.add_argument("--extended", action="store_true")
  parser.add_argument("--traffic", action="store_true")
  parser.add_argument("--cluster", action="store_true")
  args = parser.parse_args()
  if args.logs:
    actual = capture(args)
    args.output.write_text(json.dumps(actual))
  else:
    actual = json.loads(args.output.read_text())
  if args.expected:
    return int(not compare(actual, json.loads(args.expected.read_text())))
  print(json.dumps([{k: v for k, v in item.items() if k != "rows"} for item in actual], ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
