#!/usr/bin/env python3
"""Score ego lane changes using an independent camera reference and radar identities.

`cases` reads the prototype traces to fix the vehicle/event cohort. `capture`
uses the existing replay23 driver and records the source ID of each rendered
slot. `score` compares captured CAN or world-map traces on the same cohort.
"""

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np


def read_trace(path):
  geometry, vehicles = [], defaultdict(list)
  with path.open() as source:
    for line in source:
      fields = line.split()
      t = float(fields[1])
      if fields[0] == "G":
        geometry.append((t, float(fields[6]), bool(int(fields[8]))))
      elif fields[0] == "V":
        vid, rid = map(int, fields[2:4])
        x, y, r = map(float, fields[4:7])
        vehicles[t].append((vid, rid, x, y, r))
  return geometry, vehicles


def camera(geometry):
  rows = [(t, u) for t, u, _ in geometry if np.isfinite(u)]
  if len(rows) < 16:
    return np.array([]), np.array([])
  times, lane = map(np.asarray, zip(*rows, strict=True))
  padded = np.pad(lane, 2, mode="edge")
  return times, np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)


def cases(args):
  found = []
  seams = {}
  for p in args.seams:
    seams.update(json.loads(p.read_text()))
  traces = {p.stem: p for folder in reversed(args.traces) for p in folder.glob("*.trace")}
  for key, path in sorted(traces.items()):
    geometry, vehicles = read_trace(path)
    if not geometry or not geometry[0][2]:
      continue
    times, lane = camera(geometry)
    if not len(times):
      continue
    rounded = np.round(lane)
    last = -np.inf
    for i in np.flatnonzero(np.diff(rounded)) + 1:
      t = float(times[i])
      hold = (times >= t) & (times <= t + 1.)
      if t - last < 6 or hold.sum() <= 15 or not np.all(rounded[hold] == rounded[i]):
        continue
      if abs(np.diff(np.interp([t - 3., t + 3.], times, lane))[0]) < 0.7:
        continue
      parts = [key]
      if key in seams:
        before, after, middle, _ = seams[key]
        if abs(t - middle) >= 4.:
          continue
        parts = [before, after]
      window = (times >= t - 4.) & (times <= t + 4.)
      if not np.any(window):
        continue
      last = t
      tracks = defaultdict(list)
      for stamp, frame in vehicles.items():
        if abs(stamp - t) <= 4.:
          for vid, rid, x, y, _ in frame:
            tracks[vid].append((stamp, rid, x, y))
      selected = []
      for vid, track in tracks.items():
        span = [stamp for stamp, _, x, _ in track if 0. < x < 60.]
        if len(span) < 40 or span[0] > t - 1.5 or span[-1] < t + 1.5:
          continue
        endpoints = np.interp([track[0][0], track[-1][0]], times, lane)
        dy = np.mean([p[3] for p in track[-6:]]) - np.mean([p[3] for p in track[:6]])
        if abs(dy - 3.5 * (endpoints[1] - endpoints[0])) <= 1.2:
          selected.append(dict(vid=vid, track=track))
      if selected:
        # Keep context before and after the window for interpolation at replay send timestamps.
        window = (times >= t - 4.2) & (times <= t + 4.2)
        found.append(dict(segment=key, parts=parts, t=t,
                          camera=list(zip(times[window].tolist(), lane[window].tolist())), cars=selected))
  # A maneuver near the end of one segment can also occur in its stitched pair.
  # Prefer the reference with a wider observation window and retain the crossing once.
  unique = []
  for row in sorted(found, key=lambda row: -len(row['camera'])):
    route = row['parts'][0].rsplit('--', 1)[0]
    if any(p['parts'][0].rsplit('--', 1)[0] == route and abs(row['t'] - p['t']) < 1. for p in unique):
      continue
    unique.append(row)
  found = sorted(unique, key=lambda row: (row['segment'], row['t']))
  args.output.write_text(json.dumps(found))
  print(f"ref_ok={sum(len(c['cars']) for c in found)} cases={len(found)}")


def capture(args):
  """Add diagnostics to the existing replay driver without changing its inputs or CAN output."""
  source = args.driver.read_text()
  ini = 'replay_full.CACHE = Path(os.environ["POS_REFINE_CACHE"])'
  hook = '        rows.append((now, frame, rep[0] if rep else None, logged[0] if logged else None, float(cs.vEgo),'
  result = '    result[f"{route}--{segment}"] = dict(rows=rows, trace=list(TRACE), failed=radar_map._live.failed, fails=list(FAIL), pos=list(PT), restarts=restarts)'
  if not all(source.count(marker) == 1 for marker in (ini, hook, result)):
    raise ValueError("unexpected replay driver layout")
  source = source.replace(ini, '''def read_cache(log, path):
    route, segment = replay_full.segkey(path)
    name = f"{route}--{segment}.capnp"
    cache = next((root / name for root in cache_dirs if (root / name).is_file()), None)
    if cache is None:
        raise RuntimeError(f"event cache missing for {name}")
    return log.Event.read_multiple_bytes(cache.read_bytes(), traversal_limit_in_words=2**50)
replay_full.events = read_cache''')
  source = source.replace('    rows = []; TRACE.clear()', '    slot_sources = []; rows = []; TRACE.clear()')
  source = source.replace(hook, '''        if rep and len(PT) > n_trace:
            tracker = radar_map._live.tracker
            ids = (ext.state.radar_display_tracker.position_correction.display_selected if a.mode == 0 else PT[-1][2][0])
            sources = []
            for key in ids:
                obj = tracker.objects.get(key) if key is not None else None
                # A bridge can retain an identity, but its raw report can be from a different car.
                if obj is not None and not (tracker.bridge and tracker.bridge['key'] == key) and obj.report is not None:
                    if 0. <= tracker.now - obj.report[0] <= .20:
                        sources.append(obj.source)
                        continue
                sources.append(None)
            slot_sources.append((now, sources, rep[0]))
''' + hook)
  source = source.replace(result, '    result[f"{route}--{segment}"] = dict(failed=radar_map._live.failed, fails=list(FAIL), slot_sources=slot_sources)')
  # The replay driver sorts its input paths and carries state across consecutive segments.
  sys.argv = [str(args.driver), str(args.tree), str(args.output), "--mode", str(args.mode),
              *((args.logs.read_text().splitlines()))]
  exec(compile(source, str(args.driver), "exec"), {"__name__": "__main__", "cache_dirs": args.cache})


def shown_world(path):
  geometry, vehicles = read_trace(path)
  for t, _, _ in geometry:
    slots = {}
    for _, rid, x, _, r in vehicles[t]:
      slot = round(r)
      if slot in (-1, 0, 1) and 0. < x < 80. and (slot not in slots or x < slots[slot][0]):
        slots[slot] = (x, rid, -3.5 * r)
    yield t, [(rid, lateral) for _, rid, lateral in slots.values()]


def shown_can(paths):
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value

  signals = CANPacker("hyundai_canfd_generated").dbc.name_to_msg["CCNC_0x162"].sigs
  names = [(signals[base + "_DETECT"], signals[base + suffix]) for base, suffix in
           (("FF", "_LATERAL"), ("LF", "_DETECT_LATERAL"), ("RF", "_DETECT_LATERAL"))]
  for path in paths:
    with Path(path).open("rb") as source:
      replay = pickle.load(source)
    for segment, rows in replay.items():
      if rows["failed"] or rows["fails"]:
        raise RuntimeError(f"replay failed for {segment}")
      frames = []
      for stamp, identities, payload in rows["slot_sources"]:
        data = bytes.fromhex(payload)
        slots = []
        for slot, (identity, (detect, coord)) in enumerate(zip(identities, names, strict=True)):
          if identity is None or not get_raw_value(data, detect):
            continue
          v = get_raw_value(data, coord) * coord.factor + coord.offset
          if slot == 0 and v > 6.35:
            v -= 12.8
          slots.append((identity, v if slot == 1 else -v))
        frames.append((stamp / 1e9, slots))
      yield segment, frames


def score_car(frames, track, times, lane, center):
  stamps = [p[0] for p in track]
  pts = []
  for t, slots in frames:
    if abs(t - center) > 4.:
      continue
    i = bisect_left(stamps, t)
    near = min(range(max(0, i - 1), min(len(stamps), i + 1)), key=lambda k: abs(stamps[k] - t))
    if abs(stamps[near] - t) > .12:
      continue
    rid = track[near][1]
    matching = [lateral for actual, lateral in slots if actual == rid]
    if len(matching) == 1:
      pts.append((t, matching[0]))
  if len(pts) < 10:
    return None
  t, d = map(np.asarray, zip(*pts, strict=True))
  jumps = int(np.sum((np.diff(t) < .3) & (np.abs(np.diff(d)) > .5 + 1e-9)))
  d0 = np.mean(d[t <= t[0] + .3])
  u0 = np.interp(t[0], times, lane)
  tt = np.array([s for s, _ in frames if t[0] <= s <= center + 4.])
  full = 3.5 * (np.interp(tt, times, lane) - u0)
  move = full[-1]
  if abs(move) <= .3:
    return dict(samples=len(t), lag=None, reverse=None, jumps=jumps, end_error=None)
  direction = np.sign(move)
  t50 = tt[np.flatnonzero(direction * full >= abs(move) / 2)[0]]
  reached = np.flatnonzero(direction * (d - d0) >= abs(move) / 2)
  end = d[t >= tt[-1] - .5]
  return dict(samples=len(t), lag=round(float(t[reached[0]] - t50), 3) if len(reached) else None,
              reverse=round(float(max(0., np.max(-direction * (d[t <= t50 + 1.] - d0)))), 3),
              jumps=jumps,
              end_error=round(float(abs(np.mean(end) - d0 - move)), 3) if len(end) else None)


def summary(rows):
  lag = [m['lag'] for _, m in rows if m['lag'] is not None]
  reverse = [m['reverse'] for _, m in rows if m['reverse'] is not None]
  ends = [m['end_error'] for _, m in rows if m['end_error'] is not None]
  return dict(cases=len({key[:2] for key, _ in rows}), cars=len(rows),
              median_lag=round(float(np.median(lag)), 3) if lag else None,
              p90_lag=round(float(np.percentile(lag, 90)), 3) if lag else None,
              moving=len(reverse), not_reached=len(reverse) - len(lag), jumps=sum(m['jumps'] for _, m in rows),
              reverse_over_05=sum(r > .5 for r in reverse),
              reverse_over_10=sum(r > 1. for r in reverse),
              median_end_error=round(float(np.median(ends)), 3) if ends else None,
              end_coverage=len(ends))


def score(args):
  cohort = json.loads(args.cases.read_text())
  methods = {}
  for label, path in args.method:
    world = path[0].is_dir()
    if world:
      wanted = {c['segment'] for c in cohort}
      recordings = {p.stem: list(shown_world(p)) for p in path[0].glob('*.trace') if p.stem in wanted}
    else:
      recordings = dict(shown_can(path))
    stats = {}
    for c in cohort:
      frames = recordings.get(c['segment']) if world else [f for p in c['parts'] for f in recordings.get(p, ())]
      if not frames:
        continue
      t, u = zip(*c['camera'], strict=True)
      for car in c['cars']:
        m = score_car(frames, car['track'], t, u, c['t'])
        if m is not None:
          stats[c['segment'], c['t'], car['vid']] = m
    methods[label] = stats
  common = set.intersection(*(set(m) for m in methods.values()))
  output = dict(individual={k: summary(list(m.items())) for k, m in methods.items()},
                same_cars={k: summary([(key, m[key]) for key in sorted(common)]) for k, m in methods.items()},
                metrics={k: [[*key, v] for key, v in m.items()] for k, m in methods.items()})
  args.output.write_text(json.dumps(output))
  print(json.dumps({k: v for k, v in output.items() if k != 'metrics'}, indent=2))


def method_arg(value):
  label, paths = value.split('=', 1)
  return label, [Path(p) for p in paths.split(',')]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(required=True)
  case_parser = sub.add_parser('cases')
  case_parser.add_argument('output', type=Path)
  case_parser.add_argument('traces', nargs='+', type=Path)
  case_parser.add_argument('--seams', action='append', type=Path, default=[])
  case_parser.set_defaults(run=cases)
  replay = sub.add_parser('capture')
  replay.add_argument('tree', type=Path)
  replay.add_argument('driver', type=Path)
  replay.add_argument('logs', type=Path)
  replay.add_argument('output', type=Path)
  replay.add_argument('--cache', type=Path, action='append', required=True)
  replay.add_argument('--mode', type=int, choices=(0, 1), required=True)
  replay.set_defaults(run=capture)
  scoring = sub.add_parser('score')
  scoring.add_argument('cases', type=Path)
  scoring.add_argument('output', type=Path)
  scoring.add_argument('method', nargs='+', type=method_arg)
  scoring.set_defaults(run=score)
  args = parser.parse_args()
  args.run(args)


if __name__ == '__main__':
  main()
