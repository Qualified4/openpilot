"""Offline replay of the live CcncRadarMap entry point against what the car actually sent.

For every recorded sendcan carrying CCNC_0x162 on bus 0 the latest carState, modelV2 (<250ms)
and liveTracks (<150ms) are given to hyundaicanfd_ccnc_radar_map.update_vehicles, as the
controller does, and to the existing selector in a separate module instance. The recorded
0x162 is decoded for comparison. Consecutive segments share state.

Bookmarks: [{"segment": 5, "t": 44.84, "slot": "LF", "title": "..."}], t in seconds from the
segment's first sendcan. Logs, bookmarks and output are private data.
Windows can inject the local cereal bootstrap before invoking main().
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import ModuleType, SimpleNamespace as N

SIDES = ('FF', 'LF', 'RF')
DETECT = ('FF_DETECT', 'LF_DETECT', 'RF_DETECT')
DISTANCE = ('FF_DISTANCE', 'LF_DETECT_DISTANCE', 'RF_DETECT_DISTANCE')
FLAGS = ('radar_moving', 'doppler_confirmed', 'vision', 'admitted')


def load_map(path):
  spec = importlib.util.spec_from_file_location('ccnc_map_live_replay', path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def evidence(tracker, slot, interpolate):
  """Why the map shows this slot's object: the map's own flags, not video truth."""
  obj = tracker.objects.get(tracker.selected[slot]) if tracker is not None else None
  if obj is None:
    return None
  x, y = tracker.local[obj.key]
  flags = [name for name in FLAGS if getattr(obj, name)] + (['remembered'] if tracker.remembered[slot] else [])
  # Distance past ego's own lane line, the quantity BODY_CLEARANCE is compared with.
  line = interpolate(tracker.geometry.get(slot, []), x) if slot else None  # Geometry 1/2: ego's left/right line.
  return dict(key=obj.key, source=obj.source, x=round(x, 2), y=round(y, 2), flags=flags or ['none'], tracked=obj.tracked,
              hits=obj.hits, unseen=round(tracker.now - obj.seen, 2), doppler=obj.doppler, forward=round(obj.last_motion[0], 2),
              clearance=None if line is None else round(y - line if slot == 1 else line - y, 2))


def label(info):
  return '+'.join(info['flags']) if info else 'no object'


def segment_of(path):
  match = re.search(r'([0-9a-f]+--[0-9a-f]+)--(\d+)--rlog', path.name) or re.search(r'([0-9a-f]+--[0-9a-f]+)--(\d+)$', path.parent.name)
  return match.group(1), int(match.group(2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('logs', type=Path, nargs='+', help='rlog.zst files; consecutive segments of a route share state')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--bookmarks', type=Path)
  parser.add_argument('--window', type=float, default=.5, help='Seconds around each bookmark')
  parser.add_argument('--baseline', type=Path, default=Path('opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py'))
  parser.add_argument('--candidate', type=Path, default=Path('opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_radar_map.py'))
  parser.add_argument('--compare', type=Path, help='Output directory of an earlier run: report map displays removed/added since')
  args = parser.parse_args()
  from openpilot.cereal import log
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value
  args.output.mkdir(parents=True, exist_ok=True)
  clock = [0.]
  extensions = []
  for i in range(2):  # 0: conventions for the map path, 1: the existing selector.
    m = ModuleType(f'ccnc_live_replay_ext_{i}')
    source = args.baseline.read_text(encoding='utf8').replace(
      '  except:\n    values = CS.ccnc_0x162.copy()', '  except:\n    raise\n    values = CS.ccnc_0x162.copy()')
    exec(compile(source, str(args.baseline), 'exec'), m.__dict__)
    m.time = N(monotonic=lambda: clock[0])
    extensions.append(m)
  radar_map = load_map(args.candidate)
  packer = CANPacker('hyundai_canfd_generated')
  signals = packer.dbc.name_to_msg['CCNC_0x162'].sigs
  lane_fields = packer.dbc.name_to_msg['ADRV_0x161'].sigs
  bookmarks = json.loads(args.bookmarks.read_text(encoding='utf8')) if args.bookmarks else []
  hits = [dict(actual=0, map=0, legacy=0, sends=0, evidence=Counter()) for _ in bookmarks]
  totals = Counter()
  intervals = []
  changes = {kind: Counter() for kind in ('removed', 'added')}
  previous, base = None, None
  for path in sorted(args.logs, key=segment_of):
    route, segment = segment_of(path)
    if previous != (route, segment - 1):
      base, latest = None, {}
      radar_map.reset_live()
      radar_map._live.failed = False
      for m in extensions:
        m.reset()
        m.configure(True, True, True, True, 2)
    previous = route, segment
    raw = subprocess.check_output(['zstd', '-d', '-c', str(path)])
    events = sorted((e for e in log.Event.read_multiple_bytes(raw, traversal_limit_in_words=2**50)
                     if e.which() in ('carState', 'modelV2', 'liveTracks', 'sendcan')), key=lambda e: e.logMonoTime)
    zero = next(e.logMonoTime for e in events if e.which() == 'sendcan')
    reference = gzip.open(args.compare / f'{segment}.jsonl.gz', 'rt', encoding='utf8') if args.compare else None
    counts, last = Counter(), None
    with gzip.open(args.output / f'{segment}.jsonl.gz', 'wt', encoding='utf8') as output:
      for e in events:
        now, kind = e.logMonoTime, e.which()
        clock[0] = now / 1e9
        if kind != 'sendcan':
          latest[kind] = e
          continue
        if base is None:
          base = now
        sent = next((p.dat for p in e.sendcan if p.address == 0x162 and p.src == 0), None)
        if sent is None:
          continue
        if not all(k in latest for k in ('carState', 'modelV2', 'liveTracks')):
          counts['unready'] += 1
          continue
        if last is not None:
          intervals.append((now - last) / 1e9)
        last = now
        cs = latest['carState'].carState

        def fresh(key, budget):
          event = latest[key]
          return getattr(event, key) if event.valid and 0 <= now - event.logMonoTime < budget else None
        md, live = fresh('modelV2', 250000000), fresh('liveTracks', 150000000)
        frame = round((now - base) / 1e7)
        shown, distances = [], []
        for m in extensions:
          car = N(out=cs, live_tracks=live, ccnc_0x162=dict.fromkeys(signals, 0))
          m.update_lanes(dict.fromkeys(lane_fields, 0), car, md, cs.vEgo * 3.6, cs.aEgo * 3.6,
                         int(md.meta.desire.raw) if md else 0, cs.latEnabled, True, True, frame)
          if m is extensions[0]:
            v = radar_map.update_vehicles(dict(car.ccnc_0x162), car, md, frame, cs.vEgo * 3.6, cs.aEgo * 3.6, True, m)
          else:
            v = m.update_vehicles(dict(car.ccnc_0x162), car, md, frame, cs.vEgo * 3.6, cs.aEgo * 3.6, True)
          shown.append([bool(v[k]) for k in DETECT])
          distances.append([round(v[k] / .8, 1) for k in DISTANCE])
        assert not radar_map._live.failed, 'the map path raised and fell back to the existing selector'
        tracker = radar_map._live.tracker
        actual = [bool(get_raw_value(bytes(sent), signals[k])) for k in DETECT]
        sent_distance = [round((get_raw_value(bytes(sent), signals[k]) * signals[k].factor + signals[k].offset) / .8, 1) for k in DISTANCE]
        t = (now - zero) / 1e9
        info = [evidence(tracker, slot, radar_map.interpolate) if shown[0][slot] else None for slot in range(3)]
        # Longitudinal distances in metres (the field carries 0.8 of it): recorded, map, existing selector.
        row = dict(t=round(t, 3), ego=round(cs.vEgo, 2), actual=actual, map=shown[0], legacy=shown[1], objects=info,
                   distance=[sent_distance, *distances])
        output.write(json.dumps(row, separators=(',', ':')) + '\n')
        counts['sends'] += 1
        for slot, side in enumerate(SIDES):
          counts[side + ' map agrees'] += shown[0][slot] == actual[slot]
          counts[side + ' legacy agrees'] += shown[1][slot] == actual[slot]
        for mark, hit in zip(bookmarks, hits):
          if mark['segment'] == segment and abs(t - mark['t']) <= args.window + 1e-6:
            slot = SIDES.index(mark['slot'])
            hit['sends'] += 1
            hit['actual'] += actual[slot]
            hit['map'] += shown[0][slot]
            hit['legacy'] += shown[1][slot]
            if shown[0][slot]:
              hit['evidence'][label(info[slot])] += 1
        if reference is not None:
          old = json.loads(reference.readline())
          assert old['t'] == row['t'], (segment, old['t'], row['t'])
          for slot, side in enumerate(SIDES):
            if old['map'][slot] != shown[0][slot]:
              kind, who = ('removed', old['objects'][slot]) if old['map'][slot] else ('added', info[slot])
              changes[kind][(segment, side, label(who), 'legacy shows' if shown[1][slot] else 'legacy hides')] += 1
    n = max(1, counts['sends'])
    print(f"{segment}: sends {counts['sends']} unready {counts['unready']} display agreement with the recorded 0x162: "
          + ' '.join(f"{side} map {100 * counts[side + ' map agrees'] / n:.1f}% legacy {100 * counts[side + ' legacy agrees'] / n:.1f}%" for side in SIDES),
          flush=True)
    totals += counts
    del events, raw
  period = statistics.median(intervals) if intervals else 0.
  n = max(1, totals['sends'])
  print(f"all: sends {totals['sends']} period {period:.3f}s "
        + ' '.join(f"{side} map {100 * totals[side + ' map agrees'] / n:.2f}% legacy {100 * totals[side + ' legacy agrees'] / n:.2f}%" for side in SIDES))
  for mark, hit in zip(bookmarks, hits):
    print(f"bookmark {mark['segment']}/{mark['t']:.2f} {mark['slot']} {mark.get('title', '')}: sends {hit['sends']} "
          f"actual {hit['actual']} map {hit['map']} ({hit['map'] * period:.2f}s) legacy {hit['legacy']} {dict(hit['evidence'])}")
  for kind, counter in changes.items() if args.compare else ():
    print(f"map displays {kind} since {args.compare}: {sum(counter.values())} sends ({sum(counter.values()) * period:.1f}s)")
    for (segment, side, why, legacy), count in sorted(counter.items(), key=lambda item: (item[0][0], item[0][1], -item[1])):
      print(f"  {segment} {side} {why} [{legacy}]: {count} ({count * period:.2f}s)")
  (args.output / 'summary.json').write_text(json.dumps(dict(
    totals=dict(totals), period=period, bookmarks=[dict(mark, **dict(hit, evidence=dict(hit['evidence']))) for mark, hit in zip(bookmarks, hits)],
    changes={kind: [[*key, count] for key, count in counter.items()] for kind, counter in changes.items()}), ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
  main()
