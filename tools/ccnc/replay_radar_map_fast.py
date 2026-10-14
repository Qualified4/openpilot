"""Fast offline replay of the map candidate from a cache written by replay_radar_map.py --write-cache.

The cache holds the decoded inputs and the unchanged extension's outputs per send, so
neither the log nor the extension is rerun; routes are replayed in parallel. Rows have
the same fields as the slow tool (standalone with --stop-memory --explain, or --hybrid).
The cache is tied to one extension hash and one log hash per segment. Private data.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from types import SimpleNamespace as N

SIDES = ('FF', 'LF', 'RF')
BANDS = ('stopped', '<30', '30-60', '>=60')


def structure_stats(tracker, live, model_info, speed, counts):
  """Per send: resting side returns, how many the map classes as resting and as wall members,
  and what the walls keep out of the display, with the map's own evidence of that object."""
  kph = speed * 3.6
  band = BANDS[0 if kph <= .3 else 1 if kph < 30. else 2 if kph < 60. else 3]
  yaw = abs(model_info['yaw'][0]) if model_info['yaw'] else 0.
  groups = [band] + (['curve'] if speed >= 3. and yaw * 200. >= speed else [])  # Radius under 200m.
  reported = {int(p.trackId): p.vLead for p in live.points} if live is not None else {}
  add = lambda key, n=1: [counts.__setitem__((g, key), counts.get((g, key), 0) + n) for g in groups]
  add('sends')
  add('walls', len(tracker.wall_geometry))
  add('wall members', sum(o.structure for o in tracker.objects.values()))
  # Returns a band took by prediction, and those of them that later showed vehicle evidence.
  seen = tracker.__dict__.setdefault('_replay_early', {})
  for key in getattr(tracker, 'band_early', ()):
    obj = tracker.objects.get(key)
    if obj is None:
      continue
    if key not in seen:
      seen[key] = None
      add('taken by prediction')
    evidence = 'class 2' if obj.radar_moving else 'speed' if obj.doppler_confirmed else 'vision' if obj.vision_seen else None
    if evidence and seen[key] is None:
      seen[key] = evidence
      add(f'taken by prediction, later {evidence}')
  joins = getattr(tracker, 'band_joins', 0)
  add('band joins', joins - tracker.__dict__.get('_replay_joins', 0))
  tracker._replay_joins = joins
  for obj in tracker.objects.values():
    x, y = tracker.local[obj.key]
    fresh = tracker.now - obj.seen <= .15
    vehicle = obj.radar_moving or obj.doppler_confirmed
    if fresh and -5. <= x <= 20. and 1.5 <= abs(y) <= 6. and not vehicle and not obj.vision:
      add('side candidates')
      add('side stationary', obj.stationary)
      add('side structure', obj.structure)
    if fresh and obj.radar_moving and obj.stationary:
      add('class 2 stationary')
      add('class 2 stationary, reported over 2m/s', abs(reported.get(obj.source, 0.)) > 2.)
    if fresh and obj.radar_moving and obj.structure:
      add('class 2 structure')
    # The display conditions of _select that do not involve walls.
    if (tracker.now - obj.seen > tracker.PREDICTION_HOLD + 1e-6 or obj.hits < 3 or obj.seen - obj.born < .15 or obj.existence < .5
        or not 0. <= x <= (tracker.FAR_X if obj.motion_confirmed or obj.vision else tracker.MAX_X)
        or (obj.vehicle < .6 and not obj.admitted) or obj.uncertainty > tracker.MAX_DISPLAY_UNCERTAINTY):
      continue
    free = tracker._lane(x, y, moving=obj.motion_confirmed, settled=obj.settled)[0]
    if free is None and obj.vision and abs(y) <= 2.:
      free = 0
    if free is None or (free and (x > tracker.MAX_SIDE_X or obj.last_motion[0] < -1. / 3.6)):
      continue
    walled = tracker._lane(x, y, respect_walls=True, moving=obj.motion_confirmed, settled=obj.settled)[0]
    if walled is None and obj.vision and abs(y) <= 2. and not tracker._wall_outside(x, y):
      walled = 0
    reason = ('wall member' if obj.structure else 'lane cut by wall' if walled is None else
              'beyond wall' if not obj.motion_confirmed and not obj.vision and tracker._wall_outside(x, y) else None)
    if reason:
      evidence = ('class 2' if obj.radar_moving else 'speed' if obj.doppler_confirmed else 'vision' if obj.vision
                  else 'earlier vision' if obj.vision_seen else 'waiting' if obj.admitted else 'none')
      add(f'{reason}: {evidence}')
      if tracker.selected[free] is None:
        add(f'{reason}, slot left empty: {evidence}')


def load_map(path):
  spec = importlib.util.spec_from_file_location('ccnc_map_fast_replay', path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def run_route(job):
  candidate, cache_dir, output, names, hybrid, timing, maps, structure = job
  tracker = load_map(candidate).CcncRadarMap()
  fields = [k for group in tracker.SLOT_FIELDS for k in group]
  result = []
  for name in names:
    with open(Path(cache_dir) / f'{name}.pickle', 'rb') as f:
      cache = pickle.load(f)
    lives = [(N(points=[N(trackId=p[0], dRel=p[1], yRel=p[2], vRel=p[3], vLead=p[4], yvRel=p[5], measured=p[6], radarSource=p[7])
                        for p in live['points']]), live) for live in cache['lives']]
    models = [(N(laneLines=[N(x=x, y=y) for x, y in m['lines']], laneLineProbs=m['probs'],
                 roadEdges=[N(x=x, y=y) for x, y in m['edges']], orientationRate=N(z=m['yaw']),
                 leadsV3=[N(prob=prob, x=x, y=y) for prob, x, y in m['leads']]), m) for m in cache['models']]
    durations, peak, counts = [], 0, {}
    reports = {}
    with gzip.open(Path(output) / f'{name}.jsonl.gz', 'wt', encoding='utf8') as out:
      for s in cache['sends']:
        live, live_info = lives[s['live']]
        model, model_info = models[s['model']]
        start = time.perf_counter()
        tracker.update(live if s['live_fresh'] else None, model if s['model_fresh'] else None, s['now'], s['v'],
                       radar_time=live_info['time'], model_time=model_info['time'], reverse=s['reverse'],
                       evidence=live_info['evidence'],
                       **({'select': False} if hybrid else {'a_ego': s['a'], 'blindspot': s['blind']}))
        if hybrid:
          value = tracker.refine_existing(dict(s['values'][1]),
                                          [None if p is None else N(trackId=p[0], dRel=p[1], yRel=p[2]) for p in s['points']])
          value = {k: value[k] for k in fields}
        else:
          value = tracker.display_values(dict.fromkeys(fields, 0))
        durations.append((time.perf_counter() - start) * 1000.)
        peak = max(peak, len(tracker.objects))
        if structure:
          structure_stats(tracker, live if s['live_fresh'] else None, model_info, s['v'], counts)
        row = dict(t=s['t'], now=s['now'], values=[s['values'][0], s['values'][1], value], selected=tracker.selected, ego=s['v'])
        row['evidence'] = [None if (o := tracker.objects.get(k)) is None else
                           'radar' if o.radar_moving else 'doppler' if o.doppler_confirmed else
                           'vision' if o.vision else 'moved' if o.motion_confirmed else 'waiting' if o.admitted else 'none'
                           for k in tracker.selected]
        if not hybrid:
          row['remembered'] = list(tracker.remembered)
          row['why'] = [None if p is None else tracker.explain(p[0], p[1], p[2], slot) for slot, p in enumerate(s['sources'])]
          row['legacy_ids'] = [None if p is None else p[0] for p in s['sources']]
          row['candidate'] = [None if (o := tracker.objects.get(k)) is None else
                              (o.source, *tracker.local[k], tracker.cos * o.vx + tracker.sin * o.vy) for k in tracker.selected]
        if maps:
          row['map'] = tracker.snapshot()
        if reports is not None:
          # Last radar report (class, distance, track state, PROB) linked to each object, for analysis.
          for obj in tracker.objects.values():
            if obj.seen == live_info['time'] and obj.source in live_info['evidence']:
              reports[obj.key] = live_info['evidence'][obj.source]
          row['reports'] = [reports.get(k) for k in tracker.selected]
        out.write(json.dumps(row, separators=(',', ':')) + '\n')
    result.append(dict(name=name, peaks=dict(objects=peak), structure=[[*key, n] for key, n in counts.items()], timing=dict(
      mean=statistics.mean(durations) if durations else 0., maximum=max(durations, default=0.),
      p99=sorted(durations)[min(len(durations) - 1, int(len(durations) * .99))] if durations else 0.) if timing else None))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True, help='Same manifest order as the cache run; defines route continuity')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, default=Path('opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_radar_map.py'))
  parser.add_argument('--hybrid', action='store_true', help='Hide the unchanged extension displays matched to map walls')
  parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2),
                      help='Worker processes; consecutive segments of a route stay in one job')
  parser.add_argument('--maps', action='store_true', help='Store the full map snapshot per send (large)')
  parser.add_argument('--structure', action='store_true', help='Count resting side returns, walls and what walls keep out of the display')
  parser.add_argument('--timing', action='store_true', help='Record map time per send; use --jobs 1 and no other workload')
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  names = [e['name'] for e in json.loads(args.manifest.read_text(encoding='utf8'))['entries']]
  groups, previous = [], None
  for name in names:
    route, segment = name.rsplit('--', 1)
    if previous != (route, int(segment) - 1):
      groups.append([])
    groups[-1].append(name)
    previous = route, int(segment)
  jobs = [(str(args.candidate), str(args.cache), str(args.output), group, args.hybrid, args.timing, args.maps, args.structure) for group in groups]
  if args.jobs <= 1:
    results = [run_route(job) for job in jobs]
  else:
    with Pool(args.jobs) as pool:
      results = pool.map(run_route, sorted(jobs, key=lambda job: -len(job[3])), chunksize=1)
  by_name = {entry['name']: entry for group in results for entry in group}
  (args.output / 'summary.json').write_text(json.dumps([by_name[name] for name in names], indent=1))
  (args.output / 'identity.json').write_text(json.dumps(dict(
    candidate=hashlib.sha256(args.candidate.read_bytes()).hexdigest(), mode='hybrid' if args.hybrid else 'standalone',
    cache=str(args.cache), manifest=hashlib.sha256(args.manifest.read_bytes()).hexdigest()), indent=1))
  print(len(names), 'segments in', len(groups), 'routes')


if __name__ == '__main__':
  main()
