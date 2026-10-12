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
from pathlib import Path
import pickle
import statistics
import sys
import time
from types import SimpleNamespace as N

SIDES = ('FF', 'LF', 'RF')


def load_map(path):
  spec = importlib.util.spec_from_file_location('ccnc_map_fast_replay', path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def run_route(job):
  candidate, cache_dir, output, names, hybrid, timing, maps = job
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
    durations, peak = [], 0
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
    result.append(dict(name=name, peaks=dict(objects=peak), timing=dict(
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
  parser.add_argument('--jobs', type=int, default=8)
  parser.add_argument('--maps', action='store_true', help='Store the full map snapshot per send (large)')
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
  jobs = [(str(args.candidate), str(args.cache), str(args.output), group, args.hybrid, args.timing, args.maps) for group in groups]
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
