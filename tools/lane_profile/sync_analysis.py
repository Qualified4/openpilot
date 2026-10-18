"""Camera-frame / modelV2 matching study for the ccnc_lane_type daemon, from recorded publish times (offline).

Each rlog gives roadCameraState (frameId, timestampEof, logMonoTime ~ when the VisionIPC frame is available) and
modelV2 (frameId, timestampEof = EOF of the frame the model used, logMonoTime = when the daemon can see it).
The daemon is simulated as a single worker: when idle it takes the newest modelV2 (older ones that arrived while
busy are dropped, as SubMaster does), then picks a camera frame:
  exact  - the conflated newest frame only, accepted iff frameId and EOF equal the model's (previous daemon);
  buffer - the newest N frames (N = 2 or 3); exact frameId/EOF first, otherwise the frame with the nearest EOF
           within TOLERANCE of the model's timestampEof (the model's input frame time, not its receive time).
Processing time per handled model is drawn from the replay measurements, optionally scaled for a slower CPU.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def events(log):
  from tools.lane_type.retrain_data import read_events  # noqa: TID251
  cams, models = [], []
  for e in read_events(Path(log)):
    k = e.which()
    if k == 'roadCameraState':
      c = e.roadCameraState
      cams.append((e.logMonoTime, c.frameId, c.timestampEof))
    elif k == 'modelV2' and e.valid:
      m = e.modelV2
      models.append((e.logMonoTime, m.frameId, m.timestampEof))
  return sorted(cams), sorted(models)


def load(task):
  log, out = task
  out = Path(out)
  if not out.exists():
    cams, models = events(log)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cams=np.array(cams, np.int64).reshape(-1, 3), models=np.array(models, np.int64).reshape(-1, 3))
  return str(out)


def simulate(cams, models, method, n_buffer=3, tolerance_ns=50_000_000, cost_ms=lambda i: 10., rng=None):
  cam_t = cams[:, 0]
  free_at, i, handled, matched, errors, offsets, publishes = 0, 0, 0, 0, [], Counter(), []
  k = 0
  while k < len(models):
    t_arrive = models[k, 0]
    start = max(t_arrive, free_at)
    # newest model available at start time
    j = k
    while j + 1 < len(models) and models[j + 1, 0] <= start:
      j += 1
    k = j + 1
    _, fid, eof = models[j]
    handled += 1
    avail = bisect_right(cam_t, start)
    if avail == 0:
      continue
    if method == 'exact':
      candidates = cams[avail - 1:avail]
    else:
      candidates = cams[max(0, avail - n_buffer):avail]
    pick = None
    exact = [c for c in candidates if c[1] == fid and c[2] == eof]
    if exact:
      pick = exact[0]
    elif method != 'exact':
      d = np.abs(candidates[:, 2] - eof)
      if len(d) and d.min() <= tolerance_ns:
        pick = candidates[int(d.argmin())]
    if pick is None:
      continue
    matched += 1
    errors.append(abs(int(pick[2]) - int(eof))/1e6)
    offsets[int(pick[1]) - int(fid)] += 1
    cost = cost_ms(i) * 1e6
    i += 1
    free_at = start + int(cost)
    publishes.append(free_at)
  span = (models[-1, 0] - models[0, 0])/1e9 if len(models) > 1 else 0
  return {'models': len(models), 'handled': handled, 'matched': matched, 'span_s': span, 'errors_ms': errors,
          'offsets': dict(offsets), 'publish_hz': len(publishes)/span if span else None}


def main(a):
  logs = sorted({p for root in a.roots for p in Path(root).rglob('*rlog.zst')}, key=lambda p: p.name)
  seen, tasks = set(), []
  for p in logs:
    seg = p.name.replace('--rlog.zst', '')
    if seg not in seen:
      seen.add(seg)
      tasks.append((str(p), str(Path(a.cache)/f'{seg}.npz')))
  with Pool(a.jobs) as pool:
    files = list(pool.imap_unordered(load, tasks))
  replay = json.loads(Path(a.replay).read_text(encoding='utf-8')) if a.replay else None
  costs = {'scanner': (10.68, 17.87), 'stripnet': (5.05, 9.2)}
  if replay:
    costs = {m: (v['wall_ms']['mean'], v['wall_ms']['p95']) for m, v in replay['modes'].items()}
  rng = np.random.default_rng(0)
  # Basic timing facts from the logs
  lag, gaps_cam, gaps_model, missing = [], Counter(), Counter(), 0
  data = []
  for f in files:
    z = np.load(f)
    cams, models = z['cams'], z['models']
    if len(cams) < 2 or len(models) < 2:
      continue
    data.append((cams, models))
    by_fid = {int(c[1]): c for c in cams}
    for m in models:
      c = by_fid.get(int(m[1]))
      if c is None:
        missing += 1
        continue
      lag.append((int(m[0]) - int(c[0]))/1e6)
    for g in np.diff(cams[:, 1]):
      gaps_cam[int(g)] += 1
    for g in np.diff(models[:, 1]):
      gaps_model[int(g)] += 1
  report = {'segments': len(data), 'frames_model': int(sum(len(m) for _, m in data)),
            'model_after_camera_publish_ms': {q: float(np.percentile(lag, q)) for q in (5, 50, 95, 99)},
            'model_frame_without_camera_message': missing,
            'camera_frameid_steps': dict(sorted(gaps_cam.items())[:6]), 'model_frameid_steps': dict(sorted(gaps_model.items())[:6]),
            'methods': {}}
  for mode, (mean, p95) in costs.items():
    sigma = max(.1, (p95 - mean)/1.645)
    for scale in (1., 3., 6.):
      for method, n in (('exact', 1), ('buffer', 2), ('buffer', 3)):
        tot = Counter()
        errs, offs, hz = [], Counter(), []
        for cams, models in data:
          r = simulate(cams, models, method, n, a.tolerance_ms*1e6, lambda i, m=mean, s=sigma, k=scale: max(.5, rng.normal(m, s))*k)
          tot['models'] += r['models']
          tot['handled'] += r['handled']
          tot['matched'] += r['matched']
          errs += r['errors_ms']
          offs.update(r['offsets'])
          if r['publish_hz']:
            hz.append(r['publish_hz'])
        key = f'{mode}|x{scale:g}|{method}{n if method == "buffer" else ""}'
        report['methods'][key] = {'match_of_handled': tot['matched']/tot['handled'], 'match_of_models': tot['matched']/tot['models'],
                                  'publish_hz_median': float(np.median(hz)) if hz else 0., 'error_ms_p50': float(np.percentile(errs, 50)) if errs else None,
                                  'error_ms_p95': float(np.percentile(errs, 95)) if errs else None, 'error_ms_max': float(max(errs)) if errs else None,
                                  'exact_fraction_of_matches': offs.get(0, 0)/max(1, sum(offs.values())),
                                  'frame_offsets': dict(sorted(offs.items()))}
        print(key, json.dumps(report['methods'][key]), flush=True)
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
  print(json.dumps({k: v for k, v in report.items() if k != 'methods'}, indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--roots', nargs='+', required=True)
  p.add_argument('--cache', required=True)
  p.add_argument('--replay', default='.analysis/lane-marking-integration-20261018/replay.json')
  p.add_argument('--tolerance-ms', type=float, default=50.)
  p.add_argument('--output', required=True)
  p.add_argument('--jobs', type=int, default=12)
  main(p.parse_args())
