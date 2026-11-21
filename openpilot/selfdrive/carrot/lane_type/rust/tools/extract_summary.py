#!/usr/bin/env python3
"""Summarise combo_bench.sh output: per engine, extract / infer / whole (median, p95, p99 in ms over all repeats, the
first 20 modelV2 of each segment dropped), the median of each repeat, the Rust extraction stages (median ms, summed
over both sides), map cache hits, peak RSS and CPU per modelV2, and the parity run's strip agreement with py.
Usage: extract_summary.py OUT_DIR (holding parity/ and time1..N/)."""
import json
import sys
from pathlib import Path

import numpy as np


def pct(v):
  v = np.asarray(v, float)
  return [round(float(np.median(v)), 2), round(float(np.percentile(v, 95)), 2), round(float(np.percentile(v, 99)), 2)] if len(v) else None


def main(root):
  root = Path(root)
  out = {}
  runs = sorted(p for p in root.iterdir() if p.name.startswith('time') and p.is_dir())
  per = {}
  for run in runs:
    rows = json.loads((run/'rows.json').read_text())
    summ = json.loads((run/'summary.json').read_text())['engines']
    seen = {}
    for r in rows:
      seen[r['segment']] = seen.get(r['segment'], 0) + 1
      if seen[r['segment']] <= 20:
        continue
      for eng in summ:
        if eng in r:
          d = per.setdefault(eng, {'extract': [], 'infer': [], 'wall': [], 'stages': [], 'runs': {}, 'hits': 0, 'strips': 0})
          for k, v in (('minflt', r[eng].get('minflt')), ('wait', r[eng].get('wait')),
                       ('inferSideMax', max(r[eng]['inferSide']) if r[eng].get('inferSide') else None)):
            if v is not None:
              d.setdefault(k, []).append(v)
          d['hits'] += r[eng].get('cacheHits') or 0
          d['strips'] += 1  # modelV2 published
          for k in ('extract', 'infer', 'wall'):
            d[k].append(r[eng][k])
          if r[eng].get('stages'):
            d['stages'].append(r[eng]['stages'])
          d['runs'].setdefault(run.name, []).append(r[eng]['extract'])
          d.setdefault('iruns', {}).setdefault(run.name, []).append(r[eng]['infer'])
          d.setdefault('wruns', {}).setdefault(run.name, []).append(r[eng]['wall'])
    for eng, s in summ.items():
      per.setdefault(eng, {'extract': [], 'infer': [], 'wall': [], 'stages': [], 'runs': {}, 'hits': 0, 'strips': 0}).setdefault('stat', []).append(
        {'maxRssMiB': round(s['maxRssMiB'], 1), 'cpuMsPerModel': round(s['cpuS']*1000/len(rows), 2)})
  for eng, d in per.items():
    res = {'n': len(d['extract'])}
    for k in ('extract', 'infer', 'wall'):
      res[k] = pct(d[k])
    for k in ('wait', 'inferSideMax'):
      if d.get(k):
        res[k] = pct(d[k])
    if d.get('minflt'):
      f = np.asarray(d['minflt'], float)
      res['minfltPerModel'] = {'mean': round(float(f.mean()), 1), 'median': float(np.median(f)), 'p95': float(np.percentile(f, 95)),
                               'p99': float(np.percentile(f, 99))}
    res['inferMedianPerRun'] = {k: round(float(np.median(v)), 2) for k, v in d.get('iruns', {}).items()}
    res['extractMedianPerRun'] = {k: round(float(np.median(v)), 2) for k, v in d['runs'].items()}
    res['wallMedianPerRun'] = {k: round(float(np.median(v)), 2) for k, v in d.get('wruns', {}).items()}
    res['mapCacheHitsPerModels'] = f"{d['hits']} hits / {d['strips']} modelV2" 
    if d['stages']:
      st = np.asarray(d['stages'])
      res['stagesMedianMs'] = dict(zip(('rows', 'maps', 'remap', 'reduce'), (round(float(x), 3) for x in np.median(st, 0))))
    res['process'] = d.get('stat')
    out[eng] = res
  par = root/'parity'
  if par.exists():
    out['parity_totals'] = json.loads((par/'summary.json').read_text())['totals']
    rows = json.loads((par/'rows.json').read_text())
    agree = {}
    for r in rows:
      if 'py' not in r:
        continue
      for eng in [k for k in r if isinstance(r[k], dict) and k != 'py']:
        if r[eng].get('strips') is not None:
          a = agree.setdefault(eng, [0, 0])
          for x, y in zip(r[eng]['strips'], r['py']['strips']):
            a[0] += 1
            a[1] += x == y
    out['parity_strips_vs_py'] = {k: f'{m}/{n}' for k, (n, m) in agree.items()}
  print(json.dumps(out, indent=1))


if __name__ == '__main__':
  main(sys.argv[1])
