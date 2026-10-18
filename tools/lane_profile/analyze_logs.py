"""Analyse whole-log scanner runs: dash length distribution, reviewed-frame accuracy, stability, cost.

Solid/dashed can be recomputed for any solid threshold from the stored latest-solid length and gap
count, so the threshold is chosen from TRAIN-route dash measurements only and then applied to test.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from tools.lane_profile.run_logs import REVIEWED  # noqa: TID251

TYPES, COLORS = ('unknown', 'dashed', 'solid'), ('unknown', 'white', 'yellow')


def load(folder):
  for path in sorted((folder/'segments').glob('*.json.gz')):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
      yield json.load(f)


def kind(entry, solid_m):
  if entry[3] >= solid_m:
    return 'solid'
  stale = len(entry) > 10 and entry[10] >= solid_m
  return 'dashed' if entry[4] and not stale else 'unknown'


def unique_dashes(dashes):
  """Merge measurements of the same dash (same side/generation, overlapping start)."""
  out = []
  for d in sorted(dashes, key=lambda d: (d['side'], d['generation'], d['start_m'])):
    if out and out[-1]['side'] == d['side'] and out[-1]['generation'] == d['generation'] and abs(d['start_m'] - out[-1]['start_m']) <= 1.5:
      out[-1] = max(out[-1], d, key=lambda x: x['length_m'])
      continue
    out.append(d)
  return out


def percentiles(values):
  if not values:
    return None
  v = np.asarray(values)
  return {'count': len(v), **{f'p{q}': float(np.percentile(v, q)) for q in (1, 5, 25, 50, 75, 95, 99, 99.5)}, 'max': float(v.max())}


def main(folder, solid_m, output):
  segments = list(load(folder))
  labels = []
  for path in REVIEWED:
    labels.extend(json.loads(path.read_text(encoding='utf-8')))
  by_frame = {}
  for seg in segments:
    for f in seg['frames']:
      by_frame[(seg['route'], seg['segment'], seg['quality'], f['frame_id'])] = (seg, f)
  # 1. dash lengths, train routes only for the threshold
  dash = defaultdict(list)
  for seg in segments:
    for d in unique_dashes(seg['dashes']):
      dash[seg['split']].append(d['length_m'])
      dash[seg['quality']].append(d['length_m'])
  hist = {k: percentiles(v) for k, v in dash.items()}
  train = np.asarray(dash['train'])
  histogram = {f'{a:.1f}-{a+1:.1f}': int(((train >= a) & (train < a + 1)).sum()) for a in np.arange(0, 40)}
  # 2. reviewed frames
  rows = []
  missing = Counter()
  for lab in labels:
    key = (lab['route'], int(lab['segment']), lab['video_source'], int(lab['frame_id']))
    hit = by_frame.get(key)
    if hit is None:
      missing[lab['split']] += 1
      continue
    seg, f = hit
    side = lab['side']
    e = f[side]
    row = {'split': lab['split'], 'quality': lab['video_source'], 'truth_type': lab['type'], 'truth_color': lab['color'],
           'profile_type': kind(e, solid_m), 'profile_color': COLORS[e[1]], 'reason': e[2], 'speed': f['v'], 'lc': f['lc'],
           'id': lab['id'], 'chroma': e[9] if len(e) > 9 else None}
    if 'onnx_' + side in f:
      o = f['onnx_' + side]
      row.update(onnx_type=TYPES[o[0]], onnx_color=COLORS[o[1]], onnx_raw_type=TYPES[o[2]], onnx_raw_color=COLORS[o[3]])
    rows.append(row)
  evaluation = {}
  for split in ('train', 'validation', 'test'):
    for quality in ('all', 'fcamera', 'qcamera'):
      sel = [r for r in rows if r['split'] == split and (quality == 'all' or r['quality'] == quality)]
      if not sel:
        continue
      entry = {'labels': len(sel)}
      for method in ('profile', 'onnx', 'onnx_raw'):
        if method != 'profile' and any(method + '_type' not in r for r in sel):
          continue
        res = {}
        for head, classes in (('type', ('dashed', 'solid')), ('color', ('white', 'yellow'))):
          out = [(r['truth_' + head], r[f'{method}_{head}']) for r in sel]
          decided = [(t, p) for t, p in out if p in classes]
          res[head] = {'decided': len(decided), 'decided_rate': len(decided)/len(out), 'correct': sum(t == p for t, p in decided),
                       'accuracy_of_decided': sum(t == p for t, p in decided)/len(decided) if decided else None,
                       'correct_over_all': sum(t == p for t, p in decided)/len(out),
                       'confusion': dict(Counter(f'{t}->{p}' for t, p in out))}
        solid = [r for r in sel if r['truth_type'] == 'solid']
        res['false_dashed'] = f"{sum(r[method + '_type'] == 'dashed' for r in solid)}/{len(solid)}"
        yellow = [r for r in sel if r['truth_color'] == 'yellow']
        res['yellow_recall'] = f"{sum(r[method + '_color'] == 'yellow' for r in yellow)}/{len(yellow)}"
        entry[method] = res
      evaluation[f'{split}/{quality}'] = entry
  # 3. whole-log behaviour (no truth): decided fraction, flicker per minute
  behaviour = defaultdict(Counter)
  for seg in segments:
    for side in ('left', 'right'):
      prev = None
      for f in seg['frames']:
        t = kind(f[side], solid_m)
        behaviour[seg['quality']]['frames'] += 1
        behaviour[seg['quality']][t] += 1
        behaviour[seg['quality']]['color_' + COLORS[f[side][1]]] += 1
        if prev is not None and t != prev and 'unknown' not in (t, prev):
          behaviour[seg['quality']]['dashed_solid_switches'] += 1
        if t != 'unknown':
          prev = t
      behaviour[seg['quality']]['side_seconds'] += (seg['frames'][-1]['eof'] - seg['frames'][0]['eof'])/1e9 if seg['frames'] else 0
  timing = defaultdict(list)
  for seg in segments:
    if seg['timing_ms']['mean'] is not None:
      timing[seg['quality']].append((seg['timing_ms']['mean'], seg['timing_ms']['p95'], seg['timing_ms']['frames']))
  timing = {q: {'weighted_mean_ms': float(np.average([t[0] for t in v], weights=[t[2] for t in v])),
                'median_segment_p95_ms': float(np.median([t[1] for t in v])), 'frames': int(sum(t[2] for t in v))} for q, v in timing.items()}
  report = {'solid_m': solid_m, 'segments': len(segments), 'dash_length_m': hist, 'train_dash_histogram_1m': histogram,
            'reviewed_missing_frames': dict(missing), 'evaluation': evaluation, 'behaviour': {k: dict(v) for k, v in behaviour.items()},
            'timing_both_sides': timing, 'stats': dict(sum((Counter(s['stats']) for s in segments), Counter()))}
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=1), encoding='utf-8')
  (output.with_name(output.stem + '-frames.json')).write_text(json.dumps(rows), encoding='utf-8')
  return report


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--input', type=Path, required=True)
  p.add_argument('--solid-m', type=float, default=12.)
  p.add_argument('--output', type=Path, required=True)
  a = p.parse_args()
  r = main(a.input, a.solid_m, a.output)
  print(json.dumps({k: r[k] for k in ('dash_length_m', 'train_dash_histogram_1m', 'reviewed_missing_frames', 'timing_both_sides')}, indent=1))
  for k, v in r['evaluation'].items():
    for m in ('profile', 'onnx', 'onnx_raw'):
      if m in v:
        print(k, m, v['labels'], 'type', v[m]['type']['decided'], v[m]['type']['correct'], 'color', v[m]['color']['decided'], v[m]['color']['correct'],
              'FD', v[m]['false_dashed'], 'Y', v[m]['yellow_recall'])
