"""Same-frame comparison on the frozen 5 clips: profile scanner vs existing Pixel Ribbon modes vs active ONNX.

Truth comes from interval labels. Frames inside 'unknown' intervals are excluded from accuracy.
Existing modes are read from their frozen result files (not re-run); the ONNX model is run read-only.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace as N

import cv2
import numpy as np

from tools.lane_profile.labels import Lookup, load  # noqa: TID251
from tools.lane_profile.run_clips import load_rows, run  # noqa: TID251

HYBRID = Path('.analysis/pixel-ribbon-hybrid-20261018')
EXISTING = {'pixel_5hz_registered': '5hz-registered-sides2', 'pixel_20hz_registered': '20hz-registered-sides2',
            'pixel_hybrid_registered': 'hybrid-registered-sides2', 'pixel_5hz_legacy_left': '5hz-legacy-sides1',
            'pixel_20hz_legacy_left': '20hz-legacy-sides1'}


def existing(name):
  rows = json.loads((HYBRID/EXISTING[name]/'results.json').read_text(encoding='utf-8'))
  return {(r['clip'], r['side'], r['frame_id']): (r['type'], r['color']) for r in rows}


def onnx_outputs(rows, every):
  """Active PP-LCNet ONNX on 6-46m strips with the daemon's ObservationGuard (read-only use)."""
  from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
  from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard, visibility
  from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, nv12_strip
  classifier = Classifier('onnx')
  out, raw, times = {}, {}, []
  for clip in sorted({r['clip'] for r in rows}):
    guard = ObservationGuard()
    group = sorted((r for r in rows if r['clip'] == clip), key=lambda r: r['stamp'])
    for i, r in enumerate(group):
      if i % every:
        continue
      nv12 = np.load(r['nv12'])
      h, w = nv12.shape[0]*2//3, nv12.shape[1]
      start = time.perf_counter_ns()
      model = N(leadsV3=[N(prob=lead['prob'], x=lead['x'], y=lead['y']) for lead in r['leads']])
      for side, line, prob in (('left', r['line'], r['probability']), ('right', r['right_line'], r['right_probability'])):
        line_ns = N(**line)
        maps = lane_maps(line_ns, prob, r['rpy'], np.array(r['intrinsics']), w, h)
        key = (clip, side, r['frame_id'])
        if maps is None:
          out[key] = raw[key] = ('unknown', 'unknown')
          continue
        clear, observation = visibility(maps, model, line_ns, r['rpy'], np.array(r['intrinsics']))
        strip = nv12_strip(nv12.ravel(), w, h, w, w*h, maps)
        strip[~clear] = 0
        result = classifier.infer([strip])[0]
        raw[key] = (result['typeCandidate'], result['colorCandidate'])
        boundary = tuple(float(np.interp(x, line['x'], line['y'])) for x in (10, 30))
        g = guard.update(side, result, observation, boundary, now_ns=r['stamp'], camera_ns=r['stamp'], lane_change=r['lane_change'])
        out[key] = (g['type'] if g['type'] in ('dashed', 'solid') else 'unknown', g['color'] if g['color'] in ('white', 'yellow') else 'unknown')
      times.append((time.perf_counter_ns() - start)/1e6)
  return out, raw, times


def score(outputs, rows, lookup, every=1):
  frames = []
  for clip in sorted({r['clip'] for r in rows}):
    group = sorted((r for r in rows if r['clip'] == clip), key=lambda r: r['stamp'])
    for i, r in enumerate(group):
      if i % every:
        continue
      for side in ('left', 'right'):
        key = (clip, side, r['frame_id'])
        if key not in outputs:
          continue
        truth = lookup(r['route'], r['segment'], side, r['stamp'])
        frames.append({'clip': clip, 'side': side, 'index': i, 'truth': truth, 'out': outputs[key]})
  report = {}
  for head, pos, classes in (('type', 0, ('dashed', 'solid')), ('color', 1, ('white', 'yellow'))):
    known = [f for f in frames if f['truth'] and f['truth'][head] in classes]
    decided = [f for f in known if f['out'][pos] in classes]
    correct = [f for f in decided if f['out'][pos] == f['truth'][head]]
    per = {}
    for c in classes:
      t = [f for f in known if f['truth'][head] == c]
      p = [f for f in decided if f['out'][pos] == c]
      tp = sum(f['truth'][head] == c for f in p)
      per[c] = {'truth_frames': len(t), 'output_frames': len(p), 'recall': tp/len(t) if t else None,
                'precision': tp/len(p) if p else None}
    report[head] = {'labelled_frames': len(known), 'decided': len(decided), 'decided_rate': len(decided)/len(known) if known else None,
                    'accuracy_of_decided': len(correct)/len(decided) if decided else None,
                    'correct_over_labelled': len(correct)/len(known) if known else None,
                    'unknown_rate': 1 - len(decided)/len(known) if known else None, 'classes': per}
  solid = [f for f in frames if f['truth'] and f['truth']['type'] == 'solid']
  dashed = [f for f in frames if f['truth'] and f['truth']['type'] == 'dashed']
  report['false_dashed'] = {'frames': sum(f['out'][0] == 'dashed' for f in solid), 'truth_solid_frames': len(solid)}
  report['false_solid'] = {'frames': sum(f['out'][0] == 'solid' for f in dashed), 'truth_dashed_frames': len(dashed)}
  report['excluded_unknown_interval_outputs'] = dict(Counter(f['out'][0] for f in frames if f['truth'] and f['truth']['type'] == 'unknown'))
  report['per_clip_side'] = {}
  for f in frames:
    k = f"{f['clip'][-12:]}/{f['side']}"
    report['per_clip_side'].setdefault(k, Counter())[f"{f['out'][0]}/{f['out'][1]}"] += 1
  report['per_clip_side'] = {k: dict(v) for k, v in report['per_clip_side'].items()}
  return report


def stats(times):
  return {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95)), 'p99_ms': float(np.percentile(times, 99)),
          'frames': len(times)} if times else None


def main(output, labels, solid_m):
  cv2.setNumThreads(1)
  rows = load_rows()
  lookup = Lookup(load(labels))
  results = {}
  for every, name in ((1, 'profile_20hz'), (4, 'profile_5hz')):
    records, times = run(rows, every=every, solid_m=solid_m)
    outputs = {(r['clip'], r['side'], r['frame_id']): (r['type'], r['color']) for r in records}
    results[name] = {'score': score(outputs, rows, lookup, every), 'timing_both_sides': stats(times),
                     'resets': sum(r['reset'] for r in records)}
  for name in EXISTING:
    outputs = existing(name)
    perf = json.loads((HYBRID/EXISTING[name]/'performance.json').read_text(encoding='utf-8'))
    every = 4 if name.startswith('pixel_5hz') else 1
    results[name] = {'score': score(outputs, rows, lookup, every),
                     'timing_reported': {'sides': perf['sides'], 'total_wall_ms': perf['total_wall_ms']}}
  for every, name in ((1, 'onnx_20hz_guarded'), (4, 'onnx_5hz_guarded')):
    guarded, raw, times = onnx_outputs(rows, every)
    results[name] = {'score': score(guarded, rows, lookup, every), 'timing_both_sides': stats(times)}
    results[name.replace('guarded', 'raw_argmax')] = {'score': score(raw, rows, lookup, every)}
  output.mkdir(parents=True, exist_ok=True)
  (output/'clip-comparison.json').write_text(json.dumps(results, indent=1, default=float), encoding='utf-8')
  for name, value in results.items():
    s = value['score']
    t = value.get('timing_both_sides') or {}
    fd, fs = s['false_dashed'], s['false_solid']
    print(' '.join((f"{name:28s} type decided {s['type']['decided']}/{s['type']['labelled_frames']}",
                    f"acc {s['type']['accuracy_of_decided']} color decided {s['color']['decided']}/{s['color']['labelled_frames']}",
                    f"acc {s['color']['accuracy_of_decided']} FD {fd['frames']}/{fd['truth_solid_frames']}",
                    f"FS {fs['frames']}/{fs['truth_dashed_frames']} ms {t.get('mean_ms')}")))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--labels', type=Path, required=True)
  p.add_argument('--solid-m', type=float, default=12.)
  a = p.parse_args()
  main(a.output, a.labels, a.solid_m)
