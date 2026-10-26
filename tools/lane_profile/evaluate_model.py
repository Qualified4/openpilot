"""Evaluate the frozen strip model (OpenCV DNN, the device's inference path) against the 1-D scanner and the
active lane_type.onnx on the SAME frames: validation routes (all / human-only) and the held-out human test set.

Run once after training; nothing here feeds back into training. Also benchmarks single-thread CPU latency.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np

from tools.lane_profile.dataset_io import COLORS_V1 as COLORS, TYPES_V1 as TYPES, load_set  # noqa: TID251  (2-class heads)

SCAN_T, SCAN_C = ('unknown', 'dashed', 'solid'), ('unknown', 'white', 'yellow')


def run_model(net, x, batch=64):
  pt, pc = [], []
  for i in range(0, len(x), batch):
    net.setInput(np.ascontiguousarray(x[i:i + batch].transpose(0, 3, 1, 2).astype(np.float32)))
    t, c = net.forward(['line_type', 'color'])
    pt.append(t)
    pc.append(c)
  return np.concatenate(pt), np.concatenate(pc)


def model_outputs(pt, pc, threshold):
  out = []
  for t, c in zip(pt, pc, strict=True):
    ty = TYPES[t.argmax()] if t.max() >= threshold else 'unknown'
    co = COLORS[c.argmax()] if c.max() >= threshold else 'unknown'
    if ty == 'dashed' and co == 'yellow':
      ty = co = 'unknown'  # yellow dashed is not a learned class
    out.append((ty, co))
  return out


def score(pred, rows):
  res = {}
  for head, k, classes in (('type', 0, ('dashed', 'solid')), ('color', 1, ('white', 'yellow'))):
    truth = [r[head] for r in rows]
    p = [q[k] for q in pred]
    decided = [i for i, v in enumerate(p) if v in classes]
    res[head] = {'frames': len(rows), 'decided_rate': len(decided)/len(rows) if rows else None,
                 'accuracy_decided': float(np.mean([p[i] == truth[i] for i in decided])) if decided else None,
                 'correct_over_all': float(np.mean([p[i] == truth[i] for i in range(len(rows))])) if rows else None,
                 'recall': {c: (float(np.mean([p[i] == c for i in range(len(rows)) if truth[i] == c])) if c in truth else None) for c in classes},
                 'support': {c: truth.count(c) for c in classes}}
  solid = [i for i, r in enumerate(rows) if r['type'] == 'solid']
  res['false_dashed'] = f"{sum(pred[i][0] == 'dashed' for i in solid)}/{len(solid)}"
  res['per_interval'] = {}
  for i, r in enumerate(rows):
    e = res['per_interval'].setdefault(r['label_ref'], {'truth': f"{r['type']}/{r['color']}", 'n': 0, 'type_ok': 0, 'color_ok': 0, 'type_unknown': 0})
    e['n'] += 1
    e['type_ok'] += pred[i][0] == r['type']
    e['color_ok'] += pred[i][1] == r['color']
    e['type_unknown'] += pred[i][0] == 'unknown'
  return res


def onnx_lookup(logs, rows):
  wanted = defaultdict(set)
  for r in rows:
    wanted[(r['route'], r['segment'], r['quality'])].add(r['video_index'])
  found = {}
  for path in Path(logs, 'segments').glob('*.json.gz'):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
      seg = json.load(f)
    key = (seg['route'], seg['segment'], seg['quality'])
    if key not in wanted:
      continue
    for fr in seg['frames']:
      if fr['index'] in wanted[key]:
        for side in ('left', 'right'):
          found[key + (fr['index'], side)] = fr.get('onnx_' + side)
  return found


def baselines(rows, onnx):
  scanner = [(SCAN_T[r['scanner'][0]], SCAN_C[r['scanner'][1]]) for r in rows]  # per-frame codes stored at run time
  guarded, raw = [], []
  for r in rows:
    o = r.get('onnx_guarded') and (r['onnx_guarded'] + r['onnx_raw']) or onnx.get((r['route'], r['segment'], r['quality'], r['video_index'], r['side']))
    o = o or [0, 0, 0, 0]
    guarded.append((SCAN_T[o[0]], SCAN_C[o[1]]))
    raw.append((SCAN_T[o[2]], SCAN_C[o[3]]))
  return {'scanner_frame': scanner, 'active_onnx_guarded': guarded, 'active_onnx_raw': raw}


def benchmark(path, inputs, iters=500):
  net = cv2.dnn.readNetFromONNX(str(path))
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  names = net.getUnconnectedOutLayersNames()
  for _ in range(20):
    net.setInput(inputs)
    net.forward(names)
  times = []
  for _ in range(iters):
    s = time.perf_counter_ns()
    net.setInput(inputs)
    net.forward(names)
    times.append((time.perf_counter_ns() - s)/1e6)
  return {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95)), 'p99_ms': float(np.percentile(times, 99))}


def strip_cost(nv12_path, line, rpy, k, iters=200, half_m=.75, cols=96):
  from tools.lane_profile.scanner import nv12_planes, sample, strip_geometry  # noqa: TID251
  nv12 = np.load(nv12_path)
  h, w = nv12.shape[0]*2//3, nv12.shape[1]
  y, uv = nv12_planes(nv12, w, h)
  times = []
  for _ in range(iters):
    s = time.perf_counter_ns()
    geo = strip_geometry(line, rpy, k, w, h, half_m=half_m, cols=cols)
    lum, cb, cr = sample(y, uv, geo)
    cv2.resize(np.stack((lum, cb, cr), -1), (cols, 128), interpolation=cv2.INTER_AREA)
    times.append((time.perf_counter_ns() - s)/1e6)
  return {'mean_ms_per_side': float(np.mean(times)), 'p95_ms_per_side': float(np.percentile(times, 95))}


def main(a):
  cv2.setNumThreads(1)
  model_dir = Path(a.model)
  contract = json.loads((model_dir/'model-contract.json').read_text(encoding='utf-8'))
  threshold = contract['threshold']
  strip = contract.get('strip_geometry') or {'half_m': .75, 'cols': 96, 'rows': 128}  # train_strip.py; absent = v1-v3 width
  net = cv2.dnn.readNetFromONNX(str(model_dir/'strip_v3.onnx'))
  rows, x = load_set(a.data)
  val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  vh = [i for i in val if rows[i]['origin'] == 'human']
  test_rows, xt = load_set(a.test)
  onnx = onnx_lookup(a.logs, [rows[i] for i in val])
  report = {'threshold': threshold, 'yellow_dashed': 'reported as unknown (not learned)', 'sets': {}}
  for name, rs, xs in (('validation_all', [rows[i] for i in val], x[val]), ('validation_human', [rows[i] for i in vh], x[vh]),
                       ('test_human', test_rows, xt)):
    pt, pc = run_model(net, xs)
    methods = {'strip_model_threshold': model_outputs(pt, pc, threshold), 'strip_model_argmax': model_outputs(pt, pc, 0.),
               **baselines(rs, onnx)}
    report['sets'][name] = {m: score(p, rs) for m, p in methods.items()}
  frame = json.loads(Path('.analysis/pixel-ribbon-hybrid-20261018/frames.json').read_text(encoding='utf-8'))[0]
  report['cpu'] = {'opencv': cv2.__version__, 'threads': 1,
                   'strip_model_batch2_ms': benchmark(model_dir/'strip_v3.onnx', np.full((2, 3, strip['rows'], strip['cols']), 100., np.float32)),
                   'active_onnx_batch2_ms': benchmark(Path('openpilot/selfdrive/carrot/lane_type/assets/lane_type.onnx'),
                                                       np.zeros((2, 3, 192, 64), np.float32)),
                   'strip_extraction_fcamera': strip_cost(frame['nv12'], frame['line'], frame['rpy'], frame['intrinsics'],
                                                          half_m=strip['half_m'], cols=strip['cols']),
                   'model_bytes': (model_dir/'strip_v3.onnx').stat().st_size,
                   'active_onnx_bytes': Path('openpilot/selfdrive/carrot/lane_type/assets/lane_type.onnx').stat().st_size}
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
  for name, ms in report['sets'].items():
    for m, s in ms.items():
      ty, co = s['type'], s['color']
      print(f"{name:17s} {m:22s} type dec {ty['decided_rate']:.3f} acc {ty['accuracy_decided']} rec {ty['recall']} |",
            f"color dec {co['decided_rate']:.3f} acc {co['accuracy_decided']} rec {co['recall']} | FD {s['false_dashed']}")
  print(json.dumps(report['cpu'], indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--model', required=True)
  p.add_argument('--data', required=True)
  p.add_argument('--test', required=True)
  p.add_argument('--logs', required=True)
  p.add_argument('--output', required=True)
  main(p.parse_args())
