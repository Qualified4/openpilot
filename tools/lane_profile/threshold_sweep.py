"""Confidence-threshold sweep on VALIDATION for already-trained runs (no retraining, no test).

Per run: decided rate / decided accuracy / class recall for type and colour, False Dashed (dashed decided
on solid truth), at each threshold; plus the display-style rule "same decision on two consecutive 5 Hz
frames of the same boundary" (DisplayPolicy requires two fresh identical results) and the intervals the
False Dashed frames come from. Reference: the frozen StripNet baseline ONNX (OpenCV DNN) and the scanner.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from tools.lane_profile.compare_models import SCAN_C, SCAN_T, decisions, score  # noqa: TID251
from tools.lane_profile.dataset_io import load_set  # noqa: TID251

THRESHOLDS = (.8, .85, .9, .95, .99)


def consecutive(t, c, rows):
  """Keep a decision only if the previous 5 Hz frame of the same boundary made the same one."""
  t2, c2 = np.full_like(t, -1), np.full_like(c, -1)
  key = [(r['route'], r['segment'], r['quality'], r['side'], r['video_index']) for r in rows]
  index = {k: i for i, k in enumerate(key)}
  for i, k in enumerate(key):
    j = index.get(k[:4] + (k[4] - 4,))
    if j is not None:
      if t[i] >= 0 and t[j] == t[i]:
        t2[i] = t[i]
      if c[i] >= 0 and c[j] == c[i]:
        c2[i] = c[i]
  return t2, c2


def sweep(pt, pc, rows):
  out = {}
  for th in THRESHOLDS:
    t, c = decisions(pt, pc, th)
    s = score(t, c, rows)
    fd = [rows[i]['label_ref'] for i in np.flatnonzero((t == 0) & np.array([r['type'] == 'solid' for r in rows]))]
    s['false_dashed_sources'] = dict(Counter(fd))
    s['two_frame'] = score(*consecutive(t, c, rows), rows)
    s['by_quality'] = {q: score(t[idx], c[idx], [rows[i] for i in idx]) for q in ('fcamera', 'qcamera')
                       for idx in [[i for i, r in enumerate(rows) if r['quality'] == q]] if idx}
    out[str(th)] = s
  return out


def main(a):
  rows, x = load_set(a.data)
  val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  vrows, xv = [rows[i] for i in val], x[val]
  report = {'thresholds': THRESHOLDS, 'runs': {}}
  for path in sorted(Path(a.runs).glob('*-val.npz')):
    z = np.load(path)
    report['runs'][path.name[:-8]] = sweep(z['pt'], z['pc'], vrows)
  net = cv2.dnn.readNetFromONNX(str(a.baseline_onnx))
  pts, pcs = [], []
  for i in range(0, len(xv), 64):
    net.setInput(np.ascontiguousarray(xv[i:i + 64].transpose(0, 3, 1, 2).astype(np.float32)))
    p, q = net.forward(['line_type', 'color'])
    pts.append(p)
    pcs.append(q)
  report['runs']['stripnet-frozen-baseline'] = sweep(np.concatenate(pts), np.concatenate(pcs), vrows)
  st = np.array([{'dashed': 0, 'solid': 1}.get(SCAN_T[r['scanner'][0]], -1) for r in vrows])
  sc = np.array([{'white': 0, 'yellow': 1}.get(SCAN_C[r['scanner'][1]], -1) for r in vrows])
  report['scanner_frame'] = score(st, sc, vrows)
  report['scanner_frame']['two_frame'] = score(*consecutive(st, sc, vrows), vrows)
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--runs', required=True)
  p.add_argument('--baseline-onnx', required=True)
  p.add_argument('--output', required=True)
  main(p.parse_args())
