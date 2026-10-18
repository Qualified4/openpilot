"""Display-level evaluation on VALIDATION (evaluation only; models and code unchanged).

Each boundary's 5 Hz frames are replayed in time through a reimplementation of the CCNC preview
DisplayPolicy rules (openpilot/selfdrive/carrot/lane_type/display.py): per head, a value is confirmed by two
identical results within 0.4 s, a confirmed value is refreshed by the same value, and expires 0.5 s after its
last evidence; yellow renders orange solid, otherwise white dashed / white solid; anything else is hidden.
Metrics: display rate, display accuracy, wrong-display episodes (count, total, longest, >=1 s), False-Dashed
display episodes (white dashed shown on solid truth), and dashed<->solid display flips while the truth did
not change (flicker per displayed minute).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.lane_profile.compare_models import SCAN_C, SCAN_T  # noqa: TID251
from tools.lane_profile.dataset_io import COLORS, TYPES, load_set  # noqa: TID251

FPS, STEP = 20, 4
REPEAT_S, HOLD_S, BREAK_S = .4, .5, 1.
THRESHOLDS = (.8, .85, .9, .95, .99)
SUSTAINED_S = 1.


def semantic(kind, color):
  if kind in ('dashed', 'solid') and color in ('white', 'yellow'):
    return 'orange_solid' if color == 'yellow' else 'white_' + kind
  return 'hidden'


def simulate(decisions, times):
  """decisions: list of (type|None, colour|None) per frame; returns displayed semantic per frame."""
  heads, pending, shown = {}, {}, []
  for (kind, color), t in zip(decisions, times, strict=True):
    for head, value in (('type', kind), ('color', color)):
      old = heads.get(head)
      if old and t - old[1] > HOLD_S:
        heads.pop(head)
        old = None
      if value is not None:
        if old and old[0] == value:
          heads[head] = (value, t)
          pending.pop(head, None)
        else:
          p = pending.get(head)
          if p and p[0] == value and p[1] < t <= p[1] + REPEAT_S:
            heads[head] = (value, t)
            pending.pop(head, None)
          else:
            pending[head] = (value, t)
      p = pending.get(head)
      if p and t - p[1] > REPEAT_S:
        pending.pop(head)
    shown.append(semantic(heads.get('type', (None,))[0], heads.get('color', (None,))[0]))
  return shown


def sequences(rows):
  seqs = {}
  for i, r in enumerate(rows):
    seqs.setdefault((r['route'], r['segment'], r['quality'], r['side']), []).append(i)
  out = []
  for idx in seqs.values():
    idx.sort(key=lambda i: rows[i]['video_index'])
    cur = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:], strict=True):
      if (rows[b]['video_index'] - rows[a]['video_index'])/FPS > BREAK_S:
        out.append(cur)
        cur = []
      cur.append(b)
    out.append(cur)
  return out


def evaluate(decide, rows):
  """decide(i) -> (type|None, colour|None) for frame i."""
  frames = shown_n = correct = 0
  wrong_eps, fd_eps, flips, shown_seconds = [], [], 0, 0.
  for seq in sequences(rows):
    times = [rows[i]['video_index']/FPS for i in seq]
    shown = simulate([decide(i) for i in seq], times)
    truth = [semantic(rows[i]['type'], rows[i]['color']) for i in seq]
    run, fd_run, last_visible = 0, 0, None
    for k, (s, g) in enumerate(zip(shown, truth, strict=True)):
      frames += 1
      dt = STEP/FPS
      wrong = s != 'hidden' and s != g
      if s != 'hidden':
        shown_n += 1
        shown_seconds += dt
        correct += s == g
        if last_visible and last_visible[0] != s and {last_visible[0], s} == {'white_dashed', 'white_solid'} and last_visible[1] == g:
          flips += 1
        last_visible = (s, g)
      fd = s == 'white_dashed' and g == 'white_solid'
      last = k == len(shown) - 1
      run, fd_run = run + wrong, fd_run + fd
      if run and (not wrong or last):  # an episode ends on the first correct/hidden frame or at the sequence end
        wrong_eps.append(run*dt)
        run = 0
      if fd_run and (not fd or last):
        fd_eps.append(fd_run*dt)
        fd_run = 0
  return {'frames': frames, 'display_rate': shown_n/frames, 'display_accuracy': correct/shown_n if shown_n else None,
          'wrong_episodes': len(wrong_eps), 'wrong_seconds': round(sum(wrong_eps), 2), 'longest_wrong_s': max(wrong_eps, default=0.),
          'sustained_wrong_episodes': sum(e >= SUSTAINED_S for e in wrong_eps),
          'false_dashed_episodes': len(fd_eps), 'false_dashed_seconds': round(sum(fd_eps), 2), 'longest_false_dashed_s': max(fd_eps, default=0.),
          'sustained_false_dashed_episodes': sum(e >= SUSTAINED_S for e in fd_eps),
          'flips': flips, 'flips_per_displayed_minute': flips/(shown_seconds/60) if shown_seconds else None,
          'displayed_minutes': round(shown_seconds/60, 2)}


def model_decider(pt, pc, th_dashed, th_solid, th_color):
  def decide(i):
    kind = 'dashed' if pt[i, 0] >= th_dashed else 'solid' if pt[i, 1] >= th_solid else None
    color = COLORS[pc[i].argmax()] if pc[i].max() >= th_color else None
    return kind, color
  return decide


def main(a):
  rows, x = load_set(a.data)
  val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  vr = [rows[i] for i in val]
  runs = {p.name[:-8]: np.load(p) for p in sorted(Path(a.runs).glob('*-val.npz'))}
  net = cv2.dnn.readNetFromONNX(str(a.baseline_onnx))
  pts, pcs = [], []
  xv = x[val]
  for i in range(0, len(xv), 64):
    net.setInput(np.ascontiguousarray(xv[i:i + 64].transpose(0, 3, 1, 2).astype(np.float32)))
    p, q = net.forward(['line_type', 'color'])
    pts.append(p)
    pcs.append(q)
  runs['stripnet-frozen-baseline-0'] = {'pt': np.concatenate(pts), 'pc': np.concatenate(pcs)}
  settings = [(f'{t}', t, t, t) for t in THRESHOLDS] + [('asym d.99/s.80/c.90', .99, .8, .9)]
  report = {'policy': __doc__.split('\n\n')[1], 'runs': {}}
  for name, z in runs.items():
    report['runs'][name] = {label: evaluate(model_decider(z['pt'], z['pc'], d, s, c), vr) for label, d, s, c in settings}
  scan = lambda i: (SCAN_T[vr[i]['scanner'][0]] if vr[i]['scanner'][0] else None,  # noqa: E731
                    SCAN_C[vr[i]['scanner'][1]] if vr[i]['scanner'][1] else None)
  report['scanner_frame'] = evaluate(scan, vr)
  # Truth-only upper bound: perfect per-frame decisions through the same policy.
  report['perfect_per_frame'] = evaluate(lambda i: (vr[i]['type'], vr[i]['color']), vr)
  assert all(t in TYPES for t in {r['type'] for r in vr})
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--runs', required=True)
  p.add_argument('--baseline-onnx', required=True)
  p.add_argument('--output', required=True)
  main(p.parse_args())
