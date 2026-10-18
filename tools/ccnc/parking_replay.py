"""Replay parking_scan.py series through the CCNC parking-lot lane hide state machine (offline).

Uses opendbc/car/hyundai/ccnc_parking.py itself (loaded by file path: the opendbc.car package needs capnp, which
cannot load from a non-ASCII checkout on Windows). Reports hide intervals per segment, entry delay on the parking
logs, any hide on public-road logs, lane-change guard activity, and a spliced lot->road exit test.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np

STATES = {0: 'off', 1: 'preLaneChange', 2: 'laneChangeStarting', 3: 'laneChangeFinishing'}


def load_parking():
  path = Path(__file__).resolve().parents[2]/'opendbc_repo/opendbc/car/hyundai/ccnc_parking.py'
  spec = importlib.util.spec_from_file_location('ccnc_parking', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def run(module, d, t0=None):
  """d: parking_scan rows. Returns hidden flag per row and guard-active flag per row."""
  hide = module.ParkingLaneHide()
  hidden, guard = np.zeros(len(d), bool), np.zeros(len(d), bool)
  for i, r in enumerate(d):
    md = N(laneLineProbs=[r[13], r[7], r[8], r[14]], roadEdgeStds=[r[9], r[10]], meta=N(laneChangeState=STATES[int(r[11])]))
    hidden[i] = hide.update(r[0], r[1], md, bool(r[2]), bool(r[3]), int(r[12]))
    guard[i] = r[0] < hide.guard_until
  return hidden, guard


def intervals(t, flag):
  out, start = [], None
  for i, f in enumerate(flag):
    if f and start is None:
      start = i
    if (not f or i == len(flag) - 1) and start is not None:
      end = i if f else i - 1
      out.append((round(float(t[start] - t[0]), 2), round(float(t[end] - t[0]), 2)))
      start = None
  return out


def main(a):
  module = load_parking()
  positives = set(a.parking)
  report = {'thresholds': {k: getattr(module, k) for k in ('PROB_ON', 'EDGE_STD_ON', 'MIN_SPEED', 'MAX_SPEED', 'ENTRY_M', 'PROB_OFF',
                                                         'EXIT_M', 'EXIT_SPEED', 'EXIT_SPEED_S', 'GRACE_S', 'NO_MODEL_S')},
            'parking': {}, 'false_hides': [], 'public_segments': 0, 'public_hours': 0., 'guard': {}}
  series = {p.stem: np.load(p)['data'] for p in sorted(Path(a.scan).glob('*.npz'))}
  guard_low_speed = 0
  for seg, d in series.items():
    if len(d) < 2:
      continue
    hidden, guard = run(module, d)
    t = d[:, 0]
    if seg in positives:
      first = np.flatnonzero(hidden)
      report['parking'][seg] = {'duration_s': round(float(t[-1] - t[0]), 1), 'hidden_fraction': round(float(hidden.mean()), 3),
                                'entry_s': round(float(t[first[0]] - t[0]), 2) if len(first) else None,
                                'hidden_intervals_s': intervals(t, hidden), 'moving_fraction': round(float((d[:, 1] >= module.MIN_SPEED).mean()), 3)}
      continue
    report['public_segments'] += 1
    report['public_hours'] += float(t[-1] - t[0])/3600
    guard_low_speed += int((guard & (d[:, 1] < module.MAX_SPEED)).sum())
    if hidden.any():
      report['false_hides'].append({'segment': seg, 'hidden_intervals_s': intervals(t, hidden), 'hidden_s': round(float(hidden.mean()*(t[-1] - t[0])), 2)})
  report['guard']['public_low_speed_frames_with_guard'] = guard_low_speed
  report['public_hours'] = round(report['public_hours'], 2)
  # Splice: parking log, then a public-road segment continuing 50 ms later (artificial but continuous in time).
  lot, road = series[a.splice[0]], series[a.splice[1]].copy()
  road[:, 0] = road[:, 0] - road[0, 0] + lot[-1, 0] + .05
  d = np.concatenate((lot, road))
  hidden, _ = run(module, d)
  t = d[:, 0]
  boundary = len(lot)
  release = np.flatnonzero(~hidden[boundary:])
  report['splice_exit'] = {'lot': a.splice[0], 'road': a.splice[1], 'hidden_at_boundary': bool(hidden[boundary - 1]),
                           'release_after_s': round(float(t[boundary + release[0]] - t[boundary]), 2) if len(release) else None,
                           'road_speed_at_release': round(float(d[boundary + release[0], 1]), 2) if len(release) else None,
                           'rehidden_after_release': bool(hidden[boundary + release[0]:].any()) if len(release) else None}
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
  print(json.dumps(report, indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--scan', required=True)
  p.add_argument('--parking', nargs='+', required=True)
  p.add_argument('--splice', nargs=2, required=True)
  p.add_argument('--output', required=True)
  main(p.parse_args())
