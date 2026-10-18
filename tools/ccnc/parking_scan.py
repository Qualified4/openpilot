"""Extract the parking-lot lane-hide signals from rlogs (offline, read-only).

Per modelV2 message: time, vEgo/blinkers/gear from the latest carState, laneLineProbs[1,2], roadEdgeStds[0,1],
meta.laneChangeState and desire. One compressed npz per segment, used to choose and replay the CCNC parking-lot
lane-hide thresholds (opendbc/car/hyundai/ccnc_parking.py).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

LANE_CHANGE_STATES = {'off': 0, 'preLaneChange': 1, 'laneChangeStarting': 2, 'laneChangeFinishing': 3}


def scan(task):
  from tools.lane_type.retrain_data import read_events  # noqa: TID251
  log, out = task
  out = Path(out)
  if out.exists():
    return str(out)
  car = (0., False, False, 'unknown', False)
  rows = []
  for event in read_events(Path(log)):
    kind = event.which()
    if kind == 'carState':
      c = event.carState
      car = (float(c.vEgo), bool(c.leftBlinker), bool(c.rightBlinker), str(c.gearShifter), bool(c.standstill))
    elif kind == 'modelV2' and event.valid:
      m = event.modelV2
      probs, stds = list(m.laneLineProbs), list(m.roadEdgeStds)
      if len(probs) < 3 or len(stds) < 2:
        continue
      rows.append((event.logMonoTime/1e9, car[0], car[1], car[2], car[3] == 'drive', car[3] in ('park', 'reverse'), car[4],
                   probs[1], probs[2], stds[0], stds[1], LANE_CHANGE_STATES.get(str(m.meta.laneChangeState), 0), int(m.meta.desire.raw),
                   probs[0], probs[3]))
  a = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 15))
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out, data=a, columns=np.array(['t', 'v', 'lblink', 'rblink', 'drive', 'park_or_reverse', 'standstill',
                                                     'p_left', 'p_right', 'edge_std_left', 'edge_std_right', 'lc_state', 'desire',
                                                     'p_far_left', 'p_far_right']))
  return str(out)


def main(a):
  logs = sorted({p for root in a.roots for p in Path(root).rglob('*rlog.zst')}, key=lambda p: p.name)
  seen, tasks = set(), []
  for p in logs:
    seg = p.name.replace('--rlog.zst', '')
    if seg in seen:
      continue
    seen.add(seg)
    tasks.append((str(p), str(Path(a.output)/f'{seg}.npz')))
  print('segments', len(tasks), flush=True)
  with Pool(a.jobs) as pool:
    for i, path in enumerate(pool.imap_unordered(scan, tasks), 1):
      if i % 20 == 0:
        print(i, path, flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--roots', nargs='+', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--jobs', type=int, default=12)
  main(p.parse_args())
