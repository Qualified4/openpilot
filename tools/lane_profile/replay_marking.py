"""Replay recorded segments through the runtime lane marking path (PC, offline; no CAN is sent).

Same path as the device: exact frameId/EOF model-frame sync -> marking_daemon.Recogniser (Scanner or StripNet)
-> display.DisplayPolicy.confirm -> marking_daemon.packet -> marking.parse (card) -> ccnc_lane_marking.lane_code
(CCNC sender) on the existing lane code (green 6 when modelV2 laneChangeAvailable, else white 2).
Both modes see the same decoded frames. Per-frame wall/thread-CPU time is measured on this PC; a real-time
schedule (20 Hz input, newest-input-wins while busy) is simulated from those times, optionally scaled.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.carrot.lane_type import marking
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from openpilot.selfdrive.carrot.lane_type.marking_daemon import Recogniser, packet
from tools.lane_profile.labels import load as load_labels  # noqa: TID251
from tools.lane_profile.run_logs import INVENTORY  # noqa: TID251
from tools.lane_type.retrain_data import metadata  # noqa: TID251


def _load_lane_code():
  """The CCNC sender's mapping, loaded by file: importing the opendbc.car package pulls in capnp, which
  cannot load from this PC's non-ASCII checkout path. The module itself has no imports."""
  import importlib.util
  path = Path(__file__).resolve().parents[2]/'opendbc_repo/opendbc/car/hyundai/ccnc_lane_marking.py'
  spec = importlib.util.spec_from_file_location('ccnc_lane_marking', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.lane_code


lane_code = _load_lane_code()


def replay(source, quality, modes):
  import av
  import cv2
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  cv2.setNumThreads(1)
  identity, models, calibrations, cars, cameras, encodes = metadata(source)
  ct, st = [c[0] for c in calibrations], [c[0] for c in cars]
  engines = {m: (Recogniser(m), DisplayPolicy()) for m in modes}
  rows, odometer, last = [], 0., None
  with av.open(source['videos'][quality]['path']) as container:
    for index, frame in enumerate(container.decode(video=0)):
      key = encodes[quality].get(index)
      model = models.get(key) if key else None
      sensor = cameras.get(key) if key else None
      if model is None or sensor is None:
        continue
      ci, si = bisect_right(ct, key[1]) - 1, bisect_right(st, key[1]) - 1
      if ci < 0 or si < 0 or key[1] - ct[ci] > 2e9 or key[1] - st[si] > 2e8:
        last = None
        continue
      speed, blinker = cars[si][1], cars[si][2]
      if last is not None:
        odometer += max(0., speed)*(key[1] - last)/1e9
      last = key[1]
      cam = DEVICE_CAMERAS[identity['device'], sensor].fcam
      k = cam.intrinsics.copy()
      k[0] *= frame.width/cam.width
      k[1] *= frame.height/cam.height
      h, w = frame.height, frame.width
      planar = frame.to_ndarray(format='yuv420p')
      c = planar[h:].reshape(-1)
      uv = np.stack((c[:h*w//4].reshape(h//2, w//2), c[h*w//4:].reshape(h//2, w//2)), -1)
      f = {'lines': {'left': model['lines'][1], 'right': model['lines'][2]},
           'probs': {'left': model['probs'][1], 'right': model['probs'][2]}, 'rpy': calibrations[ci][1], 'k': k,
           'stamp': key[1], 'odometer': odometer, 'leads': model['leads'], 'lane_change': model['lane_change'] or blinker}
      row = {'index': index, 'eof': key[1]}
      for mode, (rec, policy) in engines.items():
        w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        raw = rec.process(planar[:h], uv, f)
        conf = policy.confirm({'cameraMonoTimeNanos': key[1], **raw}, key[1])
        wall, cpu = (time.perf_counter_ns() - w0)/1e6, (time.thread_time_ns() - c0)/1e6
        pub = packet(mode, key[0], key[1], raw, conf, key[1], {}, {})
        parsed = marking.parse(json.dumps(pub).encode(), key[1])
        row[mode] = {'wall_ms': wall, 'cpu_ms': cpu, 'shown': {s: marking.current(parsed, s, key[1]) for s in ('left', 'right')},
                     'parsed': parsed}
      rows.append(row)
  return rows, identity


def lane_change_available(source):
  """modelV2 laneChangeAvailableLeft/Right per (frameId, EOF) from the log."""
  from tools.lane_type.retrain_data import read_events  # noqa: TID251
  out = {}
  for event in read_events(Path(source['log']['path'])):
    if event.which() == 'modelV2' and event.valid:
      m = event.modelV2
      out[m.timestampEof] = (bool(m.meta.laneChangeAvailableLeft), bool(m.meta.laneChangeAvailableRight),
                             (event.logMonoTime - m.timestampEof)/1e6)
  return out


def schedule(times_ms, period_ms=50., scale=1.):
  """Real-time simulation: a new input every period; while the worker is busy only the newest input is kept
  (conflated VisionIPC + latest modelV2), and it starts as soon as the worker is free."""
  free_at, pending, latencies, finishes = 0., None, [], []

  def run(item):
    nonlocal free_at
    arrival, cost = item
    free_at = max(free_at, arrival) + cost
    latencies.append(free_at - arrival)
    finishes.append(free_at)
  for i, t in enumerate(times_ms):
    arrival = i*period_ms
    if pending is not None and free_at <= arrival:
      run(pending)
      pending = None
    pending = (arrival, t*scale)
    if free_at <= arrival:
      run(pending)
      pending = None
  n = len(times_ms)
  gaps = np.diff(finishes)
  return {'inputs': n, 'processed': len(finishes), 'processed_fraction': len(finishes)/n if n else None,
          'update_hz': float(1000./gaps.mean()) if len(gaps) else None,
          'latency_mean_ms': float(np.mean(latencies)) if latencies else None,
          'latency_p95_ms': float(np.percentile(latencies, 95)) if latencies else None, 'scale': scale}


def main(a):
  inventory = {s['id']: s for s in json.loads(INVENTORY.read_text(encoding='utf-8'))['sources']}
  human = [i for i in load_labels(a.human) if i['status'] == 'human_confirmed']
  segments = [tuple(s.split(':')) for s in a.segments]
  modes = (1, 2)
  report = {'segments': [], 'modes': {}}
  per_mode = {m: defaultdict(list) for m in modes}
  scenario = {m: Counter() for m in modes}
  truth_eval = {m: Counter() for m in modes}
  model_lag = []
  per_segment = defaultdict(Counter)
  for seg_id, quality in segments:
    source = inventory[seg_id]
    rows, identity = replay(source, quality, modes)
    lca = lane_change_available(source)
    labels = [h for h in human if f"{h['route']}--{h['segment']}" == seg_id]
    for r in rows:
      avail = lca.get(r['eof'], (False, False, None))
      if avail[2] is not None:
        model_lag.append(avail[2])
      for m in modes:
        x = r[m]
        per_mode[m]['wall'].append(x['wall_ms'])
        per_mode[m]['cpu'].append(x['cpu_ms'])
        for side, ok in (('left', avail[0]), ('right', avail[1])):
          existing = 6 if ok else 2
          code = lane_code(existing, x['parsed'], side, r['eof'])
          kind, color = x['shown'][side]
          scenario[m][f'existing{existing}->code{code}'] += 1
          if color == 'yellow' and existing == 6:
            scenario[m]['yellow_over_green'] += code == 4
          if code != existing:
            scenario[m]['changed'] += 1
          lab = [h for h in labels if h['side'] == side and h['start_eof_ns'] <= r['eof'] <= h['end_eof_ns'] and h['type'] in ('dashed', 'solid')]
          if lab:
            t = lab[0]
            want = 4 if t['color'] == 'yellow' else (10 if t['type'] == 'dashed' else 6) if existing == 6 else (8 if t['type'] == 'dashed' else 2)
            truth_eval[m]['labelled'] += 1
            truth_eval[m]['changed'] += code != existing
            truth_eval[m]['correct_when_changed'] += code != existing and code == want
            truth_eval[m]['wrong_when_changed'] += code != existing and code != want
            truth_eval[m]['false_dashed'] += t['type'] == 'solid' and code in (8, 10)
            per_segment[(seg_id, m)]['labelled'] += 1
            per_segment[(seg_id, m)]['wrong_when_changed'] += code != existing and code != want
            per_segment[(seg_id, m)]['false_dashed'] += t['type'] == 'solid' and code in (8, 10)
    report['segments'].append({'segment': seg_id, 'quality': quality, 'frames': len(rows), 'device': identity.get('device')})
    print(seg_id, quality, len(rows), flush=True)
  for m in modes:
    w, c = np.array(per_mode[m]['wall']), np.array(per_mode[m]['cpu'])
    report['modes'][marking.MODES[m]] = {
      'frames': len(w), 'wall_ms': {'mean': float(w.mean()), 'p95': float(np.percentile(w, 95)), 'p99': float(np.percentile(w, 99))},
      'thread_cpu_ms_mean': float(c.mean()), 'cpu_one_core_percent_at_20hz': float(c.mean()/50*100),
      'realtime_pc': schedule(list(w)), 'realtime_scaled_x3_estimate': schedule(list(w), scale=3.),
      'realtime_scaled_x6_estimate': schedule(list(w), scale=6.),
      'display_codes': dict(scenario[m]), 'human_labelled_frames': dict(truth_eval[m])}
  report['human_by_segment'] = {f'{k[0]}|{marking.MODES[k[1]]}': dict(v) for k, v in per_segment.items()}
  report['camera_to_modelV2_ms_from_log'] = {'mean': float(np.mean(model_lag)), 'p95': float(np.percentile(model_lag, 95))}
  Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
  print(json.dumps(report['modes'], indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--segments', nargs='+', required=True, help='route--segment:quality')
  p.add_argument('--human', required=True)
  p.add_argument('--output', required=True)
  main(p.parse_args())
