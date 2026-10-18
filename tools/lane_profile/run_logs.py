"""Run the profile scanner over whole recorded segments (every exact-synced frame, both sides).

Read-only on logs/videos. Per segment and video quality it stores per-frame outputs, complete
dash measurements (paint groups bounded by verified road gaps) and, for segments that contain
reviewed frame labels, the active ONNX classifier's guarded/raw outputs for comparison.
"""
import argparse
from bisect import bisect_right
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
import time
from types import SimpleNamespace as N

import numpy as np

INVENTORY = Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json')
REVIEWED = [Path('.analysis/lane-model-comparison-20261018')/f for f in
            ('approved-test-long.json', 'approved-train-validation.json', 'approved-extra-train.json', 'approved-extra-validation.json')]
TYPE = {'unknown': 0, 'dashed': 1, 'solid': 2}
COLOR = {'unknown': 0, 'white': 1, 'yellow': 2}


def paint_chroma(res):
  """Median near-paint chroma offset (dCb, dCr) of this frame, for colour threshold study."""
  ev = res.get('evidence')
  if ev is None:
    return None
  near = (ev['state'] == 1) & (res['geometry']['d'] <= 12.) & np.isfinite(ev['cb'])
  if near.sum() < 3:
    return None
  return [round(float(np.median(ev['cb'][near])), 1), round(float(np.median(ev['cr'][near])), 1), int(near.sum())]


def reviewed_segments():
  out = set()
  for path in REVIEWED:
    for r in json.loads(path.read_text(encoding='utf-8')):
      out.add((r['route'], int(r['segment']), r['video_source']))
  return out


def process(task):
  source, quality, output, solid_m, with_onnx = task
  import av
  import cv2
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from tools.lane_profile.scanner import ProfileScanner  # noqa: TID251
  from tools.lane_type.retrain_data import metadata  # noqa: TID251
  cv2.setNumThreads(1)
  path = Path(output)/'segments'/f"{source['id']}-{quality}.json.gz"
  if path.exists():
    return str(path)
  identity, models, calibrations, cars, cameras, encodes = metadata(source)
  ct, st = [c[0] for c in calibrations], [c[0] for c in cars]
  scanners = {side: ProfileScanner(side, solid_m=solid_m) for side in ('left', 'right')}
  onnx = None
  if with_onnx:
    from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
    from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard, visibility
    from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, nv12_strip
    onnx = (Classifier('onnx'), ObservationGuard())
  frames, dashes, times, stats = [], {}, [], {'decoded': 0, 'matched': 0, 'no_match': 0, 'stale': 0}
  odometer, last_eof = 0., None
  with av.open(source['videos'][quality]['path']) as container:
    stream = container.streams.video[0]
    stream.thread_count = 1
    for index, frame in enumerate(container.decode(video=0)):
      stats['decoded'] += 1
      key = encodes[quality].get(index)
      model = models.get(key) if key else None
      sensor = cameras.get(key) if key else None
      if model is None or sensor is None:
        stats['no_match'] += 1
        continue
      ci, si = bisect_right(ct, key[1]) - 1, bisect_right(st, key[1]) - 1
      if ci < 0 or si < 0 or key[1] - ct[ci] > 2_000_000_000 or key[1] - st[si] > 200_000_000:
        stats['stale'] += 1
        last_eof = None
        continue
      stats['matched'] += 1
      speed, blinker = cars[si][1], cars[si][2]
      if last_eof is not None:
        odometer += max(0., speed)*(key[1] - last_eof)/1e9
      last_eof = key[1]
      camera = DEVICE_CAMERAS[identity['device'], sensor].fcam
      k = camera.intrinsics.copy()
      k[0] *= frame.width/camera.width
      k[1] *= frame.height/camera.height
      h, w = frame.height, frame.width
      planar = frame.to_ndarray(format='yuv420p')
      start = time.perf_counter_ns()
      y_plane = planar[:h]
      chroma = planar[h:].reshape(-1)
      uv = np.stack((chroma[:h*w//4].reshape(h//2, w//2), chroma[h*w//4:].reshape(h//2, w//2)), -1)
      rpy = calibrations[ci][1]
      lane_change = model['lane_change'] or blinker
      row = {'eof': key[1], 'frame_id': key[0], 'index': index, 'odo': round(odometer, 3), 'v': round(speed, 2), 'lc': lane_change}
      for side, li in (('left', 1), ('right', 2)):
        res = scanners[side].update(y_plane, uv, model['lines'][li], model['probs'][li], rpy, k, key[1], odometer,
                                    leads=model['leads'], lane_change=lane_change)
        row[side] = [TYPE[res['type']], COLOR[res['color']], res['reason'], round(res['solid_run_m'], 2), len(res['gaps']),
                     int(res['reset']), res['paint_rows'], res['road_rows'], res["rows"], paint_chroma(res),
                     round(res.get('paint_since_gap_m', 0.), 2)]
        for dash in res.get('complete_dashes', ()):
          dashes[(side, scanners[side].generation, round(dash['start_m']))] = (dash['length_m'], round(speed, 1), key[1])
      times.append((time.perf_counter_ns() - start)/1e6)
      if onnx is not None:
        classifier, guard = onnx
        nv12 = np.empty((h*3//2, w), np.uint8)
        nv12[:h] = y_plane
        nv12[h:] = uv.reshape(h//2, w)
        lead_model = N(leadsV3=[N(**lead) for lead in model['leads']])
        for side, li in (('left', 1), ('right', 2)):
          line = N(**model['lines'][li])
          maps = lane_maps(line, model['probs'][li], rpy, k, w, h)
          if maps is None:
            row['onnx_' + side] = [0, 0, 0, 0]
            guard.previous.pop(side, None)
            continue
          clear, observation = visibility(maps, lead_model, line, rpy, k)
          strip = nv12_strip(nv12.ravel(), w, h, w, w*h, maps)
          strip[~clear] = 0
          result = classifier.infer([strip])[0]
          boundary = tuple(float(np.interp(x, line.x, line.y)) for x in (10, 30))
          g = guard.update(side, result, observation, boundary, now_ns=key[1], camera_ns=key[1], lane_change=lane_change)
          row['onnx_' + side] = [TYPE.get(g['type'], 0), COLOR.get(g['color'], 0), TYPE[result['typeCandidate']],
                                 COLOR[result['colorCandidate']]]
      frames.append(row)
  data = {'id': source['id'], 'route': source['route'], 'segment': int(source['segment']), 'split': source['split'],
          'quality': quality, 'identity': identity, 'stats': stats, 'solid_m': solid_m, 'frames': frames,
          'dashes': [{'side': k[0], 'generation': k[1], 'start_m': k[2], 'length_m': v[0], 'speed': v[1], 'eof': v[2]}
                     for k, v in dashes.items()],
          'timing_ms': {'mean': float(np.mean(times)) if times else None, 'p95': float(np.percentile(times, 95)) if times else None,
                        'p99': float(np.percentile(times, 99)) if times else None, 'frames': len(times)}}
  path.parent.mkdir(parents=True, exist_ok=True)
  with gzip.open(path, 'wt', encoding='utf-8') as f:
    json.dump(data, f, separators=(',', ':'))
  return str(path)


def main(output, jobs, solid_m, only_reviewed, limit, onnx_all=False):
  inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
  reviewed = reviewed_segments()
  tasks = []
  for source in inventory['sources']:
    for quality in ('fcamera', 'qcamera'):
      if quality not in source['videos']:
        continue
      labelled = (source['route'], int(source['segment']), quality) in reviewed
      if only_reviewed and not labelled:
        continue
      tasks.append((source, quality, str(output), solid_m, labelled or onnx_all))
  tasks = tasks[:limit] if limit else tasks
  print('tasks', len(tasks), flush=True)
  with Pool(jobs) as pool:
    for i, path in enumerate(pool.imap_unordered(process, tasks), 1):
      print(i, path, flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--jobs', type=int, default=8)
  p.add_argument('--solid-m', type=float, default=12.)
  p.add_argument('--only-reviewed', action='store_true')
  p.add_argument('--limit', type=int, default=0)
  p.add_argument('--onnx-all', action='store_true', help='run the active ONNX on every segment (auto-label agreement)')
  a = p.parse_args()
  main(a.output, a.jobs, a.solid_m, a.only_reviewed, a.limit, a.onnx_all)
