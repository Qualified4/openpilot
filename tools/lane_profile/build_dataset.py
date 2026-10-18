"""Training dataset from interval labels: human > auto > auto_pool > ai_reviewed, train/validation routes only.

1. Merge label sources with tools.lane_profile.labels.training_intervals (overlap-only replacement).
   Existing single-frame AI reviews are added only where no interval label covers the frame.
2. Per-frame manifest at 5 Hz (every 4th video frame) from whole-log scanner runs: frames inside a
   labelled interval whose boundary geometry is observable (no lane change, scanner path present,
   >= 60 strip rows, >= 30% rows paint/road). The scanner's own per-frame verdict is stored for
   analysis only; the label is the interval's.
3. --extract: decode each video once and store, per manifest frame, the row-aligned 4.5-20m strip
   (Y/Cb/Cr, 128 x 96, rows area-resampled) and the legacy 6-46m 192 x 64 RGB strip for a baseline.
No training, no runtime change. Test routes never enter the dataset.
"""
import argparse
from collections import Counter, defaultdict
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.lane_profile.analyze_logs import load  # noqa: TID251
from tools.lane_profile.labels import load as load_labels, training_intervals  # noqa: TID251
from tools.lane_profile.run_logs import INVENTORY, REVIEWED  # noqa: TID251

WEIGHT = {'human': 1., 'auto': 1., 'auto_pool': .7, 'ai_reviewed': .5, 'ai_frame_review': .5}
STRIDE, MIN_ROWS, MIN_KNOWN = 4, 60, .3
STRIP_ROWS = 128


def sources(args, split_of):
  items = []
  for path in (args.human, args.auto, args.pool, args.clips):
    if path:
      items.extend(load_labels(path))  # validates every interval against the label schema
  return items, training_intervals(items, split_of)


def main(args):
  inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
  split_of = inventory['route_splits'].get
  items, merged = sources(args, split_of)
  index = defaultdict(list)
  for m in merged:
    index[(m['route'], int(m['segment']), m['side'])].append(m)
  frame_reviews = defaultdict(dict)
  for path in REVIEWED:
    for r in json.loads(path.read_text(encoding='utf-8')):
      if split_of(r['route']) in ('train', 'validation') and r['type'] in ('dashed', 'solid'):
        frame_reviews[(r['route'], int(r['segment']), r['video_source'])][(int(r['video_index']), r['side'])] = r
  rows, skipped = [], Counter()
  for seg in load(Path(args.logs)):
    split = split_of(seg['route'])
    if split not in ('train', 'validation'):
      continue
    reviews = frame_reviews.get((seg['route'], seg['segment'], seg['quality']), {})
    for f in seg['frames']:
      for side in ('left', 'right'):
        hit = [m for m in index.get((seg['route'], seg['segment'], side), ()) if m['start_eof_ns'] <= f['eof'] <= m['end_eof_ns']]
        review = reviews.get((f['index'], side))
        if len(hit) > 1:
          skipped['overlapping_labels'] += 1
          continue
        if hit and f['index'] % STRIDE:
          continue
        if not hit and review is None:
          continue
        e = f[side]
        if f['lc']:
          skipped['lane_change'] += 1
          continue
        if e[2] in ('no_geometry', 'no_path', 'lane_change') or e[8] < MIN_ROWS or (e[6] + e[7]) < MIN_KNOWN*e[8]:
          skipped['boundary_not_observable'] += 1
          continue
        if hit:
          m = hit[0]
          label, origin, ref = (m['type'], m['color']), m['origin'], m.get('review_item_id') or m.get('auto_item_id') or m.get('clip')
        else:
          label, origin, ref = (review['type'], review['color']), 'ai_frame_review', review['id']
        rows.append({'route': seg['route'], 'segment': seg['segment'], 'quality': seg['quality'], 'split': split, 'side': side,
                     'video_index': f['index'], 'frame_id': f['frame_id'], 'eof': f['eof'], 'type': label[0], 'color': label[1],
                     'origin': origin, 'weight': WEIGHT[origin], 'label_ref': ref, 'speed': f['v'],
                     'scanner': [e[0], e[1]], 'scanner_rows': [e[6], e[7], e[8]]})
  out = Path(args.output)
  out.mkdir(parents=True, exist_ok=True)
  with open(out/'manifest.jsonl', 'w', encoding='utf-8') as fh:
    for r in rows:
      fh.write(json.dumps(r) + '\n')
  (out/'merged-intervals.json').write_text(json.dumps(merged, indent=1, default=str), encoding='utf-8')
  summary = {'label_sources_in': dict(Counter((i.get('origin') or ('human' if i['status'] == 'human_confirmed' else i['status'])) for i in items)),
             'merged_intervals_train_val': dict(Counter(m['origin'] for m in merged)),
             'clipped_by_higher_priority': sum(m['clipped'] for m in merged),
             'frames': len(rows), 'skipped': dict(skipped),
             'by_split_origin': {f'{k[0]}/{k[1]}': v for k, v in Counter((r['split'], r['origin']) for r in rows).items()},
             'by_split_class': {f'{k[0]}/{k[1]}/{k[2]}': v for k, v in Counter((r['split'], r['type'], r['color']) for r in rows).items()},
             'by_split_quality': {f'{k[0]}/{k[1]}': v for k, v in Counter((r['split'], r['quality']) for r in rows).items()},
             'routes': {s: sorted({r['route'] for r in rows if r['split'] == s}) for s in ('train', 'validation')},
             'scanner_frame_agreement_with_label': {o: round(float(np.mean([('unknown', 'dashed', 'solid')[r['scanner'][0]] == r['type']
                                                                             for r in rows if r['origin'] == o])), 3)
                                                    for o in sorted({r['origin'] for r in rows})},
             'test_routes_excluded': sorted(r for r, s in inventory['route_splits'].items() if s == 'test'),
             'weights': WEIGHT, 'stride_frames': STRIDE}
  (out/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  if args.extract:
    extract(rows, out, args.jobs)
  return summary


def extract_segment(task):
  import av
  import cv2
  from bisect import bisect_right
  from types import SimpleNamespace as N
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, nv12_strip
  from tools.lane_profile.scanner import sample, strip_geometry  # noqa: TID251
  from tools.lane_type.retrain_data import metadata  # noqa: TID251
  source, quality, wanted, path = task
  cv2.setNumThreads(1)
  identity, models, calibrations, _, cameras, encodes = metadata(source)
  ct = [c[0] for c in calibrations]
  need = defaultdict(list)
  for i, (index, side) in enumerate(wanted):
    need[index].append((i, side))
  strips = np.zeros((len(wanted), STRIP_ROWS, 96, 3), np.uint8)
  legacy = np.zeros((len(wanted), 192, 64, 3), np.uint8)
  ok = np.zeros((len(wanted), 2), bool)
  last = max(need)
  with av.open(source['videos'][quality]['path']) as container:
    for index, frame in enumerate(container.decode(video=0)):
      if index > last:
        break
      if index not in need:
        continue
      key = encodes[quality].get(index)
      model, sensor = models.get(key), cameras.get(key)
      ci = bisect_right(ct, key[1]) - 1
      cam = DEVICE_CAMERAS[identity['device'], sensor].fcam
      k = cam.intrinsics.copy()
      k[0] *= frame.width/cam.width
      k[1] *= frame.height/cam.height
      h, w = frame.height, frame.width
      planar = frame.to_ndarray(format='yuv420p')
      chroma = planar[h:].reshape(-1)
      uv = np.stack((chroma[:h*w//4].reshape(h//2, w//2), chroma[h*w//4:].reshape(h//2, w//2)), -1)
      nv12 = np.concatenate((planar[:h], uv.reshape(h//2, w)))
      rpy = calibrations[ci][1]
      for i, side in need[index]:
        line = model['lines'][1 if side == 'left' else 2]
        geo = strip_geometry(line, rpy, k, w, h)
        if geo is not None:
          lum, cb, cr = sample(planar[:h], uv, geo)
          ycc = np.clip(np.stack((lum, cb, cr), -1), 0, 255).astype(np.uint8)
          strips[i] = cv2.resize(ycc, (96, STRIP_ROWS), interpolation=cv2.INTER_AREA if len(ycc) > STRIP_ROWS else cv2.INTER_LINEAR)
          ok[i, 0] = True
        maps = lane_maps(N(**line), model['probs'][1 if side == 'left' else 2], rpy, k, w, h)
        if maps is not None:
          legacy[i] = nv12_strip(nv12.ravel(), w, h, w, w*h, maps)
          ok[i, 1] = True
  np.savez_compressed(path, row_strip_ycbcr=strips, legacy_6_46_rgb=legacy, available=ok)
  return str(path)


def extract(rows, out, jobs):
  inventory = {s['id']: s for s in json.loads(INVENTORY.read_text(encoding='utf-8'))['sources']}
  groups = defaultdict(list)
  for n, r in enumerate(rows):
    groups[(r['route'], r['segment'], r['quality'])].append(n)
  tasks, index = [], {}
  (out/'shards').mkdir(exist_ok=True)
  for (route, segment, quality), ns in groups.items():
    path = out/'shards'/f'{route}--{segment}-{quality}.npz'
    for j, n in enumerate(ns):
      index[n] = (path.name, j)
    tasks.append((inventory[f'{route}--{segment}'], quality, [(rows[n]['video_index'], rows[n]['side']) for n in ns], path))
  with Pool(jobs) as pool:
    for i, p in enumerate(pool.imap_unordered(extract_segment, tasks), 1):
      print(i, '/', len(tasks), p, flush=True)
  with open(out/'manifest.jsonl', 'w', encoding='utf-8') as fh:
    for n, r in enumerate(rows):
      fh.write(json.dumps(dict(r, shard=index[n][0], shard_row=index[n][1])) + '\n')


def build_test(args):
  """Held-out evaluation set: ONLY human_confirmed intervals of test routes. Same frame eligibility and
  inputs as training; also stores the scanner's and the active ONNX's per-frame outputs for comparison."""
  inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
  split_of = inventory['route_splits'].get
  human = [i for i in load_labels(args.human) if i['status'] == 'human_confirmed' and split_of(i['route']) == 'test']
  rows = []
  for seg in load(Path(args.logs)):
    for f in seg['frames']:
      if f['index'] % STRIDE:
        continue
      for side in ('left', 'right'):
        hit = [h for h in human if (h['route'], int(h['segment']), h['side'], h['quality']) == (seg['route'], seg['segment'], side, seg['quality'])
               and h['start_eof_ns'] <= f['eof'] <= h['end_eof_ns']]
        e = f[side]
        if len(hit) != 1 or f['lc'] or e[2] in ('no_geometry', 'no_path', 'lane_change') or e[8] < MIN_ROWS or (e[6] + e[7]) < MIN_KNOWN*e[8]:
          continue
        o = f.get('onnx_' + side) or [0, 0, 0, 0]
        rows.append({'route': seg['route'], 'segment': seg['segment'], 'quality': seg['quality'], 'split': 'test', 'side': side,
                     'video_index': f['index'], 'frame_id': f['frame_id'], 'eof': f['eof'], 'type': hit[0]['type'], 'color': hit[0]['color'],
                     'origin': 'human', 'weight': 1., 'label_ref': hit[0]['review_item_id'], 'speed': f['v'],
                     'scanner': [e[0], e[1]], 'scanner_rows': [e[6], e[7], e[8]], 'onnx_guarded': o[:2], 'onnx_raw': o[2:]})
  out = Path(args.output)
  out.mkdir(parents=True, exist_ok=True)
  extract(rows, out, args.jobs)
  (out/'summary.json').write_text(json.dumps({'intervals': len(human), 'frames': len(rows),
    'by_class': dict(Counter(f"{r['type']}/{r['color']}" for r in rows)), 'by_interval': dict(Counter(r['label_ref'] for r in rows)),
    'note': 'human_confirmed test intervals only (selected by the audit/conflict process, not a random sample)'}, indent=1), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--logs', required=True)
  p.add_argument('--human', required=True)
  p.add_argument('--auto')
  p.add_argument('--pool')
  p.add_argument('--clips', help='ai_reviewed interval labels (dev clips)')
  p.add_argument('--output', required=True)
  p.add_argument('--extract', action='store_true')
  p.add_argument('--jobs', type=int, default=12)
  p.add_argument('--test-human', action='store_true', help='build the held-out human test set instead')
  a = p.parse_args()
  if a.test_human:
    build_test(a)
  else:
    print(json.dumps(main(a), indent=1))
