"""Full-timeline distinct-pixel observations and reviewable native crops."""
import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.lane_type.prepare_dual_data import json_write  # noqa: TID251 - offline repository tooling
from tools.lane_type.retrain_data import metadata  # noqa: TID251 - offline repository tooling
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.carrot.lane_type.native import native_observation


def read_image(root, head, cache=None):
  if 'array_file' not in head:
    return np.asarray(Image.open(root/head['image']).convert('RGB')).copy()
  path = str(root/head['array_file'])
  if cache is not None:
    if path not in cache:
      with np.load(path) as data:
        cache[path] = {k: data[k].copy() for k in data.files}
    return cache[path][head['array_key']][head['array_index']].copy()
  with np.load(path) as data:
    return data[head['array_key']][head['array_index']].copy()


def process(arguments):
  import av
  import cv2
  source, old_root, output, review_ids = arguments
  old_root, output = Path(old_root), Path(output)
  target = output/'segments'/f"{source['id']}.json.gz"
  if target.exists():
    return {'id': source['id'], 'counts': json.loads(gzip.decompress(target.read_bytes()))['counts']}
  cv2.setNumThreads(1)
  identity, models, calibrations, cars, cameras, encodes = metadata(source)
  rows, stats = [], Counter()
  if not calibrations or not cars:
    return {'id': source['id'], 'counts': {'missing_metadata': 1}}
  ct, st = [v[0] for v in calibrations], [v[0] for v in cars]
  arrays = {'type': [], 'color': []}
  for quality, video in sorted(source['videos'].items()):
    f = old_root/'candidates'/f"{source['id']}-{quality}.json.gz"
    old = json.loads(gzip.decompress(f.read_bytes())) if f.exists() and not source.get('only_exact_extra') else []
    wanted = {(r['frame_id'], r['camera_eof_ns'], r['side']) for r in old}
    wanted.update((r['frame_id'], r['camera_eof_ns'], r['side']) for r in source.get('exact_extra', ()) if r['video_source'] == quality)
    needed = {index: key for index, key in encodes[quality].items() if any((*key, side) in wanted for side in ('left', 'right'))}
    with av.open(video['path']) as container:
      container.streams.video[0].thread_count = 1
      for index, frame in enumerate(container.decode(video=0)):
        if index not in needed:
          continue
        key = needed[index]
        if key not in models or key not in cameras:
          stats['not_exact_match'] += 1
          continue
        ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
        if ci < 0 or si < 0 or key[1]-ct[ci] > 2_000_000_000 or key[1]-st[si] > 200_000_000:
          stats['stale_metadata'] += 1
          continue
        model, rpy, car = models[key], calibrations[ci][1], cars[si]
        camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
        k = camera.intrinsics.copy()
        k[0] *= frame.width/camera.width
        k[1] *= frame.height/camera.height
        rgb = frame.to_ndarray(format='rgb24')
        for side, li in [('left', 1), ('right', 2)]:
          if (*key, side) not in wanted:
            continue
          line = model['lines'][li]
          observation = native_observation(rgb, line, model['probs'][li], rpy, k, model['leads'])
          sample_id = f"{source['id']}-{key[0]}-{side}-{quality}"
          row = {'id': sample_id, 'scene_id': f"{source['id']}-{key[0]}-{side}", 'route': source['route'],
            'segment': source['segment'], 'split': source['split'], 'video_source': quality, 'side': side,
            'frame_id': key[0], 'camera_eof_ns': key[1], 'model_eof_ns': model['eof'], 'video_index': index,
            'speed_mps': car[1], 'lane_change': bool(model['lane_change'] or car[2]),
            'boundary': [float(np.interp(x, line['x'], line['y'])) for x in (10, 25)],
            'profile': observation['profile'], 'heads': {}, 'reason': observation['reason'],
            'sync': 'exact_frame_id_and_eof', 'log_sha256': source['log']['sha256']}
          for head, h in observation['heads'].items():
            im = h.pop('image')
            row['heads'][head] = dict(h, array_file=str((output/'arrays'/f"{source['id']}.npz").resolve()),
              array_key=head, array_index=len(arrays[head]), image_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
              image_shape=list(im.shape))
            arrays[head].append(im)
            if sample_id in review_ids:
              path = output/'review-roi'/f'{sample_id}-{head}.png'
              path.parent.mkdir(exist_ok=True, parents=True)
              Image.fromarray(im).save(path)
          stats['observations'] += 1
          stats['two_heads' if len(row['heads']) == 2 else 'one_or_no_head'] += 1
          stats['native_paint_rows'] += sum(p['ink'] == 1 for p in row['profile'])
          rows.append(row)
  (output/'arrays').mkdir(exist_ok=True, parents=True)
  np.savez_compressed(output/'arrays'/f"{source['id']}.npz", **{
    head: np.stack(images) if images else np.empty((0, 64 if head == 'type' else 32, 16, 3), np.uint8)
    for head, images in arrays.items()})
  target.parent.mkdir(exist_ok=True, parents=True)
  target.write_bytes(gzip.compress(json.dumps({'rows': rows, 'counts': dict(stats)}, separators=(',', ':')).encode()))
  return {'id': source['id'], 'counts': dict(stats)}


def build(old_root, output, review_pool, jobs):
  output.mkdir(exist_ok=True, parents=True)
  inventory = json.loads((old_root/'inventory.json').read_text(encoding='utf-8'))
  review_ids = {r['id'] for r in json.loads(review_pool.read_text(encoding='utf-8'))}
  tasks = [(s, str(old_root), str(output), review_ids) for s in inventory['sources'] if s['videos']]
  counts = Counter()
  with ProcessPoolExecutor(max_workers=jobs) as executor:
    for i, result in enumerate(executor.map(process, tasks)):
      counts.update(result['counts'])
      print(i+1, '/', len(tasks), result['id'], result['counts'].get('observations', 0), flush=True)
  json_write(output/'summary.json', {'counts': dict(counts), 'source': str(old_root), 'route_splits': inventory['route_splits'],
    'sampling': 'all prior 5Hz exact matched, geometry-valid candidates; unknowns retained',
    'native_input_shapes': {'type': [64, 16, 3], 'color': [32, 16, 3]}, 'pixel_interpolation': False,
    'can_transmission_enabled': False})


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--old-data', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--review-pool', type=Path, required=True)
  parser.add_argument('--jobs', type=int, default=6)
  args = parser.parse_args()
  build(args.old_data, args.output, args.review_pool, args.jobs)
