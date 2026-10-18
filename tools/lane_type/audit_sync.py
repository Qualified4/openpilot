"""Independently decode selected original videos and audit exact metadata joins."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

from tools.lane_type.retrain_data import metadata  # noqa: TID251 - offline repository tooling
from tools.lane_type.dataset import read_events  # noqa: TID251 - offline repository tooling
from openpilot.common.transformations.camera import DEVICE_CAMERAS


def run(inventory_path, selected_path, output):
  inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
  selected = json.loads(selected_path.read_text(encoding='utf-8'))
  # Selection is declared before decoding. Test scenes excluded from rule work.
  selected = [r for r in selected if r['split'] != 'test' and r['color'] == 'yellow' and r['video_source'] == 'fcamera']
  groups = {}
  for r in selected:
    groups.setdefault((r['route'], r['segment']), []).append(r)
  output.mkdir(parents=True, exist_ok=True)
  frames = output/'frames'
  frames.mkdir(exist_ok=True)
  cache, results = [], []
  for (route, segment), rows in groups.items():
    source = next(s for s in inventory['sources'] if s['route'] == route and s['segment'] == segment)
    identity, models, calibrations, cars, cameras, encodes = metadata(source)
    raw_references, transforms = 0, set()
    for event in read_events(Path(source['log']['path'])):
      if event.which() == 'roadCameraState':
        raw_references += bool(len(event.roadCameraState.image))
        transforms.add(tuple(event.roadCameraState.transform))
    ct = [c[0] for c in calibrations]
    needed = {r['video_index']: r for r in rows}
    with av.open(source['videos']['fcamera']['path']) as container:
      stream = container.streams.video[0]
      stream.thread_count = 1
      codec = stream.codec_context
      info = {'codec': codec.name, 'has_b_frames': codec.has_b_frames, 'width': codec.width, 'height': codec.height}
      count, matched = 0, 0
      for index, frame in enumerate(container.decode(video=0)):
        count += 1
        if index not in needed:
          continue
        r = needed[index]
        key = (r['frame_id'], r['camera_eof_ns'])
        assert encodes['fcamera'][index] == key
        assert key in models and key in cameras
        assert models[key]['frame'] == key[0] and models[key]['eof'] == key[1]
        ci = bisect_right(ct, key[1])-1
        assert ci >= 0 and 0 <= key[1]-ct[ci] <= 2_000_000_000
        camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
        k = camera.intrinsics.copy()
        k[0] *= frame.width/camera.width
        k[1] *= frame.height/camera.height
        rgb = frame.to_ndarray(format='rgb24')
        target = frames/f"{r['id']}.png"
        Image.fromarray(rgb).save(target)
        assert np.array_equal(np.asarray(Image.open(target)), rgb)
        li = 1 if r['side'] == 'left' else 2
        cache.append({'id': r['id'], 'split': r['split'], 'frame': str(target.resolve()),
          'frame_size': [frame.width, frame.height], 'intrinsics': k.tolist(), 'rpy': calibrations[ci][1],
          'line': models[key]['lines'][li], 'model_leads': models[key]['leads'], 'video_source': 'fcamera'})
        results.append(dict(id=r['id'], exact_frame_id=key[0], exact_eof_ns=key[1], decode_index=index,
          calibration_age_ns=key[1]-ct[ci], device=identity, sensor=cameras[key],
          intrinsics=k.tolist(), original_video=str(source['videos']['fcamera']['path']),
          decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), image_saved_losslessly=True,
          encoded_metadata_agrees=True, physical_sync_independently_proven=False, **info))
        matched += 1
      indices = sorted(encodes['fcamera'])
      results.append({'route': route, 'segment': segment, 'decoded_frames': count,
        'encode_index_count': len(indices), 'contiguous_indices_starting_zero': indices == list(range(len(indices))),
        'decode_count_matches_index_count': count == len(indices), 'selected_frames_matched': matched,
        'uncompressed_camera_image_records': raw_references, 'recorded_camera_transforms': [list(t) for t in sorted(transforms)]})
  (output/'audit.json').write_text(json.dumps(cache, indent=2), encoding='utf-8')
  annotation = output/'annotation'
  annotation.mkdir(exist_ok=True)
  if not (annotation/'paint-annotations.json').exists():
    (annotation/'paint-annotations.json').write_text('[]', encoding='utf-8')
  (output/'sync-audit.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--inventory', type=Path, required=True)
  p.add_argument('--selected', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  a = p.parse_args()
  run(a.inventory, a.selected, a.output)
