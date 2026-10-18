"""All recording videos -> native source-context review candidates, never labels."""
import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels, source_box
from tools.lane_type.retrain_data import inventory, metadata, digest  # noqa: TID251
from tools.lane_type.validate_hybrid_tracking import read  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import write_json  # noqa: TID251


def extract_segment(task):
  source, destination = task
  output = Path(destination)
  cache = output/'segments'/f"{source['id']}.json"
  if cache.exists():
    return read(cache)
  cv2.setNumThreads(1)
  result = {'source': source['id'], 'route': source['route'], 'split': source['split'], 'videos': [], 'candidates': []}
  try:
    identity, models, calibration, cars, cameras, encodes = metadata(source)
    ct, st = [r[0] for r in calibration], [r[0] for r in cars]
    chosen_keys = []
    for quality in sorted(source['videos']):  # fcamera first, paired qcamera follows
      video = source['videos'][quality]
      status = {'path': video['path'], 'quality': quality, 'selected_frames': 0, 'exact_candidates': 0}
      result['videos'].append(status)
      pool = []
      for index, key in sorted(encodes[quality].items()):
        if key not in models or key not in cameras:
          continue
        ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
        if min(ci, si) < 0 or key[1]-ct[ci] > 2_000_000_000 or key[1]-st[si] > 200_000_000:
          continue
        model = models[key]
        pool.append({'index': index, 'key': key, 'ci': ci, 'si': si, 'speed': float(cars[si][1]),
          'curve': float(max(abs(np.diff(model['lines'][j]['y'])[:15]).max() for j in (1, 2))),
          'lane_change': bool(model['lane_change'] or cars[si][2]),
          'near_lead': any(l['prob'] >= .5 and l['x'] and l['x'][0] < 25 for l in model['leads'])})
      status['exact_candidates'] = len(pool)
      if not pool:
        status['excluded'] = 'no_exact_camera_model_encode_join_with_fresh_calibration_and_car'
        continue
      by_key = {r['key']: r for r in pool}
      proposals = [by_key[key] for key in chosen_keys if key in by_key]
      proposals += [pool[len(pool)//4], pool[3*len(pool)//4], min(pool, key=lambda r: r['speed']), max(pool, key=lambda r: r['curve'])]
      for flag in ('lane_change', 'near_lead'):
        if any(r[flag] for r in pool):
          proposals.append(next(r for r in pool if r[flag]))
      selected = []
      for p in proposals:
        if all(abs(p['key'][1]-other['key'][1]) >= 3_000_000_000 for other in selected):
          selected.append(p)
      needed = {p['index']: p for p in selected}
      if quality == 'fcamera':
        chosen_keys = [p['key'] for p in selected]
      with av.open(video['path']) as container:
        container.streams.video[0].thread_count = 1
        decoded = 0
        for index, frame in enumerate(container.decode(video=0)):
          decoded += 1
          if index not in needed:
            continue
          p = needed[index]
          key = p['key']
          camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
          k = camera.intrinsics.copy()
          k[0] *= frame.width/camera.width
          k[1] *= frame.height/camera.height
          rgb = frame.to_ndarray(format='rgb24')
          stem = f"{source['id']}-{key[0]}-{quality}"
          path = output/'frames'/f'{stem}.png'
          path.parent.mkdir(parents=True, exist_ok=True)
          Image.fromarray(rgb).save(path)
          status['selected_frames'] += 1
          for side, j in [('left', 1), ('right', 2)]:
            model = models[key]
            ribbon = collect_pixels(rgb, model['lines'][j], model['probs'][j], calibration[p['ci']][1], k, model['leads'], half_width_m=.9)
            box = source_box(rgb, ribbon)
            context = output/'native-context'/f'{stem}-{side}.png'
            context.parent.mkdir(parents=True, exist_ok=True)
            if box:
              Image.fromarray(box['image']).save(context)
              Image.fromarray(box['ribbon_mask'].astype(np.uint8)*255).save(context.with_name(context.stem+'-search-mask.png'))
            result['candidates'].append({'id': f'{stem}-{side}', 'scene_id': f"{source['id']}-{key[0]}-{side}",
              'route': source['route'], 'segment': source['segment'], 'split': source['split'], 'side': side, 'video_source': quality,
              'frame_id': key[0], 'stamp': key[1], 'video_index': index, 'frame': str(path.resolve()),
              'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'context': str(context.resolve()) if box else None,
              'origin_xy': box['origin_xy'].tolist() if box else None, 'line': model['lines'][j], 'intrinsics': k.tolist(),
              'rpy': calibration[p['ci']][1], 'probability': model['probs'][j], 'speed_mps': p['speed'],
              'curve_proposal': p['curve'], 'lane_change_proposal': p['lane_change'], 'lead_proposal': p['near_lead'],
              'vehicle_identity': identity, 'sync': 'exact_encode_frameId_timestampEof_camera_model',
              'type': 'unknown', 'color': 'unknown', 'verified': False, 'admitted_to_training': False,
              'context_excluded': None if box else 'no_observable_projected_source_pixels'})
        status['decoded_frames'] = decoded
        if not status['selected_frames']:
          status['excluded'] = 'selected_encode_indices_not_in_decoded_video'
  except Exception as e:
    result['excluded'] = f'{type(e).__name__}: {e}'
    for quality, video in source['videos'].items():
      status = next((v for v in result['videos'] if v['quality'] == quality), None)
      if status is None:
        result['videos'].append({'path': video['path'], 'quality': quality, 'selected_frames': 0,
                                'excluded': result['excluded']})
      elif 'decoded_frames' not in status:
        status['excluded'] = f'incomplete_decode: {result["excluded"]}'
  write_json(cache, result)
  return result


def video_fingerprint(path):
  h, count = hashlib.sha256(), 0
  with av.open(str(path)) as container:
    container.streams.video[0].thread_count = 1
    for frame in container.decode(video=0):
      h.update(frame.to_ndarray(format='rgb24').tobytes())
      count += 1
  return count, h.hexdigest()


def run(root, output, jobs):
  old = read(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json'))
  write_json(output/'split-lock.json', [{'route': route, 'split': split} for route, split in old['route_splits'].items()])
  inv = inventory(root, output/'data', output/'split-lock.json')
  write_json(output/'ROI_REVIEW_HOLD.json', {'training_admitted': False, 'model_promotion': False, 'labels': 'all unknown'})
  results = []
  with ProcessPoolExecutor(max_workers=jobs) as executor:
    for result in executor.map(extract_segment, [(source, str(output)) for source in inv['sources']]):
      results.append(result)
      print('covered', result['source'], len(result['candidates']), result.get('excluded', ''), flush=True)
  candidates = [r for group in results for r in group['candidates']]
  write_json(output/'candidates.json', candidates)
  coverage = []
  for file in inv['files']:
    if 'camera.' not in file['kind']:
      continue
    matches = [v for group in results for v in group['videos'] if v['quality'] == file['kind'].split('.')[0] and
      any(s['id'] == group['source'] and s['route'] == file['route'] and s['segment'] == file['segment'] for s in inv['sources'])]
    coverage.append(dict(file, extraction=matches[0] if matches else {'excluded': 'no_paired_rlog_source'},
      deduplicated_copy=bool(matches and matches[0]['path'] != file['path'])))
  for path in sorted(root.rglob('*')):
    if path.suffix.lower() not in ('.mp4', '.mkv', '.avi', '.mov'):
      continue
    record = {'path': str(path), 'sha256': digest(path), 'extraction': {'excluded': 'nonstandard_container_index_not_verified'}}
    original = next((Path(r['path']) for r in inv['files'] if r['kind'] == 'qcamera.ts' and
                     Path(r['path']).stem == path.stem), None)
    if original:
      try:
        a, b = video_fingerprint(path), video_fingerprint(original)
        record.update(decoded_frames=a[0], original=str(original), decoded_rgb_sequence_identical=a == b)
        if a == b:
          record['extraction'] = {'status': 'exact_decoded_duplicate_covered_by_ts', 'original': str(original)}
      except Exception as e:
        record['extraction']['error'] = f'{type(e).__name__}: {e}'
    coverage.append(record)
  accounted = {r['path'] for r in coverage}
  for path in sorted(root.rglob('*')):
    if path.suffix.lower() in ('.hevc', '.ts') and str(path) not in accounted:
      coverage.append({'path': str(path), 'sha256': digest(path),
                       'extraction': {'excluded': 'filename_not_recognized_as_route_segment_camera'}})
  write_json(output/'video-coverage.json', coverage)
  links = ['<!doctype html><meta charset="utf-8"><h2>전체 영상 후보 검수 — 학습 편입 보류</h2>',
    '<p>원본 문맥과 modelV2 탐색 mask입니다. 실제 목표 도색 선택이나 정답이 아닙니다. 이미지 원본은 별도 PNG이며 검수 페이지에서만 축소 표시합니다.</p>']
  for route in sorted({r['route'] for r in candidates}):
    rows = [r for r in candidates if r['route'] == route]
    body = [f'<!doctype html><meta charset="utf-8"><h2>{route} · {rows[0]["split"]} · UNKNOWN / 학습 보류</h2>']
    for frame in dict.fromkeys(r['frame'] for r in rows):
      path = Path(frame)
      body.append(f'<h3>{html.escape(path.stem)}</h3><a href="frames/{path.name}"><img loading="lazy" width="800" src="frames/{path.name}"></a>')
      for r in [r for r in rows if r['frame'] == frame]:
        if r['context']:
          name = Path(r['context']).name
          body.append(f'<p>{r["side"]}: native search context, UNKNOWN</p><img loading="lazy" src="native-context/{name}">')
    page = f'REVIEW-{route}.html'
    (output/page).write_text('\n'.join(body), encoding='utf-8')
    links.append(f'<p><a href="{page}">{route} · {rows[0]["split"]} · {len(rows)} boundary candidates</a></p>')
  (output/'PREVIEW.html').write_text('\n'.join(links), encoding='utf-8')
  write_json(output/'summary.json', {'video_files_accounted': len(coverage), 'source_segments': len(results),
    'boundary_candidates': len(candidates), 'source_frames': len({r['frame'] for r in candidates}),
    'admitted_to_training': 0, 'all_labels_unknown': True, 'route_splits_preserved': True,
    'excluded_videos': [r for r in coverage if 'excluded' in r['extraction']]})
  print('ALL VIDEO EXTRACTION COMPLETE', len(coverage), len(candidates), flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--root', type=Path, required=True, help='local OpenpilotLogs folder')
  p.add_argument('--output', type=Path, default=Path('.analysis/pixel-ribbon-hybrid-20261018/all-videos'))
  p.add_argument('--jobs', type=int, default=4)
  args = p.parse_args()
  run(args.root, args.output, args.jobs)
