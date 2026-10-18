"""Read-only added-HD inventory and exact-frame native-pixel review candidates.

No predicted class labels, training admission or threshold/model selection.
"""
import argparse
from bisect import bisect_right
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels, source_box
from tools.lane_type.retrain_data import inventory, metadata  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import write_json  # noqa: TID251
from tools.lane_type.validate_hybrid_tracking import read  # noqa: TID251


def extract(root, output):
  previous = read(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json'))
  write_json(output/'split-lock.json', [{'route': route, 'split': split} for route, split in previous['route_splits'].items()])
  inv = inventory(root, output/'data', output/'split-lock.json')
  records = []
  for source in inv['sources']:
    identity, models, calibration, cars, cameras, encodes = metadata(source)
    if not calibration or not cars or 'fcamera' not in source['videos']:
      records.append({'source': source['id'], 'excluded': 'missing_calibration_car_or_video'})
      continue
    ct, st = [r[0] for r in calibration], [r[0] for r in cars]
    pool = []
    for index, key in sorted(encodes['fcamera'].items()):
      if key not in models or key not in cameras:
        continue
      ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
      if min(ci, si) < 0 or key[1]-ct[ci] > 2_000_000_000 or key[1]-st[si] > 200_000_000:
        continue
      m = models[key]
      curvature = max(abs(np.diff(m['lines'][j]['y'])[:15]).max() for j in (1, 2))
      pool.append({'index': index, 'key': key, 'ci': ci, 'si': si, 'speed': float(cars[si][1]),
        'curve_proposal': float(curvature), 'lane_change_proposal': bool(m['lane_change'] or cars[si][2]),
        'lead_proposal': any(l['prob'] >= .5 and l['x'] and l['x'][0] < 25 for l in m['leads'])})
    if not pool:
      records.append({'source': source['id'], 'excluded': 'no_exact_fresh_camera_model_join'})
      continue
    # Metadata-only coverage: two timeline anchors, slowest, largest curve.
    ordered = [pool[len(pool)//4], pool[3*len(pool)//4], min(pool, key=lambda r: r['speed']),
               max(pool, key=lambda r: r['curve_proposal'])]
    for condition in ('lane_change_proposal', 'lead_proposal'):
      if any(p[condition] for p in pool):
        ordered.append(next(p for p in pool if p[condition]))
    selected = []
    for row in ordered:
      if all(abs(row['key'][1]-other['key'][1]) >= 3_000_000_000 for other in selected):
        selected.append(row)
    needed = {r['index']: r for r in selected}
    with av.open(source['videos']['fcamera']['path']) as container:
      container.streams.video[0].thread_count = 1
      for index, frame in enumerate(container.decode(video=0)):
        if index not in needed:
          continue
        p = needed[index]
        key = p['key']
        rgb = frame.to_ndarray(format='rgb24')
        stem = f"{source['id']}-{key[0]}"
        frame_path = output/'frames'/f'{stem}.png'
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(frame_path)
        camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
        k = camera.intrinsics.copy()
        k[0] *= frame.width/camera.width
        k[1] *= frame.height/camera.height
        rpy = calibration[p['ci']][1]
        m = models[key]
        for side, j in [('left', 1), ('right', 2)]:
          ribbon = collect_pixels(rgb, m['lines'][j], m['probs'][j], rpy, k, m['leads'], half_width_m=.9)
          box = source_box(rgb, ribbon)
          crop_path = output/'native-context'/f'{stem}-{side}.png'
          crop_path.parent.mkdir(parents=True, exist_ok=True)
          if box:
            Image.fromarray(box['image']).save(crop_path)
            Image.fromarray(box['ribbon_mask'].astype(np.uint8)*255).save(crop_path.with_name(crop_path.stem+'-search-mask.png'))
          records.append({'source': source['id'], 'route': source['route'], 'segment': source['segment'], 'split': source['split'],
            'side': side, 'frame_id': key[0], 'stamp': key[1], 'video_index': index, 'frame': str(frame_path.resolve()),
            'context': str(crop_path.resolve()) if box else None, 'origin_xy': box['origin_xy'].tolist() if box else None,
            'intrinsics': k.tolist(), 'rpy': rpy, 'line': m['lines'][j], 'probability': m['probs'][j],
            'vehicle_identity': identity, 'model_leads': m['leads'], 'speed_mps': p['speed'],
            'lane_change_proposal': p['lane_change_proposal'], 'curve_proposal': p['curve_proposal'], 'lead_proposal': p['lead_proposal'],
            'type': 'unknown', 'color': 'unknown', 'verified': False, 'admitted_to_training': False,
            'sync': 'exact_encode_frameId_timestampEof_camera_model', 'review_role': 'source-only holdout preview, never rule tuning'})
    print('extracted', source['id'], source['split'], len(needed), flush=True)
  write_json(output/'candidates.json', records)
  images = sorted((output/'frames').glob('*.png'))
  columns, tw, th = 4, 400, 250
  contact = Image.new('RGB', (columns*tw, ((len(images)+columns-1)//columns)*th), 'white')
  for i, path in enumerate(images):
    im = Image.open(path)
    im.thumbnail((tw, th-28))  # review contact sheet only; native inputs stay untouched
    x, y = i % columns*tw, i//columns*th
    contact.paste(im, (x, y+28))
    ImageDraw.Draw(contact).text((x+3, y+3), path.stem, fill='black')
  contact.save(output/'CONTACT_SHEET.jpg', quality=92)
  body = ['<!doctype html><meta charset="utf-8"><title>추가 HD 원본 검수</title>',
    '<h2>추가 HD 원본·4.5~20m 탐색 영역 — 정답/학습 편입 보류</h2>',
    '<p>modelV2 주변의 원본 픽셀 문맥입니다. 목표 도색 선택 mask나 정답이 아닙니다. 테스트 route는 규칙 조정에 사용하지 않습니다.</p>']
  for path in images:
    matching = [r for r in records if r.get('frame') == str(path.resolve())]
    body.append(f'<h3>{path.stem} · {matching[0]["split"]}</h3><a href="frames/{path.name}"><img width="800" src="frames/{path.name}"></a>')
    for row in matching:
      if row['context']:
        body.append(f'<p>{row["side"]}, native context, class UNKNOWN</p><img src="native-context/{Path(row["context"]).name}">')
  (output/'PREVIEW.html').write_text('\n'.join(body), encoding='utf-8')
  write_json(output/'ROI_REVIEW_HOLD.json', {'training_admitted': False, 'model_promotion': False, 'predicted_labels_as_truth': False})
  print('source frames', len(images), 'boundary candidates', sum('frame_id' in r for r in records), flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--root', type=Path, required=True, help='local OpenpilotLogs/고화질 영상 folder')
  p.add_argument('--output', type=Path, default=Path('.analysis/pixel-ribbon-hybrid-20261018/added-hd'))
  args = p.parse_args()
  extract(args.root, args.output)
