"""Reassess the locked full-log candidates; evidence is not automatically truth."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.lane_type.retrain_data import metadata, digest  # noqa: TID251 - offline repository tooling
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.carrot.lane_type.dual_roi import extract_pair


def json_write(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def scan(arguments):
  import av
  import cv2
  source, old_root, output, forced = arguments
  old_root, output = Path(old_root), Path(output)
  output.mkdir(parents=True, exist_ok=True)
  target = output/'segments'/f"{source['id']}.json"
  if target.exists():
    return json.loads(target.read_text(encoding='utf-8'))
  cv2.setNumThreads(1)
  identity, models, calibrations, cars, cameras, encodes = metadata(source)
  result = {'id': source['id'], 'rows': [], 'counts': {}}
  stats = Counter()
  if not calibrations or not cars or identity.get('car') not in ('HYUNDAI_SONATA_2024', None):
    result['excluded'] = 'identity_or_metadata'
    json_write(target, result)
    return result
  if identity.get('car') is None:
    evidence = json.loads((old_root/'segments'/f"{source['id']}.json").read_text(encoding='utf-8'))
    if not evidence.get('identity_evidence'):
      result['excluded'] = 'uncorroborated_identity'
      json_write(target, result)
      return result
  ct, st = [v[0] for v in calibrations], [v[0] for v in cars]
  for quality, video in sorted(source['videos'].items()):
    candidate_file = old_root/'candidates'/f"{source['id']}-{quality}.json.gz"
    old = json.loads(gzip.decompress(candidate_file.read_bytes())) if candidate_file.exists() else []
    wanted = {}
    for row in old:
      eligible = row['observation']['sufficient'] and not row['lane_change']
      if eligible:
        stats['previously_eligible_reassessed'] += 1
      sample_id = row['scene_id']+'-'+quality
      if eligible or sample_id in forced:
        wanted[row['frame_id'], row['camera_eof_ns'], row['side']] = bool(eligible)
    for r in forced.values():
      if r['route'] == source['route'] and r['segment'] == source['segment'] and r['video_source'] == quality:
        wanted.setdefault((r['frame_id'], r['camera_eof_ns'], r['side']), False)
    needed = {index: key for index, key in encodes[quality].items()
              if any((*key, side) in wanted for side in ('left', 'right'))}
    if not needed:
      continue
    states, last_saved, pool = {}, {}, []
    with av.open(video['path']) as container:
      container.streams.video[0].thread_count = 1
      for index, frame in enumerate(container.decode(video=0)):
        if index not in needed:
          continue
        key = needed[index]
        if key not in models or key not in cameras:
          stats['no_exact_model_camera'] += 1
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
          stats['observations'] += 1
          scene = f"{source['id']}-{key[0]}-{side}"
          sample_id = scene+'-'+quality
          is_forced = sample_id in forced
          line = model['lines'][li]
          pair = extract_pair(rgb, line, model['probs'][li], rpy, k, model['leads'])
          heads = pair['heads']
          boundary = [float(np.interp(x, line['x'], line['y'])) for x in (10, 25)]
          color = heads.get('color', {}).get('evidence', {}).get('color', 'unknown')
          kind = heads.get('type', {}).get('evidence', {}).get('type', 'unknown')
          prev = states.get(side)
          change = bool(model['lane_change'] or car[2])
          continuity = (prev and 0 < key[1]-prev['stamp'] <= 300_000_000 and
                        max(abs(a-b) for a, b in zip(boundary, prev['boundary'], strict=True)) <= .25 and not change)
          labels = kind, color
          agrees = continuity and labels == prev['labels'] and 'unknown' not in labels
          votes = prev['votes']+1 if agrees else 1
          travel = prev['travel']+car[1]*(key[1]-prev['stamp'])/1e9 if agrees else 0.
          states[side] = {'stamp': key[1], 'boundary': boundary, 'labels': labels, 'votes': votes, 'travel': travel}
          auto = votes >= 5 and travel >= 1. and not change and 'unknown' not in labels
          stats['pixel_'+kind+'_'+color] += 1
          stats['temporally_supported_proposals'] += int(auto)
          if not (auto or is_forced):
            stats['not_temporally_supported'] += 1
            continue
          # Cap near duplicates by elapsed time, ego travel and thumbnail change.
          # These are proposal caps, not a claim of statistical independence.
          if len(heads) < 2:
            stats['missing_head'] += 1
            continue
          feature = cv2.resize(heads['type']['image'], (8, 24), interpolation=cv2.INTER_AREA).astype(float)/255
          bucket = (side, kind, color)
          last = last_saved.get(bucket)
          if not is_forced and last and (key[1]-last[0] < 2_000_000_000 or
              (key[1]-last[0])*car[1]/1e9 < 5. or np.abs(feature-last[1]).mean() < .025):
            stats['near_duplicate'] += 1
            continue
          if not is_forced and sum(r[0]['side'] == side and r[0]['proposed_type'] == kind and r[0]['proposed_color'] == color for r in pool) >= 12:
            stats['segment_class_cap'] += 1
            continue
          last_saved[bucket] = key[1], feature
          row = {'id': sample_id, 'scene_id': scene, 'route': source['route'], 'segment': source['segment'],
            'video_size': [frame.width, frame.height],
            'split': source['split'], 'video_source': quality, 'side': side, 'frame_id': key[0],
            'camera_eof_ns': key[1], 'model_eof_ns': model['eof'], 'video_index': index,
            'sync': 'exact_frame_id_and_eof', 'boundary': boundary, 'speed_mps': car[1],
            'lane_change': change, 'curve': abs(boundary[1]-boundary[0]) > .65,
            'lighting_proposal': 'night' if np.mean(rgb[frame.height//3:2*frame.height//3]) < 40 else 'day',
            'proposed_type': kind, 'proposed_color': color, 'temporal_votes': votes,
            'temporal_travel_m': travel, 'automatic_proposal': bool(auto), 'verified': False,
            'previously_eligible': wanted[*key, side], 'baseline_reviewed': bool(is_forced and forced[sample_id].get('verified')),
            'type': None, 'color': None, 'label_source': 'unapproved_pixel_temporal_proposal',
            'log_sha256': source['log']['sha256'], 'video_sha256': video['sha256'], 'heads': {}}
          for head, data in heads.items():
            path = f"roi/{sample_id}-{head}.png"
            (output/'roi').mkdir(exist_ok=True)
            Image.fromarray(data['image']).save(output/path)
            row['heads'][head] = {name: v for name, v in data.items() if name not in ('image', 'maps')}
            row['heads'][head].update(image=path, image_sha256=digest(output/path),
              outline=np.stack((data['maps'][0][::12, [0, -1]], data['maps'][1][::12, [0, -1]]), -1).tolist())
          context = f"context/{source['id']}-{index}-{quality}.jpg"
          (output/'context').mkdir(exist_ok=True)
          thumb = Image.fromarray(rgb)
          thumb.thumbnail((1008, 570))
          thumb.save(output/context, quality=93)
          row['context'] = context
          pool.append((row, feature))
    result['rows'].extend(r[0] for r in pool)
  result['counts'] = dict(stats)
  json_write(target, result)
  return result


def build(old_root, output, jobs=6, review_pool=False):
  output.mkdir(parents=True, exist_ok=True)
  inventory = json.loads((old_root/'inventory.json').read_text(encoding='utf-8'))
  forced_rows = json.loads((old_root/'roi-audited-manifest.json').read_text(encoding='utf-8'))
  if review_pool:
    candidates = json.loads((old_root/'manifest.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for row in candidates:
      groups[row['route'], row['segment'], row['video_source']].append(row)
    for group in groups.values():
      # The old queue already used farthest-point appearance diversity. Spread
      # review picks across its entire timeline instead of taking first frames.
      indices = np.linspace(0, len(group)-1, min(10, len(group))).astype(int)
      forced_rows.extend(group[int(i)] for i in indices)
  forced = {r['id']: r for r in forced_rows}
  tasks = [(s, str(old_root), str(output), {k: r for k, r in forced.items() if r['route'] == s['route'] and r['segment'] == s['segment']})
           for s in inventory['sources'] if s['videos']]
  rows, counts = [], Counter()
  with ProcessPoolExecutor(max_workers=jobs) as executor:
    for i, result in enumerate(executor.map(scan, tasks)):
      rows.extend(result['rows'])
      counts.update(result['counts'])
      print(i+1, '/', len(tasks), result['id'], len(result['rows']), flush=True)
  # Exact f/q pairs share scene/split. Approval is a later audit step, and an
  # invisible quality never inherits a visible quality's training eligibility.
  json_write(output/'proposals.json', rows)
  json_write(output/'scan-summary.json', {'counts': dict(counts), 'proposals': len(rows),
    'route_splits': inventory['route_splits'], 'auto_approved': 0, 'test_used_for_rules': False})
  return rows


def sheets(rows, root, output, per_page=12):
  """Review context plus both heads; no auto label printed beside the crops."""
  output.mkdir(parents=True, exist_ok=True)
  index = []
  for start in range(0, len(rows), per_page):
    batch = rows[start:start+per_page]
    page = Image.new('RGB', (1800, 250*((len(batch)+2)//3)), '#202020')
    draw = ImageDraw.Draw(page)
    for j, row in enumerate(batch):
      x, y = (j % 3)*600, (j//3)*250
      context = Image.open(root/row['context']).convert('RGB')
      native = row.get('video_size', [1344, 760] if row['video_source'] == 'fcamera' else [526, 330])
      overlay = ImageDraw.Draw(context)
      for head, color in [('type', '#40ffff'), ('color', '#ffff40')]:
        for edge in (0, 1):
          points = [(p[edge][0]*context.width/native[0], p[edge][1]*context.height/native[1])
                    for p in row['heads'][head]['outline']]
          overlay.line(points, fill=color, width=1)
      context = context.crop((0, int(context.height*.25), context.width, int(context.height*.96)))
      context.thumbnail((420, 200))
      page.paste(context, (x, y+28))
      for k, head in enumerate(('color', 'type')):
        image = Image.open(root/row['heads'][head]['image']).resize((64, 192))
        page.paste(image, (x+430+k*75, y+30))
      draw.text((x+5, y+3), f"{start+j}: {row['id']}", fill='white')
      draw.text((x+5, y+227), f"{row['split']} {row['side']} C/T  {row['speed_mps']:.1f}m/s", fill='white')
    name = f'{start//per_page:03}.jpg'
    page.save(output/name, quality=94)
    index.append({'page': name, 'ids': [r['id'] for r in batch]})
  json_write(output/'index.json', index)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--old-data', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--jobs', type=int, default=6)
  parser.add_argument('--review-pool', action='store_true')
  args = parser.parse_args()
  build(args.old_data, args.output, args.jobs, args.review_pool)
