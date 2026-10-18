"""PC-only full-timeline inventory, exact synchronization and diverse review pool.

All extracted labels remain proposals until an explicit image review. Never
overwrite source recordings, the pilot dataset, or the installed model.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace as N

import numpy as np
from PIL import Image, ImageDraw

from tools.lane_type.dataset import read_events  # noqa: TID251 - PC repository tooling
from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, rgb_strip
from openpilot.selfdrive.carrot.lane_type.observation import visibility
from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip


def digest(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(root, output, pilot_sources):
  pattern = re.compile(r'(\w{8}--[0-9a-f]+)--(\d+)--(rlog\.zst|qlog\.zst|fcamera\.hevc|qcamera\.ts)$')
  groups = defaultdict(lambda: defaultdict(list))
  files = []
  for path in sorted(root.rglob('*')):
    match = pattern.fullmatch(path.name)
    if not path.is_file() or not match:
      continue
    route, segment, kind = match.groups()
    record = {'path': str(path), 'route': route, 'segment': int(segment), 'kind': kind,
              'bytes': path.stat().st_size, 'sha256': digest(path)}
    groups[route, int(segment)][kind].append(record)
    files.append(record)
  splits = {row['route']: row['split'] for row in json.loads(pilot_sources.read_text(encoding='utf-8'))}
  # Lock before looking at new labels. Old validation/test routes never enter training.
  for route in sorted({key[0] for key in groups}):
    if route not in splits:
      bucket = int(hashlib.sha256(('lane-retrain-v2/'+route).encode()).hexdigest()[:8], 16) % 10
      splits[route] = 'test' if bucket < 2 else 'validation' if bucket < 4 else 'train'
  sources = []
  for (route, segment), kinds in sorted(groups.items()):
    for kind, copies in kinds.items():
      if len({c['sha256'] for c in copies}) != 1:
        raise ValueError(f'conflicting recording copies: {route}/{segment}/{kind}')
    if 'rlog.zst' in kinds:
      sources.append({'id': f'{route}--{segment}', 'route': route, 'segment': segment, 'split': splits[route],
                      'log': kinds['rlog.zst'][0],
                      'videos': {kind.split('.')[0]: copies[0] for kind, copies in kinds.items() if 'camera.' in kind}})
  output.mkdir(parents=True, exist_ok=True)
  result = {'root': str(root), 'files': files, 'sources': sources, 'route_splits': splits,
            'policy': 'route lock before new image review; pilot validation/test routes preserved',
            'duplicate_file_copies': len(files)-sum(len(v) for v in groups.values())}
  target = output/'inventory.json'
  if target.exists() and json.loads(target.read_text(encoding='utf-8')) != result:
    raise ValueError('inventory already locked; use a new output directory')
  target.write_text(json.dumps(result, indent=2), encoding='utf-8')
  return result


def metadata(source):
  models, calibration, cars, cameras = {}, [], [], {}
  encodes = {'fcamera': {}, 'qcamera': {}}
  identity = {}
  for event in read_events(Path(source['log']['path'])):
    kind = event.which()
    if kind == 'initData':
      identity.update(device=str(event.initData.deviceType), dongle=str(event.initData.dongleId))
    elif kind == 'carParams':
      identity['car'] = str(event.carParams.carFingerprint)
    elif kind == 'modelV2' and event.valid:
      m = event.modelV2
      models[m.frameId, m.timestampEof] = {'eof': m.timestampEof, 'frame': m.frameId, 'probs': list(m.laneLineProbs),
        'lines': [{'x': list(line.x), 'y': list(line.y), 'z': list(line.z)} for line in m.laneLines],
        'leads': [{'prob': lead.prob, 'x': list(lead.x), 'y': list(lead.y)} for lead in m.leadsV3],
        'lane_change': str(m.meta.laneChangeState) != 'off'}
    elif kind == 'carState' and event.valid:
      cars.append((event.logMonoTime, event.carState.vEgo, bool(event.carState.leftBlinker or event.carState.rightBlinker)))
    elif kind == 'liveCalibration' and event.valid and str(event.liveCalibration.calStatus) == 'calibrated':
      calibration.append((event.logMonoTime, list(event.liveCalibration.rpyCalib)))
    elif kind == 'roadCameraState' and event.valid:
      c = event.roadCameraState
      cameras[c.frameId, c.timestampEof] = str(c.sensor)
    elif kind in ('roadEncodeIdx', 'qRoadEncodeIdx'):
      e = getattr(event, kind)
      if e.segmentNum == source['segment']:
        encodes['fcamera' if kind == 'roadEncodeIdx' else 'qcamera'][e.segmentId] = (e.frameId, e.timestampEof)
  return identity, models, sorted(calibration), sorted(cars), cameras, encodes


def diverse_indices(features, limit):
  """Deterministic farthest-point selection; no truth labels or model scores."""
  if not features:
    return []
  x = np.asarray(features, dtype=np.float32)
  span = np.ptp(x, axis=0)
  x = (x-x.min(0))/np.maximum(span, .05)
  chosen = [int(np.argmax(np.linalg.norm(x-x.mean(0), axis=1)))]
  distance = np.full(len(x), np.inf)
  for _ in range(min(limit, len(x))-1):
    distance = np.minimum(distance, np.square(x-x[chosen[-1]]).mean(1))
    distance[chosen] = -1
    best = int(np.argmax(distance))
    if distance[best] < .012:
      break
    chosen.append(best)
  return sorted(chosen)


def scan_source(args):
  import av
  import cv2
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  source, destination = args
  output = Path(destination)
  cache = output/'segments'/f"{source['id']}.json"
  if cache.exists() and not source.get('route_identity'):
    return json.loads(cache.read_text(encoding='utf-8'))
  cv2.setNumThreads(1)
  identity, models, calibrations, cars, cameras, encodes = metadata(source)
  stats = Counter()
  result = {'id': source['id'], 'identity': identity, 'videos': {}, 'rows': []}
  evidence = source.get('route_identity', {})
  corroborated = (identity.get('car') is None and evidence.get('car') == 'HYUNDAI_SONATA_2024' and
                  identity.get('dongle') == evidence.get('dongle') and evidence.get('route') == source['route'])
  if corroborated:
    result['identity_evidence'] = evidence
  if identity.get('car') != 'HYUNDAI_SONATA_2024' and not corroborated:
    result['excluded'] = 'unverified_or_different_vehicle'
  elif not calibrations or not cars:
    result['excluded'] = 'missing_calibration_or_car_state'
  else:
    ct, st = [c[0] for c in calibrations], [c[0] for c in cars]
    paired_scenes = set()
    for quality in sorted(source['videos']):
      video = source['videos'][quality]
      pool, features = [], []
      indices = encodes[quality]
      first = min((v[1] for v in indices.values()), default=0)
      last_bucket = -1
      candidates = []
      with av.open(video['path']) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        info = {'width': stream.width, 'height': stream.height, 'decoded': 0, 'exact_matches': 0,
                'sha256': video['sha256'], 'pixel_format': stream.format.name if stream.format else None}
        for index, frame in enumerate(container.decode(video=0)):
          info['decoded'] += 1
          key = indices.get(index)
          if key is None:
            stats['frame_without_encode_index'] += 1
            continue
          bucket = (key[1]-first)//200_000_000
          if bucket <= last_bucket:
            continue
          model = models.get(key)
          sensor = cameras.get(key)
          # Exact frameId AND timestampEof; no nearest-time or frame-rate interpolation.
          if model is None or sensor is None:
            stats['no_exact_model_camera_match'] += 1
            continue
          last_bucket = bucket
          ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
          if ci < 0 or si < 0 or key[1]-ct[ci] > 2_000_000_000 or key[1]-st[si] > 200_000_000:
            stats['stale_calibration_or_car_state'] += 1
            continue
          info['exact_matches'] += 1
          rpy, car = calibrations[ci][1], cars[si]
          camera = DEVICE_CAMERAS[identity['device'], sensor].fcam
          k = camera.intrinsics.copy()
          k[0] *= frame.width/camera.width
          k[1] *= frame.height/camera.height
          rgb = frame.to_ndarray(format='rgb24')
          lighting = float(np.mean(rgb[frame.height//3:2*frame.height//3]))/255
          for side, li in [('left', 1), ('right', 2)]:
            stats['side_observations'] += 1
            line = N(**model['lines'][li])
            maps = lane_maps(line, model['probs'][li], rpy, k, frame.width, frame.height)
            if maps is None:
              stats['incomplete_geometry_or_low_probability'] += 1
              continue
            strip = rgb_strip(rgb, maps)
            clear, observation = visibility(maps, N(leadsV3=[N(**lead) for lead in model['leads']]), line, rpy, k)
            strip[~clear] = 0
            suggestion = classify_strip(strip)
            gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            yellow = float(((hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 60)).mean())
            curve = abs(float(np.interp(46, line.x, line.y)-np.interp(6, line.x, line.y)))
            lane_change = model['lane_change'] or car[2]
            profile = cv2.resize(gray, (4, 12), interpolation=cv2.INTER_AREA).flatten()/255
            feature = [lighting, min(car[1]/30, 1), min(curve/4, 1), float(lane_change), observation['occludedFraction'],
                       yellow, float(gray.std())/128, *profile.tolist()]
            scene = f"{source['id']}-{key[0]}-{side}"
            row = {'id': f'{scene}-{quality}', 'scene_id': scene, 'route': source['route'], 'segment': source['segment'],
              'split': source['split'], 'frame_id': key[0], 'camera_eof_ns': key[1], 'model_eof_ns': model['eof'],
              'sync': 'exact_frame_id_and_eof', 'video_index': index, 'video_seconds': (key[1]-first)/1e9,
              'video_source': quality, 'side': side, 'model_probability': model['probs'][li],
              'curve': curve > 1, 'lane_change': lane_change, 'speed_mps': car[1], 'lighting': 'unreviewed',
              'image_luminance': lighting, 'observation': observation, 'tracked': True,
              'boundary': [float(np.interp(x, line.x, line.y)) for x in (10, 30)], 'suggestion': suggestion,
              'verified': False, 'type': None, 'color': None, 'roi_range_m': [6, 46], 'log_sha256': source['log']['sha256']}
            outline = np.stack((maps[0][::12, [0, -1]], maps[1][::12, [0, -1]]), -1).tolist()
            pool.append((row, strip, outline))
            features.append(feature)
            candidates.append({'scene_id': scene, 'frame_id': key[0], 'side': side, 'camera_eof_ns': key[1],
                               'feature': feature, 'observation': observation, 'lane_change': lane_change})
      chosen = sorted(set(diverse_indices(features, 32)) |
                      {i for i, (row, _, _) in enumerate(pool) if row['scene_id'] in paired_scenes})
      selected = [pool[i] for i in chosen]
      if quality == 'fcamera':
        paired_scenes.update(row['scene_id'] for row, _, _ in selected)
      needed = {row['video_index'] for row, _, _ in selected}
      with av.open(video['path']) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
          if index not in needed:
            continue
          context = f"context/{source['id']}-{index}-{quality}.jpg"
          thumb = Image.fromarray(frame.to_ndarray(format='rgb24'))
          thumb.thumbnail((1008, 570))
          thumb.save(output/context, quality=94)
          for row, strip, outline in selected:
            if row['video_index'] != index:
              continue
            row['context'] = context
            row['roi_outline'] = outline
            row['video_size'] = [frame.width, frame.height]
            row['image'] = 'roi/'+row['id']+'.png'
            Image.fromarray(strip).save(output/row['image'])
            row['image_sha256'] = digest(output/row['image'])
            result['rows'].append(row)
      info['diverse_rois'] = len(selected)
      result['videos'][quality] = info
      with gzip.open(output/'candidates'/f"{source['id']}-{quality}.json.gz", 'wt', encoding='utf-8') as stream:
        json.dump(candidates, stream)
  result['counts'] = dict(stats)
  cache.write_text(json.dumps(result, indent=2), encoding='utf-8')
  return result


def build(index, output, jobs):
  for name in ('roi', 'context', 'segments', 'candidates'):
    (output/name).mkdir(parents=True, exist_ok=True)
  results = []
  with ProcessPoolExecutor(max_workers=jobs) as executor:
    for result in executor.map(scan_source, [(source, str(output)) for source in index['sources']]):
      results.append(result)
      print(result['id'], result.get('excluded', ''), len(result['rows']), flush=True)
  rows = [row for result in results for row in result['rows']]
  (output/'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
  (output/'scan-summary.json').write_text(json.dumps({
    'segments': len(results), 'identity': dict(Counter(str(r['identity']) for r in results)),
    'counts': dict(sum((Counter(r['counts']) for r in results), Counter())),
    'selected_rois': len(rows), 'route_counts': dict(Counter(r['split'] for r in rows)),
    'policy': 'full decode, exact synchronized 5Hz candidate sweep; image-feature farthest-point diversity, up to32 ROIs/video; proposals only'
  }, indent=2), encoding='utf-8')


def review_pages(rows, root, output):
  """Context plus unmasked strip identity, not classifier predictions, for review."""
  output.mkdir(parents=True, exist_ok=True)
  for start in range(0, len(rows), 6):
    page = Image.new('RGB', (1800, 1050), '#202020')
    draw = ImageDraw.Draw(page)
    for i, row in enumerate(rows[start:start+6]):
      x, y = (i % 2)*900, (i//2)*350
      context = Image.open(root/row['context']).convert('RGB')
      w, h = context.size
      paint = ImageDraw.Draw(context)
      outline = np.asarray(row['roi_outline'])
      outline *= [w/row['video_size'][0], h/row['video_size'][1]]
      for edge in (0, 1):
        paint.line([tuple(p) for p in outline[:, edge]], fill='#00ff80', width=2)
      context.thumbnail((700, 310))
      page.paste(context, (x, y+35))
      page.paste(Image.open(root/row['image']).resize((100, 300)), (x+715, y+40))
      draw.text((x+5, y+3), f"{start+i} {row['id']} {row['split']}", fill='white')
      draw.text((x+5, y+18), f"v={row['speed_mps']:.1f} visible={row['observation']['observedLengthM']:.0f}m change={row['lane_change']}", fill='white')
    page.save(output/f'page-{start//6:03}.jpg', quality=95)


def paired_variants(rows, sources, root):
  """Extract the other recorded quality at the identical physical camera frame.

  This does not approve a label. HD context and each quality's actual strip
  still require review; synthetic downsampling is never called recorded qcamera.
  """
  import av
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  by_segment = defaultdict(list)
  for row in rows:
    by_segment[row['route'], row['segment']].append(row)
  result = []
  for source in sources:
    selected = by_segment[source['route'], source['segment']]
    if len(source['videos']) != 2 or not selected:
      continue
    identity, models, calibrations, _, cameras, encodes = metadata(source)
    times = [c[0] for c in calibrations]
    for quality in ('fcamera', 'qcamera'):
      targets = {(r['frame_id'], r['camera_eof_ns'], r['side']): r for r in selected if r['video_source'] != quality}
      needed = {index: key for index, key in encodes[quality].items() if any((key[0], key[1], side) in targets for side in ('left', 'right'))}
      if not needed:
        continue
      with av.open(source['videos'][quality]['path']) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
          if index not in needed:
            continue
          key = needed[index]
          model = models[key]
          rpy = calibrations[bisect_right(times, key[1])-1][1]
          camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
          k = camera.intrinsics.copy()
          k[0] *= frame.width/camera.width
          k[1] *= frame.height/camera.height
          rgb = frame.to_ndarray(format='rgb24')
          context = f"context/{source['id']}-{index}-{quality}.jpg"
          thumb = Image.fromarray(rgb)
          thumb.thumbnail((1008, 570))
          thumb.save(root/context, quality=94)
          for side, li in [('left', 1), ('right', 2)]:
            original = targets.get((key[0], key[1], side))
            if original is None:
              continue
            line = N(**model['lines'][li])
            maps = lane_maps(line, model['probs'][li], rpy, k, frame.width, frame.height)
            if maps is None:
              continue
            strip = rgb_strip(rgb, maps)
            clear, observation = visibility(maps, N(leadsV3=[N(**lead) for lead in model['leads']]), line, rpy, k)
            strip[~clear] = 0
            row = dict(original, id=original['scene_id']+'-'+quality, video_source=quality, video_index=index,
                       video_size=[frame.width, frame.height], context=context, observation=observation,
                       suggestion=classify_strip(strip), verified=False, type=None, color=None)
            row['roi_outline'] = np.stack((maps[0][::12, [0, -1]], maps[1][::12, [0, -1]]), -1).tolist()
            row['image'] = 'roi/'+row['id']+'.png'
            Image.fromarray(strip).save(root/row['image'])
            row['image_sha256'] = digest(root/row['image'])
            result.append(row)
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--pilot-sources', type=Path, required=True)
  parser.add_argument('--jobs', type=int, default=4)
  args = parser.parse_args()
  locked = inventory(args.root, args.output, args.pilot_sources)
  build(locked, args.output, args.jobs)
