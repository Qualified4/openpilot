"""PC-only synchronized log/video extraction and explicit label review queue."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, rgb_strip
from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip
from openpilot.selfdrive.carrot.lane_type.observation import visibility

ROOT = Path(__file__).resolve().parents[2]


def read_events(path):
  """Full schema, including modelV2 and encode indices; never the reduced DBC schema."""
  import capnp
  if os.name == 'nt':
    staging = Path(tempfile.mkdtemp(prefix='lane-schema-'))
    for directory in ('openpilot/cereal', 'opendbc_repo/opendbc/car'):
      for source in (ROOT/directory).rglob('*.capnp'):
        target = staging/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    c, car = staging/'openpilot/cereal', staging/'opendbc_repo/opendbc/car'
    parser = capnp.SchemaParser()
    log = parser.load(str(c/'log.capnp'), imports=[str(c), str(c/'include'), str(car), str(car/'include')])
  else:
    from openpilot.cereal import log
  try:
    import zstandard
    with path.open('rb') as f:
      raw = zstandard.ZstdDecompressor().stream_reader(f).read()
  except ImportError:
    raw = subprocess.check_output(['zstd', '-d', '-c', str(path)])
  yield from log.Event.read_multiple_bytes(raw, traversal_limit_in_words=2**30)


def validate_rows(rows, root=None):
  routes = defaultdict(set)
  image_splits = defaultdict(set)
  scene_splits = defaultdict(set)
  ids = set()
  for row in rows:
    if not isinstance(row.get('verified', False), bool):
      raise ValueError('verified must be an explicit boolean')
    if row['split'] not in ('train', 'validation', 'test') or not row['route']:
      raise ValueError('invalid route split')
    routes[row['route']].add(row['split'])
    if row.get('scene_id'):
      scene_splits[row['scene_id']].add(row['split'])
    if row['id'] in ids:
      raise ValueError('duplicate sample id')
    ids.add(row['id'])
    if row.get('verified'):
      image_splits[row['image_sha256']].add(row['split'])
    if root is not None and hashlib.sha256((root/row['image']).read_bytes()).hexdigest() != row['image_sha256']:
      raise ValueError('ROI image checksum mismatch')
    if row.get('verified'):
      if not row.get('reviewer') or not row.get('review_note') or row.get('label_source') != 'visual_review':
        raise ValueError('verified labels require explicit visual review provenance')
      if row.get('type') not in ('dashed', 'solid') or row.get('color') not in ('white', 'yellow'):
        raise ValueError('unconfirmed labels cannot be ground truth')
  if any(len(splits) != 1 for splits in routes.values()):
    raise ValueError('route leakage across train/validation/test')
  if any(len(splits) != 1 for splits in image_splits.values()):
    raise ValueError('duplicate ROI pixels across train/validation/test')
  if any(len(splits) != 1 for splits in scene_splits.values()):
    raise ValueError('paired scene leakage across train/validation/test')


def extract(sources, output, times, include_missing=False):
  import av
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  output.mkdir(parents=True, exist_ok=True)
  rows = []
  source_routes = defaultdict(set)
  for source in sources:
    source_routes[source['route']].add(source['split'])
  if any(len(v) != 1 for v in source_routes.values()):
    raise ValueError('source route leakage')
  for source in sources:
    video, path = Path(source['video']), Path(source['log'])
    models, calibrations, cameras, encodes, cars = [], [], [], [], []
    device = None
    for event in read_events(path):
      kind = event.which()
      if kind == 'initData':
        device = str(event.initData.deviceType)
      elif kind == 'modelV2' and event.valid:
        m = event.modelV2
        models.append({'eof': m.timestampEof, 'frame': m.frameId, 'probs': list(m.laneLineProbs),
                           'lines': [{'x': list(line.x), 'y': list(line.y), 'z': list(line.z)} for line in m.laneLines],
                           'leads': [{'prob': lead.prob, 'x': list(lead.x), 'y': list(lead.y)} for lead in m.leadsV3],
                           'lane_change': str(m.meta.laneChangeState) != 'off'})
      elif kind == 'carState':
        cars.append((event.logMonoTime, event.carState.vEgo, bool(event.carState.leftBlinker or event.carState.rightBlinker)))
      elif kind == 'liveCalibration' and event.valid and str(event.liveCalibration.calStatus) == 'calibrated':
        calibrations.append((event.logMonoTime, list(event.liveCalibration.rpyCalib)))
      elif kind == 'roadCameraState':
        cameras.append((event.logMonoTime, str(event.roadCameraState.sensor)))
      elif kind == ('roadEncodeIdx' if 'fcamera' in video.name else 'qRoadEncodeIdx'):
        e = getattr(event, kind)
        encodes.append({'index': e.segmentId, 'frame': e.frameId, 'eof': e.timestampEof})
    if not models or not calibrations or not encodes or not cameras:
      print('skip incomplete log/video metadata', source['id'], flush=True)
      continue
    models.sort(key=lambda m: m['eof'])
    model_times = [m['eof'] for m in models]
    encodes.sort(key=lambda e:e['index'])
    first_eof = encodes[0]['eof']
    selected = {min(encodes, key=lambda e:abs((e['eof']-first_eof)/1e9-t))['index'] for t in times}
    encode_by_index = {e['index']:e for e in encodes}
    camera = DEVICE_CAMERAS[(device, cameras[0][1])].fcam
    with av.open(str(video)) as container:
      for index, frame in enumerate(container.decode(video=0)):
        if index not in selected:
          continue
        encode = encode_by_index[index]
        t = encode['eof']
        candidates = models[max(0, bisect_left(model_times, t)-1):bisect_left(model_times, t)+1]
        model = min(candidates, key=lambda m:abs(m['eof']-t))
        if abs(model['eof']-t) > 75_000_000:
          continue
        previous = [c for c in calibrations if c[0] <= t]
        if not previous or t-previous[-1][0] > 2_000_000_000:
          continue
        rpy = previous[-1][1]
        rgb = frame.to_ndarray(format='rgb24')
        k = camera.intrinsics.copy()
        k[0] *= frame.width/camera.width
        k[1] *= frame.height/camera.height
        context = f"context/{source['id']}-{index}.jpg"
        (output/'context').mkdir(exist_ok=True)
        Image.fromarray(rgb).resize((672, 380)).save(output/context)
        for side, line_index in [('left', 1), ('right', 2)]:
          if len(model['lines']) <= line_index or len(model['probs']) <= line_index:
            continue
          line = SimpleNamespace(**model['lines'][line_index])
          maps = lane_maps(line, model['probs'][line_index], rpy, k, frame.width, frame.height)
          if maps is None and not include_missing:
            continue
          if maps is None:
            strip = np.zeros((192, 64, 3), np.uint8)
            observation = {'sufficient': False, 'observedLengthM': 0.0, 'occludedFraction': 1.0, 'visibilitySource': 'geometry_unavailable'}
          else:
            strip = rgb_strip(rgb, maps)
            clear, observation = visibility(maps, SimpleNamespace(leadsV3=[SimpleNamespace(**lead) for lead in model['leads']]), line, rpy, k)
            strip[~clear] = 0
          sample_id = f"{source['id']}-{index}-{side}"
          image = 'roi/'+sample_id+'.png'
          (output/'roi').mkdir(exist_ok=True)
          Image.fromarray(strip).save(output/image)
          suggestion = classify_strip(strip)
          near_y, far_y = np.interp([6, 46], line.x, line.y)
          car = min(cars, key=lambda c:abs(c[0]-t)) if cars else None
          rows.append({'id': sample_id, 'route': source['route'], 'segment': source['segment'], 'split': source['split'],
                           'image': image, 'context': context, 'image_sha256': hashlib.sha256((output/image).read_bytes()).hexdigest(),
                           'frame_id': encode['frame'], 'camera_eof_ns': t, 'model_eof_ns': model['eof'],
                           'model_probability': model['probs'][line_index], 'side': side,
                           'video_seconds': (t-first_eof)/1e9, 'video_source': video.name,
                           'curve': bool(abs(far_y-near_y) > 1.0), 'lane_change': model['lane_change'] or bool(car and car[2]), 'lighting': 'unreviewed',
                           'speed_mps': car[1] if car and abs(car[0]-t) < 200_000_000 else None, 'observation': observation,
                           'tracked': maps is not None,
                           'boundary': [float(np.interp(x, line.x, line.y)) for x in (10, 30)],
                           'suggestion': suggestion, 'verified': False, 'type': None, 'color': None,
                           'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if index >= max(selected):
          break
    print(source['id'], 'total samples', len(rows), flush=True)
  validate_rows(rows, output)
  (output/'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
  return rows


def sheets(rows, root):
  for page in range((len(rows)+23)//24):
    image = Image.new('RGB', (1440, 620), '#202020')
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows[page*24:(page+1)*24]):
      col, y = index%12, (index//12)*310
      image.paste(Image.open(root/row['image']).resize((96, 240)), (col*120+12, y+45))
      suggestion = row['suggestion']
      draw.text((col*120+3, y+2), f"{page*24+index}: {row['id'].split('-')[0]}", fill='white')
      draw.text((col*120+3, y+16), row['side']+' '+row['split'][:3], fill='white')
      draw.text((col*120+3, y+30), suggestion['type'][:3]+'/'+suggestion['color'][:3], fill='white')
    image.save(root/f'review-{page:02}.jpg', quality=95)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sources', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--times', type=float, nargs='+', default=[10, 30, 50])
  parser.add_argument('--include-missing', action='store_true', help='Record invalid geometry as unconfirmed instead of dropping that observation')
  args = parser.parse_args()
  records = extract(json.loads(args.sources.read_text(encoding='utf-8')), args.output, args.times, args.include_missing)
  sheets(records, args.output)
