"""Exact reviewed clips + annotated synthetic scenes; never starts training."""
import argparse
import base64
from bisect import bisect_right
import hashlib
import io
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.lane_type.dataset import read_events  # noqa: TID251 - offline repository tooling
from tools.lane_type.retrain_data import metadata  # noqa: TID251 - offline repository tooling
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import PaintTracker, road_points, rotation, to_local
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels


def write_json(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def synthetic_frame(pose, *, dashed=False, double=False, crosswalk=False, guide=False, shadow=False, occluded=False,
                    curved=False, white=False, paint_m=3., gap_m=5., phase_m=0.):
  h, w = 760, 1344
  k = np.array([[1141.5, 0, 672], [0, 1141.5, 380], [0, 0, 1.]])
  rpy = [0., -.024, 0.]
  matrix = k @ view_frame_from_device_frame @ rot_from_euler(rpy)
  x = np.arange(0., 80.1, .25)
  wx = np.arange(-10., 130., .05)
  wy = .0015*wx**2 if curved else np.zeros_like(wx)
  local = to_local(np.column_stack((wx, wy-1.6)), pose)
  order = np.argsort(local[:, 0])
  # Deliberate model error: tracker must use real pixels to correct 18cm.
  line = {'x': x.tolist(), 'y': (np.interp(x, local[order, 0], local[order, 1])+.18).tolist(), 'z': [1.3]*len(x)}
  yy, xx = np.mgrid[:h, :w]
  world = road_points(np.column_stack((xx.ravel(), yy.ravel())), line, matrix)[:, :2]
  world = world @ rotation(pose[2]).T+pose[:2]
  # Static world-anchored road texture supplies independent optical-flow cues.
  safe = np.nan_to_num(world, nan=0., posinf=0., neginf=0.)
  # Nonperiodic texture avoids a repeating checkerboard's optical-flow aliases.
  cell_x, cell_y = np.floor(safe[:, 0]*12), np.floor(safe[:, 1]*12)
  texture = 68+24*np.mod(np.sin(cell_x*12.9898+cell_y*78.233)*43758.5453, 1)
  rgb = np.repeat(texture.reshape(h, w, 1), 3, axis=2).astype(np.uint8)
  rgb[yy < 380] = (60, 80, 90)
  target = np.zeros((h, w), np.uint8)
  nuisance = np.zeros_like(target)

  def paint_polygon(points, color, mask):
    local = to_local(np.asarray(points), pose)
    xyz = np.column_stack((local, np.full(len(local), 1.3)))
    p = xyz @ matrix.T
    if (p[:, 2] <= .1).any():
      return
    polygon = np.rint(p[:, :2]/p[:, 2, None]).astype(np.int32)
    cv2.fillPoly(rgb, [polygon], color)
    cv2.fillPoly(mask, [polygon], 1)

  def stripe(offset, mask, diagonal=False, force_solid=False):
    # Render small connected source polygons, not warped training strips.
    for a in np.arange(-5., 100., .10):
      if dashed and mask is target and not force_solid and ((a+phase_m) % (paint_m+gap_m)) >= paint_m:
        continue
      if a < pose[0]+2:
        continue
      b = a+.11
      ya = (.0015*a*a if curved else 0)-1.6+offset
      yb = (.0015*b*b if curved else 0)-1.6+offset
      if diagonal:
        ya += .35*(a-pose[0]-12)
        yb += .35*(b-pose[0]-12)
      color = (235, 235, 230) if white else (235, 195, 35)
      paint_polygon([[a, ya-.065], [b, yb-.065], [b, yb+.065], [a, ya+.065]], color, mask)
  stripe(0., target)
  if double:
    stripe(-.28, target, force_solid=True)
  if guide:
    stripe(.55, nuisance, diagonal=True)
  if crosswalk:
    for y in np.arange(-4., 4., .9):
      paint_polygon([[12, y], [13.1, y], [13.1, y+.4], [12, y+.4]], (230, 230, 230), nuisance)
    target[nuisance.astype(bool)] = 0
  if shadow:
    sm = np.zeros_like(target)
    old = rgb.copy()
    paint_polygon([[14, -6], [16, -6], [16, 6], [14, 6]], (0, 0, 0), sm)
    rgb[sm.astype(bool)] = (old[sm.astype(bool)]*.3).astype(np.uint8)
  leads = []
  if occluded:
    points = np.array([[9., -2.], [9., -1.2]])
    p = np.column_stack((points, np.full(2, 1.3))) @ matrix.T
    p = p[:, :2]/p[:, 2, None]
    a, b = np.rint(p).astype(int)
    cv2.rectangle(rgb, (a[0], a[1]-120), (b[0], b[1]), (110, 110, 110), -1)
    cv2.rectangle(target, (a[0], a[1]-120), (b[0], b[1]), 0, -1)
    leads = [{'prob': 1., 'x': [9.], 'y': [-1.6]}]
  return rgb, line, k, rpy, target.astype(bool), nuisance.astype(bool), leads


def mask_metrics(predicted, target, nuisance, eligible):
  truth = target & eligible
  pred = predicted & eligible
  return {'target_pixels': int(truth.sum()), 'selected_pixels': int(pred.sum()),
          'target_included': int((pred & truth).sum()), 'nuisance_included': int((pred & nuisance).sum()),
          'paint_recall': float((pred & truth).sum()/truth.sum()) if truth.any() else None,
          'nuisance_fraction': float((pred & nuisance).sum()/pred.sum()) if pred.any() else None}


def preview(path, rgb, result, target=None, nuisance=None):
  source = Image.fromarray(rgb)
  mask = result['target_mask']
  overlay = rgb.copy()
  overlay[mask] = np.array([0, 255, 100])
  # Native-resolution panels; no transformed colour strips.
  panel = Image.new('RGB', (rgb.shape[1]*2, rgb.shape[0]+65), 'white')
  panel.paste(source, (0, 65))
  panel.paste(Image.fromarray(overlay), (rgb.shape[1], 65))
  d = ImageDraw.Draw(panel)
  d.text((10, 8), 'SOURCE (unchanged native pixels)', fill='black')
  d.text((rgb.shape[1]+10, 8), 'GREEN = selected target paint; other pixels stay unchanged', fill='black')
  d.text((10, 30), f"type={result['type']} color={result['color']} reason={result['reason']} tracks={len(result['tracks'])}", fill='black')
  if target is not None:
    d.text((rgb.shape[1]+10, 30), 'Masks/coordinates saved separately for pixel-level review', fill='black')
  path.parent.mkdir(parents=True, exist_ok=True)
  panel.save(path)
  Image.fromarray(mask.astype(np.uint8)*255).save(path.with_name(path.stem+'-selected.png'))
  if target is not None:
    Image.fromarray(target.astype(np.uint8)*255).save(path.with_name(path.stem+'-truth.png'))
  if nuisance is not None:
    Image.fromarray(nuisance.astype(np.uint8)*255).save(path.with_name(path.stem+'-nuisance.png'))


def synthetic(output, scenarios=None):
  summaries = []
  for scenario in scenarios or ('solid', 'dashed', 'dashed_slow', 'white_dashed', 'white_shadow', 'double', 'double_mixed',
                                'crosswalk', 'guide', 'shadow', 'occluded', 'curve', 'stopped'):
    tracker = PaintTracker('left')
    records = []
    for i in range(16):
      d = 0. if scenario == 'stopped' else i*(1. if scenario in ('dashed', 'double_mixed', 'white_dashed') else .6)
      pose = np.array([d, .0015*d*d if scenario == 'curve' else 0., np.arctan(.003*d) if scenario == 'curve' else 0.])
      rgb, line, k, rpy, target, nuisance, leads = synthetic_frame(pose, dashed=scenario in ('dashed', 'dashed_slow', 'double_mixed', 'white_dashed'),
        double=scenario in ('double', 'double_mixed'), crosswalk=scenario == 'crosswalk', guide=scenario == 'guide',
        shadow=scenario in ('shadow', 'white_shadow'), occluded=scenario == 'occluded', curved=scenario == 'curve', white=scenario.startswith('white'))
      result = tracker.update(rgb, line, 1., rpy, k, i*200_000_000, pose=pose, pose_error_m=.02, leads=leads)
      eligible = np.zeros(rgb.shape[:2], bool)
      for row in result['profile']:
        if row['geometry_visible']:
          eligible[row['y']] = True
      m = mask_metrics(result['target_mask'], target, nuisance, eligible)
      false_gap = source_paint = unknown = 0
      for row in result['profile']:
        if 'local_xy' not in row:
          continue
        # Synthetic true marking is defined in world distance, independently of
        # detector masks. Occlusion/shadow cannot change a solid into a true gap.
        world_x = float((np.asarray(row['local_xy']) @ rotation(pose[2]).T+pose[:2])[0])
        ink = scenario not in ('dashed', 'dashed_slow', 'double_mixed', 'white_dashed') or world_x % 8 < 3
        source_paint += ink
        false_gap += ink and row['ink'] == 0
        unknown += row['ink'] is None
      records.append(dict(frame=i, **m, false_gap_rows=int(false_gap), source_paint_rows=int(source_paint),
        unknown_rows=unknown, profile_rows=len(result['profile']), type=result['type'], color=result['color'],
        distinct_m=result['distinct_m'], gap_m=result['gap_m'], generation=result['generation'],
        tracking_error_m=result['tracking_error_m'], reason=result['reason']))
      records[-1].update(paint_runs=result.get('paint_runs'), verified_gap_runs=result.get('verified_gap_runs'))
      if i in (0, 8, 15):
        preview(output/'synthetic'/f'{scenario}-{i:02d}.png', rgb, result, target, nuisance)
    sums = {key: sum(r[key] for r in records) for key in ('target_pixels', 'target_included', 'selected_pixels', 'nuisance_included',
                                                        'source_paint_rows', 'false_gap_rows')}
    sums.update(scenario=scenario, frames=len(records), records=records,
      target_inclusion_fraction=sums['target_included']/max(1, sums['target_pixels']),
      nuisance_fraction=sums['nuisance_included']/max(1, sums['selected_pixels']),
      false_gap_fraction=sums['false_gap_rows']/max(1, sums['source_paint_rows']))
    summaries.append(sums)
    print(scenario, sums['target_inclusion_fraction'], sums['nuisance_fraction'], 'false gaps',sums['false_gap_rows'],
      'types',[r['type'] for r in records], flush=True)
  write_json(output/'synthetic.json', summaries)


def prepare_clips(inventory_path, output):
  inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
  # Fixed existing reviewed train/validation anchors, not randomly added logs.
  anchors = [('000003d8--26e55cbb04', 15, 18379), ('000003d8--26e55cbb04', 32, 38875),
             ('000003ea--6a2371afe5', 6, 8042), ('000003ea--6a2371afe5', 6, 8355),
             ('000003dd--26eb77e00d', 5, 7039)]
  clips = []
  for route, segment, anchor in anchors:
    name = f'{route}--{segment}-{anchor}'
    cache = output/'clips'/name/'frames.json'
    if cache.exists():
      clips.extend(json.loads(cache.read_text(encoding='utf-8')))
      continue
    source = next(s for s in inventory['sources'] if s['route'] == route and s['segment'] == segment)
    identity, models, calibrations, cars, cameras, encodes = metadata(source)
    poses = []
    for event in read_events(Path(source['log']['path'])):
      if event.which() == 'livePose':
        p = event.livePose
        poses.append({'stamp': int(p.timestamp), 'available': int(event.logMonoTime),
          'omega': [p.angularVelocityDevice.x, p.angularVelocityDevice.y, p.angularVelocityDevice.z],
          'omega_std': [p.angularVelocityDevice.xStd, p.angularVelocityDevice.yStd, p.angularVelocityDevice.zStd],
          'valid': bool(event.valid and p.inputsOK and p.sensorsOK and p.angularVelocityDevice.valid)})
    ct, st, pt = [c[0] for c in calibrations], [c[0] for c in cars], [p['available'] for p in poses]
    index = next(i for i, key in encodes['fcamera'].items() if key[0] == anchor)
    needed = set(range(max(0, index-28), min(1199, index+33), 4))
    rows = []
    xy, yaw, last_stamp, prev_speed, prev_rate = np.zeros(2), 0., None, None, None
    with av.open(source['videos']['fcamera']['path']) as container:
      container.streams.video[0].thread_count = 1
      for i, frame in enumerate(container.decode(video=0)):
        if i not in needed:
          continue
        key = encodes['fcamera'][i]
        if key not in models or key not in cameras:
          continue
        ci, si, pi = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1, bisect_right(pt, key[1])-1
        if min(ci, si, pi) < 0:
          continue
        rpy, car, p = calibrations[ci][1], cars[si], poses[pi]
        assert key[1]-ct[ci] <= 2_000_000_000 and key[1]-st[si] <= 200_000_000
        pose_age = key[1]-p['stamp']
        valid = p['valid'] and 0 <= pose_age <= 150_000_000
        rate = float((rot_from_euler(rpy).T @ p['omega'])[2])
        # Conservative upper bound for any rotated yaw-axis standard deviation.
        rate_std = float(np.linalg.norm(p['omega_std']))
        speed = float(car[1])
        dt = (key[1]-last_stamp)/1e9 if last_stamp is not None else 0.
        if last_stamp is not None:
          dyaw = .5*(rate+prev_rate)*dt
          distance = .5*(speed+prev_speed)*dt
          xy += distance*np.array([np.cos(yaw+dyaw/2), np.sin(yaw+dyaw/2)])
          yaw += dyaw
        # Explicit per-step uncertainty proxy; not a calibrated odometry bound.
        error = .02+20*rate_std*dt+.10*dt
        last_stamp, prev_speed, prev_rate = key[1], speed, rate
        camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
        k = camera.intrinsics.copy()
        k[0] *= frame.width/camera.width
        k[1] *= frame.height/camera.height
        rgb = frame.to_ndarray(format='rgb24')
        target = cache.parent/f'{key[0]}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(target)
        model = models[key]
        rows.append({'clip': name, 'route': route, 'segment': segment, 'split': source['split'], 'anchor': anchor,
          'frame_id': key[0], 'stamp': key[1], 'video_index': i, 'frame': str(target.resolve()), 'rpy': rpy,
          'intrinsics': k.tolist(), 'line': model['lines'][1], 'probability': model['probs'][1], 'leads': model['leads'],
          'pose': [*xy.tolist(), yaw], 'pose_error_m': error if valid else None, 'pose_age_ns': pose_age,
          'lane_change': bool(model['lane_change'] or car[2]), 'sync': 'exact_encode_frameId_EOF_modelV2',
          'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
    write_json(cache, rows)
    clips.extend(rows)
    print('prepared',name,len(rows),flush=True)
  write_json(output/'clips.json', clips)


def real(output):
  clips = json.loads((output/'clips.json').read_text(encoding='utf-8'))
  reports = []
  for clip in sorted({r['clip'] for r in clips}):
    tracker = PaintTracker('left')
    rows = [r for r in clips if r['clip'] == clip]
    for i, row in enumerate(rows):
      rgb = np.asarray(Image.open(row['frame']).convert('RGB'))
      result = tracker.update(rgb, row['line'], row['probability'], row['rpy'], np.array(row['intrinsics']), row['stamp'],
        pose=row['pose'], pose_error_m=row['pose_error_m'], road_pose=row.get('road_pose'), leads=row['leads'], lane_change=row['lane_change'])
      path = output/'real'/f"{clip}-{row['frame_id']}"
      path.parent.mkdir(parents=True, exist_ok=True)
      xy = np.column_stack(np.nonzero(result['target_mask']))[:, ::-1]
      np.savez_compressed(path.with_suffix('.npz'), selected_xy=xy, selected_rgb=rgb[xy[:, 1], xy[:, 0]])
      info = {k: v for k, v in result.items() if k not in ('target_mask',)}
      info.update(clip=clip, frame_id=row['frame_id'], source=row['frame'], stamp=row['stamp'], selected_pixels=len(xy))
      reports.append(info)
      if i in (0, 7, len(rows)-1) or row['frame_id'] == row['anchor']:
        preview(path.with_suffix('.png'), rgb, result)
    print('replayed',clip,len(rows),flush=True)
  write_json(output/'real-results.json', reports)


def score(output):
  """Partial source masks and state counts; physical registration is audited separately."""
  rows = json.loads((output/'clips.json').read_text(encoding='utf-8'))
  results = json.loads((output/'real-results.json').read_text(encoding='utf-8'))
  annotation_file = output/'annotation/paint-revised.json'
  if not annotation_file.exists():
    annotation_file = output/'annotation/paint.json'
  annotations = json.loads(annotation_file.read_text(encoding='utf-8'))
  traces = []
  for a in annotations:
    row = next(r for r in rows if r['frame_id'] == a['frame_id'])
    result = next(r for r in results if r['frame_id'] == a['frame_id'])
    rgb = np.asarray(Image.open(row['frame']).convert('RGB'))
    masks = {}
    for key in ('paint_polygons', 'nuisance_polygons'):
      mask = np.zeros(rgb.shape[:2], np.uint8)
      for polygon in a[key]:
        points = np.rint(np.asarray(polygon)*np.array([rgb.shape[1], rgb.shape[0]])/a['coordinate_size']).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
      masks[key] = mask
    xy = np.load(output/'real'/f"{row['clip']}-{row['frame_id']}.npz")['selected_xy']
    selected = np.zeros(rgb.shape[:2], bool)
    selected[xy[:, 1], xy[:, 0]] = True
    narrow = collect_pixels(rgb, row['line'], 1., row['rpy'], np.array(row['intrinsics']))
    baseline = np.zeros(rgb.shape[:2], bool)
    baseline[narrow['xy'][:, 1], narrow['xy'][:, 0]] = True
    eligible = np.zeros(rgb.shape[:2], bool)
    # Use the same available geometry for both selectors, even on abstention.
    wide = collect_pixels(rgb, row['line'], 1., row['rpy'], np.array(row['intrinsics']), half_width_m=.9)
    for r in wide['rows']:
      if r['geometry_visible']:
        eligible[r['y']] = True
    paint, nuisance = masks['paint_polygons'].astype(bool), masks['nuisance_polygons'].astype(bool)
    nominal = mask_metrics(selected, paint, nuisance, eligible)
    core = cv2.erode(masks['paint_polygons'], np.ones((3, 3), np.uint8), iterations=a['tolerance_px']).astype(bool)
    core_metrics = mask_metrics(selected, core, nuisance, eligible)
    false_gap = sum(r['ink'] == 0 and (core[r['y']] & eligible[r['y']]).any() for r in result['profile'])
    known_paint_rows = int(np.any(core & eligible, axis=1).sum())
    traces.append({'frame_id': a['frame_id'], 'split': row['split'], 'annotation_scope': a['scope'],
      'nominal': nominal, 'interior_core': core_metrics, 'false_gap_rows_on_known_target_paint': int(false_gap),
      'known_target_paint_rows': known_paint_rows, 'false_gap_fraction': false_gap/known_paint_rows if known_paint_rows else None,
      'baseline': mask_metrics(baseline, paint, nuisance, eligible),
      'unknown_rows': sum(r['ink'] is None for r in result['profile']), 'profile_rows': len(result['profile']),
      'reason': result['reason'], 'tracks': result['tracks']})
    preview(output/'anchor-review'/f"{row['frame_id']}.png", rgb, dict(result, target_mask=selected), paint, nuisance)
  flow = []
  for clip in sorted({r['clip'] for r in rows}):
    source = [r for r in rows if r['clip'] == clip]
    measurements = [r for r in results if r['clip'] == clip]
    resets = sum(a['generation'] != b['generation'] for a, b in zip(measurements[:-1], measurements[1:], strict=True))
    flow.append({'clip': clip, 'frames': len(source),
      'track_resets': resets, 'no_path_frames': sum(not r['tracks'] for r in measurements),
      'confirmed_type_frames': sum(r['type'] != 'unknown' for r in measurements),
      'type_state_switches': sum(a['type'] != b['type'] for a, b in zip(measurements[:-1], measurements[1:], strict=True))})
  write_json(output/'quality.json', {'manual_source_traces': traces, 'track_summary': flow, 'annotation_file': str(annotation_file),
    'annotation_limitation': 'Partial single-assistant source traces, not independent human approval; unspecified pixels are not negative truth.',
    'physical_registration_audit': 'audit_pixel_alignment.py; long-solid LK residual is not a physical accuracy measurement',
    'model_promotion_allowed': False, 'gpu_training_resumed': False})
  print(json.dumps({'traces': traces, 'flow': flow}, indent=2))
  gallery(output, rows, results)


def gallery(output, rows, results):
  entries = []
  for row in rows:
    result = next(r for r in results if r['clip'] == row['clip'] and r['frame_id'] == row['frame_id'])
    xy = np.load(output/'real'/f"{row['clip']}-{row['frame_id']}.npz")['selected_xy']
    with Image.open(row['frame']) as im:
      w, h = im.size
    mask = np.zeros((h, w, 4), np.uint8)
    mask[xy[:, 1], xy[:, 0]] = (0, 255, 100, 210)
    stream = io.BytesIO()
    Image.fromarray(mask).save(stream, format='PNG')
    entries.append({'clip': row['clip'], 'frame': row['frame_id'], 'source': os.path.relpath(row['frame'], output.resolve()).replace('\\', '/'),
      'mask': 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'), 'type': result['type'], 'color': result['color'],
      'reason': result['reason'], 'distinct_m': result['distinct_m'], 'generation': result['generation']})
  html = '''<!doctype html><meta charset="utf-8"><title>Pixel Ribbon v2 도색 추적 검수</title>
<style>body{font:16px sans-serif;background:#eee} .panels{display:flex;gap:10px;overflow:auto}
.pane{position:relative;width:48%;flex-shrink:0}.pane img{width:100%;display:block}.overlay{position:absolute;top:0;left:0}
pre{white-space:pre-wrap}input{width:65vw}.native .pane{width:auto}.native .pane img{width:auto}</style>
<h2>원본 도색과 선택 결과 — 학습·모델 승격 보류</h2>
<p>왼쪽은 원본 PNG, 오른쪽 초록색은 선택한 원본 픽셀입니다. 색을 덮은 화면은 검수용이며 모델 입력이 아닙니다.</p>
<select id="clip"></select><button id="play">재생</button><button id="native">원본 크기 전환</button>
<input type="range" id="frame" min="0" value="0"><pre id="info"></pre>
<div class="panels"><div class="pane"><img id="source"></div><div class="pane"><img id="copy"><img class="overlay" id="mask"></div></div>
<p>UNKNOWN은 공백이 아닙니다. 일부 도색 누락·추적 단절 및 실사 ego 정렬 미검증이 남아 있습니다. 실제 CAN 연결 없음.</p>
<script>const rows=PAYLOAD;const sel=document.querySelector('#clip'),slider=document.querySelector('#frame');let timer=null;
for(const clip of [...new Set(rows.map(r=>r.clip))]){let o=document.createElement('option');o.text=clip;sel.add(o)}
function update(){let a=rows.filter(r=>r.clip===sel.value);slider.max=a.length-1;let r=a[Number(slider.value)];
document.querySelector('#source').src=r.source;document.querySelector('#copy').src=r.source;document.querySelector('#mask').src=r.mask;
document.querySelector('#info').textContent=JSON.stringify({...r,mask:undefined},null,2)}
sel.onchange=()=>{slider.value=0;update()};slider.oninput=update;
document.querySelector('#native').onclick=()=>document.body.classList.toggle('native');
document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}
timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);update()},200)};update();</script>'''
  (output/'PREVIEW.html').write_text(html.replace('PAYLOAD', json.dumps(entries)), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, default=Path('.analysis/pixel-ribbon-tracking-20261018'))
  p.add_argument('--synthetic', action='store_true')
  p.add_argument('--scenario', action='append')
  p.add_argument('--prepare', action='store_true')
  p.add_argument('--real', action='store_true')
  p.add_argument('--score', action='store_true')
  p.add_argument('--inventory', type=Path, default=Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json'))
  args = p.parse_args()
  if args.synthetic:
    synthetic(args.output, args.scenario)
  if args.prepare:
    prepare_clips(args.inventory, args.output)
  if args.real:
    real(args.output)
  if args.score:
    score(args.output)
