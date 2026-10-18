"""Summarize frozen offline experiments, with no inference or training."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels
from tools.lane_type.validate_hybrid_tracking import OUTPUT, read  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import mask_metrics, write_json  # noqa: TID251


def counts(rows):
  return {'frames': len(rows), 'type': dict(Counter(r['type'] for r in rows)),
          'color': dict(Counter(r['color'] for r in rows)),
          'qualified_frames': sum(r.get('registration_qualified', False) for r in rows),
          'type_confirmation_fraction': sum(r['type'] != 'unknown' for r in rows)/max(1, len(rows)),
          'color_confirmation_fraction': sum(r['color'] != 'unknown' for r in rows)/max(1, len(rows)),
          'alternative_type': dict(Counter(r.get('alternative_type', r['type']) for r in rows))}


def summarize(output):
  rows = read(output/'frames.json')
  folders = [output/f'{mode}-registered-sides2' for mode in ('5hz', '20hz', 'hybrid')]
  folders += [output/f'{mode}-legacy-sides1' for mode in ('5hz', '20hz')]
  result = {}
  for folder in folders:
    records = read(folder/'results.json')
    left = [r for r in records if r['side'] == 'left']
    events = read(folder/'failure-events.json')
    clips = []
    for clip in sorted({r['clip'] for r in left}):
      group = [r for r in left if r['clip'] == clip]
      onset = next((r['stamp'] for r in group if r['type'] == 'dashed'), None)
      clips.append(dict(clip=clip, **counts(group),
        first_dashed_since_clip_start_s=(onset-group[0]['stamp'])/1e9 if onset else None,
        type_transitions_including_unknown=sum(a['type'] != b['type'] for a, b in zip(group[:-1], group[1:], strict=True))))
    timing = read(folder/'timing.json')
    budgets = []
    budget = 200 if folder.name.startswith('5hz') else 50
    for repetition in timing:
      finish, clip, delays = 0., None, []
      for r in repetition:
        if r['clip'] != clip:
          clip, finish, index = r['clip'], 0., 0
        arrival = index*budget
        finish = max(finish, arrival)+r['total_wall_ms']
        delays.append(max(0., finish-arrival-budget))
        index += 1
      budgets.append({'service_time_exceeds_period': sum(r['total_wall_ms'] > budget for r in repetition),
        'frames': len(repetition), 'simulated_deadline_misses': sum(d > 0 for d in delays),
        'simulated_max_backlog_ms': max(delays)})
    phase_costs = {}
    for precise in (True, False):
      values = [r['total_wall_ms'] for rep in timing for r in rep if r['precise'] == precise]
      if values:
        phase_costs['precise' if precise else 'narrow'] = {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)),
                                                        'p99': float(np.percentile(values, 99))}
    result[folder.name] = {'left': counts(left), 'left_common_80': counts([r for r in left if r['common_5hz_time']]),
      'right_unlabelled': counts([r for r in records if r['side'] == 'right']), 'clips': clips,
      'reset_calls_left': sum(r['reset_calls'] for r in events if r['side'] == 'left'),
      'failure_events_left': sum(r['failure_event_count'] for r in events if r['side'] == 'left'),
      'reset_calls_both': sum(r['reset_calls'] for r in events), 'failure_events_both': sum(r['failure_event_count'] for r in events),
      'performance': read(folder/'performance.json'), 'phase_wall_ms': phase_costs,
      'deadline_simulation': {'nominal_period_ms': budget, 'is_actual_frame_drop_measurement': False, 'repetitions': budgets}}
  synthetic = []
  for scenario in read(output/'synthetic-comparison.json'):
    for mode in ('5hz', '20hz', 'hybrid'):
      group = [r for r in scenario['records'] if r['mode'] == mode]
      color = [r for r in group if r['color'] != 'unknown']
      synthetic.append({'scenario': scenario['scenario'], 'mode': mode, **counts(group),
        'false_dashed': sum(r['false_dashed'] for r in group), 'false_gap_rows': sum(r['false_gap_rows'] for r in group),
        'alternative_false_solid': sum(r['alternative_false_solid'] for r in group),
        'color_correct': sum(r['color'] == r['truth_color'] for r in color), 'color_confirmed': len(color),
        'color_selective_accuracy': sum(r['color'] == r['truth_color'] for r in color)/len(color) if color else None,
        'target_pixels': sum(r['target_pixels'] for r in group), 'paint_included': sum(r['paint_included'] for r in group),
        'nuisance_selected': sum(r['nuisance_selected'] for r in group),
        'first_confirmation_s': next((r['time_s'] for r in group if r['type'] != 'unknown'), None),
        'max_distinct_m': max(r['distinct_m'] for r in group),
        'max_continuous_paint_m': max(r['continuous_paint_m'] for r in group)})
  write_json(output/'comparison.json', {'experiments': result, 'synthetic': synthetic,
    'real_accuracy_precision_recall_F1': None, 'real_label_limitation': 'No independent sequence-level type/color/endpoint/occlusion truth',
    'comma_CPU_latency_RSS_modeld_controlsd_frame_drops': None, 'training_resumed': False, 'models_promoted': False})
  annotations = read(Path('.analysis/pixel-ribbon-tracking-20261018/annotation/paint-revised.json'))
  quality = []
  for annotation in annotations:
    row = next(r for r in rows if r['frame_id'] == annotation['frame_id'])
    with Image.open(row['frame']) as im:
      shape = (im.height, im.width)
    masks = {}
    for key in ('paint_polygons', 'nuisance_polygons'):
      mask = np.zeros(shape, np.uint8)
      for polygon in annotation[key]:
        points = np.rint(np.asarray(polygon)*np.array(shape[::-1])/annotation['coordinate_size']).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
      masks[key] = mask
    core = cv2.erode(masks['paint_polygons'], np.ones((3, 3), np.uint8), iterations=annotation['tolerance_px']).astype(bool)
    geometry = collect_pixels(np.zeros((*shape, 3), np.uint8), row['line'], 1., row['rpy'], np.array(row['intrinsics']),
                              half_width_m=.9, include_pixels=False)
    eligible = np.zeros(shape, bool)
    for item in geometry['rows']:
      if item['geometry_visible']:
        eligible[item['y']] = True
    for folder in folders:
      record = next(r for r in read(folder/'results.json') if r['clip'] == row['clip'] and
                    r['frame_id'] == row['frame_id'] and r['side'] == 'left')
      xy = np.load(folder/'masks'/f'{row["clip"]}-{row["frame_id"]}-left.npz')['selected_xy']
      selected = np.zeros(shape, bool)
      selected[xy[:, 1], xy[:, 0]] = True
      quality.append({'frame_id': row['frame_id'], 'experiment': folder.name,
        'nominal': mask_metrics(selected, masks['paint_polygons'].astype(bool), masks['nuisance_polygons'].astype(bool), eligible),
        'core': mask_metrics(selected, core, masks['nuisance_polygons'].astype(bool), eligible),
        'known_core_rows': int(np.any(core & eligible, axis=1).sum()),
        'false_gap_rows': int(sum(p['ink'] == 0 and (core[p['y']] & eligible[p['y']]).any() for p in record['profile']))})
  write_json(output/'partial-trace-quality.json', {'limitation': 'Five partial assistant traces; no independent approval, untraced pixels not negatives',
                                                 'records': quality})
  print(json.dumps({name: {'counts': r['left_common_80'], 'CPU': r['performance']['total_wall_ms']} for name, r in result.items()}, indent=2))


def preview(output):
  rows = read(output/'frames.json')
  folders = [output/f'{mode}-registered-sides2' for mode in ('5hz', '20hz', 'hybrid')]
  lookup = {folder.name: {(r['clip'], r['frame_id']): r for r in read(folder/'results.json') if r['side'] == 'left'} for folder in folders}
  entries = []
  for row in rows:
    nv12 = np.load(row['nv12'])
    rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
    # A native crop for viewing only; the experiment still examines source pixels.
    geo = collect_pixels(rgb, row['line'], 1., row['rpy'], np.array(row['intrinsics']), half_width_m=.9, include_pixels=False)
    visible = [r for r in geo['rows'] if r['geometry_visible']]
    if not visible:
      continue
    a, b = max(0, min(r['left_x'] for r in visible)-20), max(0, min(r['y'] for r in visible)-10)
    c, d = min(rgb.shape[1], max(r['right_x'] for r in visible)+21), min(rgb.shape[0], max(r['y'] for r in visible)+11)
    name = f'{row["clip"]}-{row["frame_id"]}'
    source = output/'preview'/f'{name}-source.png'
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[b:d, a:c]).save(source)
    methods = {}
    for folder in folders:
      record = lookup[folder.name].get((row['clip'], row['frame_id']))
      if record is None:
        methods[folder.name] = {'status': '5Hz observation absent at this timestamp'}
        continue
      xy = np.load(folder/'masks'/f'{name}-left.npz')['selected_xy']
      mask = np.zeros((*rgb.shape[:2], 4), np.uint8)
      mask[xy[:, 1], xy[:, 0]] = (0, 255, 100, 210)
      path = output/'preview'/f'{name}-{folder.name}.png'
      Image.fromarray(mask[b:d, a:c]).save(path)
      methods[folder.name] = {k: record.get(k) for k in ('type', 'color', 'reason', 'distinct_m', 'generation', 'registration', 'precise')}
      methods[folder.name]['mask'] = 'preview/'+path.name
    entries.append({'clip': row['clip'], 'frame': row['frame_id'], 'stamp': row['stamp'], 'origin_xy': [int(a), int(b)],
      'source': 'preview/'+source.name, 'original_PyAV_reference': os.path.relpath(row['frame'], output.resolve()).replace('\\', '/'),
      'methods': methods})
  html = '''<!doctype html><meta charset="utf-8"><title>5Hz / 20Hz / Hybrid native pixel comparison</title>
<style>body{font:16px sans-serif;background:#eee}#panes{display:flex;gap:10px;overflow:auto}.pane{position:relative}
.overlay{position:absolute;top:0;left:0}pre{white-space:pre-wrap}input{width:60vw}</style>
<h2>5Hz 정밀 / 20Hz 정밀 / 5Hz 정밀+20Hz 경량 추적</h2>
<p>원본 좌표의 동일 크기 crop입니다. 녹색은 선택 픽셀이며 UNKNOWN은 공백이 아닙니다. NV12 변환 입력, 학습·CAN 연결 없음.</p>
<select id="clip"></select><select id="method"></select><button id="play">재생/정지</button>
<input id="frame" type="range" min="0" value="0"><pre id="info"></pre>
<div id="panes"><div><img id="source"></div><div class="pane"><img id="copy"><img id="mask" class="overlay"></div></div>
<script>const rows=PAYLOAD;
const clip=document.querySelector('#clip'),method=document.querySelector('#method'),slider=document.querySelector('#frame');let timer;
for(const c of [...new Set(rows.map(r=>r.clip))]){const o=document.createElement('option');o.text=c;clip.add(o)}
for(const c of Object.keys(rows[0].methods)){const o=document.createElement('option');o.text=c;method.add(o)}
function update(){const group=rows.filter(r=>r.clip===clip.value);slider.max=group.length-1;
const r=group[Number(slider.value)],m=r.methods[method.value];
document.querySelector('#source').src=r.source;document.querySelector('#copy').src=r.source;
document.querySelector('#mask').style.display=m.mask?'block':'none';if(m.mask)document.querySelector('#mask').src=m.mask;
document.querySelector('#info').textContent=JSON.stringify({...r,methods:undefined,method:method.value,result:m},null,2)}
clip.onchange=()=>{slider.value=0;update()};method.onchange=update;slider.oninput=update;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);update()},50)}};update();</script>'''
  (output/'PREVIEW.html').write_text(html.replace('PAYLOAD', json.dumps(entries)), encoding='utf-8')


def preservation(output):
  before = read(output/'protected-before.json')
  def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
      for block in iter(lambda: f.read(4*1024*1024), b''):
        h.update(block)
    return h.hexdigest()
  changed = [name for name, expected in before.items() if not Path(name).is_file() or digest(Path(name)) != expected]
  expected = {'openpilot/selfdrive/carrot/lane_type/pixel_ribbon.py', 'openpilot/selfdrive/carrot/lane_type/pixel_tracker.py',
              'tools/lane_type/validate_pixel_tracking.py', 'tools/lane_type/README.md'}
  unexpected = [name for name in changed if name.replace('\\', '/') not in expected]
  write_json(output/'preservation.json', {'protected_files': len(before), 'intentional_changes': changed,
    'unexpected_changes': unexpected, 'active_model_sha256': digest(Path('openpilot/selfdrive/carrot/lane_type/assets/lane_type.onnx')),
    'training_resumed': False, 'new_models_promoted': False})
  assert not unexpected, unexpected


if __name__ == '__main__':
  cv2.setNumThreads(1)
  summarize(OUTPUT)
  preview(OUTPUT)
  preservation(OUTPUT)
