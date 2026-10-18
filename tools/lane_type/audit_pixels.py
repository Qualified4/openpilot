"""Reproduce pixel correspondence failures; never trains or changes source data."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from openpilot.common.transformations.camera import get_view_frame_from_calib_frame
from openpilot.selfdrive.carrot.lane_type.dual_roi import project
from openpilot.selfdrive.carrot.lane_type.native import native_observation
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels, source_box


def legacy_native_map(observation, head):
  """Reconstruct v1's exact selected original pixels, not an approximate overlay."""
  h = observation['heads'][head]
  rows = [r for r in observation['profile'] if h['near']-1e-8 <= r['distance'] <= h['far']+1e-8]
  count = h['image'].shape[0]
  selected = np.rint(np.linspace(0, len(rows)-1, count)).astype(int)[::-1]
  result = []
  for i in selected:
    r = rows[i]
    # a and b are recovered by projecting this native source row below.
    result.append(r)
  return result


def native_xy(observation, head, line, rpy, k, width):
  distances = np.arange(4.5, 25.001, .025)
  mx, _ = project(line, rpy, k, distances)
  coords = []
  for r in legacy_native_map(observation, head):
    a = int(np.ceil(np.interp(r['distance'], distances, mx[:, 0])))
    b = int(np.floor(np.interp(r['distance'], distances, mx[:, -1])))
    center = (a+b)/2+(r['centerOffsetM'] or 0.)*(b-a+1)/.9
    half = min((b-a)/2, width-1-center, center)
    xs = np.rint(np.linspace(center-half, center+half, 16)).astype(int)
    coords.append(np.column_stack((xs, np.full(16, r['imageRow']))))
  return np.stack(coords)


def summarize(values):
  return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
          'p95': float(np.percentile(values, 95)) if values else None,
          'max': float(np.max(values)) if values else None}


def viewer(path, rgb, observation, maps):
  def png(im):
    stream = io.BytesIO()
    Image.fromarray(im).save(stream, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')
  heads = {name: {'image': png(h['image']), 'xy': maps[name].tolist(), 'rgb': h['image'].tolist()}
           for name, h in observation['heads'].items()}
  payload = json.dumps({'source': png(rgb), 'heads': heads})
  html = '''<!doctype html><meta charset="utf-8"><title>원본 픽셀 대응 감사</title>
<style>body{font:16px sans-serif;background:#eee}canvas{border:1px solid #777;image-rendering:pixelated;vertical-align:top}
#source{max-width:76vw;height:auto}#strip{margin-left:12px}pre{white-space:pre-wrap}</style>
<h2>원본 ↔ 기존 v1 입력 픽셀 대응</h2>
<p>학습 보류. 원본 또는 입력을 클릭하면 대응 픽셀과 RGB를 확인합니다. 확대 화면은 검수용이며 새 학습 입력이 아닙니다.</p>
<select id="head"></select><pre id="info">로딩 중</pre><canvas id="source"></canvas><canvas id="strip"></canvas>
<script>const data=PAYLOAD;
const sc=document.querySelector('#source'),st=document.querySelector('#strip'),sel=document.querySelector('#head'),info=document.querySelector('#info');
const ctx=sc.getContext('2d'),sctx=st.getContext('2d');let image=new Image(),strip=new Image();
for(const name of Object.keys(data.heads)){let option=document.createElement('option');option.text=name;sel.add(option)}
function draw(){ctx.drawImage(image,0,0);sctx.imageSmoothingEnabled=false;sctx.drawImage(strip,0,0,st.width,st.height)}
function choose(){const h=data.heads[sel.value];if(!h)return;strip=new Image();strip.onload=()=>{st.width=192;st.height=h.xy.length*9;draw()};strip.src=h.image}
function mark(y,x,requested){let h=data.heads[sel.value],p=h.xy[y][x];draw();ctx.strokeStyle='red';ctx.lineWidth=2;ctx.strokeRect(p[0]-5,p[1]-5,10,10);
sctx.strokeStyle='red';sctx.lineWidth=2;sctx.strokeRect(x*12,y*9,12,9);
info.textContent=`입력 [row=${y}, col=${x}] → 원본 [x=${p[0]}, y=${p[1]}], RGB=${h.rgb[y][x]}`;
if(requested)info.textContent+=` / 클릭한 원본=${requested}; 직접 대응 여부=${p[0]===requested[0]&&p[1]===requested[1]}`;}
st.onclick=e=>{let h=data.heads[sel.value],r=st.getBoundingClientRect();
mark(Math.min(h.xy.length-1,Math.floor((e.clientY-r.top)/r.height*h.xy.length)),Math.min(15,Math.floor((e.clientX-r.left)/r.width*16)))};
sc.onclick=e=>{let r=sc.getBoundingClientRect();
let p=[Math.floor((e.clientX-r.left)/r.width*sc.width),Math.floor((e.clientY-r.top)/r.height*sc.height)],h=data.heads[sel.value],best=[Infinity,0,0];
for(let y=0;y<h.xy.length;y++)for(let x=0;x<16;x++){let q=h.xy[y][x],d=(q[0]-p[0])**2+(q[1]-p[1])**2;if(d<best[0])best=[d,y,x]}mark(best[1],best[2],p)};
sel.onchange=choose;image.onload=()=>{sc.width=image.width;sc.height=image.height;choose()};image.src=data.source;
</script>'''
  path.write_text(html.replace('PAYLOAD', payload), encoding='utf-8')


def synthetic_audit(output):
  """Uncompressed, known continuous paint isolates transformation-only errors."""
  k = np.array([[550., 0, 320], [0, 550, 180], [0, 0, 1.]])
  line = {'x': [0, 5, 10, 20, 30, 50], 'y': [-1.6]*6, 'z': [1.3]*6}
  results = []
  for width in (.035, .05, .10, .15):
    rgb = np.full((360, 640, 3), 80, np.uint8)
    distances = np.linspace(4.5, 25, 1000)
    mx, my = project(line, [0, 0, 0], k, distances, half_width=width/2, columns=2)
    polygon = np.r_[np.column_stack((mx[:, 0], my[:, 0])), np.column_stack((mx[::-1, 1], my[::-1, 1]))]
    cv2.fillPoly(rgb, [np.rint(polygon).astype(np.int32)], (230, 190, 30))
    observation = native_observation(rgb, line, 1., [0, 0, 0], k)
    ribbon = collect_pixels(rgb, line, 1., [0, 0, 0], k)
    retained = np.zeros(rgb.shape[:2], bool)
    retained[ribbon['xy'][:, 1], ribbon['xy'][:, 0]] = True
    source_paint = rgb[..., 0] > 200
    rr = {'paint_width_m': width, 'source': 'synthetic uncompressed continuous yellow solid; no codec', 'heads': {}}
    for head, h in observation['heads'].items():
      pixels = native_xy(observation, head, line, [0, 0, 0], k, rgb.shape[1])
      original = np.array([source_paint[y].any() for y in pixels[:, 0, 1]])
      lost = original & ~(h['image'][..., 0] > 200).any(1)
      rr['heads'][head] = {'source_rows_with_paint': int(original.sum()), 'v1_rows_with_no_remaining_paint': int(lost.sum()),
        'v1_manufactured_missing_paint_row_fraction': float(lost.sum()/original.sum())}
    valid_rows = np.array([r['y'] for r in ribbon['rows'] if r['geometry_visible']])
    rr['ribbon_paint_pixels_lost_in_visible_rows'] = int((source_paint[valid_rows] & ~retained[valid_rows]).sum())
    Image.fromarray(rgb).save(output/f'synthetic-{width:g}m-source.png')
    for head, h in observation['heads'].items():
      Image.fromarray(h['image']).save(output/f'synthetic-{width:g}m-v1-{head}.png')
    results.append(rr)
  return results


def run(old_root, output):
  output.mkdir(parents=True, exist_ok=True)
  records = json.loads((old_root/'audit.json').read_text(encoding='utf-8'))
  annotations = {r['id']: r for r in json.loads((old_root/'annotation/paint-annotations.json').read_text(encoding='utf-8'))}
  results, traced, errors, losses = [], [], [], []
  for row in records:
    rgb = np.asarray(Image.open(old_root/row['frame']).convert('RGB'))
    line, k, rpy = row['line'], np.asarray(row['intrinsics']), row['rpy']
    observation = native_observation(rgb, line, 1., rpy, k, row['model_leads'])
    ribbon = collect_pixels(rgb, line, 1., rpy, k, row['model_leads'])
    xy = ribbon['xy']
    assert np.array_equal(ribbon['rgb'], rgb[xy[:, 1], xy[:, 0]])
    ds = np.linspace(4.5, 20., 192)
    mx, my = project(line, rpy, k, ds)
    # Independently use the canonical camera/UI convention with no extra height.
    points = np.column_stack((ds, np.interp(ds, line['x'], line['y']), np.interp(ds, line['x'], line['z']), np.ones(len(ds))))
    canonical = points @ (k @ get_view_frame_from_calib_frame(*rpy, 0.)).T
    canonical = canonical[:, :2]/canonical[:, 2, None]
    cx, cy = project(line, rpy, k, ds, columns=1, half_width=0.)
    error = float(np.max(np.abs(canonical-np.column_stack((cx[:, 0], cy[:, 0])))))
    errors.append(error)
    native_heads, maps = {}, {}
    for head, h in observation['heads'].items():
      pixels = native_xy(observation, head, line, rpy, k, rgb.shape[1])
      assert np.array_equal(h['image'], rgb[pixels[..., 1], pixels[..., 0]])
      maps[head] = pixels
      source_rows = {int(p[1]) for p in pixels.reshape(-1, 2)}
      available = [r for r in observation['profile'] if h['near'] <= r['distance'] <= h['far']]
      native_heads[head] = {'sampled_rows': len(source_rows), 'available_rows': len(available),
        'sampled_pixels': int(pixels.shape[0]*pixels.shape[1]),
        'mean_source_column_step_px': float(np.mean(np.diff(pixels[..., 0], axis=1))),
        'center_offset_adjacent_row_max_m': max((abs(a['centerOffsetM']-b['centerOffsetM'])
          for a, b in zip(available[:-1], available[1:], strict=True)
          if a['centerOffsetM'] is not None and b['centerOffsetM'] is not None), default=None),
        'source_row_omission_fraction': 1-len(source_rows)/len(available)}
      native_heads[head]['effective_center_reset_max_m'] = max((abs((a['centerOffsetM'] or 0.)-(b['centerOffsetM'] or 0.))
          for a, b in zip(available[:-1], available[1:], strict=True)), default=0.)
      losses.append(native_heads[head]['source_row_omission_fraction'])
    entry = {'id': row['id'], 'split': row['split'], 'source_shape': list(rgb.shape),
      'source_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'canonical_projection_max_error_px': error,
      'uniform_4_5_20_output_adjacent_rows_below_one_source_pixel_fraction': float((np.abs(np.diff(my[:, 32])) < 1).mean()),
      'native_v1': native_heads, 'ribbon_pixel_count': len(xy), 'ribbon_pixel_mismatch_count': 0,
      'ribbon_semantic_target_verified': False}
    if row['id'] in annotations or len(records) <= 8:
      a = annotations.get(row['id'], {'paint_polygons': [], 'coordinate_size': [rgb.shape[1], rgb.shape[0]]})
      mask = np.zeros(rgb.shape[:2], np.uint8)
      for polygon in a['paint_polygons']:
        p = np.asarray(polygon, float)*np.array([rgb.shape[1], rgb.shape[0]])/a['coordinate_size']
        cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 1)
      inclusion = np.zeros_like(mask)
      inclusion[xy[:, 1], xy[:, 0]] = 1
      native_mask = None
      annotation = {'id': row['id'], 'annotation_tolerance_in_annotation_coordinates_px': a.get('annotation_tolerance_px', 2),
        'explicitly_traced_paint_pixels': int(mask.sum()),
        'ribbon_traced_pixel_inclusion_fraction_all_traced_distances': float((mask & inclusion).sum()/max(1, mask.sum()))}
      eligible = np.zeros_like(mask)
      for pr in ribbon['rows']:
        if pr['geometry_visible']:
          eligible[pr['y']] = 1
      scoped_mask = mask & eligible
      annotation.update(traced_pixels_in_geometry_visible_4_5_20_rows=int(scoped_mask.sum()),
        traced_pixels_outside_visible_4_5_20_rows=int(mask.sum()-scoped_mask.sum()),
        ribbon_traced_pixel_inclusion_fraction_in_visible_rows=float((mask & inclusion).sum()/scoped_mask.sum()) if scoped_mask.sum() else None)
      if 'type' in maps:
        sampled = np.zeros_like(mask)
        pp = maps['type'].reshape(-1, 2)
        sampled[pp[:, 1], pp[:, 0]] = 1
        annotation['v1_retained_fraction_of_traced_pixels_in_visible_rows'] = (
          float((sampled & scoped_mask).sum()/scoped_mask.sum()) if scoped_mask.sum() else None)
      if 'type' in maps:
        pixels = maps['type']
        native_mask = mask[pixels[..., 1], pixels[..., 0]]
        original_painted = np.array([bool(mask[y].any()) for y in pixels[:, 0, 1]])
        missing = original_painted & ~native_mask.any(1)
        annotation.update(v1_selected_source_rows_with_traced_paint=int(original_painted.sum()),
          v1_rows_losing_all_traced_paint=int(missing.sum()),
          v1_complete_traced_paint_loss_fraction=float(missing.sum()/max(1, original_painted.sum())))
      if a['paint_polygons']:
        traced.append(annotation)
      np.savez_compressed(output/f"{row['id']}-mapping.npz", ribbon_xy=xy, ribbon_rgb=ribbon['rgb'],
        ribbon_distance=ribbon['distance_m'], ribbon_row_offsets=ribbon['row_offsets'],
        traced_source_mask=mask, **{f'v1_{head}_xy': value for head, value in maps.items()})
      # Honest preview: no remap, mask saved separately; rescaling below is only
      # a labelled viewing aid, never a source of training or gap evidence.
      source = Image.fromarray(rgb)
      draw = ImageDraw.Draw(source)
      poly = np.column_stack((cx[:, 0], cy[:, 0])).tolist()
      draw.line([tuple(p) for p in poly], fill=(0, 255, 255), width=2)
      for edge in ('left_x', 'right_x'):
        outline = [(r[edge], r['y']) for r in ribbon['rows'] if r['geometry_visible']]
        if len(outline) > 1:
          draw.line(outline, fill=(0, 255, 0), width=1)
      panel = Image.new('RGB', (1820, 860), 'white')
      source.thumbnail((1100, 790))
      panel.paste(source, (0, 40))
      d = ImageDraw.Draw(panel)
      d.text((10, 10), 'Decoded source / cyan=model boundary / green=ribbon edges (not paint segmentation)', fill='black')
      for j, head in enumerate(('color', 'type')):
        if head in observation['heads']:
          im = Image.fromarray(observation['heads'][head]['image']).resize((192, 576), Image.Resampling.NEAREST)
          panel.paste(im, (1120+j*210, 65))
          d.text((1120+j*210, 40), f'v1 {head} (nearest view)', fill='black')
      if native_mask is not None:
        im = Image.fromarray(native_mask*255).resize((192, 576), Image.Resampling.NEAREST).convert('RGB')
        panel.paste(im, (1540, 65))
        d.text((1540, 40), 'Traced paint after v1', fill='black')
      box = source_box(rgb, ribbon)
      if box:
        Image.fromarray(box['image']).save(output/f"{row['id']}-source-box.png")
        Image.fromarray(box['ribbon_mask'].astype(np.uint8)*255).save(output/f"{row['id']}-ribbon-mask.png")
      d.text((1110, 680), 'v2 keeps a ragged source-pixel array, not a resized strip.', fill='black')
      d.text((1110, 710), 'Mask does NOT prove target identity or absence of occlusion.', fill='black')
      panel.save(output/f"{row['id']}-correspondence.png")
      viewer(output/f"{row['id']}-interactive.html", rgb, observation, maps)
    results.append(entry)
  summary = {'records': len(results), 'canonical_projection_error_px': summarize(errors),
    'synthetic_transform_only_audit': synthetic_audit(output),
    'v1_omitted_native_rows_fraction': summarize(losses), 'traced_paint_audit': traced,
    'ribbon_pixel_equality_verified_records': len(results), 'semantic_quality_passed': False,
    'training_resume_allowed': False, 'compression_error_measurable': False,
    'compression_limitation': 'No uncompressed sensor reference; decoded source artifacts cannot be attributed quantitatively to the codec.',
    'limitations': [f'{len(traced)} partial single-reviewer masks; not an independent segmentation benchmark.',
      'Exact RGB equality proves preservation of included pixels only, not correct target selection.',
      'Original model probabilities not available in this audit cache; projection diagnostic uses 1.0, not label admission.',
      'No classifier is connected to the ragged representation; no accuracy improvement is claimed.']}
  (output/'pixel-audit.json').write_text(json.dumps({'summary': summary, 'records': results}, indent=2), encoding='utf-8')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--old-root', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  run(args.old_root, args.output)
