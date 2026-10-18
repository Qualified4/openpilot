"""Exception review page: only the intervals auto_label.py could not approve, most important first.

The page (review_page.html) plays a browser-playable copy of the original recording around each
interval, draws the modelV2 boundary under review on the video, and shows the scanner's per-frame
verdicts and existing frame labels on a timeline. Intervals can be split at a marking change.

Instead of one decision per frame ROI, the reviewer confirms/corrects one interval of one boundary
(typically 3-40 s). Existing reviewed frame labels inside an interval are shown as anchors; intervals
where they disagree with the scanner come first. Decisions are exported as interval labels
(tools.lane_profile.labels schema, status human_confirmed). Nothing is admitted automatically.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from tools.lane_profile.analyze_logs import kind, load  # noqa: TID251
from tools.lane_profile.run_logs import INVENTORY  # noqa: TID251

COLORS = ('unknown', 'white', 'yellow')
FPS = 20


def proposals(seg, solid_m, min_s=3.0):
  """Per-side intervals: short UNKNOWN breaks and colour-unknown frames join a same-type
  neighbour; anything shorter than ``min_s`` is absorbed into its neighbour and marked mixed."""
  out = []
  for side in ('left', 'right'):
    runs = []
    for f in seg['frames']:
      value = [kind(f[side], solid_m), COLORS[f[side][1]]]
      if runs and f['eof'] - runs[-1]['end'] <= 300_000_000 and _joinable(runs[-1]['value'], value):
        r = runs[-1]
        r['end'], r['frames'] = f['eof'], r['frames'] + [f]
        r['value'] = [r['value'][0] if r['value'][0] != 'unknown' else value[0], r['value'][1] if r['value'][1] != 'unknown' else value[1]]
        continue
      runs.append({'side': side, 'value': value, 'start': f['eof'], 'end': f['eof'], 'frames': [f]})
    merged = []
    for r in runs:
      short = (r['end'] - r['start'])/1e9 < min_s
      if merged and (short or (merged[-1]['end'] - merged[-1]['start'])/1e9 < min_s):
        m = merged[-1]
        m['end'], m['frames'], m['mixed'] = r['end'], m['frames'] + r['frames'], m.get('mixed') or m['value'] != r['value']
        if (r['end'] - r['start']) > (m['frames'][-len(r['frames'])]['eof'] - m['start']):
          m['value'] = r['value']
        continue
      merged.append(r)
    for m in merged:
      m['value'] = tuple(m['value'])
    out.extend(merged)
  return out


def _joinable(a, b):
  return all(x == y or 'unknown' in (x, y) for x, y in zip(a, b, strict=True))


OVERLAY_M = (4.5, 6., 8., 10., 12., 15., 20., 25., 30., 40.)
TYPE_CHAR, COLOR_CHAR = {'unknown': '.', 'dashed': 'd', 'solid': 's'}, {'unknown': '.', 'white': 'w', 'yellow': 'y'}


def build_media(task):
  """Browser-playable H.264 copy of one recorded video (pts = original decode index, so
  time*fps recovers the frame index and its camera EOF) plus a JS sidecar with per-frame
  modelV2 boundary pixels and the scanner's per-frame verdicts for the timeline."""
  from bisect import bisect_right
  from fractions import Fraction
  import av
  import numpy as np
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from tools.lane_profile.scanner import projection  # noqa: TID251
  from tools.lane_type.retrain_data import metadata  # noqa: TID251
  source, quality, key, out_dir, seg_frames, solid_m, crf = task
  video, sidecar = out_dir/f'{key}.mp4', out_dir/f'{key}.js'
  if video.exists() and sidecar.exists():
    return key
  identity, models, calibrations, _, cameras, encodes = metadata(source)
  ct = [c[0] for c in calibrations]
  by_index = {f['index']: f for f in seg_frames}
  eof, lines, scan = [], {'left': [], 'right': []}, {'left': [], 'right': []}
  tmp = video.with_suffix('.part.mp4')
  with av.open(source['videos'][quality]['path']) as src, av.open(str(tmp), 'w', format='mp4', options={'movflags': 'faststart'}) as dst:
    ins = src.streams.video[0]
    ins.thread_count = 4
    out = dst.add_stream('libx264', rate=FPS)
    out.width, out.height, out.pix_fmt = ins.codec_context.width, ins.codec_context.height, 'yuv420p'
    out.codec_context.time_base = Fraction(1, FPS)
    out.options = {'crf': str(crf), 'preset': 'veryfast', 'g': '10', 'bf': '0'}
    for index, frame in enumerate(src.decode(video=0)):
      ekey = encodes[quality].get(index)
      model, sensor = (models.get(ekey), cameras.get(ekey)) if ekey else (None, None)
      ci = bisect_right(ct, ekey[1]) - 1 if ekey else -1
      eof.append(ekey[1] if ekey else None)
      for side, li in (('left', 1), ('right', 2)):
        pts = None
        if model is not None and sensor is not None and ci >= 0:
          cam = DEVICE_CAMERAS[identity['device'], sensor].fcam
          k = cam.intrinsics.copy()
          k[0] *= frame.width/cam.width
          k[1] *= frame.height/cam.height
          line = model['lines'][li]
          d = [x for x in OVERLAY_M if line['x'][0] <= x <= line['x'][-1]]
          xyz = np.column_stack((d, np.interp(d, line['x'], line['y']), np.interp(d, line['x'], line['z']))) @ projection(calibrations[ci][1], k).T
          if len(d) and (xyz[:, 2] > .1).all():
            pts = [v for q in (xyz[:, :2]/xyz[:, 2:]) for v in (round(float(q[0]), 1), round(float(q[1]), 1))]
        lines[side].append(pts)
        f = by_index.get(index)
        scan[side].append((TYPE_CHAR[kind(f[side], solid_m)] + COLOR_CHAR[COLORS[f[side][1]]]) if f else '..')
      frame.pts, frame.time_base = index, Fraction(1, FPS)
      for packet in out.encode(frame):
        dst.mux(packet)
    for packet in out.encode():
      dst.mux(packet)
    width, height = out.width, out.height
  tmp.replace(video)
  data = {'fps': FPS, 'w': width, 'h': height, 'n': len(eof), 'eof': eof, 'lines': lines,
          'scan': {s: ''.join(v) for s, v in scan.items()}, 'dist': OVERLAY_M}
  sidecar.write_text(f'(window.SEG=window.SEG||{{}})[{json.dumps(key)}]={json.dumps(data, separators=(",", ":"))};', encoding='utf-8')
  return key


def main(folder, queue, output, solid_m=12., limit=0, jobs=8, crf=23):
  """Build the exception review page from auto_label.py's review queue (already priority-ordered)."""
  inventory = {s['id']: s for s in json.loads(INVENTORY.read_text(encoding='utf-8'))['sources']}
  rows = json.loads(Path(queue).read_text(encoding='utf-8'))
  rows = rows[:limit] if limit else rows
  segs = {(s['route'], s['segment'], s['quality']): s for s in load(folder)}
  items = []
  for r in rows:
    x = r['features']
    conflict = [a for a in r['anchors'] if a['type'] != x['type'] or a['color'] != x['color']]
    items.append({'id': r['id'], 'route': r['route'], 'segment': r['segment'], 'split': r['split'], 'quality': r['quality'],
                  'side': r['side'], 'start_eof_ns': r['start_eof_ns'], 'end_eof_ns': r['end_eof_ns'],
                  'start_index': r['start_index'], 'end_index': r['end_index'], 'duration_s': round(x['duration_s'], 1),
                  'proposal_type': x['type'], 'proposal_color': x['color'], 'mixed': 'dashed_solid_transition' in r['reasons'],
                  'reasons': r['reasons'], 'priority': r['priority'], 'anchors': r['anchors'], 'anchors_conflict': len(conflict),
                  'anchor_labels': [f"{a['type']}/{a['color']}" for a in r['anchors']],
                  'onnx': {k: x[k] for k in ('onnx_raw_type_agree', 'onnx_guarded_type_disagree', 'onnx_raw_color_agree')},
                  'media': f"{r['route']}--{r['segment']}-{r['quality']}"})
  media = output/'media'
  media.mkdir(parents=True, exist_ok=True)
  tasks = {}
  for it in items:
    if it['media'] not in tasks:
      sid = f"{it['route']}--{it['segment']}"
      tasks[it['media']] = (inventory[sid], it['quality'], it['media'], media, segs[(it['route'], it['segment'], it['quality'])]['frames'], solid_m, crf)
  with Pool(jobs) as pool:
    for i, key in enumerate(pool.imap_unordered(build_media, tasks.values()), 1):
      print(i, '/', len(tasks), key, flush=True)
  (output/'review-items.json').write_text(json.dumps(items, indent=1), encoding='utf-8')
  (output/'REVIEW.html').write_text(PAGE.read_text(encoding='utf-8').replace('__DATA__', json.dumps(items)), encoding='utf-8')
  stats = {'items': len(items), 'videos': len(tasks), 'total_interval_seconds': round(sum(i['duration_s'] for i in items), 1)}
  (output/'review-stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
  print(stats)


PAGE = Path(__file__).with_name('review_page.html')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--input', type=Path, required=True, help='run_logs output (scanner per-frame results)')
  p.add_argument('--queue', type=Path, required=True, help='auto_label.py review-queue.json')
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--solid-m', type=float, default=12.)
  p.add_argument('--limit', type=int, default=0)
  p.add_argument('--jobs', type=int, default=8)
  p.add_argument('--crf', type=int, default=23)
  a = p.parse_args()
  main(a.input, a.queue, a.output, a.solid_m, a.limit, a.jobs, a.crf)
