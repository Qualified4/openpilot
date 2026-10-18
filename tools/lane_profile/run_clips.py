"""Run the profile scanner on the frozen 5-clip 20Hz NV12 set used by the Pixel Ribbon reports."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from tools.lane_profile.scanner import ProfileScanner, nv12_planes  # noqa: TID251

FRAMES = Path('.analysis/pixel-ribbon-hybrid-20261018/frames.json')


def load_rows():
  rows = json.loads(FRAMES.read_text(encoding='utf-8'))
  for clip in {r['clip'] for r in rows}:
    group = sorted((r for r in rows if r['clip'] == clip), key=lambda r: r['stamp'])
    odo, last = 0., None
    for r in group:
      if last is not None:
        odo += float(np.hypot(r['pose'][0] - last[0], r['pose'][1] - last[1]))
      r['odometer'] = odo
      last = r['pose']
  return rows


def run(rows, *, every=1, solid_m=12., debug=None, confirm_travel_m=1.):
  cv2.setNumThreads(1)
  records, timing = [], []
  for clip in sorted({r['clip'] for r in rows}):
    group = sorted((r for r in rows if r['clip'] == clip), key=lambda r: r['stamp'])
    scanners = {side: ProfileScanner(side, solid_m=solid_m, confirm_travel_m=confirm_travel_m) for side in ('left', 'right')}
    for i, r in enumerate(group):
      if i % every:
        continue
      nv12 = np.load(r['nv12'])
      h, w = nv12.shape[0]*2//3, nv12.shape[1]
      start = time.perf_counter_ns()
      y_plane, uv = nv12_planes(nv12, w, h)
      out = {}
      for side, line, prob in (('left', r['line'], r['probability']), ('right', r['right_line'], r['right_probability'])):
        out[side] = scanners[side].update(y_plane, uv, line, prob, r['rpy'], r['intrinsics'], r['stamp'], r['odometer'],
                                          leads=r['leads'], lane_change=r['lane_change'])
      timing.append((time.perf_counter_ns() - start)/1e6)
      for side, res in out.items():
        records.append({'clip': clip, 'index': i, 'frame_id': r['frame_id'], 'side': side, 'stamp': r['stamp'],
                        'type': res['type'], 'color': res['color'], 'reason': res['reason'], 'reset': res['reset'],
                        'rows': res['rows'], 'paint_rows': res['paint_rows'], 'road_rows': res['road_rows'],
                        'gaps': res['gaps'], 'solid_run_m': res['solid_run_m'], 'color_bins': res.get('color_bins', 0),
                        'path': {k: v for k, v in (res.get('path') or {}).items() if k != 'members'} or None,
                        'double': bool(res.get('path') and len(res['path']['members']) > 1)})
        if debug is not None and i % 10 == 0 and 'evidence' in res:
          save_debug(debug/f"{clip[-10:]}-{i:02d}-{side}.png", y_plane, res)
  return records, timing


def save_debug(path, y_plane, res):
  from tools.lane_profile.scanner import HALF_M, COL_M, REF_M  # noqa: TID251
  geo, ev = res['geometry'], res['evidence']
  lum = cv2.remap(y_plane, geo['mx'], geo['my'], cv2.INTER_LINEAR).reshape(len(geo['ys']), -1, 4).mean(2).astype(np.uint8)
  rgb = np.repeat(lum[..., None], 3, 2)
  color = {1: (0, 220, 0), 0: (40, 80, 255), -1: (200, 0, 0)}
  bar = np.array([color[int(s)] for s in ev['state']], np.uint8)[:, None].repeat(8, 1)
  if ev['path'] is not None:
    e = ev['path']['offset'] + ev['path']['slope']*(geo['d'] - REF_M)
    col = np.clip(((e + HALF_M)/COL_M).astype(int), 0, rgb.shape[1] - 1)
    rgb[np.arange(len(col)), col] = (255, 0, 255)
  image = np.concatenate((rgb, bar), 1)
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(cv2.resize(image, (image.shape[1]*3, image.shape[0]*2), interpolation=cv2.INTER_NEAREST)).save(path)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--every', type=int, default=1)
  p.add_argument('--solid-m', type=float, default=12.)
  p.add_argument('--debug', action='store_true')
  a = p.parse_args()
  recs, times = run(load_rows(), every=a.every, solid_m=a.solid_m, debug=a.output/'debug' if a.debug else None)
  a.output.mkdir(parents=True, exist_ok=True)
  (a.output/'records.json').write_text(json.dumps(recs, default=float), encoding='utf-8')
  print('frames', len(times), 'mean ms', np.mean(times), 'p95', np.percentile(times, 95))
