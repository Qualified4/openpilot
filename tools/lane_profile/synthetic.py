"""Synthetic comparison on the exact scenario definitions used by the Pixel Ribbon reports.

Scenes come from tools.lane_type.validate_pixel_tracking.synthetic_frame (unchanged), passed
through RGB->I420->NV12 like the hybrid report. Stored per-frame results of the existing
trackers are read back for the same scenario/frame; nothing of the old code is re-tuned.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from tools.lane_profile.scanner import ProfileScanner  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251

ALIGN = Path('.analysis/pixel-ribbon-alignment-20261018/synthetic.json')
HYBRID = Path('.analysis/pixel-ribbon-hybrid-20261018/synthetic-comparison.json')

# (A) 13 scenarios, 16 frames at 5Hz (validate_pixel_tracking.synthetic)
A = ('solid', 'dashed', 'dashed_slow', 'white_dashed', 'white_shadow', 'double', 'double_mixed', 'crosswalk', 'guide',
     'shadow', 'occluded', 'curve', 'stopped')
# (B) 8 scenarios, 61 frames at 20Hz (validate_hybrid_tracking.synthetic_comparison)
B = [('white_dashed_fast', 20., {'dashed': True, 'white': True}), ('yellow_dashed', 5., {'dashed': True}),
     ('solid_stopped', 0., {}), ('solid_slow1', 1., {}), ('solid_slow2', 2., {}),
     ('long_dash_stopped', 0., {'dashed': True, 'paint_m': 18., 'gap_m': 5., 'phase_m': -4.}),
     ('solid_occluded', 3., {'occluded': True}), ('white_shadow', 3., {'shadow': True, 'white': True})]


def to_nv12_planes(rgb):
  h, w = rgb.shape[:2]
  planar = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
  u = planar[h:].ravel()
  return planar[:h], np.stack((u[:h*w//4].reshape(h//2, w//2), u[h*w//4:].reshape(h//2, w//2)), -1)


def scenario_a(name):
  frames = []
  for i in range(16):
    d = 0. if name == 'stopped' else i*(1. if name in ('dashed', 'double_mixed', 'white_dashed') else .6)
    pose = np.array([d, .0015*d*d if name == 'curve' else 0., np.arctan(.003*d) if name == 'curve' else 0.])
    kw = {'dashed': name in ('dashed', 'dashed_slow', 'double_mixed', 'white_dashed'), 'double': name in ('double', 'double_mixed'),
          'crosswalk': name == 'crosswalk', 'guide': name == 'guide', 'shadow': name in ('shadow', 'white_shadow'),
          'occluded': name == 'occluded', 'curved': name == 'curve', 'white': name.startswith('white')}
    frames.append((i*200_000_000, pose, kw))
  truth = ('dashed' if name in ('dashed', 'dashed_slow', 'white_dashed', 'double_mixed') else 'solid',
           'white' if name.startswith('white') else 'yellow')
  return frames, truth


def scenario_b(speed, settings):
  frames = [(i*50_000_000, np.array([speed*i*.05, 0., 0.]), settings) for i in range(61)]
  return frames, ('dashed' if settings.get('dashed') else 'solid', 'white' if settings.get('white') else 'yellow')


def run_scanner(frames, solid_m, every=1):
  scanner = ProfileScanner('left', solid_m=solid_m)
  out, last = [], None
  odo = 0.
  for i, (stamp, pose, kw) in enumerate(frames):
    if i % every:
      continue
    if last is not None:
      odo += float(np.hypot(*(pose[:2] - last[:2])))
    last = pose
    rgb, line, k, rpy, _, _, leads = synthetic_frame(pose, **kw)
    y, uv = to_nv12_planes(rgb)
    r = scanner.update(y, uv, line, 1., rpy, k, stamp, odo, leads=leads)
    out.append((i, r['type'], r['color']))
  return out


def summary(types, colors, truth):
  n = len(types)
  return {'frames': n, 'type': dict(Counter(types)), 'color': dict(Counter(colors)),
          'type_correct': sum(t == truth[0] for t in types), 'false_dashed': sum(t == 'dashed' for t in types) if truth[0] == 'solid' else 0,
          'false_solid': sum(t == 'solid' for t in types) if truth[0] == 'dashed' else 0,
          'first_correct_index': next((i for i, t in enumerate(types) if t == truth[0]), None),
          'color_correct': sum(c == truth[1] for c in colors), 'wrong_color': sum(c not in ('unknown', truth[1]) for c in colors)}


def main(output, solid_m):
  stored_a = {s['scenario']: s['records'] for s in json.loads(ALIGN.read_text(encoding='utf-8'))}
  stored_b = {s['scenario']: s['records'] for s in json.loads(HYBRID.read_text(encoding='utf-8'))}
  report = {'A_5hz_16frames': {}, 'B_20hz_61frames': {}}
  for name in A:
    frames, truth = scenario_a(name)
    mine = run_scanner(frames, solid_m)
    report['A_5hz_16frames'][name] = {'truth': truth,
      'profile': summary([t for _, t, _ in mine], [c for _, _, c in mine], truth),
      'pixel_ribbon_5hz': summary([r['type'] for r in stored_a[name]], [r['color'] for r in stored_a[name]], truth)}
    print('A', name, report['A_5hz_16frames'][name]['profile']['type'], '| old', report['A_5hz_16frames'][name]['pixel_ribbon_5hz']['type'], flush=True)
  for name, speed, settings in B:
    frames, truth = scenario_b(speed, settings)
    entry = {'truth': truth, 'speed_mps': speed}
    for mode, every in (('profile_20hz', 1), ('profile_5hz', 4)):
      mine = run_scanner(frames, solid_m, every)
      entry[mode] = summary([t for _, t, _ in mine], [c for _, _, c in mine], truth)
    for mode in ('5hz', '20hz', 'hybrid'):
      recs = [r for r in stored_b[name] if r['mode'] == mode]
      entry['pixel_' + mode] = summary([r['type'] for r in recs], [r['color'] for r in recs], truth)
    report['B_20hz_61frames'][name] = entry
    print('B', name, entry['profile_20hz']['type'], '| hybrid', entry['pixel_hybrid']['type'], flush=True)
  output.mkdir(parents=True, exist_ok=True)
  (output/'synthetic-comparison.json').write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--solid-m', type=float, default=12.)
  a = p.parse_args()
  main(a.output, a.solid_m)
