"""Frozen-log NV12 cadence/CPU comparison; no training, runtime or CAN setup."""
import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time

import av
import cv2
import numpy as np
from PIL import Image
import psutil

from openpilot.selfdrive.carrot.lane_type.hybrid_tracker import HybridPaintTracker, RegisteredPaintTracker
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import PaintTracker, to_world
from tools.lane_type.retrain_data import metadata  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import write_json, synthetic_frame  # noqa: TID251

OUTPUT = Path('.analysis/pixel-ribbon-hybrid-20261018')
FROZEN = Path('.analysis/pixel-ribbon-alignment-20261018')


def read(path):
  return json.loads(path.read_text(encoding='utf-8'))


def prepare(output):
  rows = read(FROZEN/'dense-control/clips.json')
  inventory = read(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json'))
  result, errors = [], []
  for route, segment in sorted({(r['route'], r['segment']) for r in rows}):
    source = next(s for s in inventory['sources'] if (s['route'], s['segment']) == (route, segment))
    _, models, _, _, cameras, encodes = metadata(source)
    wanted = {r['video_index']: r for r in rows if (r['route'], r['segment']) == (route, segment)}
    with av.open(source['videos']['fcamera']['path']) as container:
      container.streams.video[0].thread_count = 1
      for i, frame in enumerate(container.decode(video=0)):
        if i not in wanted:
          continue
        row = wanted[i]
        key = encodes['fcamera'][i]
        assert key == (row['frame_id'], row['stamp']) and key in models and key in cameras
        h, w = frame.height, frame.width
        assert h % 2 == w % 2 == 0
        # Plane packing only: no RGB round trip. Codec decode remains offline.
        planar = frame.to_ndarray(format='yuv420p')
        nv12 = np.empty_like(planar)
        nv12[:h] = planar[:h]
        uv = planar[h:].ravel()
        nv12[h:, ::2] = uv[:h*w//4].reshape(h//2, w//2)
        nv12[h:, 1::2] = uv[h*w//4:].reshape(h//2, w//2)
        path = output/'nv12'/f'{route}-{segment}-{key[0]}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, nv12)
        converted = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        reference = np.array(Image.open(row['frame']))
        delta = abs(converted.astype(np.int16)-reference.astype(np.int16))
        errors.append({'frame_id': key[0], 'mean_absolute_rgb_difference': float(delta.mean()),
          'p95_absolute_rgb_difference': float(np.percentile(delta, 95))})
        model = models[key]
        result.append(dict(row, nv12=str(path.resolve()), nv12_sha256=hashlib.sha256(nv12.tobytes()).hexdigest(),
          source_pixel_format=frame.format.name, source_colorspace=str(getattr(frame, 'colorspace', 'not_exposed')),
          right_line=model['lines'][2], right_probability=model['probs'][2]))
    print('prepared NV12', route, segment, flush=True)
  write_json(output/'frames.json', result)
  write_json(output/'nv12-conversion.json', {'source': 'HEVC decoded YUV420 planes, lossless U/V repacking',
    'is_device_nv12_capture': False, 'opencv_conversion': 'COLOR_YUV2RGB_NV12',
    'limitation': 'Recorded codec/range/matrix differs from live NV12; all compared modes share this path.', 'rows': errors})


class FailureEvents:
  """Repeated unavailable frames are one episode until a qualified recovery."""
  def __init__(self):
    self.active = False
    self.events = []
    self.reset_calls = 0

  def update(self, frame, stamp, generation_before, result):
    self.reset_calls += max(0, result['generation']-generation_before)
    failed = not result['tracks'] or not result.get('registration_qualified', True)
    if failed:
      if not self.active:
        self.events.append({'start_frame': frame, 'start_stamp': stamp, 'end_stamp': stamp, 'failed_frames': 0, 'reasons': {}})
      event = self.events[-1]
      event['failed_frames'] += 1
      event['end_stamp'] = stamp
      reason = result['reason']
      event['reasons'][reason] = event['reasons'].get(reason, 0)+1
    self.active = failed


def run(output, mode, *, repeats=3, sides=1, legacy_registration=False):
  cv2.setNumThreads(1)
  rows = read(output/'frames.json')
  anchors = {(r['clip'], r['frame_id']) for r in read(FROZEN/'clips.json')}
  all_times, first_records, episodes = [], [], []
  process = psutil.Process()
  peak_rss = process.memory_info().rss
  for repetition in range(repeats):
    records, times, events = [], [], []
    for clip in sorted({r['clip'] for r in rows}):
      group = [r for r in rows if r['clip'] == clip]
      constructor = HybridPaintTracker if mode == 'hybrid' else PaintTracker if legacy_registration else RegisteredPaintTracker
      trackers = [constructor(side) for side in ('left', 'right')[:sides]]
      failures = [FailureEvents() for _ in trackers]
      for row in group:
        keyframe = (row['frame_id']-group[0]['frame_id']) % 4 == 0
        if mode == '5hz' and not keyframe:
          continue
        load_start = time.perf_counter_ns()
        nv12 = np.load(row['nv12'])  # input buffer readiness, not runtime preprocessing
        load_ms = (time.perf_counter_ns()-load_start)/1e6
        wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
        rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pre_wall, pre_cpu = time.perf_counter_ns(), time.process_time_ns()
        outputs = []
        for j, tracker in enumerate(trackers):
          generation = tracker.generation
          line = row['line'] if j == 0 else row['right_line']
          probability = row['probability'] if j == 0 else row['right_probability']
          kw = {'pose': row['pose'], 'pose_error_m': row['pose_error_m'], 'road_pose': row['road_pose'],
                'leads': row['leads'], 'lane_change': row['lane_change'], 'gray': gray}
          if not legacy_registration:
            kw['continuous_alternative'] = True
          if mode == 'hybrid':
            kw['precise'] = keyframe
          value = tracker.update(rgb, line, probability, row['rpy'], np.array(row['intrinsics']), row['stamp'], **kw)
          outputs.append((generation, value))
        cpu_end, wall_end = time.process_time_ns(), time.perf_counter_ns()
        timing = {'clip': clip, 'frame_id': row['frame_id'], 'precise': keyframe if mode == 'hybrid' else True,
          'preprocess_cpu_ms': (pre_cpu-cpu_start)/1e6, 'algorithm_cpu_ms': (cpu_end-pre_cpu)/1e6,
          'preprocess_wall_ms': (pre_wall-wall_start)/1e6, 'algorithm_wall_ms': (wall_end-pre_wall)/1e6,
          'total_cpu_ms': (cpu_end-cpu_start)/1e6, 'total_wall_ms': (wall_end-wall_start)/1e6,
          'input_file_load_wall_ms_excluded': load_ms}
        times.append(timing)
        peak_rss = max(peak_rss, process.memory_info().rss)
        for j, (generation, value) in enumerate(outputs):
          failures[j].update(row['frame_id'], row['stamp'], generation, value)
          if repetition == 0:
            folder = output/f'{mode}-{"legacy" if legacy_registration else "registered"}-sides{sides}'/'masks'
            path = folder/f"{clip}-{row['frame_id']}-{'left' if j == 0 else 'right'}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            xy = np.column_stack(np.nonzero(value['target_mask']))[:, ::-1]
            np.savez_compressed(path, selected_xy=xy)
            record = {k: v for k, v in value.items() if k != 'target_mask'}
            record.update(clip=clip, side='left' if j == 0 else 'right', frame_id=row['frame_id'], stamp=row['stamp'],
              common_5hz_time=(clip, row['frame_id']) in anchors, precise=timing['precise'], selected_pixels=len(xy))
            records.append(record)
      if repetition == 0:
        for j, state in enumerate(failures):
          events.append({'clip': clip, 'side': 'left' if j == 0 else 'right', 'reset_calls': state.reset_calls,
            'failure_event_count': len(state.events), 'failure_events': state.events})
    all_times.append(times)
    if repetition == 0:
      first_records, episodes = records, events
    print('measured', mode, 'sides', sides, 'repeat', repetition+1, 'frames', len(times), flush=True)
  folder = output/f'{mode}-{"legacy" if legacy_registration else "registered"}-sides{sides}'
  write_json(folder/'results.json', first_records)
  write_json(folder/'timing.json', all_times)
  write_json(folder/'failure-events.json', episodes)
  span_s = sum((max(r['stamp'] for r in rows if r['clip'] == clip)-min(r['stamp'] for r in rows if r['clip'] == clip))/1e9
               for clip in {r['clip'] for r in rows})
  flattened = [r for repeat in all_times for r in repeat]
  def stats(key):
    values = [r[key] for r in flattened]
    return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'p99': float(np.percentile(values, 99))}
  write_json(folder/'performance.json', {'mode': mode, 'sides': sides, 'repeats': repeats, 'input_frames': len(rows),
    'processed_per_repeat': len(all_times[0]), 'video_span_s': span_s, 'preprocess_cpu_ms': stats('preprocess_cpu_ms'),
    'algorithm_cpu_ms': stats('algorithm_cpu_ms'), 'total_cpu_ms': stats('total_cpu_ms'), 'total_wall_ms': stats('total_wall_ms'),
    'preprocess_wall_ms': stats('preprocess_wall_ms'), 'algorithm_wall_ms': stats('algorithm_wall_ms'),
    'cpu_one_core_equivalent_percent': 100*np.mean([sum(r['total_cpu_ms'] for r in rep) for rep in all_times])/(span_s*1000),
    'sampled_peak_process_rss_mib': peak_rss/2**20, 'platform': platform.platform(), 'processor': platform.processor(),
    'opencv_threads': cv2.getNumThreads(), 'onnx_calls': 0, 'gpu_calls': 0, 'comma_device_measurement': False,
    'cpu_clock_note': 'Windows process CPU accounting can advance in ~15.625ms quanta. Use wall clock for per-frame p95/p99 latency; aggregate CPU for load.',
    'included': 'NV12 to RGB and gray, selection, feature tracking, geometry, 3s accumulation, type/color and alternative decision',
    'excluded': 'video decoding, input file load, output serialization/previews, process startup; no ONNX inference is part of this algorithm'})
  print('type outputs', Counter(r['type'] for r in first_records), flush=True)


def synthetic_comparison(output):
  cv2.setNumThreads(1)
  scenarios = [
    ('white_dashed_fast', 20., {'dashed': True, 'white': True}),
    ('yellow_dashed', 5., {'dashed': True}),
    ('solid_stopped', 0., {}), ('solid_slow1', 1., {}), ('solid_slow2', 2., {}),
    ('long_dash_stopped', 0., {'dashed': True, 'paint_m': 18., 'gap_m': 5., 'phase_m': -4.}),
    ('solid_occluded', 3., {'occluded': True}), ('white_shadow', 3., {'shadow': True, 'white': True}),
  ]
  reports = []
  for name, speed, settings in scenarios:
    trackers = {'5hz': RegisteredPaintTracker('left'), '20hz': RegisteredPaintTracker('left'), 'hybrid': HybridPaintTracker('left')}
    rows = []
    for i in range(61):
      pose = np.array([speed*i*.05, 0., 0.])
      rgb, line, k, rpy, target, nuisance, leads = synthetic_frame(pose, **settings)
      h, w = rgb.shape[:2]
      planar = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
      nv12 = np.empty_like(planar)
      nv12[:h] = planar[:h]
      uv = planar[h:].ravel()
      nv12[h:, ::2] = uv[:h*w//4].reshape(h//2, w//2)
      nv12[h:, 1::2] = uv[h*w//4:].reshape(h//2, w//2)
      rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
      gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
      for mode, tracker in trackers.items():
        if mode == '5hz' and i % 4:
          continue
        kw = {'pose': pose, 'pose_error_m': .02, 'leads': leads, 'gray': gray, 'continuous_alternative': True}
        if mode == 'hybrid':
          kw['precise'] = i % 4 == 0
        generation = tracker.generation
        r = tracker.update(rgb, line, 1., rpy, k, i*50_000_000, **kw)
        eligible = np.zeros(target.shape, bool)
        false_gap = 0
        for p in r['profile']:
          if p['geometry_visible']:
            eligible[p['y']] = True
          if p['ink'] == 0 and 'local_xy' in p:
            world_x = to_world(np.array([p['local_xy']]), pose)[0, 0]
            paint = not settings.get('dashed') or (world_x+settings.get('phase_m', 0)) % (
              settings.get('paint_m', 3)+settings.get('gap_m', 5)) < settings.get('paint_m', 3)
            false_gap += paint
        mask = r['target_mask']
        truth_type = 'dashed' if settings.get('dashed') else 'solid'
        rows.append({'frame': i, 'time_s': i*.05, 'mode': mode, 'common_5hz': i % 4 == 0,
          'type': r['type'], 'alternative_type': r['alternative_type'], 'color': r['color'], 'truth_type': truth_type,
          'truth_color': 'white' if settings.get('white') else 'yellow', 'registration_qualified': r.get('registration_qualified', False),
          'false_dashed': r['type'] == 'dashed' and truth_type == 'solid',
          'alternative_false_solid': r['alternative_type'] == 'solid' and truth_type == 'dashed',
          'distinct_m': r['distinct_m'], 'continuous_paint_m': r['continuous_paint_m'],
          'target_pixels': int((target & eligible).sum()), 'paint_included': int((mask & target & eligible).sum()),
          'background_selected': int((mask & ~target & eligible).sum()), 'nuisance_selected': int((mask & nuisance).sum()),
          'false_gap_rows': int(false_gap), 'reset_calls': r['generation']-generation, 'reason': r['reason']})
    reports.append({'scenario': name, 'speed_mps': speed, 'settings': settings, 'records': rows})
    print('synthetic', name, Counter((r['mode'], r['type']) for r in rows), flush=True)
  write_json(output/'synthetic-comparison.json', reports)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, default=OUTPUT)
  p.add_argument('--prepare', action='store_true')
  p.add_argument('--mode', choices=('5hz', '20hz', 'hybrid'))
  p.add_argument('--repeats', type=int, default=3)
  p.add_argument('--sides', type=int, choices=(1, 2), default=1)
  p.add_argument('--legacy-registration', action='store_true')
  p.add_argument('--synthetic', action='store_true')
  args = p.parse_args()
  if args.prepare:
    prepare(args.output)
  if args.mode:
    run(args.output, args.mode, repeats=args.repeats, sides=args.sides, legacy_registration=args.legacy_registration)
  if args.synthetic:
    synthetic_comparison(args.output)
