"""Identical held-out ROIs, abstention-aware metrics and isolated PC benchmarks."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from tools.lane_type.dataset import validate_rows  # noqa: TID251 - repository-level tooling, outside openpilot.tools
from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard
from openpilot.selfdrive.carrot.lane_type.result import TYPE_CLASSES, COLOR_CLASSES


def metrics(rows, predictions, head, classes):
  if not rows:
    return {'samples': 0, 'accuracy': None, 'macro_f1': None, 'unconfirmed_fraction': None,
            'confirmation_rate': None, 'confirmed_accuracy': None, 'classes': {}}
  truth = [r[head] for r in rows]
  predicted = [p[head] for p in predictions]
  details = {}
  for label in classes:
    tp = sum(a == label and b == label for a, b in zip(truth, predicted, strict=True))
    fp = sum(a != label and b == label for a, b in zip(truth, predicted, strict=True))
    fn = sum(a == label and b != label for a, b in zip(truth, predicted, strict=True))
    details[label] = {'support': truth.count(label), 'tp': tp, 'fp': fp, 'fn': fn,
                          'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
                          'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
  f1s = [v['f1'] for v in details.values() if v['f1'] is not None]
  confirmed = sum(v in classes for v in predicted)
  correct = sum(a == b for a, b in zip(truth, predicted, strict=True))
  return {'samples': len(rows), 'accuracy': sum(a == b for a, b in zip(truth, predicted, strict=True))/len(rows),
              'macro_f1': sum(f1s)/len(f1s) if f1s else None,
              'confirmation_rate': confirmed/len(rows), 'confirmed_accuracy': correct/confirmed if confirmed else None,
              'unconfirmed_fraction': sum(v not in classes for v in predicted)/len(rows), 'classes': details,
              'missing_truth_classes': [c for c in classes if c not in truth]}


def scenario(row, name):
  return {'day':row['lighting'] == 'day', 'night':row['lighting'] == 'night', 'curve':row['curve'],
          'lane_change':row['lane_change'], 'low_speed':row['speed_mps'] is not None and 0.1 <= row['speed_mps'] < 2,
          'stopped':row['speed_mps'] is not None and row['speed_mps'] < .1,
          'lead_occluded':row['observation']['occludedFraction'] > .15 and
                          row['observation']['visibilitySource'] == 'model_lead_silhouette_geometric_bound',
          'tracking_lost':not row.get('tracked', True)}.get(name, True)


def evaluate(manifest, directory, output):
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  validate_rows(rows, manifest.parent)
  rows = sorted([r for r in rows if r['split'] == 'test'], key=lambda r:(r['route'], r['camera_eof_ns'], r['side']))
  report = {'environment': 'PC offline encoded-video ROI classification; not comma measurements',
                'samples_total': len(rows), 'reviewed_truth': sum(r.get('verified', False) for r in rows), 'engines': {}}
  for engine in ('onnx', 'opencv'):
    classifier = Classifier(engine, directory)
    guards = {}
    predictions, raw_predictions = [], []
    previous_route = None
    for row in rows:
      if row['route'] != previous_route:
        guards.clear()
        previous_route = row['route']
      # Paired encodings are alternative input streams, never temporal evidence
      # for one another. An HD confirmation must not inflate qcamera coverage.
      quality = row['video_source'].split('.')[0]
      guard = guards.setdefault(quality, ObservationGuard())
      raw = classifier.infer([np.asarray(Image.open(manifest.parent/row['image']).convert('RGB'))])[0]
      raw_predictions.append(dict(raw, type=raw.get('typeCandidate', raw['type']), color=raw.get('colorCandidate', raw['color'])))
      prediction = guard.update(row['side'], raw, row['observation'], row['boundary'], now_ns=row['camera_eof_ns'],
                                camera_ns=row['camera_eof_ns'], lane_change=row['lane_change'], tracked=row.get('tracked', True))
      predictions.append(prediction)
    result = {}
    for name in ('all', 'day', 'night', 'curve', 'lane_change', 'low_speed', 'stopped', 'lead_occluded', 'tracking_lost'):
      indices = [i for i, r in enumerate(rows) if r.get('verified') and scenario(r, name)]
      truth = [rows[i] for i in indices]
      ps = [predictions[i] for i in indices]
      result[name] = {head:metrics(truth, ps, head, classes) for head, classes in [('type', TYPE_CLASSES), ('color', COLOR_CLASSES)]}
      # These rates need no guessed ground truth. Incorrect solids do.
      observed = [i for i, r in enumerate(rows) if scenario(r, name)]
      result[name]['observations'] = {'samples': len(observed),
        'unconfirmed_fraction': sum(predictions[i]['type'] not in TYPE_CLASSES for i in observed)/len(observed) if observed else None,
        'new_solid_confirmations': sum(predictions[i]['type'] == 'solid' and not predictions[i]['held'] for i in observed),
        'incorrect_solid_confirmations': sum(rows[i]['type'] == 'dashed' and predictions[i]['type'] == 'solid' and
                                          not predictions[i]['held'] for i in indices) if indices else None}
      pairs, switches = 0, 0
      last = {}
      for i in observed:
        row, prediction = rows[i], predictions[i]
        key = row['route'], row['segment'], row['side'], row['video_source'].split('.')[0]
        old = last.get(key)
        if old and 0 < row['camera_eof_ns']-old[0] <= 250_000_000 and not row['lane_change']:
          if old[1] in TYPE_CLASSES and prediction['type'] in TYPE_CLASSES:
            pairs += 1
            switches += old[1] != prediction['type']
        last[key] = row['camera_eof_ns'], prediction['type']
      result[name]['observations'].update(confirmed_adjacent_pairs=pairs, type_switches=switches if pairs else None)
    indices = [i for i, r in enumerate(rows) if r.get('verified') and r['observation']['sufficient'] and not r['lane_change']]
    result['raw_candidate_on_sufficient_roi'] = {head:metrics([rows[i] for i in indices], [raw_predictions[i] for i in indices], head, classes)
                                              for head, classes in [('type', TYPE_CLASSES), ('color', COLOR_CLASSES)]}
    result['errors'] = [{'id': r['id'], 'truth': {'type': r['type'], 'color': r['color']}, 'predicted': p,
                             'image': r['image'], 'context': r['context']} for r, p in zip(rows, predictions, strict=True)
                        if r.get('verified') and (r['type'] != p['type'] or r['color'] != p['color'])]
    result['predictions'] = [dict(id=r['id'], **p) for r, p in zip(rows, predictions, strict=True)]
    report['engines'][engine] = result
  output.write_text(json.dumps(report, indent=2), encoding='utf-8')
  return report


def benchmark(manifest, directory, engine, seconds=10):
  """One fresh process per engine/OFF; precomputed identical ROIs at 5Hz.

  Includes normalization/classification, excludes shared camera/ROI extraction.
  CPU is process CPU / wall time, 100% = one core. RSS is this PC process only.
  """
  import psutil
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  images = [np.asarray(Image.open(manifest.parent/r['image']).convert('RGB')) for r in rows if r['split'] == 'test']
  if len(images) < 2:
    raise ValueError('need test ROIs')
  process = psutil.Process()
  before = process.memory_info().rss
  classifier = None if engine == 'off' else Classifier(engine, directory)
  loaded = process.memory_info().rss
  if classifier:
    for _ in range(10):
      classifier.infer(images[:2])
  latencies = []
  rss = []
  start = time.perf_counter()
  cpu_start = time.process_time()
  index = 0
  while time.perf_counter()-start < seconds:
    cycle = time.perf_counter()
    batch = [images[index % len(images)], images[(index+1) % len(images)]]
    index += 2
    begin = time.perf_counter()
    if classifier:
      classifier.infer(batch)
      latencies.append((time.perf_counter()-begin)*1000)
    rss.append(process.memory_info().rss)
    time.sleep(max(0, .2-(time.perf_counter()-cycle)))
  elapsed = time.perf_counter()-start
  cpu = time.process_time()-cpu_start
  return {'environment': 'PC, not comma', 'platform': platform.platform(), 'processor': platform.processor(), 'engine': engine,
              'seconds': elapsed, 'samples': len(latencies), 'sides_per_call': 2, 'target_hz': 5, 'cpu_one_core_percent': cpu/elapsed*100,
              'cpu_counter_seconds': cpu, 'cpu_measurement_status': 'not_detected_at_counter_resolution' if cpu == 0 else 'process_time_counter',
              'cycles': index//2,
              'rss_before_engine_mib': before/2**20, 'rss_after_load_mib': loaded/2**20, 'rss_peak_mib': max(rss)/2**20,
              'latency_ms': {'mean': float(np.mean(latencies)), 'p95': float(np.percentile(latencies, 95)),
                              'p99': float(np.percentile(latencies, 99)), 'max': max(latencies)} if latencies else None,
              'observed_calls_hz': len(latencies)/elapsed if classifier else None,
              'modeld_period': 'unmeasured: no device', 'controlsd_period': 'unmeasured: no device', 'frame_drops': 'unmeasured: no device'}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--model-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--benchmark', choices=['off', 'onnx', 'opencv'])
  parser.add_argument('--seconds', type=float, default=10)
  args = parser.parse_args()
  if args.benchmark:
    args.output.write_text(json.dumps(benchmark(args.manifest, args.model_dir, args.benchmark, args.seconds), indent=2), encoding='utf-8')
  else:
    evaluate(args.manifest, args.model_dir, args.output)
