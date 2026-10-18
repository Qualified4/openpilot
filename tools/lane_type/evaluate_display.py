"""Offline display-policy metrics; warning priority is separately unit-tested.

Sparse reviewed stills cannot establish transition stability. A dense input
without reviewed truth cannot establish False Dashed or orange error rates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.lane_type.dataset import validate_rows  # noqa: TID251 - PC tooling
from tools.lane_type.evaluate import scenario  # noqa: TID251
from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard
from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip


def evaluate(manifest, model_directory, output):
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  validate_rows(rows, manifest.parent)
  rows = sorted([r for r in rows if r['split'] == 'test'], key=lambda r:(r['route'], r['camera_eof_ns'], r['side']))
  report = {'model_directory': str(model_directory), 'samples': len(rows), 'device': 'PC offline, not comma',
            'warning_context': 'No live CAN baseline supplied; no-warning classification preview. Actual warnings are not inferred from labels.',
            'engines': {}}
  for engine in ('onnx', 'opencv'):
    classifier = Classifier(engine, model_directory)
    guards, policies = {}, {}
    predictions = []
    for row in rows:
      quality = row['video_source'].split('.')[0]
      key = row['route'], quality
      guard, policy = guards.setdefault(key, ObservationGuard()), policies.setdefault(key, DisplayPolicy())
      strip = np.asarray(Image.open(manifest.parent/row['image']).convert('RGB'))
      raw = classifier.infer([strip])[0]
      result = guard.update(row['side'], raw, row['observation'], row['boundary'],
        now_ns=row['camera_eof_ns'], camera_ns=row['camera_eof_ns'], lane_change=row['lane_change'], tracked=row.get('tracked', True))
      if (raw['type'] == 'dashed' and row['observation']['occludedFraction'] == 0 and
          result.get('typeMonoTimeNanos') == row['camera_eof_ns']):
        evidence = classify_strip(strip)
        result.update(dashedPatternObserved=bool(evidence.get('repeatingPatternObserved')),
                      dashedPatternMonoTimeNanos=row['camera_eof_ns'])
      packet = {'type': 'ccncLaneType', 'version': 1, 'cameraMonoTimeNanos': row['camera_eof_ns'], row['side']: result}
      display = policy.update({}, packet, enabled=True, model_lanes=2, now_ns=row['camera_eof_ns'])['sides'][row['side']]
      predictions.append({'id': row['id'], 'raw': raw, 'guarded': result, 'display': display})
    summaries = {}
    for condition in ('all', 'day', 'night', 'curve', 'low_speed', 'stopped', 'lead_occluded', 'lane_change'):
      ix = [i for i, row in enumerate(rows) if scenario(row, condition)]
      labeled = [i for i in ix if rows[i].get('verified')]
      shown = [i for i in labeled if predictions[i]['display']['semantic'] not in ('hidden', 'existing_warning')]
      correct = sum(predictions[i]['display']['semantic'] ==
                    ('orange_solid' if rows[i]['color'] == 'yellow' else 'white_'+rows[i]['type']) for i in shown)
      dashed = [i for i in shown if predictions[i]['display']['semantic'] == 'white_dashed']
      orange = [i for i in shown if predictions[i]['display']['semantic'] == 'orange_solid']
      false_dashed = sum(rows[i]['type'] == 'solid' for i in dashed)
      false_orange = sum(rows[i]['color'] != 'yellow' for i in orange)
      yellow = [i for i in labeled if rows[i]['color'] == 'yellow']
      last, pairs, transitions, known_pairs, type_flips = {}, 0, 0, 0, 0
      for i in ix:
        row = rows[i]
        key = row['route'], row['segment'], row['side'], row['video_source'].split('.')[0]
        semantic = predictions[i]['display']['semantic']
        old = last.get(key)
        if old and 0 < row['camera_eof_ns']-old[0] <= 250_000_000 and not row['lane_change']:
          pairs += 1
          transitions += old[1] != semantic
          if old[1] in ('white_solid', 'white_dashed') and semantic in ('white_solid', 'white_dashed'):
            known_pairs += 1
            type_flips += old[1] != semantic
        last[key] = row['camera_eof_ns'], semantic
      summaries[condition] = {'observations': len(ix), 'reviewed_truth': len(labeled), 'shown_reviewed': len(shown),
        'confirmed_accuracy': correct/len(shown) if shown else None,
        'end_to_end_correct_fraction': correct/len(labeled) if labeled else None,
        'unconfirmed_or_hidden_fraction': sum(predictions[i]['display']['semantic'] == 'hidden' for i in ix)/len(ix) if ix else None,
        'dashed_confirmations': len(dashed), 'false_dashed': false_dashed if labeled else None,
        'false_dashed_rate': false_dashed/len(dashed) if dashed else None,
        'orange_boundary_confirmations': len(orange), 'false_orange_boundary': false_orange if labeled else None,
        'false_orange_boundary_rate': false_orange/len(orange) if orange else None,
        'yellow_boundary_truth': len(yellow),
        'yellow_hidden_or_missed': sum(predictions[i]['display']['semantic'] != 'orange_solid' for i in yellow),
        'adjacent_observation_pairs': pairs, 'display_transitions': transitions if pairs else None,
        'confirmed_white_type_pairs': known_pairs, 'dashed_solid_switches': type_flips if known_pairs else None}
    report['engines'][engine] = {'summaries': summaries, 'predictions': predictions}
  output.write_text(json.dumps(report, indent=2), encoding='utf-8')
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--model-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  evaluate(args.manifest, args.model_dir, args.output)
