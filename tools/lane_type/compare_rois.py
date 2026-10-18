"""Frozen-model ROI diagnostics. Invalid requested crops never enter a model."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.lane_type.evaluate import metrics, scenario  # noqa: TID251 - PC tooling
from tools.lane_type.roi_quality import projection_maps, in_frame  # noqa: TID251
from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard, visibility
from openpilot.selfdrive.carrot.lane_type.result import TYPE_CLASSES, COLOR_CLASSES, unknown
from openpilot.selfdrive.carrot.lane_type.roi import rgb_strip
from types import SimpleNamespace as N

HEADS = (('type', TYPE_CLASSES), ('color', COLOR_CLASSES))


def compare(audit, manifest, model_directory, output):
  records = json.loads(audit.read_text(encoding='utf-8'))
  original = {r['id']: r for r in json.loads(manifest.read_text(encoding='utf-8'))}
  mixed = {r['id'] for r in records if r['video_source'] == 'fcamera' and r['side'] == 'left' and
           r['id'].startswith(('000003dd--26eb77e00d--5-', '000003ea--6a2371afe5--6-'))}
  rows = sorted([r for r in records if r['verified'] and r['id'] not in mixed],
                key=lambda r:(original[r['id']]['route'], original[r['id']]['camera_eof_ns'], r['side']))
  report = {'status': 'diagnostic_only_no_roi_or_model_approved', 'model_directory': str(model_directory),
    'label_scope': 'Physical boundary labels from reviewed 6-46m context; adaptive crops need separate per-head pixel review',
    'selection_policy': 'train/validation only; test is final diagnostic, never threshold or ROI tuning',
    'mixed_marking_excluded': sorted(mixed), 'samples': len(rows), 'engines': {}}
  for engine in ('onnx', 'opencv'):
    classifier = Classifier(engine, model_directory)
    results = {}
    for tag in ('3-15', '3-25', '3-35', '6-46', 'adaptive-to15', 'adaptive-to25', 'adaptive-to35'):
      raw, final, traces = [], [], []
      guards = {}
      for row in rows:
        old = original[row['id']]
        k = np.asarray(row['intrinsics'])
        width, height = row['frame_size']
        adaptive = tag.startswith('adaptive-')
        if adaptive:
          far = float(tag.removeprefix('adaptive-to'))
          maps, ds = projection_maps(row['line'], row['rpy'], k, 3, far, rows=1024)
          valid = in_frame(maps, width, height) & (maps[1] < height*.86)
          complete = valid.all(1)
          near = float(np.ceil(ds[complete].min()*10)/10) if complete.any() else far
          maps, ds = projection_maps(row['line'], row['rpy'], k, near, far)
          available = bool(in_frame(maps, width, height).all())
        else:
          near, far = map(float, tag.split('-'))
          maps, ds = projection_maps(row['line'], row['rpy'], k, near, far)
          available = row['ranges'][tag]['complete_requested_roi']
        result = unknown('out_of_frame')
        observation = {'sufficient': False, 'observedLengthM': 0., 'occludedFraction': 1.}
        if available:
          rgb = np.asarray(Image.open(audit.parent/row['frame']).convert('RGB'))
          strip = rgb_strip(rgb, maps)
          clear, observation = visibility(maps, N(leadsV3=[N(**lead) for lead in row['model_leads']]),
                                          N(**row['line']), row['rpy'], k)
          strip[~clear] = 0
          clear_rows = clear[:, 21:43].mean(1) >= .9
          length = float(clear_rows.sum()*(far-near)/192)
          occlusion = float(1-clear_rows.mean())
          observation.update(observedLengthM=length, occludedFraction=occlusion,
                             sufficient=length >= 32 and occlusion <= .15)
          result = classifier.infer([strip], distance_range=(near, far))[0]
        raw.append({head: result.get(head+'Candidate', result[head]) for head, _ in HEADS})
        # Encodings and disjoint time windows never lend confirmations to each other.
        key = old['route'], row['video_source']
        guard = guards.setdefault(key, ObservationGuard())
        prediction = guard.update(row['side'], result, observation, old['boundary'],
          now_ns=old['camera_eof_ns'], camera_ns=old['camera_eof_ns'], lane_change=row['lane_change'], tracked=available)
        final.append(prediction)
        traces.append({'id': row['id'], 'available': available, 'effective_range_m': [near, far],
                       'observation': observation, 'raw': raw[-1], 'final': prediction})
      summaries = {}
      for split in ('validation', 'test'):
        summaries[split] = {}
        for quality in ('all', 'fcamera', 'qcamera'):
          for condition in ('all', 'night', 'curve', 'low_speed', 'stopped', 'lead_occluded', 'lane_change'):
            ix = [i for i, r in enumerate(rows) if r['split'] == split and (quality == 'all' or r['video_source'] == quality)
                  and scenario(original[r['id']], condition)]
            truth = [original[rows[i]['id']] for i in ix]
            usable = [i for i in ix if traces[i]['available']]
            entry = {'samples': len(ix), 'input_coverage': len(usable)/len(ix) if ix else None,
              'raw': {head: metrics([original[rows[i]['id']] for i in usable], [raw[i] for i in usable], head, classes)
                      for head, classes in HEADS},
              'final': {head: metrics(truth, [final[i] for i in ix], head, classes) for head, classes in HEADS},
              'wrong_solid_confirmations': sum(original[rows[i]['id']]['type'] == 'dashed' and final[i]['type'] == 'solid' and
                                              not final[i].get('held', False) for i in ix) if ix else None}
            summaries[split][quality+'/'+condition] = entry
      results[tag] = {'summaries': summaries, 'predictions': traces}
    report['engines'][engine] = results
  output.write_text(json.dumps(report, indent=2), encoding='utf-8')
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--audit', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--model-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  compare(args.audit, args.manifest, args.model_dir, args.output)
