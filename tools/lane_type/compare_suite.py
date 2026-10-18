"""Locked validation selection, exact held-out sequence replay and CPU-only cost."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np

from tools.lane_type.prepare_dual_data import json_write  # noqa: TID251 - offline repository tooling
from tools.lane_type.native_data import read_image  # noqa: TID251 - offline repository tooling
from tools.lane_type.evaluate import metrics  # noqa: TID251 - offline repository tooling
from openpilot.selfdrive.carrot.lane_type.roi import input_blob
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from openpilot.selfdrive.carrot.lane_type.temporal import DistanceEvidence

CLASSES = {'type': ('dashed', 'solid'), 'color': ('white', 'yellow')}
THRESHOLDS = (.85, .9, .95, .98, .995, .999, 1.001)


def predict(rows, directory):
  """GPU used for offline replay throughput only; never a CPU latency claim."""
  import torch
  from tools.lane_type.multi_model import DualInputModel  # noqa: TID251 - offline repository tooling
  info = json.loads((directory/'manifest.json').read_text())
  model = DualInputModel(info['architecture']).cuda().eval()
  state = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
  model.load_state_dict(state['state_dict'])
  result = {r['id']: {'type': None, 'color': None} for r in rows}
  cache = {}
  for head in ('type', 'color'):
    selected = [r for r in rows if head in r['heads']]
    for start in range(0, len(selected), 128):
      batch = selected[start:start+128]
      images = [read_image(Path.cwd(), r['heads'][head], cache) for r in batch]
      x = torch.from_numpy(input_blob(images)).cuda()
      with torch.inference_mode():
        values = getattr(model, head+'_head')(model.feature(x)).softmax(1).cpu().numpy()
      for r, v in zip(batch, values, strict=True):
        result[r['id']][head] = v.tolist()
    cache.clear()
  del model
  torch.cuda.empty_cache()
  return result


def raw_prediction(probabilities):
  return {head: CLASSES[head][int(np.argmax(probabilities[head]))] if probabilities[head] is not None else 'unknown'
          for head in CLASSES}


def choose_thresholds(rows, predictions):
  chosen, evidence = {}, {}
  for head, classes in CLASSES.items():
    attempts = []
    for threshold in THRESHOLDS:
      ps = []
      for row in rows:
        p = predictions[row['id']][head]
        ps.append({head: classes[int(np.argmax(p))] if p is not None and max(p) >= threshold else 'unknown'})
      m = metrics(rows, ps, head, classes)
      errors = sum(p[head] != 'unknown' and p[head] != r[head] for r, p in zip(rows, ps, strict=True))
      dangerous = sum((r[head], p[head]) == (('solid', 'dashed') if head == 'type' else ('yellow', 'white'))
                      for r, p in zip(rows, ps, strict=True))
      correct = sum(p[head] == r[head] for r, p in zip(rows, ps, strict=True))
      # Small validation sets cannot certify rare error rates. This zero-error
      # empirical selection is recorded as an experiment, never certification.
      attempts.append({'threshold': threshold, 'metrics': m, 'errors': errors, 'dangerous': dangerous,
                       'correct': correct, 'eligible': dangerous == 0 and errors <= .02*len(rows)})
    candidate = max((a for a in attempts if a['eligible']), key=lambda a: (a['correct'], -a['threshold']))
    chosen[head] = candidate['threshold']
    evidence[head] = attempts
  return chosen, evidence


def replay(rows, probabilities, thresholds, temporal=False, native_only=False):
  policies, accumulators, predictions = {}, {}, {}
  for r in sorted(rows, key=lambda r: (r.get('stream', ''), r['route'], r['segment'], r['camera_eof_ns'], r['side'])):
    stream = (r.get('stream', ''), r['route'], r['segment'], r['video_source'], r['side'])
    stamp = r['camera_eof_ns']
    policy = policies.setdefault(stream, DisplayPolicy())
    result = {'type': 'unknown', 'color': 'unknown', 'typeConfidence': 0., 'colorConfidence': 0.}
    if temporal:
      a = accumulators.setdefault(stream, DistanceEvidence())
      result = a.update(r['profile'], r['boundary'], r['speed_mps'], stamp, lane_change=r['lane_change'])
    for head, classes in CLASSES.items():
      p = probabilities[r['id']][head] if probabilities else None
      value = classes[int(np.argmax(p))] if p is not None else 'unknown'
      confidence = max(p) if p is not None else 0.
      h = r['heads'].get(head)
      if native_only:
        continue
      if temporal:
        if confidence < thresholds[head] or value != result[head]:
          result[head], result[head+'Confidence'] = 'unknown', 0.
        else:
          result[head+'Confidence'] = confidence
      else:
        supported = (h and h['evidence'][head] == value and
                     (head == 'type' or h['evidence']['nativePaintRows'] >= 5))
        if supported and confidence >= thresholds[head] and not r['lane_change']:
          result.update({head: value, head+'Confidence': confidence, head+'MonoTimeNanos': stamp})
          if head == 'type' and value == 'dashed':
            result.update(dashedPatternObserved=True, dashedPatternMonoTimeNanos=stamp)
    if r['lane_change']:
      result.update(trackingReset=True, type='unknown', color='unknown')
    packet = {'type': 'ccncLaneType', 'version': 1, 'cameraMonoTimeNanos': stamp, r['side']: result}
    display = policy.update({}, packet, enabled=True, model_lanes=2, now_ns=stamp)
    predictions[r['id']] = dict(display['sides'][r['side']], evidence=result)
  return predictions


def summarize(rows, probabilities, display):
  ps = [raw_prediction(probabilities[r['id']]) for r in rows] if probabilities else None
  raw = {head: metrics(rows, ps, head, classes) for head, classes in CLASSES.items()} if ps is not None else {}
  semantics = [display[r['id']]['semantic'] for r in rows]
  shown = sum(s not in ('hidden', 'existing_warning') for s in semantics)
  good = sum((s == 'orange_solid' and r['color'] == 'yellow') or
             (s == 'white_solid' and (r['type'], r['color']) == ('solid', 'white')) or
             (s == 'white_dashed' and (r['type'], r['color']) == ('dashed', 'white'))
             for r, s in zip(rows, semantics, strict=True))
  dashed = semantics.count('white_dashed')
  false = sum(r['type'] == 'solid' and s == 'white_dashed' for r, s in zip(rows, semantics, strict=True))
  yellow = sum(r['color'] == 'yellow' for r in rows)
  yellow_shown = sum(r['color'] == 'yellow' and s == 'orange_solid' for r, s in zip(rows, semantics, strict=True))
  orange = semantics.count('orange_solid')
  wrong_orange = sum(r['color'] != 'yellow' and s == 'orange_solid' for r, s in zip(rows, semantics, strict=True))
  white_on_yellow = sum(r['color'] == 'yellow' and s in ('white_solid', 'white_dashed') for r, s in zip(rows, semantics, strict=True))
  return {'samples': len(rows), 'raw': raw, 'display': {'shown': shown, 'confirmed_rate': shown/len(rows) if rows else None,
    'hidden_rate': 1-shown/len(rows) if rows else None, 'conditional_accuracy': good/shown if shown else None,
    'correct_over_all': good/len(rows) if rows else None, 'dashed_shown': dashed, 'false_dashed': false,
    'false_dashed_per_dashed': false/dashed if dashed else None, 'yellow_truth': yellow, 'yellow_shown': yellow_shown,
    'yellow_display_recall': yellow_shown/yellow if yellow else None, 'orange_shown': orange, 'false_orange': wrong_orange,
    'white_on_yellow_display': white_on_yellow},
    'false_dashed_raw': sum(r['type'] == 'solid' and p['type'] == 'dashed' for r, p in zip(rows, ps, strict=True)) if ps else None,
    'yellow_as_white_raw': sum(r['color'] == 'yellow' and p['color'] == 'white' for r, p in zip(rows, ps, strict=True)) if ps else None}


def transitions(rows, display):
  last, pairs, switches, type_switches, type_pairs = {}, 0, 0, 0, 0
  for r in sorted(rows, key=lambda r: (r['camera_eof_ns'], r['side'])):
    key = r.get('stream', ''), r['route'], r['segment'], r['video_source'], r['side']
    s = display[r['id']]['semantic']
    old = last.get(key)
    if old and 0 < r['camera_eof_ns']-old[0] <= 250_000_000:
      pairs += 1
      switches += old[1] != s
      if old[1] in ('white_solid', 'white_dashed') and s in ('white_solid', 'white_dashed'):
        type_pairs += 1
        type_switches += old[1] != s
    last[key] = r['camera_eof_ns'], s
  return {'adjacent_pairs': pairs, 'all_display_switches': switches, 'confirmed_white_type_pairs': type_pairs,
          'dashed_solid_switches': type_switches if type_pairs else None}


def run(root):
  val = json.loads((root/'native-v2-expanded-train-validation.json').read_text(encoding='utf-8'))
  val = [r for r in val if r['split'] == 'validation']
  test = json.loads((root/'native-test.json').read_text(encoding='utf-8'))
  dense_source = json.loads(Path('.analysis/archive/2026-10-18/lane-roi-quality/policy/fresh-sequences/reviewed-manifest.json').read_text(encoding='utf-8'))
  dense_truth = {r['scene_id']+'-'+r['video_source']: r for r in dense_source if r['verified']}
  wanted_segments = {(r['route'], r['segment']) for r in val+test}
  sequences = []
  for route, seg in sorted(wanted_segments):
    f = root/'native-data/segments'/f'{route}--{seg}.json.gz'
    block = json.loads(gzip.decompress(f.read_bytes()))
    sequences += [dict(r, stream='route') for r in block['rows']]
  dense_rows = []
  for f in (root/'native-dense-extra/segments').glob('*.json.gz'):
    block = json.loads(gzip.decompress(f.read_bytes()))
    dense_rows += [dict(r, stream='dense_clip') for r in block['rows']]
  for r in dense_rows:
    if r['id'] in dense_truth:
      truth_row = dense_truth[r['id']]
      assert (r['frame_id'], r['camera_eof_ns']) == (truth_row['frame_id'], truth_row['camera_eof_ns'])
  # Keep streams distinct even when a physical ID occurs in both corpora.
  # Probabilities can be shared; each display/odometry state is independent.
  merged = {r['id']: r for r in sequences+dense_rows}
  unique = list(merged.values())
  report = {'test_used_for_selection': False, 'device_measured': False,
    'accuracy_inference': 'PyTorch CUDA offline, ONNX parity checked separately; not CPU latency',
    'test_samples': len(test), 'dense_truth_samples': len(dense_truth), 'dense_observations': len(dense_rows),
    'route_observations': len(sequences), 'models': {}}
  from tools.lane_type.multi_model import NAMES  # noqa: TID251 - offline repository tooling
  validation_scores = {}
  for cohort in ('baseline', 'expanded'):
    for name in NAMES:
      key = cohort+'/'+name
      directory = root/('native-'+cohort)/name
      predictions = predict(unique, directory)
      threshold, selection = choose_thresholds(val, predictions)
      json_write(directory/'validation-thresholds.json', {'thresholds': threshold, 'attempts': selection, 'test_used': False})
      single = replay(sequences, predictions, threshold)
      temporal = replay(sequences, predictions, threshold, temporal=True)
      # dense clip replay starts from no prior state: startup, not marking-onset latency.
      dense_single = replay(dense_rows, predictions, threshold)
      dense_temporal = replay(dense_rows, predictions, threshold, temporal=True)
      groups = {'all': test, 'fcamera': [r for r in test if r['video_source']=='fcamera'],
        'qcamera': [r for r in test if r['video_source']=='qcamera'],
        'night': [r for r in test if r['lighting']=='night'], 'curve': [r for r in test if r['curve']],
        'low_speed': [r for r in test if 0.1 <= r['speed_mps'] < 2],
        'stopped': [r for r in test if r['speed_mps'] < .1], 'lane_change': [r for r in test if r['lane_change']],
        'unobservable': [r for r in test if len(merged[r['id']]['heads']) < 2]}
      dense_gt = [dict(r, **{k: dense_truth[r['id']][k] for k in ('type', 'color')})
                  for r in dense_rows if r['id'] in dense_truth]
      raw_val = [raw_prediction(predictions[r['id']]) for r in val]
      vm = {h: metrics(val, raw_val, h, c) for h, c in CLASSES.items()}
      validation_scores[key] = sum(vm[h]['macro_f1'] or 0 for h in CLASSES)/2
      result = {'thresholds': threshold, 'validation_raw': vm,
        'single': {s: summarize(rs, predictions, single) for s, rs in groups.items()},
        'temporal': {s: summarize(rs, predictions, temporal) for s, rs in groups.items()},
        'dense_single': summarize(dense_gt, predictions, dense_single),
        'dense_temporal': summarize(dense_gt, predictions, dense_temporal),
        'transitions_single': transitions(dense_rows, dense_single), 'transitions_temporal': transitions(dense_rows, dense_temporal),
        'marking_onset_delay_seconds': None,
        'marking_onset_delay_reason': 'No independently annotated physical marking-onset timestamp. Clip startup is reported separately.'}
      delays = []
      for stream in sorted({(r['route'], r['segment'], r['side'], r['video_source']) for r in dense_rows}):
        group = sorted([r for r in dense_rows if (r['route'],r['segment'],r['side'],r['video_source'])==stream], key=lambda r:r['camera_eof_ns'])
        relevant = [r for r in group if r['id'] in dense_truth and dense_truth[r['id']]['type']=='dashed']
        if relevant:
          found = next((r for r in relevant if dense_temporal[r['id']]['semantic']=='white_dashed'), None)
          delays.append({'stream': list(stream), 'first_correct_dashed_from_clip_start_s':
                        (found['camera_eof_ns']-group[0]['camera_eof_ns'])/1e9 if found else None,
                        'censored_if_null': True, 'observed_clip_s':(group[-1]['camera_eof_ns']-group[0]['camera_eof_ns'])/1e9})
      result['startup_delays'] = delays
      report['models'][key] = result
      json_write(root/'predictions'/f'{cohort}-{name}.json', {'probabilities': predictions,
        'dense_single': dense_single, 'dense_temporal': dense_temporal})
      json_write(root/'comparison-in-progress.json', report)
      print('EVALUATED', key, 'raw', result['single']['all']['raw'], flush=True)
  native_route = replay(sequences, None, None, temporal=True, native_only=True)
  native_dense = replay(dense_rows, None, None, temporal=True, native_only=True)
  dense_gt = [dict(r, **{k: dense_truth[r['id']][k] for k in ('type','color')}) for r in dense_rows if r['id'] in dense_truth]
  report['native_temporal_only'] = {'all': summarize(test, None, native_route), 'dense': summarize(dense_gt, None, native_dense),
                                   'transitions': transitions(dense_rows, native_dense)}
  report['validation_selection_scores'] = validation_scores
  expanded = {k:v for k,v in validation_scores.items() if k.startswith('expanded/')}
  sizes = {key: json.loads((root/'native-expanded'/key.split('/')[1]/'manifest.json').read_text())['parameters'] for key in expanded}
  report['validation_best'] = max(expanded, key=lambda k: (expanded[k], -sizes[k]))
  report['adoption'] = 'hold; no device validation, no independent human audit, native visibility/pixel evidence experimental'
  json_write(root/'comparison.json', report)
  return report


def refresh_dense(root):
  """Correct legacy human-readable aliases using exact scene/frame/EOF keys.

  Uses the already saved predictions and validation thresholds. Does not train,
  change rules, select a new model, or evaluate another operating threshold.
  """
  path = root/'comparison.json'
  report = json.loads(path.read_text(encoding='utf-8'))
  backup = root/'comparison-diagnostic-before-dense-alias-fix.json'
  if not backup.exists():
    backup.write_bytes(path.read_bytes())
  source = json.loads(Path('.analysis/archive/2026-10-18/lane-roi-quality/policy/fresh-sequences/reviewed-manifest.json').read_text(encoding='utf-8'))
  truth = {r['scene_id']+'-'+r['video_source']: r for r in source if r['verified']}
  rows = []
  for f in (root/'native-dense-extra/segments').glob('*.json.gz'):
    rows += [dict(r, stream='dense_clip') for r in json.loads(gzip.decompress(f.read_bytes()))['rows']]
  gt = []
  for r in rows:
    if r['id'] in truth:
      t = truth[r['id']]
      assert (r['frame_id'],r['camera_eof_ns']) == (t['frame_id'],t['camera_eof_ns'])
      gt.append(dict(r,type=t['type'],color=t['color']))
  report['dense_truth_samples'] = len(gt)
  report['dense_alias_note'] = ('Legacy aliases resolved by exact scene_id/quality plus frameId/EOF equality. '+
                               'No nearest-time mapping, retraining or threshold change.')
  for key, result in report['models'].items():
    probabilities = json.loads((root/'predictions'/(key.replace('/','-')+'.json')).read_text(encoding='utf-8'))['probabilities']
    single = replay(rows, probabilities, result['thresholds'])
    temporal = replay(rows, probabilities, result['thresholds'], temporal=True)
    result.update(dense_single=summarize(gt, probabilities, single), dense_temporal=summarize(gt, probabilities, temporal),
                  transitions_single=transitions(rows,single), transitions_temporal=transitions(rows,temporal))
    result['startup_delays'] = []
    for stream in sorted({(r['route'],r['segment'],r['side'],r['video_source']) for r in rows}):
      group=sorted([r for r in rows if (r['route'],r['segment'],r['side'],r['video_source'])==stream],key=lambda r:r['camera_eof_ns'])
      relevant=[r for r in group if r['id'] in truth and truth[r['id']]['type']=='dashed']
      if relevant:
        found=next((r for r in relevant if temporal[r['id']]['semantic']=='white_dashed'),None)
        result['startup_delays'].append({'stream':list(stream),'first_correct_dashed_from_clip_start_s':
          (found['camera_eof_ns']-group[0]['camera_eof_ns'])/1e9 if found else None,'censored_if_null':True,
          'observed_clip_s':(group[-1]['camera_eof_ns']-group[0]['camera_eof_ns'])/1e9})
  native=replay(rows,None,None,temporal=True,native_only=True)
  report['native_temporal_only'].update(dense=summarize(gt,None,native),transitions=transitions(rows,native))
  json_write(root/'dense-display-details.json',{'truth_samples':len(gt),'native_temporal':native})
  json_write(path,report)
  return report


def compare_active(root):
  from PIL import Image
  from openpilot.selfdrive.carrot.lane_type.daemon import Classifier
  old_root = Path('.analysis/archive/2026-10-18/lane-type-retrain/data')
  old = {r['id']: r for r in json.loads((old_root/'roi-audited-manifest.json').read_text(encoding='utf-8')) if r['verified']}
  truth = [r for r in json.loads((root/'native-test.json').read_text(encoding='utf-8')) if r['id'] in old]
  assert all((r['type'],r['color']) == (old[r['id']]['type'],old[r['id']]['color']) for r in truth)
  results = {}
  for engine in ('onnx','opencv'):
    classifier = Classifier(engine)
    predictions = []
    for r in truth:
      image = np.asarray(Image.open(old_root/old[r['id']]['image']).convert('RGB'))
      p = classifier.infer([image])[0]
      predictions.append({h:p.get(h+'Candidate',p[h]) for h in CLASSES})
    results['active_6_46' if engine=='onnx' else 'legacy_opencv_6_46'] = {
      h:metrics(truth,predictions,h,c) for h,c in CLASSES.items()}
  from tools.lane_type.multi_model import NAMES  # noqa: TID251 - offline repository tooling
  for name in NAMES:
    probabilities = json.loads((root/'predictions'/f'expanded-{name}.json').read_text(encoding='utf-8'))['probabilities']
    predictions = [raw_prediction(probabilities[r['id']]) for r in truth]
    results['expanded/'+name] = {h:metrics(truth,predictions,h,c) for h,c in CLASSES.items()}
  report = {'samples':len(truth),'sample_ids':[r['id'] for r in truth],'same_truth_and_scene':True,
    'scope':'System comparison: ROI, resolution, input contract and training corpus differ; not a causal isolated ROI improvement.',
    'results':results}
  json_write(root/'matched-active-comparison.json',report)
  return report


def benchmark(directory, manifest, output, seconds=32., active=False):
  import cv2
  import psutil
  cv2.setNumThreads(1)
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  rows = [r for r in rows if r['split']=='validation'][:2]
  cache = {}
  if active:
    old = json.loads(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/roi-audited-manifest.json').read_text(encoding='utf-8'))
    old = [r for r in old if r['verified'] and r['split']=='validation'][:2]
    from PIL import Image
    images = [np.asarray(Image.open(Path('.analysis/archive/2026-10-18/lane-type-retrain/data')/r['image'])) for r in old]
    a, b = input_blob(images), None
  else:
    a,b = [input_blob([read_image(manifest.parent, r['heads'][h], cache) for r in rows]) for h in ('type','color')]
  process = psutil.Process()
  before = process.memory_info().rss
  path = directory/'lane_type.onnx'
  net = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(),np.uint8))
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  def infer():
    if active:
      net.setInput(a)
      return net.forward(['line_type','color'])
    else:
      net.setInput(a, 'type_image')
      net.setInput(b, 'color_image')
    return net.forward(['type_prob','color_prob'])
  for _ in range(10):
    infer()
  after = process.memory_info().rss
  samples = []
  peak = after
  start = time.perf_counter()
  cpu0 = time.process_time()
  for i in range(round(seconds*5)):
    t = time.perf_counter()
    infer()
    samples.append((time.perf_counter()-t)*1000)
    peak=max(peak,process.memory_info().rss)
    time.sleep(max(0,start+(i+1)/5-time.perf_counter()))
  elapsed = time.perf_counter()-start
  cpu = time.process_time()-cpu0
  result={'environment':'Windows PC OpenCV DNN CPU, NOT Snapdragon/comma','calls':len(samples),'sides_per_call':2,
    'seconds':elapsed,'calls_hz':len(samples)/elapsed,'cpu_one_core_percent':cpu/elapsed*100,
    'cpu_counter_seconds':cpu,'cpu_measurement_status':'process_time_counter' if cpu else 'below_counter_resolution',
    'rss_before_load_mib':before/2**20,'rss_after_load_mib':after/2**20,
    'rss_sample_max_mib':peak/2**20,'latency_ms':{'mean':float(np.mean(samples)),'p95':float(np.percentile(samples,95)),
    'p99':float(np.percentile(samples,99))},'scope':'pre-extracted ROI normalization excluded; two heads on two sides; feature extraction/IPC/display excluded',
    'device_cpu_latency':'unmeasured: no comma','modeld_controlsd':'unmeasured: no comma','frame_drops':'unmeasured: no comma'}
  json_write(output, result)
  return result


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,default=Path('.analysis/lane-model-comparison-20261018'))
  parser.add_argument('--benchmark',type=Path)
  parser.add_argument('--manifest',type=Path)
  parser.add_argument('--output',type=Path)
  parser.add_argument('--active',action='store_true')
  parser.add_argument('--seconds',type=float,default=32.)
  parser.add_argument('--refresh-dense',action='store_true')
  parser.add_argument('--compare-active',action='store_true')
  args=parser.parse_args()
  if args.compare_active:
    compare_active(args.root)
  elif args.refresh_dense:
    refresh_dense(args.root)
  elif args.benchmark:
    benchmark(args.benchmark, args.manifest, args.output, args.seconds, args.active)
  else:
    run(args.root)
