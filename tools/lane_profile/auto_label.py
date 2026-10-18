"""Automatic interval labels from whole-log scanner runs, split into confidence tiers.

Every segment/side is cut into intervals (make_review.proposals). Each interval gets the scanner's
majority type/colour and agreement features: scanner temporal consistency, physical evidence strength,
lane-change/occlusion exposure, and agreement with the active ONNX run on the same frames.

  auto      - auto-approval candidate: consistent scanner verdict with strong evidence and no ONNX
              objection. status auto_proposed, origin auto. Usable as TRAINING label candidates for
              train/validation routes only; never accuracy ground truth.
  review    - only real conflicts of strong evidence: a strong scanner verdict against a confident (guarded)
              ONNX verdict, or an auto/strong interval against an existing reviewed frame label; plus a
              small stratified random AUDIT of auto intervals (precision estimate) and of the deferred
              'strong scanner, raw ONNX doubts' pool (can it be promoted later?).
  defer     - everything else that is uncertain (mostly UNKNOWN, transitions, unstable, weak evidence,
              raw-ONNX doubt, lane change): no label, dropped from training, not reviewed.

Tier rules are fixed before looking at existing reviewed frame labels. Those labels are only used
afterwards (a) to report how often auto intervals agree with them and (b) to send conflicts to review.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from tools.lane_profile.analyze_logs import kind, load  # noqa: TID251
from tools.lane_profile.labels import SCHEMA  # noqa: TID251
from tools.lane_profile.make_review import COLORS, proposals  # noqa: TID251
from tools.lane_profile.run_logs import REVIEWED  # noqa: TID251

ONNX_TYPES = ('unknown', 'dashed', 'solid')
MIN_S = 3.
# Fixed tier thresholds (not fitted to reviewed labels).
AUTO = {'type_decided': .8, 'type_consistency': .97, 'color_decided': .6, 'color_consistency': .97,
        'dashed_two_gap_fraction': .5, 'solid_median_run_m': 15., 'onnx_raw_type_agree': .6,
        'onnx_guarded_type_disagree': .05, 'onnx_raw_white_agree': .6, 'yellow_chroma_score': 8.}
REVIEW_MIN_DECIDED = .5
HARD_ONNX_DISAGREE = .2
AUDIT_AUTO, AUDIT_POOL, AUDIT_SEED = 20, 10, 20261018


def strong(x):
  """Scanner verdict that is stable and physically well supported, regardless of ONNX."""
  return (x['type_decided'] >= AUTO['type_decided'] and x['type_consistency'] >= AUTO['type_consistency'] and
          x['lane_change_fraction'] == 0 and (x['type'] == 'dashed' and x['dashed_two_gap_fraction'] >= AUTO['dashed_two_gap_fraction'] or
                                               x['type'] == 'solid' and x['solid_median_run_m'] >= AUTO['solid_median_run_m']))


def features(frames, side, solid_m):
  types = [kind(f[side], solid_m) for f in frames]
  colors = [COLORS[f[side][1]] for f in frames]
  dt = [t for t in types if t != 'unknown']
  dc = [c for c in colors if c != 'unknown']
  t_major = Counter(dt).most_common(1)[0][0] if dt else 'unknown'
  c_major = Counter(dc).most_common(1)[0][0] if dc else 'unknown'
  out = {'frames': len(frames), 'duration_s': (frames[-1]['eof'] - frames[0]['eof'])/1e9,
         'type': t_major, 'color': c_major,
         'type_decided': len(dt)/len(frames), 'type_consistency': dt.count(t_major)/len(dt) if dt else 0.,
         'color_decided': len(dc)/len(frames), 'color_consistency': dc.count(c_major)/len(dc) if dc else 0.,
         'dashed_frames': dt.count('dashed')/len(frames), 'solid_frames': dt.count('solid')/len(frames),
         'lane_change_fraction': float(np.mean([f['lc'] for f in frames])), 'median_speed': float(np.median([f['v'] for f in frames])),
         'resets': int(sum(f[side][5] for f in frames))}
  dashed = [f[side][4] for f, t in zip(frames, types, strict=True) if t == 'dashed']
  solid = [f[side][3] for f, t in zip(frames, types, strict=True) if t == 'solid']
  out['dashed_two_gap_fraction'] = float(np.mean([g >= 2 for g in dashed])) if dashed else 0.
  out['solid_median_run_m'] = float(np.median(solid)) if solid else 0.
  chroma = [f[side][9] for f in frames if len(f[side]) > 9 and f[side][9]]
  out['chroma_score'] = float(np.median([c[1] - c[0] for c in chroma])) if chroma else None
  onnx = [f.get('onnx_' + side) for f in frames]
  onnx = [o for o in onnx if o]
  if onnx:
    raw_t = [ONNX_TYPES[o[2]] for o in onnx]
    raw_c = [COLORS[o[3]] for o in onnx]
    guard = [ONNX_TYPES[o[0]] for o in onnx if o[0]]
    out.update(onnx_frames=len(onnx), onnx_raw_type_agree=raw_t.count(t_major)/len(raw_t),
               onnx_raw_color_agree=raw_c.count(c_major)/len(raw_c),
               onnx_guarded_decided=len(guard)/len(onnx),
               onnx_guarded_type_disagree=sum(g != t_major for g in guard)/len(onnx))
  else:
    out.update(onnx_frames=0, onnx_raw_type_agree=None, onnx_raw_color_agree=None, onnx_guarded_decided=0.,
               onnx_guarded_type_disagree=0.)
  return out


def tier(x):
  """Return (tier, reasons). Reasons explain why an interval is not auto-approved."""
  if x['duration_s'] < MIN_S or x['type_decided'] < REVIEW_MIN_DECIDED or x['lane_change_fraction'] > .5:
    return 'defer', ['mostly_unknown_or_short' if x['lane_change_fraction'] <= .5 else 'lane_change']
  reasons = []
  if x['type_decided'] < AUTO['type_decided']:
    reasons.append('scanner_often_unknown')
  if x['type_consistency'] < AUTO['type_consistency']:
    reasons.append('dashed_solid_transition' if min(x['dashed_frames'], x['solid_frames']) >= .1 else 'unstable_type')
  if x['lane_change_fraction'] > 0:
    reasons.append('lane_change_inside')
  if x['type'] == 'dashed' and x['dashed_two_gap_fraction'] < AUTO['dashed_two_gap_fraction']:
    reasons.append('weak_gap_evidence')
  if x['type'] == 'solid' and x['solid_median_run_m'] < AUTO['solid_median_run_m']:
    reasons.append('short_solid_evidence')
  if x['onnx_raw_type_agree'] is None:
    reasons.append('no_onnx')
  elif x['onnx_raw_type_agree'] < AUTO['onnx_raw_type_agree'] or x['onnx_guarded_type_disagree'] > AUTO['onnx_guarded_type_disagree']:
    reasons.append('onnx_type_disagrees')
  if x['color'] == 'unknown' or x['color_decided'] < AUTO['color_decided'] or x['color_consistency'] < AUTO['color_consistency']:
    reasons.append('color_uncertain')
  elif x['color'] == 'white' and (x['onnx_raw_color_agree'] or 0) < AUTO['onnx_raw_white_agree']:
    reasons.append('onnx_color_disagrees')
  elif x['color'] == 'yellow' and (x['chroma_score'] or 0) < AUTO['yellow_chroma_score']:
    reasons.append('weak_yellow_chroma')  # the ONNX misses yellow (recall 0/21 on test): chroma decides
  return ('auto' if not reasons else 'review'), reasons


WEIGHT = {'onnx_confident_opposite': 4., 'onnx_type_disagrees': 3., 'anchor_conflict': 3., 'dashed_solid_transition': 2., 'onnx_color_disagrees': 1.5,
          'unstable_type': 1.5, 'weak_yellow_chroma': 1.2, 'color_uncertain': 1., 'weak_gap_evidence': 1., 'short_solid_evidence': 1.,
          'scanner_often_unknown': .7, 'lane_change_inside': .7, 'no_onnx': 1.}


def priority(x, reasons, split):
  """Importance: labelled seconds x how informative the disagreement is. False Dashed risk (scanner
  dashed, ONNX solid) and held-out test routes come first because they gate safety and evaluation."""
  w = max(WEIGHT.get(r, 1.) for r in reasons)
  if x['type'] == 'dashed' and 'onnx_type_disagrees' in reasons:
    w *= 1.5
  if split == 'test':
    w *= 1.5
  return round(w*min(x['duration_s'], 30.), 2)


def interval_record(r):
  x = r['features']
  return {'route': r['route'], 'segment': r['segment'], 'side': r['side'], 'start_eof_ns': r['start_eof_ns'],
          'end_eof_ns': r['end_eof_ns'], 'type': x['type'], 'color': x['color'], 'double': False, 'member': 'ego_side',
          'status': 'auto_proposed', 'origin': 'auto', 'source': 'profile scanner v4 + active ONNX agreement (auto_label.py)',
          'note': '', 'quality': r['quality'], 'split': r['split'], 'training_eligible': r['split'] in ('train', 'validation'),
          'evaluation_eligible': False, 'auto_item_id': r['id'], 'audit': r.get('audit'), 'features': x}


def main(folder, output, solid_m):
  anchors = defaultdict(list)
  for path in REVIEWED:
    for r in json.loads(path.read_text(encoding='utf-8')):
      anchors[(r['route'], int(r['segment']), r['video_source'], r['side'])].append(r)
  rows = []
  for seg in load(folder):
    if not seg['frames']:
      continue
    for p in proposals(seg, solid_m, min_s=MIN_S):
      x = features(p['frames'], p['side'], solid_m)
      t, reasons = tier(x)
      inside = [a for a in anchors[(seg['route'], seg['segment'], seg['quality'], p['side'])] if p['start'] <= int(a['camera_eof_ns']) <= p['end']]
      rows.append({'id': f"{seg['id']}-{seg['quality']}-{p['side']}-{p['start']}", 'route': seg['route'], 'segment': seg['segment'],
                   'split': seg['split'], 'quality': seg['quality'], 'side': p['side'], 'start_eof_ns': p['start'], 'end_eof_ns': p['end'],
                   'start_index': p['frames'][0]['index'], 'end_index': p['frames'][-1]['index'], 'tier_before_anchors': t, 'tier': t,
                   'reasons': reasons, 'features': x, 'anchors': [{'index': int(a['video_index']), 'type': a['type'], 'color': a['color']} for a in inside]})
  # Agreement of auto intervals with existing reviewed frame labels, measured BEFORE anchors are used.
  check = Counter()
  for r in rows:
    for a in r['anchors']:
      check[(r['tier_before_anchors'], 'type_agree' if a['type'] == r['features']['type'] else 'type_disagree')] += 1
      check[(r['tier_before_anchors'], 'color_agree' if a['color'] == r['features']['color'] else 'color_disagree')] += 1
  have_f = {(r['route'], r['segment']) for r in rows if r['quality'] == 'fcamera'}
  for r in rows:
    # Both video qualities of one segment describe the same road: label/review it once (prefer fcamera).
    r['duplicate_of_fcamera'] = r['quality'] == 'qcamera' and (r['route'], r['segment']) in have_f
    x = r['features']
    conflict = any(a['type'] != x['type'] or a['color'] != x['color'] for a in r['anchors'])
    r['pool'] = None
    if r['tier'] == 'auto':
      if conflict:
        r['tier'], r['reasons'] = 'review', ['anchor_conflict']
    elif r['tier'] == 'review' and strong(x) and x['onnx_guarded_type_disagree'] >= HARD_ONNX_DISAGREE:
      r['reasons'] = ['onnx_confident_opposite'] + r['reasons']
    elif r['tier'] == 'review' and strong(x) and conflict:
      r['reasons'] = r['reasons'] + ['anchor_conflict']
    else:
      if r['tier'] == 'review' and strong(x) and set(r['reasons']) <= {'onnx_type_disagrees', 'onnx_color_disagrees'}:
        r['pool'] = 'strong_scanner_onnx_raw_doubt'
      r['tier'] = 'defer'
  unique = [r for r in rows if not r['duplicate_of_fcamera']]
  rng = np.random.default_rng(AUDIT_SEED)

  def sample(pool, n, tag):
    groups = defaultdict(list)
    for r in sorted(pool, key=lambda r: r['id']):
      groups[(r['features']['type'], r['features']['color'])].append(r)
    picked = []
    while len(picked) < min(n, len(pool)):
      for g in sorted(groups):
        if groups[g] and len(picked) < n:
          picked.append(groups[g].pop(int(rng.integers(len(groups[g])))))
    for r in picked:
      r['audit'] = tag
    return picked
  audits = (sample([r for r in unique if r['tier'] == 'auto'], AUDIT_AUTO, 'audit_auto') +
            sample([r for r in unique if r['pool']], AUDIT_POOL, 'audit_pool'))
  auto, review = [], []
  for r in unique:
    x = r['features']
    if r['tier'] == 'auto':
      auto.append(interval_record(r))
    elif r['tier'] == 'review':
      r['priority'] = 1000 + priority(x, r['reasons'], r['split'])
      review.append(r)
  for r in audits:
    r['reasons'] = [r['audit']] + r['reasons']
    r['priority'] = round(min(r['features']['duration_s'], 30.), 2)  # audits come after the conflicts
    review.append(r)
  review.sort(key=lambda r: -r['priority'])
  output.mkdir(parents=True, exist_ok=True)
  (output/'auto-labels.json').write_text(json.dumps({'schema': SCHEMA, 'origin': 'auto', 'thresholds': AUTO,
    'warning': 'auto_proposed labels: training candidates only (train/validation routes). Never accuracy ground truth.',
    'intervals': auto}, indent=1), encoding='utf-8')
  pool = [dict(interval_record(r), origin='auto_pool', source='profile scanner v4, strong evidence; raw ONNX objected (auto_label.py pool)')
          for r in unique if r['pool']]
  (output/'pool-labels.json').write_text(json.dumps({'schema': SCHEMA, 'origin': 'auto_pool', 'intervals': pool,
    'warning': 'deferred strong-scanner pool; training candidates only after audit. Never accuracy ground truth.'}, indent=1), encoding='utf-8')
  (output/'review-queue.json').write_text(json.dumps(review, indent=1), encoding='utf-8')
  (output/'all-intervals.json').write_text(json.dumps(rows), encoding='utf-8')

  def total(items, split=None):
    sel = [r for r in items if split is None or r['split'] == split]
    return {'count': len(sel), 'seconds': round(sum(r['features']['duration_s'] for r in sel), 1)}
  summary = {'intervals': len(rows), 'unique_intervals': len(unique),
             'tiers': {t: {s or 'all': total([r for r in unique if r['tier'] == t], s) for s in (None, 'train', 'validation', 'test')}
                       for t in ('auto', 'review', 'defer')},
             'auto_training_eligible': total([r for r in unique if r['tier'] == 'auto' and r['split'] in ('train', 'validation')]),
             'review_items': len(review), 'review_conflicts': sum(r['tier'] == 'review' for r in review),
             'review_audits': dict(Counter(r['audit'] for r in review if r.get('audit'))),
             'review_seconds': round(sum(r['features']['duration_s'] for r in review), 1),
             'deferred_strong_pool': sum(r['pool'] is not None for r in unique),
             'review_reasons': dict(Counter(x for r in review for x in r['reasons'])),
             'auto_by_value': dict(Counter(f"{a['type']}/{a['color']}" for a in auto)),
             'anchor_agreement_before_anchor_use': {f'{k[0]}/{k[1]}': v for k, v in check.items()}}
  (output/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  return summary

if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--input', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--solid-m', type=float, default=12.)
  a = p.parse_args()
  print(json.dumps(main(a.input, a.output, a.solid_m), indent=1, default=str))
