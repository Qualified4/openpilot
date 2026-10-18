"""Explicit review admission and conservative audit; never equates proposals to truth."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from tools.lane_type.prepare_dual_data import json_write  # noqa: TID251 - offline repository tooling

CODES = {'DW': ('dashed', 'white'), 'SW': ('solid', 'white'), 'SY': ('solid', 'yellow'), 'DY': ('dashed', 'yellow')}


def wilson(errors, count):
  if not count:
    return None
  z = 1.959963984540054
  p = errors/count
  return (p+z*z/(2*count)+z*math.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count)


def admit(root, decisions_paths, output, old_reviewed):
  proposals = json.loads((root/'proposals.json').read_text(encoding='utf-8'))
  decisions = {}
  for path in decisions_paths:
    for key, value in json.loads(path.read_text(encoding='utf-8')).items():
      if key in decisions and decisions[key] != value:
        raise ValueError('conflicting review decisions')
      decisions[key] = value
  old = {r['id']: r for r in json.loads(old_reviewed.read_text(encoding='utf-8')) if r['verified']}
  rows, excluded, audit = [], [], defaultdict(list)
  for p in proposals:
    review = decisions.get(p['id'])
    if review is None:
      continue
    code = review['decision']
    known = code in CODES
    if not known:
      excluded.append({'id': p['id'], 'reason': review['review_note']})
    # The automatic-rule error audit uses ONLY training routes. Validation and
    # test labels cannot approve/tune the label generator.
    if p['automatic_proposal'] and p['split'] == 'train' and p['id'] not in old:
      cell = p['proposed_type']+'/'+p['proposed_color']+'/'+p['video_source']
      audit[cell].append({'id': p['id'], 'known': known, 'error': known and CODES[code] != (p['proposed_type'], p['proposed_color']),
        'group': f"{p['route']}/{p['segment']}/{p['video_index']//200}", 'unidentifiable': not known})
    if not known:
      continue
    row = dict(p, type=CODES[code][0], color=CODES[code][1], verified=True,
      label_source='auto_individually_reviewed' if p['automatic_proposal'] and p['id'] not in old else 'visual_review',
      reviewer=review['reviewer'], review_note=review['review_note'], lighting=review['lighting'],
      source_root=str(root.resolve()), baseline_reviewed=p['id'] in old)
    if p['id'] in old and CODES[code] != (old[p['id']]['type'], old[p['id']]['color']):
      excluded.append({'id': p['id'], 'reason': 'new-ROI/old-label disagreement requires separate review'})
      continue
    for head in ('type', 'color'):
      row['heads'][head]['image'] = str((root/row['heads'][head]['image']).resolve())
    row['context'] = str((root/row['context']).resolve())
    rows.append(row)
  # Identical paired scene labels must agree; no quality inherits an unreviewed
  # label. The HD source is explicitly recorded when a reviewed pair exists.
  paired = defaultdict(list)
  for r in rows:
    paired[r['scene_id']].append(r)
  conflict = set()
  for group in paired.values():
    if len({(r['type'], r['color']) for r in group}) != 1:
      conflict.update(r['id'] for r in group)
    else:
      hd = next((r for r in group if r['video_source'] == 'fcamera'), None)
      if hd:
        for r in group:
          r['truth_reference'] = hd['id']
          r['paired_hd_reviewed'] = len(group) > 1
  rows = [r for r in rows if r['id'] not in conflict]
  excluded.extend({'id': key, 'reason': 'reviewed_exact_pair_label_conflict'} for key in sorted(conflict))
  audit_report = {}
  for cell, records in audit.items():
    n = len(records)
    errors = sum(r['error'] for r in records)
    unsupported = sum(r['unidentifiable'] for r in records)
    groups = defaultdict(list)
    for r in records:
      groups[r['group']].append(r)
    group_failures = sum(any(r['error'] or r['unidentifiable'] for r in v) for v in groups.values())
    audit_report[cell] = {'reviewed_proposals': n, 'identifiable_truth': n-unsupported, 'incorrect': errors,
      'unidentifiable': unsupported, 'conditional_error_rate': errors/(n-unsupported) if n > unsupported else None,
      'conservative_non_admissible_rate': (errors+unsupported)/n, 'temporal_scene_groups_10s': len(groups),
      'scene_group_error_upper95': wilson(group_failures, len(groups)),
      'unreviewed_auto_admission_allowed': wilson(group_failures, len(groups)) <= .02,
      'records': records}
  counts = Counter((r['split'], r['type'], r['color'], r['label_source'], r['video_source']) for r in rows)
  json_write(output, rows)
  json_write(output.with_name(output.stem+'-audit.json'), {'counts': [{'cell': list(k), 'count': v} for k, v in counts.items()],
    'admitted': len(rows), 'excluded': excluded, 'auto_rule_audit_train_only': audit_report,
    'unreviewed_auto_training_labels': 0, 'review_limitation': 'single assistant visual review, no independent human audit',
    'scene_group_definition': 'route/segment/10s: a correlation proxy, not guaranteed statistical independence'})
  return rows


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data', type=Path, required=True)
  parser.add_argument('--decisions', type=Path, nargs='+', required=True)
  parser.add_argument('--old-reviewed', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  admit(args.data, args.decisions, args.output, args.old_reviewed)
