"""User-approved corrections of manifest labels (label_fixes.json), applied when a manifest is read, and exclusion windows
(stretches that are not a lane line: dropped from scoring, see excluded()).

Source interval files and frozen manifests are never rewritten: a fix names a route--segment, side and eof range in seconds,
and every row inside it whose type is still `from_type` gets the new type/color. Rows already corrected are left alone, so
applying twice changes nothing. Fixes whose new type is outside `types` are skipped (a 2-class StripNet v1-v3 set keeps
its labels; the double-line classes exist only in the extended vocabulary, dataset_io.TYPES)."""
import json
from pathlib import Path

FIXES = Path(__file__).resolve().parent/'label_fixes.json'


def load(path=FIXES):
  return json.loads(Path(path).read_text(encoding='utf-8'))['fixes']


def load_exclusions(path=FIXES):
  return json.loads(Path(path).read_text(encoding='utf-8')).get('exclusions', [])


def excluded(route_segment, side, eof_ns, exclusions=None):
  """Boolean mask (array eof_ns) or bool (scalar): frames inside an exclusion window -- not a lane line, drop from scoring."""
  import numpy as np
  exclusions = load_exclusions() if exclusions is None else exclusions
  t = np.asarray(eof_ns)/1e9
  mask = np.zeros(t.shape, bool)
  for x in exclusions:
    if x['route_segment'] == route_segment and x['side'] == side:
      mask |= (t >= x['start_s']) & (t <= x['end_s'])
  return mask if mask.ndim else bool(mask)


def apply(rows, fixes=None, types=None):
  """Mutates rows in place; returns {fix id: rows changed}. Each changed row keeps its old label in row['label_fix_from']."""
  fixes = load() if fixes is None else fixes
  fixes = [f for f in fixes if types is None or f['type'] in types]
  changed = {f['id']: 0 for f in fixes}
  for r in rows:
    key, t = f"{r['route']}--{r['segment']}", r['eof']/1e9
    for f in fixes:
      if key == f['route_segment'] and r['side'] == f['side'] and f['start_s'] <= t <= f['end_s'] and r['type'] == f['from_type']:
        r['label_fix_from'] = dict(type=r['type'], color=r['color'], fix=f['id'])
        r['type'], r['color'] = f['type'], f['color']
        changed[f['id']] += 1
        break
  return changed
