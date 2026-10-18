"""Interval ground truth: one label per (route, segment, side, time span), not per frame ROI.

A label covers a continuous driving interval of one boundary. Frames inherit the label of the
interval containing their camera EOF. Transitions, intersections and anything not judged are
'unknown' and excluded from accuracy (but still counted for UNKNOWN/hold behaviour).

status:
  human_confirmed  - confirmed by the user in the review page (only this is ground truth for adoption)
  ai_reviewed      - an assistant looked at the video; usable for development, must be confirmed
  auto_proposed    - produced by the scanner (origin 'auto'); never ground truth
"""
import json
from pathlib import Path

SCHEMA = 'ccnc-lane-interval-v1'
TYPES, COLORS = ('dashed', 'solid', 'unknown'), ('white', 'yellow', 'unknown')
STATUSES = ('human_confirmed', 'ai_reviewed', 'auto_proposed', 'rejected')


def interval(route, segment, side, start_ns, end_ns, type_, color, *, status, source, double=False, note='', **extra):
  if side not in ('left', 'right') or type_ not in TYPES or color not in COLORS or status not in STATUSES:
    raise ValueError('invalid interval label')
  if not 0 <= start_ns <= end_ns:
    raise ValueError('invalid interval span')
  return {'route': route, 'segment': int(segment), 'side': side, 'start_eof_ns': int(start_ns), 'end_eof_ns': int(end_ns),
          'type': type_, 'color': color, 'double': bool(double), 'member': 'ego_side', 'status': status,
          'source': source, 'note': note, **extra}


def load(path):
  data = json.loads(Path(path).read_text(encoding='utf-8'))
  if data.get('schema') != SCHEMA:
    raise ValueError('unknown label schema')
  for item in data['intervals']:
    interval(item['route'], item['segment'], item['side'], item['start_eof_ns'], item['end_eof_ns'], item['type'],
             item['color'], status=item['status'], source=item['source'])
  return data['intervals']


def save(path, intervals, **meta):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps({'schema': SCHEMA, **meta, 'intervals': intervals}, ensure_ascii=False, indent=1),
                        encoding='utf-8')


class Lookup:
  """Ground-truth lookup for evaluation. Automatic labels can never be ground truth."""
  def __init__(self, intervals, statuses=('human_confirmed', 'ai_reviewed')):
    if 'auto_proposed' in statuses:
      raise ValueError('auto_proposed labels are training candidates, never evaluation ground truth')
    self.index = {}
    for item in intervals:
      if item['status'] in statuses:
        self.index.setdefault((item['route'], int(item['segment']), item['side']), []).append(item)

  def __call__(self, route, segment, side, eof_ns):
    hits = [i for i in self.index.get((route, int(segment), side), ()) if i['start_eof_ns'] <= eof_ns <= i['end_eof_ns']]
    if len(hits) != 1:
      return None  # unlabelled or overlapping (conflicting) labels are not truth
    return hits[0]


def merge_frames(frames, *, key=('type', 'color'), min_frames=1, max_gap_ns=150_000_000):
  """Merge consecutive per-frame proposals of one boundary into interval proposals."""
  out = []
  for f in sorted(frames, key=lambda f: f['eof']):
    value = tuple(f[k] for k in key)
    if out and out[-1]['value'] == value and f['eof'] - out[-1]['end'] <= max_gap_ns:
      out[-1]['end'], out[-1]['frames'] = f['eof'], out[-1]['frames'] + 1
    else:
      out.append({'value': value, 'start': f['eof'], 'end': f['eof'], 'frames': 1})
  return [o for o in out if o['frames'] >= min_frames]


PRIORITY = ('human', 'auto', 'auto_pool', 'ai_reviewed')  # higher sources replace only the overlapped time
MIN_PIECE_NS = 1_000_000_000


def origin_of(item):
  if item['status'] == 'human_confirmed':
    return 'human'
  if item['status'] == 'auto_proposed':
    return item.get('origin', 'auto')
  return item['status']


def _subtract(span, cuts):
  pieces = [span]
  for s, e in cuts:
    pieces = [q for a, b in pieces for q in ((a, min(b, s - 1)), (max(a, e + 1), b)) if q[1] >= q[0]]
  return pieces


def training_intervals(intervals, split_of, splits=('train', 'validation')):
  """Training labels of train/validation routes only (``split_of(route)``); test routes are excluded
  whatever the source. A higher-priority source replaces a lower one only over the overlapped time:
  the rest of the lower-priority interval is kept if it is at least MIN_PIECE_NS long."""
  out, taken = [], {}
  for origin in PRIORITY:
    for item in intervals:
      if origin_of(item) != origin or split_of(item['route']) not in splits or item['type'] not in ('dashed', 'solid'):
        continue
      key = (item['route'], int(item['segment']), item['side'])
      for a, b in _subtract((item['start_eof_ns'], item['end_eof_ns']), taken.get(key, [])):
        if b - a >= MIN_PIECE_NS or origin == 'human':
          out.append(dict(item, origin=origin, start_eof_ns=a, end_eof_ns=b, split=split_of(item['route']),
                          clipped=(a, b) != (item['start_eof_ns'], item['end_eof_ns'])))
      taken.setdefault(key, []).append((item['start_eof_ns'], item['end_eof_ns']))
  return out
