from tools.lane_profile import dataset_io, label_fixes

FIX = dict(id='X', route_segment='r--3', side='left', start_s=10.0, end_s=12.0, from_type='solid', type='dbl_near', color='white')


def row(t, type_='solid', side='left', segment=3):
  return dict(route='r', segment=segment, side=side, eof=int(t*1e9), type=type_, color='yellow')


def test_apply_inside_range_only():
  rows = [row(9.9), row(10.0), row(11.0), row(12.0), row(12.1), row(11.0, side='right'), row(11.0, segment=4), row(11.0, 'dashed')]
  assert label_fixes.apply(rows, [FIX]) == {'X': 3}
  assert [r['type'] for r in rows] == ['solid', 'dbl_near', 'dbl_near', 'dbl_near', 'solid', 'solid', 'solid', 'dashed']
  assert rows[1]['color'] == 'white' and rows[1]['label_fix_from'] == dict(type='solid', color='yellow', fix='X')


def test_idempotent():
  rows = [row(11.0)]
  label_fixes.apply(rows, [FIX])
  assert label_fixes.apply(rows, [FIX]) == {'X': 0} and rows[0]['type'] == 'dbl_near'


def test_vocabulary_filter_keeps_two_class_sets():
  rows = [row(11.0)]
  assert label_fixes.apply(rows, [FIX], types=dataset_io.TYPES_V1) == {}
  assert rows[0]['type'] == 'solid' and 'label_fix_from' not in rows[0]


def test_shipped_fixes_are_valid():
  for f in label_fixes.load():
    assert f['type'] in dataset_io.TYPES and f['color'] in dataset_io.COLORS and f['from_type'] in dataset_io.TYPES
    assert f['start_s'] < f['end_s'] and f['side'] in ('left', 'right')


def test_excluded_window():
  X = [dict(id='Z', set='s', route_segment='r--3', side='left', start_s=10.0, end_s=12.0)]
  assert label_fixes.excluded('r--3', 'left', int(11e9), X) is True
  assert label_fixes.excluded('r--3', 'right', int(11e9), X) is False
  assert list(label_fixes.excluded('r--3', 'left', [int(9e9), int(10e9), int(12.5e9)], X)) == [False, True, False]
  for x in label_fixes.load_exclusions():
    assert x['start_s'] < x['end_s'] and x['side'] in ('left', 'right')
