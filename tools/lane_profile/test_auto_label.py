import pytest

from tools.lane_profile.auto_label import features, strong, tier  # noqa: TID251
from tools.lane_profile.labels import Lookup, interval, training_intervals  # noqa: TID251


def frames(n, side='left', t=1, c=1, gaps=3, solid=0., onnx=(1, 1, 1, 1), lc=False, chroma=(0., 0., 5)):
  # entry: type, color, reason, solid_run_m, n_gaps, reset, paint, road, rows, chroma, paint_since_gap
  return [{'eof': i*50_000_000, 'index': i, 'v': 20., 'lc': lc, side: [t, c, 'observed', solid, gaps, 0, 50, 50, 200, list(chroma), 0.],
           'onnx_' + side: list(onnx)} for i in range(n)]


def test_consistent_dashed_with_onnx_agreement_is_auto():
  x = features(frames(100), 'left', 12.)
  assert x['type'] == 'dashed' and x['color'] == 'white' and strong(x)
  assert tier(x) == ('auto', [])


def test_onnx_raw_doubt_blocks_auto_but_is_not_a_review_conflict_by_itself():
  x = features(frames(100, onnx=(0, 0, 2, 1)), 'left', 12.)  # guarded ONNX unknown, raw says solid
  t, reasons = tier(x)
  assert t == 'review' and reasons == ['onnx_type_disagrees']
  assert x['onnx_guarded_type_disagree'] == 0  # main() defers this (no confident opposite)


def test_confident_onnx_opposite_is_measured():
  x = features(frames(100, onnx=(2, 1, 2, 1)), 'left', 12.)
  assert x['onnx_guarded_type_disagree'] == 1. and strong(x)


def test_weak_evidence_lane_change_and_unknown_are_not_auto():
  assert 'weak_gap_evidence' in tier(features(frames(100, gaps=1), 'left', 12.))[1]
  assert tier(features(frames(100, lc=True), 'left', 12.))[0] == 'defer'
  assert tier(features(frames(100, t=0, gaps=0), 'left', 12.))[0] == 'defer'  # type is re-derived from gaps/solid run
  solid = features(frames(100, t=2, gaps=0, solid=13., onnx=(2, 1, 2, 1)), 'left', 12.)
  assert 'short_solid_evidence' in tier(solid)[1]
  yellow = features(frames(100, t=2, c=2, gaps=0, solid=16., onnx=(2, 0, 2, 1), chroma=(-3., 2., 5)), 'left', 12.)
  assert 'weak_yellow_chroma' in tier(yellow)[1]


def test_auto_labels_are_never_ground_truth_and_humans_replace_only_the_overlap():
  split = {'r': 'train', 't': 'test'}.get
  auto = dict(interval('r', 1, 'left', 0, 10_000_000_000, 'dashed', 'white', status='auto_proposed', source='auto'), origin='auto')
  pool = dict(interval('r', 1, 'right', 0, 5_000_000_000, 'solid', 'white', status='auto_proposed', source='auto'), origin='auto_pool')
  test_auto = dict(auto, route='t')
  human = interval('r', 1, 'left', 4_000_000_000, 6_000_000_000, 'solid', 'white', status='human_confirmed', source='page')
  test_human = dict(human, route='t')
  with pytest.raises(ValueError):
    Lookup([auto], statuses=('auto_proposed',))
  assert Lookup([auto, human])('r', 1, 'left', 2_000_000_000) is None
  train = training_intervals([auto, pool, test_auto, human, test_human], split)
  spans = sorted((i['origin'], i['side'], i['start_eof_ns'], i['end_eof_ns'], i['type']) for i in train)
  assert spans == [('auto', 'left', 0, 3_999_999_999, 'dashed'), ('auto', 'left', 6_000_000_001, 10_000_000_000, 'dashed'),
                   ('auto_pool', 'right', 0, 5_000_000_000, 'solid'), ('human', 'left', 4_000_000_000, 6_000_000_000, 'solid')]
  assert all(i['route'] != 't' for i in train)  # test routes never train
