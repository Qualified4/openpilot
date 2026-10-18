import numpy as np
import pytest

from tools.lane_profile import scanner as S  # noqa: TID251
from tools.lane_profile.labels import Lookup, interval, merge_frames  # noqa: TID251
from tools.lane_profile.synthetic import to_nv12_planes  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251


def drive(speed, frames, *, step_ns=50_000_000, solid_m=12., side='left', **kw):
  scanner = S.ProfileScanner(side, solid_m=solid_m)
  out = []
  for i in range(frames):
    pose = np.array([speed*i*step_ns/1e9, 0., 0.])
    rgb, line, k, rpy, _, _, leads = synthetic_frame(pose, **kw)
    y, uv = to_nv12_planes(rgb)
    out.append(scanner.update(y, uv, line, 1., rpy, k, i*step_ns, float(pose[0]), leads=leads))
  return out


def test_strip_rows_are_integer_source_rows():
  _, line, k, rpy, *_ = synthetic_frame(np.zeros(3))
  geo = S.strip_geometry(line, rpy, k, 1344, 760)
  assert np.all(np.diff(geo['ys']) == 1)
  assert np.all(geo['my'] == geo['ys'][:, None])  # no vertical interpolation
  assert np.all(np.diff(geo['d']) < 0)  # far -> near: rows run from top (far) to bottom (near)
  assert geo['d'].max() <= S.FAR_M + .1 and geo['d'].min() >= S.NEAR_M - .1


def test_gap_and_run_helpers():
  P, R, U = 1, 0, None
  states = [P]*8 + [R]*6 + [P]*8
  assert len(S.find_gaps(states)) == 1
  assert S.find_gaps([P]*8 + [U]*6 + [P]*8) == []  # unknown is never a gap
  assert S.find_gaps([P]*8 + ['r']*6 + [P]*8) == []  # unconfirmed road is never a gap
  assert S.find_gaps([P]*8 + [R]*3 + [P]*8) == []  # shorter than GAP_MIN_M
  runs = S.paint_runs([R]*5 + [P]*20 + [U]*2 + [P]*20 + [R]*5)
  assert len(runs) == 1 and runs[0]['length_m'] == pytest.approx(42*S.BIN_M) and runs[0]['bounded']
  assert len(S.paint_runs([P]*20 + [U]*4 + [P]*20)) == 2  # 1m of unknown splits a run


def test_moving_dashed_and_solid_are_classified():
  dashed = drive(10., 30, dashed=True)
  assert dashed[-1]['type'] == 'dashed' and dashed[-1]['color'] == 'yellow'
  solid = drive(10., 20, white=True)
  assert solid[-1]['type'] == 'solid' and solid[-1]['color'] == 'white'
  assert not any(r['type'] == 'dashed' for r in solid)


def test_stopped_dashed_is_never_confirmed_from_one_viewpoint():
  out = drive(0., 15, dashed=True)
  assert not any(r['type'] == 'dashed' for r in out)  # road bins need travel between views


def test_occlusion_and_shadow_never_create_a_gap():
  for kw in ({'occluded': True}, {'shadow': True}, {'shadow': True, 'white': True}, {'crosswalk': True}):
    assert not any(r['type'] == 'dashed' for r in drive(5., 25, **kw)), kw


def test_lane_change_resets_evidence():
  scanner = S.ProfileScanner('left')
  rgb, line, k, rpy, *_ = synthetic_frame(np.zeros(3))
  y, uv = to_nv12_planes(rgb)
  scanner.update(y, uv, line, 1., rpy, k, 0, 0.)
  generation = scanner.generation
  r = scanner.update(y, uv, line, 1., rpy, k, 50_000_000, 0., lane_change=True)
  assert r['reset'] and r['type'] == 'unknown' and scanner.generation == generation + 1 and not scanner.bins


def test_interval_lookup():
  a = interval('r', 1, 'left', 100, 200, 'dashed', 'white', status='ai_reviewed', source='t')
  b = interval('r', 1, 'left', 150, 300, 'solid', 'white', status='ai_reviewed', source='t')
  c = interval('r', 1, 'left', 400, 500, 'solid', 'yellow', status='auto_proposed', source='scanner')
  look = Lookup([a, b, c])
  assert look('r', 1, 'left', 120)['type'] == 'dashed'
  assert look('r', 1, 'left', 170) is None  # overlapping labels are not truth
  assert look('r', 1, 'left', 450) is None  # scanner proposals are never truth
  merged = merge_frames([{'eof': i*50_000_000, 'type': 'dashed', 'color': 'white'} for i in range(5)])
  assert len(merged) == 1 and merged[0]['frames'] == 5
