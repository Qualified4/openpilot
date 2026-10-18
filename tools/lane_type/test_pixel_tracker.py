import json

import numpy as np
import pytest
import cv2

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import PaintTracker, nearest_s, to_local, to_world, project_static_points
from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251 - offline repository tooling
from tools.lane_type.audit_pixel_alignment import integrate_pose, integrate_road_pose  # noqa: TID251


def test_se2_and_curved_boundary_keep_the_same_world_location():
  angles = np.linspace(0, .7, 100)
  curve = np.column_stack((20*np.sin(angles), 20*(1-np.cos(angles))))
  pose = np.array([5., 1., .22])
  local = to_local(curve, pose)
  np.testing.assert_allclose(to_world(local, pose), curve, atol=1e-12)
  s0, _ = nearest_s(curve[10:80], curve)
  s1, error = nearest_s(to_world(local[10:80], pose), curve)
  np.testing.assert_allclose(s0, s1, atol=1e-12)
  assert error.max() < 1e-12


def test_real_paint_corrects_model_bias_and_source_pixels_are_preserved():
  rgb, line, k, rpy, target, nuisance, leads = synthetic_frame(np.zeros(3))
  result = PaintTracker('left').update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  assert abs(result['tracks'][0]['offset']+.18) < .03
  assert result['target_mask'].sum() > 1000
  assert not (result['target_mask'] & ~target).any()
  assert not (result['target_mask'] & nuisance).any()
  assert result['type'] == 'unknown'  # a short solid cannot establish a type


def test_stationary_views_do_not_multiply_distance_or_create_gaps():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3), dashed=True)
  tracker = PaintTracker('left')
  results = [tracker.update(rgb, line, 1., rpy, k, i*200_000_000, pose=np.zeros(3), pose_error_m=.02) for i in range(6)]
  assert len({r['distinct_m'] for r in results}) == 1
  assert all(r['gap_m'] == 0 and r['type'] == 'unknown' for r in results)


def test_double_boundary_keeps_the_ego_adjacent_dashed_stripe_separate():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3), dashed=True, double=True)
  tracker = PaintTracker('left')
  result = tracker.update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  assert len(result['tracks']) == 2
  assert result['tracks'][0]['offset'] > result['tracks'][1]['offset']
  assert any(r.get('gap_candidate') for r in result['profile'])
  assert result['type'] == 'unknown'


def test_desaturated_yellow_is_not_counted_as_white_and_needs_two_views():
  rgb, line, k, rpy, target, _, _ = synthetic_frame(np.zeros(3))
  rgb[target] = (110, 103, 87)
  tracker = PaintTracker('left')
  first = tracker.update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  second = tracker.update(rgb, line, 1., rpy, k, 200_000_000, pose=np.zeros(3), pose_error_m=.02)
  assert first['color'] == 'unknown' and second['color'] == 'yellow'
  assert all(p['color'] == 'yellow' for p in second['tracks'])


def test_tiny_bright_fragment_inside_a_gap_remains_unknown():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3), dashed=True)
  p = k @ view_frame_from_device_frame @ rot_from_euler(rpy) @ np.array([13., -1.6, 1.3])
  x, y = np.rint(p[:2]/p[2]).astype(int)
  cv2.rectangle(rgb, (x-5, y), (x+5, y+1), (235, 195, 35), -1)
  result = PaintTracker('left').update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  assert not result['target_mask'][y:y+2, x-5:x+6].any()
  assert all(r['ink'] is None for r in result['profile'] if y <= r['y'] <= y+1)


@pytest.mark.parametrize('reason', ['lane_change', 'tracking_loss', 'motion_uncertainty', 'clock'])
def test_boundary_and_motion_discontinuities_reset_history(reason):
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3))
  tracker = PaintTracker('left')
  tracker.update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  generation = tracker.generation
  result = tracker.update(rgb, line, .1 if reason == 'tracking_loss' else 1., rpy, k,
    0 if reason == 'clock' else 200_000_000, pose=np.zeros(3),
    pose_error_m=.5 if reason == 'motion_uncertainty' else .02, lane_change=reason == 'lane_change')
  assert tracker.generation > generation
  assert result['type'] == 'unknown'


def test_unknown_intervals_are_not_gaps_and_do_not_split_one_paint_into_strokes():
  tracker = PaintTracker('left')
  for key in range(80):
    # Three 4m paint intervals; four UNKNOWN bins at the true road gap edges.
    ink = 1 if key % 32 < 16 else 0
    if key in (16, 31, 48, 63):
      ink = None
    tracker.bins[key] = {'ink': ink, 'stamp': 0, 'near': False, 'color': None}
  assert tracker._classification(0)['type'] == 'dashed'
  for value in tracker.bins.values():
    if value['ink'] == 0:
      value['ink'] = None
  assert tracker._classification(0)['type'] == 'unknown'
  assert tracker._classification(0)['paint_runs'] == 1


def test_old_evidence_expires_in_half_a_second_without_being_refreshed_by_unknown():
  tracker = PaintTracker('left')
  tracker.bins = {i: {'ink': 1, 'color': 'yellow', 'near': True, 'stamp': 0} for i in range(100)}
  assert tracker._classification(0)['type'] == 'solid'
  assert tracker._classification(500_000_001)['type'] == 'unknown'
  assert tracker._classification(500_000_001)['color'] == 'unknown'


def test_recorded_sample_integration_has_correct_turn_translation_and_camera_time():
  times = np.arange(101, dtype=np.int64)*10_000_000+987_000_000_000
  query = np.array([times[0], times[0]+555_000_000, times[-1]])
  poses = integrate_pose(times, np.full(101, 10.), np.full(101, .2), query)
  t = (query-query[0])/1e9
  np.testing.assert_allclose(poses[:, 0], 50*np.sin(.2*t), atol=2e-6)
  np.testing.assert_allclose(poses[:, 1], 50*(1-np.cos(.2*t)), atol=2e-6)
  np.testing.assert_allclose(poses[:, 2], .2*t, atol=1e-12)
  # A stationary road endpoint gets closer and moves left during a right turn.
  assert to_local(np.array([[15., 0.]]), poses[-1])[0, 1] < 0


def test_blue_shadow_white_paint_keeps_a_trace_without_selecting_asphalt_texture():
  rgb, line, k, rpy, target, _, _ = synthetic_frame(np.zeros(3), white=True)
  rgb[~target] = (rgb[~target]*.4).astype(np.uint8)
  rgb[target] = (55, 75, 110)  # source-like blue cast; HSV saturation >75
  tracker = PaintTracker('left')
  result = tracker.update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  visible = np.zeros(target.shape, bool)
  for row in result['profile']:
    if row['geometry_visible']:
      visible[row['y']] = True
  assert (target & visible).sum() > 1000
  assert (result['target_mask'] & target).sum()/(target & visible).sum() > .95
  assert not (result['target_mask'] & ~target).any()
  assert result['tracks'][0]['color'] == 'white'


@pytest.mark.parametrize('white', [True, False])
def test_translated_dashes_require_repeated_physical_gaps_independent_of_paint_brightness(white):
  tracker = PaintTracker('left')
  for i in range(16):
    pose = np.array([float(i), 0., 0.])
    rgb, line, k, rpy, _, _, _ = synthetic_frame(pose, dashed=True, white=white)
    result = tracker.update(rgb, line, 1., rpy, k, i*200_000_000, pose=pose, pose_error_m=.02)
    for row in result['profile']:
      if row['ink'] == 0:
        assert (row['local_xy'][0]+i) % 8 >= 3
  assert result['type'] == 'dashed'
  assert result['verified_gap_runs'] >= 2
  assert result['distinct_m'] >= 18


def test_twenty_two_metre_solid_requirement_does_not_invent_evidence_when_stopped():
  tracker = PaintTracker('left')
  tracker.bins = {i: {'ink': 1, 'color': 'white', 'near': True, 'stamp': 0} for i in range(62)}
  assert tracker._classification(0)['distinct_m'] == 15.5
  assert tracker._classification(0)['type'] == 'unknown'


def test_static_endpoint_height_survives_pitch_and_a_noisy_new_road_prediction():
  times = np.array([0, 50_000_000], np.int64)
  poses = integrate_road_pose(times, [10., 10.], np.array([[0., .1, 0.], [0., .1, 0.]]), times)
  endpoint = np.array([[12., -1.6, 1.3]])
  k = np.array([[1141.5, 0, 672], [0, 1141.5, 380], [0, 0, 1.]])
  matrix = k @ view_frame_from_device_frame
  pixel, local = project_static_points(endpoint, poses[1], matrix)
  recovered_world = local @ poses[1, :3, :3].T+poses[1, :3, 3]
  np.testing.assert_allclose(recovered_world, endpoint, atol=1e-12)
  expected = local @ matrix.T
  np.testing.assert_allclose(pixel, expected[:, :2]/expected[:, 2, None])
  assert abs(local[0, 2]-1.3) > .05  # resetting z to a new flat model would lose this


def test_invalid_attitude_cannot_supply_temporal_evidence():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3))
  transform = np.eye(4)
  transform[0, 0] = 2  # scaling is not a rigid camera pose
  result = PaintTracker('left').update(rgb, line, 1., rpy, k, 0,
    pose=np.zeros(3), pose_error_m=.02, road_pose=transform)
  assert result['type'] == 'unknown' and not result['pose_qualified']


def test_attitude_representation_change_resets_and_results_are_serializable():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3))
  tracker = PaintTracker('left')
  tracker.update(rgb, line, 1., rpy, k, 0, pose=np.zeros(3), pose_error_m=.02)
  generation = tracker.generation
  result = tracker.update(rgb, line, 1., rpy, k, 200_000_000,
    pose=np.zeros(3), pose_error_m=.02, road_pose=np.eye(4))
  assert tracker.generation > generation
  assert result['type'] == 'unknown'
  json.dumps({k: v for k, v in result.items() if k != 'target_mask'})
