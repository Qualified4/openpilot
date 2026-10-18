import numpy as np
import cv2

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import candidates
from openpilot.selfdrive.carrot.lane_type.hybrid_tracker import HybridPaintTracker, RegisteredPaintTracker
from tools.lane_type.validate_hybrid_tracking import FailureEvents  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251


def test_geometry_only_and_cropped_source_colour_candidates_equal_full_source():
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3))
  matrix = k @ view_frame_from_device_frame @ rot_from_euler(rpy)
  full = collect_pixels(rgb, line, 1., rpy, k)
  geometry = collect_pixels(rgb, line, 1., rpy, k, include_pixels=False)
  assert full['rows'] == geometry['rows']
  assert len(geometry['rgb']) == 0
  visible = [r for r in geometry['rows'] if r['geometry_visible']]
  bounds = (min(r['left_x'] for r in visible), min(r['y'] for r in visible),
            max(r['right_x'] for r in visible)+1, max(r['y'] for r in visible)+1)
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  _, rows1, found1 = candidates(rgb, full, line, matrix)
  _, rows2, found2 = candidates(rgb, geometry, line, matrix, gray=gray, bounds=bounds)
  assert rows1 == rows2 and found1 == found2
  _, rows3, found3 = candidates(rgb, geometry, line, matrix, gray=gray, bounds=bounds, batched=True)
  assert rows2 == rows3 and len(found2) == len(found3)
  for a, b in zip(found2, found3, strict=True):
    for key in a:
      if isinstance(a[key], float):
        assert np.isclose(a[key], b[key], rtol=0, atol=1e-10)
      else:
        assert a[key] == b[key]


def test_intermediate_tracking_never_runs_precise_path_search(monkeypatch):
  tracker = HybridPaintTracker('left')
  rgb, line, k, rpy, _, _, _ = synthetic_frame(np.zeros(3))
  result = tracker.update(rgb, line, 1., rpy, k, 0, precise=True, pose=np.zeros(3), pose_error_m=.02)
  assert result['tracks']
  def unexpected(*args, **kwargs):
    raise AssertionError('precise robust fit ran between keyframes')
  monkeypatch.setattr('openpilot.selfdrive.carrot.lane_type.pixel_tracker.fit_path', unexpected)
  pose = np.array([.15, 0., 0.])
  rgb, line, k, rpy, target, _, _ = synthetic_frame(pose)
  result = tracker.update(rgb, line, 1., rpy, k, 50_000_000, precise=False, pose=pose, pose_error_m=.02)
  assert result['selection_mode'] == 'narrow_tracking' and result['tracks']
  assert result['target_mask'].sum() > 1000
  assert not (result['target_mask'] & ~target).any()


def test_unidentifiable_uniform_road_does_not_admit_spatial_evidence():
  tracker = RegisteredPaintTracker('left')
  rgb, line, k, rpy, paint, _, _ = synthetic_frame(np.zeros(3))
  rgb[~paint] = 80
  for i in range(4):
    result = tracker.update(rgb, line, 1., rpy, k, i*50_000_000, pose=np.zeros(3), pose_error_m=.02)
    assert result['type'] == 'unknown' and not result.get('registration_qualified', False)
    assert not tracker.bins


def test_failure_frames_and_independent_episodes_are_different():
  state = FailureEvents()
  for i, failed in enumerate([True, True, False, True]):
    state.update(i, i*50_000_000, i, {'generation': i+int(failed), 'tracks': [] if failed else [{}],
      'registration_qualified': not failed, 'reason': 'unknown' if failed else 'observed'})
  assert state.reset_calls == 3
  assert len(state.events) == 2
  assert [e['failed_frames'] for e in state.events] == [2, 1]
