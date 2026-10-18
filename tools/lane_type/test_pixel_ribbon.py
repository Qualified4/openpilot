import json

import cv2
import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels, source_box
from tools.lane_type.audit_pixels import synthetic_audit  # noqa: TID251 - offline repository tooling


def geometry():
  return {'x': [0, 5, 10, 20, 30, 50], 'y': [-1.6]*6, 'z': [1.3]*6}, np.array([[550., 0, 320], [0, 550, 180], [0, 0, 1.]])


def test_every_included_pixel_and_its_source_coordinate_survive():
  line, k = geometry()
  rgb = np.random.default_rng(7).integers(0, 256, (360, 640, 3), dtype=np.uint8)
  r = collect_pixels(rgb, line, .99, [0, 0, 0], k)
  xy = r['xy']
  assert len(xy) > 1000
  assert np.array_equal(r['rgb'], rgb[xy[:, 1], xy[:, 0]])
  assert len(np.unique(xy, axis=0)) == len(xy)
  for row, a, b in zip(r['rows'], r['row_offsets'][:-1], r['row_offsets'][1:], strict=True):
    if row['geometry_visible']:
      assert b-a == row['right_x']-row['left_x']+1
      assert np.array_equal(xy[a:b, 0], np.arange(row['left_x'], row['right_x']+1))
    else:
      assert a == b
  box = source_box(rgb, r)
  x, y = box['origin_xy']
  h, w = box['image'].shape[:2]
  assert np.array_equal(box['image'], rgb[y:y+h, x:x+w])


def test_codec_free_solid_can_lose_paint_in_v1_but_not_in_lossless_ribbon(tmp_path):
  results = synthetic_audit(tmp_path)
  thin = results[0]
  assert thin['heads']['type']['v1_rows_with_no_remaining_paint'] == 16
  assert thin['heads']['color']['v1_rows_with_no_remaining_paint'] == 12
  assert all(r['ribbon_paint_pixels_lost_in_visible_rows'] == 0 for r in results)
  # Controls prevent claiming that every normal-width solid is fragmented.
  assert results[-1]['heads']['type']['v1_rows_with_no_remaining_paint'] == 0


def test_bright_crosswalk_does_not_become_a_verified_paint_gap_or_class():
  line, k = geometry()
  rgb = np.full((360, 640, 3), 80, np.uint8)
  cv2.rectangle(rgb, (0, 240), (639, 260), (255, 255, 255), -1)
  r = collect_pixels(rgb, line, .99, [0, 0, 0], k)
  assert not r['gap_evidence'] and r['classification'] == 'unknown'
  assert all(row['paint_state'] == 'unknown' for row in r['rows'])
  assert not any(row['visibility_semantically_verified'] for row in r['rows'])


def test_low_confidence_returns_no_padding_or_invented_pixels():
  line, k = geometry()
  r = collect_pixels(np.zeros((360, 640, 3), np.uint8), line, .1, [0, 0, 0], k)
  assert not len(r['rgb']) and not r['padding_used']


def test_training_hold_is_checked_before_cuda_or_output_creation(tmp_path):
  from tools.lane_type.train_compare import train  # noqa: TID251 - offline repository tooling
  root = tmp_path/'held'
  root.mkdir()
  (root/'ROI_REVIEW_HOLD.json').write_text(json.dumps({'status': 'HOLD'}))
  manifest = root/'nested'/'manifest.json'
  with pytest.raises(RuntimeError, match='ROI pixel audit hold'):
    train(manifest, tmp_path, tmp_path/'new-output', 'pplcnet_025')
  assert not (tmp_path/'new-output').exists()
