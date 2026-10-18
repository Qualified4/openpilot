import ast
import builtins
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type.daemon import Classifier, requested
from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard, visibility
from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip
from openpilot.selfdrive.carrot.lane_type.result import classify, unknown
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, nv12_strip, input_blob
from tools.lane_type.dataset import validate_rows  # noqa: TID251 - repository-level tooling
from tools.lane_type.evaluate import metrics  # noqa: TID251

ROOT = Path(__file__).resolve().parents[2]
GOOD = {'sufficient': True, 'observedLengthM': 40.0, 'occludedFraction': 0.0}
BAD = {'sufficient': False, 'observedLengthM': 8.0, 'occludedFraction': .8}
BOUNDARY = (-1.7, -1.7)


def confirmed(kind='dashed', color='white'):
  return {'type': kind, 'color': color, 'typeConfidence': .95, 'colorConfidence': .95, 'reason': 'classified'}


def test_default_off_manager_predicate_and_manual_entrypoint_do_not_load_or_import_engine(monkeypatch):
  from openpilot.selfdrive.carrot.lane_type import daemon
  params = N(get_bool=lambda key:False)
  monkeypatch.setitem(sys.modules, 'openpilot.common.params', N(Params=lambda:params))
  loaded = []
  def constructor(*args, **kwargs):
    loaded.append(True)
    raise AssertionError('OFF loaded classifier')
  monkeypatch.setattr(daemon, 'Classifier', constructor)
  original_import = builtins.__import__
  def guarded_import(name, *args, **kwargs):
    if name in ('cv2', 'numpy', 'msgq.visionipc', 'openpilot.cereal.messaging'):
      raise AssertionError('OFF imported inference or IPC: '+name)
    return original_import(name, *args, **kwargs)
  with monkeypatch.context() as guarded:
    guarded.setattr(builtins, '__import__', guarded_import)
    daemon.main()
  assert not loaded
  assert not requested(True, params)
  assert not requested(False, N(get_bool=lambda key:True))
  assert requested(True, N(get_bool=lambda key:True))
  catalog = json.loads((ROOT/'openpilot/selfdrive/carrot_settings.json').read_text(encoding='utf-8'))
  item = next(p for p in catalog['params'] if p['name'] == 'CcncLaneTypeDetection')
  # The option became the 3-way display mode (0 off, 1 Scanner, 2 StripNet); this legacy preview daemon is no
  # longer registered with the manager (marking_daemon is), but it still stays inert with the option off.
  assert (item['min'], item['max'], item['default']) == (0, 2, 0)


def test_lane_marking_reaches_only_the_ccnc_display_path():
  """Display-only integration contract: no control/lane-change code reads the recogniser; card imports only
  the light packet parser; on the Hyundai side only the CCNC sender and its mapping module do."""
  for path in (ROOT/'openpilot/selfdrive/controls').rglob('*.py'):
    text = path.read_text(encoding='utf-8', errors='replace')
    assert 'CcncLaneTypeDetection' not in text and 'carrot.lane_type' not in text and 'ccnc_lane_marking' not in text
  for path in (ROOT/'openpilot/selfdrive/car').glob('*.py'):
    text = path.read_text(encoding='utf-8', errors='replace')
    allowed = 'from openpilot.selfdrive.carrot.lane_type import marking as lane_marking' if path.name == 'card.py' else None
    for line in text.splitlines():
      if 'carrot.lane_type' in line:
        assert line.strip() == allowed, (path.name, line)
  for path in (ROOT/'opendbc_repo/opendbc/car/hyundai').glob('*.py'):
    text = path.read_text(encoding='utf-8', errors='replace')
    assert 'carrot.lane_type' not in text
    if path.name not in ('hyundaicanfd.py', 'ccnc_lane_marking.py'):
      assert 'CcncLaneTypeDetection' not in text and 'ccnc_lane_marking' not in text, path.name
  for module in ('daemon.py', 'marking_daemon.py', 'strip_model.py', 'profile_scanner.py'):
    source = ast.parse((ROOT/'openpilot/selfdrive/carrot/lane_type'/module).read_text(encoding='utf-8'))
    assert all('can' not in alias.name.split('.')[-1] for node in ast.walk(source) if isinstance(node, ast.Import) for alias in node.names)


@pytest.mark.parametrize('values', [[np.nan, .5], [.3, .3], [-.1, 1.1], [.5, .3, .2]])
def test_malformed_probabilities_fail(values):
  with pytest.raises(ValueError):
    classify(values, [.5, .5])


def test_type_color_and_sides_are_independent():
  r = classify([.55, .45], [.99, .01])
  assert r['type'] == 'unknown' and r['color'] == 'white'
  guard = ObservationGuard()
  left = guard.update('left', confirmed(), GOOD, BOUNDARY, now_ns=1, camera_ns=1)
  right = guard.update('right', r, GOOD, (1.7, 1.7), now_ns=1, camera_ns=1)
  assert left['type'] == 'dashed' and right['type'] == 'unknown' and right['color'] == 'white'
  assert unknown('not_detected')['type'] == 'not_detected'


def test_insufficient_gap_cannot_create_a_solid_and_stationary_repeats_never_accumulate_length():
  guard = ObservationGuard()
  for index in range(100):
    r = guard.update('left', confirmed('solid'), BAD, BOUNDARY, now_ns=index*200_000_000, camera_ns=index*200_000_000)
    assert r['type'] == 'unknown'
    assert r['observedLengthM'] == 8


def test_hold_has_absolute_deadline_and_cannot_be_refreshed_by_occlusion():
  guard = ObservationGuard()
  guard.update('left', confirmed(), GOOD, BOUNDARY, now_ns=1, camera_ns=1)
  assert guard.update('left', confirmed('solid'), BAD, BOUNDARY, now_ns=400_000_001, camera_ns=400_000_001)['type'] == 'dashed'
  assert guard.update('left', confirmed('solid'), BAD, BOUNDARY, now_ns=500_000_002, camera_ns=500_000_002)['type'] == 'unknown'


@pytest.mark.parametrize('kwargs,boundary', [({'lane_change':True}, BOUNDARY), ({'tracked':False}, BOUNDARY), ({}, (1.8, 1.8)), ({}, None)])
def test_lane_change_renumber_and_tracking_loss_clear_memory(kwargs, boundary):
  guard = ObservationGuard()
  guard.update('left', confirmed(), GOOD, BOUNDARY, now_ns=1, camera_ns=1)
  r = guard.update('left', unknown(), BAD, boundary, now_ns=100_000_001, camera_ns=100_000_001, **kwargs)
  assert r['type'] == 'unknown' and 'left' not in guard.previous


@pytest.mark.parametrize('now,camera', [(600_000_001, 1), (1, 2)])
def test_stale_or_future_camera_never_creates_result(now, camera):
  assert ObservationGuard().update('left', confirmed(), GOOD, BOUNDARY, now_ns=now, camera_ns=camera)['type'] == 'unknown'


def test_preview_copies_baseline_gates_option_mode_age_and_expiry():
  baseline = {'LANELINE_LEFT':2, 'LANELINE_RIGHT':6, 'COUNTER':99}
  policy = DisplayPolicy()
  for stamp in (100, 200_000_100):
    left = dict(confirmed(), typeMonoTimeNanos=stamp, colorMonoTimeNanos=stamp,
                dashedPatternObserved=True, dashedPatternMonoTimeNanos=stamp)
    right = dict(confirmed('solid'), typeMonoTimeNanos=stamp, colorMonoTimeNanos=stamp)
    packet = {'type': 'ccncLaneType', 'version': 1, 'cameraMonoTimeNanos': stamp, 'left': left, 'right': right}
    result = policy.update(baseline, packet, enabled=True, model_lanes=1, now_ns=stamp)['values']
  assert result == {'LANELINE_LEFT':8, 'LANELINE_RIGHT':2, 'COUNTER':99}
  assert baseline['LANELINE_LEFT'] == 2
  assert policy.update(baseline, packet, enabled=False, model_lanes=1, now_ns=100)['values'] == baseline
  assert policy.update(baseline, packet, enabled=True, model_lanes=0, now_ns=100)['values'] == baseline
  assert policy.update(baseline, packet, enabled=True, model_lanes=2, now_ns=900_000_101)['values']['LANELINE_LEFT'] == 1


def test_long_roi_projection_and_reject_short_low_probability_or_offscreen():
  line = N(x=[0, 10, 30, 60], y=[-1.7]*4, z=[1.2]*4)
  k = np.array([[1141.5, 0, 672], [0, 1141.5, 380], [0, 0, 1.]])
  maps = lane_maps(line, .9, [0, 0, 0], k, 1344, 760)
  assert maps[0].shape == (192, 64)
  assert maps[1][0, 32] < maps[1][-1, 32]
  assert lane_maps(line, .1, [0, 0, 0], k, 1344, 760) is None
  assert lane_maps(N(x=[0, 10, 20], y=[0]*3, z=[1.2]*3), .9, [0, 0, 0], k, 1344, 760) is None
  assert lane_maps(N(x=line.x, y=[-20]*4, z=line.z), .9, [0, 0, 0], k, 1344, 760) is None


def test_visibility_blocks_shortened_strip_behind_lead():
  line = N(x=[0, 10, 30, 60], y=[-1.7]*4, z=[1.2]*4)
  k = np.array([[1141.5, 0, 672], [0, 1141.5, 380], [0, 0, 1.]])
  maps = lane_maps(line, .9, [0, 0, 0], k, 1344, 760)
  clear, observation = visibility(maps, N(leadsV3=[N(prob=.9, x=[10.], y=[0.])]), line, [0, 0, 0], k)
  assert not observation['sufficient'] and observation['observedLengthM'] < 32
  assert not clear.all()


def test_nv12_reads_padded_allocation_without_modifying_it():
  raw = bytearray([80]*64+[128]*32)
  original = bytes(raw)
  maps = np.meshgrid(np.linspace(0, 5, 64, dtype=np.float32), np.linspace(0, 5, 192, dtype=np.float32))
  image = nv12_strip(raw, 6, 6, 8, 64, maps)
  assert image.shape == (192, 64, 3) and bytes(raw) == original
  assert np.max(image)-np.min(image) <= 1
  with pytest.raises(ValueError):
    nv12_strip(raw[:20], 6, 6, 8, 64, maps)
  assert input_blob([image, image]).shape == (2, 3, 192, 64)


def test_opencv_synthetic_continuity_and_yellow_masks():
  solid = np.full((192, 64, 3), 50, np.uint8)
  solid[:, 28:36] = [235, 235, 235]
  assert classify_strip(solid)['type'] == 'solid'
  dashed = solid.copy()
  for start in range(0, 192, 40):
    dashed[start:start+20] = 50
  assert classify_strip(dashed)['type'] == 'dashed'
  assert classify_strip(dashed)['repeatingPatternObserved']
  json.dumps(classify_strip(dashed), allow_nan=False)
  # Shorter physical crops cannot inherit forty-metre gap lengths.
  short = classify_strip(dashed, near=3, far=15)
  assert short['type'] == 'unknown' and not short['repeatingPatternObserved']
  yellow = solid.copy()
  yellow[:, 28:36] = [240, 210, 30]
  assert classify_strip(yellow)['color'] == 'yellow'
  assert classify_strip(np.full_like(solid, 50))['type'] == 'unknown'


def test_dataset_rejects_unreviewed_truth_route_and_pixel_leakage(tmp_path):
  base = {'id': 'a', 'route': 'r1', 'split': 'train', 'image_sha256': 'a', 'verified': True, 'type': 'solid', 'color': 'white',
              'reviewer': 'human', 'review_note': 'checked', 'label_source': 'visual_review'}
  validate_rows([base])
  with pytest.raises(ValueError, match='route leakage'):
    validate_rows([base, dict(base, id='b', split='test', image_sha256='b')])
  with pytest.raises(ValueError, match='duplicate ROI'):
    validate_rows([base, dict(base, id='b', route='r2', split='test')])
  with pytest.raises(ValueError, match='review provenance'):
    validate_rows([dict(base, label_source='automatic')])
  with pytest.raises(ValueError, match='explicit boolean'):
    validate_rows([dict(base, verified='true')])


def test_abstentions_are_false_negatives_and_empty_scenarios_are_not_zero_accuracy():
  r = metrics([{'type':'dashed'}, {'type':'solid'}], [{'type':'unknown'}, {'type':'solid'}], 'type', ('dashed', 'solid'))
  assert r['accuracy'] == .5 and r['classes']['dashed']['fn'] == 1
  assert r['confirmation_rate'] == .5 and r['confirmed_accuracy'] == 1
  assert metrics([{'type': 'solid'}], [{'type': 'unknown'}], 'type', ('dashed', 'solid'))['confirmed_accuracy'] is None
  assert metrics([], [], 'type', ('dashed', 'solid'))['accuracy'] is None


def test_paired_variants_cannot_cross_splits():
  base = {'id': 'f', 'scene_id': 'physical-scene', 'route': 'a', 'split': 'train', 'image_sha256': 'f', 'verified': False}
  validate_rows([base, dict(base, id='q', image_sha256='q')])
  with pytest.raises(ValueError, match='paired scene leakage'):
    validate_rows([base, dict(base, id='q', image_sha256='q', route='b', split='test')])


def test_three_metre_crop_cannot_turn_out_of_frame_pixels_into_a_gap():
  from tools.lane_type.roi_quality import projection_maps, geometry_quality  # noqa: TID251 - offline test
  line = {'x': [0., 10., 60.], 'y': [-1.8]*3, 'z': [1.5]*3}
  k = np.array([[1141.5, 0., 672.], [0., 1141.5, 380.], [0., 0., 1.]])
  maps, distances = projection_maps(line, [0., 0., 0.], k, 3., 35.)
  q = geometry_quality(maps, distances, 1344, 760)
  assert not q['complete_requested_roi'] and q['first_full_width_visible_m'] > 4
  maps, distances = projection_maps(line, [0., 0., 0.], k, 6., 46.)
  assert geometry_quality(maps, distances, 1344, 760)['complete_requested_roi']


def test_diversity_pool_does_not_expand_identical_views():
  from tools.lane_type.retrain_data import diverse_indices  # noqa: TID251 - offline test
  assert len(diverse_indices([[.1, .1]]*100, 32)) == 1


def test_roi_review_hold_blocks_training_before_dataset_loading(tmp_path):
  from tools.lane_type.train import reviewed_data  # noqa: TID251 - PC training-only test
  (tmp_path/'ROI_REVIEW_HOLD.json').write_text('{}')
  with pytest.raises(ValueError, match='ROI review hold'):
    reviewed_data(tmp_path/'manifest.json')


def test_artifact_contract_and_real_inference(tmp_path):
  directory = ROOT/'openpilot/selfdrive/carrot/lane_type/assets'
  metadata = json.loads((directory/'manifest.json').read_text())
  assert not metadata['accuracyValidated'] and not metadata['deviceValidated'] and not metadata['canTransmitEnabled']
  assert hashlib.sha256((directory/'lane_type.onnx').read_bytes()).hexdigest() == metadata['model_sha256']
  classifier = Classifier(directory=directory)
  result = classifier.infer([np.full((192, 64, 3), 127, np.uint8)]*2)
  assert len(result) == 2 and all(0 <= r['typeConfidence'] <= 1 for r in result)
  (tmp_path/'manifest.json').write_text(json.dumps(metadata))
  (tmp_path/'lane_type.onnx').write_bytes(b'corrupt')
  with pytest.raises(ValueError, match='artifact contract'):
    Classifier(directory=tmp_path)
