"""Execute modeld's actual calibration/read/publish statements without starting GPU inference.

Messaging, schemas and warp calculations are real; only camera/Params inputs are supplied.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix


def test_trim_read_apply_and_real_schema_publish():
  path = Path(__file__).parents[1] / 'modeld.py'
  tree = ast.parse(path.read_text())
  main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
  loop = [n for n in main.body if isinstance(n, ast.While)][-1]
  poll = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'frame % 100 == 0')
  calibration = next(n for n in loop.body if isinstance(n, ast.If) and 'sm.updated' in ast.unparse(n.test))

  def run(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)

  def reads(nodes):
    return [n for n in nodes if any(name in ast.unparse(n) for name in
            ('camera_yaw_trim', 'applied_yaw_trim', 'applied_calibration_yaw'))]

  device, sensor = next(iter(DEVICE_CAMERAS))
  class SubMaster(dict):
    updated = {'liveCalibration': True}
    seen = {'roadCameraState': True, 'deviceState': True}

  calib = log.LiveCalibrationData.new_message(rpyCalib=[0., -.006, .00242], calStatus='calibrated')
  sm = SubMaster(liveCalibration=calib, deviceState=SimpleNamespace(deviceType=device),
                roadCameraState=SimpleNamespace(sensor=sensor))
  params, cloudlog = Mock(), Mock()
  params.get_int.return_value = 50
  env = dict(np=np, log=log, sm=sm, params=params, cloudlog=cloudlog,
             DEVICE_CAMERAS=DEVICE_CAMERAS, get_warp_matrix=get_warp_matrix,
             main_wide_camera=False, use_extra_client=True, live_calib_seen=False)
  run(reads(main.body[:main.body.index(loop)]), env)
  publish = [n for n in ast.walk(loop) if isinstance(n, ast.Assign) and
             ast.unparse(n.targets[0]).startswith('modelv2_send.modelV2.meta.') and
             ast.unparse(n.targets[0]).split('.')[-1] in ('cameraYawTrim', 'calibrationYaw', 'calibrationYawValid')]
  assert len(publish) == 3
  # Real PubMaster and serialization: no fake schema can hide missing fields.
  pm = messaging.PubMaster(['modelV2'])
  def snapshot():
    env['modelv2_send'] = messaging.new_message('modelV2')
    run(publish, env)
    data = env['modelv2_send'].to_bytes()
    pm.send('modelV2', data)
    with log.Event.from_bytes(data) as msg:
      return msg.modelV2.meta.to_dict()

  assert snapshot()['calibrationYawValid'] is False
  warps = []
  for trim in (50, 55, 50, -50, 0):
    before = snapshot()
    params.get_int.return_value = trim
    run(reads(poll.body), env)
    assert snapshot() == before  # a Params read is not an applied warp
    run([calibration], env)
    result = snapshot()
    assert result['cameraYawTrim'] == trim
    assert result['calibrationYawValid']
    assert result['calibrationYaw'] == pytest.approx(.00242 - np.radians(trim * .01), abs=1e-9)
    warps.append(env['model_transform_main'].copy())
  np.testing.assert_array_equal(warps[0], warps[2])
  assert not np.array_equal(warps[0], warps[1])
  assert cloudlog.info.call_count == 5  # startup plus changes, not the repeated 50
  params.get_int.return_value = 55
  run(reads(poll.body), env)
  calib.calStatus = 'uncalibrated'
  run([calibration], env)
  assert snapshot()['cameraYawTrim'] == 0
  assert snapshot()['calibrationYaw'] == pytest.approx(.00242)
