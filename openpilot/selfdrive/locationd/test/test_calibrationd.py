from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.cereal import car, log
from openpilot.selfdrive.locationd import calibrationd


@pytest.mark.parametrize('trims, statuses, writes', [
  ([50, 50, 55, 50], [True] * 4, 1),  # nonzero at startup and nonzero toggles
  ([0, 50, 50, 0, 50], [True] * 5, 2),  # resume then freeze again
  ([50, 50, 50], [False, True, True], 1),  # wait for completed calibration
  ([0, 0], [True, True], 0),
])
def test_freeze_saves_current_calibration_once(monkeypatch, trims, statuses, writes):
  params = Mock()
  params.get.return_value = car.CarParams.new_message().to_bytes()
  params.get_float.side_effect = trims
  monkeypatch.setattr(calibrationd, 'Params', lambda: params)
  calibrator = calibrationd.Calibrator()
  calibrator.reset(np.array([0., -.006, .001179]), valid_blocks=11)
  calibrator.update_status()
  calibrator.handle_cam_odom = Mock(return_value=None)
  assert calibrator.block_idx == 0  # not the periodic save block
  monkeypatch.setattr(calibrationd, 'Calibrator', lambda **kwargs: calibrator)
  monkeypatch.setattr(calibrationd, 'config_realtime_process', lambda *args: None)
  monkeypatch.setattr(calibrationd.messaging, 'PubMaster', lambda *args: Mock())

  class Finished(Exception):
    pass

  class SubMaster(dict):
    frame = -1
    updated = {'cameraOdometry': True}

    def update(self, timeout):
      self.frame += 1
      if self.frame == len(trims):
        raise Finished
      calibrator.cal_status = (log.LiveCalibrationData.Status.calibrated if statuses[self.frame]
                               else log.LiveCalibrationData.Status.uncalibrated)

    def all_checks(self):
      return True

  sm = SubMaster(carState=SimpleNamespace(vEgo=0.), cameraOdometry=SimpleNamespace(
    trans=[], rot=[], wideFromDeviceEuler=[], transStd=[], roadTransformTrans=[], roadTransformTransStd=[]))
  monkeypatch.setattr(calibrationd.messaging, 'SubMaster', lambda *args, **kwargs: sm)
  with pytest.raises(Finished):
    calibrationd.main()
  assert params.put_nonblocking.call_count == writes
  assert calibrator.handle_cam_odom.call_count == sum(not (trim and done) for trim, done in zip(trims, statuses))
  for call in params.put_nonblocking.call_args_list:
    key, data = call.args
    assert key == 'CalibrationParams'
    with log.Event.from_bytes(data) as msg:
      assert msg.valid and msg.liveCalibration.validBlocks == 11
      assert msg.liveCalibration.calStatus == log.LiveCalibrationData.Status.calibrated
      assert msg.liveCalibration.rpyCalib[2] == pytest.approx(.001179)
