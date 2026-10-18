"""5Hz target, independent camera consumer. Logs/bench preview only; no CAN."""
import hashlib
import json
import os
from pathlib import Path
import time

from openpilot.selfdrive.carrot.lane_type.result import classify, unknown
from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy

ASSETS = Path(__file__).with_name('assets')
INTERVAL = .2
MAX_MODEL_SKEW_NS = 150_000_000
MAX_INFERENCE_NS = 200_000_000


def requested(started, params, CP=None):
  return started and params.get_bool('CcncLaneTypeDetection')


class Classifier:
  def __init__(self, engine='onnx', directory=ASSETS):
    import cv2
    cv2.setNumThreads(1)
    self.engine = engine
    self.model_hash = None
    self.net = None
    if engine == 'onnx':
      import numpy as np
      directory = Path(directory)
      manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
      path = directory/'lane_type.onnx'
      model_bytes = path.read_bytes()
      self.model_hash = hashlib.sha256(model_bytes).hexdigest()
      if (manifest.get('model_sha256') != self.model_hash or manifest.get('architecture') != 'PP-LCNet-0.25x' or
          manifest.get('input_shape') != [3, 192, 64] or manifest.get('type_classes') != ['dashed', 'solid'] or
          manifest.get('color_classes') != ['white', 'yellow']):
        raise ValueError('lane classifier artifact contract mismatch')
      self.net = cv2.dnn.readNetFromONNX(np.frombuffer(model_bytes, dtype=np.uint8))
      self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    elif engine != 'opencv':
      raise ValueError('engine must be onnx or opencv')

  def infer(self, strips, *, distance_range=None):
    if self.engine == 'opencv':
      from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip
      return [classify_strip(strip, *distance_range) if distance_range is not None else classify_strip(strip) for strip in strips]
    from openpilot.selfdrive.carrot.lane_type.roi import input_blob
    self.net.setInput(input_blob(strips))
    types, colors = self.net.forward(['line_type', 'color'])
    if len(types) != len(strips) or len(colors) != len(strips):
      raise ValueError('lane classifier output batch mismatch')
    return [classify(t, c) for t, c in zip(types, colors, strict=True)]


def main():
  # OFF must not import OpenCV, load an artifact, connect IPC, or infer, even
  # when launched manually rather than by the manager predicate.
  from openpilot.common.params import Params
  params = Params()
  if not params.get_bool('CcncLaneTypeDetection'):
    return
  import cv2
  cv2.setNumThreads(1)
  import numpy as np
  from msgq.visionipc import VisionIpcClient, VisionStreamType
  import openpilot.cereal.messaging as messaging
  from openpilot.common.swaglog import cloudlog
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.selfdrive.carrot.lane_type.roi import lane_maps, nv12_strip
  from openpilot.selfdrive.carrot.lane_type.observation import ObservationGuard, visibility

  if hasattr(os, 'sched_setaffinity'):
    background = set(os.sched_getaffinity(0)) & {0, 1, 2, 3}
    if not background:
      cloudlog.warning('CCNC lane classifier: background CPUs unavailable')
      return
    os.sched_setaffinity(0, background)
  if hasattr(os, 'nice'):
    os.nice(10)
  services = ['modelV2', 'liveCalibration', 'roadCameraState', 'deviceState', 'carState']
  sm = messaging.SubMaster(services+['carControl'])
  pm = messaging.PubMaster(['customReservedRawData2'])
  client = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_ROAD, True)
  guards = ObservationGuard()
  display = DisplayPolicy()
  classifier = None
  last_frame = None
  next_run = 0.0
  engine = os.getenv('CCNC_LANE_TYPE_ENGINE', 'onnx')
  while params.get_bool('CcncLaneTypeDetection'):
    sm.update(0)
    if time.monotonic() < next_run:
      time.sleep(.01)
      continue
    next_run = time.monotonic()+INTERVAL
    if not sm.all_checks(services) or not sm['deviceState'].started:
      guards.reset()
      display.reset()
      continue
    if not client.is_connected() and not client.connect(False):
      guards.reset()
      display.reset()
      continue
    buffer = client.recv(timeout_ms=0)
    if buffer is None or client.frame_id == last_frame:
      continue
    frame_id = client.frame_id
    if last_frame is not None and frame_id <= last_frame:
      guards.reset()
      display.reset()
    last_frame = frame_id
    camera_ns = int(client.timestamp_eof)
    model, calibration = sm['modelV2'], sm['liveCalibration']
    packet = {'type': 'ccncLaneType', 'version': 1, 'frameId': frame_id, 'engine': engine,
                  'cameraMonoTimeNanos': camera_ns, 'modelMonoTimeNanos': int(model.timestampEof),
                  'left': unknown(), 'right': unknown(), 'canTransmitEnabled': False,
                  'confidenceKind': 'uncalibrated_softmax' if engine == 'onnx' else 'uncalibrated_heuristic_evidence'}
    started_ns = time.monotonic_ns()
    cpu_start = time.thread_time_ns()
    try:
      if (str(calibration.calStatus) != 'calibrated' or len(calibration.rpyCalib) != 3 or
          not 0 <= camera_ns-model.timestampEof <= MAX_MODEL_SKEW_NS or
          not 0 <= started_ns-camera_ns <= 200_000_000 or len(model.laneLines) < 3 or len(model.laneLineProbs) < 3):
        guards.reset()
        raise ValueError('stale camera/model or missing calibration')
      camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))].fcam
      k = camera.intrinsics.copy()
      k[0] *= buffer.width/camera.width
      k[1] *= buffer.height/camera.height
      strips, sides = [], []
      observations, boundaries = {}, {}
      cs = sm['carState']
      lane_change = bool(cs.leftBlinker or cs.rightBlinker or str(model.meta.laneChangeState) != 'off')
      for side, index in [('left', 1), ('right', 2)]:
        line = model.laneLines[index]
        try:
          maps = lane_maps(line, model.laneLineProbs[index], calibration.rpyCalib, k, buffer.width, buffer.height)
        except (ValueError, AttributeError):
          guards.previous.pop(side, None)
          packet[side] = unknown('invalid_lane_geometry')
          continue
        if maps is None:
          guards.previous.pop(side, None)
          packet[side] = unknown('not_detected' if model.laneLineProbs[index] < .5 else 'tracking_lost')
          continue
        clear, observation = visibility(maps, model, line, calibration.rpyCalib, k)
        boundary = tuple(float(np.interp(x, line.x, line.y)) for x in (10, 30))
        if not observation['sufficient'] or lane_change:
          packet[side] = guards.update(side, unknown('insufficient_observation'), observation, boundary,
                                      now_ns=time.monotonic_ns(), camera_ns=camera_ns, lane_change=lane_change)
          continue
        rgb = nv12_strip(buffer.data, buffer.width, buffer.height, buffer.stride, buffer.uv_offset, maps)
        rgb[~clear] = 0
        strips.append(rgb)
        sides.append(side)
        observations[side] = observation
        boundaries[side] = boundary
      if strips and classifier is None:
        classifier = Classifier(engine)
      infer_start = time.monotonic_ns()
      results = classifier.infer(strips) if strips else []
      infer_ns = time.monotonic_ns()-infer_start
      packet['inferenceMs'] = infer_ns/1e6 if strips else None
      packet['inferredSides'] = len(strips)
      packet['modelHash'] = classifier.model_hash if classifier else None
      for side, result in zip(sides, results, strict=True):
        if time.monotonic_ns()-started_ns > MAX_INFERENCE_NS:
          result = unknown('late_result')
          observations[side]['sufficient'] = False
        packet[side] = guards.update(side, result, observations[side], boundaries[side], now_ns=time.monotonic_ns(),
                                    camera_ns=camera_ns, lane_change=lane_change)
        packet[side]['raw'] = dict(result)
        if result['type'] == 'dashed' and observations[side]['occludedFraction'] == 0:
          from openpilot.selfdrive.carrot.lane_type.opencv_classifier import classify_strip
          evidence = classify_strip(strips[sides.index(side)])
          packet[side]['dashedPatternObserved'] = bool(evidence.get('repeatingPatternObserved'))
          packet[side]['dashedPatternMonoTimeNanos'] = camera_ns
    except (ValueError, RuntimeError, OSError, KeyError, AttributeError, cv2.error) as error:
      guards.reset()
      display.reset()
      packet['error'] = str(error)
      cloudlog.warning('CCNC lane classifier: %s', error)
    display_start = time.monotonic_ns()
    warnings = {}
    warning_context_valid = sm.valid['carControl'] and sm.alive['carControl']
    for side in ('left', 'right'):
      line = int(getattr(sm['carState'], side+'LaneLine', 0))
      line_warning = line % 10 not in (0, 5) if params.get_int('LaneLineCheck') >= 1 else line // 10 == 2
      warnings[side] = {'laneDeparture': bool(getattr(sm['carControl'].hudControl, side+'LaneDepart', False)) if warning_context_valid else False,
                        'blindSpot': bool(getattr(sm['carState'], side+'Blindspot', False)),
                        'laneLineWarning': line_warning, 'unknown': not warning_context_valid}
    packet['displayPreview'] = display.update({}, packet, enabled=params.get_bool('CcncLaneTypeDetection'),
      model_lanes=params.get_int('CcncModelLanes'), now_ns=time.monotonic_ns(), warnings=warnings)
    packet['ccncPreview'] = packet['displayPreview']['values']
    packet['displayMs'] = (time.monotonic_ns()-display_start)/1e6
    packet['processingMs'] = (time.monotonic_ns()-started_ns)/1e6
    packet['threadCpuMs'] = (time.thread_time_ns()-cpu_start)/1e6
    if not params.get_bool('CcncLaneTypeDetection'):
      break
    payload = json.dumps(packet, separators=(',', ':'), allow_nan=False).encode()
    msg = messaging.new_message('customReservedRawData2', size=len(payload), valid=True)
    msg.customReservedRawData2 = payload
    pm.send('customReservedRawData2', msg)
  guards.reset()


if __name__ == '__main__':
  main()
