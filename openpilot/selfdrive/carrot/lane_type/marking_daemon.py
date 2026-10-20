"""CCNC lane marking recogniser process (display only): Scanner or StripNet, never both.

Runs outside modeld/controlsd/card. Every new full-resolution road camera NV12 frame goes into a small buffer
of the newest FRAME_BUFFER frames (older ones are dropped; buffers are camerad's shared memory, referenced not
copied). For each new modelV2 the frame the model actually used (same frameId and EOF) is taken first; if it
is not buffered, the frame whose EOF is nearest the model's timestampEof (the model's input frame time, not its
receive time) within MAX_FRAME_OFFSET_NS, with the model's lane geometry shifted by the travel in between. The selected recogniser's
per-frame result goes through the existing display filter (display.DisplayPolicy: two identical results
within 0.4 s, 0.5 s expiry); the confirmed left/right type/colour, their age and timing are published on
customReservedRawData2 for card to read without waiting. No CAN, steering, longitudinal or lane-change
decision is computed here.
"""
import json
import os
import time

from openpilot.selfdrive.carrot.lane_type.marking import MODES, PACKET_TYPE, PACKET_VERSION, mode_of

PARAM_CHECK_S = 1.
# Recorded timing (190 segments): modelV2 arrives 37-53 ms (p5-p99) after its frame, so with 50 ms frames the
# newest frame is sometimes already the next one. 3 frames cover that with margin; camerad recycles its 18
# VisionIPC buffers only after ~0.9 s, so referencing 3 recent buffers during processing is safe.
FRAME_BUFFER = 3
MAX_FRAME_OFFSET_NS = 50_000_000  # one frame period; a two-frame (100 ms) offset is never accepted
# A held VisionIPC buffer is camerad's shared slot, not a copy: the ISP rewrites the slot of frame r while reading out
# frame r+18, before camerad republishes it, so the oldest buffered frame has ~0.7 s from recv() until its pixels can
# change (.analysis/lane-marking-vipc-buffer-20261019). Frames held this long are not processed or published.
MAX_HOLD_NS = 500_000_000
# StripNet: a lane line prob below LANE_PROB_MIN is not inferred. A short dip (LANE_PROB_LOST <= prob, for at most
# LANE_DIP_MAX_NS since the side first fell below LANE_PROB_MIN) is 'lane_prob_dip': the frame is skipped (unknown, not
# entered in the strip filter's windows) and neither those windows nor DisplayPolicy are emptied. Below LANE_PROB_LOST,
# or a dip lasting longer, is 'not_detected', which empties both.
LANE_PROB_MIN, LANE_PROB_LOST, LANE_DIP_MAX_NS = .5, .3, 1_000_000_000
# StripNet, blinker alone (laneChangeState off): at lane-change speed (vEgo >= BLINKER_INFER_SPEED, 30 km/h) a side
# whose line is confident (prob >= BLINKER_PROB) and beside the car (|y at the first model point| >= BLINKER_Y m, not
# the line being crossed) is inferred as usual; otherwise the blinker resets it ('lane_change'), as does any
# laneChangeState other than off (10-20 segment 9: a blinker held through two lane changes kept the old white dashed
# line for 10 s while the model saw the yellow solid one). Below the speed (turn lanes, intersections) the blinker
# resets as before (10-20 segment 3 L 14.85 s: a turn-lane block marking taken for a white solid, DaBin 10-20).
BLINKER_PROB, BLINKER_Y, BLINKER_INFER_SPEED = .7, 1., 8.3


def requested(started, params, CP=None):
  return started and mode_of(params) in MODES


def line_dict(line, shift_m=0.):
  """Model lane line; ``shift_m`` moves it towards the car by the distance driven between the model's frame and
  the processed frame (straight-motion approximation, used only for a neighbouring frame)."""
  return {'x': [x - shift_m for x in line.x], 'y': list(line.y), 'z': list(line.z)}


class FrameBuffer:
  """Newest few camera frames with their own frameId/EOF (captured at recv time, not from the client later)
  and the monotonic time they were received."""
  def __init__(self, size=FRAME_BUFFER):
    from collections import deque
    self.frames = deque(maxlen=size)

  def drain(self, client):
    """Receive everything queued without waiting; only the newest ``size`` frames are kept."""
    received = 0
    while True:
      buf = client.recv(timeout_ms=0)
      if buf is None:
        return received
      self.frames.append((int(client.frame_id), int(client.timestamp_eof), buf, time.monotonic_ns()))
      received += 1

  def select(self, frame_id, eof, max_offset_ns=MAX_FRAME_OFFSET_NS):
    """(frame_id, eof, buf, kind, recv_ns) for the model's frame: exact first, else nearest EOF within the offset."""
    for fid, feof, buf, recv_ns in self.frames:
      if fid == frame_id and feof == eof:
        return fid, feof, buf, 'exact', recv_ns
    if not self.frames:
      return None
    fid, feof, buf, recv_ns = min(self.frames, key=lambda f: abs(f[1] - eof))
    return (fid, feof, buf, 'nearest', recv_ns) if abs(feof - eof) <= max_offset_ns else None

  def clear(self):
    self.frames.clear()


def buffer_drop_reason(frame_id, buf, recv_ns, now_ns, max_hold_ns=MAX_HOLD_NS):
  """None while the held buffer still holds frame ``frame_id``, else 'hold' or 'frame_id'.

  The hold time is the guarantee: the ISP overwrites a slot's pixels before camerad updates the frame id stored
  in it, so a matching ``buf.frame_id`` alone does not prove the pixels are untouched. A different id means the
  slot has already been republished (or camerad restarted) and is always a drop."""
  if now_ns - recv_ns >= max_hold_ns:
    return 'hold'
  if int(buf.frame_id) != frame_id:
    return 'frame_id'
  return None


class Recogniser:
  """Only the selected engine is constructed (the other is never imported or loaded).

  StripNet (mode 2) always runs its type and colour results through a strip_filter.StripFilter, by default the
  asymmetric A8/4 type filter (strip_filter.LONG_M / SHORT_M) with the strip_filter.COLOR_M colour window and the
  latch (strip_filter.LATCH: the last confident type / colour is kept until the other one is confident), applied
  to each side's filter_input result before it leaves process(). ``strip_filter`` is for tests and replays only:
  None builds that default, another StripFilter is used instead of it, and False turns the filter off (the raw
  filter_input results, both heads). The Scanner (mode 1) has no filter unless one is passed, and ignores it."""
  def __init__(self, mode, strip_filter=None):
    self.mode = mode
    if strip_filter is None and mode == 2:
      from openpilot.selfdrive.carrot.lane_type.strip_filter import COLOR_M, LATCH, LONG_M, SHORT_M, StripFilter
      strip_filter = StripFilter('asym', long_m=LONG_M, short_m=SHORT_M, color_m=COLOR_M, latch=LATCH)
    self.strip_filter = strip_filter or None
    self.dip_since = {}  # side -> camera stamp of the first frame below LANE_PROB_MIN (StripNet)
    if mode == 1:
      from openpilot.selfdrive.carrot.lane_type.profile_scanner import ProfileScanner
      self.scanners = {side: ProfileScanner(side) for side in ('left', 'right')}
      self.model = None
    elif mode == 2:
      from openpilot.selfdrive.carrot.lane_type.strip_model import StripModel
      self.model = StripModel()
      self.scanners = None
    else:
      raise ValueError('lane marking mode must be 1 (Scanner) or 2 (StripNet)')

  def reset(self):
    """Forget tracking that may hold samples from a reused buffer (StripNet itself keeps no per-frame state; its
    strip filter's windows are emptied)."""
    if self.scanners:
      for scanner in self.scanners.values():
        scanner.reset()
    if self.strip_filter is not None:
      self.strip_filter.reset()
    self.dip_since.clear()

  def _skip_reason(self, side, frame):
    """StripNet: why this side is not inferred ('lane_change', 'not_detected', 'lane_prob_dip'), or None."""
    prob, stamp = frame['probs'][side], frame['stamp']
    if prob >= LANE_PROB_MIN:
      self.dip_since.pop(side, None)
    else:
      since = self.dip_since.setdefault(side, stamp)
      if stamp < since:
        since = self.dip_since[side] = stamp
    if 'blinker' in frame:
      blinker = frame['blinker'] and not (frame.get('v_ego', 0.) >= BLINKER_INFER_SPEED and prob >= BLINKER_PROB
                                          and abs(frame['lines'][side]['y'][0]) >= BLINKER_Y)
      if frame['lcs_change'] or blinker:
        return 'lane_change'
    elif frame['lane_change']:  # a frame without the split flags (older callers): any lane change resets
      return 'lane_change'
    if prob >= LANE_PROB_MIN:
      return None
    return 'lane_prob_dip' if prob >= LANE_PROB_LOST and stamp - since <= LANE_DIP_MAX_NS else 'not_detected'

  def process(self, y_plane, uv_plane, frame):
    """frame: lines/probs per side, rpy, k, stamp (camera EOF ns), odometer, leads, lane_change (blinker or
    laneChangeState), optionally blinker, lcs_change (laneChangeState != off) and v_ego separately for StripNet."""
    out = {}
    if self.mode == 1:
      for side in ('left', 'right'):
        r = self.scanners[side].update(y_plane, uv_plane, frame['lines'][side], frame['probs'][side], frame['rpy'], frame['k'],
                                       frame['stamp'], frame['odometer'], leads=frame['leads'], lane_change=frame['lane_change'])
        out[side] = filter_input(r['type'], r['color'], .99, .99, frame['stamp'], dashed_observed=r['type'] == 'dashed',
                                 reason='lane_change' if frame['lane_change'] else r['reason'], reset=r['reset'])
      return out
    from openpilot.selfdrive.carrot.lane_type.strip_model import strip_input
    strips, sides = [], []
    for side in ('left', 'right'):
      skip = self._skip_reason(side, frame)
      if skip:
        out[side] = filter_input('unknown', 'unknown', 0., 0., frame['stamp'], reason=skip)
        continue
      s = strip_input(y_plane, uv_plane, frame['lines'][side], frame['rpy'], frame['k'])
      if s is None:
        out[side] = filter_input('unknown', 'unknown', 0., 0., frame['stamp'], reason='invalid_lane_geometry')
        continue
      strips.append(s)
      sides.append(side)
    models = {}
    for side, r in zip(sides, self.model.infer(strips), strict=True):
      # The strip model's own confident dashed decision is its pattern evidence.
      out[side] = filter_input(r['type'], r['color'], r['typeConfidence'], r['colorConfidence'], frame['stamp'],
                               dashed_observed=r['type'] == 'dashed', reason=r['reason'])
      models[side] = r
    if self.strip_filter is not None:
      from openpilot.selfdrive.carrot.lane_type.strip_filter import p_dashed, p_yellow
      for side in ('left', 'right'):
        # filter_input drops the model's probabilities, so P(dashed) / P(yellow) come from the model output;
        # reset-reason sides (no inference) are passed too and empty the windows; a lane_prob_dip side is passed
        # without probabilities, so it is skipped and keeps them. Unlearned yellow-dashed frames stay out of both.
        r = models.get(side)
        learned = r is not None and r['reason'] == 'strip_model'
        p, py = (p_dashed(r), p_yellow(r)) if learned else (None, None)
        out[side] = dict(self.strip_filter.update(side, out[side], p, frame['stamp'], frame['odometer'], py),
                         typePDashed=p, colorPYellow=py)
    return out


def filter_input(kind, color, type_conf, color_conf, stamp, *, dashed_observed=False, reason='', reset=False):
  """Result dict in the form display.DisplayPolicy expects."""
  return {'type': kind, 'color': color, 'typeConfidence': float(type_conf if kind in ('dashed', 'solid') else 0.),
          'colorConfidence': float(color_conf if color in ('white', 'yellow') else 0.), 'typeMonoTimeNanos': stamp,
          'colorMonoTimeNanos': stamp, 'dashedPatternObserved': bool(dashed_observed), 'dashedPatternMonoTimeNanos': stamp,
          'reason': reason, 'trackingReset': bool(reset)}


def packet(mode, frame_id, camera_ns, raw, confirmed, now_ns, timing, counters):
  data = {'type': PACKET_TYPE, 'version': PACKET_VERSION, 'mode': MODES[mode], 'frameId': frame_id, 'cameraMonoTimeNanos': camera_ns,
          'publishMonoTimeNanos': now_ns, 'timing': timing, 'counters': counters}
  for side in ('left', 'right'):
    heads = confirmed.get(side, {})
    entry = {'raw': {k: raw.get(side, {}).get(k) for k in ('type', 'color', 'typeConfidence', 'colorConfidence', 'reason')}}
    for head in ('type', 'color'):
      h = heads.get(head)
      entry[head] = h['value'] if h else 'unknown'
      entry[head + 'AgeNs'] = int(max(0, now_ns - h['stamp'])) if h else None
    data[side] = entry
  return data


def planes(buffer):
  """Y and interleaved UV views of the shared NV12 buffer (no full-frame copy)."""
  import numpy as np
  w, h, stride, uv_offset = buffer.width, buffer.height, buffer.stride, buffer.uv_offset
  raw = np.frombuffer(buffer.data, dtype=np.uint8)
  if raw.size < uv_offset + stride*(h//2) or w % 2 or h % 2:
    raise ValueError('invalid NV12 layout')
  y = raw[:stride*h].reshape(h, stride)[:, :w]
  uv = raw[uv_offset:uv_offset + stride*(h//2)].reshape(h//2, stride//2, 2)[:, :w//2]
  return y, uv


def main():
  from openpilot.common.params import Params
  params = Params()
  mode = mode_of(params)
  if mode not in MODES:
    return  # OFF: no OpenCV import, no model load, no IPC
  import cv2
  cv2.setNumThreads(1)
  from msgq.visionipc import VisionIpcClient, VisionStreamType
  import openpilot.cereal.messaging as messaging
  from openpilot.common.swaglog import cloudlog
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy

  if hasattr(os, 'sched_setaffinity'):
    background = set(os.sched_getaffinity(0)) & {0, 1, 2, 3}
    if background:
      os.sched_setaffinity(0, background)
  if hasattr(os, 'nice'):
    os.nice(10)
  sm = messaging.SubMaster(['modelV2', 'liveCalibration', 'carState', 'deviceState', 'roadCameraState'], poll='modelV2')
  pm = messaging.PubMaster(['customReservedRawData2'])
  client = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_ROAD, False)  # every frame; FrameBuffer keeps 3
  frames = FrameBuffer()
  recogniser, policy = Recogniser(mode), DisplayPolicy()
  counters = {'processed': 0, 'exact': 0, 'nearest': 0, 'frame_mismatch': 0, 'stale_inputs': 0, 'errors': 0,
              'drop_hold_before': 0, 'drop_frame_id_before': 0, 'drop_hold_after': 0, 'drop_frame_id_after': 0}
  odometer, last_stamp, next_param = 0., None, time.monotonic() + PARAM_CHECK_S
  while True:
    if time.monotonic() >= next_param:
      next_param = time.monotonic() + PARAM_CHECK_S
      new_mode = mode_of(params)
      if new_mode not in MODES:
        break
      if new_mode != mode:
        mode, recogniser = new_mode, Recogniser(new_mode)
        policy.reset()
        odometer, last_stamp = 0., None
    sm.update(100)
    if not client.is_connected():
      if not client.connect(False):
        continue
      frames.clear()
    frames.drain(client)
    if not sm.updated['modelV2']:
      continue
    model = sm['modelV2']
    picked = frames.select(int(model.frameId), int(model.timestampEof))
    if picked is None:
      counters['frame_mismatch'] += 1
      continue
    frame_id, frame_eof, buffer, kind, recv_ns = picked
    counters[kind] += 1
    calibration, car, device = sm['liveCalibration'], sm['carState'], sm['deviceState']
    if (not sm.valid['liveCalibration'] or str(calibration.calStatus) != 'calibrated' or len(calibration.rpyCalib) != 3 or
        len(model.laneLines) < 3 or not sm.alive['carState'] or not device.started):
      counters['stale_inputs'] += 1
      policy.reset()
      continue
    drop = buffer_drop_reason(frame_id, buffer, recv_ns, time.monotonic_ns())
    if drop:  # nothing read from the buffer yet: skip the frame, keep all state
      counters[f'drop_{drop}_before'] += 1
      continue
    start_ns, cpu_start = time.monotonic_ns(), time.thread_time_ns()
    stamp = frame_eof  # the processed pixels' time
    offset_s = (frame_eof - int(model.timestampEof))/1e9
    if last_stamp is not None and 0 < stamp - last_stamp < 2_000_000_000:
      odometer += max(0., float(car.vEgo))*(stamp - last_stamp)/1e9
    last_stamp = stamp
    try:
      camera = DEVICE_CAMERAS[(str(device.deviceType), str(sm['roadCameraState'].sensor))].fcam
      k = camera.intrinsics.copy()
      k[0] *= buffer.width/camera.width
      k[1] *= buffer.height/camera.height
      y_plane, uv_plane = planes(buffer)
      shift = max(0., float(car.vEgo))*offset_s
      frame = {'lines': {'left': line_dict(model.laneLines[1], shift), 'right': line_dict(model.laneLines[2], shift)},
               'probs': {'left': float(model.laneLineProbs[1]), 'right': float(model.laneLineProbs[2])},
               'rpy': list(calibration.rpyCalib), 'k': k, 'stamp': stamp, 'odometer': odometer,
               'leads': [{'prob': lead.prob, 'x': [x - shift for x in lead.x], 'y': list(lead.y)} for lead in model.leadsV3],
               'lane_change': bool(car.leftBlinker or car.rightBlinker or str(model.meta.laneChangeState) != 'off'),
               'blinker': bool(car.leftBlinker or car.rightBlinker),
               'lcs_change': str(model.meta.laneChangeState) != 'off', 'v_ego': float(car.vEgo)}
      raw = recogniser.process(y_plane, uv_plane, frame)
    except (ValueError, KeyError, AttributeError, cv2.error) as error:
      counters['errors'] += 1
      cloudlog.warning('CCNC lane marking: %s', error)
      policy.reset()
      continue
    drop = buffer_drop_reason(frame_id, buffer, recv_ns, time.monotonic_ns())
    if drop:  # the pixels may have changed while they were read: do not publish, forget what the scanner took in
      counters[f'drop_{drop}_after'] += 1
      recogniser.reset()
      continue
    now_ns = time.monotonic_ns()
    confirmed = policy.confirm({'cameraMonoTimeNanos': stamp, **raw}, now_ns)
    counters['processed'] += 1
    timing = {'processingMs': (now_ns - start_ns)/1e6, 'threadCpuMs': (time.thread_time_ns() - cpu_start)/1e6,
              'cameraToPublishMs': (now_ns - stamp)/1e6, 'frameOffsetMs': offset_s*1e3, 'frameMatch': kind}
    payload = json.dumps(packet(mode, int(model.frameId), stamp, raw, confirmed, now_ns, timing, counters),
                         separators=(',', ':'), allow_nan=False).encode()
    msg = messaging.new_message('customReservedRawData2', size=len(payload), valid=True)
    msg.customReservedRawData2 = payload
    pm.send('customReservedRawData2', msg)


if __name__ == '__main__':
  main()
