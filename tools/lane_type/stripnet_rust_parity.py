"""Offline equivalence of the Rust StripNet daemon core (openpilot/selfdrive/carrot/lane_type/rust) with marking_daemon.py.

Each segment's rlog events (their original bytes) and its road video frames go, in log order, to
  * the Python daemon path: marking_daemon's Recogniser / DisplayPolicy / packet / line_dict / lead_d_rel / FrameBuffer,
    with the main loop's SubMaster state (latest message, carState alive within 10 periods, liveCalibration valid);
  * the Rust replay binary (stripnet_replay), which parses the same bytes and frames itself.
Per modelV2 both outputs are compared: status, published packet (confirmed type/colour and ages, raw result, reason,
counters), the strip bytes (sha256) and the model logits. The publish time is the modelV2's logMonoTime for both.

  python tools/lane_type/stripnet_rust_parity.py --rust <stripnet_replay> --out <dir> ROUTE_DIR:SEG [...]
  python tools/lane_type/stripnet_rust_parity.py --scan ROUTE_DIR:SEG [...]   # blinker / laneChangeState counts only
"""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

SERVICES = ('modelV2', 'carState', 'liveCalibration', 'deviceState', 'roadCameraState', 'radarState')
CAR_ALIVE_NS = 100_000_000  # SubMaster: 10 periods of carState (100 Hz)


def framed(data):
  """The framed messages of a decompressed rlog, as their original bytes."""
  at = 0
  while at + 8 <= len(data):
    count = struct.unpack_from('<I', data, at)[0] + 1
    header = (4 + 4*count + 7)//8*8
    size = header + 8*sum(struct.unpack_from(f'<{count}I', data, at + 4))
    yield data[at:at + size]
    at += size


def events(path):
  import zstandard
  from openpilot.cereal import log
  data = zstandard.ZstdDecompressor().stream_reader(open(path, 'rb')).read()
  for raw, e in zip(framed(data), log.Event.read_multiple_bytes(data), strict=True):
    yield e.which(), raw, e


def segment_paths(spec):
  route_dir, seg = spec.rsplit(':', 1)
  route_dir = Path(route_dir)
  stem = f'{route_dir.name}--{seg}'
  return route_dir/f'{stem}--rlog.zst', route_dir/f'{stem}--fcamera.mp4', int(seg)


class FakeBuffer:
  def __init__(self, nv12, width, height, frame_id):
    self.data, self.width, self.height, self.stride, self.uv_offset = nv12, width, height, width, width*height
    self.frame_id = frame_id


class RecordingModel:
  """StripModel pass-through that records each inferred strip's sha256 and logits (inference order = left, right)."""
  def __init__(self, model):
    self.model, self.strip, self.mirror_right = model, model.strip, model.mirror_right
    self.last, self.infer_ms = [], 0.

  def infer(self, strips):
    self.last = []
    if not strips:
      return []
    types, colors = self.model.forward(strips)
    for s, t, c in zip(strips, types, colors, strict=True):
      self.last.append({'strip': hashlib.sha256(np.ascontiguousarray(s, dtype='<f4').tobytes()).hexdigest()[:16],
                        'typeLogits': [float(v) for v in t], 'colorLogits': [float(v) for v in c]})
    return [self.model.decide4(t, c) for t, c in zip(types, colors, strict=True)]


class PyDaemon:
  """marking_daemon.main's per-modelV2 body with replay inputs (no VisionIPC hold / slot checks)."""
  def __init__(self, model, record=True):
    from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy
    from openpilot.selfdrive.carrot.lane_type.marking_daemon import FrameBuffer, Recogniser
    self.model = RecordingModel(model) if record else model
    self.recogniser, self.policy = Recogniser(2, model=self.model, lanechange='C2'), DisplayPolicy()
    self.frames = FrameBuffer()
    self.latest, self.mono = {}, {}
    self.counters = {'processed': 0, 'exact': 0, 'nearest': 0, 'frame_mismatch': 0, 'stale_inputs': 0, 'errors': 0,
                     'drop_hold_before': 0, 'drop_frame_id_before': 0, 'drop_hold_after': 0, 'drop_frame_id_after': 0}
    self.odometer, self.last_stamp = 0., None
    self.info = model.info()

  def take(self, which, event):
    self.latest[which] = (getattr(event, which), event.valid)
    self.mono[which] = event.logMonoTime

  def step(self, model, model_mono, now_ns):
    import cv2
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.selfdrive.carrot.lane_type.marking_daemon import lead_d_rel, line_dict, packet, planes
    picked = self.frames.select(int(model.frameId), int(model.timestampEof))
    if picked is None:
      self.counters['frame_mismatch'] += 1
      return {'status': 'frame_mismatch'}
    frame_id, frame_eof, buffer, kind, recv_ns = picked
    self.counters[kind] += 1
    calibration, cal_valid = self.latest.get('liveCalibration', (None, False))
    car = self.latest.get('carState', (None, False))[0]
    device = self.latest.get('deviceState', (None, False))[0]
    car_alive = 'carState' in self.mono and model_mono - self.mono['carState'] < CAR_ALIVE_NS
    if (calibration is None or not cal_valid or str(calibration.calStatus) != 'calibrated' or len(calibration.rpyCalib) != 3 or
        len(model.laneLines) < 3 or not car_alive or device is None or not device.started):
      self.counters['stale_inputs'] += 1
      self.policy.reset()
      return {'status': 'stale_inputs'}
    start, cpu = time.perf_counter_ns(), time.thread_time_ns()
    stamp = frame_eof
    offset_s = (frame_eof - int(model.timestampEof))/1e9
    if self.last_stamp is not None and 0 < stamp - self.last_stamp < 2_000_000_000:
      self.odometer += max(0., float(car.vEgo))*(stamp - self.last_stamp)/1e9
    self.last_stamp = stamp
    sensor = str(self.latest['roadCameraState'][0].sensor) if 'roadCameraState' in self.latest else 'unknown'
    try:
      camera = DEVICE_CAMERAS[(str(device.deviceType), sensor)].fcam
      k = camera.intrinsics.copy()
      k[0] *= buffer.width/camera.width
      k[1] *= buffer.height/camera.height
      y_plane, uv_plane = planes(buffer)
      shift = max(0., float(car.vEgo))*offset_s
      radar = self.latest.get('radarState', (None, False))[0]
      frame = {'lines': {'left': line_dict(model.laneLines[1], shift), 'right': line_dict(model.laneLines[2], shift)},
               'probs': {'left': float(model.laneLineProbs[1]), 'right': float(model.laneLineProbs[2])},
               'rpy': list(calibration.rpyCalib), 'k': k, 'stamp': stamp, 'odometer': self.odometer,
               'leads': [{'prob': lead.prob, 'x': [x - shift for x in lead.x], 'y': list(lead.y)} for lead in model.leadsV3],
               'lane_change': bool(car.leftBlinker or car.rightBlinker or str(model.meta.laneChangeState) != 'off'),
               'blinker': bool(car.leftBlinker or car.rightBlinker), 'lcs': str(model.meta.laneChangeState),
               'lcs_change': str(model.meta.laneChangeState) != 'off', 'v_ego': float(car.vEgo),
               'lead_d_rel': lead_d_rel(radar, self.mono.get('radarState', 0), stamp) if radar is not None else None}
      if hasattr(self.model, 'last'):
        self.model.last = []
      raw = self.recogniser.process(y_plane, uv_plane, frame)
    except (ValueError, KeyError, AttributeError, cv2.error) as error:
      self.counters['errors'] += 1
      self.policy.reset()
      return {'status': 'error', 'error': str(error)}
    wall_ms, cpu_ms = (time.perf_counter_ns() - start)/1e6, (time.thread_time_ns() - cpu)/1e6
    confirmed = self.policy.confirm({'cameraMonoTimeNanos': stamp, **raw}, now_ns)
    self.recogniser.observe_confirmation(confirmed, frame)
    self.counters['processed'] += 1
    pub = packet(2, int(model.frameId), stamp, raw, confirmed, now_ns, {}, dict(self.counters), self.info)
    pub = json.loads(json.dumps(pub, separators=(',', ':'), allow_nan=False))
    return {'status': 'published', 'packet': pub, 'diag': getattr(self.model, 'last', []), 'wallMs': wall_ms, 'cpuMs': cpu_ms,
            'extractMs': self.recogniser.timing['extractMs'], 'inferMs': self.recogniser.timing['inferMs'],
            'lane_change': frame['lane_change'], 'luma': None}


class Rust:
  def __init__(self, binary, assets, threads):
    self.p = subprocess.Popen([binary, str(assets), '--threads', str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

  def send(self, tag, payload):
    self.p.stdin.write(tag + struct.pack('<I', len(payload)) + payload)

  def result(self):
    self.p.stdin.flush()
    return json.loads(self.p.stdout.readline())

  def close(self):
    self.p.stdin.close()
    self.p.wait()


def compare(py, rs, out):
  """Fold one modelV2's two results into the Counter ``out``; returns the list of differences (empty: equal)."""
  diffs = []
  out['modelV2'] += 1
  if py['status'] != rs['status']:
    return [('status', py['status'], rs['status'])]
  out[py['status']] += 1
  if py['status'] != 'published':
    return diffs
  a, b = py['packet'], rs['packet']
  if a['counters'] != b['counters']:
    diffs.append(('counters', a['counters'], b['counters']))
  for side in ('left', 'right'):
    pa, pb = a[side], b[side]
    for head in ('type', 'color'):
      out[f'{head}_frames'] += 1
      if pa[head] == pb[head]:
        out[f'{head}_match'] += 1
      else:
        diffs.append((f'{side}.{head}', pa[head], pb[head]))
      if pa[head + 'AgeNs'] != pb[head + 'AgeNs']:
        diffs.append((f'{side}.{head}AgeNs', pa[head + 'AgeNs'], pb[head + 'AgeNs']))
    ra, rb = pa['raw'], pb['raw']
    out['reason_frames'] += 1
    out['reason_match'] += ra['reason'] == rb['reason']
    for key in ('type', 'color', 'reason', 'typeClass', 'colorClass', 'double', 'blue'):
      if ra.get(key) != rb.get(key):
        diffs.append((f'{side}.raw.{key}', ra.get(key), rb.get(key)))
    for key in ('typeConfidence', 'colorConfidence', 'pUnknown'):
      if (key in ra) != (key in rb):
        diffs.append((f'{side}.raw.{key}', ra.get(key), rb.get(key)))
      elif key in ra:
        d = abs(ra[key] - rb[key])
        out[f'max_{key}'] = max(out[f'max_{key}'], d)
  da = py['diag']
  db = [d for d in rs['diag'] if d is not None]
  if len(da) != len(db):
    diffs.append(('inferred_sides', len(da), len(db)))
  for x, y in zip(da, db, strict=False):
    out['strips'] += 1
    out['strip_match'] += x['strip'] == y['strip']
    if x['strip'] != y['strip']:
      diffs.append(('strip', x['strip'], y['strip']))
    d = max(abs(p - q) for p, q in zip(x['typeLogits'] + x['colorLogits'], y['typeLogits'] + y['colorLogits'], strict=True))
    out['max_logit'] = max(out['max_logit'], d)
  return diffs


def run_segment(spec, rust, py, rows, totals, diffs_file):
  import av
  rlog, video, seg = segment_paths(spec)
  encode = {}
  stream = list(events(rlog))
  for which, _, e in stream:
    if which == 'roadEncodeIdx' and e.roadEncodeIdx.segmentNum == seg:
      encode[e.roadEncodeIdx.segmentId] = (e.roadEncodeIdx.frameId, e.roadEncodeIdx.timestampEof)
  by_frame = {}
  container = av.open(str(video))
  decoded = container.decode(video=0)
  next_index = 0
  stats = Counter()
  seg_totals = Counter()
  lumas = []

  def frame_for(frame_id):
    """Decode up to the video frame with this frameId (roadEncodeIdx) and hand it to both sides' frame buffers."""
    nonlocal next_index
    while frame_id not in by_frame:
      try:
        f = next(decoded)
      except StopIteration:
        return
      key = encode.get(next_index)
      next_index += 1
      if key is None:
        continue
      h, w = f.height, f.width
      p = f.to_ndarray(format='yuv420p')
      nv12 = np.empty((h*3//2, w), np.uint8)
      nv12[:h] = p[:h]
      c = p[h:].reshape(-1)
      nv12[h:].reshape(-1)[0::2] = c[:h*w//4]
      nv12[h:].reshape(-1)[1::2] = c[h*w//4:]
      data = nv12.tobytes()
      by_frame[key[0]] = True
      if len(by_frame) > 8:
        by_frame.pop(next(iter(by_frame)))
      lumas.append(float(nv12[:h:16, ::16].mean()))
      py.frames.frames.append((key[0], key[1], FakeBuffer(data, w, h, key[0]), 0))
      rust.send(b'F', struct.pack('<IQII', key[0], key[1], w, h) + data)

  for which, raw, e in stream:
    if which not in SERVICES:
      continue
    if which != 'modelV2':
      py.take(which, e)
      rust.send(b'E', raw)
      continue
    m = e.modelV2
    frame_for(int(m.frameId))
    now = int(e.logMonoTime)
    rust.send(b'N', struct.pack('<q', now))
    rust.send(b'E', raw)
    a = py.step(m, now, now)
    b = rust.result()
    d = compare(a, b, seg_totals)
    if d:
      stats['frames_with_diff'] += 1
      diffs_file.write(json.dumps({'segment': spec, 'frameId': int(m.frameId), 'diffs': d}, default=str) + '\n')
    if a['status'] == 'published' and b['status'] == 'published':
      p = a['packet']
      rows.append({'segment': spec, 'frameId': int(m.frameId), 'py': {'wall': a['wallMs'], 'cpu': a['cpuMs'], 'extract': a['extractMs'],
                   'infer': a['inferMs']}, 'rs': {'wall': b['wallMs'], 'cpu': b['cpuMs'], 'extract': b['extractMs'], 'infer': b['inferMs']},
                   'lane_change': a['lane_change'], 'inferred': len(a['diag']),
                   'yellow_solid': any(p[s]['type'] == 'solid' and p[s]['color'] == 'yellow' for s in ('left', 'right'))})
  container.close()
  seg_totals['frames_with_diff'] = stats['frames_with_diff']
  for k, v in seg_totals.items():
    totals[k] = max(totals[k], v) if k.startswith('max_') else totals[k] + v
  seg_rows = [r for r in rows if r['segment'] == spec]
  summary = {'segment': spec, 'luma_mean': float(np.mean(lumas)) if lumas else None, **dict(seg_totals),
             'lane_change_frames': sum(r['lane_change'] for r in seg_rows), 'yellow_solid_frames': sum(r['yellow_solid'] for r in seg_rows)}
  print(json.dumps(summary), flush=True)
  return summary


def serve_python(record, threads):
  """The Python daemon path as an engine process with stripnet_replay's stdin protocol (one JSON line per modelV2)."""
  import sys
  import cv2
  from openpilot.cereal import log
  from openpilot.selfdrive.carrot.lane_type.strip_extractor import tune_allocator
  from openpilot.selfdrive.carrot.lane_type.strip_model import StripModel
  cv2.setNumThreads(threads)
  tune_allocator()
  d = PyDaemon(StripModel(), record=record)
  stdin, stdout = sys.stdin.buffer, sys.stdout
  now = 0
  while True:
    head = stdin.read(5)
    if len(head) < 5:
      return
    tag, size = head[:1], struct.unpack('<I', head[1:])[0]
    payload = stdin.read(size)
    if tag == b'N':
      now = struct.unpack('<q', payload)[0]
    elif tag == b'F':
      fid, eof, w, h = struct.unpack_from('<IQII', payload)
      d.frames.frames.append((fid, eof, FakeBuffer(payload[20:], w, h, fid), 0))
    elif tag == b'E':
      with log.Event.from_bytes(payload) as e:
        which = e.which()
        if which != 'modelV2':
          d.take(which, e.as_builder().as_reader())
          continue
        r = d.step(e.modelV2, int(e.logMonoTime), now)
      r.pop('lane_change', None)
      r.pop('luma', None)
      stdout.write(json.dumps(r) + '\n')
      stdout.flush()


class Engine(Rust):
  """Any engine process speaking the replay protocol; its own peak RSS and CPU time are read from /proc before it exits."""
  def __init__(self, cmd):
    self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    self.start = time.monotonic()

  def stats(self):
    status = Path(f'/proc/{self.p.pid}/status').read_text()
    hwm = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('VmHWM:'))
    stat = Path(f'/proc/{self.p.pid}/stat').read_text().rsplit(')', 1)[1].split()
    import os
    tick = os.sysconf('SC_CLK_TCK')
    return {'maxRssMiB': hwm/1024, 'cpuS': (int(stat[11]) + int(stat[12]))/tick, 'wallS': time.monotonic() - self.start}


def run_engines(spec, engines, limit, rows, totals, diffs_file):
  """run_segment with every engine in its own process (device runs): the same events and frames to each, in lockstep."""
  import av
  rlog, video, seg = segment_paths(spec)
  stream = list(events(rlog))
  encode = {e.roadEncodeIdx.segmentId: (e.roadEncodeIdx.frameId, e.roadEncodeIdx.timestampEof)
            for which, _, e in stream if which == 'roadEncodeIdx' and e.roadEncodeIdx.segmentNum == seg}
  container = av.open(str(video))
  decoded = container.decode(video=0)
  sent, next_index, models = set(), 0, 0
  seg_totals = Counter()

  def frame_for(frame_id):
    nonlocal next_index
    while frame_id not in sent:
      try:
        f = next(decoded)
      except StopIteration:
        return
      key = encode.get(next_index)
      next_index += 1
      if key is None:
        continue
      h, w = f.height, f.width
      p = f.to_ndarray(format='yuv420p')
      nv12 = np.empty((h*3//2, w), np.uint8)
      nv12[:h] = p[:h]
      c = p[h:].reshape(-1)
      nv12[h:].reshape(-1)[0::2] = c[:h*w//4]
      nv12[h:].reshape(-1)[1::2] = c[h*w//4:]
      sent.add(key[0])
      record = struct.pack('<IQII', key[0], key[1], w, h) + nv12.tobytes()
      for e in engines.values():
        e.send(b'F', record)

  for which, raw, e in stream:
    if which not in SERVICES:
      continue
    if which != 'modelV2':
      for eng in engines.values():
        eng.send(b'E', raw)
      continue
    if models >= limit:
      break
    models += 1
    frame_for(int(e.modelV2.frameId))
    now = int(e.logMonoTime)
    for eng in engines.values():
      eng.send(b'N', struct.pack('<q', now))
      eng.send(b'E', raw)
    res = {name: eng.result() for name, eng in engines.items()}
    if 'py' in res and 'rs' in res:
      d = compare(res['py'], res['rs'], seg_totals)
      if d:
        seg_totals['frames_with_diff'] += 1
        diffs_file.write(json.dumps({'segment': spec, 'frameId': int(e.modelV2.frameId), 'diffs': d}, default=str) + '\n')
    for other in ('rsb',):
      if 'py' in res and other in res:
        sub = Counter()
        if compare(res['py'], res[other], sub):
          sub['frames_with_diff'] += 1
        for k, v in sub.items():
          k = f'{other}.{k}'
          seg_totals[k] = max(seg_totals[k], v) if k.split('.', 1)[1].startswith('max_') else seg_totals[k] + v
    row = {'segment': spec, 'index': models}
    for name, r in res.items():
      if r['status'] == 'published':
        row[name] = {'wall': r['wallMs'], 'extract': r['extractMs'], 'infer': r['inferMs'],
                     'inferred': sum(x is not None for x in r['diag']) if r.get('diag') else None}
    rows.append(row)
  container.close()
  for k, v in seg_totals.items():
    totals[k] = max(totals[k], v) if k.rsplit('.', 1)[-1].startswith('max_') else totals[k] + v
  print(json.dumps({'segment': spec, 'models': models, **dict(seg_totals)}), flush=True)


def scan(specs):
  for spec in specs:
    rlog, _, _ = segment_paths(spec)
    c = Counter()
    for which, _, e in events(rlog):
      if which == 'carState':
        c['blinker'] += bool(e.carState.leftBlinker or e.carState.rightBlinker)
      elif which == 'modelV2':
        c['models'] += 1
        c['lcs'] += str(e.modelV2.meta.laneChangeState) != 'off'
    print(spec, dict(c), flush=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('segments', nargs='*', help='ROUTE_DIR:SEGMENT')
  ap.add_argument('--rust', help='stripnet_replay binary')
  ap.add_argument('--out')
  ap.add_argument('--threads', type=int, default=2)
  ap.add_argument('--scan', action='store_true')
  ap.add_argument('--serve-python', action='store_true', help='run as the Python engine process (stdin protocol)')
  ap.add_argument('--record', action='store_true', help='--serve-python: strip sha256 / logits per side')
  ap.add_argument('--engine', action='append', default=[], help='NAME=COMMAND (space separated), each its own process')
  ap.add_argument('--limit', type=int, default=10**9, help='modelV2 per segment')
  a = ap.parse_args()
  if a.serve_python:
    return serve_python(a.record, a.threads)
  if a.scan:
    return scan(a.segments)
  if a.engine:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, totals = [], Counter()
    engines = {n: Engine(c.split()) for n, c in (x.split('=', 1) for x in a.engine)}
    with open(out/'diffs.jsonl', 'w') as diffs_file:
      for spec in a.segments:
        run_engines(spec, engines, a.limit, rows, totals, diffs_file)
    stats = {n: e.stats() for n, e in engines.items()}
    for e in engines.values():
      e.close()
    (out/'rows.json').write_text(json.dumps(rows))
    (out/'summary.json').write_text(json.dumps({'totals': dict(totals), 'engines': stats}, indent=1))
    print(json.dumps({'totals': dict(totals), 'engines': stats}, indent=1))
    return
  import cv2
  cv2.setNumThreads(a.threads)
  from openpilot.selfdrive.carrot.lane_type.strip_extractor import tune_allocator
  from openpilot.selfdrive.carrot.lane_type.strip_model import ASSETS, StripModel
  tune_allocator()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  model = StripModel()
  rows, totals, summaries = [], Counter(), []
  with open(out/'diffs.jsonl', 'w') as diffs_file:
    for spec in a.segments:
      # A fresh daemon per segment (a segment is its own replay; the odometer and filters start empty on both sides).
      rust = Rust(a.rust, ASSETS, a.threads)
      summaries.append(run_segment(spec, rust, PyDaemon(model), rows, totals, diffs_file))
      rust.close()
  (out/'timing.json').write_text(json.dumps(rows))
  report = {'segments': summaries, 'totals': dict(totals)}
  for eng in ('py', 'rs'):
    for key in ('wall', 'cpu', 'extract', 'infer'):
      v = np.array([r[eng][key] for r in rows])
      report.setdefault('timing', {})[f'{eng}.{key}'] = {'mean': float(v.mean()), 'p50': float(np.percentile(v, 50)),
                                                       'p95': float(np.percentile(v, 95)), 'p99': float(np.percentile(v, 99))} if len(v) else None
  two = [r for r in rows if r['inferred'] == 2]
  for eng in ('py', 'rs'):
    for key in ('wall', 'extract', 'infer'):
      v = np.array([r[eng][key] for r in two])
      report.setdefault('timing_two_sides', {})[f'{eng}.{key}'] = {'mean': float(v.mean()), 'p50': float(np.percentile(v, 50)),
                                                                 'p95': float(np.percentile(v, 95))} if len(v) else None
  (out/'summary.json').write_text(json.dumps(report, indent=1))
  print(json.dumps(report['totals'], indent=1))
  print(json.dumps(report['timing_two_sides'], indent=1))


if __name__ == '__main__':
  main()
