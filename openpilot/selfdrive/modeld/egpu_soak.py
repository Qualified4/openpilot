"""Parked eGPU soak: run the active precompiled model on zero frames until a worker timeout reproduces.

Stop modeld first (ignition off, or modeld not using the eGPU): the GPU lock must be free.

  cd /data/openpilot && python3 -m openpilot.selfdrive.modeld.egpu_soak --frames 11000
  cd /data/openpilot && python3 -m openpilot.selfdrive.modeld.egpu_soak --model-sha <64 hex> --hz 20

Progress and mean milliseconds are printed every 500 frames. On a failure the worker
diagnostics (threads, blocked fd, pipe peers, Python stack) are printed as JSON and the
exit code is 1; they are also kept in /data/media/egpu_timeout/ and last_failure.json.
Exit code 2: the GPU lock is held (a modeld worker is running).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

GPU_LOCK = Path('/tmp/carrot-precompiled-gpu.lock')
REPORT_EVERY = 500


def gpu_lock_busy(path: Path = GPU_LOCK) -> bool:
  if sys.platform != 'linux':
    return False
  import fcntl
  try:
    with path.open('a') as f:
      fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
      fcntl.flock(f, fcntl.LOCK_UN)  # the soak's own worker takes it next
    return False
  except BlockingIOError:
    return True
  except OSError:
    return False


def model_path(sha: str | None) -> Path | None:
  from openpilot.selfdrive.modeld.precompiled_model import installed
  return installed(SimpleNamespace(sha256=sha) if sha else None)


def failure(frame: int, exc: Exception) -> dict:
  return {'frame': frame, 'error': f'{type(exc).__name__}: {exc}',
          'worker_diagnostics': getattr(exc, 'worker_diagnostics', None)}


def soak(path: Path, frames: int, hz: float, camera=(1928, 1208), out=print) -> int:
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.modeld.precompiled_runner import PrecompiledModelState
  try:
    model = PrecompiledModelState(*camera, path)
  except Exception as exc:
    out(json.dumps(failure(0, exc), indent=2, default=str))
    return 1
  try:
    buffers = {k: SimpleNamespace(data=np.zeros(model.frame_size, dtype=np.uint8)) for k in ('img', 'big_img')}
    transforms = {k: np.eye(3, dtype=np.float32) for k in buffers}
    inputs = {'desire_pulse': np.zeros(ModelConstants.DESIRE_LEN, np.float32),
              'traffic_convention': np.array([1, 0], np.float32), 'action_t': np.zeros(2, np.float32)}
    period = 1 / hz if hz > 0 else 0
    timings, held, worst, next_time = [], 0, 0., time.monotonic()
    for frame in range(1, frames + 1):
      started = time.monotonic()
      try:
        if model.run(buffers, transforms, inputs, False) is None:
          held += 1  # absorbed late reply or held output: not a failure
      except Exception as exc:
        out(json.dumps(failure(frame, exc), indent=2, default=str))
        return 1
      elapsed = time.monotonic() - started
      timings.append(elapsed)
      worst = max(worst, elapsed)
      if frame % REPORT_EVERY == 0 or frame == frames:
        out(f'frame {frame}/{frames}: mean {np.mean(timings) * 1000:.2f} ms, max {worst * 1000:.1f} ms, held {held}')
        timings, worst = [], 0.
      if period:
        next_time += period
        time.sleep(max(0., next_time - time.monotonic()))
    return 0
  finally:
    model.close()


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--frames', type=int, default=11000)
  parser.add_argument('--model-sha', help='installed precompiled model sha256 (default: active model)')
  parser.add_argument('--hz', type=float, default=0, help='request rate; 0 = as fast as possible')
  args = parser.parse_args(argv)
  if gpu_lock_busy():
    print(f'{GPU_LOCK} is locked: a modeld eGPU worker is running. Stop modeld (ignition off) and retry.')
    return 2
  path = model_path(args.model_sha)
  if path is None:
    print('no installed precompiled model' + (f' for {args.model_sha}' if args.model_sha else ''))
    return 1
  print(f'model {path}, {args.frames} frames, {args.hz or "max"} Hz')
  return soak(path, args.frames, args.hz)


if __name__ == '__main__':
  sys.exit(main())
