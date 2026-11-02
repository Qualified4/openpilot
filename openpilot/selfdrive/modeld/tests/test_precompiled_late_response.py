"""Late eGPU worker replies are absorbed briefly, one request at a time, through real pipes."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld import precompiled_runner as runner

# Each request stores its 1-based sequence number in the shared output, then replies.
WORKER = """
import struct, sys, time
output, delays = sys.argv[1], [float(x) for x in sys.argv[2].split(',')]
count = 0
while sys.stdin.buffer.read(1):
  count += 1
  time.sleep(delays[count - 1] if count <= len(delays) else 0)
  with open(output, 'r+b') as f:
    f.write(struct.pack('<ff', count, count))
  with open(output + '.count', 'w') as f:
    f.write(str(count))
  sys.stdout.buffer.write(b'1\\n')
  sys.stdout.buffer.flush()
"""


@pytest.fixture
def make_model(monkeypatch, tmp_path):
  events, warnings, failures, models = [], [], [], []
  monkeypatch.setattr(runner, 'log_event', lambda event, **fields: events.append((event, fields)))
  monkeypatch.setitem(sys.modules, 'openpilot.common.swaglog', SimpleNamespace(cloudlog=SimpleNamespace(warning=warnings.append)))
  from openpilot.selfdrive.modeld import precompiled_model
  monkeypatch.setattr(precompiled_model, 'record_failure', lambda path, exc, phase: failures.append((exc, phase)))

  def make(delays):
    shared = tmp_path / 'shared'
    shared.write_bytes(bytes(8))
    model = object.__new__(runner.PrecompiledModelState)
    model.process = subprocess.Popen([sys.executable, '-c', WORKER, str(shared), ','.join(map(str, delays))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    model.file = SimpleNamespace(name=str(shared))
    model.pkl_path, model.recovery_cancel = tmp_path / 'model.pkl', None
    model.first_run, model.frame_size = False, 16
    model.last_good_time, model.held_output = None, False
    model.prev_desire = np.zeros(8, np.float32)
    model.views = {name: np.zeros(shape, np.float32) for name, shape in
                   {'img': 16, 'big_img': 16, 'desire': 8, 'traffic_convention': 2, 'action_t': 2,
                    'tfm': (3, 3), 'big_tfm': (3, 3)}.items()}
    model.output = np.memmap(shared, np.float32, 'r', shape=(2,))
    model.output_slices = {'action': slice(0, 2)}
    model.parser = SimpleNamespace(parse_outputs=lambda value: value)
    model.close = lambda: (model.process.kill(), model.process.wait())
    model.requests = lambda: int((tmp_path / 'shared.count').read_text())
    models.append(model)
    return model

  make.events, make.warnings, make.failures = events, warnings, failures
  yield make
  for model in models:
    if model.process.poll() is None:
      model.process.kill()
    model.process.wait()


def frame(model, desire=0):
  frames = {name: SimpleNamespace(data=np.zeros(16, np.uint8)) for name in ('img', 'big_img')}
  transforms = {name: np.eye(3, dtype=np.float32) for name in frames}
  pulse = np.zeros(8, np.float32)
  pulse[desire] = desire > 0
  inputs = {'desire_pulse': pulse, 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)}
  output = model.run(frames, transforms, inputs, False)
  return None if output is None else float(output['action'][0, 0])


def test_one_late_reply_is_absorbed_and_pairs_stay_matched(make_model):
  model = make_model([0, .2, 0])
  assert frame(model) == 1
  assert frame(model, desire=1) is None  # request 2 is 200 ms late: held, worker kept
  np.testing.assert_array_equal(model.prev_desire, np.zeros(8))
  # Request 2's reply is consumed and discarded; this frame returns request 3, never 2.
  assert frame(model, desire=1) == 3
  assert frame(model) == 4
  assert model.requests() == 4 and model.process.poll() is None
  assert [e for e, _ in make_model.events] == ['late_response']
  assert 200 <= make_model.events[0][1]['ms'] < runner.LATE_RESPONSE_LIMIT * 1000
  assert len(make_model.warnings) == 1 and make_model.failures == []
  # The late request carried the pulse; it was consumed, so request 3 does not repeat it.
  assert model.views['desire'][1] == 0


def test_silent_worker_falls_back_after_the_late_limit_without_extra_requests(make_model):
  model = make_model([0, 1.1])
  assert frame(model) == 1
  started = runner.time.monotonic()
  results = []
  with pytest.raises(TimeoutError, match='^precompiled eGPU worker timed out$'):
    for _ in range(20):
      results.append(frame(model))
  elapsed = runner.time.monotonic() - started
  assert runner.LATE_RESPONSE_LIMIT <= elapsed < runner.LATE_RESPONSE_LIMIT + .15
  assert results and all(r is None for r in results)
  assert model.process.poll() is not None  # killed by the fallback
  assert model.requests() == 1  # the pending request was never followed by another 'r'
  assert make_model.events == [] and [phase for _, phase in make_model.failures] == ['inference']
  assert make_model.failures[0][0].worker_diagnostics['late_reply_ms'] >= runner.LATE_RESPONSE_LIMIT * 1000


def test_three_late_replies_within_the_window_fall_back(make_model):
  model = make_model([.2, .2, .2])
  assert frame(model) is None
  assert frame(model) is None  # request 1 consumed; request 2 is late too
  with pytest.raises(TimeoutError, match='^precompiled eGPU worker timed out$'):
    frame(model)  # request 2 consumed; request 3 is the third late reply
  assert [e for e, _ in make_model.events] == ['late_response'] * 2
  assert len(make_model.failures) == 1


def test_late_replies_outside_the_window_are_forgotten(make_model, monkeypatch):
  model = make_model([.2, 0, .2, 0, .2, 0])
  base = runner.time.monotonic
  offset = [0.]
  monkeypatch.setattr(runner.time, 'monotonic', lambda: base() + offset[0])
  for expected in (2, 4, 6):
    assert frame(model) is None
    assert frame(model) == expected
    offset[0] += runner.LATE_WINDOW_SECONDS
  assert len(make_model.events) == 3 and make_model.failures == []


def test_first_run_keeps_its_own_budget(make_model):
  model = make_model([.2])
  model.first_run = True
  assert frame(model) == 1
  assert make_model.events == []
