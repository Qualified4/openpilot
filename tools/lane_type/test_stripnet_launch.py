"""stripnet_launch.sh (process ccnc_lane_type): Rust daemon first, Python marking_daemon on a missing or failing binary."""
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

RUST = Path(__file__).resolve().parents[2] / 'openpilot/selfdrive/carrot/lane_type/rust'
ROOT = RUST.parents[4]


def stub(path: Path, body: str) -> Path:
  path.write_text('#!/usr/bin/env bash\n' + body)
  path.chmod(0o755)
  return path


@pytest.fixture
def env(tmp_path):
  log = tmp_path / 'log'
  python = stub(tmp_path / 'python', f'echo "python $PWD $*" >> {log}\n')
  (tmp_path / 'CcncExtendedDisplay').write_text('0')
  (tmp_path / 'CcncStripNetRust').write_text('1')
  e = dict(os.environ, STRIPNET_LAUNCH_STATE=str(tmp_path / 'state'), PARAMS_DIR=str(tmp_path), STRIPNET_STALL_S='2')
  return tmp_path, log, python, e


def launch(e, python, binary, **kw):
  return subprocess.Popen([str(RUST / 'stripnet_launch.sh'), str(python)], cwd=RUST, env=dict(e, STRIPNET_LAUNCH_BIN=str(binary)),
                          stderr=subprocess.DEVNULL, **kw)


def test_missing_binary_runs_python_from_the_checkout_root(env):
  tmp, log, python, e = env
  p = launch(e, python, tmp / 'none')
  assert p.wait(10) == 0
  assert log.read_text().split() == ['python', str(ROOT), '-m', 'openpilot.selfdrive.carrot.lane_type.marking_daemon']


def test_default_backend_is_python(env):
  tmp, log, python, e = env
  (tmp / 'CcncStripNetRust').write_text('0')
  assert launch(e, python, stub(tmp / 'rust', 'exit 0\n')).wait(10) == 0
  assert log.read_text().count('python') == 1 and not (tmp / 'state.failed').exists()


def test_normal_exit_is_not_a_failure(env):
  tmp, log, python, e = env
  p = launch(e, python, stub(tmp / 'rust', 'exit 0\n'))
  assert p.wait(10) == 0
  assert not log.exists() and not (tmp / 'state.failed').exists()


@pytest.mark.parametrize('body', ('exit 0\n', 'while :; do sleep 0.1; done\n'))
def test_enabled_daemon_without_publication_falls_back(env, body):
  tmp, log, python, e = env
  (tmp / 'CcncExtendedDisplay').write_text('1')
  assert launch(e, python, stub(tmp / 'rust', body)).wait(10) == 0
  assert (tmp / 'state.failed').exists()
  assert log.read_text().count('python') == 1


def test_fast_crash_falls_back_and_stays_on_python(env):
  tmp, log, python, e = env
  ran = tmp / 'ran'
  rust = stub(tmp / 'rust', f'echo x >> {ran}\nexit 3\n')
  assert launch(e, python, rust).wait(10) == 0
  assert (tmp / 'state.failed').exists() and log.read_text().count('python') == 1
  assert launch(e, python, rust).wait(10) == 0  # manager restart: the marker skips Rust
  assert ran.read_text().count('x') == 1 and log.read_text().count('python') == 2


def test_manager_sigint_stops_the_daemon_and_is_not_a_failure(env):
  tmp, log, python, e = env
  got = tmp / 'got'
  rust = stub(tmp / 'rust', f'trap "echo term > {got}; exit 143" TERM\nwhile :; do sleep 0.1; done\n')
  p = launch(e, python, rust)
  time.sleep(0.5)
  p.send_signal(signal.SIGINT)
  assert p.wait(10) == 143
  assert got.read_text().strip() == 'term'
  assert not log.exists() and not (tmp / 'state.failed').exists()
