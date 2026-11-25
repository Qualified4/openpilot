from itertools import count
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


LAUNCH = Path(__file__).resolve().parents[1] / "mapd_launch.sh"


@pytest.fixture
def launcher(tmp_path):
  binary = tmp_path / "mapd"
  state = tmp_path / "ccnc_mapd"
  env = {**os.environ, "MAPD_LAUNCH_BIN": str(binary), "MAPD_LAUNCH_STATE": str(state)}
  return binary, state, env


def make_binary(binary, script):
  binary.write_text("#!/usr/bin/env bash\n" + script)
  binary.chmod(0o755)


def test_no_binary_runs_nothing(launcher):
  _, state, env = launcher
  assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 0
  assert state.with_suffix(".failed").is_file()


def test_fast_crash_marks_failed(launcher):
  binary, state, env = launcher
  make_binary(binary, "exit 17\n")
  assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 17
  assert state.with_suffix(".crashes").read_text().strip() == "1"
  assert state.with_suffix(".failed").is_file()
  assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 0


def test_third_crash_marks_failed(launcher):
  binary, state, env = launcher
  env["MAPD_LAUNCH_FAST_S"] = "0"
  make_binary(binary, "exit 19\n")
  for n in range(1, 4):
    assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 19
    assert state.with_suffix(".crashes").read_text().strip() == str(n)
    assert state.with_suffix(".failed").exists() == (n == 3)


def test_clean_exit_is_not_failure(launcher):
  binary, state, env = launcher
  make_binary(binary, "exit 0\n")
  assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 0
  assert not state.with_suffix(".failed").exists()
  assert not state.with_suffix(".crashes").exists()


def test_sigint_forwarded_as_sigterm(launcher):
  binary, state, env = launcher
  ready = state.with_suffix(".ready")
  make_binary(binary, f"trap 'exit 42' TERM\ntouch '{ready}'\nwhile :; do sleep 0.05; done\n")
  with subprocess.Popen([LAUNCH], env=env, start_new_session=True) as proc:
    try:
      deadline = time.monotonic() + 3
      while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
      assert ready.is_file()
      proc.send_signal(signal.SIGINT)
      assert proc.wait(timeout=3) == 42
    finally:
      if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
  assert not state.with_suffix(".failed").exists()
  assert not state.with_suffix(".crashes").exists()


def test_enable_ccnc_mapd_conditions(tmp_path, monkeypatch):
  from openpilot.cereal import car
  from openpilot.common.params import Params
  from openpilot.system.manager import process_config
  from opendbc.car.hyundai.values import HyundaiExtFlags, HyundaiFlags

  cp = car.CarParams.new_message(brand="hyundai", openpilotLongitudinalControl=True,
                                 flags=int(HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC),
                                 extFlags=int(HyundaiExtFlags.CCNC_CLUSTER_RX))

  cases = count()

  def make_params(options):
    params = Params(str(tmp_path / str(next(cases))))
    for key, value in options.items():
      params.put(key, value)
    return params

  options = dict(CcncExtendedDisplay=True, CcncPosDisplayMode=1,
                 CarParamsPersistent=cp.to_bytes())
  params = make_params(options)
  failed = tmp_path / "mapd.failed"
  monkeypatch.setattr(process_config, "CCNC_MAPD_FAILED", failed)
  enabled = process_config.enable_ccnc_mapd
  assert enabled(True, params, cp)
  assert enabled(True, params, car.CarParams.new_message())
  assert not enabled(False, params, cp)
  for name in ("CcncExtendedDisplay", "CcncPosDisplayMode", "CarParamsPersistent"):
    assert not enabled(True, make_params({key: value for key, value in options.items() if key != name}), cp)
  assert not enabled(True, make_params(dict(options, CcncPosDisplayMode=2)), cp)
  for name, value in (("flags", int(cp.flags | HyundaiFlags.CANFD_HDA2)), ("extFlags", 0), ("brand", "toyota")):
    with car.CarParams.from_bytes(options["CarParamsPersistent"]) as persistent:
      unsupported = persistent.as_builder()
    setattr(unsupported, name, value)
    assert not enabled(True, params, unsupported)
    assert not enabled(True, make_params(dict(options, CarParamsPersistent=unsupported.to_bytes())), cp)
  assert not enabled(True, make_params(dict(options, CarParamsPersistent=b"invalid")), cp)
  failed.touch()
  assert not enabled(True, params, cp)
