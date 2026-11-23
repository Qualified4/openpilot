from itertools import count
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


LAUNCH = Path(__file__).resolve().parents[1] / "visiond_launch.sh"


@pytest.fixture
def launcher(tmp_path):
  binary = tmp_path / "visiond"
  state = tmp_path / "ccnc_visiond"
  env = {**os.environ, "VISIOND_LAUNCH_BIN": str(binary), "VISIOND_LAUNCH_STATE": str(state)}
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
  env["VISIOND_LAUNCH_FAST_S"] = "0"
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


def test_halt_reason_is_kept_apart_from_crashes(launcher):
  binary, state, env = launcher
  halted = state.with_suffix(".halted")
  halted.write_text("load: left by the previous start\n")
  make_binary(binary, f"[[ ! -e '{halted}' ]] || exit 23\necho 'load: no model' > \"$VISIOND_STATE.halted\"\nexit 0\n")
  assert subprocess.run([LAUNCH], env=env, timeout=3).returncode == 0
  assert halted.read_text() == "load: no model\n"
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


def test_enable_ccnc_visiond_conditions(tmp_path, monkeypatch):
  from openpilot.cereal import car
  from openpilot.common.params import Params
  from openpilot.system.manager import process_config
  from opendbc.car.hyundai.values import HyundaiExtFlags, HyundaiFlags

  # The same vehicles and display setting as the CCNC map process.
  hda1 = car.CarParams.new_message(brand="hyundai", openpilotLongitudinalControl=True,
                                   flags=int(HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC),
                                   extFlags=int(HyundaiExtFlags.CCNC_CLUSTER_RX))
  hda2 = car.CarParams.new_message(brand="hyundai", flags=int(HyundaiFlags.CANFD))
  stored_bytes = {id(hda1): hda1.to_bytes(), id(hda2): hda2.to_bytes()}
  cases = count()

  def make_params(value=None, display=True, persistent=hda1):
    params = Params(str(tmp_path / str(next(cases))))
    assert params.get_default_value("CcncVisionProcess") == 0
    if value is not None:
      with open(params.get_param_path("CcncVisionProcess"), "wb") as out:
        out.write(value)
    params.put_bool("CcncExtendedDisplay", display)
    if persistent is not None:
      params.put("CarParamsPersistent", stored_bytes[id(persistent)])
    return params

  failed = tmp_path / "visiond.failed"
  monkeypatch.setattr(process_config, "CCNC_VISIOND_FAILED", failed)
  enabled = process_config.enable_ccnc_visiond
  cp = car.CarParams.new_message()
  assert enabled(True, make_params(b"1"), cp)
  assert enabled(True, make_params(b"1"), hda1)
  assert not enabled(False, make_params(b"1"), cp)
  assert not enabled(True, make_params(), cp)
  for stored in (b"0", b"2", b"-1", b"11", b"1.5", b"on", b""):
    assert not enabled(True, make_params(stored), cp), stored
  assert not enabled(True, make_params(b"1", display=False), cp)
  assert not enabled(True, make_params(b"1", persistent=None), cp)
  assert not enabled(True, make_params(b"1", persistent=hda2), cp)
  assert not enabled(True, make_params(b"1"), hda2), "a live fingerprint vetoes the stored one"
  failed.touch()
  assert not enabled(True, make_params(b"1"), cp)


BUILT = LAUNCH.parent / "target/release/ccnc_visiond"


@pytest.mark.skipif(not BUILT.is_file(), reason="needs cargo build --release")
def test_missing_onnxruntime_halts_without_crash(launcher, tmp_path):
  # The default build has only ort, so without CCNC_VISIOND_ENGINE it loads onnxruntime, finds none and halts.
  _, state, env = launcher
  prefix = f"visiond-launch-{os.getpid()}-{time.monotonic_ns()}"
  shm = Path("/dev/shm") / f"msgq_{prefix}"
  shm.mkdir()
  env.pop("CCNC_VISIOND_ENGINE", None)
  env.update(VISIOND_LAUNCH_BIN=str(BUILT), OPENPILOT_PREFIX=prefix, CCNC_VISIOND_ASSETS=str(tmp_path),
             ORT_DYLIB_PATH=str(tmp_path / "missing" / "libonnxruntime.so"))
  halted = state.with_suffix(".halted")
  try:
    with subprocess.Popen([LAUNCH], env=env, start_new_session=True, stderr=subprocess.DEVNULL) as proc:
      try:
        deadline = time.monotonic() + 5
        while not halted.exists() and time.monotonic() < deadline:
          time.sleep(0.02)
        assert halted.read_text().startswith(f"load: loading {tmp_path / 'missing' / 'libonnxruntime.so'}")
        time.sleep(0.3)
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=3) == 0
      finally:
        if proc.poll() is None:
          os.killpg(proc.pid, signal.SIGKILL)
  finally:
    for path in shm.iterdir():
      path.unlink()
    shm.rmdir()
  assert not state.with_suffix(".failed").exists()
  assert not state.with_suffix(".crashes").exists()
