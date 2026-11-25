import hashlib
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
  env = {**os.environ, "VISIOND_LAUNCH_BIN": str(binary), "VISIOND_LAUNCH_STATE": str(state),
         "VISIOND_LAUNCH_ASSETS": str(tmp_path / "assets"), "VISIOND_LAUNCH_MANIFEST": str(tmp_path / "assets.sha256")}
  return binary, state, env


ASSET_NAMES = ("yolox_tiny_int8.onnx", "kind_mnv3s_128.onnx", "libonnxruntime.so.1.32.0")
ORT_LIB = ASSET_NAMES[-1]


def make_assets(tmp_path, names=ASSET_NAMES):
  """Stand-in asset files and a manifest with their sha256, as the launcher reads them."""
  assets = tmp_path / "assets"
  assets.mkdir(exist_ok=True)
  lines = []
  for name in names:
    data = f"not really {name}\n".encode()
    (assets / name).write_bytes(data)
    lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
  (tmp_path / "assets.sha256").write_text("".join(lines))
  return assets


def test_manifest_lists_the_launcher_assets():
  manifest = LAUNCH.parent / "assets.sha256"
  rows = [line.split() for line in manifest.read_text().splitlines() if line.strip()]
  assert tuple(name for _, name in rows) == ASSET_NAMES
  assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest, _ in rows)
  assert "ORT_LIB=" + ORT_LIB in LAUNCH.read_text()


def run_env_dump(binary, env, tmp_path):
  dump = tmp_path / "env"
  make_binary(binary, f"echo \"halt=${{CCNC_VISIOND_HALT-unset}}\" > '{dump}'\n"
                      f"echo \"assets=$CCNC_VISIOND_ASSETS\" >> '{dump}'\necho \"ort=$ORT_DYLIB_PATH\" >> '{dump}'\n"
                      f"echo \"policy=$CCNC_VISIOND_POLICY\" >> '{dump}'\necho \"cpus=$CCNC_VISIOND_CPUS\" >> '{dump}'\n"
                      f"echo \"threads=$CCNC_VISIOND_THREADS\" >> '{dump}'\necho \"min_period=$CCNC_VISIOND_MIN_PERIOD_MS\" >> '{dump}'\n"
                      f"exit 0\n")
  assert subprocess.run([LAUNCH], env=env, timeout=10).returncode == 0
  return dict(line.split("=", 1) for line in dump.read_text().splitlines())


def test_matching_assets_start_without_halt(launcher, tmp_path):
  binary, state, env = launcher
  assets = make_assets(tmp_path)
  clean = {k: v for k, v in env.items() if not k.startswith("CCNC_VISIOND_")}
  got = run_env_dump(binary, {**clean, "CCNC_VISIOND_HALT": "left over"}, tmp_path)
  assert got == {"halt": "unset", "assets": str(assets), "ort": str(assets / ORT_LIB),
                 "policy": "idle", "cpus": "0-2", "threads": "2", "min_period": "0"}
  assert not state.with_suffix(".failed").exists()


def test_launcher_forwards_scheduling_overrides(launcher, tmp_path):
  binary, state, env = launcher
  make_assets(tmp_path)
  got = run_env_dump(binary, {**env, "CCNC_VISIOND_POLICY": "other", "CCNC_VISIOND_CPUS": "4",
                              "CCNC_VISIOND_THREADS": "1", "CCNC_VISIOND_MIN_PERIOD_MS": "715"}, tmp_path)
  assert got["policy"] == "other"
  assert got["cpus"] == "4"
  assert got["threads"] == "1"
  assert got["min_period"] == "715"
  assert not state.with_suffix(".failed").exists()


@pytest.mark.parametrize("damage", ["missing", "changed", "no manifest"])
def test_unusable_assets_halt_the_process(launcher, tmp_path, damage):
  binary, state, env = launcher
  assets = make_assets(tmp_path)
  if damage == "missing":
    (assets / ASSET_NAMES[0]).unlink()
    reason = f"missing {assets / ASSET_NAMES[0]}"
  elif damage == "changed":
    (assets / ASSET_NAMES[1]).write_bytes(b"other bytes")
    reason = f"sha256 of {assets / ASSET_NAMES[1]} is {hashlib.sha256(b'other bytes').hexdigest()}, expected "
  else:
    (tmp_path / "assets.sha256").unlink()
    reason = f"no manifest {tmp_path / 'assets.sha256'}"
  got = run_env_dump(binary, env, tmp_path)
  assert got["halt"].startswith(reason), got
  # A halt is not a crash: the launcher starts the process, which stays alive and says why.
  assert not state.with_suffix(".failed").exists()
  assert not state.with_suffix(".crashes").exists()


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

  def make_params(display=True, persistent=hda1, pos_mode=1):
    params = Params(str(tmp_path / str(next(cases))))
    params.put_bool("CcncExtendedDisplay", display)
    params.put("CcncPosDisplayMode", pos_mode)
    if persistent is not None:
      params.put("CarParamsPersistent", stored_bytes[id(persistent)])
    return params

  failed = tmp_path / "visiond.failed"
  monkeypatch.setattr(process_config, "CCNC_VISIOND_FAILED", failed)
  map_failed = tmp_path / "mapd.failed"
  monkeypatch.setattr(process_config, "CCNC_MAPD_FAILED", map_failed)
  enabled = process_config.enable_ccnc_visiond
  cp = car.CarParams.new_message()
  # Its only reader is ccnc_mapd: it runs exactly when the map process does.
  assert enabled(True, make_params(), cp)
  assert enabled(True, make_params(), hda1)
  assert not enabled(False, make_params(), cp)
  assert not enabled(True, make_params(pos_mode=2), cp)
  map_failed.touch()
  assert not enabled(True, make_params(), cp)
  map_failed.unlink()
  assert enabled(True, make_params(), cp)
  assert not enabled(True, make_params(display=False), cp)
  assert not enabled(True, make_params(persistent=None), cp)
  assert not enabled(True, make_params(persistent=hda2), cp)
  assert not enabled(True, make_params(), hda2), "a live fingerprint vetoes the stored one"
  failed.touch()
  assert not enabled(True, make_params(), cp)


BUILT = LAUNCH.parent / "target/release/ccnc_visiond"


@pytest.mark.skipif(not BUILT.is_file(), reason="needs cargo build --release")
@pytest.mark.parametrize("assets_present", [False, True])
def test_unusable_assets_halt_without_crash(launcher, tmp_path, assets_present):
  # Missing assets: the launcher's reason. Present with the listed hashes but not a real onnxruntime: loading fails.
  _, state, env = launcher
  prefix = f"visiond-launch-{os.getpid()}-{time.monotonic_ns()}"
  shm = Path("/dev/shm") / f"msgq_{prefix}"
  shm.mkdir()
  for name in ("CCNC_VISIOND_ENGINE", "CCNC_VISIOND_ASSETS", "ORT_DYLIB_PATH", "CCNC_VISIOND_HALT",
               "CCNC_VISIOND_POLICY", "CCNC_VISIOND_CPUS", "CCNC_VISIOND_THREADS", "CCNC_VISIOND_MIN_PERIOD_MS"):
    env.pop(name, None)
  env.update(VISIOND_LAUNCH_BIN=str(BUILT), OPENPILOT_PREFIX=prefix, CCNC_VISIOND_CPUS="0")
  assets = tmp_path / "assets"
  if assets_present:
    make_assets(tmp_path)
    reason = f"load: loading {assets / ORT_LIB}"
  else:
    make_assets(tmp_path)
    for name in ASSET_NAMES:
      (assets / name).unlink()
    reason = f"assets: missing {assets / ASSET_NAMES[0]}"
  halted = state.with_suffix(".halted")
  try:
    with subprocess.Popen([LAUNCH], env=env, start_new_session=True, stderr=subprocess.DEVNULL) as proc:
      try:
        deadline = time.monotonic() + 5
        while not halted.exists() and time.monotonic() < deadline:
          time.sleep(0.02)
        assert halted.read_text().startswith(reason), halted.read_text()
        time.sleep(0.3)
        assert proc.poll() is None
        # Verify the real child binary took SCHED_IDLE and pinned affinity (even before model load / in halt).
        child_pids = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdecimal()
                      and (p / "stat").exists() and int((p / "stat").read_text().rsplit(")", 1)[1].split()[1]) == proc.pid]
        assert len(child_pids) == 1
        cpid = child_pids[0]
        for task in (Path("/proc") / str(cpid) / "task").iterdir():
          tid = int(task.name)
          assert os.sched_getscheduler(tid) == os.SCHED_IDLE
          cpus = next(l.split()[1] for l in (task / "status").read_text().splitlines() if l.startswith("Cpus_allowed_list:"))
          assert cpus == "0"
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
