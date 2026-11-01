import asyncio
import os
import subprocess
import sys
import threading
import time

import pytest

from openpilot.common import async_process, repo_update


@pytest.fixture
def repo(tmp_path, monkeypatch):
  monkeypatch.setattr(repo_update, "LOCK_PATH", str(tmp_path / "operation.lock"))
  subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
  monkeypatch.setattr(repo_update, "git_process_running", lambda: False)
  return tmp_path


def old_lock(repo):
  path = repo / ".git/index.lock"
  path.write_bytes(b"unfinished index")
  old = time.time() - 600  # noqa: TID251 -- set a filesystem mtime
  os.utime(path, (old, old))
  return path


def test_orphaned_index_is_recovered_without_touching_index_or_other_locks(repo):
  lock = old_lock(repo)
  index = repo / ".git/index"
  index.write_bytes(b"existing index")
  other = repo / ".git/config.lock"
  other.write_bytes(b"leave alone")
  with repo_update.repo_lock():
    assert repo_update.recover_stale_index_lock(str(repo))
  assert not lock.exists()
  assert index.read_bytes() == b"existing index"
  assert other.read_bytes() == b"leave alone"


def test_recent_index_lock_is_never_removed(repo):
  lock = old_lock(repo)
  os.utime(lock, None)
  with repo_update.repo_lock(), pytest.raises(repo_update.RepoBusyError):
    repo_update.recover_stale_index_lock(str(repo))
  assert lock.exists()


def test_active_git_keeps_even_an_old_lock(repo, monkeypatch):
  lock = old_lock(repo)
  monkeypatch.setattr(repo_update, "git_process_running", lambda: True)
  with repo_update.repo_lock(), pytest.raises(repo_update.RepoBusyError):
    repo_update.recover_stale_index_lock(str(repo))
  assert lock.exists()


def test_lock_changed_during_inspection_is_preserved(repo, monkeypatch):
  lock = old_lock(repo)
  monkeypatch.setattr(repo_update.time, "sleep", lambda _: lock.write_bytes(b"another writer"))
  with repo_update.repo_lock(), pytest.raises(repo_update.RepoBusyError):
    repo_update.recover_stale_index_lock(str(repo))
  assert lock.read_bytes() == b"another writer"


def test_separate_process_cannot_enter_a_held_checkout_lock(repo):
  code = "from openpilot.common.repo_update import repo_lock;\nwith repo_lock(): print('entered')"
  env = {**os.environ, "CARROT_REPO_LOCK_PATH": repo_update.LOCK_PATH}
  with repo_update.repo_lock():
    blocked = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
  allowed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
  assert blocked.returncode != 0
  assert "RepoBusyError" in blocked.stderr
  assert allowed.returncode == 0, allowed.stderr


def test_same_event_loop_tasks_cannot_share_a_transaction(repo):
  async def attempt():
    with pytest.raises(repo_update.RepoBusyError), repo_update.repo_lock():
      pytest.fail("concurrent task acquired the transaction lock")

  async def scenario():
    with repo_update.repo_lock():
      await asyncio.create_task(attempt())
  asyncio.run(scenario())


def test_cancelled_threaded_repair_keeps_lock_until_thread_finishes(repo):
  started, finish = threading.Event(), threading.Event()

  def repair():
    started.set()
    assert finish.wait(5)

  async def operation():
    with repo_update.repo_lock():
      await async_process.run_locked_thread(repair)

  async def scenario():
    task = asyncio.create_task(operation())
    try:
      assert await asyncio.to_thread(started.wait, 5)
      task.cancel()
      await asyncio.sleep(0)
      with pytest.raises(repo_update.RepoBusyError), repo_update.repo_lock():
        pytest.fail("cancelled operation released its lock while the thread was still writing")
    finally:
      finish.set()
      with pytest.raises(asyncio.CancelledError):
        await task
    with repo_update.repo_lock():
      pass
  asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_reap_the_child(repo, monkeypatch, cancel):
  children = []
  created = None
  create = asyncio.create_subprocess_exec

  async def record(*args, **kwargs):
    child = await create(*args, **kwargs)
    children.append(child)
    created.set()
    return child

  monkeypatch.setattr(asyncio, "create_subprocess_exec", record)

  async def scenario():
    nonlocal created
    created = asyncio.Event()
    task = asyncio.create_task(async_process.run_process(
      [sys.executable, "-c", "import time; time.sleep(60)"], timeout=None if cancel else 0.2,
    ))
    if cancel:
      await created.wait()
      task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
      await task
    assert children[0].returncode is not None
  asyncio.run(scenario())


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups and flock inheritance")
def test_timeout_kills_grandchild_and_releases_inherited_lock(repo):
  # The grandchild keeps stdout and the lock FD open and ignores SIGTERM.
  script = "import os,signal,time; pid=os.fork(); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

  async def scenario():
    with repo_update.repo_lock(), pytest.raises(asyncio.TimeoutError):
      await async_process.run_process([sys.executable, "-c", script], timeout=0.2)
    with repo_update.repo_lock():
      pass
  asyncio.run(scenario())


@pytest.mark.skipif(os.name == "nt", reason="launcher passes a POSIX file descriptor")
def test_manager_releases_inherited_boot_lock(repo, monkeypatch):
  fd = os.open(repo_update.LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
  repo_update._flock(fd)
  monkeypatch.setenv("CARROT_BOOT_LOCK_FD", str(fd))
  repo_update.release_boot_lock()
  assert "CARROT_BOOT_LOCK_FD" not in os.environ
  with repo_update.repo_lock():
    pass


@pytest.mark.skipif(os.name == "nt", reason="Linux /proc process inspection")
def test_real_git_process_prevents_recovery(tmp_path, monkeypatch):
  monkeypatch.setattr(repo_update, "LOCK_PATH", str(tmp_path / "operation.lock"))
  subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
  lock = old_lock(tmp_path)
  child = subprocess.Popen(["git", "hash-object", "--stdin"], cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  try:
    with repo_update.repo_lock(), pytest.raises(repo_update.RepoBusyError):
      repo_update.recover_stale_index_lock(str(tmp_path))
    assert lock.exists()
  finally:
    child.communicate(b"")


def test_busy_error_names_holder_stage_and_elapsed(repo, monkeypatch):
  import json
  with repo_update.repo_lock(owner="auto_update", detail="check"):
    repo_update.set_lock_detail("fetch")
    with open(repo_update.owner_path(), encoding="utf-8") as f:
      record = json.load(f)
    assert record["owner"] == "auto_update" and record["pid"] == os.getpid() and record["detail"] == "fetch"
    monkeypatch.setattr(repo_update.time, "time", lambda: record["started_at"] + 42)
    with pytest.raises(repo_update.RepoBusyError) as busy, repo_update.repo_lock(owner="git_pull_tool"):
      pass
  err = busy.value
  assert err.owner == "auto_update" and err.detail == "fetch" and err.elapsed == 42 and not err.presumed
  assert "Automatic update is running git pull (started 42 s ago, fetch stage)" in str(err)
  assert err.fields()["busy_since"] == record["started_at"]
  assert not os.path.exists(repo_update.owner_path())


def test_stale_owner_file_without_flock_is_not_busy(repo):
  import json
  with open(repo_update.owner_path(), "w", encoding="utf-8") as f:
    json.dump({"owner": "auto_update", "pid": os.getpid(), "started_at": 1, "detail": "fetch"}, f)
  with repo_update.repo_lock(owner="git_pull_tool"):
    assert json.loads(open(repo_update.owner_path(), encoding="utf-8").read())["owner"] == "git_pull_tool"
  assert not os.path.exists(repo_update.owner_path())


@pytest.mark.skipif(os.name == "nt", reason="dead-pid probe is POSIX only")
def test_held_lock_with_dead_owner_is_presumed_startup(repo):
  import json
  fd = os.open(repo_update.LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
  child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
  with open(repo_update.owner_path(), "w", encoding="utf-8") as f:
    json.dump({"owner": "auto_update", "pid": int(child.stdout), "started_at": 1, "detail": "fetch"}, f)
  try:
    repo_update._flock(fd)
    with pytest.raises(repo_update.RepoBusyError) as busy, repo_update.repo_lock():
      pass
  finally:
    os.close(fd)
  assert busy.value.owner == "startup" and busy.value.presumed
