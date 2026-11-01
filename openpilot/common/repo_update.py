"""Cooperative device checkout lock and conservative orphaned-index recovery.

The launcher, web tools and updater share this lock. Never unlink the lock file:
flock ownership, unlike a sentinel file, is released when its owner exits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import stat
import subprocess
import time


LOCK_PATH = os.environ.get("CARROT_REPO_LOCK_PATH", "/tmp/carrot_repo_update.lock")
STALE_INDEX_SECONDS = 60.0
_lock_fd: ContextVar[int | None] = ContextVar("carrot_repo_lock_fd", default=None)
_lock_owner: ContextVar[dict | None] = ContextVar("carrot_repo_lock_owner", default=None)

# The launcher takes the flock in shell without an owner file; a held lock with
# no live owner record is therefore presumed to be startup (bootstrap/build/init).
PRESUMED_OWNER = "startup"
_OWNER_LABELS = {
  "auto_update": "Automatic update is running git pull",
  "git_pull_tool": "Manual git pull is running",
  "git_tool": "Another Git tool is running",
  "build": "A build is running",
  "git_status": "Git status check is running",
  "recovery": "Startup recovery is running",
  PRESUMED_OWNER: "Startup (bootstrap, build or manager init) is probably running",
}


class RepoBusyError(RuntimeError):
  def __init__(self, message: str = "", *, owner: str = "", since: int | None = None, detail: str = "",
               elapsed: int | None = None, presumed: bool = False):
    super().__init__(message)
    self.owner = owner
    self.since = since
    self.detail = detail
    self.elapsed = elapsed
    self.presumed = presumed

  def fields(self) -> dict:
    """Response fields for GIT_BUSY; empty when the holder is not known."""
    if not self.owner:
      return {}
    return {"busy_owner": self.owner, "busy_since": self.since, "busy_detail": self.detail,
            "busy_elapsed": self.elapsed, "busy_presumed": self.presumed}


def owner_path() -> str:
  base = LOCK_PATH[:-len(".lock")] if LOCK_PATH.endswith(".lock") else LOCK_PATH
  return base + ".owner.json"


def _pid_alive(pid) -> bool:
  if not isinstance(pid, int) or pid <= 0:
    return False
  if os.name == "nt":
    return True  # os.kill(pid, 0) is not a probe on Windows
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except OSError:
    return True
  return True


def _write_owner(info: dict) -> None:
  # Diagnostics only: never fail or block the lock because of this file.
  path = owner_path()
  tmp = f"{path}.{os.getpid()}.tmp"
  try:
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(info, f)
    os.replace(tmp, path)
  except OSError:
    pass


def read_lock_owner() -> dict:
  """Only meaningful while the flock is held by someone else (busy path).

  A crashed holder leaves its file behind but its flock is released, so a stale
  file is never consulted unless the lock is actually held; a dead pid falls
  back to the presumed startup owner.
  """
  try:
    with open(owner_path(), encoding="utf-8") as f:
      info = json.load(f)
  except (OSError, ValueError):
    info = None
  if isinstance(info, dict) and info.get("owner") and _pid_alive(info.get("pid")):
    return {"owner": str(info["owner"]), "pid": info.get("pid"), "started_at": info.get("started_at"),
            "detail": str(info.get("detail") or ""), "presumed": False}
  return {"owner": PRESUMED_OWNER, "pid": None, "started_at": None, "detail": "", "presumed": True}


def busy_error() -> RepoBusyError:
  info = read_lock_owner()
  started = info["started_at"]
  elapsed = max(0, int(time.time() - started)) if isinstance(started, (int, float)) else None  # noqa: TID251 -- compared with a wall-clock record from another process
  label = _OWNER_LABELS.get(info["owner"], f"Another Git operation ({info['owner']}) is running")
  parts = []
  if elapsed is not None:
    parts.append(f"started {elapsed} s ago")
  if info["detail"]:
    parts.append(f"{info['detail']} stage")
  message = label + (f" ({', '.join(parts)})" if parts else "") + "; retry when it finishes."
  return RepoBusyError(message, owner=info["owner"], since=int(started) if elapsed is not None else None,
                       detail=info["detail"], elapsed=elapsed, presumed=info["presumed"])


def set_lock_detail(detail: str) -> None:
  """Update the current holder's stage (e.g. fetch/reset/merge)."""
  info = _lock_owner.get()
  if info is None:
    return
  info["detail"] = str(detail or "")
  _write_owner(info)


def _flock(fd: int, *, unlock: bool = False) -> None:
  if os.name == "nt":
    import msvcrt
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
  else:
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


def child_lock_kwargs() -> dict:
  fd = _lock_fd.get()
  return {"pass_fds": (fd,)} if fd is not None and os.name != "nt" else {}


@contextmanager
def repo_lock(owner: str = "", detail: str = ""):
  # Do not make this reentrant: separate asyncio tasks must also exclude each
  # other. Callers take one lock around the complete multi-command transaction.
  fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
  try:
    try:
      _flock(fd)
    except OSError as exc:
      raise busy_error() from exc
    info = {"owner": owner or "unknown", "pid": os.getpid(), "started_at": int(time.time()), "detail": str(detail or "")}  # noqa: TID251 -- read by other processes
    _write_owner(info)
    token = _lock_fd.set(fd)
    owner_token = _lock_owner.set(info)
    try:
      yield
    finally:
      _lock_owner.reset(owner_token)
      _lock_fd.reset(token)
      try:
        os.unlink(owner_path())
      except OSError:
        pass
      # close, rather than LOCK_UN, keeps children protected if the caller dies.
  finally:
    os.close(fd)


def release_boot_lock() -> None:
  """Manager releases the launcher's inherited lock only after initialization."""
  value = os.environ.pop("CARROT_BOOT_LOCK_FD", "")
  if not value:
    return
  fd = int(value)
  if not os.path.samestat(os.fstat(fd), os.stat(LOCK_PATH)):
    raise RuntimeError("Unexpected boot repository lock descriptor")
  _flock(fd, unlock=True)
  os.close(fd)


def git_process_running() -> bool:
  """Fail closed if we cannot inspect processes. Any Git process defers repair."""
  proc_root = Path("/proc")
  if not proc_root.is_dir():
    return True
  try:
    for entry in proc_root.iterdir():
      if not entry.name.isdigit():
        continue
      try:
        name = (entry / "comm").read_text().strip()
      except FileNotFoundError:
        continue  # process exited while scanning
      if name == "git" or name.startswith("git-"):
        return True
    return False
  except OSError:
    return True


def recover_stale_index_lock(repo_dir: str) -> bool:
  """Called under repo_lock, before starting any child Git command.

  Only index.lock is recoverable here. Ref/config locks need explicit diagnosis.
  Age alone is never evidence that an index lock is abandoned.
  """
  result = subprocess.run(
    ["git", "rev-parse", "--path-format=absolute", "--git-path", "index.lock"], cwd=repo_dir,
    capture_output=True, text=True, timeout=10, **child_lock_kwargs(),
  )
  if result.returncode:
    raise RuntimeError(result.stderr.strip() or "Unable to locate Git index lock")
  path = Path(repo_dir) / result.stdout.strip()
  try:
    before = path.lstat()
  except FileNotFoundError:
    return False
  if not stat.S_ISREG(before.st_mode):
    raise RepoBusyError("Git index lock is not a regular file; manual inspection required.")
  if time.time() - before.st_mtime < STALE_INDEX_SECONDS or git_process_running():  # noqa: TID251 -- filesystem mtime is wall-clock time
    raise RepoBusyError("Git index is in use; waiting before retrying.")
  time.sleep(0.1)
  try:
    after = path.lstat()
  except FileNotFoundError:
    return False
  if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
    after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size
  ) or git_process_running():
    raise RepoBusyError("Git index lock changed or Git is running; retry later.")
  path.unlink()
  print("[repo_update] removed abandoned index.lock after checking Git processes", flush=True)
  return True


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", default="/data/openpilot")
  parser.add_argument("command", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  try:
    with repo_lock(owner="recovery", detail="repo_update"):
      recover_stale_index_lock(args.repo)
      if args.command:
        return subprocess.call(args.command, cwd=args.repo, **child_lock_kwargs())
      return 0
  except RepoBusyError as exc:
    print(str(exc), flush=True)
    return 75


if __name__ == "__main__":
  raise SystemExit(main())
