"""Small shared progress record: no per-frame files, GPU probes or network I/O."""
import json
import mmap
import os
from pathlib import Path
import platform
import signal
import struct
import sys
import threading
import time

STAGES = ('starting', 'verify_model', 'load_model', 'prepare_runtime', 'bind_buffers', 'idle',
          'input_upload', 'warp', 'model_call', 'output_read', 'publish', 'local_prepare')
RECORD = struct.Struct('<QQQddQ')  # generation, stage, frame, monotonic, CPU, thread id
# Timeout evidence survives a reboot only on the userdata partition (/var is tmpfs).
MEDIA_ROOT = Path('/data/media')
TIMEOUT_DIR = 'egpu_timeout'
STACK_FILE = 'worker_stack.txt'
STACK_LIMIT = 2 * 1024 * 1024
STACK_TAIL = 8 * 1024
STACK_WAIT_POLLS, STACK_POLL_SECONDS = 3, .05  # at most 150 ms before the worker is killed
SNAPSHOT_KEEP = 10
SNAPSHOT_BUDGET = .05
# Threads are capped so last_failure.json stays within its 32 KiB reader limit.
MAX_THREADS, MAX_FDS, MAX_PROCESSES, MAX_PEERS = 32, 64, 200, 16
# Syscalls whose first argument may be the file descriptor a stuck thread waits on.
# poll/ppoll/futex take a pointer: such a value is only resolved if it is a plausible fd.
BLOCKING_SYSCALLS = {
  'aarch64': {63: 'read', 64: 'write', 65: 'readv', 66: 'writev', 67: 'pread64', 68: 'pwrite64',
              69: 'preadv', 70: 'pwritev', 72: 'pselect6', 73: 'ppoll', 22: 'epoll_pwait', 98: 'futex',
              29: 'ioctl'},
  'x86_64': {0: 'read', 1: 'write', 17: 'pread64', 18: 'pwrite64', 19: 'readv', 20: 'writev', 7: 'poll',
             271: 'ppoll', 232: 'epoll_wait', 281: 'epoll_pwait', 202: 'futex', 16: 'ioctl'},
}


class WorkerProgress:
  def __init__(self, path: Path):
    self.frame = self.generation = 0
    self.current = ('starting', 0, time.monotonic())
    self.memory = None
    try:
      with path.open('w+b') as f:
        f.truncate(RECORD.size)
        self.memory = mmap.mmap(f.fileno(), RECORD.size)
      self.mark('starting')
    except OSError:
      pass  # optional diagnostics must not prevent inference

  def mark(self, stage: str):
    self.current = (stage, self.frame, time.monotonic())
    if self.memory is None:
      return
    self.generation += 2
    struct.pack_into('<Q', self.memory, 0, self.generation - 1)
    RECORD.pack_into(self.memory, 0, self.generation - 1, STAGES.index(stage), self.frame,
                     self.current[2], time.thread_time(), threading.get_native_id())
    struct.pack_into('<Q', self.memory, 0, self.generation)

  def close(self):
    if self.memory is not None:
      self.memory.close()
      self.memory = None


def _read(path: Path, limit: int = 2048) -> str:
  with path.open() as f:
    return f.read(limit).strip()


def _fd_links(directory: Path, limit: int, deadline: float) -> dict:
  links = {}
  try:
    names = sorted((n for n in os.listdir(directory) if n.isdigit()), key=int)[:limit]
  except OSError:
    return links
  for name in names:
    if time.perf_counter() > deadline:
      break
    try:
      links[name] = os.readlink(directory / name)
    except OSError:
      pass
  return links


def _thread_state(task: Path, syscalls: dict, fd_root: Path) -> dict:
  entry = {'tid': int(task.name)}
  try:
    entry['state'] = _read(task / 'stat').rsplit(')', 1)[1].split()[0]
  except (OSError, IndexError):
    pass
  try:
    entry['wchan'] = _read(task / 'wchan', 128)
  except OSError:
    pass
  try:
    fields = _read(task / 'syscall', 512).split()
    if fields and fields[0].lstrip('-').isdigit():
      entry['syscall_nr'] = int(fields[0])
      entry['syscall_args'] = fields[1:3]
      name = syscalls.get(entry['syscall_nr'])
      if name is not None:
        entry['syscall_name'] = name
        fd = int(fields[1], 16) if len(fields) > 1 else -1
        if 0 <= fd < 65536:
          try:
            entry['blocked_fd'] = {'fd': fd, 'target': os.readlink(fd_root / str(fd))}
          except OSError:
            pass
    elif fields:
      entry['syscall_nr'] = fields[0]  # 'running'
  except (OSError, ValueError):
    pass
  return entry


def _pipe_peers(pid: int, pipes: set, proc_root: Path, deadline: float) -> list:
  """Other holders of the worker's pipes: this process (modeld) and the worker's children."""
  candidates = []
  try:
    candidates.append((int(os.readlink(proc_root / 'self')), proc_root / 'self'))
  except (OSError, ValueError):
    if (proc_root / 'self').is_dir():
      candidates.append(('self', proc_root / 'self'))
  try:
    scanned = 0
    for name in os.listdir(proc_root):
      if not name.isdigit() or int(name) == pid:
        continue
      scanned += 1
      if scanned > MAX_PROCESSES or time.perf_counter() > deadline:
        break
      try:
        if int(_read(proc_root / name / 'stat').rsplit(')', 1)[1].split()[1]) == pid:
          candidates.append((int(name), proc_root / name))
      except (OSError, IndexError, ValueError):
        pass
  except OSError:
    pass
  peers = []
  for peer_pid, root in candidates:
    for fd, target in _fd_links(root / 'fd', 1024, deadline).items():
      if target in pipes and len(peers) < MAX_PEERS:
        try:
          cmdline = _read(root / 'cmdline', 256).replace('\0', ' ').strip()[:80]
        except OSError:
          cmdline = ''
        peers.append({'pid': peer_pid, 'cmdline': cmdline, 'fd': int(fd), 'target': target})
  return peers


def _extend_snapshot(result: dict, pid: int, proc_root: Path, deadline: float) -> None:
  process = proc_root / str(pid)
  syscalls = BLOCKING_SYSCALLS.get(platform.machine(), {})
  main = result.get('thread_id', pid)
  try:
    tids = sorted((int(n) for n in os.listdir(process / 'task') if n.isdigit()), key=lambda t: (t != main, t))
  except OSError:
    tids = []
  threads = []
  for tid in tids[:MAX_THREADS]:
    if time.perf_counter() > deadline:
      break
    threads.append(_thread_state(process / 'task' / str(tid), syscalls, process / 'fd'))
  blocked = [t for t in threads if 'blocked_fd' in t]
  if threads:
    result['threads'] = threads
  if blocked:
    # The progress thread (main) is listed first, so its fd wins when several threads wait.
    result['blocked_fd'] = {**blocked[0]['blocked_fd'], 'tid': blocked[0]['tid']}
  fds = _fd_links(process / 'fd', MAX_FDS, deadline)
  if fds:
    result['fds'] = fds
  pipes = {target for target in fds.values() if target.startswith('pipe:[')}
  pipes |= {t['blocked_fd']['target'] for t in blocked if t['blocked_fd']['target'].startswith('pipe:[')}
  if pipes:
    peers = _pipe_peers(pid, pipes, proc_root, deadline)
    if peers:
      result['pipe_peers'] = peers
  if time.perf_counter() > deadline:
    result['snapshot_truncated'] = True


def save_snapshot(result: dict, media_root: Path = MEDIA_ROOT) -> None:
  """Keep the newest SNAPSHOT_KEEP snapshots on persistent device storage."""
  try:
    if not media_root.is_dir():
      return  # not a device
    directory = media_root / TIMEOUT_DIR
    directory.mkdir(exist_ok=True)
    name = f"{int(time.time())}-{result.get('frame', 0)}.json"  # noqa: TID251 - correlate with boot logs
    (directory / name).write_text(json.dumps(result, indent=1))
    saved = sorted(directory.glob('*-*.json'), key=lambda p: (p.stat().st_mtime, p.name))
    for old in saved[:-SNAPSHOT_KEEP]:
      old.unlink(missing_ok=True)
  except (OSError, TypeError, ValueError):
    pass


def stack_dump_path(media_root: Path = MEDIA_ROOT) -> Path:
  return media_root / TIMEOUT_DIR / STACK_FILE


def register_stack_dump(media_root: Path = MEDIA_ROOT):
  """Worker side: SIGUSR1 appends every thread's Python stack. Returns the open file or None."""
  if sys.platform != 'linux' or not media_root.is_dir():
    return None
  import faulthandler
  try:
    path = stack_dump_path(media_root)
    path.parent.mkdir(exist_ok=True)
    large = path.exists() and path.stat().st_size > STACK_LIMIT
    f = path.open('w' if large else 'a')
    faulthandler.register(signal.SIGUSR1, file=f, all_threads=True, chain=False)
    return f
  except (OSError, ValueError, AttributeError, RuntimeError):
    return None


def request_stack_dump(pid: int, header: str, media_root: Path = MEDIA_ROOT):
  """Runner side, once a timeout is final and before the kill: adds at most ~150 ms."""
  if sys.platform != 'linux' or not media_root.is_dir():
    return None
  path = stack_dump_path(media_root)
  try:
    path.parent.mkdir(exist_ok=True)
    with path.open('a') as f:
      header_at = f.tell()
      f.write(f'=== {time.strftime("%Y-%m-%dT%H:%M:%S%z")} {header} ===\n')
    start = path.stat().st_size
    os.kill(pid, signal.SIGUSR1)
    size = start
    for _ in range(STACK_WAIT_POLLS):
      time.sleep(STACK_POLL_SECONDS)
      size = path.stat().st_size
      if size > start:
        break
    with path.open('rb') as f:
      f.seek(max(header_at, size - STACK_TAIL))  # this header and the dump after it
      text = f.read(STACK_TAIL).decode('utf-8', errors='replace')
    return text if size > start else text + '[no Python stack within 150 ms]\n'
  except (OSError, ValueError):
    return None


def worker_snapshot(path: Path, pid: int, proc_root: Path = Path('/proc'), media_root: Path = MEDIA_ROOT) -> dict:
  """Best-effort bounded reads before termination, without waiting for a stuck worker."""
  deadline = time.perf_counter() + SNAPSHOT_BUDGET
  result = {'pid': pid}
  try:
    with path.open('rb') as f:
      first = f.read(RECORD.size)
      f.seek(0)
      second = f.read(RECORD.size)
    if first == second and len(first) == RECORD.size:
      generation, stage, frame, mono, cpu, tid = RECORD.unpack(first)
      age = time.monotonic() - mono
      if generation and generation % 2 == 0 and stage < len(STAGES) and 0 <= age < 86400:
        result.update(stage=STAGES[stage], frame=frame, stage_age_ms=round(age * 1000, 3),
                      thread_cpu_seconds=cpu, thread_id=tid)
  except (OSError, ValueError, struct.error):
    pass
  task = proc_root / str(pid) / 'task' / str(result.get('thread_id', pid))
  for name in ('stat', 'schedstat', 'wchan'):
    try:
      with (task / name).open() as f:
        result[name] = f.read(2048).strip()
    except OSError:
      pass
  try:
    _extend_snapshot(result, pid, proc_root, deadline)
  except Exception:
    pass  # optional evidence; the fallback itself must still happen
  save_snapshot(result, media_root)
  return result


def runtime_failure_code(error: str) -> str:
  text = error.lower()
  if 'pcie link not up' in text:
    return 'pcie'
  if 'precompiled egpu worker timed out' in text:
    return 'timeout'
  if any(marker in text for marker in ('libusb', 'bulk out', 'bulk in', 'input/output error')):
    return 'usb'
  return 'runtime'


def boot_identity() -> str:
  try:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  except OSError:
    return ''


def current_failure(cache: Path, sha: str) -> dict:
  """Only use current-boot evidence for a currently failed runtime badge."""
  if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
    return {}
  try:
    root = cache / 'precompiled' / sha
    with (root / 'last_failure.json').open() as f:
      value = json.loads(f.read(32768))
    with (root / 'installed.json').open() as f:
      installed = json.loads(f.read(32768))
    boot_id = boot_identity()
    # Legacy catalogs use an ONNX source hash for the directory, unlike generic PKLs.
    if not boot_id or value.get('boot_id') != boot_id or value.get('pickle_sha256') != installed['pickle']['sha256']:
      return {}
    worker = value.get('worker')
    return {'error_code': runtime_failure_code(str(value.get('error', ''))),
            'worker': worker if isinstance(worker, dict) else {}, 'phase': value.get('phase')}
  except (OSError, ValueError, TypeError, AttributeError, KeyError):
    return {}


def saved_failure(cache: Path, sha: str) -> dict:
  """Export matching historical failure evidence, without unrelated device data."""
  if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
    return {}
  try:
    root = cache / 'precompiled' / sha
    with (root / 'last_failure.json').open() as f:
      value = json.loads(f.read(32768))
    with (root / 'installed.json').open() as f:
      installed = json.loads(f.read(32768))
    if value.get('pickle_sha256') != installed['pickle']['sha256'] or not isinstance(value.get('error'), str):
      return {}
    return {key: value[key] for key in ('time', 'phase', 'rejected', 'classification', 'pickle_sha256', 'error', 'boot_id', 'worker') if key in value}
  except (OSError, ValueError, TypeError, AttributeError, KeyError):
    return {}
