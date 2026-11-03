import json
import struct

import pytest

from openpilot.selfdrive.modeld import egpu_worker_progress as progress


def test_snapshot_captures_last_call_and_bounded_thread_state(tmp_path):
  path = tmp_path / 'progress'
  worker = progress.WorkerProgress(path)
  try:
    worker.frame = 17
    worker.mark('output_read')
    snapshot = progress.worker_snapshot(path, 123, tmp_path)
    task = tmp_path / '123' / 'task' / str(snapshot['thread_id'])
    task.mkdir(parents=True)
    (task / 'schedstat').write_text('100 200 3')
    (task / 'wchan').write_text('x' * 3000)
    snapshot = progress.worker_snapshot(path, 123, tmp_path)
    assert snapshot['stage'] == 'output_read' and snapshot['frame'] == 17
    assert snapshot['stage_age_ms'] >= 0 and snapshot['thread_cpu_seconds'] >= 0
    assert snapshot['schedstat'] == '100 200 3' and len(snapshot['wchan']) == 2048
    # A record interrupted while publishing must not be presented as evidence.
    struct.pack_into('<Q', worker.memory, 0, 7)
    assert 'stage' not in progress.worker_snapshot(path, 123, tmp_path)
  finally:
    worker.close()


def test_missing_diagnostics_do_not_prevent_inference(tmp_path):
  path = tmp_path / 'missing' / 'progress'
  worker = progress.WorkerProgress(path)
  worker.mark('model_call')
  worker.close()
  assert progress.worker_snapshot(path, 123, tmp_path) == {'pid': 123}


@pytest.mark.parametrize('error,code', [
  ('RuntimeError: PCIe link not up (LTSSM=0x00)', 'pcie'),
  ('precompiled eGPU worker timed out', 'timeout'),
  ('LIBUSB_ERROR_IO', 'usb'), ('bulk OUT failed', 'usb'),
  ('Input/output error', 'usb'), ('unexpected exit', 'runtime'),
])
def test_classify_observed_failure_without_claiming_hardware_cause(error, code):
  assert progress.runtime_failure_code(error) == code


@pytest.mark.parametrize('boot,sha,worker,expected', [
  ('current', 'a' * 64, {'stage': 'output_read'}, 'pcie'),
  ('previous', 'a' * 64, {}, None), ('current', 'b' * 64, {}, None),
  ('current', 'a' * 64, 'malformed', 'pcie'),
])
def test_only_current_boot_and_model_can_supply_failure_details(tmp_path, monkeypatch, boot, sha, worker, expected):
  monkeypatch.setattr(progress, 'boot_identity', lambda: 'current')
  folder = tmp_path / 'precompiled' / ('a' * 64)
  folder.mkdir(parents=True)
  (folder / 'installed.json').write_text(json.dumps({'pickle': {'sha256': 'a' * 64}}))
  (folder / 'last_failure.json').write_text(json.dumps({
    'boot_id': boot, 'pickle_sha256': sha, 'worker': worker, 'error': 'PCIe link not up',
  }))
  result = progress.current_failure(tmp_path, 'a' * 64)
  assert result.get('error_code') == expected
  assert isinstance(result.get('worker', {}), dict)


def test_legacy_onnx_directory_uses_installed_pickle_identity(tmp_path, monkeypatch):
  monkeypatch.setattr(progress, 'boot_identity', lambda: 'current')
  folder = tmp_path / 'precompiled' / ('a' * 64)
  folder.mkdir(parents=True)
  (folder / 'installed.json').write_text(json.dumps({'pickle': {'sha256': 'b' * 64}}))
  (folder / 'last_failure.json').write_text(json.dumps({
    'boot_id': 'current', 'pickle_sha256': 'b' * 64, 'error': 'PCIe link not up',
  }))
  assert progress.current_failure(tmp_path, 'a' * 64)['error_code'] == 'pcie'


def test_bad_json_is_ignored(tmp_path, monkeypatch):
  monkeypatch.setattr(progress, 'boot_identity', lambda: 'current')
  folder = tmp_path / 'precompiled' / ('a' * 64)
  folder.mkdir(parents=True)
  (folder / 'last_failure.json').write_text('[')
  assert progress.current_failure(tmp_path, 'a' * 64) == {}
  assert progress.current_failure(tmp_path, '../bad') == {}


@pytest.mark.parametrize('contents', ['[]', '{', '{}', '{"error":"' + 'x' * 40000 + '"}'],
                         ids=['list', 'invalid', 'missing', 'oversized'])
def test_saved_failure_is_bounded_and_ignores_malformed_evidence(tmp_path, contents):
  folder = tmp_path / 'precompiled' / ('a' * 64)
  folder.mkdir(parents=True)
  (folder / 'last_failure.json').write_text(contents)
  (folder / 'installed.json').write_text('{}')
  assert progress.saved_failure(tmp_path, 'a' * 64) == {}
  assert progress.saved_failure(tmp_path, '../bad') == {}


def _proc(root, pid, ppid, fds, cmdline='', threads=None):
  process = root / str(pid)
  (process / 'fd').mkdir(parents=True)
  (process / 'stat').write_text(f'{pid} (a b) S {ppid} 1 1 0')
  (process / 'cmdline').write_text(cmdline)
  for fd, target in fds.items():
    (process / 'fd' / str(fd)).symlink_to(target)
  for tid, (state, wchan, syscall) in (threads or {}).items():
    task = process / 'task' / str(tid)
    task.mkdir(parents=True)
    (task / 'stat').write_text(f'{tid} (py (worker)) {state} {ppid} 1')
    (task / 'wchan').write_text(wchan)
    (task / 'syscall').write_text(syscall)
  return process


def test_snapshot_resolves_blocked_fd_threads_and_pipe_owners(tmp_path, monkeypatch):
  monkeypatch.setattr(progress.platform, 'machine', lambda: 'aarch64')
  proc, media = tmp_path / 'proc', tmp_path / 'media'
  media.mkdir()
  _proc(proc, 123, 50, {0: 'pipe:[700]', 1: 'pipe:[701]', 5: '/dev/null'}, threads={
    123: ('S', 'pipe_wait', '63 0x0 0x7fd0 0x1 0x0 0x0 0x0 0x7ff0 0x7fa0'),
    124: ('S', 'futex_wait_queue', '98 0x7f00aa 0x80 0x0 0x0 0x0 0x0 0x7ff0 0x7fa0'),
    125: ('R', '0', 'running')})
  _proc(proc, 'self', 1, {9: 'pipe:[700]', 10: 'socket:[5]'}, 'modeld\0--x')
  _proc(proc, 200, 123, {3: 'pipe:[701]'}, 'child\0arg')
  _proc(proc, 300, 1, {4: 'pipe:[700]'}, 'unrelated')
  snapshot = progress.worker_snapshot(tmp_path / 'missing', 123, proc, media)
  assert [t['tid'] for t in snapshot['threads']] == [123, 124, 125]
  main, futex, running = snapshot['threads']
  assert main == {'tid': 123, 'state': 'S', 'wchan': 'pipe_wait', 'syscall_nr': 63, 'syscall_args': ['0x0', '0x7fd0'],
                  'syscall_name': 'read', 'blocked_fd': {'fd': 0, 'target': 'pipe:[700]'}}
  assert futex['syscall_name'] == 'futex' and 'blocked_fd' not in futex  # a pointer, not an fd
  assert running['state'] == 'R' and running['syscall_nr'] == 'running'
  assert snapshot['blocked_fd'] == {'fd': 0, 'target': 'pipe:[700]', 'tid': 123}
  assert snapshot['fds'] == {'0': 'pipe:[700]', '1': 'pipe:[701]', '5': '/dev/null'}
  assert snapshot['pipe_peers'] == [{'pid': 'self', 'cmdline': 'modeld --x', 'fd': 9, 'target': 'pipe:[700]'},
                                    {'pid': 200, 'cmdline': 'child arg', 'fd': 3, 'target': 'pipe:[701]'}]
  saved = list((media / 'egpu_timeout').glob('*.json'))
  assert len(saved) == 1 and saved[0].name.endswith('-0.json')
  assert json.loads(saved[0].read_text()) == snapshot


def test_snapshot_files_rotate_and_are_skipped_off_device(tmp_path):
  media = tmp_path / 'media'
  folder = media / 'egpu_timeout'
  folder.mkdir(parents=True)
  for index in range(12):
    (folder / f'{1000 + index}-{index}.json').write_text('{}')
  (folder / 'worker_stack.txt').write_text('kept')
  progress.save_snapshot({'pid': 1, 'frame': 10500}, media)
  names = sorted(p.name for p in folder.glob('*.json'))
  assert len(names) == 10 and any(name.endswith('-10500.json') for name in names)
  assert '1000-0.json' not in names and (folder / 'worker_stack.txt').exists()
  progress.save_snapshot({'pid': 1}, tmp_path / 'not-a-device')
  assert not (tmp_path / 'not-a-device').exists()
  assert progress.request_stack_dump(1, 'x', tmp_path / 'not-a-device') is None
  assert progress.register_stack_dump(tmp_path / 'not-a-device') is None
