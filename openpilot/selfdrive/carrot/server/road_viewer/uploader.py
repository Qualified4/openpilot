"""Protocol adapter for the existing dashcam upload Job and file catalog."""
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import time

import aiohttp
from aiohttp import web

from ..features.dashcam import upload_jobs as jobs
from ..features.dashcam.catalog import segment_file_summary, segment_is_complete
from ..features.dashcam.paths import file_size_label, route_name, segment_dir, segment_index
from . import client, config

BLOCK_SIZE = 256 * 1024  # Local read size for hashing.
PARALLEL_FILES = 3  # Different files only: the server appends each file strictly in order.
ROUTE = re.compile(r'(?:[0-9a-fA-F]{8}--[0-9a-fA-F]{10}|\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})')


def discover(segments, options=None):
  options = options or {}
  if not 1 <= len(segments) <= 50:
    raise client.Error('invalid_segments')
  summaries = []
  for segment in segments:
    if not ROUTE.fullmatch(route_name(segment)) or not 0 <= segment_index(segment) <= 999999:
      raise client.Error('invalid_segment')
    directory = segment_dir(segment)  # Existing path validation and discovery.
    if os.path.islink(directory) or not segment_is_complete(segment):
      raise client.Error('segment_incomplete')
    files = segment_file_summary(directory)
    if any(item['name'] not in ('rlog.zst', 'qcamera.ts') for item in files):
      raise client.Error('file_not_allowed')
    for option, name in (('include_front', 'fcamera.hevc'), ('include_wide', 'ecamera.hevc')):
      if not options.get(option):
        continue
      try:
        info = os.stat(os.path.join(directory, name), follow_symlinks=False)
      except FileNotFoundError:
        continue
      if not stat.S_ISREG(info.st_mode):
        raise client.Error('file_not_allowed')
      if info.st_size > 0:
        files.append({'kind': name.split('.')[0], 'name': name, 'size': info.st_size, 'sizeLabel': file_size_label(info.st_size)})
    total = sum(item['size'] for item in files)
    summaries.append({'segment': segment, 'route': route_name(segment), 'segmentIndex': segment_index(segment),
                      'files': files, 'totalSize': total, 'totalSizeLabel': file_size_label(total)})
  return summaries


def fingerprint(source):
  value = os.fstat(source.fileno())
  if not stat.S_ISREG(value.st_mode):
    raise client.Error('local_file_error')
  return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def open_source(item):
  if not segment_is_complete(item['segment']):
    raise client.Error('segment_incomplete')
  directory = os.open(segment_dir(item['segment']), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
  try:
    fd = os.open(item['name'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
  finally:
    os.close(directory)
  source = os.fdopen(fd, 'rb')
  try:
    current = fingerprint(source)
    if current[2] != item['size'] or ('fingerprint' in item and current != item['fingerprint']):
      raise client.Error('file_changed')
    return source
  except BaseException:
    source.close()
    raise


def read_payload(source, offset, size, digest):
  source.seek(offset)
  payload = source.read(size)
  if digest is not None:
    digest.update(payload)
  return payload


def next_chunk_size(maximum, size, seconds):
  # Aim for four seconds of useful work per request, bounded by the server's
  # negotiated cap. Fast links amortize RTT; slow links keep requests small.
  return min(maximum, max(256 * 1024, int(size * 4 / max(.01, seconds) / 65536) * 65536))


def progress(job, *, phase, message, current_bytes=0, **kwargs):
  checkpoint = job.get('_road_viewer', {})
  batches = checkpoint.get('batches', [])
  completed = [batch for batch in batches if batch and batch.get('complete')]
  done_count = sum(len(batch['summaries']) for batch in completed)
  done_bytes = sum(sum(item['size'] for item in batch['files']) for batch in completed)
  total_bytes = sum(summary['totalSize'] for summary in checkpoint.get('summaries', []))
  current_bytes += done_bytes
  jobs.progress(job, phase=phase, message=message, current=done_count,
                bytes_current=current_bytes, bytes_total=total_bytes,
                percent=97 * current_bytes / max(1, total_bytes) + 2 * done_count / len(job['segments']), **kwargs)


async def prepare(job, segments=None, *, hash_files=True, report=True):
  summaries = await asyncio.to_thread(discover, segments or job['segments'], job.get('road_viewer_options'))
  files = []
  manifest = {'batch_id': secrets.token_urlsafe(24), 'segments': []}
  for summary in summaries:
    entry = {'route': summary['route'], 'segment': summary['segmentIndex'], 'files': []}
    for file in summary['files']:
      jobs.ensure_not_canceled(job)
      item = dict(file, segment=summary['segment'])
      digest = hashlib.sha256()
      with await asyncio.to_thread(open_source, item) as source:
        item['fingerprint'] = fingerprint(source)
        read = 0
        while hash_files:
          jobs.ensure_not_canceled(job)
          block = await asyncio.to_thread(source.read, BLOCK_SIZE)
          if not block:
            break
          digest.update(block)
          read += len(block)
          if report:
            progress(job, phase='preparing', message=f"{item['segment']} / {item['name']}",
                     phase_current=read, phase_total=item['size'], bytes_per_second=0)
        if fingerprint(source) != item['fingerprint'] or (hash_files and read != item['size']):
          raise client.Error('file_changed')
      files.append(item)
      entry['files'].append({'kind': item['name'], 'size': item['size'], 'sha256': digest.hexdigest() if hash_files else None})
    manifest['segments'].append(entry)
  return {'manifest': manifest, 'files': files, 'summaries': summaries, 'streaming': not hash_files}


def offsets(response, files):
  values = response.get('files')
  if not isinstance(values, list) or len(values) != len(files):
    raise client.Error('invalid_response')
  received = []
  for i, (remote, local) in enumerate(zip(values, files, strict=True)):
    if (not isinstance(remote, dict) or remote.get('index') != i or remote.get('size') != local['size']
        or remote.get('name') != f"{route_name(local['segment'])}--{segment_index(local['segment'])}--{local['name']}"
        or type(remote.get('received')) is not int or not 0 <= remote['received'] <= local['size']):
      raise client.Error('invalid_response')
    received.append(remote['received'])
  return received


RETRIES = 7  # Attempts per request on transient errors: rides out short Wi-Fi/LTE drops.


def backoff(attempt):
  return min(30.0, 0.5 * 2 ** attempt)  # 0.5, 1, 2, 4, 8, 16, 30 s


async def retry(job, operation):
  for attempt in range(RETRIES):
    jobs.ensure_not_canceled(job)
    try:
      return await operation()
    except client.Error as exc:
      if exc.code not in client.TRANSIENT or attempt == RETRIES - 1:
        raise
      jobs.progress(job, message=exc.code)
      await asyncio.sleep(backoff(attempt))
  raise AssertionError('unreachable')


async def transfer(job, http, credentials, checkpoint, defer_finish=False):
  """Send the segment's files. With defer_finish the registration is returned as a coroutine function, so the
  caller can start the next segment while the server registers this one."""
  files = checkpoint['files']
  # Re-check every source before reattaching: stale files must never silently
  # replace a manifest whose earlier chunks already exist on the server.
  for item in files:
    with await asyncio.to_thread(open_source, item):
      pass
  session = await retry(job, lambda: client.begin(http, credentials, checkpoint['manifest']))
  config.last_error = ''
  if session.get('state') == 'completed':
    validate_result(session.get('result'))
    return None
  if checkpoint.get('streaming') and session.get('checksum_mode') != client.STREAM_CHECKSUM:
    raise client.Error('invalid_response')
  checkpoint['session'] = session
  chunk_size = session.get('chunk_size')
  if type(chunk_size) is not int or not 1 <= chunk_size <= client.MAX_CHUNK_SIZE:
    raise client.Error('invalid_response')
  received = offsets(session, files)
  checkpoint['received'] = received
  started, original = time.monotonic(), sum(received)
  finished = set()
  checksums = [None] * len(files)

  def report(item):
    current = sum(received)
    progress(job, phase='uploading', message=f"{item['segment']} / {item['name']}", current_bytes=current,
             phase_current=len(finished), phase_total=len(files),
             bytes_per_second=int((current - original) / max(0.01, time.monotonic() - started)))

  async def send(index):
    item = files[index]
    with await asyncio.to_thread(open_source, item) as source:
      failures = 0
      send_size = min(chunk_size, 1024 * 1024)
      digest, hashed = hashlib.sha256(), 0

      async def align_digest(offset):
        nonlocal digest, hashed
        if not checkpoint.get('streaming') or hashed == offset:
          return
        # Resume/reconcile only: ordinary uploads hash each transmitted byte once.
        digest, hashed = hashlib.sha256(), 0
        await asyncio.to_thread(source.seek, 0)
        while hashed < offset:
          jobs.ensure_not_canceled(job)
          block = await asyncio.to_thread(source.read, min(BLOCK_SIZE, offset - hashed))
          if not block:
            raise client.Error('file_changed')
          digest.update(block)
          hashed += len(block)

      while received[index] < item['size']:
        jobs.ensure_not_canceled(job)
        if fingerprint(source) != item['fingerprint']:
          raise client.Error('file_changed')
        offset = received[index]
        await align_digest(offset)
        previous_digest = digest.copy()
        payload = await asyncio.to_thread(read_payload, source, offset, min(send_size, item['size'] - offset),
                                         digest if checkpoint.get('streaming') else None)
        if not payload:
          raise client.Error('file_changed')
        if checkpoint.get('streaming'):
          hashed += len(payload)
        try:
          requested_at = time.monotonic()
          response = await client.session_request(http, credentials['url'], session, 'PUT',
                                                   f'/files/{index}?offset={offset}', payload)
          if response.get('received') != offset + len(payload):
            raise client.Error('invalid_response')
          received[index] = response['received']
          send_size = next_chunk_size(chunk_size, len(payload), time.monotonic() - requested_at)
          failures = 0
        except client.Error as exc:
          if exc.code not in client.TRANSIENT or failures >= RETRIES - 1:
            raise
          failures += 1
          send_size = max(min(chunk_size, 256 * 1024), send_size // 2)
          await asyncio.sleep(backoff(failures - 1))
          status = await retry(job, lambda: client.session_request(http, credentials['url'], session))
          # Response may have been lost AFTER a successful write. Only this file: other
          # files have their own requests in flight and track their own acknowledged offsets.
          received[index] = offsets(status, files)[index]
          if checkpoint.get('streaming') and received[index] == offset:
            digest, hashed = previous_digest, offset
        report(item)
      if checkpoint.get('streaming') and received[index] == item['size'] and item.get('sha256'):
        checksums[index] = item['sha256']
      else:
        await align_digest(item['size'])
        checksums[index] = digest.hexdigest()
      if fingerprint(source) != item['fingerprint'] or not await asyncio.to_thread(segment_is_complete, item['segment']):
        raise client.Error('file_changed')
      if checkpoint.get('streaming'):
        item['sha256'] = checksums[index]
    finished.add(index)
    report(item)

  # Up to PARALLEL_FILES files at once, each one strictly in order. A single request at a
  # time leaves the uplink idle for every round trip and server write.
  queue = iter(range(len(files)))

  async def worker():
    for index in queue:  # Shared iterator: each file is taken by exactly one worker.
      await send(index)

  tasks = [asyncio.create_task(worker()) for _ in range(min(PARALLEL_FILES, len(files)))]
  try:
    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in tasks:
      if task in done and task.exception():
        raise task.exception()
  finally:
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
  jobs.ensure_not_canceled(job)
  body = json.dumps({'sha256': checksums}).encode() if checkpoint.get('streaming') else None

  async def finish():
    result = await retry(job, lambda: client.session_request(http, credentials['url'], session, 'POST', '/finish', body))
    validate_result(result)

  if defer_finish:
    return finish
  progress(job, phase='notifying', message=f"Road Viewer: registering {checkpoint['summaries'][0]['segment']}",
           current_bytes=sum(received), bytes_per_second=0)
  await finish()
  return None


def validate_result(result):
  if not isinstance(result, dict) or any(not isinstance(result.get(key), list) for key in ('logs', 'updated', 'duplicates')):
    raise client.Error('invalid_response')


def segment_results(checkpoint):
  results, received_bytes = [], 0
  for summary in checkpoint.get('summaries', []):
    batch = next((b for b in checkpoint['batches'] if b and any(s['segment'] == summary['segment'] for s in b['summaries'])), None)
    complete = bool(batch and batch.get('complete'))
    received = batch.get('received', []) if batch else []
    selected = [(i, f) for i, f in enumerate(batch['files']) if f['segment'] == summary['segment']] if batch else []
    transferred = bool(selected) and all(i < len(received) and received[i] == f['size'] for i, f in selected)
    received_bytes += sum(f['size'] if complete else received[i] if i < len(received) else 0 for i, f in selected)
    results.append(dict(summary, ok=complete, transferred=complete or transferred))
  return results, received_bytes


async def run_job(job):
  checkpoint = job.get('_road_viewer')
  code, canceled, success = '', False, False
  credentials = None
  active = None
  # Checksums one segment ahead: while segment N is sent, segment N+1 is hashed in the background, so its
  # manifest already carries checksums and the transfer loop only reads and sends. Only the first segment
  # (nothing to overlap with) is hashed while it is sent.
  ahead = {}
  # The previous segment's registration (index, batch, task) runs while the next segment is sent. A segment
  # is marked complete only when its registration succeeded, and before the next one is registered.
  finishing = None

  async def settle():
    nonlocal finishing
    if finishing is None:
      return
    index, batch, task = finishing
    finishing = None
    await task
    complete(index, batch)

  def complete(index, batch):
    batch['complete'] = True
    job['partial_results'], _ = segment_results(checkpoint)
    jobs.append(job, 'Road Viewer: completed ' + ', '.join(s['segment'] for s in batch['summaries']))

  def look_ahead(index):
    following = index + 1
    if following < len(checkpoint['batches']) and checkpoint['batches'][following] is None and following not in ahead:
      ahead[following] = asyncio.create_task(prepare(job, [job['segments'][following]], report=False))

  async with aiohttp.ClientSession() as http:
    try:
      credentials = await asyncio.to_thread(config.load)
      if not credentials:
        raise client.Error('disconnected')
      if checkpoint and checkpoint.get('device_id') != credentials['device_id']:
        raise client.Error('retry_unavailable')
      if checkpoint is None:
        summaries = await asyncio.to_thread(discover, job['segments'], job.get('road_viewer_options'))
        checkpoint = {'device_id': credentials['device_id'], 'summaries': summaries,
                      'batches': [None] * len(summaries), 'streaming': True}
      elif 'batches' not in checkpoint:
        # Keep an already prepared RV1 batch resumable across an in-process update.
        checkpoint = {'device_id': credentials['device_id'], 'summaries': checkpoint['summaries'],
                      'batches': [checkpoint], 'streaming': False}
      job['_road_viewer'] = checkpoint
      for index, batch in enumerate(checkpoint['batches']):
        jobs.ensure_not_canceled(job)
        if batch and batch.get('complete'):
          continue
        active = index
        if batch is None:
          if index in ahead:
            batch = await ahead.pop(index)
          else:
            progress(job, phase='preparing', message=job['segments'][index], bytes_per_second=0)
            batch = await prepare(job, [job['segments'][index]], hash_files=not checkpoint['streaming'])
          checkpoint['batches'][index] = batch
          checkpoint['summaries'][index] = batch['summaries'][0]
        look_ahead(index)
        try:
          finish = await transfer(job, http, credentials, batch, defer_finish=True)
        except client.Error as exc:
          if exc.code not in ('invalid_file', 'invalid_batch') or not batch.get('streaming') or batch.get('session'):
            raise
          # Older servers require hashes in the initial manifest. Probe only once,
          # then retain their existing protocol for the rest of this selection.
          checkpoint['streaming'] = False
          batch = await prepare(job, [job['segments'][index]])
          checkpoint['batches'][index] = batch
          checkpoint['summaries'][index] = batch['summaries'][0]
          finish = await transfer(job, http, credentials, batch, defer_finish=True)
        await settle()  # Registrations stay in order: the previous segment first.
        if finish is None:
          complete(index, batch)  # The server already had it (receipt of a lost response).
        else:
          finishing = (index, batch, asyncio.create_task(finish()))
      if finishing is not None:
        progress(job, phase='notifying', message='Road Viewer: registering ' + job['segments'][finishing[0]], bytes_per_second=0)
      await settle()
      progress(job, phase='notifying', message='Road Viewer: segment complete', bytes_per_second=0)
      success = True
    except jobs.UploadCanceled:
      canceled = True
    except client.Error as exc:
      code = exc.code
    except ValueError:
      code = 'credentials_corrupt'
    except (OSError, web.HTTPException):
      code = 'local_file_error'
    except Exception:
      code = 'upload_failed'  # Never publish raw exceptions, URLs with tokens, or response bodies.
    finally:
      if finishing is not None and not canceled:
        # A later failure must not throw away a fully sent segment: let its registration finish first.
        try:
          await settle()
        except Exception:
          pass
      background = [*ahead.values(), *([finishing[2]] if finishing else [])]
      for task in background:
        task.cancel()
      await asyncio.gather(*background, return_exceptions=True)
      if (canceled or code in ('file_changed', 'segment_incomplete', 'local_file_error')) and checkpoint and active is not None:
        batch = checkpoint['batches'][active]
        if credentials and batch and not batch.get('complete') and batch.get('session'):
          try:
            await client.session_request(http, credentials['url'], batch['session'], 'DELETE')
          except Exception:
            pass  # Offline sessions expire on the server (15 min idle / 2 h absolute).
        if canceled:
          job.pop('_road_viewer', None)
        elif batch and not batch.get('complete'):
          checkpoint['batches'][active] = None
      if canceled:
        job.pop('_road_viewer', None)
  if code:
    config.last_error = code
  results, received_bytes = segment_results(checkpoint) if checkpoint and 'batches' in checkpoint else ([], 0)
  result = {'ok': success, 'target': 'road_viewer', 'uploaded': sum(item['ok'] for item in results),
            'total': len(job['segments']), 'results': results,
            'bytes_received': received_bytes,
            'error': code or None, 'message': code or ('Upload complete' if success else 'Upload canceled'),
            'canceled': canceled}
  if not success and not canceled and job.get('_road_viewer'):
    result['retry_job_id'] = job['id']
  jobs.finish(job, ok=success, result=result, status='canceled' if canceled else None)
