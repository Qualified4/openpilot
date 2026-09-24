"""Protocol adapter for the existing dashcam upload Job and file catalog."""
import asyncio
import hashlib
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

BLOCK_SIZE = 256 * 1024
ROUTE = re.compile(r'(?:[0-9a-fA-F]{8}--[0-9a-fA-F]{10}|\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})')


def discover(segments):
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


async def prepare(job):
  summaries = await asyncio.to_thread(discover, job['segments'])
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
        while True:
          jobs.ensure_not_canceled(job)
          block = await asyncio.to_thread(source.read, BLOCK_SIZE)
          if not block:
            break
          digest.update(block)
          read += len(block)
          jobs.progress(job, phase='preparing', message=f"{item['segment']} / {item['name']}",
                        phase_current=read, phase_total=item['size'])
        if fingerprint(source) != item['fingerprint'] or read != item['size']:
          raise client.Error('file_changed')
      files.append(item)
      entry['files'].append({'kind': item['name'], 'size': item['size'], 'sha256': digest.hexdigest()})
    manifest['segments'].append(entry)
  return {'manifest': manifest, 'files': files, 'summaries': summaries}


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


async def retry(job, operation):
  for attempt in range(3):
    jobs.ensure_not_canceled(job)
    try:
      return await operation()
    except client.Error as exc:
      if exc.code not in client.TRANSIENT or attempt == 2:
        raise
      jobs.progress(job, message=exc.code)
      await asyncio.sleep(0.5 * (attempt + 1))
  raise AssertionError('unreachable')


async def transfer(job, http, credentials, checkpoint):
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
    return
  checkpoint['session'] = session
  chunk_size = session.get('chunk_size')
  if type(chunk_size) is not int or not 1 <= chunk_size <= BLOCK_SIZE:
    raise client.Error('invalid_response')
  received = offsets(session, files)
  checkpoint['received'] = received
  total = sum(item['size'] for item in files)
  started, original = time.monotonic(), sum(received)
  # One reader/request at a time both onroad and offroad; no new worker pool.
  for index, item in enumerate(files):
    with await asyncio.to_thread(open_source, item) as source:
      failures = 0
      while received[index] < item['size']:
        jobs.ensure_not_canceled(job)
        if fingerprint(source) != item['fingerprint']:
          raise client.Error('file_changed')
        offset = received[index]
        await asyncio.to_thread(source.seek, offset)
        payload = await asyncio.to_thread(source.read, min(chunk_size, item['size'] - offset))
        if not payload:
          raise client.Error('file_changed')
        try:
          response = await client.session_request(http, credentials['url'], session, 'PUT',
                                                   f'/files/{index}?offset={offset}', payload)
          if response.get('received') != offset + len(payload):
            raise client.Error('invalid_response')
          received[index] = response['received']
          failures = 0
        except client.Error as exc:
          if exc.code not in client.TRANSIENT or failures >= 2:
            raise
          failures += 1
          await asyncio.sleep(0.5 * failures)
          status = await retry(job, lambda: client.session_request(http, credentials['url'], session))
          received[:] = offsets(status, files)  # Response may have been lost AFTER a successful write.
        current = sum(received)
        jobs.progress(job, phase='uploading', message=f"{item['segment']} / {item['name']}",
                      bytes_current=current, bytes_total=total, percent=8 + 89 * current / total,
                      current=sum(all(received[i] == f['size'] for i, f in enumerate(files) if f['segment'] == segment)
                                  for segment in job['segments']),
                      phase_current=index + 1, phase_total=len(files),
                      bytes_per_second=int((current - original) / max(0.01, time.monotonic() - started)))
      if fingerprint(source) != item['fingerprint'] or not await asyncio.to_thread(segment_is_complete, item['segment']):
        raise client.Error('file_changed')
  jobs.ensure_not_canceled(job)
  jobs.progress(job, phase='notifying', message='Road Viewer: registering batch', percent=98)
  result = await retry(job, lambda: client.session_request(http, credentials['url'], session, 'POST', '/finish'))
  validate_result(result)


def validate_result(result):
  if not isinstance(result, dict) or any(not isinstance(result.get(key), list) for key in ('logs', 'updated', 'duplicates')):
    raise client.Error('invalid_response')


async def run_job(job):
  checkpoint = job.get('_road_viewer')
  code, canceled, success = '', False, False
  credentials = None
  async with aiohttp.ClientSession() as http:
    try:
      credentials = await asyncio.to_thread(config.load)
      if not credentials:
        raise client.Error('disconnected')
      if checkpoint and checkpoint.get('device_id') != credentials['device_id']:
        raise client.Error('retry_unavailable')
      if checkpoint is None:
        checkpoint = await prepare(job)
        checkpoint['device_id'] = credentials['device_id']
        job['_road_viewer'] = checkpoint
      await transfer(job, http, credentials, checkpoint)
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
      if (canceled or code in ('file_changed', 'segment_incomplete', 'local_file_error')) and checkpoint:
        if credentials and checkpoint.get('session'):
          try:
            await client.session_request(http, credentials['url'], checkpoint['session'], 'DELETE')
          except Exception:
            pass  # Offline sessions expire on the server (15 min idle / 2 h absolute).
        job.pop('_road_viewer', None)
  if code:
    config.last_error = code
  summaries = checkpoint['summaries'] if checkpoint else []
  received = checkpoint.get('received', []) if checkpoint else []
  files = checkpoint['files'] if checkpoint else []
  results = [dict(summary, ok=success, transferred=success or all(i < len(received) and received[i] == f['size']
             for i, f in enumerate(files) if f['segment'] == summary['segment'])) for summary in summaries]
  result = {'ok': success, 'target': 'road_viewer', 'uploaded': len(job['segments']) if success else 0,
            'total': len(job['segments']), 'results': results,
            'bytes_received': sum(f['size'] for f in files) if success else sum(received),
            'error': code or None, 'message': code or ('Upload complete' if success else 'Upload canceled'),
            'canceled': canceled}
  if not success and not canceled and job.get('_road_viewer'):
    result['retry_job_id'] = job['id']
  jobs.finish(job, ok=success, result=result, status='canceled' if canceled else None)
