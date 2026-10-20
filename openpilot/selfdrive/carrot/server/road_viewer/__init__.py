"""Small HTTP integration; credentials and session tokens stay backend-only."""
import asyncio
import copy
import secrets
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from . import client, config


def register(app):
  from ..features.dashcam import upload_jobs as jobs
  from ..features.dashcam.routes import request_upload_segments
  from . import uploader

  lock = asyncio.Lock()

  async def handle(request):
    action = request.match_info['action']
    if request.method == 'GET':
      if action != 'status':
        raise web.HTTPNotFound()
      return web.json_response(await asyncio.to_thread(config.status), headers={'Cache-Control': 'no-store'})
    # Protect the pairing destination/credentials from cross-origin browser POSTs.
    origin = request.headers.get('Origin')
    if request.content_type != 'application/json' or (origin and urlsplit(origin).netloc != request.host):
      raise web.HTTPForbidden()
    if lock.locked():
      return web.json_response({'ok': False, 'error': 'pairing_busy'}, status=409)
    async with lock:
      try:
        body = await request.json()
        if not isinstance(body, dict):
          raise client.Error('invalid_pairing_request')
        if action == 'disconnect':
          await asyncio.to_thread(config.clear)
          for job in list(jobs.jobs().values()):
            if job.get('upload_target') != 'road_viewer':
              continue
            jobs.cancel_job(job['id'])
            task = job.get('_task')
            if task:
              await task  # Current bounded request completes, then cancellation releases the session.
            job.pop('_road_viewer', None)
            if job.get('result'):
              job['result'].pop('retry_job_id', None)
          config.last_error = ''
        elif action == 'pair':
          if any(j.get('status') == 'running' and j.get('upload_target') == 'road_viewer' for j in jobs.jobs().values()):
            raise client.Error('upload_running')
          url = config.normalize_url(body.get('url'))
          code = body.get('code')
          if not isinstance(code, str) or len(code) > 128:
            raise client.Error('invalid_pairing_request')
          async with aiohttp.ClientSession() as http:
            credentials = await client.pair(http, url, code.strip())
          await asyncio.to_thread(config.save, credentials)
          config.last_error = ''
          for job in jobs.jobs().values():
            job.pop('_road_viewer', None)
            if job.get('result'):
              job['result'].pop('retry_job_id', None)
        elif action in ('summary', 'start'):
          selected = body.get('segments', [body.get('segment')])
          if not isinstance(selected, list) or not 1 <= len(selected) <= 50:
            raise client.Error('invalid_segments')
          previous = jobs.jobs().get(str(body.get('retry_job_id')), {}) if body.get('retry_job_id') else None
          if previous is not None:
            # Completed originals may have been rotated off the device already.
            # Only the exact earlier, validated selection can resume its checkpoint.
            if previous.get('segments') != selected:
              raise client.Error('retry_unavailable')
            segments = previous['segments']
          else:
            segments, _ = await request_upload_segments(request)
            segments = list(dict.fromkeys(segments))
          credentials = await asyncio.to_thread(config.load)
          if not credentials:
            raise client.Error('disconnected')
          options = {key: body.get(key, False) for key in ('include_front', 'include_wide')}
          if any(type(value) is not bool for value in options.values()):
            raise client.Error('invalid_file')
          checkpoint = previous.get('_road_viewer') if previous is not None else None
          if previous is not None and (not checkpoint or checkpoint['device_id'] != credentials['device_id']):
            raise client.Error('retry_unavailable')
          completed = {summary['segment'] for batch in checkpoint.get('batches', []) if batch and batch.get('complete')
                       for summary in batch['summaries']} if checkpoint else set()
          pending = [segment for segment in segments if segment not in completed]
          summaries = await asyncio.to_thread(uploader.discover, pending, previous.get('road_viewer_options', {}) if previous else options) if pending else []
          if action == 'summary':
            return web.json_response({'ok': True, 'summaries': summaries})
          running = jobs.running_job()
          if running:
            return web.json_response({'ok': False, 'error': 'upload_running', 'job_id': running['id']}, status=409)
          if body.get('retry_job_id'):
            checkpoint = copy.deepcopy(checkpoint)
            options = previous.get('road_viewer_options', {})
            if previous.get('error') in ('session_expired', 'session_not_found'):
              for batch in checkpoint.get('batches', [checkpoint]):
                if batch and not batch.get('complete'):
                  batch['manifest']['batch_id'] = secrets.token_urlsafe(24)
                  batch.pop('session', None)
                  batch.pop('received', None)
          job = jobs.create_job(segments)
          job['upload_target'] = 'road_viewer'
          job['road_viewer_options'] = options
          if checkpoint:
            job['_road_viewer'] = checkpoint
          jobs.start_job(job, runner=uploader.run_job)
          return web.json_response({'ok': True, 'job_id': job['id'], 'status': job['status']})
        else:
          raise web.HTTPNotFound()
        return web.json_response({'ok': True, **await asyncio.to_thread(config.status)}, headers={'Cache-Control': 'no-store'})
      except client.Error as exc:
        config.last_error = exc.code
        return web.json_response({'ok': False, 'error': exc.code}, status=400)
      except ValueError as exc:
        code = str(exc) if str(exc) in ('invalid_url', 'credentials_corrupt') else 'invalid_pairing_request'
        return web.json_response({'ok': False, 'error': code}, status=400)
      except web.HTTPException as exc:
        return web.json_response({'ok': False, 'error': 'segment_incomplete' if exc.status == 409 else 'invalid_segment'}, status=400)
      except Exception:
        return web.json_response({'ok': False, 'error': 'local_settings_error'}, status=500)

  app.router.add_get('/api/road-viewer/{action}', handle)
  app.router.add_post('/api/road-viewer/{action}', handle)
