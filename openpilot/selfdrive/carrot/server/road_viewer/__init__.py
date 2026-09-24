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
          segments = list(dict.fromkeys(await request_upload_segments(request)))
          credentials = await asyncio.to_thread(config.load)
          if not credentials:
            raise client.Error('disconnected')
          summaries = await asyncio.to_thread(uploader.discover, segments)
          if action == 'summary':
            return web.json_response({'ok': True, 'summaries': summaries})
          running = jobs.running_job()
          if running:
            return web.json_response({'ok': False, 'error': 'upload_running', 'job_id': running['id']}, status=409)
          checkpoint = None
          if body.get('retry_job_id'):
            old = jobs.jobs().get(str(body['retry_job_id']), {})
            checkpoint = old.get('_road_viewer')
            if old.get('segments') != segments or not checkpoint or checkpoint['device_id'] != credentials['device_id']:
              raise client.Error('retry_unavailable')
            checkpoint = copy.deepcopy(checkpoint)
            if old.get('error') in ('session_expired', 'session_not_found'):
              checkpoint['manifest']['batch_id'] = secrets.token_urlsafe(24)
              checkpoint.pop('session', None)
              checkpoint.pop('received', None)
          job = jobs.create_job(segments)
          job['upload_target'] = 'road_viewer'
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
