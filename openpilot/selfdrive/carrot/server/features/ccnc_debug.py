"""One browser owns a volatile debug lease; disconnect/OFF removes it."""
import asyncio
import json
import time
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import WSMsgType, web
from opendbc.car.hyundai import ccnc_debug as debug

OWNER = web.AppKey('ccnc_debug_owner', dict)


async def get_schema(request):
  try:
    return web.json_response(debug.schema(), headers={'Cache-Control': 'no-store'})
  except (ValueError, KeyError) as exc:
    raise web.HTTPServiceUnavailable(text=str(exc)) from exc


def status(session):
  try:
    value = debug.read_json(debug.STATUS_PATH)
    if value['session'] == session and 0 <= time.monotonic() - value['updated_at'] < debug.LEASE_SECONDS:
      return value
  except (OSError, ValueError, TypeError, KeyError):
    pass
  return {'active': False, 'reason': 'CCNC generator unavailable / waiting', 'disarmed': False}


async def websocket(request):
  # No cross-origin write access, including from another page on the LAN.
  origin = urlsplit(request.headers.get('Origin', ''))
  if origin.scheme not in ('http', 'https') or origin.netloc != request.host:
    raise web.HTTPForbidden(text='same-origin browser required')
  owner = request.app[OWNER]
  if owner['socket'] is not None:
    raise web.HTTPConflict(text='CCNC Debug already open in another browser')
  debug.schema()  # Validate the real DBC before allowing any lease.
  ws = web.WebSocketResponse(max_msg_size=8192, compress=False)
  owner['socket'] = ws
  transmitting = False
  session = uuid4().hex
  try:
    await ws.prepare(request)
    while not ws.closed:
      try:
        message = await asyncio.wait_for(ws.receive(), timeout=debug.LEASE_SECONDS)
      except TimeoutError:
        break
      if message.type != WSMsgType.TEXT:
        break
      try:
        config = debug.validate(json.loads(message.data))
        # Every edit goes through the production DBC encoder and decoder.
        packed = debug.preview(config)
        if config['transmit'] and not transmitting:
          session = uuid4().hex
        transmitting = config['transmit']
        current = status(session)
        if transmitting and current['disarmed']:
          debug.COMMAND_PATH.unlink(missing_ok=True)
        elif transmitting:
          debug.write_json(debug.COMMAND_PATH, {'session': session, 'updated_at': time.monotonic(), 'config': config})
        else:
          debug.COMMAND_PATH.unlink(missing_ok=True)
          current = {'active': False, 'reason': 'preview', 'disarmed': False}
        await ws.send_json({'preview': packed, 'status': current, 'transmit_requested': transmitting})
      except (ValueError, TypeError, KeyError, OSError) as exc:
        debug.COMMAND_PATH.unlink(missing_ok=True)
        transmitting = False
        await ws.send_json({'error': str(exc), 'status': {'active': False, 'disarmed': True, 'reason': 'invalid configuration'}})
  finally:
    debug.COMMAND_PATH.unlink(missing_ok=True)
    owner['socket'] = None
    await ws.close()
  return ws


async def cleanup_context(app):
  debug.COMMAND_PATH.unlink(missing_ok=True)
  debug.STATUS_PATH.unlink(missing_ok=True)
  yield
  owner = app[OWNER]['socket']
  if owner is not None:
    await owner.close()
  debug.COMMAND_PATH.unlink(missing_ok=True)


def register(app):
  app[OWNER] = {'socket': None}
  app.cleanup_ctx.append(cleanup_context)
  app.router.add_get('/api/ccnc_debug', get_schema)
  app.router.add_get('/ws/ccnc_debug', websocket)
