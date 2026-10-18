"""One browser owns a volatile debug lease; disconnect/OFF removes it."""
import asyncio
import json
import time
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import WSMsgType, web
from opendbc.car.hyundai import ccnc_debug as debug

OWNER = web.AppKey('ccnc_debug_owner', dict)
REFRESH_SECONDS = 0.2  # well inside the controller's lease
HEARTBEAT_SECONDS = 2.0  # aiohttp closes when a ping goes unanswered for half of this


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
  # The browser sends only edits. This process renews the controller lease while
  # the socket is alive; protocol pings (answered by the browser itself, not page
  # timers) detect a vanished phone, and the controller lease still expires if
  # this process stops. A slow tab or Wi-Fi hiccup no longer ends the session.
  ws = web.WebSocketResponse(max_msg_size=8192, compress=False, heartbeat=HEARTBEAT_SECONDS)
  owner['socket'] = ws
  state = {'config': None, 'transmitting': False, 'session': uuid4().hex}

  def publish():
    if state['transmitting']:
      debug.write_json(debug.COMMAND_PATH, {'session': state['session'], 'updated_at': time.monotonic(),
                                            'config': state['config']})
    else:
      debug.COMMAND_PATH.unlink(missing_ok=True)

  def current():
    if not state['transmitting']:
      return {'active': False, 'reason': 'preview', 'disarmed': False}
    return status(state['session'])

  async def renew():
    last = None
    while not ws.closed:
      await asyncio.sleep(REFRESH_SECONDS)
      value = current()
      if state['transmitting'] and value['disarmed']:
        # The controller ended this session (vehicle state); never renew it.
        state['transmitting'] = False
        debug.COMMAND_PATH.unlink(missing_ok=True)
        await ws.send_json({'status': value, 'transmit_requested': False})
        await ws.close()
        return
      publish()
      if value != last:
        await ws.send_json({'status': value, 'transmit_requested': state['transmitting']})
        last = value

  renewer = None
  try:
    await ws.prepare(request)
    renewer = asyncio.create_task(renew())
    async for message in ws:
      if message.type != WSMsgType.TEXT:
        break
      try:
        config = debug.validate(json.loads(message.data))
        # Every edit goes through the production DBC encoder and decoder.
        packed = debug.preview(config)
        if config['transmit'] and not state['transmitting']:
          state['session'] = uuid4().hex
        state['config'], state['transmitting'] = config, config['transmit']
        publish()
        await ws.send_json({'preview': packed, 'status': current(), 'transmit_requested': state['transmitting']})
      except (ValueError, TypeError, KeyError, OSError) as exc:
        state['transmitting'] = False
        debug.COMMAND_PATH.unlink(missing_ok=True)
        await ws.send_json({'error': str(exc), 'status': {'active': False, 'disarmed': True, 'reason': 'invalid configuration'}})
  finally:
    if renewer is not None:
      renewer.cancel()
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
