import asyncio
import importlib.util
from pathlib import Path

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from opendbc.car.hyundai import ccnc_debug as debug


@pytest.fixture
def feature(monkeypatch, tmp_path):
  spec = importlib.util.spec_from_file_location('ccnc_debug_routes', Path(__file__).parents[1] / 'features/ccnc_debug.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  monkeypatch.setattr(debug, 'COMMAND_PATH', tmp_path / 'command')
  monkeypatch.setattr(debug, 'STATUS_PATH', tmp_path / 'status')
  return module


def test_web_lease_preview_live_edits_off_disconnect_and_timeout(feature):
  async def run():
    app = web.Application()
    feature.register(app)
    async with TestClient(TestServer(app)) as client:
      response = await client.get('/api/ccnc_debug')
      assert response.status == 200
      schema = await response.json()
      assert schema['defaults']['vehicles']['LF']['detect'] == 6
      origin = str(client.make_url('/')).rstrip('/')
      ws = await client.ws_connect('/ws/ccnc_debug', headers={'Origin': origin})
      config = debug.defaults()
      config['vehicles_enabled'] = True
      config['vehicles']['LF']['enabled'] = True
      await ws.send_json(config)
      message = await ws.receive_json()
      assert message['preview']['CCNC_0x162']['fields']['LF_DETECT']['raw'] == 6
      assert not debug.COMMAND_PATH.exists()  # Preview never publishes a lease.
      config['transmit'] = True
      await ws.send_json(config)
      await ws.receive_json()
      original = debug.read_json(debug.COMMAND_PATH)
      config['vehicles']['LF']['distance'] = 22.2
      await ws.send_json(config)
      await ws.receive_json()
      edited = debug.read_json(debug.COMMAND_PATH)
      assert edited['session'] == original['session']
      assert edited['config']['vehicles']['LF']['distance'] == 22.2
      assert edited['updated_at'] >= original['updated_at']
      # A controller safety rejection cannot be bypassed with heartbeat traffic.
      debug.write_json(debug.STATUS_PATH, {'session': original['session'], 'updated_at': feature.time.monotonic(),
                                           'active': False, 'disarmed': True, 'reason': 'vehicle moving'})
      await ws.send_json(config)
      assert (await ws.receive_json())['status']['disarmed']
      assert not debug.COMMAND_PATH.exists()
      config['transmit'] = False
      await ws.send_json(config)
      await ws.receive_json()
      config['transmit'] = True
      await ws.send_json(config)
      await ws.receive_json()
      assert debug.read_json(debug.COMMAND_PATH)['session'] != original['session']
      await ws.close()
      # A subsequent connection is also an acknowledgement that finally ran.
      ws = await client.ws_connect('/ws/ccnc_debug', headers={'Origin': origin})
      assert not debug.COMMAND_PATH.exists()
      await ws.send_json(config)
      await ws.receive_json()
      assert debug.COMMAND_PATH.exists()
      assert (await ws.receive(timeout=2.)).type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
      assert not debug.COMMAND_PATH.exists()
  asyncio.run(run())


def test_web_rejects_cross_origin_second_owner_and_invalid_values(feature):
  async def run():
    app = web.Application()
    feature.register(app)
    async with TestClient(TestServer(app)) as client:
      for origin in ('http://evil.invalid', 'null', ''):
        response = await client.get('/ws/ccnc_debug', headers={'Origin': origin})
        assert response.status == 403
      origin = str(client.make_url('/')).rstrip('/')
      ws = await client.ws_connect('/ws/ccnc_debug', headers={'Origin': origin})
      response = await client.get('/ws/ccnc_debug', headers={'Origin': origin})
      assert response.status == 409
      config = debug.defaults()
      config['transmit'] = True
      await ws.send_json(config)
      await ws.receive_json()
      assert debug.COMMAND_PATH.exists()
      config['vehicles']['LF']['lateral'] = -.1
      await ws.send_json(config)
      assert (await ws.receive_json())['error']
      assert not debug.COMMAND_PATH.exists()
      await ws.close()
  asyncio.run(run())


def test_startup_and_cleanup_clear_stale_commands(feature):
  async def run():
    debug.write_json(debug.COMMAND_PATH, {'stale': True})
    app = web.Application()
    feature.register(app)
    async with TestClient(TestServer(app)):
      assert not debug.COMMAND_PATH.exists()
      debug.write_json(debug.COMMAND_PATH, {'stale': True})
    assert not debug.COMMAND_PATH.exists()
  asyncio.run(run())
