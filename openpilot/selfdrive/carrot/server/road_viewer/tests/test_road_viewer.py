"""Client checks and real HTTPS Road Viewer integration.

ROAD_VIEWER_SOURCE must point to a Road Viewer checkout. The fixture uses that
server unmodified, a disposable storage directory, and a locally trusted cert.
The TCP listener is 8098, as required by the server's device API boundary.
"""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import ssl
import subprocess
import sys
import threading
from pathlib import Path

import aiohttp
import pytest

from openpilot.selfdrive.carrot.server.road_viewer import client, config, register, uploader
from openpilot.selfdrive.carrot.server.features.dashcam import catalog, paths, upload_jobs as jobs


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
  monkeypatch.setattr(config, 'CREDENTIAL_PATH', tmp_path / 'state' / 'road_viewer_secret.json')
  monkeypatch.setattr(config, 'last_error', '')
  monkeypatch.setattr(paths, 'DASHCAM_ROOT', str(tmp_path / 'segments'))
  monkeypatch.setattr(catalog, 'DASHCAM_ROOT', str(tmp_path / 'segments'))
  jobs.jobs().clear()
  yield
  jobs.jobs().clear()


def recording(index=0, video=True):
  segment = f'00000395--0d0eda17c5--{index}'
  directory = Path(paths.DASHCAM_ROOT) / segment
  directory.mkdir(parents=True, exist_ok=True)
  (directory / 'rlog.zst').write_bytes(bytes([index + 1]) * (uploader.BLOCK_SIZE + 17))
  if video:
    (directory / 'qcamera.ts').write_bytes(b'video' * 80001)
  return segment


@pytest.mark.parametrize('url', ['http://host', 'https://user:secret@host', 'https://host/api', 'https://host?x',
                                 'https://host#x', 'https://host:0', 'https://host:65536', 'https://host:',
                                 'https://a\\b', 'https://a%2eb', 'https://-bad.host', None])
def test_invalid_urls(url):
  with pytest.raises(ValueError, match='invalid_url'):
    config.normalize_url(url)


def test_credentials_and_url(tmp_path):
  assert config.normalize_url(' https://EXAMPLE.com:18443/ ') == 'https://example.com:18443'
  assert config.normalize_url('https://[::1]:18443') == 'https://[::1]:18443'
  value = {'url': 'https://example.com:18443', 'device_id': 'a' * 32, 'device_secret': 'b' * 64}
  config.save(value)
  assert config.load() == value
  assert config.CREDENTIAL_PATH.stat().st_mode & 0o777 == 0o600
  public = json.dumps(config.status())
  assert value['device_secret'] not in public and value['device_id'] not in public
  config.CREDENTIAL_PATH.write_text('{bad json')
  assert config.status()['state'] == 're_pair_required'
  config.clear()
  assert config.status()['state'] == 'disconnected'


def test_active_changed_and_symlink_files():
  segment = recording()
  directory = Path(paths.segment_dir(segment))
  lock = directory / 'rlog.zst.lock'
  lock.touch()
  with pytest.raises(client.Error, match='segment_incomplete'):
    uploader.discover([segment])
  lock.unlink()
  checkpoint = asyncio.run(uploader.prepare(jobs.create_job([segment])))
  item = checkpoint['files'][0]
  (directory / item['name']).write_bytes(b'changed')
  with pytest.raises(client.Error, match='file_changed'):
    uploader.open_source(item)
  (directory / 'qcamera.ts').unlink()
  (directory / 'qcamera.ts').symlink_to(directory / 'rlog.zst')
  with pytest.raises(OSError):
    uploader.open_source({'segment': segment, 'name': 'qcamera.ts', 'size': (directory / 'rlog.zst').stat().st_size})
  with pytest.raises(client.Error, match="invalid_segment"):
    uploader.discover(['../../outside--0'])


@pytest.fixture
def server(tmp_path, monkeypatch):
  source = os.environ.get('ROAD_VIEWER_SOURCE')
  if not source:
    pytest.skip('Set ROAD_VIEWER_SOURCE to run the real server integration')
  from werkzeug.serving import WSGIRequestHandler, make_server
  app_dir = Path(source) / 'roadviewer' / 'app'
  monkeypatch.syspath_prepend(str(app_dir))
  monkeypatch.setenv('RV_DATA', str(tmp_path / 'server'))
  monkeypatch.setenv('RV_INGRESS_ONLY', '0')
  monkeypatch.setenv('RV_DEVICE_HOST_PORT', '8098')
  spec = importlib.util.spec_from_file_location('road_viewer_test_server', app_dir / 'server.py')
  implementation = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = implementation
  spec.loader.exec_module(implementation)
  implementation.auto_convert = False
  cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
  subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', str(key), '-out', str(cert),
                  '-days', '1', '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                 check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  class Handler(WSGIRequestHandler):
    def make_environ(self):
      environ = super().make_environ()
      environ['gunicorn.socket'] = self.connection  # Real TLS socket/local port, same guard as production.
      return environ

    def log_request(self, *args):
      pass

  httpd = make_server('127.0.0.1', 8098, implementation.app, threaded=True, ssl_context=(str(cert), str(key)), request_handler=Handler)
  worker = threading.Thread(target=httpd.serve_forever, daemon=True)
  worker.start()
  context = ssl.create_default_context(cafile=str(cert))  # Verification stays on, including host name.
  original = aiohttp.ClientSession
  monkeypatch.setattr(aiohttp, 'ClientSession', lambda **kw: original(connector=aiohttp.TCPConnector(ssl=context), **kw))
  try:
    yield implementation, os.environ.get('ROAD_VIEWER_TEST_URL', 'https://localhost:8098')
  finally:
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=5)
    implementation.pool.shutdown()


def pairing_code(server):
  implementation, _ = server
  with implementation.app.test_client() as admin:
    response = admin.post('/api/settings/devices/pairing', json={}, headers={'X-RoadViewer-Request': '1'})
    assert response.status_code == 200
    return response.json['code']


async def connect(server):
  async with aiohttp.ClientSession() as http:
    credentials = await client.pair(http, server[1], pairing_code(server))
  config.save(credentials)
  return credentials


def test_real_pair_persistence_and_auth(server):
  async def run():
    code = pairing_code(server)
    async with aiohttp.ClientSession() as http:
      with pytest.raises(client.Error, match='invalid_pairing_code'):
        await client.pair(http, server[1], 'wrong')
      credentials = await client.pair(http, server[1], code)
      config.save(credentials)
      with pytest.raises(client.Error, match='invalid_pairing_code'):
        await client.pair(http, server[1], code)
      segment = recording(video=False)
      checkpoint = await uploader.prepare(jobs.create_job([segment]))
      saved = config.load()
      # A fresh interpreter can use persisted credentials to authenticate again.
      script = '\n'.join([
        'import asyncio,json,sys; from pathlib import Path; import aiohttp',
        'from openpilot.selfdrive.carrot.server.road_viewer import config,client',
        'config.CREDENTIAL_PATH=Path(sys.argv[1])',
        'async def run():',
        ' async with aiohttp.ClientSession() as http:',
        '  session=await client.begin(http,config.load(),json.loads(sys.argv[2]))',
        '  assert session["id"]',
        'asyncio.run(run())',
      ])
      process = await asyncio.create_subprocess_exec(sys.executable, '-c', script,
        str(config.CREDENTIAL_PATH), json.dumps(checkpoint['manifest']),
        env={**os.environ, 'SSL_CERT_FILE': str(config.CREDENTIAL_PATH.parents[1] / 'cert.pem')},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
      _, error = await process.communicate()
      assert process.returncode == 0, error.decode()
      session = await client.begin(http, saved, checkpoint['manifest'])
      assert session['files'][0]['received'] == 0
      invalid = dict(saved, device_secret='0' * 64)
      with pytest.raises(client.Error, match='invalid_signature'):
        await client.begin(http, invalid, checkpoint['manifest'])
      bad_time = client.signed_headers(saved, b'{}')
      bad_time['X-RV-Timestamp'] = '1000000000'
      with pytest.raises(client.Error, match='stale_timestamp'):
        await client.request(http, saved['url'], 'POST', '/api/device/uploads', data=b'{}', headers=bad_time)
      api = server[0].devices
      new_code = pairing_code(server)
      api.pairing['expires_at'] = 1
      with pytest.raises(client.Error, match='invalid_pairing_code'):
        await client.pair(http, server[1], new_code)
      with server[0].app.test_client() as admin:
        assert admin.post(f'/api/settings/devices/{saved["device_id"]}/revoke', json={}, headers={'X-RoadViewer-Request': '1'}).status_code == 200
      with pytest.raises(client.Error, match='invalid_device'):
        await client.begin(http, saved, checkpoint['manifest'])
  asyncio.run(run())


def test_real_multi_segment_response_loss_and_retry(server, monkeypatch):
  async def run():
    await connect(server)
    segments = [recording(0), recording(1, video=False)]
    job = jobs.create_job(segments)
    job['upload_target'] = 'road_viewer'
    original = client.session_request
    puts = []
    lost = False
    failed = False

    async def interrupted(http, url, session, method='GET', suffix='', data=None):
      nonlocal lost, failed
      if method == 'PUT':
        puts.append((suffix, len(data)))
        if '/files/1?' in suffix and not failed:
          failed = True
          raise client.Error('tls_error')  # Permanent failure leaves later files pending.
        result = await original(http, url, session, method, suffix, data)
        if not lost:
          lost = True
          raise client.Error('unreachable')  # The actual server accepted bytes but the response was lost.
        return result
      return await original(http, url, session, method, suffix, data)

    monkeypatch.setattr(client, 'session_request', interrupted)
    await jobs.start_job(job, runner=uploader.run_job)
    assert job['status'] == 'failed', job['result']
    checkpoint = job['_road_viewer']
    assert checkpoint['received'][0] == checkpoint['files'][0]['size']
    assert len(list(server[0].UPLOADS.iterdir())) == 1
    first_puts = len(puts)
    retry_job = jobs.create_job(segments)
    retry_job['_road_viewer'] = copy.deepcopy(checkpoint)
    await jobs.start_job(retry_job, runner=uploader.run_job)
    assert retry_job['status'] == 'done', retry_job['result']
    assert retry_job['result']['uploaded'] == 2
    assert not any('/files/0?' in suffix for suffix, _ in puts[first_puts:])
    assert sum('/files/0?offset=0' == suffix for suffix, _ in puts) == 1
    assert all(size <= uploader.BLOCK_SIZE for _, size in puts)
    stored = list(server[0].recording_paths())
    assert len(stored) == 2
    for directory in stored:
      metadata = server[0].read_meta(directory)
      source = Path(paths.DASHCAM_ROOT) / metadata['files']['rlog.zst'].removesuffix('--rlog.zst')
      assert (directory / 'rlog.zst').read_bytes() == (source / 'rlog.zst').read_bytes()
      if (directory / 'qcamera.ts').exists():
        assert (directory / 'qcamera.ts').read_bytes() == (source / 'qcamera.ts').read_bytes()
    secret = config.load()['device_secret']
    public = json.dumps(jobs.snapshot(retry_job))
    assert secret not in public and checkpoint['session']['token'] not in public
  asyncio.run(run())


def test_real_storage_errors_expiration_cancel_and_repair(server, monkeypatch):
  async def run():
    credentials = await connect(server)
    segment = recording(video=False)
    checkpoint = await uploader.prepare(jobs.create_job([segment]))
    implementation = server[0]
    async with aiohttp.ClientSession() as http:
      policy = implementation.storage_policy
      old = copy.deepcopy(policy.settings)
      policy.settings.update(max_bytes=1, policy='reject_new')
      with pytest.raises(client.Error, match='storage_limit_exceeded'):
        await client.begin(http, credentials, checkpoint['manifest'])
      assert not list(implementation.UPLOADS.iterdir())
      policy.settings = old
      original_disk = implementation.shutil.disk_usage
      monkeypatch.setattr(implementation.shutil, 'disk_usage', lambda _: type(original_disk(implementation.ROOT))(1000, 999, 1))
      with pytest.raises(client.Error, match='insufficient_disk_space'):
        await client.begin(http, credentials, checkpoint['manifest'])
      policy.settings['policy'] = 'delete_oldest'
      with pytest.raises(client.Error, match='no_deletable_logs'):
        await client.begin(http, credentials, checkpoint['manifest'])
      policy.settings = old
      monkeypatch.setattr(implementation.shutil, 'disk_usage', original_disk)
      session = await client.begin(http, credentials, checkpoint['manifest'])
      session_path = implementation.UPLOADS / session['id']
      state_path = session_path / 'device.json'
      state = json.loads(state_path.read_text())
      state['expires_at'] = 1
      state_path.write_text(json.dumps(state))
      with pytest.raises(client.Error, match='session_expired'):
        await client.session_request(http, credentials['url'], session)
      state['expires_at'] = 4000000000
      state_path.write_text(json.dumps(state))
      await client.session_request(http, credentials['url'], session, 'DELETE')
      assert not session_path.exists()
    config.clear()
    assert config.load() is None
    new = await connect(server)
    assert new['device_id'] != credentials['device_id']
    job = jobs.create_job([segment])
    await jobs.start_job(job, runner=uploader.run_job)
    assert job['status'] == 'done'
  asyncio.run(run())


def test_cancel_removes_partial_server_data(server, monkeypatch):
  async def run():
    await connect(server)
    job = jobs.create_job([recording()])
    original = client.session_request

    async def cancel_after_chunk(*args, **kwargs):
      result = await original(*args, **kwargs)
      if len(args) > 3 and args[3] == 'PUT':
        jobs.cancel_job(job['id'])
      return result

    monkeypatch.setattr(client, 'session_request', cancel_after_chunk)
    await jobs.start_job(job, runner=uploader.run_job)
    assert job['status'] == 'canceled'
    assert not list(server[0].UPLOADS.iterdir())
    assert '_road_viewer' not in job
  asyncio.run(run())


def test_finish_response_loss_does_not_duplicate_logs(server, monkeypatch):
  async def run():
    await connect(server)
    original = client.session_request
    lost = False

    async def lose_finish(http, url, session, method='GET', suffix='', data=None):
      nonlocal lost
      result = await original(http, url, session, method, suffix, data)
      if suffix == '/finish' and not lost:
        lost = True
        raise client.Error('unreachable')
      return result

    monkeypatch.setattr(client, 'session_request', lose_finish)
    job = jobs.create_job([recording(video=False)])
    await jobs.start_job(job, runner=uploader.run_job)
    assert job['status'] == 'done'
    assert len(list(server[0].recording_paths())) == 1
  asyncio.run(run())


def test_real_recording(server):
  source = os.environ.get('ROAD_VIEWER_TEST_SEGMENT')
  if not source:
    pytest.skip('Set ROAD_VIEWER_TEST_SEGMENT to a completed real segment containing rlog.zst and qcamera.ts')
  import shutil
  source = Path(source)
  segment = source.name
  destination = Path(paths.DASHCAM_ROOT) / segment
  destination.mkdir(parents=True)
  for name in ('rlog.zst', 'qcamera.ts'):
    shutil.copyfile(source / name, destination / name)

  async def run():
    await connect(server)
    import time
    idle_cpu = time.process_time()
    await asyncio.sleep(0.3)
    idle_cpu = time.process_time() - idle_cpu
    ticks = []
    running = True

    async def heartbeat():
      previous = time.monotonic()
      while running:
        await asyncio.sleep(0.01)
        now = time.monotonic()
        ticks.append(now - previous)
        previous = now

    monitor = asyncio.create_task(heartbeat())
    started, cpu = time.monotonic(), time.process_time()
    job = jobs.create_job([segment])
    await jobs.start_job(job, runner=uploader.run_job)
    elapsed, cpu = time.monotonic() - started, time.process_time() - cpu
    running = False
    await monitor
    print(f'\nDesktop client + test server: idle CPU={idle_cpu:.3f}s/0.3s, upload wall={elapsed:.3f}s, '
          + f'CPU={cpu:.3f}s, max 10ms heartbeat interval={max(ticks, default=0):.3f}s')
    assert job['status'] == 'done', job['result']
    stored = list(server[0].recording_paths())
    assert len(stored) == 1
    for name in ('rlog.zst', 'qcamera.ts'):
      with (source / name).open('rb') as before, (stored[0] / name).open('rb') as after:
        assert hashlib.file_digest(before, 'sha256').digest() == hashlib.file_digest(after, 'sha256').digest()
  asyncio.run(run())


def test_carrot_routes_pair_upload_disconnect(server):
  from aiohttp import web
  from aiohttp.test_utils import TestClient, TestServer

  async def run():
    app = web.Application()
    register(app)
    async with TestClient(TestServer(app)) as browser:
      response = await browser.post('/api/road-viewer/pair', json={'url': server[1], 'code': pairing_code(server)},
                                    headers={'Origin': 'https://evil.invalid'})
      assert response.status == 403
      response = await browser.post('/api/road-viewer/pair', json={'url': server[1], 'code': pairing_code(server)})
      assert response.status == 200, await response.text()
      status = await response.json()
      assert status['state'] == 'connected'
      assert 'device_secret' not in status and 'device_id' not in status
      segment = recording(video=False)
      response = await browser.post('/api/road-viewer/summary', json={'segments': [segment]})
      assert (await response.json())['summaries'][0]['files'][0]['name'] == 'rlog.zst'
      response = await browser.post('/api/road-viewer/start', json={'segments': [segment]})
      job = jobs.jobs()[(await response.json())['job_id']]
      await job['_task']
      assert job['status'] == 'done'
      response = await browser.post('/api/road-viewer/disconnect', json={})
      assert (await response.json())['state'] == 'disconnected'
      assert not config.CREDENTIAL_PATH.exists()
      assert '_road_viewer' not in job
  asyncio.run(run())
