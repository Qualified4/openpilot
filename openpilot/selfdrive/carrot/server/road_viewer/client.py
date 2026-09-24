"""Road Viewer RV1 protocol. No credentials or remote error text leave this module."""
import hashlib
import hmac
import json
import secrets
import socket
import time

import aiohttp

# Only trusted, fixed error codes are returned to Carrot Web.
ERRORS = frozenset('''device_limit_reached invalid_pairing_code pairing_rate_limited invalid_pairing_request
invalid_authentication invalid_signature invalid_device stale_timestamp nonce_replayed
invalid_upload_token session_not_found session_expired session_limit_reached authentication_rate_limited
storage_limit_exceeded insufficient_disk_space no_deletable_logs batch_too_large request_too_large
invalid_batch invalid_segments invalid_segment duplicate_segment invalid_file duplicate_file rlog_required
invalid_chunk invalid_offset incomplete_upload batch_conflict offset_conflict upload_incomplete upload_busy checksum_mismatch session_completed
unreachable dns_error tls_error invalid_response invalid_url credentials_corrupt disconnected
file_changed file_not_allowed segment_incomplete local_file_error local_settings_error retry_unavailable
upload_running pairing_busy upload_failed'''.split())
TRANSIENT = frozenset(('unreachable', 'dns_error', 'upload_busy', 'offset_conflict'))


class Error(Exception):
  def __init__(self, code):
    self.code = code if isinstance(code, str) and code in ERRORS else 'upload_failed'
    super().__init__(self.code)


def signed_headers(credentials, body):
  stamp = str(int(time.time()))  # noqa: TID251
  nonce = secrets.token_urlsafe(24)
  canonical = '\n'.join(('RV1', credentials['device_id'], stamp, nonce, 'POST', '/api/device/uploads',
                         hashlib.sha256(body).hexdigest()))
  signature = hmac.new(credentials['device_secret'].encode('ascii'), canonical.encode(), hashlib.sha256).hexdigest()
  return {'Content-Type': 'application/json', 'X-RV-Device': credentials['device_id'], 'X-RV-Timestamp': stamp,
          'X-RV-Nonce': nonce, 'X-RV-Signature': signature}


async def request(http, url, method, path, *, data=None, headers=None, deadline=45):
  try:
    async with http.request(method, url + path, data=data, headers=headers, allow_redirects=False,
                            timeout=aiohttp.ClientTimeout(total=deadline, connect=15)) as response:
      payload = bytearray()
      async for block in response.content.iter_chunked(65536):
        payload.extend(block)
        if len(payload) > 2 * 1024 * 1024:
          raise Error('invalid_response')
      try:
        result = json.loads(payload)
      except (ValueError, UnicodeError):
        raise Error('invalid_response') from None
      if not isinstance(result, dict):
        raise Error('invalid_response')
      if not 200 <= response.status < 300:
        raise Error(result.get('error'))
      return result
  except (aiohttp.ClientConnectorCertificateError, aiohttp.ClientSSLError):
    raise Error('tls_error') from None
  except aiohttp.ClientConnectorError as exc:
    raise Error('dns_error' if isinstance(exc.os_error, socket.gaierror) else 'unreachable') from None
  except (aiohttp.ClientError, TimeoutError):
    raise Error('unreachable') from None


async def pair(http, url, code):
  data = json.dumps({'code': code, 'metadata': {'name': 'Carrot Web'}}).encode()
  result = await request(http, url, 'POST', '/api/device/pair', data=data, headers={'Content-Type': 'application/json'})
  import re
  if (result.get('algorithm') != 'HMAC-SHA256' or not re.fullmatch(r'[0-9a-f]{32}', str(result.get('device_id', '')))
      or not re.fullmatch(r'[0-9a-f]{64}', str(result.get('device_secret', '')))):
    raise Error('invalid_response')
  return {'url': url, 'device_id': result['device_id'], 'device_secret': result['device_secret']}


async def begin(http, credentials, manifest):
  body = json.dumps(manifest, separators=(',', ':')).encode()
  return await request(http, credentials['url'], 'POST', '/api/device/uploads', data=body,
                       headers=signed_headers(credentials, body))


async def session_request(http, url, session, method='GET', suffix='', data=None):
  import re
  if not re.fullmatch(r'[0-9a-f]{32}', str(session.get('id', ''))) or not isinstance(session.get('token'), str):
    raise Error('invalid_response')
  return await request(http, url, method, '/api/device/uploads/' + session['id'] + suffix, data=data,
                       headers={'Authorization': 'Bearer ' + session['token'], 'Content-Type': 'application/octet-stream'},
                       deadline=180 if suffix == '/finish' else 45)
