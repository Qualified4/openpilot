"""Private Road Viewer credentials, alongside (never inside) public web settings."""
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from ..config import CARROT_STATE_DIR

CREDENTIAL_PATH = Path(CARROT_STATE_DIR) / 'road_viewer_secret.json'
last_error = ''


def normalize_url(value):
  if not isinstance(value, str) or len(value) > 2048 or re.search(r'[\s\\\x00-\x1f]', value.strip()):
    raise ValueError('invalid_url')
  try:
    url = urlsplit(value.strip())
    host = url.hostname
    port = url.port
    if (url.scheme != 'https' or not host or url.username is not None or url.password is not None
        or url.path not in ('', '/') or url.query or url.fragment or '?' in value or '#' in value
        or '%' in host or url.netloc.endswith(':') or (port is not None and not 1 <= port <= 65535)):
      raise ValueError
    if ':' in host:
      import ipaddress
      host = f'[{ipaddress.IPv6Address(host)}]'
    else:
      host = host.encode('idna').decode('ascii').lower()
      if len(host) > 253 or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) for label in host.split('.')):
        raise ValueError
    return f'https://{host}' + (f':{port}' if port is not None else '')
  except (ValueError, UnicodeError):
    raise ValueError('invalid_url') from None


def load():
  try:
    with CREDENTIAL_PATH.open() as stream:
      data = json.loads(stream.read(8193))
    if (not isinstance(data, dict) or set(data) != {'url', 'device_id', 'device_secret'}
        or not re.fullmatch(r'[0-9a-f]{32}', data.get('device_id', ''))
        or not re.fullmatch(r'[0-9a-f]{64}', data.get('device_secret', ''))):
      raise ValueError
    data['url'] = normalize_url(data['url'])
    return data
  except FileNotFoundError:
    return None
  except (OSError, ValueError, TypeError):
    raise ValueError('credentials_corrupt') from None


def save(data):
  CREDENTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(dir=CREDENTIAL_PATH.parent, prefix='.road_viewer-')
  try:
    with os.fdopen(fd, 'w') as stream:
      json.dump(data, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, CREDENTIAL_PATH)  # mkstemp creates mode 0600.
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def clear():
  global last_error
  CREDENTIAL_PATH.unlink(missing_ok=True)
  last_error = ''


def status():
  try:
    data = load()
  except ValueError:
    return {'state': 're_pair_required', 'error': 'credentials_corrupt', 'url': ''}
  state = 'disconnected' if data is None else 'connected'
  if data and last_error:
    if last_error == 'invalid_device':
      state = 'revoked_or_unknown'
    elif last_error in ('invalid_authentication', 'invalid_signature', 'stale_timestamp'):
      state = 'authentication_failed'
    elif last_error in ('unreachable', 'dns_error', 'tls_error'):
      state = 'unreachable'
  return {'state': state, 'url': data['url'] if data else '', 'error': last_error}
