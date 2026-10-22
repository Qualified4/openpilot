"""Small deterministic pixels/geometry, compared with the unchanged a412f001 scalar sampler."""
import ctypes
import io
import itertools
import json
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking, marking_daemon as md, profile_scanner as ps, strip_extractor as fast, strip_model as sm

WIDTHS = ((.75, 96, 128), (1., 128, 128))


@pytest.fixture
def pixels():
  # Padded NV12-style views; a bright/dark/chroma pattern catches rounding and remap-coordinate changes.
  h, w = 192, 320
  yy, xx = np.indices((h, w + 32))
  y = ((xx*13 + yy*7 + (xx//3)*29) % 256).astype(np.uint8)[:, :w]
  yy, xx, cc = np.indices((h//2, w//2 + 16, 2))
  uv = ((xx*17 + yy*11 + cc*61) % 256).astype(np.uint8)[:, :w//2]
  x = np.linspace(0., 80., 33)
  lines = [{'x': x, 'y': sign*(1.8 + .0004*x*x), 'z': np.full_like(x, 1.2)} for sign in (-1, 1)]
  k = np.array([[250., 0., w/2], [0., 250., h/2], [0., 0., 1.]])
  return y, uv, lines, np.zeros(3), k


def scalar(y, uv, lines, rpy, k, widths=WIDTHS):
  return [sm.strip_input(y, uv, line, rpy, k, half_m=h, cols=c, rows=r) for h, c, r in widths for line in lines]


def equal(want, got):
  assert len(want) == len(got)
  for a, b in zip(want, got, strict=True):
    assert (a is None) == (b is None)
    if a is not None:
      assert b.dtype == np.float32 and np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize('switches', list(itertools.product((False, True), repeat=4)))
def test_all_switch_combinations_bit_exact(pixels, switches):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor(**dict(zip(('buffers', 'planar', 'geometry', 'repair'), switches, strict=True)))
  want = scalar(*pixels)
  assert all(s is not None for s in want)
  equal(want, ex.extract(*pixels))
  for widths in ((WIDTHS[0],), (WIDTHS[1],)):
    equal(scalar(*pixels, widths), ex.extract(*pixels, widths))


def test_owned_outputs_and_layout_changes(pixels):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor()
  first = ex.extract(*pixels)
  saved = [s.copy() for s in first]
  assert all(s.flags.owndata for s in first)
  y[:] ^= 255
  ex.extract(*pixels)
  equal(saved, first)
  old = ex.scratch['plane0']
  narrow_y, narrow_uv = y[:, :300], uv[:, :150]
  equal(scalar(narrow_y, narrow_uv, lines, rpy, k), ex.extract(narrow_y, narrow_uv, lines, rpy, k))
  assert ex.scratch['plane0'] is not old
  old = ex.scratch['plane0']
  ex.extract(narrow_y.copy(), narrow_uv.copy(), lines, rpy, k)
  assert ex.scratch['plane0'] is not old  # stride changed, not just resolution
  assert ex.buf('shrink', (17, 23)).flags.c_contiguous
  assert ex.buf('shrink', (9, 11)).flags.c_contiguous


def test_projection_invalidates_for_rpy_and_inplace_K_change(pixels, monkeypatch):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor()
  calls = []
  original = ps.projection
  def projection(*args):
    calls.append(1)
    return original(*args)
  monkeypatch.setattr(ps, 'projection', projection)
  ex.extract(*pixels)
  ex.extract(*pixels)
  assert len(calls) == 1
  for change in ('rpy', 'K'):
    if change == 'rpy':
      rpy[1] += .002
    else:
      k[0, 0] += .5
    want = scalar(*pixels)
    before = len(calls)
    equal(want, ex.extract(*pixels))
    assert len(calls) == before + 1
  assert len(ex.y_maps) <= fast.MAP_CACHE_SIZE


def test_invalid_one_side_and_bounded_maps(pixels):
  y, uv, lines, rpy, k = pixels
  lines[1] = {'x': [1., 1., 1.], 'y': [0.]*3, 'z': [0.]*3}
  ex = fast.Extractor()
  equal(scalar(*pixels), ex.extract(*pixels))
  row = ex.rows(lines[0], ex.matrix, len(y))
  for offset in range(20):
    changed = (row[0] + offset, *row[1:])
    ex.maps(changed, ex.matrix, 1., 128)
  assert len(ex.y_maps) == fast.MAP_CACHE_SIZE
  assert len(ex.scratch) < 40


def test_sparse_and_dense_correction(pixels, monkeypatch):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor()
  equal(scalar(*pixels), ex.extract(*pixels))
  row = ex.rows(lines[0], ex.matrix, len(y))
  n, w = ex.maps(row, ex.matrix, .75, 96), ex.maps(row, ex.matrix, 1., 128)
  # Change both Y and UV interpolation table coordinates. Use the original sampler on those exact maps as reference.
  n = dict(n, mx=n['mx'] + .1)
  lum, cb, cr = ps.sample(y, uv, {'ys': row[0], 'cols': 96, **n})
  import cv2
  image = np.stack((lum, cb, cr), -1).astype(np.uint8)
  want = cv2.resize(image, (96, 128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).astype(np.float32)
  for limit in (10**9, 0):
    monkeypatch.setattr(fast, 'DENSE_LIMIT', limit)
    got, _ = ex.pair(y, uv, n, w, 128)
    equal([want], [got])
  assert ex.counts['dense_fallback'] == 2


def test_mallopt_portable_and_failure_is_ignored(monkeypatch):
  monkeypatch.setattr(platform, 'libc_ver', lambda: ('musl', '1'))
  assert not fast.tune_allocator()
  monkeypatch.setattr(platform, 'libc_ver', lambda: ('glibc', '2.39'))
  def unavailable(*args):
    raise OSError('no libc')
  monkeypatch.setattr(ctypes, 'CDLL', unavailable)
  assert not fast.tune_allocator()
  called = []
  def mallopt(key, value):
    called.append((key, value))
    return int(key != -1)
  monkeypatch.setattr(ctypes, 'CDLL', lambda _: SimpleNamespace(mallopt=mallopt))
  assert not fast.tune_allocator()
  assert called == [(-3, 64 << 20), (-1, 256 << 20), (-2, 16 << 20)]
  monkeypatch.setattr(fast, 'REUSE_BUFFERS', False)
  called.clear()
  assert not fast.tune_allocator() and not called
  monkeypatch.setattr(fast.sys, 'platform', 'win32')
  assert not fast.tune_allocator()


def test_proc_stat_comm_and_missing_procfs(monkeypatch):
  monkeypatch.setattr('builtins.open', lambda *a, **kw: io.StringIO('123 (worker (x)) R 1 2 3 4 5 6 777 8'))
  assert md.minor_faults() == 777
  monkeypatch.setattr('builtins.open', lambda *a, **kw: io.StringIO('malformed'))
  assert md.minor_faults() is None
  def denied(*args, **kw):
    raise OSError('procfs unavailable')
  monkeypatch.setattr('builtins.open', denied)
  assert md.minor_faults() is None


def test_card_parser_ignores_new_timing_keys():
  data = {'type': marking.PACKET_TYPE, 'version': 1, 'mode': 'stripnet', 'frameId': 4,
          'left': {'type': 'solid', 'typeAgeNs': 0, 'color': 'white', 'colorAgeNs': 0}}
  original = marking.parse(json.dumps(data), 100)
  data['timing'] = {'minfltDelta': 7, 'extractMs': 1.2, 'inferMs': .4}
  assert marking.parse(json.dumps(data), 100) == original


@pytest.mark.parametrize('faults', (None, 123))
def test_daemon_timing_fields_and_optional_faults(monkeypatch, tmp_path, faults):
  from tools.lane_type.test_lane_strip_toggle import assets, run_daemon  # noqa: TID251 - repository test helper
  monkeypatch.setattr(md, 'minor_faults', lambda: faults)
  _, _, _, packets = run_daemon(monkeypatch, assets(tmp_path), [0, 0, 0])
  assert packets
  for data in packets:
    timing = data['timing']
    assert timing['extractMs'] >= 0 and timing['inferMs'] >= 0
    assert ('minfltDelta' in timing) == (faults is not None)
    if faults is not None:
      assert timing['minfltDelta'] == 0
