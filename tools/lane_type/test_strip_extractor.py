"""Small deterministic pixels/geometry, compared with the unchanged a412f001 scalar sampler."""
import ctypes
import io
import itertools
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from openpilot.selfdrive.carrot.lane_type import marking, marking_daemon as md, profile_scanner as ps, strip_extractor as fast, strip_model as sm

WIDTHS = ((.75, 96, 128), (1., 128, 128))
REAL = Path(__file__).resolve().parents[2]/'.analysis/strip-cache/decoded/412/00000412--4440b61281--9/chunk-0003.npz'
needs_native = pytest.mark.skipif(fast.native is None, reason='_strip_native not built (SCons / tools/build_native_cpu.py)')


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


@pytest.mark.parametrize('switches', list(itertools.product((False, True), repeat=3)))
def test_all_switch_combinations_bit_exact(pixels, switches):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor(**dict(zip(('buffers', 'planar', 'geometry'), switches, strict=True)))
  want = scalar(*pixels)
  assert all(s is not None for s in want)
  equal(want, ex.extract(*pixels))
  for widths in ((WIDTHS[0],), (WIDTHS[1],)):
    equal(scalar(*pixels, widths), ex.extract(*pixels, widths))


def test_owned_outputs_and_layout_changes(pixels):
  y, uv, lines, rpy, k = pixels
  ex = fast.Extractor(planar=True)
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


def real_frames(path=REAL):
  """Same reader as .analysis/ccnc-field-412-20261022/codex/device1023/bench_device.py frames() (that script runs on import)."""
  if not path.exists():
    pytest.skip(f'{path} not present')
  with np.load(path) as z:
    meta = json.loads(str(z['meta']))
    yy, uu, yo, uo = (z[k] for k in ('y', 'uv', 'yoff', 'uvoff'))
  for i, f in enumerate(meta):
    h, w = f['shape']
    y, uv = np.zeros((h, w), np.uint8), np.zeros((h//2, w//2, 2), np.uint8)
    yb, ub = f['yb'], f['ub']
    y[yb[0]:yb[1], yb[2]:yb[3]] = yy[yo[i]:yo[i + 1]].reshape(yb[1] - yb[0], yb[3] - yb[2])
    uv[ub[0]:ub[1], ub[2]:ub[3]] = uu[uo[i]:uo[i + 1]].reshape(ub[1] - ub[0], ub[3] - ub[2], 2)
    yield y, uv, [f['lines']['left'], f['lines']['right']], f['rpy'], np.array(f['k'])


def area_reference(planes, cols):
  """The pre-native finish() loop: uint8 -> float32, exact 4:1 INTER_AREA, truncating assignment."""
  image = np.empty((len(planes[0]), cols, 3), np.uint8)
  for i, plane in enumerate(planes):
    image[..., i] = cv2.resize(plane.astype(np.float32), (cols, len(plane)), interpolation=cv2.INTER_AREA)
  return image


@needs_native
def test_native_matches_python_on_real_frames(monkeypatch):
  calls = {'column_map': 0, 'area4_interleave': 0}
  class Spy:
    def __getattr__(self, name):
      def call(*args):
        calls[name] += 1
        return getattr(fast.native, name)(*args)
      return call
  on, off = fast.Extractor(), fast.Extractor(use_native=False)
  assert on.native is fast.native and off.native is None
  on.native = Spy()
  compared = {(c, side): 0 for _, c, _ in WIDTHS for side in ('left', 'right')}
  for frame in real_frames():
    want, got = off.extract(*frame), on.extract(*frame)
    equal(want, got)
    for (_, c, _), side, strip in zip([w for w in WIDTHS for _ in range(2)], ('left', 'right')*2, got, strict=True):
      compared[(c, side)] += strip is not None
  assert all(n >= 20 for n in compared.values()), compared
  assert calls['column_map'] == calls['area4_interleave'] == sum(compared.values())
  print('bit-identical strips', compared)


@needs_native
@pytest.mark.parametrize('rows,cols', ((1, 1), (1, 128), (7, 96), (300, 96), (213, 128)))
def test_area4_kernel_matches_cv2(rows, cols):
  rng = np.random.default_rng(rows*1000 + cols)
  cases = [rng.integers(0, 256, (3, rows, 4*cols), dtype=np.uint8), np.full((3, rows, 4*cols), 255, np.uint8),
           np.zeros((3, rows, 4*cols), np.uint8), rng.choice(np.array([0, 1, 254, 255], np.uint8), (3, rows, 4*cols))]
  for case in cases:
    planes = [np.ascontiguousarray(p) for p in case]
    out = np.full((rows, cols, 3), 77, np.uint8)
    fast.native.area4_interleave(*planes, out)
    assert np.array_equal(out, area_reference(planes, cols))
  with pytest.raises(ValueError):
    fast.native.area4_interleave(*planes[:2], planes[2][:, :-4], out)


@needs_native
def test_column_map_matches_numpy():
  rng = np.random.default_rng(23)
  for rows, cols in ((1, 96), (311, 128), (57, 96)):
    t = fast.COLUMN_GRID[cols]
    e0 = rng.uniform(-2000., 3000., rows)
    e1 = e0 + rng.normal(0., 300., rows)
    e1[:3] = e0[:3]  # zero span
    want = (e0[:, None] + t[None]*(e1 - e0)[:, None]).astype(np.float32)
    mx, half = np.empty_like(want), np.empty_like(want)
    fast.native.column_map(e0, e1, t, mx, half)
    assert np.array_equal(mx.view(np.uint32), want.view(np.uint32))
    assert np.array_equal(half.view(np.uint32), (want*.5).view(np.uint32))


def test_python_fallback_without_native(pixels, monkeypatch):
  monkeypatch.setattr(fast, 'native', None)
  ex = fast.Extractor(use_native=True)
  assert ex.native is None
  equal(scalar(*pixels), ex.extract(*pixels))
  monkeypatch.setattr(fast, 'NATIVE', False)
  monkeypatch.setattr(fast, 'native', SimpleNamespace())  # any attribute access would fail
  assert fast.Extractor().native is None
  equal(scalar(*pixels), fast.Extractor().extract(*pixels))


@needs_native
def test_native_skips_unsupported_planes(pixels):
  # Interleaved UV gives strided chroma views: finish() must use the OpenCV loop, maps() still the native column map.
  ex = fast.Extractor(planar=False)
  ex.native = SimpleNamespace(column_map=fast.native.column_map)
  equal(scalar(*pixels), ex.extract(*pixels))


@needs_native
@pytest.mark.parametrize('rows,cols', ((1, 1), (7, 96), (171, 128)))
def test_area4_strided_chroma_and_reversed_views_match_cv2(rows, cols):
  if not getattr(fast.native, 'STRIDED_AREA4', False):
    pytest.skip('older optional binary')
  rng = np.random.default_rng(1024)
  lum = rng.integers(0, 256, (rows * 2, cols * 8), dtype=np.uint8)[::2, ::2]
  chroma = rng.integers(0, 256, (rows * 2, cols * 4, 2), dtype=np.uint8)[::2]
  for reverse in (False, True):
    planes = [lum, chroma[..., 0], chroma[..., 1]]
    if reverse:
      planes = [p[:, ::-1] for p in planes]
    for p in planes:
      p.flags.writeable = False
    out = np.empty((rows, cols, 3), np.uint8)
    fast.native.area4_interleave(*planes, out)
    assert out.tobytes() == area_reference(planes, cols).tobytes()


def timing(repeats=3):
  """PC median extract() ms per frame, native vs Python, 1 OpenCV thread (python -m tools.lane_type.test_strip_extractor)."""
  cv2.setNumThreads(1)
  frames = [f for f in real_frames() if all(s is not None for s in fast.Extractor(use_native=False).extract(*f))]
  result = {}
  for name, ex in (('python', fast.Extractor(use_native=False)), ('native', fast.Extractor(use_native=True))):
    for f in frames[:5]:
      ex.extract(*f)
    ms = []
    for _ in range(repeats):
      for f in frames:
        t0 = time.perf_counter_ns()
        ex.extract(*f)
        ms.append((time.perf_counter_ns() - t0)/1e6)
    result[name] = float(np.median(ms))
  return dict(frames=len(frames), repeats=repeats, native_built=fast.native is not None, **result)


if __name__ == '__main__':
  print(json.dumps(timing(10)))
