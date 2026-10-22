"""Bit-exact batched strip extraction. Only constant maps and scratch storage persist between frames."""
from collections import OrderedDict
import math
import sys

import cv2
import numpy as np

from openpilot.selfdrive.carrot.lane_type import profile_scanner as ps

# Restart the daemon after changing a switch. Buffer reuse and allocator tuning can be disabled together.
REUSE_BUFFERS = True
TUNE_ALLOCATOR = True
REUSE_WIDE = True
PLANAR_UV = True
SHARE_GEOMETRY = True
DENSE_LIMIT = 16_384
MAP_CACHE_SIZE = 8
DISTANCES = np.arange(ps.NEAR_M, ps.FAR_M + .001, .02)
COLUMN_GRID = {c: (np.arange(c*ps.SUB) + .5)/(c*ps.SUB) for c in (96, 128)}


def tune_allocator():
  """Best effort, glibc/Linux only; a missing/failed mallopt must not prevent inference."""
  if not REUSE_BUFFERS or not TUNE_ALLOCATOR or not sys.platform.startswith('linux'):
    return False
  try:
    import ctypes
    import platform
    if platform.libc_ver()[0] != 'glibc':
      return False
    mallopt = ctypes.CDLL(None).mallopt
    mallopt.argtypes, mallopt.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
    results = [mallopt(key, value) for key, value in ((-3, 64 << 20), (-1, 256 << 20), (-2, 16 << 20))]
    return all(results)
  except (AttributeError, OSError, TypeError, ValueError):
    return False


class Extractor:
  def __init__(self, *, buffers=None, planar=None, geometry=None, repair=None):
    self.buffers = REUSE_BUFFERS if buffers is None else buffers
    self.planar = PLANAR_UV if planar is None else planar
    self.geometry = SHARE_GEOMETRY if geometry is None else geometry
    self.repair = REUSE_WIDE if repair is None else repair
    self.scratch, self.y_maps = {}, OrderedDict()
    self.layout = self.key = self.matrix = None
    self.counts = {'repair_y': 0, 'repair_uv': 0, 'dense_fallback': 0}

  def buf(self, tag, shape, dtype=np.uint8):
    if not self.buffers:
      return np.empty(shape, dtype)
    size = math.prod(shape)
    old = self.scratch.get(tag)
    if old is None or old.size < size or old.dtype != dtype:
      old = self.scratch[tag] = np.empty(size, dtype)
    # A linear capacity buffer gives OpenCV a contiguous destination even as the sampled row count shrinks.
    return old[:size].reshape(shape)

  @staticmethod
  def rows(line, matrix, height):
    x, y, z = (np.asarray(line[c], float) for c in 'xyz')
    if len(x) < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0) or x[0] > ps.NEAR_M or x[-1] < ps.FAR_M + 1:
      return None
    lat, zz = np.interp(DISTANCES, x, y), np.interp(DISTANCES, x, z)
    p = np.column_stack((DISTANCES, lat, zz)) @ matrix.T
    if (p[:, 2] <= .1).any():
      return None
    v = p[:, 1]/p[:, 2]
    if np.any(np.diff(v) >= 0):
      return None
    top = max(0, math.ceil(v[-1]))
    bottom = min(math.floor(v[0]), int(height*ps.HOOD_FRACTION) - 1, height - 1)
    if bottom - top < 8:
      return None
    ys = np.arange(top, bottom + 1)
    d = np.interp(ys, v[::-1], DISTANCES[::-1])
    return ys, d, np.interp(d, x, y), np.interp(d, x, z)

  def maps(self, row, matrix, half, cols):
    ys, d, lat, zz = row
    edges = []
    for off in (-half, half):
      q = np.column_stack((d, lat + off, zz)) @ matrix.T
      edges.append(q[:, 0]/q[:, 2])
    t = COLUMN_GRID.get(cols)
    if t is None:
      t = (np.arange(cols*ps.SUB) + .5)/(cols*ps.SUB)
    mx = (edges[0][:, None] + t[None]*(edges[1] - edges[0])[:, None]).astype(np.float32)
    key = int(ys[0]), int(ys[-1]), cols
    pair = self.y_maps.get(key)
    if pair is None:
      my = np.repeat(ys[:, None].astype(np.float32), cols*ps.SUB, 1)
      pair = (my, my*.5)
      self.y_maps[key] = pair
      if len(self.y_maps) > MAP_CACHE_SIZE:
        self.y_maps.popitem(last=False)
    else:
      self.y_maps.move_to_end(key)
    return {'mx': mx, 'my': pair[0], 'my_uv': pair[1]}

  def remap(self, source, mx, my, tag):
    return cv2.remap(source, mx, my, cv2.INTER_LINEAR,
                     dst=self.buf(tag, mx.shape + source.shape[2:]), borderMode=cv2.BORDER_REPLICATE)

  def raw(self, y, uv, geo, tag):
    mx, my = geo['mx'], geo['my']
    lum = self.remap(y, mx, my, tag+'y')
    mx_uv = mx*.5
    my_uv = geo['my_uv'] if 'my_uv' in geo else my*.5
    if isinstance(uv, tuple):
      return [lum] + [self.remap(p, mx_uv, my_uv, tag+str(i)) for i, p in enumerate(uv)]
    chroma = self.remap(uv, mx_uv, my_uv, tag+'uv')
    return [lum, chroma[..., 0], chroma[..., 1]]

  def finish(self, planes, cols, rows):
    image = self.buf('image', (len(planes[0]), cols, 3))
    for i, plane in enumerate(planes):
      cast = self.buf('float', plane.shape, np.float32)
      np.copyto(cast, plane, casting='unsafe')
      area = cv2.resize(cast, (cols, len(plane)), dst=self.buf('area', (len(plane), cols), np.float32),
                        interpolation=cv2.INTER_AREA)
      # The box mean is bounded by its uint8 source. Assignment truncates exactly as the original float32 -> uint8 cast.
      image[..., i] = area
    image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA if len(image) > rows else cv2.INTER_LINEAR)
    return image.transpose(2, 0, 1).astype(np.float32)  # caller-owned, never a scratch view

  def pair(self, y, uv, narrow, wide, rows):
    wide_planes = self.raw(y, uv, wide, 'wide')
    wide_output = self.finish(wide_planes, 128, rows)
    narrow_planes = []
    for i, plane in enumerate(wide_planes):
      target = self.buf('narrow'+str(i), (len(plane), 384))
      np.copyto(target, plane[:, 64:448])
      narrow_planes.append(target)
    center = wide['mx'][:, 64:448]
    for scale, indices, name in ((1., (0,), 'y'), (.5, (1, 2), 'uv')):
      a = self.buf('round_n', narrow['mx'].shape, np.float32)
      b = self.buf('round_w', center.shape, np.float32)
      np.multiply(narrow['mx'], 32*scale, out=a)
      np.multiply(center, 32*scale, out=b)
      np.rint(a, out=a)
      np.rint(b, out=b)
      mask = np.not_equal(a, b)
      n = int(mask.sum())
      self.counts['repair_'+name] += n
      if not n:
        continue
      sources = [y] if scale == 1. else uv if isinstance(uv, tuple) else (uv[..., 0], uv[..., 1])
      if n > DENSE_LIMIT:
        self.counts['dense_fallback'] += 1
        for i, src in zip(indices, sources, strict=True):
          narrow_planes[i][:] = self.remap(src, narrow['mx']*scale, narrow['my']*scale, 'patch')
      else:
        mx = (narrow['mx'][mask]*scale).reshape(1, -1)
        my = (narrow['my'][mask]*scale).reshape(1, -1)
        for i, src in zip(indices, sources, strict=True):
          narrow_planes[i][mask] = self.remap(src, mx, my, 'patch').ravel()
    return self.finish(narrow_planes, 96, rows), wide_output

  def extract(self, y, uv, lines, rpy, k, widths=((.75, 96, 128), (1., 128, 128))):
    """Unmirrored strips, width-major then side-major. No line geometry/pixels survive into the next call."""
    layout = y.shape, y.strides, uv.shape, uv.strides, tuple(widths)
    if layout != self.layout:
      self.layout = layout
      self.scratch.clear()
      self.y_maps.clear()
    if self.planar:
      uv = cv2.split(uv, [self.buf('plane0', uv.shape[:2]), self.buf('plane1', uv.shape[:2])])
    if self.geometry:
      key = tuple(rpy), np.asarray(k).tobytes()
      if key != self.key:
        self.key, self.matrix = key, ps.projection(rpy, k)
      common = [self.rows(line, self.matrix, y.shape[0]) for line in lines]
      geos = [[None if row is None else self.maps(row, self.matrix, h, c) for row in common] for h, c, _ in widths]
    else:
      geos = [[ps.strip_geometry(line, rpy, k, y.shape[1], y.shape[0], half_m=h, cols=c) for line in lines] for h, c, _ in widths]
    if self.repair and tuple(widths) == ((.75, 96, 128), (1., 128, 128)):
      result = [[], []]
      for n, w in zip(*geos, strict=True):
        a, b = (None, None) if n is None or w is None else self.pair(y, uv, n, w, 128)
        result[0].append(a)
        result[1].append(b)
      return result[0] + result[1]
    return [None if g is None else self.finish(self.raw(y, uv, g, 'remap'), cols, rows)
            for group, (_, cols, rows) in zip(geos, widths, strict=True) for g in group]
