# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# distutils: language=c++
"""Bit-identical loops for Extractor.maps/finish; the numpy/OpenCV path in strip_extractor.py is the reference."""
from libc.stdint cimport uint8_t

STRIDED_AREA4 = True


def column_map(const double[::1] edge0, const double[::1] edge1, const double[::1] t,
               float[:, ::1] out_mx, float[:, ::1] out_mx_half):
  """out_mx = float32(e0 + t*(e1 - e0)) in numpy's double order; out_mx_half = float32 product out_mx*0.5."""
  cdef Py_ssize_t rows = edge0.shape[0], n = t.shape[0], r, j
  cdef double e0, span, prod, value
  cdef float x
  if edge1.shape[0] != rows or out_mx.shape[0] != rows or out_mx.shape[1] != n or \
     out_mx_half.shape[0] != rows or out_mx_half.shape[1] != n:
    raise ValueError('column_map shape mismatch')
  with nogil:
    for r in range(rows):
      e0 = edge0[r]
      span = edge1[r] - e0
      for j in range(n):
        # Separate statements and -ffp-contract=off: no fused multiply-add, same as numpy's two ufunc passes.
        prod = t[j]*span
        value = e0 + prod
        x = <float>value
        out_mx[r, j] = x
        out_mx_half[r, j] = x*<float>0.5


def area4_interleave(const uint8_t[:, :] p0, const uint8_t[:, :] p1, const uint8_t[:, :] p2,
                     uint8_t[:, :, ::1] out):
  """out[r, c, i] = floor(mean of plane i's 4 samples 4c..4c+3), the exact 4:1 INTER_AREA float mean truncated to uint8."""
  cdef Py_ssize_t rows = out.shape[0], cols = out.shape[1], r, c, k
  if out.shape[2] != 3:
    raise ValueError('area4_interleave output must have 3 channels')
  if p0.shape[0] != rows or p1.shape[0] != rows or p2.shape[0] != rows or \
     p0.shape[1] != 4*cols or p1.shape[1] != 4*cols or p2.shape[1] != 4*cols:
    raise ValueError('area4_interleave plane shape mismatch')
  with nogil:
    for r in range(rows):
      for c in range(cols):
        k = 4*c
        out[r, c, 0] = <uint8_t>((<unsigned int>p0[r, k] + p0[r, k + 1] + p0[r, k + 2] + p0[r, k + 3]) >> 2)
        out[r, c, 1] = <uint8_t>((<unsigned int>p1[r, k] + p1[r, k + 1] + p1[r, k + 2] + p1[r, k + 3]) >> 2)
        out[r, c, 2] = <uint8_t>((<unsigned int>p2[r, k] + p2[r, k + 1] + p2[r, k + 2] + p2[r, k + 3]) >> 2)
