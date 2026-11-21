//! The 4.5-20 m row-aligned YCbCr strip (strip_extractor.Extractor / profile_scanner.strip_geometry), bit for bit.
//!
//! Each numpy / OpenCV step is reproduced with its own arithmetic: numpy's arange fill and interp, OpenCV's
//! INTER_LINEAR remap on 8-bit images (1/32 pixel map rounding, 2^15 weights, BORDER_REPLICATE), the exact 4:1 column
//! box mean truncated to uint8, and the row resize (INTER_AREA table for more than `rows` source rows, INTER_LINEAR's
//! 11-bit fixed point for fewer). The OpenCV rules were matched against cv2 4.13 (the M8 replay); the device's OpenCV
//! build (KleidiCV) is not checked here.

pub const NEAR_M: f64 = 4.5;
pub const FAR_M: f64 = 20.;
pub const SUB: usize = 4;
pub const HOOD_FRACTION: f64 = 0.86;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StripShape {
    pub half_m: f64,
    pub cols: usize,
    pub rows: usize,
}

pub const DEFAULT_STRIP: StripShape = StripShape { half_m: 1., cols: 128, rows: 128 };

/// A lane line as the model gives it (float32 values read as Python floats).
#[derive(Clone, Debug, Default)]
pub struct LineF64 {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// One NV12 frame: the Y plane and the interleaved CbCr plane, each with its row stride.
pub struct Nv12<'a> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub y: &'a [u8],
    pub uv: &'a [u8],
}

impl<'a> Nv12<'a> {
    /// The VisionIPC buffer layout: Y rows of `stride` bytes, CbCr from `uv_offset` (marking_daemon.planes).
    pub fn from_buffer(data: &'a [u8], width: usize, height: usize, stride: usize, uv_offset: usize) -> Option<Self> {
        if width % 2 != 0 || height % 2 != 0 || stride < width || data.len() < uv_offset + stride * (height / 2) || uv_offset < stride * height {
            return None;
        }
        Some(Nv12 { width, height, stride, y: &data[..stride * height], uv: &data[uv_offset..uv_offset + stride * (height / 2)] })
    }
}

type Mat3 = [[f64; 3]; 3];

fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

/// profile_scanner.projection: K @ view_frame_from_device_frame @ rot_from_euler(rpy).
pub fn projection(rpy: [f64; 3], k: &Mat3) -> Mat3 {
    let (cx, sx) = (rpy[0].cos(), rpy[0].sin());
    let (cy, sy) = (rpy[1].cos(), rpy[1].sin());
    let (cz, sz) = (rpy[2].cos(), rpy[2].sin());
    let rx = [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]];
    let ry = [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]];
    let rz = [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]];
    let rot = matmul(&matmul(&rz, &ry), &rx);
    let view = [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]; // device_frame_from_view_frame.T
    matmul(&matmul(k, &view), &rot)
}

fn project(m: &Mat3, p: [f64; 3], r: usize) -> f64 {
    p[0] * m[r][0] + p[1] * m[r][1] + p[2] * m[r][2]
}

/// numpy.arange(start, stop, step) for float64 (length from the ceil, values start + i*(buffer[1] - start)).
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil().max(0.) as usize;
    let mut out = Vec::with_capacity(len);
    if len > 0 {
        out.push(start);
    }
    if len > 1 {
        out.push(start + step);
    }
    let delta = start + step - start;
    for i in 2..len {
        out.push(start + i as f64 * delta);
    }
    out
}

pub fn distances() -> Vec<f64> {
    arange(NEAR_M, FAR_M + 0.001, 0.02)
}

/// numpy.interp for increasing `xp` (binary search, x == xp[j] gives fp[j], slope*(x - xp[j]) + fp[j]).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    // The largest j with xp[j] <= x.
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        let r2 = slope * (x - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

/// The integer source rows of a line (Extractor.rows): (ys, d, lat, zz), or None when its geometry is unusable.
pub struct Rows {
    pub ys: Vec<i64>,
    pub d: Vec<f64>,
    pub lat: Vec<f64>,
    pub zz: Vec<f64>,
}

pub fn rows(line: &LineF64, m: &Mat3, height: usize, dist: &[f64]) -> Option<Rows> {
    let (x, y, z) = (&line.x, &line.y, &line.z);
    if x.len() < 3 || y.len() != x.len() || z.len() != x.len() || !x.iter().all(|v| v.is_finite())
        || x.windows(2).any(|w| w[1] - w[0] <= 0.) || x[0] > NEAR_M || x[x.len() - 1] < FAR_M + 1. {
        return None;
    }
    let mut v = Vec::with_capacity(dist.len());
    for &d in dist {
        let p = [d, interp(d, x, y), interp(d, x, z)];
        let p2 = project(m, p, 2);
        if p2 <= 0.1 {
            return None;
        }
        v.push(project(m, p, 1) / p2);
    }
    if v.windows(2).any(|w| w[1] - w[0] >= 0.) {
        return None;
    }
    let top = (v[v.len() - 1].ceil() as i64).max(0);
    let hood = (height as f64 * HOOD_FRACTION) as i64 - 1;
    let bottom = (v[0].floor() as i64).min(hood).min(height as i64 - 1);
    if bottom - top < 8 {
        return None;
    }
    let vr: Vec<f64> = v.iter().rev().copied().collect();
    let dr: Vec<f64> = dist.iter().rev().copied().collect();
    let ys: Vec<i64> = (top..=bottom).collect();
    let d: Vec<f64> = ys.iter().map(|&r| interp(r as f64, &vr, &dr)).collect();
    let lat = d.iter().map(|&q| interp(q, x, y)).collect();
    let zz = d.iter().map(|&q| interp(q, x, z)).collect();
    Some(Rows { ys, d, lat, zz })
}

/// Column sample positions (luma and chroma maps) of one strip: rows x cols*SUB float32.
pub struct Maps {
    pub rows: usize,
    pub n: usize,
    pub mx: Vec<f32>,
    pub mx_uv: Vec<f32>,
    pub ys: Vec<i64>,
}

pub fn maps(r: &Rows, m: &Mat3, half_m: f64, cols: usize) -> Maps {
    let n = cols * SUB;
    let t: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let mut mx = Vec::with_capacity(r.ys.len() * n);
    for i in 0..r.ys.len() {
        let edge = |off: f64| {
            let p = [r.d[i], r.lat[i] + off, r.zz[i]];
            project(m, p, 0) / project(m, p, 2)
        };
        let (e0, e1) = (edge(-half_m), edge(half_m));
        let span = e1 - e0;
        mx.extend(t.iter().map(|&tj| (e0 + tj * span) as f32));
    }
    let mx_uv = mx.iter().map(|&v| v * 0.5).collect();
    Maps { rows: r.ys.len(), n, mx, mx_uv, ys: r.ys.clone() }
}

const TAB_BITS: i32 = 5;
const TAB_SIZE: i32 = 1 << TAB_BITS;
const COEF_BITS: i32 = 15;

/// OpenCV's INTER_LINEAR 2-D weight table for 8-bit remap: (1-a)(1-b), a(1-b), (1-a)b, ab at a, b = i/32, times 2^15.
/// Every product is a multiple of 1/1024, so all entries are exact integers summing to 2^15 (no OpenCV sum fix-up).
fn remap_weights() -> [[i32; 4]; 1024] {
    let mut out = [[0; 4]; 1024];
    let scale = 1. / TAB_SIZE as f32;
    for fy in 0..TAB_SIZE {
        for fx in 0..TAB_SIZE {
            let (a, b) = (fx as f32 * scale, fy as f32 * scale);
            let w = [(1. - b) * (1. - a), (1. - b) * a, b * (1. - a), b * a];
            out[(fy * TAB_SIZE + fx) as usize] = w.map(|v| (v * (1 << COEF_BITS) as f32).round_ties_even() as i32);
        }
    }
    out
}

/// cv2.remap(plane, mx, my, INTER_LINEAR, borderMode=BORDER_REPLICATE) of one channel of an 8-bit plane.
/// `channel`/`step`: the byte offset and element step inside a row (interleaved CbCr: 0 or 1, step 2).
#[allow(clippy::too_many_arguments)]
fn remap(plane: &[u8], width: usize, height: usize, stride: usize, channel: usize, step: usize, mx: &[f32], my_row: &[f32],
         n: usize, weights: &[[i32; 4]; 1024], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if simd() && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is present (checked above); every gather is bounds-checked per row inside.
        unsafe { remap_avx2(plane, width, height, stride, channel, step, mx, my_row, n, weights, out) };
        return;
    }
    remap_scalar(plane, width, height, stride, channel, step, mx, my_row, n, weights, out, 0..my_row.len())
}

#[allow(clippy::too_many_arguments)]
fn remap_scalar(plane: &[u8], width: usize, height: usize, stride: usize, channel: usize, step: usize, mx: &[f32], my_row: &[f32],
                n: usize, weights: &[[i32; 4]; 1024], out: &mut [u8], rows: std::ops::Range<usize>) {
    let (w1, h1) = (width as i32 - 1, height as i32 - 1);
    let px = |x: i32, y: i32| plane[y as usize * stride + x as usize * step + channel] as i32;
    for r in rows {
        let my = my_row[r];
        let yy = (my * TAB_SIZE as f32).round_ties_even() as i32;
        let (y0, fy) = (yy >> TAB_BITS, yy & (TAB_SIZE - 1));
        let row = &mx[r * n..(r + 1) * n];
        let dst = &mut out[r * n..(r + 1) * n];
        for (j, &m) in row.iter().enumerate() {
            let xx = (m * TAB_SIZE as f32).round_ties_even() as i32;
            let (x0, fx) = (xx >> TAB_BITS, xx & (TAB_SIZE - 1));
            let w = &weights[(fy * TAB_SIZE + fx) as usize];
            let (xa, xb) = (x0.clamp(0, w1), (x0 + 1).clamp(0, w1));
            let (ya, yb) = (y0.clamp(0, h1), (y0 + 1).clamp(0, h1));
            let v = px(xa, ya) * w[0] + px(xb, ya) * w[1] + px(xa, yb) * w[2] + px(xb, yb) * w[3];
            dst[j] = ((v + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8;
        }
    }
}

/// remap() with AVX2, 8 columns at a time: the same 1/32 rounding (cvtps_epi32 rounds to nearest even, as
/// round_ties_even) and integer weights computed as (32-fx)(32-fy)*32 etc. — the remap_weights() entries exactly.
/// Chunks with a coordinate beyond +-1e6 or NaN (where `as i32` saturates) and rows whose 4-byte gathers could pass
/// the end of the plane go through the scalar form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn remap_avx2(plane: &[u8], width: usize, height: usize, stride: usize, channel: usize, step: usize, mx: &[f32], my_row: &[f32],
                     n: usize, weights: &[[i32; 4]; 1024], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let (w1, h1) = (width as i32 - 1, height as i32 - 1);
    let full = n / 8 * 8;
    let v32 = _mm256_set1_ps(TAB_SIZE as f32);
    let lim = _mm256_set1_ps(1e6);
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let (zero, vw1, one) = (_mm256_setzero_si256(), _mm256_set1_epi32(w1), _mm256_set1_epi32(1));
    let (mask31, c32, byte) = (_mm256_set1_epi32(TAB_SIZE - 1), _mm256_set1_epi32(TAB_SIZE), _mm256_set1_epi32(0xff));
    let vstep = _mm256_set1_epi32(step as i32);
    let half = _mm256_set1_epi32(1 << (COEF_BITS - 1));
    for (r, &my) in my_row.iter().enumerate() {
        let yy = (my * TAB_SIZE as f32).round_ties_even() as i32;
        let (y0, fy) = (yy >> TAB_BITS, yy & (TAB_SIZE - 1));
        let (ya, yb) = (y0.clamp(0, h1) as usize, (y0 + 1).clamp(0, h1) as usize);
        let (ra, rb) = (ya * stride + channel, yb * stride + channel);
        if ra.max(rb) + w1 as usize * step + 4 > plane.len() || stride * height > i32::MAX as usize {
            remap_scalar(plane, width, height, stride, channel, step, mx, my_row, n, weights, out, r..r + 1);
            continue;
        }
        let vfy = _mm256_set1_epi32(fy);
        let vfy1 = _mm256_sub_epi32(c32, vfy);
        let (pa, pb) = (plane.as_ptr().add(ra) as *const i32, plane.as_ptr().add(rb) as *const i32);
        let row = &mx[r * n..(r + 1) * n];
        let dst = &mut out[r * n..(r + 1) * n];
        let mut j = 0;
        while j < full {
            let m = _mm256_loadu_ps(row.as_ptr().add(j));
            let ok = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(m, absmask), lim));
            if ok != 0xff {
                for (k, d) in dst[j..j + 8].iter_mut().enumerate() {
                    *d = remap_px(plane, stride, channel, step, w1, ya, yb, fy, row[j + k], weights);
                }
                j += 8;
                continue;
            }
            let xx = _mm256_cvtps_epi32(_mm256_mul_ps(m, v32));
            let x0 = _mm256_srai_epi32::<TAB_BITS>(xx);
            let fx = _mm256_and_si256(xx, mask31);
            let xa = _mm256_min_epi32(_mm256_max_epi32(x0, zero), vw1);
            let xb = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(x0, one), zero), vw1);
            let (oa, ob) = (_mm256_mullo_epi32(xa, vstep), _mm256_mullo_epi32(xb, vstep));
            let p00 = _mm256_and_si256(_mm256_i32gather_epi32::<1>(pa, oa), byte);
            let p01 = _mm256_and_si256(_mm256_i32gather_epi32::<1>(pa, ob), byte);
            let p10 = _mm256_and_si256(_mm256_i32gather_epi32::<1>(pb, oa), byte);
            let p11 = _mm256_and_si256(_mm256_i32gather_epi32::<1>(pb, ob), byte);
            let fx1 = _mm256_sub_epi32(c32, fx);
            let w0 = _mm256_slli_epi32::<5>(_mm256_mullo_epi32(fx1, vfy1));
            let w1v = _mm256_slli_epi32::<5>(_mm256_mullo_epi32(fx, vfy1));
            let w2 = _mm256_slli_epi32::<5>(_mm256_mullo_epi32(fx1, vfy));
            let w3 = _mm256_slli_epi32::<5>(_mm256_mullo_epi32(fx, vfy));
            let v = _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p00, w0), _mm256_mullo_epi32(p01, w1v)),
                                     _mm256_add_epi32(_mm256_mullo_epi32(p10, w2), _mm256_mullo_epi32(p11, w3)));
            let v = _mm256_srai_epi32::<COEF_BITS>(_mm256_add_epi32(v, half));
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
            for (d, &l) in dst[j..j + 8].iter_mut().zip(&lanes) {
                *d = l.clamp(0, 255) as u8;
            }
            j += 8;
        }
        for jj in full..n {
            dst[jj] = remap_px(plane, stride, channel, step, w1, ya, yb, fy, row[jj], weights);
        }
    }
}

/// One output pixel of remap() at the clamped rows ya / yb (the AVX2 path's fallback).
#[allow(clippy::too_many_arguments, dead_code)]
#[inline]
fn remap_px(plane: &[u8], stride: usize, channel: usize, step: usize, w1: i32, ya: usize, yb: usize, fy: i32, m: f32,
            weights: &[[i32; 4]; 1024]) -> u8 {
    let xx = (m * TAB_SIZE as f32).round_ties_even() as i32;
    let (x0, fx) = (xx >> TAB_BITS, xx & (TAB_SIZE - 1));
    let w = &weights[(fy * TAB_SIZE + fx) as usize];
    let (xa, xb) = (x0.clamp(0, w1) as usize, (x0 + 1).clamp(0, w1) as usize);
    let px = |x: usize, y: usize| plane[y * stride + x * step + channel] as i32;
    let v = px(xa, ya) * w[0] + px(xb, ya) * w[1] + px(xa, yb) * w[2] + px(xb, yb) * w[3];
    ((v + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8
}

/// cv2.resize INTER_AREA's per-axis table (computeResizeAreaTab): (dst index, src index, weight).
fn area_tab(ssize: usize, dsize: usize) -> Vec<(usize, usize, f32)> {
    let scale = 1. / (dsize as f64 / ssize as f64);
    let mut tab = Vec::new();
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let sx2 = (fsx2.floor() as i64).min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push((dx, (sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            tab.push((dx, sx as usize, (1. / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push((dx, sx2 as usize, ((fsx2 - sx2 as f64).min(1.).min(cell) / cell) as f32));
        }
    }
    tab
}

/// Row resize of an (src_rows, width) 8-bit image to `dst_rows` rows (column count unchanged), as cv2.resize with
/// INTER_AREA when shrinking and INTER_LINEAR otherwise (equal sizes: a copy).
pub fn resize_rows(src: &[u8], src_rows: usize, width: usize, dst_rows: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst_rows * width];
    if src_rows == dst_rows {
        out.copy_from_slice(&src[..dst_rows * width]);
    } else if src_rows > dst_rows {
        let mut sum = vec![0f32; width];
        let mut prev: Option<usize> = None;
        for (dy, sy, beta) in area_tab(src_rows, dst_rows) {
            let row = &src[sy * width..(sy + 1) * width];
            if prev != Some(dy) {
                if let Some(p) = prev {
                    for (o, s) in out[p * width..(p + 1) * width].iter_mut().zip(&sum) {
                        *o = s.round_ties_even().clamp(0., 255.) as u8;
                    }
                }
                for (s, &v) in sum.iter_mut().zip(row) {
                    *s = beta * v as f32;
                }
                prev = Some(dy);
            } else {
                for (s, &v) in sum.iter_mut().zip(row) {
                    *s += beta * v as f32;
                }
            }
        }
        if let Some(p) = prev {
            for (o, s) in out[p * width..(p + 1) * width].iter_mut().zip(&sum) {
                *o = s.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    } else {
        let scale = 1. / (dst_rows as f64 / src_rows as f64);
        for dy in 0..dst_rows {
            let mut fy = ((dy as f64 + 0.5) * scale - 0.5) as f32;
            let mut sy = fy.floor() as i64;
            fy -= sy as f32;
            if sy < 0 {
                fy = 0.;
                sy = 0;
            }
            if sy >= src_rows as i64 - 1 {
                fy = 0.;
                sy = src_rows as i64 - 1;
            }
            let b0 = ((1. - fy) * 2048.).round_ties_even() as i32;
            let b1 = (fy * 2048.).round_ties_even() as i32;
            let s0 = &src[sy as usize * width..(sy as usize + 1) * width];
            let s1 = &src[((sy + 1) as usize).min(src_rows - 1) * width..][..width];
            for x in 0..width {
                let (a, b) = ((s0[x] as i32) << 11, (s1[x] as i32) << 11);
                let v = (((b0 * (a >> 4)) >> 16) + ((b1 * (b >> 4)) >> 16) + 2) >> 2;
                out[dy * width + x] = v.clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// The 3 x rows x cols float32 strip of one line (`mirror`: columns reversed), or None without usable geometry.
pub fn extract(frame: &Nv12, line: &LineF64, m: &Mat3, shape: StripShape, mirror: bool, dist: &[f64]) -> Option<Vec<f32>> {
    extract_timed(frame, line, m, shape, mirror, dist, &mut Stages::default())
}

/// Wall time of the extraction stages of one strip, in nanoseconds (added to `st`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stages {
    /// rows(): projecting the line at the 776 distances and the per-row interp.
    pub rows: u64,
    /// maps(): the per-row edge projections and the float32 column maps (luma and chroma).
    pub maps: u64,
    /// The luma and CbCr remaps (the backend's work).
    pub remap: u64,
    /// The 4:1 column mean, the row resize and the float32 (mirrored) output.
    pub reduce: u64,
    /// Source rows sampled (strip height before the resize).
    pub src_rows: u64,
    /// Strips whose rows / maps came from the MapCache.
    pub cache_hits: u64,
}

/// The last rows()/maps() result of one side, reused while its inputs are bit-identical: the lane line's x / y / z,
/// the projection matrix (calibration rpy and intrinsics), the frame height and the strip shape (the distances are a
/// constant). Any change in any of these bits is a miss and replaces the entry. STRIPNET_MAP_CACHE=1 turns it on.
#[derive(Default)]
pub struct MapCache {
    key: Vec<u64>,
    scratch: Vec<u64>,
    maps: Option<Maps>,
    filled: bool,
}

pub fn map_cache_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("STRIPNET_MAP_CACHE").as_deref() == Ok("1"))
}

fn cache_key(line: &LineF64, m: &Mat3, height: usize, shape: StripShape, key: &mut Vec<u64>) {
    key.clear();
    key.extend([height as u64, shape.half_m.to_bits(), shape.cols as u64, shape.rows as u64]);
    key.extend(m.iter().flatten().map(|v| v.to_bits()));
    for v in [&line.x, &line.y, &line.z] {
        key.push(v.len() as u64);
        key.extend(v.iter().map(|x| x.to_bits()));
    }
}

impl Stages {
    pub fn add(&mut self, o: &Stages) {
        self.rows += o.rows;
        self.maps += o.maps;
        self.remap += o.remap;
        self.reduce += o.reduce;
        self.src_rows += o.src_rows;
        self.cache_hits += o.cache_hits;
    }
}

fn ns(t: std::time::Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

#[allow(clippy::too_many_arguments)]
pub fn extract_timed(frame: &Nv12, line: &LineF64, m: &Mat3, shape: StripShape, mirror: bool, dist: &[f64],
                     st: &mut Stages) -> Option<Vec<f32>> {
    extract_cached(frame, line, m, shape, mirror, dist, st, None)
}

#[allow(clippy::too_many_arguments)]
pub fn extract_cached(frame: &Nv12, line: &LineF64, m: &Mat3, shape: StripShape, mirror: bool, dist: &[f64],
                      st: &mut Stages, cache: Option<&mut MapCache>) -> Option<Vec<f32>> {
    if let Some(c) = cache {
        let t = std::time::Instant::now();
        let mut key = std::mem::take(&mut c.scratch);
        cache_key(line, m, frame.height, shape, &mut key);
        let hit = c.filled && key == c.key;
        if hit {
            st.cache_hits += 1;
            st.maps += ns(t);
        } else {
            let r = rows(line, m, frame.height, dist);
            st.rows += ns(t);
            let t = std::time::Instant::now();
            c.maps = r.map(|r| maps(&r, m, shape.half_m, shape.cols));
            c.filled = true;
            st.maps += ns(t);
        }
        c.scratch = std::mem::replace(&mut c.key, key);
        let g = c.maps.as_ref()?;
        return Some(sample_timed(frame, g, shape, mirror, remap_rule(), backend(), st));
    }
    let t = std::time::Instant::now();
    let r = rows(line, m, frame.height, dist);
    st.rows += ns(t);
    let r = r?;
    let t = std::time::Instant::now();
    let g = maps(&r, m, shape.half_m, shape.cols);
    st.maps += ns(t);
    Some(sample_timed(frame, &g, shape, mirror, remap_rule(), backend(), st))
}

/// Which OpenCV remap the Python path meets: the generic 8-bit fixed-point one (x86 / M8 cv2), or KleidiCV's float
/// remap (the device's cv2 4.13 + KleidiCV 0.7.0 HAL takes 1- and 2-channel 8-bit INTER_LINEAR maps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemapRule {
    OpenCv,
    /// `fused`: vmlaq_f32 compiled to a fused multiply-add (GCC/clang contract it on AArch64), else mul then add.
    KleidiCv { fused: bool },
}

/// The rule for this build: KleidiCV (fused) on aarch64, the OpenCV one elsewhere; STRIPNET_REMAP=opencv|kleidicv|kleidicv-unfused
/// overrides it (replays on M8).
pub fn remap_rule() -> RemapRule {
    static RULE: std::sync::OnceLock<RemapRule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("STRIPNET_REMAP").as_deref() {
        Ok("opencv") => RemapRule::OpenCv,
        Ok("kleidicv") => RemapRule::KleidiCv { fused: true },
        Ok("kleidicv-unfused") => RemapRule::KleidiCv { fused: false },
        _ if cfg!(target_arch = "aarch64") => RemapRule::KleidiCv { fused: true },
        _ => RemapRule::OpenCv,
    })
}

/// Whether the remaps use SIMD: NEON for the KleidiCV rule (aarch64), AVX2 for the OpenCV rule (x86_64, when the CPU
/// has it). Both are bit-identical to the scalar forms (tests `neon_matches_scalar`, `avx2_matches_scalar`).
/// STRIPNET_SIMD=0|1; unset: the build default (`SIMD_DEFAULT`).
pub fn simd() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| match std::env::var("STRIPNET_SIMD").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => SIMD_DEFAULT && std::env::var_os("STRIPNET_SCALAR").is_none(),
    })
}

/// On everywhere: bit-identical, and the remap stage drops 3.4 -> 1.6 ms on M8 (AVX2), 23.0 -> 17.4 ms on the device
/// (NEON) in the 2026-11-21 combo bench.
pub const SIMD_DEFAULT: bool = true;

/// Who computes the KleidiCV-rule remap (RemapRule::KleidiCv): this crate's NEON / scalar code, KleidiCV's own
/// `kleidicv_remap_f32_u8` linked statically (feature `kleidicv`), or OpenCV's `cv::remap` built with the KleidiCV HAL
/// (feature `opencv`, the call cv2.remap makes on the device). STRIPNET_BACKEND=rust|kleidicv|opencv; default kleidicv
/// when the binary has the `kleidicv` feature (remap stage 7.4 ms against the crate's NEON 17.4 ms on the device,
/// same bits), else rust.
/// A backend that is not compiled in, or that returns an error, falls back to rust (counted in `native::fallbacks`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Rust,
    KleidiCv,
    OpenCv,
}

pub fn backend() -> Backend {
    static B: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
    *B.get_or_init(|| match std::env::var("STRIPNET_BACKEND").as_deref() {
        Ok("kleidicv") => Backend::KleidiCv,
        Ok("opencv") => Backend::OpenCv,
        Ok("rust") => Backend::Rust,
        _ if cfg!(feature = "kleidicv") => Backend::KleidiCv,
        _ => Backend::Rust,
    })
}

/// KleidiCV remap_f32 (u8, INTER_LINEAR, BORDER_REPLICATE) of an image with `outs.len()` interleaved channels (1 or 2;
/// one set of coordinates for all of them, as KleidiCV's 2-channel path): x0 = min(floor(x) saturated to u32, xmax),
/// x1 = x0 + 1 only when x >= 0 and x0 < xmax; frac = x - floor(x); pixels as f32; line0 = a + (b - a) * xfrac,
/// line1 likewise, result = line0 + (line1 - line0) * yfrac, rounded to nearest with ties away (vcvtaq_u32_f32).
/// aarch64: NEON for 4 pixels at a time (`remap_kleidi_neon`), bit-identical to this scalar form (test `neon_matches_scalar`).
#[allow(clippy::too_many_arguments)]
fn remap_kleidi(plane: &[u8], width: usize, height: usize, stride: usize, mx: &[f32], my_row: &[f32], n: usize, fused: bool,
                outs: &mut [&mut [u8]]) {
    #[cfg(target_arch = "aarch64")]
    if simd() {
        // SAFETY: NEON is part of the aarch64 baseline; all indices are bounds-checked inside.
        unsafe { remap_kleidi_neon(plane, width, height, stride, mx, my_row, n, fused, outs) };
        return;
    }
    remap_kleidi_scalar(plane, width, height, stride, mx, my_row, n, fused, outs, 0)
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32, fused: bool) -> f32 {
    if fused { (b - a).mul_add(t, a) } else { a + (b - a) * t }
}

/// Columns `from..n` of every row (the NEON path's tail).
#[allow(clippy::too_many_arguments)]
fn remap_kleidi_scalar(plane: &[u8], width: usize, height: usize, stride: usize, mx: &[f32], my_row: &[f32], n: usize,
                       fused: bool, outs: &mut [&mut [u8]], from: usize) {
    let (xmax, ymax) = (width as u32 - 1, height as u32 - 1);
    let step = outs.len();
    for (r, &yf) in my_row.iter().enumerate() {
        let y0 = (yf.floor() as u32).min(ymax);
        let y1 = if yf >= 0. && y0 < ymax { y0 + 1 } else { y0 };
        let yfrac = yf - yf.floor();
        let (row0, row1) = (&plane[y0 as usize * stride..], &plane[y1 as usize * stride..]);
        for j in from..n {
            let xf = mx[r * n + j];
            let x0 = (xf.floor() as u32).min(xmax);
            let x1 = if xf >= 0. && x0 < xmax { x0 + 1 } else { x0 };
            let xfrac = xf - xf.floor();
            for (c, out) in outs.iter_mut().enumerate() {
                let px = |row: &[u8], x: u32| row[x as usize * step + c] as f32;
                let line0 = lerp(px(row0, x0), px(row0, x1), xfrac, fused);
                let line1 = lerp(px(row1, x0), px(row1, x1), xfrac, fused);
                out[r * n + j] = lerp(line0, line1, yfrac, fused).round() as u32 as u8;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn remap_kleidi_neon(plane: &[u8], width: usize, height: usize, stride: usize, mx: &[f32], my_row: &[f32], n: usize,
                            fused: bool, outs: &mut [&mut [u8]]) {
    use std::arch::aarch64::*;
    let (xmax, ymax) = (width as u32 - 1, height as u32 - 1);
    let step = outs.len();
    let vxmax = vdupq_n_u32(xmax);
    let zero = vdupq_n_f32(0.);
    let mla = |a: float32x4_t, b: float32x4_t, t: float32x4_t| -> float32x4_t {
        let d = vsubq_f32(b, a);
        if fused { vfmaq_f32(a, d, t) } else { vaddq_f32(a, vmulq_f32(d, t)) }
    };
    let full = n / 4 * 4;
    let (mut i0, mut i1) = ([0u32; 4], [0u32; 4]);
    let mut q = [[0f32; 4]; 4];
    for (r, &yf) in my_row.iter().enumerate() {
        let y0 = (yf.floor() as u32).min(ymax);
        let y1 = if yf >= 0. && y0 < ymax { y0 + 1 } else { y0 };
        let vyfrac = vdupq_n_f32(yf - yf.floor());
        let (row0, row1) = (&plane[y0 as usize * stride..], &plane[y1 as usize * stride..]);
        let mrow = &mx[r * n..(r + 1) * n];
        let mut j = 0;
        while j < full {
            let xf = vld1q_f32(mrow[j..j + 4].as_ptr());
            let x0 = vminq_u32(vcvtmq_u32_f32(xf), vxmax);
            let inside = vandq_u32(vcgeq_f32(xf, zero), vcltq_u32(x0, vxmax));
            let x1 = vsubq_u32(x0, inside);
            let xfrac = vsubq_f32(xf, vrndmq_f32(xf));
            vst1q_u32(i0.as_mut_ptr(), x0);
            vst1q_u32(i1.as_mut_ptr(), x1);
            for (c, out) in outs.iter_mut().enumerate() {
                for k in 0..4 {
                    let (a0, a1) = (i0[k] as usize * step + c, i1[k] as usize * step + c);
                    q[0][k] = row0[a0] as f32;
                    q[1][k] = row0[a1] as f32;
                    q[2][k] = row1[a0] as f32;
                    q[3][k] = row1[a1] as f32;
                }
                let line0 = mla(vld1q_f32(q[0].as_ptr()), vld1q_f32(q[1].as_ptr()), xfrac);
                let line1 = mla(vld1q_f32(q[2].as_ptr()), vld1q_f32(q[3].as_ptr()), xfrac);
                let v = vcvtaq_u32_f32(mla(line0, line1, vyfrac));
                let mut lanes = [0u32; 4];
                vst1q_u32(lanes.as_mut_ptr(), v);
                let dst = &mut out[r * n + j..r * n + j + 4];
                for k in 0..4 {
                    dst[k] = lanes[k] as u8;
                }
            }
            j += 4;
        }
    }
    if full < n {
        remap_kleidi_scalar(plane, width, height, stride, mx, my_row, n, fused, outs, full);
    }
}

pub fn sample(frame: &Nv12, g: &Maps, shape: StripShape, mirror: bool) -> Vec<f32> {
    sample_with(frame, g, shape, mirror, remap_rule())
}

pub fn sample_with(frame: &Nv12, g: &Maps, shape: StripShape, mirror: bool, rule: RemapRule) -> Vec<f32> {
    sample_timed(frame, g, shape, mirror, rule, backend(), &mut Stages::default())
}

fn weights() -> &'static [[i32; 4]; 1024] {
    static W: std::sync::OnceLock<[[i32; 4]; 1024]> = std::sync::OnceLock::new();
    W.get_or_init(remap_weights)
}

pub fn sample_timed(frame: &Nv12, g: &Maps, shape: StripShape, mirror: bool, rule: RemapRule, backend: Backend,
                    st: &mut Stages) -> Vec<f32> {
    let t = std::time::Instant::now();
    let (rows, n, cols) = (g.rows, g.n, shape.cols);
    let my: Vec<f32> = g.ys.iter().map(|&y| y as f32).collect();
    let my_uv: Vec<f32> = my.iter().map(|&v| v * 0.5).collect();
    let mut planes = [vec![0u8; rows * n], vec![0u8; rows * n], vec![0u8; rows * n]];
    let [lum, cb, cr] = &mut planes;
    let (cw, ch) = (frame.width / 2, frame.height / 2);
    match rule {
        RemapRule::OpenCv => {
            let w = weights();
            remap(frame.y, frame.width, frame.height, frame.stride, 0, 1, &g.mx, &my, n, w, lum);
            remap(frame.uv, cw, ch, frame.stride, 0, 2, &g.mx_uv, &my_uv, n, w, cb);
            remap(frame.uv, cw, ch, frame.stride, 1, 2, &g.mx_uv, &my_uv, n, w, cr);
        }
        RemapRule::KleidiCv { fused } => {
            let native = backend != Backend::Rust
                && crate::native::remap(backend, frame.y, frame.width, frame.height, frame.stride, &g.mx, &my, n, &mut [lum])
                && crate::native::remap(backend, frame.uv, cw, ch, frame.stride, &g.mx_uv, &my_uv, n, &mut [cb, cr]);
            if !native {
                remap_kleidi(frame.y, frame.width, frame.height, frame.stride, &g.mx, &my, n, fused, &mut [lum]);
                remap_kleidi(frame.uv, cw, ch, frame.stride, &g.mx_uv, &my_uv, n, fused, &mut [cb, cr]);
            }
        }
    }
    st.remap += ns(t);
    st.src_rows += rows as u64;
    let t = std::time::Instant::now();
    let mut out = vec![0f32; 3 * shape.rows * cols];
    let mut boxed = vec![0u8; rows * cols];
    for (c, plane) in planes.iter().enumerate() {
        // Exact 4:1 column box mean truncated to uint8.
        for (o, q) in boxed.iter_mut().zip(plane.chunks_exact(SUB)) {
            *o = ((q[0] as u32 + q[1] as u32 + q[2] as u32 + q[3] as u32) >> 2) as u8;
        }
        let resized = resize_rows(&boxed, rows, cols, shape.rows);
        let dst = &mut out[c * shape.rows * cols..(c + 1) * shape.rows * cols];
        for (row_out, row_in) in dst.chunks_exact_mut(cols).zip(resized.chunks_exact(cols)) {
            if mirror {
                for (o, &v) in row_out.iter_mut().zip(row_in.iter().rev()) {
                    *o = v as f32;
                }
            } else {
                for (o, &v) in row_out.iter_mut().zip(row_in) {
                    *o = v as f32;
                }
            }
        }
    }
    st.reduce += ns(t);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_is_numpy() {
        let d = distances();
        assert_eq!(d.len(), 776);
        assert_eq!(d[0], 4.5);
        assert_eq!(d[1], 4.52);
        let delta = 4.52 - 4.5;
        assert_eq!(d[775], 4.5 + 775. * delta);
    }

    #[test]
    fn interp_edges() {
        let xp = [0., 1., 2.];
        let fp = [10., 20., 40.];
        assert_eq!(interp(-1., &xp, &fp), 10.);
        assert_eq!(interp(3., &xp, &fp), 40.);
        assert_eq!(interp(1., &xp, &fp), 20.);
        assert_eq!(interp(2., &xp, &fp), 40.);
        assert_eq!(interp(1.5, &xp, &fp), 30.);
    }

    /// The KleidiCV remap's NEON path against its scalar form (aarch64; on x86 both calls are the scalar one).
    #[test]
    fn neon_matches_scalar() {
        let (w, h) = (61usize, 37usize);
        let plane: Vec<u8> = (0..w * h * 2).map(|i| ((i * 7919) % 256) as u8).collect();
        let n = 30; // not a multiple of 4: the tail too
        let mut seed = 12345u32;
        let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed as f32 / u32::MAX as f32 };
        let rows = 9;
        let mx: Vec<f32> = (0..rows * n).map(|_| rnd() * 40. - 4. + 0.5).collect();
        let my: Vec<f32> = (0..rows).map(|i| [-2.5, -0.25, 0., 3.5, 17.25, 35.75, 36., 36.5, 50.][i]).collect();
        let mut mx2 = mx.clone();
        mx2[3] = 30.5; mx2[4] = -0.5; mx2[5] = 59.999; mx2[6] = 60.; mx2[7] = 61.5; mx2[8] = 0.5;
        for fused in [true, false] {
            for ch in [1usize, 2] {
                let (mut a, mut b) = (vec![vec![0u8; rows * n]; ch], vec![vec![0u8; rows * n]; ch]);
                let mut ra: Vec<&mut [u8]> = a.iter_mut().map(|v| v.as_mut_slice()).collect();
                remap_kleidi(&plane, w, h, w * ch, &mx2, &my, n, fused, &mut ra);
                let mut rb: Vec<&mut [u8]> = b.iter_mut().map(|v| v.as_mut_slice()).collect();
                remap_kleidi_scalar(&plane, w, h, w * ch, &mx2, &my, n, fused, &mut rb, 0);
                assert_eq!(a, b, "fused {fused} channels {ch}");
            }
        }
    }

    /// The OpenCV-rule remap's AVX2 path against its scalar form (x86_64 with AVX2; elsewhere nothing to compare).
    #[test]
    fn avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let (w, h) = (61usize, 37usize);
            let plane: Vec<u8> = (0..w * h * 2).map(|i| ((i * 7919) % 256) as u8).collect();
            let n = 29;
            let mut seed = 777u32;
            let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed as f32 / u32::MAX as f32 };
            let my: Vec<f32> = vec![-2.5, -0.25, 0., 3.5, 17.25, 35.75, 36., 36.5, 50., 17.53125];
            let rows = my.len();
            let mut mx: Vec<f32> = (0..rows * n).map(|_| rnd() * 40. - 4. + 0.515625).collect();
            mx[3] = 30.5; mx[4] = -0.5; mx[5] = 59.999; mx[6] = 60.; mx[7] = 61.5; mx[8] = 0.015625; mx[9] = f32::NAN;
            mx[12] = 3e9; mx[13] = -3e9; mx[40] = 0.046875; mx[41] = 0.078125;
            let wt = remap_weights();
            for (ch, step) in [(0usize, 1usize), (0, 2), (1, 2)] {
                let (mut a, mut b) = (vec![0u8; rows * n], vec![0u8; rows * n]);
                unsafe { remap_avx2(&plane, w, h, w * step, ch, step, &mx, &my, n, &wt, &mut a) };
                remap_scalar(&plane, w, h, w * step, ch, step, &mx, &my, n, &wt, &mut b, 0..rows);
                assert_eq!(a, b, "channel {ch} step {step}");
            }
        }
    }

    #[test]
    fn weights_sum_exactly() {
        for w in remap_weights().iter() {
            assert_eq!(w.iter().sum::<i32>(), 1 << COEF_BITS);
        }
    }

    #[test]
    fn resize_area_integer_and_fraction() {
        let src: Vec<u8> = (0..300 * 2).map(|i| (i * 7 % 256) as u8).collect();
        let out = resize_rows(&src, 300, 2, 128);
        assert_eq!(out.len(), 256);
        let same = resize_rows(&src, 128, 2, 128);
        assert_eq!(same, src[..256]);
    }
}
