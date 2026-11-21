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
    let (w1, h1) = (width as i32 - 1, height as i32 - 1);
    let px = |x: i32, y: i32| plane[y as usize * stride + x as usize * step + channel] as i32;
    for (r, &my) in my_row.iter().enumerate() {
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
    let r = rows(line, m, frame.height, dist)?;
    let g = maps(&r, m, shape.half_m, shape.cols);
    Some(sample(frame, &g, shape, mirror))
}

pub fn sample(frame: &Nv12, g: &Maps, shape: StripShape, mirror: bool) -> Vec<f32> {
    let weights = remap_weights();
    let (rows, n, cols) = (g.rows, g.n, shape.cols);
    let my: Vec<f32> = g.ys.iter().map(|&y| y as f32).collect();
    let my_uv: Vec<f32> = my.iter().map(|&v| v * 0.5).collect();
    let mut planes = [vec![0u8; rows * n], vec![0u8; rows * n], vec![0u8; rows * n]];
    let [lum, cb, cr] = &mut planes;
    remap(frame.y, frame.width, frame.height, frame.stride, 0, 1, &g.mx, &my, n, &weights, lum);
    let (cw, ch) = (frame.width / 2, frame.height / 2);
    remap(frame.uv, cw, ch, frame.stride, 0, 2, &g.mx_uv, &my_uv, n, &weights, cb);
    remap(frame.uv, cw, ch, frame.stride, 1, 2, &g.mx_uv, &my_uv, n, &weights, cr);
    let mut out = vec![0f32; 3 * shape.rows * cols];
    for (c, plane) in planes.iter().enumerate() {
        // Exact 4:1 column box mean truncated to uint8.
        let boxed: Vec<u8> = plane.chunks_exact(SUB).map(|q| ((q[0] as u32 + q[1] as u32 + q[2] as u32 + q[3] as u32) >> 2) as u8).collect();
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
