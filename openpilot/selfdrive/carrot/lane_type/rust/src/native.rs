//! The KleidiCV-rule remap through native code, for comparing against the crate's own NEON (`strip::Backend`):
//! `kleidicv_remap_f32_u8` (feature `kleidicv`) or `cv::remap` (feature `opencv`, OpenCV built with the KleidiCV HAL).
//! Both take full float32 x and y maps and write interleaved channels; this module builds the y map from the
//! per-row value and splits CbCr. Without the feature, or on an error code, `remap` returns false (the caller then
//! uses the Rust remap) and counts it in `fallbacks()`.

use crate::strip::Backend;
use std::sync::atomic::{AtomicU64, Ordering};

static FALLBACKS: AtomicU64 = AtomicU64::new(0);

pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

#[allow(dead_code)]
extern "C" {
    fn stripnet_kleidicv_remap(src: *const u8, stride: usize, w: usize, h: usize, dst: *mut u8, dst_w: usize, dst_h: usize,
                               channels: usize, mx: *const f32, my: *const f32) -> i32;
    fn stripnet_opencv_remap(src: *const u8, stride: usize, w: usize, h: usize, dst: *mut u8, dst_w: usize, dst_h: usize,
                             channels: usize, mx: *const f32, my: *const f32) -> i32;
    fn stripnet_opencv_set_threads(n: i32);
}

/// Whether `backend` is compiled into this binary.
pub fn available(backend: Backend) -> bool {
    match backend {
        Backend::Rust => true,
        Backend::KleidiCv => cfg!(feature = "kleidicv"),
        Backend::OpenCv => cfg!(feature = "opencv"),
    }
}

#[cfg(feature = "opencv")]
fn opencv_threads() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // The two sides already run on two threads; STRIPNET_CV_THREADS sets cv::setNumThreads (default 1).
    ONCE.call_once(|| {
        let n = std::env::var("STRIPNET_CV_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        unsafe { stripnet_opencv_set_threads(n) }
    });
}

/// Remap `outs.len()` (1 or 2) interleaved channels of `plane` at (mx[r*n + j], my_row[r]) into `outs`.
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn remap(backend: Backend, plane: &[u8], width: usize, height: usize, stride: usize, mx: &[f32], my_row: &[f32], n: usize,
             outs: &mut [&mut [u8]]) -> bool {
    if !available(backend) || backend == Backend::Rust {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let rows = my_row.len();
    let ch = outs.len();
    assert!(mx.len() >= rows * n && plane.len() >= (height - 1) * stride + width * ch && n >= 4);
    let mut my = Vec::with_capacity(rows * n);
    for &y in my_row {
        my.extend(std::iter::repeat(y).take(n));
    }
    let mut inter;
    let dst: *mut u8 = if ch == 1 {
        outs[0].as_mut_ptr()
    } else {
        inter = vec![0u8; rows * n * ch];
        inter.as_mut_ptr()
    };
    // SAFETY: every pointer covers the sizes passed (asserted above); the C side only reads src / maps and writes dst.
    #[allow(unused_unsafe)]
    let rc = unsafe {
        match backend {
            #[cfg(feature = "kleidicv")]
            Backend::KleidiCv => stripnet_kleidicv_remap(plane.as_ptr(), stride, width, height, dst, n, rows, ch, mx.as_ptr(), my.as_ptr()),
            #[cfg(feature = "opencv")]
            Backend::OpenCv => {
                opencv_threads();
                stripnet_opencv_remap(plane.as_ptr(), stride, width, height, dst, n, rows, ch, mx.as_ptr(), my.as_ptr())
            }
            _ => -1,
        }
    };
    if rc != 0 {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    if ch > 1 {
        // SAFETY: dst points at `inter` (rows * n * ch bytes), written by the call above.
        let inter = unsafe { std::slice::from_raw_parts(dst, rows * n * ch) };
        for (c, out) in outs.iter_mut().enumerate() {
            for (o, px) in out[..rows * n].iter_mut().zip(inter.chunks_exact(ch)) {
                *o = px[c];
            }
        }
    }
    true
}
