//! Image preprocessing that matches OpenCV `cv2.resize(..., INTER_LINEAR)` on 8-bit 3-channel images.
//! OpenCV uses 11-bit fixed-point weights; the vertical pass follows its SIMD formula (mulhi on value>>4).

/// Interleaved 8-bit image (3 channels, BGR as read by cv2.imread).
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;

/// Per destination index: (source index, weight0, weight1) like OpenCV's xofs/ialpha (x) and yofs/ibeta (y) tables.
/// OpenCV clamps the weight to (1, 0) at the borders only for x; for y it keeps the weight and clamps the row index.
fn taps(src: usize, dst: usize, clamp_weight: bool) -> Vec<(i64, i16, i16)> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = f.floor() as i64;
            let mut f = f - s as f32;
            if clamp_weight && s < 0 {
                s = 0;
                f = 0.0;
            }
            if clamp_weight && s >= src as i64 - 1 {
                s = src as i64 - 1;
                f = 0.0;
            }
            // saturate_cast<short>(float) rounds half to even
            let a0 = ((1.0 - f) * COEF_SCALE).round_ties_even() as i16;
            let a1 = (f * COEF_SCALE).round_ties_even() as i16;
            (s, a0, a1)
        })
        .collect()
}

pub fn resize_linear(src: &Img, dw: usize, dh: usize) -> Img {
    let xt = taps(src.w, dw, true);
    let yt = taps(src.h, dh, false);
    let row = |y: usize, out: &mut [i32]| {
        let s = &src.data[y * src.w * 3..(y + 1) * src.w * 3];
        for (dx, &(sx, a0, a1)) in xt.iter().enumerate() {
            let sx = sx as usize;
            for c in 0..3 {
                let p0 = s[sx * 3 + c] as i32;
                let p1 = if a1 != 0 { s[(sx + 1) * 3 + c] as i32 } else { 0 };
                out[dx * 3 + c] = p0 * a0 as i32 + p1 * a1 as i32;
            }
        }
    };
    let mut r0 = vec![0i32; dw * 3];
    let mut r1 = vec![0i32; dw * 3];
    let (mut c0, mut c1) = (usize::MAX, usize::MAX);
    let mut data = vec![0u8; dw * dh * 3];
    let last = src.h as i64 - 1;
    for (dy, &(sy, b0, b1)) in yt.iter().enumerate() {
        let (sy, sy1) = (sy.clamp(0, last) as usize, (sy + 1).clamp(0, last) as usize);
        if c0 != sy {
            if c1 == sy {
                std::mem::swap(&mut r0, &mut r1);
                c0 = sy;
                c1 = usize::MAX;
            } else {
                row(sy, &mut r0);
                c0 = sy;
            }
        }
        if c1 != sy1 {
            row(sy1, &mut r1);
            c1 = sy1;
        }
        let o = &mut data[dy * dw * 3..(dy + 1) * dw * 3];
        let (b0, b1) = (b0 as i32, b1 as i32);
        for i in 0..dw * 3 {
            let v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
            o[i] = v.clamp(0, 255) as u8;
        }
    }
    Img { w: dw, h: dh, data }
}

/// YOLOX letterbox: scale to fit, top-left placement, pad 114. Returns (image, ratio).
pub fn letterbox(src: &Img, h: usize, w: usize, clahe: Option<&crate::night::Clahe>) -> (Img, f64) {
    let r = (h as f64 / src.h as f64).min(w as f64 / src.w as f64);
    let nh = (src.h as f64 * r).round() as usize;
    let nw = (src.w as f64 * r).round() as usize;
    let mut rs = resize_linear(src, nw, nh);
    if let Some(clahe) = clahe {
        clahe.apply(&mut rs);
    }
    let mut data = vec![114u8; w * h * 3];
    for y in 0..nh {
        data[y * w * 3..y * w * 3 + nw * 3].copy_from_slice(&rs.data[y * nw * 3..(y + 1) * nw * 3]);
    }
    (Img { w, h, data }, r)
}

/// HWC BGR u8 -> NCHW f32 (raw 0..255, BGR), as YOLOX expects.
pub fn to_chw_raw(img: &Img) -> Vec<f32> {
    let n = img.w * img.h;
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = img.data[i * 3 + c] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_geometry_and_padding() {
        let src = Img { w: 1344, h: 760, data: vec![10; 1344 * 760 * 3] };
        let (lb, r) = letterbox(&src, 416, 416, None);
        assert!((r - 416.0 / 1344.0).abs() < 1e-12);
        assert_eq!((lb.w, lb.h), (416, 416));
        // content rows 0..235 (760*r = 235.2 -> 235), pad below
        assert_eq!(lb.data[(234 * 416 + 415) * 3], 10);
        assert_eq!(lb.data[(235 * 416) * 3], 114);
    }

    #[test]
    fn resize_constant_and_identity() {
        let src = Img { w: 7, h: 5, data: (0..7 * 5 * 3).map(|i| (i * 7 % 251) as u8).collect() };
        let same = resize_linear(&src, 7, 5);
        assert_eq!(same.data, src.data);
        let flat = Img { w: 9, h: 9, data: vec![200; 9 * 9 * 3] };
        assert!(resize_linear(&flat, 4, 3).data.iter().all(|&v| v == 200));
        assert!(resize_linear(&flat, 20, 17).data.iter().all(|&v| v == 200));
    }

    #[test]
    fn resize_y_border_keeps_weight() {
        // 1 col x 2 rows -> 4 rows upscale on 10/20: top row uses rows (0, 0) with weights (0.75, 0.25)
        // -> SIMD rounding of two halves gives 10 (not a clamped 1/0 weight); same values as cv2.resize
        let src = Img { w: 1, h: 2, data: vec![10, 10, 10, 20, 20, 20] };
        let d = resize_linear(&src, 1, 4);
        let ch0: Vec<u8> = d.data.iter().step_by(3).copied().collect();
        assert_eq!(ch0, vec![10, 13, 18, 20]);
    }

    #[test]
    fn resize_half_pixel_center() {
        // 2 px -> 4 px upscale: OpenCV gives 10, 12.5->(13), 17.5->(18), 20 pattern on a 10..20 ramp
        let src = Img { w: 2, h: 1, data: vec![10, 10, 10, 20, 20, 20] };
        let d = resize_linear(&src, 4, 1);
        let ch0: Vec<u8> = d.data.iter().step_by(3).copied().collect();
        assert_eq!(ch0, vec![10, 13, 18, 20]);
    }
}
