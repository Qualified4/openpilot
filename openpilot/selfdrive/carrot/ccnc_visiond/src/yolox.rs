//! YOLOX output decoding and NMS, matching scripts/run_bench.py (yolox / yolox_raw families).

/// One detection in original-image pixels: [x1, y1, x2, y2], score, COCO class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Det {
    pub b: [f64; 4],
    pub score: f32,
    pub cls: usize,
}

/// Grid decode for official YOLOX ONNX releases (demo_postprocess): xy=(p+grid)*stride, wh=exp(p)*stride.
pub fn grid_decode(out: &mut [f32], nc: usize, h: usize, w: usize) {
    let row = 5 + nc;
    let mut i = 0;
    for st in [8usize, 16, 32] {
        for gy in 0..h / st {
            for gx in 0..w / st {
                let p = &mut out[i * row..i * row + 4];
                let s = st as f32;
                p[0] = (p[0] + gx as f32) * s;
                p[1] = (p[1] + gy as f32) * s;
                p[2] = p[2].exp() * s;
                p[3] = p[3].exp() * s;
                i += 1;
            }
        }
    }
    assert_eq!(i * row, out.len(), "YOLOX output rows do not match the strides 8/16/32 grid");
}

/// Rows [cx, cy, w, h, obj, cls...] (input pixels) -> candidates with score = obj * best class score >= keep.
pub fn candidates(out: &[f32], nc: usize, keep: f32) -> Vec<([f32; 4], f32, usize)> {
    let row = 5 + nc;
    let mut v = Vec::new();
    for r in out.chunks_exact(row) {
        let obj = r[4];
        let (mut best, mut cls) = (f32::NEG_INFINITY, 0);
        for (c, &p) in r[5..].iter().enumerate() {
            let s = obj * p;
            if s > best {
                best = s;
                cls = c;
            }
        }
        if best >= keep {
            let (hw, hh) = (r[2] / 2.0, r[3] / 2.0);
            v.push(([r[0] - hw, r[1] - hh, r[0] + hw, r[1] + hh], best, cls));
        }
    }
    v
}

fn rect_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    // OpenCV Rect2d (x, y, w, h) overlap = 1 - jaccardDistance
    let (aw, ah, bw, bh) = (a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]);
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let iw = (a[0] + aw).min(b[0] + bw) - x1;
    let ih = (a[1] + ah).min(b[1] + bh) - y1;
    let inter = if iw <= 0.0 || ih <= 0.0 { 0.0 } else { iw * ih };
    let uni = aw * ah + bw * bh - inter;
    if uni <= 0.0 { 0.0 } else { inter / uni }
}

/// Class-aware NMS like cv2.dnn.NMSBoxes on boxes offset by class*10000: stable sort by score desc,
/// score > score_thr, keep while IoU <= iou_thr against every kept box. Returns kept indices.
pub fn nms(c: &[([f32; 4], f32, usize)], score_thr: f32, iou_thr: f32) -> Vec<usize> {
    let boxes: Vec<[f64; 4]> = c
        .iter()
        .map(|(b, _, cls)| {
            let off = *cls as f32 * 10000.0;
            [(b[0] + off) as f64, (b[1] + off) as f64, (b[2] + off) as f64, (b[3] + off) as f64]
        })
        .collect();
    let mut order: Vec<usize> = (0..c.len()).filter(|&i| c[i].1 > score_thr).collect();
    order.sort_by(|&a, &b| c[b].1.partial_cmp(&c[a].1).unwrap());
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        if keep.iter().all(|&k| rect_iou(boxes[i], boxes[k]) as f32 <= iou_thr) {
            keep.push(i);
        }
    }
    keep
}

fn round_to(v: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (v * m).round() / m
}

/// Undo letterbox (top-left, ratio r), clip to the image, round like the Python runner (1 / 4 decimals).
pub fn to_image(c: &[([f32; 4], f32, usize)], keep: &[usize], r: f64, w0: usize, h0: usize) -> Vec<Det> {
    keep.iter()
        .map(|&i| {
            let (b, s, cls) = c[i];
            let lim = [w0 as f64, h0 as f64, w0 as f64, h0 as f64];
            let mut o = [0f64; 4];
            for k in 0..4 {
                o[k] = round_to((b[k] as f64 / r).clamp(0.0, lim[k]), 1);
            }
            Det { b: o, score: round_to(s as f64, 4) as f32, cls }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_decode_416() {
        let nc = 1;
        let n = 52 * 52 + 26 * 26 + 13 * 13;
        let mut out = vec![0f32; n * 6];
        grid_decode(&mut out, nc, 416, 416);
        // first stride-8 cell: (0+0)*8, exp(0)*8
        assert_eq!(&out[0..4], &[0.0, 0.0, 8.0, 8.0]);
        // cell (gx=3, gy=2) at stride 8
        let i = 2 * 52 + 3;
        assert_eq!(&out[i * 6..i * 6 + 4], &[24.0, 16.0, 8.0, 8.0]);
        // first stride-16 cell, then last stride-32 cell (12, 12)
        let j = 52 * 52;
        assert_eq!(out[j * 6 + 2], 16.0);
        let k = n - 1;
        assert_eq!(&out[k * 6..k * 6 + 3], &[12.0 * 32.0, 12.0 * 32.0, 32.0]);
    }

    #[test]
    fn candidates_score_is_obj_times_cls() {
        // cx cy w h obj c0 c1
        let out = [100.0, 50.0, 20.0, 10.0, 0.5, 0.2, 0.8, /* low */ 0.0, 0.0, 1.0, 1.0, 0.1, 0.3, 0.3];
        let c = candidates(&out, 2, 0.05);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0, [90.0, 45.0, 110.0, 55.0]);
        assert!((c[0].1 - 0.4).abs() < 1e-6);
        assert_eq!(c[0].2, 1);
    }

    #[test]
    fn nms_is_class_aware_and_greedy() {
        let c = vec![
            ([0.0, 0.0, 10.0, 10.0], 0.9, 2),
            ([1.0, 0.0, 11.0, 10.0], 0.8, 2), // IoU 0.82 with #0 -> dropped
            ([1.0, 0.0, 11.0, 10.0], 0.7, 7), // other class -> kept
            ([20.0, 0.0, 30.0, 10.0], 0.6, 2), // no overlap -> kept
            ([0.0, 0.0, 10.0, 10.0], 0.04, 2), // below score threshold
        ];
        assert_eq!(nms(&c, 0.05, 0.45), vec![0, 2, 3]);
        // IoU exactly at the threshold is kept (overlap <= thr): boxes 0..10 and 5..15 -> 50/150 = 1/3
        let d = vec![([0.0, 0.0, 10.0, 10.0], 0.9, 0), ([5.0, 0.0, 15.0, 10.0], 0.8, 0)];
        assert_eq!(nms(&d, 0.05, 1.0 / 3.0), vec![0, 1]);
        assert_eq!(nms(&d, 0.05, 0.3), vec![0]);
    }

    #[test]
    fn undo_letterbox_and_clip() {
        let c = vec![([-4.0, 10.0, 420.0, 100.0], 0.123456, 2)];
        let d = to_image(&c, &[0], 416.0 / 1344.0, 1344, 760);
        assert_eq!(d[0].b, [0.0, 32.3, 1344.0, 323.1]);
        assert_eq!(d[0].score, 0.1235);
    }
}
