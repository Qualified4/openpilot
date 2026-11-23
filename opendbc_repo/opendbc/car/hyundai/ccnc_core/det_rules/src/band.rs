//! Own-car band: the part of the wide frame covered by the bonnet and dashboard, and the two box statistics on it.
use crate::band_top::{BAND_H, BAND_TOP, BAND_W};

/// Traced bonnet line in the raw 1344x760 wide frame. The band is everything below it, raised 6 px, with flat ends;
/// `BAND_TOP` is that polygon as cv2.fillPoly rasters it.
pub const HOOD: [(i32, i32); 7] = [(173, 513), (240, 513), (400, 497), (600, 482), (800, 481), (1000, 487), (1133, 503)];

/// Boolean image with a summed-area table, so a box count costs four lookups.
#[derive(Clone, Debug)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    sat: Vec<u32>, // (w + 1) x (h + 1)
}
impl Mask {
    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> bool) -> Mask {
        let mut sat = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                row += f(x, y) as u32;
                sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
            }
        }
        Mask { w, h, sat }
    }
    pub fn band() -> Mask {
        Mask::from_fn(BAND_W, BAND_H, |x, y| y >= BAND_TOP[x] as usize)
    }
    /// set pixels in columns x1..x2 and rows y1..y2 (both ends exclusive on the right).
    pub fn count(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u32 {
        let s = |x: usize, y: usize| self.sat[y * (self.w + 1) + x];
        s(x2, y2) + s(x1, y1) - s(x1, y2) - s(x2, y1)
    }
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.count(x, y, x + 1, y + 1) == 1
    }
    pub fn mean(&self) -> f64 {
        self.count(0, 0, self.w, self.h) as f64 / (self.w * self.h) as f64
    }
}

/// (fraction of the box inside the band, bottom touches the band). 'touch' = band rows y2-3..y2+5 more than half band.
pub fn band_stats(m: &Mask, bbox: &[f64; 4]) -> (f64, bool) {
    let (w, h) = (m.w as i64, m.h as i64);
    let r = |v: f64| v.round_ties_even() as i64; // Python round(): half to even
    let (x1, x2) = (r(bbox[0]).clamp(0, w - 1), r(bbox[2]).clamp(0, w - 1));
    let (y1, y2) = (r(bbox[1]).clamp(0, h - 1), r(bbox[3]).clamp(0, h - 1));
    if x2 <= x1 || y2 <= y1 {
        return (0.0, false);
    }
    let (xa, xb) = (x1 as usize, x2 as usize + 1);
    let area = (xb - xa) * (y2 - y1 + 1) as usize;
    let inside = m.count(xa, y1 as usize, xb, y2 as usize + 1) as f64 / area as f64;
    let (ra, rb) = ((y2 - 3).max(0) as usize, (y2 + 6).min(h) as usize);
    let n = (xb - xa) * (rb - ra);
    let touch = n > 0 && m.count(xa, ra, xb, rb) as f64 / n as f64 > 0.5;
    (inside, touch)
}
