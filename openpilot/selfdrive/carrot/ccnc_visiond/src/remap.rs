//! Wide side view: OpenCV 5.1-compatible `cv2.remap(img, mx, my, INTER_LINEAR, borderValue=0)` with float maps
//! (Virt.mx / Virt.my of wcam.py, yaw -+55 deg, 512x384), built by `wcam::side_map` or read from a file.
use crate::pre::Img;
use anyhow::{bail, Result};

/// Map file: u32 width, u32 height (little endian), then width*height f32 x, then width*height f32 y.
pub struct Map {
    pub w: usize,
    pub h: usize,
    pub mx: Vec<f32>,
    pub my: Vec<f32>,
}

impl Map {
    pub fn from_bytes(b: &[u8]) -> Result<Map> {
        if b.len() < 8 {
            bail!("map file too short");
        }
        let w = u32::from_le_bytes(b[0..4].try_into()?) as usize;
        let h = u32::from_le_bytes(b[4..8].try_into()?) as usize;
        let n = w * h;
        if b.len() != 8 + 8 * n {
            bail!("map file size {} != 8 + 8*{}*{}", b.len(), w, h);
        }
        let f = |o: usize| -> Vec<f32> { (0..n).map(|i| f32::from_le_bytes(b[o + 4 * i..o + 4 * i + 4].try_into().unwrap())).collect() };
        Ok(Map { w, h, mx: f(8), my: f(8 + 4 * n) })
    }
    pub fn load(path: &str) -> Result<Map> {
        Map::from_bytes(&std::fs::read(path)?)
    }
}

pub fn remap_linear(src: &Img, m: &Map) -> Img {
    // OpenCV 5.x (the version the offline tests used, 5.1.0) computes INTER_LINEAR remap of 8-bit images with float maps in
    // f32: lerp along x on both rows, then along y, round half to even. Checked bit-exact on a 512x384 side view.
    let (sw, sh) = (src.w as i64, src.h as i64);
    let mut data = vec![0u8; m.w * m.h * 3];
    for i in 0..m.w * m.h {
        let (x, y) = (m.mx[i], m.my[i]);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (sx, sy) = (x0 as i64, y0 as i64);
        let px = |x: i64, y: i64, c: usize| -> f32 {
            if x >= 0 && y >= 0 && x < sw && y < sh {
                src.data[((y * sw + x) * 3) as usize + c] as f32
            } else {
                0.0
            }
        };
        for c in 0..3 {
            let (p00, p01, p10, p11) = (px(sx, sy, c), px(sx + 1, sy, c), px(sx, sy + 1, c), px(sx + 1, sy + 1, c));
            let t = p00 + (p01 - p00) * fx;
            let b = p10 + (p11 - p10) * fx;
            let v = t + (b - t) * fy;
            data[i * 3 + c] = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    Img { w: m.w, h: m.h, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn img(w: usize, h: usize) -> Img {
        Img { w, h, data: (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect() }
    }
    #[test]
    fn integer_map_copies_pixels() {
        let s = img(8, 6);
        let m = Map { w: 3, h: 2, mx: vec![1., 2., 3., 1., 2., 3.], my: vec![2., 2., 2., 4., 4., 4.] };
        let o = remap_linear(&s, &m);
        for (i, (x, y)) in [(1, 2), (2, 2), (3, 2), (1, 4), (2, 4), (3, 4)].iter().enumerate() {
            assert_eq!(&o.data[i * 3..i * 3 + 3], &s.data[(y * 8 + x) * 3..(y * 8 + x) * 3 + 3]);
        }
    }
    #[test]
    fn half_pixel_is_average_and_outside_is_border() {
        let mut s = Img { w: 2, h: 1, data: vec![0; 6] };
        s.data[3..6].copy_from_slice(&[100, 200, 50]);
        let m = Map { w: 2, h: 1, mx: vec![0.5, -5.0], my: vec![0.0, 0.0] };
        let o = remap_linear(&s, &m);
        assert_eq!(&o.data[0..3], &[50, 100, 25]);
        assert_eq!(&o.data[3..6], &[0, 0, 0]);
    }
    #[test]
    fn map_file_roundtrip() {
        let mut b = vec![];
        b.extend(2u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        for v in [0.25f32, 1.0, 3.0, 4.0] {
            b.extend(v.to_le_bytes());
        }
        let m = Map::from_bytes(&b).unwrap();
        assert_eq!((m.w, m.h), (2, 1));
        assert_eq!(m.mx, vec![0.25, 1.0]);
        assert_eq!(m.my, vec![3.0, 4.0]);
        assert!(Map::from_bytes(&b[..10]).is_err());
    }
}
