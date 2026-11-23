//! 8-bit sRGB/Lab and CLAHE matching OpenCV's CPU imgproc path.
//! Based on OpenCV color_lab.cpp, softfloat.cpp and clahe.cpp (Apache-2.0).

use crate::pre::Img;
use std::sync::OnceLock;

pub struct Clahe {
    pub clip: f64,
    pub tiles: usize,
}

impl Clahe {
    pub fn apply(&self, image: &mut Img) {
        let tab = LAB.get_or_init(LabTables::new);
        for pixel in image.data.chunks_exact_mut(3) {
            pixel.copy_from_slice(&tab.to_lab(pixel));
        }
        self.equalize_luma(image);
        for pixel in image.data.chunks_exact_mut(3) {
            pixel.copy_from_slice(&tab.to_bgr(pixel));
        }
    }

    fn equalize_luma(&self, lab: &mut Img) {
        let n = self.tiles;
        let (w, h) = (lab.w, lab.h);
        // OpenCV pads both axes if either axis needs padding.
        let (tw, th) = if w % n == 0 && h % n == 0 {
            (w / n, h / n)
        } else {
            (w / n + 1, h / n + 1)
        };
        let area = tw * th;
        let clip = ((self.clip * area as f64 / 256.0) as usize).max(1);
        let lut_scale = 255.0 / area as f32;
        let mut luts = vec![[0u8; 256]; n * n];
        let xs: Vec<_> = (0..tw * n).map(|x| reflect101(x, w)).collect();
        for (tile_y, row) in luts.chunks_exact_mut(n).enumerate() {
            for (tile_x, lut) in row.iter_mut().enumerate() {
                let mut hist = [0usize; 256];
                let cols = &xs[tile_x * tw..][..tw];
                for y in tile_y * th..(tile_y + 1) * th {
                    let offset = w * reflect101(y, h);
                    for &x in cols {
                        hist[usize::from(lab.data[3 * (offset + x)])] += 1;
                    }
                }
                if self.clip > 0.0 {
                    let mut excess = 0;
                    for bin in &mut hist {
                        excess += bin.saturating_sub(clip);
                        *bin = (*bin).min(clip);
                    }
                    let (batch, residual) = (excess / 256, excess % 256);
                    for bin in &mut hist {
                        *bin += batch;
                    }
                    if residual > 0 {
                        for bin in hist.iter_mut().step_by(256 / residual).take(residual) {
                            *bin += 1;
                        }
                    }
                }
                let mut sum = 0;
                for (out, bin) in lut.iter_mut().zip(hist) {
                    sum += bin;
                    *out = (sum as f32 * lut_scale).round_ties_even() as u8;
                }
            }
        }
        let xtaps: Vec<_> = (0..w).map(|x| taps(x, tw, n)).collect();
        for (y, row) in lab.data.chunks_exact_mut(w * 3).enumerate() {
            let (top, bottom, ya) = taps(y, th, n);
            let (top, bottom) = (&luts[top * n..][..n], &luts[bottom * n..][..n]);
            for (pixel, &(left, right, xa)) in row.chunks_exact_mut(3).zip(&xtaps) {
                let l = usize::from(pixel[0]);
                let upper = top[left][l] as f32 * (1.0 - xa) + top[right][l] as f32 * xa;
                let lower = bottom[left][l] as f32 * (1.0 - xa) + bottom[right][l] as f32 * xa;
                pixel[0] = (upper * (1.0 - ya) + lower * ya).round_ties_even() as u8;
            }
        }
    }
}

fn reflect101(pos: usize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let pos = pos % period;
    pos.min(period - pos)
}

fn taps(pos: usize, tile: usize, n: usize) -> (usize, usize, f32) {
    let f = pos as f32 * (1.0 / tile as f32) - 0.5;
    let first = f.floor();
    (
        (first as isize).max(0) as usize,
        (first as isize + 1).min(n as isize - 1) as usize,
        f - first,
    )
}

const LAB_BASE: i32 = 1 << 14;
const LAB_SHIFT: u32 = 15;
const XYZ_SHIFT: u32 = 12;
const BGR_TO_XYZ: [[i32; 3]; 3] = [[778, 1541, 1777], [296, 2929, 871], [3575, 448, 73]];
const XYZ_TO_BGR: [[i32; 3]; 3] = [[217, -836, 4715], [-3773, 7684, 185], [12615, -6296, -2223]];
static LAB: OnceLock<LabTables> = OnceLock::new();

struct LabTables {
    gamma: [i32; 256],
    inv_gamma: [u8; 4096],
    cube_root: [i32; 3072],
    y_fy: [(i32, i32); 256],
}

impl LabTables {
    fn new() -> Self {
        let gamma = std::array::from_fn(|i| {
            let x = f64::from(i as f32 / 255.0);
            let y = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
            (2040.0 * y as f32).round_ties_even() as i32
        });
        let inv_gamma = std::array::from_fn(|i| {
            let x = i as f64 / 4096.0;
            let y = if x <= 0.0031308 {
                x * 12.92
            } else {
                x.powf(1.0 / 2.4) * 1.055 - 0.055
            };
            (255.0 * y as f32).round_ties_even() as u8
        });
        let cube_root = std::array::from_fn(|i| {
            let x = (1.0f32 / 2040.0) * i as f32;
            let f = if x < 216.0 / 24389.0 {
                x.mul_add(841.0 / 108.0, 16.0 / 116.0)
            } else {
                opencv_cube_root(x)
            };
            (f * (1 << LAB_SHIFT) as f32).round_ties_even() as i32
        });
        let y_fy = std::array::from_fn(|i| {
            let i = i as i32;
            let (y, fy) = if i <= 20 {
                (
                    (i * LAB_BASE * 20 * 9) as f32 / (17.0 * 29.0 * 29.0 * 29.0),
                    LAB_BASE as f32 * (16.0 / 116.0 + (i * 5) as f32 / (3.0 * 17.0 * 29.0)),
                )
            } else {
                let fy =
                    (i * 100 * LAB_BASE) as f32 / (255.0 * 116.0) + (16 * LAB_BASE) as f32 / 116.0;
                (fy * fy * fy / (LAB_BASE * LAB_BASE) as f32, fy)
            };
            (y.round_ties_even() as i32, fy.round_ties_even() as i32)
        });
        Self {
            gamma,
            inv_gamma,
            cube_root,
            y_fy,
        }
    }

    fn to_lab(&self, p: &[u8]) -> [u8; 3] {
        let bgr = [
            self.gamma[p[0] as usize],
            self.gamma[p[1] as usize],
            self.gamma[p[2] as usize],
        ];
        let [fx, fy, fz] =
            BGR_TO_XYZ.map(|row| self.cube_root[descale(dot(row, bgr), XYZ_SHIFT) as usize]);
        let l = 296 * fy - (16 * 255 * (1 << LAB_SHIFT) + 50) / 100;
        let a = 500 * (fx - fy) + (128 << LAB_SHIFT);
        let b = 200 * (fy - fz) + (128 << LAB_SHIFT);
        [l, a, b].map(|v| descale(v, LAB_SHIFT).clamp(0, 255) as u8)
    }

    fn to_bgr(&self, p: &[u8]) -> [u8; 3] {
        let (y, fy) = self.y_fy[p[0] as usize];
        let a = ((5 * i32::from(p[1]) * 53_687 + 128) >> 13) - 128 * LAB_BASE / 500;
        let b = ((i32::from(p[2]) * 41_943 + 16) >> 9) - 128 * LAB_BASE / 200 + 1;
        let xyz = [lab_axis_to_xyz(fy + a), y, lab_axis_to_xyz(fy - b)];
        XYZ_TO_BGR.map(|row| self.inv_gamma[descale(dot(row, xyz), 14).clamp(0, 4095) as usize])
    }
}

fn dot(a: [i32; 3], b: [i32; 3]) -> i32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn descale(x: i32, shift: u32) -> i32 {
    (x + (1 << (shift - 1))) >> shift
}

fn lab_axis_to_xyz(f: i32) -> i32 {
    if f <= 3390 {
        f * 108 / 841 - LAB_BASE * 16 / 116 * 108 / 841
    } else {
        f * f / LAB_BASE * f / LAB_BASE
    }
}

// OpenCV's softfloat cbrt uses Turkowski's rational polynomial and truncates
// the f64 mantissa to f32. This table only calls it with positive, normal values.
fn opencv_cube_root(x: f32) -> f32 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let mut scaled_exp = exponent % 3;
    if scaled_exp >= 0 {
        scaled_exp -= 3;
    }
    let out_exp = (exponent - scaled_exp) / 3 - 1;
    let fr =
        f64::from_bits(((scaled_exp + 1023) as u64) << 52 | (u64::from(bits & 0x7f_ffff) << 29));
    let a = [
        0x4046a09e6653ba70,
        0x406808f46c6116e0,
        0x405dca97439cae14,
        0x402add70d2827500,
        0x3fc4f15f83f55d2d,
        0x402d9e20660edb21,
        0x4062ff15c0285815,
        0x406510d06a8112ce,
        0x4040fecbc9e2c375,
    ]
    .map(f64::from_bits);
    let numer = (((a[0] * fr + a[1]) * fr + a[2]) * fr + a[3]) * fr + a[4];
    let denom = (((a[5] * fr + a[6]) * fr + a[7]) * fr + a[8]) * fr + 1.0;
    let fraction = (((numer / denom).to_bits() >> 29) as u32) & 0x7f_ffff;
    f32::from_bits(((out_exp + 127) as u32) << 23 | fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_reference_colors() {
        let lab = LabTables::new();
        assert_eq!(lab.to_lab(&[0, 0, 0]), [0, 128, 128]);
        assert_eq!(lab.to_lab(&[0, 0, 255]), [136, 208, 195]);
        assert_eq!(lab.to_lab(&[255, 0, 0]), [82, 207, 20]);
        assert_eq!(lab.to_bgr(&[255, 128, 128]), [255, 255, 255]);
    }

    #[test]
    fn clahe_uses_luma_and_reflects_at_the_border() {
        let mut lab = Img {
            w: 3,
            h: 2,
            data: [7, 103, 207].repeat(6),
        };
        Clahe {
            clip: 3.0,
            tiles: 2,
        }
        .equalize_luma(&mut lab);
        assert_eq!(lab.data, [128, 103, 207].repeat(6));
        assert_eq!(
            (0..8).map(|x| reflect101(x, 3)).collect::<Vec<_>>(),
            [0, 1, 2, 1, 0, 1, 2, 1]
        );
    }
}
