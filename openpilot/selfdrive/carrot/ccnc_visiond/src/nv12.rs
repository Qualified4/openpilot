//! The 8-bit BT.601 limited-range conversion used by OpenCV's NV12 CPU path.

use crate::pre::Img;
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy)]
pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub uv_offset: usize,
}

impl Layout {
    pub fn validate(self, len: usize) -> Result<()> {
        let Self {
            width,
            height,
            stride,
            uv_offset,
        } = self;
        ensure!(
            width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
            "invalid NV12 dimensions"
        );
        ensure!(stride >= width, "NV12 stride is smaller than width");
        ensure!(
            stride
                .checked_mul(height)
                .is_some_and(|y_end| y_end <= uv_offset),
            "NV12 planes overlap"
        );
        let uv_end = stride
            .checked_mul(height / 2)
            .and_then(|size| uv_offset.checked_add(size))
            .context("invalid NV12 plane size")?;
        ensure!(uv_end <= len, "NV12 UV plane exceeds the buffer");
        ensure!(
            width
                .checked_mul(height)
                .and_then(|n| n.checked_mul(3))
                .is_some(),
            "BGR size overflow"
        );
        Ok(())
    }

    pub fn to_bgr(self, src: &[u8]) -> Result<Img> {
        self.validate(src.len())?;
        let Self {
            width,
            height,
            stride,
            uv_offset,
        } = self;
        let mut data = vec![0; width * height * 3];
        for (row, pair) in data.chunks_exact_mut(width * 6).enumerate() {
            let (out0, out1) = pair.split_at_mut(width * 3);
            let y0 = &src[2 * row * stride..][..width];
            let y1 = &src[(2 * row + 1) * stride..][..width];
            let uv = &src[uv_offset + row * stride..][..width];
            for (i, chroma) in uv.chunks_exact(2).enumerate() {
                let u = i32::from(chroma[0]) - 128;
                let v = i32::from(chroma[1]) - 128;
                let b = 2_116_026 * u + (1 << 19);
                let g = -409_993 * u - 852_492 * v + (1 << 19);
                let r = 1_673_527 * v + (1 << 19);
                for (ys, out) in [(y0, &mut *out0), (y1, &mut *out1)] {
                    for j in 2 * i..2 * i + 2 {
                        let y = (i32::from(ys[j]) - 16).max(0) * 1_220_542;
                        out[3 * j] = channel(y + b);
                        out[3 * j + 1] = channel(y + g);
                        out[3 * j + 2] = channel(y + r);
                    }
                }
            }
        }
        Ok(Img {
            w: width,
            h: height,
            data,
        })
    }
}

fn channel(value: i32) -> u8 {
    (value >> 20).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_comparison_with_padding() {
        let layout = Layout {
            width: 4,
            height: 2,
            stride: 6,
            uv_offset: 16,
        };
        let src = [
            16, 235, 81, 81, 255, 255, 0, 128, 81, 81, 255, 255, 255, 255, 255, 255, 128, 128, 90,
            240, 255, 255,
        ];
        assert_eq!(
            layout.to_bgr(&src).unwrap().data,
            [
                0, 0, 0, 255, 255, 255, 0, 0, 254, 0, 0, 254, 0, 0, 0, 130, 130, 130, 0, 0, 254, 0,
                0, 254,
            ]
        );
        assert!(layout.validate(src.len() - 1).is_err());
    }
}
