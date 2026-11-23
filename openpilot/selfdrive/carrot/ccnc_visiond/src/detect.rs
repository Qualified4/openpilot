//! One camera frame to boxes of one side view: NV12 to BGR, night correction, side view, YOLOX.
//! The steps and constants are those of the offline runner's wide configuration with a 416x320 input.

use crate::{engine, gate::Side, night::Clahe, nv12::Layout, pre, remap, yolox};
use anyhow::{Context, Result};
use std::path::Path;

pub const MODEL: &str = "yolox_tiny_dyn.onnx";
pub const MAPS: [&str; 2] = ["map_vL.bin", "map_vR.bin"];
pub const THREADS: usize = 2;
/// Detector input height and width. A 512x384 side view fills 416x312 of it.
const INPUT: [usize; 2] = [320, 416];
/// One box per detector output row at most: the 8/16/32 stride grids of the input (2730 rows).
#[cfg(test)]
pub const MAX_BOXES: usize = (INPUT[0] / 8) * (INPUT[1] / 8) + (INPUT[0] / 16) * (INPUT[1] / 16) + (INPUT[0] / 32) * (INPUT[1] / 32);
const NUM_CLASSES: usize = 80;
const KEEP_SCORE: f32 = 0.05;
const NMS_IOU: f32 = 0.45;
const CLAHE: Clahe = Clahe { clip: 3.0, tiles: 8 };
/// A frame is night when the mean of the upper half (all channels) is below this.
const NIGHT_MEAN: f64 = 35.0;

pub struct Detector {
    engine: Box<dyn engine::Engine>,
    maps: [remap::Map; 2],
}

pub struct Found {
    pub night: bool,
    pub boxes: Vec<yolox::Det>,
}

fn is_night(image: &pre::Img) -> bool {
    let upper = &image.data[..image.h / 2 * image.w * 3];
    let sum: u64 = upper.iter().map(|&v| u64::from(v)).sum();
    !upper.is_empty() && (sum as f64 / upper.len() as f64) < NIGHT_MEAN
}

impl Detector {
    pub fn load(assets: &Path, engine: &str) -> Result<Self> {
        let path = |name: &str| assets.join(name).to_string_lossy().into_owned();
        let load_map = |name: &str| remap::Map::load(&path(name)).with_context(|| format!("map {}", path(name)));
        let maps = [load_map(MAPS[0])?, load_map(MAPS[1])?];
        let engine = engine::open(engine, &path(MODEL), [1, 3, INPUT[0], INPUT[1]], THREADS)?;
        Ok(Self { engine, maps })
    }

    pub fn run(&mut self, layout: Layout, nv12: &[u8], side: Side) -> Result<Found> {
        let mut frame = layout.to_bgr(nv12)?;
        let night = is_night(&frame);
        if night {
            CLAHE.apply(&mut frame);
        }
        let view = remap::remap_linear(&frame, &self.maps[side as usize]);
        let [height, width] = INPUT;
        let (input, ratio) = pre::letterbox(&view, height, width, None);
        let (_, mut out) = self.engine.run([1, 3, height, width], pre::to_chw_raw(&input))?;
        yolox::grid_decode(&mut out, NUM_CLASSES, height, width);
        let candidates = yolox::candidates(&out, NUM_CLASSES, KEEP_SCORE);
        let keep = yolox::nms(&candidates, KEEP_SCORE, NMS_IOU);
        Ok(Found { night, boxes: yolox::to_image(&candidates, &keep, ratio, view.w, view.h) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn night_is_a_dark_upper_half() {
        let image = |upper: u8, lower: u8| {
            let mut data = vec![upper; 4 * 2 * 3];
            data.extend(vec![lower; 4 * 2 * 3]);
            pre::Img { w: 4, h: 4, data }
        };
        assert!(is_night(&image(34, 255)));
        assert!(!is_night(&image(35, 0)));
    }
}
