//! One camera frame to boxes of one side view: NV12 to BGR, night correction, side view, YOLOX.
//! The steps and constants are those of the offline runner's wide configuration with a 416x320 input.

use crate::{engine, gate::Target, night::Clahe, nv12::Layout, pre, remap, wcam, yolox};
use anyhow::{Context, Result};
use ccnc_vision_msg::Calibration;
use std::path::Path;
use std::time::{Duration, Instant};

pub const MODEL: &str = "yolox_tiny_dyn.onnx";
/// Map files for `Detector::fixed_maps` (replay checks against maps made offline).
pub const MAPS: [&str; 2] = ["map_vL.bin", "map_vR.bin"];
pub const THREADS: usize = 2;
/// Side view detector input height and width. A 512x384 side view fills 416x312 of it.
const SIDE_INPUT: [usize; 2] = [320, 416];
/// Full wide frame detector input height and width (1344x760 -> 416x235 inside 416x416).
const RAW_INPUT: [usize; 2] = [416, 416];
/// One box per detector output row at most: the 8/16/32 stride grids of the largest input (3549 rows for 416x416).
#[cfg(test)]
pub const MAX_BOXES: usize = (RAW_INPUT[0] / 8) * (RAW_INPUT[1] / 8) + (RAW_INPUT[0] / 16) * (RAW_INPUT[1] / 16) + (RAW_INPUT[0] / 32) * (RAW_INPUT[1] / 32);
const NUM_CLASSES: usize = 80;
const KEEP_SCORE: f32 = 0.05;
const NMS_IOU: f32 = 0.45;
const CLAHE: Clahe = Clahe { clip: 3.0, tiles: 8 };
/// A frame is night when the mean of the upper half (all channels) is below this.
const NIGHT_MEAN: f64 = 35.0;

/// Both side views' remap tables and what they were built from.
struct Views {
    maps: [remap::Map; 2],
    calibration: Calibration,
}

pub struct Detector {
    engine: Box<dyn engine::Engine>,
    views: Option<Views>,
    /// The views came from map files and stay as they are.
    fixed: bool,
}

pub struct Found {
    pub night: bool,
    pub boxes: Vec<yolox::Det>,
    pub calibration: Calibration,
}

fn is_night(image: &pre::Img) -> bool {
    let upper = &image.data[..image.h / 2 * image.w * 3];
    let sum: u64 = upper.iter().map(|&v| u64::from(v)).sum();
    !upper.is_empty() && (sum as f64 / upper.len() as f64) < NIGHT_MEAN
}

impl Detector {
    pub fn load(assets: &Path, engine: &str) -> Result<Self> {
        let model = assets.join(MODEL).to_string_lossy().into_owned();
        let engine = engine::open(engine, &model, [1, 3, SIDE_INPUT[0], SIDE_INPUT[1]], THREADS)?;
        Ok(Self { engine, views: None, fixed: false })
    }

    /// Use the map files in `dir` for every frame instead of building the views from liveCalibration.
    pub fn fixed_maps(&mut self, dir: &Path) -> Result<()> {
        let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
        let load_map = |name: &str| remap::Map::load(&path(name)).with_context(|| format!("map {}", path(name)));
        self.views = Some(Views { maps: [load_map(MAPS[0])?, load_map(MAPS[1])?], calibration: Calibration::UNKNOWN });
        self.fixed = true;
        Ok(())
    }

    /// Have views for `calibration`: built again when an angle moved by `calib::REBUILD_RAD` or more from the
    /// one they were built with. Returns how long building took, if it did.
    pub fn calibrate(&mut self, calibration: &Calibration) -> Option<Duration> {
        if self.fixed || self.views.as_ref().is_some_and(|v| !crate::calib::needs_rebuild(&v.calibration, calibration)) {
            return None;
        }
        let start = Instant::now();
        let maps = std::thread::scope(|scope| {
            let left = scope.spawn(|| wcam::side_map(calibration, 0));
            let right = wcam::side_map(calibration, 1);
            [left.join().expect("side map thread"), right]
        });
        self.views = Some(Views { maps, calibration: *calibration });
        Some(start.elapsed())
    }

    pub fn fixed(&self) -> bool {
        self.fixed
    }

    pub fn run(&mut self, layout: Layout, nv12: &[u8], target: Target) -> Result<Found> {
        let Some(views) = &self.views else { anyhow::bail!("no side views yet") };
        let mut frame = layout.to_bgr(nv12)?;
        let night = is_night(&frame);
        if night {
            CLAHE.apply(&mut frame);
        }
        let calibration = views.calibration;
        let ([height, width], image) = match target {
            Target::Left => (SIDE_INPUT, remap::remap_linear(&frame, &views.maps[0])),
            Target::Right => (SIDE_INPUT, remap::remap_linear(&frame, &views.maps[1])),
            Target::Raw => (RAW_INPUT, frame),
        };
        let (input, ratio) = pre::letterbox(&image, height, width, None);
        let (_, mut out) = self.engine.run([1, 3, height, width], pre::to_chw_raw(&input))?;
        yolox::grid_decode(&mut out, NUM_CLASSES, height, width);
        let candidates = yolox::candidates(&out, NUM_CLASSES, KEEP_SCORE);
        let keep = yolox::nms(&candidates, KEEP_SCORE, NMS_IOU);
        Ok(Found { night, boxes: yolox::to_image(&candidates, &keep, ratio, image.w, image.h), calibration })
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
