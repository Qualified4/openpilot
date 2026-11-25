//! One camera frame to boxes of one side view: NV12 to BGR, night correction, side view, YOLOX.
//! The steps and constants are those of the offline runner's wide configuration with a 416x320 input.

use crate::{engine, gate::Target, night::Clahe, nv12::Layout, pre, remap, wcam, yolox};
use anyhow::{Context, Result};
use ccnc_vision_msg::Calibration;
use std::path::Path;
use std::time::{Duration, Instant};

/// YOLOX-tiny quantized to INT8 (QDQ, the output concat/reshape left in float: det-light 2026-11-23), for the side
/// views and the full wide frame alike.
pub const MODEL: &str = "yolox_tiny_int8.onnx";
/// Map files for `Detector::fixed_maps` (replay checks against maps made offline).
pub const MAPS: [&str; 2] = ["map_vL.bin", "map_vR.bin"];
/// Side view detector input height and width. A 512x384 side view fills 416x312 of it.
const SIDE_INPUT: [usize; 2] = [320, 416];
/// Full wide frame detector input height and width (1344x760 -> 416x235 inside 416x416).
const RAW_INPUT: [usize; 2] = [416, 416];
/// Road camera above the bonnet (rows 0..620 of 1344x760) -> 416x192 (det_rules FRONT_IN).
const FRONT_INPUT: [usize; 2] = [192, 416];
pub const FRONT_CROP_Y: usize = 620;
/// A front box whose bottom is this close to the crop line (detector input pixels) is cut (det_rules `front_box`).
const FRONT_CUT_TOL: f64 = 2.0;
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
    threads: usize,
    views: Option<Views>,
    /// The views came from map files and stay as they are.
    fixed: bool,
}

pub struct Found {
    pub night: bool,
    pub boxes: Vec<yolox::Det>,
    pub calibration: Calibration,
    /// The image the boxes are measured in (the view, or the frame), for the classifier's crops.
    pub image: pre::Img,
}

/// Road camera boxes, pixels of the 1344x760 frame.
pub struct FrontFound {
    pub boxes: Vec<(yolox::Det, bool)>,
    /// The frame as captured (the classifier's crops).
    pub image: pre::Img,
    pub night: bool,
}

/// Whether the upper half of a frame is night.
fn is_night(image: &pre::Img) -> bool {
    let upper = &image.data[..image.h / 2 * image.w * 3];
    let sum: u64 = upper.iter().map(|&v| u64::from(v)).sum();
    !upper.is_empty() && (sum as f64 / upper.len() as f64) < NIGHT_MEAN
}

impl Detector {
    pub fn load(assets: &Path, engine: &str, threads: usize) -> Result<Self> {
        let model = assets.join(MODEL).to_string_lossy().into_owned();
        let engine = engine::open(engine, &model, [1, 3, SIDE_INPUT[0], SIDE_INPUT[1]], threads)?;
        Ok(Self { engine, threads, views: None, fixed: false })
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
        let maps = if self.threads <= 1 {
            [wcam::side_map(calibration, 0), wcam::side_map(calibration, 1)]
        } else {
            std::thread::scope(|scope| {
                let left = scope.spawn(|| wcam::side_map(calibration, 0));
                let right = wcam::side_map(calibration, 1);
                [left.join().expect("side map thread"), right]
            })
        };
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
        let (boxes, _) = self.detect(&image, height, width, None)?;
        Ok(Found { night, boxes, calibration, image })
    }

    /// `clahe`: applied to the scaled image inside the letterbox.
    fn detect(&mut self, image: &pre::Img, height: usize, width: usize, clahe: Option<&Clahe>) -> Result<(Vec<yolox::Det>, f64)> {
        let (input, ratio) = pre::letterbox(image, height, width, clahe);
        let (_, mut out) = self.engine.run([1, 3, height, width], pre::to_chw_raw(&input))?;
        yolox::grid_decode(&mut out, NUM_CLASSES, height, width);
        let candidates = yolox::candidates(&out, NUM_CLASSES, KEEP_SCORE);
        let keep = yolox::nms(&candidates, KEEP_SCORE, NMS_IOU);
        Ok((yolox::to_image(&candidates, &keep, ratio, image.w, image.h), ratio))
    }

    /// Road camera frame: the part above the bonnet. At night the detector input (416x192) goes through CLAHE, as the
    /// wide camera's frames do; the classifier still crops the frame as captured (item 80: crops of a corrected frame
    /// gave six times as many car->truck answers on 47f's night, and CLAHE at full size costs ten times as much).
    /// Each box comes with whether it is cut by the crop line (its bottom is the bonnet, not the car's).
    pub fn run_front(&mut self, layout: Layout, nv12: &[u8]) -> Result<FrontFound> {
        anyhow::ensure!(layout.height >= FRONT_CROP_Y, "road frame {} rows, the crop line is at {FRONT_CROP_Y}", layout.height);
        let frame = layout.to_bgr(nv12)?;
        let night = is_night(&frame);
        let top = pre::Img { w: frame.w, h: FRONT_CROP_Y, data: frame.data[..FRONT_CROP_Y * frame.w * 3].to_vec() };
        let (boxes, ratio) = self.detect(&top, FRONT_INPUT[0], FRONT_INPUT[1], night.then_some(&CLAHE))?;
        let cut_at = FRONT_CROP_Y as f64 - FRONT_CUT_TOL / ratio;
        Ok(FrontFound { boxes: boxes.into_iter().map(|b| { let cut = b.b[3] >= cut_at; (b, cut) }).collect(), image: frame, night })
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

    /// Keeps the detector input it was given; finds nothing.
    struct Recorder(std::rc::Rc<std::cell::RefCell<Vec<f32>>>);

    impl engine::Engine for Recorder {
        fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            *self.0.borrow_mut() = data;
            let rows: usize = [8, 16, 32].iter().map(|s| (shape[2] / s) * (shape[3] / s)).sum();
            Ok((vec![1, rows, 5 + NUM_CLASSES], vec![0.; rows * (5 + NUM_CLASSES)]))
        }
    }

    /// A 1344x760 NV12 frame of grey `level` with a fine pattern (so CLAHE has something to change).
    fn frame(level: u8) -> (Layout, Vec<u8>) {
        let (w, h) = (1344, 760);
        let mut nv12: Vec<u8> = (0..w * h).map(|i| level.saturating_add(((i % w / 7 + i / w / 5) % 3 * 4) as u8)).collect();
        nv12.extend(vec![128u8; w * h / 2]);
        (Layout { width: w, height: h, stride: w, uv_offset: w * h }, nv12)
    }

    fn front(level: u8) -> (FrontFound, Vec<f32>, pre::Img) {
        let input = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut detector = Detector { engine: Box::new(Recorder(input.clone())), threads: 1, views: None, fixed: false };
        let (layout, nv12) = frame(level);
        let found = detector.run_front(layout, &nv12).unwrap();
        let bgr = layout.to_bgr(&nv12).unwrap();
        let top = pre::Img { w: bgr.w, h: FRONT_CROP_Y, data: bgr.data[..FRONT_CROP_Y * bgr.w * 3].to_vec() };
        let input = input.borrow().clone();
        (found, input, top)
    }

    #[test]
    fn night_road_frames_are_detected_through_clahe_and_classified_as_captured() {
        let (found, input, top) = front(20);
        assert!(found.night);
        let plain = pre::to_chw_raw(&pre::letterbox(&top, FRONT_INPUT[0], FRONT_INPUT[1], None).0);
        let corrected = pre::to_chw_raw(&pre::letterbox(&top, FRONT_INPUT[0], FRONT_INPUT[1], Some(&CLAHE)).0);
        assert_ne!(plain, corrected, "the pattern is one CLAHE changes");
        assert_eq!(input, corrected, "the detector sees the corrected input");
        assert_eq!(found.image.data[..top.data.len()], top.data[..], "the classifier crops the frame as captured");
    }

    #[test]
    fn day_road_frames_are_not_corrected() {
        let (found, input, top) = front(120);
        assert!(!found.night);
        assert_eq!(input, pre::to_chw_raw(&pre::letterbox(&top, FRONT_INPUT[0], FRONT_INPUT[1], None).0));
    }
}
