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

/// Whether the upper `rows` of a frame are night.
fn is_night_rows(image: &pre::Img, rows: usize) -> bool {
    let rows = rows.min(image.h);
    let upper = &image.data[..rows * image.w * 3];
    let sum: u64 = upper.iter().map(|&v| u64::from(v)).sum();
    !upper.is_empty() && (sum as f64 / upper.len() as f64) < NIGHT_MEAN
}

/// Whether the upper half of a frame is night.
fn is_night(image: &pre::Img) -> bool {
    is_night_rows(image, image.h / 2)
}

impl Detector {
    pub fn load(assets: &Path, engine_name: &str, threads: usize) -> Result<Self> {
        let engine: Box<dyn engine::Engine> = match engine_name {
            "mnn" => match crate::mnn::MnnDetectorEngine::open(assets, crate::mnn::Backend::OpenClFp16, threads) {
                Ok(mnn) => {
                    eprintln!(
                        "ccnc_visiond: detector using MNN OpenCL FP16 (YOLOX-s 416x192/416x320/416x416, cache={})",
                        crate::mnn::cache_dir().display()
                    );
                    Box::new(mnn)
                }
                Err(err) => {
                    eprintln!("ccnc_visiond: MNN OpenCL detector unavailable ({err:#}); falling back to CPU ORT");
                    let model = assets.join(MODEL).to_string_lossy().into_owned();
                    engine::open("ort", &model, [1, 3, SIDE_INPUT[0], SIDE_INPUT[1]], threads)?
                }
            },
            "mnn-cpu" => {
                let mnn = crate::mnn::MnnDetectorEngine::open(assets, crate::mnn::Backend::Cpu, threads)?;
                eprintln!("ccnc_visiond: detector using MNN CPU FP16 (YOLOX-s 416x192/416x320/416x416, threads={threads})");
                Box::new(mnn)
            }
            other => {
                let model = assets.join(MODEL).to_string_lossy().into_owned();
                engine::open(other, &model, [1, 3, SIDE_INPUT[0], SIDE_INPUT[1]], threads)?
            }
        };
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
        let frame = layout.to_bgr(nv12)?;
        let night = is_night(&frame);
        let calibration = views.calibration;
        let ([height, width], image) = match target {
            Target::Left => (SIDE_INPUT, remap::remap_linear(&frame, &views.maps[0])),
            Target::Right => (SIDE_INPUT, remap::remap_linear(&frame, &views.maps[1])),
            Target::Raw => (RAW_INPUT, frame),
        };
        // Apply night CLAHE on the letterboxed detector input (416x312 for SideL/R, 416x235 for Raw)
        // instead of the full 1344x760 frame (~8-10x fewer pixels on CPU), matching `run_front` and
        // keeping `Found.image` unmodified for the classifier.
        let (boxes, _) = self.detect(&image, height, width, night.then_some(&CLAHE))?;
        Ok(Found { night, boxes, calibration, image })
    }

    /// `clahe`: applied to the scaled image inside the letterbox.
    fn detect(&mut self, image: &pre::Img, height: usize, width: usize, clahe: Option<&Clahe>) -> Result<(Vec<yolox::Det>, f64)> {
        let (input, ratio) = pre::letterbox(image, height, width, clahe);
        let (_, mut out) = self.engine.run([1, 3, height, width], pre::to_chw_raw(&input))?;
        if self.engine.needs_grid_decode() {
            yolox::grid_decode(&mut out, NUM_CLASSES, height, width);
        }
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
        let top = layout.to_bgr_rows(nv12, FRONT_CROP_Y)?;
        let night = is_night_rows(&top, layout.height / 2);
        let (boxes, ratio) = self.detect(&top, FRONT_INPUT[0], FRONT_INPUT[1], night.then_some(&CLAHE))?;
        let cut_at = FRONT_CROP_Y as f64 - FRONT_CUT_TOL / ratio;
        Ok(FrontFound { boxes: boxes.into_iter().map(|b| { let cut = b.b[3] >= cut_at; (b, cut) }).collect(), image: top, night })
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

    struct PreDecodedEngine;

    impl engine::Engine for PreDecodedEngine {
        fn run(&mut self, shape: [usize; 4], _data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            let rows: usize = [8, 16, 32].iter().map(|s| (shape[2] / s) * (shape[3] / s)).sum();
            let mut out = vec![0.0f32; rows * (5 + NUM_CLASSES)];
            // Row 0: person (cls=0) at (100, 60, 40, 80) in input pixels, obj=0.9, cls0=0.8 -> score=0.72
            out[0] = 100.0;
            out[1] = 60.0;
            out[2] = 40.0;
            out[3] = 80.0;
            out[4] = 0.9;
            out[5 + 0] = 0.8;
            // Row 1: motorcycle (cls=3) at (220, 90, 30, 50), obj=0.8, cls3=0.7 -> score=0.56
            let r1 = 5 + NUM_CLASSES;
            out[r1] = 220.0;
            out[r1 + 1] = 90.0;
            out[r1 + 2] = 30.0;
            out[r1 + 3] = 50.0;
            out[r1 + 4] = 0.8;
            out[r1 + 5 + 3] = 0.7;
            Ok((vec![1, rows, 5 + NUM_CLASSES], out))
        }

        fn needs_grid_decode(&self) -> bool {
            false
        }
    }

    #[test]
    fn predecoded_mnn_skips_grid_decode_and_preserves_person_and_motorcycle() {
        let mut detector = Detector { engine: Box::new(PreDecodedEngine), threads: 1, views: None, fixed: false };
        let img = pre::Img { w: 416, h: 192, data: vec![128u8; 416 * 192 * 3] };
        let (boxes, _ratio) = detector.detect(&img, 192, 416, None).unwrap();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].cls, 0); // person
        assert_eq!(boxes[0].b, [80.0, 20.0, 120.0, 100.0]);
        assert_eq!(boxes[1].cls, 3); // motorcycle
        assert_eq!(boxes[1].b, [205.0, 65.0, 235.0, 115.0]);
        // Classifier pick must exclude both person (0) and motorcycle (3).
        assert!(crate::classify::pick(&boxes, |_| false).is_empty());
    }

    fn identity_side_map() -> remap::Map {
        let (w, h) = (512, 384);
        let mut mx = Vec::with_capacity(w * h);
        let mut my = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                mx.push((x as f32) * (1344.0 / w as f32));
                my.push((y as f32) * (760.0 / h as f32));
            }
        }
        remap::Map { w, h, mx, my }
    }

    #[test]
    fn night_wide_frames_apply_clahe_after_remap_at_input_size() {
        let input = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut detector = Detector {
            engine: Box::new(Recorder(input.clone())),
            threads: 1,
            views: Some(Views {
                maps: [identity_side_map(), identity_side_map()],
                calibration: Calibration::UNKNOWN,
            }),
            fixed: true,
        };
        let (layout, nv12) = frame(20);
        let raw_bgr = layout.to_bgr(&nv12).unwrap();
        let remapped = remap::remap_linear(&raw_bgr, &identity_side_map());

        // SideL night: CLAHE is applied inside letterbox (416x312), Found.image stays as-captured.
        let found_side = detector.run(layout, &nv12, Target::Left).unwrap();
        assert!(found_side.night);
        assert_eq!(found_side.image.data, remapped.data, "Found.image stays unmodified for classifier");
        let plain_side = pre::to_chw_raw(&pre::letterbox(&remapped, SIDE_INPUT[0], SIDE_INPUT[1], None).0);
        let clahe_side = pre::to_chw_raw(&pre::letterbox(&remapped, SIDE_INPUT[0], SIDE_INPUT[1], Some(&CLAHE)).0);
        assert_ne!(plain_side, clahe_side);
        assert_eq!(*input.borrow(), clahe_side, "detector sees post-remap CLAHE input");

        // Raw night: CLAHE is applied inside letterbox (416x235), Found.image stays as-captured.
        let found_raw = detector.run(layout, &nv12, Target::Raw).unwrap();
        assert!(found_raw.night);
        assert_eq!(found_raw.image.data, raw_bgr.data);
        let clahe_raw = pre::to_chw_raw(&pre::letterbox(&raw_bgr, RAW_INPUT[0], RAW_INPUT[1], Some(&CLAHE)).0);
        assert_eq!(*input.borrow(), clahe_raw);

        // Day wide frame: no CLAHE applied.
        let (day_layout, day_nv12) = frame(120);
        let day_bgr = day_layout.to_bgr(&day_nv12).unwrap();
        let day_remap = remap::remap_linear(&day_bgr, &identity_side_map());
        let found_day = detector.run(day_layout, &day_nv12, Target::Left).unwrap();
        assert!(!found_day.night);
        assert_eq!(*input.borrow(), pre::to_chw_raw(&pre::letterbox(&day_remap, SIDE_INPUT[0], SIDE_INPUT[1], None).0));
    }

    #[test]
    fn night_wide_clahe_bench_and_quality_comparison() {
        // Synthetic dark scene with realistic low-light gradients + vehicle-like bright patches.
        let (w, h) = (1344, 760);
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let base = 10u8
                    .saturating_add(((x * 18) / w) as u8)
                    .saturating_add(((y * 16) / h) as u8);
                // Brighter rectangle (car silhouette + headlights)
                let patch = if (400..680).contains(&x) && (260..480).contains(&y) { 28 } else { 0 };
                let v = base.saturating_add(patch);
                let idx = (y * w + x) * 3;
                data[idx] = v;
                data[idx + 1] = v.saturating_add(2);
                data[idx + 2] = v.saturating_add(4);
            }
        }
        let full_img = pre::Img { w, h, data };
        let map = identity_side_map();

        // Warm up OnceLock LabTables before timing.
        let mut dummy = pre::Img { w: 16, h: 16, data: vec![20; 16 * 16 * 3] };
        CLAHE.apply(&mut dummy);

        // Old path: full-frame 1344x760 CLAHE -> remap 512x384 -> letterbox 416x320.
        let t_old = Instant::now();
        let mut full_clahe = pre::Img { w: full_img.w, h: full_img.h, data: full_img.data.clone() };
        CLAHE.apply(&mut full_clahe);
        let old_remap = remap::remap_linear(&full_clahe, &map);
        let (old_lb, _) = pre::letterbox(&old_remap, SIDE_INPUT[0], SIDE_INPUT[1], None);
        let old_ms = t_old.elapsed().as_secs_f64() * 1000.0;

        // New path: remap 512x384 -> letterbox 416x320 with CLAHE on 416x312.
        let t_new = Instant::now();
        let new_remap = remap::remap_linear(&full_img, &map);
        let (new_lb, _) = pre::letterbox(&new_remap, SIDE_INPUT[0], SIDE_INPUT[1], Some(&CLAHE));
        let new_ms = t_new.elapsed().as_secs_f64() * 1000.0;

        // Pure CLAHE-only timing comparison (1344x760 vs 416x312 vs 416x235).
        let mut c_full = pre::Img { w: full_img.w, h: full_img.h, data: full_img.data.clone() };
        let t_cf = Instant::now();
        CLAHE.apply(&mut c_full);
        let clahe_full_ms = t_cf.elapsed().as_secs_f64() * 1000.0;

        let mut c_side = pre::resize_linear(&new_remap, 416, 312);
        let t_cs = Instant::now();
        CLAHE.apply(&mut c_side);
        let clahe_side_ms = t_cs.elapsed().as_secs_f64() * 1000.0;

        let mut c_raw = pre::resize_linear(&full_img, 416, 235);
        let t_cr = Instant::now();
        CLAHE.apply(&mut c_raw);
        let clahe_raw_ms = t_cr.elapsed().as_secs_f64() * 1000.0;

        // Quality correlation over the active 416x312 region (excluding letterbox pad rows 312..320).
        let active_bytes = 416 * 312 * 3;
        let a = &old_lb.data[..active_bytes];
        let b = &new_lb.data[..active_bytes];
        let n = active_bytes as f64;
        let mean_a = a.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mean_b = b.iter().map(|&v| v as f64).sum::<f64>() / n;
        let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
        for (&va, &vb) in a.iter().zip(b.iter()) {
            let da = va as f64 - mean_a;
            let db = vb as f64 - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        let corr = cov / (var_a * var_b).sqrt();

        eprintln!(
            "CLAHE bench: full_pipeline={old_ms:.2}ms -> post_remap_pipeline={new_ms:.2}ms ({:.2}x); \
             clahe_only: 1344x760={clahe_full_ms:.2}ms vs 416x312={clahe_side_ms:.2}ms ({:.2}x) vs 416x235={clahe_raw_ms:.2}ms ({:.2}x); \
             mean={mean_a:.1}/{mean_b:.1}, pearson_r={corr:.4}",
            old_ms / new_ms.max(1e-6),
            clahe_full_ms / clahe_side_ms.max(1e-6),
            clahe_full_ms / clahe_raw_ms.max(1e-6),
        );

        assert!(clahe_full_ms > clahe_side_ms * 3.0, "expected >=3x CLAHE speedup (got {clahe_full_ms:.2}ms vs {clahe_side_ms:.2}ms)");
        assert!(corr > 0.90, "expected strong structural correlation between pre/post-remap CLAHE, got {corr:.4}");
    }
}
