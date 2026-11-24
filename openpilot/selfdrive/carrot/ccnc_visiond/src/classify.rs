//! Vehicle kind of detector boxes (car, truck, bus) on a thread of its own: the detector decides whether a car is
//! shown, this only which shape.
//!
//! The model is det-light's MobileNetV3-Small 128x128 fp32 with its linear head folded in (output 0: softmax over
//! car, truck, bus). A box is cut out with 10 % margin, squashed to 128x128 and normalised as torchvision does.
//! The thread runs at the process's policy and affinity (SCHED_IDLE, little cores by default) with one ORT thread.
//! It holds one waiting job: a newer frame's boxes replace one that has not started, so a slow classifier never
//! holds detection up, it only answers fewer frames.

use crate::{engine::Engine, pre::Img, yolox::Det};
use anyhow::{Context, Result};
use ccnc_vision_msg::{Calibration, KindResult, View, KIND_BUS, KIND_CAR, KIND_TRUCK};
use std::sync::{mpsc, Arc, Condvar, Mutex};

pub const MODEL: &str = "kind_mnv3s_128.onnx";
const SIZE: usize = 128;
const PAD: f64 = 0.1;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Output order of the model.
const KINDS: [u8; 3] = [KIND_CAR, KIND_TRUCK, KIND_BUS];
/// COCO car, bus, truck. Motorcycles have no shape here.
const VEHICLE_CLASSES: [usize; 3] = [2, 5, 7];
/// det_rules SCORE_MIN.
const MIN_SCORE: f32 = 0.3;
/// Boxes lower than this (pixels of their view) are too small to tell a truck from a car (det-light 7).
const MIN_HEIGHT: f64 = 32.0;
/// The largest boxes of a frame that are classified.
pub const MAX_CROPS: usize = 4;
/// Side view boxes are classified only when this is on (`ccnc_map::vision::kind::SIDE_KINDS` has the reasons).
pub const SIDE_KINDS: bool = false;

/// One frame's boxes to classify.
pub struct Job {
    pub frame_mono_time: u64,
    pub view: View,
    /// Calibration active when the frame was captured, carried through to `Status::Kinds`.
    pub calibration: Calibration,
    pub boxes: Vec<[f32; 4]>,
    /// NCHW, `boxes.len()` crops.
    pub input: Vec<f32>,
}

/// The boxes of `boxes` worth classifying: vehicles, not cut, tall enough; the largest `MAX_CROPS`.
pub fn pick<'a>(boxes: &'a [Det], cut: impl Fn(&Det) -> bool) -> Vec<&'a Det> {
    let mut picked: Vec<&Det> = boxes
        .iter()
        .filter(|d| VEHICLE_CLASSES.contains(&d.cls) && d.score >= MIN_SCORE && d.b[3] - d.b[1] >= MIN_HEIGHT && !cut(d))
        .collect();
    let area = |d: &Det| (d.b[2] - d.b[0]) * (d.b[3] - d.b[1]);
    picked.sort_by(|a, b| area(b).total_cmp(&area(a)));
    picked.truncate(MAX_CROPS);
    picked
}

/// The box with its margin cut out of `image` (cv2 slicing of det-light `cls_front.crop`), 128x128, RGB, normalised,
/// CHW, appended to `out`.
fn crop_into(image: &Img, b: [f64; 4], out: &mut Vec<f32>) {
    let (pw, ph) = (PAD * (b[2] - b[0]), PAD * (b[3] - b[1]));
    // int(max(0, v)) .. int(min(size, v)) as the Python slice.
    let clamp = |v: f64, hi: usize| (v.max(0.) as usize).min(hi);
    let (x1, x2) = (clamp(b[0] - pw, image.w), clamp(b[2] + pw, image.w));
    let (y1, y2) = (clamp(b[1] - ph, image.h), clamp(b[3] + ph, image.h));
    let (w, h) = ((x2 - x1).max(1), (y2 - y1).max(1));
    let mut data = Vec::with_capacity(w * h * 3);
    for y in y1..y1 + h {
        let y = y.min(image.h - 1);
        let row = &image.data[y * image.w * 3..(y + 1) * image.w * 3];
        for x in x1..x1 + w {
            let x = x.min(image.w - 1);
            data.extend_from_slice(&row[x * 3..x * 3 + 3]);
        }
    }
    let small = crate::pre::resize_linear(&Img { w, h, data }, SIZE, SIZE);
    let n = SIZE * SIZE;
    let start = out.len();
    out.resize(start + 3 * n, 0.);
    for i in 0..n {
        for c in 0..3 {
            // BGR in, RGB planes out.
            let v = f32::from(small.data[i * 3 + (2 - c)]) / 255.;
            out[start + c * n + i] = (v - MEAN[c]) / STD[c];
        }
    }
}

impl Job {
    pub fn new(frame_mono_time: u64, view: View, calibration: Calibration, image: &Img, boxes: &[&Det]) -> Self {
        let mut input = Vec::with_capacity(boxes.len() * 3 * SIZE * SIZE);
        for d in boxes {
            crop_into(image, d.b, &mut input);
        }
        Self { frame_mono_time, view, calibration, boxes: boxes.iter().map(|d| d.b.map(|v| v as f32)).collect(), input }
    }
}

/// The kind with the highest probability for each crop of `probs` ([n, 3]).
fn answer_job(job: &Job, probs: &[f32]) -> Vec<KindResult> {
    job.boxes
        .iter()
        .zip(probs.chunks_exact(KINDS.len()))
        .map(|(xyxy, p)| {
            let (best, prob) = p.iter().enumerate().fold((0, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
            KindResult { frame_mono_time: job.frame_mono_time, view: job.view, xyxy: *xyxy, kind: KINDS[best], prob }
        })
        .collect()
}

#[derive(Default)]
struct Slot {
    job: Option<Job>,
    stop: bool,
}

/// The classifier thread. Dropping it stops the thread.
pub struct Classifier {
    slot: Arc<(Mutex<Slot>, Condvar)>,
    answers: mpsc::Receiver<(Vec<KindResult>, Calibration)>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Jobs replaced by a newer one before the thread took them.
    pub dropped: u64,
    pub submitted: u64,
}

impl Classifier {
    /// Starts the thread, which makes its engine with `make` (so the engine never crosses threads) and reports
    /// whether that worked before this returns.
    pub fn start<F>(make: F) -> Result<Self>
    where
        F: FnOnce() -> Result<Box<dyn Engine>> + Send + 'static,
    {
        let slot = Arc::new((Mutex::new(Slot::default()), Condvar::new()));
        let (ready_tx, ready_rx) = mpsc::sync_channel::<std::result::Result<(), String>>(1);
        let (answer_tx, answers) = mpsc::channel();
        let shared = Arc::clone(&slot);
        let thread = std::thread::Builder::new().name("visiond-kind".into()).spawn(move || {
            let mut engine = match make() {
                Ok(engine) => {
                    let _ = ready_tx.send(Ok(()));
                    engine
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("{err:#}")));
                    return;
                }
            };
            let (lock, cv) = &*shared;
            loop {
                let job = {
                    let mut slot = lock.lock().unwrap_or_else(|p| p.into_inner());
                    loop {
                        if slot.stop {
                            return;
                        }
                        if let Some(job) = slot.job.take() {
                            break job;
                        }
                        slot = cv.wait(slot).unwrap_or_else(|p| p.into_inner());
                    }
                };
                let n = job.boxes.len();
                match engine.run([n, 3, SIZE, SIZE], job.input.clone()) {
                    Ok((shape, probs)) if shape == [n, KINDS.len()] => {
                        if answer_tx.send((answer_job(&job, &probs), job.calibration)).is_err() {
                            return;
                        }
                    }
                    Ok((shape, _)) => eprintln!("ccnc_visiond: kind model gave {shape:?} for {n} crops"),
                    Err(err) => eprintln!("ccnc_visiond: kind of {n} boxes failed: {err:#}"),
                }
            }
        })?;
        let ready = ready_rx.recv().context("kind thread")?;
        let mut classifier = Self { slot, answers, thread: Some(thread), dropped: 0, submitted: 0 };
        if let Err(err) = ready {
            classifier.stop();
            anyhow::bail!(err);
        }
        Ok(classifier)
    }

    /// Hand a frame's boxes over without waiting. A job the thread has not started yet is replaced.
    pub fn submit(&mut self, job: Job) {
        if job.boxes.is_empty() {
            return;
        }
        let (lock, cv) = &*self.slot;
        let mut slot = lock.lock().unwrap_or_else(|p| p.into_inner());
        if slot.job.replace(job).is_some() {
            self.dropped += 1;
        }
        self.submitted += 1;
        cv.notify_one();
    }

    /// Whether the thread still runs (it ends on a panic, or when its answers have nowhere to go).
    pub fn alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|thread| !thread.is_finished())
    }

    /// Every completed job's answers paired with its capture-time calibration since the last call.
    pub fn answers(&self) -> Vec<(Vec<KindResult>, Calibration)> {
        self.answers.try_iter().collect()
    }

    fn stop(&mut self) {
        {
            let (lock, cv) = &*self.slot;
            lock.lock().unwrap_or_else(|p| p.into_inner()).stop = true;
            cv.notify_one();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Classifier {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Answers truck for every crop after `delay`.
    struct Slow {
        delay: Duration,
        runs: Arc<Mutex<Vec<usize>>>,
    }

    impl Engine for Slow {
        fn run(&mut self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            assert_eq!(data.len(), shape.iter().product::<usize>());
            std::thread::sleep(self.delay);
            self.runs.lock().unwrap().push(shape[0]);
            Ok((vec![shape[0], 3], [0.1, 0.8, 0.1].repeat(shape[0])))
        }
    }

    fn det(b: [f64; 4], cls: usize, score: f32) -> Det {
        Det { b, score, cls }
    }

    fn image() -> Img {
        Img { w: 64, h: 48, data: (0..64 * 48 * 3).map(|i| (i % 251) as u8).collect() }
    }

    #[test]
    fn picks_tall_uncut_vehicles_largest_first() {
        let boxes = [
            det([0., 0., 40., 40.], 2, 0.9),
            det([0., 0., 100., 31.], 7, 0.9),   // too low
            det([0., 0., 50., 50.], 0, 0.9),    // a person
            det([0., 0., 60., 60.], 5, 0.29),   // weak
            det([0., 0., 80., 80.], 7, 0.5),
            det([0., 0., 90., 90.], 2, 0.5),    // cut below
            det([10., 0., 45., 33.], 5, 0.31),
            det([0., 0., 33., 33.], 2, 0.9),
            det([0., 0., 34., 34.], 2, 0.9),
        ];
        let picked: Vec<_> = pick(&boxes, |d| d.b[3] >= 85.).into_iter().map(|d| d.b[2]).collect();
        assert_eq!(picked, [80., 40., 34., 45.]);
    }

    #[test]
    fn a_crop_is_normalised_rgb_planes() {
        let img = image();
        let mut out = Vec::new();
        crop_into(&img, [-5., -5., 70., 60.], &mut out);
        crop_into(&img, [10., 10., 20., 20.], &mut out);
        assert_eq!(out.len(), 2 * 3 * SIZE * SIZE);
        // The whole image squashed: the top-left output pixel is the top-left input pixel (B, G, R = 0, 1, 2).
        let n = SIZE * SIZE;
        let rgb = [out[0], out[n], out[2 * n]];
        let want = [2., 1., 0.].map(|v: f32| v / 255.);
        for c in 0..3 {
            assert!((rgb[c] - (want[c] - MEAN[c]) / STD[c]).abs() < 1e-6, "{c}");
        }
    }

    #[test]
    fn a_slow_classifier_never_holds_detection_up() {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let engine_runs = Arc::clone(&runs);
        let mut classifier =
            Classifier::start(move || Ok(Box::new(Slow { delay: Duration::from_millis(150), runs: engine_runs }) as Box<dyn Engine>)).unwrap();
        let img = image();
        let boxes = [det([0., 0., 40., 40.], 2, 0.9), det([5., 5., 45., 45.], 7, 0.9)];
        let picked: Vec<&Det> = boxes.iter().collect();
        // A detector at 50 ms per frame for 20 frames: each submit returns at once.
        let start = Instant::now();
        let mut slowest = Duration::ZERO;
        for i in 0..20u64 {
            let t = Instant::now();
            classifier.submit(Job::new(i, View::CropFront, Calibration::UNKNOWN, &img, &picked));
            slowest = slowest.max(t.elapsed());
            std::thread::sleep(Duration::from_millis(50));
        }
        let detect = start.elapsed();
        assert!(slowest < Duration::from_millis(20), "submit took {slowest:?}");
        assert!(detect < Duration::from_millis(20 * 50 + 300), "the detector loop took {detect:?}");
        std::thread::sleep(Duration::from_millis(400));
        let got: Vec<_> = classifier.answers().into_iter().flat_map(|(kinds, _)| kinds).collect();
        let frames: Vec<u64> = got.iter().map(|a| a.frame_mono_time).collect();
        assert!(classifier.dropped >= 10, "about two of every three jobs replaced, {} dropped", classifier.dropped);
        assert_eq!(got.len(), 2 * (classifier.submitted - classifier.dropped) as usize);
        assert_eq!(*frames.last().unwrap(), 19, "the newest frame is always answered");
        assert!(got.iter().all(|a| a.kind == KIND_TRUCK && (a.prob - 0.8).abs() < 1e-6));
        assert!(runs.lock().unwrap().iter().all(|&n| n == 2), "one batch per frame");
        drop(classifier);
    }

    struct Panics;

    impl Engine for Panics {
        fn run(&mut self, _: [usize; 4], _: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
            panic!("kind engine test panic");
        }
    }

    #[test]
    fn a_thread_that_dies_shows() {
        let mut classifier = Classifier::start(|| Ok(Box::new(Panics) as Box<dyn Engine>)).unwrap();
        assert!(classifier.alive());
        let img = image();
        let boxes = [det([0., 0., 40., 40.], 2, 0.9)];
        classifier.submit(Job::new(1, View::CropFront, Calibration::UNKNOWN, &img, &boxes.iter().collect::<Vec<_>>()));
        let start = Instant::now();
        while classifier.alive() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!classifier.alive());
        assert!(classifier.answers().is_empty());
        classifier.submit(Job::new(2, View::CropFront, Calibration::UNKNOWN, &img, &boxes.iter().collect::<Vec<_>>()));
        drop(classifier);
    }

    #[test]
    fn an_engine_that_cannot_load_fails_the_start() {
        let err = Classifier::start(|| anyhow::bail!("no model")).err().unwrap();
        assert!(format!("{err:#}").contains("no model"));
    }

    #[test]
    fn answers_carry_capture_time_calibration_after_live_calibration_is_lost() {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let mut classifier =
            Classifier::start(move || Ok(Box::new(Slow { delay: Duration::from_millis(50), runs }) as Box<dyn Engine>)).unwrap();
        let img = image();
        let boxes = [det([0., 0., 40., 40.], 7, 0.9)];
        let picked: Vec<&Det> = boxes.iter().collect();
        let captured = Calibration {
            rpy_calib: [0.01, 0.02, -0.01],
            wide_from_device_euler: [0.0, 0.01, -0.005],
            height: 1.22,
        };
        let mut latest = captured;
        classifier.submit(Job::new(100, View::CropFront, latest, &img, &picked));
        // Calibration is lost before the classifier finishes.
        latest = Calibration::UNKNOWN;
        assert!(latest.rpy_calib.iter().all(|v| v.is_nan()));
        let start = Instant::now();
        let mut batches = Vec::new();
        while batches.is_empty() && start.elapsed() < Duration::from_secs(2) {
            batches = classifier.answers();
            if batches.is_empty() {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        assert_eq!(batches.len(), 1);
        let (kinds, cal) = &batches[0];
        assert_eq!(kinds.len(), 1);
        assert_eq!(*cal, captured);
        assert!(cal.rpy_calib.iter().all(|v| v.is_finite()) && cal.height.is_finite());
        let out = ccnc_vision_msg::VisionOut {
            status: ccnc_vision_msg::Status::Kinds,
            frame_id: 0,
            timestamp_sof: 0,
            frame_mono_time: 0,
            view: View::CropFront,
            processing_ms: 0.0,
            night_corrected: false,
            calibration: *cal,
            boxes: Vec::new(),
            kinds: kinds.clone(),
        };
        let encoded = ccnc_vision_msg::encode(&out, 200);
        assert!(!encoded.is_empty());
        assert_eq!(out.calibration, captured);
    }
}
