//! ccnc_visiond: vehicle boxes from the wide camera's side views while the car stands or creeps, and the kind
//! (car, truck, bus) of the radar cars ahead from the road camera.
//! Input: VisionIPC `wideRoad` and `road`, `carState.vEgo`, `liveCalibration` and CcncMap. Output: one CcncVision
//! (`customReserved4`) per processed frame, one with the classifier's answers when it has new ones, and a status
//! message about once a second while nothing is processed. The side views are built from liveCalibration; no side
//! frame is processed before calibrationd says calibrated.
//!
//! The road camera is looked at only while CcncMap has a radar car in FF, LF or RF without a confirmed kind, by
//! day, at `gate::FrontPace`. The classifier runs on its own thread (`classify`) and never holds detection up.
//!
//! Environment:
//! * `CCNC_VISIOND_ASSETS`: directory with `yolox_tiny_int8.onnx` and `kind_mnv3s_128.onnx` (default `assets`).
//! * `CCNC_VISIOND_ENGINE`: `ort` (default) or `tract` when built with the `tract` feature.
//! * `ORT_DYLIB_PATH`: libonnxruntime.so for the `ort` backend.
//! * `CCNC_VISIOND_HALT`: set by the launcher when the assets are missing or their sha256 differs. The process
//!   halts with this reason instead of loading anything.
//! * `CCNC_VISIOND_FIXED_MAPS`: replay checks only. Directory with `map_vL.bin` and `map_vR.bin` used for every
//!   frame instead of liveCalibration, which is then not needed.
//! * `CCNC_VISIOND_POLICY`: `idle` (default, `SCHED_IDLE`), `batch`, or `other`.
//! * `CCNC_VISIOND_CPUS`: CPU affinity list (default `0-2`, or `all`/`none` to inherit).
//! * `CCNC_VISIOND_THREADS`: ORT/tract intra-op threads (default `2`, max `4`).
//! * `CCNC_VISIOND_NICE`: nice value when policy is `other` or `batch` (default `19`).
//! * `CCNC_VISIOND_MIN_PERIOD_MS`: optional minimum interval between inferences in ms (default `0`).
//! * `CCNC_VISIOND_FRONT_MS`, `CCNC_VISIOND_FRONT_STOP_MS`: road camera period (`sched`).

mod calib;
mod classify;
mod detect;
mod engine;
mod gate;
mod night;
mod nv12;
mod pre;
mod remap;
mod sched;
mod vipc;
mod wcam;
mod yolox;

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use ccnc_vision_msg::{Calibration, KindResult, Status, View, VisionBox, VisionOut};
use stripnet::ipc::{Pub, Sub};

const SPEED_SERVICE: &str = "carState";
/// How long the road camera rests after a night frame (only the top half is converted to tell), whatever the map
/// asks: the standing pace, so one dark frame (a tunnel, a glitch) costs little and night costs a top half every 2 s.
const FRONT_NIGHT_REST_NS: u64 = 2_000_000_000;
const CALIBRATION_SERVICE: &str = "liveCalibration";
const MAP_SERVICE: &str = "customReserved3";
const CAMERA_SERVER: &str = "camerad";
/// Sleep while the speed keeps the detector off.
const IDLE: Duration = Duration::from_millis(100);
/// Sleep while waiting for the next camera frame (20 Hz).
const FRAME_WAIT: Duration = Duration::from_millis(5);

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn request_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn configure(sched_cfg: &sched::Config) -> io::Result<()> {
    let launcher_pid = std::env::var("VISIOND_PARENT_PID").ok().and_then(|pid| pid.parse::<libc::pid_t>().ok());
    unsafe {
        let parent = launcher_pid.unwrap_or_else(|| libc::getppid());
        let mut action: libc::sigaction = std::mem::zeroed();
        libc::sigemptyset(&mut action.sa_mask);
        action.sa_sigaction = request_stop as *const () as libc::sighandler_t;
        if libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) != 0
            || libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) != 0
            || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0
        {
            return Err(io::Error::last_os_error());
        }
        if libc::getppid() != parent {
            STOP.store(true, Ordering::Relaxed);
            return Ok(());
        }
    }
    // Applied before ORT loads so inference worker threads inherit both affinity and policy.
    let _ = sched_cfg.apply_current()?;
    Ok(())
}

fn sleep_interruptible(mut remaining: Duration) {
    while !remaining.is_zero() && !STOP.load(Ordering::Relaxed) {
        let step = remaining.min(Duration::from_millis(20));
        sleep(step);
        remaining = remaining.saturating_sub(step);
    }
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    assert_eq!(unsafe { libc::clock_gettime(clock, &mut ts) }, 0);
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn state_path() -> String {
    std::env::var("VISIOND_STATE").unwrap_or_else(|_| "/dev/shm/ccnc_visiond".into())
}

/// The backend named by CCNC_VISIOND_ENGINE, else ort (tract in a tract-only build).
fn engine_name() -> String {
    std::env::var("CCNC_VISIOND_ENGINE").unwrap_or_else(|_| if cfg!(feature = "ort") { "ort" } else { "tract" }.into())
}

/// Status messages while no frame is processed, at most this often.
const STATUS_PERIOD: Duration = Duration::from_secs(1);
/// Inference failures in a row before the process stops. One failed frame (a buffer replaced while it was copied,
/// an allocation that failed) is dropped and the next frame is tried. Ten in a row are several seconds at the
/// device's rate: the cause is not passing, and a restart would meet it again.
const MAX_FAILURES: u32 = 10;

/// `sof` (CLOCK_BOOTTIME) on CLOCK_MONOTONIC, from both clocks read at the same moment. The two differ by the time
/// the device was suspended.
fn frame_mono_time(sof: u64, boot_now: u64, mono_now: u64) -> u64 {
    sof.saturating_sub(boot_now.saturating_sub(mono_now))
}

/// Publish with logMonoTime on CLOCK_MONOTONIC, the clock of the Python publishers this process is read next to.
fn send(publisher: &Pub, out: &VisionOut) -> io::Result<()> {
    if publisher.send(&ccnc_vision_msg::encode(out, clock_ns(libc::CLOCK_MONOTONIC))) {
        Ok(())
    } else {
        Err(io::Error::other(format!("send {}", ccnc_vision_msg::SERVICE)))
    }
}

/// The once-a-second message while no frame is published.
struct Quiet {
    last: Option<Instant>,
}

impl Quiet {
    fn report(&mut self, publisher: &Pub, status: Status, view: ccnc_vision_msg::View, calibration: Calibration) -> io::Result<()> {
        if self.last.is_some_and(|last| last.elapsed() < STATUS_PERIOD) {
            return Ok(());
        }
        self.last = Some(Instant::now());
        let out = VisionOut {
            status,
            frame_id: 0,
            timestamp_sof: 0,
            frame_mono_time: 0,
            view,
            processing_ms: 0.0,
            night_corrected: false,
            calibration,
            boxes: Vec::new(),
            kinds: Vec::new(),
        };
        send(publisher, &out)
    }

    /// The classifier's answers go out as they come, in a message of their own.
    fn kinds(&mut self, publisher: &Pub, kinds: Vec<KindResult>, calibration: Calibration) -> io::Result<()> {
        let out = VisionOut {
            status: Status::Kinds,
            frame_id: 0,
            timestamp_sof: 0,
            frame_mono_time: 0,
            view: View::CropFront,
            processing_ms: 0.0,
            night_corrected: false,
            calibration,
            boxes: Vec::new(),
            kinds,
        };
        send(publisher, &out)
    }

    /// A frame went out: the next status message waits a full period.
    fn frame(&mut self) {
        self.last = Some(Instant::now());
    }
}

/// Stay alive and say so once a second: a restart would fail the same way, and the reason file tells the two
/// apart from a crash.
/// After the detector's first inference, whichever camera it was: catch any ORT worker thread created lazily then.
fn recheck_threads(sched_cfg: &sched::Config, applied: &sched::Applied, done: &mut bool) {
    if *done {
        return;
    }
    let after = sched_cfg.apply_all_threads();
    if after.threads_seen != applied.threads_seen {
        eprintln!(
            "ccnc_visiond: sched after first run active={} tasks={}/{}/{}",
            after.active_cpus, after.policy_ok, after.affinity_ok, after.threads_seen,
        );
    }
    *done = true;
}

fn halt(stage: &str, err: anyhow::Error, publisher: &Pub) -> io::Result<()> {
    let reason = format!("{stage}: {err:#}");
    eprintln!("ccnc_visiond: publishing stopped, {reason}");
    let path = format!("{}.halted", state_path());
    if let Err(err) = std::fs::write(&path, format!("{reason}\n")) {
        eprintln!("ccnc_visiond: cannot write {path}: {err}");
    }
    let mut quiet = Quiet { last: None };
    while !STOP.load(Ordering::Relaxed) {
        quiet.report(publisher, Status::Halted, ccnc_vision_msg::View::SideL, Calibration::UNKNOWN)?;
        sleep(IDLE);
    }
    Ok(())
}

fn open<T>(queue: Option<T>, what: &str) -> io::Result<T> {
    queue.ok_or_else(|| io::Error::other(format!("open {what}")))
}

fn run() -> io::Result<()> {
    let sched_cfg = sched::Config::from_env();
    configure(&sched_cfg)?;
    if STOP.load(Ordering::Relaxed) {
        return Ok(());
    }
    let speed = open(Sub::new(SPEED_SERVICE, true), SPEED_SERVICE)?;
    // Every message is looked at: one with bad values is dropped, the one before it still counts.
    let calibration = open(Sub::new(CALIBRATION_SERVICE, false), CALIBRATION_SERVICE)?;
    // CcncMap (conflated): tells visiond when LF or RF has a camera-backed car so raw is inserted conditionally.
    let map_sub = open(Sub::new(MAP_SERVICE, true), MAP_SERVICE)?;
    let publisher = open(Pub::new(ccnc_vision_msg::SERVICE), ccnc_vision_msg::SERVICE)?;
    if let Some(reason) = std::env::var_os("CCNC_VISIOND_HALT") {
        return halt("assets", anyhow::anyhow!(reason.to_string_lossy().into_owned()), &publisher);
    }
    let assets = PathBuf::from(std::env::var("CCNC_VISIOND_ASSETS").unwrap_or_else(|_| "assets".into()));
    let loaded = detect::Detector::load(&assets, &engine_name(), sched_cfg.threads).and_then(|mut detector| {
        if let Some(dir) = std::env::var_os("CCNC_VISIOND_FIXED_MAPS") {
            detector.fixed_maps(std::path::Path::new(&dir))?;
            eprintln!("ccnc_visiond: fixed side view maps from {}, liveCalibration not used", dir.to_string_lossy());
        }
        Ok(detector)
    });
    let mut detector = match loaded {
        Ok(detector) => detector,
        Err(err) => return halt("load", err, &publisher),
    };
    let kind_model = assets.join(classify::MODEL).to_string_lossy().into_owned();
    let kind_engine = engine_name();
    let mut classifier = match classify::Classifier::start(move || engine::open(&kind_engine, &kind_model, [1, 3, 128, 128], 1)) {
        Ok(classifier) => classifier,
        Err(err) => return halt("load", err.context("kind model"), &publisher),
    };
    let applied = sched_cfg.apply_all_threads();
    eprintln!(
        "ccnc_visiond: sched policy={} cpus={} active={} threads={} min_period_ms={} front_ms={} tasks={}/{}/{}",
        sched_cfg.policy.name(),
        sched_cfg.cpus.format(),
        applied.active_cpus,
        sched_cfg.threads,
        sched_cfg.min_period.as_millis(),
        sched_cfg.front.map_or("off".into(), |p| format!("{}/{}", p.moving_ns / 1_000_000, p.standing_ns / 1_000_000)),
        applied.policy_ok,
        applied.affinity_ok,
        applied.threads_seen,
    );
    let mut camera = vipc::Client::new(vipc::Stream::Wide, CAMERA_SERVER);
    let mut road = vipc::Client::new(vipc::Stream::Road, CAMERA_SERVER);
    // When the last road frame's detection ended: the pace is the rest after it, so a slow detection never runs
    // back to back.
    let mut last_front = None;
    let mut front_pace = sched_cfg.front;
    // After a night road frame the road camera rests (night frames get no answer, so the map's recheck stays open).
    let mut front_rest_until = 0u64;
    let mut road_wrong_size = None;
    let mut gate = gate::Gate::default();
    let mut calib = calib::Calib::default();
    let mut rejected = None;
    let mut calibrated = false;
    let mut quiet = Quiet { last: None };
    let mut failures = 0;
    let mut wrong_size = None;
    let mut threads_rechecked = false;
    while !STOP.load(Ordering::Relaxed) {
        for bytes in speed.drain() {
            if let Some(v_ego) = ccnc_vision_msg::car_state_speed(&bytes) {
                gate.speed(v_ego, clock_ns(libc::CLOCK_MONOTONIC));
            }
        }
        for bytes in map_sub.drain() {
            if let Some(front_on) = ccnc_vision_msg::map_front_vision_on(&bytes) {
                gate.front_vision(front_on, clock_ns(libc::CLOCK_MONOTONIC));
            }
            if let Some(ask) = ccnc_vision_msg::map_front_kind(&bytes) {
                gate.front_kind(ask, clock_ns(libc::CLOCK_MONOTONIC));
            }
        }
        for bytes in calibration.drain() {
            let Some(message) = ccnc_vision_msg::live_calibration(&bytes) else { continue };
            match calib.update(&message) {
                Ok(()) => rejected = None,
                Err(why) => {
                    if rejected != Some(why) {
                        eprintln!("ccnc_visiond: liveCalibration ignored ({why:?}), the earlier values stay");
                    }
                    rejected = Some(why);
                }
            }
        }
        if calibrated != calib.usable().is_some() {
            calibrated = !calibrated;
            eprintln!("ccnc_visiond: liveCalibration {}", if calibrated { "calibrated" } else { "not calibrated, nothing is processed" });
        }
        let latest = calib.usable().copied().unwrap_or(Calibration::UNKNOWN);
        for (answers, calibration) in classifier.answers() {
            if !answers.is_empty() {
                quiet.kinds(&publisher, answers, calibration)?;
            }
        }
        if front_pace.is_some() && !classifier.alive() {
            // No restart: the side views go on, the road camera stops (its boxes are only for the classifier).
            eprintln!("ccnc_visiond: kind thread ended; road camera off until restart");
            front_pace = None;
        }
        // Road camera answers are placed with liveCalibration: none without it.
        if let Some(pace) = front_pace.filter(|_| calibrated) {
            let now = clock_ns(libc::CLOCK_MONOTONIC);
            if now >= front_rest_until && gate.front_due(now, last_front, pace) {
                if let Some(frame) = road.poll() {
                    let size = (frame.layout.width, frame.layout.height);
                    if size != wcam::WIDE_SIZE {
                        last_front = Some(now);
                        if road_wrong_size != Some(size) {
                            eprintln!("ccnc_visiond: road camera frame {}x{}, not {}x{}; not used", size.0, size.1, wcam::WIDE_SIZE.0, wcam::WIDE_SIZE.1);
                        }
                        road_wrong_size = Some(size);
                        continue;
                    }
                    let started = Instant::now();
                    let found = detector.run_front(frame.layout, &frame.nv12);
                    last_front = Some(clock_ns(libc::CLOCK_MONOTONIC));
                    if found.is_ok() {
                        recheck_threads(&sched_cfg, &applied, &mut threads_rechecked);
                    }
                    match found {
                        Ok(found) if !found.night => {
                            let (boot_now, mono_now) = (clock_ns(libc::CLOCK_BOOTTIME), clock_ns(libc::CLOCK_MONOTONIC));
                            let frame_mono = frame_mono_time(frame.timestamp_sof, boot_now, mono_now);
                            let dets: Vec<_> = found.boxes.iter().map(|(d, _)| *d).collect();
                            let picked = classify::pick(&dets, |d| found.boxes.iter().any(|(b, cut)| *cut && b.b == d.b));
                            classifier.submit(classify::Job::new(frame_mono, View::CropFront, latest, &found.image, &picked));
                            let out = VisionOut {
                                status: Status::Frame,
                                frame_id: frame.frame_id,
                                timestamp_sof: frame.timestamp_sof,
                                frame_mono_time: frame_mono,
                                view: View::CropFront,
                                processing_ms: started.elapsed().as_secs_f32() * 1000.0,
                                night_corrected: false,
                                calibration: latest,
                                boxes: found
                                    .boxes
                                    .iter()
                                    .map(|(b, cut)| VisionBox { xyxy: b.b.map(|v| v as f32), confidence: b.score, class_id: b.cls as u8, cut: *cut })
                                    .collect(),
                                kinds: Vec::new(),
                            };
                            send(&publisher, &out)?;
                            quiet.frame();
                        }
                        Ok(_) => front_rest_until = last_front.unwrap_or(now) + FRONT_NIGHT_REST_NS,
                        Err(err) => eprintln!("ccnc_visiond: road frame {} dropped: {err:#}", frame.frame_id),
                    }
                    continue;
                }
            }
        }
        let ready = detector.fixed() || calibrated;
        match gate.phase(clock_ns(libc::CLOCK_MONOTONIC), ready) {
            gate::Phase::Closed => {
                camera.discard();
                let status = if ready { Status::Resting } else { Status::NotCalibrated };
                quiet.report(&publisher, status, gate.next_target().view(), latest)?;
                sleep(IDLE);
                continue;
            }
            // Frames of the time above the speed limit are never processed later, including the ones that came
            // during the last sleep.
            gate::Phase::Opened => camera.discard(),
            gate::Phase::Open => {}
        }
        if let Some(took) = calib.usable().and_then(|c| detector.calibrate(c)) {
            // Frames that came while building are skipped: the camera gives only the latest one.
            eprintln!(
                "ccnc_visiond: side views built in {:.1} ms for rpyCalib {:?} wideFromDeviceEuler {:?}",
                took.as_secs_f64() * 1000.0,
                latest.rpy_calib,
                latest.wide_from_device_euler
            );
        }
        let started = Instant::now();
        let Some(frame) = camera.poll() else {
            // Without a camera server, try again at the idle rate instead of the frame rate.
            if camera.connected() {
                sleep(FRAME_WAIT);
            } else {
                quiet.report(&publisher, Status::NoCamera, gate.next_target().view(), latest)?;
                sleep(IDLE);
            }
            continue;
        };
        let size = (frame.layout.width, frame.layout.height);
        if size != wcam::WIDE_SIZE {
            if wrong_size != Some(size) {
                eprintln!("ccnc_visiond: wide camera frame {}x{}, the lens model is for {}x{}; frames are not processed", size.0, size.1, wcam::WIDE_SIZE.0, wcam::WIDE_SIZE.1);
            }
            wrong_size = Some(size);
            quiet.report(&publisher, Status::UnsupportedCamera, gate.next_target().view(), latest)?;
            sleep(IDLE);
            continue;
        }
        wrong_size = None;
        let target = gate.take_target(clock_ns(libc::CLOCK_MONOTONIC));
        let found = match detector.run(frame.layout, &frame.nv12, target) {
            Ok(found) => {
                failures = 0;
                recheck_threads(&sched_cfg, &applied, &mut threads_rechecked);
                found
            }
            Err(err) => {
                failures += 1;
                if failures >= MAX_FAILURES {
                    return halt("inference", err, &publisher);
                }
                if failures == 1 {
                    eprintln!("ccnc_visiond: frame {} dropped, inference failed: {err:#}", frame.frame_id);
                }
                continue;
            }
        };
        // The view's angles are the ones it was built with. The height is not part of the view: the latest one.
        let calibration = if detector.fixed() { found.calibration } else { Calibration { height: latest.height, ..found.calibration } };
        let (boot_now, mono_now) = (clock_ns(libc::CLOCK_BOOTTIME), clock_ns(libc::CLOCK_MONOTONIC));
        let frame_mono = frame_mono_time(frame.timestamp_sof, boot_now, mono_now);
        if classify::SIDE_KINDS && target != gate::Target::Raw && !found.night {
            let picked = classify::pick(&found.boxes, |_| false);
            classifier.submit(classify::Job::new(frame_mono, target.view(), calibration, &found.image, &picked));
        }
        let out = VisionOut {
            status: Status::Frame,
            frame_id: frame.frame_id,
            timestamp_sof: frame.timestamp_sof,
            frame_mono_time: frame_mono,
            view: target.view(),
            processing_ms: started.elapsed().as_secs_f32() * 1000.0,
            night_corrected: found.night,
            calibration,
            boxes: found
                .boxes
                .iter()
                .map(|b| VisionBox {
                    xyxy: b.b.map(|v| v as f32),
                    confidence: b.score,
                    class_id: b.cls as u8,
                    cut: false,
                })
                .collect(),
            kinds: Vec::new(),
        };
        send(&publisher, &out)?;
        quiet.frame();
        if !sched_cfg.min_period.is_zero() {
            sleep_interruptible(sched_cfg.min_period.saturating_sub(started.elapsed()));
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ccnc_visiond: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process::Command};

    fn cereal_queue_sizes(services: &[&str]) -> Vec<u64> {
        let sizes = Command::new("python3")
            .arg("-c")
            .arg("import runpy,sys; s=runpy.run_path('../../../cereal/services.py')['SERVICE_LIST']; print(*(int(s[n].queue_size) for n in sys.argv[1:]))")
            .args(services)
            .output()
            .unwrap();
        assert!(sizes.status.success(), "{}", String::from_utf8_lossy(&sizes.stderr));
        let sizes: Vec<u64> = String::from_utf8(sizes.stdout).unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
        assert_eq!(sizes.len(), services.len());
        sizes
    }

    #[test]
    fn largest_message_fits_the_queue() {
        // One box per detector output row at most. msgq asserts 3 * (8-byte length + message, 8-byte aligned) <= queue.
        let boxes = vec![VisionBox { xyxy: [0.0; 4], confidence: 1.0, class_id: 79, cut: false }; detect::MAX_BOXES];
        let out = VisionOut {
            status: Status::Frame,
            frame_id: 1,
            timestamp_sof: 1,
            frame_mono_time: 1,
            view: ccnc_vision_msg::View::SideR,
            processing_ms: 1.0,
            night_corrected: true,
            calibration: ccnc_vision_msg::Calibration::UNKNOWN,
            boxes,
            kinds: Vec::new(),
        };
        let bytes = ccnc_vision_msg::encode(&out, 1).len() as u64;
        let queue = cereal_queue_sizes(&[ccnc_vision_msg::SERVICE])[0];
        assert!(3 * (8 + bytes.div_ceil(8) * 8) <= queue, "{bytes} bytes with {} boxes, queue {queue}", detect::MAX_BOXES);
        // Answers wait at most for one detection: a few jobs' worth.
        let kind = KindResult { frame_mono_time: 1, view: View::CropFront, xyxy: [0.0; 4], kind: 2, prob: 1.0 };
        let answers = VisionOut { status: Status::Kinds, boxes: Vec::new(), kinds: vec![kind; 16 * classify::MAX_CROPS], ..out };
        assert!(ccnc_vision_msg::encode(&answers, 1).len() < bytes as usize);
    }

    #[test]
    fn frame_time_moves_to_the_monotonic_clock() {
        // A device that was suspended for an hour: CLOCK_BOOTTIME runs 3600 s ahead of CLOCK_MONOTONIC.
        let (mono_now, offset) = (50_000_000_000, 3_600_000_000_000);
        let boot_now = mono_now + offset;
        let sof = boot_now - 120_000_000;
        assert_eq!(frame_mono_time(sof, boot_now, mono_now), mono_now - 120_000_000);
        assert_eq!(frame_mono_time(sof, mono_now, mono_now), sof, "same clocks");
        assert_eq!(frame_mono_time(5, boot_now, mono_now), 0, "never below zero");
    }

    #[test]
    fn opens_every_queue_with_cereal_size() {
        const HEADER_BYTES: u64 = 8 * (3 + 3 * 40); // msgq_header_t in msgq_repo/msgq/msgq.h
        let prefix = format!("visiond_queues_{}_{}", std::process::id(), clock_ns(libc::CLOCK_MONOTONIC));
        let path = PathBuf::from(format!("/dev/shm/msgq_{prefix}"));
        fs::create_dir(&path).unwrap();
        std::env::set_var("OPENPILOT_PREFIX", prefix);
        let _speed = Sub::new(SPEED_SERVICE, true).unwrap();
        let _calibration = Sub::new(CALIBRATION_SERVICE, false).unwrap();
        let _map = Sub::new(MAP_SERVICE, true).unwrap();
        let _publisher = Pub::new(ccnc_vision_msg::SERVICE).unwrap();
        let services = [SPEED_SERVICE, CALIBRATION_SERVICE, MAP_SERVICE, ccnc_vision_msg::SERVICE];
        let expected = cereal_queue_sizes(&services);
        for (name, queue_size) in services.into_iter().zip(expected) {
            assert_eq!(fs::metadata(path.join(name)).unwrap().len(), HEADER_BYTES + queue_size, "{name}");
        }
        fs::remove_dir_all(&path).unwrap();
    }
}
