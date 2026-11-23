//! ccnc_visiond: vehicle boxes from the wide camera's side views while the car stands or creeps.
//! Input: VisionIPC `wideRoad` and `carState.vEgo`. Output: one CcncVision (`customReserved4`) per processed frame.
//!
//! Environment:
//! * `CCNC_VISIOND_ASSETS`: directory with `yolox_tiny_dyn.onnx`, `map_vL.bin` and `map_vR.bin` (default `assets`).
//! * `CCNC_VISIOND_ENGINE`: `ort` or `tract`. Needed while both backends are built in.
//! * `ORT_DYLIB_PATH`: libonnxruntime.so for the `ort` backend.

mod detect;
mod engine;
mod gate;
mod night;
mod nv12;
mod pre;
mod remap;
mod vipc;
mod yolox;

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use ccnc_vision_msg::{VisionBox, VisionOut};
use stripnet::ipc::{Pub, Sub};

const SPEED_SERVICE: &str = "carState";
const CAMERA_SERVER: &str = "camerad";
/// Sleep while the speed keeps the detector off.
const IDLE: Duration = Duration::from_millis(100);
/// Sleep while waiting for the next camera frame (20 Hz).
const FRAME_WAIT: Duration = Duration::from_millis(5);
const NICE: libc::c_int = 10;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn request_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn configure() -> io::Result<()> {
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
        // The inference threads start later and inherit this.
        if libc::setpriority(libc::PRIO_PROCESS, 0, NICE) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    assert_eq!(unsafe { libc::clock_gettime(clock, &mut ts) }, 0);
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn state_path() -> String {
    std::env::var("VISIOND_STATE").unwrap_or_else(|_| "/dev/shm/ccnc_visiond".into())
}

/// The backend named by CCNC_VISIOND_ENGINE, or the only one built in.
fn engine_name() -> anyhow::Result<String> {
    if let Ok(name) = std::env::var("CCNC_VISIOND_ENGINE") {
        return Ok(name);
    }
    match (cfg!(feature = "ort"), cfg!(feature = "tract")) {
        (true, false) => Ok("ort".into()),
        (false, true) => Ok("tract".into()),
        _ => anyhow::bail!("CCNC_VISIOND_ENGINE must name the backend (ort or tract)"),
    }
}

/// Stay alive without publishing: a restart would fail the same way, and the reason file tells the two apart
/// from a crash.
fn halt(stage: &str, err: anyhow::Error) {
    let reason = format!("{stage}: {err:#}");
    eprintln!("ccnc_visiond: publishing stopped, {reason}");
    let path = format!("{}.halted", state_path());
    if let Err(err) = std::fs::write(&path, format!("{reason}\n")) {
        eprintln!("ccnc_visiond: cannot write {path}: {err}");
    }
    while !STOP.load(Ordering::Relaxed) {
        sleep(IDLE);
    }
}

fn open<T>(queue: Option<T>, what: &str) -> io::Result<T> {
    queue.ok_or_else(|| io::Error::other(format!("open {what}")))
}

fn run() -> io::Result<()> {
    configure()?;
    if STOP.load(Ordering::Relaxed) {
        return Ok(());
    }
    let speed = open(Sub::new(SPEED_SERVICE, true), SPEED_SERVICE)?;
    let publisher = open(Pub::new(ccnc_vision_msg::SERVICE), ccnc_vision_msg::SERVICE)?;
    let assets = PathBuf::from(std::env::var("CCNC_VISIOND_ASSETS").unwrap_or_else(|_| "assets".into()));
    let loaded = engine_name().and_then(|engine| detect::Detector::load(&assets, &engine));
    let mut detector = match loaded {
        Ok(detector) => detector,
        Err(err) => {
            halt("load", err);
            return Ok(());
        }
    };
    let mut camera = vipc::Client::new(vipc::Stream::Wide, CAMERA_SERVER);
    let mut gate = gate::Gate::default();
    while !STOP.load(Ordering::Relaxed) {
        for bytes in speed.drain() {
            if let Some(v_ego) = ccnc_vision_msg::car_state_speed(&bytes) {
                gate.speed(v_ego, clock_ns(libc::CLOCK_MONOTONIC));
            }
        }
        match gate.phase(clock_ns(libc::CLOCK_MONOTONIC)) {
            gate::Phase::Closed => {
                camera.discard();
                sleep(IDLE);
                continue;
            }
            // Frames of the time above the speed limit are never processed later, including the ones that came
            // during the last sleep.
            gate::Phase::Opened => camera.discard(),
            gate::Phase::Open => {}
        }
        let started = Instant::now();
        let Some(frame) = camera.poll() else {
            // Without a camera server, try again at the idle rate instead of the frame rate.
            sleep(if camera.connected() { FRAME_WAIT } else { IDLE });
            continue;
        };
        let side = gate.take_side();
        let found = match detector.run(frame.layout, &frame.nv12, side) {
            Ok(found) => found,
            Err(err) => {
                halt("inference", err);
                return Ok(());
            }
        };
        let out = VisionOut {
            frame_id: frame.frame_id,
            timestamp_sof: frame.timestamp_sof,
            view: side.view(),
            processing_ms: started.elapsed().as_secs_f32() * 1000.0,
            night_corrected: found.night,
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
        };
        // logMonoTime is on the camera's clock, so a reader gets the age of the frame from the two stamps.
        if !publisher.send(&ccnc_vision_msg::encode(&out, clock_ns(libc::CLOCK_BOOTTIME))) {
            return Err(io::Error::other(format!("send {}", ccnc_vision_msg::SERVICE)));
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
        let out = VisionOut { frame_id: 1, timestamp_sof: 1, view: ccnc_vision_msg::View::SideR, processing_ms: 1.0, night_corrected: true, boxes };
        let bytes = ccnc_vision_msg::encode(&out, 1).len() as u64;
        let queue = cereal_queue_sizes(&[ccnc_vision_msg::SERVICE])[0];
        assert!(3 * (8 + bytes.div_ceil(8) * 8) <= queue, "{bytes} bytes with {} boxes, queue {queue}", detect::MAX_BOXES);
    }

    #[test]
    fn opens_every_queue_with_cereal_size() {
        const HEADER_BYTES: u64 = 8 * (3 + 3 * 40); // msgq_header_t in msgq_repo/msgq/msgq.h
        let prefix = format!("visiond_queues_{}_{}", std::process::id(), clock_ns(libc::CLOCK_MONOTONIC));
        let path = PathBuf::from(format!("/dev/shm/msgq_{prefix}"));
        fs::create_dir(&path).unwrap();
        std::env::set_var("OPENPILOT_PREFIX", prefix);
        let _speed = Sub::new(SPEED_SERVICE, true).unwrap();
        let _publisher = Pub::new(ccnc_vision_msg::SERVICE).unwrap();
        let services = [SPEED_SERVICE, ccnc_vision_msg::SERVICE];
        let expected = cereal_queue_sizes(&services);
        for (name, queue_size) in services.into_iter().zip(expected) {
            assert_eq!(fs::metadata(path.join(name)).unwrap().len(), HEADER_BYTES + queue_size, "{name}");
        }
        fs::remove_dir_all(&path).unwrap();
    }
}
