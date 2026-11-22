//! The Rust ccnc_lane_type process (feature `ipc`): marking_daemon.main's loop on msgq / VisionIPC.
//! process_config runs it through `stripnet_launch.sh` (prebuilt/linux-aarch64), which falls back to marking_daemon.py.
//!
//! Differences from the Python process that remain: the HDA1 CarParams check is left to process_config's enable
//! function (this loop only re-reads CcncExtendedDisplay every second), no swaglog (stderr), no minflt / thread CPU timing.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use stripnet::daemon::{buffer_drop_reason, Buffered, Daemon, FrameBuffer, Step, EVENT_CUSTOM_RAW_2};
use stripnet::ipc::{Pub, Sub, Vipc, VipcFrame, VISION_STREAM_ROAD};
use stripnet::messages::{self, Payload};
use stripnet::model::StripModel;
use stripnet::recogniser::Recogniser;
use stripnet::strip::Nv12;

const PARAM_CHECK: Duration = Duration::from_secs(1);
const CEREAL_OPEN_ATTEMPTS: usize = 3;
const SERVICES: [&str; 6] = ["modelV2", "liveCalibration", "carState", "deviceState", "roadCameraState", "radarState"];

fn monotonic_ns() -> i64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { clock_gettime(1, &mut ts) }; // CLOCK_MONOTONIC, the clock logMonoTime and camerad stamps use
    ts.tv_sec * 1_000_000_000 + ts.tv_nsec
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32;
}

fn enabled(params: &std::path::Path) -> bool {
    std::fs::read(params.join("CcncExtendedDisplay")).is_ok_and(|v| v.first() == Some(&b'1'))
}

fn open_cereal(params: &std::path::Path) -> anyhow::Result<Option<(Vec<Sub>, Pub)>> {
    for attempt in 1..=CEREAL_OPEN_ATTEMPTS {
        if !enabled(params) {
            return Ok(None);
        }
        let subs: Option<Vec<_>> = SERVICES.iter().map(|s| Sub::new(s, false)).collect();
        if let Some(subs) = subs {
            if let Some(publisher) = Pub::new("customReservedRawData2") {
                return Ok(Some((subs, publisher)));
            }
        }
        if attempt < CEREAL_OPEN_ATTEMPTS {
            std::thread::sleep(PARAM_CHECK);
        }
    }
    anyhow::bail!("StripNet cereal queues unavailable after {CEREAL_OPEN_ATTEMPTS} attempts")
}

fn main() -> anyhow::Result<()> {
    let params = PathBuf::from(std::env::var("PARAMS_DIR").unwrap_or_else(|_| "/data/params/d".into()));
    if !enabled(&params) {
        return Ok(());
    }
    stripnet::tail::tune_allocator();
    let assets = PathBuf::from(std::env::var("STRIPNET_ASSETS").unwrap_or_else(|_| "../assets".into()));
    let model = match StripModel::load(&assets, "lane_strip_v4") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("CCNC lane marking: lane strip model lane_strip_v4 not loaded, lane types are not shown: {e}");
            while enabled(&params) {
                std::thread::sleep(PARAM_CHECK);
            }
            return Ok(());
        }
    };
    eprintln!("CCNC lane marking: Rust StripNet daemon, model lane_strip_v4 loaded");
    unsafe {
        // The launcher waits on this process; if it is killed, do not keep publishing as an orphan.
        prctl(1, 15); // PR_SET_PDEATHSIG, SIGTERM
        if getppid() == 1 {
            return Ok(());
        }
        nice(10);
    }
    // Before Recogniser::new starts the side worker, so it inherits the mask.
    let cpus = stripnet::tail::cpu_mask();
    if !stripnet::tail::pin_cpus(cpus) {
        eprintln!("CCNC lane marking: CPU mask {cpus:#x} refused, affinity unchanged");
    }
    let Some((subs, publisher)) = open_cereal(&params)? else { return Ok(()) };
    let client = Vipc::new("camerad", VISION_STREAM_ROAD, false);
    let mut frames: FrameBuffer<VipcFrame> = FrameBuffer::default();
    let mut daemon = Daemon::new(Recogniser::new(model));
    let mut next_param = Instant::now() + PARAM_CHECK;
    loop {
        if Instant::now() >= next_param {
            next_param = Instant::now() + PARAM_CHECK;
            if !enabled(&params) {
                break;
            }
        }
        subs[0].poll(100);
        let mut model_events = Vec::new();
        for (i, s) in subs.iter().enumerate() {
            for bytes in s.drain() {
                let Some(e) = messages::parse(&bytes) else { continue };
                if i == 0 {
                    model_events.push(e);
                } else {
                    daemon.latest.take(&e);
                }
            }
        }
        if !client.connected() {
            if !client.connect(false) {
                continue;
            }
            frames.clear();
        }
        while let Some(f) = client.recv(0) {
            frames.push(Buffered { frame_id: f.frame_id, eof: f.timestamp_eof, buf: f, recv_ns: monotonic_ns() });
        }
        // SubMaster keeps only the newest modelV2.
        let Some(event) = model_events.pop() else { continue };
        let Payload::ModelV2(model) = &event.payload else { continue };
        let picked = frames.select(model.frame_id, model.timestamp_eof);
        let image = picked.and_then(|(f, _)| Nv12::from_buffer(f.buf.bytes(), f.buf.width, f.buf.height, f.buf.stride, f.buf.uv_offset));
        let arg = picked.zip(image.as_ref()).map(|((f, kind), img)| (f.frame_id, f.eof, img, kind, f.recv_ns));
        let held = picked.map(|(f, _)| (f.frame_id, f.recv_ns, &f.buf));
        let step = daemon.step(model, monotonic_ns(), arg, monotonic_ns, |now| {
            held.and_then(|(id, recv, buf)| buffer_drop_reason(id, buf.current_frame_id(), recv, now))
        });
        match step {
            Step::Published { packet, .. } => {
                let payload = serde_json::to_vec(&packet)?;
                publisher.send(&messages::raw_data_event(EVENT_CUSTOM_RAW_2, monotonic_ns() as u64, &payload));
            }
            Step::Error(e) => eprintln!("CCNC lane marking: {e}"),
            _ => {}
        }
    }
    Ok(())
}

extern "C" {
    fn nice(inc: i32) -> i32;
    fn prctl(option: i32, arg2: u64, ...) -> i32;
    fn getppid() -> i32;
}
