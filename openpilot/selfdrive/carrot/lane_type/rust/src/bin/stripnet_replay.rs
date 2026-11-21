//! Offline replay: the daemon step on recorded messages and decoded frames streamed on stdin, one JSON line per modelV2
//! on stdout (tools/lane_type/stripnet_rust_parity.py drives it next to the Python daemon code).
//!
//! stdin records: tag byte, u32 LE length, payload.
//!   b'E' a framed log.Event (capnp bytes as in the rlog);
//!   b'F' a camera frame: u32 frameId, u64 timestampEof, u32 width, u32 height, then NV12 (stride = width);
//!   b'N' u64 monotonic now for the next modelV2's processing (publish time; the replay has no clock).
//! Frames are kept like the device FrameBuffer (newest 3); there are no hold / slot-reuse drops offline.
//! Args: <assets dir> [--threads N]

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;
use stripnet::daemon::{Buffered, Daemon, FrameBuffer, Step};
use stripnet::messages::{self, Payload};
use stripnet::model::StripModel;
use stripnet::recogniser::Recogniser;
use stripnet::strip::Nv12;

struct Owned {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let assets = PathBuf::from(args.get(1).map_or("../assets", |s| s.as_str()));
    let threads = args.iter().position(|a| a == "--threads").and_then(|i| args.get(i + 1)).map_or(2, |v| v.parse().unwrap());
    let model = StripModel::load(&assets, "lane_strip_v4")?;
    let mut recogniser = Recogniser::new(model);
    recogniser.threads = threads;
    recogniser.record = true;
    let mut daemon = Daemon::new(recogniser);
    let mut frames: FrameBuffer<Owned> = FrameBuffer::default();
    let mut stdin = std::io::stdin().lock();
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut now_next: i64 = 0;
    let mut head = [0u8; 5];
    let mut buf = Vec::new();
    loop {
        if stdin.read_exact(&mut head).is_err() {
            break;
        }
        let len = u32::from_le_bytes(head[1..5].try_into().unwrap()) as usize;
        buf.resize(len, 0);
        stdin.read_exact(&mut buf)?;
        match head[0] {
            b'N' => now_next = i64::from_le_bytes(buf[..8].try_into().unwrap()),
            b'F' => {
                let frame_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                let eof = u64::from_le_bytes(buf[4..12].try_into().unwrap());
                let width = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
                let height = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
                frames.push(Buffered { frame_id, eof, buf: Owned { width, height, data: buf[20..].to_vec() }, recv_ns: 0 });
            }
            b'E' => {
                let Some(event) = messages::parse(&buf) else {
                    writeln!(out, "{}", json!({"status": "unreadable_event"}))?;
                    continue;
                };
                let Payload::ModelV2(model) = &event.payload else {
                    daemon.latest.take(&event);
                    continue;
                };
                let picked = frames.select(model.frame_id, model.timestamp_eof);
                let image = picked.map(|(f, _)| Nv12::from_buffer(&f.buf.data, f.buf.width, f.buf.height, f.buf.width, f.buf.width * f.buf.height).unwrap());
                let arg = picked.zip(image.as_ref()).map(|((f, kind), img)| (f.frame_id, f.eof, img, kind, f.recv_ns));
                let t0 = Instant::now();
                let cpu0 = thread_cpu_ns();
                let step = daemon.step(model, event.log_mono_time as i64, arg, || now_next, |_| None);
                let wall_ms = t0.elapsed().as_nanos() as f64 / 1e6;
                let cpu_ms = (thread_cpu_ns() - cpu0) as f64 / 1e6;
                let line = match step {
                    Step::Published { packet, .. } => {
                        let diag: Vec<_> = daemon.recogniser.last.iter().map(|d| d.as_ref().map(|d| json!({
                            "strip": d.strip_sha, "typeLogits": d.type_logits, "colorLogits": d.color_logits}))).collect();
                        let t = daemon.recogniser.timing;
                        json!({"status": "published", "frameId": model.frame_id, "packet": packet, "diag": diag,
                               "wallMs": wall_ms, "cpuMs": cpu_ms, "extractMs": t.extract_ms, "inferMs": t.infer_ms})
                    }
                    Step::Mismatch => json!({"status": "frame_mismatch", "frameId": model.frame_id}),
                    Step::Stale => json!({"status": "stale_inputs", "frameId": model.frame_id}),
                    Step::Error(e) => json!({"status": "error", "frameId": model.frame_id, "error": e}),
                    Step::Dropped(d) => json!({"status": d, "frameId": model.frame_id}),
                };
                writeln!(out, "{line}")?;
                out.flush()?;
            }
            t => anyhow::bail!("unknown record {t}"),
        }
    }
    Ok(())
}

/// Process CPU time (all threads) — the extraction / inference threads are scoped per frame.
fn thread_cpu_ns() -> i64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    // CLOCK_PROCESS_CPUTIME_ID = 2 on Linux.
    unsafe { clock_gettime(2, &mut ts) };
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
