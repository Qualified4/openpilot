//! Feed the production pipeline an ordered stream of cached cereal Events and emit CcncMap Events.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use ccnc_map::vision::{Geometry, Skipped};
use ccnc_map::{cmap, input::framing, pipeline::Pipeline};

#[path = "../vision.rs"]
#[allow(dead_code)] // The geometry thread is the daemon's; the replay builds in line.
mod vision;

struct RetainedHeap;
static HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: RetainedHeap = RetainedHeap;

unsafe impl GlobalAlloc for RetainedHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            HEAP_BYTES.fetch_add(layout.size(), Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HEAP_BYTES.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn thread_cpu_ns() -> u128 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) }, 0);
    time.tv_sec as u128 * 1_000_000_000 + time.tv_nsec as u128
}

fn write_evidence(out: &mut impl Write, stamp: u64, cpu_ns: u128, pipeline: &Pipeline) -> io::Result<()> {
    let now = stamp as f64 * 1e-9;
    write!(out, "{stamp}\t{cpu_ns}\t")?;
    for o in pipeline.objects() {
        write!(
            out,
            "{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{},{},{};",
            o.obj_id, o.evidence.last_track_id.unwrap_or(0), o.s, o.d, o.lane_u, o.v_s, o.existence_p(),
            u8::from(o.evidence.last_radar_t == Some(now)),
            u8::from(o.evidence.last_vision_t.is_some_and(|t| now - t <= 0.14)),
            u8::from(o.evidence.last_lead_t.is_some_and(|t| now - t <= 0.16))
        )?;
    }
    writeln!(out)
}

fn main() -> io::Result<()> {
    let mut paths = std::env::args_os().skip(1).peekable();
    let bench = paths.next_if(|arg| arg == "--bench").is_some();
    let chain = paths.next_if(|arg| arg == "--chain").is_some();
    let mut trace = if paths.next_if(|arg| arg == "--trace").is_some() {
        let path = paths.next().ok_or_else(|| io::Error::other("--trace requires a path"))?;
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };
    if paths.peek().is_none() {
        return Err(io::Error::other(
            "usage: ccnc_mapd_replay [--bench] [--chain] [--trace FILE] CACHE.capnp ...",
        ));
    }

    let mut out = BufWriter::new(io::stdout().lock());
    let mut frame = Vec::new();
    let mut times = Vec::new();
    let mut vision_times = Vec::new();
    let mut max_state = 0;
    let mut valid = 0;
    let mut pipeline = None;
    let mut baseline = None;
    for path in paths {
        let path = Path::new(&path);
        let mut reader = BufReader::new(File::open(path)?);
        if let Some(trace) = &mut trace {
            writeln!(trace, "# {}", path.file_stem().unwrap_or_default().to_string_lossy())?;
        }
        let buffers = |frame: &Vec<u8>, times: &Vec<u128>| {
            frame.capacity() + times.capacity() * std::mem::size_of::<u128>()
        };
        if !chain {
            pipeline = None;
            baseline = None;
        }
        let base = *baseline.get_or_insert_with(|| {
            HEAP_BYTES.load(Relaxed).saturating_sub(buffers(&frame, &times))
        });
        let pipeline = pipeline.get_or_insert_with(|| Pipeline::new(0));
        while framing::next(&mut reader, &mut frame)? {
            if let Some(message) = vision::message(&frame) {
                message.observe_front(pipeline);
                let Some(camera) = message.frame else { continue };
                let stamp = message.stamp;
                // Built in line so a replay uses the first frame too, where the daemon skips frames until built.
                let start = Instant::now();
                if let Err(Skipped::NeedsGeometry(calib)) = pipeline.vision(stamp, &camera) {
                    pipeline.set_vision_geometry(Geometry::build(&calib));
                    let start = Instant::now();
                    let _ = pipeline.vision(stamp, &camera);
                    vision_times.push(start.elapsed().as_nanos());
                } else {
                    vision_times.push(start.elapsed().as_nanos());
                }
                continue;
            }
            let start = bench.then(thread_cpu_ns);
            let result = pipeline
                .ingest(&frame)
                .map_err(|err| io::Error::other(format!("{err:?}")))?;
            if let Some(map) = result {
                let bytes = cmap::encode(&map);
                let cpu_ns = start.map_or(0, |start| thread_cpu_ns() - start);
                if bench {
                    times.push(cpu_ns);
                    valid += usize::from(map.valid);
                }
                if !bench || trace.is_some() {
                    out.write_all(&bytes)?;
                }
                if let Some(trace) = &mut trace {
                    write_evidence(trace, map.radar_log_mono_time, cpu_ns, pipeline)?;
                }
            }
            if bench {
                // Map state retained after the update, excluding input and timing buffers.
                let buffer_growth =
                    frame.capacity() + times.capacity() * std::mem::size_of::<u128>();
                let heap = HEAP_BYTES
                    .load(Relaxed)
                    .saturating_sub(base + buffer_growth);
                max_state = max_state.max(std::mem::size_of_val(pipeline) + heap);
            }
        }
    }
    if bench {
        times.sort_unstable();
        let count = times.len();
        if count == 0 {
            return Err(io::Error::other("no liveTracks events"));
        }
        let at = |q: usize| times[(count * q).div_ceil(100) - 1] as f64 / 1e3;
        eprintln!(
            "radar_frames={count} valid={valid} p50_us={:.2} p95_us={:.2} p99_us={:.2} max_us={:.2} state_bytes={max_state}",
            at(50), at(95), at(99), at(100)
        );
        if !vision_times.is_empty() {
            vision_times.sort_unstable();
            let at = |q: usize| vision_times[(vision_times.len() * q).div_ceil(100).max(1) - 1] as f64 / 1e3;
            eprintln!("vision_frames={} p50_us={:.2} p99_us={:.2}", vision_times.len(), at(50), at(99));
        }
    }
    if let Some(trace) = &mut trace {
        trace.flush()?;
    }
    out.flush()
}
