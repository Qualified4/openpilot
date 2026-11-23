//! Feed the production pipeline an ordered stream of cached cereal Events and emit CcncMap Events.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
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

fn main() -> io::Result<()> {
    let mut paths = std::env::args_os().skip(1).peekable();
    let bench = paths.next_if(|arg| arg == "--bench").is_some();
    let chain = paths.next_if(|arg| arg == "--chain").is_some();
    if paths.peek().is_none() {
        return Err(io::Error::other(
            "usage: ccnc_mapd_replay [--bench] [--chain] CACHE.capnp ...",
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
        let mut reader = BufReader::new(File::open(path)?);
        let buffers = |frame: &Vec<u8>, times: &Vec<u128>| {
            frame.capacity() + times.capacity() * std::mem::size_of::<u128>()
        };
        if !chain {
            pipeline = None;
            baseline = None;
        }
        let base =
            *baseline.get_or_insert_with(|| HEAP_BYTES.load(Relaxed) - buffers(&frame, &times));
        let pipeline = pipeline.get_or_insert_with(|| Pipeline::new(0));
        while framing::next(&mut reader, &mut frame)? {
            if let Some(message) = vision::message(&frame) {
                pipeline.kinds(&message.kinds, &message.calibration);
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
            let start = bench.then(Instant::now);
            let result = pipeline
                .ingest(&frame)
                .map_err(|err| io::Error::other(format!("{err:?}")))?;
            if let Some(map) = result {
                let bytes = cmap::encode(&map);
                if let Some(start) = start {
                    times.push(start.elapsed().as_nanos());
                    valid += usize::from(map.valid);
                } else {
                    out.write_all(&bytes)?;
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
        let p99 = times[(count * 99).div_ceil(100) - 1] as f64 / 1e3;
        eprintln!("radar_frames={count} valid={valid} p99_us={p99:.2} state_bytes={max_state}");
        if !vision_times.is_empty() {
            vision_times.sort_unstable();
            let at = |q: usize| vision_times[(vision_times.len() * q).div_ceil(100).max(1) - 1] as f64 / 1e3;
            eprintln!("vision_frames={} p50_us={:.2} p99_us={:.2}", vision_times.len(), at(50), at(99));
        }
    }
    out.flush()
}
