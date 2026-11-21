//! Switches for the inference tail (p95 / p99), read once per process. Defaults: worker, state reuse on, mallopt off
//! (device 2026-11-21, README "Inference tail").
//!
//! * `STRIPNET_MALLOPT=0|1` — glibc mallopt as strip_extractor.tune_allocator (mmap threshold 64 MB, trim threshold
//!   256 MB, top pad 16 MB), so tract's per-op buffers stay on the heap instead of being returned and faulted in again.
//! * `STRIPNET_SIDES=spawn|worker|seq` — how the second side runs: a new scoped thread per phase (spawn), one resident
//!   worker thread fed through a channel (worker), or both sides one after the other on the caller (seq).
//! * `STRIPNET_TRACT_STATE=0|1` — keep one tract SimpleState per side instead of a new one per run (`model`).
//! * `STRIPNET_CPUS=0-3` — the daemon's CPU affinity, a list such as `0-2`, `1-2`, `4` or `0,2,5-6` (CPUs 0-63).

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::OnceLock;
use std::time::Instant;

fn flag(name: &str, default: bool) -> bool {
    std::env::var(name).ok().map_or(default, |v| v == "1")
}

pub fn mallopt_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("STRIPNET_MALLOPT", false))
}

pub fn tract_state_reuse() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("STRIPNET_TRACT_STATE", true))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sides {
    Spawn,
    Worker,
    Seq,
}

pub fn sides() -> Sides {
    static V: OnceLock<Sides> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("STRIPNET_SIDES").as_deref() {
        Ok("spawn") => Sides::Spawn,
        Ok("worker") => Sides::Worker,
        Ok("seq") => Sides::Seq,
        Ok(v) => {
            eprintln!("STRIPNET_SIDES={v:?} unknown: worker");
            Sides::Worker
        }
        Err(_) => Sides::Worker,
    })
}

/// CPUs the daemon may run on, as the bits of an affinity mask; the default is CPU 0-3 (the Python process's set).
pub const DEFAULT_CPUS: u64 = 0b1111;

/// A CPU list (`0-3`, `1-2`, `4`, `0,2,5-6`) as a mask; None when empty, malformed or above CPU 63.
pub fn parse_cpus(spec: &str) -> Option<u64> {
    let mut mask = 0u64;
    for part in spec.split(',') {
        let (lo, hi) = match part.trim().split_once('-') {
            Some((a, b)) => (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?),
            None => {
                let c = part.trim().parse::<u32>().ok()?;
                (c, c)
            }
        };
        if lo > hi || hi > 63 {
            return None;
        }
        for c in lo..=hi {
            mask |= 1 << c;
        }
    }
    (mask != 0).then_some(mask)
}

/// `STRIPNET_CPUS` as a mask: DEFAULT_CPUS when unset or invalid.
pub fn cpu_mask() -> u64 {
    match std::env::var("STRIPNET_CPUS") {
        Ok(v) => parse_cpus(&v).unwrap_or_else(|| {
            eprintln!("STRIPNET_CPUS={v:?} invalid: 0-3");
            DEFAULT_CPUS
        }),
        Err(_) => DEFAULT_CPUS,
    }
}

/// Pins the calling thread (and threads it starts later) to `mask`; false when the kernel refuses it (e.g. every
/// CPU in the mask is offline) and the affinity stays as it was.
pub fn pin_cpus(mask: u64) -> bool {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let mut set = [0u64; 16];
        set[0] = mask;
        return unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) } == 0;
    }
    #[allow(unreachable_code)]
    false
}

/// strip_extractor.tune_allocator; true when every mallopt call succeeded (false when off or not glibc).
pub fn tune_allocator() -> bool {
    if !mallopt_enabled() {
        return false;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        // M_MMAP_THRESHOLD, M_TRIM_THRESHOLD, M_TOP_PAD
        return [(-3, 64 << 20), (-1, 256 << 20), (-2, 16 << 20)].iter().all(|&(k, v)| unsafe { mallopt(k, v) } == 1);
    }
    #[allow(unreachable_code)]
    false
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct SlotPtr<T>(*mut Option<std::thread::Result<T>>);
unsafe impl<T: Send> Send for SlotPtr<T> {}
impl<T> SlotPtr<T> {
    fn get(self) -> *mut Option<std::thread::Result<T>> {
        self.0
    }
}

/// One resident thread that runs the second side's closures.
pub struct Worker {
    jobs: Sender<Job>,
    done: Receiver<()>,
}

impl Worker {
    pub fn new() -> Self {
        let (jobs, rx) = channel::<Job>();
        let (tx, done) = channel();
        std::thread::Builder::new()
            .name("stripnet-side".into())
            .spawn(move || {
                for job in rx {
                    job();
                    if tx.send(()).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the StripNet side worker");
        Worker { jobs, done }
    }

    /// `a` on the worker and `b` here; returns both results and the ns spent waiting for `a` after `b` finished.
    pub fn join<A: Send, B>(&self, a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B) -> (A, B, u64) {
        let mut slot: Option<std::thread::Result<A>> = None;
        let ptr = SlotPtr(&mut slot as *mut _);
        let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
            let p = ptr.get();
            let r = catch_unwind(AssertUnwindSafe(a));
            unsafe { *p = Some(r) };
        });
        // SAFETY: `job` borrows from this frame; the guard below blocks until the worker has run it (also when `b`
        // panics), so nothing it borrows is gone while it runs.
        let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, Job>(job) };
        self.jobs.send(job).expect("StripNet side worker alive");
        struct Wait<'a>(&'a Receiver<()>);
        impl Drop for Wait<'_> {
            fn drop(&mut self) {
                let _ = self.0.recv();
            }
        }
        let guard = Wait(&self.done);
        let rb = b();
        let t = Instant::now();
        drop(guard);
        let wait = t.elapsed().as_nanos() as u64;
        match slot.take().expect("StripNet side worker ran the job") {
            Ok(ra) => (ra, rb, wait),
            Err(p) => resume_unwind(p),
        }
    }
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs `a` (side 1) and `b` (side 0) the `Sides` way: (a, b, ns the caller waited for a).
pub fn pair<A: Send, B>(mode: Sides, worker: Option<&Worker>, a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B) -> (A, B, u64) {
    match (mode, worker) {
        (Sides::Worker, Some(w)) => w.join(a, b),
        (Sides::Spawn, _) => std::thread::scope(|s| {
            let h = s.spawn(a);
            let rb = b();
            let t = Instant::now();
            let ra = h.join().unwrap_or_else(|p| resume_unwind(p));
            (ra, rb, t.elapsed().as_nanos() as u64)
        }),
        _ => {
            let rb = b();
            (a(), rb, 0)
        }
    }
}

/// Minor page faults of the whole process so far (getrusage RUSAGE_SELF ru_minflt).
pub fn minflt() -> i64 {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
        }
        let mut u = [0i64; 18];
        // struct rusage: ru_utime, ru_stime (2 x timeval), then maxrss, ixrss, idrss, isrss, minflt, ...
        if unsafe { getrusage(0, &mut u) } == 0 {
            return u[8];
        }
    }
    #[allow(unreachable_code)]
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_runs_both_and_borrows() {
        let w = Worker::new();
        let mut left = vec![0u32; 4];
        let right = vec![1u32, 2, 3];
        for i in 0..100 {
            let (a, b, _) = w.join(|| right.iter().sum::<u32>() + i, || {
                left[0] += 1;
                left[0]
            });
            assert_eq!((a, b), (6 + i, i + 1));
        }
        for mode in [Sides::Spawn, Sides::Seq, Sides::Worker] {
            let (a, b, _) = pair(mode, Some(&w), || 2, || 3);
            assert_eq!((a, b), (2, 3));
        }
    }

    #[test]
    fn worker_panic_reaches_caller_and_worker_survives() {
        let w = Worker::new();
        let r = catch_unwind(AssertUnwindSafe(|| w.join(|| panic!("side 1"), || 1)));
        assert!(r.is_err());
        assert_eq!(w.join(|| 5, || 6).0, 5);
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(parse_cpus("0-3"), Some(0b1111));
        assert_eq!(parse_cpus("0-2"), Some(0b0111));
        assert_eq!(parse_cpus("1-2"), Some(0b0110));
        assert_eq!(parse_cpus("4"), Some(0b1_0000));
        assert_eq!(parse_cpus(" 0, 2,5-6 "), Some(0b110_0101));
        for bad in ["", "a", "3-1", "64", "0-", ","] {
            assert_eq!(parse_cpus(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn minflt_counts() {
        assert!(minflt() >= 0);
    }
}
