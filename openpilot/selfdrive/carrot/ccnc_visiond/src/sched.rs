//! CPU affinity, scheduling policy, thread count, and optional minimum frame period for `ccnc_visiond`.
//!
//! Defaults (little cores 0-2 under `SCHED_IDLE`, 2 threads):
//! * `CCNC_VISIOND_CPUS=0-2`: stays completely off all big driving cores (4: radard/plannerd/radarcan/ccnc_mapd,
//!   5: card, 6: camerad/controlsd/selfdrived, 7: modeld) and off core 3 (pandad FIFO 55 + encoderd).
//! * `CCNC_VISIOND_POLICY=idle`: `SCHED_IDLE` sits strictly below `SCHED_FIFO` (calibrationd/locationd/sensord) and
//!   below all `SCHED_OTHER` processes on cores 0-2 (loggerd, ui, ccnc_lane_type at nice 10).
//! * `CCNC_VISIOND_THREADS=2`: 2 ORT intra-op threads (spinning disabled) across 3 little cores keeps per-side
//!   cadence near ~1.3-1.7 s while leaving at least one core in 0-2 plus core 3 untouched.
//! * `CCNC_VISIOND_NICE=19`: used only when policy is `other` or `batch` (strictly below `ccnc_lane_type` nice 10).
//! * `CCNC_VISIOND_MIN_PERIOD_MS=0`: optional minimum wall-clock interval between inferences (ms).
//!
//! Big-core comparison alternative: `CCNC_VISIOND_POLICY=idle CCNC_VISIOND_CPUS=4 CCNC_VISIOND_THREADS=1`.

use std::fs;
use std::io;
use std::time::Duration;

pub const DEFAULT_CPUS_STR: &str = "0-2";
pub const DEFAULT_THREADS: usize = 2;
pub const MAX_THREADS: usize = 4;
pub const DEFAULT_NICE: libc::c_int = 19;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Idle,
    Batch,
    Other,
}

impl Policy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "idle" | "sched_idle" => Some(Self::Idle),
            "batch" | "sched_batch" => Some(Self::Batch),
            "other" | "normal" | "sched_other" => Some(Self::Other),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Batch => "batch",
            Self::Other => "other",
        }
    }

    pub fn raw(self) -> libc::c_int {
        match self {
            Self::Idle => libc::SCHED_IDLE,
            Self::Batch => libc::SCHED_BATCH,
            Self::Other => libc::SCHED_OTHER,
        }
    }
}

/// CPU affinity selection. `Some(mask)` pins to the listed CPUs (bits 0..63); `None` keeps inherited affinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuSpec {
    Mask(u64),
    Inherit,
}

impl CpuSpec {
    /// Parse a CPU list (`4`, `0-2`, `0,2,5-6`) or `all`/`none` for inherit.
    pub fn parse(spec: &str) -> Option<Self> {
        let s = spec.trim();
        if s.eq_ignore_ascii_case("all") || s.eq_ignore_ascii_case("none") || s.eq_ignore_ascii_case("inherit") {
            return Some(Self::Inherit);
        }
        let mut mask = 0u64;
        for part in s.split(',') {
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
                mask |= 1u64 << c;
            }
        }
        (mask != 0).then_some(Self::Mask(mask))
    }

    pub fn format(self) -> String {
        match self {
            Self::Inherit => "inherit".into(),
            Self::Mask(mask) => format_mask(mask),
        }
    }
}

fn format_mask(mask: u64) -> String {
    let mut parts = Vec::new();
    let mut c = 0u32;
    while c < 64 {
        if (mask & (1u64 << c)) != 0 {
            let start = c;
            while c + 1 < 64 && (mask & (1u64 << (c + 1))) != 0 {
                c += 1;
            }
            if start == c {
                parts.push(format!("{start}"));
            } else {
                parts.push(format!("{start}-{c}"));
            }
        }
        c += 1;
    }
    parts.join(",")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub policy: Policy,
    pub nice: libc::c_int,
    pub cpus: CpuSpec,
    pub threads: usize,
    pub min_period: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            policy: Policy::Idle,
            nice: DEFAULT_NICE,
            cpus: CpuSpec::parse(DEFAULT_CPUS_STR).unwrap(),
            threads: DEFAULT_THREADS,
            min_period: Duration::ZERO,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub threads_seen: usize,
    pub policy_ok: usize,
    pub affinity_ok: usize,
    pub active_cpus: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    pub fn from_lookup(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        let mut cfg = Self::default();
        if let Some(v) = get("CCNC_VISIOND_POLICY") {
            match Policy::parse(&v) {
                Some(p) => cfg.policy = p,
                None => eprintln!("ccnc_visiond: CCNC_VISIOND_POLICY={v:?} invalid, using {}", cfg.policy.name()),
            }
        }
        if let Some(v) = get("CCNC_VISIOND_NICE") {
            match v.trim().parse::<libc::c_int>() {
                Ok(n) => cfg.nice = n.clamp(-20, 19),
                Err(_) => eprintln!("ccnc_visiond: CCNC_VISIOND_NICE={v:?} invalid, using {}", cfg.nice),
            }
        }
        if let Some(v) = get("CCNC_VISIOND_CPUS") {
            match CpuSpec::parse(&v) {
                Some(spec) => cfg.cpus = spec,
                None => eprintln!("ccnc_visiond: CCNC_VISIOND_CPUS={v:?} invalid, using {DEFAULT_CPUS_STR}"),
            }
        }
        if let Some(v) = get("CCNC_VISIOND_THREADS") {
            match v.trim().parse::<usize>() {
                Ok(n) if n >= 1 => cfg.threads = n.clamp(1, MAX_THREADS),
                _ => eprintln!("ccnc_visiond: CCNC_VISIOND_THREADS={v:?} invalid, using {DEFAULT_THREADS}"),
            }
        }
        if let Some(v) = get("CCNC_VISIOND_MIN_PERIOD_MS") {
            match v.trim().parse::<u64>() {
                Ok(ms) => cfg.min_period = Duration::from_millis(ms),
                Err(_) => eprintln!("ccnc_visiond: CCNC_VISIOND_MIN_PERIOD_MS={v:?} invalid, using 0"),
            }
        }
        cfg
    }

    /// Apply CPU affinity and scheduling policy to one Linux thread (`tid == 0` means the calling thread).
    /// Returns `(affinity_ok, policy_ok)`. Refusal of an offline CPU mask is non-fatal (logs once via caller).
    pub fn apply_tid(&self, tid: libc::pid_t) -> (bool, bool) {
        let affinity_ok = match self.cpus {
            CpuSpec::Inherit => true,
            CpuSpec::Mask(mask) => set_affinity(tid, mask).is_ok(),
        };
        let policy_ok = set_policy(tid, self.policy, self.nice).is_ok();
        (affinity_ok, policy_ok)
    }

    /// Apply to the calling thread before spawning ORT / calibration worker threads so they inherit both.
    pub fn apply_current(&self) -> io::Result<bool> {
        let affinity_ok = match self.cpus {
            CpuSpec::Inherit => true,
            CpuSpec::Mask(mask) => match set_affinity(0, mask) {
                Ok(()) => true,
                Err(err) => {
                    eprintln!(
                        "ccnc_visiond: cannot pin CPUs {} ({err}), keeping {}",
                        self.cpus.format(),
                        read_cpus_allowed(0)
                    );
                    false
                }
            },
        };
        set_policy(0, self.policy, self.nice)?;
        Ok(affinity_ok)
    }

    /// Walk `/proc/self/task` and enforce affinity and scheduling policy on every thread in the process
    /// (including ORT's internal thread pool).
    pub fn apply_all_threads(&self) -> Applied {
        let mut threads_seen = 0;
        let mut policy_ok = 0;
        let mut affinity_ok = 0;
        if let Ok(entries) = fs::read_dir("/proc/self/task") {
            for entry in entries.flatten() {
                let Some(tid) = entry.file_name().to_str().and_then(|s| s.parse::<libc::pid_t>().ok()) else {
                    continue;
                };
                threads_seen += 1;
                let (aff, pol) = self.apply_tid(tid);
                affinity_ok += usize::from(aff);
                policy_ok += usize::from(pol);
            }
        }
        if threads_seen == 0 {
            threads_seen = 1;
            let (aff, pol) = self.apply_tid(0);
            affinity_ok = usize::from(aff);
            policy_ok = usize::from(pol);
        }
        Applied {
            threads_seen,
            policy_ok,
            affinity_ok,
            active_cpus: read_cpus_allowed(0),
        }
    }
}

fn set_affinity(tid: libc::pid_t, mask: u64) -> io::Result<()> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for c in 0..64 {
            if (mask & (1u64 << c)) != 0 {
                libc::CPU_SET(c, &mut set);
            }
        }
        if libc::sched_setaffinity(tid, std::mem::size_of_val(&set), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn set_policy(tid: libc::pid_t, policy: Policy, nice: libc::c_int) -> io::Result<()> {
    unsafe {
        let param = libc::sched_param { sched_priority: 0 };
        if libc::sched_setscheduler(tid, policy.raw(), &param) != 0 {
            return Err(io::Error::last_os_error());
        }
        if matches!(policy, Policy::Other | Policy::Batch) {
            // `tid == 0` targets the calling thread; on Linux PRIO_PROCESS takes a TID.
            if libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, nice) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

pub fn read_cpus_allowed(tid: libc::pid_t) -> String {
    let path = if tid == 0 {
        "/proc/self/status".to_string()
    } else {
        format!("/proc/self/task/{tid}/status")
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:").map(|v| v.trim().to_string()))
        })
        .unwrap_or_else(|| "?".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier};

    #[test]
    fn cpu_spec_parses_lists_ranges_and_rejects_bad() {
        assert_eq!(CpuSpec::parse("4"), Some(CpuSpec::Mask(1 << 4)));
        assert_eq!(CpuSpec::parse("0-2"), Some(CpuSpec::Mask(0b0111)));
        assert_eq!(CpuSpec::parse("0, 2, 5-6"), Some(CpuSpec::Mask(0b110_0101)));
        assert_eq!(CpuSpec::parse("all"), Some(CpuSpec::Inherit));
        assert_eq!(CpuSpec::parse("none"), Some(CpuSpec::Inherit));
        assert_eq!(CpuSpec::parse("0-2").unwrap().format(), "0-2");
        assert_eq!(CpuSpec::parse("0,2,5-6").unwrap().format(), "0,2,5-6");
        for bad in ["", "x", "3-1", "64", "0-", ","] {
            assert_eq!(CpuSpec::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn config_defaults_and_overrides() {
        let def = Config::from_lookup(|_| None);
        assert_eq!(def.policy, Policy::Idle);
        assert_eq!(def.nice, 19);
        assert_eq!(def.cpus, CpuSpec::Mask(0b0111));
        assert_eq!(def.threads, 2);
        assert_eq!(def.min_period, Duration::ZERO);

        let map = HashMap::from([
            ("CCNC_VISIOND_POLICY", "batch"),
            ("CCNC_VISIOND_NICE", "15"),
            ("CCNC_VISIOND_CPUS", "4"),
            ("CCNC_VISIOND_THREADS", "1"),
            ("CCNC_VISIOND_MIN_PERIOD_MS", "715"),
        ]);
        let custom = Config::from_lookup(|k| map.get(k).map(|v| (*v).to_string()));
        assert_eq!(custom.policy, Policy::Batch);
        assert_eq!(custom.nice, 15);
        assert_eq!(custom.cpus, CpuSpec::Mask(1 << 4));
        assert_eq!(custom.threads, 1);
        assert_eq!(custom.min_period, Duration::from_millis(715));

        let bad = HashMap::from([
            ("CCNC_VISIOND_POLICY", "fifo"),
            ("CCNC_VISIOND_CPUS", "99"),
            ("CCNC_VISIOND_THREADS", "0"),
            ("CCNC_VISIOND_MIN_PERIOD_MS", "-5"),
        ]);
        let fallback = Config::from_lookup(|k| bad.get(k).map(|v| (*v).to_string()));
        assert_eq!(fallback, def);
    }

    #[test]
    fn thread_inherits_idle_and_affinity_and_apply_tid_updates_worker() {
        // Run inside a dedicated child thread so the cargo test runner's thread keeps its original policy/affinity.
        let handle = std::thread::spawn(|| {
            // Find an online CPU from our own current affinity mask.
            let mut orig_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&orig_set), &mut orig_set) }, 0);
            let cpu = (0..64).find(|&c| unsafe { libc::CPU_ISSET(c, &orig_set) }).expect("at least one online cpu");
            let cfg = Config {
                policy: Policy::Idle,
                nice: 10,
                cpus: CpuSpec::Mask(1u64 << cpu),
                threads: 1,
                min_period: Duration::ZERO,
            };
            // Spawn a worker BEFORE configuring, then enforce on that worker's tid explicitly.
            let b_start = Arc::new(Barrier::new(2));
            let b_check = Arc::new(Barrier::new(2));
            let b_done = Arc::new(Barrier::new(2));
            let (tx, rx) = std::sync::mpsc::channel::<libc::pid_t>();
            let (bs, bc, bd) = (Arc::clone(&b_start), Arc::clone(&b_check), Arc::clone(&b_done));
            let pre_worker = std::thread::spawn(move || {
                let tid = unsafe { libc::gettid() };
                tx.send(tid).unwrap();
                bs.wait();
                bc.wait();
                let pol = unsafe { libc::sched_getscheduler(0) };
                let cpus = read_cpus_allowed(tid);
                bd.wait();
                (pol, cpus)
            });
            let worker_tid = rx.recv().unwrap();
            b_start.wait();

            assert!(cfg.apply_current().unwrap());
            let (aff_ok, pol_ok) = cfg.apply_tid(worker_tid);
            assert!(aff_ok && pol_ok, "apply_tid failed on worker");

            // Also verify a thread spawned AFTER apply_current inherits SCHED_IDLE and affinity automatically.
            let post_worker = std::thread::spawn(|| {
                let tid = unsafe { libc::gettid() };
                (unsafe { libc::sched_getscheduler(0) }, read_cpus_allowed(tid))
            });
            let (post_pol, post_cpus) = post_worker.join().unwrap();
            assert_eq!(post_pol, libc::SCHED_IDLE);
            assert_eq!(post_cpus, cpu.to_string());

            b_check.wait();
            b_done.wait();
            let (pre_pol, pre_cpus) = pre_worker.join().unwrap();
            assert_eq!(pre_pol, libc::SCHED_IDLE);
            assert_eq!(pre_cpus, cpu.to_string());
        });
        handle.join().unwrap();
    }
}
