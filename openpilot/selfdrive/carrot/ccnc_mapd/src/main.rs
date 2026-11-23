mod inputs;
mod output;
mod vision;

use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use ccnc_map::pipeline::Pipeline;
use ccnc_map::vision::Skipped;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn request_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn configure() -> io::Result<()> {
    let launcher_pid = std::env::var("MAPD_PARENT_PID")
        .ok()
        .and_then(|pid| pid.parse::<libc::pid_t>().ok());
    let tici = Path::new("/TICI").exists();
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

        let mut cpus: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(4, &mut cpus);
        if libc::sched_setaffinity(0, std::mem::size_of_val(&cpus), &cpus) != 0 {
            let err = io::Error::last_os_error();
            if tici {
                return Err(err);
            }
            eprintln!("ccnc_mapd: cannot pin CPU 4: {err}");
        }
        if tici {
            let sched = libc::sched_param {
                sched_priority: env!("MAPD_RT_PRIORITY").parse().unwrap(),
            };
            if libc::sched_setscheduler(0, libc::SCHED_FIFO, &sched) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) },
        0
    );
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    configure()?;
    if STOP.load(Ordering::Relaxed) {
        return Ok(());
    }
    let mut input = inputs::Inputs::open()?;
    let publisher = output::open()?;
    // Start each process with a different generation so a subscriber can see restarts.
    let mut pipeline = Pipeline::new((monotonic_ns() / 1_000) as u32);
    let mut geometry = vision::Builder::start()
        .map_err(|err| eprintln!("ccnc_mapd: no side view geometry thread, camera results are not used: {err}"))
        .ok();
    let mut dropped = None;
    // The last result of each side that came while the geometry was being built: used once it is there.
    let mut waiting: Vec<(u64, ccnc_map::vision::Frame)> = Vec::new();
    while !STOP.load(Ordering::Relaxed) {
        if let Some(built) = geometry.as_ref().and_then(vision::Builder::take) {
            pipeline.set_vision_geometry(built);
            for (stamp, frame) in waiting.drain(..) {
                let _ = pipeline.vision(stamp, &frame);
            }
        }
        for (_, bytes) in input.read() {
            if let Some(message) = vision::message(&bytes) {
                pipeline.kinds(message.stamp, message.front_frame, &message.kinds, &message.calibration);
                let Some(frame) = message.frame else { continue };
                let stamp = message.stamp;
                match pipeline.vision(stamp, &frame) {
                    Err(Skipped::NeedsGeometry(calib)) => {
                        if let Some(builder) = geometry.as_mut() {
                            builder.request(calib);
                            waiting.retain(|(_, other)| other.view != frame.view);
                            waiting.push((stamp, frame));
                        }
                    }
                    Err(why) => {
                        if dropped != Some(why) {
                            eprintln!("ccnc_mapd: camera result dropped ({why:?}), frame {} ms before {}", stamp.saturating_sub(frame.time_ns) / 1_000_000, stamp);
                        }
                        dropped = Some(why);
                        continue;
                    }
                    Ok(_) => {}
                }
                dropped = None;
                continue;
            }
            if let Some(map) = pipeline
                .ingest(&bytes)
                .map_err(|err| io::Error::other(format!("map update: {err:?}")))?
            {
                output::publish(&publisher, map, monotonic_ns())?;
            }
        }
        let now = monotonic_ns();
        if let Some(map) = pipeline.expire(now) {
            output::publish(&publisher, map, now)?;
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ccnc_mapd: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};
    use std::{fs, process::Command};

    /// A private msgq directory for one test. OPENPILOT_PREFIX is process wide, so the tests take turns.
    pub struct QueuePrefix {
        pub path: PathBuf,
        _turn: MutexGuard<'static, ()>,
    }

    pub fn queue_prefix(name: &str) -> QueuePrefix {
        static TURN: Mutex<()> = Mutex::new(());
        let turn = TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let prefix = format!("{name}_{}_{}", std::process::id(), monotonic_ns());
        let path = PathBuf::from(format!("/dev/shm/msgq_{prefix}"));
        fs::create_dir(&path).unwrap();
        std::env::set_var("OPENPILOT_PREFIX", prefix);
        QueuePrefix { path, _turn: turn }
    }

    impl Drop for QueuePrefix {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn runs_below_plannerd_and_radard() {
        let source = fs::read_to_string("../../../common/realtime.py").unwrap();
        let ctrl_low: i32 = source
            .lines()
            .find_map(|line| line.trim().strip_prefix("CTRL_LOW ="))
            .and_then(|value| value.split_whitespace().next()?.parse().ok())
            .unwrap();
        assert_eq!(env!("MAPD_RT_PRIORITY").parse::<i32>().unwrap(), ctrl_low - 1);
    }

    #[test]
    fn opens_every_queue_with_cereal_size() {
        const HEADER_BYTES: u64 = 8 * (3 + 3 * 40); // msgq_header_t in msgq_repo/msgq/msgq.h
        let prefix = queue_prefix("mapd_queues");
        let path = &prefix.path;
        let _inputs = inputs::Inputs::open().unwrap();
        let _output = output::open().unwrap();
        let services = inputs::SERVICES.into_iter().chain([output::SERVICE]);
        let sizes = Command::new("python3")
            .arg("-c")
            .arg("import runpy,sys; s=runpy.run_path('../../../cereal/services.py')['SERVICE_LIST']; print(*(int(s[n].queue_size) for n in sys.argv[1:]))")
            .args(services.clone())
            .output().unwrap();
        assert!(
            sizes.status.success(),
            "{}",
            String::from_utf8_lossy(&sizes.stderr)
        );
        let sizes = String::from_utf8(sizes.stdout).unwrap();
        let expected: Vec<u64> = sizes
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(expected.len(), 7);
        for (name, queue_size) in services.zip(expected) {
            assert_eq!(
                fs::metadata(path.join(name)).unwrap().len(),
                HEADER_BYTES + queue_size,
                "{name}"
            );
        }
    }
}
