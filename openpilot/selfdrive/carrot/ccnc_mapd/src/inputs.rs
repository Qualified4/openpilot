use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ccnc_map::input::map_events::Event;
use stripnet::ipc::Sub;

pub const SERVICES: [&str; 6] = [
    "liveTracks",
    "livePose",
    "carState",
    "modelV2",
    "radarState",
    // CcncVision. Present only while ccnc_visiond runs (CcncVisionProcess=1); without it nothing changes.
    "customReserved4",
];

/// Where msgq_header_t keeps `write_uid` (msgq_repo/msgq/msgq.h). A publisher writes a new one when it starts.
const WRITE_UID_OFFSET: u64 = 16;

/// The queues this process only reads when liveTracks arrives.
struct Context(Vec<Sub>);

// A msgq subscriber is shared memory and heap state without thread affinity: any thread may read it, and the
// thread that first reads it after its publisher starts gets the publisher's wake signal. The Mutex lets one
// thread at a time use it.
unsafe impl Send for Context {}

impl Context {
    fn open(services: &[&str]) -> io::Result<Self> {
        services.iter().map(|name| open_sub(name)).collect()
    }
}

impl FromIterator<Sub> for Context {
    fn from_iter<I: IntoIterator<Item = Sub>>(subs: I) -> Self {
        Context(subs.into_iter().collect())
    }
}

type Shared = Arc<Mutex<Context>>;

fn lock(context: &Shared) -> MutexGuard<'_, Context> {
    context.lock().unwrap_or_else(PoisonError::into_inner)
}

fn open_sub(name: &str) -> io::Result<Sub> {
    Sub::new(name, false).ok_or_else(|| io::Error::other(format!("open {name} subscriber")))
}

fn block_wake_signal() -> io::Result<()> {
    unsafe {
        let mut signals: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut signals);
        libc::sigaddset(&mut signals, libc::SIGUSR2);
        match libc::pthread_sigmask(libc::SIG_BLOCK, &signals, std::ptr::null_mut()) {
            0 => Ok(()),
            err => Err(io::Error::from_raw_os_error(err)),
        }
    }
}

fn queue_file(service: &str) -> io::Result<File> {
    // The path of msgq_new_queue.
    let prefix =
        std::env::var("OPENPILOT_PREFIX").map_or_else(|_| String::new(), |prefix| prefix + "/");
    File::open(format!("/dev/shm/msgq_{prefix}{service}"))
}

/// The thread that blocks the wake signal and takes it for the context queues, and what the main thread needs
/// to send it to each new publisher.
struct Registrar {
    headers: Vec<File>,
    publishers: Vec<Option<u64>>,
    request: Sender<()>,
    done: Receiver<()>,
}

impl Registrar {
    fn start(services: &[&'static str]) -> io::Result<(Shared, Self)> {
        let names = services.to_vec();
        let (request, requests) = mpsc::channel();
        let (reply, done) = mpsc::channel();
        let (report, opened) = mpsc::channel();
        std::thread::Builder::new()
            .name("mapd-context".into())
            .stack_size(256 * 1024)
            .spawn(move || {
                let context = block_wake_signal()
                    .and_then(|()| Context::open(&names))
                    .map(|context| Arc::new(Mutex::new(context)));
                let shared = context.as_ref().ok().cloned();
                let _ = report.send(context);
                if let Some(context) = shared {
                    for () in requests {
                        // Looking at a queue registers this thread if its publisher has no reader for it yet.
                        for queue in &lock(&context).0 {
                            queue.poll(0);
                        }
                        if reply.send(()).is_err() {
                            break;
                        }
                    }
                }
                // msgq signals the thread registered for a queue until the queue's publisher restarts. Staying
                // alive keeps another thread, possibly of another process, from reusing this thread's id and
                // getting those signals.
                loop {
                    std::thread::park();
                }
            })?;
        let context = opened
            .recv()
            .map_err(|_| io::Error::other("context thread exited"))??;
        let registrar = Registrar {
            headers: services
                .iter()
                .map(|name| queue_file(name))
                .collect::<io::Result<_>>()?,
            publishers: vec![None; services.len()],
            request,
            done,
        };
        Ok((context, registrar))
    }

    /// Have the context thread look first at every context queue whose publisher started since the last call.
    fn register_new_publishers(&mut self) -> io::Result<()> {
        let mut started = false;
        for (header, seen) in self.headers.iter().zip(&mut self.publishers) {
            let mut uid = [0; 8];
            header.read_exact_at(&mut uid, WRITE_UID_OFFSET)?;
            let uid = u64::from_ne_bytes(uid);
            started |= seen.replace(uid) != Some(uid);
        }
        if started {
            // The main thread waits here, so the two threads never use the context queues at the same time.
            self.request
                .send(())
                .ok()
                .and_then(|()| self.done.recv().ok())
                .ok_or_else(|| io::Error::other("context thread exited"))?;
        }
        Ok(())
    }
}

pub struct Inputs {
    tracks: Sub,
    context: Shared,
    registrar: Option<Registrar>,
}

impl Inputs {
    /// A publisher wakes a subscriber with a signal to the thread that first read the queue after the publisher
    /// started, and every signal ends that thread's sleep. On the main thread the context queues ended 160 sleeps a
    /// second (without CcncVision) for messages that wait for the next liveTracks anyway. They are opened, and registered with
    /// each new publisher, by a thread that blocks the signal. The main thread reads them and takes the signal
    /// for liveTracks alone.
    ///
    /// When that thread cannot do its part, the main thread reads and takes the signal for every queue, as
    /// before the context thread.
    pub fn open() -> io::Result<Self> {
        let [tracks, context @ ..] = SERVICES;
        let tracks = open_sub(tracks)?;
        Self::with_registrar(tracks, &context, Registrar::start(&context))
    }

    fn with_registrar(
        tracks: Sub,
        services: &[&str],
        started: io::Result<(Shared, Registrar)>,
    ) -> io::Result<Self> {
        let (context, registrar) = match started {
            Ok((context, registrar)) => (context, Some(registrar)),
            Err(err) => {
                eprintln!("ccnc_mapd: every queue wakes the main thread: {err}");
                (Arc::new(Mutex::new(Context::open(services)?)), None)
            }
        };
        Ok(Self {
            tracks,
            context,
            registrar,
        })
    }

    pub fn read(&mut self) -> Vec<(u64, Vec<u8>)> {
        // A liveTracks update drives the process. Wake periodically for the stale timer too.
        self.tracks.poll(20);
        if let Some(Err(err)) = self
            .registrar
            .as_mut()
            .map(Registrar::register_new_publishers)
        {
            // From the next publisher start on, the main thread is the one registered.
            eprintln!("ccnc_mapd: every queue wakes the main thread from now on: {err}");
            self.registrar = None;
        }
        let context = lock(&self.context);
        let mut inputs: Vec<_> = std::iter::once(&self.tracks)
            .chain(&context.0)
            .flat_map(Sub::drain)
            .filter_map(|bytes| {
                let stamp = Event::read(&bytes)?.stamp;
                Some((stamp, bytes))
            })
            .collect();
        inputs.sort_by_key(|(stamp, _)| *stamp);
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccnc_map::input::map_events::CAR_STATE;
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};
    use stripnet::ipc::Pub;

    const CAR: usize = 1; // carState in the context queues

    fn sleeps_of_this_thread() -> u64 {
        let status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let line = status
            .lines()
            .find_map(|line| line.strip_prefix("voluntary_ctxt_switches:"));
        line.unwrap().trim().parse().unwrap()
    }

    /// The smallest Event the map reads: one segment, a root struct with logMonoTime and the union tag.
    fn car_state(stamp: u64) -> Vec<u8> {
        let words: [u64; 3] = [2 << 32, stamp, u64::from(CAR_STATE)];
        let mut bytes = vec![0, 0, 0, 0, 3, 0, 0, 0];
        bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
        bytes
    }

    /// A publisher that starts after the subscriber, as card and radarcan do after ccnc_mapd. Each list sent to
    /// it publishes those stamps as carState messages (or 8-byte messages for other queues), 1 ms apart.
    fn late_publisher(service: &'static str) -> Sender<Vec<u64>> {
        let (publish, batches) = channel::<Vec<u64>>();
        let (started, ready) = channel();
        std::thread::spawn(move || {
            let publisher = Pub::new(service).unwrap();
            started.send(()).unwrap();
            for stamps in batches {
                for stamp in stamps {
                    std::thread::sleep(Duration::from_millis(1));
                    let bytes = if service == "carState" {
                        car_state(stamp)
                    } else {
                        vec![stamp as u8; 8]
                    };
                    assert!(publisher.send(&bytes));
                }
            }
        });
        ready.recv().unwrap();
        publish
    }

    fn read_header(service: &str, offset: u64) -> u64 {
        let mut word = [0; 8];
        queue_file(service)
            .unwrap()
            .read_exact_at(&mut word, offset)
            .unwrap();
        u64::from_ne_bytes(word)
    }

    /// carState stamps read in `read` calls until `count` arrive or a second passes.
    fn read_car_states(inputs: &mut Inputs, count: usize) -> Vec<u64> {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut stamps = Vec::new();
        while stamps.len() < count && Instant::now() < deadline {
            stamps.extend(inputs.read().into_iter().map(|(stamp, _)| stamp));
        }
        stamps
    }

    #[test]
    fn only_live_tracks_ends_the_wait() {
        let _prefix = crate::tests::queue_prefix("mapd_wakeups");
        let mut inputs = Inputs::open().unwrap();
        let live_tracks = late_publisher("liveTracks");
        // The second carState publisher is a card that restarted.
        for _ in 0..2 {
            let car_state = late_publisher("carState");
            assert!(inputs.read().is_empty());

            let before = sleeps_of_this_thread();
            car_state.send((0..100).collect()).unwrap();
            assert!(
                !inputs.tracks.poll(300),
                "a carState message ended the liveTracks wait"
            );
            // One sleep for the whole wait. When this thread took the signal, each message ended a sleep.
            let sleeps = sleeps_of_this_thread() - before;
            assert!(
                sleeps < 10,
                "{sleeps} sleeps while 100 carState messages arrived"
            );
            assert_eq!(lock(&inputs.context).0[CAR].drain().len(), 100);
        }
        assert!(inputs.registrar.is_some());

        live_tracks.send(vec![0]).unwrap();
        assert!(inputs.tracks.poll(1000));
        assert_eq!(inputs.tracks.drain(), vec![vec![0; 8]]);
    }

    /// The process reads `write_uid` and nothing else of msgq_header_t. A publisher start writes msgq_get_uid():
    /// random bits above the publishing thread's id. The reader count before it moves when a subscriber registers.
    #[test]
    fn write_uid_offset_matches_msgq_header() {
        let _prefix = crate::tests::queue_prefix("mapd_header");
        let subscriber = open_sub("carState").unwrap();
        for _ in 0..2 {
            let (sent, publisher_thread) = channel();
            let (stop, stopped) = channel::<()>();
            let publisher = std::thread::spawn(move || {
                let publisher = Pub::new("carState").unwrap();
                sent.send(unsafe { libc::syscall(libc::SYS_gettid) } as u64)
                    .unwrap();
                let _ = stopped.recv();
                drop(publisher);
            });
            let tid = publisher_thread.recv().unwrap();
            let uid = read_header("carState", WRITE_UID_OFFSET);
            assert_eq!(uid & 0xFFFF_FFFF, tid, "write_uid {uid:#x}");
            assert_eq!(
                read_header("carState", 0),
                0,
                "num_readers after the publisher start"
            );
            subscriber.poll(0);
            assert_eq!(
                read_header("carState", 0),
                1,
                "num_readers after one subscriber registered"
            );
            stop.send(()).unwrap();
            publisher.join().unwrap();
        }
    }

    /// A restarted card: nothing is lost once the process has looked at the queue, the messages come in order, and
    /// the process misses no message that a main-thread subscriber, as before the context thread, gets.
    #[test]
    fn keeps_every_message_across_a_publisher_restart() {
        let _prefix = crate::tests::queue_prefix("mapd_restart");
        let mut inputs = Inputs::open().unwrap();
        let first = late_publisher("carState");
        inputs.read();
        first.send((1..=50).collect()).unwrap();
        assert_eq!(
            read_car_states(&mut inputs, 50),
            (1..=50).collect::<Vec<_>>()
        );
        drop(first);

        // The new publisher sends from its first moment, before the process has looked at the queue again.
        let reference = open_sub("carState").unwrap();
        let second = late_publisher("carState");
        second.send((101..=200).collect()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let (mut stamps, mut previous) = (Vec::new(), Vec::new());
        while stamps.last() != Some(&200) && Instant::now() < deadline {
            stamps.extend(inputs.read().into_iter().map(|(stamp, _)| stamp));
            previous.extend(
                reference
                    .drain()
                    .iter()
                    .map(|bytes| Event::read(bytes).unwrap().stamp),
            );
        }
        let start = *stamps.first().unwrap();
        assert_eq!(stamps, (start..=200).collect::<Vec<_>>());
        // msgq drops what a publisher sends before a subscriber's first look after its start (msgq.cc
        // msgq_init_subscriber). That is one read period here; card publishes carState seconds after it starts.
        assert!(start <= 101 + 40, "lost {} messages", start - 101);
        assert!(previous.iter().all(|stamp| stamps.contains(stamp)));
        assert!(inputs.registrar.is_some());
    }

    /// manager's full restart removes the queue files and the new processes create them again.
    #[test]
    fn survives_queue_files_made_again() {
        let prefix = crate::tests::queue_prefix("mapd_remade");
        let mut inputs = Inputs::open().unwrap();
        let car_state = late_publisher("carState");
        inputs.read();
        car_state.send(vec![1]).unwrap();
        assert_eq!(read_car_states(&mut inputs, 1), vec![1]);
        for entry in std::fs::read_dir(&prefix.path).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        let _publishers: Vec<_> = SERVICES.iter().map(|name| late_publisher(name)).collect();
        for _ in 0..3 {
            assert!(inputs.read().is_empty());
        }
        assert!(inputs.registrar.is_some());
    }

    #[test]
    fn reads_every_queue_on_the_main_thread_when_the_context_thread_fails() {
        let _prefix = crate::tests::queue_prefix("mapd_fallback");
        let broken = |inputs: &mut Inputs, why| {
            let registrar = inputs.registrar.as_mut().unwrap();
            match why {
                "header" => registrar.headers[CAR] = File::open("/dev/null").unwrap(),
                _ => registrar.done = channel().1,
            }
        };
        for why in ["header", "thread", "start"] {
            let mut inputs = if why == "start" {
                let services = &SERVICES[1..];
                let failed = Err(io::Error::other("context thread did not start"));
                Inputs::with_registrar(open_sub(SERVICES[0]).unwrap(), services, failed).unwrap()
            } else {
                let mut inputs = Inputs::open().unwrap();
                broken(&mut inputs, why);
                inputs
            };
            // A publisher start makes the process look at the context queues.
            let first = late_publisher("carState");
            inputs.read();
            assert!(inputs.registrar.is_none(), "{why}");
            drop(first);
            // The main thread takes the signal of the next publisher, as before the context thread.
            let car_state = late_publisher("carState");
            inputs.read();
            car_state.send((1..=20).collect()).unwrap();
            assert_eq!(
                read_car_states(&mut inputs, 20),
                (1..=20).collect::<Vec<_>>(),
                "{why}"
            );
            let before = sleeps_of_this_thread();
            car_state.send((21..=40).collect()).unwrap();
            inputs.tracks.poll(300);
            assert!(sleeps_of_this_thread() - before >= 10, "{why}");
            assert_eq!(
                read_car_states(&mut inputs, 20),
                (21..=40).collect::<Vec<_>>(),
                "{why}"
            );
        }
    }
}
