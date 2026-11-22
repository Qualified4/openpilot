use std::io;

use ccnc_map::input::map_events::Event;
use stripnet::ipc::Sub;

pub const SERVICES: [&str; 5] = [
    "liveTracks",
    "livePose",
    "carState",
    "modelV2",
    "radarState",
];

pub struct Inputs {
    queues: [Sub; SERVICES.len()],
}

impl Inputs {
    pub fn open() -> io::Result<Self> {
        let queues = SERVICES.map(|name| {
            Sub::new(name, false).ok_or_else(|| io::Error::other(format!("open {name} subscriber")))
        });
        let [tracks, pose, car, model, radar] = queues;
        Ok(Self {
            queues: [tracks?, pose?, car?, model?, radar?],
        })
    }

    pub fn read(&self) -> Vec<(u64, Vec<u8>)> {
        // A liveTracks update drives the process. Wake periodically for the stale timer too.
        self.queues[0].poll(20);
        let mut inputs: Vec<_> = self
            .queues
            .iter()
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
