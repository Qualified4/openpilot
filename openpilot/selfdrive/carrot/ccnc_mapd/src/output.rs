use std::io;

use ccnc_map::{cmap, pipeline::STALE_NS};
use stripnet::ipc::Pub;

pub const SERVICE: &str = "customReserved3";

pub fn open() -> io::Result<Pub> {
    Pub::new(SERVICE).ok_or_else(|| io::Error::other(format!("open {SERVICE} publisher")))
}

pub fn publish(publisher: &Pub, mut map: cmap::MapOut, now: u64) -> io::Result<()> {
    if !now
        .checked_sub(map.radar_log_mono_time)
        .is_some_and(|age| age <= STALE_NS)
    {
        map.slots = [None; 5];
        map.valid = false;
    }
    if !publisher.send(&cmap::encode_at(&map, now)) {
        return Err(io::Error::other(format!("send {SERVICE}")));
    }
    Ok(())
}
