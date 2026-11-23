//! CcncVision messages for the map, and the side view geometry built off the map's thread.

use std::sync::mpsc::{self, Receiver, Sender};

use ccnc_map::vision::{Box, Calib, Frame, Geometry, Side};
use ccnc_vision_msg::{Status, View};

/// A frame result of a CcncVision Event, with the Event's logMonoTime. The frame's time is `frameMonoTime`, on
/// liveTracks' clock; `timestampSof` (the camera's CLOCK_BOOTTIME) is not used. Status messages give None.
pub fn frame(bytes: &[u8]) -> Option<(u64, Frame)> {
    let (stamp, out) = ccnc_vision_msg::decode(bytes)?;
    let side = match (out.status, out.view) {
        (Status::Frame, View::SideL) => Side::Left,
        (Status::Frame, View::SideR) => Side::Right,
        _ => return None,
    };
    let c = out.calibration;
    let [a, b, d] = c.rpy_calib;
    let [e, f, g] = c.wide_from_device_euler;
    let boxes = out
        .boxes
        .iter()
        .map(|b| Box { xyxy: b.xyxy.map(f64::from), confidence: f64::from(b.confidence), class_id: b.class_id })
        .collect();
    Some((stamp, Frame { time_ns: out.frame_mono_time, side, calibration: [a, b, d, e, f, g, c.height], boxes }))
}

/// Builds `Geometry` (about 50 ms on M8) on its own thread, as a normal thread at nice 19: the map's SCHED_FIFO
/// thread and the real-time processes on its core never wait for it. Not SCHED_IDLE, which can wait without end
/// on a busy core.
pub struct Builder {
    requests: Sender<Calib>,
    built: Receiver<(Geometry, std::time::Duration)>,
    asked: Option<Calib>,
}

impl Builder {
    pub fn start() -> std::io::Result<Self> {
        let (requests, wanted) = mpsc::channel::<Calib>();
        let (done, built) = mpsc::channel();
        std::thread::Builder::new().name("mapd-vision".into()).spawn(move || {
            // The main thread may already be SCHED_FIFO; this thread leaves it and takes nice 19.
            let normal = libc::sched_param { sched_priority: 0 };
            let tid = unsafe { libc::gettid() } as libc::id_t;
            if unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &normal) } != 0
                || unsafe { libc::setpriority(libc::PRIO_PROCESS, tid, 19) } != 0
            {
                eprintln!("ccnc_mapd: side view geometry thread keeps its priority: {}", std::io::Error::last_os_error());
            }
            for calib in wanted {
                let start = std::time::Instant::now();
                let geometry = Geometry::build(&calib);
                if done.send((geometry, start.elapsed())).is_err() {
                    break;
                }
            }
        })?;
        Ok(Self { requests, built, asked: None })
    }

    /// Ask once per calibration; frames until it is built are not used.
    pub fn request(&mut self, calib: Calib) {
        if self.asked != Some(calib) && self.requests.send(calib).is_ok() {
            self.asked = Some(calib);
        }
    }

    pub fn take(&self) -> Option<Geometry> {
        let (geometry, took) = self.built.try_iter().last()?;
        eprintln!("ccnc_mapd: side view geometry built in {:.1} ms", took.as_secs_f64() * 1e3);
        Some(geometry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccnc_map::vision::{Context, Skipped, Vision};
    use ccnc_map::world::PoseAt;
    use ccnc_vision_msg::{Calibration, VisionOut};

    fn message(status: Status, view: View, sof: u64, frame_mono_time: u64) -> VisionOut {
        VisionOut {
            status,
            frame_id: 9,
            timestamp_sof: sof,
            frame_mono_time,
            view,
            processing_ms: 40.,
            night_corrected: false,
            calibration: Calibration { rpy_calib: [0., -0.0106, 0.004], wide_from_device_euler: [0., -0.0098, -0.018], height: 1.22 },
            boxes: Vec::new(),
        }
    }

    #[test]
    fn status_messages_are_not_frames() {
        let resting = ccnc_vision_msg::encode(&message(Status::Resting, View::SideL, 0, 0), 1);
        assert!(frame(&resting).is_none());
        let other = ccnc_vision_msg::encode(&message(Status::Frame, View::Full, 5, 5), 1);
        assert!(frame(&other).is_none());
    }

    #[test]
    fn camera_clock_apart_from_the_map_clock() {
        // The camera's CLOCK_BOOTTIME runs an hour ahead (a suspended device). Only frameMonoTime counts: read on
        // the camera clock, every result would be an hour old or in the future.
        let hour = 3_600_000_000_000;
        let mut vision = Vision::default();
        let context = Context { v_ego: 0., tracks: &[], lprob: None, slots: Default::default(), guard: true };
        let mut decide = |frame_mono: u64, published: u64| {
            let bytes = ccnc_vision_msg::encode(&message(Status::Frame, View::SideL, frame_mono + hour, frame_mono), published);
            let (stamp, frame) = frame(&bytes).unwrap();
            assert_eq!((stamp, frame.time_ns), (published, frame_mono));
            let arrival = stamp as f64 * 1e-9;
            match vision.frame(&frame, arrival, &context, PoseAt::default()) {
                Err(Skipped::NeedsGeometry(calib)) => {
                    vision.set_geometry(Geometry::build(&calib));
                    vision.frame(&frame, arrival, &context, PoseAt::default()).map(|_| ())
                }
                other => other.map(|_| ()),
            }
        };
        assert_eq!(decide(2_000_000_000, 2_400_000_000), Ok(()));
        assert_eq!(decide(2_000_000_000, 3_499_000_000), Ok(()), "1.5 s late is still used");
        assert_eq!(decide(2_000_000_000, 3_500_000_100), Err(Skipped::Late));
        assert_eq!(decide(2_000_000_000, 1_999_000_000), Err(Skipped::Late), "published before its frame");
    }
}
