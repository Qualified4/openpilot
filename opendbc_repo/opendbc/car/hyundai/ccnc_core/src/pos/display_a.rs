//! Display A (DESIGN 6.6, 11): the estimated centre, no lane-centre lock, behind a distance-scaled backlash
//! (A_IMPROVE_KO.md). A vehicle keeps its value when it changes slot, because the value belongs to its vid.
//!
//! The backlash scales with distance. The front slot has slightly more play to absorb residual heading and
//! radar noise. Each vehicle still keeps its displayed value across slot changes.
//!
//! The heading term (A_IMPROVE_KO.md, "A 앞섬"): the estimate is in the ego frame, so when ego turns towards a new
//! lane a car at x moves by -x * (road heading) before ego has moved sideways, and A drew lane changes about 0.5 s
//! early. A subtracts the low-passed road heading times distance before the backlash. The soft deadband removes
//! small heading changes from the correction while retaining larger changes during an ego lane change.

use std::collections::BTreeMap;

use super::common::{Common, FrameIn, FrameOut, Vid};

const PLAY_PER_M: f64 = 0.0025;
const FF_PLAY_PER_M: f64 = 0.00325;
const HEADING_TAU: f64 = 0.3;
const HEADING_DEADBAND: f64 = 0.006;

#[derive(Default)]
pub struct DisplayA {
    shown: BTreeMap<Vid, f64>,
    /// The last road heading the frames carried, and the low-passed heading with its time.
    last_heading: Option<f64>,
    heading: Option<(f64, f64)>,
}

impl DisplayA {
    pub fn draw(&mut self, common: &Common, out: &FrameOut, f: &FrameIn) -> [Option<f64>; 3] {
        self.shown.retain(|vid, _| common.tracks.contains_key(vid));
        let raw = f.heading.or(self.last_heading).unwrap_or(0.);
        self.last_heading = Some(raw);
        let (h, t0) = self.heading.unwrap_or((raw, f.now));
        let h = h + (raw - h) * ((f.now - t0).max(0.) / HEADING_TAU).min(1.);
        self.heading = Some((h, f.now));
        let head = (h.abs() - HEADING_DEADBAND).max(0.).copysign(h);
        for (vid, track) in &common.tracks {
            let y = track.y - track.x * head;
            let play_per_m = if out.assigned[0] == Some(*vid) { FF_PLAY_PER_M } else { PLAY_PER_M };
            let play = play_per_m * track.x.max(0.);
            let s = self.shown.entry(*vid).or_insert(y);
            if y - *s > play {
                *s = y - play;
            } else if *s - y > play {
                *s = y + play;
            }
        }
        out.assigned.map(|vid| vid.map(|vid| self.shown[&vid]))
    }
}
