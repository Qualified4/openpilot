//! Display A (DESIGN 6.6, 11): the estimated centre, no lane-centre lock, behind a distance-scaled backlash
//! (A_IMPROVE_KO.md). A vehicle keeps its value when it changes slot, because the value belongs to its vid.
//!
//! The backlash: the drawn value moves only once the estimate is more than `PLAY_PER_M * x` away, and then trails
//! it by that much, so a steady move passes at full speed and a back-and-forth smaller than twice the play does
//! not reach the cluster. The play grows with distance like the radar's lateral noise.
//!
//! The heading term (A_IMPROVE_KO.md, "A 앞섬"): the estimate is in the ego frame, so when ego turns towards a new
//! lane a car at x moves by -x * (road heading) before ego has moved sideways, and A drew lane changes about 0.5 s
//! early. The drawn value adds back HEADING_GAIN * x times the road heading, low-passed over HEADING_TAU, before
//! the backlash.

use std::collections::BTreeMap;

use super::common::{Common, FrameIn, FrameOut, Vid};

/// Backlash per metre of distance (0.1 m at 40 m).
pub const PLAY_PER_M: f64 = 0.0025;
/// Share of the road heading taken out, and its low-pass time constant (s).
pub const HEADING_GAIN: f64 = 0.8;
pub const HEADING_TAU: f64 = 0.3;

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
        let head = HEADING_GAIN * h;
        for (vid, track) in &common.tracks {
            let y = track.y - track.x * head;
            let play = PLAY_PER_M * track.x.max(0.);
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
