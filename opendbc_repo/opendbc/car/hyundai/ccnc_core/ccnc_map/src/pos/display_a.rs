//! Display A (DESIGN 6.6, 11): the estimated centre, no lane-centre lock, behind a distance-scaled backlash
//! (A_IMPROVE_KO.md). A vehicle keeps its value when it changes slot, because the value belongs to its vid.
//!
//! The backlash: the drawn value moves only once the estimate is more than `PLAY_PER_M * x` away, and then trails
//! it by that much, so a steady move passes at full speed and a back-and-forth smaller than twice the play does
//! not reach the cluster. The play grows with distance like the radar's lateral noise.

use std::collections::BTreeMap;

use super::common::{Common, FrameOut, Vid};

/// Backlash per metre of distance (0.1 m at 40 m).
pub const PLAY_PER_M: f64 = 0.0025;

#[derive(Default)]
pub struct DisplayA {
    shown: BTreeMap<Vid, f64>,
}

impl DisplayA {
    pub fn draw(&mut self, common: &Common, out: &FrameOut) -> [Option<f64>; 3] {
        self.shown.retain(|vid, _| common.tracks.contains_key(vid));
        for (vid, track) in &common.tracks {
            let y = track.y;
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
