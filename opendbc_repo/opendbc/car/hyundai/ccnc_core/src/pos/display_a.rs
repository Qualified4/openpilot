//! Display A (DESIGN 6.6, 11): the estimated centre as it is, no lane-centre lock. A vehicle keeps its value when
//! it changes slot, because the estimate belongs to its vid.

use super::common::{Common, FrameOut};

pub fn draw(common: &Common, out: &FrameOut) -> [Option<f64>; 3] {
    out.assigned.map(|vid| vid.map(|vid| common.tracks[&vid].y))
}
