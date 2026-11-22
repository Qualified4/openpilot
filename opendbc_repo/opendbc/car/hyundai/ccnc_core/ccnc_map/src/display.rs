//! Display A in the world map, with lateral positive to the left.

use crate::pos::{common::LANE_WIDTH, display_a::backlash};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadingTerm {
    #[default]
    RoadFrame,
    Keep,
    Removed,
    PoseIntegrated,
}

pub(crate) fn draw_road_frame(shown: &mut Option<f64>, x: f64, r: f64) -> f64 {
    // A lane of ego movement must have the same display scale for every vid.
    // In particular, a change in the fitted width at x is not vehicle motion.
    let y = -LANE_WIDTH * r;
    let drawn = shown.get_or_insert(y);
    backlash(drawn, y, x.abs());
    *drawn
}
