//! Display A in the world map, with lateral positive to the left.

use crate::pos::display_a::backlash;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadingTerm {
    #[default]
    RoadFrame,
    Keep,
    Removed,
    PoseIntegrated,
}

pub(crate) fn draw_road_frame(shown: &mut Option<f64>, x: f64, width: f64, r: f64) -> f64 {
    let y = -width * r;
    let drawn = shown.get_or_insert(y);
    backlash(drawn, y, x.abs());
    *drawn
}
