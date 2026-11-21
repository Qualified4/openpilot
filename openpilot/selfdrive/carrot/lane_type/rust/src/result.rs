//! The per-side result dict marking_daemon.filter_input builds and the strip filter / DisplayPolicy pass along.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extras {
    pub type_class: &'static str,
    pub color_class: &'static str,
    pub double: bool,
    pub blue: bool,
    pub p_unknown: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SideResult {
    pub kind: &'static str,
    pub color: &'static str,
    pub type_confidence: f64,
    pub color_confidence: f64,
    pub type_mono: i64,
    pub color_mono: i64,
    pub dashed_observed: bool,
    pub dashed_mono: i64,
    pub reason: &'static str,
    pub tracking_reset: bool,
    pub extras: Option<Extras>,
}

/// marking_daemon.filter_input.
pub fn filter_input(kind: &'static str, color: &'static str, type_conf: f64, color_conf: f64, stamp: i64, dashed_observed: bool,
                    reason: &'static str) -> SideResult {
    SideResult {
        kind,
        color,
        type_confidence: if kind == "dashed" || kind == "solid" { type_conf } else { 0. },
        color_confidence: if color == "white" || color == "yellow" { color_conf } else { 0. },
        type_mono: stamp,
        color_mono: stamp,
        dashed_observed,
        dashed_mono: stamp,
        reason,
        tracking_reset: false,
        extras: None,
    }
}

pub const RESET_REASONS: [&str; 5] = ["lane_change", "tracking_lost", "tracking_lost_or_stale", "invalid_lane_geometry", "not_detected"];

pub fn is_reset(r: &SideResult) -> bool {
    r.tracking_reset || RESET_REASONS.contains(&r.reason)
}
