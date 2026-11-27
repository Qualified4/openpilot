//! Camera vehicle detection rules (item 68), ported from det-ready-20261121/scripts/pipeline.py (+ geom.py, wcam.py).
//!
//! Pure computation: no I/O, no messaging, no dependencies. One `DetRules` holds every piece of state (rule E per
//! slot, the split gate per side, class confirmation per car). Feed it one camera frame at a time:
//!
//! * wide camera: detections (raw frame + the two side views) -> `WideGeometry::candidates` -> `DetRules::wide`
//!   -> per side the LF/RF (front) and LR/RR (rear) slot candidate, whether it is displayed, the confirm count, the
//!   fill/swap decision against the current radar slot and the reason.
//! * front camera: `front_box` flags a box cut by the bonnet crop; `DetRules::front_class` confirms the class of a car.
//!
//! Coordinates: calib frame x fwd, y right; candidate x is from the front bumper (camera -> bumper 1.52 m), yc is
//! the estimated lateral centre of the other car, right positive (radar yRel is left positive).
pub mod band;
mod band_top;
pub mod geom;
pub mod rules;
pub mod state;
pub mod wide;

pub use band::{band_stats, Mask};
pub use geom::{Calib, Ground, WideGeometry};
pub use rules::{front_box, swap_decision, Act, ClassConfirm, FrontBox, Reason, RuleE, SplitGate, SwapCfg, Timing, VehicleClass};
pub use state::{DetRules, SideOutput, SlotOutput, SlotState, StartGuard, WideFrame, WideOutput};
pub use wide::{Cand, Filters};

pub const VEH: [i32; 4] = [2, 3, 5, 7]; // COCO car, motorcycle, bus, truck
pub const SCORE_MIN: f64 = 0.3;
pub const CAM2B: f64 = 1.52; // camera -> front bumper (openpilot RADAR_TO_CAMERA)
pub const HALFW: f64 = 0.9; // half car width, inner corner -> centre
pub const LANE_IN: f64 = 1.9; // |yc| < 1.9 ego lane
pub const LANE_OUT: f64 = 5.6; // |yc| < 5.6 adjacent lane
pub const JITTER_S: f64 = 0.2; // frame timestamp jitter allowed on every time threshold

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Raw,   // full wide frame, 1344x760
    SideL, // virtual side view, yaw -55 deg, 512x384
    SideR, // virtual side view, yaw +55 deg
}
impl View {
    pub fn name(self) -> &'static str {
        match self {
            View::Raw => "raw",
            View::SideL => "vL",
            View::SideR => "vR",
        }
    }
    pub fn from_name(s: &str) -> Option<View> {
        match s {
            "raw" => Some(View::Raw),
            "vL" => Some(View::SideL),
            "vR" => Some(View::SideR),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Ego,
    Left,
    Right,
    Outside,
}
impl Lane {
    pub fn name(self) -> &'static str {
        match self {
            Lane::Ego => "ego",
            Lane::Left => "left",
            Lane::Right => "right",
            Lane::Outside => "outside",
        }
    }
}

/// One detector box of a wide view, in that view's pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WideDet {
    pub view: View,
    pub bbox: [f64; 4], // x1 y1 x2 y2
    pub score: f64,
    pub class: i32, // COCO id
}
impl WideDet {
    /// vehicle class with the minimum score: the only boxes the rules look at.
    pub fn is_vehicle(&self) -> bool {
        VEH.contains(&self.class) && self.score >= SCORE_MIN
    }
}

/// One radar point at the frame time. dRel from the front bumper, yRel left positive, vRel relative speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    pub id: i64,
    pub d_rel: f64,
    pub y_rel: f64,
    pub v_rel: f64,
}

/// Python round(x, 2): the nearest two-decimal number of the exact binary value, exact ties to even.
pub fn round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.2}").parse().unwrap_or(x)
}

impl WideGeometry {
    /// vehicle boxes -> candidates, in detection order (band drop and the cut-box exception applied).
    pub fn candidates(&self, dets: &[WideDet]) -> Vec<Cand> {
        self.candidates_with_pitch(dets, 0.0)
    }
    /// vehicle boxes -> candidates with dynamic pitch compensation.
    pub fn candidates_with_pitch(&self, dets: &[WideDet], delta_pitch: f64) -> Vec<Cand> {
        dets.iter()
            .filter(|d| d.is_vehicle())
            .filter_map(|d| wide::wide_candidate(d, &self.ground_with_pitch(d, delta_pitch), self.mask(d.view), true))
            .collect()
    }
    /// person boxes (COCO class 0) that do not overlap any vehicle box -> ground (x from bumper, y right, score).
    pub fn person_points(&self, dets: &[WideDet], delta_pitch: f64) -> Vec<(f64, f64, f64)> {
        let vehicles: Vec<&WideDet> = dets.iter().filter(|d| d.is_vehicle()).collect();
        dets.iter()
            .filter(|d| d.class == 0 && d.score >= 0.45)
            .filter(|d| !vehicles.iter().any(|v| v.view == d.view && box_overlap_ratio(d.bbox, v.bbox) >= 0.3))
            .filter_map(|d| {
                let (inside, touch) = band::band_stats(self.mask(d.view), &d.bbox);
                if inside >= 0.5 || touch {
                    return None;
                }
                let g = self.ground_with_pitch(d, delta_pitch).bc?;
                let x = g[0] - CAM2B;
                ((-5.0..=30.0).contains(&x) && g[1].abs() <= 10.0).then_some((x, g[1], d.score))
            })
            .collect()
    }
}

fn box_overlap_ratio(a: [f64; 4], b: [f64; 4]) -> f64 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = ((a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)).max(1.0);
    inter / area_a
}
