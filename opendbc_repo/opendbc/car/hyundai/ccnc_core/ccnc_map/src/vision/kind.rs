//! Vehicle kinds (car, truck, bus) from `ccnc_visiond`'s classifier, put on the radar vehicles (item 68).
//!
//! The detector decides whether a car is drawn, the classifier only which shape: a kind never adds, moves or
//! removes a slot. `ccnc_visiond` sends each answer with the box and frame it is for, one or two messages after the
//! frame. Here a front box is tied to the radar track it covers at the frame's time (the det-bench `xmatch` gate),
//! the track to its vehicle (vid), and the vid's answers are voted on:
//!
//! * only answers with a probability of `MIN_PROB` or more count;
//! * two answers of the same kind at least `GAP_S` apart confirm it (det-light 6: no "different view" condition,
//!   a standing car never gives one);
//! * a confirmed kind changes after `N_FLIP` agreeing answers of another kind;
//! * the vote goes when its vid goes.
//!
//! Night frames are not classified (`ccnc_visiond`), as R1 had it: there is no labelled night data.
//! Side view answers take the same path but are not used (`SIDE_KINDS`): nothing has measured how right they are.

use std::collections::HashMap;

use super::Side;
use crate::vehicles::{Kind, Vehicle};
use det_rules::{Cand, Track, CAM2B};

/// `Kind::class` and the CcncVision `Kind.kind` values.
pub const KIND_CAR: u8 = 1;
pub const KIND_TRUCK: u8 = 2;
pub const KIND_BUS: u8 = 3;

pub const MIN_PROB: f64 = 0.6;
pub const GAP_S: f64 = 0.8;
pub const N_CONFIRM: u32 = 2;
pub const N_FLIP: u32 = 4;
/// Side view answers are tied to vids only when this is on. To turn it on: score the classifier on hand labelled
/// side view boxes (the front labels say nothing about the side views' angle and distortion), check the confirmed
/// kinds of the 58 scenes and the truck/bus scenes 30, 31, 44, then turn on `ccnc_visiond`'s `SIDE_KINDS` too.
pub const SIDE_KINDS: bool = false;

/// Road camera of the 1344x760 device (det-cost `geom.py`).
const FOCAL: f64 = 1141.5;
const CENTER: (f64, f64) = (672.0, 380.0);
/// Rows below this are the bonnet: a box ending there is cut (det_rules `FRONT_CROP_Y` minus 2 px).
const FRONT_BOTTOM: f64 = 618.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindView {
    Front,
    Side(Side),
}

/// One classifier answer, from a CcncVision `Kind`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindResult {
    /// The box's frame, ns on CLOCK_MONOTONIC.
    pub time_ns: u64,
    pub view: KindView,
    pub xyxy: [f64; 4],
    pub kind: u8,
    pub prob: f64,
}

/// The road camera for points in the calibrated frame (x forward, y right, z down from the camera).
pub struct FrontCam {
    m: [[f64; 3]; 3],
    inv: [[f64; 3]; 3],
    height: f64,
}

fn rot_from_euler(r: f64, p: f64, y: f64) -> [[f64; 3]; 3] {
    let (sr, cr) = r.sin_cos();
    let (sp, cp) = p.sin_cos();
    let (sy, cy) = y.sin_cos();
    // Rz @ Ry @ Rx, openpilot orientation.rot_from_euler.
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| m[i][k] * v[k]).sum())
}

fn inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |i: usize, j: usize| {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (x, y) = ((j + 1) % 3, (j + 2) % 3);
        m[a][x] * m[b][y] - m[a][y] * m[b][x]
    };
    let det: f64 = (0..3).map(|j| m[0][j] * c(0, j)).sum();
    (det.abs() > 1e-12).then(|| std::array::from_fn(|i| std::array::from_fn(|j| c(j, i) / det)))
}

impl FrontCam {
    /// `rpy`: liveCalibration rpyCalib, `height`: its height. None for values that are not finite.
    pub fn new(rpy: [f64; 3], height: f64) -> Option<Self> {
        if !rpy.iter().chain([&height]).all(|v| v.is_finite()) {
            return None;
        }
        let k = [[FOCAL, 0., CENTER.0], [0., FOCAL, CENTER.1], [0., 0., 1.]];
        let view_from_device = [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]];
        let m = mul(&mul(&k, &view_from_device), &rot_from_euler(rpy[0], rpy[1], rpy[2]));
        Some(Self { inv: inverse(&m)?, m, height })
    }

    fn project(&self, p: [f64; 3]) -> Option<(f64, f64)> {
        let q = apply(&self.m, p);
        (q[2] > 0.1).then(|| (q[0] / q[2], q[1] / q[2]))
    }

    /// The pixel of the road under a radar point (dRel from the radar, yRel left positive).
    pub fn radar_ground(&self, d_rel: f64, y_rel: f64) -> Option<(f64, f64)> {
        self.project([d_rel + CAM2B, -y_rel, self.height])
    }

    /// Where the ray of a pixel meets the flat road: (x forward from the camera, y right).
    pub fn ground_from_pixel(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let d = apply(&self.inv, [u, v, 1.]);
        (d[2] > 1e-4).then(|| (self.height / d[2] * d[0], self.height / d[2] * d[1]))
    }
}

/// det-bench `xmatch.gate` for vehicle boxes: whether `track` can be the car of `bbox` (road frame pixels), and how
/// far the box centre is from the track's pixel (in box widths; lower is better).
pub fn front_gate(cam: &FrontCam, track: &Track, bbox: &[f64; 4]) -> Option<f64> {
    let x = track.d_rel + CAM2B;
    let (u, v) = cam.radar_ground(track.d_rel, track.y_rel)?;
    let [x1, y1, x2, y2] = *bbox;
    let (w, h) = (x2 - x1, y2 - y1);
    let margin = (0.2 * w).max(8.) + if x < 12. { 0.5 * w } else { 0. };
    if !(x1 - margin <= u && u <= x2 + margin) {
        return None;
    }
    let bottom = y2 >= FRONT_BOTTOM;
    if !(y1 + 0.3 * h <= v && (bottom || v <= y2 + (0.35 * h).max(15.))) {
        return None;
    }
    if !bottom {
        if let Some((gx, _)) = cam.ground_from_pixel(0.5 * (x1 + x2), y2) {
            if gx > 0. && !(0.5..=2.0).contains(&(gx / x)) {
                return None;
            }
        }
    }
    let width_m = w * x / FOCAL;
    if !((if x < 12. { 0.3 } else { 0.8 })..=8.).contains(&width_m) {
        return None;
    }
    Some((u - 0.5 * (x1 + x2)).abs() / w.max(1.))
}

/// A side view box's car is the radar track this close to its ground position (the camera car merge gate).
const SIDE_MATCH_X: f64 = 2.0;
const SIDE_MATCH_Y: f64 = 1.0;

/// The track a side view candidate stands on: candidate x is from the front bumper, yc right positive.
pub fn side_gate(cand: &Cand, track: &Track) -> Option<f64> {
    let (dx, dy) = ((track.d_rel - cand.x).abs(), (-track.y_rel - cand.yc).abs());
    (dx <= SIDE_MATCH_X && dy <= SIDE_MATCH_Y).then_some(dx + dy)
}

/// The answers of one vid.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Vote {
    shown: u8,
    confirmed_at: f64,
    /// Time of the last counted answer.
    last: f64,
    streak: (u8, u32),
    counted: u32,
    prob: f64,
}

impl Vote {
    fn new() -> Self {
        Self { shown: KIND_CAR, confirmed_at: f64::NAN, last: f64::NEG_INFINITY, streak: (0, 0), counted: 0, prob: 0. }
    }

    fn observe(&mut self, t: f64, kind: u8, prob: f64) {
        if prob < MIN_PROB || !matches!(kind, KIND_CAR | KIND_TRUCK | KIND_BUS) || t - self.last < GAP_S {
            return;
        }
        self.last = t;
        self.counted += 1;
        self.streak = if self.streak.0 == kind { (kind, self.streak.1 + 1) } else { (kind, 1) };
        let confirmed = self.confirmed_at.is_finite();
        let need = if confirmed { N_FLIP } else { N_CONFIRM };
        if self.streak.1 >= need && (!confirmed || kind != self.shown) {
            self.shown = kind;
            self.confirmed_at = t;
            self.streak = (0, 0);
        }
        if kind == self.shown {
            self.prob = prob;
        }
    }

    fn kind(&self) -> Kind {
        Kind { class: self.shown, conf: self.prob as f32, confirmed_at: self.confirmed_at, frames: self.counted }
    }
}

/// Every vid's vote.
#[derive(Default)]
pub struct Kinds {
    votes: HashMap<u32, Vote>,
}

impl Kinds {
    /// Tie `results` to vids and count them. `tracks_at` gives the radar tracks at a frame's time, `vid_of` the vid
    /// of a radar track now, `side_cands` the candidates a side frame was stepped with.
    pub fn observe(
        &mut self,
        results: &[KindResult],
        cam: Option<&FrontCam>,
        tracks_at: impl Fn(u64) -> Option<Vec<Track>>,
        side_cands: impl Fn(u64, Side) -> Option<Vec<Cand>>,
        vid_of: impl Fn(i64) -> Option<u32>,
    ) {
        // Per frame and track the best placed box only.
        let mut best: HashMap<(u64, i64), (f64, &KindResult)> = HashMap::new();
        for result in results {
            let Some(tracks) = tracks_at(result.time_ns) else { continue };
            let placed = match result.view {
                KindView::Front => cam.and_then(|cam| {
                    tracks.iter().filter_map(|t| Some((front_gate(cam, t, &result.xyxy)?, t.id))).min_by(|a, b| a.0.total_cmp(&b.0))
                }),
                KindView::Side(side) if SIDE_KINDS => side_cands(result.time_ns, side).and_then(|cands| {
                    let cand = cands.iter().find(|c| c.bbox.iter().zip(&result.xyxy).all(|(a, b)| (a - b).abs() < 0.5))?;
                    tracks.iter().filter_map(|t| Some((side_gate(cand, t)?, t.id))).min_by(|a, b| a.0.total_cmp(&b.0))
                }),
                KindView::Side(_) => None,
            };
            if let Some((q, id)) = placed {
                let entry = best.entry((result.time_ns, id)).or_insert((q, result));
                if q < entry.0 {
                    *entry = (q, result);
                }
            }
        }
        let mut placed: Vec<_> = best.into_iter().filter_map(|((t, id), (_, r))| Some((t, vid_of(id)?, r))).collect();
        placed.sort_by_key(|(t, vid, _)| (*t, *vid));
        for (t, vid, result) in placed {
            self.votes.entry(vid).or_insert_with(Vote::new).observe(t as f64 * 1e-9, result.kind, result.prob);
        }
    }

    /// Put the votes on the vehicles; a vid that is gone takes its vote with it.
    pub fn apply(&mut self, cars: &mut [Vehicle]) {
        self.votes.retain(|vid, _| cars.iter().any(|car| car.vid == *vid));
        for car in cars {
            car.kind = self.votes.get(&car.vid).map(Vote::kind);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.votes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_answers_apart_confirm_and_four_flip() {
        let mut vote = Vote::new();
        vote.observe(0.0, KIND_TRUCK, 0.9);
        assert!(!vote.confirmed_at.is_finite());
        vote.observe(0.5, KIND_TRUCK, 0.9);
        assert!(!vote.confirmed_at.is_finite(), "closer than GAP_S: not counted");
        vote.observe(1.0, KIND_TRUCK, 0.59);
        assert!(!vote.confirmed_at.is_finite(), "below MIN_PROB: not counted");
        vote.observe(1.0, KIND_TRUCK, 0.7);
        assert_eq!((vote.shown, vote.confirmed_at), (KIND_TRUCK, 1.0));
        for (i, t) in [2.0, 3.0, 4.0].into_iter().enumerate() {
            vote.observe(t, KIND_CAR, 0.9);
            assert_eq!(vote.shown, KIND_TRUCK, "{i}");
        }
        vote.observe(5.0, KIND_CAR, 0.9);
        assert_eq!((vote.shown, vote.confirmed_at), (KIND_CAR, 5.0));
        let mut mixed = Vote::new();
        for (t, k) in [(0.0, KIND_TRUCK), (1.0, KIND_CAR), (2.0, KIND_TRUCK), (3.0, KIND_CAR)] {
            mixed.observe(t, k, 0.9);
        }
        assert!(!mixed.confirmed_at.is_finite(), "never two in a row");
    }

    #[test]
    fn a_radar_point_projects_back_onto_its_ground_pixel() {
        let cam = FrontCam::new([0.0, 0.02, -0.01], 1.22).unwrap();
        let (u, v) = cam.radar_ground(18.0, 1.5).unwrap();
        let (x, y) = cam.ground_from_pixel(u, v).unwrap();
        assert!((x - (18.0 + CAM2B)).abs() < 1e-6 && (y + 1.5).abs() < 1e-6, "{x} {y}");
        assert!(FrontCam::new([f64::NAN, 0., 0.], 1.2).is_none());
    }

    #[test]
    fn a_front_box_takes_the_track_under_it() {
        let cam = FrontCam::new([0.0, 0.0, 0.0], 1.22).unwrap();
        let track = |id, d_rel, y_rel| Track { id, d_rel, y_rel, v_rel: 0. };
        let ahead = track(7, 20.0, 0.0);
        let (u, v) = cam.radar_ground(20.0, 0.0).unwrap();
        // A 1.8 m wide, 1.5 m high car at 21.5 m from the camera.
        let w = 1.8 * FOCAL / 21.52;
        let h = 1.5 * FOCAL / 21.52;
        let bbox = [u - w / 2., v - h, u + w / 2., v + 2.];
        assert!(front_gate(&cam, &ahead, &bbox).is_some_and(|q| q < 0.01));
        assert_eq!(front_gate(&cam, &track(8, 20.0, 3.5), &bbox), None, "the next lane");
        assert_eq!(front_gate(&cam, &track(9, 45.0, 0.0), &bbox), None, "farther than the box's ground point");
        let results = [KindResult { time_ns: 2_000_000_000, view: KindView::Front, xyxy: bbox, kind: KIND_TRUCK, prob: 0.9 }];
        let mut kinds = Kinds::default();
        let tracks = vec![ahead, track(8, 20.0, 3.5)];
        let vid_of = |id: i64| (id == 7).then_some(70);
        kinds.observe(&results, Some(&cam), |_| Some(tracks.clone()), |_, _| None, vid_of);
        let later = [KindResult { time_ns: 3_000_000_000, ..results[0] }];
        kinds.observe(&later, Some(&cam), |_| Some(tracks.clone()), |_, _| None, vid_of);
        assert_eq!(kinds.votes[&70].shown, KIND_TRUCK);
        assert!(kinds.votes[&70].confirmed_at.is_finite());
        let side = [KindResult { view: KindView::Side(Side::Left), ..later[0] }];
        let before = kinds.votes.clone();
        kinds.observe(&side, Some(&cam), |_| Some(tracks.clone()), |_, _| Some(vec![Cand::at(det_rules::View::SideL, 20.0, 0.0)]), vid_of);
        assert_eq!(kinds.votes, before, "side answers are not used");
    }
}
