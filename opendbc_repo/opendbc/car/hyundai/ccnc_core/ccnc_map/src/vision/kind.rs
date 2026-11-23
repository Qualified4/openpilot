//! Vehicle kinds (car, truck, bus) from `ccnc_visiond`'s classifier, put on the radar vehicles (item 68).
//!
//! The detector decides whether a car is drawn, the classifier only which shape: a kind never adds, moves or
//! removes a slot. `ccnc_visiond` sends each answer with the box and frame it is for, one or two messages after the
//! frame. Here a front box is tied to the radar track it covers at the frame's time (the det-bench `xmatch` gate),
//! the track to its vehicle (vid), and the vid's answers are voted on:
//!
//! * only answers with a probability of `MIN_PROB` (car) or `MIN_PROB_TRUCK` (truck, bus) or more count;
//! * two answers of the same shape at least `GAP_S` and at most `MAX_GAP_S` apart confirm it (det-light 6: no "different view" condition,
//!   a standing car never gives one). The shape is what the cluster draws: car, or truck for a truck and a bus
//!   alike, so a truck the classifier calls truck once and bus once is confirmed; the kind shown is the last one;
//! * a confirmed shape changes after `N_FLIP` agreeing answers of the other shape;
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

/// Car answers count from this probability, truck and bus answers from `MIN_PROB_TRUCK` (the det-kind replay's one
/// wrong truck was a dusk SUV at 0.82 and 0.76).
pub const MIN_PROB: f64 = 0.6;
pub const MIN_PROB_TRUCK: f64 = 0.8;
pub const GAP_S: f64 = 0.8;
pub const MAX_GAP_S: f64 = 10.0;
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

/// The cluster shape of a kind: car, or truck (truck and bus).
fn shape(kind: u8) -> u8 {
    if kind == KIND_CAR {
        KIND_CAR
    } else {
        KIND_TRUCK
    }
}

/// The least probability an answer of this shape needs to count.
fn min_prob(shape: u8) -> f64 {
    if shape == KIND_CAR {
        MIN_PROB
    } else {
        MIN_PROB_TRUCK
    }
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

    fn confirmed(&self) -> bool {
        self.confirmed_at.is_finite()
    }

    /// Count an answer; whether it counted.
    fn observe(&mut self, t: f64, kind: u8, prob: f64) -> bool {
        if !matches!(kind, KIND_CAR | KIND_TRUCK | KIND_BUS) || prob < min_prob(shape(kind)) || t - self.last < GAP_S {
            return false;
        }
        let confirmed = self.confirmed();
        if !confirmed && t - self.last > MAX_GAP_S {
            // An answer that is not confirmed yet expires: two of a shape count only within MAX_GAP_S.
            self.streak = (0, 0);
        }
        self.last = t;
        self.counted += 1;
        let answer = shape(kind);
        self.streak = if self.streak.0 == answer { (answer, self.streak.1 + 1) } else { (answer, 1) };
        let need = if confirmed { N_FLIP } else { N_CONFIRM };
        if self.streak.1 >= need && (!confirmed || answer != shape(self.shown)) {
            self.confirmed_at = t;
            self.streak = (0, 0);
        }
        if answer == shape(self.shown) || self.confirmed_at == t {
            // The shown kind follows the latest answer of the shown shape (truck or bus).
            self.shown = kind;
            self.prob = prob;
        }
        true
    }

    fn kind(&self) -> Kind {
        Kind { class: self.shown, conf: self.prob as f32, confirmed_at: self.confirmed_at, frames: self.counted }
    }
}

/// How long the map asks for road camera answers about one vid while it is in FF, LF or RF. `ccnc_visiond` looks at
/// the road camera only while some slot asks, so these bound its load (one frame answers every car in it).
///
/// Before the kind is confirmed: a round of `ASK_S` asking or `ASK_ANSWERS` answers, then `ASK_AGAIN_S` rest and one
/// more round, then never again for that vid. A car that is never answered (cut, small, unsure, the classifier
/// gone) ends by time alike. 12 s is twice the slowest truck confirmation of the det-kind replay (6.1 s from its
/// first slot; median 2.3 s); 8 answers are four times the two a confirmation needs.
pub const ASK_S: f64 = 12.0;
pub const ASK_ANSWERS: u32 = 8;
pub const ASK_AGAIN_S: f64 = 60.0;
pub const ASK_ROUNDS: u32 = 2;
/// Once confirmed: from `RECHECK_S` after it, every `RECHECK_S` one road frame for every confirmed car ahead at once
/// (night frames are not answered: `ccnc_visiond` then rests its road camera), asked for at most
/// `RECHECK_ASK_S` (two standing periods). An answer of the other shape asks at the full pace until the vote flips
/// (`N_FLIP`) or one answer of the confirmed shape comes, within `FLIP_ASK_S` of asking per vid.
pub const RECHECK_S: f64 = 5.0;
pub const RECHECK_ASK_S: f64 = 4.0;
pub const FLIP_ASK_S: f64 = 12.0;
/// A radar update counts for this much asking at most (liveTracks is 20 Hz; a slot gap is not asking time).
const ASK_STEP_S: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Ask {
    /// Before the kind is confirmed.
    round: u32,
    asked_s: f64,
    answers: u32,
    rest_until: f64,
    /// After it: asking at the full pace for a flip, and the time that took.
    flipping: bool,
    flip_asked_s: f64,
    last_step: f64,
}

impl Ask {
    fn new() -> Self {
        Self {
            round: 1,
            asked_s: 0.,
            answers: 0,
            rest_until: f64::NEG_INFINITY,
            flipping: false,
            flip_asked_s: 0.,
            last_step: f64::NAN,
        }
    }

    fn step(&mut self, now: f64) -> f64 {
        let step = if self.last_step.is_nan() { 0. } else { (now - self.last_step).clamp(0., ASK_STEP_S) };
        self.last_step = now;
        step
    }

    /// Before the kind is confirmed.
    fn unconfirmed(&mut self, now: f64, step: f64) -> bool {
        if now < self.rest_until {
            return false;
        }
        if self.asked_s >= ASK_S || self.answers >= ASK_ANSWERS {
            self.rest_until = if self.round < ASK_ROUNDS { now + ASK_AGAIN_S } else { f64::INFINITY };
            self.round += 1;
            self.asked_s = 0.;
            self.answers = 0;
            return false;
        }
        self.asked_s += step;
        true
    }

    /// While a confirmed vid's last answer was of the other shape.
    fn flip(&mut self, step: f64) -> bool {
        if self.flipping && self.flip_asked_s < FLIP_ASK_S {
            self.flip_asked_s += step;
            return true;
        }
        self.flipping = false;
        false
    }
}

/// Every vid's vote and asking.
pub struct Kinds {
    votes: HashMap<u32, Vote>,
    asks: HashMap<u32, Ask>,
    /// The recheck of the confirmed cars: when the current one started and whether a road frame came since.
    recheck_from: f64,
    recheck_served: bool,
}

impl Default for Kinds {
    fn default() -> Self {
        Self { votes: HashMap::new(), asks: HashMap::new(), recheck_from: f64::NEG_INFINITY, recheck_served: false }
    }
}

impl Kinds {
    /// Tie `results` to vids and count them. `tracks_near` gives the radar tracks recorded closest to a frame's time
    /// and that record's time (None when it is farther than the tolerance), `vid_at` the vid of a radar track at a
    /// record's time, `side_cands` the candidates a side frame was stepped with.
    pub fn observe(
        &mut self,
        results: &[KindResult],
        cam: Option<&FrontCam>,
        tracks_near: impl Fn(u64) -> Option<(u64, Vec<Track>)>,
        side_cands: impl Fn(u64, Side) -> Option<Vec<Cand>>,
        vid_at: impl Fn(u64, i64) -> Option<u32>,
    ) {
        // Per frame and track the best placed box only.
        let mut best: HashMap<(u64, i64), (f64, u64, &KindResult)> = HashMap::new();
        for result in results {
            let Some((stamp, tracks)) = tracks_near(result.time_ns) else { continue };
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
                let entry = best.entry((result.time_ns, id)).or_insert((q, stamp, result));
                if q < entry.0 {
                    *entry = (q, stamp, result);
                }
            }
        }
        let mut placed: Vec<_> = best.into_iter().filter_map(|((t, id), (_, stamp, r))| Some((t, vid_at(stamp, id)?, r))).collect();
        placed.sort_by_key(|(t, vid, _)| (*t, *vid));
        for (t, vid, result) in placed {
            self.count(vid, t as f64 * 1e-9, result.kind, result.prob);
        }
    }

    /// One answer about `vid` for a frame at `t`.
    fn count(&mut self, vid: u32, t: f64, kind: u8, prob: f64) {
        let vote = self.votes.entry(vid).or_insert_with(Vote::new);
        let before = (vote.confirmed(), vote.confirmed_at, shape(vote.shown));
        let counted = vote.observe(t, kind, prob);
        let ask = self.asks.entry(vid).or_insert_with(Ask::new);
        if t >= ask.rest_until {
            ask.answers += 1;
        }
        if before.0 && counted {
            // A confirmed car: the other shape asks for more until it flips or the confirmed shape comes back.
            ask.flipping = shape(kind) != before.2 && vote.confirmed_at == before.1 && ask.flip_asked_s < FLIP_ASK_S;
        }
    }

    /// A road camera frame (its frame time) reached the map: it serves the recheck under way.
    pub fn front_frame(&mut self, time_ns: u64) {
        if time_ns as f64 * 1e-9 >= self.recheck_from {
            self.recheck_served = true;
        }
    }

    /// Whether to ask `ccnc_visiond` for road camera answers about `vid`, a radar car in FF, LF or RF now. Called
    /// once per radar update for each such slot; the time between calls is the asking time.
    pub fn wanted(&mut self, vid: u32, now: f64) -> bool {
        let confirmed_at = self.votes.get(&vid).map_or(f64::NAN, |vote| vote.confirmed_at);
        let ask = self.asks.entry(vid).or_insert_with(Ask::new);
        let step = ask.step(now);
        if !confirmed_at.is_finite() {
            return ask.unconfirmed(now, step);
        }
        if ask.flip(step) {
            return true;
        }
        if now < confirmed_at + RECHECK_S {
            return false;
        }
        if now >= self.recheck_from + RECHECK_S {
            self.recheck_from = now;
            self.recheck_served = false;
        }
        !self.recheck_served && now < self.recheck_from + RECHECK_ASK_S
    }

    /// Put the votes on the vehicles; a vid that is gone takes its vote and asking with it.
    pub fn apply(&mut self, cars: &mut [Vehicle]) {
        self.votes.retain(|vid, _| cars.iter().any(|car| car.vid == *vid));
        self.asks.retain(|vid, _| cars.iter().any(|car| car.vid == *vid));
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
        vote.observe(1.0, KIND_TRUCK, 0.79);
        assert!(!vote.confirmed_at.is_finite(), "a truck needs MIN_PROB_TRUCK");
        vote.observe(1.0, KIND_TRUCK, 0.8);
        assert_eq!((vote.shown, vote.confirmed_at), (KIND_TRUCK, 1.0));
        for (i, t) in [2.0, 3.0, 4.0].into_iter().enumerate() {
            vote.observe(t, KIND_CAR, 0.9);
            assert_eq!(vote.shown, KIND_TRUCK, "{i}");
        }
        vote.observe(5.0, KIND_CAR, 0.9);
        assert_eq!((vote.shown, vote.confirmed_at), (KIND_CAR, 5.0));
        let mut big = Vote::new();
        big.observe(0.0, KIND_TRUCK, 0.9);
        big.observe(1.0, KIND_BUS, 0.8);
        assert_eq!((big.shown, big.confirmed_at), (KIND_BUS, 1.0), "truck then bus: the truck shape, the last kind shown");
        big.observe(2.0, KIND_TRUCK, 0.85);
        assert_eq!((big.shown, big.confirmed_at), (KIND_TRUCK, 1.0));
        let mut mixed = Vote::new();
        for (t, k) in [(0.0, KIND_TRUCK), (1.0, KIND_CAR), (2.0, KIND_TRUCK), (3.0, KIND_CAR)] {
            mixed.observe(t, k, 0.9);
        }
        assert!(!mixed.confirmed_at.is_finite(), "never two in a row");
        let mut late = Vote::new();
        late.observe(0.0, KIND_CAR, 0.9);
        late.observe(10.5, KIND_CAR, 0.9);
        assert!(!late.confirmed_at.is_finite(), "farther than MAX_GAP_S: the first answer expired");
        late.observe(20.4, KIND_CAR, 0.9);
        assert_eq!(late.confirmed_at, 20.4, "two within MAX_GAP_S");
        late.observe(60.0, KIND_TRUCK, 0.9);
        late.observe(90.0, KIND_TRUCK, 0.9);
        late.observe(120.0, KIND_TRUCK, 0.9);
        assert_eq!(late.shown, KIND_CAR);
        late.observe(150.0, KIND_TRUCK, 0.9);
        assert_eq!((late.shown, late.confirmed_at), (KIND_TRUCK, 150.0), "a flip does not expire");
    }

    /// Ask about `vid` at 20 Hz from `from` for `secs`; how long it was wanted.
    fn ask_for(kinds: &mut Kinds, vid: u32, from: f64, secs: f64) -> f64 {
        let steps = (secs / 0.05).round() as usize;
        (0..steps).filter(|i| kinds.wanted(vid, from + *i as f64 * 0.05)).count() as f64 * 0.05
    }

    fn answer(kinds: &mut Kinds, vid: u32, t: f64, kind: u8, prob: f64) {
        kinds.count(vid, t, kind, prob);
    }

    #[test]
    fn an_unconfirmed_car_is_asked_about_two_rounds_at_most() {
        let mut kinds = Kinds::default();
        // Never answered: 12 s, 60 s rest, 12 s more, then never again.
        let first = ask_for(&mut kinds, 3, 0.0, 20.0);
        assert!((first - ASK_S).abs() < 0.11, "{first}");
        assert_eq!(ask_for(&mut kinds, 3, 20.0, 50.0), 0.0, "resting");
        let second = ask_for(&mut kinds, 3, 72.1, 20.0);
        assert!((second - ASK_S).abs() < 0.11, "{second}");
        assert_eq!(ask_for(&mut kinds, 3, 100.0, 300.0), 0.0, "two rounds only");
        // Answered but never sure: the eighth answer ends the round.
        let mut unsure = Kinds::default();
        for i in 0..ASK_ANSWERS {
            assert!(unsure.wanted(4, f64::from(i)));
            answer(&mut unsure, 4, f64::from(i), KIND_TRUCK, 0.7);
        }
        assert!(!unsure.wanted(4, 8.0));
        assert!(!unsure.votes[&4].confirmed());
        // Out of the slots ahead is not asking time.
        let mut gap = Kinds::default();
        assert!(gap.wanted(5, 0.0));
        assert!(gap.wanted(5, 30.0));
        assert!(gap.asks[&5].asked_s <= ASK_STEP_S);
    }

    #[test]
    fn confirmed_cars_are_rechecked_together_and_asked_more_on_a_disagreement() {
        let mut kinds = Kinds::default();
        for vid in [1, 2] {
            answer(&mut kinds, vid, 0.0, KIND_CAR, 0.9);
            answer(&mut kinds, vid, 1.0, KIND_CAR, 0.9);
            assert!(kinds.votes[&vid].confirmed());
        }
        assert!(!kinds.wanted(1, 1.0) && !kinds.wanted(2, 5.9), "the first recheck RECHECK_S after the confirmation");
        assert!(kinds.wanted(1, 6.0) && kinds.wanted(2, 6.0), "it opens for both");
        kinds.front_frame(6_200_000_000);
        assert!(!kinds.wanted(1, 6.5) && !kinds.wanted(2, 6.5), "one road frame serves every car");
        assert!(!kinds.wanted(1, 10.9));
        assert!(kinds.wanted(1, 11.0) && kinds.wanted(2, 11.0), "the next one RECHECK_S later");
        assert!(!kinds.wanted(1, 15.1), "unserved, it ends after RECHECK_ASK_S");
        // Frames per minute while nothing disagrees: one per recheck.
        let mut frames = 0;
        for i in 0..1200 {
            let now = 16.0 + f64::from(i) * 0.05;
            if kinds.wanted(1, now) {
                frames += 1;
                kinds.front_frame((now * 1e9) as u64);
            }
        }
        assert_eq!(frames, 12);
        // A truck answer for car 1 asks at the full pace until a car answer comes back.
        answer(&mut kinds, 1, 76.0, KIND_TRUCK, 0.9);
        assert!(kinds.asks[&1].flipping);
        assert!(kinds.wanted(2, 76.5), "the next recheck");
        kinds.front_frame(76_500_000_000);
        assert!(kinds.wanted(1, 76.6) && !kinds.wanted(2, 76.6), "car 1 alone at the full pace");
        answer(&mut kinds, 1, 77.0, KIND_CAR, 0.9);
        assert!(!kinds.wanted(1, 77.5), "the confirmed shape came back");
        assert_eq!(kinds.votes[&1].shown, KIND_CAR);
        // Four truck answers flip it, and asking stops with the flip.
        for t in [85.0, 86.0, 87.0, 88.0] {
            answer(&mut kinds, 1, t, KIND_TRUCK, 0.9);
            kinds.wanted(1, t + 0.5);
        }
        assert_eq!(kinds.votes[&1].shown, KIND_TRUCK);
        assert!(!kinds.asks[&1].flipping);
        // An answer that never settles it is asked about for FLIP_ASK_S at most.
        answer(&mut kinds, 2, 95.0, KIND_TRUCK, 0.9);
        let asked = ask_for(&mut kinds, 2, 95.0, 40.0);
        assert!(asked < FLIP_ASK_S + 8.0 * RECHECK_ASK_S, "{asked}");
        assert!(kinds.asks[&2].flip_asked_s >= FLIP_ASK_S && !kinds.asks[&2].flipping);
    }

    #[test]
    fn an_answer_goes_to_the_car_its_track_was_at_the_frame_time() {
        use crate::vehicles::{LinkGates, Tracker};
        use crate::world::World;
        let world = World::default();
        let mut tracker = Tracker::new(LinkGates::BODY);
        // Radar id 10 is one car at 1.00 s and another one at 1.05 s (det_rules same id lateral jump).
        tracker.update(1., &[(10, 20., 0., 0.), (11, 20., 1.6, 0.)], &world);
        tracker.update(1.05, &[(10, 20., 1.6, 0.)], &world);
        let (old, new) = (tracker.vid_at(1., 10).unwrap(), tracker.vid_at(1.05, 10).unwrap());
        assert_ne!(old, new);
        let cam = FrontCam::new([0.0, 0.0, 0.0], 1.22).unwrap();
        let (u, v) = cam.radar_ground(20.0, 0.0).unwrap();
        let (w, h) = (1.8 * FOCAL / 21.52, 1.5 * FOCAL / 21.52);
        let bbox = [u - w / 2., v - h, u + w / 2., v + 2.];
        let history = [
            (1_000_000_000u64, vec![Track { id: 10, d_rel: 20., y_rel: 0., v_rel: 0. }]),
            (1_050_000_000, vec![Track { id: 10, d_rel: 20., y_rel: 1.6, v_rel: 0. }]),
        ];
        let near = |t: u64| {
            let (stamp, tracks) = history.iter().min_by_key(|(s, _)| s.abs_diff(t))?;
            (stamp.abs_diff(t) <= 150_000_000).then(|| (*stamp, tracks.clone()))
        };
        let vid_at = |stamp: u64, id: i64| tracker.vid_at(stamp as f64 * 1e-9, u32::try_from(id).ok()?);
        let mut kinds = Kinds::default();
        // The answer for the 1.00 s frame arrives after the handoff.
        let result = KindResult { time_ns: 1_000_000_000, view: KindView::Front, xyxy: bbox, kind: KIND_TRUCK, prob: 0.9 };
        kinds.observe(&[result], Some(&cam), near, |_, _| None, vid_at);
        assert!(kinds.votes.contains_key(&old) && !kinds.votes.contains_key(&new));
        // A frame farther than the tolerance from every record is dropped.
        let far = KindResult { time_ns: 1_300_000_000, ..result };
        kinds.observe(&[far], Some(&cam), near, |_, _| None, vid_at);
        assert_eq!((kinds.votes.len(), kinds.votes[&old].counted), (1, 1));
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
        let vid_of = |_, id: i64| (id == 7).then_some(70);
        kinds.observe(&results, Some(&cam), |t| Some((t, tracks.clone())), |_, _| None, vid_of);
        let later = [KindResult { time_ns: 3_000_000_000, ..results[0] }];
        kinds.observe(&later, Some(&cam), |t| Some((t, tracks.clone())), |_, _| None, vid_of);
        assert_eq!(kinds.votes[&70].shown, KIND_TRUCK);
        assert!(kinds.votes[&70].confirmed_at.is_finite());
        let side = [KindResult { view: KindView::Side(Side::Left), ..later[0] }];
        let before = kinds.votes.clone();
        kinds.observe(&side, Some(&cam), |t| Some((t, tracks.clone())), |_, _| Some(vec![Cand::at(det_rules::View::SideL, 20.0, 0.0)]), vid_of);
        assert_eq!(kinds.votes, before, "side answers are not used");
    }
}
