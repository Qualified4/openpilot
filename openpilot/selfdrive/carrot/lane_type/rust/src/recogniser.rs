//! marking_daemon.Recogniser (StripNet, mode 2) with the product switches: lane-change variant C2, gate G1, the default
//! strip filter. The replay variants (CCNC_LANECHANGE_INFER other than C2, GATE_RELAX, HOLD_COLOR) are not ported.

use std::time::Instant;

use crate::filter::{p_of, StripFilter};
use crate::display::Confirmed;
use crate::model::{Decision, StripModel};
use crate::result::{filter_input, Extras, SideResult};
use crate::strip::{self, LineF64, Nv12};

pub const LANE_PROB_MIN: f64 = 0.25;
pub const LANE_PROB_LOST: f64 = 0.15;
pub const LANE_DIP_MAX_NS: i64 = 1_000_000_000;
pub const LANECHANGE_SPEED: f64 = 24. / 3.6;
pub const LEAD_GATE_DREL: f64 = 4.;
pub const C2_BLINKER_PROB: f64 = 0.3;
pub const LANECHANGE_CONFIDENCE: f64 = 0.9;

/// The frame dict marking_daemon.main builds for Recogniser.process.
#[derive(Clone, Debug)]
pub struct Frame {
    pub lines: [LineF64; 2],
    pub probs: [f64; 2],
    pub rpy: [f64; 3],
    pub k: [[f64; 3]; 3],
    pub stamp: i64,
    pub odometer: f64,
    pub blinker: bool,
    /// modelV2.meta.laneChangeState (0 = off).
    pub lcs: u16,
    pub v_ego: f64,
    pub lead_d_rel: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timing {
    pub extract_ms: f64,
    pub infer_ms: f64,
    /// Extraction stages summed over the sides (both threads' wall time, so it can exceed extract_ms).
    pub stages: strip::Stages,
}

pub struct Recogniser {
    pub model: StripModel,
    filter: StripFilter,
    dip_since: [Option<i64>; 2],
    crossing_y: [Option<f64>; 2],
    crossing_min: [Option<f64>; 2],
    crossing_stamp: Option<i64>,
    lc_seen: [[bool; 2]; 2],
    lc_frame_stamp: Option<i64>,
    lc_fresh: [[bool; 2]; 2],
    dist: Vec<f64>,
    pub threads: usize,
    /// Reuse each side's rows / maps while the geometry inputs are bit-identical (strip::MapCache).
    pub map_cache_on: bool,
    map_cache: [strip::MapCache; 2],
    pub timing: Timing,
    /// Replay diagnostics: keep each inferred side's strip sha256 (first 16 hex) and logits.
    pub record: bool,
    pub last: [Option<Diag>; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Diag {
    pub strip_sha: String,
    pub type_logits: Vec<f32>,
    pub color_logits: Vec<f32>,
}

fn strip_sha(s: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for v in s {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
}

impl Recogniser {
    pub fn new(model: StripModel) -> Self {
        Recogniser {
            model,
            filter: StripFilter::default(),
            dip_since: [None; 2],
            crossing_y: [None; 2],
            crossing_min: [None; 2],
            crossing_stamp: None,
            lc_seen: [[false; 2]; 2],
            lc_frame_stamp: None,
            lc_fresh: [[false; 2]; 2],
            dist: strip::distances(),
            threads: 2,
            map_cache_on: strip::map_cache_enabled(),
            map_cache: Default::default(),
            timing: Timing::default(),
            record: false,
            last: [None, None],
        }
    }

    pub fn reset(&mut self) {
        self.filter.reset();
        self.dip_since = [None; 2];
        self.crossing_y = [None; 2];
        self.crossing_min = [None; 2];
        self.crossing_stamp = None;
        self.lc_seen = [[false; 2]; 2];
        self.lc_frame_stamp = None;
        self.lc_fresh = [[false; 2]; 2];
    }

    fn lifecycle_resets(&mut self, frame: &Frame) -> [bool; 2] {
        if self.lc_frame_stamp.is_some_and(|s| frame.stamp <= s) {
            self.lc_seen = [[false; 2]; 2];
        }
        self.lc_frame_stamp = Some(frame.stamp);
        let mut resets = [false; 2];
        for side in 0..2 {
            if self.lc_seen[side].iter().any(|&h| h) && frame.v_ego < LANECHANGE_SPEED {
                resets[side] = true;
                self.lc_seen[side] = [false; 2];
            }
        }
        resets
    }

    pub fn observe_confirmation(&mut self, confirmed: &[Confirmed; 2], frame: &Frame) {
        for side in 0..2 {
            for (head, h) in [confirmed[side].kind, confirmed[side].color].into_iter().enumerate() {
                let Some(h) = h else { continue };
                if h.stamp != frame.stamp || !self.lc_fresh[side][head] {
                    continue;
                }
                if frame.blinker && frame.v_ego >= LANECHANGE_SPEED {
                    self.lc_seen[side][head] = true;
                } else if !frame.blinker {
                    self.lc_seen[side][head] = false;
                }
            }
        }
    }

    fn lanechange_infer(frame: &Frame) -> bool {
        frame.v_ego >= LANECHANGE_SPEED && (frame.blinker || frame.lcs != 0)
    }

    fn crossing(&mut self, frame: &Frame) -> bool {
        let stamp = frame.stamp;
        let changing = Self::lanechange_infer(frame);
        if !changing || self.crossing_stamp.is_some_and(|c| !(0 < stamp - c && stamp - c <= 300_000_000)) {
            self.crossing_y = [None; 2];
            self.crossing_min = [None; 2];
        }
        self.crossing_stamp = Some(stamp);
        if !changing {
            return false;
        }
        let mut crossed = false;
        let mut current = [None; 2];
        for side in 0..2 {
            let ys = &frame.lines[side].y;
            if ys.is_empty() || !ys[0].is_finite() {
                self.crossing_y[side] = None;
                self.crossing_min[side] = None;
                continue;
            }
            let y = ys[0];
            current[side] = Some(y);
            let old = self.crossing_y[side];
            let near = self.crossing_min[side].unwrap_or(y.abs());
            crossed |= old.is_some_and(|o| o * y < 0.) || (near < 1. && y.abs() - near > 1.5);
            self.crossing_min[side] = Some(near.min(y.abs()));
            if y != 0. {
                self.crossing_y[side] = Some(y);
            }
        }
        if crossed {
            self.crossing_min = current.map(|y| y.map(f64::abs));
            self.crossing_y = current;
        }
        crossed
    }

    fn skip_reason(&mut self, side: usize, frame: &Frame) -> Option<&'static str> {
        let (prob, stamp) = (frame.probs[side], frame.stamp);
        let lead = || frame.lead_d_rel.is_some_and(|d| d < LEAD_GATE_DREL).then_some("lead_close");
        if Self::lanechange_infer(frame) {
            self.dip_since[side] = None;
            if frame.lcs == 0 && !(prob >= C2_BLINKER_PROB) {
                return Some("lane_prob_dip");
            }
            return lead();
        }
        if frame.blinker || frame.lcs != 0 {
            return Some("lane_change");
        }
        let since = if prob >= LANE_PROB_MIN {
            self.dip_since[side] = None;
            stamp
        } else {
            let since = self.dip_since[side].get_or_insert(stamp);
            if stamp < *since {
                *since = stamp;
            }
            *since
        };
        if prob >= LANE_PROB_MIN {
            return lead();
        }
        Some(if prob >= LANE_PROB_LOST && stamp - since <= LANE_DIP_MAX_NS { "lane_prob_dip" } else { "not_detected" })
    }

    /// Recogniser.process: the per-side raw (filtered) results.
    pub fn process(&mut self, image: &Nv12, frame: &Frame) -> anyhow::Result<[SideResult; 2]> {
        let lifecycle = self.lifecycle_resets(frame);
        let extract_start = Instant::now();
        let mut out: [Option<SideResult>; 2] = [None, None];
        let mut eligible = Vec::new();
        for side in 0..2 {
            match self.skip_reason(side, frame) {
                Some(skip) => out[side] = Some(filter_input("unknown", "unknown", 0., 0., frame.stamp, false, skip)),
                None => eligible.push(side),
            }
        }
        let m = strip::projection(frame.rpy, &frame.k);
        let shape = self.model.strip;
        let mirror = self.model.mirror_right;
        let dist = &self.dist;
        let cache_on = self.map_cache_on;
        let extract = |side: usize, cache: &mut strip::MapCache| {
            let mut st = strip::Stages::default();
            let s = strip::extract_cached(image, &frame.lines[side], &m, shape, mirror && side == 1, dist, &mut st,
                                          cache_on.then_some(cache));
            (s, st)
        };
        // Both sides in parallel (2 threads, as the Python daemon's cv2.setNumThreads(2)).
        let threads = self.threads;
        let caches = &mut self.map_cache;
        let extracted: Vec<(usize, (Option<Vec<f32>>, strip::Stages))> = if threads > 1 && eligible.len() == 2 {
            let [c0, c1] = caches; // eligible == [0, 1]
            std::thread::scope(|s| {
                let h = s.spawn(|| extract(1, c1));
                let a = extract(0, c0);
                vec![(0, a), (1, h.join().unwrap())]
            })
        } else {
            eligible.iter().map(|&s| (s, extract(s, &mut caches[s]))).collect()
        };
        let mut stages = strip::Stages::default();
        let mut strips: Vec<(usize, Option<Vec<f32>>)> = extracted
            .into_iter()
            .map(|(side, (s, st))| {
                stages.add(&st);
                (side, s)
            })
            .collect();
        strips.retain(|(side, s)| {
            if s.is_none() {
                out[*side] = Some(filter_input("unknown", "unknown", 0., 0., frame.stamp, false, "invalid_lane_geometry"));
            }
            s.is_some()
        });
        let infer_start = Instant::now();
        let model = &self.model;
        let record = self.record;
        let run = |s: &Vec<f32>| -> anyhow::Result<(Decision, Option<Diag>)> {
            let (t, c) = model.forward(s)?;
            let d = model.decide4(&t, &c);
            Ok((d, record.then(|| Diag { strip_sha: strip_sha(s), type_logits: t, color_logits: c })))
        };
        let decisions: Vec<(usize, (Decision, Option<Diag>))> = if model.batched() && strips.len() == 2 {
            let (a, b) = (strips[0].1.as_ref().unwrap(), strips[1].1.as_ref().unwrap());
            let [x, y] = model.forward2(a, b)?;
            [(strips[0].0, a, x), (strips[1].0, b, y)]
                .into_iter()
                .map(|(side, s, (t, c))| {
                    let d = model.decide4(&t, &c);
                    (side, (d, record.then(|| Diag { strip_sha: strip_sha(s), type_logits: t, color_logits: c })))
                })
                .collect()
        } else if self.threads > 1 && strips.len() == 2 {
            std::thread::scope(|s| -> anyhow::Result<_> {
                let h = s.spawn(|| run(strips[1].1.as_ref().unwrap()));
                let a = run(strips[0].1.as_ref().unwrap())?;
                Ok(vec![(strips[0].0, a), (strips[1].0, h.join().unwrap()?)])
            })?
        } else {
            strips.iter().map(|(side, s)| Ok((*side, run(s.as_ref().unwrap())?))).collect::<anyhow::Result<_>>()?
        };
        let now = Instant::now();
        self.timing = Timing {
            extract_ms: (infer_start - extract_start).as_nanos() as f64 / 1e6,
            infer_ms: (now - infer_start).as_nanos() as f64 / 1e6,
            stages,
        };
        let mut models: [Option<Decision>; 2] = [None, None];
        self.last = [None, None];
        for (side, (r, diag)) in decisions {
            self.last[side] = diag;
            let mut o = filter_input(r.kind, r.color, r.type_confidence, r.color_confidence, frame.stamp, r.kind == "dashed", r.reason);
            o.extras = Some(Extras { type_class: r.type_class, color_class: r.color_class, double: r.double, blue: r.blue, p_unknown: r.p_unknown });
            out[side] = Some(o);
            models[side] = Some(r);
        }
        let mut out = out.map(|o| o.expect("every side has a result"));
        let changing = Self::lanechange_infer(frame);
        let crossed = self.crossing(frame);
        if changing {
            for side in 0..2 {
                let Some(r) = &models[side] else { continue };
                let o = &mut out[side];
                if !r.type_confidence.is_finite() || r.type_confidence < LANECHANGE_CONFIDENCE {
                    o.kind = "unknown";
                    o.type_confidence = 0.;
                }
                if !r.color_confidence.is_finite() || r.color_confidence < LANECHANGE_CONFIDENCE {
                    o.color = "unknown";
                    o.color_confidence = 0.;
                }
                if o.kind == "unknown" && o.color == "unknown" {
                    o.reason = "lane_prob_dip";
                }
            }
        }
        for side in 0..2 {
            if crossed || lifecycle[side] {
                out[side].tracking_reset = true;
                self.dip_since[side] = None;
            }
        }
        for side in 0..2 {
            let r = &models[side];
            let learned = r.as_ref().is_some_and(|r| r.reason == "strip_model");
            let (mut p, mut py) = match r {
                Some(r) if learned => (Some(p_of(r.type_candidate, r.type_confidence, "dashed")),
                                       Some(p_of(r.color_candidate, r.color_confidence, "yellow"))),
                _ => (None, None),
            };
            if changing {
                if out[side].kind == "unknown" {
                    p = None;
                }
                if out[side].color == "unknown" {
                    py = None;
                }
            }
            out[side] = self.filter.update(side, out[side].clone(), p, frame.stamp, frame.odometer, py);
        }
        for side in 0..2 {
            self.lc_fresh[side] = [false; 2];
            if let Some(r) = &models[side] {
                let ok = |value: &str, mine: &str, conf: f64| r.reason == "strip_model" && value == mine && value != "unknown" && conf >= 0.9;
                self.lc_fresh[side] = [ok(r.kind, out[side].kind, r.type_confidence), ok(r.color, out[side].color, r.color_confidence)];
            }
        }
        Ok(out)
    }
}
