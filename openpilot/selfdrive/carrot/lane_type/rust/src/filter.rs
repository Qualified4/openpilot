//! strip_filter.StripFilter in the product configuration only: 'asym' A8/4 type windows, the 8 m colour window, the
//! latch (colour change after COLOR_LATCH_FRAMES confident frames), type confidence 'abs', no type latch confirmation,
//! no graded latch, no hold_latch. The replay variants are not ported.

use std::collections::VecDeque;

use crate::result::{SideResult, RESET_REASONS};

pub const MAX_GAP_NS: i64 = 300_000_000;
pub const WINDOW_NS: i64 = 3_000_000_000;
pub const THRESHOLD: f64 = 0.90;
pub const LONG_M: f64 = 8.;
pub const SHORT_M: f64 = 4.;
pub const COLOR_M: f64 = 8.;
pub const COLOR_LATCH_FRAMES: u32 = 10;

#[derive(Default)]
struct Side {
    items: VecDeque<(i64, f64, f64, f64)>,
    state: Option<&'static str>,
    colors: VecDeque<(i64, f64, f64, f64)>,
    latched_type: Option<(&'static str, f64)>,
    latched_color: Option<(&'static str, f64)>,
    flips: u32,
    last_stamp: Option<i64>,
}

impl Side {
    fn clear(&mut self) {
        self.items.clear();
        self.state = None;
        self.colors.clear();
        self.latched_type = None;
        self.latched_color = None;
        self.flips = 0;
    }
}

#[derive(Default)]
pub struct StripFilter {
    sides: [Option<Side>; 2],
}

/// strip_filter.p_dashed / p_yellow on a decide4 result (candidate + its confidence).
pub fn p_of(candidate: &str, confidence: f64, first: &str) -> f64 {
    if candidate == first { confidence } else { 1. - confidence }
}

fn weighted_mean<'a>(it: impl Iterator<Item = &'a (i64, f64, f64, f64)>) -> f64 {
    let (mut total, mut sum) = (0., 0.);
    let mut xs = Vec::new();
    for x in it {
        xs.push(x);
    }
    for x in &xs {
        total += x.3;
    }
    for x in &xs {
        sum += x.2 * x.3;
    }
    if total > 0. { sum / total } else { 0.5 }
}

impl StripFilter {
    pub fn reset(&mut self) {
        self.sides = [None, None];
    }

    pub fn update(&mut self, side: usize, result: SideResult, p: Option<f64>, stamp: i64, odometer: f64, py: Option<f64>) -> SideResult {
        let s = self.sides[side].get_or_insert_with(Side::default);
        if result.tracking_reset || RESET_REASONS.contains(&result.reason)
            || s.last_stamp.is_some_and(|last| !(0 < stamp - last && stamp - last <= MAX_GAP_NS)) {
            s.clear();
        }
        s.last_stamp = Some(stamp);
        if RESET_REASONS.contains(&result.reason) {
            return result;
        }
        let out = Self::type_window(s, result, p, stamp, odometer);
        Self::color_window(s, out, py, stamp, odometer)
    }

    fn type_window(s: &mut Side, result: SideResult, p: Option<f64>, stamp: i64, odometer: f64) -> SideResult {
        if let Some(p) = p {
            s.items.push_back((stamp, odometer, p, (p - 0.5).abs() * 2.));
        }
        let span = LONG_M.max(SHORT_M);
        while s.items.front().is_some_and(|f| odometer - f.1 > span || stamp - f.0 > WINDOW_NS) {
            s.items.pop_front();
        }
        if p.is_none() {
            return result;
        }
        let mean = |items: &VecDeque<(i64, f64, f64, f64)>, metres: f64| weighted_mean(items.iter().filter(|x| odometer - x.1 <= metres));
        let (short, long) = (mean(&s.items, SHORT_M), mean(&s.items, LONG_M));
        let (kind, m) = if s.state == Some("dashed") {
            if short < 0.5 { (Some("solid"), short) } else { (Some("dashed"), short) }
        } else if long > 0.5 {
            (Some("dashed"), long)
        } else if short < 0.5 {
            (Some("solid"), short)
        } else if s.state == Some("solid") {
            (Some("solid"), long)
        } else {
            (None, 0.5)
        };
        if kind == Some("solid") && s.state != Some("solid") {
            while s.items.front().is_some_and(|f| odometer - f.1 > SHORT_M) {
                s.items.pop_front();
            }
        }
        s.state = kind;
        let kind = kind.unwrap_or("unknown");
        let confidence = (m - 0.5).abs() * 2.;
        // _typed with the plain type latch.
        let mut confident = (kind == "dashed" || kind == "solid") && confidence >= THRESHOLD;
        let (mut kind, mut confidence) = (kind, confidence);
        if confident {
            s.latched_type = Some((kind, confidence));
        } else if let Some((k, c)) = s.latched_type {
            (kind, confidence, confident) = (k, c, true);
        }
        let mut out = result;
        out.kind = if confident { kind } else { "unknown" };
        out.type_confidence = if confident { confidence } else { 0. };
        out.type_mono = stamp;
        out.dashed_observed = out.kind == "dashed";
        out.dashed_mono = stamp;
        out
    }

    fn color_window(s: &mut Side, result: SideResult, py: Option<f64>, stamp: i64, odometer: f64) -> SideResult {
        if let Some(py) = py {
            s.colors.push_back((stamp, odometer, py, (py - 0.5).abs() * 2.));
        }
        while s.colors.front().is_some_and(|f| odometer - f.1 > COLOR_M || stamp - f.0 > WINDOW_NS) {
            s.colors.pop_front();
        }
        if py.is_none() {
            return result;
        }
        let m = weighted_mean(s.colors.iter());
        let mut confidence = if m >= 1. - m { m } else { 1. - m };
        let mut color = if m > 0.5 { "yellow" } else if m < 0.5 { "white" } else { "unknown" };
        let mut confident = color != "unknown" && confidence >= THRESHOLD;
        // _latch for the colour head.
        let mut held = false;
        if confident && s.latched_color.is_some_and(|l| l.0 != color) {
            s.flips += 1;
            if s.flips < COLOR_LATCH_FRAMES {
                let (k, c) = s.latched_color.unwrap();
                (color, confidence, confident) = (k, c, true);
                held = true;
            }
        }
        if !held {
            s.flips = 0;
            if confident {
                s.latched_color = Some((color, confidence));
            } else if let Some((k, c)) = s.latched_color {
                (color, confidence, confident) = (k, c, true);
            }
        }
        let mut out = result;
        out.color = if confident { color } else { "unknown" };
        out.color_confidence = if confident { confidence } else { 0. };
        out.color_mono = stamp;
        out
    }
}
