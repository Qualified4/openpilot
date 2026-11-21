//! display.DisplayPolicy.confirm (two fresh identical results within 0.4 s confirm a head, 0.5 s expiry).

use crate::result::{is_reset, SideResult};

pub const MAX_AGE_NS: i64 = 500_000_000;
pub const MIN_CONFIDENCE: f64 = 0.85;
pub const REPEAT_WINDOW_NS: i64 = 400_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    pub value: &'static str,
    pub stamp: i64,
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    value: &'static str,
    stamp: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Confirmed {
    pub kind: Option<Head>,
    pub color: Option<Head>,
}

#[derive(Default)]
struct State {
    heads: [Option<Head>; 2],
    pending: [Option<Pending>; 2],
    last_now: i64,
}

#[derive(Default)]
pub struct DisplayPolicy {
    sides: [Option<State>; 2],
}

pub fn fresh(stamp: i64, now: i64) -> bool {
    0 <= now - stamp && now - stamp <= MAX_AGE_NS
}

impl DisplayPolicy {
    pub fn reset(&mut self) {
        self.sides = [None, None];
    }

    /// `result`: None for an empty dict (a stale packet).
    fn confirm_side(&mut self, side: usize, result: Option<&SideResult>, now: i64) -> Confirmed {
        if result.is_some_and(is_reset) {
            self.sides[side] = None;
        }
        let state = self.sides[side].get_or_insert_with(|| State { last_now: now, ..Default::default() });
        if now < state.last_now {
            state.heads = [None, None];
            state.pending = [None, None];
        }
        state.last_now = now;
        for head in 0..2 {
            let classes: [&str; 2] = if head == 0 { ["dashed", "solid"] } else { ["white", "yellow"] };
            let mut old = state.heads[head];
            if old.is_some_and(|o| !fresh(o.stamp, now)) {
                state.heads[head] = None;
                old = None;
            }
            if let Some(r) = result {
                let (value, confidence, stamp) =
                    if head == 0 { (r.kind, r.type_confidence, r.type_mono) } else { (r.color, r.color_confidence, r.color_mono) };
                let mut eligible = classes.contains(&value) && confidence.is_finite() && (MIN_CONFIDENCE..=1.).contains(&confidence)
                    && fresh(stamp, now);
                if head == 0 && value == "dashed" {
                    eligible &= r.dashed_observed && r.dashed_mono == stamp;
                }
                if eligible && old.is_none_or(|o| stamp > o.stamp) {
                    if old.is_some_and(|o| o.value == value) {
                        state.heads[head] = Some(Head { value, stamp, confidence });
                        state.pending[head] = None;
                    } else {
                        let pending = state.pending[head];
                        if pending.is_some_and(|p| p.value == value && p.stamp < stamp && stamp <= p.stamp + REPEAT_WINDOW_NS) {
                            state.heads[head] = Some(Head { value, stamp, confidence });
                            state.pending[head] = None;
                        } else if pending.is_none_or(|p| stamp != p.stamp) {
                            state.pending[head] = Some(Pending { value, stamp });
                        }
                    }
                }
            }
            if state.pending[head].is_some_and(|p| now - p.stamp > REPEAT_WINDOW_NS) {
                state.pending[head] = None;
            }
        }
        Confirmed { kind: state.heads[0], color: state.heads[1] }
    }

    pub fn confirm(&mut self, camera_ns: i64, raw: &[SideResult; 2], now: i64) -> [Confirmed; 2] {
        let valid = fresh(camera_ns, now);
        [0, 1].map(|side| self.confirm_side(side, valid.then_some(&raw[side]), now))
    }
}
