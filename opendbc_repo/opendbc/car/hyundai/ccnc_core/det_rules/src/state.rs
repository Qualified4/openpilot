//! All rule state in one struct, and the per-frame entry points (port of step5_lrrr.run_scene's frame body).
use crate::rules::{swap_decision, Act, ClassConfirm, Reason, RuleE, SplitGate, SwapCfg, Timing, VehicleClass};
use crate::wide::{counts_as_car, display_filters, pick_side, rear_distance, split_side, Cand, Filters};
use crate::{Lane, Track, JITTER_S};
use std::collections::HashMap;

/// Current radar slot of 0x162 the camera car is compared with (LF/RF/LR/RR _DETECT and _DETECT_DISTANCE).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlotState {
    pub detect: bool,
    pub dist: f64,
}

/// Everything the wide rules need besides the candidates of one frame.
#[derive(Clone, Copy, Debug)]
pub struct WideFrame<'a> {
    pub t_ns: u64,               // frame capture time (timestampSof)
    pub v_ego: f64,              // ego speed at the frame time
    pub start_guard_ok: bool,    // the drive has exceeded 5.5 m/s once (`StartGuard`)
    pub tracks: &'a [Track],     // radar points at the frame time
    pub lprob: Option<&'a [f64]>, // model lane line probabilities (4) at the frame time
    pub slots: [[SlotState; 2]; 2], // [left, right][front, rear]
}

/// Parking-lot start guard: rule E stays off until the drive has exceeded 5.5 m/s once.
#[derive(Clone, Copy, Debug, Default)]
pub struct StartGuard {
    pub v_max: f64,
}
impl StartGuard {
    pub const V_MIN: f64 = 5.5;
    pub fn observe(&mut self, v_ego: f64) {
        self.v_max = self.v_max.max(v_ego);
    }
    pub fn ok(&self) -> bool {
        self.v_max >= StartGuard::V_MIN
    }
}

/// One slot (LF/RF = front, LR/RR = rear) of one side after a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotOutput {
    pub cand: Option<usize>,   // index into the frame's candidates
    pub display_on: bool,      // rule E on (which includes passing the display filters)
    pub confirm_run: u32,      // hits counted by rule E
    pub action: Option<Act>,   // against the current radar slot; None = camera not shown
    pub filters: Filters,      // display filters that veto the candidate
    pub reason: Reason,        // why rule E is on or off
    pub rear_dist: Option<f64>, // rear slot: metres behind the front bumper (0 = beside)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideOutput {
    pub front: SlotOutput,
    pub rear: SlotOutput,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WideOutput {
    pub t: f64,                // frame time in s
    pub sides: [SideOutput; 2], // left, right
}

#[derive(Clone, Debug)]
struct SideRules {
    front: RuleE,
    rear: RuleE,
    gate: SplitGate,
    /// The proposed rear car while the split proposal stays unbroken (`Timing::rear_span_s`).
    pair: RuleE,
}

/// The rule state: rule E of the four side slots, the split gate of each side and R1 of every front car.
#[derive(Clone, Debug)]
pub struct DetRules {
    pub timing: Timing,
    pub swap: SwapCfg,
    sides: [SideRules; 2],
    classes: HashMap<i64, ClassConfirm>,
}
impl Default for DetRules {
    fn default() -> DetRules {
        DetRules::new(Timing::default())
    }
}
impl DetRules {
    pub fn new(timing: Timing) -> DetRules {
        let side = || SideRules { front: timing.rule_e(), rear: timing.rule_e(), gate: timing.split_gate(), pair: timing.rule_e() };
        DetRules { timing, swap: SwapCfg { min_run: timing.n_on, ..SwapCfg::default() }, sides: [side(), side()], classes: HashMap::new() }
    }

    /// One wide frame. cands: `WideGeometry::candidates` of all three views of that frame.
    pub fn wide(&mut self, f: &WideFrame, cands: &[Cand]) -> WideOutput {
        let t = f.t_ns as f64 * 1e-9;
        let refs: Vec<&Cand> = cands.iter().collect();
        let index = |c: Option<&Cand>| c.map(|c| refs.iter().position(|r| std::ptr::eq(*r, c)).unwrap_or(0));
        let shown = |c: &Cand| display_filters(c, f.v_ego, f.tracks, f.lprob);
        let mut out = [None, None];
        for (i, side) in [Lane::Left, Lane::Right].into_iter().enumerate() {
            let st = &mut self.sides[i];
            let single = pick_side(&refs, side);
            let split = split_side(&refs, side, |c| counts_as_car(c, f.v_ego, f.tracks, f.lprob));
            let (front, rear) = st.gate.apply(split, single, f.v_ego, f.start_guard_ok, |c| shown(c).is_empty(), t);
            // rear_span_s: the rear car of the split proposal while the gate's count runs (it resets with the gate)
            let pair_run = self.timing.rear_span_s.and_then(|span| {
                let proposed = split.1.filter(|_| split.0.is_some() && st.gate.rule.run > 0);
                let ok = proposed.map_or(false, |c| shown(c).is_empty());
                st.pair.step(proposed, f.v_ego, f.start_guard_ok, ok, t);
                let spanned = st.pair.first_hit().is_some_and(|t0| t - t0 >= span - JITTER_S);
                (rear.is_some() && st.pair.run >= self.timing.n_on && spanned).then_some(st.pair.run)
            });
            let slot = |rule: &mut RuleE, c: Option<&Cand>, radar: SlotState, is_rear: bool, also: Option<u32>| {
                let filters = c.map_or(Filters::default(), shown);
                let mut display_on = rule.step(c, f.v_ego, f.start_guard_ok, filters.is_empty(), t);
                let (mut run, mut reason) = (rule.run, rule.reason);
                if let Some(pair_run) = also.filter(|_| !display_on) {
                    (display_on, run, reason) = (true, pair_run, Reason::On);
                }
                let dist = c.map(|c| (if is_rear { rear_distance(c) } else { c.x }, c.cut));
                SlotOutput {
                    cand: index(c),
                    display_on,
                    confirm_run: run,
                    action: swap_decision(display_on, dist, radar.detect, radar.dist, run, &self.swap),
                    filters,
                    reason,
                    rear_dist: c.filter(|_| is_rear).map(rear_distance),
                }
            };
            let front = slot(&mut st.front, front, f.slots[i][0], false, None);
            let rear = slot(&mut st.rear, rear, f.slots[i][1], true, pair_run);
            out[i] = Some(SideOutput { front, rear });
        }
        WideOutput { t, sides: out.map(|o| o.expect("both sides are filled above")) }
    }

    /// One classified front view of the car `id` (radar track or SCC episode id). Returns (class to show, confirmed).
    /// day: usable evidence (not night); cut: `FrontBox::cut`.
    #[allow(clippy::too_many_arguments)]
    pub fn front_class(&mut self, id: i64, t: f64, dist: f64, cx: f64, cls: VehicleClass, day: bool, cut: bool) -> (VehicleClass, bool) {
        let timing = self.timing;
        let k = self.classes.entry(id).or_insert_with(|| timing.class_confirm());
        let shown = k.observe(t, dist, cx, cls, day, cut);
        (shown, k.confirmed)
    }
    /// Forget a car whose id is gone (R1: a new id starts from the default).
    pub fn forget_front(&mut self, id: i64) {
        self.classes.remove(&id);
    }
}
