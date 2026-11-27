//! When the wide side views are processed, and which side comes next.

use ccnc_vision_msg::{KindAsk, View};

/// Standstill / creeping threshold (m/s): high-rate wide scan and standing front-recheck pace.
pub const STANDING_MAX_SPEED: f32 = 3.0;
/// Maximum speed (m/s) where conditional full-frame `Raw` (416x416) is inserted for 2-car splits.
/// Above this speed, `SideL`/`SideR` alone cover the diagonal blind spot without extra GPU load.
pub const RAW_MAX_SPEED: f32 = 5.0;
/// Upper speed (m/s, ~29 km/h) of the urban high-rate wide scan regime.
pub const LOW_SPEED_MAX: f32 = 8.0;
/// vEgo of a standing car dips below zero (down to -0.0011 m/s in the 16 recorded routes, never below zero while
/// moving); reversing is faster than this.
pub const STANDSTILL_NOISE: f32 = 0.01;
/// carState comes at 100 Hz. Without a recent one the speed is unknown and nothing is processed.
pub const SPEED_TIMEOUT_NS: u64 = 500_000_000;
/// CcncMap comes at 20 Hz. Without a recent one, conditional full-frame raw inference stays off.
pub const MAP_TIMEOUT_NS: u64 = 1_000_000_000;

/// Adaptive wide-camera cadence periods (inter-frame interval across alternating L/R views).
/// Kept conservative so Adreno 630v2 retains headroom for `dmonitoringmodeld` (20 Hz, 0 drops):
/// * Standstill / creep (`<= 3 m/s`) or active turn-signal lane change: 150 ms (~6.67 Hz total, ~3.33 Hz/side).
/// * Urban low speed (`3..=8 m/s`): 200 ms (5.0 Hz total, 2.5 Hz/side).
/// * Cruise (`> 8 m/s`, no blinker): 500 ms (2.0 Hz total, 1.0 Hz/side; <10% GPU).
pub const WIDE_HIGH_PERIOD_NS: u64 = 150_000_000;
pub const WIDE_URBAN_PERIOD_NS: u64 = 200_000_000;
pub const WIDE_CRUISE_PERIOD_NS: u64 = 500_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    #[default]
    Left,
    Right,
}

impl Side {
    #[cfg(test)]
    pub fn view(self) -> View {
        match self {
            Self::Left => View::SideL,
            Self::Right => View::SideR,
        }
    }
}

/// Which view to infer on the next camera frame: alternating left/right side views, plus an optional full-frame
/// raw view inserted once per L/R pair while either LF or RF has a camera-backed car on CcncMap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Left,
    Right,
    Raw,
}

impl Target {
    pub fn view(self) -> View {
        match self {
            Self::Left => View::SideL,
            Self::Right => View::SideR,
            Self::Raw => View::Full,
        }
    }
}

// Match the side deadlines in ccnc_map::vision. A Raw result extends each side from
// its own arrival, so it can give Left more time than Right.
const SIDE_HOLD_NS: u64 = 3_500_000_000;
const RAW_EXTEND_NS: u64 = 1_500_000_000;
const RAW_MAX_HOLD_NS: u64 = 4_500_000_000;
const FRAME_MARGIN_NS: u64 = 50_000_000;
/// Minimum front-inference budget before a measured `front_cost` is observed.
/// Realistic for MNN OpenCL FP16 (~74 ms measured on comma 3X); `front_cost.max(FRONT_BUDGET_NS)`
/// still expands automatically when running on CPU ORT fallback.
pub const FRONT_BUDGET_NS: u64 = 120_000_000;

/// How often the road camera is looked at, while a radar car ahead (FF, LF, RF) has no confirmed kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontPace {
    /// Above `WIDE_MAX_SPEED`, when nothing else runs: the rest after a road frame's detection, 500 ms by default
    /// (two answers 0.8 s apart confirm a kind, so a shorter rest would only spend the little cores).
    pub moving_ns: u64,
    /// Minimum rest between rechecks of a confirmed kind while the side views run. All road frames also
    /// need an opening between wide cycles whenever the wide stream is available.
    pub standing_ns: u64,
}

impl FrontPace {
    pub const DEFAULT: FrontPace = FrontPace { moving_ns: 500_000_000, standing_ns: 2_000_000_000 };
}

/// What the loop does with the camera now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Closed,
    /// Open again since the last check: frames that came while it was closed are still queued.
    Opened,
    Open,
}

#[derive(Default)]
pub struct Gate {
    speed: Option<(f32, u64)>,
    blinker: Option<(bool, u64)>,
    front_vision: Option<(bool, u64)>,
    next: Side,
    pending_raw: bool,
    was_open: bool,
    front_kind: Option<(KindAsk, u64)>,
    /// Last successful publish time and measured cost of each side.
    side_frames: [Option<(u64, u64)>; 2],
    raw_at: Option<u64>,
    last_wide_ns: Option<u64>,
    front_turn: bool,
    front_cost: u64,
    /// When true (CPU ORT fallback), degrade wide inference back to low-speed (`<= STANDING_MAX_SPEED`).
    low_speed_only: bool,
}

impl Gate {
    pub fn with_low_speed_only(low_speed_only: bool) -> Self {
        Self { low_speed_only, ..Self::default() }
    }

    pub fn speed(&mut self, v_ego: f32, now_ns: u64) {
        self.speed = Some((v_ego, now_ns));
    }

    pub fn blinker(&mut self, left_blinker: bool, right_blinker: bool, now_ns: u64) {
        self.blinker = Some((left_blinker || right_blinker, now_ns));
    }

    pub fn car_state(&mut self, v_ego: f32, left_blinker: bool, right_blinker: bool, now_ns: u64) {
        self.speed(v_ego, now_ns);
        self.blinker(left_blinker, right_blinker, now_ns);
    }

    pub fn front_vision(&mut self, on: bool, now_ns: u64) {
        self.front_vision = Some((on, now_ns));
    }

    pub fn front_kind(&mut self, ask: KindAsk, now_ns: u64) {
        self.front_kind = Some((ask, now_ns));
    }

    fn is_low_speed(&self, now_ns: u64) -> bool {
        self.speed.is_some_and(|(v_ego, at)| {
            (-STANDSTILL_NOISE..=STANDING_MAX_SPEED).contains(&v_ego) && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS
        })
    }

    fn blinker_active(&self, now_ns: u64) -> bool {
        self.blinker.is_some_and(|(on, at)| on && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS)
    }

    /// Whether to take a road camera frame now, the last one's detection having ended at `last_ns` (`pace` is the
    /// rest after a detection). Only forward (or standing) at a known speed, while the map asks for a kind (a recent
    /// CcncMap).
    pub fn front_due(&self, now_ns: u64, last_ns: Option<u64>, pace: FrontPace) -> bool {
        let ask = self.front_kind.filter(|(_, at)| now_ns.saturating_sub(*at) <= MAP_TIMEOUT_NS).map_or(KindAsk::None, |(ask, _)| ask);
        let forward = self.speed.is_some_and(|(v_ego, at)| v_ego >= -STANDSTILL_NOISE && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS);
        let period = if self.is_low_speed(now_ns) && ask == KindAsk::Confirmed { pace.standing_ns } else { pace.moving_ns };
        ask != KindAsk::None && forward && last_ns.is_none_or(|last| now_ns.saturating_sub(last) >= period)
    }

    /// Take at most one road frame after L/R or L/R/Raw, and only if both sides can be refreshed before
    /// the map's deadline at the observed processing costs. Under load or with a slow Raw turn, skip the road frame.
    pub fn front_fits(&self, now_ns: u64) -> bool {
        let (Some((left, left_cost)), Some((right, right_cost))) = (self.side_frames[0], self.side_frames[1]) else {
            return false;
        };
        let deadline = |side| {
            let hold = side + SIDE_HOLD_NS;
            self.raw_at.filter(|&raw| raw > side).map_or(hold, |raw| {
                hold.max(raw + RAW_EXTEND_NS).min(side + RAW_MAX_HOLD_NS)
            })
        };
        let front = self.front_cost.max(FRONT_BUDGET_NS) + FRAME_MARGIN_NS;
        self.front_turn
            && now_ns + front + left_cost + FRAME_MARGIN_NS <= deadline(left)
            && now_ns + front + left_cost + right_cost + 2 * FRAME_MARGIN_NS <= deadline(right)
    }

    pub fn front_finished(&mut self, cost_ns: u64) {
        self.front_cost = cost_ns;
        self.front_turn = false;
    }

    pub fn wide_finished(&mut self, target: Target, now_ns: u64, cost_ns: u64) {
        match target {
            Target::Left => self.side_frames[0] = Some((now_ns, cost_ns)),
            Target::Right => self.side_frames[1] = Some((now_ns, cost_ns)),
            Target::Raw => self.raw_at = Some(now_ns),
        }
        self.last_wide_ns = Some(now_ns);
        self.front_turn = self.next_target() == Target::Left;
    }

    fn want_raw(&self, now_ns: u64) -> bool {
        self.front_vision
            .is_some_and(|(on, at)| on && now_ns.saturating_sub(at) <= MAP_TIMEOUT_NS)
            && self.speed.is_none_or(|(v_ego, _)| v_ego <= RAW_MAX_SPEED)
    }

    /// Wide side views run at any forward or standing speed on GPU (not reversing, fresh `carState`).
    /// On CPU ORT fallback (`low_speed_only == true`), automatically degrades to `<= STANDING_MAX_SPEED` (3.0 m/s)
    /// to prevent saturating the little cores at driving speeds.
    pub fn open(&self, now_ns: u64) -> bool {
        if self.low_speed_only {
            return self.is_low_speed(now_ns);
        }
        self.speed.is_some_and(|(v_ego, at)| {
            v_ego.is_finite() && v_ego >= -STANDSTILL_NOISE && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS
        })
    }

    /// Adaptive wide-camera target period (ns between successive wide inferences), or `None` when closed.
    pub fn wide_period_ns(&self, now_ns: u64) -> Option<u64> {
        if !self.open(now_ns) {
            return None;
        }
        let (v_ego, _) = self.speed?;
        if v_ego <= STANDING_MAX_SPEED || self.blinker_active(now_ns) {
            Some(WIDE_HIGH_PERIOD_NS)
        } else if v_ego <= LOW_SPEED_MAX {
            Some(WIDE_URBAN_PERIOD_NS)
        } else {
            Some(WIDE_CRUISE_PERIOD_NS)
        }
    }

    /// Remaining wait (ns) before the next wide inference is due, respecting both adaptive cadence and `min_period_ns`.
    pub fn wide_wait_ns(&self, now_ns: u64, min_period_ns: u64) -> u64 {
        let Some(period) = self.wide_period_ns(now_ns) else {
            return 0;
        };
        let effective = period.max(min_period_ns);
        let Some(last) = self.last_wide_ns else {
            return 0;
        };
        effective.saturating_sub(now_ns.saturating_sub(last))
    }

    /// Whether a wide inference can start now.
    pub fn wide_due(&self, now_ns: u64, min_period_ns: u64) -> bool {
        self.open(now_ns) && self.wide_wait_ns(now_ns, min_period_ns) == 0
    }

    /// `ready`: the rest of what processing needs (the side views' calibration) is there.
    pub fn phase(&mut self, now_ns: u64, ready: bool) -> Phase {
        let open = ready && self.open(now_ns);
        let phase = match (self.was_open, open) {
            (_, false) => {
                self.pending_raw = false;
                self.last_wide_ns = None;
                Phase::Closed
            }
            (false, true) => {
                self.next = Side::Left;
                self.side_frames = [None; 2];
                self.raw_at = None;
                self.last_wide_ns = None;
                self.front_turn = false;
                Phase::Opened
            }
            (true, true) => Phase::Open,
        };
        self.was_open = open;
        phase
    }

    /// The view the next frame goes to, without taking it.
    pub fn next_target(&self) -> Target {
        if self.pending_raw {
            Target::Raw
        } else {
            match self.next {
                Side::Left => Target::Left,
                Side::Right => Target::Right,
            }
        }
    }

    /// The side to process now. The sides take turns, one camera frame each.
    pub fn take_side(&mut self) -> Side {
        let side = self.next;
        self.next = match side {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        };
        side
    }

    /// The target to process now: L, R alternating, with Raw inserted after R while LF or RF has a camera car.
    pub fn take_target(&mut self, now_ns: u64) -> Target {
        self.front_turn = false;
        if self.pending_raw {
            self.pending_raw = false;
            return Target::Raw;
        }
        let side = self.take_side();
        if side == Side::Right && self.want_raw(now_ns) {
            self.pending_raw = true;
        }
        match side {
            Side::Left => Target::Left,
            Side::Right => Target::Right,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_at_any_forward_or_standstill_speed() {
        let mut gate = Gate::default();
        assert!(!gate.open(0), "no speed yet");
        for (v_ego, open) in [
            (0.0, true),
            (3.0, true),
            (3.01, true),
            (5.0, true),
            (15.0, true),
            (25.0, true),
            (f32::NAN, false),
            (-0.1, false),
            (-0.0011, true),
            (-0.0101, false),
        ] {
            gate.speed(v_ego, 1_000);
            assert_eq!(gate.open(1_000), open, "{v_ego}");
        }
    }

    #[test]
    fn closes_when_the_speed_is_old() {
        let mut gate = Gate::default();
        gate.speed(0.0, 1_000);
        assert!(gate.open(1_000 + SPEED_TIMEOUT_NS));
        assert!(!gate.open(1_001 + SPEED_TIMEOUT_NS));
        gate.speed(0.0, 2_000 + SPEED_TIMEOUT_NS);
        assert!(gate.open(2_000 + SPEED_TIMEOUT_NS));
    }

    #[test]
    fn reports_each_opening_once() {
        let mut gate = Gate::default();
        assert_eq!(gate.phase(0, true), Phase::Closed);
        gate.speed(-0.5, 10);
        assert_eq!(gate.phase(10, true), Phase::Closed);
        gate.speed(0.0, 20);
        assert_eq!(gate.phase(20, true), Phase::Opened);
        assert_eq!(gate.phase(30, true), Phase::Open);
        assert_eq!(gate.phase(21 + SPEED_TIMEOUT_NS, true), Phase::Closed);
        gate.speed(15.0, 40 + SPEED_TIMEOUT_NS);
        assert_eq!(gate.phase(40 + SPEED_TIMEOUT_NS, true), Phase::Opened);
        // Not ready (no calibration) closes it the same way, and it opens again once.
        assert_eq!(gate.phase(41 + SPEED_TIMEOUT_NS, false), Phase::Closed);
        assert_eq!(gate.phase(42 + SPEED_TIMEOUT_NS, true), Phase::Opened);
        assert_eq!(gate.phase(43 + SPEED_TIMEOUT_NS, true), Phase::Open);
    }

    #[test]
    fn front_runs_while_wanted_at_its_pace() {
        let pace = FrontPace { moving_ns: 500, standing_ns: 2_000 };
        let mut gate = Gate::default();
        assert!(!gate.front_due(0, None, pace), "nothing known");
        gate.speed(20.0, 0);
        assert!(!gate.front_due(0, None, pace), "the map does not ask");
        gate.front_kind(KindAsk::Unconfirmed, 0);
        assert!(gate.front_due(0, None, pace), "any forward speed");
        assert!(!gate.front_due(499, Some(0), pace));
        assert!(gate.front_due(500, Some(0), pace));
        gate.speed(0.0, 500);
        gate.front_kind(KindAsk::Confirmed, 500);
        assert!(!gate.front_due(1_000, Some(500), pace), "standing recheck: the side views' pace");
        assert!(gate.front_due(2_500, Some(500), pace));
        gate.front_kind(KindAsk::Confirmed, 2_500);
        gate.speed(20.0, 2_500);
        assert!(gate.front_due(3_000, Some(2_500), pace), "a moving recheck keeps the moving pace");
        gate.speed(-0.5, 2_500);
        assert!(!gate.front_due(5_000, Some(500), pace), "reversing");
        gate.speed(5.0, 10_000);
        gate.front_kind(KindAsk::None, 10_000);
        assert!(!gate.front_due(10_000, None, pace), "every car ahead has its kind");
        gate.front_kind(KindAsk::Unconfirmed, 10_000);
        assert!(!gate.front_due(11_001 + MAP_TIMEOUT_NS, None, pace), "the map went quiet");
        assert!(!gate.front_due(10_001 + SPEED_TIMEOUT_NS, None, pace), "the speed is old");
    }

    #[test]
    fn a_standing_car_ahead_without_a_kind_takes_turns_with_the_side_views() {
        let pace = FrontPace { moving_ns: 500, standing_ns: 2_000 };
        let mut gate = Gate::default();
        gate.speed(0.0, 0);
        gate.front_kind(KindAsk::Unconfirmed, 0);
        assert!(!gate.front_due(499, Some(0), pace));
        assert!(gate.front_due(500, Some(0), pace), "after one side frame, before the next one");
        gate.front_kind(KindAsk::Confirmed, 500);
        assert!(!gate.front_due(1_999, Some(0), pace), "confirmed: back to the standing pace");
        gate.front_kind(KindAsk::None, 2_000);
        assert!(!gate.front_due(2_000, Some(0), pace), "nothing asked: no road frame");
    }

    #[test]
    fn sides_take_turns() {
        let mut gate = Gate::default();
        let sides: Vec<_> = (0..4).map(|_| gate.take_side()).collect();
        assert_eq!(sides, [Side::Left, Side::Right, Side::Left, Side::Right]);
        assert_eq!((Side::Left.view(), Side::Right.view()), (View::SideL, View::SideR));
    }

    #[test]
    fn raw_inserted_only_while_front_vision_is_on_and_fresh() {
        let mut gate = Gate::default();
        // Without CcncMap, never runs Raw.
        let seq: Vec<_> = (0..4).map(|_| gate.take_target(1_000)).collect();
        assert_eq!(seq, [Target::Left, Target::Right, Target::Left, Target::Right]);
        // When front vision turns on, Raw follows each L, R pair.
        gate.front_vision(true, 2_000);
        let seq: Vec<_> = (0..6).map(|_| gate.take_target(2_500)).collect();
        assert_eq!(seq, [Target::Left, Target::Right, Target::Raw, Target::Left, Target::Right, Target::Raw]);
        assert_eq!(Target::Raw.view(), View::Full);
        // Stale front vision (> MAP_TIMEOUT_NS) stops Raw.
        let seq: Vec<_> = (0..4).map(|_| gate.take_target(2_000 + MAP_TIMEOUT_NS + 1)).collect();
        assert_eq!(seq, [Target::Left, Target::Right, Target::Left, Target::Right]);
        // Closing the gate clears a pending Raw so the next stop starts on a side view.
        gate.front_vision(true, 10_000);
        assert_eq!(gate.take_target(10_000), Target::Left);
        assert_eq!(gate.take_target(10_000), Target::Right);
        assert_eq!(gate.next_target(), Target::Raw);
        assert_eq!(gate.phase(10_000, false), Phase::Closed);
        assert_eq!(gate.next_target(), Target::Left);
    }
    #[test]
    fn a_standing_front_frame_waits_for_the_wide_round_and_both_deadlines() {
        fn wide(gate: &mut Gate, target: Target, end_ms: u64, cost_ms: u64) {
            let ns = end_ms * 1_000_000;
            assert_eq!(gate.take_target(ns), target);
            gate.wide_finished(target, ns, cost_ms * 1_000_000);
        }
        let mut gate = Gate::default();
        wide(&mut gate, Target::Left, 1100, 1100);
        assert!(!gate.front_fits(1_100_000_000), "wide right goes first");
        wide(&mut gate, Target::Right, 2200, 1100);
        assert!(gate.front_fits(2_200_000_000), "two side views and one front fit in 3.5 s");
        gate.front_finished(790_000_000);
        assert!(!gate.front_fits(2_200_000_000), "only one front per round");

        wide(&mut gate, Target::Left, 4100, 1100);
        gate.front_vision(true, 5_200_000_000);
        wide(&mut gate, Target::Right, 5200, 1100);
        assert!(!gate.front_fits(5_200_000_000), "the pending Raw must go first");
        wide(&mut gate, Target::Raw, 6300, 1100);
        assert!(!gate.front_fits(6_300_000_000), "Raw alone does not extend Right's deadline enough for another front");

        wide(&mut gate, Target::Left, 6900, 600);
        gate.front_vision(true, 7_500_000_000);
        wide(&mut gate, Target::Right, 7500, 600);
        wide(&mut gate, Target::Raw, 8100, 600);
        assert!(gate.front_fits(8_100_000_000), "even with Raw, the faster wide cycle leaves a road turn");
    }

    #[test]
    fn gpu_front_budget_120ms_fits_between_wide_cycles() {
        assert_eq!(FRONT_BUDGET_NS, 120_000_000);
        let mut gate = Gate::default();
        // Before any measured front_cost, a tight 3.5 s deadline where 800 ms would fail still fits under 120 ms.
        assert_eq!(gate.take_target(1_000_000_000), Target::Left);
        gate.wide_finished(Target::Left, 1_000_000_000, 1_500_000_000);
        assert_eq!(gate.take_target(2_500_000_000), Target::Right);
        gate.wide_finished(Target::Right, 2_500_000_000, 1_500_000_000);
        // Remaining until left deadline (4500 ms) is 2000 ms; left_cost (1500) + margin (50) + front (120+50) = 1720 <= 2000.
        // With old 800 ms budget this would be 2400 > 2000 (rejected).
        assert!(gate.front_fits(2_500_000_000));
        // Once a slow CPU ORT front_cost (790 ms) is observed, front_fits expands automatically and rejects.
        gate.front_finished(790_000_000);
        assert_eq!(gate.take_target(2_600_000_000), Target::Left);
        gate.wide_finished(Target::Left, 2_600_000_000, 1_500_000_000);
        assert_eq!(gate.take_target(4_100_000_000), Target::Right);
        gate.wide_finished(Target::Right, 4_100_000_000, 1_500_000_000);
        assert!(!gate.front_fits(4_100_000_000), "CPU ORT fallback cost automatically respected");
    }

    #[test]
    fn adaptive_wide_cadence_by_speed_and_blinker() {
        let mut gate = Gate::default();
        let t0 = 1_000_000_000u64;

        // 0 m/s (standstill): 150 ms (~6.67 Hz)
        gate.car_state(0.0, false, false, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_HIGH_PERIOD_NS));

        // 5 m/s (urban low speed): 200 ms (5.0 Hz)
        gate.car_state(5.0, false, false, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_URBAN_PERIOD_NS));

        // 15 m/s (cruise, no blinker): 500 ms (2.0 Hz)
        gate.car_state(15.0, false, false, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_CRUISE_PERIOD_NS));

        // 25 m/s (highway, no blinker): 500 ms (2.0 Hz)
        gate.car_state(25.0, false, false, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_CRUISE_PERIOD_NS));

        // 15 m/s + left blinker (lane change): boosts to 150 ms (~6.67 Hz)
        gate.car_state(15.0, true, false, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_HIGH_PERIOD_NS));

        // 25 m/s + right blinker (lane change): boosts to 150 ms (~6.67 Hz)
        gate.car_state(25.0, false, true, t0);
        assert_eq!(gate.wide_period_ns(t0), Some(WIDE_HIGH_PERIOD_NS));

        // Reversing: closed
        gate.car_state(-0.5, true, false, t0);
        assert_eq!(gate.wide_period_ns(t0), None);

        // Verify wide_due / wide_wait_ns pacing at 25 m/s cruise (500 ms) vs blinker (150 ms).
        gate.car_state(25.0, false, false, t0);
        assert!(gate.wide_due(t0, 0), "first frame runs immediately");
        gate.wide_finished(Target::Left, t0, 80_000_000);
        assert!(!gate.wide_due(t0 + 200_000_000, 0));
        assert_eq!(gate.wide_wait_ns(t0 + 200_000_000, 0), 300_000_000);
        assert!(gate.wide_due(t0 + 500_000_000, 0));

        // Turning on blinker at t0 + 150 ms immediately makes wide_due true!
        gate.car_state(25.0, true, false, t0 + 150_000_000);
        assert!(gate.wide_due(t0 + 150_000_000, 0));

        // Explicit min_period_ns floor still overrides adaptive period if larger.
        assert!(!gate.wide_due(t0 + 150_000_000, 300_000_000));
        assert_eq!(gate.wide_wait_ns(t0 + 150_000_000, 300_000_000), 150_000_000);

        // Conditional Raw is enabled <= 5.0 m/s and suppressed > 5.0 m/s.
        let mut raw_gate = Gate::default();
        raw_gate.front_vision(true, t0);
        raw_gate.car_state(5.0, false, false, t0);
        let seq_low: Vec<_> = (0..3).map(|_| raw_gate.take_target(t0)).collect();
        assert_eq!(seq_low, [Target::Left, Target::Right, Target::Raw]);

        raw_gate.car_state(15.0, false, false, t0);
        let seq_cruise: Vec<_> = (0..4).map(|_| raw_gate.take_target(t0)).collect();
        assert_eq!(seq_cruise, [Target::Left, Target::Right, Target::Left, Target::Right]);
    }

    #[test]
    fn cpu_fallback_degrades_wide_gate_to_low_speed_only() {
        let mut gate = Gate::with_low_speed_only(true);
        let t0 = 1_000_000_000u64;
        for (v_ego, blinker, expect_open) in [
            (0.0, false, true),
            (3.0, false, true),
            (-0.0011, false, true),
            (-0.1, false, false),
            (3.01, false, false),
            (5.0, true, false),
            (15.0, false, false),
            (25.0, true, false),
        ] {
            gate.car_state(v_ego, blinker, false, t0);
            assert_eq!(gate.open(t0), expect_open, "v_ego={v_ego} blinker={blinker}");
            assert_eq!(gate.wide_period_ns(t0).is_some(), expect_open);
            assert_eq!(gate.wide_due(t0, 0), expect_open);
        }
        // Road camera front_due still runs at driving speeds for vehicle-kind classification.
        gate.car_state(20.0, false, false, t0);
        gate.front_kind(KindAsk::Unconfirmed, t0);
        assert!(!gate.open(t0));
        assert!(gate.front_due(t0, None, FrontPace::DEFAULT));
    }
}
