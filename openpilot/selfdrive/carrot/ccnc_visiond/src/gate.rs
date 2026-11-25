//! When the wide side views are processed, and which side comes next.

use ccnc_vision_msg::{KindAsk, View};

/// The side views are only used for cars next to a standing or creeping car, not while reversing.
pub const WIDE_MAX_SPEED: f32 = 3.0;
/// vEgo of a standing car dips below zero (down to -0.0011 m/s in the 16 recorded routes, never below zero while
/// moving); reversing is faster than this.
pub const STANDSTILL_NOISE: f32 = 0.01;
/// carState comes at 100 Hz. Without a recent one the speed is unknown and nothing is processed.
pub const SPEED_TIMEOUT_NS: u64 = 500_000_000;
/// CcncMap comes at 20 Hz. Without a recent one, conditional full-frame raw inference stays off.
pub const MAP_TIMEOUT_NS: u64 = 1_000_000_000;

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
const FRONT_BUDGET_NS: u64 = 800_000_000;

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
    front_vision: Option<(bool, u64)>,
    next: Side,
    pending_raw: bool,
    was_open: bool,
    front_kind: Option<(KindAsk, u64)>,
    /// Last successful publish time and measured cost (including the minimum period) of each side.
    side_frames: [Option<(u64, u64)>; 2],
    raw_at: Option<u64>,
    front_turn: bool,
    front_cost: u64,
}

impl Gate {
    pub fn speed(&mut self, v_ego: f32, now_ns: u64) {
        self.speed = Some((v_ego, now_ns));
    }

    pub fn front_vision(&mut self, on: bool, now_ns: u64) {
        self.front_vision = Some((on, now_ns));
    }

    pub fn front_kind(&mut self, ask: KindAsk, now_ns: u64) {
        self.front_kind = Some((ask, now_ns));
    }

    /// Whether to take a road camera frame now, the last one's detection having ended at `last_ns` (`pace` is the
    /// rest after a detection). Only forward (or standing) at a known speed, while the map asks for a kind (a recent
    /// CcncMap).
    pub fn front_due(&self, now_ns: u64, last_ns: Option<u64>, pace: FrontPace) -> bool {
        let ask = self.front_kind.filter(|(_, at)| now_ns.saturating_sub(*at) <= MAP_TIMEOUT_NS).map_or(KindAsk::None, |(ask, _)| ask);
        let forward = self.speed.is_some_and(|(v_ego, at)| v_ego >= -STANDSTILL_NOISE && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS);
        let period = if self.open(now_ns) && ask == KindAsk::Confirmed { pace.standing_ns } else { pace.moving_ns };
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
        self.front_turn = self.next_target() == Target::Left;
    }

    fn want_raw(&self, now_ns: u64) -> bool {
        self.front_vision
            .is_some_and(|(on, at)| on && now_ns.saturating_sub(at) <= MAP_TIMEOUT_NS)
    }

    pub fn open(&self, now_ns: u64) -> bool {
        self.speed.is_some_and(|(v_ego, at)| (-STANDSTILL_NOISE..=WIDE_MAX_SPEED).contains(&v_ego) && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS)
    }

    /// `ready`: the rest of what processing needs (the side views' calibration) is there.
    pub fn phase(&mut self, now_ns: u64, ready: bool) -> Phase {
        let open = ready && self.open(now_ns);
        let phase = match (self.was_open, open) {
            (_, false) => {
                self.pending_raw = false;
                Phase::Closed
            }
            (false, true) => {
                self.next = Side::Left;
                self.side_frames = [None; 2];
                self.raw_at = None;
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
    fn open_only_at_low_known_speed() {
        let mut gate = Gate::default();
        assert!(!gate.open(0), "no speed yet");
        for (v_ego, open) in [(0.0, true), (3.0, true), (3.01, false), (25.0, false), (f32::NAN, false), (-0.1, false), (-0.0011, true), (-0.0101, false)] {
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
        gate.speed(5.0, 10);
        assert_eq!(gate.phase(10, true), Phase::Closed);
        gate.speed(0.0, 20);
        assert_eq!(gate.phase(20, true), Phase::Opened);
        assert_eq!(gate.phase(30, true), Phase::Open);
        assert_eq!(gate.phase(21 + SPEED_TIMEOUT_NS, true), Phase::Closed);
        gate.speed(0.0, 40 + SPEED_TIMEOUT_NS);
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

}
