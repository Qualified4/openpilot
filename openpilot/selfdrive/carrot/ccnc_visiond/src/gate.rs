//! When the wide side views are processed, and which side comes next.

use ccnc_vision_msg::View;

/// The side views are only used for cars next to a standing or creeping car, not while reversing.
pub const WIDE_MAX_SPEED: f32 = 3.0;
/// vEgo of a standing car dips below zero (down to -0.0011 m/s in the 16 recorded routes, never below zero while
/// moving); reversing is faster than this.
pub const STANDSTILL_NOISE: f32 = 0.01;
/// carState comes at 100 Hz. Without a recent one the speed is unknown and nothing is processed.
pub const SPEED_TIMEOUT_NS: u64 = 500_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    #[default]
    Left,
    Right,
}

impl Side {
    pub fn view(self) -> View {
        match self {
            Self::Left => View::SideL,
            Self::Right => View::SideR,
        }
    }
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
    next: Side,
    was_open: bool,
}

impl Gate {
    pub fn speed(&mut self, v_ego: f32, now_ns: u64) {
        self.speed = Some((v_ego, now_ns));
    }

    pub fn open(&self, now_ns: u64) -> bool {
        self.speed.is_some_and(|(v_ego, at)| (-STANDSTILL_NOISE..=WIDE_MAX_SPEED).contains(&v_ego) && now_ns.saturating_sub(at) <= SPEED_TIMEOUT_NS)
    }

    /// `ready`: the rest of what processing needs (the side views' calibration) is there.
    pub fn phase(&mut self, now_ns: u64, ready: bool) -> Phase {
        let open = ready && self.open(now_ns);
        let phase = match (self.was_open, open) {
            (_, false) => Phase::Closed,
            (false, true) => Phase::Opened,
            (true, true) => Phase::Open,
        };
        self.was_open = open;
        phase
    }

    /// The side the next frame goes to, without taking it.
    pub fn next_side(&self) -> Side {
        self.next
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
    fn sides_take_turns() {
        let mut gate = Gate::default();
        let sides: Vec<_> = (0..4).map(|_| gate.take_side()).collect();
        assert_eq!(sides, [Side::Left, Side::Right, Side::Left, Side::Right]);
        assert_eq!((Side::Left.view(), Side::Right.view()), (View::SideL, View::SideR));
    }
}
