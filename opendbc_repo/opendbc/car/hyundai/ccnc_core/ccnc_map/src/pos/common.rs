//! The common part of the redesigned front-slot display (DESIGN_KO.md 6.1-6.4 and 11): a per-vehicle lateral
//! estimate, lane coordinates and the FF/LF/RF assignment. Displays A and B draw from what this produces.
//!
//! Everything here works on plain values so that the Python reference (`hyundaicanfd_ccnc_pos_display.py`) can
//! repeat it operation for operation; keep the two in step (`test_ccnc_pos_display.py` compares them).
//!
//! The vehicle id (`vid`) is the radar map's object key: the map already joins a vehicle's radar points, keeps one
//! key across radar id changes and decides which objects may be shown (DESIGN 10: display qualification unchanged).

use std::collections::BTreeMap;

/// A radar measurement jump this far from the filtered measurement in one 50 ms step is a jump (DESIGN 11.1).
pub const JUMP: f64 = 0.3;
/// The time constant over which the drawn centre absorbs a jump (DESIGN 11.1: 1.5-2 s).
pub const ABSORB_TAU: f64 = 1.75;

pub(crate) fn reflection_jump(step: f64, gap: f64) -> bool {
    step.abs() > JUMP * (gap / 0.05).max(1.)
}

pub(crate) fn decay_reflection(offset: f64, dt: f64) -> f64 {
    offset - offset * (dt / ABSORB_TAU).min(1.)
}
/// Measurement noise at distance d is NOISE_BASE + NOISE_SLOPE * d (E2), the steady gain GAIN_REF / that, clipped.
pub const NOISE_BASE: f64 = 0.06;
pub const NOISE_SLOPE: f64 = 0.004;
pub const GAIN_REF: f64 = 0.05;
pub const GAIN_MIN: f64 = 0.15;
pub const GAIN_MAX: f64 = 0.6;
/// Lane width without usable lines, and the accepted width of the ego lane.
pub const LANE_WIDTH: f64 = 3.5;
pub const WIDTH_MIN: f64 = 1.8;
pub const WIDTH_MAX: f64 = 5.0;
/// A centre within this of a lane line straddles it (DESIGN 6.7, 11).
pub const STRADDLE_BAND: f64 = 0.4;
/// leadOne matches an FF/LF/RF vehicle within these gates.
pub const LEAD_GATE_X: f64 = 3.0;
pub const LEAD_GATE_Y: f64 = 1.5;
/// A different vid may not take back a slot from its new owner sooner than this after the switch.
pub const PINGPONG_HOLD: f64 = 1.0;
/// An owner kept against a ping-pong must have been measured this recently.
pub const OWNER_FRESH: f64 = 0.3;

pub type Vid = i64;

/// One map object the display may need this update. `lines` are the ego lane view's lines at `x` in the ego frame
/// (left +): left outer, left inner, right inner, right outer; None where the map has no usable line.
#[derive(Clone, Debug)]
pub struct Obj {
    pub vid: Vid,
    pub x: f64,
    pub y: f64,
    pub seen: f64,
    pub report: Option<(f64, f64, f64)>,
    pub lines: [Option<f64>; 4],
}

/// One display update's input.
#[derive(Clone, Debug)]
pub struct FrameIn {
    pub now: f64,
    /// livePose yaw rate, left +, rad/s; None without a valid livePose.
    pub yaw: Option<f64>,
    /// The existing selector's FF, LF, RF keys (display qualification and its switch guard as they are).
    pub selected: [Option<Vid>; 3],
    /// radarState leadOne (dRel, yRel) when its status is set.
    pub lead: Option<(f64, f64)>,
    pub objs: Vec<Obj>,
    /// The road's heading in the ego frame (left +, rad) at ego, from the ego lane's inner lines; None without them.
    pub heading: Option<f64>,
}

/// The lateral estimate of one vehicle (DESIGN 11.1-11.3). `m` follows the radar measurement with a light,
/// distance-scaled filter. A step from the last measurement `p` larger than JUMP moves `m` with it and goes into the
/// offset `b` as well, which decays over ABSORB_TAU, so the drawn centre `y = m - b` moves only slowly after a jump
/// and keeps up with a steady move. Ego rotation moves `m` and `p` before the measurement is compared, so a turn is
/// not taken for a jump.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub x: f64,
    pub p: f64,
    pub m: f64,
    pub b: f64,
    pub y: f64,
    pub stamp: f64,
    pub now: f64,
}

impl Track {
    pub(crate) fn new(now: f64, (stamp, x, m): (f64, f64, f64)) -> Self {
        Track { x, p: m, m, b: 0., y: m, stamp, now }
    }

    pub fn gain(x: f64) -> f64 {
        let noise = NOISE_BASE + NOISE_SLOPE * x.max(0.);
        (GAIN_REF / noise).clamp(GAIN_MIN, GAIN_MAX)
    }

    pub(crate) fn step(&mut self, report: Option<(f64, f64, f64)>, now: f64, yaw: Option<f64>) {
        let dt = (now - self.now).max(0.);
        if let Some(yaw) = yaw {
            let turn = self.x * yaw * dt;
            self.m -= turn;
            self.p -= turn;
        }
        if let Some((stamp, rx, ry)) = report {
            if stamp > self.stamp {
                let gap = if self.stamp.is_finite() { stamp - self.stamp } else { 0.05 };
                let jump = ry - self.p;
                if reflection_jump(jump, gap) {
                    self.b += jump;
                    self.m += jump;
                } else {
                    self.m += Track::gain(rx) * (ry - self.m);
                }
                self.p = ry;
                self.stamp = stamp;
                self.x = rx;
            }
        }
        self.b = decay_reflection(self.b, dt);
        self.y = self.m - self.b;
        self.now = now;
    }
}

/// The ego-relative lanes at one distance: boundaries[i] is the line between lane i-2 and i-1 (lane 0 is ego's,
/// left is -1), from left to right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneView {
    pub boundaries: [f64; 4],
}

impl LaneView {
    pub fn new(lines: &[Option<f64>; 4]) -> Self {
        let (mut left, mut right) = match (lines[1], lines[2]) {
            (Some(l), Some(r)) => (l, r),
            (Some(l), None) => (l, l - LANE_WIDTH),
            (None, Some(r)) => (r + LANE_WIDTH, r),
            (None, None) => (0.5 * LANE_WIDTH, -0.5 * LANE_WIDTH),
        };
        let width = left - right;
        if !(WIDTH_MIN..=WIDTH_MAX).contains(&width) {
            let mid = 0.5 * (left + right);
            left = mid + 0.5 * LANE_WIDTH;
            right = mid - 0.5 * LANE_WIDTH;
        }
        let width = left - right;
        let outer = |line: Option<f64>, inner: f64, sign: f64| match line {
            Some(v) if (WIDTH_MIN..=WIDTH_MAX).contains(&(sign * (v - inner))) => v,
            _ => inner + sign * width,
        };
        LaneView { boundaries: [outer(lines[0], left, 1.), left, right, outer(lines[3], right, -1.)] }
    }

    /// The lane (-2..=2) y lies in.
    pub fn lane(&self, y: f64) -> i64 {
        let b = &self.boundaries;
        if y > b[0] {
            -2
        } else if y > b[1] {
            -1
        } else if y >= b[2] {
            0
        } else if y >= b[3] {
            1
        } else {
            2
        }
    }

    pub fn center(&self, lane: i64) -> f64 {
        let b = &self.boundaries;
        let width = b[1] - b[2];
        match lane {
            -2 => b[0] + 0.5 * width,
            -1 => 0.5 * (b[0] + b[1]),
            0 => 0.5 * (b[1] + b[2]),
            1 => 0.5 * (b[2] + b[3]),
            _ => b[3] - 0.5 * width,
        }
    }

    /// The boundary (0..4) y straddles, the nearest within STRADDLE_BAND.
    pub fn straddled(&self, y: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &line) in self.boundaries.iter().enumerate() {
            let d = (y - line).abs();
            if d <= STRADDLE_BAND && best.is_none_or(|(_, e)| d < e) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }
}

/// Where a vehicle belongs: FF (0) only fully inside ego's lane, LF (1) on or left of its left line, RF (2) on or
/// right of its right line.
pub fn side_of(view: &LaneView, y: f64) -> usize {
    match view.straddled(y) {
        Some(0) | Some(1) => 1,
        Some(_) => 2,
        None => match view.lane(y) {
            0 => 0,
            l if l < 0 => 1,
            _ => 2,
        },
    }
}

/// The per-slot memory of the ping-pong guard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotMemory {
    pub owner: Option<Vid>,
    pub previous: Option<Vid>,
    pub since: f64,
}

/// The common state across updates.
#[derive(Clone, Debug, Default)]
pub struct Common {
    pub tracks: BTreeMap<Vid, Track>,
    pub slots: [SlotMemory; 3],
}

/// What the common part hands the displays this update.
pub struct FrameOut<'a> {
    pub assigned: [Option<Vid>; 3],
    pub objs: BTreeMap<Vid, &'a Obj>,
}

impl Common {
    /// Advance the tracks of the objects in `f` (a track whose object left the input is dropped) and assign slots.
    pub fn update<'a>(&mut self, f: &'a FrameIn) -> FrameOut<'a> {
        let objs: BTreeMap<Vid, &Obj> = f.objs.iter().map(|o| (o.vid, o)).collect();
        self.tracks.retain(|vid, _| objs.contains_key(vid));
        for (&vid, o) in &objs {
            match self.tracks.get_mut(&vid) {
                Some(track) => track.step(o.report, f.now, f.yaw),
                None => {
                    let report = o.report.unwrap_or((f64::NEG_INFINITY, o.x, o.y));
                    self.tracks.insert(vid, Track::new(f.now, report));
                }
            }
        }
        let lead = self.lead(f, &objs);
        let assigned = self.assign(f, &objs, lead);
        FrameOut { assigned, objs }
    }

    fn y(&self, vid: Vid) -> f64 {
        self.tracks[&vid].y
    }

    fn lead(&self, f: &FrameIn, objs: &BTreeMap<Vid, &Obj>) -> Option<Vid> {
        let (d, y) = f.lead?;
        let mut best: Option<(f64, Vid)> = None;
        for vid in f.selected.iter().flatten() {
            let Some(o) = objs.get(vid) else { continue };
            let dx = (o.x - d).abs();
            if dx <= LEAD_GATE_X && (self.y(*vid) - y).abs() <= LEAD_GATE_Y && best.is_none_or(|(e, _)| dx < e) {
                best = Some((dx, *vid));
            }
        }
        best.map(|(_, vid)| vid)
    }

    /// DESIGN 6.4 on the existing selector's choice.
    fn assign(&mut self, f: &FrameIn, objs: &BTreeMap<Vid, &Obj>, lead: Option<Vid>) -> [Option<Vid>; 3] {
        let mut slots = [None; 3];
        let x = |vid: Vid| objs[&vid].x;
        let nearer = |a: Vid, b: Option<Vid>| b.is_none_or(|b| x(a) < x(b));
        // Side vehicles first, so an FF vehicle that moves aside meets them.
        for s in [1, 2] {
            if let Some(vid) = f.selected[s].filter(|v| objs.contains_key(v)) {
                slots[s] = Some(vid);
            }
        }
        if let Some(vid) = f.selected[0].filter(|v| objs.contains_key(v)) {
            let side = side_of(&LaneView::new(&objs[&vid].lines), self.y(vid));
            if side == 0 || Some(vid) == lead {
                slots[0] = Some(vid);
            } else if nearer(vid, slots[side]) {
                slots[side] = Some(vid);
            } else {
                slots[0] = Some(vid);
            }
        }
        for s in [1, 2] {
            if let Some(vid) = slots[s] {
                let side = side_of(&LaneView::new(&objs[&vid].lines), self.y(vid));
                if (side == 0 && slots[0].is_none()) || Some(vid) == lead {
                    if let Some(old) = slots[0].filter(|_| Some(vid) == lead) {
                        // The vehicle openpilot follows takes FF; the one it displaces stays only on that side.
                        let old_side = side_of(&LaneView::new(&objs[&old].lines), self.y(old));
                        slots[s] = (old_side == s).then_some(old);
                    } else {
                        slots[s] = None;
                    }
                    slots[0] = Some(vid);
                }
            }
        }
        for s in 0..3 {
            let new = slots[s];
            let memory = &self.slots[s];
            if let Some(owner) = memory.owner {
                let fresh = objs.get(&owner).is_some_and(|o| f.now - o.seen <= OWNER_FRESH);
                let elsewhere = slots.iter().enumerate().any(|(t, v)| t != s && *v == Some(owner));
                let lead_in = s == 0 && new.is_some() && new == lead;
                if new.is_some()
                    && new != Some(owner)
                    && new == memory.previous
                    && f.now - memory.since < PINGPONG_HOLD
                    && fresh
                    && !elsewhere
                    && !lead_in
                {
                    slots[s] = Some(owner);
                }
            }
            let memory = &mut self.slots[s];
            if slots[s] != memory.owner {
                if slots[s].is_some() && memory.owner.is_some() {
                    memory.previous = memory.owner;
                    memory.since = f.now;
                }
                memory.owner = slots[s];
            }
        }
        slots
    }
}
