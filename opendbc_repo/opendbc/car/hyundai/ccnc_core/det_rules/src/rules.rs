//! The stateful rules (port of pipeline.py): rule E (display), swap/fill, the split gate and class confirmation R1,
//! with their timing in one place.
use crate::wide::{consistent, Cand};
use crate::JITTER_S;

/// The frame-count rules in one place: rule E on, swap/fill min_run, the split gate (all `n_on`) and R1.
/// `jump_rate` reads the stability check in time: (m/s, m/s) times the time since the last hit. `win_s` tolerates frames
/// without a candidate: `n_on` hits whose span fits in `win_s` (None = consecutive only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub n_on: u32,
    pub win_s: Option<f64>,
    pub jump_rate: Option<(f64, f64)>,
    pub r1_gap_s: f64, // R1: distinct views at least this far apart
    pub r1_confirm: u32,
    pub r1_flip: u32,
}
impl Timing {
    /// The rules as written for consecutive processed frames (stability 1.0 / 0.7 m per frame, R1 views 1.0 s apart).
    pub const FRAMES: Timing = Timing { n_on: 2, win_s: None, jump_rate: None, r1_gap_s: 1.0, r1_confirm: 2, r1_flip: 4 };
    /// 1 Hz recommendation (REPORT_1HZ_KO section 6): stability per second, R1 views one frame apart count.
    pub const TIME_1HZ: Timing =
        Timing { n_on: 2, win_s: None, jump_rate: Some((2.0, 1.4)), r1_gap_s: 1.0 - JITTER_S, r1_confirm: 2, r1_flip: 4 };
    /// TIME_1HZ + '2 hits within 2 s' (one miss tolerated). Tested and not recommended.
    pub const TIME_1HZ_WIN2: Timing =
        Timing { n_on: 2, win_s: Some(2.0), jump_rate: Some((2.0, 1.4)), r1_gap_s: 1.0 - JITTER_S, r1_confirm: 2, r1_flip: 4 };
    pub fn rule_e(&self) -> RuleE {
        RuleE::new(self.n_on, self.win_s, self.jump_rate)
    }
    pub fn split_gate(&self) -> SplitGate {
        SplitGate { rule: self.rule_e() }
    }
    pub fn class_confirm(&self) -> ClassConfirm {
        ClassConfirm::new(self.r1_gap_s, self.r1_confirm, self.r1_flip)
    }
}
impl Default for Timing {
    fn default() -> Timing {
        Timing::TIME_1HZ
    }
}

/// Why rule E is on or off after a step. Reporting only; the rule itself does not read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    On,
    Waiting,      // stable so far, fewer than n_on hits
    Jump,         // position jumped: the count restarted at this frame
    NoCandidate,
    Moving,       // ego v > 3 m/s
    StartGuard,   // the drive has not exceeded 5.5 m/s yet
    Inconsistent, // side-view bottom and size distances disagree
    Filtered,     // vetoed by a display filter
}
impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::On => "on",
            Reason::Waiting => "waiting",
            Reason::Jump => "jump",
            Reason::NoCandidate => "no_candidate",
            Reason::Moving => "moving",
            Reason::StartGuard => "start_guard",
            Reason::Inconsistent => "inconsistent",
            Reason::Filtered => "filtered",
        }
    }
}

/// Rule E (display) of one slot: on when `n_on` consecutive processed frames have a candidate with ego v <= 3 m/s,
/// position stable, side-view bottom/size distance consistent, and the start guard passed.
#[derive(Clone, Debug)]
pub struct RuleE {
    pub n_on: u32,
    pub win_s: Option<f64>,
    pub jump_rate: Option<(f64, f64)>,
    pub run: u32,
    pub reason: Reason,
    prev: Option<(f64, f64)>, // x, yc of the last accepted candidate
    hits: Vec<f64>,           // capture times of the hits counted in `run`
}
impl Default for RuleE {
    fn default() -> RuleE {
        Timing::FRAMES.rule_e()
    }
}
impl RuleE {
    pub fn new(n_on: u32, win_s: Option<f64>, jump_rate: Option<(f64, f64)>) -> RuleE {
        RuleE { n_on, win_s, jump_rate, run: 0, reason: Reason::NoCandidate, prev: None, hits: Vec::new() }
    }
    /// t: capture time of the frame in s (unused when both `win_s` and `jump_rate` are None).
    pub fn step(&mut self, c: Option<&Cand>, v_ego: f64, guard_ok: bool, extra_ok: bool, t: f64) -> bool {
        let c = match c {
            Some(c) if v_ego <= 3.0 && consistent(c) && guard_ok && extra_ok => c,
            _ => {
                // with win_s a frame without any candidate is a miss (hits kept); every other failure resets
                if c.is_some() || self.win_s.is_none() || v_ego > 3.0 || !guard_ok {
                    self.hits.clear();
                    self.prev = None;
                }
                self.run = 0;
                self.reason = match c {
                    None => Reason::NoCandidate,
                    Some(_) if v_ego > 3.0 => Reason::Moving,
                    Some(_) if !guard_ok => Reason::StartGuard,
                    Some(c) if !consistent(c) => Reason::Inconsistent,
                    Some(_) => Reason::Filtered,
                };
                return false;
            }
        };
        let mut jumped = false;
        if let Some((px, pyc)) = self.prev {
            let (dx, dy) = match self.jump_rate {
                None => (1.0, 0.7),
                Some((kx, ky)) => {
                    let dt = t - self.hits.last().copied().unwrap_or(t);
                    (kx * dt, ky * dt)
                }
            };
            if (c.x - px).abs() > dx || (c.yc - pyc).abs() > dy {
                self.hits.clear();
                jumped = true;
            }
        }
        if let Some(w) = self.win_s {
            self.hits.retain(|h| t - h <= w + JITTER_S);
        }
        self.hits.push(t);
        self.prev = Some((c.x, c.yc));
        self.run = self.hits.len() as u32;
        let on = self.run >= self.n_on;
        self.reason = if on {
            Reason::On
        } else if jumped {
            Reason::Jump
        } else {
            Reason::Waiting
        };
        on
    }
}

// ---------------------------------------------------------------- swap to the closer car
pub const SWAP_MARGIN: f64 = 3.0;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapCfg {
    pub min_run: u32,
    pub allow_cut: bool,
    pub max_x: f64,
    pub margin: f64,
}
impl Default for SwapCfg {
    fn default() -> SwapCfg {
        SwapCfg { min_run: 2, allow_cut: true, max_x: 6.0, margin: SWAP_MARGIN }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Fill, // slot empty, camera fills
    Swap, // slot holds a farther radar target, the camera's closer car replaces it
    Keep, // radar slot kept
}
impl Act {
    pub fn name(self) -> &'static str {
        match self {
            Act::Fill => "fill",
            Act::Swap => "swap",
            Act::Keep => "keep",
        }
    }
}
/// c: (distance to compare with the slot, cut) of the camera car, None without a candidate. None = camera not shown.
pub fn swap_decision(cam_on: bool, c: Option<(f64, bool)>, slot_detect: bool, slot_dist: f64, run: u32, cfg: &SwapCfg) -> Option<Act> {
    let (x, cut) = c.filter(|_| cam_on)?;
    let keep = slot_detect.then_some(Act::Keep);
    if run < cfg.min_run || (cut && !cfg.allow_cut) || x > cfg.max_x {
        return keep;
    }
    if !slot_detect {
        return Some(Act::Fill);
    }
    Some(if slot_dist - x > cfg.margin { Act::Swap } else { Act::Keep })
}

/// Keeps a two-car split only while its front car is on by rule E. The LF/RF rule E follows the shown candidate (the
/// single one until a split is accepted), so it cannot judge the proposed front car; this one follows the proposed
/// front car every frame (no candidate when there is no split proposal).
#[derive(Clone, Debug, Default)]
pub struct SplitGate {
    pub rule: RuleE,
}
impl SplitGate {
    pub fn apply<'a>(
        &mut self,
        split: (Option<&'a Cand>, Option<&'a Cand>),
        single: Option<&'a Cand>,
        v_ego: f64,
        guard_ok: bool,
        extra_ok: impl Fn(&Cand) -> bool,
        t: f64,
    ) -> (Option<&'a Cand>, Option<&'a Cand>) {
        let (front, rear) = split;
        let proposed = front.filter(|_| rear.is_some());
        let on = self.rule.step(proposed, v_ego, guard_ok, proposed.map_or(false, |f| extra_ok(f)), t);
        if rear.is_some() && on {
            split
        } else {
            (single, None)
        }
    }
}

// ---------------------------------------------------------------- front camera
pub const FRONT_CROP_Y: f64 = 620.0; // bonnet crop: rows < 620 of the 1344x760 road frame
pub const FRONT_IN: (u32, u32) = (416, 192); // W, H fed to the detector
pub const FRONT_W: f64 = 1344.0;

/// Front box in frame coordinates with what it may be used for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontBox {
    pub bbox: [f64; 4],
    pub cut: bool,                 // bottom on the crop line: not the car's ground contact
    pub use_bottom_distance: bool, // cut: bottom distance is ~6.2 m regardless of truth
    pub may_confirm_class: bool,   // cut: no class confirmation
    pub camera_only_ok: bool,      // cut: never a camera-only candidate (needs a radar point)
}
/// detector-input box -> frame box + flags. scale = input px per frame px (416 / 1344), tol in input px (Python: 2.0).
pub fn front_box(bbox: &[f64; 4], scale: f64, tol: f64) -> FrontBox {
    let b = bbox.map(|v| v / scale);
    let cut = b[3] >= FRONT_CROP_Y - tol / scale;
    FrontBox { bbox: b, cut, use_bottom_distance: !cut, may_confirm_class: !cut, camera_only_ok: !cut }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VehicleClass {
    Car,
    Truck,
    Bus,
}
impl VehicleClass {
    pub fn name(self) -> &'static str {
        match self {
            VehicleClass::Car => "car",
            VehicleClass::Truck => "truck",
            VehicleClass::Bus => "bus",
        }
    }
    pub fn from_name(s: &str) -> Option<VehicleClass> {
        match s {
            "car" => Some(VehicleClass::Car),
            "truck" => Some(VehicleClass::Truck),
            "bus" => Some(VehicleClass::Bus),
            _ => None,
        }
    }
}

/// R1, per physical-car id. Day only. Two observations at least `gap_s` apart that differ by >= 1 m distance or
/// >= 20 px box position and agree -> confirmed. Default car. Flip after `n_flip` agreeing opposite views.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassConfirm {
    pub shown: VehicleClass,
    pub confirmed: bool,
    pub gap_s: f64,
    pub n_confirm: u32,
    pub n_flip: u32,
    last: Option<(f64, f64, f64)>,      // t, dist, cx of the last accepted distinct view
    streak: Option<(VehicleClass, u32)>, // class of the agreeing views in a row and how many
}
impl Default for ClassConfirm {
    fn default() -> ClassConfirm {
        Timing::FRAMES.class_confirm()
    }
}
impl ClassConfirm {
    pub fn new(gap_s: f64, n_confirm: u32, n_flip: u32) -> ClassConfirm {
        ClassConfirm { shown: VehicleClass::Car, confirmed: false, gap_s, n_confirm, n_flip, last: None, streak: None }
    }
    /// One classified view of the car. day = usable evidence (not night); cut = cut box (no evidence).
    pub fn observe(&mut self, t: f64, dist: f64, cx: f64, cls: VehicleClass, day: bool, cut: bool) -> VehicleClass {
        if !day || cut {
            return self.shown;
        }
        if let Some((lt, ld, lc)) = self.last {
            if t - lt < self.gap_s || ((dist - ld).abs() < 1.0 && (cx - lc).abs() < 20.0) {
                return self.shown;
            }
        }
        self.last = Some((t, dist, cx));
        let n = match self.streak {
            Some((c, n)) if c == cls => n + 1,
            _ => 1,
        };
        self.streak = Some((cls, n));
        let need = if self.confirmed { self.n_flip } else { self.n_confirm };
        if n >= need && (!self.confirmed || cls != self.shown) {
            self.shown = cls;
            self.confirmed = true;
            self.streak = None;
        }
        self.shown
    }
}
