//! The wide camera's side cars on the map's slots (item 68 on item 80).
//!
//! `ccnc_visiond` publishes one side view per camera frame, the two sides taking turns. Each side has its own
//! `det_rules::DetRules`: a side's rules only ever look at that side's lane, so the two can step at different
//! frames. A frame is placed with the inputs of its own time (`Context`), and what the rules decided is held until
//! the next frame of that side, at most `HOLD_S`.
//!
//! Only LF, RF, LR and RR, only at or below the rules' speed limit, and the display only: nothing here feeds the
//! radar map or control.
//!
//! Times are seconds on CLOCK_MONOTONIC, the clock of liveTracks' `logMonoTime`. A frame's time is the message's
//! `frameMonoTime`, never `timestampSof` (the camera's CLOCK_BOOTTIME).

pub mod kind;
mod wide;

pub use det_rules::{Calib, Cand, SideOutput, SlotState, Track};
pub use kind::{KindResult, KindView, Kinds};
pub use wide::Geometry;

use crate::pos::common::LANE_WIDTH;
use crate::slots::{Slot, VISION_CONFIRMS, VISION_ONLY};
use crate::world::PoseAt;
use det_rules::{swap_decision, Act, DetRules, SwapCfg};

/// A result that reaches the map later than this after its frame is dropped.
/// Raised to 3.0 s for little-core SCHED_IDLE inference (~0.6-1.5 s/view, tail up to ~2.8 s); positions are stored
/// in the world frame at `frameMonoTime` (pose & input histories keep 4 s) and moved by ego motion, and camera
/// cars still turn off immediately above 3 m/s.
pub const LATE_S: f64 = 3.0;
/// Without a newer result of a side for this long, its camera cars leave the slots.
/// Covers a 3.0 s per-side cadence (1.5 s/view on little cores) without blinking while still clearing promptly if
/// `ccnc_visiond` stops or starves.
pub const HOLD_S: f64 = 3.5;
/// How long cached full-frame raw candidates stay usable for subsequent side-view frames (~1 slow L/R/raw cycle).
pub const RAW_HOLD_S: f64 = 4.0;
/// When a full-frame raw result arrives after a side result (3-view L/R/raw cycle), keep the held side car alive
/// up to this long after the raw arrival (capped at `RAW_HOLD_MAX_S` after the side arrival) so slow 3-view cycles
/// do not blink off between frames while still expiring promptly if `ccnc_visiond` stops.
const RAW_HOLD_EXTEND_S: f64 = 1.5;
const RAW_HOLD_MAX_S: f64 = 4.5;
/// Rule E's speed limit (det_rules `RuleE::step`): camera cars are shown at or below it only.
pub const SLOW: f64 = 3.0;
/// Not while reversing. A standing car's vEgo dips a little below zero (`ccnc_visiond` STANDSTILL_NOISE).
pub const STANDSTILL_NOISE: f64 = 0.01;
/// A camera car within this of a radar car in another slot is that radar car (the vid link gate).
pub const MERGE_X: f64 = 6.0;
pub const MERGE_Y: f64 = 1.0;
/// A camera car this close to the radar car of its own slot is that car: the radar's stays and the camera only
/// confirms it. 0000047f: standing in a queue, a radar car 3-4 m ahead of the camera's car crossed the swap margin
/// back and forth and the slot swapped between the two 44 times (16 56.1 s: five times in 0.9 s, 0.7 m across).
/// Across is wider than `MERGE_Y`: camera cars sit 0.7 m further out than the radar's (same slot, LF +0.74, RF -0.68).
pub const SAME_SLOT_X: f64 = 4.0;
pub const SAME_SLOT_Y: f64 = 1.5;
/// A slot changes between the radar's car and the camera's only after the camera's rules asked for the change this
/// long without a break.
pub const SWAP_HOLD_S: f64 = 0.5;
/// Where a camera-only car is drawn across: about the next lane's centre. 0000047f, ego waiting: camera-only LF
/// at 4.1 m, LR at 4.4 m (median |lateral|), the radar's at 3.2 m and 2.8 m.
pub const SIDE_LANE_PLAY: f64 = 0.3;
/// Camera cars have their own vids, one per slot, above any radar vid the map hands out.
pub const VISION_VID: [u32; 4] = [u32::MAX - 3, u32::MAX - 2, u32::MAX - 1, u32::MAX];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left = 0,
    Right = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameView {
    Side(Side),
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box {
    /// x1, y1, x2, y2 in the frame's view pixels (side view 512x384 or raw 1344x760).
    pub xyxy: [f64; 4],
    pub confidence: f64,
    /// COCO class.
    pub class_id: u8,
}

/// One CcncVision frame result.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// Start of frame, ns on CLOCK_MONOTONIC.
    pub time_ns: u64,
    pub view: FrameView,
    /// rpyCalib, wideFromDeviceEuler, height: what the view was built from (the height is the latest one).
    pub calibration: [f32; 7],
    pub boxes: Vec<Box>,
}

/// The other inputs at the frame's time.
#[derive(Clone, Copy, Debug)]
pub struct Context<'a> {
    pub v_ego: f64,
    pub tracks: &'a [Track],
    pub lprob: Option<&'a [f64]>,
    /// The map's radar slots: [left, right][front, rear].
    pub slots: [[SlotState; 2]; 2],
    /// The drive has been above 5.5 m/s once (`det_rules::StartGuard`).
    pub guard: bool,
}

/// What rule E decided for one slot, with the car's place in the world at its frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Seen {
    run: u32,
    cut: bool,
    world: (f64, f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    /// When the result reached the map.
    arrival: f64,
    /// [front, rear]: Some while rule E is on.
    seen: [Option<Seen>; 2],
}

/// Why a frame did not step the rules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Skipped {
    /// It reached the map more than `LATE_S` after its frame, or before it.
    Late,
    /// The map has no radar points or speed yet.
    NoInputs,
    /// The calibration is not usable (NaN from a fixed-map replay, or no geometry yet).
    NoCalibration,
    /// The side view geometry for these angles has to be built first (`Geometry::build`, then `set_geometry`).
    NeedsGeometry(Calib),
}

/// What one frame decided for its side, with the candidates the slot outputs index.
#[derive(Clone, Debug)]
pub struct Stepped {
    pub cands: Vec<Cand>,
    pub out: SideOutput,
}

#[derive(Clone, Debug)]
struct CachedRaw {
    time_ns: u64,
    arrival: f64,
    cands: Vec<Cand>,
}

/// Side frames whose candidates a classifier answer can still be tied to (it comes a message or two later).
const RECENT_SIDE_FRAMES: usize = 4;

/// Whether a slot shows the camera's car instead of the radar's, and since when the camera's rules want the other.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Swapped {
    camera: bool,
    since: Option<f64>,
}

impl Swapped {
    fn step(&mut self, now: f64, camera: bool) -> bool {
        if camera == self.camera {
            self.since = None;
        } else if now - *self.since.get_or_insert(now) >= SWAP_HOLD_S {
            *self = Swapped { camera, since: None };
        }
        self.camera
    }
}

#[derive(Default)]
pub struct Vision {
    geometry: Option<Geometry>,
    rules: [DetRules; 2],
    held: [Option<Held>; 2],
    /// LF, RF, LR, RR.
    swapped: [Swapped; 4],
    raw: Option<CachedRaw>,
    recent: std::collections::VecDeque<(u64, Side, Vec<Cand>)>,
}

fn calib_of(values: &[f32; 7]) -> Option<Calib> {
    values.iter().all(|v| v.is_finite()).then(|| {
        let v = values.map(f64::from);
        Calib {
            rpy: [v[0], v[1], v[2]],
            wide_from_device: [v[3], v[4], v[5]],
            height: v[6],
        }
    })
}

/// [left, right][front, rear] from the radar slots FF, LF, RF, LR, RR.
pub fn slot_states(slots: &[Option<Slot>; 5]) -> [[SlotState; 2]; 2] {
    let state = |i: usize| {
        slots[i].map_or(SlotState::default(), |s| SlotState {
            detect: true,
            dist: f64::from(s.distance).abs(),
        })
    };
    [[state(1), state(3)], [state(2), state(4)]]
}

impl Vision {
    pub fn set_geometry(&mut self, geometry: Geometry) {
        self.geometry = Some(geometry);
    }

    /// Positions of held results are in the world frame of the map: a map reset drops them. The rules keep going.
    pub fn reset_positions(&mut self) {
        self.held = [None; 2];
        self.swapped = [Swapped::default(); 4];
        self.raw = None;
        self.recent.clear();
    }

    /// The candidates the side frame of `time_ns` was stepped with, while it is one of the last few.
    pub fn side_cands(&self, time_ns: u64, side: Side) -> Option<Vec<Cand>> {
        self.recent.iter().find(|(t, s, _)| *t == time_ns && *s == side).map(|(_, _, c)| c.clone())
    }

    /// Process one CcncVision frame. `FrameView::Raw` caches raw candidates for subsequent side views and returns
    /// `Ok(None)`. `FrameView::Side` steps that side's rules (combining any fresh cached raw candidates) and returns
    /// `Ok(Some(Stepped))`.
    pub fn frame(
        &mut self,
        frame: &Frame,
        arrival: f64,
        context: &Context,
        frame_pose: PoseAt,
    ) -> Result<Option<Stepped>, Skipped> {
        let age = arrival - frame.time_ns as f64 * 1e-9;
        if !(0. ..=LATE_S).contains(&age) {
            return Err(Skipped::Late);
        }
        let calib = calib_of(&frame.calibration).ok_or(Skipped::NoCalibration)?;
        let geometry = match &mut self.geometry {
            Some(g) if g.serves(&calib) => g,
            _ => return Err(Skipped::NeedsGeometry(calib)),
        };
        let side = match frame.view {
            FrameView::Raw => {
                let cands = geometry.raw_candidates(frame, &calib);
                self.raw = Some(CachedRaw { time_ns: frame.time_ns, arrival, cands });
                return Ok(None);
            }
            FrameView::Side(side) => side,
        };
        let raw_cands: &[Cand] = self
            .raw
            .as_ref()
            .filter(|r| {
                let dt = (frame.time_ns as f64 - r.time_ns as f64) * 1e-9;
                (-0.05..=RAW_HOLD_S).contains(&dt)
            })
            .map_or(&[], |r| r.cands.as_slice());
        let (cands, out) = geometry.step(&mut self.rules[side as usize], frame, side, &calib, context, raw_cands);
        let side_idx = side as usize;
        if kind::SIDE_KINDS {
            if self.recent.len() == RECENT_SIDE_FRAMES {
                self.recent.pop_front();
            }
            self.recent.push_back((frame.time_ns, side, cands.clone()));
        }
        let seen = |slot: &det_rules::SlotOutput| {
            let cand = &cands[slot.cand.filter(|_| slot.display_on)?];
            Some(Seen {
                run: slot.confirm_run,
                cut: cand.cut,
                world: frame_pose.to_world(cand.x, -cand.yc),
            })
        };
        self.held[side_idx] = Some(Held {
            arrival,
            seen: [seen(&out.front), seen(&out.rear)],
        });
        Ok(Some(Stepped { cands, out }))
    }

    /// Put the held camera cars on the radar slots of `now`.
    pub fn apply(&mut self, now: f64, v_ego: f64, ego: PoseAt, slots: &mut [Option<Slot>; 5]) {
        if !(-STANDSTILL_NOISE..=SLOW).contains(&v_ego) {
            self.swapped = [Swapped::default(); 4];
            return;
        }
        let radar = *slots;
        let raw_arrival = self.raw.as_ref().map(|r| r.arrival);
        for (side, held) in self.held.iter().enumerate() {
            let Some(held) = held.filter(|h| {
                if now < h.arrival {
                    return false;
                }
                let mut deadline = h.arrival + HOLD_S;
                if let Some(ra) = raw_arrival.filter(|&ra| ra > h.arrival && ra <= now) {
                    deadline = deadline.max(ra + RAW_HOLD_EXTEND_S).min(h.arrival + RAW_HOLD_MAX_S);
                }
                now <= deadline
            }) else {
                for part in 0..2 {
                    self.swapped[side + 2 * part] = Swapped::default();
                }
                continue;
            };
            let swap = &self.rules[side].swap;
            for (part, seen) in held.seen.iter().enumerate() {
                let index = 1 + side + 2 * part; // LF, RF, LR, RR
                let Some(seen) = seen else {
                    self.swapped[index - 1] = Swapped::default();
                    continue;
                };
                place(now, index, part == 1, seen, swap, ego, &radar, slots, &mut self.swapped[index - 1]);
            }
        }
    }
}

/// Radar slot distance along ego's x: rear slots are behind.
fn along(index: usize, slot: &Slot) -> f64 {
    let d = f64::from(slot.distance);
    if (3..5).contains(&index) {
        -d.abs()
    } else {
        d
    }
}

#[allow(clippy::too_many_arguments)]
fn place(
    now: f64,
    index: usize,
    rear: bool,
    seen: &Seen,
    swap: &SwapCfg,
    ego: PoseAt,
    radar: &[Option<Slot>; 5],
    slots: &mut [Option<Slot>; 5],
    swapped: &mut Swapped,
) {
    let (x, y) = ego.to_ego(seen.world.0, seen.world.1);
    let distance = if rear { (-x).max(0.) } else { x };
    let here = radar[index];
    let same_car = here.is_some_and(|s| {
        (along(index, &s) - x).abs() <= SAME_SLOT_X && (f64::from(s.lateral) - y).abs() <= SAME_SLOT_Y
    });
    let act = swap_decision(
        true,
        Some((distance, seen.cut)),
        here.is_some(),
        here.map_or(0., |s| f64::from(s.distance).abs()),
        seen.run,
        swap,
    )
    .map(|act| match act {
        Act::Swap if same_car => Act::Keep,
        act => act,
    });
    // An empty slot takes the camera's car at once; between two cars the slot follows the rules only once they hold.
    let act = match act {
        Some(Act::Fill) => {
            *swapped = Swapped { camera: true, since: None };
            act
        }
        Some(Act::Swap | Act::Keep) => Some(if swapped.step(now, act == Some(Act::Swap)) { Act::Swap } else { Act::Keep }),
        None => {
            *swapped = Swapped::default();
            None
        }
    };
    match act {
        None => {}
        Some(Act::Keep) => {
            if let Some(slot) = &mut slots[index] {
                slot.vision = VISION_CONFIRMS;
            }
        }
        Some(Act::Fill | Act::Swap) => {
            // The same car already drawn by the radar in a neighbouring slot stays the radar's.
            let elsewhere = radar.iter().enumerate().any(|(i, s)| {
                i != index && s.is_some_and(|s| (along(i, &s) - x).abs() <= MERGE_X && (f64::from(s.lateral) - y).abs() <= MERGE_Y)
            });
            if !elsewhere {
                let left = index % 2 == 1; // LF, LR
                let y = if left { 1. } else { -1. } * (if left { y } else { -y }).clamp(LANE_WIDTH - SIDE_LANE_PLAY, LANE_WIDTH + SIDE_LANE_PLAY);
                slots[index] = Some(Slot {
                    vid: VISION_VID[index - 1],
                    distance: (if rear { distance } else { x }) as f32,
                    lateral: y as f32,
                    detect: 1,
                    lane_coord_rel: (-y / LANE_WIDTH) as f32,
                    v_rel: 0.,
                    kind: 0,
                    kind_confirmed: false,
                    kind_wanted: false,
                    vision: VISION_ONLY,
                });
            }
        }
    }
}
