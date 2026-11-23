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

mod kind;
mod wide;

pub use det_rules::{Calib, Cand, SideOutput, SlotState, Track};
pub use wide::Geometry;

use crate::pos::common::LANE_WIDTH;
use crate::slots::{Slot, VISION_CONFIRMS, VISION_ONLY};
use crate::world::PoseAt;
use det_rules::{swap_decision, Act, DetRules, SwapCfg};

/// A result that reaches the map later than this after its frame is dropped.
pub const LATE_S: f64 = 1.5;
/// Without a newer result of a side for this long, its camera cars leave the slots.
pub const HOLD_S: f64 = 2.0;
/// Rule E's speed limit (det_rules `RuleE::step`): camera cars are shown at or below it only.
pub const SLOW: f64 = 3.0;
/// Not while reversing. A standing car's vEgo dips a little below zero (`ccnc_visiond` STANDSTILL_NOISE).
pub const STANDSTILL_NOISE: f64 = 0.01;
/// A camera car within this of a radar car in another slot is that radar car (the vid link gate).
pub const MERGE_X: f64 = 6.0;
pub const MERGE_Y: f64 = 1.0;
/// Camera cars have their own vids, one per slot, above any radar vid the map hands out.
pub const VISION_VID: [u32; 4] = [u32::MAX - 3, u32::MAX - 2, u32::MAX - 1, u32::MAX];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left = 0,
    Right = 1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box {
    /// x1, y1, x2, y2 in side view pixels.
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
    pub side: Side,
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

#[derive(Default)]
pub struct Vision {
    geometry: Option<Geometry>,
    rules: [DetRules; 2],
    held: [Option<Held>; 2],
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
    }

    /// Step the frame's side. `arrival` is when the result reached the map, `frame_pose` the ego pose at the frame.
    pub fn frame(
        &mut self,
        frame: &Frame,
        arrival: f64,
        context: &Context,
        frame_pose: PoseAt,
    ) -> Result<Stepped, Skipped> {
        let age = arrival - frame.time_ns as f64 * 1e-9;
        if !(0. ..=LATE_S).contains(&age) {
            return Err(Skipped::Late);
        }
        let calib = calib_of(&frame.calibration).ok_or(Skipped::NoCalibration)?;
        let geometry = match &mut self.geometry {
            Some(g) if g.serves(&calib) => g,
            _ => return Err(Skipped::NeedsGeometry(calib)),
        };
        let (cands, out) = geometry.step(&mut self.rules[frame.side as usize], frame, &calib, context);
        let side = frame.side as usize;
        let seen = |slot: &det_rules::SlotOutput| {
            let cand = &cands[slot.cand.filter(|_| slot.display_on)?];
            Some(Seen {
                run: slot.confirm_run,
                cut: cand.cut,
                world: frame_pose.to_world(cand.x, -cand.yc),
            })
        };
        self.held[side] = Some(Held {
            arrival,
            seen: [seen(&out.front), seen(&out.rear)],
        });
        Ok(Stepped { cands, out })
    }

    /// Put the held camera cars on the radar slots of `now`.
    pub fn apply(&self, now: f64, v_ego: f64, ego: PoseAt, slots: &mut [Option<Slot>; 5]) {
        if !(-STANDSTILL_NOISE..=SLOW).contains(&v_ego) {
            return;
        }
        let radar = *slots;
        for (side, held) in self.held.iter().enumerate() {
            let Some(held) = held.filter(|h| (0. ..=HOLD_S).contains(&(now - h.arrival))) else {
                continue;
            };
            let swap = &self.rules[side].swap;
            for (part, seen) in held.seen.iter().enumerate() {
                let Some(seen) = seen else { continue };
                let index = 1 + side + 2 * part; // LF, RF, LR, RR
                place(index, part == 1, seen, swap, ego, &radar, slots);
            }
        }
    }
}

fn place(
    index: usize,
    rear: bool,
    seen: &Seen,
    swap: &SwapCfg,
    ego: PoseAt,
    radar: &[Option<Slot>; 5],
    slots: &mut [Option<Slot>; 5],
) {
    let (x, y) = ego.to_ego(seen.world.0, seen.world.1);
    let distance = if rear { (-x).max(0.) } else { x };
    let here = radar[index];
    let act = swap_decision(
        true,
        Some((distance, seen.cut)),
        here.is_some(),
        here.map_or(0., |s| f64::from(s.distance).abs()),
        seen.run,
        swap,
    );
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
                i != index
                    && s.is_some_and(|s| {
                        let rx = if (3..5).contains(&i) { -f64::from(s.distance).abs() } else { f64::from(s.distance) };
                        (rx - x).abs() <= MERGE_X && (f64::from(s.lateral) - y).abs() <= MERGE_Y
                    })
            });
            if !elsewhere {
                slots[index] = Some(Slot {
                    vid: VISION_VID[index - 1],
                    distance: (if rear { distance } else { x }) as f32,
                    lateral: y as f32,
                    detect: 1,
                    lane_coord_rel: (-y / LANE_WIDTH) as f32,
                    v_rel: 0.,
                    kind: 0,
                    kind_confirmed: false,
                    vision: VISION_ONLY,
                });
            }
        }
    }
}
