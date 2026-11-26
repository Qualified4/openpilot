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
use crate::slots::{Slot, SIDE_FRONT_MAX_M, VISION_CONFIRMS, VISION_ONLY};
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
/// The newest vision and radar messages in a daemon batch can be stamped slightly out of order.
const ARRIVAL_SKEW_S: f64 = 0.05;
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
    pending: Option<(f64, f64)>,
}

impl Swapped {
    fn step(&mut self, now: f64, arrival: f64, camera: bool, corroborate: bool) -> bool {
        if camera == self.camera {
            self.pending = None;
        } else {
            let (since, first_frame) = *self.pending.get_or_insert((now, arrival));
            if now - since >= SWAP_HOLD_S && (!corroborate || arrival > first_frame) {
                *self = Swapped { camera, pending: None };
            }
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

    /// Put the held camera cars on the radar slots of `now`. `protected` slots (FF, LF, RF, LR, RR) hold a car of the
    /// radar's own rules (the stop memory, a side pair): the camera does not take them over, it only confirms them.
    pub fn apply(
        &mut self, now: f64, v_ego: f64, ego: PoseAt, protected: [bool; 5],
        current_front: [bool; 2], slots: &mut [Option<Slot>; 5],
    ) {
        if !(-STANDSTILL_NOISE..=SLOW).contains(&v_ego) {
            self.swapped = [Swapped::default(); 4];
            return;
        }
        let radar = *slots;
        let raw_arrival = self.raw.as_ref().map(|r| r.arrival);
        for (side, held) in self.held.iter().enumerate() {
            let Some(held) = held.filter(|h| {
                if h.arrival - now > ARRIVAL_SKEW_S {
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
            // Give a rear camera observation first use of the rear slot.
            for part in [1, 0] {
                let index = 1 + side + 2 * part; // LF, RF, LR, RR
                if part == 1 && radar[index].is_some_and(|s| s.vid == 0) {
                    self.swapped[index - 1] = Swapped::default();
                    continue;
                }
                let Some(seen) = &held.seen[part] else {
                    self.swapped[index - 1] = Swapped::default();
                    continue;
                };
                place(now, held.arrival, v_ego, index, part == 1, seen, swap, ego, &radar,
                    protected[index], current_front[side], slots, &mut self.swapped[index - 1]);
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
    arrival: f64,
    v_ego: f64,
    index: usize,
    rear: bool,
    seen: &Seen,
    swap: &SwapCfg,
    ego: PoseAt,
    radar: &[Option<Slot>; 5],
    protected: bool,
    current_front: bool,
    slots: &mut [Option<Slot>; 5],
    swapped: &mut Swapped,
) {
    let (x, y) = ego.to_ego(seen.world.0, seen.world.1);
    let distance = if rear { (-x).max(0.) } else { x };
    if distance < 0. || (!rear && distance > SIDE_FRONT_MAX_M) || !swap.allows(distance, seen.cut, seen.run) {
        *swapped = Swapped::default();
        return;
    }
    let here = radar[index];
    let same_car = here.is_some_and(|s| {
        (along(index, &s) - x).abs() <= SAME_SLOT_X && (f64::from(s.lateral) - y).abs() <= SAME_SLOT_Y
    });
    if protected && current_front && here.is_some() {
        *swapped = Swapped::default();
        if same_car {
            slots[index].as_mut().unwrap().vision = VISION_CONFIRMS;
        }
        return;
    }
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
    // Whichever car the slot shows stays until it is gone or the other is clearly closer: a camera car goes back to
    // the radar's only once the radar car is `margin` closer, not when it is merely no farther (or the same car), and
    // a protected radar car is never taken over. 0000047f standing: the slot went back and forth 35 times between the
    // two, half of them as the stop memory or a side pair came and went under a camera car the rules kept choosing.
    let act = match act {
        Some(Act::Fill) => {
            *swapped = Swapped { camera: true, pending: None };
            act
        }
        Some(Act::Swap | Act::Keep) => {
            let radar_closer = here.is_some_and(|s| distance - f64::from(s.distance).abs() > swap.margin);
            let camera = if swapped.camera { !radar_closer } else { act == Some(Act::Swap) && !protected };
            // A single wide frame stays in memory long enough to pass the time gate on its own. At rest,
            // require another wide frame to corroborate replacing a stopped radar car (47f-14, LF at 48 s).
            let corroborate = !rear && camera && v_ego.abs() <= 0.25
                && here.is_some_and(|s| s.vid != 0 && (v_ego + f64::from(s.v_rel)).abs() <= 0.5);
            Some(if swapped.step(now, arrival, camera, corroborate) { Act::Swap } else { Act::Keep })
        }
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
            let alongside = !rear && current_front && distance <= 5. && here.is_some_and(|s| {
                f64::from(s.distance) > x + SAME_SLOT_X && s.distance <= 30.
                    && (f64::from(s.lateral) - y).abs() <= SAME_SLOT_Y
            });
            let target = if alongside { index + 2 } else { index };
            if alongside && slots[target].is_some_and(|s| s.vid != 0) {
                return;
            }
            // The same car already drawn by the radar in a neighbouring slot stays the radar's.
            let elsewhere = radar.iter().enumerate().any(|(i, s)| {
                i != index && s.is_some_and(|s| s.vid != 0 && (along(i, &s) - x).abs() <= MERGE_X && (f64::from(s.lateral) - y).abs() <= MERGE_Y)
            });
            if elsewhere {
                *swapped = Swapped::default();
            } else {
                let left = index % 2 == 1; // LF, LR
                let y = if left { 1. } else { -1. } * (if left { y } else { -y }).clamp(LANE_WIDTH - SIDE_LANE_PLAY, LANE_WIDTH + SIDE_LANE_PLAY);
                slots[target] = Some(Slot {
                    vid: VISION_VID[target - 1],
                    distance: if alongside { 0. } else { distance as f32 },
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

#[cfg(test)]
mod placement_tests {
    use super::*;
    use crate::slots::VISION_NONE;

    fn radar_slot() -> Slot {
        Slot {
            vid: 1,
            distance: 6.6,
            lateral: 3.,
            detect: 2,
            lane_coord_rel: -1.,
            v_rel: 0.,
            kind: 0,
            kind_confirmed: false,
            kind_wanted: false,
            vision: VISION_NONE,
        }
    }

    #[test]
    fn bsd_immediately_replaces_an_existing_wide_camera_rear_slot() {
        let mut vision = Vision::default();
        vision.held[1] = Some(Held {
            arrival: 1.,
            seen: [None, Some(Seen { world: (-2., -3.), run: 2, cut: false })],
        });
        let apply = |vision: &mut Vision, now, slots: &mut [Option<Slot>; 5]| {
            vision.apply(now, 0., PoseAt::default(), [false; 5], [false; 2], slots);
        };
        let mut slots = [None; 5];
        apply(&mut vision, 1., &mut slots);
        assert_eq!(slots[4].unwrap().vision, VISION_ONLY);
        assert!(vision.swapped[3].camera);

        let bsd = Slot { vid: 0, distance: 0., lateral: -3., ..radar_slot() };
        slots[4] = Some(bsd);
        apply(&mut vision, 1.05, &mut slots);
        assert_eq!(slots[4], Some(bsd));
        assert_eq!(vision.swapped[3], Swapped::default());
    }

    #[test]
    fn an_existing_camera_swap_is_cleared_when_it_leaves_the_rules_range() {
        let cfg = SwapCfg { allow_cut: false, ..SwapCfg::default() };
        let radar = [None, Some(radar_slot()), None, None, None];
        for (x, run, cut) in [(-2., 2, false), (8., 2, false), (2., 1, false), (2., 2, true)] {
            let mut slots = radar;
            let mut swap = Swapped { camera: true, pending: None };
            let seen = Seen { world: (x, 3.), run, cut };
            place(1., 1., 2., 1, false, &seen, &cfg, PoseAt::default(), &radar, false, false, &mut slots, &mut swap);
            assert_eq!(slots, radar, "x={x}, run={run}, cut={cut}");
            assert_eq!(swap, Swapped::default());
        }
    }

    #[test]
    fn a_stationary_radar_car_needs_two_wide_frames_before_it_is_replaced() {
        let mut vision = Vision::default();
        let radar = [None, Some(radar_slot()), None, None, None];
        let seen = |x| Some(Seen { world: (x, 2.5), run: 2, cut: false });
        let held = |arrival, x| Some(Held { arrival, seen: [seen(x), None] });
        vision.held[0] = held(1., 1.5);
        for now in [1., 1.5, 2.] {
            let mut slots = radar;
            vision.apply(now, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
            assert_eq!(slots[1].unwrap().vid, 1);
        }
        vision.held[0] = held(2.5, 3.5);
        let mut slots = radar;
        vision.apply(2.5, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
        assert_eq!(slots[1].unwrap().vid, 1, "the next observation no longer calls for a swap");

        for (t, expected) in [(3., VISION_CONFIRMS), (4., VISION_ONLY)] {
            vision.held[0] = held(t, 1.5);
            let mut slots = radar;
            vision.apply(t, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
            assert_eq!(slots[1].unwrap().vision, expected);
        }
    }

    #[test]
    fn a_held_wide_result_in_the_same_batch_survives_small_stamp_skew() {
        let mut vision = Vision::default();
        vision.held[0] = Some(Held {
            arrival: 2.,
            seen: [Some(Seen { world: (2., 3.), run: 2, cut: false }), None],
        });
        let mut slots = [None; 5];
        vision.apply(1.99, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
        assert_eq!(slots[1].unwrap().vision, VISION_ONLY);
        let mut slots = [None; 5];
        vision.apply(1.90, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
        assert_eq!(slots, [None; 5]);
    }
}
