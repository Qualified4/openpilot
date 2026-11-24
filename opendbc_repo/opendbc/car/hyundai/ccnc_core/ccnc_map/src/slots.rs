//! Assign FF, LF, RF, LR, RR from the reflection-corrected world lane coordinate.

use crate::display::{draw_road_frame, HeadingTerm};
use crate::pos::common::{Track, LANE_WIDTH, LEAD_GATE_X, LEAD_GATE_Y};
use crate::vehicles::{Link, Vehicle, MAX_VEHICLES};
use crate::world::{PoseAt, World};

const INNER_RADIUS: f64 = 0.48;
const OBSERVATION_MAX_AGE: f64 = 0.15;
const FF_PREFERENCE_M: f32 = 1.0;
/// The legacy selector's stop memory range ahead (`Consts::memory_max_x`); the map keeps LF and RF at x >= 0.
const MEMORY_MAX_X: f64 = 15.;
/// Two cars in one side lane: the nearer one, this close ahead or less, is drawn alongside in that side's rear slot.
const ALONGSIDE_ENTER_M: f64 = 5.0;
/// ... and back in the front slot beyond this.
const ALONGSIDE_EXIT_M: f64 = 6.0;
/// A split side stays split this long after its next car drops out ...
const ALONGSIDE_HOLD_S: f64 = 1.0;
/// ... and this long after its near car does: the radar barely sees within 5 m, and the tracker keeps a car 2 s.
const ALONGSIDE_NEAR_HOLD_S: f64 = 2.0;
const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slot {
    pub vid: u32,
    pub distance: f32,
    pub lateral: f32,
    pub detect: u8,
    pub lane_coord_rel: f32,
    pub v_rel: f32,
    pub kind: u8,
    pub kind_confirmed: bool,
    /// The map asks `ccnc_visiond` for a road camera answer about this slot's vid (FF, LF, RF only).
    pub kind_wanted: bool,
    /// What the wide camera adds: `VISION_*`.
    pub vision: u8,
}

/// The slot holds a radar car the camera has no say about.
pub const VISION_NONE: u8 = 0;
/// The slot holds a car seen by the wide camera alone (it filled an empty slot or replaced a farther radar car).
pub const VISION_ONLY: u8 = 1;
/// The slot holds a radar car, and the wide camera sees a car there too.
pub const VISION_CONFIRMS: u8 = 2;

#[derive(Clone, Copy)]
pub(crate) struct Observation {
    pub x: f64,
    /// Lane relative to ego from `Vehicle::u`, for the lateral and the rear slots.
    pub r: f64,
    /// Lane relative to ego that picks the front slot: from `Vehicle::u_slot`, or `r` while ego changes lanes, when
    /// the faster coordinate would move cars between slots early as ego crosses the line.
    pub r_slot: f64,
    pub speed: f64,
}

impl Observation {
    /// `pose` is where distance and side are measured from: `World::radar_pose` for radar cars.
    pub(crate) fn from_car(car: &Vehicle, world: &World, pose: PoseAt) -> Option<Self> {
        let (x, y) = pose.to_ego(f64::from(car.wx), f64::from(car.wy));
        let u_ego = world.u_ego();
        if !u_ego.is_finite() || !(-40. ..=80.).contains(&x) {
            return None;
        }
        let mapped = f64::from(car.u) - u_ego;
        // A valid world coordinate keeps the same reference across 5 m.
        let r = if !mapped.is_finite() && x.abs() < 5. {
            -(y - f64::from(car.off)) / LANE_WIDTH
        } else {
            mapped
        };
        let fast = f64::from(car.u_slot) - u_ego;
        let r_slot = if fast.is_finite() && !world.ego_changing() { fast } else { r };
        let (sin, cos) = pose.heading.sin_cos();
        let speed = cos * f64::from(car.vx) + sin * f64::from(car.vy);
        (r.is_finite() && speed.is_finite()).then_some(Self { x, r, r_slot, speed })
    }
}

struct Drawing {
    vid: u32,
    lane: i8,
    shown: Option<f64>,
    detect: u8,
    seen: f64,
    radar: Option<Track>,
}

impl Drawing {
    fn new(vid: u32, r: f64, now: f64) -> Self {
        Self {
            vid,
            lane: lane_at(r),
            shown: None,
            detect: 1,
            seen: now,
            radar: None,
        }
    }

    fn observe_lane(&mut self, r: f64) {
        let lane = lane_at(r);
        if (r - f64::from(lane)).abs() <= INNER_RADIUS {
            self.lane = lane;
        }
    }
}

fn lane_at(r: f64) -> i8 {
    (r + 0.5).floor() as i8
}

fn slot_index(x: f64, lane: i8) -> Option<usize> {
    match (x >= 0., lane) {
        (true, 0) => Some(0),
        (true, -1) => Some(1),
        (true, 1) => Some(2),
        (false, -1) => Some(3),
        (false, 1) => Some(4),
        _ => None,
    }
}

pub struct Assigner {
    generation: u32,
    drawings: Vec<Drawing>,
    previous_ff: Option<u32>,
    /// The last radar car drawn in LF and RF within the stop memory's range, with its world position from ego's pose
    /// at that time. A farther car passing through the slot cannot be remembered and does not replace it.
    side_last: [Option<(Slot, (f64, f64))>; 2],
    /// The second nearest radar car for LF and RF at the last update.
    runner_up: [Option<Slot>; 2],
    /// The car `split_alongside` last drew alongside in LR and RR.
    alongside: [Option<Slot>; 2],
    /// When a split side lost its next car or its near car, while it stays split without it.
    alongside_alone: [Option<f64>; 2],
}

impl Default for Assigner {
    fn default() -> Self {
        Self {
            generation: 0,
            drawings: Vec::with_capacity(MAX_VEHICLES),
            previous_ff: None,
            side_last: [None; 2],
            runner_up: [None; 2],
            alongside: [None; 2],
            alongside_alone: [None; 2],
        }
    }
}

/// A car the slots may draw from a measurement: one in the latest radar frame, at most `OBSERVATION_MAX_AGE` old.
fn current(world: &World, now: f64, stamp: f64) -> bool {
    stamp + EPS >= world.radar_stamp().unwrap_or(now) && (0. ..=OBSERVATION_MAX_AGE).contains(&(now - stamp))
}

fn shown_kind(car: &Vehicle) -> Option<crate::vehicles::Kind> {
    car.kind.filter(|k| car.link == Link::Sure || k.inherited)
}

/// The radar's own rules (the stop memory, a side pair) own their slots: where the wide camera replaced such a car,
/// it stays and the camera only confirms it. `radar` is the slots before the camera.
pub fn keep_radar(protected: [bool; 5], radar: &[Option<Slot>; 5], slots: &mut [Option<Slot>; 5]) {
    for (index, kept) in radar.iter().enumerate() {
        if let Some(kept) = kept.filter(|_| protected[index]) {
            if slots[index].is_some_and(|s| s.vision != VISION_NONE) {
                slots[index] = Some(Slot { vision: VISION_CONFIRMS, ..kept });
            }
        }
    }
}

impl Assigner {
    pub const HEADING_TERM: HeadingTerm = HeadingTerm::RoadFrame;

    /// A current per-vid estimate, before filtering for the five output slots.
    pub fn estimate_at(&self, now: f64, vid: u32) -> Option<f64> {
        let index = self.drawings.binary_search_by_key(&vid, |d| d.vid).ok()?;
        let drawing = &self.drawings[index];
        (drawing.seen == now).then_some(drawing.shown).flatten()
    }

    /// The lead is (dRel, yRel) at the radar sensor, lateral positive to the left.
    pub fn update(
        &mut self,
        now: f64,
        vehicles: &[Vehicle],
        world: &World,
        lead: Option<(f64, f64)>,
    ) -> [Option<Slot>; 5] {
        if self.generation != world.generation() {
            self.generation = world.generation();
            self.drawings.clear();
            self.previous_ff = None;
            self.side_last = [None; 2];
            self.alongside = [None; 2];
            self.alongside_alone = [None; 2];
        }
        if !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
            return [None; 5];
        }
        self.drawings.retain(|d| now - d.seen <= 2.);
        // Radar cars sit in the world at the pose RADAR_LAG before `now`. Measure their distance and side from that
        // same pose, so the slot distance is the radar's dRel rather than dRel minus ego's travel over RADAR_LAG.
        let radar = world.radar_pose(now);
        let yaw = world.yaw_at(now);
        let current = |stamp: f64| current(world, now, stamp);
        let followed = lead.and_then(|(dx, dy)| {
            vehicles
                .iter()
                .filter(|car| current(car.seen_radar) && eligible(car))
                .filter_map(|car| {
                    let (x, y) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
                    let (ex, ey) = ((x - dx).abs(), (y - dy).abs());
                    (ex <= LEAD_GATE_X
                        && ey <= LEAD_GATE_Y
                        && Observation::from_car(car, world, radar)
                            .is_some_and(|o| o.x >= 0. && o.r.abs() < 0.5))
                    .then_some((ex, ey, car.vid))
                })
                .min_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then(a.1.total_cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                })
                .map(|(_, _, vid)| vid)
        });

        // Keep a qualifying FF across small distance changes when no lead matches.
        let preferred_ff = followed.or(self.previous_ff);
        let mut best: [Option<Slot>; 5] = [None; 5];
        let mut second: [Option<Slot>; 2] = [None; 2];
        for car in vehicles {
            if !current(car.seen_radar.max(car.seen_vision)) {
                continue;
            }
            let Some(Observation { x, r, r_slot, speed }) = Observation::from_car(car, world, radar) else {
                continue;
            };
            let i = match self.drawings.binary_search_by_key(&car.vid, |d| d.vid) {
                Ok(i) => i,
                Err(i) if self.drawings.len() < MAX_VEHICLES => {
                    self.drawings.insert(i, Drawing::new(car.vid, r_slot, now));
                    i
                }
                Err(_) => continue,
            };
            let drawing = &mut self.drawings[i];
            drawing.seen = now;
            drawing.observe_lane(r_slot);
            let v_rel = speed - world.pose.speed();
            let opening_gap = ((v_rel - 0.5) / 1.5).clamp(0., 1.);
            let mut lateral_target = -LANE_WIDTH * r;
            if x >= 0. && car.radar_id.is_some() && current(car.seen_radar) {
                let (rx, ry) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
                let report = (now, rx, ry);
                let track = drawing.radar.get_or_insert_with(|| Track::new(now, report));
                track.step(Some(report), now, yaw);

                // Use the map for slot selection and the long-range drawing. A close,
                // current report also constrains changes from the fitted lane position.
                let weight = (0.7 + opening_gap * (1. - 0.7)) * ((40. - rx) / 12.).clamp(0., 1.);
                // Keep full reflection compensation at low speed. At traffic speed,
                // release less of the offset as the radar weight rises.
                let release = (0.75 - opening_gap * (0.75 - 0.5))
                    * ((speed - 6.) / 2.).clamp(0., 1.);
                let radar_y = track.y + release * track.b;
                lateral_target += weight * (radar_y - lateral_target);
            }
            let distance = x.abs();
            let near = ((40. - distance) / 12.).clamp(0., 1.);
            let play_distance = if drawing.lane == 0 {
                // Close and mid-range FF have enough noise to cross a 0.1 m CAN step.
                let extra = 18. * ((x - 10.) / 8.).clamp(0., 1.) * near;
                distance * (1. + 0.3 * near) + extra
            } else {
                distance
            };
            let lateral = draw_road_frame(
                &mut drawing.shown,
                play_distance,
                -lateral_target / LANE_WIDTH,
            );
            if !eligible(car) {
                continue;
            }
            let Some(slot) = slot_index(x, drawing.lane) else {
                continue;
            };
            if slot != 0 && (speed < -1. / 3.6 || car.oncoming) {
                continue;
            }
            if v_rel > 2. {
                drawing.detect = 1;
            } else if v_rel < -1. {
                drawing.detect = 2;
            }
            let kind = shown_kind(car);
            let candidate = Slot {
                vid: car.vid,
                distance: x as f32,
                lateral: lateral as f32,
                detect: if slot == 0 && speed < 3. {
                    2
                } else {
                    drawing.detect
                },
                lane_coord_rel: r as f32,
                v_rel: v_rel as f32,
                kind: kind.map_or(0, |k| k.class),
                kind_confirmed: kind.is_some_and(|k| k.confirmed_at.is_finite()),
                kind_wanted: false,
                vision: VISION_NONE,
            };
            let rank = |s: Slot| {
                let advantage = if slot == 0 && preferred_ff == Some(s.vid) {
                    FF_PREFERENCE_M
                } else {
                    0.
                };
                (s.distance.abs() - advantage, s.vid)
            };
            if best[slot].is_none_or(|old| rank(candidate) < rank(old)) {
                if slot == 1 || slot == 2 {
                    second[slot - 1] = best[slot];
                }
                best[slot] = Some(candidate);
            } else if (slot == 1 || slot == 2) && second[slot - 1].is_none_or(|old| rank(candidate) < rank(old)) {
                second[slot - 1] = Some(candidate);
            }
        }
        self.runner_up = second;
        self.previous_ff = best[0].map(|s| s.vid);
        let ego = world.pose.at(now);
        for side in 0..2 {
            if let Some(s) = best[side + 1].filter(|s| f64::from(s.distance) <= MEMORY_MAX_X) {
                self.side_last[side] = Some((s, ego.to_world(f64::from(s.distance), f64::from(s.lateral))));
            }
        }
        best
    }

    /// Two radar cars in one side lane, the nearer 0-5 m ahead (until 6 m): draw the nearer one alongside, at 0 m in
    /// that side's rear slot, and the next one in the front slot. It replaces any rear handover drawing there: the car
    /// alongside is nearer and measured. With one car the front slot keeps it until it falls behind and the handover
    /// takes it. Kind, camera mark, v_rel and lateral go with the vid. A split side stays split while one of its cars
    /// drops out: without the next car for up to 1 s, with the front slot empty, and the next car is drawn as soon as
    /// one is back; without a current report of the near car (radar coverage ends alongside) for up to 2 s, drawn at
    /// 0 m while the tracker keeps it and no other slot draws it, until the rear slot draws it itself. Both count from
    /// when the side first lost a car. Returns the sides split.
    pub fn split_alongside(&mut self, now: f64, vehicles: &[Vehicle], slots: &mut [Option<Slot>; 5]) -> [bool; 2] {
        let mut split = [false; 2];
        for side in 0..2 {
            let (front, rear) = (side + 1, side + 3);
            let kept = self.alongside[side];
            let held = slots[front].filter(|near| kept.is_some_and(|kept| kept.vid == near.vid));
            let limit = if held.is_some() { ALONGSIDE_EXIT_M } else { ALONGSIDE_ENTER_M };
            let near = slots[front].filter(|near| (0. ..=limit).contains(&f64::from(near.distance)));
            let next = self.runner_up[side].filter(|next| near.is_some_and(|near| near.vid != next.vid));
            let hold = |alone: &mut Option<f64>, limit: f64| now - *alone.get_or_insert(now) <= limit;
            let shown = match (near, next) {
                (Some(near), Some(next)) => {
                    self.alongside_alone[side] = None;
                    Some((near, Some(next)))
                }
                (Some(near), None) if held.is_some() => hold(&mut self.alongside_alone[side], ALONGSIDE_HOLD_S).then_some((near, None)),
                _ => kept
                    .filter(|kept| {
                        held.is_none()
                            && vehicles.iter().any(|car| car.vid == kept.vid)
                            && slots.iter().all(|slot| slot.is_none_or(|slot| slot.vid != kept.vid))
                    })
                    .filter(|_| hold(&mut self.alongside_alone[side], ALONGSIDE_NEAR_HOLD_S))
                    .map(|kept| (kept, slots[front])),
            };
            self.alongside[side] = shown.map(|(near, _)| near);
            if shown.is_none() {
                self.alongside_alone[side] = None;
            }
            if let Some((near, next)) = shown {
                slots[rear] = Some(Slot { distance: 0., ..near });
                slots[front] = next;
                split[side] = true;
            }
        }
        split
    }

    /// The vids `hold_stopped` may draw, for the tracker to keep while ego holds.
    pub fn remembered_vids(&self) -> [Option<u32>; 2] {
        self.side_last.map(|s| s.map(|(s, _)| s.vid))
    }

    /// The legacy selector's stop memory, by vid: while ego holds (stopped, or below 10 km/h and braking) and LF or
    /// RF is empty with no BSD on that side, keep the car last drawn there if the radar map would remember it
    /// (`remember`) and it is 0-15 m ahead. A new measurement of it, another car in the slot, or ego moving off
    /// ends it. Returns the sides drawn from memory.
    pub fn hold_stopped(
        &mut self,
        now: f64,
        vehicles: &[Vehicle],
        world: &World,
        blind: [bool; 2],
        remember: impl Fn(&Vehicle) -> bool,
        slots: &mut [Option<Slot>; 5],
    ) -> [bool; 2] {
        let mut remembered = [false; 2];
        if self.generation != world.generation() || !now.is_finite() || !world.accepts_radar() {
            return remembered;
        }
        let ego = world.pose.at(now);
        for side in 0..2 {
            let index = side + 1;
            let Some((last, position)) = self.side_last[side] else { continue };
            if slots[index].is_some() || blind[side] || slots.iter().flatten().any(|s| s.vid == last.vid) {
                continue;
            }
            let Some(car) = vehicles.iter().find(|car| car.vid == last.vid && !car.oncoming) else {
                continue;
            };
            // Carry the last drawn distance by ego's travel since. The car's world position sits RADAR_LAG behind:
            // measured from the lagged pose it would come out short by RADAR_LAG x the speed ego lost braking.
            let (x, _) = ego.to_ego(position.0, position.1);
            if current(world, now, car.seen_radar.max(car.seen_vision)) || !(0. ..=MEMORY_MAX_X).contains(&x) || !remember(car) {
                continue;
            }
            if let Ok(i) = self.drawings.binary_search_by_key(&car.vid, |d| d.vid) {
                self.drawings[i].seen = now;
            }
            let kind = shown_kind(car);
            slots[index] = Some(Slot {
                distance: x as f32,
                // A remembered car is at rest.
                v_rel: -world.pose.speed() as f32,
                kind: kind.map_or(0, |k| k.class),
                kind_confirmed: kind.is_some_and(|k| k.confirmed_at.is_finite()),
                vision: VISION_NONE,
                ..last
            });
            remembered[side] = true;
        }
        remembered
    }
}

fn eligible(car: &Vehicle) -> bool {
    !car.flags.structure && (car.flags.proven || car.flags.admitted || car.flags.motion_confirmed)
}

