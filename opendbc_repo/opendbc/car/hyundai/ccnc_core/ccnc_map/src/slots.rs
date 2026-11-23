//! Assign FF, LF, RF, LR, RR from the reflection-corrected world lane coordinate.

use crate::display::{draw_road_frame, HeadingTerm};
use crate::pos::common::{Track, LANE_WIDTH, LEAD_GATE_X, LEAD_GATE_Y};
use crate::vehicles::{Link, Vehicle, MAX_VEHICLES};
use crate::world::{PoseAt, World};

const INNER_RADIUS: f64 = 0.48;
const OBSERVATION_MAX_AGE: f64 = 0.15;
const FF_PREFERENCE_M: f32 = 1.0;
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
    pub r: f64,
    pub speed: f64,
}

impl Observation {
    pub(crate) fn from_car(car: &Vehicle, world: &World, ego: PoseAt) -> Option<Self> {
        let (x, y) = ego.to_ego(f64::from(car.wx), f64::from(car.wy));
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
        let (sin, cos) = ego.heading.sin_cos();
        let speed = cos * f64::from(car.vx) + sin * f64::from(car.vy);
        (r.is_finite() && speed.is_finite()).then_some(Self { x, r, speed })
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
}

impl Default for Assigner {
    fn default() -> Self {
        Self {
            generation: 0,
            drawings: Vec::with_capacity(MAX_VEHICLES),
            previous_ff: None,
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
        }
        if !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
            return [None; 5];
        }
        self.drawings.retain(|d| now - d.seen <= 2.);
        let ego = world.pose.at(now);
        let radar = world.radar_pose(now);
        let yaw = world.yaw_at(now);
        let radar_stamp = world.radar_stamp().unwrap_or(now);
        let current = |stamp: f64| {
            stamp + EPS >= radar_stamp && (0. ..=OBSERVATION_MAX_AGE).contains(&(now - stamp))
        };
        let followed = lead.and_then(|(dx, dy)| {
            vehicles
                .iter()
                .filter(|car| current(car.seen_radar) && eligible(car))
                .filter_map(|car| {
                    let (x, y) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
                    let (ex, ey) = ((x - dx).abs(), (y - dy).abs());
                    (ex <= LEAD_GATE_X
                        && ey <= LEAD_GATE_Y
                        && Observation::from_car(car, world, ego)
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
        for car in vehicles {
            if !current(car.seen_radar.max(car.seen_vision)) {
                continue;
            }
            let Some(Observation { x, r, speed }) = Observation::from_car(car, world, ego) else {
                continue;
            };
            let i = match self.drawings.binary_search_by_key(&car.vid, |d| d.vid) {
                Ok(i) => i,
                Err(i) if self.drawings.len() < MAX_VEHICLES => {
                    self.drawings.insert(i, Drawing::new(car.vid, r, now));
                    i
                }
                Err(_) => continue,
            };
            let drawing = &mut self.drawings[i];
            drawing.seen = now;
            drawing.observe_lane(r);
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
            if slot != 0 && speed < -1. / 3.6 {
                continue;
            }
            if v_rel > 2. {
                drawing.detect = 1;
            } else if v_rel < -1. {
                drawing.detect = 2;
            }
            let kind = car.kind.filter(|_| car.link == Link::Sure);
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
                best[slot] = Some(candidate);
            }
        }
        self.previous_ff = best[0].map(|s| s.vid);
        best
    }
}

fn eligible(car: &Vehicle) -> bool {
    !car.flags.structure && (car.flags.proven || car.flags.admitted || car.flags.motion_confirmed)
}

