//! Assign FF, LF, RF, LR, RR from the reflection-corrected world lane coordinate.

use crate::display::{draw_road_frame, HeadingTerm};
use crate::pos::common::{LEAD_GATE_X, LEAD_GATE_Y, PINGPONG_HOLD};
use crate::vehicles::{Link, Vehicle, MAX_VEHICLES};
use crate::world::{PoseAt, World};

const STRADDLE: f64 = 0.35;
const INNER_RADIUS: f64 = 0.4;
const LANE_HOLD: f64 = 0.3;
const OWNER_HOLD: f64 = 0.5;
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
}

#[derive(Clone, Copy)]
pub(crate) struct Observation {
    pub x: f64,
    pub width: f64,
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
        let width = world
            .lanes
            .width_at(ego.s + if x.abs() < 5. { 0. } else { x });
        let r = if x.abs() < 5. {
            -(y - f64::from(car.off)) / width
        } else {
            f64::from(car.u) - u_ego
        };
        let (sin, cos) = ego.heading.sin_cos();
        let speed = cos * f64::from(car.vx) + sin * f64::from(car.vy);
        (r.is_finite() && speed.is_finite()).then_some(Self { x, width, r, speed })
    }
}

struct Drawing {
    vid: u32,
    lane: i8,
    pending: Option<(i8, f64)>,
    shown: Option<f64>,
    detect: u8,
    seen: f64,
}

impl Drawing {
    fn new(vid: u32, r: f64, now: f64) -> Self {
        let lane = if r.abs() > STRADDLE && r.abs() < 0.5 {
            r.signum() as i8
        } else {
            lane_at(r)
        };
        Self {
            vid,
            lane,
            pending: None,
            shown: None,
            detect: 1,
            seen: now,
        }
    }

    fn observe_lane(&mut self, now: f64, r: f64) {
        let lane = lane_at(r);
        if lane == self.lane || (r - f64::from(lane)).abs() > INNER_RADIUS {
            self.pending = None;
        } else if let Some((target, since)) = self.pending.filter(|&(target, _)| target == lane) {
            if now - since + EPS >= LANE_HOLD {
                self.lane = target;
                self.pending = None;
            }
        } else {
            self.pending = Some((lane, now));
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

#[derive(Default)]
struct Owner {
    current: Option<u32>,
    previous: Option<u32>,
    since: f64,
}

impl Owner {
    fn select(
        &mut self,
        now: f64,
        mut best: Option<Slot>,
        retained: Option<Slot>,
        force: bool,
    ) -> Option<Slot> {
        if let Some(old) = retained.filter(|_| !force) {
            let age = now - self.since;
            if best.is_none()
                || age < OWNER_HOLD
                || (age + EPS < PINGPONG_HOLD && best.map(|s| s.vid) == self.previous)
            {
                best = Some(old);
            }
        }
        let next = best.map(|s| s.vid);
        if self.current != next {
            self.previous = self.current;
            self.current = next;
            self.since = now;
        }
        best
    }
}

pub struct Assigner {
    generation: u32,
    drawings: Vec<Drawing>,
    owners: [Owner; 5],
}

impl Default for Assigner {
    fn default() -> Self {
        Self {
            generation: 0,
            drawings: Vec::with_capacity(MAX_VEHICLES),
            owners: Default::default(),
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
            self.owners = Default::default();
        }
        if !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
            self.owners = Default::default();
            return [None; 5];
        }
        self.drawings.retain(|d| now - d.seen <= 2.);
        let ego = world.pose.at(now);
        let radar = world.radar_pose(now);
        let followed = lead.and_then(|(dx, dy)| {
            vehicles
                .iter()
                .filter(|car| (0. ..=0.15).contains(&(now - car.seen_radar)) && eligible(car))
                .filter_map(|car| {
                    let (x, y) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
                    let (ex, ey) = ((x - dx).abs(), (y - dy).abs());
                    (ex <= LEAD_GATE_X
                        && ey <= LEAD_GATE_Y
                        && Observation::from_car(car, world, ego).is_some_and(|o| o.x >= 0.))
                    .then_some((ex, ey, car.vid))
                })
                .min_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then(a.1.total_cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                })
                .map(|(_, _, vid)| vid)
        });

        let mut best: [Option<Slot>; 5] = [None; 5];
        let mut retained = [None; 5];
        for car in vehicles {
            let age = now - car.seen_radar.max(car.seen_vision);
            if !(0. ..=OWNER_HOLD + EPS).contains(&age) {
                continue;
            }
            let Some(Observation { x, width, r, speed }) = Observation::from_car(car, world, ego)
            else {
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
            drawing.observe_lane(now, r);
            let lateral = draw_road_frame(&mut drawing.shown, x, width, r);
            if !eligible(car) {
                continue;
            }
            let is_lead = followed == Some(car.vid);
            let lane = if is_lead { 0 } else { drawing.lane };
            let Some(slot) = slot_index(x, lane) else {
                continue;
            };
            if slot != 0 && speed < -1. / 3.6 {
                continue;
            }
            let v_rel = speed - world.pose.speed();
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
            };
            if self.owners[slot].current == Some(car.vid) {
                retained[slot] = Some(candidate);
            }
            if is_lead
                || (age <= 0.15
                    && best[slot].is_none_or(|old| {
                        followed != Some(old.vid)
                            && (candidate.distance.abs(), candidate.vid)
                                < (old.distance.abs(), old.vid)
                    }))
            {
                best[slot] = Some(candidate);
            }
        }
        wide_rules(&mut best);
        std::array::from_fn(|i| {
            self.owners[i].select(now, best[i], retained[i], i == 0 && followed.is_some())
        })
    }
}

fn eligible(car: &Vehicle) -> bool {
    !car.flags.structure && (car.flags.proven || car.flags.admitted || car.flags.motion_confirmed)
}

fn wide_rules(_slots: &mut [Option<Slot>; 5]) {}
