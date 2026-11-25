//! Assign FF, LF, RF, LR, RR from the reflection-corrected world lane coordinate.

use crate::display::{draw_road_frame, HeadingTerm};
use crate::pos::common::{Track, LANE_WIDTH, LEAD_GATE_X, LEAD_GATE_Y};
use crate::radar::lanes::interpolate;
use crate::vehicles::{Link, Vehicle, MAX_VEHICLES};
use crate::world::{PoseAt, World};

const INNER_RADIUS: f64 = 0.48;
/// A side car moves on to the lane beyond (no slot) only this close to that lane's center: a car near the outer line
/// of a side lane went in and out of LF or RF while the map's line wobbled across it.
const OUTER_RADIUS: f64 = 0.35;
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
/// The near car is alongside only moving with ego, within this speed of it (m/s). A car ego passes, or one passing
/// ego, goes on from the front slot to the rear one: 0000047f 6 0-18.7 s, ego at 24-42 km/h past a queue creeping in
/// the right lane at 2-4.5 m/s, RR held each passed car at 0 m in 288 of 331 frames and jumped 0 -> 0.4-4 m -> 0 at
/// each of 12 vid changes. Measured over 47f and 47d (REPORT_RAIL_RR_KO): rear slots at 0 m while passing 62.1 -> 3.5 s
/// and 50.4 -> 0.4 s.
const ALONGSIDE_SPEED: f32 = 2.0;
/// ... and a split side stays split while its near car is within this speed of ego: 00000454 18, a car alongside at
/// -2.6 to -1.8 m/s split and joined 20 times in 4 s around an ego lane change.
const ALONGSIDE_SPEED_EXIT: f32 = 3.0;
/// A vid that once moved at this ground speed (m/s) or faster is a moving car: `EDGE_ROOM_M` does not apply to it.
const MOVED: f64 = 3.0;
/// A car's center sits about half its width or more inside the road edge. A side object that never moved at `MOVED`
/// is not drawn closer than this to the road edge on its outer side. 0000047d: the median beside ego in the first
/// lane, drawn in LF for 23 s while ego waited (15 32.4 s, 0.3-0.9 m from the left edge as the model's edge drifted
/// out); a leaf bag and two pedestrians on the right edge that the radar reported at -0.5 to +2.6 m/s, drawn in RF for
/// 0.05-0.5 s (21 10.39 s, 21 14.79 s, 22 15.45 s; 0.3 m). Cars that moved are left alone: the map's edge can sit
/// inside the real one, and the rule cut real cars at 10-15 m/s there (47c side flicker 90 -> 115). Slow cars inside
/// the road are left alone too: a ground speed rule (any side object below 3 m/s while ego drives) hid a queue of
/// real cars creeping in the right lane (00000479 12 51-59 s).
const EDGE_ROOM_M: f64 = 0.9;
/// The rule above judges a vid once it has been tracked this long: a car pulling out of an alley at the road
/// edge (0000047d 21 22.89 s) is drawn as it appears.
const SETTLED_S: f64 = 1.0;
/// A vid tracked shorter than `SETTLED_S` that never moved is not drawn where the edge rule hid a side object within
/// this long, this close along and across: 0000047f 5 17-24 s, a road edge object the radar reported as a new vid
/// five times in 6 s, each drawn in RF for its first 0.25-0.65 s.
const EDGE_PLACE_S: f64 = 2.0;
const EDGE_PLACE_X: f64 = 3.0;
const EDGE_PLACE_Y: f64 = 0.7;
/// FF does not draw a car measured coming at us (`Vehicle::oncoming`) this close while ego drives at
/// `ONCOMING_FF_EGO_SPEED` or more: 0000047d 1 9.75 s, a car park pillar the radar measured at -4 m/s, drawn in FF
/// for 1 s at 10 km/h. A car standing ahead is never oncoming; while ego waits FF still draws crossing traffic.
const ONCOMING_FF_M: f64 = 15.;
const ONCOMING_FF_EGO_SPEED: f64 = 2.;
/// The display lead qualifies the car it matches only moving like it: their speeds relative to ego this close (m/s).
/// 0000047f 1 56.0 s: radarState's lead jumped to 7.7 m with yRel 0, vRel 0 and radarTrackId 0 (going at ego's
/// 4.3 m/s), on a curve, and qualified a guard rail object standing there (ground speed 0.1 m/s, never proven) in FF for
/// 0.4 s. The leads that qualified real cars standing ahead measured them: 47f 12 the bus (vRel -6, the car -6), 47d 0
/// and 10-11 standing cars.
const LEAD_SPEED_MATCH: f64 = 2.;
const EPS: f64 = 1e-9;
/// `ModelLane`: the legacy selector's ego lane between the model's two ego lines (`_lane`, a153574e
/// hyundaicanfd_ccnc_radar_map.py 446-460), from lines this probable, this far ahead, this wide, with this margin
/// inside each line.
pub const MODEL_LANE_PROB: f64 = 0.5;
const MODEL_LANE_X: std::ops::RangeInclusive<f64> = 5. ..=50.;
const MODEL_LANE_WIDTH: std::ops::RangeInclusive<f64> = 1.8..=5.;
const MODEL_LANE_MARGIN: f64 = 0.3;

/// Ego's lane by the model's lines of this frame, placed at ego's pose now (lateral positive to the left). A radar car
/// inside it is drawn in FF and never in LF or RF, whatever lane the world map gives it. The world map's lane coordinate
/// drifts by half a lane and more on sharp curves, single lane ramps and far ahead: 0000047f 6 13-57 s, the car ahead
/// in ego's lane on a single lane ramp drawn in LF and RF at 41-65 m and in RF at 13-20 m, with the wall beside it.
/// Over 47f and 47d the map drew such a car in LF or RF 1.0 and 1.5 s per minute driven, the legacy selector 0.29.
#[derive(Clone, Debug, Default)]
pub struct ModelLane {
    left: Vec<(f64, f64)>,
    right: Vec<(f64, f64)>,
}

impl ModelLane {
    pub fn new(left: Vec<(f64, f64)>, right: Vec<(f64, f64)>) -> Self {
        Self { left, right }
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        if !MODEL_LANE_X.contains(&x) {
            return false;
        }
        let (Some(a), Some(b)) = (interpolate(&self.left, x), interpolate(&self.right, x)) else {
            return false;
        };
        MODEL_LANE_WIDTH.contains(&(a - b)) && b + MODEL_LANE_MARGIN < y && y < a - MODEL_LANE_MARGIN
    }
}

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
    /// The vid once moved at `MOVED` or faster.
    fast: bool,
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
            fast: false,
        }
    }

    fn observe_lane(&mut self, r: f64) {
        let lane = lane_at(r);
        let outward = self.lane.abs() == 1 && lane == 2 * self.lane;
        if (r - f64::from(lane)).abs() <= if outward { OUTER_RADIUS } else { INNER_RADIUS } {
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
    /// The car the front slot drew 0-5 m ahead without splitting the side, too fast for it (`ALONGSIDE_SPEED`).
    passing: [Option<Slot>; 2],
    /// Where (world) and when the edge rule last hid a side object.
    edge_hidden: Vec<(f64, f64, f64)>,
    /// `set_model_lane`.
    model_lane: Option<ModelLane>,
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
            passing: [None; 2],
            edge_hidden: Vec::new(),
            model_lane: None,
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

impl Assigner {
    pub const HEADING_TERM: HeadingTerm = HeadingTerm::RoadFrame;

    /// Ego's lane by the model for the next updates, or None while the model's lines are not to be trusted (missing,
    /// below `MODEL_LANE_PROB`, stale, or ego changing lanes): the world map's lane alone picks the slots then.
    pub fn set_model_lane(&mut self, lane: Option<ModelLane>) {
        self.model_lane = lane;
    }

    /// A current per-vid estimate, before filtering for the five output slots.
    pub fn estimate_at(&self, now: f64, vid: u32) -> Option<f64> {
        let index = self.drawings.binary_search_by_key(&vid, |d| d.vid).ok()?;
        let drawing = &self.drawings[index];
        (drawing.seen == now).then_some(drawing.shown).flatten()
    }

    /// The lead is (dRel, yRel) at the radar sensor, lateral positive to the left; it only prefers among the radar's
    /// proven cars.
    pub fn update(&mut self, now: f64, vehicles: &[Vehicle], world: &World, lead: Option<(f64, f64)>) -> [Option<Slot>; 5] {
        self.update_with(now, vehicles, world, lead, None)
    }

    /// `update` with radarState's display lead (`Event::display_lead`): the radar car at that lead may be drawn in FF
    /// though the radar has not proven it a vehicle by motion or Doppler, as card drew the control lead in FF before
    /// the map (`_apply_ccnc_lead`, 93c5c6af). A car or bus standing ahead in ego's lane: 0000047f 12 39.4 s, a bus
    /// stopped 18-13 m ahead and the lead, FF blank for 0.87 s until it rolled; 0000047d 10-11, a car standing 40-4 m
    /// ahead and the lead, FF blank for 59 s. FF only: LF and RF keep the radar's own proof. The lead is (dRel, yRel,
    /// vRel), and qualifies a car only at its speed (`LEAD_SPEED_MATCH`).
    pub fn update_lead(&mut self, now: f64, vehicles: &[Vehicle], world: &World, lead: Option<(f64, f64, f64)>) -> [Option<Slot>; 5] {
        self.update_with(now, vehicles, world, lead.map(|(x, y, _)| (x, y)), lead.map(|(_, _, v)| v))
    }

    fn update_with(
        &mut self,
        now: f64,
        vehicles: &[Vehicle],
        world: &World,
        lead: Option<(f64, f64)>,
        lead_v_rel: Option<f64>,
    ) -> [Option<Slot>; 5] {
        if self.generation != world.generation() {
            self.generation = world.generation();
            self.drawings.clear();
            self.previous_ff = None;
            self.side_last = [None; 2];
            self.alongside = [None; 2];
            self.alongside_alone = [None; 2];
            self.passing = [None; 2];
            self.edge_hidden.clear();
        }
        if !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
            return [None; 5];
        }
        self.drawings.retain(|d| now - d.seen <= 2.);
        self.edge_hidden.retain(|h| (0. ..=EDGE_PLACE_S).contains(&(now - h.0)));
        let ego = world.pose.at(now);
        // Radar cars sit in the world at the pose RADAR_LAG before `now`. Measure their distance and side from that
        // same pose, so the slot distance is the radar's dRel rather than dRel minus ego's travel over RADAR_LAG.
        let radar = world.radar_pose(now);
        let yaw = world.yaw_at(now);
        let current = |stamp: f64| current(world, now, stamp);
        let followed = lead.and_then(|(dx, dy)| {
            vehicles
                .iter()
                .filter(|car| current(car.seen_radar))
                .filter_map(|car| {
                    let (x, y) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
                    let (ex, ey) = ((x - dx).abs(), (y - dy).abs());
                    let o = Observation::from_car(car, world, radar)?;
                    let qualified = eligible(car)
                        || lead_v_rel.is_some_and(|v| (o.speed - world.pose.speed() - v).abs() <= LEAD_SPEED_MATCH);
                    (qualified && ex <= LEAD_GATE_X && ey <= LEAD_GATE_Y && o.x >= 0. && o.r.abs() < 0.5)
                        .then_some((ex, ey, car.vid))
                })
                .min_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then(a.1.total_cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                })
                .map(|(_, _, vid)| vid)
        });

        // Keep a qualifying FF across small distance changes when no lead matches. A truck can reach the map as two
        // radar tracks, two vids side by side, and the radar's lead moves between them: FF went back and forth 102
        // times in 38 s behind one (11-24 00000479 segment 10). The lead keeps the FF it moved from when the two are
        // one car (`twins`).
        let preferred_ff = match (followed, self.previous_ff) {
            (Some(lead), Some(previous)) if twins(vehicles, world, radar, current, lead, previous) => Some(previous),
            _ => followed.or(self.previous_ff),
        };
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
            drawing.fast |= speed.abs() >= MOVED;
            let still = now - car.born >= SETTLED_S && !drawing.fast;
            let v_rel = speed - world.pose.speed();
            let opening_gap = ((v_rel - 0.5) / 1.5).clamp(0., 1.);
            let mut lateral_target = -LANE_WIDTH * r;
            let measured = x >= 0. && car.radar_id.is_some() && current(car.seen_radar);
            let (rx, ry) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
            // A side lane car the model puts in ego's lane is drawn in FF where the radar measures it.
            let into_ff = measured
                && drawing.lane.abs() == 1
                && self.model_lane.as_ref().is_some_and(|lane| lane.contains(rx, ry));
            if into_ff {
                lateral_target = ry;
            }
            if measured {
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
            let the_lead = lead_v_rel.is_some() && followed == Some(car.vid);
            if !eligible(car) && !the_lead {
                continue;
            }
            let Some(slot) = (if into_ff { Some(0) } else { slot_index(x, drawing.lane) }) else {
                continue;
            };
            if slot != 0 && !eligible(car) {
                continue;
            }
            let edge = (1..=2).contains(&slot) && !drawing.fast && {
                if still {
                    at_edge(world, now, car, slot)
                } else {
                    let (cx, cy) = ego.to_ego(f64::from(car.wx), f64::from(car.wy));
                    self.edge_hidden.iter().any(|&(_, hx, hy)| {
                        let (hx, hy) = ego.to_ego(hx, hy);
                        (hx - cx).abs() <= EDGE_PLACE_X && (hy - cy).abs() <= EDGE_PLACE_Y
                    })
                }
            };
            if edge {
                self.edge_hidden.push((now, f64::from(car.wx), f64::from(car.wy)));
            }
            if slot != 0 && (speed < -1. / 3.6 || car.oncoming || edge) {
                continue;
            }
            if slot == 0 && car.oncoming && x <= ONCOMING_FF_M && world.pose.speed() >= ONCOMING_FF_EGO_SPEED {
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

    /// Two radar cars in one side lane, the nearer 0-5 m ahead (until 6 m) at ego's speed (`ALONGSIDE_SPEED`): draw the
    /// nearer one alongside, at 0 m in
    /// that side's rear slot, and the next one in the front slot. It replaces any rear handover drawing there: the car
    /// alongside is nearer and measured. With one car the front slot keeps it until it falls behind and the handover
    /// takes it. Kind, camera mark, v_rel and lateral go with the vid. A split side stays split while one of its cars
    /// drops out: without the next car for up to 1 s, with the front slot empty, and the next car is drawn as soon as
    /// one is back; without a current report of the near car (radar coverage ends alongside) for up to 2 s, drawn at
    /// 0 m while the tracker keeps it and no other slot draws it, until the rear slot draws it itself. Both count from
    /// when the side first lost a car. A car passing 0-5 m ahead in the front slot, too fast to split the side, is drawn
    /// in the rear slot at 0 m the same way once the radar stops reporting it alongside, keeping the front slot: the
    /// rear handover takes it 0.25-0.7 s later (00000454 18, 28, 34). Returns the sides split.
    pub fn split_alongside(&mut self, now: f64, vehicles: &[Vehicle], slots: &mut [Option<Slot>; 5]) -> [bool; 2] {
        let mut split = [false; 2];
        for side in 0..2 {
            let (front, rear) = (side + 1, side + 3);
            let kept = self.alongside[side];
            let held = slots[front].filter(|near| kept.is_some_and(|kept| kept.vid == near.vid));
            let (limit, speed) = if held.is_some() {
                (ALONGSIDE_EXIT_M, ALONGSIDE_SPEED_EXIT)
            } else {
                (ALONGSIDE_ENTER_M, ALONGSIDE_SPEED)
            };
            let close = slots[front].filter(|near| (0. ..=limit).contains(&f64::from(near.distance)));
            let near = close.filter(|near| near.v_rel.abs() <= speed);
            let next = self.runner_up[side].filter(|next| near.is_some_and(|near| near.vid != next.vid));
            let hold = |alone: &mut Option<f64>, limit: f64| now - *alone.get_or_insert(now) <= limit;
            let shown = match (near, next) {
                (Some(near), Some(next)) => {
                    self.alongside_alone[side] = None;
                    Some((near, Some(next)))
                }
                (Some(near), None) if held.is_some() => hold(&mut self.alongside_alone[side], ALONGSIDE_HOLD_S).then_some((near, None)),
                _ => kept
                    .or(self.passing[side])
                    .filter(|kept| {
                        held.is_none()
                            && vehicles.iter().any(|car| car.vid == kept.vid)
                            && slots.iter().all(|slot| slot.is_none_or(|slot| slot.vid != kept.vid))
                    })
                    .filter(|_| hold(&mut self.alongside_alone[side], ALONGSIDE_NEAR_HOLD_S))
                    .map(|kept| (kept, slots[front])),
            };
            self.alongside[side] = shown.map(|(near, _)| near);
            self.passing[side] = match close.filter(|_| near.is_none()) {
                Some(passing) => Some(passing),
                None => self.passing[side].filter(|_| shown.is_some()),
            };
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

/// Two current, eligible radar cars this close along, across (m) and in ground speed (m/s) are one car the radar sees
/// as two tracks. 11-24 00000479 segment 10: 0-0.5 m, 0.02-0.36 m, the same speed.
const TWIN_LON_M: f64 = 1.5;
const TWIN_LAT_M: f64 = 1.0;
const TWIN_SPEED: f64 = 1.0;

fn twins(vehicles: &[Vehicle], world: &World, radar: PoseAt, current: impl Fn(f64) -> bool, a: u32, b: u32) -> bool {
    let at = |vid: u32| {
        let car = vehicles.iter().find(|car| car.vid == vid && current(car.seen_radar) && eligible(car))?;
        let (x, y) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
        Some((x, y, Observation::from_car(car, world, radar)?.speed))
    };
    a != b
        && at(a).zip(at(b)).is_some_and(|((ax, ay, av), (bx, by, bv))| {
            (ax - bx).abs() <= TWIN_LON_M && (ay - by).abs() <= TWIN_LAT_M && (av - bv).abs() <= TWIN_SPEED
        })
}

/// The side car in LF (`slot` 1) or RF (2) is within `EDGE_ROOM_M` of the road edge beyond it: the map has no lane
/// coordinate that far further out.
fn at_edge(world: &World, now: f64, car: &Vehicle, slot: usize) -> bool {
    let ego = world.pose.at(now);
    let (x, y) = ego.to_ego(f64::from(car.wx), f64::from(car.wy));
    let outward = if slot == 1 { EDGE_ROOM_M } else { -EDGE_ROOM_M };
    world.vehicle_lane_coord(now, ego.s + x, ego.to_world(x, y + outward)).is_none()
}

fn eligible(car: &Vehicle) -> bool {
    !car.flags.structure && (car.flags.proven || car.flags.admitted || car.flags.motion_confirmed)
}

