//! Assign FF, LF, RF, LR, RR from the reflection-corrected world lane coordinate.

use crate::display::{draw_road_frame, HeadingTerm, ModelPath};
use crate::input::map_events::DisplayLead;
use crate::pos::common::{Track, LANE_WIDTH, LEAD_GATE_X, LEAD_GATE_Y};
use crate::radar::lanes::interpolate;
use crate::vehicles::{Link, Vehicle, MAX_VEHICLES};
use crate::world::{PoseAt, World};

const INNER_RADIUS: f64 = 0.48;
/// A side car moves on to the lane beyond (no slot) only this close to that lane's center: a car near the outer line
/// of a side lane went in and out of LF or RF while the map's line wobbled across it.
const OUTER_RADIUS: f64 = 0.35;
pub(crate) const OBSERVATION_MAX_AGE: f64 = 0.15;
const FF_PREFERENCE_M: f32 = 1.0;
pub(crate) const FF_CONFIRM_S: f64 = 0.35;
const FF_HOLD_S: f64 = 1.0;
const FF_BOUNDARY_MAX_EGO_SPEED: f64 = 0.5;
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
const EDGE_MERGE_M: f64 = 0.5;
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
pub(crate) const LEAD_SPEED_MATCH: f64 = 2.;
const UNPROVEN_LEAD_MIN_PROB: f64 = 0.6;
/// A lead first matched in the radar's near gap can be a crossing car or a parked object.
const UNPROVEN_LEAD_ENTRY_MIN_X: f64 = 5.;
/// At parking speeds a first match also needs reliable lane geometry at the radar position (m/s).
const UNPROVEN_LEAD_ENTRY_CREEP_SPEED: f64 = 2.;
const EPS: f64 = 1e-9;
/// `ModelLane`: the legacy selector's ego lane between the model's two ego lines (`_lane`, a153574e
/// hyundaicanfd_ccnc_radar_map.py 446-460), from lines this probable, this far ahead, this wide, with this margin
/// inside each line.
pub const MODEL_LANE_PROB: f64 = 0.5;
const MODEL_LANE_X: std::ops::RangeInclusive<f64> = 5. ..=50.;
const MODEL_LANE_WIDTH: std::ops::RangeInclusive<f64> = 1.8..=5.;
const MODEL_LANE_MARGIN: f64 = 0.3;
const MODEL_LANE_KEEP_PROB: f64 = 0.45;
const MODEL_LANE_KEEP_X: std::ops::RangeInclusive<f64> = 4. ..=55.;

/// Ego's lane by the model's lines of this frame, placed at ego's pose now (lateral positive to the left). A radar car
/// inside it is drawn in FF and never in LF or RF, whatever lane the world map gives it. The world map's lane coordinate
/// drifts by half a lane and more on sharp curves, single lane ramps and far ahead: 0000047f 6 13-57 s, the car ahead
/// in ego's lane on a single lane ramp drawn in LF and RF at 41-65 m and in RF at 13-20 m, with the wall beside it.
/// Over 47f and 47d the map drew such a car in LF or RF 1.0 and 1.5 s per minute driven, the legacy selector 0.29.
#[derive(Clone, Debug, Default)]
pub struct ModelLane {
    left: Vec<(f64, f64)>,
    right: Vec<(f64, f64)>,
    probability: f64,
}

impl ModelLane {
    pub fn new(left: Vec<(f64, f64)>, right: Vec<(f64, f64)>) -> Self {
        Self::with_probability(left, right, 1.)
    }

    pub(crate) fn with_probability(left: Vec<(f64, f64)>, right: Vec<(f64, f64)>, probability: f64) -> Self {
        Self { left, right, probability }
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.contains_for(x, y, false)
    }

    fn contains_for(&self, x: f64, y: f64, already_inside: bool) -> bool {
        let (range, prob, margin) = if already_inside {
            (&MODEL_LANE_KEEP_X, MODEL_LANE_KEEP_PROB, 0.)
        } else {
            (&MODEL_LANE_X, MODEL_LANE_PROB, MODEL_LANE_MARGIN)
        };
        if !range.contains(&x) || !(self.probability >= prob) {
            return false;
        }
        let (Some(a), Some(b)) = (interpolate(&self.left, x), interpolate(&self.right, x)) else {
            return false;
        };
        MODEL_LANE_WIDTH.contains(&(a - b)) && b + margin < y && y < a - margin
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

struct Followed {
    rank: (f64, f64, u32),
    slot: Slot,
    corroborated: bool,
    new_unproven: bool,
    needs_confirmation: bool,
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
    road_lane: Option<i32>,
    shown: Option<f64>,
    bend_shown: Option<f64>,
    detect: u8,
    seen: f64,
    radar: Option<Track>,
    into_ff: bool,
    ff_change_since: Option<f64>,
    lane_entry_since: Option<f64>,
    /// The vid once moved at `MOVED` or faster.
    fast: bool,
}

impl Drawing {
    fn new(vid: u32, r: f64, now: f64) -> Self {
        Self {
            vid,
            lane: lane_at(r),
            road_lane: None,
            shown: None,
            bend_shown: None,
            detect: 1,
            seen: now,
            radar: None,
            into_ff: false,
            ff_change_since: None,
            lane_entry_since: None,
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

    fn update_model_ff(&mut self, now: f64, can_hold: bool, lane: Option<&ModelLane>, x: f64, y: f64, confirmed: bool) {
        if !can_hold {
            self.into_ff = false;
            self.ff_change_since = None;
            return;
        }
        let inside = lane.is_some_and(|lane| lane.contains_for(x, y, self.into_ff));
        if inside == self.into_ff {
            self.ff_change_since = None;
        } else {
            let since = *self.ff_change_since.get_or_insert(now);
            if (inside && confirmed) || now - since >= if inside { FF_CONFIRM_S } else { FF_HOLD_S } {
                self.into_ff = inside;
                self.ff_change_since = None;
            }
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

const ROAD_LANE_HYSTERESIS: f64 = 0.06;
const ROAD_HANDOFF_S: f64 = 0.85;

/// Schmitt trigger around the shared integer boundaries, in units of lane width.
fn road_lane(prior: Option<i32>, u: f64) -> Option<i32> {
    u.is_finite().then(|| {
        let floor = u.floor() as i32;
        prior
            .filter(|&lane| u >= f64::from(lane) - ROAD_LANE_HYSTERESIS && u < f64::from(lane + 1) + ROAD_LANE_HYSTERESIS)
            .unwrap_or(floor)
    })
}

#[derive(Clone, Copy)]
struct Plan {
    cost: f64,
    slots: [Option<Slot>; 5],
}

/// A 5-bit mask keeps one exact optimum per occupied subset. Each row (vid) is visited once.
fn match_slots(
    candidates: &[(Slot, [bool; 5])],
    ff: Option<Slot>,
    required: [bool; 5],
    last: [Option<u32>; 5],
) -> [Option<Slot>; 5] {
    const CENTRES: [f64; 5] = [0., -1., 1., -1., 1.];
    const UNFILLED: [f64; 5] = [10., 4., 4., 2., 2.];
    let base_cost: f64 = UNFILLED.iter().sum();
    let empty = Plan { cost: base_cost, slots: [None; 5] };
    let mut plans = [Plan { cost: f64::INFINITY, slots: [None; 5] }; 32];
    plans[0] = empty;
    for &(car, allowed) in candidates {
        for mask in (0..32usize).rev() {
            if !plans[mask].cost.is_finite() {
                continue;
            }
            for i in 0..5 {
                let bit = 1usize << i;
                let candidate = if i == 0 {
                    ff.filter(|s| s.vid == car.vid)
                } else {
                    allowed[i].then_some(car)
                };
                let Some(candidate) = candidate.filter(|_| mask & bit == 0) else {
                    continue;
                };
                let centre = f64::from(candidate.lane_coord_rel) - CENTRES[i];
                let changed = last.iter().position(|&v| v == Some(candidate.vid)).is_some_and(|j| i != j);
                let cost = plans[mask].cost - UNFILLED[i]
                    + centre * centre
                    + 0.004 * f64::from(candidate.distance.abs())
                    + if changed { 0.08 } else { 0. };
                if cost < plans[mask | bit].cost {
                    let mut slots = plans[mask].slots;
                    slots[i] = Some(candidate);
                    plans[mask | bit] = Plan { cost, slots };
                }
            }
        }
    }
    plans
        .into_iter()
        .filter(|p| required.iter().zip(p.slots).all(|(&needed, slot)| !needed || slot.is_some()))
        .min_by(|a, b| a.cost.total_cmp(&b.cost))
        .unwrap_or(empty)
        .slots
}

pub struct Assigner {
    last_commit_seen: Option<f64>,
    matching_last: [Option<u32>; 5],
    generation: u32,
    drawings: Vec<Drawing>,
    previous_ff: Option<u32>,
    ff_lead: Option<(u32, f64)>,
    ff_lead_pending: Option<(u32, f64)>,
    measured_lead_since: Option<(u32, f64)>,
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
    /// A passed car already in the radar gap can remain there while the next front car closes in.
    passing_hold: [Option<Slot>; 2],
    /// Where (world) and when the edge rule last hid a side object.
    edge_hidden: Vec<(f64, f64, f64)>,
    /// `set_model_lane`.
    model_lane: Option<ModelLane>,
    /// Last shown lateral kept only across consecutive reports of the same opening side car.
    side_lateral: [Option<(u32, f64, f32)>; 2],
}

impl Default for Assigner {
    fn default() -> Self {
        Self {
            last_commit_seen: None,
            matching_last: [None; 5],
            generation: 0,
            drawings: Vec::with_capacity(MAX_VEHICLES),
            previous_ff: None,
            ff_lead: None,
            ff_lead_pending: None,
            measured_lead_since: None,
            side_last: [None; 2],
            runner_up: [None; 2],
            alongside: [None; 2],
            alongside_alone: [None; 2],
            passing: [None; 2],
            passing_hold: [None; 2],
            edge_hidden: Vec::new(),
            model_lane: None,
            side_lateral: [None; 2],
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

    /// Ego's lane by the model for the next updates. An existing promotion can ride out a brief loss of the lines
    /// while the car is still measured. During an ego lane change the world map picks the slots.
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
        self.update_with(now, vehicles, world, lead, None, None, None, None)
    }

    /// `update` with the display lead's motion: a radar car at that lead may be drawn in FF
    /// though the radar has not proven it a vehicle by motion or Doppler, as card drew the control lead in FF before
    /// the map (`_apply_ccnc_lead`, 93c5c6af). A car or bus standing ahead in ego's lane: 0000047f 12 39.4 s, a bus
    /// stopped 18-13 m ahead and the lead, FF blank for 0.87 s until it rolled; 0000047d 10-11, a car standing 40-4 m
    /// ahead and the lead, FF blank for 59 s. FF only: LF and RF keep the radar's own proof. The lead is (dRel, yRel,
    /// vRel), and qualifies a car only at its speed (`LEAD_SPEED_MATCH`).
    pub fn update_lead(&mut self, now: f64, vehicles: &[Vehicle], world: &World, lead: Option<(f64, f64, f64)>) -> [Option<Slot>; 5] {
        self.update_with(now, vehicles, world, lead.map(|(x, y, _)| (x, y)), lead.map(|(_, _, v)| v), None, None, None)
    }

    pub(crate) fn update_display_lead(
        &mut self, now: f64, vehicles: &[Vehicle], world: &World, lead: Option<DisplayLead>,
        lead_one: Option<(DisplayLead, bool)>,
        bend: Option<(&ModelPath, f64)>,
    ) -> [Option<Slot>; 5] {
        self.update_with(
            now, vehicles, world, lead.map(|lead| (lead.motion.0, lead.motion.1)),
            lead.map(|lead| lead.motion.2), lead, lead_one, bend,
        )
    }

    fn update_with(
        &mut self,
        now: f64,
        vehicles: &[Vehicle],
        world: &World,
        lead: Option<(f64, f64)>,
        lead_v_rel: Option<f64>,
        display_lead: Option<DisplayLead>,
        lead_one: Option<(DisplayLead, bool)>,
        bend: Option<(&ModelPath, f64)>,
    ) -> [Option<Slot>; 5] {
        if self.generation != world.generation() {
            self.generation = world.generation();
            self.drawings.clear();
            self.last_commit_seen = None;
            self.matching_last = [None; 5];
            self.previous_ff = None;
            self.ff_lead = None;
            self.ff_lead_pending = None;
            self.measured_lead_since = None;
            self.side_last = [None; 2];
            self.clear_split();
            self.edge_hidden.clear();
            self.side_lateral = [None; 2];
        }
        if !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
            self.matching_last = [None; 5];
            self.ff_lead = None;
            self.ff_lead_pending = None;
            self.measured_lead_since = None;
            self.clear_split();
            return [None; 5];
        }
        self.drawings.retain(|d| now - d.seen <= 2.);
        self.edge_hidden.retain(|h| (0. ..=EDGE_PLACE_S).contains(&(now - h.0)));
        let lead_track = display_lead.and_then(|lead| lead.measured_track);
        let supported_lead = display_lead.filter(|lead| (UNPROVEN_LEAD_MIN_PROB..=1.).contains(&lead.model_prob));
        self.measured_lead_since = lead_track.map(|id| match self.measured_lead_since {
            Some((track_id, since)) if id == track_id => (id, since),
            _ => (id, now),
        });
        let stable_radar_lead = self.measured_lead_since
            .filter(|&(_, since)| now - since >= FF_CONFIRM_S)
            .map(|(id, _)| id);
        let ego = world.pose.at(now);
        // Radar cars sit in the world at the pose RADAR_LAG before `now`. Measure their distance and side from that
        // same pose, so the slot distance is the radar's dRel rather than dRel minus ego's travel over RADAR_LAG.
        let radar = world.radar_pose(now);
        let yaw = world.yaw_at(now);
        let current = |stamp: f64| current(world, now, stamp);
        let (sin, cos) = ego.heading.sin_cos();
        let mut followed: Option<Followed> = None;
        let mut confirmed_lead = None;
        let mut previous = None;
        let mut best: [Option<Slot>; 5] = [None; 5];
        let mut second: [Option<Slot>; 2] = [None; 2];
        let changing = world.ego_changing();
        let use_road = world.pose.speed() > 2.
            && world.status() & crate::world::STATUS_NO_LIVE_POSE == 0
            && !world.lane_held();
        let ego_committed = use_road.then(|| world.committed_lane()).flatten();
        let commit_info = world.last_commit();
        let commit_stamp = commit_info.map(|(ct, _, _)| ct);
        let post_commit = ego_committed.is_some()
            && commit_stamp.is_some_and(|ct| (0. ..=ROAD_HANDOFF_S).contains(&(now - ct)));
        let approaching = ego_committed.is_some() && world.approaching_boundary();
        if commit_stamp.is_some() && commit_stamp != self.last_commit_seen {
            self.last_commit_seen = commit_stamp;
            if use_road {
                self.side_last = [None; 2];
                self.clear_split();
                self.side_lateral = [None; 2];
            }
        }
        let mut matching = post_commit.then(Vec::new);
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
            if now - drawing.seen > OBSERVATION_MAX_AGE {
                drawing.into_ff = false;
                drawing.ff_change_since = None;
                drawing.lane_entry_since = None;
                drawing.road_lane = None;
                drawing.bend_shown = None;
            }
            drawing.seen = now;
            let original_lane = drawing.lane;
            drawing.observe_lane(r_slot);
            let proven_vehicle = eligible(car);
            let mut projection = None;
            if let Some(ego_lane) = ego_committed.filter(|_| car.u.is_finite()) {
                drawing.road_lane = road_lane(drawing.road_lane, f64::from(car.u));
                projection = drawing.road_lane.map(|l| (l - ego_lane).clamp(-3, 3) as i8);
            } else {
                drawing.road_lane = None;
            }
            let car_centered = car.u.is_finite()
                && (f64::from(car.u) - (f64::from(car.u).floor() + 0.5)).abs() <= 0.38;
            let is_prev_ff = self.previous_ff == Some(car.vid);
            let proj_safe_visibility = projection.is_some_and(|p| p.abs() <= 1 || r.abs() >= 1.35);
            let use_projection = (post_commit || approaching)
                && proven_vehicle
                && (-20. ..=50.).contains(&x)
                && car_centered
                && proj_safe_visibility
                && (post_commit
                    || (is_prev_ff && projection == Some(0) && r_slot.abs() <= 0.85));
            let measured = x >= 0. && car.radar_id.is_some() && current(car.seen_radar);
            let (rx, ry) = radar.to_ego(f64::from(car.wx), f64::from(car.wy));
            let v_rel = speed - world.pose.speed();
            let matching_track = measured && lead_track.is_some() && car.radar_id == lead_track;
            let confirmed_position = matching_track && lead.zip(lead_v_rel).is_some_and(|((dx, dy), dv)| {
                (rx - dx).abs() <= LEAD_GATE_X && (ry - dy).abs() <= LEAD_GATE_Y && (v_rel - dv).abs() <= LEAD_SPEED_MATCH
            });
            let entering_ff = original_lane.abs() == 1 && drawing.lane == 0;
            let current_lead_support = supported_lead.is_some() && confirmed_position;
            let model_boundary_entry = entering_ff && measured && proven_vehicle && !changing
                && self.model_lane.as_ref().is_some_and(|lane| lane.contains(rx, ry));
            // At low speed a moving map boundary can briefly put a stopped side car in FF.
            let boundary_entry = entering_ff && measured && !changing && world.pose.speed().abs() < FF_BOUNDARY_MAX_EGO_SPEED
                && ry.abs() >= LANE_WIDTH / 2. && !confirmed_position && !model_boundary_entry;
            if boundary_entry {
                if now - *drawing.lane_entry_since.get_or_insert(now) < FF_CONFIRM_S {
                    drawing.lane = original_lane;
                }
            } else {
                drawing.lane_entry_since = None;
            }
            let lane = if use_projection {
                let p = projection.unwrap_or(drawing.lane);
                if post_commit {
                    // Sync drawing.lane so post_commit expiry cannot snap back to a stale pre-commit lane.
                    drawing.lane = p;
                }
                p
            } else {
                drawing.lane
            };
            drawing.fast |= speed.abs() >= MOVED;
            let still = now - car.born >= SETTLED_S && !drawing.fast;
            let opening_gap = ((v_rel - 0.5) / 1.5).clamp(0., 1.);
            let mut lateral_target = -LANE_WIDTH * r;
            // A side lane car the model puts in ego's lane is drawn in FF where the radar measures it.
            let can_hold = measured && !changing && lane.abs() == 1 && MODEL_LANE_KEEP_X.contains(&rx);
            let confirmed_boundary = confirmed_position && (self.previous_ff == Some(car.vid) || r_slot.abs() <= 0.5);
            drawing.update_model_ff(now, can_hold, self.model_lane.as_ref(), rx, ry, confirmed_boundary);
            if drawing.into_ff {
                lateral_target = ry;
            }
            let mut bend_offset = None;
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
                if let Some((path, blend)) = bend.filter(|_| weight > 0. && !drawing.into_ff) {
                    bend_offset = path.offset_at(rx).map(|offset| -weight * blend * offset);
                }
            }
            let distance = x.abs();
            let near = ((40. - distance) / 12.).clamp(0., 1.);
            let play_distance = if lane == 0 || drawing.into_ff {
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
            if let Some(offset) = bend_offset {
                drawing.bend_shown.get_or_insert(lateral);
                draw_road_frame(
                    &mut drawing.bend_shown,
                    play_distance,
                    -(lateral_target + offset) / LANE_WIDTH,
                );
            } else {
                drawing.bend_shown = None;
            }
            let Some(slot) = (if drawing.into_ff { Some(0) } else { slot_index(x, lane) }) else {
                continue;
            };
            if car.flags.structure || (slot != 0 && !proven_vehicle) {
                continue;
            }
            if (1..=2).contains(&slot) && !drawing.fast {
                let (wx, wy) = (f64::from(car.wx), f64::from(car.wy));
                let at_edge = still && {
                    let outward = if slot == 1 { EDGE_ROOM_M } else { -EDGE_ROOM_M };
                    let along = cos * (wx - ego.x) + sin * (wy - ego.y);
                    world.vehicle_lane_coord(now, ego.s + along, (wx - sin * outward, wy + cos * outward)).is_none()
                };
                if at_edge {
                    remember_edge(&mut self.edge_hidden, now, wx, wy);
                    continue;
                }
                if !still && self.edge_hidden.iter().any(|&(_, hx, hy)| {
                    let (dx, dy) = (hx - wx, hy - wy);
                    (cos * dx + sin * dy).abs() <= EDGE_PLACE_X && (cos * dy - sin * dx).abs() <= EDGE_PLACE_Y
                }) {
                    continue;
                }
            }
            if slot != 0 && (speed < -1. / 3.6 || car.oncoming) {
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
            if let Some(candidates) = matching.as_mut() {
                let side_ok = proven_vehicle && !car.oncoming && speed >= -1. / 3.6;
                let continuing = x.abs() <= 50. && self.matching_last.contains(&Some(car.vid));
                let near_side = |sign: f64| side_ok && continuing && (0.40..=1.58).contains(&(sign * r));
                let allowed = [
                    false,
                    slot == 1 || (x >= 0. && near_side(-1.)),
                    slot == 2 || (x >= 0. && near_side(1.)),
                    slot == 3 || (x < 0. && near_side(-1.)),
                    slot == 4 || (x < 0. && near_side(1.)),
                ];
                candidates.push((candidate, allowed));
            }
            if slot == 0 {
                let held_lead = self.ff_lead.is_some_and(|(vid, _)| vid == car.vid);
                let primary_match = lead_one.filter(|(one, _)| {
                    let (dx, dy, dv) = one.motion;
                    measured && one.measured_track.is_some() && car.radar_id == one.measured_track
                        && (UNPROVEN_LEAD_MIN_PROB..=1.).contains(&one.model_prob)
                        && (rx - dx).abs() <= LEAD_GATE_X && (ry - dy).abs() <= LEAD_GATE_Y
                        && (v_rel - dv).abs() <= LEAD_SPEED_MATCH
                });
                let new_unproven = !proven_vehicle && self.previous_ff != Some(car.vid);
                if measured && (proven_vehicle || current_lead_support || primary_match.is_some()) && held_lead
                    && !new_unproven
                {
                    confirmed_lead = Some(candidate);
                }
                let in_lane = !proven_vehicle
                    && self.model_lane.as_ref().is_some_and(|lane| lane.contains(rx, ry));
                let unproven_entry = primary_match.is_some() && rx >= UNPROVEN_LEAD_ENTRY_MIN_X
                    && (world.pose.speed().abs() > UNPROVEN_LEAD_ENTRY_CREEP_SPEED || in_lane);
                let primary = primary_match.filter(|_| !proven_vehicle && (new_unproven || !current_lead_support));
                let (position, velocity, matching_track) = if let Some((primary, _)) = primary {
                    let (dx, dy, dv) = primary.motion;
                    (Some((dx, dy)), Some(dv), true)
                } else {
                    (lead, lead_v_rel, matching_track)
                };
                if let Some((dx, dy)) = position.filter(|_| current(car.seen_radar)) {
                    let (ex, ey) = ((rx - dx).abs(), (ry - dy).abs());
                    let rank = (ex, ey, car.vid);
                    let speed_matches = velocity.is_some_and(|v| (v_rel - v).abs() <= LEAD_SPEED_MATCH);
                    let supported = if new_unproven {
                        unproven_entry
                    } else {
                        (held_lead && (current_lead_support || primary_match.is_some())) || unproven_entry
                    };
                    let qualifies = proven_vehicle || supported || (display_lead.is_none() && speed_matches);
                    if qualifies && ex <= LEAD_GATE_X && ey <= LEAD_GATE_Y
                        && followed.as_ref().is_none_or(|old| rank < old.rank)
                    {
                        // A single matched update can be a static reflection, even with a good model lead.
                        let previously_matched = self.ff_lead_pending
                            .is_some_and(|(vid, since)| vid == car.vid && since < now);
                        let corroborated = (proven_vehicle || previously_matched) && matching_track && speed_matches
                            && (stable_radar_lead.is_some() || now - car.born >= FF_CONFIRM_S
                                || (ex <= 1. && ey <= 0.5));
                        let new_unproven = new_unproven && display_lead.is_some();
                        // Incomplete lane evidence requires a sustained measured LeadOne match at road speeds.
                        let needs_confirmation = new_unproven && !in_lane && !primary_match.is_some_and(|(_, settled)| settled);
                        followed = Some(Followed { rank, slot: candidate, corroborated, new_unproven, needs_confirmation });
                    }
                }
                if !proven_vehicle {
                    continue;
                }
                if self.previous_ff == Some(car.vid) {
                    previous = Some(candidate);
                }
            }
            let rank = |s: Slot| (s.distance.abs(), s.vid);
            if best[slot].is_none_or(|old| rank(candidate) < rank(old)) {
                if slot == 1 || slot == 2 {
                    second[slot - 1] = best[slot];
                }
                best[slot] = Some(candidate);
            } else if (slot == 1 || slot == 2) && second[slot - 1].is_none_or(|old| rank(candidate) < rank(old)) {
                second[slot - 1] = Some(candidate);
            }
        }
        // Apply the lead preference among drawable FF candidates. A truck can produce two nearby radar tracks;
        // if the lead moves between them, let the previous FF keep the preference.
        let preferred_lead = if lead_v_rel.is_none() && lead.is_some() {
            // Callers supplying only a position retain the radar-only selector.
            self.ff_lead_pending = None;
            followed.map(|lead| lead.slot)
        } else {
            // Only current radar observations can survive a brief change in the matched lead.
            if let Some(Followed { slot, corroborated, new_unproven, needs_confirmation, .. }) = followed {
                let since = match self.ff_lead_pending {
                    Some((vid, since)) if vid == slot.vid => since,
                    _ => now,
                };
                let current = self.ff_lead.is_some_and(|(vid, _)| vid == slot.vid);
                let holding = confirmed_lead.is_some()
                    && self.ff_lead.is_some_and(|(_, last)| now - last < FF_HOLD_S);
                // A closer, corroborated display lead may enter as soon as it is drawable.
                let closer = confirmed_lead.is_some_and(|old| slot.distance < old.distance);
                let replaces_previous = !new_unproven
                    || previous.is_none_or(|old| slot.distance + FF_PREFERENCE_M < old.distance);
                if !needs_confirmation && replaces_previous && (current || (corroborated && (closer || !holding))
                    || (!holding && now - since >= FF_CONFIRM_S))
                {
                    self.ff_lead = Some((slot.vid, now));
                    self.ff_lead_pending = None;
                    confirmed_lead = Some(slot);
                } else {
                    self.ff_lead_pending = Some((slot.vid, since));
                }
            } else {
                self.ff_lead_pending = None;
            }
            if confirmed_lead.is_none() || self.ff_lead.is_some_and(|(_, last)| now - last >= FF_HOLD_S) {
                self.ff_lead = None;
                confirmed_lead = None;
            }
            confirmed_lead
        };
        let preferred = match (preferred_lead, previous) {
            (Some(lead), Some(prev)) if twins(vehicles, world, radar, current, lead.vid, prev.vid) => Some(prev),
            _ => preferred_lead.or(previous),
        };
        if let Some(pref) = preferred {
            if best[0].is_none_or(|old| (pref.distance - FF_PREFERENCE_M, pref.vid) < (old.distance, old.vid)) {
                best[0] = Some(pref);
            }
        }
        if let Some(candidates) = matching.as_mut() {
            for (s, allowed) in candidates.iter_mut() {
                for (i, flag) in allowed.iter_mut().enumerate().skip(1) {
                    if let Some(picked) = best[i] {
                        *flag = picked.vid == s.vid;
                    } else if best.iter().flatten().any(|picked| picked.vid == s.vid) {
                        *flag = false;
                    }
                }
            }
            let ff = best[0];
            let required = best.map(|s| s.is_some());
            best = match_slots(candidates, ff, required, self.matching_last);
        }
        self.matching_last = best.map(|s| s.map(|s| s.vid));
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
    pub fn split_alongside(
        &mut self,
        now: f64,
        vehicles: &[Vehicle],
        protected: [bool; 2],
        slots: &mut [Option<Slot>; 5],
    ) -> [bool; 2] {
        let mut split = [false; 2];
        for side in 0..2 {
            let (front, rear) = (side + 1, side + 3);
            if protected[side] || slots[rear].is_some_and(|slot| slot.vid == 0) {
                self.alongside[side] = None;
                self.alongside_alone[side] = None;
                self.passing[side] = None;
                self.passing_hold[side] = None;
                continue;
            }
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
            let mut bridge = false;
            let shown = match (near, next) {
                (Some(near), Some(next)) => {
                    self.alongside_alone[side] = None;
                    Some((near, Some(next)))
                }
                (Some(near), None) if held.is_some() => hold(&mut self.alongside_alone[side], ALONGSIDE_HOLD_S).then_some((near, None)),
                _ => {
                    let droppable = |kept: &Slot| {
                        held.is_none()
                            && vehicles.iter().any(|car| car.vid == kept.vid)
                            && slots.iter().all(|slot| slot.is_none_or(|slot| slot.vid != kept.vid))
                    };
                    let split_near = kept.filter(droppable);
                    let lost = split_near
                        .or_else(|| self.passing_hold[side].filter(droppable))
                        .or_else(|| self.passing[side].filter(droppable));
                    bridge = split_near.is_none() && lost.is_some();
                    lost.filter(|_| hold(&mut self.alongside_alone[side], ALONGSIDE_NEAR_HOLD_S))
                        .map(|lost| (lost, slots[front]))
                }
            };
            self.alongside[side] = shown.filter(|_| !bridge).map(|(near, _)| near);
            self.passing_hold[side] = shown.filter(|_| bridge).map(|(near, _)| near);
            self.passing[side] = close.filter(|s| s.v_rel < -speed);
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

    /// Apply the road-frame correction after selection and camera fusion.
    pub(crate) fn apply_bend(&self, now: f64, slots: &mut [Option<Slot>; 5]) {
        for slot in slots[..3].iter_mut().flatten().filter(|slot| slot.vision != VISION_ONLY) {
            let Some(drawing) = self.drawings.binary_search_by_key(&slot.vid, |d| d.vid)
                .ok().map(|i| &self.drawings[i]).filter(|d| d.seen == now)
            else {
                continue;
            };
            if let Some((bent, raw)) = drawing.bend_shown.zip(drawing.shown) {
                slot.lateral += (bent - raw) as f32;
            }
        }
    }

    /// Limit lateral display jumps from a moving side radar return on a straight road. 47f-14 at 34 s:
    /// one radar ID and vid receded smoothly, but its measured lateral moved 1.6 m and partly bounced back.
    /// Apply the limit after slot selection and fusion, so it cannot alter range, lane assignment or visibility.
    pub(crate) fn smooth_opening_sides(&mut self, now: f64, straight: bool, slots: &mut [Option<Slot>; 5]) {
        const MAX_LATERAL_RATE: f64 = 2.0;
        for (current, previous) in slots[1..3].iter_mut().zip(&mut self.side_lateral) {
            let Some(slot) = current.as_mut().filter(|s| {
                straight && s.vid != 0 && s.vision != VISION_ONLY && s.distance >= 8. && s.v_rel > 2.
            }) else {
                *previous = None;
                continue;
            };
            if let Some((vid, at, lateral)) = *previous {
                let dt = now - at;
                if slot.vid == vid && dt > 0. && dt <= OBSERVATION_MAX_AGE {
                    let limit = (MAX_LATERAL_RATE * dt) as f32;
                    slot.lateral = lateral + (slot.lateral - lateral).clamp(-limit, limit);
                }
            }
            *previous = Some((slot.vid, now, slot.lateral));
        }
    }

    fn clear_split(&mut self) {
        self.runner_up = [None; 2];
        self.alongside = [None; 2];
        self.alongside_alone = [None; 2];
        self.passing = [None; 2];
        self.passing_hold = [None; 2];
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
        if self.generation != world.generation() || !now.is_finite() || !world.accepts_radar() || !world.u_ego().is_finite() {
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

/// Refresh a location only with newly observed road-edge evidence. Keep nearby reports on one anchor.
fn remember_edge(hidden: &mut Vec<(f64, f64, f64)>, now: f64, wx: f64, wy: f64) {
    if let Some(point) = hidden.iter_mut().find(|&&mut (_, x, y)| (x - wx).hypot(y - wy) <= EDGE_MERGE_M) {
        point.0 = now;
    } else if hidden.len() < MAX_VEHICLES {
        hidden.push((now, wx, wy));
    } else if let Some(oldest) = hidden.iter_mut().min_by(|a, b| a.0.total_cmp(&b.0)) {
        *oldest = (now, wx, wy);
    }
}

fn eligible(car: &Vehicle) -> bool {
    !car.flags.structure && (car.flags.proven || car.flags.admitted || car.flags.motion_confirmed)
}


#[cfg(test)]
mod review_tests {
    use super::*;
    use crate::input::model_v2::{Line, Model};
    use crate::vehicles::{Flags, Link};

    #[test]
    fn nearby_edge_measurements_refresh_one_anchor() {
        let mut hidden = Vec::new();
        for i in 0..40 {
            remember_edge(&mut hidden, f64::from(i) * 0.05, 10. + f64::from(i % 2) * 0.2, 3.);
        }
        assert_eq!(hidden, [(39. * 0.05, 10., 3.)]);
        for i in 0..(MAX_VEHICLES + 4) {
            remember_edge(&mut hidden, 2. + i as f64, i as f64 * 2., 7.);
        }
        assert_eq!(hidden.len(), MAX_VEHICLES);
        assert!(hidden.iter().all(|h| h.0 >= 6.));
    }

    #[test]
    fn a_receding_radar_car_on_a_straight_road_limits_only_lateral_jump() {
        let mut assigner = Assigner::default();
        let base = Slot {
            vid: 7325, distance: 20., lateral: -3.05, v_rel: 6.,
            detect: 1, lane_coord_rel: 1., kind: 0, kind_confirmed: false, kind_wanted: false, vision: VISION_NONE,
        };
        let mut slots = [None, None, Some(base), None, None];
        assigner.smooth_opening_sides(33., true, &mut slots);
        let measured = Slot { distance: 20.3, lateral: -3.35, ..base };
        slots[2] = Some(measured);
        assigner.smooth_opening_sides(33.05, true, &mut slots);
        assert_eq!(slots[2], Some(Slot { lateral: -3.05 - 0.1, ..measured }));

        let new_vid = Slot { vid: 7326, ..measured };
        slots[2] = Some(new_vid);
        assigner.smooth_opening_sides(33.1, true, &mut slots);
        assert_eq!(slots[2], Some(new_vid), "another car must start at its own lateral");

        slots[2] = Some(Slot { lateral: -4.2, ..new_vid });
        assigner.smooth_opening_sides(33.15, false, &mut slots);
        assert_eq!(slots[2].unwrap().lateral, -4.2, "on turns and lane changes the drawing follows the input");
    }

    #[test]
    fn a_weak_line_can_hold_a_car_but_cannot_move_it_into_ff() {
        let lane = ModelLane::with_probability(vec![(0., 1.7), (80., 1.7)], vec![(0., -1.7), (80., -1.7)], 0.49);
        assert!(!lane.contains(20., 0.));
        assert!(lane.contains_for(20., 0., true));
    }

    fn slot_test_world() -> World {
        let mut world = World::default();
        world.model(0., &Model {
            lane_lines: [-5.25, -1.75, 1.75, 5.25].into_iter()
                .map(|y| Line { x: vec![0., 5., 10., 20., 40., 80.], y: vec![y; 6] })
                .collect(),
            lane_line_probs: vec![0.9; 4],
            road_edges: Vec::new(),
        });
        assert!(world.u_ego().is_finite());
        world
    }

    fn slot_test_tick(world: &mut World, now: f64, ego_speed: f64) {
        world.live_pose(now, 0.);
        world.car_state(now, ego_speed, false);
        world.radar(now);
    }

    fn stationary_car(world: &World, now: f64, track: u32, point: (f64, f64), flags: Flags) -> Vehicle {
        let (wx, wy) = world.radar_point_world(now, point.0, point.1);
        let u = (world.u_ego() - point.1 / LANE_WIDTH) as f32;
        Vehicle {
            vid: track, wx: wx as f32, wy: wy as f32, vx: 0., vy: 0., off: 0.,
            u, u_slot: u, ids: Default::default(), hist: Default::default(),
            link: Link::Sure, kind: None, seen_radar: now, seen_vision: f64::NEG_INFINITY,
            born: 0., flags, radar_id: Some(track), oncoming: false,
            oncoming_since: f64::NAN, onward_since: f64::NAN,
        }
    }

    fn stationary_lead(track: u32, point: (f64, f64), ego_speed: f64) -> DisplayLead {
        DisplayLead {
            motion: (point.0, point.1, -ego_speed), measured_track: Some(track),
            model_prob: 0.98, confirmed_track: Some(i128::from(track)),
        }
    }

    fn stationary_match(
        assigner: &mut Assigner, world: &mut World, now: f64, ego_speed: f64,
        track: u32, point: (f64, f64), flags: Flags,
    ) -> Option<u32> {
        slot_test_tick(world, now, ego_speed);
        let car = stationary_car(world, now, track, point, flags);
        let lead = stationary_lead(track, point, ego_speed);
        assigner.update_display_lead(now, &[car], world, Some(lead), Some((lead, false)), None)[0].map(|slot| slot.vid)
    }

    #[test]
    fn brief_primary_match_does_not_displace_the_existing_car() {
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        let speed = 2.586;
        for frame in 0..6 {
            let now = f64::from(frame) * 0.05;
            slot_test_tick(&mut world, now, speed);
            let near = (10.20, 0.25);
            let mut lead = stationary_lead(44, near, speed);
            lead.motion.2 = -2.21;
            let mut reflection = stationary_car(&world, now, 44, near, Flags::default());
            reflection.vx = 0.376;
            let car = stationary_car(&world, now, 42, (40.05, -0.55), Flags { proven: true, ..Flags::default() });
            let lead = (2..=3).contains(&frame).then_some(lead);
            let ff = assigner.update_display_lead(now, &[car, reflection], &world, lead, lead.map(|l| (l, false)), None)[0];
            assert_eq!(ff.map(|slot| slot.vid), Some(42), "frame {frame}");
        }
    }

    #[test]
    fn a_parking_row_return_with_brief_primary_matches_leaves_ff_empty() {
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        for frame in 0..10 {
            let now = f64::from(frame) * 0.05;
            let speed = 7.06 + now;
            slot_test_tick(&mut world, now, speed);
            let point = (20.4 - 7. * now, 0.15);
            let car = stationary_car(&world, now, 40, point, Flags::default());
            let lead = (1..=7).contains(&frame).then(|| stationary_lead(40, point, speed));
            let ff = assigner.update_display_lead(now, &[car], &world, lead, lead.map(|l| (l, false)), None)[0];
            assert_eq!(ff, None, "frame {frame}");
        }
    }

    #[test]
    fn weak_entry_uses_primary_id_motion_and_replacement_margin() {
        for (primary_track, primary_x, primary_speed, near, expected) in [
            (41, 12., -4., 12., 42), // LeadTwo agrees, but leadOne has a different ID.
            (40, 16., -4., 12., 42), // LeadOne distance does not agree.
            (40, 12., -1., 12., 42), // LeadOne relative speed does not agree.
            (40, 12., -4., 12., 42), // Exactly 1 m closer.
            (40, 11.75, -4., 11.75, 40),
        ] {
            let mut world = slot_test_world();
            let mut assigner = Assigner::default();
            for frame in 0..4 {
                let now = f64::from(frame) * 0.05;
                slot_test_tick(&mut world, now, 4.);
                let incumbent = stationary_car(&world, now, 42, (13., 0.), Flags { proven: true, ..Flags::default() });
                let weak = stationary_car(&world, now, 40, (near, 0.), Flags::default());
                let preferred = stationary_lead(40, (near, 0.), 4.);
                let mut one = stationary_lead(primary_track, (primary_x, 0.), 4.);
                one.motion.2 = primary_speed;
                let ff = assigner.update_display_lead(now, &[weak, incumbent], &world, Some(preferred), Some((one, true)), None)[0];
                assert_eq!(ff.map(|slot| slot.vid), Some(if frame < 1 { 42 } else { expected }),
                    "primary track {primary_track}, position {primary_x}, speed {primary_speed}, distance {near}, frame {frame}");
            }
        }
    }

    #[test]
    fn parked_track_32_stays_out_of_ff_at_a_stop_and_when_ego_pulls_away() {
        // 47c-0/1: a 6.45 m stationary match at a stop persists as ego starts creeping.
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        for frame in 0..20 {
            let now = f64::from(frame) * 0.05;
            let (speed, distance) = if frame < 10 { (0., 6.45) } else { (0.85, 6.45 - 0.85 * (now - 0.45)) };
            assert_eq!(stationary_match(&mut assigner, &mut world, now, speed, 32, (distance, 0.10),
                Flags::default()), None, "frame {frame}");
        }
    }

    #[test]
    fn parking_aisle_track_47_cannot_enter_ff_while_creeping() {
        // 47d-0: ego is above 0.5 m/s when the 5.1 m stationary return first matches.
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        for frame in 0..12 {
            let now = f64::from(frame) * 0.05;
            assert_eq!(stationary_match(&mut assigner, &mut world, now, 0.9, 47, (5.1 - 0.9 * now, 0.4),
                Flags::default()), None, "frame {frame}");
        }
    }

    #[test]
    fn stopped_lead_beyond_five_metres_enters_when_vehicle_evidence_arrives() {
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        for frame in 0..9 {
            let proven = frame >= 6;
            assert_eq!(stationary_match(&mut assigner, &mut world, f64::from(frame) * 0.05, 0., 1,
                (8., 0.), Flags { proven, ..Flags::default() }), proven.then_some(1), "frame {frame}");
        }
    }

    #[test]
    fn a_road_without_current_lane_lines_uses_the_normal_vehicle_path_at_creep_speed() {
        // The world map still has a lane coordinate from earlier observations.
        for ego_speed in [2., 2.01] {
            let mut world = slot_test_world();
            let mut assigner = Assigner::default();
            world.model(0.01, &Model { lane_lines: Vec::new(), lane_line_probs: Vec::new(), road_edges: Vec::new() });
            for frame in 0..9 {
                let proven = frame >= 6;
                let shown = stationary_match(&mut assigner, &mut world, 0.05 + f64::from(frame) * 0.05,
                    ego_speed, 2, (12., 0.), Flags { proven, ..Flags::default() });
                assert_eq!(shown, proven.then_some(2),
                    "ego speed {ego_speed}, frame {frame}");
            }
        }
    }

    #[test]
    fn a_creeping_lead_can_enter_inside_trusted_lines_and_hold_when_the_lines_disappear() {
        let mut world = slot_test_world();
        let mut assigner = Assigner::default();
        let lines = || (vec![(0., 1.7), (80., 1.7)], vec![(0., -1.7), (80., -1.7)]);
        let (left, right) = lines();
        assigner.set_model_lane(Some(ModelLane::with_probability(left, right, 0.49)));
        for frame in 0..10 {
            if frame == 3 {
                let (left, right) = lines();
                assigner.set_model_lane(Some(ModelLane::new(left, right)));
            } else if frame == 6 {
                assigner.set_model_lane(None);
            }
            assert_eq!(stationary_match(&mut assigner, &mut world, f64::from(frame) * 0.05, 1., 1,
                (5.26 - f64::from(frame) * 0.05, 0.), Flags::default()), (frame >= 4).then_some(1),
                "frame {frame}");
        }
    }
}
