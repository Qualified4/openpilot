//! The map's own state that `_advance`, `_measurements`, `_associate` and `_observe` keep: ego's pose, the objects and
//! their ego-frame positions, the counters and the radar id history. The Python map keeps the same fields as
//! attributes (hyundaicanfd_ccnc_radar_map.py `CcncRadarMap.__init__`).

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use super::associate::{associate, measure, AssociateIn, Ego, Evidence, RadarPoint};
use super::consts::Consts;
use super::lanes::{LaneCache, Lanes};
use super::object::{MapObjectData, Ring};
use super::observe::{crossing, observe, ObserveIn};
use super::select::{lane_reference, vision, Bridge, Selector};
use super::walls::{self, BandData, BandPoint, IntSet, WallData};
use crate::vehicles::{Flags, Reported, Tracker, Vehicle};
use crate::world::World;

/// A failure of the map update, as the Python reference raises it (the extension turns it into that exception).
#[derive(Debug)]
pub enum Error {
    ZeroDivision(&'static str),
    Index(&'static str),
    Type(&'static str),
    Key(i64),
    Field(&'static str),
    Value(&'static str),
    Overflow(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A dict with Python's insertion order. Updating an existing key keeps its place.
#[derive(Clone, Debug)]
pub struct InsertionMap<K, V> {
    order: Vec<K>,
    values: HashMap<K, V>,
}

impl<K, V> Default for InsertionMap<K, V> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            values: HashMap::new(),
        }
    }
}

impl<K: Copy + Eq + Hash, V> InsertionMap<K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        self.values.get(key)
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.values.insert(key, value).is_none() {
            self.order.push(key);
        }
    }

    /// `dict.setdefault(key, default())`.
    pub fn entry(&mut self, key: K, default: impl FnOnce() -> V) -> &mut V {
        if !self.values.contains_key(&key) {
            self.order.push(key);
        }
        self.values.entry(key).or_insert_with(default)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.values.get_mut(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.order.iter().copied()
    }

    pub fn retain(&mut self, mut keep: impl FnMut(K, &V) -> bool) {
        self.order.retain(|key| {
            let keep = keep(*key, &self.values[key]);
            if !keep {
                self.values.remove(key);
            }
            keep
        });
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
        self.order.iter().map(|k| (*k, &self.values[k]))
    }
}

/// The map's pose and its cosine/sine: `to_ego`.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub px: f64,
    pub py: f64,
    pub c: f64,
    pub s: f64,
}

impl Frame {
    pub fn to_ego(self, x: f64, y: f64) -> (f64, f64) {
        let (c, s) = (self.c, self.s);
        (
            c * (x - self.px) + s * (y - self.py),
            -s * (x - self.px) + c * (y - self.py),
        )
    }
}

pub type Pose = (f64, f64, f64);

/// What `_advance` reads of the map's constants.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub min_x: f64,
    pub keep_x: f64,
    pub max_y: f64,
    pub far_x: f64,
    pub far_y: f64,
}

pub struct Observation<'a> {
    pub points: &'a [RadarPoint],
    pub stamp: f64,
    /// The interpolated radar pose (x, y, cos, sin).
    pub pose: Option<(f64, f64, f64, f64)>,
    pub evidence: &'a HashMap<i128, Evidence>,
    /// A current radar observation independently corroborated as the model's lead car.
    pub wall_car_track: Option<i128>,
}

pub struct UpdateIn<'a> {
    pub radar: Option<Observation<'a>>,
    pub radar_stamp: Option<f64>,
    pub a_ego: f64,
    pub yaw_rate: f64,
    pub holding: bool,
    pub ego_change: bool,
    pub model_stamp: Option<f64>,
    pub model_changing: bool,
    pub lane_change: bool,
    pub blindspot: [bool; 2],
    pub leads: &'a [(f64, f64)],
    pub vision_leads: &'a [(f64, f64, Option<f64>)],
    pub vision: &'a [(f64, f64)],
    pub select: bool,
    pub usable: bool,
}

pub struct Updated {
    pub lane_change: bool,
    pub retired_bands: Vec<BandData>,
}

#[derive(Clone, Debug)]
pub struct MapState {
    pub(super) selection_history: bool,
    pub pose: Pose,
    pub cos: f64,
    pub sin: f64,
    pub now: Option<f64>,
    pub speed: f64,
    /// Distance ego has travelled, for conditions that follow the road instead of the clock.
    pub odometer: f64,
    pub poses: Ring<(f64, Pose), 64>,
    /// Keys are given in increasing order and never reused, so key order is the Python dict's insertion order.
    pub objects: BTreeMap<i64, MapObjectData>,
    pub local: BTreeMap<i64, (f64, f64)>,
    pub next_id: i64,
    pub created: i64,
    pub evicted: i64,
    pub range_removed: i64,
    /// Radar id -> recent (stamp, x, y) reports, ego frame.
    pub reports: InsertionMap<i128, Ring<(f64, f64, f64), 4>>,
    /// Radar id -> stamp of its last lateral jump.
    pub jumps: InsertionMap<i128, f64>,
    pub walls: Vec<WallData>,
    pub wall_geometry: Vec<Vec<(f64, f64)>>,
    pub bends: Vec<(i64, f64, f64)>,
    pub bands: Vec<BandData>,
    pub band_points: InsertionMap<i64, BandPoint>,
    pub band_early: IntSet,
    pub(super) next_band: u64,
    pub selected: [Option<i64>; 3],
    pub switched: [Option<f64>; 3],
    pub absent: [Option<f64>; 3],
    pub remembered: [bool; 3],
    pub bridge: Option<Bridge>,
    /// The FF object's last report inside ego's lane: (key, lateral, centred, speed).
    pub hold: Option<(i64, f64, bool, f64)>,
    pub far_since: Option<(i64, f64)>,
    pub lane_return: Option<f64>,
    pub lane_offset: i64,
    pub lane_cache: LaneCache,
}

impl Default for MapState {
    fn default() -> Self {
        Self::new()
    }
}

impl MapState {
    pub fn new() -> Self {
        Self {
            selection_history: true,
            pose: (0., 0., 0.),
            cos: 1.,
            sin: 0.,
            now: None,
            speed: 0.,
            odometer: 0.,
            poses: Ring::new(),
            objects: BTreeMap::new(),
            local: BTreeMap::new(),
            next_id: 1,
            created: 0,
            evicted: 0,
            range_removed: 0,
            reports: InsertionMap::default(),
            jumps: InsertionMap::default(),
            walls: Default::default(),
            wall_geometry: Default::default(),
            bends: Default::default(),
            bands: Default::default(),
            band_points: Default::default(),
            band_early: Default::default(),
            next_band: Default::default(),
            selected: Default::default(),
            switched: Default::default(),
            absent: Default::default(),
            remembered: Default::default(),
            bridge: Default::default(),
            hold: Default::default(),
            far_since: Default::default(),
            lane_return: Default::default(),
            lane_offset: 0,
            lane_cache: LaneCache::default(),
        }
    }

    pub fn frame(&self) -> Frame {
        Frame {
            px: self.pose.0,
            py: self.pose.1,
            c: self.cos,
            s: self.sin,
        }
    }

    /// The existing radar proof used for both the map vehicles and the side display.
    pub(crate) fn vehicle_proven(&self, object: &MapObjectData, crossing_travel: f64) -> bool {
        !object.structure
            && !object.superseded
            && (object.radar_moving || object.doppler_confirmed)
            && !crossing(object, self.frame(), crossing_travel)
    }

    /// Populate the world-map vehicle layer after updating the radar's object and wall state.
    pub fn update_vehicles<'a>(
        &self,
        tracker: &'a mut Tracker,
        stamp: f64,
        points: &[(u32, f64, f64, f64)],
        world: &World,
        crossing_travel: f64,
    ) -> &'a [Vehicle] {
        tracker.update(stamp, points, world);
        let mut reports = HashMap::with_capacity(points.len());
        for object in self.objects.values() {
            if object.report.is_some_and(|(t, _, _)| t == stamp) {
                // A later object key wins when both objects report the same radar ID.
                reports.insert(object.source, object);
            }
        }
        for car in tracker.vehicles_mut().iter_mut().filter(|car| car.seen_radar == stamp) {
            let Some(id) = car.radar_id else { continue };
            let object = reports.get(&i128::from(id)).copied();
            let prior = car.flags;
            car.flags = object.map_or_else(Flags::default, |object| {
                let borrowed = borrowed_motion(&prior, object, id, stamp);
                let moved = borrowed.is_some();
                Flags {
                    motion_confirmed: object.motion_confirmed || moved,
                    stationary: object.stationary,
                    structure: object.structure,
                    proven: self.vehicle_proven(object, crossing_travel)
                        || (moved && prior.proven && !crossing(object, self.frame(), crossing_travel)),
                    admitted: object.admitted,
                    object: Some(object.key),
                    source: Some(id),
                    borrowed,
                    reported: Some(Reported { x: object.x, y: object.y, vx: object.vx, vy: object.vy, at: stamp }),
                }
            });
        }
        tracker.vehicles()
    }

    /// `self.local[key]`.
    pub fn local_xy(&self, key: i64) -> Result<(f64, f64)> {
        self.local.get(&key).copied().ok_or(Error::Key(key))
    }

    pub fn object(&self, key: i64) -> Result<&MapObjectData> {
        self.objects.get(&key).ok_or(Error::Key(key))
    }

    pub fn object_mut(&mut self, key: i64) -> Result<&mut MapObjectData> {
        self.objects.get_mut(&key).ok_or(Error::Key(key))
    }

    /// Update the legacy map from prepared inputs after advancing ego and placing the model lines.
    pub fn update(&mut self, k: &Consts, inp: UpdateIn<'_>) -> Result<Updated> {
        let now = self.now.ok_or(Error::Value("map not advanced"))?;
        let fresh = inp
            .model_stamp
            .is_some_and(|stamp| (0. ..=0.25).contains(&(now - stamp)));
        let cache = std::mem::take(&mut self.lane_cache);
        let mut lanes = cache.view(
            self.lane_offset,
            fresh,
            (k.outer_narrow, k.outer_far),
            inp.model_changing,
        );
        let result = self.update_stages(k, &mut lanes, inp);
        self.lane_cache = cache;
        result
    }

    fn update_stages(&mut self, k: &Consts, lanes: &mut Lanes<'_>, inp: UpdateIn<'_>) -> Result<Updated> {
        let now = self.now.ok_or(Error::Value("map not advanced"))?;
        let frame = self.frame();
        let lane_change = lane_reference(self, k, lanes, now, inp.ego_change, inp.lane_change);
        let measured = if let Some(radar) = &inp.radar {
            let ego = Ego {
                a_ego: inp.a_ego,
                yaw_rate: inp.yaw_rate,
                speed: self.speed,
            };
            let measurements = radar.pose.map_or_else(Vec::new, |pose| {
                measure(
                    self,
                    k,
                    ego,
                    inp.leads,
                    &|track| radar.evidence.get(&track).copied(),
                    radar.points,
                    radar.stamp,
                    pose,
                )
            });
            let association = AssociateIn {
                holding: inp.holding,
                lane_change,
                selected: self.selected,
            };
            let pairs = associate(self, k, association, lanes, &measurements, radar.stamp)?;
            let mut assigned = vec![None; measurements.len()];
            let mut reserved = Vec::with_capacity(pairs.len());
            for (index, key) in pairs {
                assigned[index] = Some(key);
                reserved.push(key);
            }
            let observation = ObserveIn {
                holding: inp.holding,
                a_ego: inp.a_ego,
                bridge: self.bridge.map(|b| (b.key, b.x, b.y)),
            };
            lanes.read_bands(&self.bands);
            observe(
                self,
                k,
                observation,
                lanes,
                &measurements,
                assigned,
                reserved,
                radar.stamp,
            )?
            .into_iter()
            .collect()
        } else {
            IntSet::default()
        };

        vision(self, now, inp.vision)?;
        // Same source and update only. An older object that reused this radar ID may still belong to a wall.
        let wall_car = inp.radar.and_then(|radar| {
            let track = radar.wall_car_track?;
            self.objects.values()
                .filter(|o| o.source == track && o.report.is_some_and(|(at, _, _)| at == radar.stamp))
                .last()
                .map(|o| o.key)
        });
        let retired_bands = walls::update(self, k, now, lanes, &measured, wall_car)?;
        if inp.select {
            Selector {
                st: self,
                k,
                lanes,
                frame,
                now,
                vision_leads: inp.vision_leads,
                holding: inp.holding,
                lane_change,
                usable: inp.usable,
                radar_stamp: inp.radar_stamp,
                blindspot: inp.blindspot,
            }
            .select()?;
        }
        if !inp.select || !inp.usable {
            self.selected = [None; 3];
        }
        Ok(Updated {
            lane_change,
            retired_bands,
        })
    }

    /// `_advance`: move ego's pose by `speed` and `yaw` since the last update, then each object, rebuild `local` and
    /// drop the objects out of range (except the bridged one). Returns (dt, distance).
    pub fn advance(
        &mut self,
        now: f64,
        speed: f64,
        yaw: f64,
        holding: bool,
        bridged: Option<i64>,
        k: Range,
    ) -> (f64, f64) {
        let dt = self.now.map_or(0., |last| now - last);
        let (x, y, angle) = self.pose;
        let distance = 0.5 * (speed + self.speed) * dt;
        let middle = angle + 0.5 * yaw * dt;
        self.pose = (
            x + distance * middle.cos(),
            y + distance * middle.sin(),
            angle + yaw * dt,
        );
        self.cos = self.pose.2.cos();
        self.sin = self.pose.2.sin();
        self.now = Some(now);
        self.speed = speed;
        self.odometer += distance.abs();
        self.poses.push((now, self.pose));
        self.advance_objects(now, dt, distance, holding, bridged, k);
        (dt, distance)
    }

    fn advance_objects(&mut self, now: f64, dt: f64, distance: f64, holding: bool, bridged: Option<i64>, k: Range) {
        let frame = self.frame();
        self.local.clear();
        let mut expired = Vec::new();
        for (&key, o) in self.objects.iter_mut() {
            if !holding {
                o.admitted = false;
            }
            if holding && now - o.seen > 0.3 {
                // An unseen neighbor slowing with ego is not extrapolated at its last speed.
                o.vx = 0.;
                o.vy = 0.;
            }
            o.x += o.vx * dt;
            o.y += o.vy * dt;
            // A stationary remembered vehicle can still depart outside radar coverage.
            o.uncertainty += (if o.stationary {
                0.08
            } else {
                0.35 + 0.3 * (now - o.seen)
            }) * dt
                + 0.002 * distance.abs();
            o.vision = false;
            let (x, y) = frame.to_ego(o.x, o.y);
            self.local.insert(key, (x, y));
            let in_range = k.min_x <= x && x <= k.keep_x && y.abs() <= k.max_y;
            // A far object exists only while measured with vehicle evidence; it is never remembered.
            let far_vehicle = k.keep_x < x
                && x <= k.far_x + 2.
                && y.abs() <= k.far_y
                && now - o.seen <= 1.
                && (o.motion_confirmed || o.vehicle >= 0.6);
            if bridged != Some(key) && !in_range && !far_vehicle {
                expired.push(key);
            }
        }
        for key in &expired {
            self.objects.remove(key);
            self.local.remove(key);
        }
        self.range_removed += expired.len() as i64;
    }

    /// `pose_at(stamp)`: the pose at `stamp`, interpolated in the pose history; None before it or after `now`.
    pub fn pose_at(&self, stamp: f64) -> Option<Pose> {
        let first = self.poses.front()?;
        if stamp < first.0 - 1e-6 {
            return None;
        }
        let mut previous = first;
        for current in self.poses.iter() {
            if current.0 >= stamp {
                let dt = current.0 - previous.0;
                let f = if dt > 0. { (stamp - previous.0) / dt } else { 0. };
                let (a, b) = (previous.1, current.1);
                return Some((a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1), a.2 + f * (b.2 - a.2)));
            }
            previous = current;
        }
        let now = self.now?;
        (stamp <= now + 1e-6).then_some(self.pose)
    }
}

/// The radar map starts a new object for the same radar track when a measurement jumps past its gate (a car creeping
/// 5 m ahead at 2 m/s jumped 0.8 m back in 11-24 0000047c segment 5 at 45.3 s). The new object has to prove its motion
/// again, and the car lost its slot for 4 frames. A map car keeps the motion proof of its track's earlier object, but
/// only across a jump of a moving car that continues that object's last report: both at `BORROW_MOVING` or faster,
/// within `BORROW_GAP_S`, within `BORROW_JUMP_M` of where it was heading and within `BORROW_SPEED` of its velocity.
/// - A standing object proves itself: an earlier object at 0-0.6 m/s gave its proof to a new one at 0-0.5 m/s in
///   00000479 segment 19 at 35.8 s and 0000047c segment 2 at 27.9 s, drawn in FF for 0.5-1.2 s.
/// - One car's radar velocity moves 1.3 m/s across a jump at 10 m/s (00000454 segment 34), under the tracker's own
///   2.5 m/s gate. The earlier object can be gone by then (00000450 segment 16 at 25 m/s), so the car keeps its report.
/// The borrowed proof lasts `BORROW_S` from the first jump unless the new object proves its own motion.
const BORROW_MOVING: f64 = 1.0;
const BORROW_GAP_S: f64 = 0.5;
const BORROW_JUMP_M: f64 = 1.5;
const BORROW_SPEED: f64 = 2.0;
const BORROW_S: f64 = 1.0;

/// The stamp the car started borrowing its motion proof from an earlier object of radar track `id`, or None when the
/// reporting `object` has to stand on its own proof.
fn borrowed_motion(prior: &Flags, object: &MapObjectData, id: u32, stamp: f64) -> Option<f64> {
    if object.motion_confirmed || object.structure || object.superseded || prior.source != Some(id) || !prior.motion_confirmed {
        return None;
    }
    if prior.object == Some(object.key) {
        // The same unproven object: it keeps what it borrowed until the deadline.
        return prior.borrowed.filter(|since| stamp - since <= BORROW_S);
    }
    let earlier = prior.reported?;
    let dt = stamp - earlier.at;
    let continues = earlier.vx.hypot(earlier.vy) >= BORROW_MOVING
        && object.vx.hypot(object.vy) >= BORROW_MOVING
        && dt <= BORROW_GAP_S
        && (object.x - earlier.x - earlier.vx * dt).hypot(object.y - earlier.y - earlier.vy * dt) <= BORROW_JUMP_M
        && (object.vx - earlier.vx).hypot(object.vy - earlier.vy) <= BORROW_SPEED;
    // A chain of jumps does not restart the deadline.
    let since = prior.borrowed.unwrap_or(stamp);
    (continues && stamp - since <= BORROW_S).then_some(since)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(key: i64, x: f64, vx: f64) -> MapObjectData {
        MapObjectData::new(key, 4, x, -3.5, vx, 0., 0., 0.)
    }

    /// The flags of a car whose proven object `key` reported at `x`, moving `vx`, at `at`.
    fn proven(key: i64, x: f64, vx: f64, at: f64) -> Flags {
        Flags {
            motion_confirmed: true,
            proven: true,
            object: Some(key),
            source: Some(4),
            reported: Some(Reported { x, y: -3.5, vx, vy: 0., at }),
            ..Flags::default()
        }
    }

    #[test]
    fn a_new_object_of_the_track_borrows_the_motion_proof_only_when_it_continues_the_earlier_one() {
        let prior = proven(1, 5.0, 2.0, 1.0);
        // 47c-5: 0.8 m back from where it was heading (5.1 m) at the same speed, the next frame.
        assert_eq!(borrowed_motion(&prior, &object(2, 4.3, 2.0), 4, 1.05), Some(1.05));
        assert_eq!(borrowed_motion(&prior, &object(2, 4.3, 3.8), 4, 1.05), Some(1.05), "radar speed moved 1.8 m/s");
        for (why, prior, new, stamp) in [
            ("3 m jump", prior, object(2, 2.1, 2.0), 1.05),
            ("2.5 m/s faster", prior, object(2, 4.3, 4.5), 1.05),
            ("onto a standing object", prior, object(2, 4.6, 0.0), 1.05),
            ("a slow object onto a standing one (479-19)", proven(1, 5.0, 0.5, 1.0), object(2, 4.6, 0.5), 1.05),
            ("last report 0.6 s ago", prior, object(2, 6.0, 2.0), 1.6),
            ("another track", Flags { source: Some(5), ..prior }, object(2, 4.3, 2.0), 1.05),
            ("no report", Flags { reported: None, ..prior }, object(2, 4.3, 2.0), 1.05),
        ] {
            assert_eq!(borrowed_motion(&prior, &new, 4, stamp), None, "{why}");
        }
    }

    #[test]
    fn a_borrowed_motion_proof_ends_a_second_after_the_jump_unless_the_new_object_proves_its_own() {
        let mut flags = proven(1, 5.0, 2.0, 1.0);
        let mut held = Vec::new();
        for i in 1..40 {
            let stamp = 1.0 + 0.05 * f64::from(i);
            // Two jumps: a chain of new objects keeps the first jump's deadline.
            let key = if i < 10 { 2 } else { 3 };
            let new = object(key, 5.0 + 2.0 * (stamp - 1.0) - 0.8, 2.0);
            let borrowed = borrowed_motion(&flags, &new, 4, stamp);
            held.push((i, borrowed.is_some()));
            flags = Flags {
                motion_confirmed: borrowed.is_some(),
                borrowed,
                ..proven(key, new.x, new.vx, stamp)
            };
        }
        // The first jump is at 1.05 s: borrowed through 2.0 s, gone from 2.1 s.
        assert!(held.iter().all(|&(i, on)| i == 21 || on == (i <= 20)), "{held:?}");
        let mut own = object(4, 7.0, 2.0);
        own.motion_confirmed = true;
        assert_eq!(borrowed_motion(&flags, &own, 4, 2.5), None, "own proof: nothing to borrow");
    }
}
