//! The map's own state that `_advance`, `_measurements`, `_associate` and `_observe` keep: ego's pose, the objects and
//! their ego-frame positions, the counters and the radar id history. The Python map keeps the same fields as
//! attributes (hyundaicanfd_ccnc_radar_map.py `CcncRadarMap.__init__`).

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use super::associate::{associate, measure, AssociateIn, Ego, Evidence, RadarPoint};
use super::consts::Consts;
use super::lanes::{interpolate, LaneCache, Lanes};
use super::object::{MapObjectData, Ring};
use super::observe::{crossing, observe, ObserveIn};
use super::select::{approach_only, lane_reference, vision, wall_outside, Bridge, Selector};
use super::walls::{self, BandData, BandPoint, IntSet, WallData};
use crate::vehicles::{Flags, Tracker, Vehicle};
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

    /// The world map uses the radar's observation and wall stages with its own slot selector.
    pub(crate) fn for_observation() -> Self {
        Self { selection_history: false, ..Self::new() }
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
        ego_changing: bool,
    ) -> &'a [Vehicle] {
        tracker.update(stamp, points, world);
        let mut reports = HashMap::with_capacity(points.len());
        for object in self.objects.values() {
            if object.report.is_some_and(|(t, _, _)| t == stamp) {
                // A later object key wins when both objects report the same radar ID.
                reports.insert(object.source, object);
            }
        }
        // The side checks read the lines only, not the model age, outer margins or lane change of `update`'s view.
        let lanes = self.lane_cache.view(self.lane_offset, false, (0., 0.), false);
        for car in tracker.vehicles_mut().iter_mut().filter(|car| car.seen_radar == stamp) {
            let Some(id) = car.radar_id else { continue };
            let object = reports.get(&i128::from(id)).copied();
            car.flags = object.map_or_else(Flags::default, |object| Flags {
                motion_confirmed: object.motion_confirmed,
                stationary: object.stationary,
                structure: object.structure,
                proven: self.vehicle_proven(object, crossing_travel),
                admitted: object.admitted,
                approach_only: approach_only(object),
                // The model relabels its lines while ego changes lanes: the world lane coordinate alone keeps the cars.
                side_blocked: if ego_changing {
                    [false; 2]
                } else {
                    self.side_blocked(&lanes, object)
                },
                object: Some(object.key),
            });
        }
        tracker.vehicles()
    }

    /// `_select`'s lane and wall exclusions for a side slot, as `Lanes::side_blocked`.
    fn side_blocked(&self, lanes: &Lanes<'_>, object: &MapObjectData) -> [bool; 2] {
        let Some(&(x, y)) = self.local.get(&object.key) else {
            return [false; 2];
        };
        let walls: Vec<f64> = self.wall_geometry.iter().filter_map(|points| interpolate(points, x)).collect();
        let unconfirmed_outside = !object.motion_confirmed && !object.vision && wall_outside(&self.wall_geometry, x, y);
        lanes
            .side_blocked(x, y, object.motion_confirmed, &walls)
            .map(|blocked| blocked || unconfirmed_outside)
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
        let measured = if let Some(radar) = inp.radar {
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
        let retired_bands = walls::update(self, k, now, lanes, &measured)?;
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
