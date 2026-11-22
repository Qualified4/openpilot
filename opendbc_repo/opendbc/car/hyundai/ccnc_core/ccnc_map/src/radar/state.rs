//! The map's own state that `_advance`, `_measurements`, `_associate` and `_observe` keep: ego's pose, the objects and
//! their ego-frame positions, the counters and the radar id history. The Python map keeps the same fields as
//! attributes (hyundaicanfd_ccnc_radar_map.py `CcncRadarMap.__init__`).

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use super::object::{MapObjectData, Ring};

/// A failure of the map update, as the Python reference raises it (the extension turns it into that exception).
#[derive(Debug)]
pub enum Error {
    ZeroDivision(&'static str),
    Index(&'static str),
    Key(i64),
    Value(&'static str),
    Overflow(&'static str),
    /// An error of a caller's callback (the lane queries of the extension), handed back as it was.
    External(Box<dyn std::any::Any + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A dict with Python's insertion order, for keys that are set again (they keep their place) and never removed.
#[derive(Clone, Debug)]
pub struct InsertionMap<K, V> {
    order: Vec<K>,
    values: HashMap<K, V>,
}

impl<K, V> Default for InsertionMap<K, V> {
    fn default() -> Self {
        Self { order: Vec::new(), values: HashMap::new() }
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
        (c * (x - self.px) + s * (y - self.py), -s * (x - self.px) + c * (y - self.py))
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

#[derive(Clone, Debug)]
pub struct MapState {
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
}

impl Default for MapState {
    fn default() -> Self {
        Self::new()
    }
}

impl MapState {
    pub fn new() -> Self {
        Self {
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
        }
    }

    pub fn frame(&self) -> Frame {
        Frame { px: self.pose.0, py: self.pose.1, c: self.cos, s: self.sin }
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

    /// `_advance`: move ego's pose by `speed` and `yaw` since the last update, then each object, rebuild `local` and
    /// drop the objects out of range (except the bridged one). Returns (dt, distance).
    pub fn advance(&mut self, now: f64, speed: f64, yaw: f64, holding: bool, bridged: Option<i64>, k: Range) -> (f64, f64) {
        let dt = self.now.map_or(0., |last| now - last);
        let (x, y, angle) = self.pose;
        let distance = 0.5 * (speed + self.speed) * dt;
        let middle = angle + 0.5 * yaw * dt;
        self.pose = (x + distance * middle.cos(), y + distance * middle.sin(), angle + yaw * dt);
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
            o.uncertainty += (if o.stationary { 0.08 } else { 0.35 + 0.3 * (now - o.seen) }) * dt + 0.002 * distance.abs();
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
