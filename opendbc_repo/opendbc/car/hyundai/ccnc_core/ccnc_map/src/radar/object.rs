//! A map object: the fields of `MapObject` in hyundaicanfd_ccnc_radar_map.py, in its order and with its defaults.

use std::collections::VecDeque;

/// `collections.deque(maxlen=N)`: appending to a full ring drops the oldest item.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ring<T, const N: usize>(VecDeque<T>);

impl<T: Copy, const N: usize> Ring<T, N> {
    pub fn new() -> Self {
        Self(VecDeque::with_capacity(N))
    }

    pub fn from_items(items: impl IntoIterator<Item = T>) -> Self {
        let mut ring = Self::new();
        for item in items {
            ring.push(item);
        }
        ring
    }

    pub fn push(&mut self, item: T) {
        if self.0.len() == N {
            self.0.pop_front();
        }
        self.0.push_back(item);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn front(&self) -> Option<T> {
        self.0.front().copied()
    }

    pub fn back(&self) -> Option<T> {
        self.0.back().copied()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + DoubleEndedIterator + Clone + '_ {
        self.0.iter().copied()
    }

    pub const fn capacity() -> usize {
        N
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MapObjectData {
    pub key: i64,
    pub source: i128,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub born: f64,
    pub seen: f64,
    pub hits: i64,
    pub existence: f64,
    pub vehicle: f64,
    pub uncertainty: f64,
    pub motion_confirmed: bool,
    pub stationary: bool,
    pub vision: bool,
    pub structure: bool,
    pub slot: Option<i64>,
    pub interior: bool,
    pub drift: f64,
    pub explained: f64,
    pub radar_moving: bool,
    pub doppler: i64,
    pub doppler_confirmed: bool,
    pub forward_samples: i64,
    pub vision_seen: bool,
    pub doppler_recent: Ring<f64, 5>,
    pub admitted: bool,
    pub last_motion: (f64, f64),
    pub origin: Option<(f64, f64)>,
    pub departed: bool,
    pub superseded: bool,
    pub ahead_of: Vec<(i64, f64)>,
    pub tracked: i64,
    pub beyond: Option<f64>,
    pub settled: bool,
    pub history: Ring<(f64, f64, f64, f64), 16>,
    pub trail: Ring<(f64, f64), 20>,
    pub report: Option<(f64, f64, f64)>,
    pub lateral_reports: Ring<(f64, f64), 16>,
    pub calm_since: Option<f64>,
    pub stable_y: Option<f64>,
}

impl MapObjectData {
    /// `MapObject(key, source, x, y, vx, vy, born, seen)`: every other field at its default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(key: i64, source: i128, x: f64, y: f64, vx: f64, vy: f64, born: f64, seen: f64) -> Self {
        Self::with_history(key, source, x, y, vx, vy, born, seen, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn with_history(key: i64, source: i128, x: f64, y: f64, vx: f64, vy: f64, born: f64, seen: f64, selection_history: bool) -> Self {
        Self {
            key,
            source,
            x,
            y,
            vx,
            vy,
            born,
            seen,
            hits: 1,
            existence: 0.2,
            vehicle: 0.,
            uncertainty: 0.25,
            motion_confirmed: false,
            stationary: false,
            vision: false,
            structure: false,
            slot: None,
            interior: false,
            drift: 0.,
            explained: 0.,
            radar_moving: false,
            doppler: 0,
            doppler_confirmed: false,
            forward_samples: 0,
            vision_seen: false,
            doppler_recent: Ring::new(),
            admitted: false,
            last_motion: (0., 0.),
            origin: None,
            departed: false,
            superseded: false,
            ahead_of: Vec::new(),
            tracked: 0,
            beyond: None,
            settled: false,
            history: Ring::new(),
            trail: if selection_history { Ring::new() } else { Ring::default() },
            report: None,
            lateral_reports: if selection_history { Ring::new() } else { Ring::default() },
            calm_since: None,
            stable_y: None,
        }
    }

    /// `MapObject(key, source, x, y, vx, vy, stamp, stamp, origin=(x, y))`: a new measured object.
    pub fn measured(key: i64, source: i128, x: f64, y: f64, vx: f64, vy: f64, stamp: f64) -> Self {
        Self::measured_with_history(key, source, x, y, vx, vy, stamp, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn measured_with_history(key: i64, source: i128, x: f64, y: f64, vx: f64, vy: f64, stamp: f64, selection_history: bool) -> Self {
        Self { origin: Some((x, y)), ..Self::with_history(key, source, x, y, vx, vy, stamp, stamp, selection_history) }
    }
}
