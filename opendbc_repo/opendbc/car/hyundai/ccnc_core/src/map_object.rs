//! `MapObject` of hyundaicanfd_ccnc_radar_map.py as a Rust class: the Rust loops change its fields directly.
//!
//! The constructor, the field names and `__getstate__` follow the Python dataclass, which the Python reference
//! path keeps using. A deque field reads as a tuple copy, so Python code that tries to change one in place fails
//! loudly instead of changing a copy.

use std::collections::VecDeque;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::radar_map::deque;

/// `collections.deque(maxlen=N)`: appending to a full ring drops the oldest item.
#[derive(Clone, Debug, Default)]
pub struct Ring<T, const N: usize>(VecDeque<T>);

impl<T: Copy, const N: usize> Ring<T, N> {
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

    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + Clone + '_ {
        self.0.iter().copied()
    }

    fn from_items(items: Option<Vec<T>>) -> Self {
        let mut ring = Self(VecDeque::with_capacity(N));
        for item in items.into_iter().flatten() {
            ring.push(item);
        }
        ring
    }

    fn tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>>
    where
        T: for<'a> IntoPyObject<'a>,
    {
        PyTuple::new(py, self.iter())
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>
    where
        T: for<'a> IntoPyObject<'a>,
    {
        let kwargs = PyDict::new(py);
        kwargs.set_item("maxlen", N)?;
        deque(py)?.call((self.tuple(py)?,), Some(&kwargs))
    }
}

#[pyclass(module = "ccnc_core", name = "MapObject")]
pub struct MapObject {
    #[pyo3(get, set)]
    pub key: i64,
    #[pyo3(get, set)]
    pub source: i128,
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub vx: f64,
    #[pyo3(get, set)]
    pub vy: f64,
    #[pyo3(get, set)]
    pub born: f64,
    #[pyo3(get, set)]
    pub seen: f64,
    #[pyo3(get, set)]
    pub hits: i64,
    #[pyo3(get, set)]
    pub existence: f64,
    #[pyo3(get, set)]
    pub vehicle: f64,
    #[pyo3(get, set)]
    pub uncertainty: f64,
    #[pyo3(get, set)]
    pub motion_confirmed: bool,
    #[pyo3(get, set)]
    pub stationary: bool,
    #[pyo3(get, set)]
    pub vision: bool,
    #[pyo3(get, set)]
    pub structure: bool,
    #[pyo3(get, set)]
    pub slot: Option<i64>,
    #[pyo3(get, set)]
    pub interior: bool,
    #[pyo3(get, set)]
    pub drift: f64,
    #[pyo3(get, set)]
    pub explained: f64,
    #[pyo3(get, set)]
    pub radar_moving: bool,
    #[pyo3(get, set)]
    pub doppler: i64,
    #[pyo3(get, set)]
    pub doppler_confirmed: bool,
    #[pyo3(get, set)]
    pub forward_samples: i64,
    #[pyo3(get, set)]
    pub vision_seen: bool,
    pub doppler_recent: Ring<f64, 5>,
    #[pyo3(get, set)]
    pub admitted: bool,
    #[pyo3(get, set)]
    pub last_motion: (f64, f64),
    #[pyo3(get, set)]
    pub origin: Option<(f64, f64)>,
    #[pyo3(get, set)]
    pub departed: bool,
    #[pyo3(get, set)]
    pub superseded: bool,
    pub ahead_of: Vec<(i64, f64)>,
    #[pyo3(get, set)]
    pub tracked: i64,
    #[pyo3(get, set)]
    pub beyond: Option<f64>,
    #[pyo3(get, set)]
    pub settled: bool,
    pub history: Ring<(f64, f64, f64, f64), 16>,
    pub trail: Ring<(f64, f64), 20>,
    #[pyo3(get, set)]
    pub report: Option<(f64, f64, f64)>,
    pub lateral_reports: Ring<(f64, f64), 16>,
    #[pyo3(get, set)]
    pub calm_since: Option<f64>,
    #[pyo3(get, set)]
    pub stable_y: Option<f64>,
}

impl MapObject {
    /// `MapObject(key, source, x, y, vx, vy, stamp, stamp, origin=(x, y))`: a new measured object.
    pub fn measured(key: i64, source: i128, x: f64, y: f64, vx: f64, vy: f64, stamp: f64) -> Self {
        let mut obj = Self::new(
            key, source, x, y, vx, vy, stamp, stamp, 1, 0.2, 0., 0.25, false, false, false, false, None, false, 0.,
            0., false, 0, false, 0, false, None, false, (0., 0.), None, false, false, None, 0, None, false, None,
            None, None, None, None, None,
        );
        obj.origin = Some((x, y));
        obj
    }
}

#[pymethods]
impl MapObject {
    #[new]
    #[pyo3(signature = (key, source, x, y, vx, vy, born, seen, hits=1, existence=0.2, vehicle=0., uncertainty=0.25,
                        motion_confirmed=false, stationary=false, vision=false, structure=false, slot=None,
                        interior=false, drift=0., explained=0., radar_moving=false, doppler=0,
                        doppler_confirmed=false, forward_samples=0, vision_seen=false, doppler_recent=None,
                        admitted=false, last_motion=(0., 0.), origin=None, departed=false, superseded=false,
                        ahead_of=None, tracked=0, beyond=None, settled=false, history=None, trail=None,
                        report=None, lateral_reports=None, calm_since=None, stable_y=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        key: i64,
        source: i128,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        born: f64,
        seen: f64,
        hits: i64,
        existence: f64,
        vehicle: f64,
        uncertainty: f64,
        motion_confirmed: bool,
        stationary: bool,
        vision: bool,
        structure: bool,
        slot: Option<i64>,
        interior: bool,
        drift: f64,
        explained: f64,
        radar_moving: bool,
        doppler: i64,
        doppler_confirmed: bool,
        forward_samples: i64,
        vision_seen: bool,
        doppler_recent: Option<Vec<f64>>,
        admitted: bool,
        last_motion: (f64, f64),
        origin: Option<(f64, f64)>,
        departed: bool,
        superseded: bool,
        ahead_of: Option<Vec<(i64, f64)>>,
        tracked: i64,
        beyond: Option<f64>,
        settled: bool,
        history: Option<Vec<(f64, f64, f64, f64)>>,
        trail: Option<Vec<(f64, f64)>>,
        report: Option<(f64, f64, f64)>,
        lateral_reports: Option<Vec<(f64, f64)>>,
        calm_since: Option<f64>,
        stable_y: Option<f64>,
    ) -> Self {
        Self {
            key,
            source,
            x,
            y,
            vx,
            vy,
            born,
            seen,
            hits,
            existence,
            vehicle,
            uncertainty,
            motion_confirmed,
            stationary,
            vision,
            structure,
            slot,
            interior,
            drift,
            explained,
            radar_moving,
            doppler,
            doppler_confirmed,
            forward_samples,
            vision_seen,
            doppler_recent: Ring::from_items(doppler_recent),
            admitted,
            last_motion,
            origin,
            departed,
            superseded,
            ahead_of: ahead_of.unwrap_or_default(),
            tracked,
            beyond,
            settled,
            history: Ring::from_items(history),
            trail: Ring::from_items(trail),
            report,
            lateral_reports: Ring::from_items(lateral_reports),
            calm_since,
            stable_y,
        }
    }

    #[getter]
    fn doppler_recent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.doppler_recent.tuple(py)
    }

    #[setter]
    fn set_doppler_recent(&mut self, items: Vec<f64>) {
        self.doppler_recent = Ring::from_items(Some(items));
    }

    #[getter]
    fn ahead_of<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.ahead_of.iter().copied())
    }

    #[setter]
    fn set_ahead_of(&mut self, items: Vec<(i64, f64)>) {
        self.ahead_of = items;
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.history.tuple(py)
    }

    #[setter]
    fn set_history(&mut self, items: Vec<(f64, f64, f64, f64)>) {
        self.history = Ring::from_items(Some(items));
    }

    #[getter]
    fn trail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.trail.tuple(py)
    }

    #[setter]
    fn set_trail(&mut self, items: Vec<(f64, f64)>) {
        self.trail = Ring::from_items(Some(items));
    }

    #[getter]
    fn lateral_reports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.lateral_reports.tuple(py)
    }

    #[setter]
    fn set_lateral_reports(&mut self, items: Vec<(f64, f64)>) {
        self.lateral_reports = Ring::from_items(Some(items));
    }

    /// The dataclass's `vars()`: every field in declaration order, deque fields as `deque(..., maxlen=N)`. The names
    /// are interned, as the dataclass's attribute names are, so that the two pickle alike.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item(intern!(py, "key"), self.key)?;
        d.set_item(intern!(py, "source"), self.source)?;
        d.set_item(intern!(py, "x"), self.x)?;
        d.set_item(intern!(py, "y"), self.y)?;
        d.set_item(intern!(py, "vx"), self.vx)?;
        d.set_item(intern!(py, "vy"), self.vy)?;
        d.set_item(intern!(py, "born"), self.born)?;
        d.set_item(intern!(py, "seen"), self.seen)?;
        d.set_item(intern!(py, "hits"), self.hits)?;
        d.set_item(intern!(py, "existence"), self.existence)?;
        d.set_item(intern!(py, "vehicle"), self.vehicle)?;
        d.set_item(intern!(py, "uncertainty"), self.uncertainty)?;
        d.set_item(intern!(py, "motion_confirmed"), self.motion_confirmed)?;
        d.set_item(intern!(py, "stationary"), self.stationary)?;
        d.set_item(intern!(py, "vision"), self.vision)?;
        d.set_item(intern!(py, "structure"), self.structure)?;
        d.set_item(intern!(py, "slot"), self.slot)?;
        d.set_item(intern!(py, "interior"), self.interior)?;
        d.set_item(intern!(py, "drift"), self.drift)?;
        d.set_item(intern!(py, "explained"), self.explained)?;
        d.set_item(intern!(py, "radar_moving"), self.radar_moving)?;
        d.set_item(intern!(py, "doppler"), self.doppler)?;
        d.set_item(intern!(py, "doppler_confirmed"), self.doppler_confirmed)?;
        d.set_item(intern!(py, "forward_samples"), self.forward_samples)?;
        d.set_item(intern!(py, "vision_seen"), self.vision_seen)?;
        d.set_item(intern!(py, "doppler_recent"), self.doppler_recent.state(py)?)?;
        d.set_item(intern!(py, "admitted"), self.admitted)?;
        d.set_item(intern!(py, "last_motion"), self.last_motion)?;
        d.set_item(intern!(py, "origin"), self.origin)?;
        d.set_item(intern!(py, "departed"), self.departed)?;
        d.set_item(intern!(py, "superseded"), self.superseded)?;
        d.set_item(intern!(py, "ahead_of"), self.ahead_of(py)?)?;
        d.set_item(intern!(py, "tracked"), self.tracked)?;
        d.set_item(intern!(py, "beyond"), self.beyond)?;
        d.set_item(intern!(py, "settled"), self.settled)?;
        d.set_item(intern!(py, "history"), self.history.state(py)?)?;
        d.set_item(intern!(py, "trail"), self.trail.state(py)?)?;
        d.set_item(intern!(py, "report"), self.report)?;
        d.set_item(intern!(py, "lateral_reports"), self.lateral_reports.state(py)?)?;
        d.set_item(intern!(py, "calm_since"), self.calm_since)?;
        d.set_item(intern!(py, "stable_y"), self.stable_y)?;
        Ok(d)
    }
}
