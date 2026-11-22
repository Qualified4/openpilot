//! `MapObject` of hyundaicanfd_ccnc_radar_map.py as a Rust class: the Rust loops change its fields directly.
//!
//! The constructor, the field names and `__getstate__` follow the Python dataclass, which the Python reference
//! path keeps using. A deque field reads as a tuple copy, so Python code that tries to change one in place fails
//! loudly instead of changing a copy.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::radar_map::deque;

pub use ccnc_map::radar::Ring;
use ccnc_map::radar::MapObjectData;

/// A ring as a tuple copy.
pub fn ring_tuple<'py, T, const N: usize>(py: Python<'py>, ring: &Ring<T, N>) -> PyResult<Bound<'py, PyTuple>>
where
    T: Copy + for<'a> IntoPyObject<'a>,
{
    PyTuple::new(py, ring.iter())
}

/// A ring as the Python field holds it: `deque(items, maxlen=N)`.
pub fn ring_state<'py, T, const N: usize>(py: Python<'py>, ring: &Ring<T, N>) -> PyResult<Bound<'py, PyAny>>
where
    T: Copy + for<'a> IntoPyObject<'a>,
{
    let kwargs = PyDict::new(py);
    kwargs.set_item("maxlen", N)?;
    deque(py)?.call((ring_tuple(py, ring)?,), Some(&kwargs))
}

fn ring<T: Copy, const N: usize>(items: Option<Vec<T>>) -> Ring<T, N> {
    Ring::from_items(items.into_iter().flatten())
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
    /// A Python copy of a map object of the Rust map state.
    pub fn from_data(d: MapObjectData) -> Self {
        Self {
            key: d.key,
            source: d.source,
            x: d.x,
            y: d.y,
            vx: d.vx,
            vy: d.vy,
            born: d.born,
            seen: d.seen,
            hits: d.hits,
            existence: d.existence,
            vehicle: d.vehicle,
            uncertainty: d.uncertainty,
            motion_confirmed: d.motion_confirmed,
            stationary: d.stationary,
            vision: d.vision,
            structure: d.structure,
            slot: d.slot,
            interior: d.interior,
            drift: d.drift,
            explained: d.explained,
            radar_moving: d.radar_moving,
            doppler: d.doppler,
            doppler_confirmed: d.doppler_confirmed,
            forward_samples: d.forward_samples,
            vision_seen: d.vision_seen,
            doppler_recent: d.doppler_recent,
            admitted: d.admitted,
            last_motion: d.last_motion,
            origin: d.origin,
            departed: d.departed,
            superseded: d.superseded,
            ahead_of: d.ahead_of,
            tracked: d.tracked,
            beyond: d.beyond,
            settled: d.settled,
            history: d.history,
            trail: d.trail,
            report: d.report,
            lateral_reports: d.lateral_reports,
            calm_since: d.calm_since,
            stable_y: d.stable_y,
        }
    }

    /// The object as the Rust map state keeps it.
    pub fn data(&self) -> MapObjectData {
        MapObjectData {
            key: self.key.clone(),
            source: self.source.clone(),
            x: self.x.clone(),
            y: self.y.clone(),
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            born: self.born.clone(),
            seen: self.seen.clone(),
            hits: self.hits.clone(),
            existence: self.existence.clone(),
            vehicle: self.vehicle.clone(),
            uncertainty: self.uncertainty.clone(),
            motion_confirmed: self.motion_confirmed.clone(),
            stationary: self.stationary.clone(),
            vision: self.vision.clone(),
            structure: self.structure.clone(),
            slot: self.slot.clone(),
            interior: self.interior.clone(),
            drift: self.drift.clone(),
            explained: self.explained.clone(),
            radar_moving: self.radar_moving.clone(),
            doppler: self.doppler.clone(),
            doppler_confirmed: self.doppler_confirmed.clone(),
            forward_samples: self.forward_samples.clone(),
            vision_seen: self.vision_seen.clone(),
            doppler_recent: self.doppler_recent.clone(),
            admitted: self.admitted.clone(),
            last_motion: self.last_motion.clone(),
            origin: self.origin.clone(),
            departed: self.departed.clone(),
            superseded: self.superseded.clone(),
            ahead_of: self.ahead_of.clone(),
            tracked: self.tracked.clone(),
            beyond: self.beyond.clone(),
            settled: self.settled.clone(),
            history: self.history.clone(),
            trail: self.trail.clone(),
            report: self.report.clone(),
            lateral_reports: self.lateral_reports.clone(),
            calm_since: self.calm_since.clone(),
            stable_y: self.stable_y.clone(),
        }
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
            doppler_recent: ring(doppler_recent),
            admitted,
            last_motion,
            origin,
            departed,
            superseded,
            ahead_of: ahead_of.unwrap_or_default(),
            tracked,
            beyond,
            settled,
            history: ring(history),
            trail: ring(trail),
            report,
            lateral_reports: ring(lateral_reports),
            calm_since,
            stable_y,
        }
    }

    #[getter]
    fn doppler_recent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        ring_tuple(py, &self.doppler_recent)
    }

    #[setter]
    fn set_doppler_recent(&mut self, items: Vec<f64>) {
        self.doppler_recent = Ring::from_items(items);
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
        ring_tuple(py, &self.history)
    }

    #[setter]
    fn set_history(&mut self, items: Vec<(f64, f64, f64, f64)>) {
        self.history = Ring::from_items(items);
    }

    #[getter]
    fn trail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        ring_tuple(py, &self.trail)
    }

    #[setter]
    fn set_trail(&mut self, items: Vec<(f64, f64)>) {
        self.trail = Ring::from_items(items);
    }

    #[getter]
    fn lateral_reports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        ring_tuple(py, &self.lateral_reports)
    }

    #[setter]
    fn set_lateral_reports(&mut self, items: Vec<(f64, f64)>) {
        self.lateral_reports = Ring::from_items(items);
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
        d.set_item(intern!(py, "doppler_recent"), ring_state(py, &self.doppler_recent)?)?;
        d.set_item(intern!(py, "admitted"), self.admitted)?;
        d.set_item(intern!(py, "last_motion"), self.last_motion)?;
        d.set_item(intern!(py, "origin"), self.origin)?;
        d.set_item(intern!(py, "departed"), self.departed)?;
        d.set_item(intern!(py, "superseded"), self.superseded)?;
        d.set_item(intern!(py, "ahead_of"), self.ahead_of(py)?)?;
        d.set_item(intern!(py, "tracked"), self.tracked)?;
        d.set_item(intern!(py, "beyond"), self.beyond)?;
        d.set_item(intern!(py, "settled"), self.settled)?;
        d.set_item(intern!(py, "history"), ring_state(py, &self.history)?)?;
        d.set_item(intern!(py, "trail"), ring_state(py, &self.trail)?)?;
        d.set_item(intern!(py, "report"), self.report)?;
        d.set_item(intern!(py, "lateral_reports"), ring_state(py, &self.lateral_reports)?)?;
        d.set_item(intern!(py, "calm_since"), self.calm_since)?;
        d.set_item(intern!(py, "stable_y"), self.stable_y)?;
        Ok(d)
    }
}
