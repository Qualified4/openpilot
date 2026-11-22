//! `ccnc_core.RadarMapState`: the `ccnc_map::radar::MapState` of a `CcncRadarMap` whose update runs in Rust.
//!
//! The state is attached to the map object as `_rust_state` on its first Rust update. The fields it keeps
//! (`FIELDS`) then leave the object's `__dict__`; Python reads them from the state (`CcncRadarMap.__getattr__`), as
//! copies: `objects` is a dict of `MapObject` copies, `local`, `reports` and `jumps` are dicts. Rust code borrows the
//! state directly. A map that runs the Python reference path never gets one.

use pyo3::exceptions::{PyIndexError, PyKeyError, PyOverflowError, PyRuntimeError, PyValueError, PyZeroDivisionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ccnc_map::radar::{Error, MapObjectData, MapState, Range};

use crate::map_object::{ring_state, MapObject};

/// The `CcncRadarMap` attributes the state keeps (also `_RUST_FIELDS` in hyundaicanfd_ccnc_radar_map.py).
pub const FIELDS: [&str; 15] = [
    "pose",
    "cos",
    "sin",
    "now",
    "speed",
    "odometer",
    "poses",
    "objects",
    "local",
    "next_id",
    "created",
    "evicted",
    "range_removed",
    "reports",
    "jumps",
];

/// A map error as the Python reference raises it.
pub fn py_err(error: Error) -> PyErr {
    match error {
        Error::ZeroDivision(message) => PyZeroDivisionError::new_err(message),
        Error::Index(message) => PyIndexError::new_err(message),
        Error::Key(key) => PyKeyError::new_err(key),
        Error::Value(message) => PyValueError::new_err(message),
        Error::Overflow(message) => PyOverflowError::new_err(message),
        Error::External(inner) => match inner.downcast::<PyErr>() {
            Ok(error) => *error,
            Err(_) => PyRuntimeError::new_err("ccnc_map callback failed"),
        },
    }
}

/// A Python error raised inside a map stage's callback, handed through the stage unchanged.
pub fn external(error: PyErr) -> Error {
    Error::External(Box::new(error))
}

#[pyclass(module = "ccnc_core", name = "RadarMapState")]
#[derive(Default)]
pub struct RadarMapState {
    pub st: MapState,
}

/// The state attached to a map object, if it has one.
pub fn attached<'py>(radar_map: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, RadarMapState>>> {
    let py = radar_map.py();
    let Ok(dict) = radar_map.getattr(intern!(py, "__dict__")) else { return Ok(None) };
    let Ok(dict) = dict.cast_into::<PyDict>() else { return Ok(None) };
    match dict.get_item(intern!(py, "_rust_state"))? {
        Some(state) if !state.is_none() => Ok(Some(state.cast_into::<RadarMapState>()?)),
        _ => Ok(None),
    }
}

/// The state of a map object whose update runs in Rust, attached on first use: the fields it keeps leave the
/// object's `__dict__`. Only a map that has not been updated yet gets one.
pub fn attach<'py>(radar_map: &Bound<'py, PyAny>) -> PyResult<Bound<'py, RadarMapState>> {
    if let Some(state) = attached(radar_map)? {
        return Ok(state);
    }
    let py = radar_map.py();
    let dict = radar_map.getattr(intern!(py, "__dict__"))?.cast_into::<PyDict>()?;
    if dict.get_item(intern!(py, "now"))?.is_some_and(|now| !now.is_none()) {
        return Err(PyRuntimeError::new_err("the map was updated in Python; reset it before a Rust update"));
    }
    let state = Bound::new(py, RadarMapState::default())?;
    dict.set_item(intern!(py, "_rust_state"), &state)?;
    for name in FIELDS {
        if dict.contains(name)? {
            dict.del_item(name)?;
        }
    }
    Ok(state)
}

/// `_advance`'s constants.
pub fn range(radar_map: &Bound<'_, PyAny>) -> PyResult<Range> {
    let py = radar_map.py();
    let k = |name| -> PyResult<f64> { radar_map.getattr(name)?.extract() };
    Ok(Range {
        min_x: k(intern!(py, "MIN_X"))?,
        keep_x: k(intern!(py, "KEEP_X"))?,
        max_y: k(intern!(py, "MAX_Y"))?,
        far_x: k(intern!(py, "FAR_X"))?,
        far_y: k(intern!(py, "FAR_Y"))?,
    })
}

/// `self.bridge['key']` (None without a bridge).
pub fn bridge_key(radar_map: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    let py = radar_map.py();
    let bridge = radar_map.getattr(intern!(py, "bridge"))?;
    if bridge.is_truthy()? {
        Ok(Some(bridge.get_item(intern!(py, "key"))?.extract()?))
    } else {
        Ok(None)
    }
}

/// `_advance(now, speed, yaw)` of a map whose update runs in Rust: ego's pose and the objects, in its state.
#[pyfunction]
pub fn advance_state(radar_map: &Bound<'_, PyAny>, now: f64, speed: f64, yaw: f64) -> PyResult<()> {
    crate::guard(|| {
        let py = radar_map.py();
        let holding = radar_map.getattr(intern!(py, "holding"))?.is_truthy()?;
        let (bridged, range) = (bridge_key(radar_map)?, range(radar_map)?);
        attach(radar_map)?.try_borrow_mut()?.st.advance(now, speed, yaw, holding, bridged, range);
        Ok(())
    })
}

/// `pose_at(stamp)` of a map whose update runs in Rust.
#[pyfunction]
pub fn state_pose_at(radar_map: &Bound<'_, PyAny>, stamp: f64) -> PyResult<Option<(f64, f64, f64)>> {
    crate::guard(|| Ok(attach(radar_map)?.try_borrow()?.st.pose_at(stamp)))
}

/// A map object as the Python class, a copy.
pub fn to_py<'py>(py: Python<'py>, o: &MapObjectData) -> PyResult<Bound<'py, MapObject>> {
    Bound::new(py, MapObject::from_data(o.clone()))
}

fn ring_deque<'py, T, const N: usize>(py: Python<'py>, ring: &ccnc_map::radar::Ring<T, N>) -> PyResult<Bound<'py, PyAny>>
where
    T: Copy + for<'a> IntoPyObject<'a>,
{
    ring_state(py, ring)
}

#[pymethods]
impl RadarMapState {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Back to a new map's state.
    fn reset(&mut self) {
        self.st = MapState::new();
    }

    #[getter]
    fn pose(&self) -> (f64, f64, f64) {
        self.st.pose
    }

    #[getter]
    fn cos(&self) -> f64 {
        self.st.cos
    }

    #[setter]
    fn set_cos(&mut self, value: f64) {
        self.st.cos = value;
    }

    #[getter]
    fn sin(&self) -> f64 {
        self.st.sin
    }

    #[setter]
    fn set_sin(&mut self, value: f64) {
        self.st.sin = value;
    }

    #[getter]
    fn now(&self) -> Option<f64> {
        self.st.now
    }

    #[setter]
    fn set_now(&mut self, value: Option<f64>) {
        self.st.now = value;
    }

    #[getter]
    fn speed(&self) -> f64 {
        self.st.speed
    }

    #[getter]
    fn odometer(&self) -> f64 {
        self.st.odometer
    }

    #[getter]
    fn poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ring_deque(py, &self.st.poses)
    }

    /// `{key: MapObject}`, copies in dict order.
    #[getter]
    fn objects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (key, o) in &self.st.objects {
            out.set_item(key, to_py(py, o)?)?;
        }
        Ok(out)
    }

    #[getter]
    fn local<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (key, xy) in &self.st.local {
            out.set_item(key, *xy)?;
        }
        Ok(out)
    }

    #[getter]
    fn next_id(&self) -> i64 {
        self.st.next_id
    }

    #[getter]
    fn created(&self) -> i64 {
        self.st.created
    }

    #[getter]
    fn evicted(&self) -> i64 {
        self.st.evicted
    }

    #[getter]
    fn range_removed(&self) -> i64 {
        self.st.range_removed
    }

    #[getter]
    fn reports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (track, ring) in self.st.reports.iter() {
            out.set_item(track, ring_deque(py, ring)?)?;
        }
        Ok(out)
    }

    #[getter]
    fn jumps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (track, stamp) in self.st.jumps.iter() {
            out.set_item(track, *stamp)?;
        }
        Ok(out)
    }

    /// Puts a copy of a `MapObject` in the map (and its ego-frame position, if given): for tests that build a scene.
    #[pyo3(signature = (obj, local=None))]
    fn put_object(&mut self, obj: &Bound<'_, MapObject>, local: Option<(f64, f64)>) -> PyResult<()> {
        let data = obj.try_borrow()?.data();
        let key = data.key;
        self.st.objects.insert(key, data);
        if let Some(xy) = local {
            self.st.local.insert(key, xy);
        }
        Ok(())
    }

    /// The state as the Python map keeps it: `{field: value}` for each of `FIELDS`, objects as their
    /// `__getstate__()` (the dataclass's `vars()`).
    fn snapshot<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let out = PyDict::new(py);
        for name in FIELDS {
            let value = slf.getattr(name)?;
            if name == "objects" {
                let objects = PyDict::new(py);
                for (key, o) in value.cast_into::<PyDict>()?.iter() {
                    objects.set_item(key, o.call_method0(intern!(py, "__getstate__"))?)?;
                }
                out.set_item(name, objects)?;
            } else {
                out.set_item(name, value)?;
            }
        }
        Ok(out)
    }
}
