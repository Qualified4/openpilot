use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod consts;
mod correct;
mod display_filter;
mod lanes;
mod live_tracks;
mod map_object;
mod map_state;
mod model_v2;
mod observe;
mod position_correction;
mod pos;
mod pymath;
mod radar_map;
mod rear;
mod select;
mod update;
mod vehicle_display;
mod walls;

#[pyfunction]
fn source_hash() -> &'static str {
    env!("CCNC_CORE_SOURCE_HASH")
}

#[pyfunction]
fn abi_version() -> u32 {
    12
}

/// A Rust panic reaches Python as PanicException, a BaseException that the display's `except Exception`
/// does not catch: it would end card. Entry points run through this, so a panic turns the display off instead.
pub(crate) fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(PyRuntimeError::new_err("ccnc_core panicked")))
}

/// For the loader test only: a panic behind the guard.
#[pyfunction]
fn _panic_for_test() -> PyResult<()> {
    guard(|| panic!("requested by test"))
}

/// For the parity test only: Python 3.12 `sum()` of each list of floats (None for an empty one, where Python
/// returns the int 0).
#[pyfunction]
fn _sum_for_test(lists: Vec<Vec<f64>>) -> PyResult<Vec<Option<f64>>> {
    guard(|| Ok(lists.into_iter().map(pymath::py_sum).collect()))
}

/// For the parity test only: `math.<name>` of each value, as the Rust map computes it.
#[pyfunction]
fn _math_for_test(name: &str, values: Vec<f64>) -> PyResult<Vec<f64>> {
    guard(|| {
        let f: fn(f64) -> PyResult<f64> = match name {
            "degrees" => |x| Ok(pymath::degrees(x)),
            "radians" => |x| Ok(pymath::radians(x)),
            "acos" => |x| pymath::acos(x).map_err(map_state::py_err),
            "cos" => |x| pymath::cos(x).map_err(map_state::py_err),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(name.to_owned())),
        };
        values.into_iter().map(f).collect()
    })
}

/// For the parity test only: `_lane(x, y, moving=moving, settled=settled)` of the map as it is now.
#[pyfunction]
fn _lane_for_test(
    radar_map: &Bound<'_, PyAny>,
    x: f64,
    y: f64,
    moving: bool,
    settled: bool,
) -> PyResult<(Option<i64>, bool, f64)> {
    guard(|| {
        let mut view = lanes::LaneCache::new(radar_map);
        let lanes = view.get()?;
        lanes.changing = radar_map.call_method0("_model_changing")?.is_truthy()?;
        lanes.lane(x, y, moving, settled, &[]).map_err(map_state::py_err)
    })
}

/// The Rust state of a map whose update runs in Rust (`map_state::attach`): made on first use.
#[pyfunction]
fn attach_state<'py>(radar_map: &Bound<'py, PyAny>) -> PyResult<Bound<'py, map_state::RadarMapState>> {
    guard(|| map_state::attach(radar_map))
}

/// For the parity test only: `_associate(measurements, stamp)` of the map's Rust state, as {measurement index: key}.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn _associate_for_test<'py>(
    radar_map: &Bound<'py, PyAny>,
    measurements: Vec<(i128, f64, f64, f64, f64, f64, f64, f64, bool, i32, bool)>,
    stamp: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    guard(|| {
        use ccnc_map::radar::associate::{associate, AssociateIn};
        use ccnc_map::radar::observe::Measurement;
        let py = radar_map.py();
        let k = consts::read(radar_map)?;
        let inp = AssociateIn {
            holding: radar_map.getattr("holding")?.is_truthy()?,
            lane_change: radar_map.getattr("lane_change")?.is_truthy()?,
            selected: select::read_selected(radar_map)?,
        };
        let ms: Vec<Measurement> = measurements
            .into_iter()
            .map(|(source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked)| Measurement {
                source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked,
            })
            .collect();
        let state = map_state::attach(radar_map)?;
        let mut lanes = lanes::LaneCache::new(radar_map);
        let pairs = {
            let state = state.try_borrow()?;
            associate(&state.st, &k, inp, lanes.get()?, &ms, stamp).map_err(map_state::py_err)?
        };
        let out = pyo3::types::PyDict::new(py);
        for (i, key) in pairs {
            out.set_item(i, key)?;
        }
        Ok(out)
    })
}

#[pymodule]
fn ccnc_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(source_hash, module)?)?;
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(_panic_for_test, module)?)?;
    module.add_class::<map_object::MapObject>()?;
    module.add_class::<map_state::RadarMapState>()?;
    module.add_function(wrap_pyfunction!(attach_state, module)?)?;
    module.add_function(wrap_pyfunction!(map_state::advance_state, module)?)?;
    module.add_function(wrap_pyfunction!(map_state::state_pose_at, module)?)?;
    module.add_function(wrap_pyfunction!(update::update, module)?)?;
    module.add_function(wrap_pyfunction!(update::_update_calls_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::live_tracks_copy, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::_parse_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::live_tracks_schema, module)?)?;
    module.add_function(wrap_pyfunction!(model_v2::update_lanes, module)?)?;
    module.add_function(wrap_pyfunction!(
        model_v2::_update_lanes_calls_for_test,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(model_v2::_parse_model_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(model_v2::model_v2_schema, module)?)?;
    module.add_function(wrap_pyfunction!(_sum_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(_math_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(_lane_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(_associate_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(correct::correct_positions, module)?)?;
    module.add_function(wrap_pyfunction!(correct::_position_apply_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(rear::_update_rear_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(vehicle_display::update_display, module)?)?;
    module.add_class::<pos::PosState>()?;
    module.add_function(wrap_pyfunction!(pos::_pos_run_for_test, module)?)?;
    Ok(())
}
