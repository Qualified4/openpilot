use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod consts;
mod lanes;
mod live_tracks;
mod map_object;
mod observe;
mod pymath;
mod radar_map;
mod select;
mod update;
mod walls;

#[pyfunction]
fn source_hash() -> &'static str {
    env!("CCNC_CORE_SOURCE_HASH")
}

#[pyfunction]
fn abi_version() -> u32 {
    6
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
            "acos" => pymath::acos,
            "cos" => pymath::cos,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(name.to_owned())),
        };
        values.into_iter().map(f).collect()
    })
}

/// For the parity test only: `_lane(x, y, moving=moving, settled=settled)` of the map as it is now.
#[pyfunction]
fn _lane_for_test(radar_map: &Bound<'_, PyAny>, x: f64, y: f64, moving: bool, settled: bool) -> PyResult<(Option<i64>, bool, f64)> {
    guard(|| lanes::LaneCache::new(radar_map).get()?.lane(x, y, moving, settled, &[]))
}

#[pymodule]
fn ccnc_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(source_hash, module)?)?;
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(_panic_for_test, module)?)?;
    module.add_class::<radar_map::RadarPoints>()?;
    module.add_class::<radar_map::AssociateInput>()?;
    module.add_function(wrap_pyfunction!(radar_map::radar_points, module)?)?;
    module.add_function(wrap_pyfunction!(radar_map::measurements, module)?)?;
    module.add_function(wrap_pyfunction!(radar_map::associate_input, module)?)?;
    module.add_function(wrap_pyfunction!(radar_map::associate, module)?)?;
    module.add_class::<map_object::MapObject>()?;
    module.add_function(wrap_pyfunction!(observe::observe, module)?)?;
    module.add_function(wrap_pyfunction!(observe::advance, module)?)?;
    module.add_function(wrap_pyfunction!(observe::forget_lateral, module)?)?;
    module.add_class::<walls::WallRows>()?;
    module.add_function(wrap_pyfunction!(walls::walls, module)?)?;
    module.add_function(wrap_pyfunction!(walls::bands, module)?)?;
    module.add_function(wrap_pyfunction!(update::update, module)?)?;
    module.add_function(wrap_pyfunction!(update::_update_calls_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::live_tracks_copy, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::_parse_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(live_tracks::live_tracks_schema, module)?)?;
    module.add_function(wrap_pyfunction!(_sum_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(_math_for_test, module)?)?;
    module.add_function(wrap_pyfunction!(_lane_for_test, module)?)?;
    Ok(())
}
