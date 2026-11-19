use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod pymath;
mod radar_map;

#[pyfunction]
fn source_hash() -> &'static str {
    env!("CCNC_CORE_SOURCE_HASH")
}

#[pyfunction]
fn abi_version() -> u32 {
    2
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
    Ok(())
}
