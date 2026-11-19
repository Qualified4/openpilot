use pyo3::prelude::*;

#[pyfunction]
fn source_hash() -> &'static str {
    env!("CCNC_CORE_SOURCE_HASH")
}

#[pyfunction]
fn abi_version() -> u32 {
    1
}

#[pymodule]
fn ccnc_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(source_hash, module)?)?;
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    Ok(())
}
