//! CPython float operations that must give the same bits as the Python code they replace: the pure ones live in
//! `ccnc_map::pymath`; those that raise Python errors are here.

pub use ccnc_map::pymath::*;

/// CPython `math_1`: a NaN result from a non-NaN argument is "math domain error".
fn math_1(x: f64, r: f64) -> pyo3::PyResult<f64> {
    if r.is_nan() && !x.is_nan() {
        return Err(pyo3::exceptions::PyValueError::new_err("math domain error"));
    }
    Ok(r)
}

/// `math.acos(x)`: the same libm acos() CPython calls.
pub fn acos(x: f64) -> pyo3::PyResult<f64> {
    math_1(x, x.acos())
}

/// `math.cos(x)`: the same libm cos() CPython calls.
pub fn cos(x: f64) -> pyo3::PyResult<f64> {
    math_1(x, x.cos())
}
