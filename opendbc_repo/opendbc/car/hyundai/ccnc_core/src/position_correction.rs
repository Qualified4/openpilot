//! The map-only Rust path shares the selector's Python state. Keep dict writes in Python order
//! and mutate the original deques, including those with a maxlen.

use std::cmp::Ordering;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString};

macro_rules! field {
    ($dict:expr, $key:literal) => {
        required(&$dict, pyo3::intern!($dict.py(), $key))?
    };
}

macro_rules! number {
    ($dict:expr, $key:literal) => {
        field!($dict, $key).extract::<f64>()?
    };
}

macro_rules! integer {
    ($dict:expr, $key:literal) => {
        field!($dict, $key).extract::<i64>()?
    };
}

macro_rules! truth {
    ($dict:expr, $key:literal) => {
        field!($dict, $key).is_truthy()?
    };
}

macro_rules! put {
    ($dict:expr, $key:literal, $value:expr) => {
        $dict.set_item(pyo3::intern!($dict.py(), $key), $value)?
    };
}

macro_rules! attr {
    ($obj:expr, $name:literal) => {
        $obj.getattr(pyo3::intern!($obj.py(), $name))?
    };
}

mod apply;
mod lead;
mod slots;

pub(crate) use apply::{apply, ApplyInput};
pub(crate) use slots::stabilize_slots;

// Follow Python dict subscription when a required field is absent.
fn required<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
}

fn pop<'py>(dict: &Bound<'py, PyDict>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = dict.py();
    dict.call_method1(intern!(py, "pop"), (key, py.None()))
}

fn new_deque<'py>(py: Python<'py>, maxlen: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
    static DEQUE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let deque = DEQUE.get_or_try_init(py, || {
        py.import(intern!(py, "collections"))?
            .getattr(intern!(py, "deque"))
            .map(Bound::unbind)
    })?;
    match maxlen {
        Some(limit) => deque.bind(py).call1(((), limit)),
        None => deque.bind(py).call0(),
    }
}

fn clear(deque: &Bound<'_, PyAny>) -> PyResult<()> {
    deque.call_method0(intern!(deque.py(), "clear"))?;
    Ok(())
}

fn popleft(deque: &Bound<'_, PyAny>) -> PyResult<()> {
    deque.call_method0(intern!(deque.py(), "popleft"))?;
    Ok(())
}

fn append<'py, T>(deque: &Bound<'py, PyAny>, sample: T) -> PyResult<()>
where
    T: IntoPyObject<'py>,
{
    deque.call_method1(intern!(deque.py(), "append"), (sample,))?;
    Ok(())
}

fn read_samples<'py, T>(deque: &Bound<'py, PyAny>) -> PyResult<Vec<T>>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    deque.try_iter()?.map(|sample| sample?.extract()).collect()
}

// With NaN, Python's sort does not impose a total order. Keep that ordering by calling Python.
fn sort_values(py: Python<'_>, values: &mut Vec<f64>) -> PyResult<()> {
    if values.iter().any(|value| value.is_nan()) {
        *values = py
            .import(intern!(py, "builtins"))?
            .getattr(intern!(py, "sorted"))?
            .call1((&*values,))?
            .extract()?;
    } else {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    }
    Ok(())
}

fn median(py: Python<'_>, mut values: Vec<f64>) -> PyResult<f64> {
    sort_values(py, &mut values)?;
    let middle = values.len() / 2;
    let hi = *values
        .get(middle)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("list index out of range"))?;
    Ok(if values.len() & 1 == 0 {
        (values[middle - 1] + hi) / 2.
    } else {
        hi
    })
}

fn side_sign(slot: usize) -> f64 {
    if slot == 1 {
        1.
    } else {
        -1.
    }
}

fn on_road(bounds: Option<(f64, f64)>, y: f64) -> bool {
    bounds.is_some_and(|(left, right)| right <= y && y <= left)
}

// Python counts control frames with arbitrary-width integers. The map uses i64 counters;
// saturating differences preserve the time-window comparisons even across their extremes.
fn age(now: i64, then: i64) -> i64 {
    now.saturating_sub(then)
}

// The vision input uses float(), which also accepts textual values.
fn python_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    use pyo3::types::PyFloat;
    if let Ok(value) = value.cast_exact::<PyFloat>() {
        return Ok(value.value());
    }
    value.py().get_type::<PyFloat>().call1((value,))?.extract()
}

fn require_bounds(bounds: Option<(f64, f64)>) -> PyResult<(f64, f64)> {
    bounds.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("'NoneType' object is not subscriptable")
    })
}

fn checked_add(frame: i64, offset: i64) -> PyResult<i64> {
    frame
        .checked_add(offset)
        .ok_or_else(|| PyValueError::new_err("CCNC correction frame out of range"))
}
