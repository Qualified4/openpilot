//! Filters shared by CCNC vehicle drawings. The Python reference uses the
//! same state containers so replay can compare the state between updates.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use crate::pymath::{py_max, py_min};
use crate::radar_map::deque;

pub(crate) fn reset(filter: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
    let py = filter.py();
    filter.setattr(intern!(py, "_filtered_value"), value)?;
    filter
        .getattr(intern!(py, "_buffer"))?
        .call_method0(intern!(py, "clear"))?;
    Ok(())
}

pub(crate) fn value(filter: &Bound<'_, PyAny>) -> PyResult<f64> {
    filter
        .getattr(intern!(filter.py(), "_filtered_value"))?
        .extract()
}

fn append_median(filter: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<f64> {
    let py = filter.py();
    let buffer = filter.getattr(intern!(py, "_buffer"))?;
    buffer.call_method1(intern!(py, "append"), (target,))?;
    if buffer.len()? == 1 {
        return target.extract();
    }
    let mut recent = buffer.extract::<Vec<f64>>()?;
    recent.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(recent[recent.len() / 2])
}

pub(crate) fn fixed(filter: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<f64> {
    let py = filter.py();
    let med = append_median(filter, target)?;
    let alpha: f64 = filter.getattr(intern!(py, "_alpha"))?.extract()?;
    let value = alpha * med + (1. - alpha) * value(filter)?;
    filter.setattr(intern!(py, "_filtered_value"), value)?;
    Ok(value)
}

pub(crate) fn detect<'py>(filter: &Bound<'py, PyAny>, speed: f64) -> PyResult<Bound<'py, PyAny>> {
    let py = filter.py();
    let state = intern!(py, "_current_state");
    let next = if speed
        > filter
            .getattr(intern!(py, "_upper_bound"))?
            .extract::<f64>()?
    {
        filter.getattr(intern!(py, "_state_high"))?
    } else if speed
        < filter
            .getattr(intern!(py, "_lower_bound"))?
            .extract::<f64>()?
    {
        filter.getattr(intern!(py, "_state_low"))?
    } else {
        return filter.getattr(state);
    };
    filter.setattr(state, &next)?;
    Ok(next)
}

pub(crate) fn empty_object<'py>(cls: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    py.get_type::<PyAny>()
        .getattr(intern!(py, "__new__"))?
        .call1((cls,))
}

pub(crate) fn new_position_filter<'py>(
    class: &Bound<'py, PyAny>,
    value: f64,
    distance: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = class.py();
    let filter = empty_object(class)?;
    filter.setattr(intern!(py, "_default_value"), value)?;
    filter.setattr(intern!(py, "_filtered_value"), value)?;
    filter.setattr(
        intern!(py, "_buffer"),
        deque(py)?.call1(((value, value, value), 3))?,
    )?;
    filter.setattr(intern!(py, "_a_min"), 0.3)?;
    filter.setattr(intern!(py, "_err_max"), if distance { 4.0 } else { 0.6 })?;
    filter.setattr(intern!(py, "dt"), 0.05)?;
    filter.setattr(intern!(py, "_following_step"), false)?;
    let mode = if distance {
        intern!(py, "_apply_adaptive")
    } else {
        intern!(py, "_apply_step")
    };
    filter.setattr(intern!(py, "apply"), filter.getattr(mode)?)?;
    Ok(filter)
}

pub(crate) fn position(filter: &Bound<'_, PyAny>, target: f64, distance: bool) -> PyResult<f64> {
    let py = filter.py();
    let previous = value(filter)?;
    let hard_reset: f64 = filter.getattr(intern!(py, "_err_max"))?.extract()?;
    if distance && (target - previous).abs() > hard_reset {
        reset(filter, target)?;
        return Ok(target);
    }
    let med = append_median(filter, PyFloat::new(py, target).as_any())?;
    let error = med - previous;
    let value = if distance {
        let abs = error.abs();
        let alpha = if abs <= 1. {
            0.3
        } else if abs >= 4. {
            0.9
        } else {
            ((0.9 - 0.3) / 3.) * (abs - 1.) + 0.3
        };
        alpha * med + (1. - alpha) * previous
    } else {
        let following = intern!(py, "_following_step");
        if filter.getattr(following)?.is_truthy()? || error.abs() > hard_reset {
            let dt = filter.getattr(intern!(py, "dt"))?.extract()?;
            let step = py_max(0., dt) * 5.;
            let next = previous + py_min(error.abs(), step).copysign(error);
            filter.setattr(following, error.abs() > step)?;
            next
        } else {
            let alpha: f64 = filter.getattr(intern!(py, "_a_min"))?.extract()?;
            previous + alpha * error
        }
    };
    filter.setattr(intern!(py, "_filtered_value"), value)?;
    Ok(value)
}
