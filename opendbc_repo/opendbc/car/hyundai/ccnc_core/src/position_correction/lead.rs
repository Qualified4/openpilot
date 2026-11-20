//! FF departure uses radar motion alongside a matching vision lead.

use pyo3::exceptions::PyAttributeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::pymath::py_max;

use super::apply::{ApplyInput, SettlingSample};
use super::{
    age, append, checked_add, median, new_deque, popleft, python_float, require_bounds, required,
};

pub(super) struct VisionReport {
    x: f64,
    y: f64,
    matched: bool,
}

pub(super) fn departure<'py>(saved: &Bound<'py, PyDict>) -> PyResult<Option<Bound<'py, PyDict>>> {
    saved
        .get_item(intern!(saved.py(), "ff_departure"))?
        .filter(|value| !value.is_none())
        .map(Bound::cast_into::<PyDict>)
        .transpose()
        .map_err(Into::into)
}

fn getattr_or_none<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    match obj.getattr(name) {
        Err(err) if err.is_instance_of::<PyAttributeError>(obj.py()) => {
            Ok(obj.py().None().into_bound(obj.py()))
        }
        result => result,
    }
}

pub(super) fn update_vision<'py>(
    correction: &Bound<'py, PyAny>,
    saved: &Bound<'py, PyDict>,
    radar_x: f64,
    frame: i64,
) -> PyResult<(Bound<'py, PyAny>, Option<VisionReport>)> {
    let py = correction.py();
    let history = if let Some(history) = saved.get_item(intern!(py, "ff_vision"))? {
        history
    } else {
        let history = new_deque(py, None)?;
        put!(saved, "ff_vision", &history);
        history
    };
    let model = attr!(correction, "vision_model");
    let mut report = None;
    if !model.is_none() {
        let leads = getattr_or_none(&model, intern!(py, "leadsV3"))?;
        if leads.is_truthy()? {
            let lead = leads.get_item(0)?;
            if attr!(lead, "prob").ge(0.5)? {
                let xs = attr!(lead, "x");
                if xs.len()? > 0 {
                    let ys = attr!(lead, "y");
                    if ys.len()? > 0 {
                        let x = python_float(&xs.get_item(0)?)? - 1.52;
                        let y = -python_float(&ys.get_item(0)?)?;
                        if x.is_finite() && y.is_finite() {
                            let mut stamp = getattr_or_none(&model, intern!(py, "timestampEof"))?;
                            if !stamp.is_truthy()? {
                                stamp = (model.as_ptr() as usize).into_pyobject(py)?.into_any();
                            }
                            let matched = (x - radar_x).abs() <= py_max(3., 0.2 * radar_x);
                            if !history.is_truthy()?
                                || history.get_item(-1)?.get_item(1)?.ne(&stamp)?
                            {
                                append(&history, (frame, stamp, x, y, matched))?;
                            }
                            report = Some(VisionReport { x, y, matched });
                        }
                    }
                }
            }
        }
    }
    while history.is_truthy()? && age(frame, history.get_item(0)?.get_item(0)?.extract()?) > 100 {
        popleft(&history)?;
    }
    Ok((history, report))
}

fn matched_samples(history: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, f64)>> {
    let mut paired = Vec::new();
    for sample in history.try_iter()? {
        let sample = sample?;
        if sample.get_item(4)?.is_truthy()? {
            paired.push((
                sample.get_item(0)?.extract()?,
                sample.get_item(3)?.extract()?,
            ));
        }
    }
    Ok(paired)
}

fn update_baseline(ff: &Bound<'_, PyDict>, history: &[SettlingSample], stamp: i64) -> PyResult<()> {
    let recent = || history.iter().filter(|s| age(stamp, s.0) <= 100);
    put!(
        ff,
        "baseline",
        median(ff.py(), recent().map(|s| s.2).collect())?
    );
    put!(
        ff,
        "raw_baseline",
        median(ff.py(), recent().map(|s| s.1).collect())?
    );
    Ok(())
}

fn vision_baseline(py: Python<'_>, paired: &[(i64, f64)]) -> PyResult<Option<f64>> {
    if paired.is_empty() {
        Ok(None)
    } else {
        median(py, paired.iter().map(|s| s.1).collect()).map(Some)
    }
}

fn reset_readiness(ff: &Bound<'_, PyDict>) -> PyResult<()> {
    put!(ff, "radar_ready", ff.py().None());
    put!(ff, "vision_rejected", false);
    put!(ff, "vision_reason", ff.py().None());
    Ok(())
}

pub(super) fn relock(
    ff: &Bound<'_, PyDict>,
    history: &[SettlingSample],
    stamp: i64,
    vision_history: &Bound<'_, PyAny>,
) -> PyResult<()> {
    put!(ff, "locked", true);
    put!(ff, "following", false);
    update_baseline(ff, history, stamp)?;
    put!(ff, "trail", ff.py().None());
    let paired = matched_samples(vision_history)?;
    let baseline = vision_baseline(ff.py(), &paired)?;
    put!(ff, "vision_samples", paired);
    put!(ff, "vision_baseline", baseline);
    reset_readiness(ff)
}

fn new_departure<'py>(
    saved: &Bound<'py, PyDict>,
    history: &[SettlingSample],
    vision_history: &Bound<'py, PyAny>,
    stamp: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let py = saved.py();
    let ff = PyDict::new(py);
    update_baseline(&ff, history, stamp)?;
    put!(ff, "trail", py.None());
    put!(ff, "released", false);
    put!(ff, "locked", true);
    put!(ff, "following", false);
    put!(ff, "output", field!(saved, "output"));
    put!(ff, "stamp", stamp);
    put!(ff, "catchup_until", -1000);
    put!(ff, "release_frame", -1000);
    let paired = matched_samples(vision_history)?;
    let baseline = vision_baseline(py, &paired)?;
    put!(ff, "vision_samples", paired);
    put!(ff, "vision_baseline", py.None());
    reset_readiness(&ff)?;
    if let Some(baseline) = baseline {
        put!(ff, "vision_baseline", baseline);
    }
    put!(saved, "ff_departure", &ff);
    Ok(ff)
}

fn update_trail(ff: &Bound<'_, PyDict>, input: &ApplyInput<'_, '_>, stamp: i64) -> PyResult<()> {
    let py = ff.py();
    let dev = input.aligned - number!(ff, "baseline");
    if dev.abs() < 0.25 {
        reset_readiness(ff)?;
    }
    let sign = if dev >= 0. { 1 } else { -1 };
    let (left, right) = require_bounds(input.bounds)?;
    let distance = f64::from(sign) * ((if sign > 0 { left } else { right }) - input.aligned);
    let old = field!(ff, "trail");
    let mut trail = if dev.abs() < 0.5 || old.is_none() {
        None
    } else {
        let old = old.cast_into::<PyDict>()?;
        if integer!(old, "sign") != i64::from(sign)
            || age(stamp, integer!(ff, "stamp")) > 15
            || number!(old, "peak") - dev.abs() > 0.1
        {
            None
        } else {
            Some(old)
        }
    };
    if dev.abs() >= 0.5 {
        let path = if let Some(path) = trail.take() {
            path
        } else {
            let path = PyDict::new(py);
            put!(path, "sign", sign);
            put!(path, "start", stamp);
            put!(path, "count", 0);
            put!(path, "peak", dev.abs());
            put!(path, "progress", stamp);
            put!(path, "distance", distance);
            path
        };
        if dev.abs() > number!(path, "peak") + 1e-6 {
            put!(path, "peak", dev.abs());
            put!(path, "progress", stamp);
        }
        let count = field!(path, "count").add(1)?;
        put!(path, "count", &count);
        if age(stamp, integer!(path, "progress")) <= 25 {
            if age(stamp, integer!(path, "start")) >= 30
                && count.ge(3)?
                && distance < number!(path, "distance")
                && f64::from(sign) * (input.y - number!(ff, "raw_baseline")) >= 0.35
                && field!(ff, "radar_ready").is_none()
                && !truth!(ff, "vision_rejected")
            {
                put!(ff, "radar_ready", input.frame);
                put!(ff, "vision_reason", intern!(py, "waiting"));
                if field!(ff, "vision_baseline").is_none() {
                    put!(ff, "vision_rejected", true);
                    put!(ff, "vision_reason", intern!(py, "no_baseline"));
                }
            }
            trail = Some(path);
        }
    }
    put!(ff, "trail", trail);
    put!(ff, "stamp", stamp);
    Ok(())
}

pub(super) fn track_departure<'py>(
    saved: &Bound<'py, PyDict>,
    ff: Option<Bound<'py, PyDict>>,
    history: &[SettlingSample],
    vision_history: &Bound<'py, PyAny>,
    vision: Option<&VisionReport>,
    input: &ApplyInput<'_, 'py>,
    stamp: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let ff = match ff {
        Some(ff) => ff,
        None => new_departure(saved, history, vision_history, stamp)?,
    };
    if truth!(ff, "locked") && !truth!(ff, "following") {
        if (input.aligned - number!(ff, "baseline")).abs() <= 0.25 {
            update_baseline(&ff, history, stamp)?;
            let samples = field!(ff, "vision_samples");
            let mut paired = Vec::with_capacity(samples.len()?);
            for sample in samples.try_iter()? {
                let sample: (i64, f64) = sample?.extract()?;
                if age(input.frame, sample.0) <= 100 {
                    paired.push(sample);
                }
            }
            if let Some(vision) = vision.filter(|v| v.matched) {
                paired.push((input.frame, vision.y));
            }
            put!(ff, "vision_samples", &paired);
            put!(ff, "vision_baseline", vision_baseline(ff.py(), &paired)?);
        }
        update_trail(&ff, input, stamp)?;
    }
    Ok(ff)
}

pub(super) fn release_departure(
    saved: &Bound<'_, PyDict>,
    ff: &Bound<'_, PyDict>,
    history: &Bound<'_, PyAny>,
    vision: Option<&VisionReport>,
    input: &ApplyInput<'_, '_>,
    stamp: i64,
) -> PyResult<()> {
    if !truth!(ff, "locked") {
        return Ok(());
    }
    let Some(ready) = field!(ff, "radar_ready").extract::<Option<i64>>()? else {
        return Ok(());
    };
    if truth!(ff, "vision_rejected") {
        return Ok(());
    }
    let dev = input.aligned - number!(ff, "baseline");
    let sign = if dev >= 0. { 1 } else { -1 };
    let trail = field!(ff, "trail");
    let mut still_moving = false;
    if !trail.is_none() && dev.abs() >= 0.5 {
        let trail = trail.cast::<PyDict>()?;
        still_moving = i64::from(sign) == integer!(trail, "sign")
            && number!(trail, "peak") - dev.abs() <= 0.1
            && age(stamp, integer!(trail, "progress")) <= 25
            && f64::from(sign) * (input.y - number!(ff, "raw_baseline")) >= 0.35;
    }
    let mut moved = false;
    if let Some(baseline) = field!(ff, "vision_baseline").extract::<Option<f64>>()? {
        for sample in history.try_iter()? {
            let sample = sample?;
            let frame: i64 = sample.get_item(0)?.extract()?;
            if age(ready, frame) <= 50
                && age(input.frame, frame) <= 50
                && sample.get_item(4)?.is_truthy()?
                && f64::from(sign) * (sample.get_item(3)?.extract::<f64>()? - baseline) >= 0.3
            {
                moved = true;
                break;
            }
        }
    }
    let farther = vision.is_some_and(|v| !v.matched && v.x > input.x + 5.);
    let py = ff.py();
    let elapsed = age(input.frame, ready);
    if elapsed <= 50 && still_moving && (moved || farther) {
        let output = if truth!(ff, "released") {
            field!(ff, "output")
        } else {
            field!(saved, "output")
        };
        put!(ff, "output", output);
        put!(ff, "released", true);
        put!(ff, "locked", false);
        put!(ff, "following", true);
        put!(ff, "catchup_until", checked_add(input.frame, 50)?);
        put!(ff, "release_frame", input.frame);
        put!(
            ff,
            "vision_reason",
            if moved {
                intern!(py, "moved")
            } else {
                intern!(py, "farther")
            }
        );
    } else if elapsed >= 50 {
        put!(ff, "vision_rejected", true);
        put!(ff, "vision_reason", intern!(py, "timeout"));
    }
    Ok(())
}
