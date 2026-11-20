//! Debounce vacant-slot crossings, then inherit (or resume) the displayed lateral position.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::pymath::{py_max, py_min};

use super::{age, lead, required, side_sign};

type Fields<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);

fn track<'py>(
    tracks: &Bound<'py, PyDict>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    tracks
        .get_item(key)?
        .filter(|value| !value.is_none())
        .map(Bound::cast_into::<PyDict>)
        .transpose()
        .map_err(Into::into)
}

fn fields<'py>(slot_fields: &Bound<'py, PyAny>, slot: usize) -> PyResult<Fields<'py>> {
    slot_fields.get_item(slot)?.extract()
}

fn stopped_slots(
    correction: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    selected: &Bound<'_, PyAny>,
    observations: &Bound<'_, PyDict>,
    previous: &Bound<'_, PyAny>,
    frame: i64,
) -> PyResult<()> {
    let py = correction.py();
    attr!(correction, "display_slots").cast::<PyDict>()?.clear();
    let tracks = attr!(correction, "tracks").cast_into::<PyDict>()?;
    let slot_fields = attr!(correction, "SLOT_FIELDS");
    for slot in [1, 2] {
        let key = selected.get_item(slot)?;
        let saved = track(&tracks, &key)?;
        let observation = observations.get_item(&key)?;
        if key.is_none() || previous.get_item(0)?.ne(&key)? || !selected.get_item(0)?.is_none() {
            continue;
        }
        let (Some(saved), Some(observation)) = (saved, observation) else {
            continue;
        };
        if observation.is_none()
            || field!(saved, "birth").ne(observation.get_item(0)?)?
            || age(frame, integer!(saved, "frame")) > 15
        {
            continue;
        }
        let crosses_into_ff = number!(saved, "output") * side_sign(slot) < 0.;
        if !crosses_into_ff {
            continue;
        }
        let (distance, _, detect) = fields(&slot_fields, slot)?;
        let ff_detect = intern!(py, "FF_DETECT");
        if values.get_item(ff_detect)?.eq(0)? && values.get_item(&detect)?.ne(0)? {
            let from_distance = values.get_item(distance)?;
            let from_lateral = field!(saved, "output").neg()?;
            let from_detect = values.get_item(&detect)?;
            values.set_item(intern!(py, "FF_DISTANCE"), from_distance)?;
            values.set_item(intern!(py, "FF_LATERAL"), from_lateral)?;
            values.set_item(ff_detect, from_detect)?;
            values.set_item(detect, 0)?;
            let mut destinations: Vec<_> = selected.try_iter()?.collect::<PyResult<_>>()?;
            destinations[0] = key;
            destinations[slot] = py.None().into_bound(py);
            correction.setattr(
                intern!(py, "display_selected"),
                PyTuple::new(py, destinations)?,
            )?;
        }
    }
    Ok(())
}

fn new_slot<'py>(
    py: Python<'py>,
    birth: &Bound<'py, PyAny>,
    slot: usize,
    stamp: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let saved = PyDict::new(py);
    put!(saved, "birth", birth);
    put!(saved, "slot", slot);
    put!(saved, "pending", py.None());
    put!(saved, "stamp", stamp);
    put!(saved, "count", 0);
    put!(saved, "start", stamp);
    Ok(saved)
}

fn interior_ff(
    correction: &Bound<'_, PyAny>,
    saved: &Bound<'_, PyDict>,
    slot: usize,
) -> PyResult<bool> {
    let Some(road) = saved.get_item(intern!(saved.py(), "display_road"))? else {
        return Ok(false);
    };
    if slot != 0 || road.is_none() || road.get_item(2)?.is_none() {
        return Ok(false);
    }
    let aligned: f64 = road.get_item(1)?.extract()?;
    let (left, right): (f64, f64) = road.get_item(2)?.extract()?;
    Ok([aligned, left, right].into_iter().all(f64::is_finite)
        && py_min(left - aligned, aligned - right)
            >= attr!(correction, "ROAD_BOUNDARY_RETURN_MARGIN").extract::<f64>()?)
}

fn debounce<'py>(
    correction: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    selected: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyDict>,
    frame: i64,
    bypass: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = correction.py();
    let previous = attr!(correction, "display_selected");
    correction.setattr(intern!(py, "display_selected"), selected)?;
    if bypass.is_truthy()? {
        return stopped_slots(correction, values, selected, observations, &previous, frame);
    }
    let slot_fields = attr!(correction, "SLOT_FIELDS");
    let mut snapshots = Vec::with_capacity(3);
    for fields in slot_fields.try_iter()? {
        let (distance, lateral, detect): Fields<'_> = fields?.extract()?;
        snapshots.push((
            values.get_item(distance)?,
            values.get_item(lateral)?,
            values.get_item(detect)?,
        ));
    }
    let active = PyDict::new(py);
    let mut destinations: Vec<_> = selected.try_iter()?.collect::<PyResult<_>>()?;
    let display_slots = attr!(correction, "display_slots").cast_into::<PyDict>()?;
    let tracks = attr!(correction, "tracks").cast_into::<PyDict>()?;
    for (slot, key) in selected.try_iter()?.enumerate() {
        let key = key?;
        let Some(observation) = observations
            .get_item(&key)?
            .filter(|value| !value.is_none())
        else {
            continue;
        };
        let birth = observation.get_item(0)?;
        let stamp = observation.get_item(1)?;
        let saved = match track(&display_slots, &key)? {
            Some(saved)
                if field!(saved, "birth").eq(&birth)?
                    && (0..=15).contains(&age(frame, integer!(saved, "frame"))) =>
            {
                saved
            }
            _ => new_slot(py, &birth, slot, &stamp)?,
        };
        let old: usize = field!(saved, "slot").extract()?;
        let (distance, lateral, detect) = snapshots
            .get(slot)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("list index out of range"))?;
        let mut common_y = if slot == 0 || slot == 2 {
            lateral.neg()?
        } else {
            lateral.clone()
        };
        let track = track(&tracks, &key)?;
        let mut aligned_y = None;
        if let Some(track) = &track {
            if !track.is_empty() {
                if let Some(output) = track.get_item(intern!(py, "output"))? {
                    common_y = if slot == 0 {
                        output
                    } else {
                        py_min(py_max(output.extract()?, -5.4), 5.4)
                            .into_pyobject(py)?
                            .into_any()
                    };
                }
                let samples = field!(track, "samples");
                if samples.is_truthy()? {
                    aligned_y = Some(samples.get_item(-1)?.get_item(2)?);
                }
            }
        }
        let y = common_y.extract::<f64>()?;
        let aligned = aligned_y.as_ref().unwrap_or(&common_y).extract::<f64>()?;
        let representable = old == 0
            || (old == 1 && py_min(y, aligned) >= 0.)
            || (old == 2 && py_max(y, aligned) <= 0.);
        let interior = match &track {
            Some(track) => interior_ff(correction, track, slot)?,
            None => false,
        };
        let immediate = attr!(correction, "ego_change_active").is_truthy()?
            && previous.get_item(0)?.is_none()
            && field!(saved, "pending").ne(slot)?
            && matches!(old, 1 | 2)
            && interior;
        if old == slot || !selected.get_item(old)?.is_none() || !representable || immediate {
            put!(saved, "slot", slot);
            put!(saved, "pending", py.None());
        } else {
            if field!(saved, "pending").ne(slot)? {
                put!(saved, "pending", slot);
                put!(saved, "start", &stamp);
                put!(saved, "count", 1);
                put!(saved, "stamp", &stamp);
            } else if stamp.ne(field!(saved, "stamp"))? {
                put!(saved, "stamp", &stamp);
                put!(saved, "count", field!(saved, "count").add(1)?);
            }
            let ready = slot == 0 || y * side_sign(slot) >= 0.;
            if ready
                && age(stamp.extract()?, integer!(saved, "start"))
                    >= attr!(correction, "SLOT_TIME").extract::<i64>()?
                && integer!(saved, "count") >= 3
            {
                put!(saved, "slot", slot);
                put!(saved, "pending", py.None());
            }
        }
        let dest: usize = field!(saved, "slot").extract()?;
        if dest != slot {
            let (from_distance, from_lateral, from_detect) = fields(&slot_fields, dest)?;
            destinations[slot] = py.None().into_bound(py);
            let target = destinations.get_mut(dest).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("list assignment index out of range")
            })?;
            *target = key.clone();
            values.set_item(slot_fields.get_item(slot)?.get_item(2)?, 0)?;
            let mapped_y = if dest == 0 || dest == 2 {
                common_y.neg()?
            } else {
                common_y
            };
            values.set_item(from_distance, distance)?;
            values.set_item(from_lateral, mapped_y)?;
            values.set_item(from_detect, detect)?;
        }
        put!(saved, "frame", frame);
        active.set_item(key, saved)?;
    }
    correction.setattr(intern!(py, "display_slots"), active)?;
    correction.setattr(
        intern!(py, "display_selected"),
        PyTuple::new(py, destinations)?,
    )?;
    Ok(())
}

fn inherit_ff<'py>(
    correction: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    frame: i64,
) -> PyResult<()> {
    let py = correction.py();
    let previous_field = intern!(py, "ff_previous_display");
    let previous = correction.getattr(previous_field)?;
    correction.setattr(previous_field, py.None())?;
    let key = attr!(correction, "display_selected").get_item(0)?;
    let tracks = attr!(correction, "tracks").cast_into::<PyDict>()?;
    let saved = track(&tracks, &key)?;
    if attr!(correction, "lane_change").is_truthy()?
        || !values.get_item(intern!(py, "FF_DETECT"))?.is_truthy()?
    {
        return Ok(());
    }
    let Some(saved) = saved else { return Ok(()) };
    if !saved
        .get_item(intern!(py, "frame"))?
        .map(|value| value.eq(frame))
        .transpose()?
        .unwrap_or(false)
    {
        return Ok(());
    }
    let ff = lead::departure(&saved)?;
    let effective = match ff.as_ref() {
        Some(ff) if truth!(ff, "released") => ff,
        _ => &saved,
    };
    let locked = effective
        .get_item(intern!(py, "locked"))?
        .map(|value| value.is_truthy())
        .transpose()?
        .unwrap_or(false);
    let stable = locked
        && !truth!(effective, "following")
        && !truth!(saved, "boundary_follow")
        && field!(saved, "reference")
            .get_item(0)?
            .eq(intern!(py, "lane"))?
        && (0..=attr!(correction, "GEOMETRY_HOLD").extract::<i64>()?)
            .contains(&age(frame, integer!(saved, "valid_frame")));
    let lateral = intern!(py, "FF_LATERAL");
    if stable
        && !previous.is_none()
        && previous.get_item(0)?.ne(&key)?
        && previous.get_item(3)?.is_truthy()?
        && (0..=15).contains(&age(frame, previous.get_item(1)?.extract()?))
        && previous.get_item(4)?.eq(field!(saved, "reference"))?
    {
        let samples = field!(saved, "samples");
        if samples.is_truthy()? && samples.get_item(-1)?.get_item(2)?.extract::<f64>()?.abs() <= 1.2
        {
            let output = previous.get_item(2)?;
            if output.extract::<f64>()?.abs() <= 0.4 {
                put!(saved, "output", &output);
                if let Some(ff) = ff {
                    if truth!(ff, "released") {
                        put!(ff, "output", &output);
                    }
                }
                values.set_item(lateral, output.neg()?)?;
            }
        }
    }
    let reference = saved.get_item(intern!(py, "reference"))?;
    correction.setattr(
        previous_field,
        (
            key,
            frame,
            values.get_item(lateral)?.neg()?,
            stable,
            reference,
        ),
    )?;
    Ok(())
}

fn inherit_side(
    correction: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    frame: i64,
) -> PyResult<()> {
    let py = correction.py();
    let previous_field = intern!(py, "side_previous_display");
    let previous = correction.getattr(previous_field)?.cast_into::<PyDict>()?;
    let drawn = PyDict::new(py);
    correction.setattr(previous_field, &drawn)?;
    if attr!(correction, "lane_change").is_truthy()? {
        return Ok(());
    }
    let tracks = attr!(correction, "tracks").cast_into::<PyDict>()?;
    let selected = attr!(correction, "display_selected");
    let slot_fields = attr!(correction, "SLOT_FIELDS");
    for slot in [1, 2] {
        let key = selected.get_item(slot)?;
        let saved = track(&tracks, &key)?;
        let (_, lateral, detect) = fields(&slot_fields, slot)?;
        if !values.get_item(detect)?.is_truthy()? {
            continue;
        }
        let Some(saved) = saved else { continue };
        if !saved
            .get_item(intern!(py, "frame"))?
            .map(|value| value.eq(frame))
            .transpose()?
            .unwrap_or(false)
        {
            continue;
        }
        let Some(road) = saved.get_item(intern!(py, "display_road"))? else {
            continue;
        };
        let (x, aligned, bounds): (f64, f64, Option<(f64, f64)>) = road.extract()?;
        let old = track(&previous, &(slot.into_pyobject(py)?.into_any()))?;
        let stable = truth!(saved, "locked")
            && !truth!(saved, "following")
            && !truth!(saved, "boundary_follow")
            && field!(saved, "hold").is_none()
            && field!(saved, "slot").eq(slot)?
            && field!(saved, "reference").eq((intern!(py, "lane"), intern!(py, "center")))?
            && match field!(saved, "ff_frame").extract::<Option<i64>>()? {
                None => true,
                Some(ff) => age(frame, ff) > attr!(correction, "RECENT_FF").extract::<i64>()?,
            }
            && attr!(correction, "center_valid")
                .get_item(slot - 1)?
                .is_truthy()?
            && (0..=attr!(correction, "GEOMETRY_HOLD").extract::<i64>()?)
                .contains(&age(frame, integer!(saved, "valid_frame")));
        let mut rate = 0.;
        let sign = if slot == 1 { 1 } else { -1 };
        let mut output = values.get_item(&lateral)?.mul(sign)?;
        if let Some(old) = old {
            let dt = age(frame, integer!(old, "frame")) as f64 * 0.01;
            let continuous = 0. < dt && dt <= 0.15;
            if field!(old, "key").eq(&key)? && continuous {
                rate = (x - number!(old, "x")) / dt;
            }
            if stable
                && truth!(old, "stable")
                && field!(old, "key").ne(&key)?
                && continuous
                && x < 30.
                && (x - number!(old, "x")).abs() <= 3.
                && (x - number!(old, "x") - number!(old, "rate") * dt).abs() <= 3.
                && (aligned - number!(old, "aligned")).abs() <= 0.35
            {
                if let Some((left, right)) = bounds {
                    let ys = [aligned, number!(old, "aligned"), number!(old, "output")];
                    if ys
                        .into_iter()
                        .all(|y| right + 0.35 <= y && y <= left - 0.35)
                    {
                        let center: f64 =
                            attr!(correction, "centers").get_item(slot - 1)?.extract()?;
                        if ys.into_iter().all(|y| (y - center).abs() <= 0.6) {
                            output = field!(old, "output");
                            put!(saved, "output", &output);
                            values.set_item(&lateral, output.mul(sign)?)?;
                        }
                    }
                }
            }
        }
        let row = PyDict::new(py);
        put!(row, "key", &key);
        put!(row, "frame", frame);
        put!(row, "x", x);
        put!(row, "rate", rate);
        put!(row, "aligned", aligned);
        put!(row, "output", output);
        put!(row, "stable", stable);
        drawn.set_item(slot, row)?;
    }
    Ok(())
}

pub(crate) fn stabilize_slots<'py>(
    correction: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    selected: &Bound<'py, PyAny>,
    observations: &Bound<'py, PyDict>,
    frame: i64,
    bypass: &Bound<'py, PyAny>,
) -> PyResult<()> {
    debounce(correction, values, selected, observations, frame, bypass)?;
    if !attr!(correction, "ego_follow_active").is_truthy()? {
        inherit_ff(correction, values, frame)?;
        return inherit_side(correction, values, frame);
    }
    let py = correction.py();
    correction.setattr(intern!(py, "ff_previous_display"), py.None())?;
    attr!(correction, "side_previous_display")
        .cast::<PyDict>()?
        .clear();
    let selected = attr!(correction, "display_selected");
    let targets = attr!(correction, "ego_targets").cast_into::<PyDict>()?;
    let drawn = attr!(correction, "ego_drawn").cast_into::<PyDict>()?;
    let slot_fields = attr!(correction, "SLOT_FIELDS");
    for (slot, key) in selected.try_iter()?.enumerate() {
        let key = key?;
        let target = targets.get_item(&key)?.filter(|value| !value.is_none());
        let (_, lateral, detect) = fields(&slot_fields, slot)?;
        let Some(target) = target else { continue };
        if target.get_item(0)?.ne(frame)?
            || target.get_item(3)?.get_item(0)?.ne(intern!(py, "lane"))?
            || !values.get_item(detect)?.is_truthy()?
        {
            continue;
        }
        let aligned: f64 = target.get_item(1)?.extract()?;
        let filtered: f64 = target.get_item(2)?.extract()?;
        let output = py_min(py_max(filtered, aligned - 0.25), aligned + 0.25);
        let lateral_y = if slot == 0 {
            py_min(12.7, py_max(-12.8, -output))
        } else {
            py_min(5.4, py_max(0., output * side_sign(slot)))
        };
        values.set_item(lateral, lateral_y)?;
        if let Some(observation) = observations
            .get_item(&key)?
            .filter(|value| !value.is_none())
        {
            let sign = if slot == 1 { 1. } else { -1. };
            drawn.set_item(key, (observation.get_item(0)?, frame, lateral_y * sign))?;
        }
    }
    Ok(())
}
