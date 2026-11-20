use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::pymath::{py_max, py_min, py_sum};

use super::{
    age, append, checked_add, clear, lead, median, new_deque, on_road, pop, popleft, read_samples,
    require_bounds, required, side_sign, sort_values,
};

pub(crate) struct ApplyInput<'a, 'py> {
    pub key: &'a Bound<'py, PyAny>,
    pub x: f64,
    pub y: f64,
    pub fresh: bool,
    pub aligned: f64,
    pub filtered: f64,
    pub slot: usize,
    pub reference: &'a Bound<'py, PyTuple>,
    pub bounds: Option<(f64, f64)>,
    pub observation: &'a Bound<'py, PyAny>,
    pub frame: i64,
    pub reliable: bool,
    pub lateral_filter: Option<&'a Bound<'py, PyAny>>,
}

type MotionSample = (i64, f64, f64);
pub(super) type SettlingSample = (i64, f64, f64, f64, bool);

#[allow(clippy::too_many_arguments)]
fn new_track<'py>(
    py: Python<'py>,
    birth: &Bound<'py, PyAny>,
    frame: i64,
    locked: bool,
    inside: bool,
    output: f64,
    slot: usize,
    reference: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyDict>> {
    let saved = PyDict::new(py);
    put!(saved, "birth", birth);
    put!(saved, "frame", frame);
    put!(saved, "stamp", py.None());
    put!(saved, "samples", new_deque(py, None)?);
    put!(saved, "settling", new_deque(py, None)?);
    put!(saved, "locked", locked);
    put!(saved, "following", !locked);
    put!(saved, "valid_frame", -1000);
    put!(saved, "output", output);
    put!(saved, "slot", slot);
    put!(saved, "reference", reference);
    put!(saved, "slot_frame", -1000);
    put!(saved, "boundary_follow", !inside);
    put!(saved, "fast", new_deque(py, Some(8))?);
    put!(saved, "fast_speed", 0.);
    put!(saved, "raw", new_deque(py, None)?);
    put!(saved, "drift", new_deque(py, None)?);
    put!(saved, "hold", py.None());
    put!(saved, "ff_frame", py.None());
    put!(saved, "jumped", false);
    put!(saved, "inward_frame", -1000);
    Ok(saved)
}

fn slow_outward(
    correction: &Bound<'_, PyAny>,
    saved: &Bound<'_, PyDict>,
    slot: usize,
    stamp: i64,
    net: f64,
    frame: i64,
) -> PyResult<bool> {
    let drift = field!(saved, "drift");
    if slot == 0 || !truth!(saved, "jumped") || !drift.is_truthy()? || net * side_sign(slot) <= 0. {
        return Ok(false);
    }
    let first: (i64, f64) = drift.get_item(0)?.extract()?;
    if stamp <= first.0 {
        return Ok(false);
    }
    if let Some(ff_frame) = field!(saved, "ff_frame").extract::<Option<i64>>()? {
        if age(frame, ff_frame) <= attr!(correction, "RECENT_FF").extract::<i64>()? {
            return Ok(false);
        }
    }
    let last: (i64, f64) = drift.get_item(-1)?.extract()?;
    Ok((last.1 - first.1) / (age(stamp, first.0) as f64 * 0.01)
        <= attr!(correction, "DRIFT_SPEED").extract::<f64>()?)
}

fn nets(first: MotionSample, last: MotionSample) -> [f64; 2] {
    [last.1 - first.1, last.2 - first.2]
}

fn ongoing_motion(samples: &[MotionSample], stamp: i64, total: [f64; 2]) -> bool {
    let mut recent = samples.iter().filter(|sample| age(stamp, sample.0) <= 20);
    let Some(first) = recent.next() else {
        return false;
    };
    let Some(last) = recent
        .nth(1)
        .map(|third| recent.next_back().unwrap_or(third))
    else {
        return false;
    };
    if age(stamp, first.0) < 10 {
        return false;
    }
    let span = age(stamp, first.0) as f64 * 0.01;
    nets(*first, *last)
        .into_iter()
        .zip(total)
        .all(|(net, total)| net * total > 0. && net.abs() >= py_max(0.04, 0.15 * span))
}

fn coherent_steps(
    samples: &[MotionSample],
    total: [f64; 2],
    largest: f64,
    allow: impl Fn(f64) -> bool,
) -> bool {
    total.into_iter().enumerate().all(|(axis, net)| {
        let net = net.abs();
        if !allow(net) {
            return false;
        }
        let steps = || samples.windows(2).map(|w| nets(w[0], w[1])[axis].abs());
        let sum = py_sum(steps()).unwrap_or(0.);
        let max = steps().reduce(py_max).unwrap_or(0.);
        net >= 0.8 * sum && max <= largest * net
    })
}

fn prune(deque: &Bound<'_, PyAny>, stamp: i64, window: i64) -> PyResult<()> {
    while deque.len()? > 1 && age(stamp, deque.get_item(1)?.get_item(0)?.extract()?) >= window {
        popleft(deque)?;
    }
    Ok(())
}

fn update_side(
    correction: &Bound<'_, PyAny>,
    saved: &Bound<'_, PyDict>,
    input: &ApplyInput<'_, '_>,
    stamp: i64,
) -> PyResult<()> {
    let py = saved.py();
    let raw = field!(saved, "raw");
    let drift = field!(saved, "drift");
    let hold = field!(saved, "hold");
    let sign = side_sign(input.slot);
    let outward = sign * input.y;
    let jump_time = attr!(correction, "JUMP_TIME").extract()?;
    while raw.is_truthy()? && age(stamp, raw.get_item(0)?.get_item(0)?.extract()?) > jump_time {
        popleft(&raw)?;
    }
    if !hold.is_none() {
        let hold = hold.cast::<PyDict>()?;
        let after = field!(hold, "after");
        let on_time = attr!(correction, "JUMP_ON_TIME").extract()?;
        let on = attr!(correction, "JUMP_ON").extract::<f64>()? - 1e-6;
        let mut still_moving = false;
        for sample in after.try_iter()? {
            let (s, o): (i64, f64) = sample?.extract()?;
            if age(stamp, s) <= on_time && o - outward >= on {
                still_moving = true;
                break;
            }
        }
        if still_moving {
            put!(saved, "hold", py.None());
            clear(&raw)?;
        } else if age(stamp, integer!(hold, "start"))
            > attr!(correction, "HOLD_AFTER_JUMP").extract::<i64>()?
        {
            put!(saved, "hold", py.None());
        } else {
            append(&after, (stamp, outward))?;
        }
    } else if raw.is_truthy()? {
        let raw_samples: Vec<(i64, f64)> = read_samples(&raw)?;
        let peak = raw_samples.into_iter().map(|(_, y)| y).reduce(py_max);
        let jump_size = attr!(correction, "JUMP_SIZE").extract::<f64>()? - 1e-6;
        if peak.is_some_and(|peak| peak - outward >= jump_size)
            && (input.x < attr!(correction, "INWARD_KEEP_X").extract::<f64>()?
                || age(input.frame, integer!(saved, "inward_frame"))
                    > attr!(correction, "INWARD_KEEP_TIME").extract::<i64>()?)
        {
            let hold = PyDict::new(py);
            let after = new_deque(py, None)?;
            append(&after, (stamp, outward))?;
            put!(hold, "start", stamp);
            put!(hold, "after", after);
            put!(saved, "hold", hold);
            put!(saved, "jumped", true);
        }
    }
    append(&raw, (stamp, outward))?;
    append(&drift, (stamp, sign * input.aligned))?;
    prune(&drift, stamp, attr!(correction, "DRIFT_TIME").extract()?)
}

fn follow(
    saved: &Bound<'_, PyDict>,
    fast_speed: Option<f64>,
    side: bool,
    slot: usize,
    net: f64,
    frame: i64,
) -> PyResult<()> {
    if let Some(speed) = fast_speed {
        put!(saved, "fast_speed", speed);
    }
    put!(saved, "locked", false);
    put!(saved, "following", true);
    if side && net * side_sign(slot) < 0. {
        put!(saved, "inward_frame", frame);
    }
    Ok(())
}

fn fast_motion(
    correction: &Bound<'_, PyAny>,
    saved: &Bound<'_, PyDict>,
    input: &ApplyInput<'_, '_>,
    stamp: i64,
    center_valid: bool,
    common_road: bool,
    side: bool,
) -> PyResult<()> {
    let fast = field!(saved, "fast");
    if fast.is_truthy()? {
        let (last, raw, aligned): MotionSample = fast.get_item(-1)?.extract()?;
        let elapsed = age(stamp, last) as f64 * 0.01;
        let geometry = (input.aligned - input.y) - (aligned - raw);
        if !(0. < elapsed && elapsed <= 0.15) || geometry.abs() > 0.04 + 3. * elapsed {
            clear(&fast)?;
        }
    }
    append(&fast, (stamp, input.y, input.aligned))?;
    let continuing = number!(saved, "fast_speed") > 0.;
    put!(saved, "fast_speed", 0.);
    if fast.len()? < 3
        || age(stamp, fast.get_item(0)?.get_item(0)?.extract()?)
            < attr!(correction, "FAST_TIME").extract::<i64>()?
    {
        return Ok(());
    }
    if center_valid && input.reliable {
        let (left, right) = require_bounds(input.bounds)?;
        let history: Vec<MotionSample> = read_samples(&fast)?;
        let (Some(first), Some(last)) = (history.first(), history.last()) else {
            return Ok(());
        };
        let total = nets(*first, *last);
        let span = age(stamp, first.0) as f64 * 0.01;
        let crossing = !on_road(input.bounds, input.aligned);
        let boundary = if total[1] > 0. { left } else { right };
        let approaching = crossing
            || (boundary - input.aligned).abs()
                <= attr!(correction, "FAST_BOUNDARY_MARGIN").extract::<f64>()?;
        let recent = (0..=30).contains(&age(input.frame, integer!(saved, "slot_frame")));
        let minimum = if common_road && !(crossing || recent || continuing) {
            attr!(correction, "ROAD_FAST_DISTANCE").extract()?
        } else {
            0.4
        };
        if (approaching || continuing || recent)
            && ongoing_motion(&history, stamp, total)
            && total[0] * total[1] > 0.
            && coherent_steps(&history, total, 0.7, |net| {
                (0.8..=15.).contains(&(net / span)) && net >= minimum
            })
            && field!(saved, "hold").is_none()
            && !slow_outward(correction, saved, input.slot, stamp, total[1], input.frame)?
        {
            follow(
                saved,
                Some(total[1].abs() / span),
                side,
                input.slot,
                total[1],
                input.frame,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ordinary_motion(
    correction: &Bound<'_, PyAny>,
    saved: &Bound<'_, PyDict>,
    samples: &Bound<'_, PyAny>,
    input: &ApplyInput<'_, '_>,
    stamp: i64,
    center_valid: bool,
    common_road: bool,
    side: bool,
    settle_time: i64,
) -> PyResult<()> {
    if samples.is_truthy()? {
        let (last, raw, aligned): MotionSample = samples.get_item(-1)?.extract()?;
        let dt = age(stamp, last) as f64 * 0.01;
        let raw_step = input.y - raw;
        let geometry_step = (input.aligned - input.y) - (aligned - raw);
        let valid_dt = 0. < dt && dt <= 0.15;
        if !valid_dt {
            clear(&field!(saved, "settling"))?;
        }
        if !valid_dt || raw_step.abs() > 0.04 + 2. * dt || geometry_step.abs() > 0.04 + 3. * dt {
            clear(samples)?;
        }
    }
    append(samples, (stamp, input.y, input.aligned))?;
    prune(samples, stamp, settle_time)?;
    if number!(saved, "fast_speed") == 0. && center_valid && input.reliable && samples.len()? >= 5 {
        let mut moving: Vec<MotionSample> = read_samples(samples)?;
        moving.retain(|sample| age(stamp, sample.0) <= 70);
        if moving.len() >= 5
            && age(stamp, moving[0].0) >= attr!(correction, "MOVE_TIME").extract::<i64>()?
        {
            let span = age(stamp, moving[0].0) as f64 * 0.01;
            let total = nets(moving[0], moving[moving.len() - 1]);
            let distance = if common_road {
                attr!(correction, "ROAD_MOVE_DISTANCE").extract()?
            } else {
                attr!(correction, "MOVE_DISTANCE").extract()?
            };
            let min = py_max(
                distance,
                attr!(correction, "MOVE_SPEED").extract::<f64>()? * span,
            );
            if ongoing_motion(&moving, stamp, total)
                && total[0] * total[1] > 0.
                && coherent_steps(&moving, total, 0.5, |net| net >= min)
                && field!(saved, "hold").is_none()
                && !slow_outward(correction, saved, input.slot, stamp, total[1], input.frame)?
            {
                follow(saved, None, side, input.slot, total[1], input.frame)?;
            }
        }
    }
    Ok(())
}

fn settled(
    correction: &Bound<'_, PyAny>,
    evidence: &[SettlingSample],
    stamp: i64,
    common_road: bool,
) -> PyResult<bool> {
    let py = correction.py();
    let span = age(stamp, evidence[0].0) as f64 * 0.01;
    let values: Vec<_> = evidence.iter().map(|s| s.3).collect();
    let mut ordered = values.clone();
    sort_values(py, &mut ordered)?;
    let trim = (values.len() / 10).max(1);
    let edge = (values.len() / 3).max(2);
    let start = median(py, values[..edge].to_vec())?;
    let end = median(py, values[values.len() - edge..].to_vec())?;
    let start_samples = &evidence[..edge];
    let end_samples = &evidence[evidence.len() - edge..];
    let raw_trend = median(py, end_samples.iter().map(|s| s.1).collect())?
        - median(py, start_samples.iter().map(|s| s.1).collect())?;
    let aligned_trend = median(py, end_samples.iter().map(|s| s.2).collect())?
        - median(py, start_samples.iter().map(|s| s.2).collect())?;
    let speed = attr!(correction, "SETTLE_SPEED").extract::<f64>()? * span;
    let trending = py_min(raw_trend.abs(), aligned_trend.abs()) > speed;
    let moving = raw_trend * aligned_trend > 0. && trending;
    let geometry_drift = raw_trend * aligned_trend < 0. && trending;
    let range = if common_road {
        attr!(correction, "ROAD_SETTLE_RANGE").extract()?
    } else {
        attr!(correction, "SETTLE_RANGE").extract()?
    };
    let stable =
        ordered[ordered.len() - trim - 1] - ordered[trim] <= range && (end - start).abs() <= speed;
    let inside_count = evidence.iter().filter(|s| s.4).count();
    Ok((stable || geometry_drift) && !moving && inside_count as f64 >= 0.8 * evidence.len() as f64)
}

pub(crate) fn apply(correction: &Bound<'_, PyAny>, input: ApplyInput<'_, '_>) -> PyResult<f64> {
    let py = correction.py();
    let ApplyInput {
        key,
        x,
        y,
        fresh,
        aligned,
        mut filtered,
        slot,
        reference,
        bounds,
        observation,
        frame,
        reliable,
        lateral_filter,
    } = input;
    if attr!(correction, "ego_follow_active").is_truthy()? {
        let targets = attr!(correction, "ego_targets").cast_into::<PyDict>()?;
        if !targets.contains(key)? {
            targets.set_item(key, (frame, aligned, filtered, reference))?;
        }
    }
    if observation.is_none() {
        return Ok(filtered);
    }
    let birth = observation.get_item(0)?;
    let stamp: i64 = observation.get_item(1)?.extract()?;
    let center = if slot == 0 {
        0.
    } else {
        attr!(correction, "centers").get_item(slot - 1)?.extract()?
    };
    let clearance = match bounds {
        Some((left, right))
            if [left, right, aligned, center]
                .into_iter()
                .all(f64::is_finite) =>
        {
            py_min(
                py_min(py_min(left - aligned, aligned - right), left - center),
                center - right,
            )
        }
        _ => f64::NEG_INFINITY,
    };
    let common_road = reference.eq((intern!(py, "lane"), intern!(py, "center")))?;
    let (release, return_margin): (f64, f64) = if common_road {
        (
            attr!(correction, "ROAD_BOUNDARY_RELEASE_MARGIN").extract()?,
            attr!(correction, "ROAD_BOUNDARY_RETURN_MARGIN").extract()?,
        )
    } else {
        (
            attr!(correction, "BOUNDARY_RELEASE_MARGIN").extract()?,
            attr!(correction, "BOUNDARY_RETURN_MARGIN").extract()?,
        )
    };
    let mut inside = clearance >= release;
    let tracks = attr!(correction, "tracks").cast_into::<PyDict>()?;
    let existing = tracks
        .get_item(key)?
        .filter(|value| !value.is_none())
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    let saved = match existing {
        Some(saved)
            if field!(saved, "birth").eq(&birth)?
                && (0..=15).contains(&age(frame, integer!(saved, "frame"))) =>
        {
            saved
        }
        _ => {
            let saved = new_track(
                py,
                &birth,
                frame,
                inside && reliable,
                inside,
                filtered,
                slot,
                reference,
            )?;
            tracks.set_item(key, &saved)?;
            saved
        }
    };
    put!(saved, "display_road", (x, aligned, bounds));
    let resume = pop(&attr!(correction, "ego_resume").cast_into::<PyDict>()?, key)?;
    if !resume.is_none()
        && resume.get_item(0)?.eq(&birth)?
        && (0..=15).contains(&age(frame, resume.get_item(1)?.extract()?))
    {
        let output = resume.get_item(2)?;
        filtered = output.extract()?;
        put!(saved, "output", &output);
        if let Some(ff) = lead::departure(&saved)? {
            put!(ff, "output", &output);
        }
        if let Some(filter) = lateral_filter {
            filter.call_method1(intern!(py, "reset"), (&output,))?;
            attr!(filter, "_buffer")
                .call_method1(intern!(py, "extend"), ((&output, &output, &output),))?;
            filter.setattr(intern!(py, "_following_step"), false)?;
        }
    }
    let bounds_valid =
        bounds.is_some_and(|(left, right)| left.is_finite() && right.is_finite() && left > right);
    let center_valid = bounds_valid && on_road(bounds, center);
    if center_valid && !inside {
        put!(saved, "boundary_follow", true);
    } else if clearance >= return_margin {
        put!(saved, "boundary_follow", false);
    }
    inside = inside && !truth!(saved, "boundary_follow");
    let samples = field!(saved, "samples");
    let lane_change = attr!(correction, "lane_change").is_truthy()?;
    let old_reference = field!(saved, "reference");
    let same_reference = reference.eq(&old_reference)?;
    let old_slot: usize = field!(saved, "slot").extract()?;
    if lane_change || slot == 0 || !same_reference {
        pop(&saved, intern!(py, "exit_follow_until").as_any())?;
    } else if slot != old_slot && old_slot == 0 {
        if let Some(ff) = lead::departure(&saved)? {
            if truth!(ff, "released")
                && truth!(ff, "following")
                && (0..=200).contains(&age(frame, integer!(ff, "release_frame")))
            {
                put!(saved, "output", field!(ff, "output"));
                put!(saved, "exit_follow_until", checked_add(frame, 100)?);
            }
        }
    }
    if slot != old_slot || !same_reference {
        let lane = intern!(py, "lane");
        let center = intern!(py, "center");
        let position = intern!(py, "position");
        let same_road = (common_road || reference.eq((lane, position))?)
            && (old_reference.eq((lane, center))? || old_reference.eq((lane, position))?);
        put!(
            saved,
            "slot_frame",
            if slot != old_slot && same_reference {
                frame
            } else {
                -1000
            }
        );
        clear(&samples)?;
        pop(&saved, intern!(py, "ff_departure").as_any())?;
        pop(&saved, intern!(py, "ff_vision").as_any())?;
        put!(saved, "inward_frame", -1000);
        if !same_reference {
            clear(&field!(saved, "fast"))?;
            put!(saved, "fast_speed", 0.);
        }
        clear(&field!(saved, "settling"))?;
        if slot != old_slot || !same_road {
            put!(saved, "valid_frame", -1000);
        }
        if slot != old_slot {
            put!(saved, "locked", false);
        }
    }
    put!(saved, "slot", slot);
    put!(saved, "reference", reference);
    if slot != 0 || lane_change {
        pop(&saved, intern!(py, "ff_departure").as_any())?;
        pop(&saved, intern!(py, "ff_vision").as_any())?;
    }
    let mut ff = lead::departure(&saved)?;
    let (vision_history, vision) = if slot == 0 && !lane_change {
        let (history, vision) = lead::update_vision(correction, &saved, x, frame)?;
        (Some(history), vision)
    } else {
        (None, None)
    };
    if inside && reliable && fresh {
        put!(saved, "valid_frame", frame);
    }
    let valid_age = age(frame, integer!(saved, "valid_frame"));
    let geometry_limit = attr!(correction, "GEOMETRY_HOLD").extract::<i64>()?;
    let locked = truth!(saved, "locked");
    let geometry_hold = locked && !center_valid && (0..=geometry_limit).contains(&valid_age);
    let forced_follow = (!inside && !geometry_hold) || (!reliable && valid_age > geometry_limit);
    put!(saved, "following", !locked || forced_follow);
    if slot == 0 {
        put!(saved, "ff_frame", frame);
    }
    let side = slot != 0 && x < attr!(correction, "SIDE_X").extract::<f64>()? && !lane_change;
    if !side {
        clear(&field!(saved, "raw"))?;
        clear(&field!(saved, "drift"))?;
        put!(saved, "hold", py.None());
        put!(saved, "inward_frame", -1000);
    }
    if field!(saved, "stamp").ne(stamp)? && fresh {
        put!(saved, "stamp", stamp);
        if side {
            update_side(correction, &saved, &input, stamp)?;
        }
        fast_motion(
            correction,
            &saved,
            &input,
            stamp,
            center_valid,
            common_road,
            side,
        )?;
        let settle_time: i64 = attr!(correction, "SETTLE_TIME").extract()?;
        ordinary_motion(
            correction,
            &saved,
            &samples,
            &input,
            stamp,
            center_valid,
            common_road,
            side,
            settle_time,
        )?;
        let settling = field!(saved, "settling");
        append(&settling, (stamp, y, aligned, filtered, inside))?;
        prune(&settling, stamp, settle_time)?;
        let eligible = slot == 0 && !lane_change && inside && reliable && center_valid;
        let mut evidence = None;
        if (truth!(saved, "following")
            || ff
                .as_ref()
                .map(|ff| -> PyResult<bool> { Ok(truth!(ff, "following")) })
                .transpose()?
                .unwrap_or(false))
            && inside
            && reliable
            && settling.len()? >= 5
            && age(stamp, settling.get_item(0)?.get_item(0)?.extract()?) >= settle_time
        {
            let items: Vec<SettlingSample> = read_samples(&settling)?;
            if settled(correction, &items, stamp, common_road)? {
                put!(saved, "following", false);
                put!(saved, "locked", true);
                if let (Some(ff), Some(vision_history)) = (&ff, &vision_history) {
                    if truth!(ff, "released") {
                        lead::relock(ff, &items, stamp, vision_history)?;
                    }
                }
            }
            evidence = Some(items);
        }
        if let Some(departure) = &ff {
            if truth!(saved, "following") && !truth!(departure, "released") {
                pop(&saved, intern!(py, "ff_departure").as_any())?;
                ff = None;
            } else if truth!(departure, "released") && truth!(saved, "following") {
                put!(departure, "locked", false);
                put!(departure, "following", true);
            }
        }
        if eligible && truth!(saved, "locked") && !truth!(saved, "following") {
            let items = match evidence {
                Some(items) => items,
                None => read_samples(&settling)?,
            };
            if let Some(vision_history) = &vision_history {
                ff = Some(lead::track_departure(
                    &saved,
                    ff,
                    &items,
                    vision_history,
                    vision.as_ref(),
                    &input,
                    stamp,
                )?);
            }
        } else if let Some(ff) = &ff {
            put!(ff, "trail", py.None());
        }
    }
    if let (Some(ff), Some(history)) = (&ff, &vision_history) {
        lead::release_departure(&saved, ff, history, vision.as_ref(), &input, stamp)?;
    }
    if !field!(saved, "hold").is_none() && bounds_valid {
        let (left, right) = require_bounds(bounds)?;
        if if slot == 1 {
            aligned < right
        } else {
            aligned > left
        } {
            put!(saved, "hold", py.None());
            clear(&field!(saved, "raw"))?;
        }
    }
    let dt = py_max(
        0.,
        py_min(0.15, age(frame, integer!(saved, "frame")) as f64 * 0.01),
    );
    put!(saved, "frame", frame);
    if !field!(saved, "hold").is_none() {
        return Ok(number!(saved, "output"));
    }
    let exit_follow = !lane_change
        && slot != 0
        && frame
            <= saved
                .get_item(intern!(py, "exit_follow_until"))?
                .map(|value| value.extract::<i64>())
                .transpose()?
                .unwrap_or(-1000);
    if exit_follow {
        put!(saved, "locked", false);
        put!(saved, "following", true);
    }
    let following = truth!(saved, "following");
    let mut target = if following { filtered } else { center };
    let mut rate = if following {
        py_max(
            attr!(correction, "FOLLOW_RATE").extract()?,
            if exit_follow { 3. } else { 0. },
        )
    } else {
        attr!(correction, "CENTER_RATE").extract()?
    };
    let fast_speed = number!(saved, "fast_speed");
    let mut follow_aligned = false;
    if following && fast_speed > 0. && fresh {
        follow_aligned = bounds.is_some()
            && (!on_road(bounds, aligned)
                || (0..=30).contains(&age(frame, integer!(saved, "slot_frame"))));
        if follow_aligned {
            target = aligned;
        }
        rate = py_min(15., py_max(rate, 1.4 * fast_speed + 1.));
    }
    let output = number!(saved, "output");
    let output = output + py_min(py_max(target - output, -rate * dt), rate * dt);
    put!(saved, "output", output);
    if let Some(ff) = &ff {
        if truth!(ff, "released") {
            let ff_following = truth!(ff, "following");
            let mut target = if ff_following { filtered } else { center };
            let mut ff_rate = if ff_following {
                attr!(correction, "FOLLOW_RATE").extract()?
            } else {
                attr!(correction, "CENTER_RATE").extract()?
            };
            if ff_following && fresh {
                ff_rate = py_max(
                    py_max(ff_rate, rate),
                    if frame <= integer!(ff, "catchup_until") {
                        3.
                    } else {
                        0.
                    },
                );
                if follow_aligned
                    || (fast_speed > 0.
                        && bounds.is_some()
                        && (!on_road(bounds, aligned)
                            || (0..=30).contains(&age(frame, integer!(saved, "slot_frame")))))
                {
                    target = aligned;
                }
            }
            let output = number!(ff, "output");
            let output = output + py_min(py_max(target - output, -ff_rate * dt), ff_rate * dt);
            put!(ff, "output", output);
            return Ok(output);
        }
    }
    Ok(output)
}
