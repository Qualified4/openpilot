//! State projection used to compare the Rust rear policy with the Python reference.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ccnc_map::rear::{
    side_flow::{Candidate, Drawing, Flow, Key, Motion},
    Event, Identity, Phase, Track,
};

use super::{number, RearDisplay, Values};

macro_rules! put {
    ($obj:expr, $($key:literal => $value:expr),+ $(,)?) => {{
        $($obj.set_item($key, $value)?;)+
    }};
}

fn identity(key: Identity) -> (i128, f64) {
    (key.0, key.1)
}

fn shown_key(py: Python<'_>, key: Option<Key>) -> PyResult<Bound<'_, PyAny>> {
    Ok(match key {
        Some(Key::Radar(key)) => identity(key).into_pyobject(py)?.into_any(),
        Some(Key::Display(side)) => ("display", side).into_pyobject(py)?.into_any(),
        None => py.None().into_bound(py),
    })
}

fn track<'py>(py: Python<'py>, saved: &Track) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    put!(out, "key" => identity(saved.key), "source" => saved.key.0, "phase" => saved.phase.name(),
        "seen" => saved.seen, "distance" => saved.distance, "vrel" => saved.vrel, "v_anchor" => saved.anchor,
        "lateral" => saved.lateral, "detect" => saved.detect);
    if saved.phase != Phase::Rear || saved.hits != 0 {
        put!(out, "first" => saved.first, "first_x" => saved.first_x, "hits" => saved.hits,
            "moving_hits" => saved.moving_hits, "qualified" => saved.qualified,
            "aliases" => saved.aliases.iter().copied().map(identity).collect::<Vec<_>>(),
            "vrel_hist" => &saved.vrel_hist, "vlead_hist" => &saved.vlead_hist, "vlead_long" => &saved.vlead_long,
            "x" => saved.x, "y" => saved.y, "vrel_raw" => saved.vrel_raw, "vlead" => saved.vlead,
            "update" => saved.updated, "drawn_distance" => saved.drawn_distance, "drawn_stamp" => saved.drawn_stamp,
            "drawn_lateral" => saved.drawn_lateral, "radar_moving" => saved.radar_moving);
        if saved.phase != Phase::Front {
            put!(out, "signed" => saved.signed, "rate" => saved.rate, "coast_y" => saved.coast_y);
        }
    }
    if saved.phase != Phase::Front {
        put!(out, "shown" => saved.shown, "start" => saved.start, "speed" => saved.speed);
    }
    if let Some(forced) = saved.forced {
        put!(out, "forced" => forced);
    }
    Ok(out)
}

fn motion<'py>(py: Python<'py>, motion: &Motion, live: bool) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    put!(out, "key" => identity(motion.key), "aliases" => motion.aliases.iter().copied().map(identity).collect::<Vec<_>>(),
        "phase" => motion.phase.name(), "live" => live, "signed" => motion.signed, "speed" => motion.speed,
        "distance" => motion.distance, "force_at" => motion.force_at, "deadline" => motion.deadline,
        "max_speed" => motion.max_speed, "end_reason" => motion.end_reason);
    Ok(out)
}

fn candidate<'py>(py: Python<'py>, entry: Candidate) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    put!(out, "key" => shown_key(py, Some(entry.key))?, "x" => number(py, entry.x)?,
        "y" => number(py, entry.y)?, "detect" => number(py, entry.detect)?);
    Ok(out)
}

fn drawing<'py>(py: Python<'py>, entry: &Drawing, live: Option<u64>) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    put!(out, "key" => shown_key(py, entry.key)?, "x" => number(py, entry.x)?,
        "y" => number(py, entry.y)?, "detect" => number(py, entry.detect)?, "phase" => entry.phase.name(),
        "display_pass" => entry.display_pass, "target" => entry.target.map(|c| candidate(py, c)).transpose()?,
        "motion" => entry.motion.as_ref().map(|m| motion(py, m, Some(m.serial) == live)).transpose()?,
        "model_distance" => entry.model_distance, "spacing" => entry.spacing);
    Ok(out)
}

fn flow<'py>(py: Python<'py>, flow: &Flow, live: Option<u64>) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    put!(out, "front" => flow.front.as_ref().map(|f| drawing(py, f, live)).transpose()?,
        "rear" => flow.rear.as_ref().map(|r| drawing(py, r, live)).transpose()?, "bsd" => flow.bsd,
        "bsd_off" => flow.bsd_off, "after_pass" => flow.after_pass, "events" => &flow.events);
    Ok(out)
}

fn event<'py>(py: Python<'py>, event: &Event) -> PyResult<Values<'py>> {
    let out = PyDict::new(py);
    match *event {
        Event::Drop {
            side,
            reason,
            source,
            phase,
            distance,
            age,
            qualified,
            hits,
        } => {
            put!(out, "side" => side, "event" => reason, "source" => source, "phase" => phase.name(),
                "distance" => distance, "age" => age, "qualified" => qualified, "hits" => hits);
        }
        Event::Coast {
            side,
            reason,
            source,
            x,
            vrel,
            vlead,
            anchor,
            ego,
            front_distance,
        } => {
            put!(out, "side" => side, "event" => "coast", "reason" => reason, "source" => source, "x" => x,
                "vrel" => vrel, "vlead" => vlead, "v_anchor" => anchor, "v_ego" => ego, "front_distance" => front_distance);
        }
        Event::Handoff {
            side,
            key,
            distance,
            seen_age,
            speed,
            lateral,
            coast_time,
            origin,
        } => {
            put!(out, "side" => side, "event" => "handoff", "source" => key.0, "key" => identity(key),
                "distance" => distance, "seen_age" => seen_age, "speed" => speed, "lateral" => lateral,
                "coast_time" => coast_time, "origin" => origin);
        }
        Event::Forced { side, source, distance } => {
            put!(out, "side" => side, "event" => "forced_exit", "source" => source, "distance" => distance)
        }
        Event::Flow { side, name } => put!(out, "side" => side, "event" => name),
    }
    Ok(out)
}

pub(super) fn snapshot<'py>(policy: &Bound<'py, PyAny>, holder: &RearDisplay) -> PyResult<Values<'py>> {
    let py = policy.py();
    let state = &holder.state;
    let out = PyDict::new(py);
    let fronts = PyList::empty(py);
    let rears = PyList::empty(py);
    let consumed = PyList::empty(py);
    let flows = PyList::empty(py);
    for side in &state.sides {
        fronts.append(side.front.as_ref().map(|track_| track(py, track_)).transpose()?)?;
        rears.append(side.rear.as_ref().map(|track_| track(py, track_)).transpose()?)?;
        consumed.append(side.consumed.iter().copied().map(identity).collect::<Vec<_>>())?;
        if policy.hasattr("flows")? {
            flows.append(flow(py, &side.flow, side.rear.as_ref().map(|r| r.serial))?)?;
        }
    }
    let events = state
        .events
        .iter()
        .map(|e| event(py, e))
        .collect::<PyResult<Vec<_>>>()?;
    let hist = state.ego_hist.iter().copied().collect::<Vec<_>>();
    put!(out, "fronts" => fronts, "rears" => rears, "consumed" => consumed,
        "context" => holder.context.as_ref().map(|p| p.bind(py)), "raw" => holder.raw.as_ref().map(|p| p.bind(py)),
        "last" => state.last, "raw_stamp" => state.raw_stamp, "fresh" => state.fresh,
        "unproven" => state.sides.iter().map(|s| s.unproven).collect::<Vec<_>>(),
        "events" => events, "v_ego" => state.ego, "changing" => state.changing, "ego_hist" => hist);
    if !flows.is_empty() {
        put!(out, "flows" => flows);
    }
    Ok(out)
}

/// Some tests start in a configured rear-exit phase rather than running a full pass first.
pub(super) fn seed(policy: &Bound<'_, PyAny>, holder: &mut RearDisplay) -> PyResult<()> {
    let context = policy.getattr("context")?;
    holder.context = (!context.is_none()).then_some(context.unbind());
    holder.state.last = policy.getattr("last")?.extract()?;
    for i in 0..2 {
        let old = policy.getattr("rears")?.get_item(i)?;
        if old.is_none() {
            continue;
        }
        let read = |name| old.get_item(name)?.extract::<f64>();
        let (source, epoch): (i128, f64) = old.get_item("key")?.extract()?;
        let mut saved = Track::new(Identity(source, epoch), u64::MAX - i as u64, read("seen")?, 0.);
        saved.phase = Phase::Rear;
        saved.shown = read("shown")?;
        saved.start = Some(read("start")?);
        saved.distance = read("distance")?;
        saved.vrel = read("vrel")?;
        saved.anchor = read("v_anchor")?;
        saved.lateral = read("lateral")?;
        saved.detect = old.get_item("detect")?.extract()?;
        holder.state.sides[i].rear = Some(saved);
    }
    Ok(())
}
