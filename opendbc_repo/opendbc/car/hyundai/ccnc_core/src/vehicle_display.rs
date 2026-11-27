//! One display update on the radar map. Map observation and selection, position
//! correction, and the rear policy run directly on the shared Rust stages.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

use crate::correct::correct_positions;
use crate::display_filter;
use crate::guard;
use crate::map_state;
use crate::model_v2::{
    curve, numpy_interp, prepare, read_geometry, valid_curve, GeometryInput,
};
use crate::pymath::{py_max, py_min, py_sum};
use crate::{cmap, pos, rear, update};

macro_rules! get {
    ($object:expr, $name:literal) => {
        $object.getattr(intern!($object.py(), $name))?
    };
}

macro_rules! put {
    ($object:expr, $($name:literal => $value:expr),+ $(,)?) => {
        $($object.setattr(intern!($object.py(), $name), $value)?;)+
    };
}

mod lane_curvature;
mod lane_positions;
mod panel;

fn enum_name<'py>(
    item: &Bound<'py, PyAny>,
    name: &Bound<'py, pyo3::types::PyString>,
    default: &str,
) -> PyResult<String> {
    item.getattr_opt(name)?
        .map_or_else(|| Ok(default.into()), |v| Ok(v.str()?.to_str()?.to_owned()))
}

fn optional_model_meta<'py>(model: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    Ok(model
        .getattr_opt(intern!(py, "meta"))?
        .unwrap_or_else(|| py.None().into_bound(py)))
}

fn model_desire(model: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(matches!(
        enum_name(
            &optional_model_meta(model)?,
            intern!(model.py(), "desire"),
            ""
        )?
        .as_str(),
        "laneChangeLeft" | "laneChangeRight"
    ))
}

/// The lane message used to run this for the map as a side effect. Keep its
/// cadence, and advance correction memory in the same call as its consumers.
fn update_ego_follow<'py>(
    correction: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    lines: &GeometryInput,
    now: f64,
    speed: f64,
    desire: i64,
    blinkers: (bool, bool),
) -> PyResult<()> {
    let py = md.py();
    let (left, right) = blinkers;
    let was_active = get!(correction, "ego_follow_active").is_truthy()?;
    let last: Option<f64> = get!(correction, "ego_follow_time").extract()?;
    let motion = get!(correction, "ego_follow_motion");
    let confirmed = get!(correction, "ego_change_motion");
    if last.is_some_and(|last| !(0. ..=0.15).contains(&(now - last))) {
        put!(correction, "ego_follow_until" => f64::NEG_INFINITY);
        for history in [&motion, &confirmed] {
            history.call_method0(intern!(py, "clear"))?;
        }
    }
    put!(correction, "vision_model" => md, "ego_follow_time" => now);
    get!(correction, "ego_targets").cast::<PyDict>()?.clear();
    if !md.is_none() {
        let stamp = md.getattr_opt(intern!(py, "timestampEof"))?;
        let stamp = match stamp {
            Some(stamp) if stamp.is_truthy()? => stamp,
            _ => (md.as_ptr() as usize).into_pyobject(py)?.into_any(),
        };
        if !stamp.eq(get!(correction, "ego_follow_model"))? {
            put!(correction, "ego_follow_model" => stamp);
            let mut velocities = Vec::with_capacity(2);
            if lines.lines.len() >= 3 && lines.probs.len() >= 3 {
                for i in [1, 2] {
                    let p = lines.probs[i];
                    let line = &lines.lines[i];
                    if !p.is_finite()
                        || p < 0.1
                        || !valid_curve(line)
                        || line.0[0] > 0.
                        || line.0[line.0.len() - 1] < 5.
                    {
                        continue;
                    }
                    let [y0, ym, y5] = [0., 2.5, 5.].map(|x| numpy_interp(line, x));
                    velocities.push(speed * (-3. * y0 + 4. * ym - y5) / 5.);
                }
            }
            let append = intern!(py, "append");
            let agrees = velocities.len() == 2 && (velocities[0] - velocities[1]).abs() <= 0.5;
            if agrees {
                motion.call_method1(
                    append,
                    ((now, py_sum(velocities.iter().copied()).unwrap_or(0.) * 0.5),),
                )?;
            }
            if velocities.len() == 1 || agrees {
                confirmed.call_method1(
                    append,
                    ((
                        now,
                        py_sum(velocities.iter().copied()).unwrap_or(0.) / velocities.len() as f64,
                    ),),
                )?;
            }
        }
    }
    let advancing = |history: &Bound<'py, PyAny>| -> PyResult<bool> {
        while history.len()? > 0 && now - history.get_item(0)?.get_item(0)?.extract::<f64>()? > 0.3
        {
            history.call_method0(intern!(py, "popleft"))?;
        }
        if left == right || history.len()? < 4 {
            return Ok(false);
        }
        let points: Vec<(f64, f64)> = history.extract()?;
        if points[points.len() - 1].0 - points[0].0 < 0.15 {
            return Ok(false);
        }
        let mut v: Vec<f64> = points.into_iter().map(|(_, v)| v).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = v.len() / 2;
        let median = if v.len() % 2 == 0 {
            (v[mid - 1] + v[mid]) / 2.
        } else {
            v[mid]
        };
        Ok((if left { median } else { -median }) >= 0.3)
    };
    let moving = advancing(&motion)?;
    let changing = advancing(&confirmed)?;
    let state = enum_name(
        &optional_model_meta(md)?,
        intern!(py, "laneChangeState"),
        "off",
    )?;
    let auto = matches!(desire, 3 | 4);
    let model_active =
        auto || matches!(state.as_str(), "laneChangeStarting" | "laneChangeFinishing");
    put!(correction, "lane_change" => auto || left || right,
        "ego_change_active" => (auto || left || right) && (model_active || changing));
    if model_active || moving {
        put!(correction, "ego_follow_until" => now + 1.);
    }
    let until: f64 = get!(correction, "ego_follow_until").extract()?;
    let active = now <= until;
    put!(correction, "ego_follow_active" => active);
    if was_active && !active {
        put!(correction, "ego_resume" => get!(correction, "ego_drawn"), "ego_drawn" => PyDict::new(py));
    }
    let resume = get!(correction, "ego_resume").cast_into::<PyDict>()?;
    if !was_active && active {
        resume.clear();
    }
    let keep = PyDict::new(py);
    for (key, value) in resume.iter() {
        let stamp: f64 = value.get_item(1)?.extract()?;
        if (0. ..=0.15).contains(&(now - stamp * 0.01)) {
            keep.set_item(key, value)?;
        }
    }
    put!(correction, "ego_resume" => keep);
    Ok(())
}

/// `reset()`: the Python attributes as `__init__` sets them and the Rust state as new.
fn map_reset(map: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = map.py();
    map_state::attach(map)?.try_borrow_mut()?.reset();
    map.setattr("evidence", PyDict::new(py))?;
    for attr in ["leads", "vision_leads"] {
        map.setattr(attr, PyList::empty(py))?;
    }
    for attr in ["radar_stamp", "model_stamp", "raw", "model", "shown"] {
        map.setattr(attr, py.None())?;
    }
    for attr in ["usable", "holding", "stopped", "lane_change", "ego_change"] {
        map.setattr(attr, false)?;
    }
    put!(map, "a_ego" => 0., "yaw_rate" => 0., "blindspot" => (false, false), "generation" => py.get_type::<PyAny>().call0()?);
    Ok(())
}


/// `_advance(now, speed, yaw)` on the map's Rust state.
fn advance_pose(map: &Bound<'_, PyAny>, now: f64, speed: f64, yaw: f64, holding: bool) -> PyResult<()> {
    let range = map_state::range(map)?;
    let state = map_state::attach(map)?;
    let st = &mut state.try_borrow_mut()?.st;
    st.advance(now, speed, yaw, holding, st.bridge.map(|bridge| bridge.key), range);
    Ok(())
}

#[derive(Default)]
struct Leads {
    nearby: Vec<(f64, f64)>,
    confident: Vec<(f64, f64, Option<f64>)>,
    vision: Vec<(f64, f64)>,
}

fn read_leads(model: &Bound<'_, PyAny>, bridge_prob: f64) -> PyResult<Leads> {
    let py = model.py();
    let mut leads = Leads::default();
    let Some(live) = model.getattr_opt(intern!(py, "leadsV3"))? else {
        return Ok(leads);
    };
    for lead in live.try_iter()? {
        let lead = lead?;
        let (xs, ys) = (get!(lead, "x"), get!(lead, "y"));
        if xs.len()? == 0 || ys.len()? == 0 {
            continue;
        }
        let (x, y): (f64, f64) = (xs.get_item(0)?.extract()?, ys.get_item(0)?.extract()?);
        let probability: f64 = get!(lead, "prob").extract()?;
        let point = (x - 1.52, -y);
        if probability >= 0.8 {
            leads.nearby.push(point);
        }
        if probability.partial_cmp(&0.8) != Some(std::cmp::Ordering::Less) {
            leads.vision.push(point);
        }
        if probability >= bridge_prob && x.is_finite() && y.is_finite() {
            let v = lead.getattr_opt(intern!(py, "v"))?;
            let speed = if let Some(v) = v {
                if v.len()? > 0 {
                    Some(v.get_item(0)?.extract::<f64>()?).filter(|v| v.is_finite())
                } else {
                    None
                }
            } else {
                None
            };
            leads.confident.push((point.0, point.1, speed));
        }
    }
    Ok(leads)
}

#[allow(clippy::too_many_arguments)]
fn select<'py>(
    map: &Bound<'py, PyAny>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    live: &Bound<'py, PyAny>,
    now: f64,
    lines: &GeometryInput,
    lane_type: &Bound<'py, PyAny>,
    wall_type: &Bound<'py, PyAny>,
    band_type: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = map.py();
    let radar = get!(cs, "live_tracks");
    if !radar.is_none() && !radar.is(&get!(live, "raw")) {
        put!(live, "raw" => &radar, "radar_time" => now);
    }
    if !md.is_none() && !md.is(&get!(live, "model")) {
        put!(live, "model" => md, "model_time" => now);
    }

    let out = get!(cs, "out");
    let yaw = md.getattr_opt(intern!(py, "orientationRate"))?;
    let yaw = match yaw
        .and_then(|rate| rate.getattr_opt(intern!(py, "z")).transpose())
        .transpose()?
    {
        Some(rates) if rates.len()? > 0 => -rates.get_item(0)?.extract::<f64>()?,
        _ => f64::NAN,
    };
    let ego: f64 = get!(out, "vEgo").extract()?;
    let state = map_state::attach(map)?;
    let previous_time: Option<f64> = state.try_borrow()?.st.now;
    let reverse = enum_name(&out, intern!(py, "gearShifter"), "")? == "reverse";
    if ![now, ego, yaw].iter().all(|v| v.is_finite())
        || reverse
        || ego < -0.5
        || yaw.abs() > 1.
        || previous_time.is_some_and(|last| !(0. ..=0.25).contains(&(now - last)))
    {
        return map_reset(map);
    }
    let speed = py_max(0., ego);
    let acceleration: f64 = get!(out, "aEgo").extract()?;
    let stopped = speed <= 0.3 / 3.6;
    let holding = stopped
        || (!get!(map, "stopped").is_truthy()?
            && speed <= 10. / 3.6
            && acceleration.is_finite()
            && acceleration < -0.1 / 3.6);
    let blind = (get!(out, "leftBlindspot"), get!(out, "rightBlindspot"));
    let signal = out
        .getattr_opt(intern!(py, "leftBlinker"))?
        .map_or(Ok(false), |v| v.is_truthy())?
        || out
            .getattr_opt(intern!(py, "rightBlinker"))?
            .map_or(Ok(false), |v| v.is_truthy())?;
    let changing = model_desire(md)? || signal;
    put!(map, "a_ego" => if acceleration.is_finite() { acceleration } else { 0. }, "holding" => holding, "stopped" => stopped,
        "blindspot" => blind, "lane_change" => changing, "ego_change" => changing, "evidence" => PyDict::new(py), "yaw_rate" => yaw);
    advance_pose(map, now, speed, yaw, holding)?;

    let radar_time = get!(live, "radar_time");
    let radar_stamp: f64 = radar_time.extract()?;
    let model_time = get!(live, "model_time");
    let model_stamp: f64 = model_time.extract()?;
    let fresh_model = !md.is_none() && (0. ..=0.25).contains(&(now - model_stamp));
    let refresh = fresh_model && !get!(map, "model_stamp").eq(&model_time)? && !md.is(&get!(map, "model"));
    prepare(map, refresh.then_some((lines, model_stamp)), changing, lane_type)?;
    if refresh {
        put!(map, "model" => md, "model_stamp" => &model_time);
    }
    let usable = !radar.is_none() && (0. ..=0.15).contains(&(now - radar_stamp));
    put!(map, "usable" => usable);
    let leads = if fresh_model {
        read_leads(md, get!(map, "BRIDGE_PROB").extract()?)?
    } else {
        Leads::default()
    };
    put!(map, "leads" => PyList::new(py, &leads.nearby)?, "vision_leads" => PyList::new(py, &leads.confident)?);
    let last_radar: Option<f64> = get!(map, "radar_stamp").extract()?;
    let observe =
        usable && !radar.is(&get!(map, "raw")) && last_radar.is_none_or(|last| radar_stamp > last);
    let observed = if observe {
        let pose = state.try_borrow()?.st.pose_at(radar_stamp);
        let observed = match pose {
            Some((x, y, angle)) => Some((get!(radar, "points"), (x, y, angle.cos(), angle.sin()))),
            None => None,
        };
        put!(map, "raw" => &radar, "radar_stamp" => &radar_time);
        observed
    } else {
        None
    };
    let last_model = get!(map, "model");
    let meta = optional_model_meta(&last_model)?;
    let changing = model_desire(&last_model)?
        || matches!(
            enum_name(&meta, intern!(py, "laneChangeState"), "off")?.as_str(),
            "laneChangeStarting" | "laneChangeFinishing"
        );
    let raw = cs.getattr_opt(intern!(py, "live_tracks_bytes"))?;
    let bytes = raw
        .as_ref()
        .filter(|data| !data.is_none())
        .map(|data| data.extract())
        .transpose()?;
    update::update(
        py,
        map,
        observe,
        observed,
        &radar_time,
        leads.nearby,
        leads.confident,
        leads.vision,
        changing,
        true,
        wall_type,
        band_type,
        bytes,
    )?;
    Ok(())
}

pub(crate) const FRONT_FIELDS: [[&str; 3]; 3] = [
    ["FF_DISTANCE", "FF_LATERAL", "FF_DETECT"],
    ["LF_DETECT_DISTANCE", "LF_DETECT_LATERAL", "LF_DETECT"],
    ["RF_DETECT_DISTANCE", "RF_DETECT_LATERAL", "RF_DETECT"],
];

pub(crate) const REAR_FIELDS: [[&str; 3]; 2] = [
    ["LR_DETECT_DISTANCE", "LR_DETECT_LATERAL", "LR_DETECT"],
    ["RR_DETECT_DISTANCE", "RR_DETECT_LATERAL", "RR_DETECT"],
];

fn deadband(value: f64, center: f64, radius: f64, degree: f64) -> f64 {
    let absolute = value.abs();
    let diff = absolute - center;
    if diff >= radius || diff <= -radius {
        return value;
    }
    let weight = (diff.abs() / radius).powf(degree);
    let smoothed = (1. - weight) * center + weight * absolute;
    if value >= 0. {
        smoothed
    } else {
        -smoothed
    }
}

/// `assigned` replaces the map's selection (display modes 1 and 2).
fn draw_front<'py>(
    map: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    values: &Bound<'py, PyDict>,
    assigned: Option<[Option<i64>; 3]>,
) -> PyResult<()> {
    let py = map.py();
    let map_state = map_state::attach(map)?;
    let map_state = map_state.try_borrow()?;
    let st = &map_state.st;
    let selected = get!(map, "selected");
    let speed = st.speed;
    let frame = st.frame();
    let usable = get!(map, "usable").is_truthy()?;
    let bridge = get!(map, "bridge");
    let detectors = [
        get!(state, "ff_detect"),
        get!(state, "lf_detect"),
        get!(state, "rf_detect"),
    ];
    let centers = [get!(state, "lf_center"), get!(state, "rf_center")];
    for (slot, [distance, lateral, detect]) in FRONT_FIELDS.into_iter().enumerate() {
        let key: Option<i64> = match assigned {
            Some(keys) => keys[slot],
            None => selected.get_item(slot)?.extract()?,
        };
        let pair = match key {
            Some(key) if usable => st.objects.get(&key).zip(st.local.get(&key)),
            _ => None,
        };
        let Some((object, &(x, y))) = pair else {
            values.set_item(detect, 0)?;
            continue;
        };
        let bridged = slot == 0
            && !bridge.is_none()
            && bridge.get_item(intern!(py, "key"))?.extract::<Option<i64>>()? == key;
        let forward: f64 = if bridged {
            bridge.get_item(intern!(py, "speed"))?.extract()?
        } else {
            frame.c * object.vx + frame.s * object.vy
        };
        values.set_item(distance, py_min(204.7, py_max(0., x * 0.8)))?;
        if slot == 0 {
            values.set_item(lateral, deadband(-y, 0., 0.7, 1.))?;
        } else {
            let center = display_filter::value(&centers[slot - 1])?;
            values.set_item(
                lateral,
                py_min(
                    py_max(
                        deadband(if slot == 1 { y } else { -y }, center, 0.9, 2.),
                        0.,
                    ),
                    12.7,
                ),
            )?;
        }
        if slot == 0 && forward < 3. {
            values.set_item(detect, 2)?;
        } else {
            values.set_item(
                detect,
                display_filter::detect(&detectors[slot], forward - speed)?,
            )?;
        }
    }
    Ok(())
}

/// DISPLAY_HYSTERESIS (m, 0 off): the drawn FF/LF/RF lateral keeps its last value while the slot keeps its target
/// and the new value is less than this far from it.
fn hysteresis<'py>(map: &Bound<'py, PyAny>, values: &Bound<'py, PyDict>) -> PyResult<()> {
    let py = map.py();
    let band: f64 = get!(map, "DISPLAY_HYSTERESIS").extract()?;
    if !(band > 0.) {
        return Ok(());
    }
    let selected = get!(map, "selected");
    let previous = map.getattr_opt(intern!(py, "shown"))?.filter(|v| !v.is_none());
    let mut shown = Vec::with_capacity(3);
    for (slot, [_, lateral, detect]) in FRONT_FIELDS.into_iter().enumerate() {
        let key: Option<i64> = selected.get_item(slot)?.extract()?;
        let value: f64 = values.as_any().get_item(lateral)?.extract()?;
        if key.is_none() || !values.as_any().get_item(detect)?.is_truthy()? {
            shown.push(py.None().into_bound(py));
            continue;
        }
        let last: Option<(i64, f64)> = match &previous {
            Some(p) => p.get_item(slot)?.extract()?,
            None => None,
        };
        let kept = match last {
            Some((k, v)) if Some(k) == key && (value - v).abs() < band - 1e-9 => v,
            _ => value,
        };
        values.set_item(lateral, kept)?;
        shown.push((key, kept).into_pyobject(py)?.into_any());
    }
    map.setattr(intern!(py, "shown"), PyTuple::new(py, shown)?)?;
    Ok(())
}

fn finish_fields<'py>(values: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(values.py());
    for [distance, lateral, detect] in FRONT_FIELDS.into_iter().chain(REAR_FIELDS) {
        for key in [distance, lateral] {
            fields.set_item(key, values.as_any().get_item(key)?)?;
        }
        // A slot the map left untouched still holds the parsed 0x162 value, a float
        // (LR/RR on the first map frame); keep its type as the Python path does.
        let code = values.as_any().get_item(detect)?;
        let number: f64 = code.extract()?;
        fields.set_item(
            detect,
            if number == 1. || number == 2. {
                code.add(2)?
            } else {
                code
            },
        )?;
    }
    Ok(fields)
}

#[allow(clippy::too_many_arguments)]
fn update_map_display<'py>(
    existing: &Bound<'py, PyDict>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    lines: &GeometryInput,
    live: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    lane_type: &Bound<'py, PyAny>,
    wall_type: &Bound<'py, PyAny>,
    band_type: &Bound<'py, PyAny>,
    position_filter: &Bound<'py, PyAny>,
    mode: i64,
) -> PyResult<Bound<'py, PyDict>> {
    guard(|| {
        let py = cs.py();
        let values = existing.copy()?;
        let map = get!(live, "tracker");
        let now = frame as f64 * 0.01;
        let correction = get!(get!(state, "radar_display_tracker"), "position_correction");
        select(
            &map, cs, md, live, now, lines, lane_type, wall_type, band_type,
        )?;
        let center_filters = [get!(state, "lf_center"), get!(state, "rf_center")];
        for (side, filter) in center_filters.iter().enumerate() {
            if let Some(center) = lines.side_center(side, 0.6) {
                display_filter::fixed(filter, PyFloat::new(py, center).as_any())?;
            }
        }
        if mode != 0 {
            // DESIGN 10: the position-correction path and the lateral hysteresis are bypassed in modes 1 and 2.
            let (assigned, drawn) = pos::assign_and_draw(&map, cs, state, mode)?;
            draw_front(&map, state, &values, Some(assigned))?;
            for (slot, [_, lateral, detect]) in FRONT_FIELDS.into_iter().enumerate() {
                let Some(y) = drawn[slot] else { continue };
                if !values.as_any().get_item(detect)?.is_truthy()? {
                    continue;
                }
                let value = match slot {
                    0 => -y,
                    1 => py_min(py_max(y, 0.), 12.7),
                    _ => py_min(py_max(-y, 0.), 12.7),
                };
                values.set_item(lateral, value)?;
            }
            rear::finish_map(&values, cs, md, frame, state, &map)?;
            return finish_fields(&values);
        }
        pos::shadow(&map, cs, state)?;
        draw_front(&map, state, &values, None)?;

        let model_time: f64 = get!(live, "model_time").extract()?;
        let old_change = get!(correction, "lane_change");
        let lane_change =
            old_change.is_truthy()? || get!(map, "lane_offset").extract::<i64>()? != 0;
        let min_probability: f64 = get!(correction, "MIN_LANE_PROBABILITY").extract()?;
        let inputs = if (0. ..=0.25).contains(&(now - model_time)) {
            let model = get!(live, "model");
            let active = get!(correction, "ego_follow_active").is_truthy()?;
            let blend: (Option<i64>, f64, f64) = get!(live, "ego_blend").extract()?;
            if !model.is_none() && (active || blend.1 > 0.) {
                let model_lines = if model.is(md) {
                    None
                } else {
                    Some(read_geometry(&model, None)?)
                };
                let m = model_lines.as_ref().unwrap_or(&lines);
                let inner = if m.lines.len() >= 3 {
                    m.lines[1..3].to_vec()
                } else {
                    Vec::new()
                };
                let probs = m.probs.iter().skip(1).take(2).copied().collect();
                let path = model
                    .getattr_opt(intern!(py, "position"))?
                    .filter(|x| !x.is_none())
                    .map(|x| curve(&x))
                    .transpose()?;
                Some((inner, path, probs))
            } else {
                None
            }
        } else {
            None
        };
        let centers = (
            lines
                .side_center(0, min_probability)
                .into_pyobject(py)?
                .into_any(),
            lines
                .side_center(1, min_probability)
                .into_pyobject(py)?
                .into_any(),
        );
        put!(correction, "lane_change" => lane_change);
        let result = correct_positions(
            py,
            &map,
            &correction,
            values.as_any(),
            frame,
            centers,
            inputs,
            live,
            position_filter,
            0.5,
            0.3,
        );
        put!(correction, "lane_change" => old_change);
        result?;
        rear::finish_map(&values, cs, md, frame, state, &map)?;
        hysteresis(&map, &values)?;
        finish_fields(&values)
    })
}

struct StockLead<'py> {
    point: Bound<'py, PyAny>,
    distance: f64,
    lateral: f64,
    relative_speed: f64,
}

impl StockLead<'_> {
    fn path_lateral(&self, md: &Bound<'_, PyAny>) -> PyResult<f64> {
        let py = md.py();
        let mut lateral = -self.lateral;
        if let Some(path) = md
            .getattr_opt(intern!(py, "position"))?
            .filter(|p| !p.is_none())
        {
            let line = curve(&path)?;
            if valid_curve(&line) {
                lateral -= numpy_interp(&line, self.distance);
            }
        }
        Ok(lateral)
    }
}

fn stock_lead<'py>(cs: &Bound<'py, PyAny>) -> PyResult<Option<StockLead<'py>>> {
    let py = cs.py();
    let Some(radar) = cs.getattr_opt(intern!(py, "radarState"))? else {
        return Ok(None);
    };
    let mut selected: Option<StockLead<'_>> = None;
    for name in [intern!(py, "leadOne"), intern!(py, "leadTwo")] {
        let Some(lead) = radar.getattr_opt(name)?.filter(|lead| !lead.is_none()) else {
            continue;
        };
        if !get!(lead, "status").is_truthy()? {
            continue;
        }
        let distance: f64 = get!(lead, "dRel").extract()?;
        if !distance.is_finite() || distance <= 0. {
            continue;
        }
        let lateral: f64 = get!(lead, "yRel").extract()?;
        let relative: f64 = get!(lead, "vRel").extract()?;
        if lateral.is_finite()
            && relative.is_finite()
            && selected.as_ref().is_none_or(|old| distance < old.distance)
        {
            selected = Some(StockLead {
                point: lead,
                distance,
                lateral,
                relative_speed: relative,
            });
        }
    }
    Ok(selected)
}

fn filter_stock_lead<'py>(
    owner: &Bound<'py, PyAny>,
    lead: Option<&StockLead<'py>>,
    md: &Bound<'py, PyAny>,
) -> PyResult<f64> {
    let py = owner.py();
    let filter = get!(owner, "filter");
    let Some(lead) = lead else {
        put!(filter, "initialized" => false);
        put!(owner, "target" => py.None(), "distance" => py.None());
        return Ok(0.);
    };
    let source = lead
        .point
        .getattr_opt(intern!(py, "radar"))?
        .unwrap_or_else(|| false.into_pyobject(py).unwrap().to_owned().into_any());
    let identity = lead
        .point
        .getattr_opt(intern!(py, "radarTrackId"))?
        .map_or_else(|| (-1_i32).into_pyobject(py).map(|i| i.into_any()), Ok)?;
    let target = (source, identity).into_pyobject(py)?;
    let last: Option<f64> = get!(owner, "distance").extract()?;
    let lateral = lead.path_lateral(md)?;
    if target.ne(get!(owner, "target"))?
        || last.is_some_and(|last| (lead.distance - last).abs() > 5.)
    {
        put!(filter, "initialized" => false);
    }
    put!(owner, "target" => target, "distance" => lead.distance);
    let value = if get!(filter, "initialized").is_truthy()? {
        let alpha: f64 = get!(filter, "alpha").extract()?;
        let previous: f64 = get!(filter, "x").extract()?;
        (1. - alpha) * previous + alpha * lateral
    } else {
        put!(filter, "initialized" => true);
        lateral
    };
    put!(filter, "x" => value);
    Ok(value)
}

type MapConfiguration<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

type Template<'py> = Option<Bound<'py, PyDict>>;
type Display<'py> = (
    Template<'py>,
    Template<'py>,
    Template<'py>,
    Option<panel::StockOverlay<'py>>,
);

fn template<'py>(cs: &Bound<'py, PyAny>, name: &str) -> PyResult<Template<'py>> {
    cs.getattr_opt(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.cast_into::<PyDict>()?.copy())
        .transpose()
}

/// The lead filter advances at 100 Hz. Only ticks with display inputs build CCNC fields.
#[pyfunction]
#[pyo3(signature = (cs, md, frame, inputs, state, position_filter, map, lead_filter, *, process_now_ns=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_display<'py>(
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    inputs: Option<panel::Inputs<'py>>,
    state: &Bound<'py, PyAny>,
    position_filter: &Bound<'py, PyAny>,
    map: Option<MapConfiguration<'py>>,
    lead_filter: Option<&Bound<'py, PyAny>>,
    process_now_ns: Option<u64>,
) -> PyResult<Option<Display<'py>>> {
    guard(|| {
        let process = match process_now_ns {
            Some(now) => cmap::from_card(cs, state, now, inputs.is_some())?,
            None => None,
        };
        let lead = stock_lead(cs)?;
        let Some(inputs) = inputs else {
            if let Some(filter) = lead_filter {
                filter_stock_lead(filter, lead.as_ref(), md)?;
            }
            return Ok(None);
        };
        let py = cs.py();
        let raw_model = if !md.is_none() {
            cs.getattr_opt(intern!(py, "model_v2_bytes"))?
                .filter(|v| !v.is_none())
        } else {
            None
        };
        let raw = raw_model
            .as_ref()
            .map(|v| v.extract::<&[u8]>())
            .transpose()?;
        let lines = read_geometry(md, raw)?;
        let meta = optional_model_meta(md)?;
        let desire = meta
            .getattr_opt(intern!(py, "desire"))?
            .map_or(Ok(0), |value| get!(value, "raw").extract::<i64>())?;

        let values_161 = template(cs, "adrv_0x161")?;
        let values_162 = template(cs, "ccnc_0x162")?;
        let values_1ea = template(cs, "adrv_0x1ea")?;
        let mode = if process_now_ns.is_some() || values_162.is_none() {
            0
        } else {
            pos::mode(state)?
        };
        let ff = match &process {
            Some((map, _)) => map.slots[0].map(|slot| (f64::from(slot.distance), f64::from(slot.v_rel))),
            None if mode == 0 => lead.as_ref().map(|lead| (lead.distance, lead.relative_speed)),
            None => None,
        };
        if let (Some(values), Some((map, rear_distances))) = (&values_162, &process) {
            cmap::apply_slots(values, map, Some(*rear_distances))?;
        }
        let panel_attr = intern!(py, "_rust_panel");
        let mut display_state = state.getattr_opt(panel_attr)?.map_or(
            Ok(panel::DisplayState::default()),
            |value| -> PyResult<_> {
                Ok(value
                    .cast_into::<panel::DisplayState>()?
                    .try_borrow()?
                    .clone())
            },
        )?;
        let mut overlay = None;
        if let Some(values) = &values_161 {
            overlay = Some(display_state.draw_161(
                values,
                cs,
                md,
                &lines,
                &inputs,
                frame,
                desire,
                lead.as_ref(),
                ff,
            )?);
            if !get!(cs, "trailer_connected").is_truthy()? {
                let correction = get!(get!(state, "radar_display_tracker"), "position_correction");
                let out = get!(cs, "out");
                let speed: f64 = get!(out, "vEgo").extract()?;
                let left = get!(out, "leftBlinker").is_truthy()?;
                let right = get!(out, "rightBlinker").is_truthy()?;
                let lat_enabled = get!(out, "latEnabled").is_truthy()?;
                update_ego_follow(
                    &correction,
                    md,
                    &lines,
                    frame as f64 * 0.01,
                    (speed * 3.6) / 3.6,
                    desire,
                    (left, right),
                )?;
                let lane_panel = (values.clone(), desire, lat_enabled, inputs.5);
                lane_positions::draw(state, cs, md, raw, frame, &lines, &lane_panel)?;
            }
        }
        if let Some(values) = &values_162 {
            if process.is_none() {
                let (live, lanes, walls, bands) = map.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("missing CCNC radar map state")
                })?;
                let fields = update_map_display(
                    values, cs, md, frame, &lines, &live, state, &lanes, &walls, &bands,
                    position_filter, mode,
                )?;
                values.update(fields.as_mapping())?;
            }
            panel::draw_162(values, cs, inputs.8)?;
        }
        if let Some(values) = &values_1ea {
            panel::draw_1ea(values, cs, get!(inputs.0, "latActive").is_truthy()?, desire)?;
        }
        if values_161.is_some() {
            display_state.publish_traffic(inputs.7.as_ref())?;
        }
        state.setattr(panel_attr, Bound::new(py, display_state)?)?;
        if let Some(filter) = lead_filter {
            filter_stock_lead(filter, lead.as_ref(), md)?;
        }
        Ok(Some((values_161, values_162, values_1ea, overlay)))
    })
}
