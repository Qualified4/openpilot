//! Loops of `CcncRadarMap._measurements` and `CcncRadarMap._associate` (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. Every comparison, `min`/`max` and sort key below follows the
//! Python expression it replaces, in the same order, so that the results are the same to the bit.
//! Python state the map keeps (`reports`, `jumps`, objects) is read and written in place.

use std::collections::HashMap;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::guard;
use crate::lanes::LaneCache;
use crate::map_object::MapObject;
use crate::observe::Frame;
use crate::pymath::{hypot, py_max, py_min};

static DEQUE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `collections.deque`.
pub(crate) fn deque(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    Ok(DEQUE
        .get_or_try_init(py, || -> PyResult<_> { Ok(py.import("collections")?.getattr("deque")?.unbind()) })?
        .bind(py))
}

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

/// `math.floor(v)` for the grid cell of a finite coordinate.
pub(crate) fn floor(v: f64) -> PyResult<i64> {
    if v.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    let f = v.floor();
    if !(-9.2e18..=9.2e18).contains(&f) {
        return Err(PyOverflowError::new_err("cannot convert float infinity to integer"));
    }
    Ok(f as i64)
}

struct Point {
    track: Py<PyAny>, // int(p.trackId)
    x_obj: Py<PyAny>,
    y_obj: Py<PyAny>,
    x: f64,
    y: f64,
    speed: f64,
    motion: Py<PyAny>,      // getattr(p, 'motionState', None)
    track_state: Py<PyAny>, // getattr(p, 'trackState', 0)
}

/// The front radar's measured points with finite values, in input order: the input of `measurements`.
#[pyclass(frozen, module = "ccnc_core")]
pub struct RadarPoints {
    points: Vec<Point>,
}

#[pyfunction]
pub fn radar_points(py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<RadarPoints> {
    guard(|| radar_points_inner(py, points))
}

fn radar_points_inner(py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<RadarPoints> {
    let int_type = py.get_type::<pyo3::types::PyInt>();
    let mut out = Vec::new();
    for p in points.try_iter()? {
        let p = p?;
        if p.getattr(intern!(py, "radarSource"))?.str()?.to_cow()? != "frontRadar"
            || !p.getattr(intern!(py, "measured"))?.is_truthy()?
        {
            continue;
        }
        let objs = [
            p.getattr(intern!(py, "dRel"))?,
            p.getattr(intern!(py, "yRel"))?,
            p.getattr(intern!(py, "vLead"))?,
            p.getattr(intern!(py, "vRel"))?,
        ];
        let mut values = [0.0f64; 4];
        let mut finite = true;
        for (value, obj) in values.iter_mut().zip(&objs) {
            *value = obj.extract()?;
            if !value.is_finite() {
                finite = false;
                break;
            }
        }
        if !finite {
            continue;
        }
        let track = int_type.call1((p.getattr(intern!(py, "trackId"))?,))?;
        let none = py.None().into_bound(py);
        let zero = 0i32.into_pyobject(py)?.into_any();
        let motion = p.getattr_opt(intern!(py, "motionState"))?.unwrap_or(none);
        let track_state = p.getattr_opt(intern!(py, "trackState"))?.unwrap_or(zero);
        let [x_obj, y_obj, ..] = objs;
        out.push(Point {
            track: track.unbind(),
            x_obj: x_obj.unbind(),
            y_obj: y_obj.unbind(),
            x: values[0],
            y: values[1],
            speed: values[2],
            motion: motion.unbind(),
            track_state: track_state.unbind(),
        });
    }
    Ok(RadarPoints { points: out })
}

/// `CcncRadarMap._measurements` after `pose_at`: px/py and cos/sin of the pose angle come from Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn measurements<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    points: &RadarPoints,
    stamp: &Bound<'py, PyAny>,
    px: f64,
    py_: f64,
    c: f64,
    s: f64,
) -> PyResult<Bound<'py, PyList>> {
    guard(|| measurements_inner(py, radar_map, points, stamp, px, py_, c, s))
}

#[allow(clippy::too_many_arguments)]
fn measurements_inner<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    points: &RadarPoints,
    stamp: &Bound<'py, PyAny>,
    px: f64,
    py_: f64,
    c: f64,
    s: f64,
) -> PyResult<Bound<'py, PyList>> {
    let m = radar_map;
    let k = |name: &Bound<'py, pyo3::types::PyString>| attr_f64(m, name);
    let jump_time = k(intern!(py, "BRIDGE_JUMP_TIME"))? + 1e-6;
    let reconnect = k(intern!(py, "BRIDGE_RECONNECT"))?;
    let jump = k(intern!(py, "BRIDGE_JUMP"))?;
    let far_x = k(intern!(py, "FAR_X"))?;
    let max_x = k(intern!(py, "MAX_X"))?;
    let max_y = k(intern!(py, "MAX_Y"))?;
    let far_y = k(intern!(py, "FAR_Y"))?;
    let doppler_min_x = k(intern!(py, "DOPPLER_MIN_X"))?;
    let doppler_near_y = k(intern!(py, "DOPPLER_NEAR_Y"))?;
    let doppler_band = k(intern!(py, "DOPPLER_SPEED"))?
        + k(intern!(py, "REPORT_DELAY_BOUND"))? * k(intern!(py, "a_ego"))?.abs();
    let doppler_yaw = k(intern!(py, "yaw_rate"))?.abs() <= k(intern!(py, "DOPPLER_YAW"))?;
    let ego_speed = k(intern!(py, "speed"))?;
    let reports = m.getattr(intern!(py, "reports"))?;
    let reports = reports.cast::<PyDict>()?;
    let jumps = m.getattr(intern!(py, "jumps"))?;
    let evidence = m.getattr(intern!(py, "evidence"))?;
    let evidence_dict = evidence.cast::<PyDict>().ok();
    let mut leads: Option<Vec<(f64, f64)>> = None;
    let deque = deque(py)?;
    let stamp_f: f64 = stamp.extract()?;
    let two = 2i32.into_pyobject(py)?;
    let out = PyList::empty(py);

    for p in &points.points {
        let track = p.track.bind(py);
        let (x, y, speed) = (p.x, p.y, p.speed);
        let recent = match reports.get_item(track)? {
            Some(recent) => recent,
            None => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "maxlen"), 4)?;
                let recent = deque.call((), Some(&kwargs))?;
                reports.set_item(track, &recent)?;
                recent
            }
        };
        let mut jumped = false;
        for report in recent.try_iter()? {
            let report = report?;
            let (t, rx, ry): (f64, f64, f64) = report.extract()?;
            let dt = stamp_f - t;
            if 0. < dt && dt <= jump_time && (x - rx).abs() <= reconnect && (y - ry).abs() >= jump {
                jumped = true;
                break;
            }
        }
        if jumped {
            jumps.set_item(track, stamp)?;
        }
        let entry = PyTuple::new(py, [stamp.clone(), p.x_obj.bind(py).clone(), p.y_obj.bind(py).clone()])?;
        recent.call_method1(intern!(py, "append"), (entry,))?;
        if !(0. <= x && x <= far_x) || y.abs() > (if x <= max_x { max_y } else { far_y }) || speed.abs() > 100. {
            continue;
        }
        let (wx, wy) = (px + c * x - s * y, py_ + s * x + c * y);
        let distance = hypot(x, y);
        let cosine = if distance > 0. { x / distance } else { 1. };
        let explained = py_min(py_max(speed, 0.), ego_speed * (1. - cosine));
        let report = match evidence_dict {
            Some(d) => d.get_item(track)?,
            None => Some(evidence.call_method1(intern!(py, "get"), (track,))?),
        }
        .filter(|r| !r.is_none());
        let linked = match &report {
            Some(r) => (r.get_item(1)?.extract::<f64>()? - x).abs() <= 1.5,
            None => false,
        };
        let report = report.filter(|_| linked);
        let moving = match &report {
            Some(r) => r.get_item(0)?,
            None => p.motion.bind(py).clone(),
        }
        .rich_compare(&two, CompareOp::Eq)?;
        let tracked = match &report {
            Some(r) if r.len()? > 2 => r.get_item(2)?,
            _ => p.track_state.bind(py).clone(),
        }
        .rich_compare(&two, CompareOp::Eq)?;
        let residual = speed - explained;
        let doppler = cosine >= 0.707
            && (x >= doppler_min_x || y.abs() >= doppler_near_y)
            && residual.abs() > doppler_band
            && doppler_yaw;
        let doppler: i32 = if doppler { if residual > 0. { 1 } else { -1 } } else { 0 };
        if x > max_x && !moving.is_truthy()? {
            if leads.is_none() {
                leads = Some(m.getattr(intern!(py, "leads"))?.extract()?);
            }
            let near_lead = leads.as_ref().unwrap().iter().any(|&(lx, ly)| {
                (x - lx).abs() <= py_max(3., 0.2 * lx) && (y - ly).abs() <= 1.2
            });
            if !near_lead {
                continue;
            }
        }
        let item = (
            track,
            wx,
            wy,
            speed * c,
            speed * s,
            explained,
            c,
            s,
            moving,
            doppler,
            tracked,
        )
            .into_pyobject(py)?;
        out.append(item)?;
    }
    Ok(out)
}

struct Object {
    key: i64,
    key_obj: Py<PyAny>,
    has_report: bool,
    interior: bool,
    slot: Option<i64>,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    confirmed: bool,
    source: i128,
    seen: f64,
    uncertainty: f64,
    gx: f64, // Grid position: reported coordinates at the measurement stamp.
    gy: f64,
}

struct Measurement {
    source: i128,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

/// Map objects (dict order) and measurements as numbers: the input of `associate`.
#[pyclass(frozen, module = "ccnc_core")]
pub struct AssociateInput {
    objects: Vec<Object>,
    measurements: Vec<Measurement>,
    age: f64,
    cos: f64,
    sin: f64,
}

#[pyfunction]
pub fn associate_input(
    py: Python<'_>,
    radar_map: &Bound<'_, PyAny>,
    measurements: &Bound<'_, PyAny>,
    stamp: f64,
) -> PyResult<AssociateInput> {
    guard(|| associate_input_inner(py, radar_map, measurements, stamp))
}

fn associate_input_inner(
    py: Python<'_>,
    radar_map: &Bound<'_, PyAny>,
    measurements: &Bound<'_, PyAny>,
    stamp: f64,
) -> PyResult<AssociateInput> {
    let m = radar_map;
    let age = attr_f64(m, intern!(py, "now"))? - stamp;
    let cos = attr_f64(m, intern!(py, "cos"))?;
    let sin = attr_f64(m, intern!(py, "sin"))?;
    let objects_dict = m.getattr(intern!(py, "objects"))?;
    let objects_dict = objects_dict.cast::<PyDict>()?;
    let mut objects = Vec::with_capacity(objects_dict.len());
    for (key, obj) in objects_dict.iter() {
        let o = obj.cast::<MapObject>()?.try_borrow()?;
        let confirmed = o.motion_confirmed;
        let offset = if confirmed { 0. } else { o.drift };
        let (x, y, vx, vy) = (o.x, o.y, o.vx, o.vy);
        objects.push(Object {
            key: key.extract()?,
            key_obj: key.clone().unbind(),
            has_report: o.report.is_some(),
            interior: o.interior,
            slot: o.slot,
            x,
            y,
            vx,
            vy,
            confirmed,
            source: o.source,
            seen: o.seen,
            uncertainty: o.uncertainty,
            gx: x - vx * age + offset * cos,
            gy: y - vy * age + offset * sin,
        });
    }
    let mut parsed = Vec::new();
    for item in measurements.try_iter()? {
        let item = item?;
        parsed.push(Measurement {
            source: item.get_item(0)?.extract()?,
            x: item.get_item(1)?.extract()?,
            y: item.get_item(2)?.extract()?,
            vx: item.get_item(3)?.extract()?,
            vy: item.get_item(4)?.extract()?,
        });
    }
    Ok(AssociateInput { objects, measurements: parsed, age, cos, sin })
}

/// `local[key][0]`, read only where the Python code reads it.
fn local_x(local: &Bound<'_, PyAny>, object: &Object) -> PyResult<f64> {
    let py = local.py();
    let key = object.key_obj.bind(py);
    match local.cast::<PyDict>() {
        Ok(d) => d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))?,
        Err(_) => local.get_item(key)?,
    }
    .get_item(0)?
    .extract()
}

/// `CcncRadarMap._associate`: {measurement index: object key}, in the Python insertion order.
#[pyfunction]
pub fn associate<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    input: &AssociateInput,
    stamp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    guard(|| associate_inner(py, radar_map, input, stamp))
}

fn associate_inner<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    input: &AssociateInput,
    stamp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let m = radar_map;
    let objects = &input.objects;
    let (cos, sin, age) = (input.cos, input.sin, input.age);
    let local = m.getattr(intern!(py, "local"))?;
    let id_cycle = attr_f64(m, intern!(py, "ID_CYCLE"))?;
    let match_distance = attr_f64(m, intern!(py, "MATCH_DISTANCE"))?;

    // Insertion-ordered grid cells, each listing objects in dict order.
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (index, o) in objects.iter().enumerate() {
        grid.entry((floor(o.gx / 2.)?, floor(o.gy / 2.)?)).or_default().push(index);
    }

    let mut recent: HashMap<i128, usize> = HashMap::new();
    for (index, o) in objects.iter().enumerate() {
        let dt = stamp - o.seen;
        if 0. < dt
            && dt <= id_cycle
            && o.has_report
            && local_x(&local, o)?.abs() <= 15.
        {
            // `source not in recent or seen > objects[recent[source]].seen`
            let newer = match recent.get(&o.source) {
                Some(&other) => o.seen > objects[other].seen,
                None => true,
            };
            if newer {
                recent.insert(o.source, index);
            }
        }
    }

    let mut continuations: Vec<(usize, usize)> = Vec::new();
    let holding = m.getattr(intern!(py, "holding"))?.is_truthy()?;
    if holding {
        let resting_speed = attr_f64(m, intern!(py, "RESTING_SPEED"))?;
        let selected = m.getattr(intern!(py, "selected"))?;
        let side_selected = selected.get_item(pyo3::types::PySlice::new(py, 1, isize::MAX, 1))?;
        let lane_change = m.getattr(intern!(py, "lane_change"))?.is_truthy()?;
        let frame = Frame::read(m)?;
        let mut lanes = LaneCache::new(m);
        for (i, ms) in input.measurements.iter().enumerate() {
            let Some(&index) = recent.get(&ms.source) else { continue };
            let o = &objects[index];
            let (dx, dy) = (ms.x - (o.x - o.vx * age), ms.y - (o.y - o.vy * age));
            let (forward, lateral) = (dx * cos + dy * sin, dy * cos - dx * sin);
            if side_selected.contains(o.key_obj.bind(py))?
                && cos * ms.vx + sin * ms.vy > resting_speed
                && o.confirmed
                && o.interior
                && matches!(o.slot, Some(1 | 2))
                && !lane_change
                && !continuations.iter().any(|&(_, k)| objects[k].key == o.key)
                && forward.abs() <= 2.
                && lateral.abs() <= 0.35
                && hypot(ms.vx - o.vx, ms.vy - o.vy) <= 6.
            {
                let (ex, ey) = frame.to_ego(ms.x, ms.y);
                if lanes.get()?.lane(ex, ey, true, false, &[])?.1 {
                    continuations.push((i, index));
                }
            }
        }
    }

    // (score, measurement index, object key, object index); sorted as the Python tuples (score, i, key).
    let mut pairs: Vec<(f64, usize, i64, usize)> = Vec::new();
    let mut matches: Vec<(f64, usize, i64, usize)> = Vec::new();
    for (i, ms) in input.measurements.iter().enumerate() {
        matches.clear();
        let (cell_x, cell_y) = (floor(ms.x / 2.)?, floor(ms.y / 2.)?);
        for cx in cell_x - 1..cell_x + 2 {
            for cy in cell_y - 1..cell_y + 2 {
                let Some(cell) = grid.get(&(cx, cy)) else { continue };
                for &index in cell {
                    let o = &objects[index];
                    let distance = hypot(ms.x - o.gx, ms.y - o.gy);
                    let same = ms.source == o.source;
                    let near = local_x(&local, o)?.abs() <= 15.;
                    let gate = if same {
                        py_min(1.2, (if near { 0.85 } else { 0.5 }) + o.uncertainty * 0.25)
                    } else {
                        match_distance
                    };
                    let speed_error = hypot(ms.vx - o.vx, ms.vy - o.vy);
                    if distance <= gate && speed_error <= (if same { 6. } else { 3. }) {
                        let score = distance + 0.04 * py_min(speed_error, 3.) - (if same { 0.1 } else { 0. });
                        matches.push((score, i, o.key, index));
                    }
                }
            }
        }
        sort(&mut matches);
        if matches.len() > 1 && matches[1].0 - matches[0].0 < 0.1 && objects[matches[0].3].source != ms.source {
            continue;
        }
        pairs.extend_from_slice(&matches);
    }
    sort(&mut pairs);

    let assigned = PyDict::new(py);
    let mut taken_measurements = vec![false; input.measurements.len()];
    let mut used = vec![false; objects.len()];
    for &(i, index) in &continuations {
        assigned.set_item(i, objects[index].key_obj.bind(py))?;
        taken_measurements[i] = true;
        used[index] = true;
    }
    for &(_, i, _, index) in &pairs {
        if !taken_measurements[i] && !used[index] {
            assigned.set_item(i, objects[index].key_obj.bind(py))?;
            taken_measurements[i] = true;
            used[index] = true;
        }
    }
    Ok(assigned)
}

/// Python tuple order of (score, i, key). Scores are finite and never -0.0 (a NaN distance fails the gate,
/// and the score starts from a distance >= 0), so the total order is the Python order.
fn sort(items: &mut [(f64, usize, i64, usize)]) {
    items.sort_by(|a, b| {
        a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
}
