//! `_correct_positions` of the radar map display (hyundaicanfd_ccnc_radar_map.py): the map's own part of the
//! position correction, `_road_alignment`, `_model_alignment` and `_bend` included.
//!
//! The state stays where it is: `correction` (the extension's `_CcncVehiclePositionCorrection`) and
//! `_live.filters/aligned/handover/ego_blend` are read and written in their original containers. The
//! model input supplies side lane centres and, while a blend may run, the inner lines, planned path and probabilities.

use pyo3::exceptions::{PyAttributeError, PyKeyError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PySet, PyTuple};

use crate::display_filter::{self, new_position_filter};
use crate::lanes::interpolate;
use crate::map_object::MapObject;
use crate::position_correction::{apply, stabilize_slots, ApplyInput};
use crate::pymath::{py_max, py_min, py_sum};

type Curve = (Vec<f64>, Vec<f64>);
type Points = Vec<(f64, f64)>;
/// `_road_alignment`'s result: aligned y, reference, bounds.
type Alignment<'py> = (f64, Bound<'py, PyTuple>, Option<(f64, f64)>);

/// `_bend(curve, x)`: the lateral bend of one model curve from its start to x, or None.
fn bend(curve: &Curve, x: f64) -> PyResult<Option<f64>> {
    let (xs, ys) = curve;
    let stop = (1..xs.len()).find(|&i| xs[i] <= xs[i - 1]).unwrap_or(xs.len());
    if xs.len() != ys.len() || stop < 2 {
        return Ok(None);
    }
    let points: Vec<(f64, f64)> = xs[..stop].iter().copied().zip(ys[..stop].iter().copied()).collect();
    if !points.iter().all(|&(a, b)| a.is_finite() && b.is_finite()) {
        return Ok(None);
    }
    let at = py_min(py_max(x, points[0].0), points[points.len() - 1].0);
    // Clamped into points that rise strictly, so only a NaN x finds no value: Python's None - float.
    match interpolate(&points, at) {
        Some(y) => Ok(Some(y - points[0].1)),
        None => Err(PyTypeError::new_err("unsupported operand type(s) for -: 'NoneType' and 'float'")),
    }
}

/// The model values `_model_alignment` and the path blend read, made in Python (`_model_inputs`).
struct Model {
    lines: Vec<Curve>,       // laneLines[1:3] as (x, y) lists
    path: Option<Curve>,     // position (x, y), None when the model has no position
    probs: Vec<f64>,         // laneLineProbs[1:3]
}

impl Model {
    /// `_model_alignment(md, slot, x, y, path)`.
    fn alignment(&self, slot: usize, x: f64, y: f64, path: f64) -> PyResult<Option<f64>> {
        let mut bends = Vec::with_capacity(2);
        for line in &self.lines {
            if let Some(b) = bend(line, x)? {
                bends.push(b);
            }
        }
        let mut lane = if bends.is_empty() { None } else { py_sum(bends.iter().copied()).map(|s| s / bends.len() as f64) };
        if slot == 0 && path > 0. && x <= 80. {
            let route = self.path.as_ref().ok_or_else(|| PyAttributeError::new_err("position"))?;
            if let Some(route) = bend(route, x)? {
                lane = Some(match lane {
                    None => route,
                    Some(lane) => lane + path * (route - lane),
                });
            }
        }
        Ok(lane.map(|lane| y + lane))
    }
}

/// The map's lines that `_road_alignment` reads, once per call: for i in 0..4 the points of physical line
/// i + lane_offset when it is usable (probability at least .1, some points), and which of them are weak.
struct Road {
    lines: [Option<Points>; 4],
    weak: [bool; 4],
}

impl Road {
    fn read(tracker: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(state) = crate::map_state::attached(tracker)? {
            let state = state.try_borrow()?;
            let cache = &state.st.lane_cache;
            let mut road = Road { lines: Default::default(), weak: [false; 4] };
            for i in 0..4 {
                let physical = (i as i64 + state.st.lane_offset) as usize;
                road.weak[i] = cache.weak_at(physical);
                if cache.lines.get(&physical).is_some_and(|line| line.probability.partial_cmp(&0.1) != Some(std::cmp::Ordering::Less)) {
                    road.lines[i] = cache.geometry.get(&physical).filter(|points| !points.is_empty()).cloned();
                }
            }
            return Ok(road);
        }
        let py = tracker.py();
        let offset: i64 = tracker.getattr(intern!(py, "lane_offset"))?.extract()?;
        let lanes = tracker.getattr(intern!(py, "lanes"))?;
        let geometry = tracker.getattr(intern!(py, "geometry"))?;
        let weak = tracker.getattr(intern!(py, "weak"))?;
        let (lanes, geometry) = (lanes.cast::<PyDict>()?, geometry.cast::<PyDict>()?);
        let weak = weak.cast::<PySet>()?;
        let mut road = Road { lines: [None, None, None, None], weak: [false; 4] };
        for i in 0..4 {
            let physical = i as i64 + offset;
            road.weak[i] = weak.contains(physical)?;
            let (Some(lane), Some(points)) = (lanes.get_item(physical)?, geometry.get_item(physical)?) else {
                continue;
            };
            let probability: f64 = lane.getattr(intern!(py, "probability"))?.extract()?;
            if probability < 0.1 || !points.is_truthy()? {
                continue;
            }
            road.lines[i] = Some(points.extract()?);
        }
        Ok(road)
    }

    /// `_road_alignment(tracker, slot, x, y)`: (aligned y, reference, bounds).
    fn alignment<'py>(&self, py: Python<'py>, slot: usize, x: f64, y: f64) -> PyResult<Alignment<'py>> {
        let mut lines: [Option<(f64, f64)>; 4] = [None; 4];
        for (i, line) in self.lines.iter().enumerate() {
            if let Some(points) = line {
                let near = py_min(py_max(0., points[0].0), points[points.len() - 1].0);
                if let (Some(here), Some(near)) = (interpolate(points, x), interpolate(points, near)) {
                    lines[i] = Some((here, near));
                }
            }
        }
        let map = || PyTuple::new(py, [intern!(py, "map").as_any(), py.None().bind(py)]);
        let inner: Vec<(f64, f64)> = [lines[1], lines[2]].into_iter().flatten().collect();
        if inner.is_empty() || (inner.len() == 2 && !(in_lane(inner[0].0 - inner[1].0) && in_lane(inner[0].1 - inner[1].1))) {
            return Ok((y, map()?, None));
        }
        let shift = py_sum(inner.iter().map(|(here, near)| here - near)).unwrap_or(0.) / inner.len() as f64;
        if (1..3).any(|i| lines[i].is_some() && self.weak[i]) {
            return Ok((y - shift, map()?, None));
        }
        let (left, right) = [(1, 2), (0, 1), (2, 3)][slot];
        let bounds = match (lines[left], lines[right]) {
            (Some(l), Some(r)) => Some((l.0 - shift, r.0 - shift)),
            _ => None,
        };
        let kind = if inner.len() == 2 {
            intern!(py, "center").clone().into_any()
        } else {
            pyo3::types::PyBool::new(py, lines[1].is_some()).to_owned().into_any()
        };
        Ok((y - shift, PyTuple::new(py, [intern!(py, "lane").as_any(), &kind])?, bounds))
    }
}

fn in_lane(width: f64) -> bool {
    (2.3..=4.8).contains(&width)
}

/// One displayed map object: its key, born and seen as the Python objects the Python code reads.
fn object_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let py = obj.py();
    if let Ok(o) = obj.cast::<MapObject>() {
        let o = o.borrow();
        return Ok((o.key.into_pyobject(py)?.into_any(), PyFloat::new(py, o.born).into_any(), PyFloat::new(py, o.seen).into_any()));
    }
    Ok((obj.getattr(intern!(py, "key"))?, obj.getattr(intern!(py, "born"))?, obj.getattr(intern!(py, "seen"))?))
}

/// The map's objects: its Rust state, or the `objects` and `local` dicts of a map without one (the reference
/// map, or a test's stand-in).
enum Objects<'py> {
    Rust(Bound<'py, crate::map_state::RadarMapState>),
    Py(Bound<'py, PyDict>, Bound<'py, PyAny>),
}

type Fields<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, f64, f64);

impl<'py> Objects<'py> {
    fn read(tracker: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = tracker.py();
        if let Some(state) = crate::map_state::attached(tracker)? {
            return Ok(Self::Rust(state));
        }
        Ok(Self::Py(tracker.getattr(intern!(py, "objects"))?.cast_into()?, tracker.getattr(intern!(py, "local"))?))
    }

    /// `{key: (born, round(seen * 100))}` of every object, in dict order.
    fn observations(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let observations = PyDict::new(py);
        match self {
            Self::Rust(state) => {
                for (key, o) in &state.try_borrow()?.st.objects {
                    let scaled = PyFloat::new(py, o.seen * 100.).into_any();
                    observations.set_item(key, (o.born, py_round(py, &scaled)?))?;
                }
            }
            Self::Py(objects, _) => {
                for obj in objects.values() {
                    let (key, born, seen) = object_fields(&obj)?;
                    let scaled = match seen.cast::<PyFloat>() {
                        Ok(s) => PyFloat::new(py, s.value() * 100.).into_any(),
                        Err(_) => seen.mul(100.)?,
                    };
                    observations.set_item(key, (born, py_round(py, &scaled)?))?;
                }
            }
        }
        Ok(observations)
    }

    /// `objects.get(key)`: its key, born, seen and `local[key]`.
    fn get(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Option<Fields<'py>>> {
        match self {
            Self::Rust(state) => {
                let Some(key) = key.extract::<Option<i64>>()? else { return Ok(None) };
                let state = state.try_borrow()?;
                let Some(o) = state.st.objects.get(&key) else { return Ok(None) };
                let (x, y) = state.st.local.get(&key).copied().ok_or_else(|| PyKeyError::new_err(key))?;
                Ok(Some((
                    key.into_pyobject(py)?.into_any(),
                    PyFloat::new(py, o.born).into_any(),
                    PyFloat::new(py, o.seen).into_any(),
                    x,
                    y,
                )))
            }
            Self::Py(objects, local) => {
                let Some(obj) = objects.get_item(key)? else { return Ok(None) };
                let (key, born, seen) = object_fields(&obj)?;
                let (x, y): (f64, f64) = local.get_item(&key)?.extract()?;
                Ok(Some((key, born, seen, x, y)))
            }
        }
    }
}

/// Python `round(value)` of a float: half to even, as an int (ValueError for NaN, OverflowError for infinity).
fn py_round<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(v) = value.cast::<PyFloat>() {
        let r = v.value().round_ties_even();
        if r.is_finite() && r.abs() < 9.0e15 {
            return Ok((r as i64).into_pyobject(py)?.into_any());
        }
    }
    py.import(intern!(py, "builtins"))?.getattr(intern!(py, "round"))?.call1((value,))
}

fn item<'py>(obj: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    obj.get_item(index)
}

/// `_correct_positions(tracker, correction, values, frame, md, ext)` after the model reads, which the caller does:
/// `centers` are the two `_ccnc_side_lane_center` values, `model` is None or (inner lines, path, probabilities)
/// of `_live.model` while it is fresh (given whenever the blend may be above zero).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (tracker, correction, values, frame, centers, model, live, position_filter, align_time, handover_time))]
pub fn correct_positions<'py>(
    py: Python<'py>,
    tracker: &Bound<'py, PyAny>,
    correction: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    frame: i64,
    centers: (Bound<'py, PyAny>, Bound<'py, PyAny>),
    model: Option<(Vec<Curve>, Option<Curve>, Vec<f64>)>,
    live: &Bound<'py, PyAny>,
    position_filter: &Bound<'py, PyAny>,
    align_time: f64,
    handover_time: f64,
) -> PyResult<()> {
    crate::guard(|| {
        let model = model.map(|(lines, path, probs)| Model { lines, path, probs });
        correct(py, tracker, correction, values, frame, centers, model.as_ref(), live, position_filter, align_time, handover_time)
    })
}

/// Exercise apply with and without a filter: measured map slots always supply one.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _position_apply_for_test<'py>(
    correction: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
    position: (f64, f64, bool),
    aligned: f64,
    filtered: f64,
    slot: usize,
    reference: &Bound<'py, PyTuple>,
    bounds: Option<(f64, f64)>,
    observation: &Bound<'py, PyAny>,
    frame: i64,
    reliable: bool,
    lateral_filter: Option<&Bound<'py, PyAny>>,
) -> PyResult<f64> {
    crate::guard(|| {
        let (x, y, fresh) = position;
        apply(correction, ApplyInput {
            key, x, y, fresh, aligned, filtered, slot, reference, bounds, observation, frame,
            reliable, lateral_filter,
        })
    })
}

pub(crate) fn update_centers<'py>(
    correction: &Bound<'py, PyAny>,
    centers: (Bound<'py, PyAny>, Bound<'py, PyAny>),
    frame: i64,
    observations: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let py = correction.py();
    let stamp: Option<i64> = correction.getattr(intern!(py, "center_frame"))?.extract()?;
    let dt = stamp.map_or(0., |stamp| py_max(0., py_min(0.15, (frame - stamp) as f64 * 0.01)));
    correction.setattr(intern!(py, "center_frame"), frame)?;
    let previous = correction.getattr(intern!(py, "centers"))?;
    let valid = correction.getattr(intern!(py, "center_valid"))?;
    for (i, target) in [centers.0, centers.1].into_iter().enumerate() {
        let target: Option<f64> = target.extract()?;
        let target = target.filter(|x| x.is_finite());
        valid.set_item(i, target.is_some())?;
        if let Some(mut target) = target {
            if i == 1 {
                target = -target;
            }
            let center: f64 = previous.get_item(i)?.extract()?;
            let tau: f64 = correction.getattr(intern!(py, "CENTER_TAU"))?.extract()?;
            let rate: f64 = correction.getattr(intern!(py, "CENTER_RATE"))?.extract()?;
            let limit = dt * rate;
            let step = (target - center) * -(-dt / tau).exp_m1();
            previous.set_item(i, center + py_min(py_max(step, -limit), limit))?;
        }
    }
    let old = correction.getattr(intern!(py, "tracks"))?.cast_into::<PyDict>()?;
    let retained = PyDict::new(py);
    for (key, saved) in old.iter() {
        if let Some(observation) = observations.get_item(&key)? {
            if observation.get_item(0)?.eq(saved.get_item(intern!(py, "birth"))?)? {
                retained.set_item(key, saved)?;
            }
        }
    }
    correction.setattr(intern!(py, "tracks"), retained)
}

#[allow(clippy::too_many_arguments)]
fn correct<'py>(
    py: Python<'py>,
    tracker: &Bound<'py, PyAny>,
    correction: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    frame: i64,
    centers: (Bound<'py, PyAny>, Bound<'py, PyAny>),
    model: Option<&Model>,
    live: &Bound<'py, PyAny>,
    position_filter: &Bound<'py, PyAny>,
    align_time: f64,
    handover_time: f64,
) -> PyResult<()> {
    let objects = Objects::read(tracker)?;
    let observations = objects.observations(py)?;
    update_centers(correction, centers, frame, &observations)?;
    let filters = PyDict::new(py);

    if !live.getattr(intern!(py, "generation"))?.is(&tracker.getattr(intern!(py, "generation"))?) {
        live.setattr(intern!(py, "generation"), tracker.getattr(intern!(py, "generation"))?)?;
        live.setattr(intern!(py, "ego_blend"), (py.None(), 0., 0.))?;
        live.setattr(intern!(py, "aligned"), PyDict::new(py))?;
        live.setattr(intern!(py, "handover"), PyDict::new(py))?;
    }
    let (stamp, blend, path): (Option<i64>, f64, f64) = live.getattr(intern!(py, "ego_blend"))?.extract()?;
    let dt = match stamp {
        Some(stamp) => py_min(py_max((frame - stamp) as f64 * 0.01, 0.), 0.25),
        None => 0.,
    };
    let active = correction.getattr(intern!(py, "ego_follow_active"))?.is_truthy()?;
    let blend = py_min(py_max(blend + (if active { 1. } else { -1. }) * dt / align_time, 0.), 1.);
    let probs: &[f64] = model.map(|m| m.probs.as_slice()).unwrap_or(&[]);
    let uncertain = probs.len() == 2 && py_max(probs[0], probs[1]) < 0.5;
    let path = if blend > 0. {
        py_min(py_max(path + (if uncertain { 1. } else { -1. }) * dt / align_time, 0.), 1.)
    } else {
        0.
    };
    live.setattr(intern!(py, "ego_blend"), (frame, blend, path))?;
    let (handover, drawn) = if blend > 0. {
        (live.getattr(intern!(py, "handover"))?.cast_into::<PyDict>()?, PyDict::new(py))
    } else {
        (PyDict::new(py), PyDict::new(py))
    };
    let live_aligned = live.getattr(intern!(py, "aligned"))?;
    let live_filters = live.getattr(intern!(py, "filters"))?;
    let selected = tracker.getattr(intern!(py, "selected"))?;
    let remembered = tracker.getattr(intern!(py, "remembered"))?;
    let slot_fields = tracker.getattr(intern!(py, "SLOT_FIELDS"))?;
    let tracks = correction.getattr(intern!(py, "tracks"))?;
    let mut road: Option<Road> = None;
    for (slot, fields) in slot_fields.try_iter()?.enumerate() {
        let fields = fields?;
        let (distance, lateral, detect): (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>) = fields.extract()?;
        let Some((key, born, seen, x, y)) = objects.get(py, &item(&selected, slot)?)? else { continue };
        if !values.get_item(&detect)?.is_truthy()? {
            continue;
        }
        let mut output: f64;
        if item(&remembered, slot)?.is_truthy()? {
            let saved = tracks.call_method1(intern!(py, "get"), (&key,))?;
            output = if !saved.is_none() && saved.get_item(intern!(py, "birth"))?.eq(&born)? {
                saved.get_item(intern!(py, "output"))?.extract()?
            } else {
                y
            };
        } else {
            if road.is_none() {
                road = Some(Road::read(tracker)?);
            }
            let (mut aligned, reference, bounds) = road.as_ref().unwrap().alignment(py, slot, x, y)?;
            let target = match (blend > 0., model) {
                (true, Some(model)) => model.alignment(slot, x, y, path)?,
                _ => None,
            };
            if let Some(target) = target {
                aligned += blend * (target - aligned);
            }
            if blend > 0. {
                let last = live_aligned.call_method1(intern!(py, "get"), (&key,))?;
                if !last.is_none() && item(&last, 0)?.eq(&born)? && item(&last, 1)?.ne(slot)? {
                    let since = frame - item(&last, 3)?.extract::<i64>()?;
                    if (0..=15).contains(&since) {
                        let drawn_y: f64 = item(&last, 2)?.extract()?;
                        handover.set_item(&key, (&born, drawn_y - aligned, frame))?;
                    }
                }
                if let Some(carried) = handover.get_item(&key)? {
                    if item(&carried, 0)?.eq(&born)? {
                        let start: i64 = item(&carried, 2)?.extract()?;
                        let left = 1. - (frame - start) as f64 * 0.01 / handover_time;
                        if left > 0. {
                            aligned += item(&carried, 1)?.extract::<f64>()? * left;
                        } else {
                            handover.del_item(&key)?;
                        }
                    }
                }
                drawn.set_item(&key, (&born, slot, aligned, frame))?;
            }
            let previous = live_filters.call_method1(intern!(py, "get"), (&key,))?;
            let renew = previous.is_none() || item(&previous, 0)?.ne(&born)? || {
                let since = frame - item(&previous, 1)?.extract::<i64>()?;
                !(0..=15).contains(&since)
            };
            let previous = if renew {
                PyTuple::new(py, [
                    born.clone(),
                    frame.into_pyobject(py)?.into_any(),
                    new_position_filter(position_filter, x, true)?,
                    new_position_filter(position_filter, aligned, false)?,
                ])?
                .into_any()
            } else {
                previous
            };
            let (_, last, far, side): (Bound<'py, PyAny>, i64, Bound<'py, PyAny>, Bound<'py, PyAny>) = previous.extract()?;
            side.setattr(intern!(py, "dt"), (frame - last) as f64 * 0.01)?;
            filters.set_item(&key, (&born, frame, &far, &side))?;
            let far_x: f64 = display_filter::position(&far, x, true)?;
            values.set_item(&distance, py_min(204.7, py_max(0., far_x * 0.8)))?;
            let fresh = seen.eq(tracker.getattr(intern!(py, "radar_stamp"))?)?;
            let reliable = match bounds {
                None => false,
                Some(_) if slot == 0 => true,
                Some(_) => item(&correction.getattr(intern!(py, "center_valid"))?, slot - 1)?.is_truthy()?,
            };
            let filtered = display_filter::position(&side, aligned, false)?;
            let observation = observations.get_item(&key)?.ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))?;
            output = apply(correction, ApplyInput {
                key: &key, x, y, fresh, aligned, filtered, slot, reference: &reference, bounds,
                observation: &observation, frame, reliable, lateral_filter: Some(&side),
            })?;
        }
        if slot == 0 {
            values.set_item(&lateral, py_min(12.7, py_max(-12.8, -output)))?;
        } else {
            output = py_min(py_max(output, -5.4), 5.4);
            values.set_item(&lateral, py_min(py_max(if slot == 1 { output } else { -output }, 0.), 12.7))?;
        }
    }
    live.setattr(intern!(py, "filters"), filters)?;
    live.setattr(intern!(py, "aligned"), drawn)?;
    live.setattr(intern!(py, "handover"), &handover)?;
    let bypass = tracker.getattr(intern!(py, "holding"))?;
    stabilize_slots(correction, values, &selected, &observations, frame, &bypass)
}
