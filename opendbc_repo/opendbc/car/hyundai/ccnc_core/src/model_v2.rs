//! The lane lines of a `modelV2` `log.Event` message read from its bytes, and `CcncRadarMap._update_lanes` on them
//! (hyundaicanfd_ccnc_radar_map.py; its Cython twin `_ccnc_map_native.update_lanes`).
//!
//! As for liveTracks (`live_tracks.rs`), the positions read are constants here; `tests/test_ccnc_core_model_v2.py`
//! and card's start (`hyundaicanfd_ccnc_points.model_schema_mismatch`) compare them with the schema pycapnp loads.
//! A message this does not read exactly as pycapnp would gives None, and the map reads the Cap'n Proto objects.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

use crate::guard;
use crate::lanes::interpolate;
use crate::pymath::py_max;
use crate::live_tracks::{data_bytes, Message, EVENT_DISCRIMINANT_OFFSET};

/// `Event`: the modelV2 member of the union.
pub const EVENT_MODEL_V2: u16 = 73;
pub const EVENT_MODEL_V2_POINTER: usize = 0;
/// `ModelDataV2` pointer indices.
pub const LANE_LINES_POINTER: usize = 4; // List(XYZTData)
pub const LANE_LINE_PROBS_POINTER: usize = 5; // List(Float32)
pub const ROAD_EDGES_POINTER: usize = 6; // List(XYZTData)
/// `XYZTData` pointer indices, both List(Float32).
pub const XYZT_X_POINTER: usize = 0;
pub const XYZT_Y_POINTER: usize = 1;

/// Floats and list elements read from one message at most: overlapping pointers could otherwise make a small message
/// read as a huge one (messaging reads without a traversal limit; this gives None instead).
const READ_LIMIT: usize = 1 << 24;

pub struct Line {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

pub struct Model {
    pub lane_lines: Vec<Line>,
    pub lane_line_probs: Vec<f32>,
    pub road_edges: Vec<Line>,
}

/// laneLines, laneLineProbs and roadEdges of a modelV2 Event message, or None where this does not read them exactly
/// as pycapnp.
pub fn parse(bytes: &[u8]) -> Option<Model> {
    let message = Message::new(bytes)?;
    let event = message.read_struct(0, 0)?;
    if u16::from_le_bytes(data_bytes(event.data, EVENT_DISCRIMINANT_OFFSET * 2)) != EVENT_MODEL_V2 {
        return None;
    }
    let model = message.struct_pointer(&event, EVENT_MODEL_V2_POINTER)?;
    let mut read = 0usize;
    let mut lines = |index: usize| -> Option<Vec<Line>> {
        let elements = message.struct_elements(&model, index)?;
        read += elements.len();
        let mut out = Vec::with_capacity(elements.len());
        for e in &elements {
            let line = Line { x: message.f32_list(e, XYZT_X_POINTER)?, y: message.f32_list(e, XYZT_Y_POINTER)? };
            read += line.x.len() + line.y.len();
            if read > READ_LIMIT {
                return None;
            }
            out.push(line);
        }
        Some(out)
    };
    let lane_lines = lines(LANE_LINES_POINTER)?;
    let road_edges = lines(ROAD_EDGES_POINTER)?;
    let lane_line_probs = message.f32_list(&model, LANE_LINE_PROBS_POINTER)?;
    Some(Model { lane_lines, lane_line_probs, road_edges })
}

fn floats<'py>(py: Python<'py>, values: &[f32]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().map(|&v| PyFloat::new(py, v as f64)))
}

/// For the parser test only: ([(x, y) per lane line], laneLineProbs, [(x, y) per road edge]) as Python floats, or
/// None where `parse` gives None.
#[pyfunction]
pub fn _parse_model_for_test<'py>(py: Python<'py>, raw: &[u8]) -> PyResult<Option<Bound<'py, PyTuple>>> {
    guard(|| {
        let Some(m) = parse(raw) else { return Ok(None) };
        let lines = |lines: &[Line]| -> PyResult<Bound<'py, PyList>> {
            let out = PyList::empty(py);
            for l in lines {
                out.append((floats(py, &l.x)?, floats(py, &l.y)?))?;
            }
            Ok(out)
        };
        Ok(Some(PyTuple::new(py, [lines(&m.lane_lines)?.into_any(), floats(py, &m.lane_line_probs)?.into_any(),
                                  lines(&m.road_edges)?.into_any()])?))
    })
}

/// The constants above, by schema name: compared with the schema pycapnp loads, by the test and when card starts.
#[pyfunction]
pub fn model_v2_schema(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("event_discriminant_offset", EVENT_DISCRIMINANT_OFFSET)?;
    d.set_item("event_model_v2", EVENT_MODEL_V2)?;
    d.set_item("event_model_v2_pointer", EVENT_MODEL_V2_POINTER)?;
    d.set_item("lane_lines_pointer", LANE_LINES_POINTER)?;
    d.set_item("lane_line_probs_pointer", LANE_LINE_PROBS_POINTER)?;
    d.set_item("road_edges_pointer", ROAD_EDGES_POINTER)?;
    d.set_item("xyzt_x_pointer", XYZT_X_POINTER)?;
    d.set_item("xyzt_y_pointer", XYZT_Y_POINTER)?;
    Ok(d)
}

/// Python's `<` on two-float tuples: the first unequal element decides (NaN is unequal to everything).
fn tuple_lt(a: &(f64, f64), b: &(f64, f64)) -> bool {
    if a.0 != b.0 {
        a.0 < b.0
    } else {
        a.1 < b.1
    }
}

/// `sorted()` of two-float tuples. Without NaN the order is total and a stable sort gives Timsort's result; with a
/// NaN it is not, and Python sorts them.
fn sorted(py: Python<'_>, mut points: Vec<(f64, f64)>) -> PyResult<Vec<(f64, f64)>> {
    if points.iter().any(|p| p.0.is_nan() || p.1.is_nan()) {
        let list = PyList::new(py, &points)?;
        list.sort()?;
        return list.extract();
    }
    points.sort_by(|a, b| if tuple_lt(a, b) { Ordering::Less } else if tuple_lt(b, a) { Ordering::Greater } else { Ordering::Equal });
    Ok(points)
}

/// `bisect_left(values, x)` on floats.
fn bisect_left(values: &[f64], x: f64) -> usize {
    let (mut lo, mut hi) = (0, values.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if values[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `[p for j, p in enumerate(points) if j == 0 or p[0] > points[j - 1][0] + .001]` as tuples.
fn deduplicated<'py>(py: Python<'py>, points: &[(f64, f64)]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for (j, p) in points.iter().enumerate() {
        if j == 0 || p.0 > points[j - 1].0 + 0.001 {
            out.append(*p)?;
        }
    }
    Ok(out)
}

static CALLS: AtomicU64 = AtomicU64::new(0);

/// For the parity test only: how many `update_lanes` calls updated the lanes from the bytes.
#[pyfunction]
pub fn _update_lanes_calls_for_test() -> u64 {
    CALLS.load(AtomicOrdering::Relaxed)
}

/// `_update_lanes(model, stamp, None, previous_placed)` on the message in `raw`, after the caller's
/// `pose = self.pose_at(stamp)` (not None) and `pc, ps = math.cos(pose[2]), math.sin(pose[2])`. False, with nothing
/// changed, where `parse` gives None: the caller then runs the Cap'n Proto path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update_lanes<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    raw: &[u8],
    px: f64,
    py_: f64,
    pc: f64,
    ps: f64,
    previous_placed: &Bound<'py, PyAny>,
    boundary_type: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    guard(|| {
        let Some(model) = parse(raw) else { return Ok(false) };
        CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        let m = radar_map;
        let (min_x, far_x): (f64, f64) = (m.getattr(intern!(py, "MIN_X"))?.extract()?, m.getattr(intern!(py, "FAR_X"))?.extract()?);
        let ego_change = m.getattr(intern!(py, "ego_change"))?.is_truthy()?;
        let pose: (f64, f64, f64) = m.getattr(intern!(py, "pose"))?.extract()?;
        let (c, s): (f64, f64) = (m.getattr(intern!(py, "cos"))?.extract()?, m.getattr(intern!(py, "sin"))?.extract()?);
        let lanes = m.getattr(intern!(py, "lanes"))?.cast_into::<PyDict>()?;
        let geometry = m.getattr(intern!(py, "geometry"))?.cast_into::<PyDict>()?;
        let weak = m.getattr(intern!(py, "weak"))?;
        let to_ego = |(x, y): (f64, f64)| {
            let (dx, dy) = (x - pose.0, y - pose.1);
            (c * dx + s * dy, -s * dx + c * dy)
        };
        // lines = list(zip(laneLines, laneLineProbs))[:4], padded to 4, then the first two road edges at probability 1.
        let count = model.lane_lines.len().min(model.lane_line_probs.len()).min(4);
        let entries = (0..4).map(|i| (i < count).then(|| (&model.lane_lines[i], model.lane_line_probs[i] as f64)))
            .chain(model.road_edges.iter().take(2).map(|edge| Some((edge, 1.))));
        for (i, entry) in entries.enumerate() {
            let Some((line, mut probability)) = entry else { continue };
            let n = line.x.len().min(line.y.len());
            let coords: Vec<(f64, f64)> = (0..n).map(|k| (line.x[k] as f64, line.y[k] as f64)).collect();
            let valid = n >= 2 && line.x.len() == line.y.len()
                && coords.iter().all(|p| p.0.is_finite() && p.1.is_finite())
                && coords.windows(2).all(|w| w[1].0 > w[0].0);
            if !valid || !probability.is_finite() {
                continue;
            }
            let old = lanes.get_item(i)?;
            if probability < 0.1 {
                if !ego_change || i > 3 {
                    if let Some(old) = &old {
                        let value: f64 = old.getattr(intern!(py, "probability"))?.extract()?;
                        old.setattr(intern!(py, "probability"), value * 0.9)?;
                    }
                    continue;
                }
                probability = 0.1;
                weak.call_method1(intern!(py, "add"), (i,))?;
            } else {
                weak.call_method1(intern!(py, "discard"), (i,))?;
            }
            let distances: Vec<f64> = coords.iter().map(|p| p.0).collect();
            let first = bisect_left(&distances, min_x).saturating_sub(1);
            let last = (bisect_left(&distances, far_x) + 1).min(n);
            let world: Vec<(f64, f64)> = coords[first..last.max(first)].iter()
                .map(|&(x, y)| (px + pc * x + ps * y, py_ + ps * x - pc * y)).collect();
            let placed: Vec<(f64, f64)> = world.iter().copied().map(to_ego).collect();
            let local = sorted(py, placed.clone())?;
            let previous_geometry: Vec<(f64, f64)> = match geometry.get_item(i)? {
                Some(points) => points.extract()?,
                None => Vec::new(),
            };
            let mut max_error: Option<f64> = None;
            for &(x, y) in &local {
                if (0. ..=30.).contains(&x) {
                    if let Some(previous) = interpolate(&previous_geometry, x) {
                        let error = (y - previous).abs();
                        max_error = Some(max_error.map_or(error, |m| py_max(m, error))); // max(errors)
                    }
                }
            }
            let consistent = match (&old, max_error) {
                (Some(old), Some(e)) if e <= 0.35 => old.getattr(intern!(py, "consistent"))?.extract::<i64>()? + 1,
                _ => 1,
            };
            let limit = local[0].0 - 0.1;
            let mut behind: Vec<(Bound<'py, PyAny>, (f64, f64))> = Vec::new();
            if let Some(old) = &old {
                let old_points = old.getattr(intern!(py, "points"))?;
                let old_ego = previous_placed.get_item(i)?;
                for (point, ego) in old_points.try_iter()?.zip(old_ego.try_iter()?) {
                    let (point, ego) = (point?, ego?.extract::<(f64, f64)>()?);
                    if -5. <= ego.0 && ego.0 < limit {
                        behind.push((point, ego));
                    }
                }
                behind.drain(..behind.len().saturating_sub(24));
            }
            let points = PyList::empty(py);
            for (point, _) in &behind {
                points.append(point)?;
            }
            for p in world.iter().take(40) {
                points.append(*p)?;
            }
            lanes.set_item(i, boundary_type.call1((points, probability, consistent.min(20)))?)?;
            let ego: Vec<(f64, f64)> = behind.iter().map(|(_, ego)| *ego).chain(placed.iter().take(40).copied()).collect();
            geometry.set_item(i, deduplicated(py, &sorted(py, ego)?)?)?;
        }
        Ok(true)
    })
}
