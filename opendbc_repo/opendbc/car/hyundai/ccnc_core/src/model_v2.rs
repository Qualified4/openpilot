//! The lane lines of a `modelV2` `log.Event` message read from its bytes, and `CcncRadarMap._update_lanes` on them
//! (hyundaicanfd_ccnc_radar_map.py; its Cython twin `_ccnc_map_native.update_lanes`).
//!
//! As for liveTracks (`live_tracks.rs`), the positions read are constants here; `tests/test_ccnc_core_model_v2.py`
//! and card's start (`hyundaicanfd_ccnc_points.model_schema_mismatch`) compare them with the schema pycapnp loads.
//! A message this does not read exactly as pycapnp would gives None, and the map reads the Cap'n Proto objects.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

use crate::guard;
use crate::observe::Frame;
use ccnc_map::input::live_tracks::EVENT_DISCRIMINANT_OFFSET;
use ccnc_map::input::model_v2::{
    parse, parse_path, Line, EVENT_MODEL_V2, EVENT_MODEL_V2_POINTER, LANE_LINES_POINTER, LANE_LINE_PROBS_POINTER,
    POSITION_POINTER, ROAD_EDGES_POINTER, XYZT_X_POINTER, XYZT_Y_POINTER, XYZT_Y_STD_POINTER,
};
pub(crate) use ccnc_map::radar::lanes::{numpy_interp, valid_curve, Curve, GeometryInput};

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
        Ok(Some(PyTuple::new(
            py,
            [
                lines(&m.lane_lines)?.into_any(),
                floats(py, &m.lane_line_probs)?.into_any(),
                lines(&m.road_edges)?.into_any(),
            ],
        )?))
    })
}

/// The constants above, by schema name: compared with the schema pycapnp loads, by the test and when card starts.
#[pyfunction]
pub fn model_v2_schema(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("event_discriminant_offset", EVENT_DISCRIMINANT_OFFSET)?;
    d.set_item("event_model_v2", EVENT_MODEL_V2)?;
    d.set_item("event_model_v2_pointer", EVENT_MODEL_V2_POINTER)?;
    d.set_item("position_pointer", POSITION_POINTER)?;
    d.set_item("lane_lines_pointer", LANE_LINES_POINTER)?;
    d.set_item("lane_line_probs_pointer", LANE_LINE_PROBS_POINTER)?;
    d.set_item("road_edges_pointer", ROAD_EDGES_POINTER)?;
    d.set_item("xyzt_x_pointer", XYZT_X_POINTER)?;
    d.set_item("xyzt_y_pointer", XYZT_Y_POINTER)?;
    d.set_item("xyzt_y_std_pointer", XYZT_Y_STD_POINTER)?;
    Ok(d)
}

static CALLS: AtomicU64 = AtomicU64::new(0);

/// For the parity test only: how many `update_lanes` calls updated the lanes from the bytes.
#[pyfunction]
pub fn _update_lanes_calls_for_test() -> u64 {
    CALLS.load(AtomicOrdering::Relaxed)
}

pub(crate) fn curve(line: &Bound<'_, PyAny>) -> PyResult<Curve> {
    let py = line.py();
    Ok((
        line.getattr(intern!(py, "x"))?.extract()?,
        line.getattr(intern!(py, "y"))?.extract()?,
    ))
}

/// Read the display curvature's position samples. A reader without matching raw bytes uses its fields.
pub(crate) fn read_path(md: &Bound<'_, PyAny>, raw: Option<&[u8]>) -> PyResult<(Curve, Vec<f64>)> {
    if let Some(path) = raw.and_then(parse_path) {
        return Ok(path);
    }
    let py = md.py();
    let position = md.getattr(intern!(py, "position"))?;
    Ok((curve(&position)?, position.getattr(intern!(py, "yStd"))?.extract()?))
}

pub fn read_geometry(md: &Bound<'_, PyAny>, raw: Option<&[u8]>) -> PyResult<GeometryInput> {
    if let Some(model) = raw.and_then(parse) {
        return Ok(GeometryInput::from_raw(model));
    }
    if md.is_none() {
        return Ok(GeometryInput::default());
    }
    let py = md.py();
    let read = |name, n| -> PyResult<Vec<Curve>> {
        let Some(list) = md.getattr_opt(name)? else {
            return Ok(Vec::new());
        };
        list.try_iter()?
            .take(n)
            .map(|line| {
                let line = line?;
                let x = line
                    .getattr_opt(intern!(py, "x"))?
                    .map_or(Ok(Vec::new()), |v| v.extract())?;
                let y = line.getattr(intern!(py, "y"))?.extract()?;
                Ok((x, y))
            })
            .collect()
    };
    let lanes = intern!(py, "laneLines");
    let edges = intern!(py, "roadEdges");
    let probs = md.getattr_opt(intern!(py, "laneLineProbs"))?;
    Ok(GeometryInput {
        lines: read(lanes, 4)?,
        probs: probs.map_or(Ok(Vec::new()), |p| {
            p.try_iter()?.take(4).map(|v| v?.extract()).collect()
        })?,
        edges: read(edges, 2)?,
    })
}

/// Prepare geometry from the current pose and ingest new model curves if available at their stamp.
pub(crate) fn prepare(
    map: &Bound<'_, PyAny>,
    model: Option<(&GeometryInput, f64)>,
    ego_change: bool,
    boundary_type: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let state = crate::map_state::attach(map)?;
    let mut state = state.try_borrow_mut()?;
    if state.boundary_type.is_none() {
        state.boundary_type = Some(boundary_type.clone().unbind());
    }
    let st = &mut state.st;
    let frame = st.frame();
    let previous = st.lane_cache.place(frame);
    if let Some((model, stamp)) = model {
        if let Some((px, py, angle)) = st.pose_at(stamp) {
            let origin = Frame {
                px,
                py,
                c: angle.cos(),
                s: angle.sin(),
            };
            let window = (map.getattr("MIN_X")?.extract()?, map.getattr("FAR_X")?.extract()?);
            st.lane_cache
                .refresh(model, origin, frame, &previous, window, ego_change);
            return Ok(true);
        }
    }
    Ok(false)
}

/// Python map entry point: gather its lane samples and pass them into the Rust cache.
#[pyfunction]
#[pyo3(signature = (map, model, stamp, curves, raw, boundary_type))]
pub fn prepare_lanes(
    map: &Bound<'_, PyAny>,
    model: &Bound<'_, PyAny>,
    stamp: f64,
    curves: &Bound<'_, PyAny>,
    raw: Option<&[u8]>,
    boundary_type: &Bound<'_, PyAny>,
) -> PyResult<()> {
    guard(|| {
        let mut from_bytes = false;
        let input = if model.is_none() {
            None
        } else if let Some(model) = raw.filter(|_| curves.is_none()).and_then(parse) {
            from_bytes = true;
            Some(GeometryInput::from_raw(model))
        } else {
            let mut input = read_geometry(model, None)?;
            if !curves.is_none() {
                let read = |i| -> PyResult<Curve> {
                    let line = curves.call1((model, i))?;
                    if !line.get_item(2)?.is_truthy()? {
                        return Ok(Curve::default());
                    }
                    Ok((
                        line.get_item(0)?.call_method0("tolist")?.extract()?,
                        line.get_item(1)?.call_method0("tolist")?.extract()?,
                    ))
                };
                for (i, line) in input.lines.iter_mut().take(input.probs.len()).enumerate() {
                    *line = read(i)?;
                }
                for (i, line) in input.edges.iter_mut().enumerate() {
                    *line = read(i + 4)?;
                }
            }
            Some(input)
        };
        let ego_change = map.getattr("ego_change")?.is_truthy()?;
        if prepare(
            map,
            input.as_ref().map(|input| (input, stamp)),
            ego_change,
            boundary_type,
        )? && from_bytes
        {
            CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        }
        Ok(())
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update_lanes(
    radar_map: &Bound<'_, PyAny>,
    raw: &[u8],
    px: f64,
    py_: f64,
    pc: f64,
    ps: f64,
    previous_placed: &Bound<'_, PyAny>,
    boundary_type: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    guard(|| {
        let _ = previous_placed; // The attached cache reprojects the preceding lines itself.
        let Some(state) = crate::map_state::attached(radar_map)? else {
            return Ok(false);
        };
        let Some(model) = parse(raw) else { return Ok(false) };
        let window = (
            radar_map.getattr("MIN_X")?.extract()?,
            radar_map.getattr("FAR_X")?.extract()?,
        );
        let changing = radar_map.getattr("ego_change")?.is_truthy()?;
        let mut state = state.try_borrow_mut()?;
        if state.boundary_type.is_none() {
            state.boundary_type = Some(boundary_type.clone().unbind());
        }
        let st = &mut state.st;
        let frame = st.frame();
        let previous = st.lane_cache.place(frame);
        let origin = Frame {
            px,
            py: py_,
            c: pc,
            s: ps,
        };
        st.lane_cache.refresh(
            &GeometryInput::from_raw(model),
            origin,
            frame,
            &previous,
            window,
            changing,
        );
        CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        Ok(true)
    })
}
