//! The liveTracks entry points over `ccnc_map::input::live_tracks`, which reads the message bytes.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};

use crate::guard;
use ccnc_map::input::live_tracks::{
    parse, A_LEAD, A_REL, D_REL, EVENT_DISCRIMINANT_OFFSET, EVENT_LIVE_TRACKS, EVENT_LIVE_TRACKS_POINTER, J_LEAD, MEASURED,
    MOTION_STATE, RADAR_DATA_POINTS_POINTER, RADAR_SOURCE, RADAR_SOURCES, TRACK_ID, TRACK_STATE, V_LEAD, V_REL, YV_REL, Y_REL,
};

/// `hyundaicanfd_ccnc_points.snapshot` of the message in `raw`: a `tracks` object whose `points` list holds a `point`
/// object per RadarPoint with the 13 fields as pycapnp gives them. None where `parse` gives None.
#[pyfunction]
pub fn live_tracks_copy<'py>(
    py: Python<'py>,
    raw: &[u8],
    point: &Bound<'py, PyAny>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    guard(|| {
        let Some(parsed) = parse(raw) else { return Ok(None) };
        let points = PyList::empty(py);
        for p in &parsed {
            let out = point.call0()?;
            let float = |v: f32| PyFloat::new(py, v as f64);
            out.setattr(intern!(py, "trackId"), PyInt::new(py, p.track_id))?;
            out.setattr(intern!(py, "dRel"), float(p.d_rel))?;
            out.setattr(intern!(py, "yRel"), float(p.y_rel))?;
            out.setattr(intern!(py, "vRel"), float(p.v_rel))?;
            out.setattr(intern!(py, "aRel"), float(p.a_rel))?;
            out.setattr(intern!(py, "yvRel"), float(p.yv_rel))?;
            out.setattr(intern!(py, "measured"), PyBool::new(py, p.measured))?;
            out.setattr(intern!(py, "vLead"), float(p.v_lead))?;
            out.setattr(intern!(py, "aLead"), float(p.a_lead))?;
            out.setattr(intern!(py, "jLead"), float(p.j_lead))?;
            out.setattr(intern!(py, "trackState"), PyInt::new(py, p.track_state))?;
            out.setattr(intern!(py, "motionState"), PyInt::new(py, p.motion_state))?;
            out.setattr(intern!(py, "radarSource"), PyString::intern(py, RADAR_SOURCES[p.source as usize]))?;
            points.append(out)?;
        }
        let out = tracks.call0()?;
        out.setattr(intern!(py, "points"), points)?;
        Ok(Some(out))
    })
}

/// For the parser test only: each point as a tuple in RadarPoint field order, as `live_tracks_copy` gives its fields,
/// or None where `parse` gives None.
#[pyfunction]
pub fn _parse_for_test<'py>(py: Python<'py>, raw: &[u8]) -> PyResult<Option<Bound<'py, PyList>>> {
    guard(|| {
        let Some(parsed) = parse(raw) else { return Ok(None) };
        let out = PyList::empty(py);
        for p in &parsed {
            let float = |v: f32| PyFloat::new(py, v as f64).into_any();
            let fields = [
                PyInt::new(py, p.track_id).into_any(), float(p.d_rel), float(p.y_rel), float(p.v_rel), float(p.a_rel),
                float(p.yv_rel), PyBool::new(py, p.measured).to_owned().into_any(), float(p.v_lead), float(p.a_lead),
                float(p.j_lead), PyString::intern(py, RADAR_SOURCES[p.source as usize]).into_any(),
                PyInt::new(py, p.track_state).into_any(), PyInt::new(py, p.motion_state).into_any(),
            ];
            out.append(pyo3::types::PyTuple::new(py, fields)?)?;
        }
        Ok(Some(out))
    })
}

/// The constants above, by schema name: compared with the schema pycapnp loads, by the test and when card starts.
#[pyfunction]
pub fn live_tracks_schema(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("event_discriminant_offset", EVENT_DISCRIMINANT_OFFSET)?;
    d.set_item("event_live_tracks", EVENT_LIVE_TRACKS)?;
    d.set_item("event_live_tracks_pointer", EVENT_LIVE_TRACKS_POINTER)?;
    d.set_item("radar_data_points_pointer", RADAR_DATA_POINTS_POINTER)?;
    let fields = pyo3::types::PyDict::new(py);
    fields.set_item("trackId", ("uint64", TRACK_ID.0, TRACK_ID.1))?;
    for (name, (offset, default)) in [("dRel", D_REL), ("yRel", Y_REL), ("vRel", V_REL), ("aRel", A_REL), ("yvRel", YV_REL),
                                      ("vLead", V_LEAD), ("aLead", A_LEAD), ("jLead", J_LEAD)] {
        fields.set_item(name, ("float32", offset, default))?;
    }
    fields.set_item("measured", ("bool", MEASURED.0, MEASURED.1))?;
    fields.set_item("radarSource", ("enum", RADAR_SOURCE.0, RADAR_SOURCE.1))?;
    fields.set_item("trackState", ("uint8", TRACK_STATE.0, TRACK_STATE.1))?;
    fields.set_item("motionState", ("uint8", MOTION_STATE.0, MOTION_STATE.1))?;
    d.set_item("radar_point_fields", fields)?;
    d.set_item("radar_sources", RADAR_SOURCES.to_vec())?;
    Ok(d)
}
