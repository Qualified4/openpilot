//! The extension side of `CcncRadarMap._measurements` and `_associate`: the radar points and the offline radar class
//! reports as `ccnc_map::radar::associate` reads them (hyundaicanfd_ccnc_radar_map.py).

use std::collections::HashMap;

use pyo3::basic::CompareOp;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use ccnc_map::input::live_tracks::RawPoint;
use ccnc_map::radar::associate::{Evidence, RadarPoint};

static DEQUE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `collections.deque`.
pub(crate) fn deque(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    Ok(DEQUE
        .get_or_try_init(py, || -> PyResult<_> { Ok(py.import("collections")?.getattr("deque")?.unbind()) })?
        .bind(py))
}

/// The front radar's measured points with finite values, in input order: what `_measurements` reads of
/// `live.points`.
pub(crate) fn radar_points_inner(py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<Vec<RadarPoint>> {
    let int_type = py.get_type::<pyo3::types::PyInt>();
    let two = 2i32.into_pyobject(py)?;
    let mut out = Vec::new();
    for p in points.try_iter()? {
        let p = p?;
        if p.getattr(intern!(py, "radarSource"))?.str()?.to_cow()? != "frontRadar"
            || !p.getattr(intern!(py, "measured"))?.is_truthy()?
        {
            continue;
        }
        let mut values = [0.0f64; 4];
        let mut finite = true;
        for (value, name) in values.iter_mut().zip([intern!(py, "dRel"), intern!(py, "yRel"), intern!(py, "vLead"), intern!(py, "vRel")]) {
            *value = p.getattr(name)?.extract()?;
            if !value.is_finite() {
                finite = false;
                break;
            }
        }
        if !finite {
            continue;
        }
        let track: i128 = int_type.call1((p.getattr(intern!(py, "trackId"))?,))?.extract()?;
        let moving = match p.getattr_opt(intern!(py, "motionState"))? {
            Some(motion) => motion.rich_compare(&two, CompareOp::Eq)?.is_truthy()?,
            None => false,
        };
        let tracked = match p.getattr_opt(intern!(py, "trackState"))? {
            Some(state) => state.rich_compare(&two, CompareOp::Eq)?.is_truthy()?,
            None => false,
        };
        out.push(RadarPoint { track, x: values[0], y: values[1], speed: values[2], moving, tracked });
    }
    Ok(out)
}

/// `radar_points_inner` of points read from the message bytes: the same filter and values.
pub(crate) fn radar_points_raw(points: &[RawPoint]) -> Vec<RadarPoint> {
    let mut out = Vec::new();
    for p in points {
        // RADAR_SOURCES[0] is frontRadar.
        if p.source != 0 || !p.measured {
            continue;
        }
        let (x, y, speed) = (p.d_rel as f64, p.y_rel as f64, p.v_lead as f64);
        if !(x.is_finite() && y.is_finite() && speed.is_finite() && (p.v_rel as f64).is_finite()) {
            continue;
        }
        out.push(RadarPoint {
            track: p.track_id as i128,
            x,
            y,
            speed,
            moving: p.motion_state == 2,
            tracked: p.track_state == 2,
        });
    }
    out
}

/// `self.evidence.get(track)` of each point's track, as `_measurements` reads it: (radar motion class == 2,
/// reported distance, track state == 2 where the report has one).
pub(crate) fn read_evidence(m: &Bound<'_, PyAny>, points: &[RadarPoint]) -> PyResult<HashMap<i128, Evidence>> {
    let py = m.py();
    let evidence = m.getattr(intern!(py, "evidence"))?;
    let mut out = HashMap::new();
    if let Ok(d) = evidence.cast::<PyDict>() {
        if d.is_empty() {
            return Ok(out);
        }
    }
    let two = 2i32.into_pyobject(py)?;
    for p in points {
        let report = match evidence.cast::<PyDict>() {
            Ok(d) => d.get_item(p.track)?,
            Err(_) => Some(evidence.call_method1(intern!(py, "get"), (p.track,))?),
        };
        let Some(report) = report.filter(|r| !r.is_none()) else { continue };
        let tracked = if report.len()? > 2 {
            Some(report.get_item(2)?.rich_compare(&two, CompareOp::Eq)?.is_truthy()?)
        } else {
            None
        };
        out.insert(
            p.track,
            Evidence {
                moving: report.get_item(0)?.rich_compare(&two, CompareOp::Eq)?.is_truthy()?,
                distance: report.get_item(1)?.extract()?,
                tracked,
            },
        );
    }
    Ok(out)
}
