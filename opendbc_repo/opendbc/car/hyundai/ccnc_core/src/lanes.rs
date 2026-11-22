//! Read the Python lane cache for the Rust radar map.

pub use ccnc_map::radar::lanes::interpolate;
use ccnc_map::radar::lanes::{Lanes, Line};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

/// The lane-query parity test can use a map that runs the Python reference.
pub fn read(m: &Bound<'_, PyAny>) -> PyResult<Lanes<'static>> {
    let py = m.py();
    let lanes = m.getattr(intern!(py, "lanes"))?.cast_into::<PyDict>()?;
    let geometry = m.getattr(intern!(py, "geometry"))?.cast_into::<PyDict>()?;
    let mut lines: [Line; 6] = std::array::from_fn(|_| Line {
        lane: None,
        geometry: None,
    });
    for (i, line) in lines.iter_mut().enumerate() {
        if let Some(lane) = lanes.get_item(i)? {
            line.lane = Some((
                attr_f64(&lane, intern!(py, "probability"))?,
                attr_f64(&lane, intern!(py, "consistent"))?,
            ));
        }
        line.geometry = geometry
            .get_item(i)?
            .map(|points| points.extract::<Vec<_>>().map(std::borrow::Cow::Owned))
            .transpose()?;
    }
    let stamp: Option<f64> = m.getattr(intern!(py, "model_stamp"))?.extract()?;
    let fresh = match stamp {
        None => false,
        Some(stamp) => (0. ..=0.25).contains(&(attr_f64(m, intern!(py, "now"))? - stamp)),
    };
    let outer = (
        attr_f64(m, intern!(py, "OUTER_NARROW"))?,
        attr_f64(m, intern!(py, "OUTER_FAR"))?,
    );
    let changing = m.call_method0(intern!(py, "_model_changing"))?.is_truthy()?;
    let offset = m.getattr(intern!(py, "lane_offset"))?.extract()?;
    Ok(Lanes::new(lines, offset, fresh, !geometry.is_empty(), outer, changing))
}
