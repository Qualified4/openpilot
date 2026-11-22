//! Read the Python lane cache for the Rust radar map.

pub use ccnc_map::radar::lanes::interpolate;
use ccnc_map::radar::lanes::{Lanes, Line};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The snapshot, made on first use.
pub struct LaneCache<'a, 'py> {
    radar_map: &'a Bound<'py, PyAny>,
    lanes: Option<Lanes>,
}

impl<'a, 'py> LaneCache<'a, 'py> {
    pub fn new(radar_map: &'a Bound<'py, PyAny>) -> Self {
        Self { radar_map, lanes: None }
    }

    pub fn get(&mut self) -> PyResult<&mut Lanes> {
        if self.lanes.is_none() {
            self.lanes = Some(read(self.radar_map, None, None)?);
        }
        Ok(self.lanes.as_mut().unwrap())
    }
}

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

pub fn read(m: &Bound<'_, PyAny>, preset: Option<(f64, f64, bool)>, now: Option<f64>) -> PyResult<Lanes> {
    let py = m.py();
    let lanes = m.getattr(intern!(py, "lanes"))?;
    let lanes = lanes.cast::<PyDict>()?;
    let geometry = m.getattr(intern!(py, "geometry"))?;
    let geometry = geometry.cast::<PyDict>()?;
    let mut lines = Vec::with_capacity(6);
    for i in 0..6 {
        let lane = match lanes.get_item(i)? {
            Some(lane) => Some((
                attr_f64(&lane, intern!(py, "probability"))?,
                attr_f64(&lane, intern!(py, "consistent"))?,
            )),
            None => None,
        };
        let geometry = match geometry.get_item(i)? {
            Some(points) => Some(points.extract()?),
            None => None,
        };
        lines.push(Line { lane, geometry });
    }
    let model_stamp = m.getattr(intern!(py, "model_stamp"))?;
    let model_fresh = if model_stamp.is_none() {
        false
    } else {
        let now = match now {
            Some(now) => now,
            None => match crate::map_state::attached(m)? {
                Some(state) => state
                    .try_borrow()?
                    .st
                    .now
                    .ok_or_else(|| PyValueError::new_err("map not advanced"))?,
                None => attr_f64(m, intern!(py, "now"))?,
            },
        };
        let age = now - model_stamp.extract::<f64>()?;
        (0. ..=0.25).contains(&age)
    };
    let (outer_narrow, outer_far, changing) = match preset {
        Some((narrow, far, changing)) => (narrow, far, changing),
        None => (
            attr_f64(m, intern!(py, "OUTER_NARROW"))?,
            attr_f64(m, intern!(py, "OUTER_FAR"))?,
            false,
        ),
    };
    let lane_offset = if preset.is_some() {
        0
    } else {
        m.getattr(intern!(py, "lane_offset"))?.extract()?
    };
    Ok(Lanes::new(
        lines,
        lane_offset,
        model_fresh,
        !geometry.is_empty(),
        (outer_narrow, outer_far),
        changing,
    ))
}
