//! The extension side of `_advance` and `_observe`: the stages run on the map's Rust state
//! (`ccnc_map::radar::observe`); this reads what they need of the Python map object.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

pub use ccnc_map::radar::observe::crossing;
pub use ccnc_map::radar::Frame;

use crate::map_state::attached;

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

/// The map's pose and its cosine/sine: from its Rust state if it has one (which must not be borrowed), otherwise
/// from its attributes (the Python reference map, or a stand-in with the same fields).
pub fn read_frame(m: &Bound<'_, PyAny>) -> PyResult<Frame> {
    if let Some(state) = attached(m)? {
        return Ok(state.try_borrow()?.st.frame());
    }
    let py = m.py();
    let (px, py_, _): (f64, f64, f64) = m.getattr(intern!(py, "pose"))?.extract()?;
    Ok(Frame { px, py: py_, c: attr_f64(m, intern!(py, "cos"))?, s: attr_f64(m, intern!(py, "sin"))? })
}
