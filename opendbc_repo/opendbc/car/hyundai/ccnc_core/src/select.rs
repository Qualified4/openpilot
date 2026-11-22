use pyo3::intern;
use pyo3::prelude::*;

/// `self.selected` of the previous update.
pub(crate) fn read_selected(m: &Bound<'_, PyAny>) -> PyResult<[Option<i64>; 3]> {
    let selected = m.getattr(intern!(m.py(), "selected"))?;
    Ok([selected.get_item(0)?.extract()?, selected.get_item(1)?.extract()?, selected.get_item(2)?.extract()?])
}
