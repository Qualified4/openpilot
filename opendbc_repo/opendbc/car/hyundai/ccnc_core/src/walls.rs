//! Python views of the Rust wall and band state.

use std::collections::HashMap;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use ccnc_map::radar::walls::{BandData, WallData};

use crate::display_filter::empty_object;

#[derive(Default)]
pub struct Structures {
    types: Option<(Py<PyAny>, Py<PyAny>)>,
    /// Preserve the identity of a Python Band once observed, without reading it into the map.
    bands: HashMap<u64, Py<PyAny>>,
}

impl Structures {
    pub fn set_types(&mut self, wall: &Bound<'_, PyAny>, band: &Bound<'_, PyAny>) {
        if self.types.is_none() {
            self.types = Some((wall.clone().unbind(), band.clone().unbind()));
        }
    }

    pub fn walls<'py>(&self, py: Python<'py>, walls: &[WallData]) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for wall in walls {
            let wall_type = &self.types.as_ref().expect("a wall has been updated").0;
            let value = empty_object(wall_type.bind(py))?;
            value.setattr(intern!(py, "members"), PyTuple::new(py, wall.members.iter().copied())?)?;
            value.setattr(intern!(py, "points"), PyList::new(py, wall.points.iter().copied())?)?;
            result.append(value)?;
        }
        Ok(result)
    }

    pub fn bands<'py>(&mut self, py: Python<'py>, bands: &[BandData]) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for band in bands {
            let value = match self.bands.entry(band.id) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let band_type = &self.types.as_ref().expect("a band has been updated").1;
                    let value = empty_object(band_type.bind(py))?;
                    value.setattr(intern!(py, "sign"), band.sign)?;
                    value.setattr(intern!(py, "reference"), band.reference)?;
                    update_band(&value, band)?;
                    entry.insert(value.unbind())
                }
            };
            result.append(value.bind(py))?;
        }
        Ok(result)
    }

    pub fn sync_bands(&mut self, py: Python<'_>, active: &[BandData], retired: Vec<BandData>) -> PyResult<()> {
        for band in active {
            if let Some(value) = self.bands.get(&band.id) {
                update_band(value.bind(py), band)?;
            }
        }
        for band in retired {
            if let Some(value) = self.bands.remove(&band.id) {
                update_band(value.bind(py), &band)?;
            }
        }
        Ok(())
    }
}

fn update_band(value: &Bound<'_, PyAny>, band: &BandData) -> PyResult<()> {
    let py = value.py();
    let members = PyDict::new(py);
    for (key, distance) in band.members.iter() {
        members.set_item(key, *distance)?;
    }
    value.setattr(intern!(py, "offset"), band.offset)?;
    value.setattr(intern!(py, "members"), members)?;
    value.setattr(intern!(py, "supported"), band.supported)?;
    value.setattr(intern!(py, "spacing"), band.spacing)
}
