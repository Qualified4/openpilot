//! The map part of `CcncRadarMap.update` in one call: `_update_lane_reference`, `_observe` (`_measurements`,
//! `_associate`, `_reconnect` and the measurement loop), `_vision`, `_walls` (with `_bands`) and `_select`.
//!
//! Python runs the input checks, `_advance` (its own Rust call), the Cython lane update and builds the model's values
//! as plain lists before this call. Constants, the pose frame and the lane snapshot are read once here; the stages
//! hand their intermediate values to each other in Rust. The map's state stays in its Python attributes.

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PySet};

use crate::consts::Consts;
use crate::guard;
use crate::lanes::LaneCache;
use crate::observe::{observe_core, Measurement, State};
use crate::radar_map::{associate_rows, measure, pair, radar_points_inner, Ego};
use crate::select::{lane_reference, read_selected, vision, Selector};
use crate::walls::{bands_core, walls_core};

static CALLS: AtomicU64 = AtomicU64::new(0);
static ROUND: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// For the parity test only: how many one-call updates this process ran.
#[pyfunction]
pub fn _update_calls_for_test() -> u64 {
    CALLS.load(Ordering::Relaxed)
}

/// The radar input of an update that observes: `live.points` and `pose_at(stamp)` as (px, py, cos, sin), or None
/// where `_measurements` finds no pose.
type Radar<'py> = Option<(Bound<'py, PyAny>, (f64, f64, f64, f64))>;

/// `update()` after `self.usable` and the model's lists are set, up to `_select`.
///
/// `observe` says whether this update observes the radar at `stamp` (Python has set `raw` and `radar_stamp`), and
/// `radar` holds its points and pose. `leads` is `self.leads`, `vision_leads` `self.vision_leads`, `vision` the
/// (x, y) of each model lead `_vision` takes (none without a fresh model), `changing` `self._model_changing()`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    observe: bool,
    radar: Radar<'py>,
    stamp: &Bound<'py, PyAny>,
    leads: Vec<(f64, f64)>,
    vision_leads: Vec<(f64, f64, Option<f64>)>,
    vision_input: Vec<(f64, f64)>,
    changing: bool,
    select: bool,
    wall_type: &Bound<'py, PyAny>,
    band_type: &Bound<'py, PyAny>,
) -> PyResult<()> {
    guard(|| {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let m = radar_map;
        let k = Consts::read(m)?;
        let state = State::read(m)?;
        let now_obj = m.getattr(intern!(py, "now"))?;
        let previous = read_selected(m)?;
        let mut lanes = LaneCache::preset(m, k.outer_narrow, k.outer_far, changing);
        lane_reference(m, &k, lanes.get()?, state.now, &now_obj, previous)?;

        let measured = if observe {
            let stamp_f: f64 = stamp.extract()?;
            let rows = match radar {
                Some((points, pose)) => {
                    let points = radar_points_inner(py, &points)?;
                    measure(py, m, &k, Ego::read(m)?, &mut Some(leads), &points, stamp, pose)?
                }
                None => Vec::new(),
            };
            let input = associate_rows(py, m, &rows, stamp_f, state.now, state.frame)?;
            let pairs = pair(py, m, &k, &mut lanes, Some(state.frame), &input, stamp_f, state.holding)?;
            let ms = rows.iter().map(Measurement::from_row).collect::<PyResult<Vec<_>>>()?;
            let mut assigned = vec![None; ms.len()];
            let mut reserved = Vec::with_capacity(pairs.len());
            for (i, index) in pairs {
                assigned[i] = Some(input.key(index));
                reserved.push(input.key(index));
            }
            observe_core(py, m, &k, &state, &mut lanes, &ms, assigned, reserved, None, stamp_f)?
        } else {
            PySet::empty(py)?
        };

        let objects = m.getattr(intern!(py, "objects"))?.cast_into::<PyDict>()?;
        let local = m.getattr(intern!(py, "local"))?.cast_into::<PyDict>()?;
        vision(&objects, &local, state.now, &vision_input)?;

        let measured = measured.into_any();
        let (rows, bends, mut walls) = walls_core(py, m, &k, state.now, state.frame, &measured, wall_type)?;
        let round = ROUND.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("builtins")?.getattr("round")?.unbind()) })?;
        let round = round.bind(py);
        let rounded = PyList::empty(py);
        for (key, angle, gap) in bends {
            rounded.append((key, round.call1((angle, 1))?, round.call1((gap, 2))?))?;
        }
        m.setattr(intern!(py, "bends"), rounded)?;
        let view = lanes.get()?;
        bands_core(py, m, &k, state.odometer, &*view, &rows, &measured, band_type, Some(&mut walls))?;

        if select {
            let radar_stamp = m.getattr(intern!(py, "radar_stamp"))?;
            let mut selector = Selector {
                m,
                k: &k,
                lanes: view,
                frame: state.frame,
                walls: &walls,
                now: state.now,
                now_obj: &now_obj,
                vision_leads: &vision_leads,
                objects,
                local,
                previous,
                holding: state.holding,
                lane_change: m.getattr(intern!(py, "lane_change"))?.is_truthy()?,
                usable: m.getattr(intern!(py, "usable"))?.is_truthy()?,
                radar_stamp: if radar_stamp.is_none() { None } else { Some(radar_stamp.extract()?) },
                jumps: m.getattr(intern!(py, "jumps"))?.cast_into::<PyDict>()?,
            };
            selector.select()?;
        }
        Ok(())
    })
}
