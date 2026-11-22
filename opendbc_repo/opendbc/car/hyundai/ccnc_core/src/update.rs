//! The map part of `CcncRadarMap.update` in one call: `_update_lane_reference`, `_observe` (`_measurements`,
//! `_associate`, `_reconnect` and the measurement loop), `_vision`, `_walls` (with `_bands`) and `_select`.
//!
//! Python (or the display's Rust caller) runs the input checks, `_advance` on the map's Rust state, the lane update
//! and builds the model's values as plain lists before this call. Constants and the lane snapshot are read once
//! here. Ego's pose, the objects and the radar id history are the map's Rust state (`map_state`); the rest of the
//! map stays in its Python attributes.

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFloat, PyList, PySet};

use ccnc_map::input::live_tracks::parse;
use ccnc_map::radar::associate::{associate, measure, AssociateIn, Ego};
use ccnc_map::radar::observe::{observe as observe_stage, ObserveIn};

use crate::guard;
use crate::lanes::LaneCache;
use crate::map_state::{attach, py_err};
use crate::observe::State;
use crate::radar_map::{radar_points_inner, radar_points_raw, read_evidence};
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

/// `self.bridge` as (key, x, y).
fn read_bridge(m: &Bound<'_, PyAny>) -> PyResult<Option<(i64, f64, f64)>> {
    let py = m.py();
    let bridge = m.getattr(intern!(py, "bridge"))?;
    if bridge.is_none() {
        return Ok(None);
    }
    Ok(Some((
        bridge.get_item(intern!(py, "key"))?.extract()?,
        bridge.get_item(intern!(py, "x"))?.extract()?,
        bridge.get_item(intern!(py, "y"))?.extract()?,
    )))
}

/// `update()` after `self.usable`, `_advance` and the model's lists are set, up to `_select`.
///
/// `observe` says whether this update observes the radar at `stamp` (the caller has set `raw` and `radar_stamp`), and
/// `radar` holds its points and pose. `live_bytes`, the bytes of the liveTracks message those points are from, gives
/// the points instead where it reads (`live_tracks::parse`). `leads` is `self.leads`, `vision_leads` `self.vision_leads`, `vision` the
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
    live_bytes: Option<&[u8]>,
) -> PyResult<()> {
    guard(|| {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let m = radar_map;
        let k = crate::consts::read(m)?;
        let state_obj = attach(m)?;
        let mut borrowed = state_obj.try_borrow_mut()?;
        let st = &mut borrowed.st;
        let state = State::read(m, st)?;
        let now_obj = PyFloat::new(py, state.now).into_any();
        let previous = read_selected(m)?;
        let mut lanes = LaneCache::preset(m, k.outer_narrow, k.outer_far, changing, state.now);
        lane_reference(m, &k, lanes.get()?, state.now, &now_obj, previous)?;

        let measured = if observe {
            let stamp_f: f64 = stamp.extract()?;
            let ms = match radar {
                Some((points, pose)) => {
                    let points = match live_bytes.and_then(parse) {
                        Some(parsed) => radar_points_raw(&parsed),
                        None => radar_points_inner(py, &points)?,
                    };
                    let evidence = read_evidence(m, &points)?;
                    let ego = Ego {
                        a_ego: state.a_ego,
                        yaw_rate: m.getattr(intern!(py, "yaw_rate"))?.extract()?,
                        speed: st.speed,
                    };
                    measure(st, &k, ego, &leads, &|track| evidence.get(&track).copied(), &points, stamp_f, pose)
                }
                None => Vec::new(),
            };
            let inp = AssociateIn {
                holding: state.holding,
                lane_change: m.getattr(intern!(py, "lane_change"))?.is_truthy()?,
                selected: previous,
            };
            let pairs = associate(st, &k, inp, &mut lanes, &ms, stamp_f).map_err(py_err)?;
            let mut assigned = vec![None; ms.len()];
            let mut reserved = Vec::with_capacity(pairs.len());
            for (i, key) in pairs {
                assigned[i] = Some(key);
                reserved.push(key);
            }
            let inp = ObserveIn { holding: state.holding, a_ego: state.a_ego, bridge: read_bridge(m)? };
            let keys = observe_stage(st, &k, inp, &mut lanes, &ms, assigned, reserved, stamp_f).map_err(py_err)?;
            let measured = PySet::empty(py)?;
            for key in keys {
                measured.add(key)?;
            }
            measured
        } else {
            PySet::empty(py)?
        };

        vision(st, state.now, &vision_input)?;

        let measured = measured.into_any();
        let (rows, bends, mut walls) = walls_core(py, m, st, &k, state.now, state.frame, &measured, wall_type)?;
        let round = ROUND.get_or_try_init(py, || -> PyResult<_> { Ok(py.import("builtins")?.getattr("round")?.unbind()) })?;
        let round = round.bind(py);
        let rounded = PyList::empty(py);
        for (key, angle, gap) in bends {
            rounded.append((key, round.call1((angle, 1))?, round.call1((gap, 2))?))?;
        }
        m.setattr(intern!(py, "bends"), rounded)?;
        let view = lanes.get()?;
        bands_core(py, m, st, &k, state.odometer, &*view, &rows, &measured, band_type, Some(&mut walls))?;

        if select {
            let radar_stamp = m.getattr(intern!(py, "radar_stamp"))?;
            let jumps = st.jumps.iter().map(|(t, s)| (t, *s)).collect();
            let mut selector = Selector {
                m,
                st,
                k: &k,
                lanes: view,
                frame: state.frame,
                walls: &walls,
                now: state.now,
                now_obj: &now_obj,
                vision_leads: &vision_leads,
                previous,
                holding: state.holding,
                lane_change: m.getattr(intern!(py, "lane_change"))?.is_truthy()?,
                usable: m.getattr(intern!(py, "usable"))?.is_truthy()?,
                radar_stamp: if radar_stamp.is_none() { None } else { Some(radar_stamp.extract()?) },
                jumps,
            };
            selector.select()?;
        }
        Ok(())
    })
}
