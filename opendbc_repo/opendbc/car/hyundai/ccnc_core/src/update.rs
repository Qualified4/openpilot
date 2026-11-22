//! Inputs for the pure Rust radar-map update.

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::intern;
use pyo3::prelude::*;

use ccnc_map::input::live_tracks::parse;
use ccnc_map::radar::state::{Observation, UpdateIn};

use crate::guard;
use crate::lanes;
use crate::map_state::{attach, py_err};
use crate::radar_map::{radar_points_inner, radar_points_raw, read_evidence};
static CALLS: AtomicU64 = AtomicU64::new(0);

/// For the parity test only: how many one-call updates this process ran.
#[pyfunction]
pub fn _update_calls_for_test() -> u64 {
    CALLS.load(Ordering::Relaxed)
}

/// The radar input of an update that observes: `live.points` and `pose_at(stamp)` as (px, py, cos, sin), or None
/// where `_measurements` finds no pose.
type Radar<'py> = Option<(Bound<'py, PyAny>, (f64, f64, f64, f64))>;

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
        let state = attach(m)?;
        let mut state = state.try_borrow_mut()?;
        let crate::map_state::RadarMapState { st, structures } = &mut *state;
        let mut lanes = lanes::read(m, Some((k.outer_narrow, k.outer_far, changing)), st.now)?;
        let (points, pose) = match radar {
            Some((points, pose)) if observe => {
                let points = match live_bytes.and_then(parse) {
                    Some(parsed) => radar_points_raw(&parsed),
                    None => radar_points_inner(py, &points)?,
                };
                (points, Some(pose))
            }
            _ => (Vec::new(), None),
        };
        let evidence = if pose.is_some() {
            read_evidence(m, &points)?
        } else {
            Default::default()
        };
        let blindspot = m.getattr(intern!(py, "blindspot"))?;
        let radar_stamp = m.getattr(intern!(py, "radar_stamp"))?.extract()?;
        let inp = UpdateIn {
            radar: if observe {
                Some(Observation {
                    points: &points,
                    stamp: stamp.extract()?,
                    pose,
                    evidence: &evidence,
                })
            } else {
                None
            },
            radar_stamp,
            a_ego: m.getattr(intern!(py, "a_ego"))?.extract()?,
            yaw_rate: m.getattr(intern!(py, "yaw_rate"))?.extract()?,
            holding: m.getattr(intern!(py, "holding"))?.is_truthy()?,
            ego_change: m.getattr(intern!(py, "ego_change"))?.is_truthy()?,
            lane_change: m.getattr(intern!(py, "lane_change"))?.is_truthy()?,
            blindspot: [blindspot.get_item(0)?.is_truthy()?, blindspot.get_item(1)?.is_truthy()?],
            leads: &leads,
            vision_leads: &vision_leads,
            vision: &vision_input,
            select,
            usable: m.getattr(intern!(py, "usable"))?.is_truthy()?,
        };
        let result = st.update(&k, &mut lanes, inp).map_err(py_err)?;
        m.setattr(intern!(py, "lane_change"), result.lane_change)?;
        structures.set_types(wall_type, band_type);
        structures.sync_bands(py, &st.bands, result.retired_bands)
    })
}
