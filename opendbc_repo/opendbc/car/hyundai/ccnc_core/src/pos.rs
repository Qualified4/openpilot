//! The redesigned FF/LF/RF position display, chosen by the internal `CcncPosDisplayMode` (DESIGN_KO.md 6.8):
//! 0 keeps the existing path (this module does not run), 1 draws display A, 2 display B. Both sit on the common part
//! in `pos/core.rs`; the existing selector still decides which map objects qualify, and the rear slots keep their
//! own path.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::guard;
use crate::lanes::LaneCache;
use crate::map_object::MapObject;
use crate::select::approach_only;

mod common;
mod display_a;
mod display_b;

use self::common::{Common, FrameIn, Obj, Vid};
use self::display_a::DisplayA;
use self::display_b::DisplayB;

/// (vid, x, y, seen, report, lines, side_ok) of an `Obj`, as the trace and the test entry carry it.
type ObjTuple = (i64, f64, f64, f64, Option<(f64, f64, f64)>, [Option<f64>; 4], bool);
type FrameTuple = (f64, Option<f64>, [Option<i64>; 3], Option<(f64, f64)>, Vec<ObjTuple>, Option<f64>);
type OutTuple = ([Option<i64>; 3], [Option<f64>; 3]);

fn frame_from(t: FrameTuple) -> FrameIn {
    let (now, yaw, selected, lead, objs, heading) = t;
    let objs = objs
        .into_iter()
        .map(|(vid, x, y, seen, report, lines, side_ok)| Obj { vid, x, y, seen, report, lines, side_ok })
        .collect();
    FrameIn { now, yaw, selected, lead, objs, heading }
}

fn frame_tuple(f: &FrameIn) -> FrameTuple {
    let objs = f.objs.iter().map(|o| (o.vid, o.x, o.y, o.seen, o.report, o.lines, o.side_ok)).collect();
    (f.now, f.yaw, f.selected, f.lead, objs, f.heading)
}

/// The display's memory across updates, kept on the extension state as `_rust_pos`.
#[pyclass(module = "ccnc_core", name = "PosState")]
#[derive(Default)]
pub struct PosState {
    common: Common,
    a: DisplayA,
    b: DisplayB,
    mode: i64,
}

impl PosState {
    fn run(&mut self, mode: i64, f: &FrameIn) -> OutTuple {
        if mode != self.mode {
            *self = PosState { mode, ..Default::default() };
        }
        let out = self.common.update(f);
        let drawn = if mode == 2 {
            self.b.draw(&self.common, &out, f.now)
        } else {
            self.a.draw(&self.common, &out, f)
        };
        (out.assigned, drawn)
    }
}

#[pymethods]
impl PosState {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

/// For the parity test only: the frames through a fresh state in `mode`, each frame's (assigned, drawn).
#[pyfunction]
pub fn _pos_run_for_test(mode: i64, frames: Vec<FrameTuple>) -> PyResult<Vec<OutTuple>> {
    guard(|| {
        let mut state = PosState::default();
        Ok(frames.into_iter().map(|t| state.run(mode, &frame_from(t))).collect())
    })
}

fn yaw_rate(cs: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let py = cs.py();
    let Some(pose) = cs.getattr_opt(intern!(py, "livePose"))?.filter(|p| !p.is_none()) else {
        return Ok(None);
    };
    let rate = pose.getattr(intern!(py, "angularVelocityDevice"))?;
    if !rate.getattr(intern!(py, "valid"))?.is_truthy()? {
        return Ok(None);
    }
    let z: f64 = rate.getattr(intern!(py, "z"))?.extract()?;
    Ok(z.is_finite().then_some(-z))
}

fn lead_one(cs: &Bound<'_, PyAny>) -> PyResult<Option<(f64, f64)>> {
    let py = cs.py();
    let Some(radar) = cs.getattr_opt(intern!(py, "radarState"))?.filter(|r| !r.is_none()) else {
        return Ok(None);
    };
    let lead = radar.getattr(intern!(py, "leadOne"))?;
    if !lead.getattr(intern!(py, "status"))?.is_truthy()? {
        return Ok(None);
    }
    let (d, y): (f64, f64) = (lead.getattr(intern!(py, "dRel"))?.extract()?, lead.getattr(intern!(py, "yRel"))?.extract()?);
    Ok((d.is_finite() && y.is_finite()).then_some((d, y)))
}

/// The mode the extension state asks for; 0 without one.
pub fn mode(state: &Bound<'_, PyAny>) -> PyResult<i64> {
    match state.getattr_opt(intern!(state.py(), "pos_display_mode"))? {
        Some(v) if !v.is_none() => v.extract(),
        _ => Ok(0),
    }
}

fn trace_list<'py>(state: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    match state.getattr_opt(intern!(state.py(), "_pos_trace"))?.filter(|v| !v.is_none()) {
        Some(v) => Ok(Some(v.cast_into::<PyList>()?)),
        None => Ok(None),
    }
}

/// This update's input from the map: the selected objects and those the state still follows. None after the map
/// reset this update (no clock yet); the map then shows nothing.
fn build_frame(map: &Bound<'_, PyAny>, cs: &Bound<'_, PyAny>, followed: &[Vid]) -> PyResult<Option<FrameIn>> {
    let py = map.py();
    let Some(now): Option<f64> = map.getattr(intern!(py, "now"))?.extract()? else { return Ok(None) };
    let selected: [Option<i64>; 3] = map.getattr(intern!(py, "selected"))?.extract()?;
    let objects = map.getattr(intern!(py, "objects"))?.cast_into::<PyDict>()?;
    let local = map.getattr(intern!(py, "local"))?.cast_into::<PyDict>()?;
    let usable = map.getattr(intern!(py, "usable"))?.is_truthy()?;
    let mut keys: Vec<Vid> = Vec::new();
    if usable {
        keys.extend(selected.iter().flatten());
        keys.extend(followed);
    }
    keys.sort_unstable();
    keys.dedup();
    let mut cache = LaneCache::new(map);
    let lanes = cache.get()?;
    let offset = lanes.lane_offset();
    let bridge = map.getattr(intern!(py, "bridge"))?;
    let bridged: Option<i64> =
        if bridge.is_truthy()? { Some(bridge.get_item(intern!(py, "key"))?.extract()?) } else { None };
    let mut objs = Vec::with_capacity(keys.len());
    for vid in keys {
        let (Some(object), Some(position)) = (objects.get_item(vid)?, local.get_item(vid)?) else { continue };
        let (x, y): (f64, f64) = position.extract()?;
        let o = object.cast::<MapObject>()?.try_borrow()?;
        let lines = std::array::from_fn(|i| lanes.boundary(i as i64 + offset, x));
        // The selector's own side-slot rules: no opposing or approach-only object, and its bridged FF stays FF.
        let side_ok = !(o.last_motion.0 < -1. / 3.6 || approach_only(&o) || bridged == Some(vid));
        objs.push(Obj { vid, x, y, seen: o.seen, report: o.report, lines, side_ok });
    }
    let heading = road_heading(lanes, offset);
    Ok(Some(FrameIn { now, yaw: yaw_rate(cs)?, selected: if usable { selected } else { [None; 3] }, lead: lead_one(cs)?, objs, heading }))
}

/// The road's heading in the ego frame (left +): the slope at x = 0 of a quadratic fit to the midline of ego's
/// inner lines at 0-30 m, where the lane is 1.8-5.5 m wide; None with fewer than four such points.
fn road_heading(lanes: &crate::lanes::Lanes<'_>, offset: i64) -> Option<f64> {
    let mut s = [0f64; 5];
    let mut t = [0f64; 3];
    let mut n = 0;
    for x in [0., 5., 10., 15., 20., 30.] {
        let (Some(a), Some(b)) = (lanes.boundary(offset + 1, x), lanes.boundary(offset + 2, x)) else { continue };
        if !(1.8..=5.5).contains(&(a - b)) {
            continue;
        }
        let y = 0.5 * (a + b);
        let mut xp = 1.;
        for k in 0..5 {
            s[k] += xp;
            if k < 3 {
                t[k] += xp * y;
            }
            xp *= x;
        }
        n += 1;
    }
    if n < 4 {
        return None;
    }
    let m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-9 {
        return None;
    }
    let mut q = m;
    for r in 0..3 {
        q[r][1] = t[r];
    }
    Some((det(q) / d).atan())
}

/// Mode 0 with a `_pos_trace` list (replay only): record the input the redesigned display would get, nothing else.
pub fn shadow(map: &Bound<'_, PyAny>, cs: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Some(trace) = trace_list(state)? {
        let Some(frame) = build_frame(map, cs, &[])? else { return Ok(()) };
        let offset: i64 = map.getattr(intern!(map.py(), "lane_offset"))?.extract()?;
        trace.append((0, frame_tuple(&frame), (frame.selected, [None::<f64>; 3]), offset))?;
    }
    Ok(())
}

/// Mode 1 or 2 after the map has selected: the FF/LF/RF keys to draw and their lateral (ego frame, left +).
/// The state's `_pos_trace` list, when set (replay), gets each update's mode, input, output and map lane offset.
pub fn assign_and_draw(
    map: &Bound<'_, PyAny>,
    cs: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    mode: i64,
) -> PyResult<OutTuple> {
    let py = map.py();
    let attr = intern!(py, "_rust_pos");
    let holder = match state.getattr_opt(attr)?.filter(|v| !v.is_none()) {
        Some(v) => v.cast_into::<PosState>()?,
        None => {
            let v = Bound::new(py, PosState::default())?;
            state.setattr(attr, &v)?;
            v
        }
    };
    let mut pos = holder.try_borrow_mut()?;
    let mut followed: Vec<Vid> = pos.common.tracks.keys().copied().collect();
    followed.extend(pos.common.slots.iter().filter_map(|s| s.owner));
    let Some(frame) = build_frame(map, cs, &followed)? else { return Ok(([None; 3], [None; 3])) };
    let out = pos.run(mode, &frame);
    if let Some(trace) = trace_list(state)? {
        let offset: i64 = map.getattr(intern!(py, "lane_offset"))?.extract()?;
        trace.append((mode, frame_tuple(&frame), out, offset))?;
    }
    Ok(out)
}
