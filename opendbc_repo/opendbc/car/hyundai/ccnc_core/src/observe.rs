//! `CcncRadarMap._observe` after association (with `_reconnect`, `_stable`, `_settle`, `_crossing`,
//! `_forget_lateral`) and the object part of `_advance` (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. Each expression follows the Python one in the same order and with the
//! same grouping; `objects` and `local` stay Python dicts, read and changed in place in the Python order.

use std::collections::HashSet;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyZeroDivisionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString};

use crate::consts::Consts;
use crate::guard;
use crate::lanes::{LaneCache, Lanes};
use crate::radar_map::Row;
use crate::map_object::MapObject;
use crate::pymath::{hypot, py_max, py_min, py_sum};

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

fn attr_bool(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    obj.getattr(name)?.is_truthy()
}

/// The map's pose and its cosine/sine: `to_ego`.
#[derive(Clone, Copy)]
pub struct Frame {
    pub px: f64,
    pub py: f64,
    pub c: f64,
    pub s: f64,
}

impl Frame {
    pub fn read(m: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = m.py();
        let (px, py_, _): (f64, f64, f64) = m.getattr(intern!(py, "pose"))?.extract()?;
        Ok(Self { px, py: py_, c: attr_f64(m, intern!(py, "cos"))?, s: attr_f64(m, intern!(py, "sin"))? })
    }

    pub fn to_ego(self, x: f64, y: f64) -> (f64, f64) {
        let (c, s) = (self.c, self.s);
        (c * (x - self.px) + s * (y - self.py), -s * (x - self.px) + c * (y - self.py))
    }
}

/// `self.local[key]`.
fn local_xy(local: &Bound<'_, PyDict>, key: i64) -> PyResult<(f64, f64)> {
    local.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key))?.extract()
}

fn object<'py>(objects: &Bound<'py, PyDict>, key: i64) -> PyResult<Bound<'py, MapObject>> {
    objects.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key))?.cast_into::<MapObject>().map_err(PyErr::from)
}

/// `_forget_lateral(obj)`.
pub(crate) fn forget(o: &mut MapObject, c: f64, s: f64) {
    o.history.clear();
    let forward = c * o.vx + s * o.vy;
    o.vx = forward * c;
    o.vy = forward * s;
}

#[pyfunction]
pub fn forget_lateral(obj: &Bound<'_, MapObject>, cos: f64, sin: f64) -> PyResult<()> {
    guard(|| {
        forget(&mut *obj.try_borrow_mut()?, cos, sin);
        Ok(())
    })
}

/// `_crossing(obj)`.
pub(crate) fn crossing(o: &MapObject, frame: Frame, travel: f64) -> bool {
    let Some((ox, oy)) = o.origin else { return false };
    let (dx, dy) = (o.x - ox, o.y - oy);
    let lateral = (frame.c * dy - frame.s * dx).abs();
    lateral >= travel && lateral > (frame.c * dx + frame.s * dy).abs()
}

/// Python `sum(values) / len(values)`.
fn mean(values: &[f64]) -> PyResult<f64> {
    match py_sum(values.iter().copied()) {
        Some(total) => Ok(total / values.len() as f64),
        None => Err(PyZeroDivisionError::new_err("division by zero")),
    }
}

/// One `_measurements` tuple: (source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked).
pub(crate) struct Measurement {
    source: i128,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    explained: f64,
    hc: f64,
    hs: f64,
    radar_moving: bool,
    doppler: i64,
    tracked: i64,
}

impl Measurement {
    fn read(item: &Bound<'_, PyAny>) -> PyResult<Self> {
        let f = |i: usize| -> PyResult<f64> { item.get_item(i)?.extract() };
        Ok(Self {
            source: item.get_item(0)?.extract()?,
            x: f(1)?,
            y: f(2)?,
            vx: f(3)?,
            vy: f(4)?,
            explained: f(5)?,
            hc: f(6)?,
            hs: f(7)?,
            radar_moving: item.get_item(8)?.is_truthy()?,
            doppler: item.get_item(9)?.extract()?,
            tracked: item.get_item(10)?.is_truthy()? as i64,
        })
    }

    /// The same values from a Rust measurement row.
    pub(crate) fn from_row(row: &Row<'_>) -> PyResult<Self> {
        Ok(Self {
            source: row.track.extract()?,
            x: row.wx,
            y: row.wy,
            vx: row.vx,
            vy: row.vy,
            explained: row.explained,
            hc: row.c,
            hs: row.s,
            radar_moving: row.moving.is_truthy()?,
            doppler: row.doppler as i64,
            tracked: row.tracked.is_truthy()? as i64,
        })
    }
}

/// `_stable(obj)`.
fn stable(o: &mut MapObject, lanes: &Lanes<'_>, bridge_key: Option<i64>, k: &Consts) -> PyResult<()> {
    let (stamp, x, y) = o.report.expect("set just before");
    let centre = lanes.centre(x);
    // While bridged its reports are the merged track's: they would make the merged position look stable.
    let Some(centre) = centre.filter(|_| bridge_key != Some(o.key)) else {
        o.lateral_reports.clear();
        o.calm_since = None;
        return Ok(());
    };
    o.lateral_reports.push((stamp, y - centre));
    let earlier = o.lateral_reports.iter().find(|r| {
        let age = stamp - r.0;
        0.1 - 1e-6 <= age && age <= k.stable_window + 1e-6
    });
    let Some((t0, y0)) = earlier else {
        o.calm_since = None;
        return Ok(());
    };
    let last = o.lateral_reports.back().expect("appended above").1;
    let span = stamp - t0;
    if span == 0. {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    if py_max(0., (last - y0).abs() - k.report_step) / span >= k.stable_speed {
        o.calm_since = None;
        return Ok(());
    }
    let calm_since = *o.calm_since.get_or_insert(t0);
    if stamp - calm_since >= k.stable_time - 1e-6 {
        let recent: Vec<f64> =
            o.lateral_reports.iter().filter(|r| stamp - r.0 <= k.stable_time + 1e-6).map(|r| r.1).collect();
        o.stable_y = Some(mean(&recent)?);
    }
    Ok(())
}

/// `_settle(obj, x, y, stamp)`.
fn settle(o: &mut MapObject, lanes: &Lanes<'_>, x: f64, y: f64, stamp: f64, k: &Consts) -> PyResult<()> {
    let Some(outer) = lanes.outer(x, y)? else { return Ok(()) };
    if !o.motion_confirmed || (if y > 0. { y - outer } else { outer - y }) <= 0. {
        o.beyond = None;
        o.settled = false;
        return Ok(());
    }
    let beyond = *o.beyond.get_or_insert(stamp);
    o.settled = o.settled || (stamp - beyond > k.outer_time && o.last_motion.1.abs() < k.outer_lateral);
    Ok(())
}

/// Python `a < b` for the eviction priority tuples (near, evidence, existence, seen).
fn priority_lt(a: &(bool, bool, f64, f64), b: &(bool, bool, f64, f64)) -> bool {
    if a.0 != b.0 {
        return !a.0 & b.0;
    }
    if a.1 != b.1 {
        return !a.1 & b.1;
    }
    if a.2 != b.2 {
        return a.2 < b.2;
    }
    a.3 != b.3 && a.3 < b.3
}

/// Counter attributes of the map, written back as they change.
struct Counter<'a, 'py> {
    radar_map: &'a Bound<'py, PyAny>,
    name: &'a Bound<'py, PyString>,
    value: i64,
}

impl<'a, 'py> Counter<'a, 'py> {
    fn read(radar_map: &'a Bound<'py, PyAny>, name: &'a Bound<'py, PyString>) -> PyResult<Self> {
        Ok(Self { radar_map, name, value: radar_map.getattr(name)?.extract()? })
    }

    fn add(&mut self) -> PyResult<i64> {
        let old = self.value;
        self.value += 1;
        self.radar_map.setattr(self.name, self.value)?;
        Ok(old)
    }
}

/// `_observe` after `_associate`: `_reconnect`, then the measurement loop. Returns `measured_keys`.
#[pyfunction]
pub fn observe<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    measurements: &Bound<'py, PyAny>,
    assignments: &Bound<'py, PyDict>,
    stamp: f64,
) -> PyResult<Bound<'py, PySet>> {
    guard(|| observe_inner(py, radar_map, measurements, assignments, stamp))
}

fn observe_inner<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    measurements: &Bound<'py, PyAny>,
    assignments: &Bound<'py, PyDict>,
    stamp: f64,
) -> PyResult<Bound<'py, PySet>> {
    let m = radar_map;
    let k = Consts::read(m)?;
    let mut ms = Vec::new();
    for item in measurements.try_iter()? {
        ms.push(Measurement::read(&item?)?);
    }
    let mut assigned: Vec<Option<i64>> = vec![None; ms.len()];
    let mut reserved: Vec<i64> = Vec::with_capacity(assignments.len());
    for (i, key) in assignments.iter() {
        let key: i64 = key.extract()?;
        if let Some(slot) = usize::try_from(i.extract::<i64>()?).ok().and_then(|i| assigned.get_mut(i)) {
            *slot = Some(key);
        }
        reserved.push(key);
    }
    let state = State::read(m)?;
    let mut lanes = LaneCache::new(m);
    observe_core(py, m, &k, &state, &mut lanes, &ms, assigned, reserved, Some(assignments), stamp)
}

/// What `_observe` reads of the map's own state; it does not change during the update.
#[derive(Clone, Copy)]
pub(crate) struct State {
    pub now: f64,
    pub holding: bool,
    pub odometer: f64,
    pub a_ego: f64,
    pub frame: Frame,
}

impl State {
    pub fn read(m: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = m.py();
        Ok(Self {
            now: attr_f64(m, intern!(py, "now"))?,
            holding: attr_bool(m, intern!(py, "holding"))?,
            odometer: attr_f64(m, intern!(py, "odometer"))?,
            a_ego: attr_f64(m, intern!(py, "a_ego"))?,
            frame: Frame::read(m)?,
        })
    }
}

/// `_observe` after `_associate`: `_reconnect`, then the measurement loop. `assigned` holds the object key of each
/// measurement, `reserved` the assigned keys; a reconnection is also written to the Python `assignments` if given.
/// Returns `measured_keys`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn observe_core<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    k: &Consts,
    state: &State,
    lanes: &mut LaneCache<'_, 'py>,
    ms: &[Measurement],
    mut assigned: Vec<Option<i64>>,
    mut reserved: Vec<i64>,
    assignments: Option<&Bound<'py, PyDict>>,
    stamp: f64,
) -> PyResult<Bound<'py, PySet>> {
    let m = radar_map;
    let State { now, holding, odometer, a_ego, frame } = *state;
    let objects = m.getattr(intern!(py, "objects"))?.cast_into::<PyDict>()?;
    let local = m.getattr(intern!(py, "local"))?.cast_into::<PyDict>()?;
    let bridge = m.getattr(intern!(py, "bridge"))?;
    let bridge_key: Option<i64> =
        if bridge.is_none() { None } else { Some(bridge.get_item(intern!(py, "key"))?.extract()?) };

    // _reconnect: a bridged FF object continues with a new report near its bridged position in ego's lane.
    if let Some(key) = bridge_key.filter(|key| !reserved.contains(key)) {
        let bx: f64 = bridge.get_item(intern!(py, "x"))?.extract()?;
        let by: f64 = bridge.get_item(intern!(py, "y"))?.extract()?;
        let mut best: Option<(f64, usize)> = None;
        for (i, me) in ms.iter().enumerate() {
            if assigned[i].is_some() {
                continue;
            }
            let (ex, ey) = frame.to_ego(me.x, me.y);
            let distance = hypot(ex - bx, ey - by);
            if distance <= k.bridge_reconnect
                && lanes.get()?.lane(ex, ey, true, false, &[])?.0 == Some(0)
                && best.is_none_or(|b| distance < b.0)
            {
                best = Some((distance, i));
            }
        }
        if let Some((_, i)) = best {
            let obj = object(&objects, key)?;
            let mut o = obj.try_borrow_mut()?;
            // to_world(bx, by, self.pose): math.cos/sin of the pose angle, as Python computes them.
            let (px, py_, angle): (f64, f64, f64) = m.getattr(intern!(py, "pose"))?.extract()?;
            let math = py.import(intern!(py, "math"))?;
            let c: f64 = math.call_method1(intern!(py, "cos"), (angle,))?.extract()?;
            let s: f64 = math.call_method1(intern!(py, "sin"), (angle,))?.extract()?;
            o.x = px + c * bx - s * by;
            o.y = py_ + s * bx + c * by;
            forget(&mut o, frame.c, frame.s);
            o.vx = ms[i].vx;
            o.vy = ms[i].vy;
            if let Some(assignments) = assignments {
                assignments.set_item(i, key)?;
            }
            assigned[i] = Some(key);
            reserved.push(key);
        }
    }
    let reserved: HashSet<i64> = reserved.into_iter().collect();

    let measured_keys = PySet::empty(py)?;
    let mut measured: HashSet<i64> = HashSet::with_capacity(ms.len());
    let mut next_id = Counter::read(m, intern!(py, "next_id"))?;
    let mut created = Counter::read(m, intern!(py, "created"))?;
    let mut evicted = Counter::read(m, intern!(py, "evicted"))?;

    for (i, me) in ms.iter().enumerate() {
        let (obj, key) = match assigned[i] {
            None => {
                if objects.len() as f64 >= k.max_objects {
                    // Retain confirmed nearby vehicles before unconfirmed distant returns.
                    let mut victim: Option<((bool, bool, f64, f64), i64)> = None;
                    for other in objects.values() {
                        let o = other.cast::<MapObject>()?.try_borrow()?;
                        if measured.contains(&o.key) || reserved.contains(&o.key) || bridge_key == Some(o.key) {
                            continue;
                        }
                        let priority = (
                            local_xy(&local, o.key)?.0.abs() <= 15.,
                            o.structure || o.motion_confirmed || o.vehicle >= 0.6,
                            o.existence,
                            o.seen,
                        );
                        if victim.as_ref().is_none_or(|v| priority_lt(&priority, &v.0)) {
                            victim = Some((priority, o.key));
                        }
                    }
                    let Some((_, victim)) = victim else { continue };
                    if frame.to_ego(me.x, me.y).0.abs() > 15. && local_xy(&local, victim)?.0.abs() <= 15. {
                        continue; // A new distant return cannot evict a remembered near object.
                    }
                    objects.del_item(victim)?;
                    local.del_item(victim)?;
                    evicted.add()?;
                }
                let key = next_id.add()?;
                if me.radar_moving {
                    for other in objects.values() {
                        let mut o = other.cast::<MapObject>()?.try_borrow_mut()?;
                        let age = stamp - o.seen;
                        if o.source == me.source && 0. < age && age <= k.id_cycle {
                            o.superseded = true;
                        }
                    }
                }
                let obj = Bound::new(py, MapObject::measured(key, me.source, me.x, me.y, me.vx, me.vy, stamp))?;
                objects.set_item(key, &obj)?;
                created.add()?;
                (obj, key)
            }
            Some(key) => {
                let obj = object(&objects, key)?;
                {
                    let mut o = obj.try_borrow_mut()?;
                    let age = stamp - o.seen;
                    if 0. < age && age <= 0.15 && !o.motion_confirmed {
                        o.drift += 0.5 * (o.explained + me.explained) * (stamp - o.seen);
                    }
                    o.source = me.source;
                    o.seen = stamp;
                    o.departed = false;
                    o.superseded = false;
                    o.hits += 1;
                    o.existence = py_min(1., o.existence + 0.15);
                }
                (obj, key)
            }
        };
        let mut o = obj.try_borrow_mut()?;
        o.explained = me.explained;
        let (rx, ry) = frame.to_ego(me.x, me.y);
        o.report = Some((stamp, rx, ry));
        stable(&mut o, lanes.get()?, bridge_key, k)?;
        o.tracked += me.tracked;
        o.radar_moving = o.radar_moving || me.radar_moving;
        if me.doppler != 0 {
            // Isolated noisy samples must not add up over a long observation.
            o.doppler += 1;
            o.forward_samples += (me.doppler > 0) as i64;
            o.doppler_recent.push(stamp);
        }
        if o.doppler_recent.len() as f64 == k.doppler_samples {
            let (Some(first), Some(last)) = (o.doppler_recent.front(), o.doppler_recent.back()) else {
                return Err(PyIndexError::new_err("deque index out of range"));
            };
            if last - first <= k.doppler_window && o.tracked as f64 >= k.doppler_tracked {
                o.doppler_confirmed = true;
            }
        }
        if o.radar_moving || o.doppler_confirmed {
            o.motion_confirmed = true;
        }
        let (cx, cy) = (me.x - o.drift * me.hc, me.y - o.drift * me.hs);
        o.history.push((stamp, cx, cy, odometer));
        measured_keys.add(key)?;
        measured.insert(key);
        let first = o.history.front().expect("appended above");
        let span = stamp - first.0;
        if o.history.len() >= 4 && span >= 0.15 {
            let last = o.history.back().expect("appended above");
            let (dx, dy) = (last.1 - first.1, last.2 - first.2);
            let net = hypot(dx, dy);
            let steps: Vec<f64> =
                o.history.iter().zip(o.history.iter().skip(1)).map(|(a, b)| hypot(b.1 - a.1, b.2 - a.2)).collect();
            let path = py_sum(steps.iter().copied()).expect("four or more reports");
            let longest = steps.iter().copied().reduce(py_max).unwrap_or(0.);
            let coherent = net > 0.10 && net >= 0.8 * path && longest <= 0.7 * path;
            if k.displacement_evidence
                && coherent
                && span >= 0.4
                && net >= 0.3
                && net / span > 0.35 + k.report_delay_bound * a_ego.abs()
            {
                o.motion_confirmed = true;
            }
            o.stationary = span >= 0.3 && net <= k.resting_travel * (last.3 - first.3) + k.resting_net && !coherent;
            if o.motion_confirmed && (me.hc * me.vx + me.hs * me.vy).abs() > k.resting_speed {
                o.stationary = false; // Slower than the travel bound, but the radar reports it driving.
            }
            if o.stationary {
                o.vx = 0.;
                o.vy = 0.;
            } else if coherent {
                o.vx = dx / span;
                o.vy = dy / span;
            } else {
                let (vx, vy) = (o.vx, o.vy);
                o.vx = 0.8 * vx + 0.2 * (me.vx - me.explained * me.hc);
                o.vy = 0.8 * vy + 0.2 * (me.vy - me.explained * me.hs);
            }
        }
        // A confirmed vehicle keeps its reported position.
        let (x, y) = if o.motion_confirmed { (me.x, me.y) } else { (cx, cy) };
        let latency = now - stamp;
        let (mx, my) = (x + o.vx * latency, y + o.vy * latency);
        if o.hits == 1 {
            o.x = mx;
            o.y = my;
        } else {
            o.x = 0.4 * o.x + 0.6 * mx;
            o.y = 0.4 * o.y + 0.6 * my;
        }
        o.uncertainty = py_max(0.15, py_min(0.4, o.uncertainty * 0.6));
        let (ex, ey) = frame.to_ego(o.x, o.y);
        local.set_item(key, (ex, ey))?;
        o.last_motion = (me.hc * o.vx + me.hs * o.vy, me.hc * o.vy - me.hs * o.vx);
        let view = lanes.get()?;
        settle(&mut o, view, ex, ey, stamp, k)?;
        let (slot, interior, _) = view.lane(ex, ey, o.motion_confirmed, o.settled, &[])?;
        o.slot = slot;
        o.interior = interior;
        let side = matches!(slot, Some(1 | 2));
        // A remembered vehicle that drives off reappears just ahead of where it stood, slowly, with the
        // radar's moving class or its old radar id, and the gap then grows. It is no longer remembered.
        if o.hits == 1 {
            if holding && side && me.hc * me.vx + me.hs * me.vy <= k.release_speed {
                let mut ahead = Vec::new();
                for other in objects.values() {
                    if other.as_ptr() == obj.as_ptr() {
                        continue;
                    }
                    let n = other.cast::<MapObject>()?.try_borrow()?;
                    let candidate = n.slot == o.slot
                        && !n.departed
                        && now - n.seen > k.prediction_hold
                        && (n.vehicle >= 0.6 || n.admitted);
                    if !candidate {
                        continue;
                    }
                    let (lx, ly) = local_xy(&local, n.key)?;
                    if !(k.memory_min_x <= lx && lx <= k.memory_max_x) {
                        continue;
                    }
                    let gap = ex - lx;
                    if !(k.release_gap.0 <= gap && gap <= k.release_gap.1 && (ey - ly).abs() <= k.release_lateral) {
                        continue;
                    }
                    ahead.push((n.key, ex - lx));
                }
                o.ahead_of = ahead;
            }
        } else {
            for (remembered, gap) in o.ahead_of.clone() {
                let Some(other) = objects.get_item(remembered)? else { continue };
                if other.as_ptr() == obj.as_ptr() {
                    if (o.radar_moving || me.source == o.source)
                        && ex - local_xy(&local, remembered)?.0 >= gap + k.release_advance
                    {
                        o.departed = true;
                    }
                    continue;
                }
                let mut n = other.cast::<MapObject>()?.try_borrow_mut()?;
                if (o.radar_moving || me.source == n.source)
                    && ex - local_xy(&local, remembered)?.0 >= gap + k.release_advance
                {
                    n.departed = true;
                }
            }
        }
        // While ego is stopped or stopping, a return that has rested for .5s near the middle of an adjacent lane
        // is a waiting vehicle, even if it never moved in view.
        if k.waiting_admission
            && holding
            && o.stationary
            && side
            && !o.structure
            && 0. < ex
            && ex <= k.memory_max_x
            && stamp - o.born >= 0.5
            && o.hits >= 7
            && o.tracked as f64 >= k.waiting_tracked
            && view.body_clear(slot.unwrap_or(0), ex, ey, k.body_clearance, k.body_far_side)
            && !crossing(&o, frame, k.crossing_travel)
            && !view.band_line(ex, ey, odometer, k.band_bin, k.band_memory)?
        {
            o.admitted = true;
        }
        if o.motion_confirmed {
            o.vehicle = py_max(0.8, o.vehicle);
            let (x, y) = (o.x, o.y);
            if o.trail.back().is_none_or(|(tx, ty)| hypot(x - tx, y - ty) >= k.trail_step) {
                o.trail.push((x, y));
            }
        } else if o.interior {
            // Lane occupancy alone cannot distinguish a stopped vehicle from an obstacle.
            o.vehicle = if o.vehicle < 0.6 { py_min(0.45, o.vehicle + 0.12) } else { o.vehicle };
        } else if o.stationary && view.has_geometry {
            o.vehicle = py_max(0., o.vehicle - 0.3);
        }
    }
    Ok(measured_keys)
}

/// The object part of `_advance`: move each object, rebuild `local`, drop the ones out of range.
#[pyfunction]
pub fn advance(py: Python<'_>, radar_map: &Bound<'_, PyAny>, dt: f64, distance: f64) -> PyResult<()> {
    guard(|| advance_inner(py, radar_map, dt, distance))
}

fn advance_inner(py: Python<'_>, radar_map: &Bound<'_, PyAny>, dt: f64, distance: f64) -> PyResult<()> {
    let m = radar_map;
    let k = |name: &Bound<'_, PyString>| attr_f64(m, name);
    let (min_x, keep_x, max_y) = (k(intern!(py, "MIN_X"))?, k(intern!(py, "KEEP_X"))?, k(intern!(py, "MAX_Y"))?);
    let (far_x, far_y) = (k(intern!(py, "FAR_X"))?, k(intern!(py, "FAR_Y"))?);
    let now = k(intern!(py, "now"))?;
    let holding = attr_bool(m, intern!(py, "holding"))?;
    let frame = Frame::read(m)?;
    let bridge = m.getattr(intern!(py, "bridge"))?;
    let bridged: Option<i64> =
        if bridge.is_truthy()? { Some(bridge.get_item(intern!(py, "key"))?.extract()?) } else { None };
    let objects = m.getattr(intern!(py, "objects"))?.cast_into::<PyDict>()?;
    let local = PyDict::new(py);
    let mut expired = Vec::new();
    for (key, obj) in objects.iter() {
        let mut o = obj.cast::<MapObject>()?.try_borrow_mut()?;
        if !holding {
            o.admitted = false;
        }
        if holding && now - o.seen > 0.3 {
            // An unseen neighbor slowing with ego is not extrapolated at its last speed.
            o.vx = 0.;
            o.vy = 0.;
        }
        o.x += o.vx * dt;
        o.y += o.vy * dt;
        // A stationary remembered vehicle can still depart outside radar coverage.
        o.uncertainty += (if o.stationary { 0.08 } else { 0.35 + 0.3 * (now - o.seen) }) * dt + 0.002 * distance.abs();
        o.vision = false;
        let (x, y) = frame.to_ego(o.x, o.y);
        local.set_item(&key, (x, y))?;
        let in_range = min_x <= x && x <= keep_x && y.abs() <= max_y;
        // A far object exists only while measured with vehicle evidence; it is never remembered.
        let far_vehicle = keep_x < x
            && x <= far_x + 2.
            && y.abs() <= far_y
            && now - o.seen <= 1.
            && (o.motion_confirmed || o.vehicle >= 0.6);
        if bridged != Some(key.extract()?) && !in_range && !far_vehicle {
            expired.push(key);
        }
    }
    m.setattr(intern!(py, "local"), &local)?;
    for key in &expired {
        objects.del_item(key)?;
        local.del_item(key)?;
    }
    let removed: i64 = m.getattr(intern!(py, "range_removed"))?.extract()?;
    m.setattr(intern!(py, "range_removed"), removed + expired.len() as i64)?;
    Ok(())
}
