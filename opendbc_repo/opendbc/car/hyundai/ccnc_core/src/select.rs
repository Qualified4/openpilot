//! `CcncRadarMap._update_lane_reference`, `_vision` and `_select` with everything `_select` calls (`_stop_memory`,
//! `_behind_lead`, `_trail`, `_wall_outside`, `_far_hold`, `_bridge`, `_bridged`, `_approach_only`, `_ego_lines`,
//! `_centre`) for the one-call update (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. The map's state stays in its Python attributes: each value is read where
//! the Python code reads it and written where the Python code writes it, so that the next stage (and the next update)
//! sees what the Python path would leave. Tuple comparisons in `min()` follow Python: the first unequal element
//! decides with `<`, and `min()` keeps the first of equal items.

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::consts::Consts;
use crate::lanes::{interpolate, Lanes};
use crate::map_object::MapObject;
use crate::observe::{crossing, forget, Frame};
use crate::pymath::{hypot, py_max};

type Xy = (f64, f64);

/// `self.local[key]`.
fn local_xy(local: &Bound<'_, PyDict>, key: i64) -> PyResult<Xy> {
    local.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key))?.extract()
}

fn object<'py>(objects: &Bound<'py, PyDict>, key: i64) -> PyResult<Bound<'py, MapObject>> {
    objects.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key))?.cast_into::<MapObject>().map_err(PyErr::from)
}

/// `self.selected` of the previous update.
pub(crate) fn read_selected(m: &Bound<'_, PyAny>) -> PyResult<[Option<i64>; 3]> {
    let selected = m.getattr(intern!(m.py(), "selected"))?;
    Ok([selected.get_item(0)?.extract()?, selected.get_item(1)?.extract()?, selected.get_item(2)?.extract()?])
}

/// `_update_lane_reference()`. `lanes` is the update's snapshot; its lane offset follows the one written here.
pub(crate) fn lane_reference(
    m: &Bound<'_, PyAny>,
    k: &Consts,
    lanes: &mut Lanes<'_>,
    now: f64,
    now_obj: &Bound<'_, PyAny>,
    selected: [Option<i64>; 3],
) -> PyResult<()> {
    let py = m.py();
    let model_stamp = m.getattr(intern!(py, "model_stamp"))?;
    let fresh = !model_stamp.is_none() && {
        let age = now - model_stamp.extract::<f64>()?;
        (0. ..=0.25).contains(&age)
    };
    if !fresh {
        m.setattr(intern!(py, "lane_offset"), 0)?;
        lanes.set_lane_offset(0);
        return Ok(());
    }
    // Require an empty FF and a displayed destination-side car before starting, unless ego is changing lanes.
    let free = m.getattr(intern!(py, "ego_change"))?.is_truthy()?;
    let x = if free { 5. } else { 10. };
    let (left, right) = (lanes.boundary(1, x), lanes.boundary(2, x));
    let mut offset: i64 = m.getattr(intern!(py, "lane_offset"))?.extract()?;
    let back = if free { 1. } else { 0.1 };
    let returned =
        (offset == -1 && left.is_some_and(|l| l >= 0.5)) || (offset == 1 && right.is_some_and(|r| r <= -0.5));
    let mut lane_return = m.getattr(intern!(py, "lane_return"))?;
    if !returned {
        lane_return = py.None().into_bound(py);
    } else if lane_return.is_none() {
        lane_return = now_obj.clone();
    }
    let cancelled = returned && now - lane_return.extract::<f64>()? >= k.lane_return_time - 1e-6;
    if (offset == -1 && (left.is_none_or(|l| l >= back) || cancelled))
        || (offset == 1 && (right.is_none_or(|r| r <= -back) || cancelled))
    {
        offset = 0;
    }
    if offset == 0 {
        lane_return = py.None().into_bound(py);
        let left_start = left.filter(|&l| {
            l < (if free { 0. } else { -0.3 })
                && (free || (selected[0].is_none() && selected[1].is_some() && selected[2].is_none()))
        });
        let right_start = right.filter(|&r| {
            r > (if free { 0. } else { 0.3 })
                && (free || (selected[0].is_none() && selected[2].is_some() && selected[1].is_none()))
        });
        if let Some(left) = left_start {
            if let Some(outer) = lanes.boundary(0, x) {
                if outer > 0.1 && 1.8 <= outer - left && outer - left <= 5. {
                    offset = -1;
                }
            }
        } else if let Some(right) = right_start {
            if let Some(outer) = lanes.boundary(3, x) {
                if outer < -0.1 && 1.8 <= right - outer && right - outer <= 5. {
                    offset = 1;
                }
            }
        }
    }
    m.setattr(intern!(py, "lane_return"), lane_return)?;
    m.setattr(intern!(py, "lane_offset"), offset)?;
    lanes.set_lane_offset(offset);
    if !m.getattr(intern!(py, "lane_change"))?.is_truthy()? {
        m.setattr(intern!(py, "lane_change"), offset != 0)?;
    }
    Ok(())
}

/// `_vision(model)` with the model's leads as (x, y): the nearest fresh object to each lead is a vision-seen vehicle.
pub(crate) fn vision(objects: &Bound<'_, PyDict>, local: &Bound<'_, PyDict>, now: f64, leads: &[Xy]) -> PyResult<()> {
    for &(x, y) in leads {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let mut best: Option<(f64, Bound<'_, MapObject>)> = None;
        for value in objects.values() {
            let obj = value.cast_into::<MapObject>()?;
            let (key, seen) = {
                let o = obj.try_borrow()?;
                (o.key, o.seen)
            };
            if now - seen > 0.15 {
                continue;
            }
            let (px, py) = local_xy(local, key)?;
            if (px - x).abs() <= py_max(3., 0.2 * x) && (py - y).abs() <= 1.2 {
                let distance = hypot(px - x, py - y);
                if best.as_ref().is_none_or(|b| distance < b.0) {
                    best = Some((distance, obj));
                }
            }
        }
        if let Some((_, obj)) = best {
            let mut o = obj.try_borrow_mut()?;
            o.vision = true;
            o.vehicle = py_max(0.8, o.vehicle);
            o.vision_seen = true;
        }
    }
    Ok(())
}

/// `_approach_only(obj)`.
pub(crate) fn approach_only(o: &MapObject) -> bool {
    !(o.radar_moving || o.forward_samples != 0 || o.vision_seen || o.admitted)
}

/// `_wall_outside(x, y)`.
fn wall_outside(walls: &[Vec<Xy>], x: f64, y: f64) -> bool {
    walls.iter().any(|points| interpolate(points, x).is_some_and(|wy| wy * y > 0. && y.abs() >= wy.abs() - 0.25))
}

/// Python `a < b` for tuples of floats: the first unequal element decides.
fn floats_lt(a: &[f64], b: &[f64]) -> bool {
    for (x, y) in a.iter().zip(b) {
        if x != y {
            return x < y;
        }
    }
    false
}

/// A selection candidate: (x, centre distance, key).
type Candidate = (f64, f64, i64);

/// `self.hold[1:]` of the FF object: (lateral, centred, speed).
type Hold<'py> = (Bound<'py, PyAny>, bool, Bound<'py, PyAny>);

/// `_select()` and what it calls, on one update's state.
pub(crate) struct Selector<'a, 'py> {
    pub m: &'a Bound<'py, PyAny>,
    pub k: &'a Consts,
    pub lanes: &'a mut Lanes<'py>,
    pub frame: Frame,
    pub walls: &'a [Vec<Xy>],
    pub now: f64,
    pub now_obj: &'a Bound<'py, PyAny>,
    pub vision_leads: &'a [(f64, f64, Option<f64>)],
    pub objects: Bound<'py, PyDict>,
    pub local: Bound<'py, PyDict>,
    pub previous: [Option<i64>; 3],
    pub holding: bool,
    pub lane_change: bool,
    pub usable: bool,
    pub radar_stamp: Option<f64>,
    pub jumps: Bound<'py, PyDict>,
}

impl<'py> Selector<'_, 'py> {
    fn py(&self) -> Python<'py> {
        self.m.py()
    }

    /// `self.jumps.get(source, -math.inf)`.
    fn jumped_at(&self, source: i128) -> PyResult<f64> {
        match self.jumps.get_item(source)? {
            Some(stamp) => stamp.extract(),
            None => Ok(f64::NEG_INFINITY),
        }
    }

    /// The wall lines' y at x, for `_lane(respect_walls=True)`.
    fn walls_at(&self, x: f64) -> Vec<f64> {
        self.walls.iter().filter_map(|points| interpolate(points, x)).collect()
    }

    pub fn select(&mut self) -> PyResult<()> {
        let py = self.py();
        let k = self.k;
        let mut groups: [Vec<Candidate>; 3] = Default::default();
        let mut far: Option<(f64, i64)> = None; // The displayed FF object beyond FF_FAR_X, not in ego's lane slot.
        for value in self.objects.values() {
            let obj = value.cast_into::<MapObject>()?;
            let o = obj.try_borrow()?;
            let (x, y) = local_xy(&self.local, o.key)?;
            // Ordinary candidates are fresh measurements or a short prediction of one.
            if self.now - o.seen > k.prediction_hold + 1e-6 {
                continue;
            }
            let limit = if o.motion_confirmed || o.vision { k.far_x } else { k.max_x };
            if !(0. <= x && x <= limit)
                || o.hits < 3
                || o.seen - o.born < 0.15
                || o.existence < 0.5
                || (o.vehicle < 0.6 && !o.admitted)
                || o.uncertainty > k.max_display_uncertainty
                || o.structure
            {
                continue;
            }
            let walls = self.walls_at(x);
            let (mut slot, _, mut centre_distance) = self.lanes.lane(x, y, o.motion_confirmed, o.settled, &walls)?;
            if slot != Some(0)
                && Some(o.key) == self.previous[0]
                && x > k.ff_far_x
                && (o.motion_confirmed || o.vision || !wall_outside(self.walls, x, y))
            {
                far = Some((x, o.key));
            }
            if slot.is_some() && slot != Some(0) && x > k.max_side_x {
                continue; // Same side-slot distance limit as the existing selector.
            }
            if slot.is_none() && o.vision && y.abs() <= 2. && !wall_outside(self.walls, x, y) {
                slot = Some(0);
                centre_distance = y.abs();
            }
            let Some(slot) = slot else { continue };
            // The direction of its last real measurement, never the zeroed velocity of an unseen object.
            if slot != 0 && (o.last_motion.0 < -1. / 3.6 || approach_only(&o)) {
                continue; // Opposing traffic stays in the map, outside same-direction side slots.
            }
            if o.structure || (!o.motion_confirmed && !o.vision && wall_outside(self.walls, x, y)) {
                continue;
            }
            let slot = usize::try_from(slot).map_err(|_| PyKeyError::new_err(slot))?;
            groups[slot].push((x, centre_distance, o.key));
        }

        let mut selected: [Option<i64>; 3] = [None; 3];
        let switched = self.m.getattr(intern!(py, "switched"))?;
        let held_since = self.m.getattr(intern!(py, "absent"))?;
        let mut absent: [Option<Bound<'py, PyAny>>; 3] = [None, None, None];
        // A side target that moved to the other side's candidates is not held (it would be drawn twice).
        let elsewhere: [bool; 3] = std::array::from_fn(|slot| {
            slot != 0 && self.previous[slot].is_some_and(|p| groups[3 - slot].iter().any(|c| c.2 == p))
        });
        for (slot, mut candidates) in groups.into_iter().enumerate() {
            if slot != 0 {
                if let Some(lead) = selected[0] {
                    let lead_x = local_xy(&self.local, lead)?.0;
                    // A bridged FF object measured beside ego's lane stays FF, not a side vehicle as well.
                    let mut kept = Vec::with_capacity(candidates.len());
                    for c in candidates {
                        if c.2 != lead && !self.behind_lead(c.2, slot as i64, lead_x)? {
                            kept.push(c);
                        }
                    }
                    candidates = kept;
                }
            }
            if slot != 0 && !candidates.is_empty() && !self.holding && !self.lane_change {
                // An absent outer line must not let the lane beyond displace a measured adjacent candidate.
                let (inner, outer, sign) = if slot == 1 { (1, 0, 1.) } else { (2, 3, -1.) };
                let line_usable = self.lanes.usable(outer);
                let mut adjacent = Vec::new();
                for &c in &candidates {
                    let (x, y) = local_xy(&self.local, c.2)?;
                    let outer_valid = line_usable && self.lanes.at(outer, x).is_some();
                    let base = self.lanes.at(inner, x);
                    if outer_valid || base.is_none_or(|base| sign * (y - base) <= k.outer_far) {
                        adjacent.push(c);
                    }
                }
                if !adjacent.is_empty() {
                    candidates = adjacent;
                }
            }
            let mut choice = None;
            if let Some(first) = candidates.first() {
                let nearest = candidates[1..].iter().fold(first.0, |m, c| if c.0 < m { c.0 } else { m });
                let close: Vec<Candidate> = candidates.iter().copied().filter(|c| c.0 <= nearest + 1.).collect();
                let old = close.iter().find(|c| Some(c.2) == self.previous[slot]);
                let mut best = close[0];
                for &c in &close[1..] {
                    if floats_lt(&[c.1, c.0], &[best.1, best.0]) || (c.1 == best.1 && c.0 == best.0 && c.2 < best.2) {
                        best = c;
                    }
                }
                choice = Some(match old {
                    Some(old) if old.1 <= best.1 + 0.25 => old.2,
                    _ => best.2,
                });
            }
            let blocked = if slot == 0 {
                None
            } else if elsewhere[slot] {
                self.previous[slot]
            } else {
                selected[0]
            };
            let since = switched.get_item(slot)?;
            let (kept, held) = self.switch_guard(slot, choice, &candidates, blocked, &since, &held_since.get_item(slot)?)?;
            selected[slot] = kept;
            absent[slot] = held;
            if slot == 0 {
                let normal = self.far_hold(selected[0], far)?;
                selected[0] = self.bridge(normal)?;
            }
        }
        let mut remembered = [false; 3];
        if self.holding {
            let blindspot = self.m.getattr(intern!(py, "blindspot"))?;
            for slot in [1usize, 2] {
                if selected[slot].is_none() && !blindspot.get_item(slot - 1)?.is_truthy()? {
                    selected[slot] = self.stop_memory(slot)?;
                    if selected[slot].is_some() && selected[slot] == selected[0] {
                        selected[slot] = None;
                    }
                    remembered[slot] = selected[slot].is_some();
                }
            }
        }
        self.m.setattr(intern!(py, "remembered"), PyList::new(py, remembered)?)?;
        let mut since = Vec::with_capacity(3);
        for slot in 0..3 {
            since.push(match (selected[slot], self.previous[slot]) {
                (Some(new), Some(old)) if new != old => self.now_obj.clone(),
                _ => switched.get_item(slot)?,
            });
        }
        self.m.setattr(intern!(py, "switched"), PyTuple::new(py, since)?)?;
        self.m.setattr(intern!(py, "absent"), PyTuple::new(py, absent)?)?;
        self.m.setattr(intern!(py, "selected"), PyTuple::new(py, selected)?)?;
        Ok(())
    }

    /// `_switch_guard(slot, choice, candidates, blocked)`: the slot's target and since when it is held outside the
    /// candidates, for a target tracked for SWITCH_SETTLE, measured this radar cycle and within FF_FAR_X.
    fn switch_guard(
        &self,
        slot: usize,
        choice: Option<i64>,
        candidates: &[Candidate],
        blocked: Option<i64>,
        since: &Bound<'py, PyAny>,
        absent: &Bound<'py, PyAny>,
    ) -> PyResult<(Option<i64>, Option<Bound<'py, PyAny>>)> {
        let k = self.k;
        let Some(current) = self.previous[slot] else { return Ok((choice, None)) };
        let Some(obj) = self.objects.get_item(current)? else { return Ok((choice, None)) };
        let obj = obj.cast_into::<MapObject>()?;
        let (born, structure, seen, sideways) = {
            let o = obj.try_borrow()?;
            (o.born, o.structure, o.seen, self.frame.c * o.vy - self.frame.s * o.vx)
        };
        let x = local_xy(&self.local, current)?.0;
        if choice == Some(current)
            || self.lane_change
            || self.m.getattr(intern!(self.py(), "lane_offset"))?.is_truthy()?
            || (slot == 0 && !self.m.getattr(intern!(self.py(), "bridge"))?.is_none())
            || self.now - born < k.switch_settle - 1e-6
            || Some(seen) != self.radar_stamp
            || !(0. <= x && x <= k.ff_far_x)
        {
            return Ok((choice, None));
        }
        if let Some(held) = candidates.iter().find(|c| c.2 == current) {
            let new = candidates.iter().find(|c| Some(c.2) == choice);
            let since: f64 = if since.is_none() { f64::NEG_INFINITY } else { since.extract()? };
            let release = self.now - since > k.switch_hold + 1e-6
                || new.is_none_or(|new| new.0 < held.0 - k.switch_margin_x || new.1 < held.1 - k.switch_margin_c);
            return Ok((if release { choice } else { Some(current) }, None));
        }
        let absent = if absent.is_none() { self.now_obj.clone() } else { absent.clone() };
        if Some(current) == blocked
            || structure
            || sideways.abs() > k.switch_depart
            || self.now - absent.extract::<f64>()? > k.switch_grace + 1e-6
        {
            return Ok((choice, None));
        }
        Ok((Some(current), Some(absent)))
    }

    /// `_trail(obj)`: for each recorded point TRAIL_NEAR-TRAIL_FAR ahead, how far it lies outside ego's lane lines.
    fn trail(&self, o: &MapObject) -> PyResult<Vec<f64>> {
        let offset = self.lanes.lane_offset();
        let mut lines = Vec::with_capacity(2);
        for i in [1 + offset, 2 + offset] {
            if self.lanes.usable(i) {
                lines.push(usize::try_from(i).ok().and_then(|i| self.lanes.geometry(i)));
            }
        }
        if lines.len() < 2 {
            return Ok(Vec::new());
        }
        let none = || PyTypeError::new_err("object of type 'NoneType' has no len()");
        let mut offsets = Vec::new();
        for (tx, ty) in o.trail.iter() {
            let (x, y) = self.frame.to_ego(tx, ty);
            if self.k.trail_near <= x && x <= self.k.trail_far {
                let left = interpolate(lines[0].ok_or_else(none)?, x);
                let right = interpolate(lines[1].ok_or_else(none)?, x);
                if let (Some(left), Some(right)) = (left, right) {
                    offsets.push(py_max(y - left, right - y));
                }
            }
        }
        Ok(offsets)
    }

    /// `_behind_lead(key, slot, lead_x)`.
    fn behind_lead(&self, key: i64, slot: i64, lead_x: f64) -> PyResult<bool> {
        let k = self.k;
        let obj = object(&self.objects, key)?;
        let (x, y) = local_xy(&self.local, key)?;
        if x <= lead_x + k.lead_beyond {
            return Ok(false);
        }
        if x > (if self.holding { k.lead_side_hold_x } else { k.lead_side_x }) && x > lead_x + k.lead_side_past {
            return Ok(true);
        }
        let offset = self.lanes.lane_offset();
        if let (Some(left), Some(right)) = (self.lanes.at(1 + offset, x), self.lanes.at(2 + offset, x)) {
            if (0.5 * (left + right)).abs() > k.lead_curve && (if slot == 1 { y - left } else { right - y }) < k.lead_margin {
                return Ok(true);
            }
        }
        let offsets = self.trail(&*obj.try_borrow()?)?;
        Ok(offsets.len() >= 2 && offsets[1..].iter().fold(offsets[0], |m, &v| py_max(m, v)) <= k.trail_room)
    }

    /// `_far_hold(normal, far)`.
    fn far_hold(&self, normal: Option<i64>, far: Option<(f64, i64)>) -> PyResult<Option<i64>> {
        let py = self.py();
        let release = match far {
            None => true,
            Some(far) => {
                self.lane_change
                    || !self.m.getattr(intern!(py, "bridge"))?.is_none()
                    || match normal {
                        Some(normal) => local_xy(&self.local, normal)?.0 <= far.0,
                        None => false,
                    }
            }
        };
        let Some((_, far_key)) = far.filter(|_| !release) else {
            self.m.setattr(intern!(py, "far_since"), py.None())?;
            return Ok(normal);
        };
        let far_since = self.m.getattr(intern!(py, "far_since"))?;
        let since = if far_since.is_none() || far_since.get_item(0)?.extract::<i64>()? != far_key {
            let value = PyTuple::new(py, [far_key.into_pyobject(py)?.into_any(), self.now_obj.clone()])?;
            self.m.setattr(intern!(py, "far_since"), value)?;
            self.now
        } else {
            far_since.get_item(1)?.extract()?
        };
        Ok(if self.now - since <= self.k.ff_far_hold + 1e-6 { Some(far_key) } else { normal })
    }

    /// `_bridge(normal)`.
    fn bridge(&self, normal: Option<i64>) -> PyResult<Option<i64>> {
        let py = self.py();
        let kept = self.bridged(normal)?;
        let Some(key) = normal.filter(|_| kept == normal) else { return Ok(kept) };
        // Record where the FF object was last reported inside ego's lane, unless that report is part of a jump.
        let obj = object(&self.objects, key)?;
        let (report, source, vx, vy) = {
            let o = obj.try_borrow()?;
            (o.report, o.source, o.vx, o.vy)
        };
        if let Some((stamp, x, y)) = report {
            if Some(stamp) == self.radar_stamp && self.now - self.jumped_at(source)? > self.k.bridge_jump_time + 1e-6 {
                let (left, right) = self.lanes.ego_lines(x);
                let speed = self.frame.c * vx + self.frame.s * vy;
                let hold = match (left, right) {
                    (Some(left), Some(right)) => {
                        (right <= y && y <= left).then_some((key, y - 0.5 * (left + right), true, speed)) // Relative to the lane centre.
                    }
                    _ => (y.abs() <= self.k.bridge_lane).then_some((key, y, false, speed)),
                };
                if let Some(hold) = hold {
                    self.m.setattr(intern!(py, "hold"), hold)?;
                }
            }
        }
        let hold = self.m.getattr(intern!(py, "hold"))?;
        if !hold.is_none() && hold.get_item(0)?.extract::<i64>()? != key {
            self.m.setattr(intern!(py, "hold"), py.None())?;
        }
        Ok(normal)
    }

    /// `self.hold` if it is the given key's, as (lateral, centred, speed).
    fn hold_of(&self, key: i64) -> PyResult<Option<Hold<'py>>> {
        let hold = self.m.getattr(intern!(self.py(), "hold"))?;
        if hold.is_none() || hold.get_item(0)?.extract::<i64>()? != key {
            return Ok(None);
        }
        Ok(Some((hold.get_item(1)?, hold.get_item(2)?.is_truthy()?, hold.get_item(3)?)))
    }

    /// `_bridged(normal)`.
    fn bridged(&self, normal: Option<i64>) -> PyResult<Option<i64>> {
        let py = self.py();
        let k = self.k;
        let bridge = self.m.getattr(intern!(py, "bridge"))?;
        let key: Option<i64> =
            if bridge.is_truthy()? { bridge.get_item(intern!(py, "key"))?.extract()? } else { self.previous[0] };
        let obj = match key {
            Some(key) => self.objects.get_item(key)?.map(|o| o.cast_into::<MapObject>()).transpose()?,
            None => None,
        };
        self.m.setattr(intern!(py, "bridge"), py.None())?;
        if let Some(obj) = &obj {
            if !bridge.is_none() && normal == key {
                forget(&mut *obj.try_borrow_mut()?, self.frame.c, self.frame.s); // Back in ego's lane on its own reports.
            }
        }
        let Some(obj) = obj.filter(|_| normal != key && !self.lane_change && self.usable) else { return Ok(normal) };
        let key = key.expect("an object was found");
        let (start, x0, dx, speed, cause);
        if bridge.is_none() {
            // A lost report, a report that jumped sideways, or one that crossed the lane line slowly.
            let o = obj.try_borrow()?;
            let unseen = Some(o.seen) != self.radar_stamp;
            let radar_stamp = self
                .radar_stamp
                .ok_or_else(|| PyTypeError::new_err("unsupported operand type(s) for -: 'NoneType' and 'float'"))?;
            let jumped = radar_stamp - self.jumped_at(o.source)? <= k.prediction_hold + 1e-6;
            let crossed = matches!(o.slot, Some(1 | 2)) && !o.structure && local_xy(&self.local, key)?.0 <= k.bridge_drift_x;
            if !(unseen || jumped || crossed) {
                return Ok(normal);
            }
            cause = PyString::new(py, if unseen { "lost" } else if jumped { "jump" } else { "drift" }).into_any();
            start = self.now_obj.clone();
            x0 = local_xy(&self.local, key)?.0;
            dx = None;
            speed = match self.hold_of(key)? {
                Some((_, _, speed)) => speed.extract::<f64>()?,
                None => self.frame.c * o.vx + self.frame.s * o.vy,
            };
        } else {
            start = bridge.get_item(intern!(py, "start"))?;
            x0 = bridge.get_item(intern!(py, "x"))?.extract()?;
            dx = bridge.get_item(intern!(py, "dx"))?.extract::<Option<f64>>()?;
            speed = bridge.get_item(intern!(py, "speed"))?.extract::<f64>()?;
            cause = bridge.get_item(intern!(py, "cause"))?;
            if self.now - start.extract::<f64>()? > k.bridge_hold {
                return Ok(normal);
            }
        }
        let mut best: Option<(f64, f64, Option<f64>)> = None;
        for &(x, y, v) in self.vision_leads {
            let centre = self.lanes.centre(x);
            if (x - x0).abs() <= py_max(3., 0.2 * x0) && (y - centre.unwrap_or(0.)).abs() <= k.bridge_lane {
                let m = ((x - x0).abs(), x, v);
                if best.is_none_or(|b| floats_lt(&[m.0, m.1], &[b.0, b.1])) {
                    best = Some(m);
                }
            }
        }
        let Some((_, vision_x, vision_speed)) = best else { return Ok(normal) };
        // The displayed speed is vision's, else the last one before: the merged track reports the other vehicle's.
        let speed = vision_speed.unwrap_or(speed);
        // Follow the vision distance from where the object was drawn.
        let dx = dx.unwrap_or(x0 - vision_x);
        let x = vision_x + dx;
        if let Some(normal) = normal {
            if local_xy(&self.local, normal)?.0 <= x + py_max(3., 0.2 * x) {
                return Ok(Some(normal)); // A nearer measured vehicle takes FF; only a farther one cannot.
            }
        }
        let hold = self.hold_of(key)?;
        let centre = self.lanes.centre(x).unwrap_or(0.);
        let (stable_y, seen) = {
            let o = obj.try_borrow()?;
            (o.stable_y, o.seen)
        };
        let y = match (stable_y, hold) {
            (Some(stable_y), _) => centre + stable_y,
            (None, None) => centre,
            (None, Some((lateral, centred, _))) => {
                let lateral: f64 = lateral.extract()?;
                if centred {
                    centre + lateral
                } else {
                    lateral
                }
            }
        };
        self.local.set_item(key, (x, y))?;
        let state = PyDict::new(py);
        state.set_item(intern!(py, "key"), key)?;
        state.set_item(intern!(py, "start"), start)?;
        state.set_item(intern!(py, "x"), x)?;
        state.set_item(intern!(py, "y"), y)?;
        state.set_item(intern!(py, "dx"), dx)?;
        state.set_item(intern!(py, "speed"), speed)?;
        state.set_item(intern!(py, "cause"), cause)?;
        state.set_item(intern!(py, "kind"), if Some(seen) != self.radar_stamp { "lost" } else { "jump" })?;
        self.m.setattr(intern!(py, "bridge"), state)?;
        Ok(Some(key))
    }

    /// `_stop_memory(slot)`: a side vehicle that left radar coverage while ego stops.
    fn stop_memory(&self, slot: usize) -> PyResult<Option<i64>> {
        let k = self.k;
        let mut best: Option<(bool, f64, i64)> = None;
        for value in self.objects.values() {
            let obj = value.cast_into::<MapObject>()?;
            let o = obj.try_borrow()?;
            let (x, _) = local_xy(&self.local, o.key)?;
            if o.slot != Some(slot as i64)
                || o.departed
                || o.superseded
                || o.structure
                || (o.vehicle < 0.6 && !o.admitted)
                || o.hits < 3
                || o.existence < 0.5
                || self.now - o.seen <= k.prediction_hold + 1e-6
                || !(k.memory_min_x <= x && x <= k.memory_max_x)
            {
                continue;
            }
            // A vehicle that was crossing, turning or oncoming when it left coverage is gone, not waiting.
            let (forward, lateral) = o.last_motion;
            let resting = o.stationary || hypot(forward, lateral) <= 0.6;
            let alongside =
                lateral.abs() <= 0.6 && (0. ..=3.).contains(&forward) && self.now - o.seen <= k.memory_moving_hold;
            if !(resting || alongside)
                || crossing(&o, self.frame, k.crossing_travel)
                || (!o.motion_confirmed && (o.tracked as f64) < k.memory_tracked)
            {
                continue;
            }
            if approach_only(&o) {
                continue;
            }
            let candidate = (Some(o.key) != self.previous[slot], x.abs(), o.key);
            let smaller = match best {
                None => true,
                Some(b) => {
                    if candidate.0 != b.0 {
                        !candidate.0 & b.0
                    } else if candidate.1 != b.1 {
                        candidate.1 < b.1
                    } else {
                        candidate.2 < b.2
                    }
                }
            };
            if smaller {
                best = Some(candidate);
            }
        }
        Ok(best.map(|b| b.2))
    }
}
