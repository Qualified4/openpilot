//! Lane reference, model-lead association, and front slot selection.

use crate::pymath::{hypot, py_max};

use super::consts::Consts;
use super::lanes::{interpolate, Lanes};
use super::observe::{crossing, forget};
use super::state::{Error, Frame, MapState, Result};
use super::MapObjectData as MapObject;

type Xy = (f64, f64);

#[derive(Clone, Copy, Debug)]
pub struct Bridge {
    pub key: i64,
    pub start: f64,
    pub x: f64,
    pub y: f64,
    pub dx: f64,
    pub speed: f64,
    pub cause: &'static str,
    pub kind: &'static str,
}

/// `_update_lane_reference()`. `lanes` is the update's snapshot; its lane offset follows the one written here.
pub fn lane_reference(
    st: &mut MapState,
    k: &Consts,
    lanes: &mut Lanes,
    now: f64,
    free: bool,
    lane_change: bool,
) -> bool {
    if !lanes.model_fresh {
        st.lane_offset = 0;
        lanes.set_lane_offset(0);
        return lane_change;
    }
    let selected = st.selected;
    // Require an empty FF and a displayed destination-side car before starting, unless ego is changing lanes.
    let x = if free { 5. } else { 10. };
    let (left, right) = (lanes.boundary(1, x), lanes.boundary(2, x));
    let mut offset = st.lane_offset;
    let back = if free { 1. } else { 0.1 };
    let returned =
        (offset == -1 && left.is_some_and(|l| l >= 0.5)) || (offset == 1 && right.is_some_and(|r| r <= -0.5));
    let mut lane_return = returned.then(|| st.lane_return.unwrap_or(now));
    let cancelled = lane_return.is_some_and(|since| now - since >= k.lane_return_time - 1e-6);
    if (offset == -1 && (left.is_none_or(|l| l >= back) || cancelled))
        || (offset == 1 && (right.is_none_or(|r| r <= -back) || cancelled))
    {
        offset = 0;
    }
    if offset == 0 {
        lane_return = None;
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
    st.lane_return = lane_return;
    st.lane_offset = offset;
    lanes.set_lane_offset(offset);
    lane_change || offset != 0
}

/// `_vision(model)` with the model's leads as (x, y): the nearest fresh object to each lead is a vision-seen vehicle.
pub fn vision(st: &mut MapState, now: f64, leads: &[Xy]) -> Result<()> {
    for &(x, y) in leads {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let mut best: Option<(f64, i64)> = None;
        for o in st.objects.values() {
            if now - o.seen > 0.15 {
                continue;
            }
            let (px, py) = st.local_xy(o.key)?;
            if (px - x).abs() <= py_max(3., 0.2 * x) && (py - y).abs() <= 1.2 {
                let distance = hypot(px - x, py - y);
                if best.as_ref().is_none_or(|b| distance < b.0) {
                    best = Some((distance, o.key));
                }
            }
        }
        if let Some((_, key)) = best {
            let o = st.objects.get_mut(&key).expect("found above");
            o.vision = true;
            o.vehicle = py_max(0.8, o.vehicle);
            o.vision_seen = true;
        }
    }
    Ok(())
}

/// `_approach_only(obj)`.
pub fn approach_only(o: &MapObject) -> bool {
    !(o.radar_moving || o.forward_samples != 0 || o.vision_seen || o.admitted)
}

/// `_stop_memory`'s test for one object at ego distance `x`, apart from its slot and how long it has been unseen:
/// a vehicle that left radar coverage resting or alongside, and that a stopping ego keeps drawing.
pub(crate) fn rememberable(k: &Consts, frame: Frame, now: f64, o: &MapObject, x: f64) -> bool {
    if o.departed
        || o.superseded
        || o.structure
        || (o.vehicle < 0.6 && !o.admitted)
        || o.hits < 3
        || o.existence < 0.5
        || !(k.memory_min_x <= x && x <= k.memory_max_x)
    {
        return false;
    }
    // A vehicle that was crossing, turning or oncoming when it left coverage is gone, not waiting.
    let (forward, lateral) = o.last_motion;
    let resting = o.stationary || hypot(forward, lateral) <= 0.6;
    let alongside = lateral.abs() <= 0.6 && (0. ..=3.).contains(&forward) && now - o.seen <= k.memory_moving_hold;
    (resting || alongside)
        && !crossing(o, frame, k.crossing_travel)
        && (o.motion_confirmed || (o.tracked as f64) >= k.memory_tracked)
        && !approach_only(o)
}

/// `_wall_outside(x, y)`.
fn wall_outside(walls: &[Vec<Xy>], x: f64, y: f64) -> bool {
    walls
        .iter()
        .any(|points| interpolate(points, x).is_some_and(|wy| wy * y > 0. && y.abs() >= wy.abs() - 0.25))
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

/// `_select()` and the inputs it reads in addition to the map.
pub struct Selector<'a> {
    pub st: &'a mut MapState,
    pub k: &'a Consts,
    pub lanes: &'a Lanes<'a>,
    pub frame: Frame,
    pub now: f64,
    pub vision_leads: &'a [(f64, f64, Option<f64>)],
    pub holding: bool,
    pub lane_change: bool,
    pub usable: bool,
    pub radar_stamp: Option<f64>,
    pub blindspot: [bool; 2],
}

impl Selector<'_> {
    fn jumped_at(&self, source: i128) -> f64 {
        self.st.jumps.get(&source).copied().unwrap_or(f64::NEG_INFINITY)
    }

    pub fn select(&mut self) -> Result<()> {
        let k = self.k;
        let mut groups: [Vec<Candidate>; 3] = Default::default();
        let mut far: Option<(f64, i64)> = None; // The displayed FF object beyond FF_FAR_X, not in ego's lane slot.
        for o in self.st.objects.values() {
            let (x, y) = self.st.local_xy(o.key)?;
            // Ordinary candidates are fresh measurements or a short prediction of one.
            if self.now - o.seen > k.prediction_hold + 1e-6 {
                continue;
            }
            let limit = if o.motion_confirmed || o.vision {
                k.far_x
            } else {
                k.max_x
            };
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
            let walls: Vec<f64> = self
                .st
                .wall_geometry
                .iter()
                .filter_map(|points| interpolate(points, x))
                .collect();
            let (mut slot, _, mut centre_distance) = self.lanes.lane(x, y, o.motion_confirmed, o.settled, &walls)?;
            if slot != Some(0)
                && Some(o.key) == self.st.selected[0]
                && x > k.ff_far_x
                && (o.motion_confirmed || o.vision || !wall_outside(&self.st.wall_geometry, x, y))
            {
                far = Some((x, o.key));
            }
            if slot.is_some() && slot != Some(0) && x > k.max_side_x {
                continue; // Same side-slot distance limit as the existing selector.
            }
            if slot.is_none() && o.vision && y.abs() <= 2. && !wall_outside(&self.st.wall_geometry, x, y) {
                slot = Some(0);
                centre_distance = y.abs();
            }
            let Some(slot) = slot else { continue };
            // The direction of its last real measurement, never the zeroed velocity of an unseen object.
            if slot != 0 && (o.last_motion.0 < -1. / 3.6 || approach_only(o)) {
                continue; // Opposing traffic stays in the map, outside same-direction side slots.
            }
            if o.structure || (!o.motion_confirmed && !o.vision && wall_outside(&self.st.wall_geometry, x, y)) {
                continue;
            }
            let group = groups.get_mut(slot as usize).ok_or(Error::Key(slot))?;
            group.push((x, centre_distance, o.key));
        }

        let mut selected: [Option<i64>; 3] = [None; 3];
        let switched = self.st.switched;
        let held_since = self.st.absent;
        let mut absent = [None; 3];
        // A side target that moved to the other side's candidates is not held (it would be drawn twice).
        let elsewhere: [bool; 3] = std::array::from_fn(|slot| {
            slot != 0 && self.st.selected[slot].is_some_and(|p| groups[3 - slot].iter().any(|c| c.2 == p))
        });
        for (slot, mut candidates) in groups.into_iter().enumerate() {
            if slot != 0 {
                if let Some(lead) = selected[0] {
                    let lead_x = self.st.local_xy(lead)?.0;
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
                    let (x, y) = self.st.local_xy(c.2)?;
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
                let nearest = candidates[1..]
                    .iter()
                    .fold(first.0, |m, c| if c.0 < m { c.0 } else { m });
                let close: Vec<Candidate> = candidates.iter().copied().filter(|c| c.0 <= nearest + 1.).collect();
                let old = close.iter().find(|c| Some(c.2) == self.st.selected[slot]);
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
                self.st.selected[slot]
            } else {
                selected[0]
            };
            let (kept, held) =
                self.switch_guard(slot, choice, &candidates, blocked, switched[slot], held_since[slot])?;
            selected[slot] = kept;
            absent[slot] = held;
            if slot == 0 {
                let normal = self.far_hold(selected[0], far)?;
                selected[0] = self.bridge(normal)?;
            }
        }
        let mut remembered = [false; 3];
        if self.holding {
            for slot in [1usize, 2] {
                if selected[slot].is_none() && !self.blindspot[slot - 1] {
                    selected[slot] = self.stop_memory(slot)?;
                    if selected[slot].is_some() && selected[slot] == selected[0] {
                        selected[slot] = None;
                    }
                    remembered[slot] = selected[slot].is_some();
                }
            }
        }
        self.st.remembered = remembered;
        self.st.switched = std::array::from_fn(|slot| match (selected[slot], self.st.selected[slot]) {
            (Some(new), Some(old)) if new != old => Some(self.now),
            _ => switched[slot],
        });
        self.st.absent = absent;
        self.st.selected = selected;
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
        since: Option<f64>,
        absent: Option<f64>,
    ) -> Result<(Option<i64>, Option<f64>)> {
        let k = self.k;
        let Some(current) = self.st.selected[slot] else {
            return Ok((choice, None));
        };
        let Some(o) = self.st.objects.get(&current) else {
            return Ok((choice, None));
        };
        let (born, structure, seen, sideways) =
            (o.born, o.structure, o.seen, self.frame.c * o.vy - self.frame.s * o.vx);
        let x = self.st.local_xy(current)?.0;
        if choice == Some(current)
            || self.lane_change
            || self.st.lane_offset != 0
            || (slot == 0 && self.st.bridge.is_some())
            || self.now - born < k.switch_settle - 1e-6
            || Some(seen) != self.radar_stamp
            || !(0. <= x && x <= k.ff_far_x)
        {
            return Ok((choice, None));
        }
        if let Some(held) = candidates.iter().find(|c| c.2 == current) {
            let new = candidates.iter().find(|c| Some(c.2) == choice);
            let since = since.unwrap_or(f64::NEG_INFINITY);
            let release = self.now - since > k.switch_hold + 1e-6
                || new.is_none_or(|new| new.0 < held.0 - k.switch_margin_x || new.1 < held.1 - k.switch_margin_c);
            return Ok((if release { choice } else { Some(current) }, None));
        }
        let absent = absent.unwrap_or(self.now);
        if Some(current) == blocked
            || structure
            || sideways.abs() > k.switch_depart
            || self.now - absent > k.switch_grace + 1e-6
        {
            return Ok((choice, None));
        }
        Ok((Some(current), Some(absent)))
    }

    /// `_trail(obj)`: for each recorded point TRAIL_NEAR-TRAIL_FAR ahead, how far it lies outside ego's lane lines.
    fn trail(&self, o: &MapObject) -> Result<Vec<f64>> {
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
        let none = || Error::Type("object of type 'NoneType' has no len()");
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
    fn behind_lead(&self, key: i64, slot: i64, lead_x: f64) -> Result<bool> {
        let k = self.k;
        let obj = self.st.object(key)?;
        let (x, y) = self.st.local_xy(key)?;
        if x <= lead_x + k.lead_beyond {
            return Ok(false);
        }
        if x > (if self.holding {
            k.lead_side_hold_x
        } else {
            k.lead_side_x
        }) && x > lead_x + k.lead_side_past
        {
            return Ok(true);
        }
        let offset = self.lanes.lane_offset();
        if let (Some(left), Some(right)) = (self.lanes.at(1 + offset, x), self.lanes.at(2 + offset, x)) {
            if (0.5 * (left + right)).abs() > k.lead_curve
                && (if slot == 1 { y - left } else { right - y }) < k.lead_margin
            {
                return Ok(true);
            }
        }
        let offsets = self.trail(obj)?;
        Ok(offsets.len() >= 2 && offsets[1..].iter().fold(offsets[0], |m, &v| py_max(m, v)) <= k.trail_room)
    }

    /// `_far_hold(normal, far)`.
    fn far_hold(&mut self, normal: Option<i64>, far: Option<(f64, i64)>) -> Result<Option<i64>> {
        let release = match far {
            None => true,
            Some(far) => {
                self.lane_change
                    || self.st.bridge.is_some()
                    || match normal {
                        Some(normal) => self.st.local_xy(normal)?.0 <= far.0,
                        None => false,
                    }
            }
        };
        let Some((_, far_key)) = far.filter(|_| !release) else {
            self.st.far_since = None;
            return Ok(normal);
        };
        let since = match self.st.far_since {
            Some((key, since)) if key == far_key => since,
            _ => {
                self.st.far_since = Some((far_key, self.now));
                self.now
            }
        };
        Ok(if self.now - since <= self.k.ff_far_hold + 1e-6 {
            Some(far_key)
        } else {
            normal
        })
    }

    /// `_bridge(normal)`.
    fn bridge(&mut self, normal: Option<i64>) -> Result<Option<i64>> {
        let kept = self.bridged(normal)?;
        let Some(key) = normal.filter(|_| kept == normal) else {
            return Ok(kept);
        };
        // Record where the FF object was last reported inside ego's lane, unless that report is part of a jump.
        let o = self.st.object(key)?;
        let (report, source, vx, vy) = (o.report, o.source, o.vx, o.vy);
        if let Some((stamp, x, y)) = report {
            if Some(stamp) == self.radar_stamp && self.now - self.jumped_at(source) > self.k.bridge_jump_time + 1e-6 {
                let (left, right) = self.lanes.ego_lines(x);
                let speed = self.frame.c * vx + self.frame.s * vy;
                let hold = match (left, right) {
                    (Some(left), Some(right)) => {
                        (right <= y && y <= left).then_some((key, y - 0.5 * (left + right), true, speed))
                        // Relative to the lane centre.
                    }
                    _ => (y.abs() <= self.k.bridge_lane).then_some((key, y, false, speed)),
                };
                if let Some(hold) = hold {
                    self.st.hold = Some(hold);
                }
            }
        }
        if self.st.hold.is_some_and(|hold| hold.0 != key) {
            self.st.hold = None;
        }
        Ok(normal)
    }

    /// `self.hold` if it is the given key's, as (lateral, centred, speed).
    fn hold_of(&self, key: i64) -> Option<(f64, bool, f64)> {
        self.st
            .hold
            .filter(|hold| hold.0 == key)
            .map(|(_, lateral, centred, speed)| (lateral, centred, speed))
    }

    /// `_bridged(normal)`.
    fn bridged(&mut self, normal: Option<i64>) -> Result<Option<i64>> {
        let k = self.k;
        let bridge = self.st.bridge.take();
        let key = bridge.map(|bridge| bridge.key).or(self.st.selected[0]);
        let found = key.is_some_and(|key| self.st.objects.contains_key(&key));
        if found && !bridge.is_none() && normal == key {
            let o = self.st.objects.get_mut(&key.expect("found")).expect("found");
            forget(o, self.frame.c, self.frame.s); // Back in ego's lane on its own reports.
        }
        if !(found && normal != key && !self.lane_change && self.usable) {
            return Ok(normal);
        }
        let key = key.expect("an object was found");
        let (start, x0, dx, speed, cause);
        if let Some(bridge) = bridge {
            (start, x0, dx, speed, cause) = (bridge.start, bridge.x, Some(bridge.dx), bridge.speed, bridge.cause);
            if self.now - start > k.bridge_hold {
                return Ok(normal);
            }
        } else {
            // A lost report, a report that jumped sideways, or one that crossed the lane line slowly.
            let o = self.st.object(key)?;
            let unseen = Some(o.seen) != self.radar_stamp;
            let radar_stamp = self
                .radar_stamp
                .ok_or_else(|| Error::Type("unsupported operand type(s) for -: 'NoneType' and 'float'"))?;
            let jumped = radar_stamp - self.jumped_at(o.source) <= k.prediction_hold + 1e-6;
            let crossed = matches!(o.slot, Some(1 | 2)) && !o.structure && self.st.local_xy(key)?.0 <= k.bridge_drift_x;
            if !(unseen || jumped || crossed) {
                return Ok(normal);
            }
            cause = if unseen {
                "lost"
            } else if jumped {
                "jump"
            } else {
                "drift"
            };
            start = self.now;
            x0 = self.st.local_xy(key)?.0;
            dx = None;
            speed = match self.hold_of(key) {
                Some((_, _, speed)) => speed,
                None => self.frame.c * o.vx + self.frame.s * o.vy,
            };
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
        let Some((_, vision_x, vision_speed)) = best else {
            return Ok(normal);
        };
        // The displayed speed is vision's, else the last one before: the merged track reports the other vehicle's.
        let speed = vision_speed.unwrap_or(speed);
        // Follow the vision distance from where the object was drawn.
        let dx = dx.unwrap_or(x0 - vision_x);
        let x = vision_x + dx;
        if let Some(normal) = normal {
            if self.st.local_xy(normal)?.0 <= x + py_max(3., 0.2 * x) {
                return Ok(Some(normal)); // A nearer measured vehicle takes FF; only a farther one cannot.
            }
        }
        let hold = self.hold_of(key);
        let centre = self.lanes.centre(x).unwrap_or(0.);
        let (stable_y, seen) = {
            let o = self.st.object(key)?;
            (o.stable_y, o.seen)
        };
        let y = match (stable_y, hold) {
            (Some(stable_y), _) => centre + stable_y,
            (None, None) => centre,
            (None, Some((lateral, centred, _))) => {
                if centred {
                    centre + lateral
                } else {
                    lateral
                }
            }
        };
        self.st.local.insert(key, (x, y));
        let kind = if Some(seen) != self.radar_stamp { "lost" } else { "jump" };
        self.st.bridge = Some(Bridge {
            key,
            start,
            x,
            y,
            dx,
            speed,
            cause,
            kind,
        });
        Ok(Some(key))
    }

    /// `_stop_memory(slot)`: a side vehicle that left radar coverage while ego stops.
    fn stop_memory(&self, slot: usize) -> Result<Option<i64>> {
        let k = self.k;
        let mut best: Option<(bool, f64, i64)> = None;
        for o in self.st.objects.values() {
            let (x, _) = self.st.local_xy(o.key)?;
            if o.slot != Some(slot as i64)
                || self.now - o.seen <= k.prediction_hold + 1e-6
                || !rememberable(k, self.frame, self.now, o, x)
            {
                continue;
            }
            let candidate = (Some(o.key) != self.st.selected[slot], x.abs(), o.key);
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
