//! `CcncRadarMap._observe` after association (with `_reconnect`, `_stable`, `_settle`, `_crossing`,
//! `_forget_lateral`) on the map state (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. Each expression follows the Python one in the same order and with the
//! same grouping; objects are visited in the Python dict order (key order).

use std::collections::HashSet;

use super::consts::Consts;
use super::lanes::LaneQuery;
use super::object::MapObjectData;
use super::state::{Error, Frame, MapState, Result};
use crate::pymath::{hypot, py_max, py_min, py_sum};

/// `_forget_lateral(obj)`.
pub fn forget(o: &mut MapObjectData, c: f64, s: f64) {
    o.history.clear();
    let forward = c * o.vx + s * o.vy;
    o.vx = forward * c;
    o.vy = forward * s;
}

/// `_crossing(obj)`.
pub fn crossing(o: &MapObjectData, frame: Frame, travel: f64) -> bool {
    let Some((ox, oy)) = o.origin else { return false };
    let (dx, dy) = (o.x - ox, o.y - oy);
    let lateral = (frame.c * dy - frame.s * dx).abs();
    lateral >= travel && lateral > (frame.c * dx + frame.s * dy).abs()
}

/// Python `sum(values) / len(values)`.
fn mean(values: &[f64]) -> Result<f64> {
    match py_sum(values.iter().copied()) {
        Some(total) => Ok(total / values.len() as f64),
        None => Err(Error::ZeroDivision("division by zero")),
    }
}

/// One `_measurements` tuple: (source, x, y, vx, vy, explained, hc, hs, radar_moving, doppler, tracked).
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub source: i128,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub explained: f64,
    pub hc: f64,
    pub hs: f64,
    pub radar_moving: bool,
    pub doppler: i32,
    pub tracked: bool,
}

/// `_stable(obj)`.
fn stable(o: &mut MapObjectData, lanes: &mut dyn LaneQuery, bridge_key: Option<i64>, k: &Consts) -> Result<()> {
    let (stamp, x, y) = o.report.expect("set just before");
    let centre = lanes.centre(x)?;
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
        return Err(Error::ZeroDivision("float division by zero"));
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
fn settle(o: &mut MapObjectData, lanes: &mut dyn LaneQuery, x: f64, y: f64, stamp: f64, k: &Consts) -> Result<()> {
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

/// What `_observe` reads of the map besides its state.
#[derive(Clone, Copy, Debug)]
pub struct ObserveIn {
    pub holding: bool,
    pub a_ego: f64,
    /// `self.bridge`: (key, x, y) of the bridged FF object.
    pub bridge: Option<(i64, f64, f64)>,
}

/// `_observe` after `_associate`: `_reconnect`, then the measurement loop. `assigned` holds the object key of each
/// measurement, `reserved` the assigned keys. Returns `measured_keys` in the order they were added.
pub fn observe(
    st: &mut MapState,
    k: &Consts,
    inp: ObserveIn,
    lanes: &mut dyn LaneQuery,
    ms: &[Measurement],
    mut assigned: Vec<Option<i64>>,
    mut reserved: Vec<i64>,
    stamp: f64,
) -> Result<Vec<i64>> {
    let now = st.now.expect("advanced before observing");
    let (holding, odometer, a_ego, frame) = (inp.holding, st.odometer, inp.a_ego, st.frame());
    let bridge_key = inp.bridge.map(|b| b.0);

    // _reconnect: a bridged FF object continues with a new report near its bridged position in ego's lane.
    if let Some((key, bx, by)) = inp.bridge.filter(|b| !reserved.contains(&b.0)) {
        let mut best: Option<(f64, usize)> = None;
        for (i, me) in ms.iter().enumerate() {
            if assigned[i].is_some() {
                continue;
            }
            let (ex, ey) = frame.to_ego(me.x, me.y);
            let distance = hypot(ex - bx, ey - by);
            if distance <= k.bridge_reconnect
                && lanes.lane(ex, ey, true, false)?.0 == Some(0)
                && best.is_none_or(|b| distance < b.0)
            {
                best = Some((distance, i));
            }
        }
        if let Some((_, i)) = best {
            // to_world(bx, by, self.pose): math.cos/sin of the pose angle, as Python computes them.
            let (px, py, angle) = st.pose;
            let (c, s) = (angle.cos(), angle.sin());
            let o = st.object_mut(key)?;
            o.x = px + c * bx - s * by;
            o.y = py + s * bx + c * by;
            forget(o, frame.c, frame.s);
            o.vx = ms[i].vx;
            o.vy = ms[i].vy;
            assigned[i] = Some(key);
            reserved.push(key);
        }
    }
    let reserved: HashSet<i64> = reserved.into_iter().collect();

    let mut measured_keys = Vec::with_capacity(ms.len());
    let mut measured: HashSet<i64> = HashSet::with_capacity(ms.len());

    for (i, me) in ms.iter().enumerate() {
        let key = match assigned[i] {
            None => {
                if st.objects.len() as f64 >= k.max_objects {
                    // Retain confirmed nearby vehicles before unconfirmed distant returns.
                    let mut victim: Option<((bool, bool, f64, f64), i64)> = None;
                    for o in st.objects.values() {
                        if measured.contains(&o.key) || reserved.contains(&o.key) || bridge_key == Some(o.key) {
                            continue;
                        }
                        let priority = (
                            st.local_xy(o.key)?.0.abs() <= 15.,
                            o.structure || o.motion_confirmed || o.vehicle >= 0.6,
                            o.existence,
                            o.seen,
                        );
                        if victim.as_ref().is_none_or(|v| priority_lt(&priority, &v.0)) {
                            victim = Some((priority, o.key));
                        }
                    }
                    let Some((_, victim)) = victim else { continue };
                    if frame.to_ego(me.x, me.y).0.abs() > 15. && st.local_xy(victim)?.0.abs() <= 15. {
                        continue; // A new distant return cannot evict a remembered near object.
                    }
                    st.objects.remove(&victim);
                    if st.local.remove(&victim).is_none() {
                        return Err(Error::Key(victim));
                    }
                    st.evicted += 1;
                }
                let key = st.next_id;
                st.next_id += 1;
                if me.radar_moving {
                    for o in st.objects.values_mut() {
                        let age = stamp - o.seen;
                        if o.source == me.source && 0. < age && age <= k.id_cycle {
                            o.superseded = true;
                        }
                    }
                }
                st.objects.insert(key, MapObjectData::measured(key, me.source, me.x, me.y, me.vx, me.vy, stamp));
                st.created += 1;
                key
            }
            Some(key) => {
                let o = st.object_mut(key)?;
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
                key
            }
        };
        let o = st.objects.get_mut(&key).expect("present");
        o.explained = me.explained;
        let (rx, ry) = frame.to_ego(me.x, me.y);
        o.report = Some((stamp, rx, ry));
        stable(o, lanes, bridge_key, k)?;
        o.tracked += me.tracked as i64;
        o.radar_moving = o.radar_moving || me.radar_moving;
        if me.doppler != 0 {
            // Isolated noisy samples must not add up over a long observation.
            o.doppler += 1;
            o.forward_samples += (me.doppler > 0) as i64;
            o.doppler_recent.push(stamp);
        }
        if o.doppler_recent.len() as f64 == k.doppler_samples {
            let (Some(first), Some(last)) = (o.doppler_recent.front(), o.doppler_recent.back()) else {
                return Err(Error::Index("deque index out of range"));
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
        measured_keys.push(key);
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
        st.local.insert(key, (ex, ey));
        let o = st.objects.get_mut(&key).expect("present");
        o.last_motion = (me.hc * o.vx + me.hs * o.vy, me.hc * o.vy - me.hs * o.vx);
        settle(o, lanes, ex, ey, stamp, k)?;
        let (slot, interior, _) = lanes.lane(ex, ey, o.motion_confirmed, o.settled)?;
        o.slot = slot;
        o.interior = interior;
        let side = matches!(slot, Some(1 | 2));
        // A remembered vehicle that drives off reappears just ahead of where it stood, slowly, with the
        // radar's moving class or its old radar id, and the gap then grows. It is no longer remembered.
        if o.hits == 1 {
            if holding && side && me.hc * me.vx + me.hs * me.vy <= k.release_speed {
                let mut ahead = Vec::new();
                for n in st.objects.values() {
                    if n.key == key {
                        continue;
                    }
                    let candidate = n.slot == slot
                        && !n.departed
                        && now - n.seen > k.prediction_hold
                        && (n.vehicle >= 0.6 || n.admitted);
                    if !candidate {
                        continue;
                    }
                    let (lx, ly) = st.local_xy(n.key)?;
                    if !(k.memory_min_x <= lx && lx <= k.memory_max_x) {
                        continue;
                    }
                    let gap = ex - lx;
                    if !(k.release_gap.0 <= gap && gap <= k.release_gap.1 && (ey - ly).abs() <= k.release_lateral) {
                        continue;
                    }
                    ahead.push((n.key, ex - lx));
                }
                st.objects.get_mut(&key).expect("present").ahead_of = ahead;
            }
        } else {
            let (radar_moving, own_source, ahead_of) = (o.radar_moving, o.source, o.ahead_of.clone());
            for (remembered, gap) in ahead_of {
                let Some(n) = st.objects.get(&remembered) else { continue };
                let n_source = n.source;
                if remembered == key {
                    if (radar_moving || me.source == own_source) && ex - st.local_xy(remembered)?.0 >= gap + k.release_advance {
                        st.objects.get_mut(&key).expect("present").departed = true;
                    }
                    continue;
                }
                if (radar_moving || me.source == n_source) && ex - st.local_xy(remembered)?.0 >= gap + k.release_advance {
                    st.objects.get_mut(&remembered).expect("present").departed = true;
                }
            }
        }
        // While ego is stopped or stopping, a return that has rested for .5s near the middle of an adjacent lane
        // is a waiting vehicle, even if it never moved in view.
        let o = st.objects.get_mut(&key).expect("present");
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
            && lanes.body_clear(slot.unwrap_or(0), ex, ey, k.body_clearance, k.body_far_side)?
            && lanes.edge_room(slot.unwrap_or(0), ex, k.waiting_edge_room)?
            && !crossing(o, frame, k.crossing_travel)
            && !lanes.band_line(ex, ey, odometer, k.band_bin, k.band_memory)?
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
        } else if o.stationary && lanes.has_geometry()? {
            o.vehicle = py_max(0., o.vehicle - 0.3);
        }
    }
    Ok(measured_keys)
}
