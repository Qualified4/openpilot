//! `CcncRadarMap._measurements` and `CcncRadarMap._associate` on the map state (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. Every comparison, `min`/`max` and sort key below follows the
//! Python expression it replaces, in the same order, so that the results are the same to the bit.

use std::collections::HashMap;

use super::consts::Consts;
use super::lanes::Lanes;
use super::object::Ring;
use super::observe::Measurement;
use super::state::{Error, MapState, Result};
use crate::pymath::{hypot, py_max, py_min};

/// `math.floor(v)` for the grid cell of a finite coordinate.
pub fn floor(v: f64) -> Result<i64> {
    if v.is_nan() {
        return Err(Error::Value("cannot convert float NaN to integer"));
    }
    let f = v.floor();
    if !(-9.2e18..=9.2e18).contains(&f) {
        return Err(Error::Overflow("cannot convert float infinity to integer"));
    }
    Ok(f as i64)
}

/// A front radar point with finite values, measured: what `_measurements` reads of it.
#[derive(Clone, Copy, Debug)]
pub struct RadarPoint {
    pub track: i128,
    pub x: f64,
    pub y: f64,
    pub speed: f64,
    /// `getattr(p, 'motionState', None) == 2`.
    pub moving: bool,
    /// `getattr(p, 'trackState', 0) == 2`.
    pub tracked: bool,
}

/// An offline radar class report of a track (`self.evidence[track]`): (motion class == 2, reported distance,
/// track state == 2 if it carries one).
#[derive(Clone, Copy, Debug)]
pub struct Evidence {
    pub moving: bool,
    pub distance: f64,
    pub tracked: Option<bool>,
}

/// What `_measurements` reads of ego's motion.
#[derive(Clone, Copy, Debug)]
pub struct Ego {
    pub a_ego: f64,
    pub yaw_rate: f64,
    pub speed: f64,
}

/// The `_measurements` loop at the pose (px, py, cos, sin): updates the radar id history and returns the
/// measurements. `leads` is `self.leads`, `evidence` `self.evidence.get`.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    st: &mut MapState,
    k: &Consts,
    ego: Ego,
    leads: &[(f64, f64)],
    evidence: &dyn Fn(i128) -> Option<Evidence>,
    points: &[RadarPoint],
    stamp: f64,
    (px, py, c, s): (f64, f64, f64, f64),
) -> Vec<Measurement> {
    let jump_time = k.bridge_jump_time + 1e-6;
    let doppler_band = k.doppler_speed + k.report_delay_bound * ego.a_ego.abs();
    let doppler_yaw = ego.yaw_rate.abs() <= k.doppler_yaw;
    let mut out = Vec::with_capacity(points.len());
    for p in points {
        let (x, y, speed) = (p.x, p.y, p.speed);
        if st.selection_history {
            let recent = st.reports.entry(p.track, Ring::new);
            let jumped = recent.iter().any(|(t, rx, ry)| {
                let dt = stamp - t;
                0. < dt && dt <= jump_time && (x - rx).abs() <= k.bridge_reconnect && (y - ry).abs() >= k.bridge_jump
            });
            recent.push((stamp, x, y));
            if jumped {
                st.jumps.insert(p.track, stamp);
            }
        }
        if !(0. <= x && x <= k.far_x) || y.abs() > (if x <= k.max_x { k.max_y } else { k.far_y }) || speed.abs() > 100. {
            continue;
        }
        let (wx, wy) = (px + c * x - s * y, py + s * x + c * y);
        let distance = hypot(x, y);
        let cosine = if distance > 0. { x / distance } else { 1. };
        let explained = py_min(py_max(speed, 0.), ego.speed * (1. - cosine));
        let report = evidence(p.track).filter(|r| (r.distance - x).abs() <= 1.5);
        let moving = report.map_or(p.moving, |r| r.moving);
        let tracked = report.and_then(|r| r.tracked).unwrap_or(p.tracked);
        let residual = speed - explained;
        let doppler = cosine >= 0.707
            && (x >= k.doppler_min_x || y.abs() >= k.doppler_near_y)
            && residual.abs() > doppler_band
            && doppler_yaw;
        let doppler: i32 = if doppler { if residual > 0. { 1 } else { -1 } } else { 0 };
        if x > k.max_x
            && !moving
            && !leads.iter().any(|&(lx, ly)| (x - lx).abs() <= py_max(3., 0.2 * lx) && (y - ly).abs() <= 1.2)
        {
            continue;
        }
        out.push(Measurement {
            source: p.track,
            x: wx,
            y: wy,
            vx: speed * c,
            vy: speed * s,
            explained,
            hc: c,
            hs: s,
            radar_moving: moving,
            doppler,
            tracked,
        });
    }
    out
}

/// One map object as `_associate` reads it, in dict order.
struct Object {
    key: i64,
    has_report: bool,
    interior: bool,
    slot: Option<i64>,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    confirmed: bool,
    source: i128,
    seen: f64,
    uncertainty: f64,
    gx: f64, // Grid position: reported coordinates at the measurement stamp.
    gy: f64,
    near: Option<bool>, // abs(local[key][0]) <= 15, None without a local position.
}

/// What `_associate` reads of the map besides its state.
#[derive(Clone, Copy, Debug)]
pub struct AssociateIn {
    pub holding: bool,
    pub lane_change: bool,
    /// `self.selected`.
    pub selected: [Option<i64>; 3],
}

/// `CcncRadarMap._associate`: (measurement index, object key) in the insertion order of the Python dict.
pub fn associate(
    st: &MapState,
    k: &Consts,
    inp: AssociateIn,
    lanes: &Lanes<'_>,
    measurements: &[Measurement],
    stamp: f64,
) -> Result<Vec<(usize, i64)>> {
    let now = st.now.expect("advanced before associating");
    let frame = st.frame();
    let (cos, sin, age) = (frame.c, frame.s, now - stamp);
    let mut objects = Vec::with_capacity(st.objects.len());
    for (&key, o) in &st.objects {
        let confirmed = o.motion_confirmed;
        let offset = if confirmed { 0. } else { o.drift };
        objects.push(Object {
            key,
            has_report: o.report.is_some(),
            interior: o.interior,
            slot: o.slot,
            x: o.x,
            y: o.y,
            vx: o.vx,
            vy: o.vy,
            confirmed,
            source: o.source,
            seen: o.seen,
            uncertainty: o.uncertainty,
            gx: o.x - o.vx * age + offset * cos,
            gy: o.y - o.vy * age + offset * sin,
            near: st.local.get(&key).map(|xy| xy.0.abs() <= 15.),
        });
    }
    let near = |o: &Object| o.near.ok_or(Error::Key(o.key));
    let (id_cycle, match_distance) = (k.id_cycle, k.match_distance);

    // Insertion-ordered grid cells, each listing objects in dict order.
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (index, o) in objects.iter().enumerate() {
        grid.entry((floor(o.gx / 2.)?, floor(o.gy / 2.)?)).or_default().push(index);
    }

    let mut recent: HashMap<i128, usize> = HashMap::new();
    for (index, o) in objects.iter().enumerate() {
        let dt = stamp - o.seen;
        if 0. < dt && dt <= id_cycle && o.has_report && near(o)? {
            // `source not in recent or seen > objects[recent[source]].seen`
            let newer = match recent.get(&o.source) {
                Some(&other) => o.seen > objects[other].seen,
                None => true,
            };
            if newer {
                recent.insert(o.source, index);
            }
        }
    }

    let mut continuations: Vec<(usize, usize)> = Vec::new();
    if inp.holding {
        let resting_speed = k.resting_speed;
        for (i, ms) in measurements.iter().enumerate() {
            let Some(&index) = recent.get(&ms.source) else { continue };
            let o = &objects[index];
            let (dx, dy) = (ms.x - (o.x - o.vx * age), ms.y - (o.y - o.vy * age));
            let (forward, lateral) = (dx * cos + dy * sin, dy * cos - dx * sin);
            if inp.selected[1..].contains(&Some(o.key))
                && cos * ms.vx + sin * ms.vy > resting_speed
                && o.confirmed
                && o.interior
                && matches!(o.slot, Some(1 | 2))
                && !inp.lane_change
                && !continuations.iter().any(|&(_, k)| objects[k].key == o.key)
                && forward.abs() <= 2.
                && lateral.abs() <= 0.35
                && hypot(ms.vx - o.vx, ms.vy - o.vy) <= 6.
            {
                let (ex, ey) = frame.to_ego(ms.x, ms.y);
                if lanes.lane(ex, ey, true, false, &[])?.1 {
                    continuations.push((i, index));
                }
            }
        }
    }

    // (score, measurement index, object key, object index); sorted as the Python tuples (score, i, key).
    let mut pairs: Vec<(f64, usize, i64, usize)> = Vec::new();
    let mut matches: Vec<(f64, usize, i64, usize)> = Vec::new();
    for (i, ms) in measurements.iter().enumerate() {
        matches.clear();
        let (cell_x, cell_y) = (floor(ms.x / 2.)?, floor(ms.y / 2.)?);
        for cx in cell_x - 1..cell_x + 2 {
            for cy in cell_y - 1..cell_y + 2 {
                let Some(cell) = grid.get(&(cx, cy)) else { continue };
                for &index in cell {
                    let o = &objects[index];
                    let distance = hypot(ms.x - o.gx, ms.y - o.gy);
                    let same = ms.source == o.source;
                    let near = near(o)?;
                    let gate = if same {
                        py_min(1.2, (if near { 0.85 } else { 0.5 }) + o.uncertainty * 0.25)
                    } else {
                        match_distance
                    };
                    let speed_error = hypot(ms.vx - o.vx, ms.vy - o.vy);
                    if distance <= gate && speed_error <= (if same { 6. } else { 3. }) {
                        let score = distance + 0.04 * py_min(speed_error, 3.) - (if same { 0.1 } else { 0. });
                        matches.push((score, i, o.key, index));
                    }
                }
            }
        }
        sort(&mut matches);
        if matches.len() > 1 && matches[1].0 - matches[0].0 < 0.1 && objects[matches[0].3].source != ms.source {
            continue;
        }
        pairs.extend_from_slice(&matches);
    }
    sort(&mut pairs);

    let mut assigned = Vec::with_capacity(measurements.len());
    let mut taken_measurements = vec![false; measurements.len()];
    let mut used = vec![false; objects.len()];
    for &(i, index) in &continuations {
        assigned.push((i, objects[index].key));
        taken_measurements[i] = true;
        used[index] = true;
    }
    for &(_, i, _, index) in &pairs {
        if !taken_measurements[i] && !used[index] {
            assigned.push((i, objects[index].key));
            taken_measurements[i] = true;
            used[index] = true;
        }
    }
    Ok(assigned)
}

/// Python tuple order of (score, i, key). Scores are finite and never -0.0 (a NaN distance fails the gate,
/// and the score starts from a distance >= 0), so the total order is the Python order.
fn sort(items: &mut [(f64, usize, i64, usize)]) {
    items.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
}
