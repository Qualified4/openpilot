//! Stationary return rows and road-relative bands in the CCNC radar map.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::associate::floor;
use super::consts::Consts;
use super::lanes::{interpolate, Lanes};
use super::state::{Error, Frame, InsertionMap, MapState, Result};
use crate::pymath::{acos, cos, degrees, hypot, py_max, py_min, py_sum, radians, round_decimal};

pub use super::int_set::IntSet;

type Xy = (f64, f64);
type Candidate = (f64, f64, i64); // (*local[key], key)
pub type BandPoint = (i64, i64, f64, f64, f64, bool); // (sign, reference, road distance, offset, odometer, inside)
type RowPoint = (f64, f64, i64, bool); // (road distance, offset, key, inside)
type Bend = (i64, f64, f64); // (key, degrees, gap) before Python's round

fn index_error() -> Error {
    Error::Index("list index out of range")
}

fn divide(a: f64, b: f64) -> Result<f64> {
    if b == 0. {
        return Err(Error::ZeroDivision("float division by zero"));
    }
    Ok(a / b)
}

/// Python `a < b` decides; neither is smaller only for equal values (or NaN).
fn fcmp(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if b < a {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Python tuple order: the first unequal element decides.
fn fcmp_seq(a: &[f64], b: &[f64]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        if x != y {
            return fcmp(*x, *y);
        }
    }
    Ordering::Equal
}

fn cmp_xy(a: &Xy, b: &Xy) -> Ordering {
    fcmp_seq(&[a.0, a.1], &[b.0, b.1])
}

fn cmp_candidate(a: &Candidate, b: &Candidate) -> Ordering {
    fcmp_seq(&[a.0, a.1], &[b.0, b.1]).then(a.2.cmp(&b.2))
}

fn cmp_row(a: &RowPoint, b: &RowPoint) -> Ordering {
    fcmp_seq(&[a.0, a.1], &[b.0, b.1])
        .then(a.2.cmp(&b.2))
        .then(a.3.cmp(&b.3))
}

fn sort_floats(values: &mut [f64]) {
    values.sort_by(|a, b| fcmp(*a, *b));
}

/// Python `min(values)` / `max(values)` of a non-empty list: the first of equal values stays.
fn fold(values: impl IntoIterator<Item = f64>, f: fn(f64, f64) -> f64) -> Option<f64> {
    values.into_iter().reduce(f)
}

/// What `_walls` and `_bands` read of one map object (and `local[key]`).
struct Obj {
    x: f64,
    y: f64,
    local: Option<Xy>,
    vision: bool,
    radar_vehicle: bool, // `_radar_vehicle(obj)`
    motion_confirmed: bool,
    stationary: bool,
    uncertainty: f64,
    born: f64,
    interior: bool,
    doppler: i64,
    vision_seen: bool,
}

impl Obj {
    fn local(&self, key: i64) -> Result<Xy> {
        self.local.ok_or_else(|| Error::Key(key))
    }
}

/// `self.objects[key]`.
fn obj(objs: &HashMap<i64, Obj>, key: i64) -> Result<&Obj> {
    objs.get(&key).ok_or_else(|| Error::Key(key))
}

/// `self.local[key]`.
fn local(objs: &HashMap<i64, Obj>, key: i64) -> Result<Xy> {
    obj(objs, key)?.local(key)
}

/// The rows `_walls` hands to `_bands`: `candidates`, `inside`, and the objects as `_walls` read them (no Python
/// code runs between the two).
pub struct WallRows {
    candidates: Vec<Candidate>,
    inside: HashSet<i64>,
    objs: HashMap<i64, Obj>,
}

#[derive(Clone, Debug, Default)]
pub struct WallData {
    pub members: Vec<i64>,
    pub points: Vec<Xy>,
}

/// `Wall(tuple(keys), [anchors[k] for k in keys])`.
fn from_anchors(anchors: &InsertionMap<i64, Xy>, keys: &[i64]) -> WallData {
    WallData {
        members: keys.to_vec(),
        points: keys.iter().map(|&k| *anchors.get(&k).expect("an anchor key")).collect(),
    }
}

/// `sorted(self.to_ego(*p) for p in points)`.
fn ego_sorted(frame: Frame, points: &[Xy]) -> Vec<Xy> {
    let mut out: Vec<Xy> = points.iter().map(|&(x, y)| frame.to_ego(x, y)).collect();
    out.sort_by(cmp_xy);
    out
}

/// `sorted(keys, key=f)` for float sort keys.
fn sort_keys(keys: impl Iterator<Item = i64>, mut f: impl FnMut(i64) -> Result<f64>) -> Result<Vec<i64>> {
    let mut keyed = Vec::new();
    for k in keys {
        keyed.push((k, f(k)?));
    }
    keyed.sort_by(|a, b| fcmp(a.1, b.1));
    Ok(keyed.into_iter().map(|(k, _)| k).collect())
}

/// `_walls` up to its `_bands` call.
fn walls_core(st: &mut MapState, k: &Consts, now: f64, measured: &IntSet) -> Result<WallRows> {
    let frame = st.frame();
    let any_measured = !measured.is_empty();
    // Every object's structure is cleared; what the rest reads of it is kept.
    let mut order = Vec::with_capacity(st.objects.len());
    let mut objs = HashMap::with_capacity(st.objects.len());
    for (&key, o) in st.objects.iter_mut() {
        let info = {
            o.structure = false;
            Obj {
                x: o.x,
                y: o.y,
                local: None,
                vision: o.vision,
                radar_vehicle: o.radar_moving || o.doppler_confirmed,
                motion_confirmed: o.motion_confirmed,
                stationary: o.stationary,
                uncertainty: o.uncertainty,
                born: o.born,
                interior: o.interior,
                doppler: o.doppler,
                vision_seen: o.vision_seen,
            }
        };
        let info = Obj {
            local: st.local.get(&key).copied(),
            ..info
        };
        order.push(key);
        objs.insert(key, info);
    }

    // Anchors stay at their measured world positions; a member with radar vehicle evidence leaves its wall.
    let mut walls: Vec<WallData> = Vec::new();
    for WallData { members, points } in &st.walls {
        let mut support = WallData::default();
        for (&key, &anchor) in members.iter().zip(points) {
            let Some(o) = objs.get(&key) else { continue };
            if o.vision || o.radar_vehicle {
                continue;
            }
            let (ex, ey) = frame.to_ego(anchor.0, anchor.1);
            if !(k.min_x <= ex && ex <= k.keep_x && ey.abs() <= k.max_y) {
                continue;
            }
            if hypot(o.x - anchor.0, o.y - anchor.1) <= 1.5 {
                support.members.push(key);
                support.points.push(anchor);
            }
        }
        if support.members.len() >= 2 {
            walls.push(support);
        }
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    if any_measured {
        for &key in &order {
            let o = &objs[&key];
            if o.stationary && !o.motion_confirmed && o.uncertainty <= 0.6 && !o.vision && now - o.born >= 0.3 {
                let (x, y) = o.local(key)?;
                candidates.push((x, y, key));
            }
        }
    }
    candidates.sort_by(cmp_candidate);
    let inside: HashSet<i64> = order.iter().copied().filter(|key| objs[key].interior).collect();

    // New straight rows on each side.
    for sign in [-1., 1.] {
        let points: Vec<Candidate> = candidates.iter().copied().filter(|p| p.1 * sign > 0.2).collect();
        let mut used: HashSet<i64> = HashSet::new();
        for start in 0..points.len().saturating_sub(2) {
            if used.contains(&points[start].2) {
                continue;
            }
            let mut row = vec![points[start]];
            for &point in &points[start + 1..] {
                let gap = point.0 - row[row.len() - 1].0;
                if gap > k.wall_max_gap {
                    break;
                }
                if gap < 0.6 || used.contains(&point.2) {
                    continue;
                }
                let (x0, y0) = (row[0].0, row[0].1);
                let slope = divide(point.1 - y0, point.0 - x0)?;
                if slope.abs() > 0.4
                    || row[1..]
                        .iter()
                        .any(|&(x, y, _)| (y - y0 - slope * (x - x0)).abs() > 0.25)
                {
                    continue;
                }
                row.push(point);
            }
            if row.iter().any(|p| inside.contains(&p.2))
                && ((row.len() as f64) < k.dense_row_points || row[row.len() - 1].0 - row[0].0 < k.dense_row_span)
            {
                row.retain(|p| !inside.contains(&p.2));
            }
            if row.len() < 3
                || row[row.len() - 1].0 - row[0].0 < 4. - 1e-5
                || row.iter().filter(|p| measured.contains(&p.2)).count() < 2
            {
                continue;
            }
            let members: Vec<i64> = row.iter().map(|p| p.2).collect();
            used.extend(&members);
            let mut anchors = InsertionMap::default();
            for &key in &members {
                let o = obj(&objs, key)?;
                anchors.insert(key, (o.x, o.y));
            }
            let owner = walls
                .iter()
                .position(|w| members.iter().filter(|key| w.members.contains(key)).count() >= 2);
            if let Some(j) = owner {
                for (&key, &point) in walls[j].members.iter().zip(&walls[j].points) {
                    anchors.insert(key, point);
                }
            }
            let mut keys = sort_keys(anchors.keys(), |key| Ok(local(&objs, key)?.0))?;
            keys.truncate(16);
            let wall = from_anchors(&anchors, &keys);
            if let Some(j) = owner {
                walls[j] = wall;
            } else if (walls.len() as f64) < k.max_walls {
                walls.push(wall);
            }
        }
    }

    // A fresh resting return on the forward continuation of an established wall joins it.
    let mut limit: Option<f64> = None;
    let mut bends = Vec::new();
    for wall in walls.iter_mut() {
        let mut line: Vec<Candidate> = wall
            .members
            .iter()
            .zip(&wall.points)
            .map(|(&key, &(x, y))| {
                let (ex, ey) = frame.to_ego(x, y);
                (ex, ey, key)
            })
            .collect();
        line.sort_by(cmp_candidate);
        if line.len() < 3 {
            continue;
        }
        let ((x0, y0, _), (mut x1, mut y1, _)) = (line[0], line[line.len() - 1]);
        let slope = if x1 - x0 >= 1. { (y1 - y0) / (x1 - x0) } else { 0. };
        for &(x, y, key) in &candidates {
            if measured.contains(&key)
                && !wall.members.contains(&key)
                && 0.6 <= x - x1
                && x - x1 <= k.wall_continue
                && (y - y1 - slope * (x - x1)).abs() <= 0.25
            {
                let o = obj(&objs, key)?;
                wall.members.push(key);
                wall.points.push((o.x, o.y));
                (x1, y1) = (x, y);
            }
        }
        if k.wall_bend > 0. {
            let limit = match limit {
                Some(limit) => limit,
                None => *limit.insert(cos(radians(k.wall_bend))?),
            };
            bend(
                wall,
                &candidates,
                measured,
                &objs,
                frame,
                limit,
                k.wall_max_gap,
                &mut bends,
            )?;
        }
        if wall.members.len() > 16 {
            let mut keep: Vec<(usize, f64)> = wall
                .points
                .iter()
                .enumerate()
                .map(|(i, &(x, y))| (i, -frame.to_ego(x, y).0))
                .collect();
            keep.sort_by(|a, b| fcmp(a.1, b.1));
            keep.truncate(16);
            *wall = WallData {
                members: keep.iter().map(|&(i, _)| wall.members[i]).collect(),
                points: keep.iter().map(|&(i, _)| wall.points[i]).collect(),
            };
        }
    }
    if any_measured {
        let outside: Vec<Candidate> = candidates.iter().copied().filter(|p| !inside.contains(&p.2)).collect();
        walls = bridge_walls(walls, &outside, measured, &objs, frame)?;
    }

    for wall in &walls {
        for key in &wall.members {
            if let Some(o) = st.objects.get_mut(key) {
                o.structure = true;
            }
        }
    }
    st.wall_geometry = walls.iter().map(|wall| ego_sorted(frame, &wall.points)).collect();
    st.walls = walls;
    st.bends = bends
        .into_iter()
        .map(|(key, angle, gap)| (key, round_decimal(angle, 1), round_decimal(gap, 2)))
        .collect();
    Ok(WallRows {
        candidates,
        inside,
        objs,
    })
}

/// `_bend(wall, candidates, measured)`: continue a wall from either end through returns that follow its end
/// segment within WALL_BEND (`limit` is its cosine).
#[allow(clippy::too_many_arguments)]
fn bend(
    wall: &mut WallData,
    candidates: &[Candidate],
    measured: &IntSet,
    objs: &HashMap<i64, Obj>,
    frame: Frame,
    limit: f64,
    max_gap: f64,
    bends: &mut Vec<Bend>,
) -> Result<()> {
    for forward in [true, false] {
        let line = ego_sorted(frame, &wall.points);
        let n = line.len();
        if n < 2 {
            return Ok(());
        }
        let (mut a, mut b) = if forward {
            (line[n - 2], line[n - 1])
        } else {
            (line[1], line[0])
        };
        while wall.members.len() < 24 {
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let segment = hypot(dx, dy);
            if segment < 0.3 {
                break;
            }
            let mut best: Option<(f64, f64, f64, i64)> = None;
            for &(x, y, key) in candidates {
                if wall.members.contains(&key) || !measured.contains(&key) {
                    continue;
                }
                let (ex, ey) = (x - b.0, y - b.1);
                let gap = hypot(ex, ey);
                if 0.6 <= gap
                    && gap <= max_gap
                    && divide(ex * dx + ey * dy, gap * segment)? >= limit
                    && best.is_none_or(|best| gap < best.0)
                {
                    best = Some((gap, x, y, key));
                }
            }
            let Some((gap, x, y, key)) = best else { break };
            let angle = degrees(acos(py_min(
                1.,
                divide((x - b.0) * dx + (y - b.1) * dy, gap * segment)?,
            ))?);
            bends.push((key, angle, gap));
            let o = obj(objs, key)?;
            wall.members.push(key);
            wall.points.push((o.x, o.y));
            (a, b) = (b, (x, y));
        }
    }
    Ok(())
}

/// `_bridge_walls(candidates, measured)`: join adjacent established rows through observed, aligned stationary
/// returns.
fn bridge_walls(
    walls: Vec<WallData>,
    candidates: &[Candidate],
    measured: &IntSet,
    objs: &HashMap<i64, Obj>,
    frame: Frame,
) -> Result<Vec<WallData>> {
    let mut result: Vec<(WallData, Vec<Xy>)> = Vec::new();
    for sign in [-1., 1.] {
        let mut rows: Vec<(f64, usize, Vec<Xy>)> = Vec::new();
        for (i, wall) in walls.iter().enumerate() {
            let local = ego_sorted(frame, &wall.points);
            let first = *local.first().ok_or_else(index_error)?;
            if first.1 * sign > 0. {
                rows.push((first.0, i, local));
            }
        }
        rows.sort_by(|a, b| fcmp(a.0, b.0));
        for (_, i, local) in rows {
            let wall = &walls[i];
            let joins = match result.last() {
                None => false,
                Some((_, left)) => left.first().ok_or_else(index_error)?.1 * sign > 0.,
            };
            if !joins {
                result.push((wall.clone(), local));
                continue;
            }
            let (prior, left) = result.last().expect("checked above");
            let (a, b) = (left[left.len() - 1], local[0]);
            let gap = b.0 - a.0;
            if gap <= 0.1 {
                result.push((wall.clone(), local));
                continue;
            }
            let before = *left
                .len()
                .checked_sub(2)
                .and_then(|i| left.get(i))
                .ok_or_else(index_error)?;
            let after = *local.get(1).ok_or_else(index_error)?;
            let (left_dx, right_dx) = (a.0 - before.0, after.0 - b.0);
            if py_min(left_dx, right_dx) <= 0.1 {
                result.push((wall.clone(), local));
                continue;
            }
            let slope = (b.1 - a.1) / gap;
            if slope.abs() > 0.4
                || (slope - (a.1 - before.1) / left_dx).abs() > 0.25
                || (slope - (after.1 - b.1) / right_dx).abs() > 0.25
            {
                result.push((wall.clone(), local));
                continue;
            }
            let middle: Vec<Candidate> = candidates
                .iter()
                .copied()
                .filter(|p| a.0 + 0.1 < p.0 && p.0 < b.0 - 0.1 && (p.1 - a.1 - slope * (p.0 - a.0)).abs() <= 0.25)
                .collect();
            let gap_limit = py_min(6., py_max(3., 2. * py_max(left_dx, right_dx)));
            let chain: Vec<f64> = [a.0]
                .into_iter()
                .chain(middle.iter().map(|p| p.0))
                .chain([b.0])
                .collect();
            if !middle.iter().any(|p| measured.contains(&p.2)) || chain.windows(2).any(|w| w[1] - w[0] > gap_limit) {
                result.push((wall.clone(), local));
                continue;
            }
            let mut anchors = InsertionMap::default();
            for p in &middle {
                let o = obj(objs, p.2)?;
                anchors.insert(p.2, (o.x, o.y));
            }
            for w in [wall, prior] {
                for (&key, &point) in w.members.iter().zip(&w.points) {
                    anchors.insert(key, point);
                }
            }
            let mut keys = sort_keys(anchors.keys(), |key| {
                let (x, y) = *anchors.get(&key).expect("an anchor key");
                Ok(frame.c * x + frame.s * y)
            })?;
            keys.truncate(16);
            let joined = from_anchors(&anchors, &keys);
            let joined_local = ego_sorted(frame, &joined.points);
            *result.last_mut().expect("checked above") = (joined, joined_local);
        }
    }
    Ok(result.into_iter().map(|(wall, _)| wall).collect())
}

/// `_band_row(points)`.
fn band_row(mut points: Vec<RowPoint>, dense_points: f64, dense_span: f64) -> Vec<RowPoint> {
    if points.iter().any(|p| p.3) {
        let high = fold(points.iter().map(|p| p.0), py_max).expect("not empty");
        let low = fold(points.iter().map(|p| p.0), py_min).expect("not empty");
        if (points.len() as f64) < dense_points || high - low < dense_span {
            points.retain(|p| !p.3);
        }
    }
    points.sort_by(cmp_row);
    points
}

#[derive(Clone, Debug)]
pub struct BandData {
    pub id: u64,
    pub sign: i64,
    pub reference: i64,
    pub offset: f64,
    pub members: InsertionMap<i64, f64>,
    pub supported: f64,
    pub spacing: f64,
}

impl BandData {
    pub fn span(&self) -> Option<Xy> {
        let values = || self.members.iter().map(|(_, value)| *value);
        Some((fold(values(), py_min)?, fold(values(), py_max)?))
    }
}

/// Update walls and bands. Return the bands retired in this frame, including their final field values.
pub fn update(st: &mut MapState, k: &Consts, now: f64, lanes: &Lanes, measured: &IntSet) -> Result<Vec<BandData>> {
    let rows = walls_core(st, k, now, measured)?;
    bands_core(st, k, lanes, &rows, measured)
}

/// `_bands`, appending band lines to `wall_geometry`.
fn bands_core(
    st: &mut MapState,
    k: &Consts,
    lanes: &Lanes,
    rows: &WallRows,
    measured: &IntSet,
) -> Result<Vec<BandData>> {
    let odometer = st.odometer;
    let (band_bin, band_memory) = (k.band_bin, k.band_memory);
    let (band_prediction, band_gap) = (k.band_prediction, k.band_gap);
    let (band_fold, wall_continue) = (k.band_fold, k.wall_continue);
    let (dense_points, dense_span) = (k.dense_row_points, k.dense_row_span);
    let max_walls = k.max_walls;
    let objs = &rows.objs;
    // {sign: (reference, geometry)}, sign 1 first.
    let mut lines: Vec<(i64, i64, &[Xy])> = Vec::with_capacity(2);
    for (sign, order) in [(1, [1, 0]), (-1, [2, 3])] {
        for i in order {
            if lanes.probability(i).is_some_and(|p| p >= 0.1) && lanes.geometry(i).is_some_and(|g| g.len() >= 2) {
                lines.push((sign, i as i64, lanes.geometry(i).expect("checked")));
                break;
            }
        }
    }
    let line_of = |sign: i64| {
        lines
            .iter()
            .find(|l| l.0 == sign)
            .map(|&(_, reference, points)| (reference, points))
    };

    let bands = &mut st.bands;
    let band_points = &mut st.band_points;
    let band_early = &mut st.band_early;
    if !measured.is_empty() {
        band_early.intersect(st.objects.keys().copied());
        let mut early = Vec::new();
        for key in band_early.iter() {
            if measured.contains(&key) && band_points.get(&key).is_some() {
                let (x, y) = local(objs, key)?;
                early.push((x, y, key));
            }
        }
        for &(x, y, key) in rows.candidates.iter().chain(&early) {
            let sign = if y > 0. { 1 } else { -1 };
            let line = if y.abs() > 0.2 { line_of(sign) } else { None };
            let Some((reference, points)) = line else { continue };
            let Some(base) = interpolate(points, x) else { continue };
            let old = band_points.get(&key).map(|p| p.4);
            let since = match old {
                Some(since) if !measured.contains(&key) => since,
                _ => odometer,
            };
            band_points.insert(
                key,
                (
                    sign,
                    reference,
                    odometer + x,
                    y - base,
                    since,
                    rows.inside.contains(&key),
                ),
            );
        }
        // A return is forgotten after BAND_MEMORY metres without a measurement, or at once with vehicle evidence.
        band_points.retain(|key, p| {
            !(odometer - p.4 > band_memory || objs.get(&key).is_some_and(|o| o.vision || o.radar_vehicle))
        });

        let mut claimed: HashSet<i64> = HashSet::new();
        for band in bands.iter_mut() {
            let mut points: InsertionMap<i64, BandPoint> = InsertionMap::default();
            for (key, &p) in band_points.iter() {
                if p.0 == band.sign && p.1 == band.reference && !claimed.contains(&key) {
                    points.insert(key, p);
                }
            }
            // The model's line moves between frames and all offsets move with it: follow the members.
            let mut current: Vec<f64> = band
                .members
                .keys()
                .filter_map(|key| points.get(&key).map(|p| p.3))
                .collect();
            sort_floats(&mut current);
            if current.is_empty() {
                band.members = InsertionMap::default();
                continue;
            }
            band.offset = current[current.len() / 2];
            let row: Vec<RowPoint> = points
                .iter()
                .filter(|(_, p)| (p.3 - band.offset).abs() <= band_bin)
                .map(|(key, p)| (p.2, p.3, key, p.5))
                .collect();
            let row = band_row(row, dense_points, dense_span);
            let positions: Vec<f64> = row
                .iter()
                .filter(|p| band.members.get(&p.2).is_some())
                .map(|p| p.0)
                .collect();
            let (Some(mut low), Some(mut high)) = (
                fold(positions.iter().copied(), py_min),
                fold(positions.iter().copied(), py_max),
            ) else {
                band.members = InsertionMap::default();
                continue;
            };
            // The band continues through returns within WALL_CONTINUE of its ends, as a straight row does.
            for p in &row {
                if high < p.0 && p.0 <= high + wall_continue {
                    high = p.0;
                }
            }
            for p in row.iter().rev() {
                if low - wall_continue <= p.0 && p.0 < low {
                    low = p.0;
                }
            }
            let mut members = InsertionMap::default();
            for p in &row {
                if low <= p.0 && p.0 <= high {
                    members.insert(p.2, p.0);
                }
            }
            band.members = members;
            claimed.extend(band.members.keys());
            if band.members.keys().any(|key| measured.contains(&key)) {
                band.supported = odometer;
            }
            let mut positions: Vec<f64> = band.members.iter().map(|(_, v)| *v).collect();
            sort_floats(&mut positions);
            let mut gaps: Vec<f64> = positions
                .windows(2)
                .map(|w| w[1] - w[0])
                .filter(|&gap| gap >= 0.6)
                .collect();
            sort_floats(&mut gaps);
            band.spacing = if gaps.is_empty() { 0. } else { gaps[gaps.len() / 2] };
            // The next post is expected one spacing beyond the last (or two, with one post missed), at the band's
            // offset. A fresh return there without any vehicle evidence is that post: it joins now.
            let Some((reference, line)) = line_of(band.sign) else {
                continue;
            };
            if band.spacing <= 0. || reference != band.reference {
                continue;
            }
            let fresh = measured.difference(band_points);
            for key in sort_keys(fresh.iter(), |key| Ok(local(objs, key)?.0))? {
                let o = obj(objs, key)?;
                let (x, y) = o.local(key)?;
                let base = if y * band.sign as f64 > 0.2 {
                    interpolate(line, x)
                } else {
                    None
                };
                let Some(base) = base else { continue };
                if o.motion_confirmed || o.doppler != 0 || o.vision || o.vision_seen || o.interior {
                    continue;
                }
                let ahead = odometer + x - positions[positions.len() - 1];
                if (y - base - band.offset).abs() <= band_prediction
                    && py_min((ahead - band.spacing).abs(), (ahead - 2. * band.spacing).abs()) <= band_prediction
                {
                    band_points.insert(
                        key,
                        (band.sign, band.reference, odometer + x, y - base, odometer, false),
                    );
                    band_early.insert(key);
                    band.members.insert(key, odometer + x);
                    claimed.insert(key);
                    positions.push(odometer + x);
                }
            }
        }

        let walls_len = st.walls.len();
        for &(sign, reference, _) in &lines {
            let mut cells: InsertionMap<i64, Vec<RowPoint>> = InsertionMap::default();
            for (key, &p) in band_points.iter() {
                if p.0 == sign && p.1 == reference && !claimed.contains(&key) {
                    let cell = floor(p.3 / band_bin)?;
                    match cells.get_mut(&cell) {
                        Some(list) => list.push((p.2, p.3, key, p.5)),
                        None => cells.insert(cell, vec![(p.2, p.3, key, p.5)]),
                    }
                }
            }
            let mut windows = Vec::with_capacity(cells.len());
            for (cell, list) in cells.iter() {
                let next = cell.checked_add(1).ok_or_else(|| Error::Overflow("cell"))?;
                let mut window = list.clone();
                window.extend(cells.get(&next).into_iter().flatten().copied());
                windows.push(window);
            }
            windows.sort_by_key(|w| std::cmp::Reverse(w.len())); // sorted(key=len, reverse=True): stable
            for window in windows {
                let window: Vec<RowPoint> = window.into_iter().filter(|p| !claimed.contains(&p.2)).collect();
                let mut run: Vec<RowPoint> = Vec::new();
                for p in band_row(window, dense_points, dense_span)
                    .into_iter()
                    .map(Some)
                    .chain([None])
                {
                    if let Some(p) = p {
                        if run.is_empty() || p.0 - run[run.len() - 1].0 <= band_gap {
                            run.push(p);
                            continue;
                        }
                    }
                    // Three posts (returns nearer than .6m are one post) no farther apart than in a straight row,
                    // over 4m.
                    let posts = if run.is_empty() {
                        0
                    } else {
                        1 + run.windows(2).filter(|w| w[1].0 - w[0].0 >= 0.6).count()
                    };
                    if posts >= 3
                        && run[run.len() - 1].0 - run[0].0 >= 4. - 1e-5
                        && run.iter().any(|q| measured.contains(&q.2))
                        && ((bands.len() + walls_len) as f64) < max_walls
                    {
                        claimed.extend(run.iter().map(|q| q.2));
                        let mut members = InsertionMap::default();
                        for q in &run {
                            members.insert(q.2, q.0);
                        }
                        let offset = py_sum(run.iter().map(|q| q.1)).expect("three posts") / run.len() as f64;
                        let id = st.next_band;
                        st.next_band += 1;
                        bands.push(BandData {
                            id,
                            sign,
                            reference,
                            offset,
                            members,
                            supported: odometer,
                            spacing: 0.,
                        });
                    }
                    run = p.into_iter().collect();
                }
            }
        }
    }

    // Fold a band after driving three of its spacings without a measured member, and never while ego stands.
    let mut retired = Vec::new();
    bands.retain(|band| {
        let keep = !band.members.is_empty() && odometer - band.supported <= py_max(3. * band.spacing, band_fold);
        if !keep {
            retired.push(band.clone());
        }
        keep
    });
    for band in bands {
        let Some((reference, line)) = line_of(band.sign) else {
            continue;
        };
        if reference != band.reference {
            continue;
        }
        let (first, last) = band.span().expect("members");
        let (first, last) = (first - odometer, last - odometer);
        let mut geometry: Vec<Xy> = Vec::new();
        for x in [first, last] {
            if let Some(y) = interpolate(line, x) {
                geometry.push((x, y + band.offset));
            }
        }
        geometry.extend(
            line.iter()
                .filter(|p| first < p.0 && p.0 < last)
                .map(|&(x, y)| (x, y + band.offset)),
        );
        geometry.sort_by(cmp_xy);
        if geometry.len() >= 2 && geometry[geometry.len() - 1].0 > geometry[0].0 {
            st.wall_geometry.push(geometry);
        }
        for key in band.members.keys() {
            if let Some(o) = objs.get(&key) {
                if !o.vision && !o.radar_vehicle {
                    st.objects.get_mut(&key).expect("an object of this update").structure = true;
                }
            }
        }
    }
    Ok(retired)
}

/// Extend established walls over a missing post on the offline hybrid display path.
pub fn extend_observed(st: &mut MapState, eligible: &[i64]) -> Result<()> {
    let frame = st.frame();
    for wall in &mut st.walls {
        let line = ego_sorted(frame, &wall.points);
        if line.len() < 3 {
            continue;
        }
        let mut gaps: Vec<f64> = line
            .windows(2)
            .map(|p| p[1].0 - p[0].0)
            .filter(|gap| (0.6..=3.).contains(gap))
            .collect();
        sort_floats(&mut gaps);
        let Some(&largest_gap) = gaps.last() else { continue };
        let ((x0, y0), (x1, y1)) = (line[0], line[line.len() - 1]);
        if x1 - x0 < 1. {
            continue;
        }
        let slope = (y1 - y0) / (x1 - x0);
        for key in eligible {
            let o = st.objects.get_mut(key).ok_or(Error::Key(*key))?;
            let now = st.now.ok_or(Error::Value("map not advanced"))?;
            if wall.members.contains(key) || !o.stationary || now - o.born < 0.3 || now - o.seen > 0.15 {
                continue;
            }
            let (x, y) = *st.local.get(key).ok_or(Error::Key(*key))?;
            let gap = if x < x0 {
                x0 - x
            } else if x > x1 {
                x - x1
            } else {
                0.
            };
            if gap < 0.6
                || gap > 2. * largest_gap + 0.35
                || !gaps
                    .iter()
                    .any(|spacing| py_min((gap - spacing).abs(), (gap - 2. * spacing).abs()) <= 0.35)
                || (y - y0 - slope * (x - x0)).abs() > 0.25
                || wall.members.len() >= 16
            {
                continue;
            }
            wall.members.push(*key);
            wall.points.push((o.x, o.y));
            o.structure = true;
        }
    }
    st.wall_geometry = st.walls.iter().map(|wall| ego_sorted(frame, &wall.points)).collect();
    Ok(())
}
