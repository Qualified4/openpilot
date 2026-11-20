//! `CcncRadarMap._walls` (with `_bend`, `_bridge_walls`) and `_bands` (with `_band_row`)
//! (hyundaicanfd_ccnc_radar_map.py).
//!
//! The Python methods stay the reference. `Wall` and `Band` stay Python dataclasses: their fields are read at the
//! start and new objects (walls) or field values (bands kept from the last update) are written at the end, as the
//! Python code leaves them. Dict orders (`band_points`, `Band.members`, the cells) are kept with `Ordered`. The sets
//! whose order changes a result (`band_early`, `measured.difference(band_points)`) are Python sets, iterated and
//! changed through their own methods. Sorts compare with `==`/`<` as Python does, and are stable, as Timsort is.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyIndexError, PyKeyError, PyOverflowError, PyZeroDivisionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};

use crate::consts::Consts;
use crate::display_filter::empty_object;
use crate::guard;
use crate::lanes::{interpolate, LaneCache, Lanes};
use crate::map_object::MapObject;
use crate::observe::Frame;
use crate::pymath::{acos, cos, degrees, hypot, py_max, py_min, py_sum, radians};
use crate::radar_map::floor;

type Xy = (f64, f64);
type Candidate = (f64, f64, i64); // (*local[key], key)
type BandPoint = (i64, i64, f64, f64, f64, bool); // (sign, reference, road distance, offset, odometer, inside)
type RowPoint = (f64, f64, i64, bool); // (road distance, offset, key, inside)
type Bend = (i64, f64, f64); // (key, degrees, gap) before Python's round

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

fn index_error() -> PyErr {
    PyIndexError::new_err("list index out of range")
}

fn divide(a: f64, b: f64) -> PyResult<f64> {
    if b == 0. {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
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
    fcmp_seq(&[a.0, a.1], &[b.0, b.1]).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3))
}

fn sort_floats(values: &mut [f64]) {
    values.sort_by(|a, b| fcmp(*a, *b));
}

/// Python `min(values)` / `max(values)` of a non-empty list: the first of equal values stays.
fn fold(values: impl IntoIterator<Item = f64>, f: fn(f64, f64) -> f64) -> Option<f64> {
    values.into_iter().reduce(f)
}

/// A dict with int keys: insertion order, an existing key keeps its place.
struct Ordered<V> {
    items: Vec<(i64, V)>,
    index: HashMap<i64, usize>,
}

impl<V> Ordered<V> {
    fn new() -> Self {
        Self { items: Vec::new(), index: HashMap::new() }
    }

    fn get(&self, key: i64) -> Option<&V> {
        self.index.get(&key).map(|&i| &self.items[i].1)
    }

    fn get_mut(&mut self, key: i64) -> Option<&mut V> {
        self.index.get(&key).map(|&i| &mut self.items[i].1)
    }

    fn contains(&self, key: i64) -> bool {
        self.index.contains_key(&key)
    }

    fn insert(&mut self, key: i64, value: V) {
        match self.index.get(&key) {
            Some(&i) => self.items[i].1 = value,
            None => {
                self.index.insert(key, self.items.len());
                self.items.push((key, value));
            }
        }
    }

    fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.items.iter().map(|(k, _)| *k)
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `for key in [k for k, v in d.items() if drop(k, v)]: del d[key]`.
    fn remove_where(&mut self, mut drop: impl FnMut(i64, &V) -> bool) {
        self.items.retain(|(k, v)| !drop(*k, v));
        self.index = self.items.iter().enumerate().map(|(i, (k, _))| (*k, i)).collect();
    }
}

/// What `_walls` and `_bands` read of one map object (and `local[key]`), and the object itself for `structure`.
struct Obj {
    obj: Py<MapObject>,
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
    fn local(&self, key: i64) -> PyResult<Xy> {
        self.local.ok_or_else(|| PyKeyError::new_err(key))
    }
}

/// `self.objects[key]`.
fn obj(objs: &HashMap<i64, Obj>, key: i64) -> PyResult<&Obj> {
    objs.get(&key).ok_or_else(|| PyKeyError::new_err(key))
}

/// `self.local[key]`.
fn local(objs: &HashMap<i64, Obj>, key: i64) -> PyResult<Xy> {
    obj(objs, key)?.local(key)
}

/// The rows `_walls` hands to `_bands`: `candidates`, `inside`, and the objects as `_walls` read them (no Python
/// code runs between the two).
#[pyclass(frozen, module = "ccnc_core")]
pub struct WallRows {
    candidates: Vec<Candidate>,
    inside: HashSet<i64>,
    objs: HashMap<i64, Obj>,
}

#[derive(Clone, Default)]
struct WallData {
    members: Vec<i64>,
    points: Vec<Xy>,
}

/// `Wall(tuple(keys), [anchors[k] for k in keys])`.
fn from_anchors(anchors: &Ordered<Xy>, keys: &[i64]) -> WallData {
    WallData { members: keys.to_vec(), points: keys.iter().map(|&k| *anchors.get(k).expect("an anchor key")).collect() }
}

/// `sorted(self.to_ego(*p) for p in points)`.
fn ego_sorted(frame: Frame, points: &[Xy]) -> Vec<Xy> {
    let mut out: Vec<Xy> = points.iter().map(|&(x, y)| frame.to_ego(x, y)).collect();
    out.sort_by(cmp_xy);
    out
}

/// `sorted(keys, key=f)` for float sort keys.
fn sort_keys(keys: impl Iterator<Item = i64>, mut f: impl FnMut(i64) -> PyResult<f64>) -> PyResult<Vec<i64>> {
    let mut keyed = Vec::new();
    for k in keys {
        keyed.push((k, f(k)?));
    }
    keyed.sort_by(|a, b| fcmp(a.1, b.1));
    Ok(keyed.into_iter().map(|(k, _)| k).collect())
}

/// `_walls(measured)` up to its `_bands` call: writes `walls`, `wall_geometry` and `structure`, and returns the rows
/// for `bands` and the bends as (key, degrees, gap) before Python's `round`.
#[pyfunction]
pub fn walls<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    measured: &Bound<'py, PyAny>,
    wall_type: &Bound<'py, PyAny>,
) -> PyResult<(WallRows, Vec<Bend>)> {
    guard(|| {
        let k = Consts::read(radar_map)?;
        let now = attr_f64(radar_map, intern!(py, "now"))?;
        let frame = Frame::read(radar_map)?;
        let (rows, bends, _) = walls_core(py, radar_map, &k, now, frame, measured, wall_type)?;
        Ok((rows, bends))
    })
}

/// `measured` as Rust keys, and whether it is truthy.
fn read_measured(measured: &Bound<'_, PyAny>) -> PyResult<HashSet<i64>> {
    let mut out = HashSet::new();
    for key in measured.try_iter()? {
        out.insert(key?.extract()?);
    }
    Ok(out)
}

/// `_walls` up to its `_bands` call; also returns `wall_geometry` as written.
#[allow(clippy::type_complexity)]
pub(crate) fn walls_core<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    k: &Consts,
    now: f64,
    frame: Frame,
    measured_obj: &Bound<'py, PyAny>,
    wall_type: &Bound<'py, PyAny>,
) -> PyResult<(WallRows, Vec<Bend>, Vec<Vec<Xy>>)> {
    let m = radar_map;
    let measured = read_measured(measured_obj)?;
    let any_measured = measured_obj.is_truthy()?;

    // Every object's structure is cleared; what the rest reads of it is kept.
    let objects = m.getattr(intern!(py, "objects"))?.cast_into::<PyDict>()?;
    let local_dict = m.getattr(intern!(py, "local"))?.cast_into::<PyDict>()?;
    let mut order = Vec::with_capacity(objects.len());
    let mut objs = HashMap::with_capacity(objects.len());
    for (key, value) in objects.iter() {
        let key: i64 = key.extract()?;
        let bound = value.cast_into::<MapObject>()?;
        let info = {
            let mut o = bound.try_borrow_mut()?;
            o.structure = false;
            Obj {
                obj: bound.clone().unbind(),
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
        let info = Obj { local: local_dict.get_item(key)?.map(|xy| xy.extract()).transpose()?, ..info };
        order.push(key);
        objs.insert(key, info);
    }

    // Anchors stay at their measured world positions; a member with radar vehicle evidence leaves its wall.
    let mut walls: Vec<WallData> = Vec::new();
    for wall in m.getattr(intern!(py, "walls"))?.try_iter()? {
        let wall = wall?;
        let members: Vec<i64> = wall.getattr(intern!(py, "members"))?.extract()?;
        let points: Vec<Xy> = wall.getattr(intern!(py, "points"))?.extract()?;
        let mut support = WallData::default();
        for (&key, &anchor) in members.iter().zip(&points) {
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
                if slope.abs() > 0.4 || row[1..].iter().any(|&(x, y, _)| (y - y0 - slope * (x - x0)).abs() > 0.25) {
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
            let mut anchors = Ordered::new();
            for &key in &members {
                let o = obj(&objs, key)?;
                anchors.insert(key, (o.x, o.y));
            }
            let owner = walls.iter().position(|w| members.iter().filter(|key| w.members.contains(key)).count() >= 2);
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
            bend(wall, &candidates, &measured, &objs, frame, limit, k.wall_max_gap, &mut bends)?;
        }
        if wall.members.len() > 16 {
            let mut keep: Vec<(usize, f64)> =
                wall.points.iter().enumerate().map(|(i, &(x, y))| (i, -frame.to_ego(x, y).0)).collect();
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
        walls = bridge_walls(walls, &outside, &measured, &objs, frame)?;
    }

    let wall_list = PyList::empty(py);
    let geometry = PyList::empty(py);
    let mut lines = Vec::with_capacity(walls.len());
    for wall in &walls {
        for key in &wall.members {
            if let Some(o) = objs.get(key) {
                o.obj.bind(py).try_borrow_mut()?.structure = true;
            }
        }
        let drawing = empty_object(wall_type)?;
        drawing.setattr(intern!(py, "members"), PyTuple::new(py, wall.members.iter().copied())?)?;
        drawing.setattr(intern!(py, "points"), PyList::new(py, wall.points.iter().copied())?)?;
        wall_list.append(drawing)?;
        let line = ego_sorted(frame, &wall.points);
        geometry.append(PyList::new(py, line.iter().copied())?)?;
        lines.push(line);
    }
    m.setattr(intern!(py, "walls"), wall_list)?;
    m.setattr(intern!(py, "wall_geometry"), geometry)?;
    Ok((WallRows { candidates, inside, objs }, bends, lines))
}

/// `_bend(wall, candidates, measured)`: continue a wall from either end through returns that follow its end
/// segment within WALL_BEND (`limit` is its cosine).
#[allow(clippy::too_many_arguments)]
fn bend(
    wall: &mut WallData,
    candidates: &[Candidate],
    measured: &HashSet<i64>,
    objs: &HashMap<i64, Obj>,
    frame: Frame,
    limit: f64,
    max_gap: f64,
    bends: &mut Vec<Bend>,
) -> PyResult<()> {
    for forward in [true, false] {
        let line = ego_sorted(frame, &wall.points);
        let n = line.len();
        if n < 2 {
            return Ok(());
        }
        let (mut a, mut b) = if forward { (line[n - 2], line[n - 1]) } else { (line[1], line[0]) };
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
            let angle = degrees(acos(py_min(1., divide((x - b.0) * dx + (y - b.1) * dy, gap * segment)?))?);
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
    measured: &HashSet<i64>,
    objs: &HashMap<i64, Obj>,
    frame: Frame,
) -> PyResult<Vec<WallData>> {
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
            let before = *left.len().checked_sub(2).and_then(|i| left.get(i)).ok_or_else(index_error)?;
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
            let chain: Vec<f64> = [a.0].into_iter().chain(middle.iter().map(|p| p.0)).chain([b.0]).collect();
            if !middle.iter().any(|p| measured.contains(&p.2)) || chain.windows(2).any(|w| w[1] - w[0] > gap_limit) {
                result.push((wall.clone(), local));
                continue;
            }
            let mut anchors = Ordered::new();
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
                let (x, y) = *anchors.get(key).expect("an anchor key");
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

struct BandData<'py> {
    object: Option<Bound<'py, PyAny>>, // None for a band made in this call
    sign: i64,
    reference: i64,
    offset: f64,
    members: Ordered<f64>,
    supported: f64,
    spacing: f64,
}

impl BandData<'_> {
    fn members_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (key, value) in &self.members.items {
            d.set_item(key, value)?;
        }
        Ok(d)
    }
}

/// `_bands(candidates, measured, inside)` with the rows `walls` returned.
#[pyfunction]
pub fn bands<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    rows: &WallRows,
    measured: &Bound<'py, PyAny>,
    band_type: &Bound<'py, PyAny>,
) -> PyResult<()> {
    guard(|| {
        let k = Consts::read(radar_map)?;
        let odometer = attr_f64(radar_map, intern!(py, "odometer"))?;
        let mut cache = LaneCache::new(radar_map);
        bands_core(py, radar_map, &k, odometer, cache.get()?, rows, measured, band_type, None)
    })
}

/// `_bands`; the band geometry it appends to `wall_geometry` is also appended to `geometry` if given.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bands_core<'py>(
    py: Python<'py>,
    radar_map: &Bound<'py, PyAny>,
    k: &Consts,
    odometer: f64,
    lanes: &Lanes<'_>,
    rows: &WallRows,
    measured_obj: &Bound<'py, PyAny>,
    band_type: &Bound<'py, PyAny>,
    mut geometry_out: Option<&mut Vec<Vec<Xy>>>,
) -> PyResult<()> {
    let m = radar_map;
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
    let line_of = |sign: i64| lines.iter().find(|l| l.0 == sign).map(|&(_, reference, points)| (reference, points));

    let mut bands: Vec<BandData<'py>> = Vec::new();
    for band in m.getattr(intern!(py, "bands"))?.try_iter()? {
        let band = band?;
        let mut members = Ordered::new();
        for (key, value) in band.getattr(intern!(py, "members"))?.cast_into::<PyDict>()?.iter() {
            members.insert(key.extract()?, value.extract()?);
        }
        bands.push(BandData {
            sign: band.getattr(intern!(py, "sign"))?.extract()?,
            reference: band.getattr(intern!(py, "reference"))?.extract()?,
            offset: band.getattr(intern!(py, "offset"))?.extract()?,
            supported: band.getattr(intern!(py, "supported"))?.extract()?,
            spacing: band.getattr(intern!(py, "spacing"))?.extract()?,
            members,
            object: Some(band),
        });
    }
    let kept = bands.len();

    let any_measured = measured_obj.is_truthy()?;
    if any_measured {
        let measured = read_measured(measured_obj)?;
        let band_points_dict = m.getattr(intern!(py, "band_points"))?.cast_into::<PyDict>()?;
        let mut band_points: Ordered<BandPoint> = Ordered::new();
        for (key, value) in band_points_dict.iter() {
            band_points.insert(key.extract()?, value.extract()?);
        }
        let band_early = m.getattr(intern!(py, "band_early"))?;
        band_early.call_method1(intern!(py, "intersection_update"), (m.getattr(intern!(py, "objects"))?,))?;
        let mut early = Vec::new();
        for key in band_early.try_iter()? {
            let key: i64 = key?.extract()?;
            if measured.contains(&key) && band_points.contains(key) {
                let (x, y) = local(objs, key)?;
                early.push((x, y, key));
            }
        }
        for &(x, y, key) in rows.candidates.iter().chain(&early) {
            let sign = if y > 0. { 1 } else { -1 };
            let line = if y.abs() > 0.2 { line_of(sign) } else { None };
            let Some((reference, points)) = line else { continue };
            let Some(base) = interpolate(points, x) else { continue };
            let old = band_points.get(key).map(|p| p.4);
            let since = match old {
                Some(since) if !measured.contains(&key) => since,
                _ => odometer,
            };
            band_points.insert(key, (sign, reference, odometer + x, y - base, since, rows.inside.contains(&key)));
        }
        // A return is forgotten after BAND_MEMORY metres without a measurement, or at once with vehicle evidence.
        band_points.remove_where(|key, p| {
            odometer - p.4 > band_memory || objs.get(&key).is_some_and(|o| o.vision || o.radar_vehicle)
        });

        let mut claimed: HashSet<i64> = HashSet::new();
        for band in bands.iter_mut() {
            let mut points: Ordered<BandPoint> = Ordered::new();
            for &(key, p) in &band_points.items {
                if p.0 == band.sign && p.1 == band.reference && !claimed.contains(&key) {
                    points.insert(key, p);
                }
            }
            // The model's line moves between frames and all offsets move with it: follow the members.
            let mut current: Vec<f64> = band.members.keys().filter_map(|key| points.get(key).map(|p| p.3)).collect();
            sort_floats(&mut current);
            if current.is_empty() {
                band.members = Ordered::new();
                continue;
            }
            band.offset = current[current.len() / 2];
            let row: Vec<RowPoint> = points
                .items
                .iter()
                .filter(|(_, p)| (p.3 - band.offset).abs() <= band_bin)
                .map(|&(key, p)| (p.2, p.3, key, p.5))
                .collect();
            let row = band_row(row, dense_points, dense_span);
            let positions: Vec<f64> = row.iter().filter(|p| band.members.contains(p.2)).map(|p| p.0).collect();
            let (Some(mut low), Some(mut high)) =
                (fold(positions.iter().copied(), py_min), fold(positions.iter().copied(), py_max))
            else {
                band.members = Ordered::new();
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
            let mut members = Ordered::new();
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
            let mut positions: Vec<f64> = band.members.items.iter().map(|(_, v)| *v).collect();
            sort_floats(&mut positions);
            let mut gaps: Vec<f64> = positions.windows(2).map(|w| w[1] - w[0]).filter(|&gap| gap >= 0.6).collect();
            sort_floats(&mut gaps);
            band.spacing = if gaps.is_empty() { 0. } else { gaps[gaps.len() / 2] };
            // The next post is expected one spacing beyond the last (or two, with one post missed), at the band's
            // offset. A fresh return there without any vehicle evidence is that post: it joins now.
            let Some((reference, line)) = line_of(band.sign) else { continue };
            if band.spacing <= 0. || reference != band.reference {
                continue;
            }
            // measured.difference(self.band_points): a set of the same keys gives Python's set the same result.
            let known = PySet::new(py, band_points.keys())?;
            let fresh = measured_obj.call_method1(intern!(py, "difference"), (known,))?;
            let mut fresh_keys = Vec::new();
            for key in fresh.try_iter()? {
                fresh_keys.push(key?.extract::<i64>()?);
            }
            for key in sort_keys(fresh_keys.into_iter(), |key| Ok(local(objs, key)?.0))? {
                let o = obj(objs, key)?;
                let (x, y) = o.local(key)?;
                let base = if y * band.sign as f64 > 0.2 { interpolate(line, x) } else { None };
                let Some(base) = base else { continue };
                if o.motion_confirmed || o.doppler != 0 || o.vision || o.vision_seen || o.interior {
                    continue;
                }
                let ahead = odometer + x - positions[positions.len() - 1];
                if (y - base - band.offset).abs() <= band_prediction
                    && py_min((ahead - band.spacing).abs(), (ahead - 2. * band.spacing).abs()) <= band_prediction
                {
                    band_points.insert(key, (band.sign, band.reference, odometer + x, y - base, odometer, false));
                    band_early.call_method1(intern!(py, "add"), (key,))?;
                    band.members.insert(key, odometer + x);
                    claimed.insert(key);
                    positions.push(odometer + x);
                }
            }
        }

        let walls_len = m.getattr(intern!(py, "walls"))?.len()?;
        for &(sign, reference, _) in &lines {
            let mut cells: Ordered<Vec<RowPoint>> = Ordered::new();
            for &(key, p) in &band_points.items {
                if p.0 == sign && p.1 == reference && !claimed.contains(&key) {
                    let cell = floor(p.3 / band_bin)?;
                    match cells.get_mut(cell) {
                        Some(list) => list.push((p.2, p.3, key, p.5)),
                        None => cells.insert(cell, vec![(p.2, p.3, key, p.5)]),
                    }
                }
            }
            let mut windows = Vec::with_capacity(cells.len());
            for (cell, list) in &cells.items {
                let next = cell.checked_add(1).ok_or_else(|| PyOverflowError::new_err("cell"))?;
                let mut window = list.clone();
                window.extend(cells.get(next).into_iter().flatten().copied());
                windows.push(window);
            }
            windows.sort_by_key(|w| std::cmp::Reverse(w.len())); // sorted(key=len, reverse=True): stable
            for window in windows {
                let window: Vec<RowPoint> = window.into_iter().filter(|p| !claimed.contains(&p.2)).collect();
                let mut run: Vec<RowPoint> = Vec::new();
                for p in band_row(window, dense_points, dense_span).into_iter().map(Some).chain([None]) {
                    if let Some(p) = p {
                        if run.is_empty() || p.0 - run[run.len() - 1].0 <= band_gap {
                            run.push(p);
                            continue;
                        }
                    }
                    // Three posts (returns nearer than .6m are one post) no farther apart than in a straight row,
                    // over 4m.
                    let posts = if run.is_empty() { 0 } else { 1 + run.windows(2).filter(|w| w[1].0 - w[0].0 >= 0.6).count() };
                    if posts >= 3
                        && run[run.len() - 1].0 - run[0].0 >= 4. - 1e-5
                        && run.iter().any(|q| measured.contains(&q.2))
                        && ((bands.len() + walls_len) as f64) < max_walls
                    {
                        claimed.extend(run.iter().map(|q| q.2));
                        let mut members = Ordered::new();
                        for q in &run {
                            members.insert(q.2, q.0);
                        }
                        let offset = py_sum(run.iter().map(|q| q.1)).expect("three posts") / run.len() as f64;
                        bands.push(BandData { object: None, sign, reference, offset, members, supported: odometer, spacing: 0. });
                    }
                    run = p.into_iter().collect();
                }
            }
        }

        band_points_dict.clear();
        for (key, p) in &band_points.items {
            band_points_dict.set_item(key, p)?;
        }
        for band in &bands[..kept] {
            let object = band.object.as_ref().expect("a band of the last update");
            object.setattr(intern!(py, "offset"), band.offset)?;
            object.setattr(intern!(py, "members"), band.members_dict(py)?)?;
            object.setattr(intern!(py, "supported"), band.supported)?;
            object.setattr(intern!(py, "spacing"), band.spacing)?;
        }
    }

    // A band ages with the road, not the clock: it is folded after driving three of its spacings (at least
    // BAND_FOLD metres) without a measured member, and never while ego stands.
    bands.retain(|band| !band.members.is_empty() && odometer - band.supported <= py_max(3. * band.spacing, band_fold));
    let band_list = PyList::empty(py);
    let wall_geometry = m.getattr(intern!(py, "wall_geometry"))?;
    for band in &bands {
        let object = match &band.object {
            Some(object) => object.clone(),
            None => {
                let object = empty_object(band_type)?;
                object.setattr(intern!(py, "sign"), band.sign)?;
                object.setattr(intern!(py, "reference"), band.reference)?;
                object.setattr(intern!(py, "offset"), band.offset)?;
                object.setattr(intern!(py, "members"), band.members_dict(py)?)?;
                object.setattr(intern!(py, "supported"), band.supported)?;
                object.setattr(intern!(py, "spacing"), 0.)?;
                object
            },
        };
        band_list.append(&object)?;
        let Some((reference, line)) = line_of(band.sign) else { continue };
        if reference != band.reference {
            continue;
        }
        let values = || band.members.items.iter().map(|(_, v)| *v);
        let first = fold(values(), py_min).expect("members") - odometer;
        let last = fold(values(), py_max).expect("members") - odometer;
        let mut geometry: Vec<Xy> = Vec::new();
        for x in [first, last] {
            if let Some(y) = interpolate(line, x) {
                geometry.push((x, y + band.offset));
            }
        }
        geometry.extend(line.iter().filter(|p| first < p.0 && p.0 < last).map(|&(x, y)| (x, y + band.offset)));
        geometry.sort_by(cmp_xy);
        if geometry.len() >= 2 && geometry[geometry.len() - 1].0 > geometry[0].0 {
            wall_geometry.call_method1(intern!(py, "append"), (PyList::new(py, geometry.iter().copied())?,))?;
            if let Some(out) = geometry_out.as_deref_mut() {
                out.push(geometry);
            }
        }
        for key in band.members.keys() {
            if let Some(o) = objs.get(&key) {
                if !o.vision && !o.radar_vehicle {
                    o.obj.bind(py).try_borrow_mut()?.structure = true;
                }
            }
        }
    }
    m.setattr(intern!(py, "bands"), band_list)?;
    Ok(())
}
