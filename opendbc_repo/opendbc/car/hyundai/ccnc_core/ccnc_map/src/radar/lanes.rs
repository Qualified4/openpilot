//! Lane queries for the radar map.

use super::state::{Error, Frame, InsertionMap, Result};
use super::walls::BandData;
use std::borrow::Cow;

use crate::input::model_v2::{Line as ModelLine, Model};
use crate::pymath::{py_max, py_min};

/// `interpolate(points, x)` (and its Cython twin): the line's y at x, or None outside it.
pub fn interpolate(points: &[(f64, f64)], x: f64) -> Option<f64> {
    let n = points.len();
    if n < 2 || !(points[0].0 <= x && x <= points[n - 1].0) {
        return None;
    }
    // bisect_left(points, (x, -inf)): a point sorts before the key when its x is smaller.
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if points[mid].0 < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let i = lo.max(1);
    let (a, b) = (points[i - 1], points[i]);
    if b.0 <= a.0 {
        return None;
    }
    Some(a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0))
}

#[derive(Clone, Debug)]
pub struct Line<'a> {
    pub lane: Option<(f64, f64)>, // lanes[i]: (probability, consistent)
    pub geometry: Option<Cow<'a, [(f64, f64)]>>,
}

#[derive(Clone, Debug)]
struct BandSpan {
    sign: i64,
    reference: i64,
    offset: f64,
    low: f64,
    high: f64,
}

#[derive(Clone, Debug)]
pub struct Lanes<'a> {
    lines: [Line<'a>; 6],
    lane_offset: i64,
    pub model_fresh: bool,
    pub has_geometry: bool,
    outer_narrow: f64,
    outer_far: f64,
    pub changing: bool,
    bands: Vec<BandSpan>,
}

impl<'a> Lanes<'a> {
    pub fn new(
        lines: [Line<'a>; 6],
        lane_offset: i64,
        model_fresh: bool,
        has_geometry: bool,
        outer: (f64, f64),
        changing: bool,
    ) -> Self {
        Self {
            lines,
            lane_offset,
            model_fresh,
            has_geometry,
            outer_narrow: outer.0,
            outer_far: outer.1,
            changing,
            bands: Vec::new(),
        }
    }

    pub fn read_bands(&mut self, bands: &[BandData]) {
        self.bands = bands
            .iter()
            .filter_map(|band| {
                let (low, high) = band.span()?;
                Some(BandSpan {
                    sign: band.sign,
                    reference: band.reference,
                    offset: band.offset,
                    low,
                    high,
                })
            })
            .collect();
    }
    fn line(&self, i: i64) -> Option<&Line<'a>> {
        usize::try_from(i).ok().and_then(|i| self.lines.get(i))
    }

    fn points(&self, i: usize) -> &[(f64, f64)] {
        self.lines[i].geometry.as_deref().unwrap_or(&[])
    }

    /// `self.lanes[i].probability`, or None without that line.
    pub fn probability(&self, i: usize) -> Option<f64> {
        self.lines
            .get(i)
            .and_then(|line| line.lane)
            .map(|(probability, _)| probability)
    }

    /// `self.geometry.get(i)`.
    pub fn geometry(&self, i: usize) -> Option<&[(f64, f64)]> {
        self.lines.get(i).and_then(|line| line.geometry.as_deref())
    }

    /// `interpolate(self.geometry.get(i, []), x)`.
    pub fn at(&self, i: i64, x: f64) -> Option<f64> {
        interpolate(self.line(i).and_then(|line| line.geometry.as_deref()).unwrap_or(&[]), x)
    }

    /// `interpolate(self.geometry.get(i, []), x)` of a line in `self.lanes` with probability .1 or more, else None.
    pub fn boundary(&self, i: i64, x: f64) -> Option<f64> {
        match self.line(i)?.lane {
            Some((probability, _)) if probability >= 0.1 => self.at(i, x),
            _ => None,
        }
    }

    /// Whether line i is in `self.lanes` with probability .1 or more.
    pub fn usable(&self, i: i64) -> bool {
        self.line(i)
            .and_then(|line| line.lane)
            .is_some_and(|(probability, _)| probability >= 0.1)
    }

    pub fn lane_offset(&self) -> i64 {
        self.lane_offset
    }

    /// After `_update_lane_reference` has set the map's lane offset.
    pub fn set_lane_offset(&mut self, offset: i64) {
        self.lane_offset = offset;
    }

    /// `_lane_view(x)`: bounds by role, and the usable/strong roles as bit masks.
    fn view(&self, x: f64) -> ([Option<f64>; 6], u8, u8) {
        let mut bounds = [None; 6];
        let (mut usable, mut strong) = (0u8, 0u8);
        for (i, bound) in bounds.iter_mut().enumerate() {
            let j = if i < 4 {
                let j = i as i64 + self.lane_offset;
                if !(0..4).contains(&j) {
                    continue;
                }
                j as usize
            } else {
                i
            };
            *bound = interpolate(self.points(j), x);
            if let Some((probability, consistent)) = self.lines[j].lane {
                if probability >= 0.1 && bound.is_some() {
                    usable |= 1 << i;
                    if probability >= 0.6 && consistent >= 3. {
                        strong |= 1 << i;
                    }
                }
            }
        }
        (bounds, usable, strong)
    }

    /// `_lane(x, y, respect_walls, moving, settled)`: (slot, interior, centre distance). `walls` holds the wall
    /// lines' y at x (`respect_walls`); the Rust callers so far pass none.
    pub fn lane(&self, x: f64, y: f64, moving: bool, settled: bool, walls: &[f64]) -> Result<(Option<i64>, bool, f64)> {
        let (bounds, usable, strong) = self.view(x);
        let has = |mask: u8, i: usize| mask & (1 << i) != 0;
        if has(usable, 1) && has(usable, 2) {
            let (mut a, mut b) = (bounds[1].unwrap(), bounds[2].unwrap());
            for &wy in walls {
                if wy > 0.2 {
                    a = py_min(a, wy);
                } else if wy < -0.2 {
                    b = py_max(b, wy);
                }
            }
            let width = a - b;
            if (1.8..=5.).contains(&width) && (b..=a).contains(&y) {
                let interior = has(strong, 1) && has(strong, 2) && py_min(a - y, y - b) >= 0.35;
                return Ok((Some(0), interior, (y - 0.5 * (a + b)).abs()));
            }
        } else if !has(usable, 1) && !has(usable, 2) {
            let changing = self.changing;
            // A distant interpolation gap alone must not enable the no-line distance cap.
            let limited = self.model_fresh
                && changing
                && !(1..3).any(|i| self.lines[i].lane.is_some_and(|(probability, _)| probability >= 0.1));
            if (!limited || x <= 80.) && y.abs() <= 1.2 {
                return Ok((Some(0), false, y.abs()));
            }
        }
        for (slot, inner, line, edge, sign) in [(1, 1, 0, 4, 1.), (2, 2, 3, 5, -1.)] {
            if !has(usable, inner) {
                continue;
            }
            let base = bounds[inner].unwrap();
            let offset = sign * (y - base);
            let mut widths = [0.; 2];
            let mut count = 0;
            for i in [line, edge] {
                if has(usable, i) {
                    widths[count] = sign * (bounds[i].unwrap() - base);
                    count += 1;
                }
            }
            let widths = &widths[..count];
            let fold = |f: fn(f64, f64) -> f64| widths[1..].iter().fold(widths[0], |acc, &w| f(acc, w));
            if offset <= 0. || widths.is_empty() {
                continue;
            }
            let room = if moving && has(usable, edge) {
                fold(py_max)
            } else {
                fold(py_min)
            };
            if room - 0.25 <= 1.8 {
                continue;
            }
            let mut margin = if !moving { -0.25 } else { 0.75 };
            if settled
                && has(usable, line)
                && sign * (bounds[line].unwrap() - base) >= self.outer_narrow
                && offset >= self.outer_far
            {
                margin = 0.;
            }
            let mut limit: Option<f64> = None;
            let mut take = |v: f64| limit = Some(limit.map_or(v, |l| py_min(l, v)));
            if has(usable, line) {
                take(sign * (bounds[line].unwrap() - base) + margin);
            }
            if has(usable, edge) {
                take(sign * (bounds[edge].unwrap() - base) - 0.95);
            }
            for &wy in walls {
                if sign * wy > 0.2 {
                    take(sign * (wy - base));
                }
            }
            let limit = limit.ok_or_else(|| Error::Value("min() iterable argument is empty"))?;
            if offset <= limit {
                let width = fold(py_min);
                let interior = has(strong, inner) && has(strong, line) && py_min(offset, width - offset) >= 0.35;
                return Ok((Some(slot), interior, (offset - 0.5 * width).abs()));
            }
        }
        Ok((None, false, f64::INFINITY))
    }

    /// `_lane`'s side-slot room for LF (index 0) and RF (index 1): true where the lines leave no adjacent lane for an
    /// object at (x, y) beyond ego's line on that side. Without ego's line there, `_lane` gives no side slot either.
    /// An object inside ego's line is left to the world lane coordinate, and the outer line's margin is not checked.
    pub fn side_blocked(&self, x: f64, y: f64, moving: bool, walls: &[f64]) -> [bool; 2] {
        let (bounds, usable, _) = self.view(x);
        let from = |i: usize, base: f64, sign: f64| (usable & (1 << i) != 0).then(|| sign * (bounds[i].unwrap() - base));
        [(1, 0, 4, 1.), (2, 3, 5, -1.)].map(|(inner, line, edge, sign)| {
            let Some(base) = (usable & (1 << inner) != 0).then(|| bounds[inner].unwrap()) else {
                return true;
            };
            let offset = sign * (y - base);
            if offset <= 0. {
                return false;
            }
            let (outer, road) = (from(line, base, sign), from(edge, base, sign));
            let room = match (outer, road) {
                (None, None) => return true,
                (Some(outer), Some(road)) if moving => py_max(outer, road),
                (Some(outer), Some(road)) => py_min(outer, road),
                (Some(width), None) | (None, Some(width)) => width,
            };
            room - 0.25 <= 1.8
                || road.is_some_and(|road| offset > road - 0.95)
                || walls.iter().any(|&wy| sign * wy > 0.2 && offset > sign * (wy - base))
        })
    }

    /// `_ego_lines(x)`: ego's left and right lane lines at x (None where unusable).
    pub fn ego_lines(&self, x: f64) -> (Option<f64>, Option<f64>) {
        let line = |i: i64| -> Option<f64> {
            if !(0..4).contains(&i) {
                return None;
            }
            match self.lines[i as usize].lane {
                Some((probability, _)) if probability >= 0.1 => interpolate(self.points(i as usize), x),
                _ => None,
            }
        };
        (line(1 + self.lane_offset), line(2 + self.lane_offset))
    }

    /// `_centre(x)`: the middle of ego's lane lines, or None.
    pub fn centre(&self, x: f64) -> Option<f64> {
        let (left, right) = self.ego_lines(x);
        Some(0.5 * (left? + right?))
    }

    /// `_settle`'s outer line of the position's side at x, or None where `_settle` returns early.
    pub fn outer(&self, x: f64, y: f64) -> Result<Option<f64>> {
        let line = if y > 0. { 0 } else { 3 };
        let Some(points) = self.lines[line].geometry.as_deref() else {
            return Ok(None);
        };
        let (probability, _) = self.lines[line].lane.ok_or_else(|| Error::Key(line as i64))?;
        if probability < 0.1 {
            return Ok(None);
        }
        Ok(interpolate(points, x))
    }

    /// `_body_clear(slot, x, y)`.
    pub fn body_clear(&self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> bool {
        let inner = if slot == 1 { 1 } else { 2 };
        let base = interpolate(self.points(inner), x);
        let (Some((probability, _)), Some(base)) = (self.lines[inner].lane, base) else {
            return false;
        };
        if probability < 0.1 {
            return false;
        }
        let v = if slot == 1 { y - base } else { base - y };
        clearance <= v && v <= far_side
    }

    /// `_edge_room(slot, x)`: the side's road edge leaves `room` past ego's lane line, or there is no usable edge.
    pub fn edge_room(&self, slot: i64, x: f64, room: f64) -> bool {
        if room == 0. {
            return true;
        }
        let (inner, edge) = if slot == 1 { (1, 4) } else { (2, 5) };
        let (Some((probability, _)), Some(base), Some(bound)) = (
            self.lines[edge].lane,
            interpolate(self.points(inner), x),
            interpolate(self.points(edge), x),
        ) else {
            return true;
        };
        if probability < 0.1 {
            return true;
        }
        (if slot == 1 { bound - base } else { base - bound }) >= room
    }

    /// `_band_line(x, y)` with the bands of the previous update.
    pub fn band_line(&self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> bool {
        for band in &self.bands {
            if y * band.sign as f64 <= 0.2 {
                continue;
            }
            let Some(line) = self.line(band.reference) else {
                continue;
            };
            match line.lane {
                Some((probability, _)) if probability >= 0.1 => {}
                _ => continue,
            }
            let base = interpolate(line.geometry.as_deref().unwrap_or(&[]), x);
            let along = odometer + x;
            if let Some(base) = base {
                if (y - base - band.offset).abs() <= bin && band.low - memory <= along && along <= band.high + memory {
                    return true;
                }
            }
        }
        false
    }
}

pub type Curve = (Vec<f64>, Vec<f64>);
pub type Xy = (f64, f64);

#[derive(Clone, Debug)]
pub struct Boundary {
    pub points: Vec<Xy>,
    pub probability: f64,
    pub consistent: i64,
}

#[derive(Clone, Debug, Default)]
pub struct LaneCache {
    pub lines: InsertionMap<usize, Boundary>,
    pub geometry: InsertionMap<usize, Vec<Xy>>,
    weak: u8,
}

pub type Placed = [Option<Vec<Xy>>; 6];

impl LaneCache {
    pub fn weak_at(&self, i: usize) -> bool {
        i < 4 && self.weak & (1 << i) != 0
    }

    pub fn view(&self, offset: i64, model_fresh: bool, outer: (f64, f64), changing: bool) -> Lanes<'_> {
        let lines = std::array::from_fn(|i| Line {
            lane: self
                .lines
                .get(&i)
                .map(|line| (line.probability, line.consistent as f64)),
            geometry: self.geometry.get(&i).map(|points| Cow::Borrowed(points.as_slice())),
        });
        Lanes::new(lines, offset, model_fresh, !self.geometry.is_empty(), outer, changing)
    }

    /// Reproject the world points after advancing ego. Retain their order for the next model update.
    pub fn place(&mut self, frame: Frame) -> Placed {
        let mut placed: Placed = Default::default();
        self.geometry = InsertionMap::default();
        for (i, lane) in self.lines.iter() {
            let local: Vec<_> = lane.points.iter().map(|&(x, y)| frame.to_ego(x, y)).collect();
            self.geometry.insert(i, deduplicated(sorted(local.clone())));
            placed[i] = Some(local);
        }
        placed
    }

    /// Refresh model lines while retaining up to 24 real points behind the model's domain.
    pub fn refresh(
        &mut self,
        model: &GeometryInput,
        origin: Frame,
        frame: Frame,
        previous: &Placed,
        window: (f64, f64),
        ego_change: bool,
    ) {
        let (px, py, pc, ps) = (origin.px, origin.py, origin.c, origin.s);
        let count = model.lines.len().min(model.probs.len()).min(4);
        let entries = (0..4)
            .map(|i| (i < count).then(|| (&model.lines[i], model.probs[i])))
            .chain(model.edges.iter().take(2).map(|edge| Some((edge, 1.))));
        for (i, entry) in entries.enumerate() {
            let Some((line, mut probability)) = entry else { continue };
            if !valid_curve(line) || !probability.is_finite() {
                continue;
            }
            if probability < 0.1 {
                if !ego_change || i > 3 {
                    if let Some(old) = self.lines.get_mut(&i) {
                        old.probability *= 0.9;
                    }
                    continue;
                }
                probability = 0.1;
                self.weak |= 1 << i;
            } else {
                self.weak &= !(1 << i);
            }
            let first = bisect_left(&line.0, window.0).saturating_sub(1);
            let last = (bisect_left(&line.0, window.1) + 1).min(line.0.len());
            let world: Vec<Xy> = (first..last)
                .map(|j| {
                    let (x, y) = (line.0[j], line.1[j]);
                    (px + pc * x + ps * y, py + ps * x - pc * y)
                })
                .collect();
            let placed: Vec<_> = world.iter().map(|&(x, y)| frame.to_ego(x, y)).collect();
            let local = sorted(placed.clone());
            let Some(first) = local.first() else { continue };
            let previous_geometry = self.geometry.get(&i).map_or(&[][..], Vec::as_slice);
            let max_error = local
                .iter()
                .filter(|p| (0. ..=30.).contains(&p.0))
                .filter_map(|&(x, y)| interpolate(previous_geometry, x).map(|previous| (y - previous).abs()))
                .reduce(py_max);
            let old = self.lines.get(&i);
            let consistent = match (old, max_error) {
                (Some(old), Some(error)) if error <= 0.35 => old.consistent + 1,
                _ => 1,
            };
            let limit = first.0 - 0.1;
            let mut behind: Vec<_> = match (old, &previous[i]) {
                (Some(old), Some(old_ego)) => old
                    .points
                    .iter()
                    .copied()
                    .zip(old_ego.iter().copied())
                    .filter(|(_, ego)| -5. <= ego.0 && ego.0 < limit)
                    .collect(),
                _ => Vec::new(),
            };
            let behind = behind.split_off(behind.len().saturating_sub(24));
            let points = behind
                .iter()
                .map(|&(point, _)| point)
                .chain(world.into_iter().take(40))
                .collect();
            let ego = behind
                .into_iter()
                .map(|(_, ego)| ego)
                .chain(placed.into_iter().take(40))
                .collect();
            self.lines.insert(
                i,
                Boundary {
                    points,
                    probability,
                    consistent: consistent.min(20),
                },
            );
            self.geometry.insert(i, deduplicated(sorted(ego)));
        }
    }
}

fn sort_key(a: Xy, b: Xy) -> std::cmp::Ordering {
    use std::cmp::Ordering::Equal;
    if a.0 != b.0 {
        a.0.partial_cmp(&b.0).unwrap_or(Equal)
    } else {
        a.1.partial_cmp(&b.1).unwrap_or(Equal)
    }
}

fn sorted(mut points: Vec<Xy>) -> Vec<Xy> {
    // Normal model curves have a total order. Pathological floating point inputs can
    // overflow the pose transform. Avoid Rust's total-order assumption in that case.
    if points.iter().any(|p| p.0.is_nan() || p.1.is_nan()) {
        // CPython's initial monotone run and stable binary insertion on these short lines.
        let n = points.len();
        let mut run = n.min(2);
        if n > 1 {
            let descending = sort_key(points[1], points[0]).is_lt();
            while run < n && sort_key(points[run], points[run - 1]).is_lt() == descending {
                run += 1;
            }
            if descending {
                points[..run].reverse();
            }
        }
        for i in run..n {
            let (mut left, mut right) = (0, i);
            while left < right {
                let middle = (left + right) / 2;
                if sort_key(points[i], points[middle]).is_lt() {
                    right = middle;
                } else {
                    left = middle + 1;
                }
            }
            points[left..=i].rotate_right(1);
        }
    } else {
        points.sort_by(|a, b| sort_key(*a, *b));
    }
    points
}

fn deduplicated(points: Vec<Xy>) -> Vec<Xy> {
    points
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, p)| (i == 0 || p.0 > points[i - 1].0 + 0.001).then_some(p))
        .collect()
}

#[derive(Default)]
pub struct GeometryInput {
    pub lines: Vec<Curve>,
    pub probs: Vec<f64>,
    pub edges: Vec<Curve>,
}

impl GeometryInput {
    pub fn from_raw(model: Model) -> Self {
        let convert = |line: ModelLine| {
            (
                line.x.into_iter().map(f64::from).collect(),
                line.y.into_iter().map(f64::from).collect(),
            )
        };
        Self {
            lines: model.lane_lines.into_iter().take(4).map(convert).collect(),
            probs: model.lane_line_probs.into_iter().take(4).map(f64::from).collect(),
            edges: model.road_edges.into_iter().take(2).map(convert).collect(),
        }
    }

    pub fn side_center(&self, side: usize, probability: f64) -> Option<f64> {
        let (i, j) = if side == 0 { (1, 0) } else { (2, 3) };
        if self.lines.len() < 4
            || self.probs.len() < 4
            || ![i, j]
                .into_iter()
                .all(|idx| self.probs[idx].is_finite() && self.probs[idx] >= probability)
        {
            return None;
        }
        let (inner, outer) = (&self.lines[i], &self.lines[j]);
        if ![inner, outer]
            .into_iter()
            .all(|p| valid_curve(p) && p.0[0] <= 20. && 20. <= p.0[p.0.len() - 1])
        {
            return None;
        }
        let (y, outer_y) = (numpy_interp(inner, 20.), numpy_interp(outer, 20.));
        let width = if side == 0 { y - outer_y } else { outer_y - y };
        let center = inner.1[0].abs() + width * 0.5;
        ((2.3..=4.8).contains(&width) && center.is_finite()).then_some(center)
    }
}

pub fn valid_curve((x, y): &Curve) -> bool {
    x.len() >= 2 && x.len() == y.len() && x.iter().chain(y).all(|p| p.is_finite()) && x.windows(2).all(|p| p[1] > p[0])
}

/// The slope-first interpolation used by numpy.interp for validated curves.
pub fn numpy_interp((x, y): &Curve, at: f64) -> f64 {
    if at.is_nan() {
        return at;
    }
    let i = bisect_left(x, at);
    if i >= x.len() {
        return y[y.len() - 1];
    }
    if i == 0 || x[i] == at {
        return y[i];
    }
    let j = i - 1;
    let slope = (y[i] - y[j]) / (x[i] - x[j]);
    let value = slope * (at - x[j]) + y[j];
    if value.is_nan() {
        let retry = slope * (at - x[i]) + y[i];
        if retry.is_nan() && y[i] == y[j] {
            return y[j];
        }
        return retry;
    }
    value
}

/// `bisect_left(values, x)` on floats.
fn bisect_left(values: &[f64], x: f64) -> usize {
    let (mut lo, mut hi) = (0, values.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if values[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
