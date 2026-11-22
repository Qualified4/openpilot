//! Lane queries for the radar map.

use super::state::{Error, Result};
use super::walls::BandData;
use crate::pymath::{py_max, py_min};

pub trait LaneQuery {
    /// `_lane(x, y, moving=moving, settled=settled)`: (slot, interior, distance past the line).
    fn lane(&mut self, x: f64, y: f64, moving: bool, settled: bool) -> Result<(Option<i64>, bool, f64)>;
    /// `_centre(x)`: ego's lane centre at x.
    fn centre(&mut self, x: f64) -> Result<Option<f64>>;
    /// The outer line of the side y is on, at x (`_settle`).
    fn outer(&mut self, x: f64, y: f64) -> Result<Option<f64>>;
    fn body_clear(&mut self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> Result<bool>;
    fn edge_room(&mut self, slot: i64, x: f64, room: f64) -> Result<bool>;
    fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> Result<bool>;
    /// Whether the map has any lane geometry (`bool(self.geometry)`).
    fn has_geometry(&mut self) -> Result<bool>;
}

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
pub struct Line {
    pub lane: Option<(f64, f64)>, // lanes[i]: (probability, consistent)
    pub geometry: Option<Vec<(f64, f64)>>,
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
pub struct Lanes {
    lines: Vec<Line>,
    lane_offset: i64,
    pub model_fresh: bool,
    pub has_geometry: bool,
    outer_narrow: f64,
    outer_far: f64,
    pub changing: bool,
    bands: Vec<BandSpan>,
}

impl Lanes {
    pub fn new(
        lines: Vec<Line>,
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
    fn line(&self, i: i64) -> Option<&Line> {
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
    pub fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> Result<bool> {
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
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

impl LaneQuery for Lanes {
    fn lane(&mut self, x: f64, y: f64, moving: bool, settled: bool) -> Result<(Option<i64>, bool, f64)> {
        Lanes::lane(self, x, y, moving, settled, &[])
    }

    fn centre(&mut self, x: f64) -> Result<Option<f64>> {
        Ok(Lanes::centre(self, x))
    }

    fn outer(&mut self, x: f64, y: f64) -> Result<Option<f64>> {
        Lanes::outer(self, x, y)
    }

    fn body_clear(&mut self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> Result<bool> {
        Ok(Lanes::body_clear(self, slot, x, y, clearance, far_side))
    }

    fn edge_room(&mut self, slot: i64, x: f64, room: f64) -> Result<bool> {
        Ok(Lanes::edge_room(self, slot, x, room))
    }

    fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> Result<bool> {
        Lanes::band_line(self, x, y, odometer, bin, memory)
    }

    fn has_geometry(&mut self) -> Result<bool> {
        Ok(self.has_geometry)
    }
}
