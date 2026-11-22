//! Lane geometry reads of `CcncRadarMap`: `interpolate`, `_lane_view`, `_lane`, `_ego_lines`/`_centre`,
//! `_settle`'s outer line, `_body_clear` and `_band_line` (hyundaicanfd_ccnc_radar_map.py).
//!
//! `lanes`, `geometry`, `lane_offset` and the model stamp do not change after `_update_lane_reference` until the
//! next update, so one snapshot serves a whole Rust call. The model's lane change state and the bands are read
//! only when first needed; the capnp names stay in Python (`_model_changing`).

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

struct Line {
    lane: Option<(f64, f64)>, // lanes[i]: (probability, consistent)
    geometry: Option<Vec<(f64, f64)>>,
}

struct Band {
    sign: f64,
    reference: i64,
    offset: f64,
    members: Option<(f64, f64)>, // min and max of members.values(); None when it has none
}

pub struct Lanes<'py> {
    radar_map: Bound<'py, PyAny>,
    lines: Vec<Line>, // physical index 0..6
    lane_offset: i64,
    model_fresh: bool,
    pub has_geometry: bool,
    outer_narrow: f64,
    outer_far: f64,
    changing: Option<bool>,
    bands: Option<Vec<Band>>,
}

/// The snapshot, made on first use.
pub struct LaneCache<'a, 'py> {
    radar_map: &'a Bound<'py, PyAny>,
    lanes: Option<Lanes<'py>>,
    preset: Option<(f64, f64, bool)>, // (OUTER_NARROW, OUTER_FAR, _model_changing()) read by the caller
    now: Option<f64>,                 // the map's `now`, read from the map when None
}

impl<'a, 'py> LaneCache<'a, 'py> {
    pub fn new(radar_map: &'a Bound<'py, PyAny>) -> Self {
        Self { radar_map, lanes: None, preset: None, now: None }
    }

    /// The one-call update's snapshot: its constants and the model's lane change state come from the caller.
    /// `now` is the map's: its state may be borrowed while the snapshot is made.
    pub fn preset(radar_map: &'a Bound<'py, PyAny>, outer_narrow: f64, outer_far: f64, changing: bool, now: f64) -> Self {
        Self { radar_map, lanes: None, preset: Some((outer_narrow, outer_far, changing)), now: Some(now) }
    }

    pub fn get(&mut self) -> PyResult<&mut Lanes<'py>> {
        if self.lanes.is_none() {
            self.lanes = Some(Lanes::read(self.radar_map, self.preset, self.now)?);
        }
        Ok(self.lanes.as_mut().unwrap())
    }
}

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

impl<'py> Lanes<'py> {
    fn read(m: &Bound<'py, PyAny>, preset: Option<(f64, f64, bool)>, now: Option<f64>) -> PyResult<Self> {
        let py = m.py();
        let lanes = m.getattr(intern!(py, "lanes"))?;
        let lanes = lanes.cast::<PyDict>()?;
        let geometry = m.getattr(intern!(py, "geometry"))?;
        let geometry = geometry.cast::<PyDict>()?;
        let mut lines = Vec::with_capacity(6);
        for i in 0..6 {
            let lane = match lanes.get_item(i)? {
                Some(lane) => Some((
                    attr_f64(&lane, intern!(py, "probability"))?,
                    attr_f64(&lane, intern!(py, "consistent"))?,
                )),
                None => None,
            };
            let geometry = match geometry.get_item(i)? {
                Some(points) => Some(points.extract()?),
                None => None,
            };
            lines.push(Line { lane, geometry });
        }
        let model_stamp = m.getattr(intern!(py, "model_stamp"))?;
        let model_fresh = if model_stamp.is_none() {
            false
        } else {
            let now = match now {
                Some(now) => now,
                None => match crate::map_state::attached(m)? {
                    Some(state) => state.try_borrow()?.st.now.ok_or_else(|| PyValueError::new_err("map not advanced"))?,
                    None => attr_f64(m, intern!(py, "now"))?,
                },
            };
            let age = now - model_stamp.extract::<f64>()?;
            (0. ..=0.25).contains(&age)
        };
        let (outer_narrow, outer_far, changing) = match preset {
            Some((narrow, far, changing)) => (narrow, far, Some(changing)),
            None => (attr_f64(m, intern!(py, "OUTER_NARROW"))?, attr_f64(m, intern!(py, "OUTER_FAR"))?, None),
        };
        Ok(Self {
            radar_map: m.clone(),
            lines,
            lane_offset: m.getattr(intern!(py, "lane_offset"))?.extract()?,
            model_fresh,
            has_geometry: !geometry.is_empty(),
            outer_narrow,
            outer_far,
            changing,
            bands: None,
        })
    }

    fn line(&self, i: i64) -> Option<&Line> {
        usize::try_from(i).ok().and_then(|i| self.lines.get(i))
    }

    fn points(&self, i: usize) -> &[(f64, f64)] {
        self.lines[i].geometry.as_deref().unwrap_or(&[])
    }

    /// `self.lanes[i].probability`, or None without that line.
    pub fn probability(&self, i: usize) -> Option<f64> {
        self.lines.get(i).and_then(|line| line.lane).map(|(probability, _)| probability)
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
        self.line(i).and_then(|line| line.lane).is_some_and(|(probability, _)| probability >= 0.1)
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

    fn changing(&mut self) -> PyResult<bool> {
        if let Some(changing) = self.changing {
            return Ok(changing);
        }
        let py = self.radar_map.py();
        let changing = self.radar_map.call_method0(intern!(py, "_model_changing"))?.is_truthy()?;
        self.changing = Some(changing);
        Ok(changing)
    }

    /// `_lane(x, y, respect_walls, moving, settled)`: (slot, interior, centre distance). `walls` holds the wall
    /// lines' y at x (`respect_walls`); the Rust callers so far pass none.
    pub fn lane(&mut self, x: f64, y: f64, moving: bool, settled: bool, walls: &[f64]) -> PyResult<(Option<i64>, bool, f64)> {
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
            let changing = self.changing()?;
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
            let room = if moving && has(usable, edge) { fold(py_max) } else { fold(py_min) };
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
            let limit = limit.ok_or_else(|| PyValueError::new_err("min() iterable argument is empty"))?;
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
    pub fn outer(&self, x: f64, y: f64) -> PyResult<Option<f64>> {
        let line = if y > 0. { 0 } else { 3 };
        let Some(points) = self.lines[line].geometry.as_deref() else { return Ok(None) };
        let (probability, _) = self.lines[line].lane.ok_or_else(|| PyKeyError::new_err(line))?;
        if probability < 0.1 {
            return Ok(None);
        }
        Ok(interpolate(points, x))
    }

    /// `_body_clear(slot, x, y)`.
    pub fn body_clear(&self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> bool {
        let inner = if slot == 1 { 1 } else { 2 };
        let base = interpolate(self.points(inner), x);
        let (Some((probability, _)), Some(base)) = (self.lines[inner].lane, base) else { return false };
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
        let (Some((probability, _)), Some(base), Some(bound)) =
            (self.lines[edge].lane, interpolate(self.points(inner), x), interpolate(self.points(edge), x))
        else {
            return true;
        };
        if probability < 0.1 {
            return true;
        }
        (if slot == 1 { bound - base } else { base - bound }) >= room
    }

    fn bands(&mut self) -> PyResult<&[Band]> {
        if self.bands.is_none() {
            let py = self.radar_map.py();
            let mut bands = Vec::new();
            for band in self.radar_map.getattr(intern!(py, "bands"))?.try_iter()? {
                let band = band?;
                let members = band.getattr(intern!(py, "members"))?;
                let members = members.cast::<PyDict>()?;
                let mut span: Option<(f64, f64)> = None;
                for value in members.values() {
                    let v: f64 = value.extract()?;
                    span = Some(span.map_or((v, v), |(lo, hi)| (py_min(lo, v), py_max(hi, v))));
                }
                bands.push(Band {
                    sign: band.getattr(intern!(py, "sign"))?.extract()?,
                    reference: band.getattr(intern!(py, "reference"))?.extract()?,
                    offset: band.getattr(intern!(py, "offset"))?.extract()?,
                    members: span,
                });
            }
            self.bands = Some(bands);
        }
        Ok(self.bands.as_deref().unwrap())
    }

    /// `_band_line(x, y)` with the bands of the previous update.
    pub fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> PyResult<bool> {
        self.bands()?;
        for band in self.bands.as_deref().unwrap() {
            let Some((low, high)) = band.members else { continue };
            if y * band.sign <= 0.2 {
                continue;
            }
            let Some(line) = self.line(band.reference) else { continue };
            match line.lane {
                Some((probability, _)) if probability >= 0.1 => {}
                _ => continue,
            }
            let base = interpolate(line.geometry.as_deref().unwrap_or(&[]), x);
            let along = odometer + x;
            if let Some(base) = base {
                if (y - base - band.offset).abs() <= bin && low - memory <= along && along <= high + memory {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn lane_err(error: PyErr) -> ccnc_map::radar::Error {
    crate::map_state::external(error)
}

impl ccnc_map::radar::lanes::LaneQuery for LaneCache<'_, '_> {
    fn lane(&mut self, x: f64, y: f64, moving: bool, settled: bool) -> ccnc_map::radar::Result<(Option<i64>, bool, f64)> {
        self.get().and_then(|l| l.lane(x, y, moving, settled, &[])).map_err(lane_err)
    }

    fn centre(&mut self, x: f64) -> ccnc_map::radar::Result<Option<f64>> {
        self.get().map(|l| l.centre(x)).map_err(lane_err)
    }

    fn outer(&mut self, x: f64, y: f64) -> ccnc_map::radar::Result<Option<f64>> {
        self.get().and_then(|l| l.outer(x, y)).map_err(lane_err)
    }

    fn body_clear(&mut self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> ccnc_map::radar::Result<bool> {
        self.get().map(|l| l.body_clear(slot, x, y, clearance, far_side)).map_err(lane_err)
    }

    fn edge_room(&mut self, slot: i64, x: f64, room: f64) -> ccnc_map::radar::Result<bool> {
        self.get().map(|l| l.edge_room(slot, x, room)).map_err(lane_err)
    }

    fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> ccnc_map::radar::Result<bool> {
        self.get().and_then(|l| l.band_line(x, y, odometer, bin, memory)).map_err(lane_err)
    }

    fn has_geometry(&mut self) -> ccnc_map::radar::Result<bool> {
        self.get().map(|l| l.has_geometry).map_err(lane_err)
    }
}
