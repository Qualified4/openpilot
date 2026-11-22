use std::collections::VecDeque;

use crate::input::model_v2::{Line, Model};

use super::{Pose, PoseAt};

pub const MODEL_GRID: [f32; 10] = [0., 5., 10., 15., 20., 30., 40., 50., 60., 80.];
const LANE_WIDTH: f64 = 3.5;
const BIN_M: f64 = 2.;
const BIN_COUNT: usize = 70;
const BOUNDARY_COUNT: usize = 8;
const PENDING_CAPACITY: usize = 512;
const NEAR: f64 = 25.;
const EXTRAPOLATION_BINS: i64 = 12;
const MAX_CURVE_EXTENSION: f64 = 45.;
const MAX_CURVE_COEFFICIENT: f64 = 0.0007;

type Point = (f64, f64);

#[derive(Clone, Copy)]
struct Bin {
    index: i64,
    point: Point,
    observations: u32,
}

struct Boundary {
    id: i32,
    bins: [Option<Bin>; BIN_COUNT],
}

impl Default for Boundary {
    fn default() -> Self {
        Self {
            id: i32::MIN,
            bins: [None; BIN_COUNT],
        }
    }
}

impl Boundary {
    fn observe(&mut self, index: i64, point: Point, x: f64, degraded: bool) {
        let slot = index.rem_euclid(BIN_COUNT as i64) as usize;
        let alpha = 0.15 * ((60. - x) / 40.).clamp(0.1, 1.);
        if let Some(bin) = self.bins[slot].as_mut().filter(|b| b.index == index) {
            bin.point.0 += alpha * (point.0 - bin.point.0);
            bin.point.1 += alpha * (point.1 - bin.point.1);
            bin.observations = bin.observations.saturating_add(1);
        } else {
            // When yaw falls back to camera motion, the first far-field batches
            // can disagree with the extrapolated line. Blend from that line.
            let point = if degraded && x > NEAR {
                self.point(index as f64 * BIN_M)
                    .map_or(point, |((x, y), _)| {
                        (x + alpha * (point.0 - x), y + alpha * (point.1 - y))
                    })
            } else {
                point
            };
            self.bins[slot] = Some(Bin {
                index,
                point,
                observations: 1,
            });
        }
    }

    fn last(&self) -> Option<i64> {
        self.bins.iter().flatten().map(|b| b.index).max()
    }

    /// Interpolate between nearby bins, using a curvature fit for long forward extensions.
    fn point(&self, s: f64) -> Option<(Point, Point)> {
        let index = bin_number(s);
        let bins = || self.bins.iter().flatten();
        let before = bins().filter(|b| b.index <= index).max_by_key(|b| b.index);
        let after = bins().filter(|b| b.index > index).min_by_key(|b| b.index);
        let (a, b) = match (before, after) {
            (Some(a), Some(b)) => (a, b),
            (Some(b), _) => (
                bins()
                    .filter(|a| (5..=15).contains(&(b.index - a.index)))
                    .max_by_key(|a| a.index)?,
                b,
            ),
            (None, Some(a)) => (
                a,
                bins()
                    .filter(|b| (5..=15).contains(&(b.index - a.index)))
                    .min_by_key(|b| b.index)?,
            ),
            _ => return None,
        };
        let (dx, dy) = (b.point.0 - a.point.0, b.point.1 - a.point.1);
        let len = dx.hypot(dy);
        if len < 1e-3 {
            return None;
        }
        let f = (s / BIN_M - a.index as f64) / (b.index - a.index) as f64;
        let linear = (a.point.0 + f * dx, a.point.1 + f * dy);
        let tangent = (dx / len, dy / len);
        if after.is_none() {
            // Close to the mapped road, retain its tangent. Blend in the curvature farther ahead.
            let distance = s - b.index as f64 * BIN_M;
            let weight = ((distance - NEAR) / NEAR).clamp(0., 1.);
            if weight > 0. {
                if let Some((curve, direction)) = self.extrapolate(s, b) {
                    let mix = |a, b| a + weight * (b - a);
                    let (ux, uy): Point =
                        (mix(tangent.0, direction.0), mix(tangent.1, direction.1));
                    let length = ux.hypot(uy);
                    if length >= 1e-3 {
                        return Some((
                            (mix(linear.0, curve.0), mix(linear.1, curve.1)),
                            (ux / length, uy / length),
                        ));
                    }
                }
            }
        }
        Some((linear, tangent))
    }

    fn extrapolate(&self, s: f64, edge: &Bin) -> Option<(Point, Point)> {
        // A single short segment amplifies lane-point error at long range.
        // Fit all samples in the last 24 m and bound the amount of curvature carried forward.
        let mut moments = [0.; 5];
        let (mut mx, mut my) = ([0.; 3], [0.; 3]);
        let mut span = 0;
        for bin in self.bins.iter().flatten() {
            let offset = bin.index - edge.index;
            if offset.abs() > EXTRAPOLATION_BINS {
                continue;
            }
            span = span.max(offset.abs());
            let offset = offset as f64 * BIN_M;
            let mut power = 1.;
            for k in 0..5 {
                moments[k] += power;
                if k < 3 {
                    mx[k] += power * (bin.point.0 - edge.point.0);
                    my[k] += power * (bin.point.1 - edge.point.1);
                }
                power *= offset;
            }
        }
        let [n, a, b, c, d] = moments;
        let linear_det = n * b - a * a;
        if span < 5 || linear_det <= 1e-6 {
            return None;
        }
        let quadratic_det = n * (b * d - c * c) - a * (a * d - c * b) + b * (a * c - b * b);
        let s = s - edge.index as f64 * BIN_M;
        let curve_s = s.clamp(-MAX_CURVE_EXTENSION, MAX_CURVE_EXTENSION);
        let quadratic_slope = (n * c - a * b) / linear_det;
        let fit = |[p, q, r]: [f64; 3]| {
            let slope = (n * q - a * p) / linear_det;
            let curve = if n >= 5. && quadratic_det > 1e-6 {
                (n * (b * r - q * c) - a * (a * r - q * b) + p * (a * c - b * b)) / quadratic_det
            } else {
                0.
            };
            let curve = curve.clamp(-MAX_CURVE_COEFFICIENT, MAX_CURVE_COEFFICIENT);
            let position = slope * s + curve * (curve_s * (2. * s - curve_s) - quadratic_slope * s);
            let tangent = slope + curve * (2. * curve_s - quadratic_slope);
            (position, tangent)
        };
        let (x, dx) = fit(mx);
        let (y, dy) = fit(my);
        let length = dx.hypot(dy);
        (length >= 1e-3).then(|| {
            (
                (edge.point.0 + x, edge.point.1 + y),
                (dx / length, dy / length),
            )
        })
    }

    fn lateral(&self, s: f64, p: Point) -> Option<f64> {
        let ((x, y), (ux, uy)) = self.point(s)?;
        Some(-uy * (p.0 - x) + ux * (p.1 - y))
    }
}

fn bin_number(s: f64) -> i64 {
    (s / BIN_M).floor() as i64
}

struct Observation {
    t: f64,
    line: usize, // lane bin ring index or one of the two road edges
    id: i32,
    index: i64,
    point: Point,
    x: f64,
    degraded: bool,
}

pub struct LaneMap {
    bounds: [Boundary; BOUNDARY_COUNT],
    edges: [Boundary; 2],
    offset: i32,
    renumbers: u32,
    pending: VecDeque<Observation>,
    frozen: bool,
    stamp: Option<f64>,
    s: f64,
}

impl Default for LaneMap {
    fn default() -> Self {
        Self {
            bounds: std::array::from_fn(|_| Boundary::default()),
            edges: std::array::from_fn(|_| Boundary::default()),
            offset: 0,
            renumbers: 0,
            pending: VecDeque::with_capacity(PENDING_CAPACITY),
            frozen: false,
            stamp: None,
            s: 0.,
        }
    }
}

pub(super) fn line_at(line: &Line, x: f64) -> Option<f64> {
    if line.x.len() != line.y.len() || line.x.len() < 2 {
        return None;
    }
    for (xs, ys) in line.x.windows(2).zip(line.y.windows(2)) {
        let (a, b) = (f64::from(xs[0]), f64::from(xs[1]));
        if a <= x && x <= b && b > a && ys.iter().all(|y| y.is_finite()) {
            let (ya, yb) = (f64::from(ys[0]), f64::from(ys[1]));
            // modelV2 is right-positive. The radar and world pose are left-positive.
            return Some(-(ya + (yb - ya) * (x - a) / (b - a)));
        }
    }
    None
}

fn usable_lane(model: &Model, i: usize) -> Option<&Line> {
    if matches!(i, 0 | 3) && !model.lane_line_probs.get(i).is_some_and(|&p| p >= 0.3) {
        return None;
    }
    model.lane_lines.get(i)
}

/// The quadratic fit to the centre of the inner lines gives heading and curvature at x = 0.
pub fn near_heading(model: &Model) -> Option<(f64, f64)> {
    let (left, right) = (model.lane_lines.get(1)?, model.lane_lines.get(2)?);
    let mut s = [0.; 5];
    let mut rhs = [0.; 3];
    let mut n = 0;
    // The far portion of modelV2 can still face the old lane during a change.
    for x in MODEL_GRID[..5].iter().copied().map(f64::from) {
        let (Some(a), Some(b)) = (line_at(left, x), line_at(right, x)) else {
            continue;
        };
        if !(1.8..=5.5).contains(&(a - b)) {
            continue;
        }
        let y = 0.5 * (a + b);
        let mut power = 1.;
        for k in 0..5 {
            s[k] += power;
            if k < 3 {
                rhs[k] += power * y;
            }
            power *= x;
        }
        n += 1;
    }
    if n < 4 {
        return None;
    }
    let m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let det = |[[a, b, c], [d, e, f], [g, h, i]]: [[f64; 3]; 3]| {
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    };
    let d = det(m);
    if d.abs() < 1e-9 {
        return None;
    }
    let coef = |k| {
        let mut replaced = m;
        for i in 0..3 {
            replaced[i][k] = rhs[i];
        }
        det(replaced) / d
    };
    Some((coef(1).atan(), 2. * coef(2)))
}

impl LaneMap {
    pub fn ingest(&mut self, t: f64, model: &Model, pose: &Pose, frozen: bool) {
        self.ingest_with_walls(t, model, pose, frozen, false, &[]);
    }

    pub(super) fn ingest_with_walls(
        &mut self,
        t: f64,
        model: &Model,
        pose: &Pose,
        frozen: bool,
        degraded: bool,
        walls: &[Vec<Point>],
    ) {
        if !t.is_finite() || self.stamp.is_some_and(|last| t <= last) {
            return;
        }
        self.stamp = Some(t);
        self.set_frozen(frozen);
        let at = pose.at(t);
        self.s = at.s;
        let min_bin = bin_number(at.s - 40.);
        for bound in self.bounds.iter_mut().chain(self.edges.iter_mut()) {
            for bin in &mut bound.bins {
                if bin.is_some_and(|b| b.index < min_bin) {
                    *bin = None;
                }
            }
        }
        self.match_lines(model, at);
        for i in 0..4 {
            let Some(line) = usable_lane(model, i) else {
                continue;
            };
            let id = self.offset + i as i32;
            let ring = id.rem_euclid(BOUNDARY_COUNT as i32) as usize;
            self.ingest_line(t, at, ring, id, degraded, |x| line_at(line, x));
        }
        for side in 0..2 {
            let outer = side * 3;
            let inner = 1 + side;
            let sign = if side == 0 { 1. } else { -1. };
            self.ingest_line(t, at, BOUNDARY_COUNT + side, side as i32, degraded, |x| {
                let inside = line_at(model.lane_lines.get(inner)?, x)?;
                if let Some(y) = model
                    .road_edges
                    .get(side)
                    .and_then(|edge| line_at(edge, x))
                    .filter(|y| sign * (y - inside) > 0.)
                {
                    return Some(y);
                }
                if usable_lane(model, outer).is_some() {
                    return None;
                }
                let expected = model
                    .lane_lines
                    .get(outer)
                    .and_then(|line| line_at(line, x))
                    .unwrap_or(inside + sign * LANE_WIDTH);
                walls
                    .iter()
                    .filter_map(|wall| wall_at(wall, x))
                    .map(|y| y - sign * 0.5)
                    .filter(|y| sign * (y - inside) > 0. && (y - expected).abs() <= 3.)
                    .min_by(|a, b| (a - expected).abs().total_cmp(&(b - expected).abs()))
            });
        }
        while self.pending.front().is_some_and(|o| t - o.t >= 1.) {
            let obs = self.pending.pop_front().unwrap();
            if obs.index >= min_bin {
                self.apply(obs);
            }
        }
    }

    fn ingest_line(
        &mut self,
        t: f64,
        at: PoseAt,
        line: usize,
        id: i32,
        degraded: bool,
        sample: impl Fn(f64) -> Option<f64>,
    ) {
        for x in MODEL_GRID[..9].iter().copied().map(f64::from) {
            if self.frozen && x > NEAR {
                break;
            }
            let Some(y) = sample(x) else {
                continue;
            };
            let obs = Observation {
                t,
                line,
                id,
                index: bin_number(at.s + x),
                point: at.to_world(x, y),
                x,
                degraded,
            };
            if x > NEAR {
                // ponytail: drop the oldest sample if modelV2 outruns the 20 Hz queue budget.
                if self.pending.len() == PENDING_CAPACITY {
                    self.pending.pop_front();
                }
                self.pending.push_back(obs);
            } else {
                self.apply(obs);
            }
        }
    }

    fn apply(&mut self, o: Observation) {
        let bound = if o.line < BOUNDARY_COUNT {
            &mut self.bounds[o.line]
        } else {
            &mut self.edges[o.line - BOUNDARY_COUNT]
        };
        if bound.id != o.id {
            // A delayed far-field observation must never take back a reused boundary slot.
            if o.x > NEAR || (self.frozen && bound.last().is_some()) {
                return;
            }
            bound.bins.fill(None);
            bound.id = o.id;
        }
        if !self.frozen || bound.last().is_none_or(|last| o.index > last) {
            bound.observe(o.index, o.point, o.x, o.degraded);
        }
    }

    fn match_lines(&mut self, model: &Model, at: PoseAt) {
        let mut votes = [None; 4];
        for (i, vote) in votes.iter_mut().enumerate() {
            let Some(line) = usable_lane(model, i) else {
                continue;
            };
            let mut best: Option<(f64, i32)> = None;
            for boundary in &self.bounds {
                let (mut sum, mut count) = (0., 0);
                for x in MODEL_GRID[..5].iter().copied().map(f64::from) {
                    if let Some(y) = line_at(line, x) {
                        if let Some(distance) = boundary.lateral(at.s + x, at.to_world(x, y)) {
                            sum += distance.abs();
                            count += 1;
                        }
                    }
                }
                let mean = sum / f64::from(count.max(1));
                if count >= 3 && mean <= 1.2 && best.is_none_or(|b| (mean, boundary.id) < b) {
                    best = Some((mean, boundary.id));
                }
            }
            *vote = best.map(|(_, id)| id - i as i32);
        }
        let count = |offset| votes.iter().filter(|&&v| v == Some(offset)).count();
        if let Some(best) = votes.iter().flatten().copied().max_by_key(|&v| count(v)) {
            let winning_votes = count(best);
            let tied = votes
                .iter()
                .flatten()
                .any(|&v| v != best && count(v) == winning_votes);
            // A lane split can move one line by a lane width. A second line must support a renumber.
            if !tied && winning_votes >= 2 && best != self.offset {
                self.offset = best;
                self.renumbers += 1;
            }
        }
    }

    pub(super) fn camera_ego(&self, model: &Model) -> Option<f64> {
        if !model.lane_line_probs.get(1..3)?.iter().all(|&p| p > 0.4) {
            return None;
        }
        let left = line_at(model.lane_lines.get(1)?, 0.)?;
        let right = line_at(model.lane_lines.get(2)?, 0.)?;
        let width = left - right;
        (1.8..=5.5)
            .contains(&width)
            .then(|| f64::from(self.offset) + 1. + left / width)
    }

    pub fn coord(&self, s: f64, p: Point) -> Option<f64> {
        if !p.0.is_finite() || !p.1.is_finite() || !(-40. ..=80.).contains(&(s - self.s)) {
            return None;
        }
        for (i, edge) in self.edges.iter().enumerate() {
            if edge
                .lateral(s, p)
                .is_some_and(|y| if i == 0 { y > 0. } else { y < 0. })
            {
                return None;
            }
        }
        let mut distances = [(0, 0.); BOUNDARY_COUNT];
        let mut n = 0;
        for b in &self.bounds {
            if let Some(y) = b.lateral(s, p) {
                distances[n] = (b.id, y);
                n += 1;
            }
        }
        let d = &mut distances[..n];
        d.sort_unstable_by_key(|&(k, _)| k);
        let (mut first, mut prior) = (*d.first()?, d[0]);
        let mut count = 1;
        for &next in &d[1..] {
            let width = (next.1 - prior.1) / f64::from(next.0 - prior.0);
            if !(1.8..=5.5).contains(&width) {
                if count == 1 {
                    first = next;
                    prior = next;
                }
                continue;
            }
            count += 1;
            if first.1 > 0. {
                return Some(f64::from(first.0) - first.1 / LANE_WIDTH);
            }
            if prior.1 <= 0. && next.1 > 0. {
                return Some(f64::from(prior.0) - prior.1 / width);
            }
            prior = next;
        }
        (count >= 2).then(|| f64::from(prior.0) - prior.1 / LANE_WIDTH)
    }

    pub fn width_at(&self, s: f64) -> f64 {
        if !(-40. ..=80.).contains(&(s - self.s)) {
            return LANE_WIDTH;
        }
        let width = |left: i32| {
            let ((x, y), _) = self.bound(left)?.point(s)?;
            self.bound(left + 1)?
                .lateral(s, (x, y))
                .filter(|w| (1.8..=5.5).contains(w))
        };
        [1, 0, 2]
            .iter()
            .find_map(|i| width(self.offset + i))
            .unwrap_or(LANE_WIDTH)
    }

    fn bound(&self, id: i32) -> Option<&Boundary> {
        let boundary = &self.bounds[id.rem_euclid(BOUNDARY_COUNT as i32) as usize];
        (boundary.id == id).then_some(boundary)
    }

    pub fn observations(&self, boundary: i32, s: f64) -> u32 {
        let index = bin_number(s);
        self.bound(boundary)
            .and_then(|b| b.bins[index.rem_euclid(BIN_COUNT as i64) as usize])
            .filter(|b| b.index == index)
            .map_or(0, |b| b.observations)
    }

    pub fn renumbers(&self) -> u32 {
        self.renumbers
    }

    pub(super) fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
        if frozen {
            self.pending.clear();
        }
    }

    pub(super) fn clear(&mut self) {
        for bound in self.bounds.iter_mut().chain(self.edges.iter_mut()) {
            bound.id = i32::MIN;
            bound.bins.fill(None);
        }
        self.offset = 0;
        self.renumbers = 0;
        self.pending.clear();
        self.frozen = false;
        self.stamp = None;
        self.s = 0.;
    }
}

fn wall_at(wall: &[Point], x: f64) -> Option<f64> {
    let i = wall.partition_point(|p| p.0 < x);
    if i == 0 {
        return wall.first().filter(|p| p.0 == x).map(|p| p.1);
    }
    let (&(a, y), &(b, z)) = (wall.get(i - 1)?, wall.get(i)?);
    (b > a).then(|| y + (z - y) * (x - a) / (b - a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_extends_a_curve_past_its_last_observation() {
        let mut boundary = Boundary::default();
        for index in 0..=12 {
            let s = index as f64 * BIN_M;
            boundary.observe(index, (s, 0.0005 * s * s), s, false);
        }
        let ((_, near_y), _) = boundary.point(30.).unwrap();
        assert!((near_y - 0.402).abs() < 1e-9);
        let ((far_x, far_y), (dx, dy)) = boundary.point(74.).unwrap();
        assert!((far_x - 74.).abs() < 1e-9);
        assert!((far_y - 2.7255).abs() < 1e-9);
        assert!((dy / dx - 0.069).abs() < 1e-9);
    }
}
