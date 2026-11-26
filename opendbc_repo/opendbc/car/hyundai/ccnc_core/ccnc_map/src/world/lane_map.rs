use std::collections::VecDeque;

use crate::{display::ModelPath, input::model_v2::{Line, Model}};

use super::{Pose, PoseAt};

pub const MODEL_GRID: [f32; 10] = [0., 5., 10., 15., 20., 30., 40., 50., 60., 80.];
const LANE_WIDTH: f64 = 3.5;
// In replay, line and path error are similar through 35–40 m. The line's
// p95 error rises abruptly by 50 m, including high-probability lines.
const PATH_BLEND_START: f64 = 35.;
const PATH_BLEND_END: f64 = 50.;
const BIN_M: f64 = 2.;
const BIN_COUNT: usize = 70;
const BOUNDARY_COUNT: usize = 8;
const PENDING_CAPACITY: usize = 512;
const NEAR: f64 = 25.;
/// The near field ends at the bin of the last near grid point (20 m), whose centre is at most 21 m ahead.
const NEAR_FIELD: f64 = 21.;
/// A near bin off the grid points takes the line when it has not for this long (s). On the move the grid points pass
/// over every bin more often than this from 10 m/s up; at a stop they never reach the bins between them.
const STALE_S: f64 = 0.5;
const EXTRAPOLATION_BINS: i64 = 12;
const MAX_CURVE_EXTENSION: f64 = 45.;
const MAX_CURVE_COEFFICIENT: f64 = 0.0007;
/// Two bins this close give no tangent (`Boundary::spread`); bins are 2 m apart.
const MIN_TANGENT_M: f64 = 0.5;

type Point = (f64, f64);

#[derive(Clone, Copy)]
struct Bin {
    index: i64,
    point: Point,
    /// Confidence-weighted distant estimate for checking a radar row.
    /// The observed point still anchors ego and near-lane tracking.
    far_point: Point,
    observations: u32,
    /// When the bin last took a point.
    seen: f64,
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
    fn observe(&mut self, index: i64, point: Point, x: f64, degraded: bool, t: f64, far: (Point, f64)) {
        let slot = index.rem_euclid(BIN_COUNT as i64) as usize;
        let alpha = 0.15 * ((60. - x) / 40.).clamp(0.1, 1.);
        if let Some(bin) = self.bins[slot].as_mut().filter(|b| b.index == index) {
            bin.point.0 += alpha * (point.0 - bin.point.0);
            bin.point.1 += alpha * (point.1 - bin.point.1);
            if x > PATH_BLEND_START {
                let (target, probability) = far;
                let history = f64::from(bin.observations.min(6)) / 6.;
                let trust = far_weight(x) * (1. - probability);
                let gain = alpha * (1. - trust * (0.25 + 0.25 * history));
                bin.far_point.0 += gain * (target.0 - bin.far_point.0);
                bin.far_point.1 += gain * (target.1 - bin.far_point.1);
            } else {
                bin.far_point = bin.point;
            }
            bin.observations = bin.observations.saturating_add(1);
            bin.seen = t;
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
                far_point: if x > PATH_BLEND_START { far.0 } else { point },
                observations: 1,
                seen: t,
            });
        }
    }

    fn last(&self) -> Option<i64> {
        self.bins.iter().flatten().map(|b| b.index).max()
    }

    /// Interpolate between nearby bins, using a curvature fit for long forward extensions.
    fn point(&self, s: f64) -> Option<(Point, Point)> {
        self.point_for(s, false)
    }

    fn point_for(&self, s: f64, far: bool) -> Option<(Point, Point)> {
        let index = bin_number(s);
        let bins = || self.bins.iter().flatten();
        let before = bins().filter(|b| b.index <= index).max_by_key(|b| b.index);
        let after = bins().filter(|b| b.index > index).min_by_key(|b| b.index);
        let (a, b) = match (before, after) {
            (Some(a), Some(b)) => self.spread(a, b),
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
        let (p, q) = if far && before.is_some() && after.is_some() {
            (a.far_point, b.far_point)
        } else {
            (a.point, b.point)
        };
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let len = dx.hypot(dy);
        if len < 1e-3 {
            return None;
        }
        let f = (s / BIN_M - a.index as f64) / (b.index - a.index) as f64;
        let linear = (p.0 + f * dx, p.1 + f * dy);
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

    /// Two neighbouring bins less than `MIN_TANGENT_M` apart along the road give no direction: their lateral
    /// difference would set it. Reach one bin farther on either side until the pair spans that much along the road,
    /// measured on the chord of the bins around the pair.
    fn spread<'a>(&'a self, mut a: &'a Bin, mut b: &'a Bin) -> (&'a Bin, &'a Bin) {
        let bins = || self.bins.iter().flatten();
        loop {
            let earlier = bins().filter(|x| x.index < a.index).max_by_key(|x| x.index);
            let later = bins().filter(|x| x.index > b.index).min_by_key(|x| x.index);
            let (from, to) = (earlier.unwrap_or(a).point, later.unwrap_or(b).point);
            let (rx, ry) = (to.0 - from.0, to.1 - from.1);
            let along = (rx * (b.point.0 - a.point.0) + ry * (b.point.1 - a.point.1)) / rx.hypot(ry).max(1e-9);
            if along >= MIN_TANGENT_M {
                return (a, b);
            }
            match (earlier, later) {
                (Some(e), Some(l)) if l.index - b.index < a.index - e.index => b = l,
                (Some(e), _) => a = e,
                (None, Some(l)) => b = l,
                (None, None) => return (a, b),
            }
        }
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
        self.lateral_for(s, p, false)
    }

    fn lateral_for(&self, s: f64, p: Point, far: bool) -> Option<f64> {
        let ((x, y), (ux, uy)) = self.point_for(s, far)?;
        Some(-uy * (p.0 - x) + ux * (p.1 - y))
    }
}

fn far_weight(x: f64) -> f64 {
    ((x - PATH_BLEND_START) / (PATH_BLEND_END - PATH_BLEND_START)).clamp(0., 1.)
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
    far: (Point, f64),
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

const BACK_SLOPE_M: f64 = 3.;

pub(super) fn line_at(line: &Line, x: f64) -> Option<f64> {
    if line.x.len() != line.y.len() || line.x.len() < 2 {
        return None;
    }
    // The bin ego stands in has its centre up to 1 m behind: carry the line back that far along its slope over the
    // first `BACK_SLOPE_M`. The first modelV2 segment is only 0.19 m long, so its slope is about 5 times noisier.
    let first = f64::from(line.x[0]);
    if x < first && x >= first - 0.5 * BIN_M {
        let points = line.x.iter().zip(&line.y).map(|(&x, &y)| (f64::from(x), f64::from(y)));
        let (b, yb) = points.clone().skip(1).find(|&(b, _)| b >= first + BACK_SLOPE_M).or(points.last())?;
        let (a, ya) = (first, f64::from(line.y[0]));
        return (b > a && ya.is_finite() && yb.is_finite()).then(|| -(ya + (yb - ya) * (x - a) / (b - a)));
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

pub(super) struct CameraLanes {
    pub(super) stamp: f64,
    pose: PoseAt,
    points: [[f32; MODEL_GRID.len()]; 4],
}

pub(crate) struct RoadAhead {
    pub(super) stamp: f64,
    pub(super) path: ModelPath,
    pub(super) changing: bool,
    pub(super) pose: PoseAt,
    weak_lines: bool,
    width: f64,
    reference_u: Option<f64>,
    sides: [[SideGeometry; MODEL_GRID.len()]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SideGeometry {
    Unknown,
    Lane,
    Edge,
}

impl RoadAhead {
    /// Lateral position relative to the predicted path. Its origin cancels
    /// when two returns are compared against the same path.
    pub(crate) fn project_row(&self, wx: f32, wy: f32) -> Option<f64> {
        let (x, y) = self.pose.to_ego(f64::from(wx), f64::from(wy));
        let (lateral, confidence) = self.path.project_lateral(x, y, self.width)?;
        (confidence >= 0.65).then_some(lateral)
    }

    fn line_reference(&self, point: Point) -> Option<f64> {
        self.reference_u?;
        let (x, y) = self.pose.to_ego(point.0, point.1);
        let (center, _, confidence) = self.path.road_sample(x, self.width)?;
        (confidence >= 0.65).then_some(y - center)
    }

    pub(super) fn new(
        model: &Model, pose: PoseAt, stamp: f64, path: ModelPath, changing: bool,
        edge_stds: [f32; 2], width: Option<(f64, f64)>,
    ) -> Self {
        let width = width.filter(|(t, _)| stamp - t <= 5.).map_or(LANE_WIDTH, |(_, w)| w);
        let weak_lines = (1..=2).all(|i| !model.lane_line_probs.get(i).is_some_and(|&p| p >= 0.4));
        let sides = std::array::from_fn(|side| {
            let outward = if side == 0 { 1. } else { -1. };
            let outer = usable_lane(model, side * 3);
            MODEL_GRID.map(|x| {
                let x = f64::from(x);
                let Some(inner) = model.lane_lines.get(1 + side).and_then(|line| line_at(line, x)) else {
                    return SideGeometry::Unknown;
                };
                let beyond_inner = |line| line_at(line, x).map(|y| outward * (y - inner));
                let marked = outer.and_then(beyond_inner).is_some_and(|room| (1.8..=5.5).contains(&room));
                let edge = model.road_edges.get(side).and_then(beyond_inner)
                    .filter(|_| edge_stds[side] <= 1.);
                if edge.is_some_and(|room| (0. ..=0.50 * width).contains(&room)) {
                    SideGeometry::Edge
                } else if marked || edge.is_some_and(|room| room >= 0.70 * width) {
                    SideGeometry::Lane
                } else {
                    SideGeometry::Unknown
                }
            })
        });
        Self { stamp, path, changing, pose, weak_lines, width, reference_u: None, sides }
    }

    pub(super) fn align(&mut self, map: &LaneMap, ego_u: f64) {
        // Preserve the nearby lane's lateral origin while following the shape
        // of position farther out. Samples in another lane cannot set it.
        const ANCHOR_X: f64 = 20.;
        let Some((y, _, _)) = self.path.road_sample(ANCHOR_X, self.width).filter(|_| !self.weak_lines) else {
            return;
        };
        self.reference_u = map.coord(self.pose.s + ANCHOR_X, self.pose.to_world(ANCHOR_X, y))
            .filter(|&u| (u - ego_u).abs() <= 0.30);
    }

    pub(super) fn coord(&self, point: Point, ego_u: f64, mapped: Option<f64>, crossing: bool, yaw: f64) -> Option<f64> {
        let observed_u = mapped?;
        let (x, y) = self.pose.to_ego(point.0, point.1);
        if !ego_u.is_finite() || x <= PATH_BLEND_START || x > 80. {
            return mapped;
        }
        let Some((lateral, confidence)) = self.path.project_lateral(x, y, self.width) else {
            return mapped;
        };
        let origin = self.reference_u.unwrap_or(ego_u);
        let path_u = origin - lateral / self.width;
        let observed = observed_u - ego_u;
        let predicted = path_u - ego_u;
        let side = usize::from(predicted > 0.);
        // A far return cannot establish a side lane past a nearby and
        // distant shoulder. Require agreement with the observed placement
        // and a confident prediction that the return is outside our lane.
        if self.weak_lines && x >= PATH_BLEND_END && confidence >= 0.5
            && observed.abs() >= 0.5 && predicted.abs() >= 0.7
            && observed.signum() == predicted.signum()
            && self.side_geometry(x, side) == SideGeometry::Edge
            && self.side_geometry(20., side) == SideGeometry::Edge
        {
            return None;
        }
        // Straight-road lateral jitter is already constrained by the
        // observed map. Correct its distant fit only with a measured bend.
        if crossing || self.path.bend_weight(yaw) == 0. {
            return mapped;
        }

        // Keep an observed lane assignment through forks and merges. Use
        // the more accurate far curve only if both fits agree on the lane
        // and the point is clear of a possibly moving boundary.
        if observed.abs().max(predicted.abs()) > 0.22 {
            return mapped;
        }
        let fade = ((x - PATH_BLEND_START) / (PATH_BLEND_END - PATH_BLEND_START)).clamp(0., 1.);
        let weight = confidence * fade * fade * (3. - 2. * fade);
        Some(observed_u + weight * (path_u - observed_u))
    }

    pub(super) fn radar_lateral(&self, x: f64, y: f64, yaw: f64) -> Option<f64> {
        let bend = self.path.bend_weight(yaw);
        if bend == 0. {
            return None;
        }
        // The radar blend expresses lateral displacement from ego. Remove
        // the model path's intercept so a shift of its origin cannot move cars.
        let (near, _) = self.path.project_lateral(0., 0., self.width)?;
        let (far, confidence) = self.path.project_lateral(x, y, self.width)?;
        let projected = far - near;
        let lane_offset = (projected / self.width).abs();
        // Paths past a junction can bend away while the tracked car is still
        // in the current lane. Fade the projection as the two disagree.
        let agreement = ((0.55 - lane_offset) / 0.20).clamp(0., 1.);
        Some(y + (bend * confidence * agreement) * (projected - y))
    }

    fn side_geometry(&self, x: f64, side: usize) -> SideGeometry {
        let x = x.clamp(0., 80.);
        let i = MODEL_GRID.partition_point(|&d| f64::from(d) < x);
        let end = self.sides[side][i];
        if f64::from(MODEL_GRID[i]) == x || end == self.sides[side][i - 1] {
            end
        } else {
            SideGeometry::Unknown
        }
    }
}

pub(super) fn observed_width(model: &Model) -> Option<f64> {
    let probs = model.lane_line_probs.get(1..3)?;
    if !probs.iter().all(|&p| p >= 0.4) {
        return None;
    }
    let (left, right) = (model.lane_lines.get(1)?, model.lane_lines.get(2)?);
    let width = line_at(left, 10.)? - line_at(right, 10.)?;
    (1.8..=5.5).contains(&width).then_some(width)
}

impl CameraLanes {
    pub(super) fn observe(model: &Model, pose: PoseAt, stamp: f64) -> Option<Self> {
        if !model.lane_line_probs.get(1..3)?.iter().all(|&p| p >= 0.4) {
            return None;
        }
        let points = std::array::from_fn(|i| {
            let line = usable_lane(model, i);
            MODEL_GRID.map(|x| {
                line.and_then(|l| line_at(l, f64::from(x)))
                    .map_or(f32::NAN, |p| p as f32)
            })
        });
        Some(Self {
            stamp,
            pose,
            points,
        })
    }

    pub(super) fn coord(&self, point: Point) -> Option<f64> {
        let (x, y) = self.pose.to_ego(point.0, point.1);
        if !(-2. ..=80.).contains(&x) {
            return None;
        }
        let x = x.max(0.);
        let next = MODEL_GRID.partition_point(|&s| f64::from(s) < x);
        let sample = |i: usize| {
            let b = f64::from(self.points[i][next]);
            let x1 = f64::from(MODEL_GRID[next]);
            if x == x1 {
                return b;
            }
            let a = f64::from(self.points[i][next - 1]);
            let x0 = f64::from(MODEL_GRID[next - 1]);
            a + (b - a) * (x - x0) / (x1 - x0)
        };
        let (left, right) = (sample(1), sample(2));
        let width = left - right;
        if !(1.8..=5.5).contains(&width) {
            return None;
        }
        if y > left {
            let w = sample(0) - left;
            if (1.8..=5.5).contains(&w) {
                return Some(1. + (left - y) / w);
            }
        } else if y < right {
            let w = right - sample(3);
            if (1.8..=5.5).contains(&w) {
                return Some(2. + (right - y) / w);
            }
        }
        Some(1. + (left - y) / width)
    }
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
        self.ingest_with_walls(t, model, pose, frozen, false, &[], None);
    }

    pub(super) fn ingest_with_walls(
        &mut self,
        t: f64,
        model: &Model,
        pose: &Pose,
        frozen: bool,
        degraded: bool,
        walls: &[Vec<Point>],
        road: Option<&RoadAhead>,
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
            let probability = model.lane_line_probs.get(i).copied()
                .filter(|p| (0. ..=1.).contains(p)).map_or(0., f64::from);
            let reference = road.zip(line_at(line, 20.))
                .and_then(|(road, y)| road.line_reference(at.to_world(20., y)));
            self.ingest_line(t, at, ring, id, degraded, probability, |x| {
                let y = line_at(line, x)?;
                let correction = reference.zip(road).filter(|_| x > PATH_BLEND_START)
                    .and_then(|(reference, road)| {
                        let offset = road.line_reference(at.to_world(x, y))?;
                        let shift = (reference - offset).clamp(-0.5 * road.width, 0.5 * road.width);
                        Some(shift * (at.heading - road.pose.heading).cos())
                    })
                    .unwrap_or(0.);
                Some((y, y + far_weight(x) * (1. - probability) * correction))
            });
        }
        for side in 0..2 {
            let outer = side * 3;
            let inner = 1 + side;
            let sign = if side == 0 { 1. } else { -1. };
            self.ingest_line(t, at, BOUNDARY_COUNT + side, side as i32, degraded, 1., |x| {
                let inside = line_at(model.lane_lines.get(inner)?, x)?;
                if let Some(y) = model
                    .road_edges
                    .get(side)
                    .and_then(|edge| line_at(edge, x))
                    .filter(|y| sign * (y - inside) > 0.)
                {
                    return Some((y, y));
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
                    .map(|y| (y, y))
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
        probability: f64,
        sample: impl Fn(f64) -> Option<(f64, f64)>,
    ) {
        // The near field (through the bin of the 20 m grid point) takes the line at the bins of its grid points, and
        // at every other bin that has not had it for STALE_S. On the grid points only, the bins between them kept their
        // last points while ego stood still (33 s in 11-24 0000047c segments 5-6). Taking every bin each frame made
        // the map follow the model's relabelled lines in an ego lane change and lose the change (65 replayed
        // segments: 50 -> 38 clean changes). Farther out the grid points feed the 1 s queue.
        let (first, frozen) = (bin_number(at.s), self.frozen);
        let grid: Vec<i64> = MODEL_GRID[..5].iter().map(|&x| bin_number(at.s + f64::from(x))).collect();
        let near: Vec<i64> = (first..)
            .take_while(|&i| (i as f64 + 0.5) * BIN_M - at.s <= NEAR_FIELD)
            .filter(|i| grid.contains(i) || self.stale(line, id, *i, t))
            .collect();
        let far = MODEL_GRID[..9]
            .iter()
            .map(|&x| f64::from(x))
            .filter(|&x| x > NEAR && !frozen)
            .map(|x| bin_number(at.s + x));
        for index in near.into_iter().chain(far) {
            // Sample the line at the centre of the bin. A bin's point anywhere in its 2 m made two neighbouring points
            // meet when ego stopped with a grid point at a bin seam, and their tangent turned by 60 to 100 degrees
            // (11-24 0000047c segment 5 at 41.8 s).
            let x = (index as f64 + 0.5) * BIN_M - at.s;
            let Some((y, far_y)) = sample(x) else {
                continue;
            };
            let obs = Observation {
                t,
                line,
                id,
                index,
                point: at.to_world(x, y),
                far: (at.to_world(x, far_y), probability),
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

    fn stale(&self, line: usize, id: i32, index: i64, t: f64) -> bool {
        let bound = if line < BOUNDARY_COUNT { &self.bounds[line] } else { &self.edges[line - BOUNDARY_COUNT] };
        bound.id != id
            || bound.bins[index.rem_euclid(BIN_COUNT as i64) as usize]
                .filter(|b| b.index == index)
                .is_none_or(|b| t - b.seen >= STALE_S - 1e-6)
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
            bound.observe(o.index, o.point, o.x, o.degraded, o.t, o.far);
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
        self.coord_for(s, p, false)
    }

    pub(super) fn far_coord(&self, s: f64, p: Point) -> Option<f64> {
        self.coord_for(s, p, true)
    }

    fn coord_for(&self, s: f64, p: Point, far: bool) -> Option<f64> {
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
            if let Some(y) = b.lateral_for(s, p, far) {
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

    pub(super) fn camera_weight(&self, s: f64, u: f64) -> f64 {
        let left = u.floor() as i32;
        let last = |id| self.bound(id).and_then(Boundary::last);
        // When either boundary still has observations at this range, prefer the world map.
        last(left).zip(last(left + 1)).map_or(0., |(a, b)| {
            ((s - a.max(b) as f64 * BIN_M) / NEAR).clamp(0., 1.)
        })
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
    fn future_path_replaces_far_lane_geometry_and_requires_room_beyond_the_edge() {
        let xs = vec![0., 20., 40., 60., 80.];
        let line = |y| Line { x: xs.clone(), y: vec![y; xs.len()] };
        let model = Model {
            lane_lines: vec![line(-5.25), line(-1.75), line(1.75), line(5.25)],
            lane_line_probs: vec![0.9, 0.9, 0.9, 0.01],
            road_edges: vec![line(-5.6), line(2.4)],
        };
        let path = ModelPath::new(xs.clone(), vec![0., 0., -1., -3., -5.]).unwrap()
            .with_deviations(vec![0.3, 0.3, 0.3, 0.3, 3.5]);
        let road = RoadAhead::new(&model, PoseAt::default(), 0., path, false, [0.5, 0.5], Some((0., 3.5)));
        assert_eq!(road.coord((20., -3.), 1.5, Some(2.4), false, -0.04), Some(2.4));
        assert_eq!(road.coord((60., -3.), 1.5, Some(1.7), false, -0.04), Some(1.5));
        assert_eq!(road.coord((60., -3.), 1.5, Some(2.4), false, -0.04), Some(2.4));
        assert_eq!(road.coord((60., -3.), 1.5, Some(1.7), true, -0.04), Some(1.7));
        assert_eq!(road.coord((60., -3.), 1.5, None, false, -0.04), None);
        assert_eq!(road.coord((80., -5.), 1.5, Some(2.4), false, -0.04), Some(2.4));
        assert_eq!(road.side_geometry(60., 0), SideGeometry::Lane);
        assert_eq!(road.side_geometry(60., 1), SideGeometry::Edge);
    }

    #[test]
    fn a_line_carried_behind_its_first_point_follows_its_slope_over_the_first_metres() {
        // modelV2 X_IDXS starts 0, 0.1875, 0.75, 1.6875, 3 m. A straight line 0.1 m/m with 2 cm of noise on its second
        // point: the first segment alone would put the line 1 m behind at 0.2 m from where it is.
        let x = [0., 0.1875, 0.75, 1.6875, 3., 4.6875, 6.75];
        let mut y: Vec<f32> = x.iter().map(|x| 0.1 * x).collect();
        y[1] += 0.02;
        let line = Line { x: x.to_vec(), y };
        let behind = line_at(&line, -1.).unwrap();
        assert!((behind - 0.1).abs() < 1e-6, "{behind}");
        assert_eq!(line_at(&line, -1.1), None, "only one metre back");
        let short = Line { x: vec![0., 0.1875], y: vec![0., 0.01875] };
        assert!((line_at(&short, -1.).unwrap() - 0.1).abs() < 1e-6, "a short line takes its last point");
    }

    #[test]
    fn boundary_extends_a_curve_past_its_last_observation() {
        let mut boundary = Boundary::default();
        for index in 0..=12 {
            let s = index as f64 * BIN_M;
            let point = (s, 0.0005 * s * s);
            boundary.observe(index, point, s, false, 0., (point, 1.));
        }
        let ((_, near_y), _) = boundary.point(30.).unwrap();
        assert!((near_y - 0.402).abs() < 1e-9);
        let ((far_x, far_y), (dx, dy)) = boundary.point(74.).unwrap();
        assert!((far_x - 74.).abs() < 1e-9);
        assert!((far_y - 2.7255).abs() < 1e-9);
        assert!((dy / dx - 0.069).abs() < 1e-9);
    }

    #[test]
    fn two_bins_that_meet_take_their_tangent_from_a_farther_bin() {
        let mut boundary = Boundary::default();
        // Bins 4 and 5 hold points 0.05 m apart along the road and 0.6 m apart across it.
        boundary.observe(3, (7., 5.), 7., false, 0., ((7., 5.), 1.));
        boundary.observe(4, (9.95, 5.), 10., false, 0., ((9.95, 5.), 1.));
        boundary.observe(5, (10., 5.6), 10., false, 0., ((10., 5.6), 1.));
        boundary.observe(6, (13., 5.6), 13., false, 0., ((13., 5.6), 1.));
        let (_, (dx, dy)) = boundary.point(9.).unwrap();
        assert!(dy.atan2(dx).to_degrees().abs() < 15., "{dx} {dy}");
    }
}
