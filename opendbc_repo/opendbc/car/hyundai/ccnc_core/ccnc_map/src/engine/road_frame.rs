//! Local Frenet frame in the model's right-positive convention.

use super::RoadObservation;
use crate::input::model_v2::{Line, Model};
use crate::world::PoseAt;

pub(crate) const LANE_WIDTH: f64 = 3.5;
const END_EXTENSION: f64 = 5.;
type Point = (f64, f64);

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: Point,
    tangent: Point,
    length: f64,
    s: f64,
}

#[derive(Clone)]
pub(super) struct RoadFrame {
    segments: Vec<Segment>,
    origin: f64,
    forward_end: f64,
    // Distance, centre offset and width from trustworthy lane pairs.
    lanes: Vec<(f64, f64, f64)>,
}

impl RoadFrame {
    pub(super) fn new(
        path: &[Point],
        model: Option<&Model>,
        model_pose: PoseAt,
        ego: PoseAt,
        past: impl Iterator<Item = PoseAt>,
        changing_lanes: bool,
    ) -> Self {
        let to_ego = |(x, y): Point| {
            let (wx, wy) = model_pose.to_world(x, -y);
            let (x, y) = ego.to_ego(wx, wy);
            (x, -y)
        };
        let mut points: Vec<_> = past
            .map(|p| {
                let (x, y) = ego.to_ego(p.x, p.y);
                (x, -y)
            })
            .collect();
        points.push((0., 0.));
        let mut direction = None;
        for point in path
            .iter()
            .copied()
            .map(to_ego)
            .take_while(|(x, y)| x.is_finite() && y.is_finite())
            .skip_while(|p| p.0 < 0.35)
        {
            let previous = points[points.len() - 1];
            let (dx, dy) = (point.0 - previous.0, point.1 - previous.1);
            if dx.hypot(dy) < 0.01 {
                continue;
            }
            // A stopping forecast can fold back on itself.
            if direction.is_some_and(|(tx, ty)| dx * tx + dy * ty <= 0.) {
                break;
            }
            direction = Some((dx, dy));
            points.push(point);
        }
        if points.last().is_none_or(|p| p.0 < 1. && p.1.abs() < 1.) {
            points.push((50., 0.));
        }
        let mut s = 0.;
        let mut segments = Vec::with_capacity(points.len() - 1);
        for p in points.windows(2) {
            let (dx, dy) = (p[1].0 - p[0].0, p[1].1 - p[0].1);
            let length = dx.hypot(dy);
            if length < 0.01 {
                continue;
            }
            segments.push(Segment {
                start: p[0],
                tangent: (dx / length, dy / length),
                length,
                s,
            });
            s += length;
        }
        let mut frame = Self {
            segments,
            origin: 0.,
            forward_end: s,
            lanes: Vec::new(),
        };
        let last = *frame.segments.last().expect("forward segment");
        let (tx, ty) = last.tangent;
        frame.origin = frame.project(0., 0.).0;
        let end = (last.start.0 + last.length * tx, last.start.1 + last.length * ty);
        let earlier = frame.cartesian((s - frame.origin - END_EXTENSION).max(0.), 0.);
        let (dx, dy) = (end.0 - earlier.0, end.1 - earlier.1);
        // Limit how far the final model tangent influences returns past the forecast.
        // A short stopping path can otherwise steer the entire radar range.
        let mut tail = end;
        let mut along = 0.;
        if dx > 0. && dy.abs() > 1e-6 {
            let span = dx.hypot(dy);
            let tangent = (dx / span, dy / span);
            along = END_EXTENSION.min(1. / tangent.1.abs());
            tail = (end.0 + along * tangent.0, end.1 + along * tangent.1);
            frame.segments.push(Segment {
                start: end,
                tangent,
                length: along,
                s,
            });
        }
        frame.segments.push(Segment {
            start: tail,
            tangent: (1., 0.),
            length: 0.,
            s: s + along,
        });
        if let Some(model) = model.filter(|m| {
            !changing_lanes
                && m.lane_lines.len() >= 3
                && m.lane_line_probs.len() >= 3
                && m.lane_line_probs[1] >= 0.75
                && m.lane_line_probs[2] >= 0.75
        }) {
            let (left, right) = (&model.lane_lines[1], &model.lane_lines[2]);
            for x in [5., 10., 15., 20., 25., 30.] {
                let (Some(yl), Some(yr)) = (line_y(left, x), line_y(right, x)) else {
                    continue;
                };
                let l = to_ego((x, yl));
                let r = to_ego((x, yr));
                let (sl, dl) = frame.project(l.0, l.1);
                let (sr, dr) = frame.project(r.0, r.1);
                let width = dr - dl;
                let offset = (dr + dl) * 0.5;
                if (2.8..=4.5).contains(&width) && offset.abs() <= 0.55 && (sl - sr).abs() <= 2.8 {
                    frame.lanes.push(((sl + sr) / 2., offset, width));
                }
            }
        }
        frame
    }

    pub(super) fn project(&self, x: f64, y: f64) -> (f64, f64) {
        self.project_in(x, y, f64::NEG_INFINITY, f64::INFINITY)
    }

    pub(super) fn project_near(&self, x: f64, y: f64, s: f64, radius: f64) -> (f64, f64) {
        self.project_in(x, y, s - radius, s + radius)
    }

    fn project_in(&self, x: f64, y: f64, min_s: f64, max_s: f64) -> (f64, f64) {
        let (mut best, mut road) = (f64::INFINITY, (x, y));
        for (i, seg) in self.segments.iter().enumerate() {
            let (tx, ty) = seg.tangent;
            let (dx, dy) = (x - seg.start.0, y - seg.start.1);
            let start = seg.s - self.origin;
            let lo = if i == 0 { min_s - start } else { (min_s - start).max(0.) };
            let end = if i + 1 == self.segments.len() { f64::INFINITY } else { seg.length };
            let hi = (max_s - start).min(end);
            if lo > hi {
                continue;
            }
            let along = (dx * tx + dy * ty).clamp(lo, hi);
            let error = (dx - along * tx).powi(2) + (dy - along * ty).powi(2);
            if error < best {
                best = error;
                road = (seg.s + along - self.origin, dy * tx - dx * ty);
            }
        }
        road
    }

    fn segment(&self, s: f64) -> &Segment {
        let i = self.segments.partition_point(|p| p.s <= s + self.origin);
        &self.segments[i.saturating_sub(1)]
    }

    pub(super) fn cartesian(&self, s: f64, d: f64) -> Point {
        let seg = self.segment(s);
        let (tx, ty) = seg.tangent;
        let along = s + self.origin - seg.s;
        (
            seg.start.0 + along * tx - d * ty,
            seg.start.1 + along * ty + d * tx,
        )
    }

    pub(super) fn beyond_forecast(&self, s: f64) -> bool {
        s + self.origin > self.forward_end
    }

    pub(super) fn disagrees(&self, old: &Self, past: PoseAt, ego: PoseAt) -> bool {
        [12., 24., 36., 48.].into_iter().any(|s| {
            if old.beyond_forecast(s) {
                return false;
            }
            let (x, y) = old.cartesian(s, 0.);
            let (wx, wy) = past.to_world(x, -y);
            let (x, y) = ego.to_ego(wx, wy);
            let (new_s, d) = self.project(x, -y);
            // A distant forecast can drift more between model updates.
            !self.beyond_forecast(new_s) && d.abs() > 1.0 + 0.035 * s
        })
    }

    pub(super) fn advance(mut self, past: PoseAt, ego: PoseAt) -> Self {
        let (sin, cos) = (past.heading - ego.heading).sin_cos();
        for seg in &mut self.segments {
            let (wx, wy) = past.to_world(seg.start.0, -seg.start.1);
            let (ex, ey) = ego.to_ego(wx, wy);
            seg.start = (ex, -ey);
            let (tx, ty) = seg.tangent;
            seg.tangent = (cos * tx + sin * ty, cos * ty - sin * tx);
        }
        let shift = self.project(0., 0.).0;
        self.origin += shift;
        for (sample, _, _) in &mut self.lanes {
            *sample -= shift;
        }
        self
    }

    fn lane_geometry(&self, s: f64) -> (f64, f64) {
        // Blend out local corrections outside the measured range. Weak or inconsistent boundaries
        // cannot steer the centreline away from position.
        let mut weights = 0.;
        let (mut offset, mut width_delta) = (0., 0.);
        for &(sample, centre, width) in &self.lanes {
            let weight = (1. - (s - sample).abs() / 12.).max(0.);
            offset += weight * centre;
            width_delta += weight * (width - LANE_WIDTH);
            weights += weight;
        }
        let norm = weights.max(1.);
        (offset / norm, LANE_WIDTH + width_delta / norm)
    }

    pub(super) fn lane_u(&self, s: f64, d: f64) -> f64 {
        let (offset, width) = self.lane_geometry(s);
        (d - offset) / width
    }

    pub(super) fn lane_center(&self, s: f64, lane: f64) -> f64 {
        let (offset, width) = self.lane_geometry(s);
        offset + lane * width
    }
}

fn line_y(line: &Line, x: f64) -> Option<f64> {
    if line.x.len() != line.y.len() || line.x.len() < 2 {
        return None;
    }
    let i = line.x.partition_point(|v| f64::from(*v) < x);
    let (&hi, &right) = (line.x.get(i)?, line.y.get(i)?);
    let (&lo, &left) = (line.x.get(i.checked_sub(1)?)?, line.y.get(i - 1)?);
    if !(lo < hi && f64::from(lo) <= x && x <= f64::from(hi)) {
        return None;
    }
    Some(f64::from(left) + (x - f64::from(lo)) / f64::from(hi - lo) * f64::from(right - left))
}

pub(super) struct CornerLane<'a> {
    pub road: RoadObservation<'a>,
    pub model_pose: PoseAt,
    pub ego: PoseAt,
    /// Right-positive curvature from live ego rotation.
    pub curvature: f64,
}

impl CornerLane<'_> {
    fn in_model_frame(&self, (x, y): Point) -> Point {
        let (wx, wy) = self.ego.to_world(x, -y);
        let (x, y) = self.model_pose.to_ego(wx, wy);
        (x, -y)
    }

    fn boundary(&self, index: usize, x: f64, confidence: f32) -> Option<f64> {
        let model = self.road.lines?;
        if *model.lane_line_probs.get(index)? < confidence {
            return None;
        }
        line_y(model.lane_lines.get(index)?, x)
    }

    fn model_lane_u(&self, (x, y): Point) -> Option<f64> {
        let (x, y) = self.in_model_frame((x, y));
        let left = self.boundary(1, x, 0.75)?;
        let right = self.boundary(2, x, 0.75)?;
        let width = right - left;
        (2.8..=4.5).contains(&width).then(|| (y - (left + right) / 2.) / width)
    }

    pub fn same_lane_ahead(&self, lead: Point, far: Point, lead_u: f64, side_u: f64) -> bool {
        let (lx, ly) = lead;
        let (x, y) = far;
        if !(4. ..=32.).contains(&(x - lx)) || !(0.16..=0.85).contains(&(self.curvature * x).abs()) {
            return false;
        }
        // Compare the radar returns to the measured lane at their own ranges. A trusted
        // line pair also takes precedence when it places the farther return in a side lane.
        if let Some(u) = self.model_lane_u(far) {
            return u.abs() <= 0.34 && self.model_lane_u(lead).is_some_and(|u| u.abs() <= 0.40);
        }
        let (mx, my) = self.in_model_frame(far);
        let boundary = if side_u < 0. { 1 } else { 2 };
        if self.boundary(boundary, mx, 0.55).is_some_and(|edge| (my - edge) * side_u.signum() > 0.45) {
            return false;
        }

        // When the lines cannot constrain the far point, check the arc through the near
        // lead against the ego's turn. The circular residual is a distance in metres.
        let k = 2. * ly / (lx * lx + ly * ly);
        if !(10. ..=35.).contains(&lx) || lead_u.abs() > 0.30 || k * side_u <= 0.
            || (k - self.curvature).abs() > 0.55 * self.curvature.abs()
            || (k * x).abs() > 0.85
        {
            return false;
        }
        let residual = (2. * y - k * (x * x + y * y)) / (1. + (1. - k * y).hypot(k * x));
        residual.abs() <= 0.85
    }
}

/// A delayed detection beyond a trusted outer lane line can be passing oncoming traffic.
pub(super) fn inside_outer_line(
    road: RoadObservation<'_>,
    model_pose: PoseAt,
    ego: PoseAt,
    lane: f64,
    x: f64,
    y: f64,
) -> bool {
    let index = if lane < 0. { 0 } else { 3 };
    let Some(line) = road.lines.filter(|model| {
        !road.changing_lanes && model.lane_line_probs.get(index).is_some_and(|&p| p >= 0.75)
    }).and_then(|model| model.lane_lines.get(index)) else {
        return true;
    };
    let (wx, wy) = ego.to_world(x, -y);
    let (mx, my) = model_pose.to_ego(wx, wy);
    line_y(line, mx).is_none_or(|edge| lane.signum() * (-my - edge) <= 0.25)
}
