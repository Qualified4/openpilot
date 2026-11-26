//! Local Frenet frame in the model's right-positive convention.

use super::RoadObservation;
use crate::input::model_v2::{Line, Model};
use crate::world::PoseAt;

pub(crate) const LANE_WIDTH: f64 = 3.5;
type Point = (f64, f64);

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: Point,
    tangent: Point,
    length: f64,
    s: f64,
}

pub(super) struct RoadFrame {
    segments: Vec<Segment>,
    origin: f64,
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
            // A stopping forecast can fold back on itself. Extrapolate the
            // forward path before that fold so distant returns keep their range.
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
            lanes: Vec::new(),
        };
        frame.origin = frame.project(0., 0.).0;
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
        let (mut best, mut road) = (f64::INFINITY, (x, y));
        for (i, seg) in self.segments.iter().enumerate() {
            let (tx, ty) = seg.tangent;
            let (dx, dy) = (x - seg.start.0, y - seg.start.1);
            let along = (dx * tx + dy * ty).clamp(
                if i == 0 { f64::NEG_INFINITY } else { 0. },
                if i + 1 == self.segments.len() {
                    f64::INFINITY
                } else {
                    seg.length
                },
            );
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

    pub(super) fn lane_u(&self, s: f64, d: f64) -> f64 {
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
        (d - offset / norm) / (LANE_WIDTH + width_delta / norm)
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
