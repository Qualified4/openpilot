//! Display A in the world map, with lateral positive to the left.

use crate::pos::{common::LANE_WIDTH, display_a::backlash};
use crate::radar::lanes::interpolate;

pub(crate) struct ModelPath {
    points: Vec<(f64, f64)>,
    deviations: Vec<f64>,
    curvature: f64,
}

impl ModelPath {
    pub(crate) fn new(xs: Vec<f32>, ys: Vec<f32>) -> Option<Self> {
        if xs.len() != ys.len() {
            return None;
        }
        let points: Vec<_> = xs.into_iter().zip(ys).map(|(x, y)| (f64::from(x), f64::from(y))).collect();
        if points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite())
            || points.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return None;
        }
        let start = 5.;
        let end = points.last()?.0.min(35.);
        let span = (end - start) / 2.;
        let curvature = if span >= 5. {
            let a = interpolate(&points, start)?;
            let b = interpolate(&points, start + span)?;
            let c = interpolate(&points, end)?;
            let slope = (c - a) / (2. * span);
            (a - 2. * b + c) / (span * span * slope.hypot(1.).powi(3))
        } else {
            0.
        };
        (points.len() >= 2).then_some(Self { points, deviations: Vec::new(), curvature })
    }

    pub(crate) fn with_deviations(mut self, deviations: Vec<f32>) -> Self {
        if self.points.len() == deviations.len() {
            self.deviations = deviations.into_iter().map(f64::from).collect();
        }
        self
    }

    pub(crate) fn bend_weight(&self, yaw: f64) -> f64 {
        if yaw * self.curvature <= 0. {
            return 0.;
        }
        // A single model frame crossing the curvature threshold must not
        // apply the full projection in one radar tick.
        let path_bend = ((self.curvature.abs() - 0.002) / 0.001).clamp(0., 1.);
        let measured_bend = ((yaw.abs() - 0.020) / 0.020).clamp(0., 1.);
        path_bend * measured_bend
    }

    pub(crate) fn project_lateral(&self, x: f64, y: f64, width: f64) -> Option<(f64, f64)> {
        let (center, slope, confidence) = self.road_sample(x, width)?;
        Some(((y - center) / slope.hypot(1.), confidence))
    }

    pub(crate) fn road_sample(&self, x: f64, lane_width: f64) -> Option<(f64, f64, f64)> {
        let end = self.points.last()?.0;
        if !(lane_width > 0.) || !(self.points[0].0..=end).contains(&x) {
            return None;
        }
        let i = self.points.partition_point(|&(s, _)| s < x).max(1);
        let (a, ya) = self.points[i - 1];
        let (b, yb) = self.points[i];
        let f = (x - a) / (b - a);
        let confidence = if self.deviations.is_empty() {
            1.
        } else {
            let (p, q) = (self.deviations[i - 1], self.deviations[i]);
            if !p.is_finite() || !q.is_finite() || p < 0. || q < 0. {
                return None;
            }
            let deviation = p + f * (q - p);
            (2. - 2. * deviation / lane_width).clamp(0., 1.)
        };
        let horizon = ((end - x) / 8.).clamp(0., 1.);
        Some((ya + f * (yb - ya), (yb - ya) / (b - a), confidence * horizon))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadingTerm {
    #[default]
    RoadFrame,
    Keep,
    Removed,
    PoseIntegrated,
}

pub(crate) fn draw_road_frame(shown: &mut Option<f64>, x: f64, r: f64) -> f64 {
    // A lane of ego movement must have the same display scale for every vid.
    // In particular, a change in the fitted width at x is not vehicle motion.
    let y = -LANE_WIDTH * r;
    let drawn = shown.get_or_insert(y);
    backlash(drawn, y, x.abs());
    *drawn
}
