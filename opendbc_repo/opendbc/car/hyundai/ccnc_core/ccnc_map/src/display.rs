//! Display A in the world map, with lateral positive to the left.

use crate::pos::{common::LANE_WIDTH, display_a::backlash};
use crate::radar::lanes::interpolate;

const BEND_MIN_CURVATURE: f64 = 0.002;
const BEND_START_YAW: f64 = 0.020;
const BEND_FULL_YAW: f64 = 0.040;

pub(crate) struct ModelPath {
    points: Vec<(f64, f64)>,
    y0: f64,
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
        let y0 = interpolate(&points, 0.)?;
        let start = 5.;
        let end = points.last()?.0.min(35.);
        let half_span = (end - start) / 2.;
        if half_span < 5. {
            return None;
        }
        let a = interpolate(&points, start)?;
        let b = interpolate(&points, start + half_span)?;
        let c = interpolate(&points, end)?;
        let slope = (c - a) / (2. * half_span);
        let curvature = (a - 2. * b + c) / (half_span * half_span * (1. + slope * slope).powf(1.5));
        Some(Self { points, y0, curvature })
    }

    pub(crate) fn bend(&self, yaw: f64) -> Option<(&Self, f64)> {
        if yaw * self.curvature <= 0. || self.curvature.abs() <= BEND_MIN_CURVATURE || yaw.abs() <= BEND_START_YAW {
            return None;
        }
        let weight = ((yaw.abs() - BEND_START_YAW) / (BEND_FULL_YAW - BEND_START_YAW)).min(1.);
        Some((self, weight))
    }

    pub(crate) fn offset_at(&self, x: f64) -> Option<f64> {
        interpolate(&self.points, x).map(|y| y - self.y0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bend_uses_matching_yaw_and_stays_inside_the_model_path() {
        let xs = vec![0., 5., 20., 35.];
        let ys = xs.iter().map(|x| 1. + 0.005 * x * x).collect();
        let path = ModelPath::new(xs, ys).unwrap();
        assert_eq!(path.offset_at(0.), Some(0.));
        assert_eq!(path.offset_at(20.), Some(2.));
        assert_eq!(path.offset_at(40.), None);
        assert!(path.bend(-0.04).is_none());
        assert!(path.bend(0.02).is_none());
        assert!((path.bend(0.03).unwrap().1 - 0.5).abs() < 1e-9);
        assert_eq!(path.bend(0.04).unwrap().1, 1.);
    }
}
