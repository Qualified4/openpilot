use pyo3::intern;
use pyo3::prelude::*;

use crate::model_v2::{numpy_interp, read_path, valid_curve, Curve, GeometryInput};
use crate::pymath::{py_max, py_min};

fn lookahead(speed: f64) -> f64 {
    30. + py_min(py_max((speed - 20.) / 80., 0.), 1.) * 50.
}

fn fit_samples(y: [f64; 5], span: f64, residual: bool) -> (f64, f64) {
    let half = span * 0.5;
    let quadratic = (4. * y[0] - 2. * y[1] - 4. * y[2] - 2. * y[3] + 4. * y[4]) / 7.;
    let slope = (-y[0] - 0.5 * y[1] + 0.5 * y[3] + y[4]) / (2.5 * half);
    let curvature = -3600. * quadratic / (half * half * (1. + slope * slope).powf(1.5));
    let max_error = if residual {
        let mean = y.into_iter().sum::<f64>() / 5.;
        [-1., -0.5, 0., 0.5, 1.]
            .into_iter()
            .zip(y)
            .map(|(t, y)| {
                let fitted = mean - quadratic * 0.5 + slope * half * t + quadratic * t * t;
                (y - fitted).abs()
            })
            .fold(0., f64::max)
    } else {
        0.
    };
    (curvature, max_error)
}

fn fit_curve(path: &Curve, start: f64, end: f64, residual: bool) -> (f64, f64) {
    let step = (end - start) * 0.25;
    let y = [
        start,
        start + step,
        start + 2. * step,
        start + 3. * step,
        end,
    ]
    .map(|x| numpy_interp(path, x));
    fit_samples(y, end - start, residual)
}

fn clip_curve(curvature: f64) -> f64 {
    py_min(py_max(curvature, -15.), 15.)
}

fn road_curve(
    mut path: Curve,
    stds: &[f64],
    lines: &GeometryInput,
    speed: f64,
    changing: bool,
) -> Option<f64> {
    if path.0.len() != path.1.len()
        || path.0.len() != stds.len()
        || !stds.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let trusted = stds.iter().position(|&std| std > 0.8).unwrap_or(stds.len());
    path.0.truncate(trusted);
    path.1.truncate(trusted);
    if !valid_curve(&path) {
        return None;
    }
    let available = path.0[trusted - 1];
    let end = py_min(
        lookahead(speed) + if changing { speed / 3.6 * 2. } else { 0. },
        available,
    );
    let start = if changing {
        py_max(20., end * 0.5)
    } else {
        py_max(0., path.0[0])
    };
    if end - start < 20. {
        return None;
    }
    let target = clip_curve(fit_curve(&path, start, end, false).0);
    if !changing
        || path.0[0] > 0.
        || fit_curve(&path, 0., end, true).1 <= 0.02
        || lines.lines.len() < 3
    {
        return Some(target);
    }
    let inner = [&lines.lines[1], &lines.lines[2]];
    if !inner
        .into_iter()
        .all(|line| valid_curve(line) && line.0[0] <= 0. && line.0[line.0.len() - 1] >= 30.)
    {
        return Some(target);
    }
    let [left, right] = inner.map(|line| [0., 7.5, 15., 22.5, 30.].map(|x| numpy_interp(line, x)));
    let left_curve = fit_samples(left, 30., false).0;
    let right_curve = fit_samples(right, 30., false).0;
    if !right
        .into_iter()
        .zip(left)
        .all(|(r, l)| (2.3..=4.8).contains(&(r - l)))
        || (right_curve - left_curve).abs() > 3.
    {
        return Some(target);
    }
    let local = (left_curve + right_curve) * 0.5;
    Some(
        if (fit_curve(&path, 0., 30., false).0 - local).abs() > 0.5 {
            clip_curve(local)
        } else {
            target
        },
    )
}

pub(super) struct Curvature {
    value: f64,
    valid_time: f64,
    target: Option<f64>,
}

impl Default for Curvature {
    fn default() -> Self {
        Self {
            value: 0.,
            valid_time: f64::NEG_INFINITY,
            target: None,
        }
    }
}

impl Curvature {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        out: &Bound<'_, PyAny>,
        md: &Bound<'_, PyAny>,
        raw: Option<&[u8]>,
        lines: &GeometryInput,
        speed: f64,
        now: f64,
        dt: f64,
        fresh: bool,
        absent: bool,
        changing: bool,
    ) -> PyResult<f64> {
        if fresh {
            self.target = read_path(md, raw)
                .ok()
                .and_then(|(path, stds)| road_curve(path, &stds, lines, speed, changing));
            if self.target.is_some() {
                self.valid_time = now;
            }
        }
        let expired = now - self.valid_time > 0.35;
        if md.is_none() || expired {
            self.target = None;
        }
        let target = self.target.unwrap_or(if expired { 0. } else { self.value });
        let step = (target - self.value) * -(-dt / 0.25).exp_m1();
        self.value += py_min(py_max(step, -10. * dt), 10. * dt);
        if absent || (self.target.is_none() && expired) {
            return Ok((get!(out, "steeringAngleDeg").extract::<f64>()? / 3.).round_ties_even());
        }
        Ok(self.value.round_ties_even())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_curve_ignores_heading_and_untrusted_tail() {
        let x = vec![0., 10., 20., 30., 40., 80., 1.];
        let y = x.iter().map(|x| x * 0.1).collect();
        let lines = GeometryInput {
            lines: vec![],
            probs: vec![],
            edges: vec![],
        };
        let path = (x, y);
        let stds = [0., 0., 0., 0., 0., 0., 0.9];
        assert!(road_curve(path, &stds, &lines, 60., false).unwrap().abs() < 1e-12);
    }

    #[test]
    fn invalid_paths_have_no_precise_target() {
        let lines = GeometryInput::default();
        for (xs, ys, stds) in [
            (vec![], vec![], vec![]),
            (vec![0., 30.], vec![], vec![0., 0.]),
            (vec![0., 30.], vec![0., f64::NAN], vec![0., 0.]),
            (vec![0., 30.], vec![0., 1.], vec![0., f64::NAN]),
        ] {
            assert_eq!(road_curve((xs, ys), &stds, &lines, 60., true), None);
        }
    }
}
