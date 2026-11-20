//! Lane positions used by both the lane message and the rear vehicle offsets.

use std::collections::VecDeque;

use pyo3::exceptions::PyIndexError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat};

use crate::display_filter::{self, fixed, value};
use crate::model_v2::{numpy_interp, valid_curve, GeometryInput};
use crate::pymath::{py_max, py_min};

#[pyclass]
struct LanePositions {
    last: Option<f64>,
    reader: Option<Py<PyAny>>,
    stamp: Option<u64>,
    direction: Option<bool>,
    motion: VecDeque<(f64, f64, f64)>,
    hold_start: Option<f64>,
    hold_speed: f64,
    phase_min: f64,
    escaped: usize,
    draw_center: bool,
    hold: bool,
    alpha: f64,
    width: f64,
}

impl LanePositions {
    fn new() -> Self {
        Self {
            last: None,
            reader: None,
            stamp: None,
            direction: None,
            motion: VecDeque::new(),
            hold_start: None,
            hold_speed: 0.,
            phase_min: 10.,
            escaped: 0,
            draw_center: false,
            hold: false,
            alpha: 0.2,
            width: 3.,
        }
    }

    fn clear_motion(&mut self) {
        self.motion.clear();
        self.direction = None;
        self.hold_start = None;
        self.hold_speed = 0.;
    }

    fn clear_phase(&mut self) {
        self.phase_min = 10.;
        self.escaped = 0;
        self.draw_center = false;
        self.hold = false;
    }

    fn start_hold(&mut self, now: f64) {
        self.hold_start = Some(now);
        self.hold_speed = 0.;
        let m = &self.motion;
        if m.len() < 4
            || now - m.back().unwrap().0 > 0.15
            || m.back().unwrap().0 - m[0].0 < 0.15
            || m.iter()
                .zip(m.iter().skip(1))
                .any(|(a, b)| !(0. < b.0 - a.0 && b.0 - a.0 <= 0.15))
        {
            return;
        }
        let mut speed = [0.; 2];
        for (i, speed) in speed.iter_mut().enumerate() {
            let mut values: Vec<_> = m.iter().map(|p| if i == 0 { p.1 } else { p.2 }).collect();
            if !values.iter().all(|v| v.is_finite()) {
                return;
            }
            values.sort_by(f64::total_cmp);
            if values[values.len() - 1] - values[0] > 0.6 {
                return;
            }
            let middle = values.len() / 2;
            *speed = if values.len() % 2 == 0 {
                (values[middle - 1] + values[middle]) / 2.
            } else {
                values[middle]
            };
        }
        if speed.into_iter().all(|v| (0.1..=1.5).contains(&v)) && (speed[0] - speed[1]).abs() <= 0.4
        {
            self.hold_speed = py_min(speed[0], speed[1]);
        }
    }

    fn observe(&mut self, lines: &GeometryInput, left: bool, fresh: bool, now: f64, speed: f64) {
        if self.direction != Some(left) {
            self.clear_motion();
        }
        self.direction = Some(left);
        if self.draw_center || !fresh {
            return;
        }
        let left_line = &lines.lines[1];
        let right_line = &lines.lines[2];
        if !(2.3..=4.8).contains(&(right_line.1[0] - left_line.1[0])) {
            self.motion.clear();
            return;
        }
        if ![left_line, right_line]
            .into_iter()
            .all(|curve| valid_curve(curve) && curve.0[0] <= 0. && curve.0[curve.0.len() - 1] >= 5.)
        {
            self.motion.clear();
            return;
        }
        let sign = if left { 1. } else { -1. };
        let [l, r] = [left_line, right_line].map(|curve| {
            let [y0, y1, y2] = [0., 2.5, 5.].map(|x| numpy_interp(curve, x));
            sign * speed / 3.6 * (-3. * y0 + 4. * y1 - y2) / 5.
        });
        if self.motion.len() == 8 {
            self.motion.pop_front();
        }
        self.motion.push_back((now, l, r));
    }

    #[allow(clippy::too_many_arguments)]
    fn draw<'py>(
        &mut self,
        display: &Bound<'py, PyDict>,
        state: &Bound<'py, PyAny>,
        cs: &Bound<'py, PyAny>,
        md: &Bound<'py, PyAny>,
        lines: &GeometryInput,
        now: f64,
        drive_speed: f64,
        desire: i64,
        lat_enabled: bool,
    ) -> PyResult<()> {
        let py = state.py();
        let left = get!(state, "l_lane_f");
        let right = get!(state, "r_lane_f");
        let filters = [&left, &right];
        let elapsed = self.last.map(|last| now - last);
        let clock_reset = elapsed.is_none_or(|dt| !(0. ..=0.15).contains(&dt));
        if clock_reset {
            self.clear_motion();
            self.clear_phase();
            self.alpha = 0.2;
            self.stamp = None;
            self.reader = None;
            if elapsed.is_some() {
                for filter in filters {
                    put!(filter, "_alpha" => get!(filter, "_a_min"));
                }
            }
        }
        let dt = elapsed
            .filter(|&dt| dt >= 0.)
            .map_or(0.05, |dt| py_min(0.15, py_max(0., dt)));
        self.last = Some(now);
        let stamp: Option<u64> = md
            .getattr_opt(intern!(py, "timestampEof"))?
            .map_or(Ok(None), |v| v.extract())?;
        let fresh = !md.is_none()
            && if stamp.is_some_and(|v| v > 0) {
                stamp != self.stamp
            } else {
                self.reader.as_ref().is_none_or(|old| !old.bind(py).is(md))
            };
        self.stamp = stamp;
        self.reader = if md.is_none() {
            None
        } else {
            Some(md.clone().unbind())
        };

        if lines.probs.len() < 3 {
            return Err(PyIndexError::new_err("missing lane probability"));
        }
        let out = get!(cs, "out");
        let left_blinker = get!(out, "leftBlinker").is_truthy()?;
        let right_blinker = get!(out, "rightBlinker").is_truthy()?;
        let auto = matches!(desire, 3 | 4);
        let blinking = left_blinker != right_blinker;
        let changing = auto || (blinking && drive_speed > 20.);
        let target_alpha = if changing { 0.6 } else { 0.2 };
        let alpha = self.alpha + py_min(py_max(target_alpha - self.alpha, -2. * dt), 2. * dt);
        if alpha != self.alpha {
            self.alpha = alpha;
            for filter in filters {
                put!(filter, "_alpha" => py_min(py_max(alpha, 0.001), 1.));
            }
        }

        let current = |i: usize| {
            lines
                .lines
                .get(i)
                .and_then(|c| c.1.first())
                .map(|v| v.abs())
                .ok_or_else(|| PyIndexError::new_err("missing lane position"))
        };
        let mut targets = [current(1)?, current(2)?];
        let l_valid = lines.probs[1] > 0.3 || auto || blinking;
        let r_valid = lines.probs[2] > 0.3 || auto || blinking;
        match (l_valid, r_valid) {
            (false, false) => targets = [1.5; 2],
            (false, true) => targets[0] = self.width - targets[1],
            (true, false) => targets[1] = self.width - targets[0],
            (true, true) => {}
        }

        let apply = |targets: [f64; 2]| -> PyResult<[f64; 2]> {
            Ok([
                fixed(filters[0], PyFloat::new(py, targets[0]).as_any())?,
                fixed(filters[1], PyFloat::new(py, targets[1]).as_any())?,
            ])
        };
        let positions = if changing {
            let moving_left = if auto { desire == 3 } else { left_blinker };
            let side = usize::from(!moving_left);
            if self.direction.is_some_and(|other| other != moving_left) {
                self.clear_motion();
                self.clear_phase();
            }
            if !self.draw_center {
                let raw = targets[side];
                let shifted = raw < 0.1 || (self.phase_min < 0.5 && raw - self.phase_min > 0.3);
                self.phase_min = py_min(self.phase_min, raw);
                if shifted {
                    self.start_hold(now);
                    self.draw_center = true;
                    self.hold = true;
                    self.phase_min = 6.;
                    self.escaped = 0;
                    display_filter::reset(filters[1 - side], value(filters[side])?)?;
                }
            }
            let positions = if self.hold {
                let swapped = targets[1 - side];
                self.phase_min = py_min(self.phase_min, swapped);
                if swapped - self.phase_min > 0.1 {
                    self.escaped += usize::from(fresh);
                    if self.escaped >= 2 {
                        self.hold = false;
                    }
                } else {
                    self.escaped = 0;
                }
                let progress = self.hold_start.map_or(0., |start| {
                    let t = py_min(0.6, py_max(0., now - start));
                    py_min(0.45, self.hold_speed * (t - t * t / 1.2))
                });
                let hold = py_max(progress, self.escaped as f64 * 0.1);
                let targets = if moving_left {
                    [self.width - hold, hold]
                } else {
                    [hold, self.width - hold]
                };
                for i in 0..2 {
                    display_filter::reset(filters[i], targets[i])?;
                }
                targets
            } else if self.draw_center {
                let step = 3. * dt;
                for i in 0..2 {
                    let previous = value(filters[i])?;
                    targets[i] = previous + py_min(py_max(targets[i] - previous, -step), step);
                }
                apply(targets)?
            } else {
                apply(targets)?
            };
            if auto {
                if self.draw_center {
                    display.set_item(intern!(py, "LANE_HIGHLIGHT"), 1)?;
                    display.set_item(intern!(py, "LANE_HIGHLIGHT_DISTANCE"), 60)?;
                } else {
                    display.set_item(
                        if desire == 3 {
                            intern!(py, "LANE_LEFT")
                        } else {
                            intern!(py, "LANE_RIGHT")
                        },
                        1,
                    )?;
                }
            } else if (positions[0] - positions[1]).abs() < self.width / 5. {
                self.clear_phase();
            }
            self.observe(lines, moving_left, fresh, now, drive_speed);
            positions
        } else {
            self.clear_motion();
            self.clear_phase();
            let positions = apply(targets)?;
            let width = positions[0] + positions[1];
            if 2. < width && width < 4.6 {
                self.width = width;
            }
            positions
        };
        let encode = |x| (py_min(py_max(x, 0.), 3.) * 10.).round_ties_even();
        let l = encode(positions[0]);
        let r = encode(positions[1]);
        if !l.is_finite() || !r.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nonfinite lane position",
            ));
        }
        display.set_item(intern!(py, "LANELINE_LEFT_POSITION"), l as i64)?;
        display.set_item(intern!(py, "LANELINE_RIGHT_POSITION"), r as i64)?;

        if lat_enabled {
            let meta = get!(md, "meta");
            let lca =
                |warning: bool, other: bool, available: &Bound<'py, PyAny>| -> PyResult<i32> {
                    Ok(if warning {
                        1
                    } else if other || !available.is_truthy()? {
                        4
                    } else {
                        2
                    })
                };
            let left = lca(
                get!(out, "leftBlindspot").is_truthy()?,
                right_blinker,
                &get!(meta, "laneChangeAvailableLeft"),
            )?;
            display.set_item(intern!(py, "LCA_LEFT_ICON"), left)?;
            let right = lca(
                get!(out, "rightBlindspot").is_truthy()?,
                left_blinker,
                &get!(meta, "laneChangeAvailableRight"),
            )?;
            display.set_item(intern!(py, "LCA_RIGHT_ICON"), right)?;
        }
        Ok(())
    }
}

pub(super) type LanePanel<'py> = Option<(Bound<'py, PyDict>, i64, bool)>;

pub(super) fn draw<'py>(
    state: &Bound<'py, PyAny>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    model_lanes: bool,
    lines: &GeometryInput,
    panel: LanePanel<'py>,
) -> PyResult<()> {
    let Some((display, desire, lat_enabled)) = panel.filter(|_| model_lanes) else {
        return Ok(());
    };
    let py = state.py();
    let geometry = get!(state, "lane_geometry");
    let name = intern!(py, "_rust_positions");
    let lane = if let Some(previous) = geometry.getattr_opt(name)? {
        previous.cast_into::<LanePositions>()?
    } else {
        let lane = Bound::new(py, LanePositions::new())?;
        geometry.setattr(name, &lane)?;
        lane
    };
    let speed: f64 = get!(get!(cs, "out"), "vEgo").extract()?;
    let result = lane.try_borrow_mut()?.draw(
        &display,
        state,
        cs,
        md,
        lines,
        frame as f64 * 0.01,
        speed * 3.6,
        desire,
        lat_enabled,
    );
    if result.is_err() {
        let absent = lines.probs.len() < 3
            || lines.lines.len() < 3
            || lines.lines[1].1.is_empty()
            || lines.lines[2].1.is_empty();
        if !absent {
            for (key, value) in [
                ("LANELINE_LEFT_POSITION", 30),
                ("LANELINE_RIGHT_POSITION", 30),
                ("LANE_HIGHLIGHT", 3),
                ("LANE_HIGHLIGHT_DISTANCE", 60),
                ("LANE_LEFT", 1),
                ("LANE_RIGHT", 1),
                ("LKA_ICON", 1),
            ] {
                display.set_item(key, value)?;
            }
        }
    }
    Ok(())
}
