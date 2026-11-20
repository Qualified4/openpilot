//! The continuous front/rear drawing for one side. Keep dict identities across
//! an origin crossing so the drawing still refers to the same rear memory.

use super::{field, flag, record, reset_filter, DisplayFields, Record};
use crate::display_filter::empty_object;
use crate::pymath::{py_max, py_min};
use pyo3::exceptions::{PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Coast,
    Exit,
    JoinFront,
    ToFront,
    ToRear,
    Virtual,
    Other,
}

fn phase(object: &Record<'_>) -> PyResult<Phase> {
    let name = get!(object, "phase");
    let Ok(name) = name.cast::<PyString>() else {
        return Ok(Phase::Other);
    };
    Ok(match name.to_str()? {
        "coast" => Phase::Coast,
        "exit" => Phase::Exit,
        "join_front" => Phase::JoinFront,
        "to_front" => Phase::ToFront,
        "to_rear" => Phase::ToRear,
        "virtual" => Phase::Virtual,
        _ => Phase::Other,
    })
}

fn owns(motion: &Record<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(key.eq(get!(motion, "key"))?
        || optional!(motion, "aliases").map_or(Ok(false), |aliases| aliases.contains(key))?)
}

fn front<'py>(candidate: &Record<'py>) -> PyResult<Record<'py>> {
    let result = candidate.copy()?;
    put!(result, "phase" => "front");
    Ok(result)
}

fn floor(x: f64) -> PyResult<f64> {
    if x.is_finite() {
        Ok(x.floor())
    } else if x.is_nan() {
        Err(PyValueError::new_err("cannot convert float NaN to integer"))
    } else {
        Err(PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ))
    }
}

fn gap(front: &Record<'_>, rear: &Record<'_>) -> PyResult<f64> {
    Ok(floor(num!(front, "x") * 10.)? + floor(-num!(rear, "x") * 10.)?)
}

// None means Python returns the target object itself (which may be an int).
fn moved(value: f64, target: f64, dt: f64, speed: f64, tau: f64) -> PyResult<Option<f64>> {
    let error = target - value;
    let limit = speed * dt;
    if error.abs() <= py_min(0.04, limit) {
        return Ok(None);
    }
    if tau == 0. {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    let exponent = -dt / tau;
    let change = exponent.exp_m1();
    if change.is_infinite() && exponent.is_finite() {
        return Err(PyOverflowError::new_err("math range error"));
    }
    let step = py_min(error.abs() * (-change), limit);
    Ok(Some(value + step.copysign(error)))
}

fn move_key(
    obj: &Record<'_>,
    key: &Bound<'_, PyString>,
    target: &Bound<'_, PyAny>,
    dt: f64,
    speed: f64,
    tau: f64,
) -> PyResult<()> {
    let value = field(obj, key)?.extract()?;
    let output = moved(value, target.extract()?, dt, speed, tau)?;
    if let Some(x) = output {
        obj.set_item(key, x)
    } else {
        obj.set_item(key, target)
    }
}

struct Flow<'a, 'py> {
    state: &'a Bound<'py, PyAny>,
    events: Bound<'py, PyList>,
    now: f64,
    dt: f64,
}

impl<'py> Flow<'_, 'py> {
    fn lateral(&self, obj: &Record<'py>, target: &Bound<'py, PyAny>, dt: f64) -> PyResult<()> {
        move_key(
            obj,
            intern!(obj.py(), "y"),
            target,
            dt,
            attr!(self.state, "LATERAL_SPEED").extract()?,
            attr!(self.state, "LATERAL_TAU").extract()?,
        )
    }

    fn advance(
        &self,
        obj: &Record<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        dt: f64,
        speed: f64,
        tau: f64,
    ) -> PyResult<()> {
        move_key(obj, intern!(obj.py(), "x"), x, dt, speed, tau)?;
        self.lateral(obj, y, dt)
    }

    fn remove_moved(
        &self,
        drawing: &Option<Record<'py>>,
        name: &Bound<'py, PyString>,
        elsewhere: &Bound<'py, PyAny>,
        motion: Option<&Record<'py>>,
    ) -> PyResult<bool> {
        let Some(obj) = drawing else { return Ok(false) };
        let mut owner = match optional!(obj, "motion") {
            Some(value) => record(value)?,
            None => None,
        };
        if owner.is_none() {
            if let Some(motion) = motion {
                if owns(motion, &get!(obj, "key"))? {
                    owner = Some(motion.clone());
                }
            }
        }
        let key = get!(obj, "key");
        let elsewhere_owner = |owner: Record<'py>| -> PyResult<bool> {
            if elsewhere.contains(get!(owner, "key"))? {
                return Ok(true);
            }
            if let Some(aliases) = optional!(owner, "aliases") {
                for alias in aliases.try_iter()? {
                    if elsewhere.contains(alias?)? {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        };
        if (!key.is_none() && elsewhere.contains(&key)?)
            || owner.map_or(Ok(false), elsewhere_owner)?
        {
            self.state.setattr(name, self.state.py().None())?;
            return Ok(true);
        }
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_rear(
        &self,
        r: &Record<'py>,
        motion: Option<&Record<'py>>,
        rear_lateral: &Bound<'py, PyAny>,
        eligible_bsd: bool,
        bsd: bool,
        lane_change: bool,
        spawned: bool,
    ) -> PyResult<bool> {
        let py = r.py();
        let mut physical = match optional!(r, "motion") {
            Some(value) => record(value)?,
            None => None,
        };
        let mut delta = 0.;
        if let Some(tracked) = &physical {
            if motion.is_some_and(|m| tracked.is(m)) {
                let distance = get!(tracked, "distance");
                delta = py_max(0., distance.extract::<f64>()? - num!(r, "model_distance"));
                put!(r, "model_distance" => distance);
            } else if eligible_bsd {
                put!(r, "motion" => py.None());
                physical = None;
            } else {
                let mut retire = false;
                if let Some(reason) = optional!(tracked, "end_reason") {
                    for end in ["distance_end", "timeout", "rear_replaced", "lane_change"] {
                        if reason.eq(end)? {
                            retire = true;
                            break;
                        }
                    }
                }
                if retire {
                    put!(r, "phase" => "exit");
                } else {
                    self.state.setattr(intern!(py, "rear"), py.None())?;
                    self.events.append("rear_cancelled")?;
                    return Ok(false);
                }
            }
        }

        let mut r_phase = phase(r)?;
        if lane_change && !eligible_bsd && physical.is_none() && r_phase == Phase::Virtual {
            put!(r, "phase" => "exit");
            r_phase = Phase::Exit;
        }
        let passing = physical.is_none() && r_phase == Phase::Virtual;
        let end: f64 = attr!(self.state, "END").extract()?;
        if let Some(physical) = physical.filter(|_| !eligible_bsd) {
            if r_phase == Phase::Exit {
                delta = num!(physical, "max_speed") * self.dt;
            }
            let x = num!(r, "x");
            if self.now >= num!(physical, "force_at") {
                let divisor = py_max(self.dt, num!(physical, "deadline") - self.now + self.dt);
                if divisor == 0. {
                    return Err(PyZeroDivisionError::new_err("float division by zero"));
                }
                let required = (end + x) * self.dt / divisor;
                delta = py_max(
                    delta,
                    py_min(required, num!(physical, "max_speed") * self.dt),
                );
            }
            put!(r, "x" => py_max(-end, x - delta));
            self.lateral(r, rear_lateral, self.dt)?;
        } else {
            let target = if bsd || (passing && eligible_bsd) {
                (-8.0_f64).into_pyobject(py)?.into_any()
            } else if eligible_bsd {
                get!(r, "x")
            } else {
                (-end).into_pyobject(py)?.into_any()
            };
            let (mut speed, tau) = if bsd {
                (6., 0.4)
            } else {
                (
                    attr!(self.state, "EXIT_SPEED").extract()?,
                    attr!(self.state, "EXIT_TAU").extract()?,
                )
            };
            if passing {
                speed = attr!(self.state, "PASS_SPEED").extract()?;
            }
            self.advance(
                r,
                &target,
                rear_lateral,
                if spawned { 0. } else { self.dt },
                speed,
                tau,
            )?;
        }
        if num!(r, "x") == -end && !eligible_bsd {
            self.state.setattr(intern!(py, "rear"), py.None())?;
            self.events.append("rear_exit")?;
            return Ok(false);
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        now_value: &Bound<'py, PyAny>,
        candidate: Option<&Record<'py>>,
        bsd: bool,
        rear_lateral: &Bound<'py, PyAny>,
        elsewhere: &Bound<'py, PyAny>,
        motion: Option<&Record<'py>>,
        lane_change: bool,
    ) -> PyResult<(Option<Record<'py>>, Option<Record<'py>>)> {
        let py = self.state.py();
        let front_name = intern!(py, "front");
        let rear_name = intern!(py, "rear");
        let bsd_off = intern!(py, "bsd_off");
        if attr!(self.state, "bsd").is_truthy()? && !bsd {
            self.state.setattr(bsd_off, now_value)?;
        } else if bsd {
            self.state.setattr(bsd_off, py.None())?;
        }
        self.state.setattr(intern!(py, "bsd"), bsd)?;
        if lane_change && !bsd {
            self.state.setattr(bsd_off, py.None())?;
        }
        let mut f = record(self.state.getattr(front_name)?)?;
        let mut r = record(self.state.getattr(rear_name)?)?;
        if f.is_none() && r.is_none() && candidate.is_none() && !bsd {
            return Ok((None, None));
        }
        let eligible_bsd = bsd || {
            let off = self.state.getattr(bsd_off)?;
            !off.is_none()
                && self.now - off.extract::<f64>()? < attr!(self.state, "HOLD").extract::<f64>()?
        };

        if self.remove_moved(&f, front_name, elsewhere, motion)? {
            f = None;
        }
        if self.remove_moved(&r, rear_name, elsewhere, motion)? {
            r = None;
        }
        let previous_rear = r.clone();
        let rear_start = r
            .as_ref()
            .map(|obj| field(obj, intern!(py, "x")))
            .transpose()?;
        let f_owned = match (&f, motion) {
            (Some(f), Some(motion)) => owns(motion, &get!(f, "key"))?,
            _ => false,
        };

        if let Some(current) = &f {
            let same = if let Some(candidate) = candidate {
                get!(candidate, "key").eq(get!(current, "key"))?
            } else {
                false
            };
            let to_rear = phase(current)? == Phase::ToRear;
            if to_rear && same {
                put!(current, "phase" => "join_front");
            } else if to_rear
                && !f_owned
                && !eligible_bsd
                && !flag(current, intern!(py, "display_pass"))?
            {
                self.state.setattr(front_name, py.None())?;
                f = None;
            } else if !to_rear && !same {
                if num!(current, "x") <= 5. {
                    put!(current, "phase" => "to_rear", "display_pass" => !f_owned);
                } else if match candidate {
                    None => true,
                    Some(candidate) => {
                        num!(candidate, "x") < attr!(self.state, "GAP").extract::<f64>()?
                            || if let Some(r) = &r {
                                get!(candidate, "key").eq(get!(r, "key"))?
                            } else {
                                false
                            }
                    }
                } {
                    self.state.setattr(front_name, py.None())?;
                    f = None;
                } else if let Some(candidate) = candidate {
                    let next = front(candidate)?;
                    self.state.setattr(front_name, &next)?;
                    f = Some(next);
                }
            }
        }

        let spawned = r.is_none() && bsd;
        if spawned {
            let obj = PyDict::new(py);
            put!(obj, "key" => py.None(), "x" => -attr!(self.state, "END").extract::<f64>()?, "y" => rear_lateral,
                "detect" => 2, "phase" => "bsd");
            self.state.setattr(rear_name, &obj)?;
            r = Some(obj);
        }

        if let Some(current) = &r {
            if let Some(candidate) = candidate.filter(|_| f.is_none()) {
                if phase(current)? != Phase::ToFront
                    && (get!(candidate, "key").eq(get!(current, "key"))?
                        || num!(candidate, "x") <= attr!(self.state, "GAP").extract::<f64>()?
                            && eligible_bsd
                            && !attr!(self.state, "after_pass").is_truthy()?)
                {
                    put!(current, "phase" => "to_front", "key" => get!(candidate, "key"), "target" => candidate.copy()?);
                    self.events.append("rear_to_front_start")?;
                }
            }

            if phase(current)? == Phase::ToFront {
                if let Some(candidate) = candidate {
                    if get!(candidate, "key").eq(get!(current, "key"))? {
                        put!(current, "target" => candidate.copy()?);
                    }
                }
                let target = get!(current, "target")
                    .cast_into::<PyDict>()
                    .map_err(PyErr::from)?;
                let x = get!(target, "x");
                let speed: f64 = attr!(self.state, "FAST_SPEED").extract()?;
                let tau: f64 = attr!(self.state, "FAST_TAU").extract()?;
                let min_x = speed * tau;
                let x = if min_x > x.extract::<f64>()? {
                    min_x.into_pyobject(py)?.into_any()
                } else {
                    x
                };
                self.advance(current, &x, &get!(current, "y"), self.dt, speed, tau)?;
                let x = py_min(0., num!(current, "x"));
                // min(0., current['x']) keeps the original number when it wins.
                if x == 0. {
                    put!(current, "x" => x);
                    self.state.setattr(front_name, current)?;
                    put!(current, "phase" => "join_front");
                    self.state.setattr(rear_name, py.None())?;
                    self.state.setattr(intern!(py, "after_pass"), false)?;
                    self.events.append("rear_to_front")?;
                    return Ok((r, None));
                }
            } else if !self.update_rear(
                current,
                motion,
                rear_lateral,
                eligible_bsd,
                bsd,
                lane_change,
                spawned,
            )? {
                r = None;
            }
        }

        let mut crossed = false;
        if let Some(current) = &f {
            if phase(current)? == Phase::ToRear {
                let physical = motion.filter(|_| f_owned);
                let (target, speed) = if let Some(motion) = physical {
                    if phase(motion)? == Phase::Coast {
                        (py_max(0., num!(motion, "signed")), 8.)
                    } else {
                        (0., num!(motion, "speed"))
                    }
                } else {
                    (0., attr!(self.state, "PASS_SPEED").extract()?)
                };
                let x = get!(current, "x");
                let position: f64 = x.extract()?;
                let limit = position - py_min(8., speed) * self.dt;
                let target_or_position = py_min(position, target);
                let next = py_max(target_or_position, limit);
                // Two equal arguments keep the first value, including its signed zero and type.
                if target < position || limit > target_or_position {
                    put!(current, "x" => next);
                }
                if target == 0. && next < 1e-9 {
                    put!(current, "x" => 0.);
                }
                if num!(current, "x") == 0. {
                    if r.is_some() {
                        self.events.append("rear_display_replaced")?;
                    }
                    self.state.setattr(rear_name, current)?;
                    let (motion, distance) = if let Some(physical) = physical {
                        (physical.clone().into_any(), get!(physical, "distance"))
                    } else {
                        (
                            py.None().into_bound(py),
                            0.0_f64.into_pyobject(py)?.into_any(),
                        )
                    };
                    put!(current, "phase" => "virtual", "motion" => motion, "model_distance" => distance);
                    r = f.take();
                    self.state.setattr(front_name, py.None())?;
                    self.state.setattr(intern!(py, "after_pass"), true)?;
                    self.events.append("front_to_rear")?;
                    crossed = true;
                }
            }
        }

        if let Some(r) = &r {
            if phase(r)? != Phase::ToFront {
                if match &f {
                    Some(f) => phase(f)? != Phase::ToRear,
                    None => true,
                } {
                    if let Some(candidate) = candidate {
                        if num!(candidate, "x") < attr!(self.state, "GAP").extract::<f64>()? {
                            put!(r, "spacing" => true);
                        }
                    }
                }
                let x = num!(r, "x");
                if flag(r, intern!(py, "spacing"))?
                    && x > -attr!(self.state, "SPACE_DISTANCE").extract::<f64>()?
                    && !crossed
                {
                    let limit = match (previous_rear, rear_start) {
                        (Some(previous), Some(start)) if r.is(&previous) => py_max(
                            -attr!(self.state, "END").extract::<f64>()?,
                            start.extract::<f64>()? - 8. * self.dt,
                        ),
                        _ => x,
                    };
                    let space = -attr!(self.state, "SPACE_DISTANCE").extract::<f64>()?;
                    let adjusted =
                        py_max(limit, moved(x, space, self.dt, 8., 0.25)?.unwrap_or(space));
                    if adjusted < x {
                        put!(r, "x" => adjusted);
                        self.events.append("rear_spacing")?;
                    }
                }
                if num!(r, "x") <= -attr!(self.state, "SPACE_DISTANCE").extract::<f64>()? {
                    put!(r, "spacing" => false);
                }
            }
        }

        if let Some(candidate) = candidate.filter(|_| f.is_none()) {
            let clear = match &r {
                Some(r) => gap(candidate, r)? >= attr!(self.state, "GAP").extract::<f64>()? * 10.,
                None => true,
            };
            if clear
                && match &r {
                    Some(r) => phase(r)? != Phase::ToFront,
                    None => true,
                }
            {
                let next = front(candidate)?;
                self.state.setattr(front_name, &next)?;
                f = Some(next);
                self.state.setattr(intern!(py, "after_pass"), false)?;
                self.events.append("front_admitted")?;
            }
        }

        if let Some(f) = &f {
            let current_phase = phase(f)?;
            if current_phase != Phase::ToRear {
                let target = if let Some(c) = candidate {
                    if get!(c, "key").eq(get!(f, "key"))? {
                        Some(c.clone())
                    } else {
                        optional!(f, "target").map(record).transpose()?.flatten()
                    }
                } else {
                    optional!(f, "target").map(record).transpose()?.flatten()
                };
                if let Some(target) = target {
                    let x = get!(target, "x");
                    if current_phase == Phase::JoinFront {
                        self.advance(
                            f,
                            &x,
                            &get!(target, "y"),
                            self.dt,
                            attr!(self.state, "FAST_SPEED").extract()?,
                            attr!(self.state, "FAST_TAU").extract()?,
                        )?;
                        if get!(f, "x").eq(x)? && get!(f, "y").eq(get!(target, "y"))? {
                            put!(f, "phase" => "front");
                        }
                    } else {
                        let target_x = x.extract::<f64>()?;
                        let old = num!(f, "x");
                        let low = old - 20. * self.dt;
                        let high = old + 20. * self.dt;
                        if low > target_x || high < target_x {
                            put!(f, "x" => py_min(py_max(target_x, low), high));
                        } else {
                            put!(f, "x" => x);
                        }
                        let (y, detect) = (get!(target, "y"), get!(target, "detect"));
                        put!(f, "y" => y, "detect" => detect);
                    }
                }
                if let Some(r) = &r {
                    if phase(r)? != Phase::ToFront
                        && gap(f, r)? < attr!(self.state, "GAP").extract::<f64>()? * 10.
                    {
                        return Ok((None, Some(r.clone())));
                    }
                }
            }
        }
        Ok((f, r))
    }
}

impl<'a, 'py> Flow<'a, 'py> {
    fn new(state: &'a Bound<'py, PyAny>, now: f64, dt: f64) -> PyResult<Self> {
        let events = PyList::empty(state.py());
        state.setattr(intern!(state.py(), "events"), &events)?;
        Ok(Self {
            state,
            events,
            now,
            dt,
        })
    }
}

pub(super) fn reset<'py>(policy: &Bound<'py, PyAny>, flows: &Bound<'py, PyAny>) -> PyResult<()> {
    let py = policy.py();
    let flow_type = flows.get_item(0)?.get_type();
    let reset = PyList::empty(py);
    for _ in 0..2 {
        let flow = empty_object(flow_type.as_any())?;
        flow.setattr(intern!(py, "front"), py.None())?;
        flow.setattr(intern!(py, "rear"), py.None())?;
        flow.setattr(intern!(py, "bsd"), false)?;
        flow.setattr(intern!(py, "bsd_off"), py.None())?;
        flow.setattr(intern!(py, "after_pass"), false)?;
        flow.setattr(intern!(py, "events"), PyList::empty(py))?;
        reset.append(flow)?;
    }
    policy.setattr(intern!(py, "flows"), reset)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render<'py>(
    policy: &Bound<'py, PyAny>,
    values: &Record<'py>,
    selected: &[Bound<'py, PyAny>; 3],
    filters: &Bound<'py, PyAny>,
    rear_laterals: [f64; 2],
    now: f64,
    dt: f64,
    blind: [bool; 2],
    lane_change: bool,
) -> PyResult<()> {
    let py = policy.py();
    let rears = attr!(policy, "rears");
    let flows = attr!(policy, "flows");
    let now_value = now.into_pyobject(py)?.into_any();
    for (side, rear_lateral) in rear_laterals.into_iter().enumerate() {
        let fields = DisplayFields::new(py, side);
        let DisplayFields {
            front_detect: fd,
            front_distance: fx,
            front_lateral: fy,
            rear_detect: rd,
            rear_distance: rx,
            rear_lateral: ry,
        } = fields;
        let state = flows.get_item(side)?;
        let front_limit = attr!(state, "FRONT_LIMIT").extract::<f64>()?;
        let (x, y) = (field(values, fx)?, field(values, fy)?);
        let key = selected[side + 1].clone();
        let detect = field(values, fd)?;
        let mut candidate = None;
        if detect.is_truthy()? {
            let distance: f64 = x.extract()?;
            if distance.is_finite()
                && y.extract::<f64>()?.is_finite()
                && (0. ..=front_limit).contains(&distance)
            {
                let key = if key.is_none() {
                    ("display", side).into_pyobject(py)?.into_any()
                } else {
                    key
                };
                let drawing = PyDict::new(py);
                put!(drawing, "key" => key, "x" => x, "y" => y, "detect" => detect);
                candidate = Some(drawing);
            }
        }
        let motion = record(rears.get_item(side)?)?;
        if let Some(motion) = &motion {
            let shown = num!(motion, "shown");
            let max_time: f64 = attr!(policy, "MAX_TIME").extract()?;
            let deadline = shown + max_time + attr!(policy, "EXIT_GRACE").extract::<f64>()?;
            let max_speed = attr!(policy, "EXIT_MAX_SPEED");
            put!(motion, "force_at" => shown + max_time, "deadline" => deadline, "max_speed" => max_speed);
        }
        let rear_lateral = rear_lateral.into_pyobject(py)?.into_any();
        let elsewhere = PyTuple::new(py, [&selected[0], &selected[2 - side]])?.into_any();
        let flow = Flow::new(&state, now, dt)?;
        let (f, r) = flow.update(
            &now_value,
            candidate.as_ref(),
            blind[side],
            &rear_lateral,
            &elsewhere,
            motion.as_ref(),
            lane_change,
        )?;
        if let Some(f) = &f {
            values.set_item(fd, get!(f, "detect"))?;
        } else {
            values.set_item(fd, 0)?;
        }
        if let Some(r) = &r {
            values.set_item(rd, get!(r, "detect"))?;
        } else {
            values.set_item(rd, 0)?;
        }
        if let Some(f) = &f {
            let (x, y) = (get!(f, "x"), get!(f, "y"));
            values.set_item(fx, x)?;
            values.set_item(fy, y)?;
        }
        let filter = filters.get_item(side)?;
        if let Some(r) = &r {
            let (x, y) = (-num!(r, "x"), get!(r, "y"));
            values.set_item(rx, x)?;
            values.set_item(ry, y)?;
            reset_filter(&filter, x)?;
        } else {
            reset_filter(&filter, 16.)?;
        }
        if !flow.events.is_empty() {
            let events = attr!(policy, "events");
            let events = events.cast::<PyList>().map_err(PyErr::from)?;
            for name in flow.events.iter() {
                let event = PyDict::new(py);
                put!(event, "side" => side, "event" => name);
                events.append(event)?;
            }
        }
    }
    Ok(())
}
