//! Rear display policy for both sides. The existing Python data containers remain
//! available for replay tools while Rust advances the observation and drawing state.

use pyo3::exceptions::{PyKeyError, PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyString, PyTuple};

use crate::display_filter::{self, reset as reset_filter};
use crate::guard;
use crate::map_object::MapObject;
use crate::observe::crossing;
use crate::pymath::{py_max, py_min, py_sum};

type Record<'py> = Bound<'py, PyDict>;

fn field<'py>(dict: &Record<'py>, key: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
}

macro_rules! get {
    ($dict:expr, $key:literal) => {
        field(&$dict, intern!($dict.py(), $key))?
    };
}

macro_rules! optional {
    ($dict:expr, $key:literal) => {
        $dict.get_item(intern!($dict.py(), $key))?
    };
}

macro_rules! num {
    ($dict:expr, $key:literal) => {
        get!($dict, $key).extract::<f64>()?
    };
}

macro_rules! put {
    ($dict:expr, $($key:literal => $value:expr),+ $(,)?) => {
        $($dict.set_item(intern!($dict.py(), $key), $value)?;)+
    };
}

macro_rules! attr {
    ($obj:expr, $name:literal) => {
        $obj.getattr(intern!($obj.py(), $name))?
    };
}

mod side_flow;

fn record(value: Bound<'_, PyAny>) -> PyResult<Option<Record<'_>>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.cast_into::<PyDict>().map_err(PyErr::from)?))
    }
}

fn phase(dict: &Record<'_>, name: &str) -> PyResult<bool> {
    get!(dict, "phase").eq(name)
}

struct Sample {
    x: f64,
    y: f64,
    vrel: f64,
    vlead: f64,
    vehicle: bool,
    stamp: f64,
}

struct Side<'a, 'py> {
    policy: &'a Bound<'py, PyAny>,
    values: &'a Record<'py>,
    filters: &'a Bound<'py, PyAny>,
    points: &'a Record<'py>,
    index: usize,
    now: f64,
    fields: &'a DisplayFields<'py>,
}

impl<'py> Side<'_, 'py> {
    fn sample(&self, source: &Bound<'py, PyAny>) -> PyResult<Option<Sample>> {
        let Some(point) = self.points.get_item(source)? else {
            return Ok(None);
        };
        let py = point.py();
        if point.is_none()
            || !attr!(point, "radarSource")
                .str()?
                .as_any()
                .eq(intern!(py, "frontRadar"))?
        {
            return Ok(None);
        }
        let raw = [
            attr!(point, "dRel"),
            attr!(point, "yRel"),
            attr!(point, "vRel"),
            attr!(point, "vLead"),
        ];
        let mut values = [0.; 4];
        for (value, raw) in values.iter_mut().zip(raw) {
            *value = raw.extract::<f64>()?;
            if !value.is_finite() {
                return Ok(None);
            }
        }
        let [x, y, vrel, vlead] = values;
        let vehicle = point
            .getattr_opt(intern!(py, "motionState"))?
            .map_or(Ok(false), |motion| motion.eq(2))?;
        Ok(Some(Sample {
            x,
            y,
            vrel,
            vlead,
            vehicle,
            stamp: attr!(self.policy, "raw_stamp").extract()?,
        }))
    }

    fn event(&self, event: Record<'py>) -> PyResult<()> {
        attr!(self.policy, "events")
            .cast::<PyList>()
            .map_err(PyErr::from)?
            .append(event)
    }

    fn drop(&self, memories: &Bound<'py, PyAny>, reason: &'static str) -> PyResult<()> {
        let py = self.policy.py();
        if let Some(saved) = record(memories.get_item(self.index)?)? {
            let event = PyDict::new(py);
            let saved_phase = get!(saved, "phase");
            put!(event, "side" => self.index, "event" => reason, "source" => get!(saved, "source"), "phase" => &saved_phase,
                "distance" => get!(saved, "distance"), "age" => self.now - num!(saved, "seen"),
                "qualified" => optional!(saved, "qualified").unwrap_or_else(|| py.None().into_bound(py)),
                "hits" => optional!(saved, "hits").unwrap_or_else(|| py.None().into_bound(py)));
            self.event(event)?;
            if reason != "bsd" && (saved_phase.eq("coast")? || saved_phase.eq("rear")?) {
                self.values.set_item(self.fields.rear_detect, 0)?;
                reset_filter(&self.filters.get_item(self.index)?, 15.)?;
            }
        }
        memories.set_item(self.index, py.None())
    }

    fn anchor_rate(&self, saved: &Record<'py>) -> PyResult<()> {
        let x = num!(saved, "x") + self.dx(saved, num!(saved, "seen"), num!(saved, "update"))?;
        let scale: f64 = attr!(self.policy, "SCALE").extract()?;
        let rate = if x > 0.1 {
            py_min(py_max(num!(saved, "signed") / x, scale), 1.5)
        } else {
            scale
        };
        put!(saved, "rate" => rate);
        Ok(())
    }

    fn start_coast(&self, reason: &'static str) -> PyResult<()> {
        let py = self.policy.py();
        let fronts = attr!(self.policy, "fronts");
        let rears = attr!(self.policy, "rears");
        let Some(saved) = record(fronts.get_item(self.index)?)? else {
            return Ok(());
        };
        if num!(saved, "drawn_stamp") > num!(saved, "update") {
            put!(saved, "distance" => get!(saved, "drawn_distance"), "update" => get!(saved, "drawn_stamp"),
                "lateral" => get!(saved, "drawn_lateral"));
        }
        put!(saved, "signed" => get!(saved, "distance"));
        self.anchor_rate(&saved)?;
        let seen = num!(saved, "seen");
        let limit: f64 = attr!(self.policy, "FRONT_COAST_TIME").extract()?;
        let mut point = self.sample(&get!(saved, "source"))?;
        if let Some(sample) = &point {
            if sample.stamp > seen && !self.continuous(&saved, sample, limit)? {
                point = None;
            }
        }
        let predicted = num!(saved, "x") + self.dx(&saved, seen, self.now)?;
        let x = point.as_ref().map_or(predicted, |sample| sample.x);
        let mut recent: Vec<f64> = if let Some(history) = optional!(saved, "vlead_hist") {
            history.extract()?
        } else {
            Vec::new()
        };
        if let Some(sample) = point.as_ref().filter(|sample| sample.stamp > seen) {
            recent.push(sample.vlead);
        }
        let start = recent.len().saturating_sub(3);
        let vlead = if recent.is_empty() {
            num!(saved, "vlead")
        } else {
            median_values(&mut recent[start..])
        };
        let reachable = flag(&saved, intern!(py, "qualified"))?
            && self.now - seen <= limit
            && predicted >= attr!(self.policy, "COAST_MIN_X").extract::<f64>()?;
        if !reachable {
            return self.drop(&fronts, reason);
        }
        if x >= 3. {
            return self.drop(&fronts, "still_ahead");
        }
        if vlead < 0. {
            return self.drop(&fronts, "stopped");
        }

        fronts.set_item(self.index, py.None())?;
        self.drop(&rears, "rear_replaced")?;
        let consumed = attr!(self.policy, "consumed").get_item(self.index)?;
        let append = intern!(py, "append");
        consumed.call_method1(append, (get!(saved, "key"),))?;
        for alias in get!(saved, "aliases").try_iter()? {
            consumed.call_method1(append, (alias?,))?;
        }
        let coast_y = point.as_ref().map_or(num!(saved, "y"), |sample| sample.y);
        put!(saved, "phase" => "coast", "distance" => 0., "shown" => self.now, "start" => py.None(), "speed" => 0.,
            "coast_y" => coast_y);
        rears.set_item(self.index, &saved)?;
        reset_filter(&self.filters.get_item(self.index)?, 0.)?;
        let event = PyDict::new(py);
        put!(event, "side" => self.index, "event" => "coast", "reason" => reason, "source" => get!(saved, "source"),
            "x" => get!(saved, "x"), "vrel" => get!(saved, "vrel"), "vlead" => get!(saved, "vlead"),
            "v_anchor" => get!(saved, "v_anchor"), "v_ego" => attr!(self.policy, "v_ego"), "front_distance" => get!(saved, "signed"));
        self.event(event)
    }

    fn alias(&self, key: &Bound<'py, PyAny>) -> PyResult<bool> {
        let rears = attr!(self.policy, "rears");
        let Some(coast) = record(rears.get_item(self.index)?)? else {
            return Ok(false);
        };
        if !phase(&coast, "coast")? {
            return Ok(false);
        }
        let point = self.sample(&key.get_item(0)?)?;
        let x = if let Some(point) = &point {
            point.x
        } else {
            field(self.values, self.fields.front_distance)?.extract::<f64>()?
                / attr!(self.policy, "SCALE").extract::<f64>()?
        };
        let predicted = num!(coast, "x") + self.dx(&coast, num!(coast, "seen"), self.now)?;
        if (x - predicted)
            .abs()
            .partial_cmp(&attr!(self.policy, "ALIAS_GAP").extract()?)
            != Some(std::cmp::Ordering::Less)
        {
            return Ok(false);
        }
        if let Some(point) = &point {
            if !((point.y - num!(coast, "y")).abs()
                <= attr!(self.policy, "ALIAS_Y").extract::<f64>()?
                && (point.vrel - num!(coast, "vrel")).abs()
                    <= attr!(self.policy, "ALIAS_VREL").extract::<f64>()?)
            {
                return Ok(false);
            }
        }
        get!(coast, "aliases").cast::<PyList>()?.append(key)?;
        attr!(self.policy, "consumed")
            .get_item(self.index)?
            .call_method1(intern!(key.py(), "append"), (key,))?;
        self.values.set_item(self.fields.front_detect, 0)?;
        Ok(true)
    }

    fn dx(&self, saved: &Record<'py>, t0: f64, t1: f64) -> PyResult<f64> {
        let vrel = num!(saved, "vrel");
        if t0 >= t1 || t0.is_nan() || t1.is_nan() {
            return Ok(vrel * 0.);
        }
        let hist = attr!(self.policy, "ego_hist");
        let anchor = num!(saved, "v_anchor");
        if anchor < attr!(self.policy, "ANCHOR_MIN_EGO").extract()? {
            return Ok(vrel * 0.);
        }
        let max_ratio = attr!(self.policy, "RATIO_MAX").extract()?;
        let mut recent = [0.; 8];
        let mut older = Vec::new();
        let mut count = 0;
        // History is kept in send order and cleared on a clock reset.
        // Scan only the suffix in range, then sum in the original order.
        for row in hist
            .call_method0(intern!(hist.py(), "__reversed__"))?
            .try_iter()?
        {
            let row = row?;
            let row = row.cast::<PyTuple>().map_err(PyErr::from)?;
            let t: f64 = row.get_item(0)?.extract()?;
            if t <= t0 {
                break;
            }
            if t <= t1 {
                let v: f64 = row.get_item(1)?.extract()?;
                let dt: f64 = row.get_item(2)?.extract()?;
                let term = dt * py_min(py_max(v / anchor, 0.), max_ratio);
                if let Some(slot) = recent.get_mut(count) {
                    *slot = term;
                    count += 1;
                } else {
                    older.push(term);
                }
            }
        }
        Ok(vrel
            * py_sum(
                older
                    .into_iter()
                    .rev()
                    .chain(recent[..count].iter().rev().copied()),
            )
            .unwrap_or(0.))
    }

    fn ratio(&self, saved: &Record<'py>, v: f64) -> PyResult<f64> {
        let anchor = num!(saved, "v_anchor");
        Ok(
            if anchor < attr!(self.policy, "ANCHOR_MIN_EGO").extract()? {
                0.
            } else {
                py_min(
                    py_max(v / anchor, 0.),
                    attr!(self.policy, "RATIO_MAX").extract()?,
                )
            },
        )
    }

    fn continuous(&self, saved: &Record<'py>, point: &Sample, limit: f64) -> PyResult<bool> {
        let seen = num!(saved, "seen");
        let elapsed = point.stamp - seen;
        Ok(0. < elapsed
            && elapsed <= limit
            && (point.x - num!(saved, "x") - self.dx(saved, seen, point.stamp)?).abs()
                <= attr!(self.policy, "X_GATE").extract::<f64>()? + 2. * elapsed
            && (point.y - num!(saved, "y")).abs()
                <= (if get!(saved, "qualified").is_truthy()? {
                    attr!(self.policy, "QUALIFIED_Y_GATE").extract::<f64>()?
                } else {
                    0.25
                }) + elapsed
            && (point.vrel - num!(saved, "vrel")).abs()
                <= attr!(self.policy, "VREL_GATE").extract::<f64>()? + 2. * elapsed)
    }

    fn signed(&self, saved: &Record<'py>) -> PyResult<f64> {
        Ok(num!(saved, "signed")
            + num!(saved, "rate") * self.dx(saved, num!(saved, "update"), self.now)?)
    }

    fn retreat(&self, saved: &Record<'py>) -> PyResult<f64> {
        Ok(py_min(
            attr!(self.policy, "EXIT_MAX_SPEED").extract()?,
            py_max(
                0.,
                -num!(saved, "vrel")
                    * self.ratio(saved, attr!(self.policy, "v_ego").extract()?)?
                    * attr!(self.policy, "SCALE").extract::<f64>()?,
            ),
        ))
    }

    fn observe_front(
        &self,
        key: &Bound<'py, PyAny>,
        proven: bool,
        hold: bool,
        front: &DisplayFields<'py>,
    ) -> PyResult<()> {
        let fronts = attr!(self.policy, "fronts");
        let mut saved = record(fronts.get_item(self.index)?)?;
        let stale = attr!(self.policy, "STALE_HOLD").extract::<f64>()?;
        let py = self.policy.py();
        if let Some(memory) = &saved {
            let mut rekey = !key.is_none() && key.ne(get!(memory, "key"))?;
            if rekey && key.get_item(0)?.eq(get!(memory, "source"))? && proven && !hold {
                if let Some(point) = self.sample(&key.get_item(0)?)? {
                    if point.stamp > num!(memory, "seen")
                        && self.continuous(memory, &point, stale + 0.05 + 1e-6)?
                    {
                        let aliases = get!(memory, "aliases")
                            .cast_into::<PyList>()
                            .map_err(PyErr::from)?;
                        let previous = get!(memory, "key");
                        if !aliases.contains(&previous)? {
                            aliases.append(previous)?;
                        }
                        let remaining = PyList::empty(py);
                        for alias in aliases.iter() {
                            if alias.ne(key)? {
                                remaining.append(alias)?;
                            }
                        }
                        put!(memory, "aliases" => remaining, "key" => key);
                        rekey = false;
                    }
                }
            }
            if rekey {
                if get!(memory, "qualified").is_truthy()? {
                    self.start_coast("object_replaced")?;
                } else {
                    self.drop(&fronts, "object_replaced")?;
                }
                fronts.set_item(self.index, py.None())?;
                saved = None;
            }
        }
        if !key.is_none() && saved.is_none() && self.alias(key)? {
            return Ok(());
        }
        if key.is_none() || !field(self.values, front.front_detect)?.is_truthy()? {
            if let Some(memory) = saved {
                if get!(memory, "qualified").is_truthy()? {
                    self.start_coast("selection_ended")?;
                } else {
                    self.drop(&fronts, "no_qualified_zero_crossing")?;
                }
            }
            return Ok(());
        }

        // Match the keyword evaluation order of dict.update: read both values before mutating the memory.
        if let Some(memory) = &saved {
            let (distance, lateral) = front.drawing(self.values)?;
            put!(memory, "drawn_distance" => distance, "drawn_stamp" => self.now, "drawn_lateral" => lateral);
        }
        let point = self.sample(&key.get_item(0)?)?;
        if hold {
            return Ok(());
        }
        if !proven || point.is_none() {
            if let Some(memory) = &saved {
                if get!(memory, "qualified").is_truthy()? {
                    if proven
                        && !attr!(self.policy, "fresh").is_truthy()?
                        && self.now - num!(memory, "seen") <= stale + 1e-6
                    {
                        return Ok(());
                    }
                    self.start_coast(if proven {
                        "measurement_lost"
                    } else {
                        "not_proven"
                    })?;
                    if attr!(self.policy, "rears").get_item(self.index)?.is(memory) {
                        self.values.set_item(front.front_detect, 0)?;
                    }
                    return Ok(());
                }
                if self.now - num!(memory, "seen") <= stale + 1e-6 {
                    return Ok(());
                }
            }
            self.drop(&fronts, "measurement_unqualified")?;
            return Ok(());
        }
        let Some(point) = point else { return Ok(()) };
        let Sample {
            stamp,
            x,
            y,
            vrel: vrel_raw,
            vlead: vlead_raw,
            vehicle,
        } = point;
        let continuous = if let Some(memory) = &saved {
            if stamp <= num!(memory, "seen") {
                return Ok(());
            }
            self.continuous(memory, &point, stale + 0.05 + 1e-6)?
        } else {
            false
        };
        let memory = match saved {
            Some(memory) if continuous => memory,
            previous => {
                if previous.is_some() {
                    self.drop(&fronts, "identity_innovation")?;
                }
                let memory = PyDict::new(py);
                put!(memory, "key" => key, "source" => key.get_item(0)?, "first" => stamp, "first_x" => x,
                    "hits" => 0, "moving_hits" => 0, "phase" => "front", "qualified" => false, "aliases" => PyList::empty(py));
                fronts.set_item(self.index, &memory)?;
                memory
            }
        };

        let size = attr!(self.policy, "VREL_N").extract()?;
        let vrel = median(&memory, intern!(py, "vrel_hist"), vrel_raw, size)?;
        let vlead = median(&memory, intern!(py, "vlead_hist"), vlead_raw, size)?;
        let moving_size: usize = attr!(self.policy, "MOVING_N").extract()?;
        let vlead_long = median(
            &memory,
            intern!(py, "vlead_long"),
            vlead_raw,
            if moving_size == 0 { size } else { moving_size },
        )?;
        let v_ego = attr!(self.policy, "v_ego").extract::<f64>()?;
        let (distance, lateral) = front.drawing(self.values)?;
        let detect = field(self.values, front.front_detect)?;
        let detect = if detect.is_exact_instance_of::<PyInt>() {
            detect
        } else {
            py.get_type::<PyInt>().call1((detect,))?
        };
        put!(memory, "x" => x, "y" => y, "vrel" => vrel, "vrel_raw" => vrel_raw, "vlead" => vlead,
            "v_anchor" => v_ego, "seen" => stamp, "update" => self.now, "distance" => distance, "lateral" => lateral,
            "drawn_distance" => distance, "drawn_stamp" => self.now, "drawn_lateral" => lateral, "detect" => detect);

        let hits = get!(memory, "hits").extract::<i64>()? + 1;
        put!(memory, "hits" => hits);
        let radar_moving = optional!(memory, "radar_moving")
            .map_or(Ok(0), |v| v.extract::<i64>())?
            + i64::from(vehicle);
        put!(memory, "radar_moving" => radar_moving);
        let moving_hits = get!(memory, "moving_hits").extract::<i64>()?
            + i64::from(
                vehicle || {
                    let context = attr!(self.policy, "context");
                    !context.is_none() && context.get_item(0)?.eq(intern!(py, "map"))?
                },
            );
        put!(memory, "moving_hits" => moving_hits);
        let qualified = get!(memory, "qualified").is_truthy()?
            || (hits >= 4
                && moving_hits >= 3
                && self.now - num!(memory, "first")
                    >= attr!(self.policy, "CONFIRM_TIME").extract::<f64>()?
                && num!(memory, "first_x") - x >= 0.25
                && -0.3 >= vrel
                && vlead_long >= {
                    let moving = attr!(self.policy, "MOVING_FRACTION").extract::<f64>()? * v_ego;
                    if radar_moving >= 3 {
                        moving - attr!(self.policy, "MOVING_SLACK").extract::<f64>()?
                    } else {
                        py_max(
                            moving - attr!(self.policy, "MOVING_SLACK_MAP").extract::<f64>()?,
                            attr!(self.policy, "MOVING_VLEAD").extract()?,
                        )
                    }
                });
        put!(memory, "qualified" => qualified);
        Ok(())
    }

    fn coast(&self, rears: &Bound<'py, PyAny>, saved: &Record<'py>) -> PyResult<bool> {
        let py = saved.py();
        if let Some(point) = self.sample(&get!(saved, "source"))? {
            let Sample {
                stamp,
                x,
                y,
                vrel: raw,
                vlead,
                ..
            } = point;
            if stamp > num!(saved, "seen")
                && self.continuous(
                    saved,
                    &point,
                    attr!(self.policy, "FRONT_COAST_TIME").extract()?,
                )?
            {
                if x >= 3. {
                    self.drop(rears, "still_ahead")?;
                    return Ok(false);
                }
                if (y - num!(saved, "coast_y")).abs() > attr!(self.policy, "LANE_HALF").extract()? {
                    self.drop(rears, "lane_departed")?;
                    return Ok(false);
                }
                let predicted = num!(saved, "x") + self.dx(saved, num!(saved, "seen"), stamp)?;
                let n = attr!(self.policy, "VREL_N").extract()?;
                let vrel = median(saved, intern!(py, "vrel_hist"), raw, n)?;
                let vlead = median(saved, intern!(py, "vlead_hist"), vlead, n)?;
                let signed = self.signed(saved)? + num!(saved, "rate") * (x - predicted);
                let anchor = attr!(self.policy, "v_ego");
                put!(saved, "signed" => signed, "update" => self.now, "x" => x, "y" => y, "vrel" => vrel,
                    "vrel_raw" => raw, "vlead" => vlead, "v_anchor" => anchor, "seen" => stamp);
                let x = num!(saved, "x")
                    + self.dx(saved, num!(saved, "seen"), num!(saved, "update"))?;
                let scale: f64 = attr!(self.policy, "SCALE").extract()?;
                let rate = if x > 0.1 {
                    py_min(py_max(num!(saved, "signed") / x, scale), 1.5)
                } else {
                    scale
                };
                put!(saved, "rate" => rate);
            }
        }
        let seen = num!(saved, "seen");
        if num!(saved, "x") + self.dx(saved, seen, self.now)? >= 3. {
            self.drop(rears, "still_ahead")?;
            return Ok(false);
        }
        let signed = self.signed(saved)?;
        put!(saved, "signed" => signed, "update" => self.now);
        if signed > 0. {
            return Ok(true);
        }
        let speed = self.retreat(saved)?;
        put!(saved, "phase" => "rear", "start" => self.now, "speed" => speed);
        let event = PyDict::new(py);
        put!(event, "side" => self.index, "event" => "handoff", "source" => get!(saved, "source"), "key" => get!(saved, "key"),
            "distance" => get!(saved, "distance"), "seen_age" => self.now - seen, "speed" => speed, "lateral" => get!(saved, "lateral"),
            "coast_time" => self.now - num!(saved, "shown"), "origin" => signed <= 0.);
        attr!(self.policy, "events")
            .cast::<PyList>()
            .map_err(PyErr::from)?
            .append(event)?;
        Ok(true)
    }

    fn draw_rear(&self, fields: &DisplayFields<'py>, dt: f64) -> PyResult<()> {
        let rears = attr!(self.policy, "rears");
        let Some(saved) = record(rears.get_item(self.index)?)? else {
            return Ok(());
        };
        let py = saved.py();
        if phase(&saved, "coast")? && !self.coast(&rears, &saved)? {
            return Ok(());
        }
        let rear = phase(&saved, "rear")?;
        let age = self.now - num!(saved, "shown");
        let max_time: f64 = attr!(self.policy, "MAX_TIME").extract()?;
        let grace = if rear {
            attr!(self.policy, "EXIT_GRACE").extract::<f64>()?
        } else {
            0.
        };
        if age >= max_time + grace {
            return self.drop(&rears, "timeout");
        }
        let mut forced = optional!(saved, "forced").filter(|value| !value.is_none());
        if rear
            && age >= max_time
            && !forced
                .as_ref()
                .map_or(Ok(false), |value| value.is_truthy())?
        {
            let at = self.now.into_pyobject(py)?;
            put!(saved, "forced" => &at);
            forced = Some(at.into_any());
            let event = PyDict::new(py);
            put!(event, "side" => self.index, "event" => "forced_exit", "source" => get!(saved, "source"), "distance" => get!(saved, "distance"));
            attr!(self.policy, "events")
                .cast::<PyList>()
                .map_err(PyErr::from)?
                .append(event)?;
        }
        if rear && num!(saved, "start") < self.now {
            let mut ramp_start = py_max(
                num!(saved, "seen") + attr!(self.policy, "EXIT_DELAY").extract::<f64>()?,
                num!(saved, "start"),
            );
            let forced = forced
                .as_ref()
                .map_or(Ok(false), |value| value.is_truthy())?;
            if forced {
                ramp_start = py_min(ramp_start, num!(saved, "forced"));
            }
            let ramp0 = py_max(0., self.now - dt - ramp_start);
            let ramp1 = py_max(0., self.now - ramp_start);
            let mut base = self.retreat(&saved)?;
            if forced {
                base = py_max(base, attr!(self.policy, "EXIT_FORCED_SPEED").extract()?);
            }
            let max_speed: f64 = attr!(self.policy, "EXIT_MAX_SPEED").extract()?;
            let accel: f64 = attr!(self.policy, "EXIT_ACCEL").extract()?;
            if accel == 0. {
                return Err(PyZeroDivisionError::new_err("float division by zero"));
            }
            let cap = py_max(0., (max_speed - base) / accel);
            let integral = |t| {
                let q = py_min(t, cap);
                0.5 * accel * q * q + accel * cap * py_max(0., t - cap)
            };
            let distance =
                num!(saved, "distance") + (base * dt + integral(ramp1) - integral(ramp0));
            put!(saved, "distance" => distance, "speed" => py_min(max_speed, base + accel * ramp1));
            if distance >= attr!(self.policy, "END_DISTANCE").extract()? {
                return self.drop(&rears, "distance_end");
            }
        }
        let front_distance = field(self.values, fields.front_distance)?;
        if field(self.values, fields.front_detect)?.is_truthy()? {
            let front_distance: f64 = front_distance.extract()?;
            let gap: f64 = attr!(self.policy, "MIN_FRONT_REAR_GAP").extract()?;
            if 0. <= front_distance && front_distance < gap {
                let clearance = 10. * (gap - front_distance);
                if clearance.is_nan() {
                    return Err(PyValueError::new_err("cannot convert float NaN to integer"));
                }
                if clearance.is_infinite() {
                    return Err(PyOverflowError::new_err(
                        "cannot convert float infinity to integer",
                    ));
                }
                put!(saved, "distance" => py_max(num!(saved, "distance"), clearance.ceil() / 10.));
            }
        }
        let distance = get!(saved, "distance");
        self.values.set_item(fields.rear_distance, &distance)?;
        self.values
            .set_item(fields.rear_lateral, get!(saved, "lateral"))?;
        self.values
            .set_item(fields.rear_detect, get!(saved, "detect"))?;
        reset_filter(&self.filters.get_item(self.index)?, distance.extract()?)?;
        Ok(())
    }
}

fn median<'py>(
    saved: &Record<'py>,
    name: &Bound<'py, PyString>,
    raw: f64,
    size: usize,
) -> PyResult<f64> {
    let hist = match saved.get_item(name)? {
        Some(hist) => hist.cast_into::<PyList>().map_err(PyErr::from)?,
        None => {
            let hist = PyList::empty(saved.py());
            saved.set_item(name, &hist)?;
            hist
        }
    };
    hist.append(raw)?;
    // In Python [: -0] is [:0], which only applies if a caller changed the window to zero.
    if size != 0 && hist.len() > size {
        hist.del_slice(0, hist.len() - size)?;
    }
    let mut short = [0.; 40];
    let mut long;
    let values = if hist.len() <= short.len() {
        &mut short[..hist.len()]
    } else {
        long = vec![0.; hist.len()];
        long.as_mut_slice()
    };
    for (value, item) in values.iter_mut().zip(hist.iter()) {
        *value = item.extract()?;
    }
    Ok(median_values(values))
}

fn median_values(values: &mut [f64]) -> f64 {
    // _sample admits only finite values. Stable sorting keeps the first sign of an equal zero.
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.
    } else {
        values[middle]
    }
}

fn flag(dict: &Record<'_>, key: &Bound<'_, PyString>) -> PyResult<bool> {
    dict.get_item(key)?
        .map_or(Ok(false), |value| value.is_truthy())
}

struct DisplayFields<'py> {
    front_detect: &'py Bound<'py, PyString>,
    front_distance: &'py Bound<'py, PyString>,
    front_lateral: &'py Bound<'py, PyString>,
    rear_detect: &'py Bound<'py, PyString>,
    rear_distance: &'py Bound<'py, PyString>,
    rear_lateral: &'py Bound<'py, PyString>,
}

impl DisplayFields<'_> {
    fn drawing(&self, values: &Record<'_>) -> PyResult<(f64, f64)> {
        Ok((
            field(values, self.front_distance)?.extract()?,
            field(values, self.front_lateral)?.extract()?,
        ))
    }
}

impl<'py> DisplayFields<'py> {
    fn new(py: Python<'py>, index: usize) -> Self {
        if index == 0 {
            DisplayFields {
                front_detect: intern!(py, "LF_DETECT"),
                front_distance: intern!(py, "LF_DETECT_DISTANCE"),
                front_lateral: intern!(py, "LF_DETECT_LATERAL"),
                rear_detect: intern!(py, "LR_DETECT"),
                rear_distance: intern!(py, "LR_DETECT_DISTANCE"),
                rear_lateral: intern!(py, "LR_DETECT_LATERAL"),
            }
        } else {
            DisplayFields {
                front_detect: intern!(py, "RF_DETECT"),
                front_distance: intern!(py, "RF_DETECT_DISTANCE"),
                front_lateral: intern!(py, "RF_DETECT_LATERAL"),
                rear_detect: intern!(py, "RR_DETECT"),
                rear_distance: intern!(py, "RR_DETECT_DISTANCE"),
                rear_lateral: intern!(py, "RR_DETECT_LATERAL"),
            }
        }
    }
}

fn out_flag(out: &Bound<'_, PyAny>, name: &Bound<'_, PyString>, default: bool) -> PyResult<bool> {
    out.getattr_opt(name)?
        .map_or(Ok(default), |value| value.is_truthy())
}

fn attr_text(
    item: &Bound<'_, PyAny>,
    name: &Bound<'_, PyString>,
    default: &str,
) -> PyResult<String> {
    item.getattr_opt(name)?.map_or_else(
        || Ok(default.into()),
        |item| Ok(item.str()?.to_str()?.to_owned()),
    )
}

#[allow(clippy::too_many_arguments)]
fn update<'py>(
    policy: &Bound<'py, PyAny>,
    values: &Record<'py>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    selected: &Bound<'py, PyAny>,
    filters: &Bound<'py, PyAny>,
    context: &Bound<'py, PyAny>,
    proven: [bool; 2],
    rear_laterals: [f64; 2],
) -> PyResult<()> {
    let py = policy.py();
    let now = (frame as f64) * 0.01;
    let last = attr!(policy, "last");
    let dt = if last.is_none() {
        0.
    } else {
        now - last.extract::<f64>()?
    };
    let events = PyList::empty(py);
    policy.setattr(intern!(py, "events"), &events)?;
    let out = attr!(cs, "out");
    let blind = [
        attr!(out, "leftBlindspot").is_truthy()?,
        attr!(out, "rightBlindspot").is_truthy()?,
    ];
    let meta = md
        .getattr_opt(intern!(py, "meta"))?
        .unwrap_or_else(|| py.None().into_bound(py));
    let changing = out_flag(&out, intern!(py, "leftBlinker"), false)?
        || out_flag(&out, intern!(py, "rightBlinker"), false)?
        || matches!(
            attr_text(&meta, intern!(py, "desire"), "")?.as_str(),
            "laneChangeLeft" | "laneChangeRight"
        )
        || attr_text(&meta, intern!(py, "laneChangeState"), "off")? != "off";
    policy.setattr(intern!(py, "changing"), changing)?;
    let probs = md.getattr_opt(intern!(py, "laneLineProbs"))?;
    let clock_reset = !attr!(policy, "context").eq(context)? || !(0. < dt && dt <= 0.2);
    let reverse = attr_text(&out, intern!(py, "gearShifter"), "")? == "reverse";
    let blocked = if clock_reset {
        Some("clock_or_context")
    } else if reverse {
        Some("reverse")
    } else if changing {
        Some("lane_change")
    } else {
        None
    };
    let can_valid = out_flag(&out, intern!(py, "canValid"), true)?;
    let mut lost = [true; 2];
    if let Some(probs) = probs.filter(|_| can_valid && !md.is_none()) {
        if probs.len()? >= 3 {
            for (index, lost) in lost.iter_mut().enumerate() {
                let p: f64 = probs.get_item(1 + index)?.extract()?;
                *lost = !p.is_finite() || p < 0.1;
            }
        }
    }

    policy.setattr(intern!(py, "context"), context)?;
    policy.setattr(intern!(py, "last"), now)?;
    let mut ego = out
        .getattr_opt(intern!(py, "vEgo"))?
        .map_or(Ok(0.0_f64), |speed| speed.extract())?;
    if !ego.is_finite() {
        ego = 0.;
    }
    policy.setattr(intern!(py, "v_ego"), ego)?;
    let hist = attr!(policy, "ego_hist");
    if clock_reset {
        hist.call_method0(intern!(py, "clear"))?;
    }
    hist.call_method1(
        intern!(py, "append"),
        ((now, ego, if clock_reset { 0. } else { dt }),),
    )?;
    let oldest = now - attr!(policy, "EGO_SPAN").extract::<f64>()?;
    while hist.len()? > 0 && hist.get_item(0)?.get_item(0)?.extract::<f64>()? < oldest {
        hist.call_method0(intern!(py, "popleft"))?;
    }

    let raw = attr!(cs, "live_tracks");
    if !attr!(policy, "raw").is(&raw) {
        policy.setattr(intern!(py, "raw"), &raw)?;
        policy.setattr(intern!(py, "raw_stamp"), now)?;
    }
    let fresh = !raw.is_none()
        && now - attr!(policy, "raw_stamp").extract::<f64>()?
            <= attr!(policy, "TRACKS_FRESH").extract::<f64>()?;
    policy.setattr(intern!(py, "fresh"), fresh)?;
    let points = PyDict::new(py);

    let fronts = attr!(policy, "fronts");
    let rears = attr!(policy, "rears");
    let flows = policy.getattr_opt(intern!(py, "flows"))?;
    let previous_rears = [record(rears.get_item(0)?)?, record(rears.get_item(1)?)?];
    let unproven = attr!(policy, "unproven");
    let consumed = attr!(policy, "consumed");
    let selected = [
        selected.get_item(0)?,
        selected.get_item(1)?,
        selected.get_item(2)?,
    ];
    if fresh
        && blocked.is_none()
        && (0..2).any(|i| !blind[i] && (!selected[i + 1].is_none() || previous_rears[i].is_some()))
    {
        load_points(&points, &raw)?;
    } else if fresh
        && blocked.is_none()
        && (0..2).any(|i| !blind[i] && fronts.get_item(i).is_ok_and(|p| !p.is_none()))
    {
        load_points(&points, &raw)?;
    }
    let fields = [DisplayFields::new(py, 0), DisplayFields::new(py, 1)];
    for index in 0..2 {
        let fields = &fields[index];
        let side = Side {
            policy,
            values,
            filters,
            points: &points,
            index,
            now,
            fields,
        };
        let once = !proven[index] && !unproven.get_item(index)?.is_truthy()?;
        unproven.set_item(index, !proven[index])?;
        if let Some(reason) = if blind[index] { Some("bsd") } else { blocked } {
            side.drop(&fronts, reason)?;
            side.drop(&rears, reason)?;
            continue;
        }
        if lost[index] {
            side.drop(&fronts, "confidence_lost")?;
        }
        let elsewhere = [&selected[0], &selected[2 - index]];
        for memories in [&fronts, &rears] {
            if let Some(saved) = record(memories.get_item(index)?)? {
                if related(&saved, elsewhere[0])? || related(&saved, elsewhere[1])? {
                    side.drop(memories, "changed_lane_slot")?;
                }
            }
        }
        let mut key = selected[index + 1].clone();
        if !key.is_none() && consumed.get_item(index)?.contains(&key)? {
            let point = side.sample(&key.get_item(0)?)?;
            let own = match record(rears.get_item(index)?)? {
                Some(saved) => related(&saved, &key)?,
                None => false,
            };
            if own && point.is_some_and(|point| point.x >= 1. && point.vrel > 0.3) {
                side.drop(&rears, "measured_front_reacquired")?;
            } else if own
                || field(values, fields.front_distance)?.extract::<f64>()?
                    <= attr!(policy, "SCALE").extract::<f64>()?
            {
                values.set_item(fields.front_detect, 0)?;
            }
            key = py.None().into_bound(py);
        }
        if !lost[index] {
            side.observe_front(&key, proven[index] || once, once, fields)?;
        }
        side.draw_rear(fields, dt)?;
    }

    let Some(flows) = flows else { return Ok(()) };
    for (index, previous) in previous_rears.iter().enumerate() {
        if let Some(saved) = previous
            .as_ref()
            .filter(|saved| rears.get_item(index).is_ok_and(|r| !r.is(*saved)))
        {
            let source = get!(saved, "source");
            let mut reason = "cancelled".into_pyobject(py)?.into_any();
            for event in events.iter() {
                let event = event.cast::<PyDict>()?;
                if get!(event, "side").eq(index)?
                    && optional!(event, "source")
                        .is_some_and(|key| key.eq(&source).unwrap_or(false))
                {
                    if let Some(phase) = optional!(event, "phase") {
                        if phase.eq("coast")? || phase.eq("rear")? {
                            reason = get!(event, "event");
                        }
                    }
                }
            }
            put!(saved, "end_reason" => reason);
        }
    }
    if clock_reset || !can_valid || reverse {
        side_flow::reset(policy, &flows)?;
    }
    if reverse || !can_valid {
        return Ok(());
    }
    side_flow::render(
        policy,
        values,
        &selected,
        filters,
        rear_laterals,
        now,
        if clock_reset { 0. } else { dt },
        blind,
        changing,
    )
}

fn related(saved: &Record<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(get!(saved, "key").eq(key)?
        || optional!(saved, "aliases").map_or(Ok(false), |aliases| aliases.contains(key))?)
}

fn load_points(points: &Record<'_>, raw: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = points.py();
    if let Some(raw_points) = raw.getattr_opt(intern!(py, "points"))? {
        for point in raw_points.try_iter()? {
            let point = point?;
            let key: i128 = attr!(point, "trackId").extract()?;
            points.set_item(key, point)?;
        }
    }
    Ok(())
}

/// The same BSD low-pass used by the Python fallback. Smooth handover is applied
/// afterwards and takes ownership of the actual drawn distances.
fn draw_bsd<'py>(
    values: &Record<'py>,
    out: &Bound<'py, PyAny>,
    filters: &Bound<'py, PyAny>,
    laterals: [f64; 2],
) -> PyResult<()> {
    let py = values.py();
    let active = [
        attr!(out, "leftBlindspot").is_truthy()?,
        attr!(out, "rightBlindspot").is_truthy()?,
    ];
    for (index, lateral) in laterals.into_iter().enumerate() {
        let filter = filters.get_item(index)?;
        if active[index] || display_filter::value(&filter)? < 15. {
            let target: i32 = if active[index] { 8 } else { 16 };
            let distance = display_filter::fixed(&filter, target.into_pyobject(py)?.as_any())?;
            let fields = DisplayFields::new(py, index);
            values.set_item(fields.rear_distance, distance)?;
            values.set_item(fields.rear_lateral, lateral)?;
            values.set_item(fields.rear_detect, 2)?;
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _update_rear_for_test<'py>(
    policy: &Bound<'py, PyAny>,
    values: &Record<'py>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    selected: &Bound<'py, PyAny>,
    filters: &Bound<'py, PyAny>,
    context: &Bound<'py, PyAny>,
    proven: [bool; 2],
    rear_laterals: [f64; 2],
) -> PyResult<()> {
    guard(|| {
        draw_bsd(values, &attr!(cs, "out"), filters, rear_laterals)?;
        if policy.is_none() {
            return Ok(());
        }
        update(
            policy,
            values,
            cs,
            md,
            frame,
            selected,
            filters,
            context,
            proven,
            rear_laterals,
        )
    })
}

fn map_selection<'py>(
    tracker: &Bound<'py, PyAny>,
    index: usize,
    key: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let py = tracker.py();
    if let Some(state) = crate::map_state::attached(tracker)? {
        let state = state.try_borrow()?;
        let Some(o) = key.extract::<Option<i64>>()?.and_then(|key| state.st.objects.get(&key)) else {
            return Ok((py.None().into_bound(py), false));
        };
        let identity = (o.source, o.born).into_pyobject(py)?.into_any();
        let proven = index > 0
            && !o.structure
            && !o.superseded
            && (o.radar_moving || o.doppler_confirmed)
            && !crossing(o, state.st.frame(), attr!(tracker, "CROSSING_TRAVEL").extract::<f64>()?);
        return Ok((identity, proven));
    }
    let objects = attr!(tracker, "objects").cast_into::<PyDict>()?;
    let Some(object) = objects.get_item(key)? else {
        return Ok((py.None().into_bound(py), false));
    };
    let identity = (attr!(object, "source"), attr!(object, "born"))
        .into_pyobject(py)?
        .into_any();
    let mut proven = false;
    if index > 0
        && !attr!(object, "structure").is_truthy()?
        && !attr!(object, "superseded").is_truthy()?
        && (attr!(object, "radar_moving").is_truthy()?
            || attr!(object, "doppler_confirmed").is_truthy()?)
    {
        let frame = crate::observe::read_frame(tracker)?;
        let travel = attr!(tracker, "CROSSING_TRAVEL").extract::<f64>()?;
        proven = if let Ok(object) = object.cast::<MapObject>() {
            !crossing(&object.try_borrow()?.data(), frame, travel)
        } else {
            // The map's reference class exposes the same fields.
            let origin = attr!(object, "origin");
            if origin.is_none() {
                true
            } else {
                let (ox, oy): (f64, f64) = origin.extract()?;
                let dx = attr!(object, "x").extract::<f64>()? - ox;
                let dy = attr!(object, "y").extract::<f64>()? - oy;
                let lateral = (frame.c * dy - frame.s * dx).abs();
                !(lateral >= travel && lateral > (frame.c * dx + frame.s * dy).abs())
            }
        };
    }
    Ok((identity, proven))
}

pub(crate) fn finish_map<'py>(
    values: &Record<'py>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    state: &Bound<'py, PyAny>,
    tracker: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = state.py();
    let left = attr!(state, "l_lane_f");
    let right = attr!(state, "r_lane_f");
    let offset = (attr!(right, "_filtered_value").extract::<f64>()?
        - attr!(left, "_filtered_value").extract::<f64>()?)
        / 2.;
    let laterals = [2.8 - offset, 2.8 + offset];
    let filters = (attr!(state, "lr_distance"), attr!(state, "rr_distance"))
        .into_pyobject(py)?
        .into_any();
    draw_bsd(values, &attr!(cs, "out"), &filters, laterals)?;
    let policy = attr!(state, "rear_handover");
    let mut chosen = Vec::with_capacity(3);
    let mut proven = [true; 2];
    for (index, key) in attr!(tracker, "selected").try_iter()?.enumerate().take(3) {
        let (identity, proof) = map_selection(tracker, index, &key?)?;
        if index > 0 {
            proven[index - 1] = proof;
        }
        chosen.push(identity);
    }
    let selected = PyTuple::new(py, chosen)?.into_any();
    let context = ("map", attr!(tracker, "generation"))
        .into_pyobject(py)?
        .into_any();
    update(
        &policy, values, cs, md, frame, &selected, &filters, &context, proven, laterals,
    )
}
