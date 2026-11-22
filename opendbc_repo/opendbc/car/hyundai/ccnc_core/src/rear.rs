//! Python inputs and CAN fields for the pure Rust rear display policy.

use pyo3::exceptions::PyAttributeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyString};

use ccnc_map::input::live_tracks::parse;
use ccnc_map::rear::{
    self, side_flow, Consts, Identity, Input, Number, RadarInput, RearSlots, RearState, Sample, Slot,
};

use crate::{display_filter, guard, map_state::py_err};

mod snapshot;

macro_rules! attr {
    ($obj:expr, $name:literal) => {
        $obj.getattr(intern!($obj.py(), $name))?
    };
}

type Values<'py> = Bound<'py, PyDict>;

#[pyclass(module = "ccnc_core")]
#[derive(Default)]
struct RearDisplay {
    state: RearState,
    context: Option<Py<PyAny>>,
    raw: Option<Py<PyAny>>,
}

struct Fields<'py> {
    distance: &'py Bound<'py, PyString>,
    lateral: &'py Bound<'py, PyString>,
    detect: &'py Bound<'py, PyString>,
}

impl<'py> Fields<'py> {
    fn front(py: Python<'py>, side: usize) -> Self {
        if side == 0 {
            Self {
                distance: intern!(py, "LF_DETECT_DISTANCE"),
                lateral: intern!(py, "LF_DETECT_LATERAL"),
                detect: intern!(py, "LF_DETECT"),
            }
        } else {
            Self {
                distance: intern!(py, "RF_DETECT_DISTANCE"),
                lateral: intern!(py, "RF_DETECT_LATERAL"),
                detect: intern!(py, "RF_DETECT"),
            }
        }
    }

    fn rear(py: Python<'py>, side: usize) -> Self {
        if side == 0 {
            Self {
                distance: intern!(py, "LR_DETECT_DISTANCE"),
                lateral: intern!(py, "LR_DETECT_LATERAL"),
                detect: intern!(py, "LR_DETECT"),
            }
        } else {
            Self {
                distance: intern!(py, "RR_DETECT_DISTANCE"),
                lateral: intern!(py, "RR_DETECT_LATERAL"),
                detect: intern!(py, "RR_DETECT"),
            }
        }
    }

    fn read(&self, values: &Values<'py>) -> PyResult<Slot> {
        let read = |field| values.get_item(field)?.map(read_number).transpose();
        Ok(Slot {
            distance: read(self.distance)?,
            lateral: read(self.lateral)?,
            detect: read(self.detect)?,
        })
    }

    fn write(&self, values: &Values<'py>, slot: Slot) -> PyResult<()> {
        for (key, value) in [
            (self.distance, slot.distance),
            (self.lateral, slot.lateral),
            (self.detect, slot.detect),
        ] {
            if let Some(value) = value {
                values.set_item(key, number(values.py(), value)?)?;
            }
        }
        Ok(())
    }
}

fn number(py: Python<'_>, value: Number) -> PyResult<Bound<'_, PyAny>> {
    match value {
        Number::Integer(value) => Ok(value.into_pyobject(py)?.into_any()),
        Number::Float(value) => Ok(value.into_pyobject(py)?.into_any()),
    }
}

fn read_number(value: Bound<'_, PyAny>) -> PyResult<Number> {
    if value.is_exact_instance_of::<PyInt>() {
        Ok(Number::Integer(value.extract()?))
    } else {
        Ok(Number::Float(value.extract()?))
    }
}

fn constants(policy: &Bound<'_, PyAny>) -> PyResult<Consts> {
    let py = policy.py();
    macro_rules! read {
        ($name:literal) => {
            attr!(policy, $name).extract()?
        };
    }
    let smooth = match policy.getattr_opt(intern!(py, "flows"))? {
        Some(flows) => {
            let read = |index| -> PyResult<side_flow::Consts> {
                let flow = flows.get_item(index)?;
                macro_rules! flow {
                    ($name:literal) => {
                        attr!(flow, $name).extract()?
                    };
                }
                Ok(side_flow::Consts {
                    hold: flow!("HOLD"),
                    end: flow!("END"),
                    gap: flow!("GAP"),
                    space_distance: flow!("SPACE_DISTANCE"),
                    front_limit: flow!("FRONT_LIMIT"),
                    fast_speed: flow!("FAST_SPEED"),
                    fast_tau: flow!("FAST_TAU"),
                    pass_speed: flow!("PASS_SPEED"),
                    exit_speed: flow!("EXIT_SPEED"),
                    exit_tau: flow!("EXIT_TAU"),
                    lateral_speed: flow!("LATERAL_SPEED"),
                    lateral_tau: flow!("LATERAL_TAU"),
                })
            };
            Some([read(0)?, read(1)?])
        }
        None => None,
    };
    Ok(Consts {
        scale: read!("SCALE"),
        end_distance: read!("END_DISTANCE"),
        min_gap: read!("MIN_FRONT_REAR_GAP"),
        confirm_time: read!("CONFIRM_TIME"),
        coast_time: read!("FRONT_COAST_TIME"),
        coast_min_x: read!("COAST_MIN_X"),
        qualified_y_gate: read!("QUALIFIED_Y_GATE"),
        exit_delay: read!("EXIT_DELAY"),
        exit_accel: read!("EXIT_ACCEL"),
        exit_max_speed: read!("EXIT_MAX_SPEED"),
        max_time: read!("MAX_TIME"),
        alias_gap: read!("ALIAS_GAP"),
        alias_y: read!("ALIAS_Y"),
        alias_vrel: read!("ALIAS_VREL"),
        lane_half: read!("LANE_HALF"),
        tracks_fresh: read!("TRACKS_FRESH"),
        stale_hold: read!("STALE_HOLD"),
        ego_span: read!("EGO_SPAN"),
        ratio_max: read!("RATIO_MAX"),
        anchor_min_ego: read!("ANCHOR_MIN_EGO"),
        vrel_n: read!("VREL_N"),
        x_gate: read!("X_GATE"),
        vrel_gate: read!("VREL_GATE"),
        moving_fraction: read!("MOVING_FRACTION"),
        moving_slack: read!("MOVING_SLACK"),
        moving_slack_map: read!("MOVING_SLACK_MAP"),
        moving_vlead: read!("MOVING_VLEAD"),
        moving_n: read!("MOVING_N"),
        exit_forced_speed: read!("EXIT_FORCED_SPEED"),
        exit_grace: read!("EXIT_GRACE"),
        smooth,
    })
}

fn valid_sample(x: f64, y: f64, vrel: f64, vlead: f64, vehicle: bool) -> Option<Sample> {
    [x, y, vrel, vlead].into_iter().all(f64::is_finite).then_some(Sample {
        x,
        y,
        vrel,
        vlead,
        vehicle,
    })
}

fn samples(raw: &Bound<'_, PyAny>, bytes: Option<&[u8]>) -> PyResult<Vec<(i128, Option<Sample>)>> {
    if let Some(points) = bytes.and_then(parse) {
        return Ok(points
            .iter()
            .map(|p| {
                let sample = if p.source == 0 {
                    valid_sample(
                        p.d_rel as f64,
                        p.y_rel as f64,
                        p.v_rel as f64,
                        p.v_lead as f64,
                        p.motion_state == 2,
                    )
                } else {
                    None
                };
                (p.track_id as i128, sample)
            })
            .collect());
    }
    let py = raw.py();
    let Some(points) = raw.getattr_opt(intern!(py, "points"))? else {
        return Ok(Vec::new());
    };
    points
        .try_iter()?
        .map(|p| {
            let p = p?;
            let id = attr!(p, "trackId").extract()?;
            let sample = if attr!(p, "radarSource").str()?.to_cow()? == "frontRadar" {
                let moving = p
                    .getattr_opt(intern!(py, "motionState"))?
                    .map_or(Ok(false), |v| v.eq(2))?;
                valid_sample(
                    attr!(p, "dRel").extract()?,
                    attr!(p, "yRel").extract()?,
                    attr!(p, "vRel").extract()?,
                    attr!(p, "vLead").extract()?,
                    moving,
                )
            } else {
                None
            };
            Ok((id, sample))
        })
        .collect()
}

fn flag(out: &Bound<'_, PyAny>, name: &Bound<'_, PyString>, default: bool) -> PyResult<bool> {
    out.getattr_opt(name)?.map_or(Ok(default), |v| v.is_truthy())
}

fn text(out: &Bound<'_, PyAny>, name: &Bound<'_, PyString>, default: &str) -> PyResult<String> {
    out.getattr_opt(name)?
        .map_or_else(|| Ok(default.to_owned()), |v| Ok(v.str()?.to_cow()?.into_owned()))
}

fn blind(out: &Bound<'_, PyAny>) -> PyResult<[bool; 2]> {
    Ok([
        attr!(out, "leftBlindspot").is_truthy()?,
        attr!(out, "rightBlindspot").is_truthy()?,
    ])
}

fn draw_bsd<'py>(
    values: &Values<'py>,
    filters: &Bound<'py, PyAny>,
    blind: [bool; 2],
    rear_lateral: [f64; 2],
) -> PyResult<()> {
    let py = values.py();
    for side in 0..2 {
        let filter = filters.get_item(side)?;
        let distance = if blind[side] {
            0.
        } else {
            display_filter::value(&filter)?
        };
        if let Some(target) = rear::bsd_target(blind[side], distance) {
            let distance = display_filter::fixed(&filter, target.into_pyobject(py)?.as_any())?;
            Fields::rear(py, side).write(
                values,
                Slot {
                    distance: Some(Number::Float(distance)),
                    lateral: Some(Number::Float(rear_lateral[side])),
                    detect: Some(Number::Integer(2)),
                },
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run<'py>(
    holder: &mut RearDisplay,
    values: &Values<'py>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    context: &Bound<'py, PyAny>,
    filters: &Bound<'py, PyAny>,
    rear_lateral: [f64; 2],
    advance: impl FnOnce(&mut RearState, Input<'_>, &mut RearSlots) -> ccnc_map::radar::Result<()>,
) -> PyResult<()> {
    let py = values.py();
    let out = attr!(cs, "out");
    let blind = blind(&out)?;
    let meta = md
        .getattr_opt(intern!(py, "meta"))?
        .unwrap_or_else(|| py.None().into_bound(py));
    let changing = flag(&out, intern!(py, "leftBlinker"), false)?
        || flag(&out, intern!(py, "rightBlinker"), false)?
        || matches!(
            text(&meta, intern!(py, "desire"), "")?.as_str(),
            "laneChangeLeft" | "laneChangeRight"
        )
        || text(&meta, intern!(py, "laneChangeState"), "off")? != "off";
    let can_valid = flag(&out, intern!(py, "canValid"), true)?;
    let mut lost = [true; 2];
    if let Some(probs) = md
        .getattr_opt(intern!(py, "laneLineProbs"))?
        .filter(|_| can_valid && !md.is_none())
    {
        if probs.len()? >= 3 {
            for (side, lost) in lost.iter_mut().enumerate() {
                let p: f64 = probs.get_item(side + 1)?.extract()?;
                *lost = !p.is_finite() || p < 0.1;
            }
        }
    }
    let reverse = text(&out, intern!(py, "gearShifter"), "")? == "reverse";
    let ego = out.getattr_opt(intern!(py, "vEgo"))?.map_or(Ok(0.), |v| v.extract())?;
    let context_changed = match &holder.context {
        Some(old) => !old.bind(py).eq(context)?,
        None => true,
    };
    holder.context = Some(context.clone().unbind());

    let raw = attr!(cs, "live_tracks");
    let unchanged = holder.raw.as_ref().map_or(raw.is_none(), |p| p.bind(py).is(&raw));
    let points;
    let radar = if unchanged {
        RadarInput::Unchanged
    } else if raw.is_none() {
        RadarInput::Missing
    } else {
        let bytes = cs.getattr_opt(intern!(py, "live_tracks_bytes"))?;
        let bytes = bytes
            .as_ref()
            .filter(|b| !b.is_none())
            .map(|bytes| bytes.extract())
            .transpose()?;
        points = samples(&raw, bytes)?;
        RadarInput::Points(&points)
    };
    holder.raw = Some(raw.unbind());
    let input = Input {
        now: frame as f64 * 0.01,
        context_changed,
        map_context: context.get_item(0)?.eq("map")?,
        changing,
        reverse,
        can_valid,
        ego,
        blind,
        lost,
        rear_lateral,
        radar,
    };
    let front = [Fields::front(py, 0), Fields::front(py, 1)];
    let rear = [Fields::rear(py, 0), Fields::rear(py, 1)];
    let mut slots = RearSlots::default();
    for i in 0..2 {
        slots[i].front = front[i].read(values)?;
        slots[i].rear = rear[i].read(values)?;
    }
    let result = advance(&mut holder.state, input, &mut slots);
    for i in 0..2 {
        front[i].write(values, slots[i].front)?;
        rear[i].write(values, slots[i].rear)?;
        if let Some(distance) = slots[i].reset_distance {
            display_filter::reset(&filters.get_item(i)?, distance)?;
        }
    }
    result.map_err(py_err)?;
    Ok(())
}

fn holder<'py>(policy: &Bound<'py, PyAny>) -> PyResult<Bound<'py, RearDisplay>> {
    let py = policy.py();
    let name = intern!(py, "_rust_rear");
    if let Some(holder) = policy.getattr_opt(name)? {
        return Ok(holder.cast_into::<RearDisplay>()?);
    }
    let holder = Bound::new(py, RearDisplay::default())?;
    policy.setattr(name, &holder)?;
    Ok(holder)
}

pub(crate) fn finish_map<'py>(
    values: &Values<'py>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    state: &Bound<'py, PyAny>,
    tracker: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = state.py();
    let laterals = rear::bsd_laterals(
        display_filter::value(&attr!(state, "l_lane_f"))?,
        display_filter::value(&attr!(state, "r_lane_f"))?,
    );
    let filters = (attr!(state, "lr_distance"), attr!(state, "rr_distance"))
        .into_pyobject(py)?
        .into_any();
    draw_bsd(values, &filters, blind(&attr!(cs, "out"))?, laterals)?;
    let policy = attr!(state, "rear_handover");
    if policy.is_none() {
        return Ok(());
    }
    let context = ("map", attr!(tracker, "generation")).into_pyobject(py)?.into_any();
    let map = crate::map_state::attached(tracker)?.ok_or_else(|| PyAttributeError::new_err("_rust_state"))?;
    let map = map.try_borrow()?;
    let travel = attr!(tracker, "CROSSING_TRAVEL").extract()?;
    let k = constants(&policy)?;
    let holder = holder(&policy)?;
    let mut holder = holder.try_borrow_mut()?;
    run(
        &mut holder,
        values,
        cs,
        md,
        frame,
        &context,
        &filters,
        laterals,
        |rear, input, slots| rear::finish_map(&map.st, rear, &k, input, slots, travel),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _update_rear_for_test<'py>(
    policy: &Bound<'py, PyAny>,
    values: &Values<'py>,
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
        draw_bsd(values, filters, blind(&attr!(cs, "out"))?, rear_laterals)?;
        if policy.is_none() {
            return Ok(());
        }
        let selected: [Option<(i128, f64)>; 3] = selected.extract()?;
        let selected = selected.map(|slot| slot.map(|(source, born)| Identity(source, born)));
        let k = constants(policy)?;
        let had_state = policy.hasattr("_rust_rear")?;
        let holder = holder(policy)?;
        let mut holder = holder.try_borrow_mut()?;
        if !had_state {
            snapshot::seed(policy, &mut holder)?;
        }
        run(
            &mut holder,
            values,
            cs,
            md,
            frame,
            context,
            filters,
            rear_laterals,
            |rear, input, slots| rear.update(&k, input, selected, proven, slots),
        )
    })
}

#[pyfunction]
pub(crate) fn _rear_snapshot_for_test<'py>(policy: &Bound<'py, PyAny>) -> PyResult<Values<'py>> {
    let holder = holder(policy)?;
    let holder = holder.try_borrow()?;
    snapshot::snapshot(policy, &holder)
}
