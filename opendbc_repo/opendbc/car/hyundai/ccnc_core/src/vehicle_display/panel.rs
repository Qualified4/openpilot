//! Display fields shared by the CCNC cluster templates. CAN packing stays with the caller.
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::model_v2::GeometryInput;
use crate::pymath::{py_max, py_min};

use super::StockLead;

const LANE_LINES: [&str; 2] = ["LANELINE_LEFT", "LANELINE_RIGHT"];
const CORNER_OBJECTS: [(&str, &str); 4] = [
    ("LF_DETECT", "LF_DETECT_DISTANCE"),
    ("RF_DETECT", "RF_DETECT_DISTANCE"),
    ("LR_DETECT", "LR_DETECT_DISTANCE"),
    ("RR_DETECT", "RR_DETECT_DISTANCE"),
];
const SOUNDS: [&str; 4] = ["SOUNDS_1", "SOUNDS_2", "SOUNDS_3", "SOUNDS_4"];

macro_rules! optional {
    ($object:expr, $key:literal, $default:expr) => {
        $object
            .getattr_opt(intern!($object.py(), $key))?
            .map_or(Ok($default), |value| value.extract())?
    };
}

fn field(values: &Bound<'_, PyDict>, name: &str) -> PyResult<f64> {
    values
        .get_item(name)?
        .map_or(Ok(0.), |value| value.extract())
}

fn elapsed(last: &mut Option<f64>, now: f64) -> f64 {
    last.replace(now)
        .map_or(0., |previous| (now - previous).clamp(0., 0.2))
}

#[derive(Clone, Copy, Default)]
struct MarkingHead {
    last: Option<bool>,
    held_at: Option<f64>,
    lapsed: bool,
}

impl MarkingHead {
    // Type: true = dashed, false = solid. Colour: true = yellow, false = white.
    fn update(
        &mut self,
        fresh: Option<bool>,
        now: f64,
        slow: bool,
        changing: bool,
    ) -> Option<bool> {
        if let Some(fresh) = fresh {
            self.last = Some(fresh);
            self.held_at = Some(now);
            self.lapsed = false;
        } else if self
            .held_at
            .is_some_and(|held| slow || changing || now - held <= 5.)
        {
            if slow || changing {
                self.held_at = Some(now);
            }
        } else if changing && self.last.is_some() {
            self.held_at = Some(now);
            self.lapsed = false;
        } else if self.held_at.take().is_some() {
            self.lapsed = true;
        }
        self.held_at.and(self.last)
    }
}

#[derive(Clone, Copy, Default)]
struct LaneSide {
    heads: [MarkingHead; 2],
    reset_ns: u64,
    green: Option<bool>,
    green_s: f64,
    low: bool,
    low_s: f64,
    last: Option<f64>,
}

#[derive(Default)]
struct Marking {
    kind: Option<bool>,
    color: Option<bool>,
    reset_ns: u64,
}

impl Marking {
    fn read(packet: Option<&Bound<'_, PyDict>>, side: &str, now: u64) -> PyResult<Self> {
        let Some(packet) = packet else {
            return Ok(Self::default());
        };
        let Some(entry) = packet.get_item(side)?.filter(|value| !value.is_none()) else {
            return Ok(Self::default());
        };
        let entry = entry.cast_into::<PyDict>()?;
        let head = |key: &str, names: [&str; 2]| -> PyResult<Option<bool>> {
            let Some(value) = entry.get_item(key)?.filter(|value| !value.is_none()) else {
                return Ok(None);
            };
            let (name, expiry): (String, i64) = value.extract()?;
            if i64::try_from(now).unwrap_or(i64::MAX) > expiry {
                return Ok(None);
            }
            Ok(names
                .iter()
                .position(|&known| known == name)
                .map(|index| index == 1))
        };
        let reset_ns = entry
            .get_item("displayResetNs")?
            .and_then(|value| value.extract().ok())
            .unwrap_or(0);
        Ok(Self {
            kind: head("type", ["solid", "dashed"])?,
            color: head("color", ["white", "yellow"])?,
            reset_ns,
        })
    }
}

impl LaneSide {
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        now: f64,
        speed: f64,
        available: bool,
        prob: Option<f64>,
        changing: bool,
        marking: Marking,
        lat_active: bool,
    ) -> i64 {
        if marking.reset_ns > self.reset_ns {
            self.reset_ns = marking.reset_ns;
            self.heads = [MarkingHead::default(); 2];
        }
        let dt = elapsed(&mut self.last, now);
        self.green_s = if available { self.green_s + dt } else { 0. };
        if self.green.is_none() || !available || self.green_s >= 1. {
            self.green = Some(available);
        }
        if changing && self.heads[1].last.is_some() {
            self.low = false;
            self.low_s = 0.;
        } else if let Some(prob) = prob.filter(|_| speed >= 0.5 || speed.is_nan()) {
            self.low_s = if prob < 0.5 { self.low_s + dt } else { 0. };
            if self.low_s >= 1.5 {
                self.low = true;
            } else if prob >= 0.5 {
                self.low = false;
            }
        }
        let dash = self.heads[0].update(marking.kind, now, speed < 2., changing);
        let yellow = self.heads[1].update(marking.color, now, speed < 2., changing);
        if self.heads[1].last.is_none() {
            14
        } else if self.low
            || (yellow.is_none() && self.heads[1].lapsed)
            || (yellow == Some(false) && dash.is_none() && self.heads[0].lapsed)
        {
            0
        } else if yellow == Some(true) {
            4
        } else {
            let base = if self.green == Some(true) && lat_active {
                6
            } else {
                2
            };
            if yellow == Some(false) && dash == Some(true) {
                if base == 6 {
                    10
                } else {
                    8
                }
            } else {
                base
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Parking {
    hidden: bool,
    distance: f64,
    exit_distance: f64,
    fast_s: f64,
    last: Option<f64>,
    guard_until: f64,
    no_model_s: f64,
}

impl Default for Parking {
    fn default() -> Self {
        Self {
            hidden: false,
            distance: 0.,
            exit_distance: 0.,
            fast_s: 0.,
            last: None,
            guard_until: -1.,
            no_model_s: 0.,
        }
    }
}

impl Parking {
    fn update(
        &mut self,
        now: f64,
        speed: f64,
        probs: &[f64],
        stds: &[f64],
        changing: bool,
    ) -> bool {
        let dt = elapsed(&mut self.last, now);
        let ([_, left, right, ..], [edge_left, edge_right, ..]) = (probs, stds) else {
            self.no_model_s += dt;
            if self.no_model_s >= 1. {
                self.hidden = false;
                self.distance = 0.;
            }
            return self.hidden;
        };
        self.no_model_s = 0.;
        let probability = py_max(*left, *right);
        if changing {
            self.guard_until = now + 3.;
        }
        if !self.hidden {
            if now < self.guard_until {
                self.distance = 0.;
            } else if speed >= 0.5 {
                let lot =
                    probability < 0.2 && py_min(*edge_left, *edge_right) > 1.5 && speed <= 4.5;
                self.distance = if lot { self.distance + speed * dt } else { 0. };
                if self.distance >= 5. {
                    self.hidden = true;
                    self.exit_distance = 0.;
                    self.fast_s = 0.;
                }
            }
        } else {
            self.fast_s = if speed >= 24. / 3.6 {
                self.fast_s + dt
            } else {
                0.
            };
            if speed >= 0.5 {
                self.exit_distance = if probability >= 0.5 {
                    self.exit_distance + speed * dt
                } else {
                    0.
                };
            }
            if self.exit_distance >= 10. || self.fast_s >= 1. {
                self.hidden = false;
                self.distance = 0.;
            }
        }
        self.hidden
    }
}

#[derive(Clone, Copy, Default)]
struct Traffic {
    shown: i64,
    shown_at: Option<i64>,
    want_since: Option<i64>,
    clear_since: Option<i64>,
    move_since: Option<i64>,
    stopped: bool,
}

impl Traffic {
    fn update(&mut self, frame: i64, stopping: bool, standstill: bool, speed: f64) -> i64 {
        let want = stopping || standstill;
        self.want_since = want.then(|| self.want_since.unwrap_or(frame));
        self.clear_since = (!want).then(|| self.clear_since.unwrap_or(frame));
        self.move_since = (!want && speed > 0.3).then(|| self.move_since.unwrap_or(frame));
        self.stopped |= standstill && self.shown == 1;
        let dwell = self.shown_at.is_none_or(|since| frame - since >= 150);
        let red = dwell && self.want_since.is_some_and(|since| frame - since >= 50);
        let next = match self.shown {
            1 if self.stopped
                && dwell
                && self.move_since.is_some_and(|since| frame - since >= 20) =>
            {
                2
            }
            1 if dwell && self.clear_since.is_some_and(|since| frame - since >= 100) => 0,
            1 => 1,
            _ if red => 1,
            2 if self.shown_at.is_some_and(|since| frame - since >= 250) => 0,
            _ => self.shown,
        };
        if next != self.shown {
            self.shown = next;
            self.shown_at = Some(frame);
            self.stopped = false;
        }
        self.shown
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub(super) struct DisplayState {
    lanes: [LaneSide; 2],
    parking: Parking,
    traffic: Traffic,
    gap: Option<i64>,
    gap_until: i64,
    sla_since: f64,
}

pub(super) type Inputs<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    u64,
    i64,
    i64,
    Option<i64>,
    i64,
    bool,
    (bool, bool, bool),
);
pub(super) type StockOverlay<'py> = ([i64; 2], bool, bool, Bound<'py, PyDict>, bool, bool);

impl DisplayState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_161<'py>(
        &mut self,
        values: &Bound<'py, PyDict>,
        cs: &Bound<'py, PyAny>,
        md: &Bound<'py, PyAny>,
        lines: &GeometryInput,
        inputs: &Inputs<'py>,
        frame: i64,
        desire: i64,
        lead: Option<&StockLead<'py>>,
    ) -> PyResult<StockOverlay<'py>> {
        let (cc, hud, now_ns, lane_check, hdp_use, _, paddle, traffic, _) = inputs;
        let py = cs.py();
        let out = get!(cs, "out");
        let cruise = get!(cc, "enabled").is_truthy()?;
        let lat = get!(cc, "latActive").is_truthy()?;
        let lat_enabled = get!(out, "latEnabled").is_truthy()?;
        let steering_pressed = optional!(out, "steeringPressed", false);
        let standstill = optional!(out, "standstill", false);
        let main = get!(get!(out, "cruiseState"), "available").is_truthy()?;
        let nav = optional!(hud, "activeCarrot", 0_i64) > 1;
        let navi_available = nav || optional!(out, "vehicleNaviAvailable", false);
        let hdp = cruise && (*hdp_use == 2 || (*hdp_use == 1 && nav));
        let trailer = get!(cs, "trailer_connected").is_truthy()?;
        let speed: f64 = get!(out, "vEgo").extract()?;
        let direction = [
            get!(out, "leftBlinker").is_truthy()?,
            get!(out, "rightBlinker").is_truthy()?,
        ];
        let set_speed: f64 = get!(hud, "setSpeed").extract()?;
        let units = if get!(cs, "is_metric").is_truthy()? {
            3.6
        } else {
            3.6 * (1. / 1.609344)
        };
        let set_speed = (set_speed * units + 0.5) as i64;
        let gap: i64 = get!(hud, "leadDistanceBars").extract()?;
        let angle = steering_curve(&out, lat)?;
        let cluster = get!(cs, "lfahda_cluster");
        let center = cluster
            .cast::<PyDict>()
            .ok()
            .map_or(Ok(0.), |v| field(v, "HDA_CntrlModSta"))?
            > 0.;
        for (key, value) in [
            (
                "SETSPEED",
                if !main {
                    0
                } else if hdp {
                    6
                } else if cruise {
                    3
                } else {
                    1
                },
            ),
            (
                "SETSPEED_HUD",
                if !main {
                    0
                } else if hdp {
                    5
                } else if cruise {
                    3
                } else {
                    1
                },
            ),
            ("vSetDis", set_speed),
            ("DISTANCE", gap),
            (
                "DISTANCE_LEAD",
                if lead.is_none() {
                    0
                } else if cruise {
                    2
                } else if main {
                    1
                } else {
                    0
                },
            ),
            (
                "DISTANCE_CAR",
                if hdp {
                    3
                } else if cruise {
                    2
                } else {
                    i64::from(main)
                },
            ),
            (
                "DISTANCE_SPACING",
                if hdp {
                    if gap == 4 {
                        5
                    } else {
                        3
                    }
                } else {
                    i64::from(cruise)
                },
            ),
            ("TARGET", i64::from(lead.is_some() && cruise)),
            ("CENTERLINE", i64::from(center)),
            ("CAR_CIRCLE", if hdp { 2 } else { i64::from(cruise) }),
            (
                "NAV_ICON",
                if !navi_available {
                    0
                } else if cruise {
                    2
                } else {
                    i64::from(main)
                },
            ),
            (
                "HDA_ICON",
                if hdp {
                    5
                } else if cruise {
                    2
                } else {
                    i64::from(main)
                },
            ),
            (
                "LFA_ICON",
                if !lat_enabled {
                    0
                } else if !lat {
                    1
                } else if steering_pressed {
                    3
                } else if hdp {
                    5
                } else {
                    2
                },
            ),
            ("FCA_ALT_ICON", 0),
            ("LANELINE_CURVATURE", angle.0),
            ("LANELINE_CURVATURE_DIRECTION", angle.1),
            ("LCA_LEFT_ARROW", 2 * i64::from(direction[0])),
            ("LCA_RIGHT_ARROW", 2 * i64::from(direction[1])),
        ] {
            values.set_item(key, value)?;
        }
        values.set_item(
            "TARGET_DISTANCE",
            lead.map_or(0., |lead| lead.distance.clamp(0.1, 204.5)),
        )?;
        let traffic = if *traffic {
            self.traffic.update(
                frame,
                optional!(hud, "trafficState", 0_i64) == 1,
                standstill,
                speed,
            )
        } else {
            0
        };
        let background = match traffic {
            1 => 5,
            2 => 8,
            _ if lat && matches!(desire, 1 | 2)
                || *paddle > 0 && optional!(cs, "paddle_button_prev", 0_i64) > 0 =>
            {
                6
            }
            _ if cruise => 1,
            _ if lat => 3,
            _ => 7,
        };
        values.set_item("BACKGROUND", background)?;
        self.speed_limit(values, cs, &out, cruise, set_speed, *now_ns)?;
        self.filter_alerts(values, cs)?;

        // The stock overlay sees filtered stock fields before the gap popup, trailer filter, HDP reminder and DM.
        let alerts = PyDict::new(py);
        for key in ["ALERTS_1", "ALERTS_3", "ALERTS_4", "ALERTS_5", "SOUNDS_4"] {
            alerts.set_item(key, field(values, key)?)?;
        }
        if hdp && self.gap.is_some_and(|previous| gap != previous) {
            self.gap_until = frame + 200;
        }
        self.gap = Some(gap);
        if frame < self.gap_until && (1..=4).contains(&gap) && field(values, "ALERTS_3")? == 0. {
            values.set_item("ALERTS_3", 6 + gap)?;
        }
        if trailer && field(values, "ALERTS_5")? == 6. {
            values.set_item("ALERTS_5", 0)?;
        }

        let mono = frame as f64 * 0.01;
        let meta = super::optional_model_meta(md)?;
        let in_change = meta
            .getattr_opt(intern!(py, "laneChangeState"))?
            .is_some_and(|value| !value.str().is_ok_and(|name| name == "off"));
        let changing = matches!(desire, 3 | 4) || in_change;
        let packet = cs
            .getattr_opt(intern!(py, "ccnc_lane_marking"))?
            .filter(|value| !value.is_none())
            .map(|value| value.cast_into::<PyDict>())
            .transpose()?;
        let mut colors = [0; 2];
        for (index, (side, visible, depart, available, camera)) in [
            (
                "left",
                optional!(hud, "leftLaneVisible", false),
                optional!(hud, "leftLaneDepart", false),
                optional!(meta, "laneChangeAvailableLeft", false),
                optional!(out, "leftLaneLine", 0_i64),
            ),
            (
                "right",
                optional!(hud, "rightLaneVisible", false),
                optional!(hud, "rightLaneDepart", false),
                optional!(meta, "laneChangeAvailableRight", false),
                optional!(out, "rightLaneLine", 0_i64),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let code = if trailer {
                2
            } else {
                let mark = Marking::read(packet.as_ref(), side, *now_ns)?;
                let prob = (lines.probs.len() >= 3).then(|| lines.probs[index + 1]);
                let mut code = self.lanes[index].update(
                    mono,
                    speed,
                    available,
                    prob,
                    changing || direction[0] || direction[1],
                    mark,
                    lat,
                );
                if if *lane_check >= 1 {
                    !matches!(camera.rem_euclid(10), 0 | 5)
                } else {
                    camera.div_euclid(10) == 2
                } {
                    code = 4;
                }
                code
            };
            colors[index] = code;
            let drawn = if !trailer && depart {
                if (frame / 50) % 2 == 0 {
                    4
                } else {
                    1
                }
            } else if visible {
                code
            } else {
                0
            };
            values.set_item(LANE_LINES[index], drawn)?;
        }
        if trailer {
            let lat_icon = lat || (cruise && lat_enabled && standstill);
            for key in ["LCA_LEFT_ICON", "LCA_RIGHT_ICON"] {
                values.set_item(key, i64::from(lat_icon))?;
            }
            for key in ["LANE_LEFT", "LANE_RIGHT"] {
                values.set_item(key, 0)?;
            }
        }
        let stds: Vec<f64> =
            md.getattr_opt(intern!(py, "roadEdgeStds"))?
                .map_or(Ok(Vec::new()), |stds| {
                    stds.try_iter()?
                        .take(2)
                        .map(|value| value?.extract())
                        .collect()
                })?;
        let hidden = self.parking.update(
            mono,
            speed,
            &lines.probs,
            &stds,
            changing || direction[0] != direction[1],
        );
        if hidden {
            for key in LANE_LINES {
                values.set_item(key, 14)?;
            }
        }
        let dm = optional!(hud, "driverMonitoringAlert", 0_i64);
        let (oem_sound, dm_chime) =
            driver_monitoring(values, &get!(cs, "adrv_0x161").cast_into::<PyDict>()?, dm)?;
        Ok((colors, hidden, lead.is_some(), alerts, oem_sound, dm_chime))
    }

    fn speed_limit(
        &mut self,
        values: &Bound<'_, PyDict>,
        cs: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
        cruise: bool,
        set_speed: i64,
        now_ns: u64,
    ) -> PyResult<()> {
        let cluster = optional!(out, "vCruiseCluster", 0_f64);
        if cruise && cluster > set_speed as f64 {
            let now = now_ns as f64 / 1e9;
            if self.sla_since < 1. {
                self.sla_since = now;
            }
            values.set_item("SETSPEED", 2)?;
            values.set_item("SETSPEED_HUD", 2)?;
            values.set_item(
                "SLA_ICON",
                if (now - self.sla_since) % 3.5 < 2. {
                    2
                } else {
                    0
                },
            )?;
        } else {
            self.sla_since = 0.;
            if cruise && field(values, "SLA_ICON")? > 0. {
                let stock = get!(cs, "ccnc_0x162");
                if let Ok(stock) = stock.cast::<PyDict>() {
                    let limit = field(stock, "SPEEDLIMIT")?;
                    values.set_item(
                        "SLA_ICON",
                        if limit > cluster {
                            3
                        } else if limit < cluster {
                            4
                        } else {
                            0
                        },
                    )?;
                }
            }
        }
        Ok(())
    }

    fn filter_alerts(&self, values: &Bound<'_, PyDict>, cs: &Bound<'_, PyAny>) -> PyResult<()> {
        if matches!(
            field(values, "ALERTS_2")? as i64,
            1 | 2 | 5 | 6 | 10 | 21 | 22
        ) {
            values.set_item("ALERTS_2", 0)?;
            values.set_item("DAW_ICON", 0)?;
        }
        if field(values, "ALERTS_1")? == 0. {
            for key in ["SOUNDS_1", "SOUNDS_2", "SOUNDS_4"] {
                values.set_item(key, 0)?;
            }
        }
        if matches!(field(values, "ALERTS_3")? as i64, 3 | 4 | 7..=14 | 17 | 19 | 20 | 26..=28) {
            values.set_item("ALERTS_3", 0)?;
            values.set_item("SOUNDS_3", 0)?;
        }
        let alert5 = field(values, "ALERTS_5")? as i64;
        if matches!(alert5, 1..=5) || (alert5 == 11 && optional!(cs, "softHoldActive", 0_i64) == 0)
        {
            values.set_item("ALERTS_5", 0)?;
        }
        Ok(())
    }
}

fn driver_monitoring(
    values: &Bound<'_, PyDict>,
    stock: &Bound<'_, PyDict>,
    level: i64,
) -> PyResult<(bool, bool)> {
    if !matches!(level, 1..=3) {
        return Ok((false, false));
    }
    let alert = field(stock, "ALERTS_2")? as i64;
    if matches!(alert, 7 | 8 | 9 | 10 | 14 | 21) {
        values.set_item("ALERTS_2", alert)?;
        for key in SOUNDS {
            values.set_item(key, field(stock, key)?)?;
        }
        return Ok((true, false));
    }
    if field(values, "ALERTS_2")? != 0. {
        return Ok((false, false));
    }
    values.set_item("ALERTS_2", if level == 3 { 2 } else { 1 })?;
    let mut all_quiet = true;
    for key in SOUNDS {
        all_quiet &= field(values, key)? == 0.;
    }
    let chime = level == 3 && all_quiet;
    if chime {
        values.set_item("SOUNDS_2", 3)?;
    }
    Ok((false, chime))
}

fn steering_curve(out: &Bound<'_, PyAny>, active: bool) -> PyResult<(i64, i64)> {
    if !active {
        return Ok((0, 0));
    }
    let curve = (get!(out, "steeringAngleDeg").extract::<f64>()? / 3.).round_ties_even();
    let direction = i64::from(curve < 0.);
    Ok((curve.abs().min(15.) as i64 - direction, direction))
}

pub(super) fn draw_1ea(
    values: &Bound<'_, PyDict>,
    cs: &Bound<'_, PyAny>,
    lat: bool,
    desire: i64,
) -> PyResult<()> {
    let out = get!(cs, "out");
    let (mut curve, mut direction) = steering_curve(&out, lat)?;
    if matches!(desire, 1..=4) {
        values.set_item("LANE_CHANGING", desire)?;
        if matches!(desire, 1 | 2) {
            curve = 15;
            direction = desire - 1;
        }
    }
    values.set_item("LANELINE_CURVATURE", curve)?;
    values.set_item("LANELINE_CURVATURE_DIRECTION", direction)?;
    for (detect, distance) in CORNER_OBJECTS {
        if field(values, distance)? != 0. && field(values, detect)? >= 4. {
            values.set_item(detect, 1)?;
        }
    }
    Ok(())
}

pub(super) fn draw_162(
    values: &Bound<'_, PyDict>,
    cs: &Bound<'_, PyAny>,
    flags: (bool, bool, bool),
) -> PyResult<()> {
    let out = get!(cs, "out");
    let (left, right, debug) = flags;
    if (left && !get!(out, "leftBlinker").is_truthy()?)
        || (right && !get!(out, "rightBlinker").is_truthy()?)
    {
        values.set_item("VIBRATE", 1)?;
    }
    let mut filtered = false;
    for key in ["FAULT_LCA", "FAULT_HDA"] {
        if field(values, key)? == 1. {
            values.set_item(key, 0)?;
            filtered = true;
        }
    }
    if debug || (filtered && field(values, "FAULT_DAS")? == 1.) {
        values.set_item("FAULT_DAS", 0)?;
    }
    if debug {
        values.set_item("FAULT_LSS", 0)?;
    }
    Ok(())
}
