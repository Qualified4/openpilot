//! Legacy selector. Radar observations and display memory remain local to the
//! CCNC drawing path, including predictions and stationary-structure decisions.

use std::collections::{HashMap, HashSet, VecDeque};

use pyo3::exceptions::PyIndexError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyTuple};

use super::{deadband, enum_name, finish_fields, optional_model_meta, FRONT_FIELDS};
use crate::correct::update_centers;
use crate::display_filter;
use crate::guard;
use crate::live_tracks;
use crate::model_v2::{curve, numpy_interp, valid_curve, Curve, GeometryInput};
use crate::position_correction::{apply, stabilize_slots, ApplyInput};
use crate::pymath::{hypot, py_max, py_min, py_sum};
use crate::rear;

mod stopped;
mod structure;
mod temporal;

use stopped::Stops;
use structure::Structure;
use temporal::Temporal;

type TrackId = i128;
type Xy = (f64, f64);
const STOP_SPEED: f64 = 2. / 3.6;

#[derive(Clone, Copy)]
struct Point {
    id: TrackId,
    source: TrackId,
    x: f64,
    y: f64,
    v: f64,
    v_lead: f64,
    vy: f64,
    front: bool,
    measured: bool,
    fresh: bool,
    continuous: bool,
    stamp: i64,
}

impl Point {
    fn eligible(&self) -> bool {
        self.front
            && self.x >= 1.
            && [self.x, self.y, self.v, self.v_lead]
                .iter()
                .all(|v| v.is_finite())
    }

    fn squared_distance(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    fn from_raw(point: live_tracks::RawPoint, stamp: i64) -> Self {
        let source = i128::from(point.track_id);
        Self {
            id: source,
            source,
            x: f64::from(point.d_rel),
            y: f64::from(point.y_rel),
            v: f64::from(point.v_rel),
            v_lead: f64::from(point.v_lead),
            vy: f64::from(point.yv_rel),
            front: point.source == 0,
            measured: point.measured,
            fresh: true,
            continuous: false,
            stamp,
        }
    }

    fn read(point: &Bound<'_, PyAny>, stamp: i64) -> PyResult<Self> {
        let py = point.py();
        let id = get!(point, "trackId").extract()?;
        Ok(Self {
            id,
            source: id,
            x: get!(point, "dRel").extract()?,
            y: get!(point, "yRel").extract()?,
            v: get!(point, "vRel").extract()?,
            v_lead: get!(point, "vLead").extract()?,
            vy: point
                .getattr_opt(intern!(py, "yvRel"))?
                .map_or(Ok(0.), |v| v.extract())?,
            front: enum_name(point, intern!(py, "radarSource"), "frontRadar")? == "frontRadar",
            measured: point
                .getattr_opt(intern!(py, "measured"))?
                .map_or(Ok(true), |v| v.is_truthy())?,
            fresh: true,
            continuous: false,
            stamp,
        })
    }
}

#[derive(Clone, Copy)]
struct Observation {
    birth: i64,
    stamp: i64,
    count: i64,
    point: Point,
}

#[derive(Clone, Copy, PartialEq)]
enum Reference {
    Lane(bool),
    Center,
    Position,
    Path,
}

impl Reference {
    fn into_tuple(self, py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let (tag, value) = match self {
            Self::Lane(left) => ("lane", left.into_pyobject(py)?.to_owned().into_any()),
            Self::Center => ("lane", "center".into_pyobject(py)?.into_any()),
            Self::Position => ("lane", "position".into_pyobject(py)?.into_any()),
            Self::Path => ("path", false.into_pyobject(py)?.to_owned().into_any()),
        };
        Ok((tag, value).into_pyobject(py)?)
    }

    fn is_lane(self) -> bool {
        self != Self::Path
    }
}

#[derive(Clone, Copy)]
struct Target {
    point: Point,
    y: f64,
    reference: Reference,
}

#[derive(Clone, Copy, PartialEq)]
enum AdmissionStatus {
    Pending,
    Blocked,
    Allowed,
}

#[derive(Clone, Copy)]
struct Motion {
    frame: i64,
    pos: Xy,
    direction: Xy,
}

#[derive(Default)]
struct Window {
    frame: i64,
    x: f64,
    y: f64,
    count: usize,
}

fn update_windows(windows: &mut VecDeque<Window>, frame: i64, (x, y): Xy) {
    if windows.back().is_some_and(|w| frame - w.frame > 35) {
        windows.clear();
    }
    if windows.back().is_none_or(|w| frame - w.frame >= 20) {
        if windows.len() == 4 {
            windows.pop_front();
        }
        windows.push_back(Window {
            frame,
            ..Window::default()
        });
    }
    let w = windows.back_mut().unwrap();
    w.x += x;
    w.y += y;
    w.count += 1;
}

fn completed_windows(windows: &VecDeque<Window>) -> Option<[Xy; 3]> {
    if windows.len() != 4 || windows.iter().take(3).any(|w| w.count < 2) {
        return None;
    }
    Some(std::array::from_fn(|i| {
        let w = &windows[i];
        (w.x / w.count as f64, w.y / w.count as f64)
    }))
}

struct Admission {
    birth: i64,
    frame: i64,
    status: AdmissionStatus,
    samples: usize,
    near: usize,
    moving: Option<Motion>,
    windows: VecDeque<Window>,
    boundary_since: Option<(i64, usize)>,
}

impl Admission {
    fn new(birth: i64) -> Self {
        Self {
            birth,
            frame: -1,
            status: AdmissionStatus::Pending,
            samples: 0,
            near: 0,
            moving: None,
            windows: VecDeque::new(),
            boundary_since: None,
        }
    }
}

#[derive(Clone)]
struct SavedWidth {
    distance: f64,
    inner: Curve,
    outer: Curve,
    edge: Option<Curve>,
}

#[derive(Clone)]
struct Crossing {
    birth: i64,
    frame: i64,
    distance: f64,
    curves: [Curve; 4],
    reference: usize,
}

struct Position {
    birth: i64,
    frame: i64,
    reference: Reference,
    raw: f64,
    input: f64,
    bias: f64,
    blending: bool,
    distance: Py<PyAny>,
    lateral: Py<PyAny>,
}

#[derive(Clone, Copy)]
struct Vision {
    probability: f64,
    x: f64,
    y: f64,
    v: Option<f64>,
}

struct Model {
    geometry: GeometryInput,
    path: Option<Curve>,
    deviation: Vec<f64>,
    vision: Vec<Option<Vision>>,
    yaw: Option<f64>,
    changing: bool,
    present: bool,
}

impl Model {
    fn read(md: &Bound<'_, PyAny>, geometry: GeometryInput) -> PyResult<Self> {
        let py = md.py();
        let position = md
            .getattr_opt(intern!(py, "position"))?
            .filter(|p| !p.is_none());
        let path = position.as_ref().map(curve).transpose()?;
        let deviation = if let Some(p) = position {
            p.getattr_opt(intern!(py, "yStd"))?
                .map_or(Ok(Vec::new()), |v| v.extract())?
        } else {
            Vec::new()
        };
        let mut vision = Vec::new();
        if let Some(leads) = md.getattr_opt(intern!(py, "leadsV3"))? {
            for lead in leads.try_iter()? {
                let lead = lead?;
                let x = get!(lead, "x");
                let y = get!(lead, "y");
                if x.len()? == 0 || y.len()? == 0 {
                    vision.push(None);
                    continue;
                }
                let v = match lead.getattr_opt(intern!(py, "v"))? {
                    Some(v) if v.len()? > 0 => Some(v.get_item(0)?.extract::<f64>()?),
                    _ => None,
                };
                vision.push(Some(Vision {
                    probability: get!(lead, "prob").extract()?,
                    x: x.get_item(0)?.extract::<f64>()? - 1.52,
                    y: y.get_item(0)?.extract()?,
                    v,
                }));
            }
        }
        let mut yaw = None;
        if let Some(rates) = md.getattr_opt(intern!(py, "orientationRate"))? {
            if let Some(z) = rates.getattr_opt(intern!(py, "z"))? {
                if z.len()? > 0 {
                    yaw = Some(z.get_item(0)?.extract()?);
                }
            }
        }
        let meta = optional_model_meta(md)?;
        let changing = enum_name(&meta, intern!(py, "laneChangeState"), "off")? != "off";
        Ok(Self {
            geometry,
            path,
            deviation,
            vision,
            yaw,
            changing,
            present: !md.is_none(),
        })
    }

    fn line(&self, i: usize) -> Option<&Curve> {
        self.geometry.lines.get(i).filter(|p| valid_curve(p))
    }

    fn curves(&self, minimum: f64) -> [Option<&Curve>; 2] {
        std::array::from_fn(|side| {
            let i = side + 1;
            let p = *self.geometry.probs.get(i)?;
            (p.is_finite() && p >= minimum)
                .then(|| self.line(i))
                .flatten()
        })
    }

    fn probabilities(&self) -> [f64; 2] {
        if self.geometry.probs.len() < 3 || self.geometry.lines.len() < 3 {
            return [0.; 2];
        }
        [1, 2].map(|i| {
            if self.line(i).is_some() {
                self.geometry.probs[i]
            } else {
                0.
            }
        })
    }
}

#[derive(Default)]
struct LaneSelector {
    left: Option<bool>,
    since: Option<i64>,
    last: Option<i64>,
}

impl LaneSelector {
    fn update(&mut self, [left, right]: [f64; 2], frame: i64) -> bool {
        if self
            .last
            .is_some_and(|last| !(0..=100).contains(&(frame - last)))
        {
            self.left = None;
            self.since = None;
        }
        self.last = Some(frame);
        let l = if left.is_finite() { left } else { 0. };
        let r = if right.is_finite() { right } else { 0. };
        if py_max(l, r) < 0.1 {
            self.left = None;
            self.since = None;
            return false;
        }
        let prefer_left = l > r;
        let current = if self.left == Some(true) { l } else { r };
        if self.left.is_none() || current < 0.1 {
            self.left = Some(prefer_left);
            self.since = None;
        } else if self.left != Some(prefer_left) && (l - r).abs() >= 0.1 {
            match self.since {
                None => self.since = Some(frame),
                Some(since) if frame - since >= 30 => {
                    self.left = Some(prefer_left);
                    self.since = None;
                }
                _ => {}
            }
        } else {
            self.since = None;
        }
        self.left == Some(true)
    }
}

#[pyclass]
struct Selector {
    raw: Option<Py<PyAny>>,
    raw_points: Vec<Point>,
    raw_generation: u64,
    display_generation: u64,
    usable: bool,
    points: Vec<Point>,
    view_points: Vec<Point>,
    last: i64,
    tracks: HashMap<TrackId, Observation>,
    order: Vec<TrackId>,
    selected: [Option<TrackId>; 3],
    model_reader: Option<Py<PyAny>>,
    model_stamp: Option<u64>,
    lane_data: bool,
    projection: Option<u64>,
    positions: HashMap<TrackId, Position>,
    lateral_grace: HashMap<TrackId, i64>,
    recent: HashMap<TrackId, (i64, i64)>,
    recent_front: HashMap<TrackId, (i64, i64)>,
    lane_sides: HashMap<TrackId, (i64, i64, f64)>,
    boundary: HashMap<TrackId, Admission>,
    rejected: HashSet<TrackId>,
    widths: [Option<SavedWidth>; 2],
    saved_side: [Option<SavedWidth>; 2],
    crossing: HashMap<TrackId, Crossing>,
    crossing_order: Vec<TrackId>,
    temporal: Option<Temporal>,
    lane: LaneSelector,
    lane_ready: bool,
    stop: Stops,
    structure: Structure,
}

impl Selector {
    fn new(corrected: bool) -> Self {
        Self {
            raw: None,
            raw_points: Vec::new(),
            raw_generation: 0,
            display_generation: 0,
            usable: false,
            points: Vec::new(),
            view_points: Vec::new(),
            last: -1,
            tracks: HashMap::new(),
            order: Vec::new(),
            selected: [None; 3],
            model_reader: None,
            model_stamp: None,
            lane_data: false,
            projection: None,
            positions: HashMap::new(),
            lateral_grace: HashMap::new(),
            recent: HashMap::new(),
            recent_front: HashMap::new(),
            lane_sides: HashMap::new(),
            boundary: HashMap::new(),
            rejected: HashSet::new(),
            widths: [None, None],
            saved_side: [None, None],
            crossing: HashMap::new(),
            crossing_order: Vec::new(),
            temporal: corrected.then(Temporal::default),
            lane: LaneSelector::default(),
            lane_ready: true,
            stop: Stops::default(),
            structure: Structure::default(),
        }
    }

    fn stable(&self, id: TrackId, frames: i64) -> bool {
        self.tracks
            .get(&id)
            .is_some_and(|p| p.count >= 3 && p.stamp - p.birth >= frames)
    }

    fn recently_selected(&self, id: TrackId, front_only: bool) -> bool {
        let history = if front_only {
            &self.recent_front
        } else {
            &self.recent
        };
        history
            .get(&id)
            .zip(self.tracks.get(&id))
            .is_some_and(|(&(birth, stamp), current)| {
                birth == current.birth && (0..=30).contains(&(self.last - stamp))
            })
    }

    fn observe(&mut self, cs: &Bound<'_, PyAny>, frame: i64) -> PyResult<()> {
        let py = cs.py();
        let data = get!(cs, "live_tracks");
        let incoming = !data.is_none();
        let fresh = match &self.raw {
            Some(raw) => !raw.bind(py).is(&data),
            None => incoming,
        };
        if fresh {
            self.raw_generation += 1;
            if incoming {
                let raw = cs.getattr_opt(intern!(py, "live_tracks_bytes"))?;
                let bytes = raw
                    .as_ref()
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract::<&[u8]>())
                    .transpose()?;
                self.raw_points = if let Some(points) = bytes.and_then(live_tracks::parse) {
                    points
                        .into_iter()
                        .map(|point| Point::from_raw(point, frame))
                        .collect()
                } else {
                    get!(data, "points")
                        .try_iter()?
                        .map(|p| Point::read(&p?, frame))
                        .collect::<PyResult<_>>()?
                };
                self.raw = Some(data.unbind());
            } else {
                self.raw = None;
                self.raw_points.clear();
            }
        }
        let (usable, changed) = match &mut self.temporal {
            Some(temporal) => {
                let updated = temporal.update(
                    incoming.then_some(self.raw_points.as_slice()),
                    fresh,
                    frame,
                    self.selected,
                    &self.tracks,
                );
                (temporal.usable, updated)
            }
            None => (incoming, fresh),
        };
        let gap = !(0..=15).contains(&(frame - self.last));
        if gap {
            self.lane_ready = true;
        }
        if gap || !usable {
            self.crossing.clear();
            self.crossing_order.clear();
            self.tracks.clear();
            self.order.clear();
            self.points.clear();
            self.selected = [None; 3];
            self.positions.clear();
            self.lateral_grace.clear();
            self.recent.clear();
            self.recent_front.clear();
            self.lane_sides.clear();
            self.boundary.clear();
            self.rejected.clear();
            self.structure.clear();
        }
        self.last = frame;
        self.usable = usable;
        if !usable || (!changed && !gap) {
            return Ok(());
        }
        self.display_generation += 1;
        self.view_points = match &self.temporal {
            Some(temporal) => temporal.view.clone(),
            None => self.raw_points.clone(),
        };
        let mut current = HashMap::new();
        self.order.clear();
        self.points.clear();
        for mut p in self.view_points.iter().copied().filter(Point::eligible) {
            if self.temporal.is_none() {
                p.stamp = frame;
            }
            self.points.push(p);
            let mut observation = Observation {
                birth: frame,
                stamp: frame,
                count: 1,
                point: p,
            };
            if let Some(previous) = self.tracks.get(&p.id) {
                if self.temporal.is_some() && p.continuous {
                    observation.birth = previous.birth;
                    observation.count = previous.count + i64::from(p.fresh);
                    observation.stamp = p.stamp;
                } else if self.temporal.is_none() {
                    let n = frame - previous.stamp;
                    let dt = n as f64 * 0.01;
                    let old = previous.point;
                    if (1..=15).contains(&n) && (p.x - old.x - old.v * dt).abs() <= 1. + 2. * dt {
                        let dy = (p.y - old.y).abs();
                        let usual = dy <= 0.5 + 5. * dt;
                        let grace = !usual
                            && self.selected[1..].contains(&Some(p.id))
                            && previous.count >= 3
                            && previous.stamp - previous.birth >= 15
                            && dy <= 0.75 + 5. * dt
                            && (p.v - old.v).abs() <= 1.
                            && frame - self.lateral_grace.get(&p.id).unwrap_or(&-1000) >= 30;
                        if usual || grace {
                            observation.birth = previous.birth;
                            observation.count = previous.count + 1;
                        }
                        if grace {
                            self.lateral_grace.insert(p.id, frame);
                            self.positions.remove(&p.id);
                        }
                    }
                }
            }
            if current.insert(p.id, observation).is_none() {
                self.order.push(p.id);
            }
        }
        self.lateral_grace
            .retain(|id, stamp| current.get(id).is_some_and(|o| o.birth <= *stamp));
        self.positions
            .retain(|id, p| current.get(id).is_some_and(|o| o.birth == p.birth));
        for memory in [&mut self.recent, &mut self.recent_front] {
            memory.retain(|id, (birth, stamp)| {
                current
                    .get(id)
                    .is_some_and(|o| o.birth == *birth && frame - *stamp <= 30)
            });
        }
        self.lane_sides.retain(|id, (birth, stamp, distance)| {
            current.get(id).is_some_and(|o| o.birth == *birth)
                && (0..=100).contains(&(frame - *stamp))
                && self.stop.distance - *distance <= 15.
        });
        self.tracks = current;
        Ok(())
    }

    fn update_boundary(&mut self, model: &Model) {
        self.boundary
            .retain(|key, saved| self.tracks.get(key).is_some_and(|o| o.birth == saved.birth));
        for id in &self.order {
            let Observation {
                birth,
                stamp,
                point: p,
                ..
            } = self.tracks[id];
            if self.temporal.is_some() && !p.fresh {
                continue;
            }
            let admission = self
                .boundary
                .entry(*id)
                .or_insert_with(|| Admission::new(birth));
            if (admission.status == AdmissionStatus::Allowed && admission.moving.is_some())
                || admission.frame == stamp
            {
                continue;
            }
            admission.frame = stamp;
            let stationary = p.v_lead.abs() <= STOP_SPEED && p.vy.abs() <= STOP_SPEED;
            let mut near = false;
            let mut probability = 0.;
            if stationary {
                let mut best: Option<(f64, f64)> = None;
                for i in 0..model.geometry.lines.len().min(model.geometry.probs.len()) {
                    if let Some(line) = model.line(i).filter(|line| in_domain(line, p.x)) {
                        let candidate = (
                            (p.y + numpy_interp(line, p.x)).abs(),
                            model.geometry.probs[i],
                        );
                        if best.is_none_or(|best| tuple_lt(candidate, best)) {
                            best = Some(candidate);
                        }
                    }
                }
                if let Some((gap, prob)) = best {
                    probability = prob;
                    near = prob > 0.1 && gap <= 0.3;
                }
            }
            if admission.status == AdmissionStatus::Allowed {
                admission.boundary_since = near.then(|| {
                    let (start, count) = admission.boundary_since.unwrap_or((stamp, 0));
                    (start, count + 1)
                });
                continue;
            }
            let position = (self.stop.distance + p.x, p.y);
            let speed = hypot(p.v_lead, p.vy);
            if !speed.is_finite() || speed <= STOP_SPEED {
                admission.moving = None;
                admission.windows.clear();
            } else {
                let direction = (p.v_lead / speed, p.vy / speed);
                let dot = |(x, y): Xy, (dx, dy): Xy| py_sum([x * dx, y * dy]).unwrap_or(0.);
                if admission
                    .moving
                    .is_none_or(|m| dot(direction, m.direction) < 0.5)
                {
                    admission.moving = Some(Motion {
                        frame: stamp,
                        pos: position,
                        direction,
                    });
                    admission.windows.clear();
                }
                let mut consistent = true;
                if speed <= 2.5 && p.vy.abs() <= 0.6 {
                    update_windows(&mut admission.windows, stamp, position);
                    consistent = completed_windows(&admission.windows).is_some_and(|v| {
                        v.windows(2)
                            .all(|w| dot((w[1].0 - w[0].0, w[1].1 - w[0].1), direction) >= 0.06)
                    });
                }
                if admission.moving.is_some_and(|m| {
                    consistent
                        && stamp - m.frame
                            >= if admission.status == AdmissionStatus::Blocked {
                                50
                            } else {
                                15
                            }
                        && dot((position.0 - m.pos.0, position.1 - m.pos.1), m.direction) >= 0.5
                }) {
                    admission.status = AdmissionStatus::Allowed;
                }
            }
            if admission.status == AdmissionStatus::Pending
                && stationary
                && (near || probability >= 0.3)
            {
                admission.samples += 1;
                admission.near += usize::from(near);
                if admission.samples >= 7 && stamp - birth >= 30 {
                    admission.status =
                        if admission.near >= 3 && admission.near * 2 >= admission.samples {
                            AdmissionStatus::Blocked
                        } else {
                            AdmissionStatus::Allowed
                        };
                }
            }
        }
        self.rejected = self
            .boundary
            .iter()
            .filter(|(_, a)| {
                a.status == AdmissionStatus::Blocked
                    || (a.status == AdmissionStatus::Pending && a.near > 0)
            })
            .map(|(id, _)| *id)
            .collect();
        self.stop.remove_rejected(&self.rejected, &self.tracks);
    }
}

fn tuple_lt(a: Xy, b: Xy) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}

fn in_domain(curve: &Curve, x: f64) -> bool {
    curve
        .0
        .first()
        .zip(curve.0.last())
        .is_some_and(|(lo, hi)| *lo <= x && x <= *hi)
}

fn speed_thresholds(v: f64, a: f64) -> [f64; 3] {
    if v.is_nan() {
        return [if a < -3. { -100. } else { v }, v, v];
    }
    let front = if a < -3. || v <= 30. {
        -100.
    } else if v < 40. {
        10. * (v - 30.) - 100.
    } else if v < 100. {
        (20. / 60.) * (v - 40.)
    } else {
        20.
    };
    let side = if v <= 0. {
        2.
    } else if v < 30. {
        (8. / 30.) * v + 2.
    } else if v < 100. {
        (10. / 70.) * (v - 30.) + 10.
    } else {
        20.
    };
    let low = if v <= 10. {
        -1.
    } else if v < 40. {
        (11. / 30.) * (v - 10.) - 1.
    } else {
        10.
    };
    [front, side, low]
}

struct LaneData<'a> {
    inner: [&'a Curve; 2],
    outer: [Option<&'a Curve>; 2],
    edge: [Option<&'a Curve>; 2],
}

impl<'a> LaneData<'a> {
    fn read(model: &'a Model, corrected: bool, minimum: f64) -> PyResult<Self> {
        let g = &model.geometry;
        if g.lines.len() < 4 || g.probs.len() < 4 || g.edges.len() < 2 {
            return Err(PyIndexError::new_err("list index out of range"));
        }
        let inner = [&g.lines[1], &g.lines[2]];
        // The reference interpolates both inner curves even when only one
        // supplied the final selection probability.
        if inner
            .iter()
            .any(|(x, y)| x.is_empty() || x.len() != y.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "interpolation array size mismatch",
            ));
        }
        let outer = [0, 3].map(|i| {
            let p = g.probs[i];
            (if corrected { p >= minimum } else { p > 0.1 })
                .then(|| model.line(i))
                .flatten()
        });
        let edge = [0, 1].map(|i| g.edges.get(i).filter(|p| valid_curve(p)));
        Ok(Self { inner, outer, edge })
    }

    fn side_at(&self, side: usize, x: f64, y: f64, saved: Option<&SavedWidth>) -> [f64; 2] {
        let current = [self.outer[side], self.edge[side]];
        let saved_curves =
            saved.map(|saved| ([Some(&saved.outer), saved.edge.as_ref()], &saved.inner));
        std::array::from_fn(|i| {
            if let Some((Some(curve), inner)) = saved_curves.map(|(c, inner)| (c[i], inner)) {
                if in_domain(curve, x) && in_domain(inner, x) {
                    y + numpy_interp(curve, x) - numpy_interp(inner, x)
                } else {
                    f64::NAN
                }
            } else {
                current[i]
                    .filter(|curve| in_domain(curve, x))
                    .map_or(f64::NAN, |curve| numpy_interp(curve, x))
            }
        })
    }
}

impl Selector {
    fn update_model(&mut self, md: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = md.py();
        let stamp: Option<u64> = md
            .getattr_opt(intern!(py, "timestampEof"))?
            .map_or(Ok(None), |v| v.extract())?;
        if stamp.is_some()
            && stamp == self.model_stamp
            && self
                .model_reader
                .as_ref()
                .is_some_and(|old| old.bind(py).is(md))
        {
            return Ok(());
        }
        self.model_reader = if md.is_none() {
            None
        } else {
            Some(md.clone().unbind())
        };
        self.model_stamp = stamp;
        self.lane_data = false;
        self.projection = None;
        Ok(())
    }

    fn remember_widths(&mut self, lanes: &LaneData<'_>, probabilities: [f64; 2]) {
        if self.lane_data {
            return;
        }
        self.saved_side = [None, None];
        for (side, probability) in probabilities.into_iter().enumerate() {
            if let Some(outer) = lanes.outer[side].filter(|_| probability >= 0.1) {
                self.widths[side] = Some(SavedWidth {
                    distance: self.stop.distance,
                    inner: lanes.inner[side].clone(),
                    outer: outer.clone(),
                    edge: lanes.edge[side].cloned(),
                });
            } else if self.stop.stopped {
                self.saved_side[side] = self.widths[side].clone();
            }
        }
        self.lane_data = true;
    }

    fn remember_lane_sides(&mut self, lanes: &LaneData<'_>, probabilities: [f64; 2]) {
        if self.projection == Some(self.display_generation) {
            return;
        }
        self.projection = Some(self.display_generation);
        if !probabilities.iter().all(|&p| p.is_finite() && p >= 0.6) {
            return;
        }
        for p in &self.points {
            if !p.fresh || !lanes.inner.iter().all(|line| in_domain(line, p.x)) {
                continue;
            }
            let [left, right] = lanes.inner.map(|line| numpy_interp(line, p.x));
            if right <= left {
                continue;
            }
            if p.y > -left + 0.35 || p.y < -right - 0.35 {
                let birth = self.tracks[&p.id].birth;
                self.lane_sides
                    .insert(p.id, (birth, self.last, self.stop.distance));
            } else {
                self.lane_sides.remove(&p.id);
            }
        }
    }

    fn side_entry_width(&self, p: Point, side: usize, lanes: &LaneData<'_>, model: &Model) -> bool {
        let outer_index = if side == 0 { 0 } else { 3 };
        if p.v_lead * 3.6 > 2.
            || model.geometry.probs[outer_index] > 0.1
            || self.recently_selected(p.id, false)
        {
            return true;
        }
        let sign = if side == 0 { -1. } else { 1. };
        let distances = [py_max(0., p.x - 20.), p.x + 20.];
        let (inner, outer, edge) = if let Some(saved) = &self.saved_side[side] {
            (&saved.inner, Some(&saved.outer), saved.edge.as_ref())
        } else {
            let Some(edge) = lanes.edge[side] else {
                return false;
            };
            (lanes.inner[side], None, Some(edge))
        };
        distances.into_iter().all(|x| {
            in_domain(inner, x)
                && [outer, edge].into_iter().flatten().all(|curve| {
                    in_domain(curve, x)
                        && sign * (numpy_interp(curve, x) - numpy_interp(inner, x)) - 0.25 > 1.8
                })
        })
    }

    fn stationary_side(&self, point: Point) -> bool {
        let vy = self
            .temporal
            .as_ref()
            .map_or(0., |t| t.lateral_speed(point.id));
        self.stop.stationary(point, vy)
    }

    fn side_representative(&self, candidates: &[(Target, f64)]) -> Option<Target> {
        let less = |a: &&(Target, f64), b: &&(Target, f64)| {
            let score = |p: Point| p.x.powf(2.) + p.y.powf(2.);
            score(a.0.point)
                .partial_cmp(&score(b.0.point))
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut best = candidates.iter().min_by(less)?;
        if (self.stop.stopped || self.stop.approaching) && self.stationary_side(best.0.point) {
            if let Some(retained) = candidates.iter().find(|c| {
                c.0.point.fresh
                    && self.selected[1..].contains(&Some(c.0.point.id))
                    && self.stationary_side(c.0.point)
            }) {
                best = retained;
            }
        }
        let p = best.0.point;
        let admission = self.boundary.get(&p.id);
        let estimate = admission.and_then(|a| a.boundary_since);
        let Some((since, n)) = estimate else {
            return Some(best.0);
        };
        if admission.unwrap().frame - since < 50 || n < 7 || best.1 > 0.3 || !p.fresh {
            return Some(best.0);
        }
        Some(
            candidates
                .iter()
                .filter(|(target, clearance)| {
                    let c = target.point;
                    c.id != p.id
                        && *clearance >= 0.6
                        && (c.x - p.x).abs() <= 1.5
                        && (c.y - p.y).abs() <= 1.5
                        && c.v_lead.abs() <= STOP_SPEED
                        && c.vy.abs() <= STOP_SPEED
                        && c.fresh
                        && self.stable(c.id, 50)
                })
                .min_by(less)
                .map_or(best.0, |c| c.0),
        )
    }

    fn front_admitted(&self, p: Point, probabilities: [f64; 2], model: &Model) -> bool {
        if p.v_lead.abs() > STOP_SPEED
            || self.recently_selected(p.id, false)
            || py_max(probabilities[0], probabilities[1]) >= 0.5
        {
            return true;
        }
        let Some(vision) = model.vision.first().and_then(|v| *v) else {
            return false;
        };
        let Some(v) = vision.v else { return false };
        if ![vision.probability, vision.x, vision.y, v]
            .iter()
            .all(|v| v.is_finite())
        {
            return false;
        }
        let (weak, strong) = vision_match(p, vision, v, false);
        weak || strong
    }

    fn path_lead(
        &self,
        model: &Model,
        probs: [f64; 2],
        minimum: f64,
        require_vision: bool,
        front_only: bool,
    ) -> Option<Target> {
        let (x, y) = model.path.as_ref()?;
        if x.len() != y.len() {
            return None;
        }
        let stop = x
            .windows(2)
            .position(|p| p[1] <= p[0])
            .map_or(x.len(), |i| i + 1);
        let path: Curve = (x[..stop].to_vec(), y[..stop].to_vec());
        if !valid_curve(&path) {
            return None;
        }
        let limits = py_max(1., x[0])..=py_min(80., x[stop - 1]);
        let vision = model.vision.first().and_then(|v| *v).and_then(|lead| {
            let v = lead.v?;
            [lead.x, lead.y, v]
                .iter()
                .all(|v| v.is_finite())
                .then_some((lead, v))
        });
        let mut best = None;
        let mut score = f64::INFINITY;
        for id in &self.order {
            let observation = self.tracks[id];
            let p = observation.point;
            if self.rejected.contains(id)
                || !self.front_admitted(p, probs, model)
                || !self.stable(p.id, 15)
                || !limits.contains(&p.x)
                || p.v_lead * 3.6 <= minimum
            {
                continue;
            }
            let d = p.squared_distance();
            let retained = self.recently_selected(p.id, true);
            if d >= score || (front_only && !retained) {
                continue;
            }
            let (vision_match, strong) = vision.map_or((false, false), |(lead, v)| {
                let (weak, strong) = vision_match(p, lead, v, retained);
                (weak || strong, strong)
            });
            if require_vision && !vision_match {
                continue;
            }
            let aligned = p.y + (numpy_interp(&path, p.x) - path.1[0]);
            let was_side = self
                .lane_sides
                .get(id)
                .is_some_and(|&(birth, stamp, distance)| {
                    birth == observation.birth
                        && (0..=100).contains(&(self.last - stamp))
                        && self.stop.distance - distance <= 15.
                });
            if (was_side || (self.recently_selected(p.id, false) && !retained))
                && !(vision_match && aligned.abs() <= 1.2)
            {
                continue;
            }
            let corridor = if strong {
                2.
            } else if retained {
                1.8
            } else {
                1.2
            };
            if (aligned.abs() <= corridor && (retained || vision_match))
                || (retained && vision_match && aligned.abs() <= 4.5)
            {
                best = Some(Target {
                    point: p,
                    y: aligned,
                    reference: Reference::Path,
                });
                score = d;
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn bridge_crossing(
        &mut self,
        leads: &mut [Option<Target>; 3],
        model: &Model,
        probabilities: [f64; 2],
        probability: f64,
        reference: Reference,
        lanes: Option<&LaneData<'_>>,
        frame: i64,
        thresholds: [f64; 3],
    ) {
        let chosen: HashSet<_> = leads.iter().filter_map(|p| p.map(|p| p.point.id)).collect();
        let left_reference = usize::from(reference == Reference::Lane(false));
        let order = std::mem::take(&mut self.crossing_order);
        for id in order {
            let saved = &self.crossing[&id];
            let decision = (|| {
                let o = self.tracks.get(&id)?;
                if self.rejected.contains(&id)
                    || o.birth != saved.birth
                    || !(0..=200).contains(&(frame - saved.frame))
                    || self.stop.distance - saved.distance > 30.
                    || probability >= 0.3
                {
                    return None;
                }
                if chosen.contains(&id) {
                    return Some(());
                }
                let p = o.point;
                if !(1. ..=80.).contains(&p.x) || !self.stable(id, 15) {
                    return None;
                }
                let y_on = |index| -numpy_interp(&saved.curves[index], p.x);
                let (left, right) = (y_on(0), y_on(1));
                let slot = if p.y > left {
                    1
                } else if p.y < right {
                    2
                } else {
                    0
                };
                if slot == 0 && !self.front_admitted(p, probabilities, model) {
                    return None;
                }
                let curve = &saved.curves[saved.reference];
                let aligned = p.y + numpy_interp(curve, p.x) - curve.1[0];
                let speed = p.v_lead * 3.6;
                let eligible = if slot == 0 {
                    speed > thresholds[0]
                } else {
                    speed > thresholds[1] || (p.x < 30. && speed > thresholds[2])
                };
                if right >= left || aligned.abs() > 5.4 || !eligible {
                    return None;
                }
                let margin = if slot == 0 { 0.25 } else { 0.95 };
                if !(y_on(3) + margin < p.y && p.y < y_on(2) - margin) || leads[slot].is_some() {
                    return None;
                }
                leads[slot] = Some(Target {
                    point: p,
                    y: aligned,
                    reference,
                });
                Some(())
            })();
            if decision.is_some() {
                self.crossing_order.push(id);
            } else {
                self.crossing.remove(&id);
            }
        }
        if let Some(lanes) = lanes.filter(|_| probability >= 0.3) {
            if let [Some(left), Some(right)] = lanes.edge {
                for p in leads[1..]
                    .iter()
                    .flatten()
                    .map(|p| p.point)
                    .filter(|p| self.stable(p.id, 15))
                    .collect::<Vec<_>>()
                {
                    let crossing = Crossing {
                        birth: self.tracks[&p.id].birth,
                        frame,
                        distance: self.stop.distance,
                        curves: [
                            lanes.inner[0].clone(),
                            lanes.inner[1].clone(),
                            left.clone(),
                            right.clone(),
                        ],
                        reference: left_reference,
                    };
                    if self.crossing.insert(p.id, crossing).is_none() {
                        self.crossing_order.push(p.id);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select<'a>(
        &mut self,
        model: &'a Model,
        minimum: f64,
        corrected: bool,
        input_present: bool,
        frame: i64,
        speed: f64,
        acceleration: f64,
    ) -> PyResult<([Option<Target>; 3], Option<LaneData<'a>>)> {
        let probabilities = model.probabilities();
        let is_left = self.lane.update(probabilities, frame);
        let lane_index = usize::from(!is_left);
        let probability = probabilities[lane_index];
        if probability < 0.1 {
            self.lane_ready = false;
        } else if probability >= 0.3 {
            self.lane_ready = true;
        }
        let reference = Reference::Lane(is_left);
        let limits = speed_thresholds(speed, acceleration);
        let mut leads = [None; 3];
        let mut at_boundary = None;
        let mut lane_data = None;
        if input_present && probability >= 0.1 {
            let lanes = LaneData::read(model, corrected, minimum)?;
            self.remember_widths(&lanes, probabilities);
            self.remember_lane_sides(&lanes, probabilities);
            let mut candidates = [Vec::new(), Vec::new()];
            let mut best = f64::INFINITY;
            for &p in &self.points {
                if self.rejected.contains(&p.id)
                    || (self.temporal.is_some() && !self.stable(p.id, 15))
                {
                    continue;
                }
                let velocity = p.v_lead * 3.6;
                if !(velocity > limits[0]
                    || velocity > limits[1]
                    || (p.x < 30. && velocity > limits[2]))
                {
                    continue;
                }
                let [left, right] = lanes.inner.map(|curve| numpy_interp(curve, p.x));
                let aligned = p.y + ([left, right][lane_index] - lanes.inner[lane_index].1[0]);
                let (l, r) = (-left, -right);
                if !l.is_finite() || !r.is_finite() || r >= l {
                    continue;
                }
                let retain =
                    if (self.stop.stopped || self.stop.approaching) && self.stationary_side(p) {
                        let birth = self.tracks[&p.id].birth;
                        [p.y > l - 0.2, p.y < r + 0.2]
                            .into_iter()
                            .enumerate()
                            .find(|&(side, inside)| {
                                inside
                                    && self.selected[side + 1] == Some(p.id)
                                    && self.stop.matched_hold(side, p.id, birth)
                            })
                            .map(|(side, _)| side)
                    } else {
                        None
                    };
                let target = Target {
                    point: p,
                    y: aligned,
                    reference,
                };
                if self.selected[0] == Some(p.id)
                    && self.recently_selected(p.id, true)
                    && velocity > limits[0]
                    && self.front_admitted(p, probabilities, model)
                    && r - 0.35 <= p.y
                    && p.y <= l + 0.35
                {
                    at_boundary = Some(target);
                }
                if r <= p.y && p.y <= l && retain.is_none() {
                    let score = p.squared_distance();
                    if score < best
                        && velocity > limits[0]
                        && self.front_admitted(p, probabilities, model)
                    {
                        leads[0] = Some(target);
                        best = score;
                    }
                    continue;
                }
                let side = if p.y > l || retain == Some(0) {
                    0
                } else if p.y < r || retain == Some(1) {
                    1
                } else {
                    continue;
                };
                if p.x > 80.
                    || !self.stable(p.id, 15)
                    || aligned.abs() > 5.4
                    || !(velocity > limits[1]
                        || (p.x < 30. && velocity > limits[2] && self.stable(p.id, 50)))
                {
                    continue;
                }
                let inner = [left, right][side];
                let [outer, edge] = lanes.side_at(side, p.x, inner, self.saved_side[side].as_ref());
                let sign = if side == 0 { 1. } else { -1. };
                let mut margin = if velocity > limits[1] && self.stable(p.id, 30) {
                    0.75
                } else {
                    -0.25
                };
                if self.selected[side + 1] == Some(p.id) {
                    margin += 0.3;
                }
                let mut width_limit = f64::INFINITY;
                let mut limit = f64::INFINITY;
                if outer.is_finite() {
                    width_limit = -sign * outer;
                    limit = width_limit + margin;
                }
                if edge.is_finite() {
                    let edge_bound = -sign * edge;
                    width_limit = py_min(width_limit, edge_bound);
                    limit = py_min(limit, edge_bound - 0.95);
                }
                let inner_bound = -sign * inner;
                if width_limit != f64::INFINITY
                    && sign * p.y < limit
                    && width_limit - 0.25 - inner_bound > 1.8
                    && self.side_entry_width(p, side, &lanes, model)
                {
                    candidates[side].push((target, sign * p.y - inner_bound));
                }
            }
            for side in 0..2 {
                leads[side + 1] = self.side_representative(&candidates[side]);
            }
            lane_data = Some(lanes);
        }
        if input_present && (!self.lane_ready || probability < 0.5) {
            let fallback = self.path_lead(
                model,
                probabilities,
                limits[0],
                self.lane_ready,
                self.lane_ready && leads[0].is_some(),
            );
            if let Some(fallback) = fallback.filter(|p| {
                leads[0].is_none_or(|current| {
                    !self.lane_ready
                        || p.point.x.powf(2.) + p.point.y.powf(2.)
                            <= current.point.x.powf(2.) + current.point.y.powf(2.)
                })
            }) {
                leads[0] = Some(fallback);
            } else if !self.lane_ready
                && leads[0].is_some_and(|p| !self.recently_selected(p.point.id, true))
            {
                leads[0] = None;
            }
        }
        if let Some(p) = at_boundary.filter(|p| {
            leads[0].is_none()
                && !leads[1..]
                    .iter()
                    .flatten()
                    .any(|selected| selected.point.id == p.point.id)
        }) {
            leads[0] = Some(p);
        }
        self.bridge_crossing(
            &mut leads,
            model,
            probabilities,
            probability,
            reference,
            lane_data.as_ref(),
            frame,
            limits,
        );
        if let Some(temporal) = &mut self.temporal {
            temporal.select(
                &mut leads,
                frame,
                self.stop.stopped || self.stop.approaching || !self.usable,
                &self.rejected,
            );
        }
        if let Some(target) = leads[0] {
            for current in &mut leads[1..] {
                if current.is_some_and(|p| p.point.id == target.point.id) {
                    *current = None;
                }
            }
        }
        Ok((leads, lane_data))
    }
}

fn vision_match(p: Point, lead: Vision, v: f64, retained: bool) -> (bool, bool) {
    let (dx, dy, dv) = (
        (p.x - lead.x).abs(),
        (p.y + lead.y).abs(),
        (p.v_lead - v).abs(),
    );
    let threshold = if retained { 0.3 } else { 0.5 };
    let minimum = if retained { 8. } else { 6. };
    (
        lead.probability >= threshold && dx <= py_max(minimum, p.x * 0.2) && dy <= 1.5 && dv <= 5.,
        lead.probability >= 0.8
            && p.x <= 30.
            && dx <= py_max(3., p.x * 0.2)
            && dy <= 2.
            && dv <= 3.,
    )
}

impl Selector {
    fn align(&self, target: &mut Target, model: &Model, changing: bool, minimum: f64) {
        if !target.reference.is_lane() {
            return;
        }
        let point = target.point;
        let samples: Vec<_> = model
            .curves(minimum)
            .into_iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let c = c.filter(|curve| {
                    in_domain(curve, 0.) && 0. <= point.x && in_domain(curve, point.x)
                })?;
                let end = numpy_interp(c, point.x);
                Some((
                    i,
                    [
                        numpy_interp(c, 0.),
                        if point.x >= 20. {
                            numpy_interp(c, 20.)
                        } else {
                            end
                        },
                        end,
                    ],
                ))
            })
            .collect();
        if samples.is_empty()
            || (samples.len() == 2
                && !(0..3).all(|i| (2.3..=4.8).contains(&(samples[1].1[i] - samples[0].1[i]))))
        {
            return;
        }
        let correction =
            py_sum(samples.iter().map(|(_, y)| y[2] - y[0])).unwrap_or(0.) / samples.len() as f64;
        if let Some(path) = model.path.as_ref().filter(|c| {
            !changing
                && point.x > 40.
                && valid_curve(c)
                && in_domain(c, 0.)
                && in_domain(c, point.x)
                && model.deviation.len() == c.0.len()
                && model.deviation.iter().all(|x| x.is_finite())
        }) {
            let uncertainty = path
                .0
                .iter()
                .zip(&model.deviation)
                .take_while(|(x, _)| **x <= point.x)
                .map(|(_, p)| *p)
                .fold(f64::NEG_INFINITY, py_max);
            if uncertainty <= 0.8 {
                let bend = numpy_interp(path, point.x) - numpy_interp(path, 0.);
                let threshold = if self
                    .positions
                    .get(&point.id)
                    .is_some_and(|p| p.reference == Reference::Position)
                {
                    0.35
                } else {
                    0.75
                };
                if (bend - correction).abs() > threshold {
                    target.y = point.y + bend;
                    target.reference = Reference::Position;
                    return;
                }
            }
        }
        target.y = point.y + correction;
        target.reference = if samples.len() == 2 {
            Reference::Center
        } else {
            Reference::Lane(samples[0].0 == 0)
        };
    }

    fn filter<'py>(
        &mut self,
        target: Target,
        frame: i64,
        filter_type: &Bound<'py, PyAny>,
    ) -> PyResult<(f64, f64)> {
        let py = filter_type.py();
        let point = target.point;
        let birth = self.tracks.get(&point.id).map_or(frame, |p| p.birth);
        let p = match self
            .positions
            .get_mut(&point.id)
            .filter(|p| p.birth == birth && (0..=15).contains(&(frame - p.frame)))
        {
            Some(p) => {
                let dt = (frame - p.frame) as f64 * 0.01;
                p.lateral.bind(py).setattr(intern!(py, "dt"), dt)?;
                let far = target.reference.is_lane() && point.x > 60.;
                let rate = if far {
                    3. - 1.5 * py_min(1., py_max(0., (point.x - 60.) / 60.))
                } else {
                    3.
                };
                let mut step = dt * rate;
                let geometry = (target.y - point.y) - (p.input - p.raw);
                let changed_reference = p.reference != target.reference;
                if far && !changed_reference {
                    let raw = point.y - p.raw;
                    let geometric = geometry + p.bias;
                    if raw * geometric < 0. {
                        step += py_min(raw.abs(), geometric.abs());
                    }
                }
                if changed_reference {
                    p.bias = p.input + (point.y - p.raw) - target.y;
                    p.blending = false;
                } else if p.blending
                    || (p.bias == 0. && geometry.abs() > if far { step } else { 0.6 })
                {
                    p.bias = p.input + (point.y - p.raw) - target.y;
                    p.blending = true;
                }
                if p.bias != 0. {
                    p.bias = py_max(0., p.bias.abs() - step).copysign(p.bias);
                }
                if p.bias == 0. {
                    p.blending = false;
                }
                p
            }
            None => {
                self.positions.insert(
                    point.id,
                    Position {
                        birth,
                        frame,
                        reference: target.reference,
                        raw: point.y,
                        input: target.y,
                        bias: 0.,
                        blending: false,
                        distance: display_filter::new_position_filter(filter_type, point.x, true)?
                            .unbind(),
                        lateral: display_filter::new_position_filter(filter_type, target.y, false)?
                            .unbind(),
                    },
                );
                self.positions.get_mut(&point.id).unwrap()
            }
        };
        p.frame = frame;
        p.reference = target.reference;
        p.raw = point.y;
        p.input = target.y + p.bias;
        Ok((
            display_filter::position(p.distance.bind(py), point.x, true)?,
            display_filter::position(p.lateral.bind(py), p.input, false)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn correct<'py>(
        &self,
        correction: &Bound<'py, PyAny>,
        observations: &Bound<'py, PyDict>,
        target: Target,
        filtered: f64,
        model: &Model,
        lanes: Option<&LaneData<'_>>,
        minimum: f64,
        slot: usize,
        frame: i64,
    ) -> PyResult<f64> {
        let py = correction.py();
        let p = target.point;
        let Some(position) = self.positions.get(&p.id) else {
            return Ok(filtered);
        };
        let aligned = position.input;
        let key = p.id.into_pyobject(py)?.into_any();
        let Some(observation) = observations.get_item(&key)? else {
            return Ok(filtered);
        };
        let reference = target.reference.into_tuple(py)?;
        let mut reliable = false;
        let mut bounds = None;
        if let Some(lanes) = lanes.filter(|_| target.reference.is_lane()) {
            let line_indices = [[1, 2], [0, 1], [2, 3]][slot];
            if slot == 0 {
                reliable = model
                    .probabilities()
                    .iter()
                    .all(|&p| p.is_finite() && p >= minimum);
            } else {
                reliable = get!(correction, "center_valid")
                    .get_item(slot - 1)?
                    .is_truthy()?
                    && lanes.outer[slot - 1].is_some();
            }
            let saved = get!(correction, "tracks")
                .cast_into::<PyDict>()?
                .get_item(&key)?;
            if let Some(saved) = saved {
                let locked = saved.get_item(intern!(py, "locked"))?.is_truthy()?;
                let saved_slot = saved.get_item(intern!(py, "slot"))?.eq(slot)?;
                let saved_frame: i64 = saved.get_item(intern!(py, "frame"))?.extract()?;
                if locked
                    && saved_slot
                    && (0..=15).contains(&(frame - saved_frame))
                    && saved
                        .get_item(intern!(py, "birth"))?
                        .eq(observation.get_item(0)?)?
                {
                    reliable = line_indices.iter().all(|&i| {
                        model.geometry.probs[i].is_finite() && model.geometry.probs[i] >= minimum
                    });
                }
            }
            let pair = match slot {
                0 => [model.line(1), model.line(2)],
                1 => [lanes.outer[0], model.line(1)],
                _ => [model.line(2), lanes.outer[1]],
            };
            if let [Some(left), Some(right)] = pair {
                if in_domain(left, p.x) && in_domain(right, p.x) {
                    let bound = |c| -numpy_interp(c, p.x) + aligned - p.y;
                    bounds = Some((bound(left), bound(right)));
                }
            }
        }
        apply(
            correction,
            ApplyInput {
                key: &key,
                x: p.x,
                y: p.y,
                fresh: p.fresh,
                aligned,
                filtered,
                slot,
                reference: &reference,
                bounds,
                observation: &observation,
                frame,
                reliable,
                lateral_filter: Some(position.lateral.bind(py)),
            },
        )
    }

    fn structure_suppresses(&self, side: usize, model: &Model, frame: i64, hold: i64) -> bool {
        let Some(id) = self.selected[side + 1] else {
            return false;
        };
        let Some(observation) = self.tracks.get(&id) else {
            return false;
        };
        let p = observation.point;
        if self
            .boundary
            .get(&id)
            .is_some_and(|a| a.status == AdmissionStatus::Allowed && a.moving.is_some())
            || self
                .stop
                .holds
                .iter()
                .flatten()
                .any(|a| a.id == id && a.birth == observation.birth)
            || self
                .stop
                .approaching_holds
                .iter()
                .flatten()
                .any(|a| a.hold.id == id && a.hold.birth == observation.birth)
            || model.vision.iter().flatten().any(|v| {
                v.probability >= 0.8
                    && (p.x - v.x).abs() <= py_max(3., p.x * 0.2)
                    && (p.y + v.y).abs() <= 1.2
            })
        {
            return false;
        }
        self.structure
            .suppresses(side, frame, p, observation.stamp, self.stop.distance, hold)
    }

    fn finish_selection(&mut self, leads: [Option<Target>; 3], frame: i64) {
        self.selected = leads.map(|p| p.map(|p| p.point.id));
        for (slot, p) in leads.into_iter().enumerate() {
            if let Some(p) =
                p.and_then(|p| self.tracks.get(&p.point.id).map(|o| (p.point.id, o.birth)))
            {
                self.recent.insert(p.0, (p.1, frame));
                if slot == 0 {
                    self.recent_front.insert(p.0, (p.1, frame));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw<'py>(
        &mut self,
        values: &Bound<'py, PyDict>,
        tracker: &Bound<'py, PyAny>,
        state: &Bound<'py, PyAny>,
        cs: &Bound<'py, PyAny>,
        md: &Bound<'py, PyAny>,
        model: &Model,
        frame: i64,
        model_lanes: bool,
        filter_type: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let py = state.py();
        let out = get!(cs, "out");
        let ego: f64 = get!(out, "vEgo").extract()?;
        let acceleration: f64 = get!(out, "aEgo").extract()?;
        let speed = ego * 3.6;
        let acceleration = acceleration * 3.6;
        let correction = get!(tracker, "position_correction");
        let corrected = !correction.is_none();
        let (left, right) = (
            get!(out, "leftBlinker").is_truthy()?,
            get!(out, "rightBlinker").is_truthy()?,
        );
        if corrected != self.temporal.is_some() {
            self.temporal = corrected.then(Temporal::default);
            self.raw = None; // The newly selected observation policy needs its own view of the held reader.
        }
        self.observe(cs, frame)?;
        let stopped = self.stop.stopped;
        let gap = self
            .stop
            .update(speed, acceleration, frame, self.usable, &self.tracks);
        if gap {
            self.widths = [None, None];
        }
        let mut invalidated = gap || stopped != self.stop.stopped;
        for width in &mut self.widths {
            if width
                .as_ref()
                .is_some_and(|saved| self.stop.distance - saved.distance >= 10.)
            {
                *width = None;
                invalidated = true;
            }
        }
        if invalidated {
            self.lane_data = false;
            self.projection = None;
        }
        self.update_model(md)?;
        self.update_boundary(model);
        let minimum = if corrected {
            get!(correction, "MIN_LANE_PROBABILITY").extract()?
        } else {
            0.6
        };
        let observations = PyDict::new(py);
        let centers = [
            model.geometry.side_center(0, minimum),
            model.geometry.side_center(1, minimum),
        ];
        if corrected {
            for id in &self.order {
                let o = self.tracks[id];
                observations.set_item(id, (o.birth, o.stamp))?;
            }
            update_centers(
                &correction,
                (
                    centers[0].into_pyobject(py)?.into_any(),
                    centers[1].into_pyobject(py)?.into_any(),
                ),
                frame,
                &observations,
            )?;
        }
        let center_filters = [get!(state, "lf_center"), get!(state, "rf_center")];
        for (i, center) in centers.into_iter().enumerate() {
            if let Some(center) = center {
                display_filter::fixed(&center_filters[i], PyFloat::new(py, center).as_any())?;
            }
        }
        let (leads, lane_data) = self.select(
            model,
            minimum,
            corrected,
            self.raw.is_some(),
            frame,
            speed,
            acceleration,
        )?;
        let changing = left || right || model.changing;
        let mut geometry = [(0., 0.); 3];
        for (slot, selected) in leads.into_iter().enumerate() {
            let Some(mut target) = selected else { continue };
            if corrected {
                self.align(&mut target, model, changing, minimum);
            }
            let (distance, mut lateral) = self.filter(target, frame, filter_type)?;
            if corrected && get!(correction, "ego_follow_active").is_truthy()? {
                let reference = target.reference.into_tuple(py)?;
                get!(correction, "ego_targets")
                    .cast::<PyDict>()?
                    .set_item(target.point.id, (frame, target.y, lateral, reference))?;
            }
            if corrected {
                lateral = self.correct(
                    &correction,
                    &observations,
                    target,
                    lateral,
                    model,
                    lane_data.as_ref(),
                    minimum,
                    slot,
                    frame,
                )?;
            }
            geometry[slot] = (distance, lateral);
        }
        self.finish_selection(leads, frame);
        let [ff, lf, rf] = self.selected;
        let selected = (ff, lf, rf).into_pyobject(py)?.into_any();
        put!(tracker, "selected" => &selected);
        let detectors = [
            get!(state, "ff_detect"),
            get!(state, "lf_detect"),
            get!(state, "rf_detect"),
        ];
        for (slot, [distance, lateral, detect]) in FRONT_FIELDS.into_iter().enumerate() {
            if let Some(p) = leads[slot].map(|p| p.point) {
                let (d, mut y) = geometry[slot];
                values.set_item(distance, d * 0.8)?;
                if slot == 0 {
                    y = -y;
                    if !corrected {
                        y = deadband(y, 0., 0.7, 1.);
                    }
                } else {
                    y = py_min(py_max(y, -5.4), 5.4) * if slot == 1 { 1. } else { -1. };
                    if !corrected {
                        y = deadband(
                            y,
                            display_filter::value(&center_filters[slot - 1])?,
                            0.9,
                            2.,
                        );
                    }
                    y = py_min(py_max(y, 0.), 12.7);
                }
                values.set_item(lateral, y)?;
                if slot == 0 && p.v_lead < 3. {
                    values.set_item(detect, 2)?;
                } else {
                    values.set_item(detect, display_filter::detect(&detectors[slot], p.v)?)?;
                }
            } else {
                values.set_item(detect, 0)?;
            }
        }

        let hold = if self.temporal.is_some() { 30 } else { 15 };
        let reverse = enum_name(&out, intern!(py, "gearShifter"), "unknown")? == "reverse";
        self.structure.update(
            frame,
            model,
            self.stop.distance,
            reverse,
            self.usable,
            self.stop.stopped,
            &self.raw_points,
            self.raw_generation,
            &self.view_points,
            self.display_generation,
            hold,
        );
        let blocked = [0, 1].map(|side| self.structure_suppresses(side, model, frame, hold));
        if self.usable {
            let stationaries =
                [0, 1].map(|side| leads[side + 1].filter(|_| !blocked[side]).map(|t| t.point));
            let vy = |id| self.temporal.as_ref().map_or(0., |t| t.lateral_speed(id));
            self.stop.draw(
                values,
                stationaries,
                frame,
                &self.tracks,
                &self.points,
                &self.view_points,
                self.selected,
                vy,
            )?;
        }
        if corrected {
            let bypass =
                self.stop.stopped || self.stop.approaching || self.stop.any_approaching_hold();
            stabilize_slots(
                &correction,
                values.as_any(),
                &selected,
                &observations,
                frame,
                bypass.into_pyobject(py)?.as_any(),
            )?;
        }
        for (side, blocked) in blocked.into_iter().enumerate() {
            if blocked && !self.stop.has_hold(side) {
                values.set_item(FRONT_FIELDS[side + 1][2], 0)?;
            }
        }
        let identities = self
            .selected
            .map(|id| self.tracks.get(&id?).map(|o| (o.point.source, o.birth)));
        rear::finish_selector(
            values,
            cs,
            md,
            frame,
            model_lanes,
            state,
            tracker,
            identities,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_selector_display<'py>(
    existing: &Bound<'py, PyDict>,
    cs: &Bound<'py, PyAny>,
    md: &Bound<'py, PyAny>,
    frame: i64,
    model_lanes: bool,
    lines: GeometryInput,
    state: &Bound<'py, PyAny>,
    filter_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    guard(|| {
        let py = state.py();
        let tracker = get!(state, "radar_display_tracker");
        let handle = intern!(py, "_rust_display");
        let selector = if let Some(previous) = tracker.getattr_opt(handle)? {
            previous.cast_into::<Selector>()?
        } else {
            let corrected = !get!(tracker, "position_correction").is_none();
            let selector = Bound::new(py, Selector::new(corrected))?;
            tracker.setattr(handle, &selector)?;
            selector
        };
        let model = Model::read(md, lines)?;
        let values = existing.copy()?;
        selector.try_borrow_mut()?.draw(
            &values,
            &tracker,
            state,
            cs,
            md,
            &model,
            frame,
            model_lanes,
            filter_type,
        )?;
        finish_fields(&values)
    })
}
