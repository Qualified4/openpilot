//! Stationary and approaching side-display memories for the legacy selector.

use std::collections::{HashMap, HashSet, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Observation, Point, TrackId, FRONT_FIELDS, STOP_SPEED};
use crate::pymath::{py_max, py_sum};

#[derive(Clone, Copy)]
struct Pending {
    id: TrackId,
    frame: i64,
    x: f64,
    y: f64,
}

#[derive(Clone, Copy)]
struct Drawing {
    code: i64,
    x: f64,
    y: f64,
}

impl Drawing {
    fn read(values: &Bound<'_, PyDict>, side: usize) -> PyResult<Self> {
        let [x, y, code] = FRONT_FIELDS[side + 1];
        Ok(Self {
            code: values.as_any().get_item(code)?.extract()?,
            x: values.as_any().get_item(x)?.extract()?,
            y: values.as_any().get_item(y)?.extract()?,
        })
    }

    fn write(&self, values: &Bound<'_, PyDict>, side: usize) -> PyResult<()> {
        let [x, y, code] = FRONT_FIELDS[side + 1];
        values.set_item(code, self.code)?;
        values.set_item(x, self.x)?;
        values.set_item(y, self.y)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Hold {
    pub id: TrackId,
    pub birth: i64,
    x: f64,
    y: f64,
    drawing: Drawing,
}

#[derive(Clone, Copy)]
pub(super) struct Approach {
    pub hold: Hold,
    distance: f64,
}

#[derive(Clone, Copy)]
struct Moved {
    id: TrackId,
    birth: i64,
    stamp: i64,
}

struct Motion {
    birth: i64,
    samples: VecDeque<(i64, f64)>,
    blocked: bool,
    settled: (i64, f64, f64),
}

#[derive(Default)]
pub(super) struct Stops {
    last: Option<i64>,
    pub stopped: bool,
    pub approaching: bool,
    pub distance: f64,
    pending: [Option<Pending>; 2],
    pub holds: [Option<Hold>; 2],
    motion: HashMap<TrackId, Motion>,
    approaching_pending: [Option<Pending>; 2],
    pub approaching_holds: [Option<Approach>; 2],
    moving: [Option<Moved>; 2],
}

impl Stops {
    pub fn update(
        &mut self,
        speed: f64,
        acceleration: f64,
        frame: i64,
        usable: bool,
        tracks: &HashMap<TrackId, Observation>,
    ) -> bool {
        let gap = self
            .last
            .is_none_or(|last| !(0..=15).contains(&(frame - last)));
        let stopped = speed.is_finite() && speed.abs() <= 0.3;
        let approaching = !self.stopped
            && speed.is_finite()
            && acceleration.is_finite()
            && speed > 0.3
            && speed <= 10.
            && acceleration < -0.1;
        if gap || !usable || !(stopped || approaching) {
            self.approaching_pending = [None; 2];
            self.approaching_holds = [None; 2];
            self.moving = [None; 2];
        }
        self.approaching = approaching;
        if !gap {
            self.distance += if speed.is_finite() {
                speed.abs() / 3.6 * (frame - self.last.unwrap()) as f64 * 0.01
            } else {
                10.
            };
        }
        if gap || !stopped || !usable {
            self.pending = [None; 2];
            self.holds = [None; 2];
            self.motion.clear();
        }
        self.stopped = stopped;
        self.last = Some(frame);
        if stopped && usable {
            self.update_motion(tracks);
        }
        gap
    }

    fn update_motion(&mut self, tracks: &HashMap<TrackId, Observation>) {
        self.motion
            .retain(|key, value| tracks.get(key).is_some_and(|p| p.birth == value.birth));
        for (
            key,
            Observation {
                birth,
                stamp,
                point,
                ..
            },
        ) in tracks
        {
            if !point.fresh {
                continue;
            }
            let history = self.motion.entry(*key).or_insert_with(|| Motion {
                birth: *birth,
                samples: VecDeque::new(),
                blocked: false,
                settled: (*stamp, point.x, point.y),
            });
            let samples = &mut history.samples;
            if samples.back().is_some_and(|last| last.0 == *stamp) {
                continue;
            }
            samples.push_back((*stamp, point.y));
            while samples.front().is_some_and(|p| stamp - p.0 > 100) {
                samples.pop_front();
            }
            if samples.len() >= 4 && stamp - samples[0].0 >= 50 {
                let steps = samples
                    .iter()
                    .zip(samples.iter().skip(1))
                    .map(|(a, b)| b.1 - a.1);
                let net = samples.back().unwrap().1 - samples[0].1;
                let travel = py_sum(steps.clone().map(f64::abs)).unwrap_or(0.);
                let sign = if net > 0. { 1. } else { -1. };
                let directed = steps.clone().filter(|&step| step * sign >= 0.04).count();
                if net.abs() >= 0.75
                    && net.abs() >= 0.8 * travel
                    && directed >= 3
                    && steps.clone().all(|step| step.abs() <= 0.6 * net.abs())
                {
                    history.blocked = true;
                    history.settled = (*stamp, point.x, point.y);
                }
            }
            let (start, x, y) = history.settled;
            if (point.x - x).abs() > 0.3
                || (point.y - y).abs() > 0.2
                || point.v_lead.abs() > STOP_SPEED
                || point.v.abs() > STOP_SPEED
            {
                history.settled = (*stamp, point.x, point.y);
            } else if history.blocked && stamp - start >= 200 {
                history.blocked = false;
                samples.clear();
                samples.push_back((*stamp, point.y));
            }
            if history.blocked {
                for holds in [&mut self.pending, &mut self.approaching_pending] {
                    for hold in holds {
                        if hold.is_some_and(|p| p.id == *key) {
                            *hold = None;
                        }
                    }
                }
                for side in 0..2 {
                    if self.holds[side].is_some_and(|p| p.id == *key && p.birth == *birth) {
                        self.holds[side] = None;
                    }
                    if self.approaching_holds[side]
                        .is_some_and(|p| p.hold.id == *key && p.hold.birth == *birth)
                    {
                        self.approaching_holds[side] = None;
                    }
                }
            }
        }
    }

    fn blocked(&self, id: TrackId) -> bool {
        self.motion.get(&id).is_some_and(|m| m.blocked)
    }

    pub fn stationary(&self, p: Point, estimated_vy: f64) -> bool {
        p.v_lead.abs() <= STOP_SPEED
            && p.vy.abs() <= STOP_SPEED
            && estimated_vy.abs() <= STOP_SPEED
            && !self.blocked(p.id)
    }

    pub fn has_hold(&self, side: usize) -> bool {
        self.holds[side].is_some() || self.approaching_holds[side].is_some()
    }

    pub fn any_approaching_hold(&self) -> bool {
        self.approaching_holds.iter().any(Option::is_some)
    }

    pub fn matched_hold(&self, side: usize, id: TrackId, birth: i64) -> bool {
        let held = self.holds[side].or_else(|| self.approaching_holds[side].map(|a| a.hold));
        held.is_some_and(|p| p.id == id && p.birth == birth)
    }

    pub fn remove_rejected(
        &mut self,
        rejected: &HashSet<TrackId>,
        tracks: &HashMap<TrackId, Observation>,
    ) {
        let remove = |p: Hold| {
            rejected.contains(&p.id) && tracks.get(&p.id).is_some_and(|o| o.birth == p.birth)
        };
        for side in 0..2 {
            if self.holds[side].is_some_and(remove) {
                self.holds[side] = None;
            }
            if self.approaching_holds[side].is_some_and(|p| remove(p.hold)) {
                self.approaching_holds[side] = None;
            }
            for pending in [&mut self.pending, &mut self.approaching_pending] {
                if pending[side].is_some_and(|p| rejected.contains(&p.id)) {
                    pending[side] = None;
                }
            }
        }
    }

    fn prefer_memory(
        &mut self,
        side: usize,
        lead: Option<Point>,
        held: Hold,
        distance: f64,
        tracks: &HashMap<TrackId, Observation>,
        vy: impl Fn(TrackId) -> f64,
    ) -> bool {
        let Some((lead, current)) = lead.and_then(|p| tracks.get(&p.id).map(|o| (p, o))) else {
            self.moving[side] = None;
            return false;
        };
        if lead.id == held.id && current.birth == held.birth {
            self.moving[side] = None;
            return false;
        }
        if lead.x > distance + 1. {
            self.moving[side] = None;
            return true;
        }
        let visible = self.moving[side].is_some_and(|m| {
            m.id == lead.id
                && m.birth == current.birth
                && (0..=15).contains(&(current.stamp - m.stamp))
        });
        if visible || !self.stationary(lead, vy(lead.id)) {
            self.moving[side] = Some(Moved {
                id: lead.id,
                birth: current.birth,
                stamp: current.stamp,
            });
            false
        } else {
            self.moving[side] = None;
            true
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_approach(
        &mut self,
        values: &Bound<'_, PyDict>,
        side: usize,
        lead: Option<Point>,
        frame: i64,
        tracks: &HashMap<TrackId, Observation>,
        view: &[Point],
        selected: [Option<TrackId>; 3],
        vy: impl Fn(TrackId) -> f64 + Copy,
    ) -> PyResult<()> {
        if let Some(lead) = lead.filter(|p| self.approaching && p.fresh) {
            let distance = lead.x + self.distance;
            let limit = if self.approaching_holds[side].is_some() {
                3. / 3.6
            } else {
                STOP_SPEED
            };
            if lead.v_lead.abs() > limit {
                self.approaching_pending[side] = None;
            } else {
                let pending = self.approaching_pending[side]
                    .filter(|p| {
                        p.id == lead.id
                            && (distance - p.x).abs() <= 1.
                            && (lead.y - p.y).abs() <= 0.75
                    })
                    .unwrap_or(Pending {
                        id: lead.id,
                        frame,
                        x: distance,
                        y: lead.y,
                    });
                self.approaching_pending[side] = Some(pending);
                let observation = tracks.get(&lead.id).unwrap();
                if lead.x <= 15.
                    && observation.stamp - pending.frame >= 50
                    && stable(observation, 50)
                    && self.approaching_holds[side].is_none_or(|p| distance <= p.hold.x + 1.)
                {
                    self.approaching_holds[side] = Some(Approach {
                        hold: Hold {
                            id: lead.id,
                            birth: observation.birth,
                            x: distance,
                            y: lead.y,
                            drawing: Drawing::read(values, side)?,
                        },
                        distance: self.distance,
                    });
                }
            }
        } else {
            self.approaching_pending[side] = None;
        }
        let Some(saved) = self.approaching_holds[side] else {
            return Ok(());
        };
        let held = saved.hold;
        let x = held.x - self.distance;
        let mut invalid = x < -1. || self.distance - saved.distance > 5.;
        for p in view.iter().filter(|p| p.front) {
            let (dx, dy) = ((p.x - x).abs(), (p.y - held.y).abs());
            if p.id == held.id
                && tracks.get(&held.id).is_some_and(|o| o.birth == held.birth)
                && (dx > 1. || dy > 0.75 || p.vy.abs() > STOP_SPEED)
            {
                invalid = true;
            }
            if p.x.is_finite()
                && p.y.is_finite()
                && dx <= 2.
                && dy <= 0.8
                && (!p.v_lead.is_finite()
                    || p.v_lead.abs() > 3. / 3.6
                    || dx > 1.
                    || dy > 0.75
                    || [selected[0], selected[2 - side]].contains(&Some(p.id)))
            {
                invalid = true;
                break;
            }
        }
        if invalid {
            self.approaching_holds[side] = None;
            return Ok(());
        }
        let drawing = Drawing {
            x: py_max(0., x) * 0.8,
            ..held.drawing
        };
        if lead.is_some_and(|p| {
            p.id != held.id || tracks.get(&p.id).is_some_and(|o| o.birth != held.birth)
        }) {
            if self.prefer_memory(side, lead, held, x, tracks, vy) {
                drawing.write(values, side)?;
            }
        } else {
            self.moving[side] = None;
        }
        if self.stopped {
            if self.holds[side].is_none_or(|p| x < p.x) {
                self.holds[side] = Some(Hold { x, drawing, ..held });
            }
            self.approaching_holds[side] = None;
        } else if lead.is_none() {
            drawing.write(values, side)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        values: &Bound<'_, PyDict>,
        leads: [Option<Point>; 2],
        frame: i64,
        tracks: &HashMap<TrackId, Observation>,
        points: &[Point],
        view: &[Point],
        selected: [Option<TrackId>; 3],
        vy: impl Fn(TrackId) -> f64 + Copy,
    ) -> PyResult<()> {
        if self.approaching || self.any_approaching_hold() {
            for (side, lead) in leads.into_iter().enumerate() {
                self.draw_approach(values, side, lead, frame, tracks, view, selected, vy)?;
            }
        }
        if !self.stopped {
            return Ok(());
        }
        for (side, lead) in leads.into_iter().enumerate() {
            if let Some(held) = self.holds[side] {
                let invalid = points.iter().any(|p| {
                    let same =
                        p.id == held.id && tracks.get(&p.id).is_some_and(|o| o.birth == held.birth);
                    let nearby = (p.x - held.x).abs() <= 2. && (p.y - held.y).abs() <= 0.8;
                    (same || nearby)
                        && (p.v_lead.abs() > STOP_SPEED
                            || p.v.abs() > STOP_SPEED
                            || [selected[0], selected[2 - side]].contains(&Some(p.id))
                            || (same && (p.x - held.x).abs() > 1.))
                });
                if invalid {
                    self.holds[side] = None;
                }
            }
            let held = self.holds[side];
            if let Some(p) = lead.filter(|p| p.fresh) {
                let stationary =
                    p.v_lead.abs() <= STOP_SPEED && p.v.abs() <= STOP_SPEED && !self.blocked(p.id);
                if !stationary {
                    self.pending[side] = None;
                    if let Some(old) = held {
                        if self.prefer_memory(side, lead, old, old.x, tracks, vy) {
                            old.drawing.write(values, side)?;
                        }
                    }
                    continue;
                }
                let pending = self.pending[side]
                    .filter(|old| {
                        old.id == p.id && (p.x - old.x).abs() <= 1. && (p.y - old.y).abs() <= 0.75
                    })
                    .unwrap_or(Pending {
                        id: p.id,
                        frame,
                        x: p.x,
                        y: p.y,
                    });
                self.pending[side] = Some(pending);
                let current = tracks.get(&p.id).unwrap();
                if p.x <= 15.
                    && current.stamp - pending.frame >= 50
                    && stable(current, 50)
                    && held.is_none_or(|old| p.x <= old.x + 1.)
                {
                    self.holds[side] = Some(Hold {
                        id: p.id,
                        birth: current.birth,
                        x: p.x,
                        y: p.y,
                        drawing: Drawing::read(values, side)?,
                    });
                }
            } else {
                self.pending[side] = None;
                if let Some(held) = held.filter(|_| lead.is_none()) {
                    held.drawing.write(values, side)?;
                }
            }
            if let Some(held) = self.holds[side] {
                if self.prefer_memory(side, lead, held, held.x, tracks, vy) {
                    held.drawing.write(values, side)?;
                }
            }
        }
        Ok(())
    }
}

fn stable(p: &Observation, time: i64) -> bool {
    p.count >= 3 && p.stamp - p.birth >= time
}
