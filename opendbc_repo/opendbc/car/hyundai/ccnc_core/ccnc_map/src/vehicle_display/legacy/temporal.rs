//! Continuity and short retention of the legacy selector's display observations.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{Observation, Point, Reference, Target, TrackId};
use crate::pymath::py_sum;

const HOLD: i64 = 30;
const RECONNECT: i64 = 35;

#[derive(Clone, Copy)]
struct Sample {
    frame: i64,
    value: f64,
    speed: f64,
}

fn push<T>(data: &mut VecDeque<T>, value: T, capacity: usize) {
    if data.len() == capacity {
        data.pop_front();
    }
    data.push_back(value);
}

fn coherent(samples: &VecDeque<Sample>, axis: usize) -> bool {
    if samples.len() < 3 || samples.back().unwrap().frame - samples[0].frame < 10 {
        return false;
    }
    let steps = samples
        .iter()
        .zip(samples.iter().skip(1))
        .map(|(a, b)| b.value - a.value);
    let travel = py_sum(steps.clone().map(f64::abs)).unwrap_or(0.);
    let rate = [4., 15., 10.][axis];
    samples.iter().zip(samples.iter().skip(1)).all(|(a, b)| {
        let n = (b.frame - a.frame) as f64;
        let predicted = if axis == 0 { a.speed * n * 0.01 } else { 0. };
        (0. < n && n <= 15.) && (b.value - a.value - predicted).abs() <= 0.15 + rate * n * 0.01
    }) && (travel < 0.2 || py_sum(steps).unwrap_or(0.).abs() >= 0.8 * travel)
}

struct Entry {
    point: Point,
    source: Option<TrackId>,
    frame: i64,
    good: i64,
    selected: Option<i64>,
    confirmed: bool,
    vy: f64,
    pending: [VecDeque<Sample>; 3],
    lateral: VecDeque<Sample>,
}

#[derive(Clone, Copy)]
struct Selection {
    id: TrackId,
    qualified: i64,
    offset: f64,
    reference: Reference,
    pending: Option<(TrackId, i64, i64, i64)>,
}

pub(super) struct Temporal {
    entries: HashMap<TrackId, Entry>,
    order: Vec<TrackId>,
    stamp: i64,
    next_id: TrackId,
    selection: [Option<Selection>; 3],
    reconnections: HashMap<TrackId, (TrackId, i64)>,
    pub view: Vec<Point>,
    pub usable: bool,
}

impl Default for Temporal {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            stamp: -1000,
            next_id: 1 << 32,
            selection: [None; 3],
            reconnections: HashMap::new(),
            view: Vec::new(),
            usable: false,
        }
    }
}

impl Temporal {
    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.reconnections.clear();
        self.selection = [None; 3];
    }

    fn new_track(&mut self, mut raw: Point) -> TrackId {
        if self.entries.contains_key(&raw.id) {
            raw.id = self.next_id;
            self.next_id += 1;
        }
        let frame = raw.stamp;
        let id = raw.id;
        self.entries.insert(
            id,
            Entry {
                point: raw,
                source: Some(raw.source),
                frame,
                good: frame,
                selected: None,
                confirmed: false,
                vy: 0.,
                pending: std::array::from_fn(|_| VecDeque::new()),
                lateral: VecDeque::new(),
            },
        );
        self.order.push(id);
        id
    }

    /// Returns whether the display view changed.
    pub fn update(
        &mut self,
        raw: Option<&[Point]>,
        new_reader: bool,
        frame: i64,
        selected: [Option<TrackId>; 3],
        observations: &HashMap<TrackId, Observation>,
    ) -> bool {
        let Some(raw) = raw else {
            let changed = self.usable;
            self.clear();
            self.view.clear();
            self.usable = false;
            self.stamp = frame;
            return changed;
        };
        if !new_reader && (0..=15).contains(&(frame - self.stamp)) {
            return false;
        }
        if !(1..=15).contains(&(frame - self.stamp)) {
            self.clear();
            if !new_reader {
                let changed = self.usable;
                self.view.clear();
                self.usable = false;
                return changed;
            }
        }
        self.stamp = frame;
        self.usable = true;
        self.entries
            .retain(|_, entry| frame - entry.good <= RECONNECT);
        self.order.retain(|key| self.entries.contains_key(key));
        for (key, entry) in &mut self.entries {
            entry.confirmed |= observations
                .get(key)
                .is_some_and(|p| p.count >= 3 && p.stamp - p.birth >= 15);
            if entry.confirmed && selected.contains(&Some(*key)) {
                entry.selected = Some(frame);
            }
        }

        let mut ordered = Vec::new();
        let mut rows = HashMap::new();
        for &point in raw.iter().filter(|p| p.eligible()) {
            if rows.insert(point.source, point).is_none() {
                ordered.push(point.source);
            }
        }
        let by_source: HashMap<_, _> = self
            .order
            .iter()
            .filter_map(|key| self.entries[key].source.map(|source| (source, *key)))
            .collect();
        let mut matched: HashMap<_, _> = ordered
            .iter()
            .filter_map(|source| by_source.get(source).map(|key| (*source, *key)))
            .collect();
        let mut candidates = Vec::new();
        let mut count = HashMap::new();
        for source in ordered
            .iter()
            .filter(|source| !matched.contains_key(source))
        {
            let point = rows[source];
            let mut options = Vec::new();
            for key in &self.order {
                let entry = &self.entries[key];
                let Some(selected) = entry.selected else {
                    continue;
                };
                if entry
                    .source
                    .is_some_and(|source| rows.contains_key(&source))
                    || !entry.confirmed
                {
                    continue;
                }
                let dt = (frame - entry.frame) as f64 * 0.01;
                let old = entry.point;
                if frame - entry.good <= RECONNECT
                    && frame - selected <= RECONNECT
                    && (point.x - old.x - old.v * dt).abs() <= 1. + 2. * dt
                    && (point.y - old.y - entry.vy * dt).abs() <= 0.5 + 2. * dt
                    && (point.v - old.v).abs() <= 2.
                {
                    options.push(*key);
                    *count.entry(*key).or_insert(0) += 1;
                }
            }
            candidates.push((*source, options));
        }
        let mut pending = HashMap::new();
        let mut reconnected = HashSet::new();
        for (source, options) in candidates {
            if let [key] = options.as_slice() {
                if count[key] != 1 {
                    continue;
                }
                if self
                    .reconnections
                    .get(&source)
                    .is_some_and(|(old, stamp)| old == key && (1..=15).contains(&(frame - stamp)))
                {
                    matched.insert(source, *key);
                    self.entries.get_mut(key).unwrap().source = Some(source);
                    reconnected.insert(source);
                } else {
                    pending.insert(source, (*key, frame));
                }
            }
        }
        self.reconnections = pending;
        let mut view = Vec::with_capacity(ordered.len());
        let mut used = HashSet::new();
        for source in ordered {
            if self.reconnections.contains_key(&source) {
                continue;
            }
            let mut raw = rows[&source];
            raw.stamp = frame;
            let Some(mut key) = matched
                .get(&source)
                .copied()
                .filter(|id| self.entries.contains_key(id))
            else {
                let key = self.new_track(raw);
                view.push(self.entries[&key].point);
                used.insert(key);
                continue;
            };
            let entry = self.entries.get_mut(&key).unwrap();
            let old = entry.point;
            let dt = (frame - entry.frame) as f64 * 0.01;
            let predicted = [old.x + old.v * dt, old.y + entry.vy * dt, old.v];
            let observed = [raw.x, raw.y, raw.v];
            let limits = [0.5 + 2. * dt, 0.35 + 2. * dt, 1. + 5. * dt];
            let mut accepted = [false; 3];
            let mut values = predicted;
            for axis in 0..3 {
                let samples = &mut entry.pending[axis];
                accepted[axis] = (observed[axis] - predicted[axis]).abs() <= limits[axis]
                    || (axis < 2 && reconnected.contains(&source));
                if accepted[axis] {
                    samples.clear();
                } else {
                    push(
                        samples,
                        Sample {
                            frame,
                            value: observed[axis],
                            speed: raw.v,
                        },
                        4,
                    );
                    accepted[axis] = coherent(samples, axis);
                }
                if accepted[axis] {
                    values[axis] = observed[axis];
                }
            }
            let relocated = (raw.x - predicted[0]).abs() > 3. || (raw.y - predicted[1]).abs() > 3.;
            let fresh = accepted[0] && accepted[1];
            let mut result = Point { id: old.id, ..raw };
            if !entry.confirmed || (relocated && fresh) {
                if relocated {
                    entry.source = None;
                    key = self.new_track(raw);
                    result.id = key;
                }
                result.continuous = !relocated && fresh;
                let entry = self.entries.get_mut(&key).unwrap();
                entry.good = frame;
                entry.vy = 0.;
            } else {
                [result.x, result.y, result.v] = values;
                result.v_lead += result.v - raw.v;
                result.fresh = fresh;
                result.continuous = true;
                result.stamp = if fresh { frame } else { old.stamp };
                result.measured &= fresh;
                if fresh {
                    entry.good = frame;
                }
                if accepted[1] {
                    let history = &mut entry.lateral;
                    if accepted[0] && accepted[2] {
                        push(
                            history,
                            Sample {
                                frame,
                                value: raw.y,
                                speed: 0.,
                            },
                            4,
                        );
                    } else {
                        history.clear();
                    }
                    entry.vy = 0.;
                    if coherent(history, 1) {
                        let first = history[0];
                        let last = *history.back().unwrap();
                        let span = (last.frame - first.frame) as f64 * 0.01;
                        let net = last.value - first.value;
                        if history
                            .iter()
                            .zip(history.iter().skip(1))
                            .all(|(a, b)| (b.value - a.value).abs() <= 0.7 * net.abs())
                        {
                            entry.vy = net / span;
                        }
                    }
                }
                if !fresh && frame - entry.good > HOLD {
                    entry.source = None;
                    key = self.new_track(raw);
                    result = self.entries[&key].point;
                }
            }
            let entry = self.entries.get_mut(&key).unwrap();
            entry.point = result;
            entry.frame = frame;
            view.push(result);
            used.insert(result.id);
        }
        for key in &self.order {
            let entry = self.entries.get_mut(key).unwrap();
            if used.contains(key)
                || !entry.confirmed
                || entry.selected.is_none()
                || frame - entry.good > HOLD
            {
                continue;
            }
            let dt = (frame - entry.frame) as f64 * 0.01;
            entry.point.x += entry.point.v * dt;
            entry.point.y += entry.vy * dt;
            entry.point.fresh = false;
            entry.point.measured = false;
            entry.point.continuous = true;
            entry.frame = frame;
            if entry.point.x >= 1. {
                view.push(entry.point);
            }
        }
        self.view = view;
        true
    }

    pub fn lateral_speed(&self, key: TrackId) -> f64 {
        self.entries.get(&key).map_or(0., |entry| entry.vy)
    }

    pub fn select(
        &mut self,
        leads: &mut [Option<Target>; 3],
        frame: i64,
        bypass: bool,
        rejected: &HashSet<TrackId>,
    ) {
        if bypass {
            self.selection = [None; 3];
            return;
        }
        let mut chosen: HashSet<_> = leads
            .iter()
            .filter_map(|lead| lead.map(|lead| lead.point.id))
            .collect();
        let available: HashSet<_> = self.view.iter().map(|p| p.id).collect();
        for (slot, lead) in leads.iter_mut().enumerate() {
            if let Some(saved) = &mut self.selection[slot] {
                if let Some(entry) = self.entries.get(&saved.id) {
                    let old = entry.point;
                    let retain = available.contains(&saved.id)
                        && entry.confirmed
                        && frame - saved.qualified <= HOLD
                        && frame - entry.good <= HOLD
                        && !rejected.contains(&saved.id)
                        && !chosen.contains(&saved.id)
                        && (1. ..=if slot == 0 { 160. } else { 80. }).contains(&old.x)
                        && (slot == 0
                            || (old.y + saved.offset) * if slot == 1 { 1. } else { -1. } > 0.)
                        && lead.is_none_or(|point| point.point.x >= old.x - 2.);
                    if retain {
                        let mut wait = true;
                        if let Some(point) = lead {
                            let stamp = point.point.stamp;
                            let (id, start, last, count) = match saved.pending {
                                Some((key, start, last, n)) if key == point.point.id => {
                                    (key, start, stamp, n + i64::from(last != stamp))
                                }
                                _ => (point.point.id, stamp, stamp, 1),
                            };
                            saved.pending = Some((id, start, last, count));
                            wait = stamp - start < 15 || count < 3;
                        }
                        if wait {
                            *lead = Some(Target {
                                point: old,
                                y: old.y + saved.offset,
                                reference: saved.reference,
                            });
                            chosen.insert(saved.id);
                            continue;
                        }
                    }
                }
            }
            match lead {
                Some(lead) if lead.point.fresh => {
                    self.selection[slot] = Some(Selection {
                        id: lead.point.id,
                        qualified: frame,
                        offset: lead.y - lead.point.y,
                        reference: lead.reference,
                        pending: None,
                    });
                }
                None => self.selection[slot] = None,
                _ => {}
            }
        }
    }
}
