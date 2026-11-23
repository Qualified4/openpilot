//! Radar identities and reflection-corrected positions for the world map.

mod link;

pub use link::LinkGates;

use crate::pos::common::{decay_reflection, reflection_jump};
use crate::world::World;

use self::link::{handoff, Axes, Decision, Sample};

pub const MAX_VEHICLES: usize = 128;
/// The 4 s the pipeline keeps the tracks a classifier answer is matched against: 81 liveTracks updates at 20 Hz,
/// both ends counted, and a few for a faster radar.
const HISTORY_LEN: usize = 84;
const RETAIN: f64 = 2.;
const RADAR_FRAME_TOLERANCE: f64 = 0.03;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Sure,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kind {
    pub class: u8,
    pub conf: f32,
    pub confirmed_at: f64,
    pub frames: u32,
    /// Taken from another vid of the same vehicle when this vid started: shown even on an `Ambiguous` link.
    pub inherited: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Flags {
    pub motion_confirmed: bool,
    pub stationary: bool,
    pub structure: bool,
    pub proven: bool,
    pub admitted: bool,
}

/// Bounded history with oldest-first iteration. Storage is only needed after the first sample.
#[derive(Debug)]
pub struct History<T, const N: usize> {
    items: Option<Box<[T; N]>>,
    next: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> Default for History<T, N> {
    fn default() -> Self {
        Self {
            items: None,
            next: 0,
            len: 0,
        }
    }
}

impl<T: Copy, const N: usize> History<T, N> {
    fn push(&mut self, item: T)
    where
        T: Default,
    {
        let items = self
            .items
            .get_or_insert_with(|| Box::new([T::default(); N]));
        items[self.next] = item;
        self.next = (self.next + 1) % N;
        self.len = (self.len + 1).min(N);
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = T> + '_ {
        let items = self.items.as_ref().map_or(&[][..], |items| &items[..]);
        (N - self.len..N).map(|i| items[(self.next + i) % N])
    }
}

#[derive(Debug)]
pub struct Vehicle {
    pub vid: u32,
    pub wx: f32,
    pub wy: f32,
    pub vx: f32,
    pub vy: f32,
    pub off: f32,
    /// Absolute lane coordinate, NaN while the map has no coordinate for this car.
    pub u: f32,
    pub ids: History<(f64, u32), 4>,
    pub hist: History<(f64, f32, f32), HISTORY_LEN>,
    pub link: Link,
    pub kind: Option<Kind>,
    pub seen_radar: f64,
    pub seen_vision: f64,
    pub born: f64,
    pub flags: Flags,
    pub radar_id: Option<u32>,
}

impl Vehicle {
    fn new(vid: u32, now: f64, link: Link) -> Self {
        Self {
            vid,
            wx: 0.,
            wy: 0.,
            vx: 0.,
            vy: 0.,
            off: 0.,
            u: f32::NAN,
            ids: History::default(),
            hist: History::default(),
            link,
            kind: None,
            seen_radar: f64::NEG_INFINITY,
            seen_vision: f64::NEG_INFINITY,
            born: now,
            flags: Flags::default(),
            radar_id: None,
        }
    }

    fn observe(
        &mut self,
        now: f64,
        sample: Sample,
        axes: Axes,
        world: &World,
        record_history: bool,
    ) {
        let mut lateral_speed = 0.;
        if self.seen_radar.is_finite() {
            let dt = now - self.seen_radar;
            let (_, step, _) = axes.error(self, sample, dt);
            let mut offset = f64::from(self.off);
            lateral_speed = axes.across(f64::from(self.vx), f64::from(self.vy));
            if reflection_jump(step, dt) {
                offset += step;
            } else {
                let observed = lateral_speed + step / dt;
                lateral_speed += 0.15 * (observed.clamp(-3., 3.) - lateral_speed);
            }
            self.off = decay_reflection(offset, dt) as f32;
        }
        if record_history && self.radar_id != Some(sample.id) {
            self.ids.push((now, sample.id));
        }
        self.radar_id = Some(sample.id);
        self.seen_radar = now;
        (self.wx, self.wy) = (sample.wx as f32, sample.wy as f32);
        (self.vx, self.vy) = (
            (axes.cos * sample.speed - axes.sin * lateral_speed) as f32,
            (axes.sin * sample.speed + axes.cos * lateral_speed) as f32,
        );
        if record_history {
            self.hist.push((now, self.wx, self.wy));
        }

        let ego = world.pose.at(now);
        let (x, y) = ego.to_ego(sample.wx, sample.wy);
        if let Some(u) =
            world.vehicle_lane_coord(now, ego.s + x, ego.to_world(x, y - f64::from(self.off)))
        {
            self.u = if self.u.is_finite() {
                // Filter the vehicle coordinate only. Ego lane movement is
                // subtracted at display time and keeps its own bandwidth.
                self.u + (0.17 * (u as f32 - self.u)).clamp(-0.1, 0.1)
            } else {
                u as f32
            };
        } else {
            self.u = f32::NAN;
        }
    }
}

pub struct Tracker {
    gates: LinkGates,
    record_history: bool,
    cars: Vec<Vehicle>,
    next_vid: u32,
    generation: u32,
    frames: History<f64, HISTORY_LEN>,
    last_radar: f64,
}

impl Default for Tracker {
    fn default() -> Self {
        // Keep the fcamera false-link rate within the gate.
        Self::new(LinkGates::CONSERVATIVE)
    }
}

impl Tracker {
    pub fn new(gates: LinkGates) -> Self {
        Self {
            gates,
            record_history: true,
            cars: Vec::with_capacity(MAX_VEHICLES),
            next_vid: 1,
            generation: 0,
            frames: History::default(),
            last_radar: f64::NEG_INFINITY,
        }
    }

    pub fn vehicles(&self) -> &[Vehicle] {
        &self.cars
    }

    pub fn vehicle_mut(&mut self, vid: u32) -> Option<&mut Vehicle> {
        self.cars.iter_mut().find(|car| car.vid == vid)
    }

    pub(crate) fn vehicles_mut(&mut self) -> &mut [Vehicle] {
        &mut self.cars
    }

    pub fn update(
        &mut self,
        now: f64,
        points: &[(u32, f64, f64, f64)],
        world: &World,
    ) -> &[Vehicle] {
        if self.generation != world.generation() {
            self.generation = world.generation();
            self.cars.clear();
            self.frames = History::default();
            self.last_radar = f64::NEG_INFINITY;
        }
        if !world.accepts_radar() || !now.is_finite() || now <= self.last_radar {
            return &self.cars;
        }
        self.last_radar = now;
        if self.record_history {
            self.frames.push(now);
        }
        self.cars
            .retain(|car| now - car.seen_radar.max(car.seen_vision) <= RETAIN);
        let pose = world.radar_pose(now);
        let (sin, cos) = pose.heading.sin_cos();
        let axes = Axes { cos, sin };
        let samples = || {
            points
                .iter()
                .filter_map(|&point| Sample::from_point(point, pose, world.pose.speed()))
        };

        // Current IDs take precedence. Process them first so the coverage check
        // sees the other cars already reported in this frame.
        let mut split_ids = [None; MAX_VEHICLES];
        for sample in samples() {
            if let Some((index, car)) = self
                .cars
                .iter_mut()
                .enumerate()
                .find(|(_, car)| car.radar_id == Some(sample.id))
            {
                let dt = now - car.seen_radar;
                if dt == 0. {
                    continue;
                }
                if LinkGates::same_id(dt, axes.error(car, sample, dt)) {
                    car.observe(now, sample, axes, world, self.record_history);
                } else {
                    split_ids[index] = car.radar_id;
                    car.radar_id = None;
                }
            }
        }
        for sample in samples() {
            if self
                .cars
                .iter()
                .any(|car| car.radar_id == Some(sample.id) && car.seen_radar == now)
            {
                continue;
            }
            let decision = if split_ids.contains(&Some(sample.id)) {
                Decision::Start
            } else {
                handoff(self.gates, axes, now, sample, &self.cars)
            };
            let link = match decision {
                Decision::Handoff(i) => {
                    self.cars[i].link = Link::Sure;
                    self.cars[i].observe(now, sample, axes, world, self.record_history);
                    continue;
                }
                Decision::Ambiguous => Link::Ambiguous,
                Decision::Start => Link::Sure,
            };
            if self.cars.len() < MAX_VEHICLES {
                let mut car = Vehicle::new(self.next_vid, now, link);
                self.next_vid = self.next_vid.wrapping_add(1).max(1);
                car.observe(now, sample, axes, world, self.record_history);
                self.cars.push(car);
            }
        }
        &self.cars
    }

    /// Look up a radar ID at the frame closest to the supplied sensor time.
    /// The frame history includes empty frames, so a gap cannot return a stale ID.
    pub fn vid_at(&self, t: f64, radar_id: u32) -> Option<u32> {
        let stamp = self
            .frames
            .iter()
            .rev()
            .filter(|stamp| (stamp - t).abs() <= RADAR_FRAME_TOLERANCE)
            .min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs()))?;
        self.cars.iter().find_map(|car| {
            let (_, id) = car.ids.iter().rev().find(|(first, _)| *first <= stamp)?;
            (id == radar_id && car.hist.iter().rev().any(|(t, _, _)| t == stamp)).then_some(car.vid)
        })
    }
}
