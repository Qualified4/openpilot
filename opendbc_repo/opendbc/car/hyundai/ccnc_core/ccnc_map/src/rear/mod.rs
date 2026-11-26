//! Rear handover and continuous side drawings for the legacy CCNC map.

mod continuity;
pub mod side_flow;

use std::collections::{HashMap, VecDeque};

use crate::pymath::{py_max, py_min, py_sum};
use crate::radar::{Error, MapState, Result};
use crate::slots::{Observation, Slot as WorldSlot};
use crate::vehicles::{Kind, Link, Vehicle, MAX_VEHICLES};
use crate::world::World;
use side_flow::{Flow, Motion};

/// Preserve the number type when a drawing copies an existing CAN field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    pub fn value(self) -> f64 {
        match self {
            Self::Integer(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

impl Default for Number {
    fn default() -> Self {
        Self::Integer(0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Slot {
    pub distance: Option<Number>,
    pub lateral: Option<Number>,
    pub detect: Option<Number>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SideSlots {
    pub front: Slot,
    pub rear: Slot,
    /// The last reset requested for the existing display filter.
    pub reset_distance: Option<f64>,
}

pub type RearSlots = [SideSlots; 2];

const FRONT: [[&str; 3]; 2] = [
    ["LF_DETECT_DISTANCE", "LF_DETECT_LATERAL", "LF_DETECT"],
    ["RF_DETECT_DISTANCE", "RF_DETECT_LATERAL", "RF_DETECT"],
];

impl SideSlots {
    fn distance(&self, side: usize) -> Result<Number> {
        self.front.distance.ok_or(Error::Field(FRONT[side][0]))
    }

    fn lateral(&self, side: usize) -> Result<Number> {
        self.front.lateral.ok_or(Error::Field(FRONT[side][1]))
    }

    fn detect(&self, side: usize) -> Result<Number> {
        self.front.detect.ok_or(Error::Field(FRONT[side][2]))
    }

    fn drawing(&self, side: usize) -> Result<(f64, f64)> {
        Ok((self.distance(side)?.value(), self.lateral(side)?.value()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Identity(pub i128, pub f64);

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub vrel: f64,
    pub vlead: f64,
    pub vehicle: bool,
}

#[derive(Clone, Copy)]
pub enum RadarInput<'a> {
    Unchanged,
    Missing,
    Points(&'a [(i128, Option<Sample>)]),
}

pub struct Input<'a> {
    pub now: f64,
    pub context_changed: bool,
    pub map_context: bool,
    pub changing: bool,
    pub reverse: bool,
    pub can_valid: bool,
    pub ego: f64,
    pub blind: [bool; 2],
    pub lost: [bool; 2],
    pub rear_lateral: [f64; 2],
    pub radar: RadarInput<'a>,
}

#[derive(Clone, Debug)]
pub struct Consts {
    pub scale: f64,
    pub end_distance: f64,
    pub min_gap: f64,
    pub confirm_time: f64,
    pub coast_time: f64,
    pub coast_min_x: f64,
    pub qualified_y_gate: f64,
    pub exit_delay: f64,
    pub exit_accel: f64,
    pub exit_max_speed: f64,
    pub max_time: f64,
    pub alias_gap: f64,
    pub alias_y: f64,
    pub alias_vrel: f64,
    pub lane_half: f64,
    pub tracks_fresh: f64,
    pub stale_hold: f64,
    pub ego_span: f64,
    pub ratio_max: f64,
    pub anchor_min_ego: f64,
    pub vrel_n: usize,
    pub x_gate: f64,
    pub vrel_gate: f64,
    pub moving_fraction: f64,
    pub moving_slack: f64,
    pub moving_slack_map: f64,
    pub moving_vlead: f64,
    pub moving_n: usize,
    pub exit_forced_speed: f64,
    pub exit_grace: f64,
    pub smooth: Option<[side_flow::Consts; 2]>,
}

impl Default for Consts {
    fn default() -> Self {
        Self {
            scale: 0.8,
            end_distance: 16.,
            min_gap: 6.,
            confirm_time: 0.3,
            coast_time: 1.,
            coast_min_x: -2.,
            qualified_y_gate: 1.,
            exit_delay: 0.8,
            exit_accel: 1.2,
            exit_max_speed: 8.,
            max_time: 6.,
            alias_gap: 2.,
            alias_y: 1.,
            alias_vrel: 2.,
            lane_half: 1.75,
            tracks_fresh: 0.15,
            stale_hold: 0.3,
            ego_span: 8.,
            ratio_max: 1.5,
            anchor_min_ego: 0.5,
            vrel_n: 7,
            x_gate: 1.,
            vrel_gate: 8.,
            moving_fraction: 0.15,
            moving_slack: 1.5,
            moving_slack_map: 0.5,
            moving_vlead: 1.,
            moving_n: 40,
            exit_forced_speed: 4.,
            exit_grace: 4.,
            smooth: Some(std::array::from_fn(|_| side_flow::Consts::default())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Front,
    Coast,
    Rear,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::Coast => "coast",
            Self::Rear => "rear",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    pub serial: u64,
    pub key: Identity,
    pub first: f64,
    pub first_x: f64,
    pub hits: i64,
    pub moving_hits: i64,
    pub phase: Phase,
    pub qualified: bool,
    pub aliases: Vec<Identity>,
    pub vrel_hist: Vec<f64>,
    pub vlead_hist: Vec<f64>,
    pub vlead_long: Vec<f64>,
    pub x: f64,
    pub y: f64,
    pub vrel: f64,
    pub vrel_raw: f64,
    pub vlead: f64,
    pub anchor: f64,
    pub seen: f64,
    pub updated: f64,
    pub distance: f64,
    pub lateral: f64,
    pub drawn_distance: f64,
    pub drawn_stamp: f64,
    pub drawn_lateral: f64,
    pub detect: i64,
    pub radar_moving: i64,
    pub signed: f64,
    pub rate: f64,
    pub shown: f64,
    pub start: Option<f64>,
    pub speed: f64,
    pub coast_y: f64,
    pub forced: Option<f64>,
}

impl Track {
    pub fn new(key: Identity, serial: u64, stamp: f64, x: f64) -> Self {
        Self {
            serial,
            key,
            first: stamp,
            first_x: x,
            hits: 0,
            moving_hits: 0,
            phase: Phase::Front,
            qualified: false,
            aliases: Vec::new(),
            vrel_hist: Vec::new(),
            vlead_hist: Vec::new(),
            vlead_long: Vec::new(),
            x,
            y: 0.,
            vrel: 0.,
            vrel_raw: 0.,
            vlead: 0.,
            anchor: 0.,
            seen: stamp,
            updated: stamp,
            distance: 0.,
            lateral: 0.,
            drawn_distance: 0.,
            drawn_stamp: stamp,
            drawn_lateral: 0.,
            detect: 0,
            radar_moving: 0,
            signed: 0.,
            rate: 0.,
            shown: 0.,
            start: None,
            speed: 0.,
            coast_y: 0.,
            forced: None,
        }
    }

    fn owns(&self, key: Identity) -> bool {
        key == self.key || self.aliases.contains(&key)
    }

    fn motion(&self, k: &Consts, end_reason: Option<&'static str>) -> Motion {
        Motion {
            serial: self.serial,
            key: self.key,
            aliases: self.aliases.clone(),
            phase: self.phase,
            signed: self.signed,
            speed: self.speed,
            distance: self.distance,
            force_at: self.shown + k.max_time,
            deadline: self.shown + k.max_time + k.exit_grace,
            max_speed: k.exit_max_speed,
            end_reason,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Drop {
        side: usize,
        reason: &'static str,
        source: i128,
        phase: Phase,
        distance: f64,
        age: f64,
        qualified: bool,
        hits: i64,
    },
    Coast {
        side: usize,
        reason: &'static str,
        source: i128,
        x: f64,
        vrel: f64,
        vlead: f64,
        anchor: f64,
        ego: f64,
        front_distance: f64,
    },
    Handoff {
        side: usize,
        key: Identity,
        distance: f64,
        seen_age: f64,
        speed: f64,
        lateral: f64,
        coast_time: f64,
        origin: bool,
    },
    Forced {
        side: usize,
        source: i128,
        distance: f64,
    },
    Flow {
        side: usize,
        name: &'static str,
    },
}

#[derive(Clone, Debug, Default)]
pub struct SideState {
    pub front: Option<Track>,
    pub rear: Option<Track>,
    pub consumed: VecDeque<Identity>,
    pub unproven: bool,
    pub flow: Flow,
}

#[derive(Clone, Debug)]
pub struct RearState {
    pub sides: [SideState; 2],
    pub last: Option<f64>,
    pub raw_stamp: f64,
    pub fresh: bool,
    pub changing: bool,
    pub ego: f64,
    pub ego_hist: VecDeque<(f64, f64, f64)>,
    pub events: Vec<Event>,
    points: HashMap<i128, Sample>,
    raw_present: bool,
    next_serial: u64,
}

impl Default for RearState {
    fn default() -> Self {
        Self {
            sides: Default::default(),
            last: None,
            raw_stamp: f64::NEG_INFINITY,
            fresh: false,
            changing: false,
            ego: 0.,
            ego_hist: VecDeque::new(),
            events: Vec::new(),
            points: HashMap::new(),
            raw_present: false,
            next_serial: 1,
        }
    }
}

/// Compute the existing BSD display targets before applying the rear handover.
pub fn bsd_target(active: bool, distance: f64) -> Option<i64> {
    (active || distance < 15.).then_some(if active { 8 } else { 16 })
}

pub fn bsd_laterals(left_filter: f64, right_filter: f64) -> [f64; 2] {
    let offset = (right_filter - left_filter) / 2.;
    [2.8 - offset, 2.8 + offset]
}

pub fn finish_map(
    map: &MapState,
    rear: &mut RearState,
    k: &Consts,
    input: Input<'_>,
    slots: &mut RearSlots,
    crossing_travel: f64,
) -> Result<()> {
    let objects = map.selected.map(|key| key.and_then(|key| map.objects.get(&key)));
    let selected = objects.map(|o| o.map(|o| Identity(o.source, o.born)));
    let proven = [1, 2].map(|i| objects[i].is_some_and(|o| map.vehicle_proven(o, crossing_travel)));
    rear.update(k, input, selected, proven, slots)
}

#[derive(Clone, Copy)]
pub struct WorldInput<'a> {
    pub now: f64,
    pub changing: bool,
    pub can_valid: bool,
    pub blind: [bool; 2],
    pub vehicles: &'a [Vehicle],
}

#[derive(Default)]
pub struct WorldRear {
    state: RearState,
    constants: Consts,
    generation: u32,
    radar_stamp: Option<f64>,
    /// Confirmed kinds by vid. The tracker forgets a car 2 s after its last radar update; the rear handover keeps
    /// drawing it for up to 10 s, with this kind.
    kinds: Vec<(u32, Kind)>,
    alongside: [Option<continuity::Alongside>; 2],
    joins: [continuity::DisplayJoin; 2],
    side_radar_ids: HashMap<u32, (u32, f64)>,
}

impl WorldRear {
    /// Continue side vehicles through zero and apply the existing BSD/exit policy.
    /// The map's front drawings already include Display A and must keep that lateral.
    pub fn finish(
        &mut self,
        world: &World,
        input: WorldInput<'_>,
        slots: &mut [Option<WorldSlot>; 5],
    ) -> Result<()> {
        let context_changed = self.generation != world.generation();
        self.generation = world.generation();
        let now = input.now;
        let ego = world.pose.at(now);
        let width = world.lanes.width_at(ego.s);
        let lateral = bsd_laterals(width, width);
        if context_changed {
            self.kinds.clear();
            self.alongside = [None; 2];
            self.joins = Default::default();
            self.side_radar_ids.clear();
        }
        for car in input.vehicles {
            if let Some(kind) = car
                .kind
                .filter(|k| (car.link == Link::Sure || k.inherited) && k.confirmed_at.is_finite())
            {
                match self.kinds.iter_mut().find(|(vid, _)| *vid == car.vid) {
                    Some(kept) => kept.1 = kind,
                    None => self.kinds.push((car.vid, kind)),
                }
            }
        }
        let to_car = |vid| input.vehicles.iter().find(|car| car.vid == vid);
        let front = [slots[0], slots[1], slots[2]].map(|slot| slot.and_then(|s| to_car(s.vid)));
        let selected = front.map(|car| car.map(|car| Identity(i128::from(car.vid), car.born)));
        let proven = [1, 2].map(|i| front[i].is_some_and(|car| car.flags.proven));
        let mut samples = [(0, None); MAX_VEHICLES];
        let mut count = 0;
        let stamp = world.radar_stamp();
        let radar = match stamp {
            None => RadarInput::Missing,
            Some(stamp) if context_changed || self.radar_stamp != Some(stamp) => {
                for car in input
                    .vehicles
                    .iter()
                    .filter(|car| car.seen_radar == stamp)
                    .take(MAX_VEHICLES)
                {
                    if let Some(Observation { x, r, speed, .. }) =
                        Observation::from_car(car, world, world.radar_pose(now))
                    {
                        // Convert r to the existing policy's metre gates with a fixed lane width.
                        samples[count] = (
                            i128::from(car.vid),
                            Some(Sample {
                                x,
                                y: -r * crate::pos::common::LANE_WIDTH,
                                vrel: speed - world.pose.speed(),
                                vlead: speed,
                                vehicle: car.flags.proven,
                            }),
                        );
                        count += 1;
                    }
                }
                RadarInput::Points(&samples[..count])
            }
            Some(_) => RadarInput::Unchanged,
        };
        self.radar_stamp = stamp;
        let mut sides = RearSlots::default();
        for (i, side) in sides.iter_mut().enumerate() {
            let (distance, y, detect) = slots[i + 1].map_or((0., 0., 0), |s| {
                (
                    s.distance as f64 * self.constants.scale,
                    f64::from(s.lateral.abs()),
                    i64::from(s.detect),
                )
            });
            side.front = Slot {
                distance: Some(Number::Float(distance)),
                lateral: Some(Number::Float(y)),
                detect: Some(Number::Integer(detect)),
            };
        }
        self.state.update(
            &self.constants,
            Input {
                now,
                context_changed,
                map_context: true,
                changing: input.changing,
                reverse: !world.accepts_radar(),
                can_valid: input.can_valid,
                ego: world.pose.speed(),
                blind: input.blind,
                lost: [false; 2],
                rear_lateral: lateral,
                radar,
            },
            selected,
            proven,
            &mut sides,
        )?;

        for (i, side) in self.state.sides.iter().enumerate() {
            // Preserve measured rear slots. A live BSD signal supplies its own drawing.
            if slots[i + 3].is_some() && !input.blind[i] {
                continue;
            }
            let Some(drawn) = side
                .flow
                .rear
                .as_ref()
                .filter(|_| input.can_valid && world.accepts_radar())
            else {
                continue;
            };
            let vid = match drawn.key {
                Some(side_flow::Key::Radar(Identity(vid, _))) => vid as u32,
                _ => 0,
            };
            if vid != 0 {
                if slots.iter().enumerate().any(|(index, s)| {
                    index != i + 1 && index != i + 3 && s.is_some_and(|s| s.vid == vid)
                }) {
                    continue;
                }
                if slots[i + 1].is_some_and(|s| s.vid == vid) {
                    if sides[i]
                        .front
                        .detect
                        .is_some_and(|detect| detect.value() != 0.)
                    {
                        continue;
                    }
                    slots[i + 1] = None;
                }
            }
            let y = if i == 0 {
                drawn.y.value()
            } else {
                -drawn.y.value()
            };
            let kind = match to_car(vid) {
                Some(car) => car.kind.filter(|k| car.link == Link::Sure || k.inherited),
                None => self.kinds.iter().find(|(kept, _)| *kept == vid).map(|&(_, kind)| kind),
            };
            slots[i + 3] = Some(WorldSlot {
                vid,
                distance: (drawn.x.value() / self.constants.scale) as f32,
                lateral: y as f32,
                detect: drawn.detect.value() as u8,
                lane_coord_rel: (-y / width) as f32,
                v_rel: side
                    .rear
                    .as_ref()
                    .filter(|t| t.key.0 == i128::from(vid))
                    .map_or(0., |t| t.vrel as f32),
                kind: kind.map_or(0, |k| k.class),
                kind_confirmed: kind.is_some_and(|k| k.confirmed_at.is_finite()),
                kind_wanted: false,
                vision: crate::slots::VISION_NONE,
            });
        }
        let sides = &self.state.sides;
        self.kinds.retain(|&(vid, _)| {
            let key = i128::from(vid);
            input.vehicles.iter().any(|car| car.vid == vid)
                || sides.iter().any(|side| {
                    [&side.front, &side.rear].into_iter().flatten().any(|track| track.key.0 == key)
                        || side.flow.rear.as_ref().is_some_and(|drawn| {
                            matches!(drawn.key, Some(side_flow::Key::Radar(Identity(v, _))) if v == key)
                        })
                })
        });
        Ok(())
    }
    pub(crate) fn continue_sides(
        &mut self,
        world: &World,
        input: WorldInput<'_>,
        remembered: [bool; 2],
        selected: &[Option<WorldSlot>; 5],
        slots: &mut [Option<WorldSlot>; 5],
    ) -> [Option<usize>; 2] {
        if !world.accepts_radar() || !input.can_valid {
            self.alongside = [None; 2];
            self.joins = Default::default();
            self.side_radar_ids.clear();
            return [None; 2];
        }
        let mut continuous = std::array::from_fn(|side| {
            continuity::Alongside::update(
                &mut self.alongside[side], side, input.now, world.pose.speed(), input.blind[side],
                remembered[side], selected, input.vehicles, slots,
            )
        });
        // Keep the originating radar ID through the longest rear animation, including after a VID
        // leaves the tracker. This prevents a retired slot drawing from coming back after one empty tick.
        self.side_radar_ids.retain(|vid, (_, seen)| {
            input.now - *seen <= 10. || slots.iter().flatten().any(|slot| slot.vid == *vid)
        });
        self.side_radar_ids.extend(input.vehicles.iter().filter_map(|car| {
            Some((car.vid, (car.radar_id.or(car.flags.source)?, car.seen_radar)))
        }));
        for side in 0..2 {
            let (front, rear) = (side + 1, side + 3);
            let radar = |index: usize| slots[index].and_then(|s| self.side_radar_ids.get(&s.vid).copied());
            if let Some(((ft, fs), (rt, rs))) = radar(front).zip(radar(rear)) {
                if ft == rt {
                    // Prefer the latest observation of this radar ID over an older drawing.
                    let drop = if fs >= rs { rear } else { front };
                    let retired = slots[drop].take().unwrap().vid;
                    if self.alongside[side].is_some_and(|motion| motion.vid() == retired) {
                        self.alongside[side] = None;
                    }
                    let state = &mut self.state.sides[side];
                    let owns = |key: Identity| key.0 == i128::from(retired);
                    if state.front.as_ref().is_some_and(|track| owns(track.key)) {
                        state.front = None;
                    }
                    if state.rear.as_ref().is_some_and(|track| owns(track.key)) {
                        state.rear = None;
                    }
                    if state.flow.rear.as_ref().is_some_and(|drawn| {
                        matches!(drawn.key, Some(side_flow::Key::Radar(key)) if owns(key))
                    }) {
                        state.flow.rear = None;
                    }
                    if continuous[side] == Some(drop) {
                        continuous[side] = None;
                    }
                }
            }
        }
        for (side, active) in continuous.iter_mut().enumerate() {
            if let Some(joined) = self.joins[side].update(input.now, side, slots) {
                *active = Some(joined);
            }
        }
        continuous
    }

}

impl RearState {
    pub fn update(
        &mut self,
        k: &Consts,
        input: Input<'_>,
        selected: [Option<Identity>; 3],
        proven: [bool; 2],
        slots: &mut RearSlots,
    ) -> Result<()> {
        let now = input.now;
        let dt = self.last.map_or(0., |last| now - last);
        let clock_reset = input.context_changed || !(0. < dt && dt <= 0.2);
        let blocked = if clock_reset {
            Some("clock_or_context")
        } else if input.reverse {
            Some("reverse")
        } else if input.changing {
            Some("lane_change")
        } else {
            None
        };
        self.events.clear();
        self.changing = input.changing;
        self.last = Some(now);
        self.ego = if input.ego.is_finite() { input.ego } else { 0. };
        if clock_reset {
            self.ego_hist.clear();
        }
        self.ego_hist
            .push_back((now, self.ego, if clock_reset { 0. } else { dt }));
        while self.ego_hist.front().is_some_and(|(t, _, _)| *t < now - k.ego_span) {
            self.ego_hist.pop_front();
        }
        match input.radar {
            RadarInput::Unchanged => {}
            RadarInput::Missing => {
                self.raw_present = false;
                self.raw_stamp = now;
                self.points.clear();
            }
            RadarInput::Points(points) => {
                self.raw_present = true;
                self.raw_stamp = now;
                self.points.clear();
                for &(id, point) in points {
                    if let Some(point) = point {
                        self.points.insert(id, point);
                    } else {
                        self.points.remove(&id);
                    }
                }
            }
        }
        self.fresh = self.raw_present && now - self.raw_stamp <= k.tracks_fresh;

        let mut retired: [Option<Motion>; 2] = [None, None];
        for (i, slot) in slots.iter_mut().enumerate() {
            let mut side = Side {
                state: &mut self.sides[i],
                slots: slot,
                k,
                side: i,
                now,
                ego: self.ego,
                hist: &self.ego_hist,
                raw_stamp: self.raw_stamp,
                fresh: self.fresh,
                points: &self.points,
                events: &mut self.events,
                retired: &mut retired[i],
                next_serial: &mut self.next_serial,
                map_context: input.map_context,
            };
            side.update(
                selected[i + 1],
                [selected[0], selected[2 - i]],
                proven[i],
                input.blind[i],
                input.lost[i],
                blocked,
                dt,
            )?;
        }

        if let Some(flow_configs) = &k.smooth {
            if clock_reset || !input.can_valid || input.reverse {
                for side in &mut self.sides {
                    side.flow = Flow::default();
                }
            }
            if !input.reverse && input.can_valid {
                for (i, slot) in slots.iter_mut().enumerate() {
                    let state = &mut self.sides[i];
                    let motion = state.rear.as_ref().map(|track| track.motion(k, None));
                    let dt = if clock_reset { 0. } else { dt };
                    state.flow.render(
                        &flow_configs[i],
                        slot,
                        i,
                        now,
                        dt,
                        selected[i + 1],
                        [selected[0], selected[2 - i]],
                        motion.as_ref(),
                        retired[i].as_ref(),
                        input.blind[i],
                        input.changing,
                        input.rear_lateral[i],
                    )?;
                    self.events
                        .extend(state.flow.events.iter().map(|&name| Event::Flow { side: i, name }));
                }
            }
        }
        Ok(())
    }
}

struct Side<'a> {
    state: &'a mut SideState,
    slots: &'a mut SideSlots,
    k: &'a Consts,
    side: usize,
    now: f64,
    ego: f64,
    hist: &'a VecDeque<(f64, f64, f64)>,
    raw_stamp: f64,
    fresh: bool,
    points: &'a HashMap<i128, Sample>,
    events: &'a mut Vec<Event>,
    retired: &'a mut Option<Motion>,
    next_serial: &'a mut u64,
    map_context: bool,
}

impl Side<'_> {
    fn sample(&self, source: i128) -> Option<Sample> {
        self.fresh.then(|| self.points.get(&source).copied()).flatten()
    }

    fn drop_track(&mut self, saved: Track, reason: &'static str) {
        self.events.push(Event::Drop {
            side: self.side,
            reason,
            source: saved.key.0,
            phase: saved.phase,
            distance: saved.distance,
            age: self.now - saved.seen,
            qualified: saved.qualified,
            hits: saved.hits,
        });
        if saved.phase != Phase::Front {
            if self.retired.is_none() {
                *self.retired = Some(saved.motion(self.k, Some(reason)));
            }
            if reason != "bsd" {
                self.slots.rear.detect = Some(Number::Integer(0));
                self.slots.reset_distance = Some(15.);
            }
        }
    }

    fn drop_rear(&mut self, reason: &'static str) {
        if let Some(track) = self.state.rear.take() {
            self.drop_track(track, reason);
        }
    }

    fn drop_front(&mut self, reason: &'static str) {
        if let Some(track) = self.state.front.take() {
            self.drop_track(track, reason);
        }
    }

    fn consume(&mut self, key: Identity) {
        if self.state.consumed.len() == 16 {
            self.state.consumed.pop_front();
        }
        self.state.consumed.push_back(key);
    }

    fn ratio(&self, saved: &Track, ego: f64) -> Result<f64> {
        if saved.anchor < self.k.anchor_min_ego {
            Ok(0.)
        } else {
            Ok(py_min(py_max(divide(ego, saved.anchor)?, 0.), self.k.ratio_max))
        }
    }

    fn dx(&self, saved: &Track, t0: f64, t1: f64) -> Result<f64> {
        if t0 >= t1 || t0.is_nan() || t1.is_nan() || saved.anchor < self.k.anchor_min_ego {
            return Ok(saved.vrel * 0.);
        }
        let mut recent = [0.; 8];
        let mut older = Vec::new();
        let mut count = 0;
        for &(_, ego, dt) in self
            .hist
            .iter()
            .rev()
            .take_while(|row| row.0 > t0)
            .filter(|row| row.0 <= t1)
        {
            let term = dt * self.ratio(saved, ego)?;
            if let Some(slot) = recent.get_mut(count) {
                *slot = term;
                count += 1;
            } else {
                older.push(term);
            }
        }
        Ok(saved.vrel * py_sum(older.into_iter().rev().chain(recent[..count].iter().rev().copied())).unwrap_or(0.))
    }

    fn continuous(&self, saved: &Track, point: Sample, limit: f64) -> Result<bool> {
        let elapsed = self.raw_stamp - saved.seen;
        Ok(0. < elapsed
            && elapsed <= limit
            && (point.x - saved.x - self.dx(saved, saved.seen, self.raw_stamp)?).abs() <= self.k.x_gate + 2. * elapsed
            && (point.y - saved.y).abs() <= (if saved.qualified { self.k.qualified_y_gate } else { 0.25 }) + elapsed
            && (point.vrel - saved.vrel).abs() <= self.k.vrel_gate + 2. * elapsed)
    }

    fn anchor_rate(&self, saved: &mut Track) -> Result<()> {
        let x = saved.x + self.dx(saved, saved.seen, saved.updated)?;
        saved.rate = if x > 0.1 {
            py_min(py_max(saved.signed / x, self.k.scale), 1.5)
        } else {
            self.k.scale
        };
        Ok(())
    }

    fn signed(&self, saved: &Track) -> Result<f64> {
        Ok(saved.signed + saved.rate * self.dx(saved, saved.updated, self.now)?)
    }

    fn retreat(&self, saved: &Track) -> Result<f64> {
        Ok(py_min(
            self.k.exit_max_speed,
            py_max(0., -saved.vrel * self.ratio(saved, self.ego)? * self.k.scale),
        ))
    }

    fn start_coast(&mut self, mut saved: Track, reason: &'static str) -> Result<bool> {
        if saved.drawn_stamp > saved.updated {
            saved.distance = saved.drawn_distance;
            saved.updated = saved.drawn_stamp;
            saved.lateral = saved.drawn_lateral;
        }
        saved.signed = saved.distance;
        self.anchor_rate(&mut saved)?;
        let mut point = self.sample(saved.key.0);
        if let Some(sample) = point {
            if self.raw_stamp > saved.seen && !self.continuous(&saved, sample, self.k.coast_time)? {
                point = None;
            }
        }
        let predicted = saved.x + self.dx(&saved, saved.seen, self.now)?;
        let x = point.map_or(predicted, |sample| sample.x);
        let mut recent = saved.vlead_hist.clone();
        if let Some(sample) = point.filter(|_| self.raw_stamp > saved.seen) {
            recent.push(sample.vlead);
        }
        let start = recent.len().saturating_sub(3);
        let vlead = if recent.is_empty() {
            saved.vlead
        } else {
            median_values(&mut recent[start..])
        };
        let reachable =
            saved.qualified && self.now - saved.seen <= self.k.coast_time && predicted >= self.k.coast_min_x;
        let reject = if !reachable {
            Some(reason)
        } else if x >= 3. {
            Some("still_ahead")
        } else if vlead < 0. {
            Some("stopped")
        } else {
            None
        };
        if let Some(reject) = reject {
            self.drop_track(saved, reject);
            return Ok(false);
        }
        self.drop_rear("rear_replaced");
        self.consume(saved.key);
        for &alias in &saved.aliases {
            self.consume(alias);
        }
        saved.phase = Phase::Coast;
        saved.distance = 0.;
        saved.shown = self.now;
        saved.start = None;
        saved.speed = 0.;
        saved.coast_y = point.map_or(saved.y, |sample| sample.y);
        self.slots.reset_distance = Some(0.);
        self.events.push(Event::Coast {
            side: self.side,
            reason,
            source: saved.key.0,
            x: saved.x,
            vrel: saved.vrel,
            vlead: saved.vlead,
            anchor: saved.anchor,
            ego: self.ego,
            front_distance: saved.signed,
        });
        self.state.rear = Some(saved);
        Ok(true)
    }

    fn alias(&mut self, key: Identity) -> Result<bool> {
        let Some(coast) = self.state.rear.as_ref() else {
            return Ok(false);
        };
        if coast.phase != Phase::Coast {
            return Ok(false);
        }
        let point = self.sample(key.0);
        let x = match point {
            Some(point) => point.x,
            None => divide(self.slots.distance(self.side)?.value(), self.k.scale)?,
        };
        let predicted = coast.x + self.dx(coast, coast.seen, self.now)?;
        if (x - predicted).abs().partial_cmp(&self.k.alias_gap) != Some(std::cmp::Ordering::Less)
            || point.is_some_and(|p| {
                !((p.y - coast.y).abs() <= self.k.alias_y && (p.vrel - coast.vrel).abs() <= self.k.alias_vrel)
            })
        {
            return Ok(false);
        }
        self.state.rear.as_mut().expect("coasting track").aliases.push(key);
        self.consume(key);
        self.slots.front.detect = Some(Number::Integer(0));
        Ok(true)
    }

    fn observe_front(&mut self, key: Option<Identity>, proven: bool, hold: bool) -> Result<()> {
        let stale = self.k.stale_hold;
        if let Some(mut saved) = self.state.front.take() {
            let mut rekey = key.is_some_and(|key| key != saved.key);
            if let Some(key) = key.filter(|key| rekey && key.0 == saved.key.0 && proven && !hold) {
                if let Some(point) = self.sample(key.0) {
                    if self.raw_stamp > saved.seen && self.continuous(&saved, point, stale + 0.05 + 1e-6)? {
                        if !saved.aliases.contains(&saved.key) {
                            saved.aliases.push(saved.key);
                        }
                        saved.aliases.retain(|alias| *alias != key);
                        saved.key = key;
                        rekey = false;
                    }
                }
            }
            if rekey {
                if saved.qualified {
                    self.start_coast(saved, "object_replaced")?;
                } else {
                    self.drop_track(saved, "object_replaced");
                }
            } else {
                self.state.front = Some(saved);
            }
        }
        if let Some(key) = key.filter(|_| self.state.front.is_none()) {
            if self.alias(key)? {
                return Ok(());
            }
        }
        let key = match key {
            Some(key) if self.slots.detect(self.side)?.value() != 0. => key,
            _ => {
                if let Some(saved) = self.state.front.take() {
                    if saved.qualified {
                        self.start_coast(saved, "selection_ended")?;
                    } else {
                        self.drop_track(saved, "no_qualified_zero_crossing");
                    }
                }
                return Ok(());
            }
        };
        if let Some(saved) = self.state.front.as_mut() {
            let (distance, lateral) = self.slots.drawing(self.side)?;
            saved.drawn_distance = distance;
            saved.drawn_stamp = self.now;
            saved.drawn_lateral = lateral;
        }
        let point = self.sample(key.0);
        if hold {
            return Ok(());
        }
        if !proven || point.is_none() {
            if let Some(saved) = self.state.front.as_ref() {
                if saved.qualified {
                    if proven && !self.fresh && self.now - saved.seen <= stale + 1e-6 {
                        return Ok(());
                    }
                    let saved = self.state.front.take().expect("front track");
                    if self.start_coast(saved, if proven { "measurement_lost" } else { "not_proven" })? {
                        self.slots.front.detect = Some(Number::Integer(0));
                    }
                    return Ok(());
                }
                if self.now - saved.seen <= stale + 1e-6 {
                    return Ok(());
                }
            }
            self.drop_front("measurement_unqualified");
            return Ok(());
        }
        let point = point.expect("proven observation");
        let continuous = if let Some(saved) = self.state.front.as_ref() {
            if self.raw_stamp <= saved.seen {
                return Ok(());
            }
            self.continuous(saved, point, stale + 0.05 + 1e-6)?
        } else {
            false
        };
        if !continuous {
            self.drop_front("identity_innovation");
            self.state.front = Some(Track::new(key, *self.next_serial, self.raw_stamp, point.x));
            *self.next_serial += 1;
        }
        let saved = self.state.front.as_mut().expect("front track");
        let n = self.k.vrel_n;
        let vrel = median(&mut saved.vrel_hist, point.vrel, n);
        let vlead = median(&mut saved.vlead_hist, point.vlead, n);
        let moving_n = if self.k.moving_n == 0 { n } else { self.k.moving_n };
        let vlead_long = median(&mut saved.vlead_long, point.vlead, moving_n);
        let (distance, lateral) = self.slots.drawing(self.side)?;
        saved.x = point.x;
        saved.y = point.y;
        saved.vrel = vrel;
        saved.vrel_raw = point.vrel;
        saved.vlead = vlead;
        saved.anchor = self.ego;
        saved.seen = self.raw_stamp;
        saved.updated = self.now;
        saved.distance = distance;
        saved.lateral = lateral;
        saved.drawn_distance = distance;
        saved.drawn_stamp = self.now;
        saved.drawn_lateral = lateral;
        let detect = self.slots.detect(self.side)?.value();
        finite_integer(detect)?;
        saved.detect = detect as i64;
        saved.hits += 1;
        saved.radar_moving += i64::from(point.vehicle);
        saved.moving_hits += i64::from(point.vehicle || self.map_context);
        saved.qualified |= saved.hits >= 4
            && saved.moving_hits >= 3
            && self.now - saved.first >= self.k.confirm_time
            && saved.first_x - point.x >= 0.25
            && vrel <= -0.3
            && vlead_long >= {
                let moving = self.k.moving_fraction * self.ego;
                if saved.radar_moving >= 3 {
                    moving - self.k.moving_slack
                } else {
                    py_max(moving - self.k.moving_slack_map, self.k.moving_vlead)
                }
            };
        Ok(())
    }

    fn coast(&mut self, saved: &mut Track) -> Result<Option<&'static str>> {
        if let Some(point) = self.sample(saved.key.0) {
            if self.raw_stamp > saved.seen && self.continuous(saved, point, self.k.coast_time)? {
                if point.x >= 3. {
                    return Ok(Some("still_ahead"));
                }
                if (point.y - saved.coast_y).abs() > self.k.lane_half {
                    return Ok(Some("lane_departed"));
                }
                let predicted = saved.x + self.dx(saved, saved.seen, self.raw_stamp)?;
                let vrel = median(&mut saved.vrel_hist, point.vrel, self.k.vrel_n);
                let vlead = median(&mut saved.vlead_hist, point.vlead, self.k.vrel_n);
                saved.signed = self.signed(saved)? + saved.rate * (point.x - predicted);
                saved.updated = self.now;
                saved.x = point.x;
                saved.y = point.y;
                saved.vrel = vrel;
                saved.vrel_raw = point.vrel;
                saved.vlead = vlead;
                saved.anchor = self.ego;
                saved.seen = self.raw_stamp;
                self.anchor_rate(saved)?;
            }
        }
        if saved.x + self.dx(saved, saved.seen, self.now)? >= 3. {
            return Ok(Some("still_ahead"));
        }
        saved.signed = self.signed(saved)?;
        saved.updated = self.now;
        if saved.signed.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) {
            saved.speed = self.retreat(saved)?;
            saved.phase = Phase::Rear;
            saved.start = Some(self.now);
            self.events.push(Event::Handoff {
                side: self.side,
                key: saved.key,
                distance: saved.distance,
                seen_age: self.now - saved.seen,
                speed: saved.speed,
                lateral: saved.lateral,
                coast_time: self.now - saved.shown,
                origin: saved.signed <= 0.,
            });
        }
        Ok(None)
    }

    fn draw_rear(&mut self, saved: &mut Track, dt: f64) -> Result<Option<&'static str>> {
        if saved.phase == Phase::Coast {
            if let Some(reason) = self.coast(saved)? {
                return Ok(Some(reason));
            }
        }
        let rear = saved.phase == Phase::Rear;
        let age = self.now - saved.shown;
        let grace = if rear { self.k.exit_grace } else { 0. };
        if age >= self.k.max_time + grace {
            return Ok(Some("timeout"));
        }
        if rear && age >= self.k.max_time && saved.forced.unwrap_or(0.) == 0. {
            saved.forced = Some(self.now);
            self.events.push(Event::Forced {
                side: self.side,
                source: saved.key.0,
                distance: saved.distance,
            });
        }
        if let Some(start) = saved.start.filter(|&start| rear && start < self.now) {
            let mut ramp_start = py_max(saved.seen + self.k.exit_delay, start);
            let forced = saved.forced.filter(|&time| time != 0.);
            if let Some(forced) = forced {
                ramp_start = py_min(ramp_start, forced);
            }
            let ramp0 = py_max(0., self.now - dt - ramp_start);
            let ramp1 = py_max(0., self.now - ramp_start);
            let mut base = self.retreat(saved)?;
            if forced.is_some() {
                base = py_max(base, self.k.exit_forced_speed);
            }
            let accel = self.k.exit_accel;
            let cap = py_max(0., divide(self.k.exit_max_speed - base, accel)?);
            let integral = |t| {
                let q = py_min(t, cap);
                0.5 * accel * q * q + accel * cap * py_max(0., t - cap)
            };
            saved.distance += base * dt + integral(ramp1) - integral(ramp0);
            saved.speed = py_min(self.k.exit_max_speed, base + accel * ramp1);
            if saved.distance >= self.k.end_distance {
                return Ok(Some("distance_end"));
            }
        }
        let front_distance = self.slots.distance(self.side)?.value();
        if self.slots.detect(self.side)?.value() != 0. && 0. <= front_distance && front_distance < self.k.min_gap {
            let clearance = 10. * (self.k.min_gap - front_distance);
            finite_integer(clearance)?;
            saved.distance = py_max(saved.distance, clearance.ceil() / 10.);
        }
        self.slots.rear = Slot {
            distance: Some(Number::Float(saved.distance)),
            lateral: Some(Number::Float(saved.lateral)),
            detect: Some(Number::Integer(saved.detect)),
        };
        self.slots.reset_distance = Some(saved.distance);
        Ok(None)
    }

    fn update(
        &mut self,
        mut key: Option<Identity>,
        elsewhere: [Option<Identity>; 2],
        proven: bool,
        bsd: bool,
        lost: bool,
        blocked: Option<&'static str>,
        dt: f64,
    ) -> Result<()> {
        let once = !proven && !self.state.unproven;
        self.state.unproven = !proven;
        if let Some(reason) = if bsd { Some("bsd") } else { blocked } {
            self.drop_front(reason);
            self.drop_rear(reason);
            return Ok(());
        }
        if lost {
            self.drop_front("confidence_lost");
        }
        let moved = |saved: &Track| elsewhere.into_iter().flatten().any(|key| saved.owns(key));
        if self.state.front.as_ref().is_some_and(moved) {
            self.drop_front("changed_lane_slot");
        }
        if self.state.rear.as_ref().is_some_and(moved) {
            self.drop_rear("changed_lane_slot");
        }
        if let Some(consumed) = key.filter(|key| self.state.consumed.contains(key)) {
            let own = self.state.rear.as_ref().is_some_and(|saved| saved.owns(consumed));
            if own
                && self
                    .sample(consumed.0)
                    .is_some_and(|point| point.x >= 1. && point.vrel > 0.3)
            {
                self.drop_rear("measured_front_reacquired");
            } else if own || self.slots.distance(self.side)?.value() <= self.k.scale {
                self.slots.front.detect = Some(Number::Integer(0));
            }
            key = None;
        }
        if !lost {
            self.observe_front(key, proven || once, once)?;
        }
        if let Some(mut saved) = self.state.rear.take() {
            match self.draw_rear(&mut saved, dt) {
                Ok(Some(reason)) => self.drop_track(saved, reason),
                result => {
                    self.state.rear = Some(saved);
                    result?;
                }
            }
        }
        Ok(())
    }
}

fn median(hist: &mut Vec<f64>, raw: f64, n: usize) -> f64 {
    hist.push(raw);
    if n != 0 && hist.len() > n {
        hist.drain(..hist.len() - n);
    }
    let mut short = [0.; 40];
    if let Some(values) = short.get_mut(..hist.len()) {
        values.copy_from_slice(hist);
        median_values(values)
    } else {
        median_values(&mut hist.clone())
    }
}

fn median_values(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.
    } else {
        values[middle]
    }
}

fn divide(a: f64, b: f64) -> Result<f64> {
    if b == 0. {
        Err(Error::ZeroDivision("float division by zero"))
    } else {
        Ok(a / b)
    }
}

fn finite_integer(x: f64) -> Result<()> {
    if x.is_nan() {
        Err(Error::Value("cannot convert float NaN to integer"))
    } else if x.is_infinite() {
        Err(Error::Overflow("cannot convert float infinity to integer"))
    } else {
        Ok(())
    }
}
