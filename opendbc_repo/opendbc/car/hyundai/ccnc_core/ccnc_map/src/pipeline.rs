//! One stream of cereal inputs for the world map, shared by the daemon and replay.

use crate::cmap::MapOut;
use crate::input::{
    live_tracks::{self, EVENT_LIVE_TRACKS},
    map_events::{CarState, DisplayLead, Event, CAR_STATE, LIVE_POSE, RADAR_STATE},
    model_v2::{self, EVENT_MODEL_V2},
};
use crate::pos::common::{LEAD_GATE_X, LEAD_GATE_Y};
use crate::radar::{
    associate::RadarPoint,
    consts::Consts,
    select::rememberable,
    lanes::GeometryInput,
    state::{Observation, UpdateIn},
    Frame, MapState, Range, Result,
};
use crate::rear::{WorldInput, WorldRear};
use crate::slots::{Assigner, Slot, FF_CONFIRM_S, LEAD_SPEED_MATCH, OBSERVATION_MAX_AGE};
use crate::vehicles::{Tracker, Vehicle, MAX_VEHICLES};
use crate::vision::{self, Vision};
use crate::world::{World, RADAR_LAG, STATUS_FROZEN};
use std::collections::VecDeque;

pub const STALE_NS: u64 = 250_000_000;
/// How far back the inputs of a camera frame's time are kept (covers LATE_S = 3 s with 1 s margin).
const HISTORY_NS: u64 = 4_000_000_000;
const RADAR: Consts = Consts::map_process();
/// A classifier answer is tied to the radar tracks recorded at most this far from its frame (liveTracks is 20 Hz).
const KIND_TRACKS_NS: u64 = 150_000_000;
/// The map asks `ccnc_visiond` for kinds only while it hears from it (a status message comes about once a second).
const VISION_ALIVE_NS: u64 = 3_000_000_000;

fn seconds(t: u64) -> f64 {
    t as f64 * 1e-9
}

fn fresh(now: u64, stamp: u64) -> bool {
    now.checked_sub(stamp).is_some_and(|age| age <= STALE_NS)
}

/// A context input for the radar update at `stamp`. The daemon reads every queue when liveTracks arrives, so a
/// carState or radarState published while radarcan was sending can be newer than the tracks.
fn current(stamp: u64, input: u64) -> bool {
    stamp.abs_diff(input) <= STALE_NS
}

/// Recent inputs by logMonoTime, for the camera frames that arrive up to a second and a half late.
struct History<T>(VecDeque<(u64, T)>);

impl<T> History<T> {
    fn push(&mut self, stamp: u64, value: T) {
        if self.0.back().is_some_and(|(last, _)| stamp < *last) {
            return;
        }
        while self.0.front().is_some_and(|(first, _)| first + HISTORY_NS < stamp) {
            self.0.pop_front();
        }
        self.0.push_back((stamp, value));
    }

    /// The last one at or before `t`, else the first (what the offline rules were scored with).
    fn at(&self, t: u64) -> Option<&T> {
        let i = self.0.partition_point(|(stamp, _)| *stamp <= t);
        self.0.get(i.saturating_sub(1)).map(|(_, value)| value)
    }

    /// The record closest to `t` and its time, when it is at most `tolerance` away (a time outside the kept range
    /// has none).
    fn near(&self, t: u64, tolerance: u64) -> Option<(u64, &T)> {
        let i = self.0.partition_point(|(stamp, _)| *stamp <= t);
        [i.checked_sub(1), Some(i)]
            .into_iter()
            .flatten()
            .filter_map(|j| self.0.get(j))
            .min_by_key(|(stamp, _)| stamp.abs_diff(t))
            .filter(|(stamp, _)| stamp.abs_diff(t) <= tolerance)
            .map(|(stamp, value)| (*stamp, value))
    }
}

impl<T> Default for History<T> {
    fn default() -> Self {
        Self(VecDeque::new())
    }
}

pub struct Pipeline {
    world: World,
    radar: MapState,
    tracker: Tracker,
    slots: Assigner,
    /// The last radar update's `stopped` (ego at or below 0.3 km/h), for the legacy `holding`.
    stopped: bool,
    rear: WorldRear,
    generation_base: u32,
    radar_generation: u32,
    last_tracks: Option<u64>,
    tracks_valid: bool,
    car: Option<(u64, CarState)>,
    model_stamp: Option<u64>,
    /// radarState at its time: leadOne position for the radar map, preferred and primary display leads for FF.
    lead: Option<(u64, Option<(f64, f64)>, Option<DisplayLead>, Option<DisplayLead>)>,
    /// Consecutive liveTracks observations matching the measured leadOne: ID, start and latest radar time.
    lead_one_match: Option<(u32, u64, u64)>,
    vision: Vision,
    start_guard: det_rules::StartGuard,
    tracks_at: History<Vec<vision::Track>>,
    speed_at: History<f64>,
    lprob_at: History<Vec<f64>>,
    /// The last update's radar slots, before the camera cars.
    radar_slots: [Option<Slot>; 5],
    kinds: vision::Kinds,
    /// logMonoTime of the last CcncVision message.
    vision_seen: Option<u64>,
}

impl Pipeline {
    pub fn new(generation_base: u32) -> Self {
        Self {
            world: World::default(),
            radar: MapState::for_observation(),
            // With its ID history: a classifier answer finds its car at the frame's radar update (`vid_at`).
            tracker: Tracker::default(),
            slots: Assigner::default(),
            stopped: false,
            rear: WorldRear::default(),
            generation_base,
            radar_generation: 0,
            last_tracks: None,
            tracks_valid: false,
            car: None,
            model_stamp: None,
            lead: None,
            lead_one_match: None,
            vision: Vision::default(),
            start_guard: det_rules::StartGuard::default(),
            tracks_at: History::default(),
            speed_at: History::default(),
            lprob_at: History::default(),
            radar_slots: [None; 5],
            kinds: vision::Kinds::default(),
            vision_seen: None,
        }
    }

    fn empty(&self) -> MapOut {
        MapOut {
            radar_log_mono_time: self.last_tracks.unwrap_or(0),
            generation: self.generation_base.wrapping_add(self.world.generation()),
            status: self.world.status(),
            ego_lane_coord: 0.,
            slots: [None; 5],
            valid: false,
        }
    }

    /// Publish one invalidation when the last radar input times out, then let the topic go quiet.
    pub fn expire(&mut self, now: u64) -> Option<MapOut> {
        if !self.tracks_valid
            || self
                .last_tracks
                .is_none_or(|stamp| fresh(now, stamp) || stamp > now)
        {
            return None;
        }
        self.tracks_valid = false;
        self.world.reset();
        self.sync_generation();
        Some(self.empty())
    }

    fn sync_generation(&mut self) {
        if self.radar_generation != self.world.generation() {
            self.radar_generation = self.world.generation();
            self.radar = MapState::for_observation();
            self.model_stamp = None;
            self.lead_one_match = None;
            self.vision.reset_positions();
        }
    }

    /// One CcncVision frame that reached the map at `stamp` (its logMonoTime). The decision shows from the next
    /// radar update on. `Skipped::NeedsGeometry` asks for `set_vision_geometry`; the frame is not used.
    pub fn vision(&mut self, stamp: u64, frame: &vision::Frame) -> std::result::Result<Option<vision::Stepped>, vision::Skipped> {
        let t = frame.time_ns;
        let (Some(tracks), Some(&v_ego)) = (self.tracks_at.at(t), self.speed_at.at(t)) else {
            return Err(vision::Skipped::NoInputs);
        };
        let context = vision::Context {
            v_ego,
            tracks,
            lprob: self.lprob_at.at(t).map(Vec::as_slice),
            slots: vision::slot_states(&self.radar_slots),
            guard: self.start_guard.ok(),
        };
        let pose = self.world.pose.at(seconds(t));
        self.vision.frame(frame, seconds(stamp), &context, pose)
    }

    /// Every CcncVision message that reached the map at `stamp` (its logMonoTime): `front_frame` is the frame time
    /// of a road camera frame, `results` the classifier answers, `calibration` the message's (rpyCalib, height used
    /// for the road camera). Answers show from the next radar update on, on cars the radar already draws.
    pub fn kinds(&mut self, stamp: u64, front_frame: Option<u64>, results: &[vision::KindResult], calibration: &[f32; 7]) {
        self.vision_seen = Some(self.vision_seen.map_or(stamp, |seen| seen.max(stamp)));
        if let Some(t) = front_frame {
            self.kinds.front_frame(t);
        }
        if results.is_empty() {
            return;
        }
        let c = calibration.map(f64::from);
        let cam = vision::kind::FrontCam::new([c[0], c[1], c[2]], c[6]);
        let (tracks_at, vision, tracker) = (&self.tracks_at, &self.vision, &self.tracker);
        self.kinds.observe(
            results,
            cam.as_ref(),
            |t| tracks_at.near(t, KIND_TRACKS_NS).map(|(stamp, tracks)| (stamp, tracks.clone())),
            |t, side| vision.side_cands(t, side),
            // The car the track was at the frame's radar update, not the car with that radar id now.
            |stamp, id| tracker.vid_at(seconds(stamp), u32::try_from(id).ok()?),
        );
    }

    pub fn set_vision_geometry(&mut self, geometry: vision::Geometry) {
        self.vision.set_geometry(geometry);
    }

    pub fn ingest(&mut self, bytes: &[u8]) -> Result<Option<MapOut>> {
        let Some(event) = Event::read(bytes) else {
            return Ok(None);
        };
        let stamp = event.stamp;
        let now = seconds(stamp);
        match event.which {
            LIVE_POSE if event.valid => {
                if let Some(yaw) = event.yaw() {
                    self.world.live_pose(now, yaw);
                }
            }
            CAR_STATE if self.car.is_none_or(|(last, _)| stamp > last) => {
                let mut car = event.car_state().unwrap_or_default();
                car.can_valid &= event.valid && car.speed.is_finite();
                if car.can_valid {
                    self.speed_at.push(stamp, car.speed);
                    self.start_guard.observe(car.speed);
                    self.world.blinker(car.blinker);
                    self.world.car_state(now, car.speed, car.reverse);
                    self.sync_generation();
                }
                self.car = Some((stamp, car));
            }
            EVENT_MODEL_V2 if event.valid && self.model_stamp.is_none_or(|last| stamp > last) => {
                if let Some(model) = model_v2::parse(bytes) {
                    self.lprob_at.push(stamp, model.lane_line_probs.iter().map(|&p| f64::from(p)).collect());
                    self.world.model_with_walls(now, &model, Some(&self.radar));
                    let (px, py, heading) = self.radar.pose_at(now).unwrap_or(self.radar.pose);
                    let (s, c) = heading.sin_cos();
                    let frame = self.radar.frame();
                    let previous = self.radar.lane_cache.place(frame);
                    self.radar.lane_cache.refresh(
                        &GeometryInput::from_raw(model),
                        Frame { px, py, c, s },
                        frame,
                        &previous,
                        (RADAR.min_x, RADAR.far_x),
                        self.world.status() & STATUS_FROZEN != 0,
                    );
                    self.model_stamp = Some(stamp);
                }
            }
            RADAR_STATE => {
                if self.lead.is_none_or(|(last, ..)| stamp >= last) {
                    let (display, one) = if event.valid { event.display_leads() } else { (None, None) };
                    self.lead = Some((stamp, event.lead().filter(|_| event.valid), display, one));
                }
            }
            EVENT_LIVE_TRACKS if self.last_tracks.is_none_or(|last| stamp > last) => {
                if self.last_tracks.is_some_and(|last| !fresh(stamp, last)) && self.tracks_valid {
                    self.world.reset();
                    self.sync_generation();
                }
                self.last_tracks = Some(stamp);
                if let Some(points) = event.valid.then(|| live_tracks::parse(bytes)).flatten() {
                    self.tracks_at.push(
                        stamp,
                        points
                            .iter()
                            .map(|p| vision::Track {
                                id: p.track_id as i64,
                                d_rel: f64::from(p.d_rel),
                                y_rel: f64::from(p.y_rel),
                                v_rel: f64::from(p.v_rel),
                            })
                            .collect(),
                    );
                    self.tracks_valid = true;
                    return self.update_radar(stamp, points).map(Some);
                }
                self.lead_one_match = None;
                return Ok(Some(self.empty()));
            }
            _ => {}
        }
        Ok(None)
    }

    /// The model's two ego lines at the radar tick, unless the data is stale or ego is changing lanes.
    fn model_lane(&self, stamp: u64) -> Option<crate::slots::ModelLane> {
        if !self.model_stamp.is_some_and(|model_stamp| current(stamp, model_stamp)) || self.world.ego_changing() {
            return None;
        }
        let cache = &self.radar.lane_cache;
        let line = |role: i64| {
            let i = usize::try_from(role + self.radar.lane_offset).ok()?;
            let probability = cache.lines.get(&i)?.probability;
            probability.is_finite().then_some((cache.geometry.get(&i)?, probability))
        };
        let (left, left_prob) = line(1)?;
        let (right, right_prob) = line(2)?;
        Some(crate::slots::ModelLane::with_probability(left.clone(), right.clone(), left_prob.min(right_prob)))
    }

    fn update_radar(&mut self, stamp: u64, input: Vec<live_tracks::RawPoint>) -> Result<MapOut> {
        let Some((_, car)) = self
            .car
            .filter(|(stamp_car, car)| car.can_valid && current(stamp, *stamp_car))
        else {
            self.lead_one_match = None;
            return Ok(self.empty());
        };
        if !self.world.accepts_radar() {
            self.lead_one_match = None;
            return Ok(self.empty());
        }
        let now = seconds(stamp);
        self.world.radar(now);
        self.sync_generation();
        let (lead, display_lead, lead_one) = self
            .lead
            .filter(|(t, ..)| current(stamp, *t))
            .map_or((None, None, None), |(_, lead, display, one)| (lead, display, one));
        // card's `holding` (vehicle_display.rs): stopped, or below 10 km/h and braking unless stopped a frame ago.
        let speed = car.speed.max(0.);
        let stopped = speed <= 0.3 / 3.6;
        let holding = stopped
            || (!self.stopped && speed <= 10. / 3.6 && car.accel.is_finite() && car.accel < -0.1 / 3.6);
        self.stopped = stopped;
        let mut tracks = Vec::with_capacity(input.len().min(MAX_VEHICLES));
        let mut measured = Vec::with_capacity(tracks.capacity());
        let mut wall_car_track = None;
        let mut measured_one = None;
        for point in input.iter().filter(|p| p.source == 0).take(MAX_VEHICLES) {
            let (x, y, v_rel) = (
                f64::from(point.d_rel),
                f64::from(point.y_rel),
                f64::from(point.v_rel),
            );
            let Ok(id) = u32::try_from(point.track_id) else {
                continue;
            };
            if id == 0 || !(0. ..=200.).contains(&x) || !y.is_finite() || !v_rel.is_finite() {
                continue;
            }
            if x > 0. && y.abs() <= 15. {
                tracks.push((id, x, y, v_rel));
            }
            if point.measured {
                let speed = f64::from(point.v_lead);
                if speed.is_finite() {
                    if lead_one.is_some_and(|one| {
                        let (lx, ly, lv) = one.motion;
                        one.measured_track == Some(id) && (x - lx).abs() <= LEAD_GATE_X
                            && (y - ly).abs() <= LEAD_GATE_Y && (v_rel - lv).abs() <= LEAD_SPEED_MATCH
                    }) {
                        measured_one = Some(id);
                    }
                    if holding && display_lead.is_some_and(|lead| {
                        let (lx, ly, lv) = lead.motion;
                        lead.confirmed_track == Some(i128::from(id)) && (x - lx).abs() <= 1.
                            && (y - ly).abs() <= 0.5 && (v_rel - lv).abs() <= 1.
                    }) {
                        wall_car_track = Some(i128::from(id));
                    }
                    measured.push(RadarPoint {
                        track: i128::from(id),
                        x,
                        y,
                        speed,
                        moving: point.motion_state == 2,
                        tracked: point.track_state == 2,
                    });
                }
            }
        }
        let settled_primary = self.observe_lead_one(stamp, measured_one);
        let primary = lead_one.map(|one| (one, settled_primary));
        let measured_yaw = self.world.yaw_at(now);
        let yaw = measured_yaw.unwrap_or(0.);
        let changing = self.world.status() & STATUS_FROZEN != 0;
        self.radar.advance(
            now,
            car.speed,
            yaw,
            holding,
            None,
            Range {
                min_x: RADAR.min_x,
                keep_x: RADAR.keep_x,
                max_y: RADAR.max_y,
                far_x: RADAR.far_x,
                far_y: RADAR.far_y,
            },
        );
        self.radar.lane_cache.place(self.radar.frame());
        let (x, y, heading) = self
            .radar
            .pose_at(now - RADAR_LAG)
            .or_else(|| self.radar.poses.front().map(|(_, pose)| pose))
            .unwrap_or(self.radar.pose);
        let (sin, cos) = heading.sin_cos();
        self.radar.update(
            &RADAR,
            UpdateIn {
                radar: Some(Observation {
                    points: &measured,
                    stamp: now,
                    pose: Some((x, y, cos, sin)),
                    evidence: &Default::default(),
                    wall_car_track,
                }),
                radar_stamp: Some(now),
                a_ego: if car.accel.is_finite() { car.accel } else { 0. },
                yaw_rate: yaw,
                holding,
                ego_change: changing,
                model_stamp: self.model_stamp.map(seconds),
                model_changing: changing,
                lane_change: changing,
                blindspot: car.blind,
                leads: lead.as_slice(),
                vision_leads: &[],
                vision: &[],
                select: false,
                usable: true,
            },
        )?;
        self.tracker.keep(if holding { self.slots.remembered_vids() } else { [None; 2] });
        self.radar.update_vehicles(
            &mut self.tracker,
            now,
            &tracks,
            &self.world,
            RADAR.crossing_travel,
        );
        self.kinds.apply(now, self.world.radar_pose(now).heading, self.tracker.vehicles_mut());
        let cars = self.tracker.vehicles();
        self.slots.set_model_lane(self.model_lane(stamp));
        let mut slots = self.slots.update_display_lead(now, cars, &self.world, display_lead, primary);
        let measured_slots = slots;
        let remembered = if holding {
            // The map's 0.15 s freshness, not the selector's 0.3 s prediction hold, decides when memory takes over.
            let (map, frame) = (&self.radar, self.radar.frame());
            let remember = |car: &Vehicle| {
                car.flags.object.is_some_and(|key| {
                    map.objects
                        .get(&key)
                        .zip(map.local_xy(key).ok())
                        .is_some_and(|(o, (x, _))| rememberable(&RADAR, frame, now, o, x))
                })
            };
            self.slots.hold_stopped(now, cars, &self.world, car.blind, remember, &mut slots)
        } else {
            [false; 2]
        };
        let ego = self.world.u_ego();
        let rear_input = WorldInput {
            now,
            changing: changing || car.blinker,
            can_valid: true,
            blind: car.blind,
            vehicles: cars,
        };
        self.rear.finish(&self.world, rear_input, &mut slots)?;
        let split = self.slots.split_alongside(now, cars, remembered, &mut slots);
        let continuous = self.rear.continue_sides(&self.world, rear_input, remembered, &measured_slots, &mut slots);
        self.radar_slots = slots;
        let mut protected = [false, remembered[0] || split[0], remembered[1] || split[1], split[0], split[1]];
        for index in continuous.into_iter().flatten() {
            protected[index] = true;
        }
        self.vision.apply(now, car.speed, self.world.pose.at(now), protected, &mut slots);
        let straight = speed >= 0.5 && !car.blinker && !self.world.ego_changing()
            && measured_yaw.is_some_and(|rate| rate.abs() <= 0.04);
        self.slots.smooth_opening_sides(now, straight, &mut slots);
        let asking = self.vision_seen.is_some_and(|seen| stamp.saturating_sub(seen) <= VISION_ALIVE_NS);
        for slot in slots[..3].iter_mut().flatten() {
            slot.kind_wanted = asking
                && slot.vid != 0
                && slot.vision != crate::slots::VISION_ONLY
                && self.kinds.wanted(slot.vid, now);
        }
        Ok(if ego.is_finite() {
            MapOut {
                ego_lane_coord: ego as f32,
                slots,
                valid: true,
                ..self.empty()
            }
        } else {
            self.empty()
        })
    }

    fn observe_lead_one(&mut self, stamp: u64, measured: Option<u32>) -> bool {
        self.lead_one_match = measured.map(|id| {
            let since = self.lead_one_match
                .filter(|&(old, _, seen)| old == id && stamp - seen <= (OBSERVATION_MAX_AGE * 1e9) as u64)
                .map_or(stamp, |(_, since, _)| since);
            (id, since, stamp)
        });
        self.lead_one_match.is_some_and(|(_, since, _)| stamp - since >= (FF_CONFIRM_S * 1e9) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_lead_confirmation_requires_current_consecutive_measurements() {
        let mut pipe = Pipeline::new(0);
        for i in 0..8 {
            assert_eq!(pipe.observe_lead_one(i * 50_000_000, Some(40)), i == 7);
        }
        assert!(!pipe.observe_lead_one(400_000_000, None));
        assert!(!pipe.observe_lead_one(450_000_000, Some(40)));
        assert!(!pipe.observe_lead_one(900_000_000, Some(40)));
        assert!(!pipe.observe_lead_one(950_000_000, Some(44)));
        assert_eq!(pipe.lead_one_match, Some((44, 950_000_000, 950_000_000)));
    }

    #[test]
    fn the_map_tracker_finds_a_radar_id_at_an_earlier_update() {
        let mut pipe = Pipeline::new(0);
        let world = World::default();
        pipe.tracker.update(1.0, &[(10, 20., 0., 0.)], &world);
        pipe.tracker.update(1.05, &[(10, 20., 0., 0.)], &world);
        assert!(pipe.tracker.vid_at(1.0, 10).is_some(), "kind answers need the tracker's ID history");
    }

    #[test]
    fn the_tracker_keeps_radar_ids_as_long_as_the_tracks_kept_for_kind_answers() {
        let mut pipe = Pipeline::new(0);
        let world = World::default();
        let kept_s = HISTORY_NS as f64 * 1e-9;
        for i in 0..=(kept_s * 20.) as i32 {
            pipe.tracker.update(1.0 + f64::from(i) * 0.05, &[(10, 20., 0., 0.)], &world);
        }
        assert!(pipe.tracker.vid_at(1.0, 10).is_some(), "an answer for a frame {kept_s} s old finds its car");
    }

    #[test]
    fn v_lead_marks_stationary_guardrail_as_wall_and_keeps_slower_real_car() {
        let mut pipe = Pipeline::new(0);
        let point = |track_id: u64, d_rel: f32, y_rel: f32, v_rel: f32, v_lead: f32| live_tracks::RawPoint {
            track_id,
            d_rel,
            y_rel,
            v_rel,
            a_rel: 0.,
            yv_rel: 0.,
            measured: true,
            v_lead,
            a_lead: 0.,
            j_lead: 0.,
            source: 0,
            track_state: 2,
            motion_state: 0,
        };
        // Warm up the pose ring past RADAR_LAG (0.2 s) so pose_at(now - RADAR_LAG) tracks ego at 20 m/s.
        for i in 0..16u64 {
            let stamp = 1_000_000_000 + i * 50_000_000;
            let now = seconds(stamp);
            pipe.world.car_state(now, 20., false);
            pipe.car = Some((
                stamp,
                CarState {
                    speed: 20.,
                    can_valid: true,
                    ..CarState::default()
                },
            ));
            let points = if i < 6 {
                Vec::new()
            } else {
                let dt_total = (i - 6) as f32 * 0.05;
                vec![
                    // Three stationary guardrail posts along the right within max_x (20 m) and wall_max_gap (3.5 m).
                    point(1, 14. - 20. * dt_total, -3.5, -20., 0.),
                    point(2, 17. - 20. * dt_total, -3.5, -20., 0.),
                    point(3, 20. - 20. * dt_total, -3.5, -20., 0.),
                    // Slower real car in the left lane (v_lead=15, v_rel=-5, motionState != 2).
                    point(4, 18. - 5. * dt_total, 3.5, -5., 15.),
                    // Non-finite v_lead must be ignored in radar measurement.
                    point(99, 15., 0., -20., f32::NAN),
                ]
            };
            pipe.update_radar(stamp, points).unwrap();
        }
        assert!(!pipe.radar.walls.is_empty(), "stationary guardrail posts must form a wall");
        assert!(!pipe.radar.wall_geometry.is_empty(), "wall geometry must be populated");
        for id in [1i128, 2, 3] {
            let obj = pipe.radar.objects.values().find(|o| o.source == id).expect("guardrail object");
            assert!(obj.stationary, "guardrail {id} should be stationary");
            assert!(!obj.motion_confirmed, "guardrail {id} must not be motion_confirmed");
            assert!(!obj.doppler_confirmed, "guardrail {id} must not be doppler_confirmed");
            assert!(obj.structure, "guardrail {id} should belong to a wall structure");
        }
        let car_obj = pipe.radar.objects.values().find(|o| o.source == 4).expect("slower real car object");
        assert!(car_obj.forward_samples > 0, "slower real car (v_lead=15) must have forward Doppler samples");
        assert!(car_obj.motion_confirmed && !car_obj.stationary);
        let car = pipe.tracker.vehicles().iter().find(|v| v.radar_id == Some(4)).expect("tracked slower car");
        assert!(car.flags.proven && !crate::radar::select::approach_only(car_obj), "slower real car must not be approach-only");
        assert!(
            !pipe.radar.objects.values().any(|o| o.source == 99),
            "non-finite v_lead must be filtered out of radar measurements"
        );
    }

    #[test]
    fn a_car_whose_radar_point_jumps_keeps_its_motion_proof() {
        let point = |d_rel: f32, v_lead: f32| live_tracks::RawPoint {
            track_id: 4,
            d_rel,
            y_rel: -3.5,
            v_rel: v_lead,
            a_rel: 0.,
            yv_rel: 0.,
            measured: true,
            v_lead,
            a_lead: 0.,
            j_lead: 0.,
            source: 0,
            track_state: 2,
            motion_state: 0,
        };
        // Ego stands; a car on the right creeps away from 5 m at 2 m/s, and its point jumps back at 1.5 s: 0.8 m as
        // in 47c-5, or 3 m, which does not continue the earlier object.
        let run = |jump: f32| {
            let mut pipe = Pipeline::new(0);
            let mut proven = Vec::new();
            for i in 0..50u64 {
                let stamp = 1_000_000_000 + i * 50_000_000;
                let now = seconds(stamp);
                pipe.world.car_state(now, 0., false);
                pipe.car = Some((stamp, CarState { speed: 0., can_valid: true, ..CarState::default() }));
                let t = i as f32 * 0.05;
                let d = 5. + 2. * t - if i >= 30 { jump } else { 0. };
                pipe.update_radar(stamp, vec![point(d, 2.)]).unwrap();
                let car = pipe.tracker.vehicles().iter().find(|v| v.radar_id == Some(4));
                proven.push((i, car.map(|c| (c.vid, c.flags.proven))));
            }
            proven
        };
        let proven = run(0.8);
        let (before, after) = (proven[29].1.unwrap(), proven[30].1.unwrap());
        assert!(before.1, "{proven:?}");
        assert_eq!(before.0, after.0, "the same map car");
        assert!(proven[30..].iter().all(|(_, c)| c.is_some_and(|c| c.1)), "{proven:?}");
        let jumped = run(3.);
        assert!(jumped[29].1.unwrap().1 && !jumped[30].1.unwrap().1, "{jumped:?}");
    }

    #[test]
    fn a_car_state_newer_than_the_tracks_still_runs_the_radar_update() {
        // The daemon reads carState when liveTracks arrives; one published while radarcan was sending is newer.
        for (car_after_ms, runs) in [(5, true), (0, true), (-5, true), (260, false), (-260, false)] {
            let mut pipe = Pipeline::new(0);
            let stamp = 2_000_000_000u64;
            let car = stamp.checked_add_signed(car_after_ms * 1_000_000).unwrap();
            pipe.world.car_state(seconds(car), 10., false);
            pipe.car = Some((car, CarState { speed: 10., can_valid: true, ..CarState::default() }));
            pipe.update_radar(stamp, Vec::new()).unwrap();
            assert_eq!(pipe.world.radar_stamp() == Some(seconds(stamp)), runs, "car state {car_after_ms} ms after the tracks");
        }
    }

    #[test]
    fn a_model_frame_newer_than_the_tracks_can_still_supply_the_ego_lane() {
        let mut pipe = Pipeline::new(0);
        pipe.radar.lane_offset = 0;
        for (role, y) in [(1, 1.7), (2, -1.7)] {
            let points = vec![(0., y), (80., y)];
            pipe.radar.lane_cache.geometry.insert(role, points.clone());
            pipe.radar.lane_cache.lines.insert(role, crate::radar::lanes::Boundary { points, probability: 1., consistent: 1 });
        }
        pipe.model_stamp = Some(2_000_000_000);
        assert!(pipe.model_lane(1_995_000_000).is_some_and(|lane| lane.contains(20., 0.)));
        assert!(pipe.model_lane(1_745_000_000).is_none());
        assert!(pipe.model_lane(2_255_000_000).is_none());
    }

    #[test]
    fn near_drops_a_time_farther_than_the_tolerance() {
        let mut history = History::default();
        for t in [1_000, 1_050, 1_100] {
            history.push(t, t);
        }
        assert_eq!(history.near(1_060, 20), Some((1_050, &1_050)));
        assert_eq!(history.near(1_090, 20), Some((1_100, &1_100)));
        assert_eq!(history.near(1_130, 20), None, "after the last record");
        assert_eq!(history.near(970, 20), None, "before the kept range");
    }
}
