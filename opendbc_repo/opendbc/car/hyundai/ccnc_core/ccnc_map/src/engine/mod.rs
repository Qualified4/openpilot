//! Display-only object tracking and slot selection.

mod display;
mod object;
mod rear_display;
mod road_frame;
mod selector;
mod tracker;

pub use object::{Evidence, TrackedObject, VehicleKind};

use crate::input::{live_tracks::RawPoint, map_events::DisplayLead, model_v2::Model};
use crate::vision::kind::{FrontCam, KindResult};
use crate::world::{Pose, PoseAt};
use display::LateralDisplay;
use road_frame::{CornerLane, RoadFrame};
use std::collections::{HashMap, VecDeque};
use tracker::Tracker;

#[derive(Clone, Copy)]
pub struct RoadObservation<'a> {
    pub time: f64,
    /// Model position x/y pairs in metres, right positive.
    pub path: &'a [(f64, f64)],
    pub lines: Option<&'a Model>,
    pub road_edge_stds: [f32; 2],
    pub changing_lanes: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionObservation {
    /// Capture time in the same monotonic clock as the radar tick.
    pub time: f64,
    /// Ground point in capture coordinates, right positive.
    pub x: f64,
    pub y: f64,
    pub kind: VehicleKind,
    pub confidence: f64,
    /// A classifier answer carries type only. A detector frame can also support position and existence.
    pub position: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PedestrianObservation {
    /// Capture time in the same monotonic clock as the radar tick.
    pub time: f64,
    /// Ground point in capture coordinates, right positive.
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

/// The model lane-change flag used by the production message reader.
pub fn model_changing_lanes(bytes: &[u8]) -> bool {
    crate::input::model_v2::parse_road_path(bytes).is_none_or(|(_, changing, _)| changing)
}

pub struct EngineInput<'a> {
    /// Seconds in a single monotonic time base.
    pub time: f64,
    pub speed: f64,
    /// livePose yaw in rad/s, left positive as returned by Event::yaw.
    pub yaw: Option<f64>,
    pub road: Option<RoadObservation<'a>>,
    pub radar: &'a [RawPoint],
    /// radarState.leadOne and its original publication time.
    pub lead: Option<(f64, DisplayLead)>,
    /// Newly arrived camera observations only, including delayed classification.
    pub vision: &'a [VisionObservation],
    pub blind: [bool; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct EngineSlot {
    pub obj_id: u64,
    pub s: f64,
    pub d: f64,
    /// Cartesian position in ego coordinates, right positive.
    pub x: f64,
    pub y: f64,
    /// Road-relative lateral position drawn in the cluster, right positive.
    pub display_d: f64,
    pub(crate) rear_animated: bool,
    pub v_s: f64,
    pub kind: VehicleKind,
    pub existence_p: f64,
    pub last_track_id: Option<u64>,
    pub lane_u: f64,
    pub camera_seen: bool,
    radar_seen: bool,
    side_range: Option<selector::SideRange>,
    ff_handoff_until: Option<f64>,
}

#[derive(Debug, Default)]
pub struct EngineOutput<'a> {
    /// FF, LF, RF, LR, RR.
    pub slots: [Option<EngineSlot>; 5],
    pub objects: &'a [TrackedObject],
    pub blind: [bool; 2],
}

#[derive(Default)]
pub struct MapEngine {
    tracker: Tracker,
    pose: Pose,
    behind: VecDeque<PoseAt>,
    previous: Option<RoadFrame>,
    displayed: [Option<EngineSlot>; 5],
    pair_locks: [Option<selector::PairLock>; 2],
    rear_display: [rear_display::SideDisplay; 2],
    lateral: HashMap<u64, LateralDisplay>,
    road_disagreement: Option<f64>,
    last_lane_change: Option<f64>,
    pending_pedestrians: Vec<PedestrianObservation>,
    ego_lane_coord: f64,
    time: Option<f64>,
}

impl MapEngine {
    pub fn objects(&self) -> &[TrackedObject] {
        &self.tracker.objects
    }

    pub fn ego_lane_coord(&self) -> f64 {
        self.ego_lane_coord
    }

    pub fn observe_pedestrian(&mut self, obs: PedestrianObservation) {
        self.pending_pedestrians.push(obs);
    }

    pub(crate) fn display_motion(&self, vid: u64) -> Option<(bool, f64)> {
        self.lateral.get(&vid).map(LateralDisplay::diagnostic)
    }

    pub(crate) fn break_radar_continuity(&mut self) {
        for object in &mut self.tracker.objects {
            object.stationary_radar_continuous = false;
        }
    }

    pub(crate) fn kind_frame(&mut self, time_ns: u64) {
        self.tracker.kinds.front_frame(time_ns);
    }

    pub(crate) fn kind_wanted(&mut self, vid: u32, now: f64) -> bool {
        self.tracker.kinds.wanted(vid, now)
    }

    /// Associate classifier answers with the radar objects at their capture time.
    pub fn observe_front_kinds(
        &mut self,
        published: u64,
        camera: &FrontCam,
        results: &[KindResult],
    ) {
        self.tracker.front_kinds(published, camera, results);
    }

    /// Associate both front and side-view classifier answers with capture-time radar objects.
    pub fn observe_kinds(
        &mut self,
        published: u64,
        camera: &FrontCam,
        results: &[KindResult],
        side_cands: impl Fn(u64, crate::vision::Side) -> Option<Vec<det_rules::Cand>>,
    ) {
        self.tracker.observe_kinds(published, camera, results, side_cands);
    }

    /// Body detections support a pair of currently measured slow side vehicles.
    pub fn observe_front_boxes(&mut self, capture: u64, camera: &FrontCam, boxes: &[[f64; 4]]) {
        self.tracker.front_boxes(capture, camera, boxes);
    }

    pub fn step(&mut self, input: EngineInput<'_>) -> EngineOutput<'_> {
        let now = input.time;
        let speed = input.speed.max(0.);
        if !now.is_finite() || !input.speed.is_finite() || self.time.is_some_and(|t| t >= now) {
            return EngineOutput::default();
        }
        // A radar outage breaks prediction. Identity numbers still advance within this engine.
        let dt = self.time.map_or(0., |t| now - t);
        if dt > 0.25 {
            self.tracker.reset();
            self.pose = Pose::default();
            self.behind.clear();
            self.previous = None;
            self.displayed = [None; 5];
            self.pair_locks = [None; 2];
            self.rear_display = Default::default();
            self.lateral.clear();
            self.road_disagreement = None;
            self.last_lane_change = None;
            self.pending_pedestrians.clear();
            self.ego_lane_coord = 0.;
        }
        let previous_pose = self.pose.current();
        self.pose.advance(now, speed, input.yaw);
        let ego = self.pose.current();
        if self.time.is_some()
            && dt <= 0.25
            && self
                .behind
                .back()
                .is_none_or(|p| previous_pose.s - p.s >= 0.6)
        {
            self.behind.push_back(previous_pose);
        }
        self.behind.retain(|p| ego.s - p.s <= 20.);

        let road = input
            .road
            .filter(|r| (0. ..=0.25).contains(&(now - r.time)));
        if road.is_some_and(|r| r.changing_lanes) {
            self.last_lane_change = Some(now);
        }
        let model_pose = road.map_or(ego, |r| self.pose.at(r.time));
        let mut frame = RoadFrame::new_with_previous(
            road.map_or(&[], |r| r.path),
            road.and_then(|r| r.lines),
            model_pose,
            ego,
            self.behind.iter().copied(),
            road.is_some_and(|r| r.changing_lanes),
            self.previous.as_ref().map(|p| (p, previous_pose)),
        );
        if let Some(previous) = self.previous.take() {
            if road.is_some() && frame.disagrees(&previous, previous_pose, ego) {
                let since = *self.road_disagreement.get_or_insert(now);
                if now - since <= 0.20 {
                    // Keep the last plausible road briefly while the new forecast is uncorroborated.
                    frame = previous.clone().advance(previous_pose, ego);
                }
            } else {
                self.road_disagreement = None;
            }
            self.tracker
                .predict(dt, speed, &previous, &frame, previous_pose, ego);
        }
        self.ego_lane_coord = frame.ego_lane_coord();
        let visible = self.tracker.radar(now, speed, &frame, ego, input.radar);
        let primary = self.tracker.lead(now, input.lead, &frame);
        for &observation in input.vision {
            let age = now - observation.time;
            if !(0. ..=1.5).contains(&age)
                || !observation.x.is_finite()
                || !observation.y.is_finite()
            {
                continue;
            }
            let capture = self.pose.at(observation.time);
            if age > 0.5 && (capture.t > observation.time || ego.s - capture.s > 1.5) {
                continue;
            }
            let (wx, wy) = capture.to_world(observation.x, -observation.y);
            let (x, y) = ego.to_ego(wx, wy);
            self.tracker.vision(now, speed, &frame, observation, x, -y, |lane, rx, ry| {
                road.is_none_or(|r| road_frame::inside_outer_line(r, model_pose, ego, lane, rx, ry))
            });
        }
        for ped in self.pending_pedestrians.drain(..) {
            let age = now - ped.time;
            if !(0. ..=1.5).contains(&age) || !ped.x.is_finite() || !ped.y.is_finite() {
                continue;
            }
            let capture = self.pose.at(ped.time);
            if age > 0.5 && (capture.t > ped.time || ego.s - capture.s > 1.5) {
                continue;
            }
            let (wx, wy) = capture.to_world(ped.x, -ped.y);
            let (x, y) = ego.to_ego(wx, wy);
            self.tracker.observe_pedestrian(ped.time, speed, &frame, ped.confidence, x, -y);
        }
        let previous_rear = [3, 4].map(|i| self.displayed[i].map(|slot| slot.obj_id));
        self.tracker.lifecycle(now, dt.min(0.25), speed, &frame, &visible, input.blind, previous_rear);
        let changing = self.last_lane_change.is_some_and(|t| now - t <= 0.32);
        let sharp_curve = input.yaw.is_some_and(|yaw| yaw.abs() / speed.max(3.) > 0.0025);
        let corner = input.yaw.filter(|yaw| !changing && speed >= 3. && yaw.abs() >= 0.065)
            .and_then(|yaw| road.map(|road| CornerLane { road, model_pose, ego, curvature: -yaw / speed }));
        let mut slots = selector::select(
            now,
            speed,
            changing,
            sharp_curve,
            &frame,
            &mut self.tracker.objects,
            self.displayed,
            &mut self.pair_locks,
            input.blind,
            primary,
            |ff, o| {
                let Some(corner) = &corner else { return false };
                if !ff.radar_seen || o.s <= ff.s || o.evidence.last_radar_t != Some(now)
                    || o.same_track_streak < 4
                {
                    return false;
                }
                let Some(lead) = input.radar.iter().find(|p| Some(p.track_id) == ff.last_track_id
                    && p.source == 0 && p.measured && p.track_state >= 2) else { return false };
                let [x, y, _] = o.last_radar_motion;
                corner.same_lane_ahead((f64::from(lead.d_rel), -f64::from(lead.y_rel)),
                                       (x, y), ff.lane_u, o.lane_u)
            },
            |side, o| {
                road.is_some_and(|r| {
                    let (x, y) = o.cartesian(&frame);
                    road_frame::outside_road_edge(r, model_pose, ego, side, x, y)
                })
            },
        );
        self.lateral.retain(|_, display| now - display.time <= 1.);
        for (index, slot) in slots.iter_mut().enumerate() {
            let Some(slot) = slot else { continue };
            let old = self.displayed.iter().flatten().find(|old| old.obj_id == slot.obj_id);
            if let Some(old) = old {
                // A refreshed path can move the Cartesian projection even while radar range is
                // steady. Allow the measured relative speed plus a bounded position innovation.
                let predicted = old.x + old.v_s * dt;
                slot.x = predicted + (slot.x - predicted).clamp(-1.25, 1.25);
                if matches!(index, 1 | 2) {
                    slot.x = slot.x.min(selector::SIDE_FRONT_LIMIT);
                }
            }
            let object = self.tracker.objects.iter().find(|o| o.obj_id == slot.obj_id).unwrap();
            let center = frame.lane_center(slot.s, slot.lane_u.round());
            let is_adjacent = matches!(index, 1 | 2);
            let adjacent_inner_snap = is_adjacent
                && old.is_none()
                && slot.ff_handoff_until.is_none()
                && slot.lane_u.round().abs() == 1.
                && object.v_d.abs() < 0.6;
            let display = self.lateral.entry(slot.obj_id)
                .or_insert_with(|| LateralDisplay::new(now, slot.d, center, adjacent_inner_snap));
            let (moving, anchor) = display.diagnostic();
            // Close, receding side vehicles can return a radar reflection from the inner corner.
            // Keep the settled anchor while the vehicle remains in the same adjacent lane.
            let corner_echo = index != 0 && !changing && !moving
                && slot.s.abs() <= 5. && slot.v_s < -selector::ALONGSIDE_SPEED
                && old.is_some_and(|old| old.lane_u.round() == slot.lane_u.round())
                && slot.lane_u.round().abs() == 1. && object.d * anchor > 0.
                && object.d.abs() < anchor.abs();
            let handoff = index >= 3
                && self.displayed[index - 2].is_some_and(|front| front.obj_id == slot.obj_id);
            let observed = !handoff && (object.evidence.last_vision_t == Some(now)
                || object.evidence.last_radar_t == Some(now) && !corner_echo);
            slot.display_d = display.update(
                now,
                observed.then_some((object.d, object.v_d)),
                center,
                slot.x,
                is_adjacent,
            );
        }
        let selected = slots;
        for (side, display) in self.rear_display.iter_mut().enumerate() {
            display.update(now, side, input.blind[side], self.displayed, &mut slots);
        }
        self.displayed = selected;
        self.previous = Some(frame);
        self.time = Some(now);
        EngineOutput {
            slots,
            objects: &self.tracker.objects,
            blind: input.blind,
        }
    }
}

#[cfg(test)]
mod tests;
