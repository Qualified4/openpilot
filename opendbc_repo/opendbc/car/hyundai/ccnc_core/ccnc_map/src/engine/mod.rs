//! Offline/shadow object tracking. The production map pipeline does not call this module.

mod object;
mod road_frame;
mod selector;
mod tracker;

pub use object::{Evidence, TrackedObject, VehicleKind};

use crate::input::{live_tracks::RawPoint, map_events::DisplayLead, model_v2::Model};
use crate::world::{Pose, PoseAt};
use road_frame::RoadFrame;
use std::collections::VecDeque;
use tracker::Tracker;

#[derive(Clone, Copy)]
pub struct RoadObservation<'a> {
    pub time: f64,
    /// Model position x/y pairs in metres, right positive.
    pub path: &'a [(f64, f64)],
    pub lines: Option<&'a Model>,
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
    pub v_s: f64,
    pub kind: VehicleKind,
    pub existence_p: f64,
    pub last_track_id: Option<u64>,
}

#[derive(Debug, Default)]
pub struct EngineOutput {
    /// FF, LF, RF, LR, RR.
    pub slots: [Option<EngineSlot>; 5],
    pub objects: Vec<TrackedObject>,
    pub blind: [bool; 2],
}

#[derive(Default)]
pub struct MapEngine {
    tracker: Tracker,
    pose: Pose,
    behind: VecDeque<PoseAt>,
    previous: Option<RoadFrame>,
    time: Option<f64>,
}

impl MapEngine {
    pub fn step(&mut self, input: EngineInput<'_>) -> EngineOutput {
        let now = input.time;
        let speed = input.speed.max(0.);
        if !now.is_finite() || !input.speed.is_finite() || self.time.is_some_and(|t| t >= now) {
            return EngineOutput::default();
        }
        // A radar outage breaks prediction. Identity numbers still advance within this engine.
        let dt = self.time.map_or(0., |t| now - t);
        if dt > 0.25 {
            self.tracker.objects.clear();
            self.pose = Pose::default();
            self.behind.clear();
            self.previous = None;
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
        let frame = RoadFrame::new(
            road.map_or(&[], |r| r.path),
            road.and_then(|r| r.lines),
            road.map_or(ego, |r| self.pose.at(r.time)),
            ego,
            self.behind.iter().copied(),
            road.is_some_and(|r| r.changing_lanes),
        );
        if let Some(previous) = &self.previous {
            self.tracker
                .predict(dt, speed, previous, &frame, previous_pose, ego);
        }
        let visible = self.tracker.radar(now, speed, &frame, input.radar);
        self.tracker.lead(now, input.lead, &frame);
        for &observation in input.vision {
            let age = now - observation.time;
            if !(0. ..=0.5).contains(&age)
                || !observation.x.is_finite()
                || !observation.y.is_finite()
            {
                continue;
            }
            let capture = self.pose.at(observation.time);
            let (wx, wy) = capture.to_world(observation.x, -observation.y);
            let (x, y) = ego.to_ego(wx, wy);
            self.tracker.vision(now, speed, &frame, observation, x, -y);
        }
        self.tracker
            .lifecycle(now, dt.min(0.25), &frame, &visible, input.blind);
        let slots = selector::select(now, &frame, &mut self.tracker.objects);
        self.previous = Some(frame);
        self.time = Some(now);
        EngineOutput {
            slots,
            objects: self.tracker.objects.clone(),
            blind: input.blind,
        }
    }
}

#[cfg(test)]
mod tests;
