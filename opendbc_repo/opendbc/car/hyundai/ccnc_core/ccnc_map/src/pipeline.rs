//! The cereal adapter for the map process and its replay.

use std::collections::VecDeque;

use crate::cmap::{MapOut, STATUS_BSD_LEFT, STATUS_BSD_RIGHT, STATUS_REAR_ANIMATED};
use crate::engine::{
    EngineInput, MapEngine, PedestrianObservation, RoadObservation, TrackedObject, VehicleKind,
    VisionObservation,
};
use crate::input::{
    live_tracks::{self, EVENT_LIVE_TRACKS},
    map_events::{CarState, DisplayLead, Event, CAR_STATE, LIVE_POSE, RADAR_STATE},
    model_v2::{self, Model, EVENT_MODEL_V2},
};
use crate::radar::Result;
use crate::slots::{Slot, VISION_CONFIRMS, VISION_NONE, VISION_ONLY};
use crate::vision::{
    self,
    kind::{FrontCam, KIND_CAR, KIND_TRUCK, SIDE_KINDS},
    Cand, FrameView, FrontBox, Geometry, Side, Skipped,
};
use crate::world::{STATUS_BLINKER, STATUS_MAP_UNKNOWN, STATUS_NO_LIVE_POSE};

pub const STALE_NS: u64 = 250_000_000;
const YAW_MAX_AGE_NS: u64 = 500_000_000;
/// A running vision daemon reports status about once a second.
const VISION_ALIVE_NS: u64 = 3_000_000_000;
/// High-pass dynamic pitch filter time constant (s) to track road grade while passing braking/accel dive.
const PITCH_LP_TAU_S: f64 = 2.0;
/// Cap dynamic pitch compensation to +/- 1.5 degrees.
const MAX_DYNAMIC_PITCH_RAD: f64 = 1.5 * std::f64::consts::PI / 180.0;
/// Half-width of a motorcycle (0.8 m width) vs passenger car HALFW (0.9 m).
const MOTO_HALFW: f64 = 0.4;
/// Number of recent side frames kept for asynchronous SIDE_KINDS matching.
const RECENT_SIDE_FRAMES: usize = 6;

fn seconds(t: u64) -> f64 {
    t as f64 * 1e-9
}

fn current(stamp: u64, input: u64) -> bool {
    stamp.abs_diff(input) <= STALE_NS
}

fn box_overlap_ratio(a: [f64; 4], b: [f64; 4]) -> f64 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area_a = ((a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)).max(1.0);
    (iw * ih) / area_a
}

struct Road {
    stamp: u64,
    path: Vec<(f64, f64)>,
    model: Option<Model>,
    changing_lanes: bool,
}

pub struct Pipeline {
    engine: MapEngine,
    generation_base: u32,
    generation: u32,
    invalid_motion: bool,
    last_car_stamp: Option<u64>,
    last_tracks: Option<u64>,
    tracks_valid: bool,
    car: Option<(u64, CarState)>,
    yaw: Option<(u64, f64)>,
    pitch_lp: Option<(u64, f64)>,
    pitch_hf: Option<(u64, f64)>,
    road: Option<Road>,
    lead: Option<(u64, Option<DisplayLead>)>,
    geometry: Option<Geometry>,
    recent_side: VecDeque<(u64, Side, Vec<Cand>)>,
    vision: Vec<VisionObservation>,
    vision_seen: Option<u64>,
}

impl Pipeline {
    pub fn new(generation_base: u32) -> Self {
        Self {
            engine: MapEngine::default(),
            generation_base,
            generation: 0,
            invalid_motion: false,
            last_car_stamp: None,
            last_tracks: None,
            tracks_valid: false,
            car: None,
            yaw: None,
            pitch_lp: None,
            pitch_hf: None,
            road: None,
            lead: None,
            geometry: None,
            recent_side: VecDeque::new(),
            vision: Vec::new(),
            vision_seen: None,
        }
    }

    fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.engine = MapEngine::default();
        self.vision.clear();
        self.recent_side.clear();
        self.invalid_motion = false;
    }

    fn yaw_at(&self, stamp: u64) -> Option<f64> {
        self.yaw
            .filter(|(t, _)| stamp.abs_diff(*t) <= YAW_MAX_AGE_NS)
            .map(|(_, y)| y)
    }

    fn dynamic_pitch_at(&self, stamp: u64) -> f64 {
        self.pitch_hf
            .filter(|(t, _)| stamp.abs_diff(*t) <= YAW_MAX_AGE_NS)
            .map_or(0., |(_, dp)| dp)
    }

    fn base_status(&self, stamp: u64) -> u8 {
        let no_live_pose = self.yaw_at(stamp).is_none();
        let map_unknown = self.road.as_ref().is_none_or(|r| !current(stamp, r.stamp));
        let blinker = self.car.is_some_and(|(_, c)| c.can_valid && c.blinker);
        (u8::from(no_live_pose) * STATUS_NO_LIVE_POSE)
            | (u8::from(map_unknown) * STATUS_MAP_UNKNOWN)
            | (u8::from(blinker) * STATUS_BLINKER)
    }

    fn empty(&self) -> MapOut {
        let stamp = self.last_tracks.unwrap_or(0);
        MapOut {
            radar_log_mono_time: stamp,
            generation: self.generation_base.wrapping_add(self.generation),
            status: self.base_status(stamp),
            ego_lane_coord: 0.,
            slots: [None; 5],
            valid: false,
        }
    }

    /// Publish one invalidation when the last radar input times out, then let the topic go quiet.
    pub fn expire(&mut self, now: u64) -> Option<MapOut> {
        let stamp = self.last_tracks?;
        if !self.tracks_valid || now.saturating_sub(stamp) <= STALE_NS {
            return None;
        }
        self.tracks_valid = false;
        self.reset();
        Some(self.empty())
    }

    pub fn kinds(&mut self, stamp: u64, results: &[vision::KindResult], calibration: &[f32; 7]) {
        self.vision_seen = Some(self.vision_seen.map_or(stamp, |seen| seen.max(stamp)));
        if !results.is_empty() {
            let dp = self.dynamic_pitch_at(stamp);
            if let Some(camera) = front_camera(calibration, dp) {
                let recent = &self.recent_side;
                self.engine.observe_kinds(stamp, &camera, results, |t, side| {
                    recent
                        .iter()
                        .find(|(ts, s, _)| *ts == t && *s == side)
                        .map(|(_, _, c)| c.clone())
                });
            }
        }
    }

    pub fn front_frame(&mut self, time_ns: u64, calibration: &[f32; 7], boxes: &[FrontBox]) {
        let dp = self.dynamic_pitch_at(time_ns);
        let Some(camera) = front_camera(calibration, dp) else {
            return;
        };
        self.engine.kind_frame(time_ns);
        let vehicle_boxes: Vec<&FrontBox> = boxes
            .iter()
            .filter(|b| det_rules::VEH.contains(&i32::from(b.detection.class_id)))
            .collect();
        let bodies: Vec<_> = vehicle_boxes
            .iter()
            .filter(|b| b.detection.confidence >= f64::from(0.55_f32))
            .map(|b| b.detection.xyxy)
            .collect();
        self.engine.observe_front_boxes(time_ns, &camera, &bodies);
        for b in vehicle_boxes.iter().filter(|b| !b.cut) {
            let [x1, _, x2, bottom] = b.detection.xyxy;
            if let Some((x, y)) = camera.ground_from_pixel((x1 + x2) / 2., bottom) {
                if (det_rules::CAM2B..=130.).contains(&x) {
                    self.vision.push(VisionObservation {
                        time: seconds(time_ns),
                        x: x - det_rules::CAM2B,
                        y,
                        kind: if b.detection.class_id == 3 {
                            VehicleKind::Motorcycle
                        } else {
                            VehicleKind::Unknown
                        },
                        confidence: b.detection.confidence,
                        position: true,
                    });
                }
            }
        }
        // Person (COCO class 0) detections that do not overlap a vehicle box veto roadside radar false positives.
        for b in boxes
            .iter()
            .filter(|b| !b.cut && b.detection.class_id == 0 && b.detection.confidence >= 0.45)
        {
            if vehicle_boxes
                .iter()
                .any(|v| box_overlap_ratio(b.detection.xyxy, v.detection.xyxy) >= 0.3)
            {
                continue;
            }
            let [x1, _, x2, bottom] = b.detection.xyxy;
            if let Some((x, y)) = camera.ground_from_pixel((x1 + x2) / 2., bottom) {
                if (det_rules::CAM2B..=40.).contains(&x) {
                    self.engine.observe_pedestrian(PedestrianObservation {
                        time: seconds(time_ns),
                        x: x - det_rules::CAM2B,
                        y,
                        confidence: b.detection.confidence,
                    });
                }
            }
        }
    }

    /// A wide frame is projected off its capture time before the engine handles association.
    pub fn vision(&mut self, stamp: u64, frame: &vision::Frame) -> std::result::Result<(), Skipped> {
        if !stamp.checked_sub(frame.time_ns).is_some_and(|age| age <= 1_500_000_000) {
            return Err(Skipped::Late);
        }
        let calib = vision::calib_of(&frame.calibration)
            .filter(|c| (0.5..=3.).contains(&c.height))
            .ok_or(Skipped::NoCalibration)?;
        let dp = self.dynamic_pitch_at(frame.time_ns);
        let geometry = self.geometry.as_mut().filter(|g| g.serves(&calib)).ok_or(Skipped::NeedsGeometry(calib))?;
        let cands = geometry.candidates_with_pitch(frame, &calib, dp);
        if SIDE_KINDS {
            if let FrameView::Side(side) = frame.view {
                if self.recent_side.len() == RECENT_SIDE_FRAMES {
                    self.recent_side.pop_front();
                }
                self.recent_side.push_back((frame.time_ns, side, cands.clone()));
            }
        }
        for (px, py, score) in geometry.person_points(frame, &calib, dp) {
            self.engine.observe_pedestrian(PedestrianObservation {
                time: seconds(frame.time_ns),
                x: px,
                y: py,
                confidence: score,
            });
        }
        self.vision.extend(cands.into_iter().filter(|c| !c.near03).map(|c| {
            // Motorcycle width is 0.8 m (half-width 0.4 m) rather than passenger car HALFW (0.9 m).
            let y = if c.class == 3 && !c.cut {
                let inset = det_rules::HALFW - MOTO_HALFW;
                if c.yc < 0. {
                    (c.yc + inset).min(0.)
                } else {
                    (c.yc - inset).max(0.)
                }
            } else {
                c.yc
            };
            VisionObservation {
                time: seconds(frame.time_ns),
                x: c.x,
                y,
                kind: match c.class {
                    2 => VehicleKind::Car,
                    3 => VehicleKind::Motorcycle,
                    // Raw side YOLOX bus/truck (5/7) must not bypass MNv3 + MIN_TRUCK_HEIGHT_M (2.05 m).
                    _ => VehicleKind::Unknown,
                },
                confidence: c.score,
                position: true,
            }
        }));
        Ok(())
    }

    pub fn set_vision_geometry(&mut self, geometry: Geometry) {
        self.geometry = Some(geometry);
    }

    /// Tracking evidence for replay diagnostics, keyed by the IDs present in the emitted MapOut.
    /// Current lateral animation for replay diagnostics (moving, settled anchor or displayed position).
    pub fn display_motion(&self, vid: u64) -> Option<(bool, f64)> {
        self.engine.display_motion(vid)
    }

    pub fn objects(&self) -> &[TrackedObject] {
        self.engine.objects()
    }

    pub fn ingest(&mut self, bytes: &[u8]) -> Result<Option<MapOut>> {
        let Some(event) = Event::read(bytes) else {
            return Ok(None);
        };
        let stamp = event.stamp;
        match event.which {
            LIVE_POSE if event.valid && self.yaw.is_none_or(|(last, _)| stamp > last) => {
                if let Some(yaw) = event.yaw() {
                    self.yaw = Some((stamp, yaw));
                }
                if let Some(pitch) = event.pitch() {
                    let lp = match self.pitch_lp {
                        Some((last, prev)) if stamp > last && stamp - last <= 2_000_000_000 => {
                            let dt = seconds(stamp - last);
                            let alpha = 1. - (-dt / PITCH_LP_TAU_S).exp();
                            prev + alpha * (pitch - prev)
                        }
                        _ => pitch,
                    };
                    self.pitch_lp = Some((stamp, lp));
                    let hf = (pitch - lp).clamp(-MAX_DYNAMIC_PITCH_RAD, MAX_DYNAMIC_PITCH_RAD);
                    self.pitch_hf = Some((stamp, hf));
                }
            }
            CAR_STATE if self.car.is_none_or(|(last, _)| stamp > last) => {
                let mut car = event.car_state().unwrap_or_default();
                car.can_valid &= event.valid && car.speed.is_finite();
                if car.can_valid {
                    let yaw = self.yaw_at(stamp);
                    let invalid = car.reverse
                        || !car.speed.is_finite()
                        || car.speed < -0.5
                        || yaw.is_some_and(|y| y.abs() > 1.);
                    if invalid {
                        if !self.invalid_motion {
                            self.reset();
                        }
                        self.invalid_motion = true;
                    } else {
                        self.invalid_motion = false;
                        if self
                            .last_car_stamp
                            .is_some_and(|last| stamp.saturating_sub(last) > STALE_NS)
                        {
                            self.reset();
                        }
                    }
                    self.last_car_stamp = Some(stamp);
                }
                self.car = Some((stamp, car));
            }
            EVENT_MODEL_V2 if event.valid && self.road.as_ref().is_none_or(|r| stamp > r.stamp) => {
                let model = model_v2::parse(bytes);
                let changing_lanes = model_v2::parse_road_path(bytes).is_none_or(|(_, changing, _)| changing);
                if let Some(((x, y), _)) = model_v2::parse_path(bytes) {
                    self.road = Some(Road {
                        stamp,
                        path: x.into_iter().zip(y).collect(),
                        model,
                        changing_lanes,
                    });
                }
            }
            RADAR_STATE if self.lead.is_none_or(|(last, _)| stamp >= last) => {
                let primary = event.valid.then(|| event.primary_display_lead()).flatten();
                self.lead = Some((stamp, primary));
            }
            EVENT_LIVE_TRACKS if self.last_tracks.is_none_or(|last| stamp > last) => {
                if self.last_tracks.is_some_and(|last| !current(stamp, last)) && self.tracks_valid {
                    self.reset();
                }
                self.last_tracks = Some(stamp);
                let map = if let Some(points) = event.valid.then(|| live_tracks::parse(bytes)).flatten() {
                    self.tracks_valid = true;
                    self.update_radar(stamp, &points)
                } else {
                    self.empty()
                };
                if !map.valid {
                    self.engine.break_radar_continuity();
                }
                self.vision.clear();
                return Ok(Some(map));
            }
            _ => {}
        }
        Ok(None)
    }

    fn update_radar(&mut self, stamp: u64, radar: &[live_tracks::RawPoint]) -> MapOut {
        let Some((_, car)) = self.car.filter(|(t, c)| c.can_valid && !c.reverse && current(stamp, *t)) else {
            return self.empty();
        };
        if self.invalid_motion {
            return self.empty();
        }
        let now = seconds(stamp);
        let yaw = self.yaw_at(stamp);
        let tracked = self.engine.step(EngineInput {
            time: now,
            speed: car.speed,
            yaw,
            road: self.road.as_ref().map(|r| RoadObservation {
                time: seconds(r.stamp),
                path: &r.path,
                lines: r.model.as_ref(),
                changing_lanes: r.changing_lanes,
            }),
            radar,
            lead: self.lead.and_then(|(t, lead)| lead.map(|p| (seconds(t), p))),
            vision: &self.vision,
            blind: car.blind,
        });
        let asking = self.vision_seen.is_some_and(|seen| stamp.saturating_sub(seen) <= VISION_ALIVE_NS);
        let (selected, blind) = (tracked.slots, tracked.blind);
        let slots = std::array::from_fn(|index| {
            let s = selected[index]?;
            let vid = u32::try_from(s.obj_id).ok()?;
            let kind = match s.kind {
                VehicleKind::Car | VehicleKind::Motorcycle => KIND_CAR,
                VehicleKind::Heavy => KIND_TRUCK,
                VehicleKind::Unknown => 0,
            };
            let vision = if s.last_track_id.is_none() {
                VISION_ONLY
            } else if s.camera_seen {
                VISION_CONFIRMS
            } else {
                VISION_NONE
            };
            Some(Slot {
                vid,
                distance: s.x as f32,
                lateral: -s.display_d as f32,
                detect: if s.v_s > 2. { 1 } else { 2 },
                lane_coord_rel: s.lane_u as f32,
                v_rel: s.v_s as f32,
                kind,
                kind_confirmed: kind != 0,
                kind_wanted: index < 3 && asking && s.last_track_id.is_some() && self.engine.kind_wanted(vid, now),
                vision,
            })
        });
        let ego = self.engine.ego_lane_coord();
        MapOut {
            ego_lane_coord: if ego.is_finite() { ego as f32 } else { 0. },
            status: self.base_status(stamp)
                | (u8::from(blind[0]) * STATUS_BSD_LEFT)
                | (u8::from(blind[1]) * STATUS_BSD_RIGHT)
                | (u8::from(selected[3].is_some_and(|s| s.rear_animated)) * STATUS_REAR_ANIMATED[0])
                | (u8::from(selected[4].is_some_and(|s| s.rear_animated)) * STATUS_REAR_ANIMATED[1]),
            slots,
            valid: true,
            ..self.empty()
        }
    }
}

fn front_camera(values: &[f32; 7], delta_pitch: f64) -> Option<FrontCam> {
    let [roll, pitch, yaw, _, _, _, height] = values.map(f64::from);
    (0.5..=3.).contains(&height).then(|| FrontCam::new([roll, pitch - delta_pitch, yaw], height)).flatten()
}
