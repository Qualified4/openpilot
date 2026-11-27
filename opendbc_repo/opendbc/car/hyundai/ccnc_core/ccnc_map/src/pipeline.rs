//! The cereal adapter for the map process and its replay.

use crate::cmap::{MapOut, STATUS_BSD_LEFT, STATUS_BSD_RIGHT};
use crate::engine::{EngineInput, MapEngine, RoadObservation, TrackedObject, VehicleKind, VisionObservation};
use crate::input::{
    live_tracks::{self, EVENT_LIVE_TRACKS},
    map_events::{CarState, DisplayLead, Event, CAR_STATE, LIVE_POSE, RADAR_STATE},
    model_v2::{self, Model, EVENT_MODEL_V2},
};
use crate::radar::Result;
use crate::slots::{Slot, VISION_CONFIRMS, VISION_NONE, VISION_ONLY};
use crate::vision::{self, kind::{FrontCam, KIND_CAR, KIND_TRUCK}, FrontBox, Geometry, Skipped};
use crate::world::World;

pub const STALE_NS: u64 = 250_000_000;
const YAW_MAX_AGE_NS: u64 = 500_000_000;
/// A running vision daemon reports status about once a second.
const VISION_ALIVE_NS: u64 = 3_000_000_000;

fn seconds(t: u64) -> f64 {
    t as f64 * 1e-9
}

fn current(stamp: u64, input: u64) -> bool {
    stamp.abs_diff(input) <= STALE_NS
}

struct Road {
    stamp: u64,
    path: Vec<(f64, f64)>,
    model: Option<Model>,
    changing_lanes: bool,
}

pub struct Pipeline {
    engine: MapEngine,
    // The existing world coordinate and status are still published for consumers of CcncMap.
    world: World,
    generation_base: u32,
    generation: u32,
    last_tracks: Option<u64>,
    tracks_valid: bool,
    car: Option<(u64, CarState)>,
    yaw: Option<(u64, f64)>,
    road: Option<Road>,
    lead: Option<(u64, Option<DisplayLead>)>,
    geometry: Option<Geometry>,
    vision: Vec<VisionObservation>,
    vision_seen: Option<u64>,
}

impl Pipeline {
    pub fn new(generation_base: u32) -> Self {
        Self {
            engine: MapEngine::default(),
            world: World::default(),
            generation_base,
            generation: 0,
            last_tracks: None,
            tracks_valid: false,
            car: None,
            yaw: None,
            road: None,
            lead: None,
            geometry: None,
            vision: Vec::new(),
            vision_seen: None,
        }
    }

    fn empty(&self) -> MapOut {
        MapOut {
            radar_log_mono_time: self.last_tracks.unwrap_or(0),
            generation: self.generation_base.wrapping_add(self.generation),
            status: self.world.status(),
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
        self.world.reset();
        self.sync_generation();
        Some(self.empty())
    }

    fn sync_generation(&mut self) {
        if self.generation != self.world.generation() {
            self.generation = self.world.generation();
            self.engine = MapEngine::default();
            self.vision.clear();
        }
    }

    pub fn kinds(&mut self, stamp: u64, results: &[vision::KindResult], calibration: &[f32; 7]) {
        self.vision_seen = Some(self.vision_seen.map_or(stamp, |seen| seen.max(stamp)));
        if !results.is_empty() {
            if let Some(camera) = front_camera(calibration) {
                self.engine.observe_front_kinds(stamp, &camera, results);
            }
        }
    }

    pub fn front_frame(&mut self, time_ns: u64, calibration: &[f32; 7], boxes: &[FrontBox]) {
        let Some(camera) = front_camera(calibration) else {
            return;
        };
        self.engine.kind_frame(time_ns);
        let vehicles = boxes.iter().filter(|b| det_rules::VEH.contains(&i32::from(b.detection.class_id)));
        let bodies: Vec<_> = vehicles.clone()
            .filter(|b| b.detection.confidence >= f64::from(0.55_f32))
            .map(|b| b.detection.xyxy)
            .collect();
        self.engine.observe_front_boxes(time_ns, &camera, &bodies);
        for b in vehicles.filter(|b| !b.cut) {
            let [x1, _, x2, bottom] = b.detection.xyxy;
            if let Some((x, y)) = camera.ground_from_pixel((x1 + x2) / 2., bottom) {
                if (det_rules::CAM2B..=130.).contains(&x) {
                    self.vision.push(VisionObservation {
                        time: seconds(time_ns),
                        x: x - det_rules::CAM2B,
                        y,
                        kind: VehicleKind::Unknown,
                        confidence: b.detection.confidence,
                        position: true,
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
        let geometry = self.geometry.as_mut().filter(|g| g.serves(&calib)).ok_or(Skipped::NeedsGeometry(calib))?;
        self.vision.extend(geometry.candidates(frame, &calib).into_iter().filter(|c| !c.near03).map(|c| {
            VisionObservation {
                time: seconds(frame.time_ns),
                x: c.x,
                y: c.yc,
                kind: match c.class {
                    2 => VehicleKind::Car,
                    5 | 7 => VehicleKind::Heavy,
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
    pub fn objects(&self) -> &[TrackedObject] {
        self.engine.objects()
    }

    pub fn ingest(&mut self, bytes: &[u8]) -> Result<Option<MapOut>> {
        let Some(event) = Event::read(bytes) else {
            return Ok(None);
        };
        let stamp = event.stamp;
        let now = seconds(stamp);
        match event.which {
            LIVE_POSE if event.valid && self.yaw.is_none_or(|(last, _)| stamp > last) => {
                if let Some(yaw) = event.yaw() {
                    self.yaw = Some((stamp, yaw));
                    self.world.live_pose(now, yaw);
                }
            }
            CAR_STATE if self.car.is_none_or(|(last, _)| stamp > last) => {
                let mut car = event.car_state().unwrap_or_default();
                car.can_valid &= event.valid && car.speed.is_finite();
                if car.can_valid {
                    self.world.blinker(car.blinker);
                    self.world.car_state(now, car.speed, car.reverse);
                    self.sync_generation();
                }
                self.car = Some((stamp, car));
            }
            EVENT_MODEL_V2 if event.valid && self.road.as_ref().is_none_or(|r| stamp > r.stamp) => {
                let model = model_v2::parse(bytes);
                let legacy_path = model_v2::parse_road_path(bytes);
                let changing_lanes = legacy_path.as_ref().is_none_or(|(_, changing, _)| *changing);
                if let Some(model) = &model {
                    self.world.model(now, model);
                    self.world.model_road_ahead(now, model, legacy_path);
                }
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
                    self.world.reset();
                    self.sync_generation();
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
        if !self.world.accepts_radar() {
            return self.empty();
        }
        let now = seconds(stamp);
        self.world.radar(now);
        self.sync_generation();
        let yaw = self.yaw.filter(|(t, _)| stamp.abs_diff(*t) <= YAW_MAX_AGE_NS).map(|(_, y)| y);
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
                VehicleKind::Car => KIND_CAR,
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
                lateral: -(if index < 3 { s.display_d } else { s.y }) as f32,
                detect: if s.v_s > 2. { 1 } else { 2 },
                lane_coord_rel: s.lane_u as f32,
                v_rel: s.v_s as f32,
                kind,
                kind_confirmed: kind != 0,
                kind_wanted: index < 3 && asking && s.last_track_id.is_some() && self.engine.kind_wanted(vid, now),
                vision,
            })
        });
        let ego = self.world.u_ego();
        MapOut {
            ego_lane_coord: if ego.is_finite() { ego as f32 } else { 0. },
            status: self.world.status()
                | (u8::from(blind[0]) * STATUS_BSD_LEFT)
                | (u8::from(blind[1]) * STATUS_BSD_RIGHT),
            slots,
            valid: true,
            ..self.empty()
        }
    }
}

fn front_camera(values: &[f32; 7]) -> Option<FrontCam> {
    let [roll, pitch, yaw, _, _, _, height] = values.map(f64::from);
    (0.5..=3.).contains(&height).then(|| FrontCam::new([roll, pitch, yaw], height)).flatten()
}
