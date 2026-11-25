//! Road-relative geometry for the map process. The in-card display uses its existing radar map.

mod lane_change;
mod lane_map;
mod pose;

use lane_change::{EgoLaneFilter, EgoLaneInput};
pub use lane_change::LaneChange;
pub use lane_map::{near_heading, LaneMap, MODEL_GRID};
pub use pose::{Pose, PoseAt};

use crate::{input::model_v2::Model, radar::MapState};
use lane_map::CameraLanes;

pub const RADAR_LAG: f64 = 0.2;
pub const STATUS_FROZEN: u8 = 1;
pub const STATUS_NO_LIVE_POSE: u8 = 2;
pub const STATUS_MAP_UNKNOWN: u8 = 4;
pub const STATUS_BLINKER: u8 = 8;
const INPUT_GAP: f64 = 0.25;
const LIVE_POSE_MAX_AGE: f64 = 0.5;
const MAX_EGO_LANE_RATE: f64 = 1.;
/// Ego is changing lanes more than this far off its lane center ...
const EGO_CHANGE_CENTER: f64 = 0.25;
/// ... and for this long after.
const EGO_CHANGE_HOLD_S: f64 = 3.;
/// The lane map can miss ego for a few frames while a road edge or the lane widths jitter past it. Ego keeps its last
/// coordinate this long (`World::lane_held`); longer, the map does not know where ego is.
const EGO_HOLD_S: f64 = 0.5;

const CAMERA_EGO_MIN_PROB: f64 = 0.20;
const UNFILTERED_CAMERA_PROB: f32 = 0.4;
const CAMERA_EGO_GAIN: f64 = 0.8;
const UNFILTERED_MAX_BIAS_U: f64 = 0.25;
const BIAS_TAPER_START_U: f64 = 0.18;
const BIAS_TAPER_WIDTH_U: f64 = 0.12;
const FILTERED_BLEND_OUTER_U: f64 = 0.32;
const FILTERED_BLEND_WIDTH_U: f64 = 0.14;

pub struct World {
    pub pose: Pose,
    pub lanes: LaneMap,
    lane_change: LaneChange,
    ego_filter: EgoLaneFilter,
    yaw: Option<(f64, f64)>,
    last_model: Option<f64>,
    last_radar: Option<f64>,
    u_ego: Option<(f64, f64)>,
    /// Rate-limited measurement track for lane interiors.
    u_ego_raw_track: Option<(f64, f64)>,
    /// When the lane map last placed ego (`u_ego` is held after it).
    u_ego_mapped: f64,
    camera_lane_bias: f64,
    camera_lane_bias_unfiltered: f64,
    last_min_inner_prob: f64,
    camera_lanes: Option<CameraLanes>,
    generation: u32,
    radar_lag: f64,
    freeze_enabled: bool,
    invalid_motion: bool,
    no_live_pose: bool,
    blinker: bool,
    ego_change_until: f64,
}

impl Default for World {
    fn default() -> Self {
        // The same-car Q2 comparison had fewer reversals with freeze disabled.
        Self::new(RADAR_LAG, false)
    }
}

impl World {
    /// The map process can supply CcncRadarLagMs in seconds. Replay can disable motion freeze.
    pub fn new(radar_lag: f64, freeze_enabled: bool) -> Self {
        let radar_lag = if radar_lag.is_finite() {
            radar_lag.clamp(0., 2.)
        } else {
            RADAR_LAG
        };
        Self {
            pose: Pose::default(),
            lanes: LaneMap::default(),
            lane_change: LaneChange::default(),
            ego_filter: EgoLaneFilter::default(),
            yaw: None,
            last_model: None,
            last_radar: None,
            u_ego: None,
            u_ego_raw_track: None,
            u_ego_mapped: f64::NEG_INFINITY,
            camera_lane_bias: 0.,
            camera_lane_bias_unfiltered: 0.,
            last_min_inner_prob: 0.,
            camera_lanes: None,
            generation: 0,
            radar_lag,
            freeze_enabled,
            invalid_motion: false,
            no_live_pose: true,
            blinker: false,
            ego_change_until: f64::NEG_INFINITY,
        }
    }

    pub fn car_state(&mut self, t: f64, v_ego: f64, reverse: bool) {
        if !t.is_finite() || self.pose.stamp().is_some_and(|last| t <= last) {
            return;
        }
        let yaw = self.yaw_at(t);
        if reverse || !v_ego.is_finite() || v_ego < -0.5 || yaw.is_some_and(|y| y.abs() > 1.) {
            if !self.invalid_motion {
                self.reset();
            }
            self.invalid_motion = true;
            return;
        }
        self.invalid_motion = false;
        if self.pose.stamp().is_some_and(|last| t - last > INPUT_GAP) {
            self.reset();
        }
        self.pose.advance(t, v_ego, yaw);
        self.update_pose_status(t);
    }

    /// Yaw rate from -angularVelocityDevice.z in rad/s (left positive).
    pub fn live_pose(&mut self, t: f64, yaw: f64) {
        if t.is_finite() && yaw.is_finite() && self.yaw.is_none_or(|(last, _)| t > last) {
            self.yaw = Some((t, yaw));
        }
    }

    pub fn blinker(&mut self, on: bool) {
        self.blinker = on;
    }

    pub fn model(&mut self, t: f64, model: &Model) {
        self.model_with_walls(t, model, None);
    }

    pub fn model_with_walls(&mut self, t: f64, model: &Model, radar: Option<&MapState>) {
        if self.invalid_motion || !t.is_finite() || self.last_model.is_some_and(|last| t <= last) {
            return;
        }
        // Task 2's wall geometry is in ego coordinates. Bands follow the static walls in this list.
        let walls = radar
            .and_then(|m| m.wall_geometry.get(..m.walls.len()))
            .unwrap_or_default();
        if let Some((heading, curvature)) = near_heading(model) {
            self.pose.observe_lane_heading(t, heading, curvature);
        }
        self.update_pose_status(t);
        self.lanes.ingest_with_walls(
            t,
            model,
            &self.pose,
            self.lane_change.frozen(),
            self.no_live_pose,
            walls,
        );
        self.camera_lanes = CameraLanes::observe(model, self.pose.at(t), t);
        self.camera_lane_bias = 0.;
        self.camera_lane_bias_unfiltered = 0.;
        self.last_min_inner_prob = model
            .lane_line_probs
            .get(1..3)
            .map_or(0., |p| f64::from(p[0].min(p[1])));
        if !self.lane_change.frozen() && self.last_min_inner_prob > CAMERA_EGO_MIN_PROB {
            let at = self.pose.at(t);
            if let Some((camera, map)) = self
                .lanes
                .camera_ego(model)
                .zip(self.lanes.coord(at.s, (at.x, at.y)))
            {
                let bias = camera - map;
                let taper = 1. - ((bias.abs() - BIAS_TAPER_START_U) / BIAS_TAPER_WIDTH_U).clamp(0., 1.);
                self.camera_lane_bias = CAMERA_EGO_GAIN * bias * taper;
                if model.lane_line_probs[1..3].iter().all(|&p| p > UNFILTERED_CAMERA_PROB)
                    && bias.abs() <= UNFILTERED_MAX_BIAS_U
                {
                    self.camera_lane_bias_unfiltered = CAMERA_EGO_GAIN * bias;
                }
            }
        }
        self.last_model = Some(t);
        self.update_lane(t);
    }

    /// Advance the lane-change state at the radar cadence and detect a broken input stream.
    pub fn radar(&mut self, t: f64) {
        if self.invalid_motion || !t.is_finite() || self.last_radar.is_some_and(|last| t <= last) {
            return;
        }
        if self.last_radar.is_some_and(|last| t - last > INPUT_GAP) {
            self.reset();
        }
        self.last_radar = Some(t);
        self.update_pose_status(t);
        self.update_lane(t);
    }

    pub fn radar_point_world(&self, stamp: f64, x: f64, y: f64) -> (f64, f64) {
        self.radar_pose(stamp).to_world(x, y)
    }

    pub(crate) fn accepts_radar(&self) -> bool {
        !self.invalid_motion
    }

    pub(crate) fn radar_stamp(&self) -> Option<f64> {
        self.last_radar
    }

    pub(crate) fn radar_pose(&self, stamp: f64) -> PoseAt {
        self.pose.at(stamp - self.radar_lag)
    }

    /// Locate a world-space radar point after lateral reflection correction.
    pub fn vehicle_lane_coord(&self, t: f64, s: f64, point: (f64, f64)) -> Option<f64> {
        let mapped = self.lanes.coord(s, point)?;
        if self.lane_change.frozen() {
            return Some(mapped);
        }
        let weight = self.lanes.camera_weight(s, mapped);
        let Some(camera) = self
            .camera_lanes
            .as_ref()
            .filter(|c| weight > 0. && (0. ..INPUT_GAP).contains(&(t - c.stamp)))
        else {
            return Some(mapped);
        };
        let ego = self.pose.at(t);
        let camera_offset = camera
            .coord(point)
            .zip(camera.coord((ego.x, ego.y)))
            .map(|(car, near)| self.u_ego() + car - near - mapped)
            .filter(|delta| delta.abs() <= 0.5);
        Some(mapped + weight * camera_offset.unwrap_or(0.))
    }

    pub fn u_ego(&self) -> f64 {
        self.u_ego.map_or(f64::NAN, |(_, u)| u)
    }

    pub(crate) fn committed_lane(&self) -> Option<i32> {
        self.ego_filter.committed_lane()
    }

    pub(crate) fn last_commit(&self) -> Option<(f64, i32, i32)> {
        self.ego_filter.last_commit()
    }

    pub(crate) fn approaching_boundary(&self) -> bool {
        self.ego_filter.approaching()
    }

    /// Ego is changing lanes: it was off its lane center within `EGO_CHANGE_HOLD_S`. This process runs with the lane
    /// change freeze off, so `STATUS_FROZEN` cannot tell. The blinker does not count: it is on in 17% of the 65
    /// replayed segments, and the cut-outs it covers kept their old slot lag.
    pub fn ego_changing(&self) -> bool {
        self.u_ego.is_some_and(|(t, _)| t <= self.ego_change_until)
    }
    /// Ego's coordinate is the last one the lane map gave, held over a short miss. The cars it cannot place keep their
    /// coordinates meanwhile.
    pub fn lane_held(&self) -> bool {
        self.u_ego.is_some_and(|(t, _)| t > self.u_ego_mapped)
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn status(&self) -> u8 {
        self.lane_change.status()
            | if self.no_live_pose {
                STATUS_NO_LIVE_POSE
            } else {
                0
            }
            | if self.u_ego.is_none() {
                STATUS_MAP_UNKNOWN
            } else {
                0
            }
            | if self.blinker { STATUS_BLINKER } else { 0 }
    }

    /// A vehicle layer using these coordinates must discard the old epoch on any generation change.
    pub fn reset(&mut self) {
        self.pose.clear();
        self.lanes.clear();
        self.lane_change = LaneChange::default();
        self.ego_filter.reset();
        self.last_model = None;
        self.last_radar = None;
        self.u_ego = None;
        self.u_ego_raw_track = None;
        self.u_ego_mapped = f64::NEG_INFINITY;
        self.camera_lane_bias = 0.;
        self.camera_lane_bias_unfiltered = 0.;
        self.last_min_inner_prob = 0.;
        self.camera_lanes = None;
        self.ego_change_until = f64::NEG_INFINITY;
        self.generation = self.generation.wrapping_add(1);
        self.invalid_motion = false;
    }

    pub(crate) fn yaw_at(&self, t: f64) -> Option<f64> {
        self.yaw
            .filter(|(stamp, _)| (0. ..=LIVE_POSE_MAX_AGE).contains(&(t - stamp)))
            .map(|(_, yaw)| yaw)
    }

    fn update_pose_status(&mut self, t: f64) {
        self.no_live_pose = self.yaw_at(t).is_none();
        if self.no_live_pose {
            self.lane_change = LaneChange::default();
            self.lanes.set_frozen(false);
        }
    }

    fn update_lane(&mut self, t: f64) {
        if self.u_ego.is_some_and(|(last, _)| t <= last) {
            return;
        }
        let at = self.pose.at(t);
        let coordinate = self.lanes.coord(at.s, (at.x, at.y));
        if coordinate.is_some() {
            self.u_ego_mapped = t;
        }
        let expired = coordinate.is_none() && t - self.u_ego_mapped > EGO_HOLD_S;
        let model_fresh = !self.lane_change.frozen()
            && self
                .last_model
                .is_some_and(|last| (0. ..INPUT_GAP).contains(&(t - last)));
        // Preserve the rate-limited measurement away from lane boundaries.
        let held_raw = self
            .u_ego_raw_track
            .filter(|_| !expired)
            .map(|(_, u)| (t, u));
        self.u_ego_raw_track = coordinate
            .map(|u| {
                let u = if model_fresh { u + self.camera_lane_bias_unfiltered } else { u };
                let u = if let Some((last, prior)) = self.u_ego_raw_track {
                    let limit = MAX_EGO_LANE_RATE * (t - last);
                    prior + (u - prior).clamp(-limit, limit)
                } else {
                    u
                };
                (t, u)
            })
            .or(held_raw);
        let filtered = self.ego_filter.step(EgoLaneInput {
            t,
            map_u: coordinate,
            raw_bias: if model_fresh { self.camera_lane_bias } else { 0. },
            min_inner_prob: if model_fresh { self.last_min_inner_prob } else { 0. },
            yaw: self.yaw_at(t),
            speed: self.pose.speed(),
            renumbers: self.lanes.renumbers(),
            map_unknown: expired,
        });
        // Blend continuously from the filtered coordinate at boundaries to the raw track in lane interiors.
        // At intersections and while re-locking after map loss, the filter has no trustworthy road lane.
        self.u_ego = if !expired && filtered.is_finite() && self.ego_filter.committed_lane().is_some() {
            let raw_u = self.u_ego_raw_track.map_or(filtered, |(_, u)| u);
            let boundary_distance = (filtered - filtered.round()).abs().min((raw_u - raw_u.round()).abs());
            let w_kf = ((FILTERED_BLEND_OUTER_U - boundary_distance) / FILTERED_BLEND_WIDTH_U).clamp(0., 1.);
            Some((t, raw_u + w_kf * (filtered - raw_u)))
        } else {
            self.u_ego_raw_track
        };
        if let Some((t, u)) = self.u_ego {
            if (u - (u.floor() + 0.5)).abs() > EGO_CHANGE_CENTER {
                self.ego_change_until = t + EGO_CHANGE_HOLD_S;
            }
        }
        let frozen = if self.freeze_enabled && !self.no_live_pose && self.u_ego.is_some() {
            self.lane_change.step(t, self.u_ego())
        } else {
            self.lane_change = LaneChange::default();
            false
        };
        self.lanes.set_frozen(frozen);
    }
}
