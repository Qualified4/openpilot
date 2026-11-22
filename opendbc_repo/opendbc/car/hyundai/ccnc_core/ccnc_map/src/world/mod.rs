//! Road-relative geometry for the map process. The in-card display uses its existing radar map.

mod lane_change;
mod lane_map;
mod pose;

pub use lane_change::LaneChange;
pub use lane_map::{near_heading, LaneMap, MODEL_GRID};
pub use pose::{Pose, PoseAt};

use crate::{input::model_v2::Model, radar::MapState};

pub const RADAR_LAG: f64 = 0.2;
pub const STATUS_FROZEN: u8 = 1;
pub const STATUS_NO_LIVE_POSE: u8 = 2;
pub const STATUS_MAP_UNKNOWN: u8 = 4;
pub const STATUS_BLINKER: u8 = 8;
const INPUT_GAP: f64 = 0.25;
const LIVE_POSE_MAX_AGE: f64 = 0.5;
const MAX_EGO_LANE_RATE: f64 = 1.;

pub struct World {
    pub pose: Pose,
    pub lanes: LaneMap,
    lane_change: LaneChange,
    yaw: Option<(f64, f64)>,
    last_model: Option<f64>,
    last_radar: Option<f64>,
    u_ego: Option<(f64, f64)>,
    generation: u32,
    radar_lag: f64,
    freeze_enabled: bool,
    invalid_motion: bool,
    no_live_pose: bool,
    blinker: bool,
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
            yaw: None,
            last_model: None,
            last_radar: None,
            u_ego: None,
            generation: 0,
            radar_lag,
            freeze_enabled,
            invalid_motion: false,
            no_live_pose: true,
            blinker: false,
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
        self.pose.at(stamp - self.radar_lag).to_world(x, y)
    }

    pub fn u_ego(&self) -> f64 {
        self.u_ego.map_or(f64::NAN, |(_, u)| u)
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
        self.last_model = None;
        self.last_radar = None;
        self.u_ego = None;
        self.generation = self.generation.wrapping_add(1);
        self.invalid_motion = false;
    }

    fn yaw_at(&self, t: f64) -> Option<f64> {
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
        self.u_ego = coordinate.map(|u| {
            let u = if let Some((last, prior)) = self.u_ego {
                // Crossing a bin seam must not turn a geometry error into an
                // implausible lateral movement of the ego car.
                let limit = MAX_EGO_LANE_RATE * (t - last);
                prior + (u - prior).clamp(-limit, limit)
            } else {
                u
            };
            (t, u)
        });
        let frozen = if self.freeze_enabled && !self.no_live_pose && self.u_ego.is_some() {
            self.lane_change.step(t, self.u_ego())
        } else {
            self.lane_change = LaneChange::default();
            false
        };
        self.lanes.set_frozen(frozen);
    }
}
