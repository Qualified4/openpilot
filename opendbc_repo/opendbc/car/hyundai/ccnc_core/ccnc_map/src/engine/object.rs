/// Vehicle type from camera votes. An unclassified radar object remains Unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VehicleKind {
    #[default]
    Unknown,
    Car,
    Heavy,
    Motorcycle,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Evidence {
    pub last_radar_t: Option<f64>,
    pub last_vision_t: Option<f64>,
    pub last_lead_t: Option<f64>,
    pub last_bsd_t: Option<f64>,
    pub last_kind_t: Option<f64>,
    pub last_track_id: Option<u64>,
}

// Sparse front detections may be several seconds apart at a stop.
pub(super) const STATIONARY_VISION_WINDOW: f64 = 8.;
pub(super) const REAR_LIMIT: f64 = -10.;
// Without BSD, require the projected pass to finish sooner.
const SPLIT_ORIGIN_WINDOW: f64 = 0.45;

#[derive(Clone, Debug)]
pub struct TrackedObject {
    pub obj_id: u64,
    pub s: f64,
    pub v_s: f64,
    pub d: f64,
    // Projection onto a polyline vertex can leave a tangential remainder.
    pub(super) projection_remainder: (f64, f64),
    pub v_d: f64,
    pub var_s: f64,
    pub var_d: f64,
    pub lane_u: f64,
    pub kind: VehicleKind,
    pub existence_log_odds: f64,
    pub evidence: Evidence,
    pub(super) lane: i8,
    pub(super) lane_filtered: f64,
    pub(super) radar_hits: u32,
    pub(super) radar_streak: u32,
    pub(super) same_track_streak: u32,
    pub(super) last_radar_motion: [f64; 3],
    pub(super) motion_hits: u8,
    pub(super) radar_sample: Option<RadarSample>,
    pub(super) motion_time: f64,
    pub(super) verified_travel: f64,
    pub(super) moved: bool,
    pub(super) range_consistent: bool,
    pub(super) range_recovered_t: Option<f64>,
    pub(super) handoff_until: f64,
    pub(super) vision_hits: u32,
    pub(super) front_box_t: Option<f64>,
    pub(super) stationary_vision_t: Option<f64>,
    pub(super) stationary_vision_confirmed: bool,
    /// Radar has measured the same stationary return on every tick since confirmation.
    pub(super) stationary_radar_continuous: bool,
    pub(super) last_lead_sample: Option<f64>,
    pub(super) rear: bool,
    pub(super) split_until: f64,
    pub(super) detector_kind: VehicleKind,
    pub(super) kind_vote: f64,
    pub(super) car_streak: u8,
    pub(super) pedestrian_veto_t: Option<f64>,
}

impl TrackedObject {
    pub(super) fn cartesian(&self, frame: &super::road_frame::RoadFrame) -> (f64, f64) {
        let (x, y) = frame.cartesian(self.s, self.d);
        (x + self.projection_remainder.0, y + self.projection_remainder.1)
    }

    pub(super) fn radar_age(&self, now: f64) -> f64 {
        self.evidence
            .last_radar_t
            .map_or(f64::INFINITY, |t| now - t)
    }

    /// Extend a confirmed near handoff only while its rear slot stays occupied.
    pub(super) fn rear_coast(&self, now: f64, was_rear: bool, blind: bool) -> bool {
        let age = self.radar_age(now);
        was_rear
            && self.handoff_until.is_finite()
            && ((REAR_LIMIT..0.).contains(&self.s) && age <= 1.65
                || self.s >= 0. && age <= 0.55
                    && (blind && self.s <= 0.65 && self.v_s <= 0.
                        // A previously split car must be closing fast enough to reach zero
                        // within the shorter uncorroborated radar gap allowance.
                        || self.evidence.last_radar_t.is_some_and(|t| t <= self.split_until)
                            && self.v_s < -0.5 && self.s <= -self.v_s * (SPLIT_ORIGIN_WINDOW - age)))
    }

    pub(super) fn supports_stationary_vision(&self, now: f64, speed: f64) -> bool {
        self.evidence.last_radar_t == Some(now)
            && self.radar_hits >= 6
            && (1.5..=30.).contains(&self.s)
            && (0.55..=1.45).contains(&self.lane_u.abs())
            && (-0.6..=1.0).contains(&(self.v_s + speed))
    }

    pub fn existence_p(&self) -> f64 {
        1. / (1. + (-self.existence_log_odds).exp())
    }

    pub(super) fn support(&mut self, log_likelihood: f64) {
        self.existence_log_odds = (self.existence_log_odds + log_likelihood).clamp(-5., 4.);
    }

    pub(super) fn vote(&mut self, kind: VehicleKind, confidence: f64) {
        if kind == VehicleKind::Motorcycle {
            if confidence >= 0.45 && self.detector_kind != VehicleKind::Heavy {
                self.detector_kind = VehicleKind::Motorcycle;
            }
            return;
        }
        self.car_streak = if kind == VehicleKind::Car && confidence >= 0.6 {
            self.car_streak.saturating_add(1)
        } else {
            0
        };
        self.kind_vote = match kind {
            VehicleKind::Unknown | VehicleKind::Motorcycle => return,
            VehicleKind::Car => self.kind_vote - confidence,
            VehicleKind::Heavy => self.kind_vote + confidence,
        }
        .clamp(-4., 4.);
        if self.kind_vote >= 1.25 {
            self.detector_kind = VehicleKind::Heavy;
        } else if self.kind_vote <= -1.25
            && (self.detector_kind != VehicleKind::Heavy
                || u32::from(self.car_streak) >= crate::vision::kind::N_FLIP)
        {
            self.detector_kind = VehicleKind::Car;
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct RadarSample {
    pub time: f64,
    pub range: f64,
    pub relative_speed: f64,
    pub world_x: f64,
    pub world_y: f64,
}
