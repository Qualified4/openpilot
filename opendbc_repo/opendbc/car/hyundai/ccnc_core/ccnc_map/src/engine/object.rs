/// Vehicle type from camera votes. An unclassified radar object remains Unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VehicleKind {
    #[default]
    Unknown,
    Car,
    Heavy,
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

#[derive(Clone, Debug)]
pub struct TrackedObject {
    pub obj_id: u64,
    pub s: f64,
    pub v_s: f64,
    pub d: f64,
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
    pub(super) motion_hits: u8,
    pub(super) radar_sample: Option<RadarSample>,
    pub(super) motion_time: f64,
    pub(super) verified_travel: f64,
    pub(super) moved: bool,
    pub(super) range_consistent: bool,
    pub(super) handoff_until: f64,
    pub(super) vision_hits: u32,
    pub(super) stationary_vision_t: Option<f64>,
    pub(super) stationary_vision_confirmed: bool,
    pub(super) last_lead_sample: Option<f64>,
    pub(super) rear: bool,
    pub(super) split_until: f64,
    pub(super) kind_vote: f64,
}

impl TrackedObject {
    pub(super) fn radar_age(&self, now: f64) -> f64 {
        self.evidence
            .last_radar_t
            .map_or(f64::INFINITY, |t| now - t)
    }

    pub(super) fn supports_stationary_vision(&self, now: f64, speed: f64) -> bool {
        self.evidence.last_radar_t == Some(now)
            && self.radar_hits >= 6
            && (1.5..=30.).contains(&self.s)
            && (0.55..=1.45).contains(&self.lane_u.abs())
            && (self.v_s + speed).abs() <= 1.
    }

    pub fn existence_p(&self) -> f64 {
        1. / (1. + (-self.existence_log_odds).exp())
    }

    pub(super) fn support(&mut self, log_likelihood: f64) {
        self.existence_log_odds = (self.existence_log_odds + log_likelihood).clamp(-5., 4.);
    }

    pub(super) fn vote(&mut self, kind: VehicleKind, confidence: f64) {
        self.kind_vote = match kind {
            VehicleKind::Unknown => return,
            VehicleKind::Car => self.kind_vote - confidence,
            VehicleKind::Heavy => self.kind_vote + confidence,
        }
        .clamp(-4., 4.);
        if self.kind_vote >= 1.25 {
            self.kind = VehicleKind::Heavy;
        } else if self.kind_vote <= -1.25 {
            self.kind = VehicleKind::Car;
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
