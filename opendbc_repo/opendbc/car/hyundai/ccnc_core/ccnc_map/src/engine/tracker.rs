use super::object::{Evidence, RadarSample, TrackedObject, STATIONARY_VISION_WINDOW};
use super::road_frame::{RoadFrame, LANE_WIDTH};
use super::{VehicleKind, VisionObservation};
use crate::input::{live_tracks::RawPoint, map_events::DisplayLead};
use crate::vision::kind::{
    front_gate, front_height, FrontCam, KindResult, Kinds, KIND_CAR, MAX_TALL_M,
};
use crate::world::PoseAt;
use det_rules::{Track, CAM2B};
use std::collections::VecDeque;

pub(super) const CONFIRM: f64 = 0.85;
const DROP: f64 = -1.8;
const MAX_OBJECTS: usize = 80;
// Keep the capture-time radar history for asynchronous visiond classifier answers.
const CAMERA_HISTORY_NS: u64 = 4_000_000_000;
const CAMERA_MATCH_NS: u64 = 150_000_000;

#[derive(Clone, Copy)]
struct Measurement {
    id: u64,
    x: f64,
    y: f64,
    s: f64,
    d: f64,
    speed: f64,
    moving: bool,
}

#[derive(Default)]
pub(super) struct Tracker {
    pub objects: Vec<TrackedObject>,
    last_id: u64,
    camera_history: VecDeque<(u64, Vec<Track>)>,
    pub(super) kinds: Kinds,
}

impl Tracker {
    fn create(&mut self, s: f64, d: f64, v_s: f64, lane_u: f64) -> Option<&mut TrackedObject> {
        if self.objects.len() >= MAX_OBJECTS {
            return None;
        }
        self.last_id += 1;
        self.objects.push(TrackedObject {
            obj_id: self.last_id,
            s,
            d,
            projection_remainder: (0., 0.),
            v_s,
            v_d: 0.,
            var_s: 1.4,
            var_d: 0.7,
            lane_u,
            kind: VehicleKind::Unknown,
            existence_log_odds: -1.3,
            evidence: Evidence::default(),
            lane: lane_u.round().clamp(-3., 3.) as i8,
            lane_filtered: lane_u,
            radar_hits: 0,
            radar_streak: 0,
            same_track_streak: 0,
            last_radar_motion: [0.; 3],
            motion_hits: 0,
            radar_sample: None,
            motion_time: 0.,
            verified_travel: 0.,
            moved: false,
            range_consistent: false,
            range_recovered_t: None,
            handoff_until: f64::NEG_INFINITY,
            vision_hits: 0,
            front_box_t: None,
            stationary_vision_t: None,
            stationary_vision_confirmed: false,
            stationary_radar_continuous: false,
            last_lead_sample: None,
            rear: s < 0.,
            split_until: f64::NEG_INFINITY,
            detector_kind: VehicleKind::Unknown,
            kind_vote: 0.,
            car_streak: 0,
            pedestrian_veto_t: None,
        });
        self.objects.last_mut()
    }

    pub fn reset(&mut self) {
        self.objects.clear();
        self.camera_history.clear();
        self.kinds = Kinds::default();
    }

    pub fn front_kinds(&mut self, published: u64, camera: &FrontCam, results: &[KindResult]) {
        self.observe_kinds(published, camera, results, |_, _| None);
    }

    pub fn observe_kinds(
        &mut self,
        published: u64,
        camera: &FrontCam,
        results: &[KindResult],
        side_cands: impl Fn(u64, crate::vision::Side) -> Option<Vec<det_rules::Cand>>,
    ) {
        let results: Vec<_> = results
            .iter()
            .copied()
            .filter(|result| {
                published >= result.time_ns && published - result.time_ns <= CAMERA_HISTORY_NS
            })
            .collect();
        let history = &self.camera_history;
        // A box votes for its best capture-time object. A nearby shadow object may
        // be a separate passenger car even when it fits within one bus length.
        self.kinds.observe_matches(
            &results,
            camera,
            |t| camera_tracks(history, t).map(|(stamp, tracks)| (*stamp, tracks.clone())),
            side_cands,
        );
    }

    /// Suppress low-speed roadside radar-only false positives that coincide with a detected person.
    /// Pedestrians are never promoted to displayed vehicles.
    pub fn observe_pedestrian(
        &mut self,
        now: f64,
        speed: f64,
        frame: &RoadFrame,
        confidence: f64,
        x: f64,
        y: f64,
    ) {
        if !(0.45..=1.).contains(&confidence) || !x.is_finite() || !y.is_finite() {
            return;
        }
        let (s, d) = frame.project(x, y);
        let range_gate = (2.0 + 0.15 * s.abs()).min(5.0);
        for o in &mut self.objects {
            if o.vision_hits > 0
                || o.evidence.last_lead_t.is_some()
                || (o.moved && o.verified_travel >= 3.0)
                || (o.v_s + speed).abs() > 3.0
                || o.lane_u.abs() < 0.55
            {
                continue;
            }
            if (o.s - s).abs() <= range_gate && (o.d - d).abs() <= 1.2 {
                o.pedestrian_veto_t = Some(now);
                o.support(-1.2 * confidence);
            }
        }
    }

    pub fn front_boxes(&mut self, capture: u64, camera: &FrontCam, boxes: &[[f64; 4]]) {
        let Some((_, tracks)) = camera_tracks(&self.camera_history, capture) else {
            return;
        };
        let t = capture as f64 * 1e-9;
        for bbox in boxes {
            let id = tracks
                .iter()
                .filter(|track| (1. ..=MAX_TALL_M).contains(&front_height(track, bbox)))
                .filter_map(|track| front_gate(camera, track, bbox).map(|cost| (track.id, cost)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id as u64);
            if let Some(o) = self.objects.iter_mut().find(|o| Some(o.obj_id) == id) {
                if o.front_box_t.is_none_or(|last| t > last) {
                    o.front_box_t = Some(t);
                }
            }
        }
    }

    pub fn predict(
        &mut self,
        dt: f64,
        speed: f64,
        old: &RoadFrame,
        new: &RoadFrame,
        past: PoseAt,
        ego: PoseAt,
    ) {
        let turning = (ego.heading - past.heading).abs() > 0.085 * dt;
        for o in &mut self.objects {
            let v_absolute = o.v_s + speed;
            let predicted_d = o.d + o.v_d * dt;
            let (x, y) = old.cartesian(o.s + v_absolute * dt, predicted_d);
            let (rx, ry) = o.projection_remainder;
            let (wx, wy) = past.to_world(x + rx, -(y + ry));
            let (ex, ey) = ego.to_ego(wx, wy);
            let (mut s, mut d) = new.project(ex, -ey);
            let chart_jump = (d - predicted_d).abs();
            if chart_jump > 0.95 || turning && chart_jump > 0.55 {
                let predicted_s = o.s + v_absolute * dt - (ego.s - past.s);
                if (s - predicted_s).abs() > 4.2 {
                    (s, d) = new.project_near(ex, -ey, predicted_s, 4.2);
                }
                // A refreshed forecast (including its tail) can move the road frame
                // several metres in one tick. Adopt it gradually while preserving the
                // Cartesian prediction for radar association via the remainder below.
                let step = (6. * dt).min(0.40);
                d = predicted_d + (d - predicted_d).clamp(-step, step);
            }
            (o.s, o.d) = (s, d);
            let (x, y) = new.cartesian(o.s, o.d);
            o.projection_remainder = (ex - x, -ey - y);
            o.v_d *= (-dt / 0.45).exp();
            o.var_s = (o.var_s + 1.8 * dt).min(7.);
            o.var_d += 0.85 * dt;
        }
    }

    pub fn radar(
        &mut self,
        now: f64,
        speed: f64,
        frame: &RoadFrame,
        ego: PoseAt,
        points: &[RawPoint],
    ) -> Vec<(f64, f64)> {
        let measurements: Vec<_> = points
            .iter()
            .filter_map(|p| {
                let (x, y, v) = (f64::from(p.d_rel), -f64::from(p.y_rel), f64::from(p.v_rel));
                if p.track_id == 0
                    || p.source != 0
                    || !p.measured
                    || p.track_state < 2
                    || !(0.1..=130.).contains(&x)
                    || y.abs() > 13.
                    || !y.is_finite()
                    || !(-3. ..=75.).contains(&(v + speed))
                {
                    return None;
                }
                let (s, d) = frame.project(x, y);
                (s >= -1.5 && d.abs() <= 8.5).then_some(Measurement {
                    id: p.track_id,
                    x,
                    y,
                    s,
                    d,
                    speed: v,
                    moving: p.motion_state == 2 || v + speed > 1.8,
                })
            })
            .collect();

        // Bounded greedy assignment on gated costs. Each observation and object is consumed once.
        // ponytail: on crowded splits, a globally optimal assignment could recover crossed matches.
        let mut pairs = Vec::new();
        for (mi, m) in measurements.iter().enumerate() {
            for (oi, o) in self.objects.iter().enumerate() {
                let same = o.evidence.last_track_id == Some(m.id);
                let (mut ds, mut dd, dv) = (
                    (o.s - m.s).abs(),
                    (o.d - m.d).abs(),
                    (o.v_s - m.speed).abs(),
                );
                if frame.beyond_forecast(o.s.max(m.s))
                    || o.projection_remainder.0.hypot(o.projection_remainder.1) > 0.25
                {
                    let (x, y) = o.cartesian(frame);
                    ds = (x - m.x).abs();
                    dd = (y - m.y).abs();
                }
                if o.s < -1.0 && m.s > o.s + 0.9
                    || ds > if same { 4.2 } else { 2.8 }
                    || dd > if same { 2.5 } else { 1.6 }
                    || dv > if same { 6. } else { 3.5 }
                {
                    continue;
                }
                let score = ds * ds / (o.var_s + 0.7) + dd * dd / (o.var_d + 0.35) + dv * dv / 6.
                    - if same { 1.8 } else { 0. };
                if score <= 8.5 {
                    pairs.push((score, mi, oi));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seen_point = vec![false; measurements.len()];
        let mut seen_object = vec![false; self.objects.len()];
        let mut camera = Vec::with_capacity(measurements.len());
        for (_, mi, oi) in pairs {
            if seen_point[mi] || seen_object[oi] {
                continue;
            }
            seen_point[mi] = true;
            seen_object[oi] = true;
            let o = &mut self.objects[oi];
            let m = measurements[mi];
            if let Some(old) = o.evidence.last_track_id.filter(|&old| old != m.id) {
                // A different, still measured return now owns that raw track. An inherited
                // classification without a supporting answer cannot follow this reassignment.
                if measurements.iter().any(|point| point.id == old) {
                    if let Ok(id) = u32::try_from(o.obj_id) {
                        self.kinds.discard_unverified_inheritance(id);
                    }
                }
            }
            update_radar(o, m, now, ego, speed);
            camera.push(camera_track(o, m));
        }
        for (i, &m) in measurements.iter().enumerate() {
            if !seen_point[i] {
                let u = frame.lane_u(m.s, m.d);
                if let Some(o) = self.create(m.s, m.d, m.speed, u) {
                    update_radar(o, m, now, ego, speed);
                    camera.push(camera_track(o, m));
                }
            }
        }
        self.handoff_kind(now);
        let stamp = (now * 1e9).round() as u64;
        self.camera_history
            .retain(|(t, _)| stamp.saturating_sub(*t) <= CAMERA_HISTORY_NS);
        self.camera_history.push_back((stamp, camera));
        measurements.iter().map(|m| (m.x, m.y)).collect()
    }

    fn handoff_kind(&mut self, now: f64) {
        for o in self
            .objects
            .iter()
            .filter(|o| o.evidence.last_radar_t == Some(now) && o.same_track_streak >= 3)
        {
            let Ok(to) = u32::try_from(o.obj_id) else {
                continue;
            };
            if self.kinds.confirmed_kind(to).is_some() {
                continue;
            }
            let mut donors = self.objects.iter().filter(|old| {
                let dt = old.radar_age(now);
                let [x, y, v] = o.last_radar_motion;
                let [old_x, old_y, old_v] = old.last_radar_motion;
                dt > 0.
                    && dt <= 0.25
                    && old.evidence.last_track_id == o.evidence.last_track_id
                    && (x - old_x - old_v * dt).abs() <= 1.
                    && (y - old_y).abs() <= 0.85
                    && (v - old_v).abs() <= 2.
            });
            // The raw radar return stayed continuous across a change of shadow object.
            // Use raw motion here because refreshed lane coordinates can diverge at range.
            if let Some(old) = donors.next().filter(|_| donors.next().is_none()) {
                if let Ok(from) = u32::try_from(old.obj_id) {
                    self.kinds.inherit_confirmed(from, to);
                }
            }
        }
    }

    pub fn lead(
        &mut self,
        now: f64,
        lead: Option<(f64, DisplayLead)>,
        frame: &RoadFrame,
    ) -> Option<u64> {
        let (stamp, lead) = lead.filter(|(t, l)| {
            (0. ..=0.18).contains(&(now - *t)) && (0.65..=1.).contains(&l.model_prob)
        })?;
        let (x, y, v) = lead.motion;
        let (s, d) = frame.project(x, -y);
        let ids_match = |o: &TrackedObject| {
            lead.measured_track
                .map(u64::from)
                .is_some_and(|id| o.evidence.last_track_id == Some(id))
        };
        // Compare independent sensors in ego coordinates before consulting the model path.
        let strong = self.objects.iter().position(|o| {
            let [rx, ry, rv] = o.last_radar_motion;
            lead.model_prob >= 0.75
                && ids_match(o)
                && o.evidence.last_radar_t == Some(now)
                && (0. ..=100.).contains(&rx)
                && (rx - x).abs() <= 2.5
                && (ry + y).abs() <= 1.8
                && (rv - v).abs() <= 3.
        });
        let matched = strong.or_else(|| {
            self.objects
                .iter()
                .enumerate()
                .filter(|(_, o)| {
                    o.s > 0.
                        && frame.lane_u(o.s, o.d).abs() < if ids_match(o) { 0.82 } else { 0.7 }
                        && (o.s - s).abs() <= if ids_match(o) { 4. } else { 2.0 }
                        && (o.d - d).abs() <= 1.8
                        && (o.v_s - v).abs() <= 3.
                })
                .min_by(|(_, a), (_, b)| {
                    let cost = |o: &TrackedObject| {
                        (o.s - s).abs() + (o.d - d).abs() - 2. * f64::from(ids_match(o))
                    };
                    cost(a).total_cmp(&cost(b))
                })
                .map(|(i, _)| i)
        });
        if let Some(i) = matched {
            let o = &mut self.objects[i];
            if o.last_lead_sample != Some(stamp) {
                o.support(0.65 * lead.model_prob);
                o.evidence.last_lead_t = Some(now);
                o.last_lead_sample = Some(stamp);
            }
        }
        strong.map(|i| self.objects[i].obj_id)
    }

    pub fn vision(
        &mut self,
        now: f64,
        speed: f64,
        frame: &RoadFrame,
        obs: VisionObservation,
        x: f64,
        y: f64,
        inside_outer_line: impl Fn(f64, f64, f64) -> bool,
    ) {
        if !(0.3..=1.).contains(&obs.confidence) {
            return;
        }
        let (s, d) = frame.project(x, y);
        let age = now - obs.time;
        if age > 0.5 {
            // Slow camera frames can confirm a stopped vehicle that radar still sees.
            // Keep these frame votes out of the position update and radar coasting rules.
            if !obs.position || obs.confidence < 0.55 || speed > 3. {
                return;
            }
            let vision_range = x + CAM2B;
            if vision_range <= CAM2B {
                return;
            }
            let candidate = self
                .objects
                .iter_mut()
                .filter(|o| o.supports_stationary_vision(now, speed))
                .filter_map(|o| {
                    let (rx, ry) = o.cartesian(frame);
                    let radar_range = rx + CAM2B;
                    // Flat-road range inferred from a box bottom is noisy. Bearing
                    // constrains the match without letting a close box label a distant echo.
                    let ratio = vision_range / radar_range;
                    let lateral_error = (y / ratio - ry).abs();
                    ((0.55..=1.6).contains(&ratio) && lateral_error <= 0.75)
                        .then_some((o, lateral_error + ratio.ln().abs()))
                })
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(o, _)| o);
            if let Some(o) = candidate {
                let gap = o
                    .stationary_vision_t
                    .map_or(f64::INFINITY, |t| obs.time - t);
                if gap >= 0.25 {
                    // A mature radar return can be confirmed immediately when the same
                    // high-confidence frame agrees by both box geometry and ground contact.
                    let [rx, ry, _] = o.last_radar_motion;
                    let adjacent = inside_outer_line(o.lane_u, rx, ry);
                    let box_match = adjacent && obs.confidence >= f64::from(0.65_f32)
                        && o.front_box_t == Some(obs.time)
                        && o.same_track_streak >= 6
                        && o.range_consistent;
                    o.stationary_vision_confirmed =
                        o.stationary_radar_continuous || box_match || gap <= STATIONARY_VISION_WINDOW;
                    o.stationary_radar_continuous = o.stationary_vision_confirmed && adjacent;
                    o.stationary_vision_t = Some(obs.time);
                }
            }
            return;
        }
        let match_cost = |o: &TrackedObject| {
            // Warp the static ground contact with ego pose, then let association account for
            // the object's own travel since capture. Camera range is noticeably noisier than radar.
            let ds = (o.s - s - (o.v_s + speed) * age).abs();
            let dd = (o.d - d).abs();
            let range_gate = (1.6 + 0.12 * s.abs()).min(5.);
            // Motorcycles (0.8 m width) use a tighter lateral gate so lane-splitting bikes stay
            // separate from neighboring passenger cars.
            let lat_gate = if obs.kind == VehicleKind::Motorcycle || o.detector_kind == VehicleKind::Motorcycle {
                0.85
            } else {
                1.3
            };
            (ds <= range_gate && dd < lat_gate).then_some(ds / range_gate + dd / lat_gate)
        };
        let matched = self
            .objects
            .iter_mut()
            .filter_map(|o| match_cost(o).map(|cost| (cost, o)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, o)| o);
        let u = frame.lane_u(s, d);
        let object = match matched {
            some @ Some(_) => some,
            None if obs.position
                && (0. ..=6.).contains(&s)
                && (0.5..=1.55).contains(&u.abs())
                && age <= 0.2 =>
            {
                self.create(s, d, -speed, u)
            }
            None => None,
        };
        if let Some(o) = object {
            if obs.kind != VehicleKind::Unknown
                && o.evidence.last_kind_t.is_none_or(|t| obs.time > t)
            {
                o.vote(obs.kind, obs.confidence);
                o.evidence.last_kind_t = Some(obs.time);
            }
            if obs.position && o.evidence.last_vision_t.is_none_or(|last| obs.time > last) {
                o.evidence.last_vision_t = Some(obs.time);
                o.vision_hits += 1;
                o.support(1.25 * obs.confidence);
                if o.evidence.last_radar_t.is_none_or(|t| now - t > 0.12) && s <= 15. {
                    let innovation = s + (o.v_s + speed) * age - o.s;
                    o.s += 0.32 * innovation.clamp(-1., 1.);
                    o.d += 0.32 * (d - o.d).clamp(-0.8, 0.8);
                    o.var_s *= 0.85;
                    o.var_d *= 0.85;
                }
            }
        }
    }

    pub fn lifecycle(
        &mut self,
        now: f64,
        dt: f64,
        speed: f64,
        frame: &RoadFrame,
        radar: &[(f64, f64)],
        blind: [bool; 2],
        previous_rear: [Option<u64>; 2],
    ) {
        for o in &mut self.objects {
            let (x, y) = o.cartesian(frame);
            o.lane_u = frame.lane_u(o.s, o.d);
            o.stationary_radar_continuous &= speed <= 3. && o.supports_stationary_vision(now, speed);
            let observed = o.evidence.last_radar_t == Some(now)
                || o.evidence.last_lead_t == Some(now)
                || o.evidence.last_vision_t.is_some_and(|t| now - t <= 0.12);
            if observed {
                continue;
            }
            let side = o.lane_u.abs() > 0.5;
            let in_bsd = side && (-8. ..=0.).contains(&o.s);
            let bsd = in_bsd && blind[usize::from(o.lane_u > 0.)];
            let occluded = x > 6.
                && radar.iter().any(|&(near_x, near_y)| {
                    let fraction = near_x / x;
                    near_x >= 3.
                        && near_x + 2. < x
                        && (y * fraction - near_y).abs() <= 0.9 - 0.7 * fraction
                });
            let visible = (6. ..=50.).contains(&o.s) && y.abs() < 1. + x * 0.55 && !occluded;
            // A current return at the predicted location consumed by another object
            // is stronger negative evidence than the absence of a return in a blind zone.
            let replaced = radar
                .iter()
                .any(|&(rx, ry)| (x - rx).abs() < 0.65 && (y - ry).abs() < 0.55);
            let decay = if replaced {
                30.
            } else if visible {
                22.
            } else if side && o.rear_coast(
                now, previous_rear[usize::from(o.lane_u > 0.)] == Some(o.obj_id),
                blind[usize::from(o.lane_u > 0.)],
            ) {
                1.5
            } else if bsd {
                o.evidence.last_bsd_t = Some(now);
                if now <= o.handoff_until {
                    0.30
                } else {
                    8.
                }
            } else if now <= o.handoff_until && o.s < 6. {
                3.2
            } else if occluded {
                8.
            } else {
                12.
            };
            o.support(-decay * dt);
        }
        self.objects.retain(|o| {
            o.existence_log_odds >= DROP
                && o.var_d <= (LANE_WIDTH * 0.5).powi(2)
                && (-23. ..=155.).contains(&o.s)
                && o.d.abs() <= 10.
        });
        self.kinds
            .retain(|vid| self.objects.iter().any(|o| o.obj_id == u64::from(vid)));
        for o in &mut self.objects {
            o.kind = match u32::try_from(o.obj_id)
                .ok()
                .and_then(|id| self.kinds.confirmed_kind(id))
            {
                Some(KIND_CAR) => VehicleKind::Car,
                Some(_) => VehicleKind::Heavy,
                None => o.detector_kind,
            };
        }
    }
}

fn camera_tracks(
    history: &VecDeque<(u64, Vec<Track>)>,
    capture: u64,
) -> Option<&(u64, Vec<Track>)> {
    let i = history.partition_point(|(stamp, _)| *stamp <= capture);
    [
        i.checked_sub(1).and_then(|j| history.get(j)),
        history.get(i),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|(stamp, _)| stamp.abs_diff(capture))
    .filter(|(stamp, _)| stamp.abs_diff(capture) <= CAMERA_MATCH_NS)
}

fn camera_track(o: &TrackedObject, m: Measurement) -> Track {
    Track {
        id: o.obj_id as i64,
        d_rel: m.x,
        y_rel: -m.y,
        v_rel: m.speed,
    }
}

fn update_radar(o: &mut TrackedObject, m: Measurement, now: f64, ego: PoseAt, speed: f64) {
    update_motion(o, m, now, ego, speed);
    let ds = m.s - o.s;
    let dd = m.d - o.d;
    // Huber influence bounds the displacement from a single inconsistent reflection.
    // Measured relative speed, rather than the position residual, drives longitudinal prediction.
    let gain_s = o.var_s / (o.var_s + 0.55);
    // Suppress small stationary innovations while responding to lateral travel.
    let near = ((15. - m.x) / 5.).clamp(0., 1.);
    let movement = ((o.v_d.abs() - 0.22) / 0.55).clamp(0., 1.);
    let response = ((dd.abs() - 0.20) / 0.45).clamp(0., 1.).max(near * movement);
    let gain_d = o.var_d / (o.var_d + 0.32) * (0.40 + 0.60 * response);
    o.s += gain_s * ds.clamp(-0.40, 0.40);
    o.d += gain_d * dd.clamp(-0.62, 0.62);
    o.v_s += 0.45 * (m.speed - o.v_s).clamp(-2.6, 2.6);
    // Non-holonomic steering kinematic bound: a vehicle's lateral velocity cannot exceed
    // tan(delta_max) * |v_abs| + sensor floor, preventing stationary reflection crab-walk.
    let v_abs = (o.v_s + speed).abs();
    let max_v_d = (0.35 * v_abs + 0.05).min(2.7);
    o.v_d = (o.v_d + (0.3 + 0.3 * response) * dd.clamp(-0.8, 0.8)).clamp(-max_v_d, max_v_d);
    o.var_s *= 1. - gain_s * (0.4 / ds.abs()).min(1.);
    o.var_d *= 1. - gain_d * (0.62 / dd.abs()).min(1.);
    let prior = o.evidence.last_radar_t;
    if o.radar_age(now) > 0.2 {
        o.radar_streak = 0;
    }
    o.radar_streak += 1;
    if o.radar_streak == 1 || o.evidence.last_track_id != Some(m.id) {
        o.same_track_streak = 0;
    }
    o.same_track_streak += 1;
    o.last_radar_motion = [m.x, m.y, m.speed];
    o.evidence.last_radar_t = Some(now);
    o.evidence.last_track_id = Some(m.id);
    o.radar_hits += 1;
    if o.motion_hits < 3 {
        o.motion_hits = if m.moving {
            o.motion_hits + 1
        } else {
            o.motion_hits.saturating_sub(1)
        };
    }
    if m.s < 6. && (o.moved || speed <= 5. && o.motion_hits >= 3) {
        o.handoff_until = now + 1.15;
    }
    let quality = if m.moving {
        0.85
    } else if o.s > 1.5 && prior.is_some() {
        0.28
    } else {
        0.10
    };
    o.support(quality);
}

fn update_motion(o: &mut TrackedObject, m: Measurement, now: f64, ego: PoseAt, speed: f64) {
    let (world_x, world_y) = ego.to_world(m.x, -m.y);
    let sample = RadarSample {
        time: now,
        range: m.x.hypot(m.y),
        relative_speed: m.speed,
        world_x,
        world_y,
    };
    let interrupted = o.radar_age(now) > 0.20;
    if interrupted || o.evidence.last_track_id != Some(m.id) {
        o.stationary_vision_t = None;
        o.stationary_vision_confirmed = false;
        o.stationary_radar_continuous = false;
        o.front_box_t = None;
        o.range_recovered_t = None;
    }
    if interrupted {
        o.radar_sample = Some(sample);
        o.motion_time = 0.;
        return;
    }
    let Some(previous) = o.radar_sample else {
        o.radar_sample = Some(sample);
        return;
    };
    let interval = now - previous.time;
    if interval < 0.24 {
        return;
    }
    o.radar_sample = Some(sample);

    // Doppler must agree with the change in slant range. Use ego-compensated
    // Cartesian displacement for motion: a near static post can have a large
    // positive vRel + vEgo because the radar observes it at an oblique angle.
    let doppler = (sample.relative_speed + previous.relative_speed) * 0.5;
    let range_error = (sample.range - previous.range - doppler * interval).abs();
    let consistent = range_error <= 0.65 + 2.0 * interval;
    if o.moved && consistent && !o.range_consistent {
        o.range_recovered_t = Some(now);
    }
    o.range_consistent = consistent;
    let (previous_x, _) = ego.to_ego(previous.world_x, previous.world_y);
    let ground_speed = (m.x - previous_x) / interval;
    let min_ground_speed = if speed <= 1.0 { 0.45 } else { 1.5 };
    if o.range_consistent
        && ground_speed > min_ground_speed
        && (m.x >= 6. || m.speed.abs() < 0.5 * speed || o.moved || speed <= 1.0)
    {
        o.motion_time += interval;
        o.verified_travel = (o.verified_travel + ground_speed * interval).min(10.);
        let need_time = if ground_speed >= 4. { 0.24 } else { 0.48 };
        let need_travel = if speed <= 1.0 && ground_speed <= 1.5 { 0.45 } else { 0. };
        o.moved |= o.motion_time >= need_time && o.verified_travel >= need_travel;
    } else {
        o.motion_time = 0.;
    }
}
