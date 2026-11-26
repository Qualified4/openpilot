use super::object::{Evidence, RadarSample, TrackedObject};
use super::road_frame::{RoadFrame, LANE_WIDTH};
use super::{VehicleKind, VisionObservation};
use crate::input::{live_tracks::RawPoint, map_events::DisplayLead};
use crate::world::PoseAt;

pub(super) const CONFIRM: f64 = 0.85;
const DROP: f64 = -1.8;
const MAX_OBJECTS: usize = 80;

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
            motion_hits: 0,
            radar_sample: None,
            motion_time: 0.,
            verified_travel: 0.,
            moved: false,
            range_consistent: false,
            handoff_until: f64::NEG_INFINITY,
            vision_hits: 0,
            last_lead_sample: None,
            rear: s < 0.,
            split_until: f64::NEG_INFINITY,
            kind_vote: 0.,
        });
        self.objects.last_mut()
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
        for o in &mut self.objects {
            let v_absolute = o.v_s + speed;
            let (x, y) = old.cartesian(o.s + v_absolute * dt, o.d + o.v_d * dt);
            let (wx, wy) = past.to_world(x, -y);
            let (ex, ey) = ego.to_ego(wx, wy);
            let (s, d) = new.project(ex, -ey);
            o.s += o.v_s * dt;
            // A refreshed path can change its length and centre quickly. Carry the coordinate
            // change across gradually so path noise cannot teleport an unobserved object.
            o.s += 0.25 * (s - o.s).clamp(-1., 1.);
            o.d += (d - o.d).clamp(-0.65, 0.65);
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
                let (ds, dd, dv) = (
                    (o.s - m.s).abs(),
                    (o.d - m.d).abs(),
                    (o.v_s - m.speed).abs(),
                );
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
        for (_, mi, oi) in pairs {
            if seen_point[mi] || seen_object[oi] {
                continue;
            }
            seen_point[mi] = true;
            seen_object[oi] = true;
            update_radar(&mut self.objects[oi], measurements[mi], now, ego, speed);
        }
        for (i, &m) in measurements.iter().enumerate() {
            if !seen_point[i] {
                let u = frame.lane_u(m.s, m.d);
                if let Some(o) = self.create(m.s, m.d, m.speed, u) {
                    update_radar(o, m, now, ego, speed);
                }
            }
        }
        measurements.iter().map(|m| (m.x, m.y)).collect()
    }

    pub fn lead(&mut self, now: f64, lead: Option<(f64, DisplayLead)>, frame: &RoadFrame) {
        let Some((stamp, lead)) = lead.filter(|(t, l)| {
            (0. ..=0.18).contains(&(now - *t)) && (0.65..=1.).contains(&l.model_prob)
        }) else {
            return;
        };
        let (x, y, v) = lead.motion;
        let (s, d) = frame.project(x, -y);
        let ids_match = |o: &TrackedObject| {
            lead.measured_track
                .map(u64::from)
                .is_some_and(|id| o.evidence.last_track_id == Some(id))
        };
        if let Some(o) = self
            .objects
            .iter_mut()
            .filter(|o| {
                o.s > 0.
                    && frame.lane_u(o.s, o.d).abs() < if ids_match(o) { 0.82 } else { 0.7 }
                    && (o.s - s).abs() <= if ids_match(o) { 4. } else { 2.0 }
                    && (o.d - d).abs() <= 1.8
                    && (o.v_s - v).abs() <= 3.
            })
            .min_by(|a, b| {
                let cost = |o: &TrackedObject| {
                    (o.s - s).abs() + (o.d - d).abs() - 2. * f64::from(ids_match(o))
                };
                cost(a).total_cmp(&cost(b))
            })
        {
            if o.last_lead_sample != Some(stamp) {
                o.support(0.65 * lead.model_prob);
                o.evidence.last_lead_t = Some(now);
                o.last_lead_sample = Some(stamp);
            }
        }
    }

    pub fn vision(
        &mut self,
        now: f64,
        speed: f64,
        frame: &RoadFrame,
        obs: VisionObservation,
        x: f64,
        y: f64,
    ) {
        if !(0.3..=1.).contains(&obs.confidence) {
            return;
        }
        let (s, d) = frame.project(x, y);
        let age = now - obs.time;
        let match_cost = |o: &TrackedObject| {
            // Warp the static ground contact with ego pose, then let association account for
            // the object's own travel since capture. Camera range is noticeably noisier than radar.
            let ds = (o.s - s - (o.v_s + speed) * age).abs();
            let dd = (o.d - d).abs();
            let range_gate = (1.6 + 0.12 * s.abs()).min(5.);
            (ds <= range_gate && dd < 1.3).then_some(ds / range_gate + dd / 1.3)
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
                if o.evidence.last_radar_t.is_none_or(|t| now - t > 0.12) {
                    let innovation = s + (o.v_s + speed) * age - o.s;
                    o.s += 0.32 * innovation.clamp(-1., 1.);
                    o.d += 0.32 * (d - o.d).clamp(-0.8, 0.8);
                    o.v_s += 0.45 * innovation.clamp(-1.5, 1.5);
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
        frame: &RoadFrame,
        radar: &[(f64, f64)],
        blind: [bool; 2],
    ) {
        for o in &mut self.objects {
            let (x, y) = frame.cartesian(o.s, o.d);
            o.lane_u = frame.lane_u(o.s, o.d);
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
    }
}

fn update_radar(o: &mut TrackedObject, m: Measurement, now: f64, ego: PoseAt, speed: f64) {
    update_motion(o, m, now, ego, speed);
    let ds = m.s - o.s;
    let dd = m.d - o.d;
    // Huber influence bounds the displacement from a single inconsistent reflection.
    // Measured relative speed, rather than the position residual, drives longitudinal prediction.
    let gain_s = o.var_s / (o.var_s + 0.55);
    let gain_d = o.var_d / (o.var_d + 0.32);
    o.s += gain_s * ds.clamp(-0.40, 0.40);
    o.d += gain_d * dd.clamp(-0.62, 0.62);
    o.v_s += 0.45 * (m.speed - o.v_s).clamp(-2.6, 2.6);
    o.v_d = (o.v_d + 0.6 * dd.clamp(-0.8, 0.8)).clamp(-2.7, 2.7);
    o.var_s *= 1. - gain_s * (0.4 / ds.abs()).min(1.);
    o.var_d *= 1. - gain_d * (0.62 / dd.abs()).min(1.);
    let prior = o.evidence.last_radar_t;
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
    if o.radar_age(now) > 0.20 {
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
    o.range_consistent = range_error <= 0.65 + 2.0 * interval;
    let (previous_x, _) = ego.to_ego(previous.world_x, previous.world_y);
    let ground_speed = (m.x - previous_x) / interval;
    if o.range_consistent
        && ground_speed > 1.5
        && (m.x >= 6. || m.speed.abs() < 0.5 * speed || o.moved)
    {
        o.motion_time += interval;
        o.verified_travel = (o.verified_travel + ground_speed * interval).min(10.);
        o.moved |= o.motion_time >= if ground_speed >= 4. { 0.24 } else { 0.48 };
    } else {
        o.motion_time = 0.;
    }
}
