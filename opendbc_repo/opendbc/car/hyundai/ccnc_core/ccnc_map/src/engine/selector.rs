use super::object::{TrackedObject, STATIONARY_VISION_WINDOW};
use super::road_frame::RoadFrame;
use super::tracker::CONFIRM;
use super::EngineSlot;

const SIDE_FRONT_LIMIT: f64 = 50.;
const SPLIT_REAR_MIN: f64 = -8.;
const SPLIT_REAR_MAX: f64 = 6.;
const TWO_CAR_GAP: f64 = 4.;
const PASS_BAND: f64 = 0.65;
const SIDE_CONFIRM: f64 = 1.35;
const HOLD: f64 = 0.5;
const FF_HANDOFF_OFFSET: f64 = 0.35;

fn lead(now: f64, obj: &TrackedObject) -> bool {
    obj.evidence.last_lead_t.is_some_and(|t| now - t <= 0.16) && obj.lane_u.abs() <= 0.82
}

fn seen(now: f64, obj: &TrackedObject) -> bool {
    obj.evidence.last_radar_t == Some(now) || vision(now, obj)
}

fn vision(now: f64, obj: &TrackedObject) -> bool {
    obj.evidence.last_vision_t.is_some_and(|t| now - t <= 0.14)
}

fn cost(now: f64, o: &TrackedObject, range: f64, occupied: Option<u64>) -> f64 {
    let age = if seen(now, o) {
        0.
    } else {
        o.radar_age(now).min(0.25)
    };
    let sticky = if occupied == Some(o.obj_id) {
        2.5 + 0.08 * range.min(50.)
    } else {
        0.
    };
    range + 6. * (1. - o.existence_p()) + 6. * age - sticky
}

pub(super) fn select(
    now: f64,
    speed: f64,
    changing_lanes: bool,
    frame: &RoadFrame,
    objects: &mut [TrackedObject],
    previous: [Option<EngineSlot>; 5],
    primary: Option<u64>,
) -> [Option<EngineSlot>; 5] {
    let occupied = previous.map(|slot| slot.map(|o| o.obj_id));
    let mut slots = [None; 5];
    let mut front = Vec::new();
    let mut ff_handoffs = [None; 3];
    let mut sides: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    let mut close: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    let mut camera_queue: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    for o in objects.iter_mut() {
        let previous_lane = o.lane;
        update_lane(o);
        let o = &*o;
        if primary == Some(o.obj_id) {
            let mut lead = slot(frame, o, now);
            let [x, y, _] = o.last_radar_motion;
            if (lead.x - x).abs() > 2.5 || (lead.y - y).abs() > 1.8 {
                lead.x = x;
                lead.y = y;
            }
            slots[0] = Some(lead);
            continue;
        }
        let incumbent = occupied.contains(&Some(o.obj_id));
        if o.radar_hits < 3 && o.vision_hits < 3 {
            continue;
        }
        let has_lead = lead(now, o);
        let observed = seen(now, o);
        let ff_incumbent = occupied[0] == Some(o.obj_id);
        let ff_lane = o.lane == 0
            || has_lead
            || (ff_incumbent && changing_lanes && o.lane_filtered.abs() <= 0.95);
        let ff_candidate = o.s > 0. && o.s <= 100. && ff_lane;
        if ff_candidate {
            let threshold = if ff_incumbent { HOLD } else { CONFIRM };
            if o.existence_log_odds >= threshold
                && (o.motion_hits >= 3 || o.vision_hits >= 3 || has_lead)
                && (has_lead || observed || ff_incumbent && o.radar_age(now) <= 0.22)
            {
                front.push(o);
            }
        }
        let handoff_side = usize::from(o.lane_u > 0.);
        let side_incumbent = occupied[handoff_side + 1] == Some(o.obj_id);
        let offset = o.lane_u.abs();
        // Enter on a fresh observation moving away from centre. A brief FF change
        // while the old lead is returning towards centre must not flash a side slot.
        let leaving_ff = ff_incumbent
            && observed
            && offset < 0.60
            && previous[0].is_some_and(|ff| {
                ff.lane_u.signum() == o.lane_u.signum() && ff.lane_u.abs() <= offset
            });
        // While lane hysteresis still says centre, keep a handoff on the same side.
        let ff_handoff = ff_candidate
            && !o.rear
            && o.lane == 0
            && (FF_HANDOFF_OFFSET..1.5).contains(&offset)
            && (leaving_ff || previous_lane == 0 && side_incumbent);
        if (!ff_handoff && (ff_candidate || o.lane.abs() != 1)) || o.s > SIDE_FRONT_LIMIT {
            continue;
        }
        let handoff = o.rear && (-10. ..=0.).contains(&o.s) && now <= o.handoff_until;
        let near = o.s <= SPLIT_REAR_MAX + o.var_s.sqrt().min(0.7);
        let near_continuity = near && (observed || incumbent && now <= o.handoff_until);
        let hold_age = if near_continuity && o.v_s < 0. {
            0.34
        } else {
            0.20
        };
        let stationary = speed > 5. && (o.v_s + speed).abs() < 1.5;
        let confirmed_stopped = speed <= 3.
            && o.stationary_vision_confirmed
            && (o.stationary_radar_continuous
                || o.stationary_vision_t.is_some_and(|t| now - t <= STATIONARY_VISION_WINDOW))
            && o.supports_stationary_vision(now, speed);
        let measured_vehicle = vision(now, o)
            || confirmed_stopped
            || o.moved
                && (o.range_consistent || near_continuity || handoff)
                && (!stationary || o.verified_travel >= 3.)
            || speed <= 5. && o.motion_hits >= 3;
        if ff_handoff {
            let threshold = if side_incumbent { HOLD } else { SIDE_CONFIRM };
            if measured_vehicle
                && o.existence_log_odds >= threshold
                && (observed || o.radar_age(now) <= hold_age)
            {
                let previous_slot = if ff_incumbent { 0 } else { handoff_side + 1 };
                ff_handoffs[previous_slot] = Some(o);
            }
            continue;
        }
        let threshold = if incumbent { HOLD } else { SIDE_CONFIRM };
        let side = usize::from(o.lane > 0);
        if o.evidence.last_radar_t == Some(now)
            && speed <= 3.
            && o.radar_streak >= 5
            && (0.55..=1.35).contains(&o.lane_u.abs())
            && (-0.6..=3.).contains(&(o.v_s + speed))
        {
            // Sparse evidence at a stop can only be used as part of a two-car pair.
            if !o.rear && near && o.s >= 1.5 && (o.range_consistent || speed <= 1. && o.v_s >= 0.) {
                close[side].push(o);
            }
            if !o.rear
                && o.range_consistent
                && o.front_box_t
                    .is_some_and(|t| (0. ..=1.5).contains(&(now - t)))
            {
                camera_queue[side].push(o);
            }
        }
        if measured_vehicle
            && o.existence_log_odds >= threshold
            && (observed || incumbent && (handoff || !o.rear && o.radar_age(now) <= hold_age))
        {
            sides[side].push(o);
        }
    }
    slots[0] = slots[0].or_else(|| {
        front.into_iter()
            .min_by(|a, b| {
                let lead_time = |o: &TrackedObject| {
                    o.last_lead_sample
                        .filter(|_| lead(now, o))
                        .unwrap_or(f64::NEG_INFINITY)
                };
                lead_time(b).total_cmp(&lead_time(a)).then_with(|| {
                    cost(now, a, a.s, occupied[0]).total_cmp(&cost(now, b, b.s, occupied[0]))
                })
            })
            .map(|o| slot(frame, o, now))
    });
    let front_within_range = |o: &TrackedObject| {
        o.s <= SIDE_FRONT_LIMIT && frame.cartesian(o.s, o.d).0 <= SIDE_FRONT_LIMIT
    };
    let mut splits = [None; 2];
    for (side, cars) in sides.iter().enumerate() {
        let front_slot = occupied[side + 1];
        let rear_slot = occupied[side + 3];
        let behind = cars.iter().copied().filter(|o| o.rear).min_by(|a, b| {
            cost(now, a, a.s.abs(), rear_slot).total_cmp(&cost(now, b, b.s.abs(), rear_slot))
        });
        let front_cost = |o: &TrackedObject| cost(now, o, o.s, front_slot);
        let front_candidates = || {
            cars.iter()
                .copied()
                .filter(|o| !o.rear && front_within_range(o))
        };
        let front_default =
            front_candidates().min_by(|a, b| front_cost(a).total_cmp(&front_cost(b)));
        let alongside = front_candidates()
            .chain(close[side].iter().copied())
            .filter(|o| {
                o.s >= SPLIT_REAR_MIN
                    && o.s - o.var_s.sqrt().min(0.7) <= SPLIT_REAR_MAX
                    && front_default.is_none_or(|front| !seen(now, front) || o.s <= front.s)
            })
            .min_by(|a, b| {
                cost(now, a, a.s.abs(), rear_slot).total_cmp(&cost(now, b, b.s.abs(), rear_slot))
            });
        let split = alongside.and_then(|near| {
            let established_near = cars.iter().any(|o| o.obj_id == near.obj_id);
            let queue_pair = |far: &TrackedObject| {
                near.evidence.last_radar_t == Some(now)
                    && far.evidence.last_radar_t == Some(now)
                    && (far.lane_u - near.lane_u).abs() <= 0.45
                    && (far.v_s - near.v_s).abs() <= 2.5
                    && far.s - near.s <= 12.
            };
            let ordered_far = |far: &TrackedObject| far.s >= near.s + TWO_CAR_GAP && far.s > 0.;
            let far = front_candidates()
                .filter(|far| {
                    ordered_far(far)
                        && seen(now, far)
                        && (established_near || queue_pair(far))
                        && behind.is_none_or(|old| {
                            !seen(now, old) && seen(now, near) && far.lane_u.abs() <= 1.35
                        })
                })
                .chain(camera_queue[side].iter().copied().filter(|far| {
                    behind.is_none()
                        && ordered_far(far)
                        && front_within_range(far)
                        && queue_pair(far)
                        && camera_queue[side].iter().any(|o| o.obj_id == near.obj_id)
                }))
                .min_by(|a, b| front_cost(a).total_cmp(&front_cost(b)))?;
            Some((near, far))
        });
        splits[side] = split.map(|(near, _)| near.obj_id);
        let (front, rear) = if let Some((near, far)) = split {
            (Some(far), Some(near))
        } else if let Some(near) = alongside.filter(|near| {
            behind.is_none()
                && now <= near.split_until
                && cars.iter().any(|o| o.obj_id == near.obj_id)
        }) {
            let front = front_candidates()
                .filter(|o| o.s >= near.s + TWO_CAR_GAP)
                .min_by(|a, b| front_cost(a).total_cmp(&front_cost(b)));
            (front, Some(near))
        } else {
            // Promote the farther vehicle only when the same assignment places the nearer one.
            (front_default, behind)
        };
        let front = front.or_else(|| {
            [ff_handoffs[0], ff_handoffs[side + 1]]
                .into_iter()
                .flatten()
                .filter(|o| {
                    usize::from(o.lane_u > 0.) == side
                        && slots[0].is_some_and(|ff| ff.obj_id != o.obj_id)
                        && front_within_range(o)
                })
                .min_by(|a, b| front_cost(a).total_cmp(&front_cost(b)))
        });
        slots[side + 1] = front.map(|o| slot(frame, o, now));
        slots[side + 3] = rear.map(|o| slot(frame, o, now));
    }
    for o in objects
        .iter_mut()
        .filter(|o| splits.contains(&Some(o.obj_id)))
    {
        o.split_until = now + 0.18;
    }
    slots
}

fn update_lane(o: &mut TrackedObject) {
    o.lane_filtered += 0.24 * (o.lane_u - o.lane_filtered);
    let lane = f64::from(o.lane);
    if (o.lane_filtered - lane).abs() > 0.59 {
        o.lane = o.lane_filtered.round().clamp(-3., 3.) as i8;
    }
    if o.s < -PASS_BAND {
        o.rear = true;
    } else if o.s > PASS_BAND {
        o.rear = false;
    }
}

fn slot(frame: &RoadFrame, o: &TrackedObject, now: f64) -> EngineSlot {
    let (x, y) = frame.cartesian(o.s, o.d);
    EngineSlot {
        obj_id: o.obj_id,
        s: o.s,
        d: o.d,
        x,
        y,
        v_s: o.v_s,
        kind: o.kind,
        existence_p: o.existence_p(),
        last_track_id: o.evidence.last_track_id,
        lane_u: o.lane_u,
        camera_seen: vision(now, o),
    }
}
