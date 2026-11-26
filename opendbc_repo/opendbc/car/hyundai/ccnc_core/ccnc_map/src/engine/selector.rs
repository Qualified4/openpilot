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
    occupied: [Option<u64>; 5],
) -> [Option<EngineSlot>; 5] {
    for o in objects.iter_mut() {
        update_lane(o);
    }
    let mut slots = [None; 5];
    let mut front = Vec::new();
    let mut sides: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    for o in objects.iter() {
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
        if o.s > 0. && o.s <= 100. && ff_lane {
            let threshold = if ff_incumbent { HOLD } else { CONFIRM };
            if o.existence_log_odds >= threshold
                && (o.motion_hits >= 3 || o.vision_hits >= 3 || has_lead)
                && (has_lead || observed || ff_incumbent && o.radar_age(now) <= 0.22)
            {
                front.push(o);
            }
            continue;
        }
        if o.lane.abs() != 1 || o.s > SIDE_FRONT_LIMIT {
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
            && o.stationary_vision_t
                .is_some_and(|t| now - t <= STATIONARY_VISION_WINDOW)
            && o.supports_stationary_vision(now, speed);
        let measured_vehicle = vision(now, o)
            || confirmed_stopped
            || o.moved
                && (o.range_consistent || near_continuity || handoff)
                && (!stationary || o.verified_travel >= 3.)
            || speed <= 5. && o.motion_hits >= 3;
        let threshold = if incumbent { HOLD } else { SIDE_CONFIRM };
        if measured_vehicle
            && o.existence_log_odds >= threshold
            && (observed || incumbent && (handoff || !o.rear && o.radar_age(now) <= hold_age))
        {
            sides[usize::from(o.lane > 0)].push(o);
        }
    }
    slots[0] = front
        .into_iter()
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
        .map(|o| slot(frame, o));
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
        let alongside = cars
            .iter()
            .copied()
            .filter(|o| {
                // Treat the alongside threshold as an interval: a car whose range uncertainty still
                // overlaps it can leave room for a separate measured car farther ahead.
                !o.rear && o.s >= SPLIT_REAR_MIN && o.s - o.var_s.sqrt().min(0.7) <= SPLIT_REAR_MAX
            })
            .min_by(|a, b| {
                cost(now, a, a.s.abs(), rear_slot).total_cmp(&cost(now, b, b.s.abs(), rear_slot))
            });
        let split = alongside.filter(|near| {
            // An observed two-car pair can replace a predicted rear picture.
            // Ensure the farther return lies comfortably inside the same lane.
            cars.iter().any(|far| {
                far.s >= near.s + TWO_CAR_GAP
                    && far.s > 0.
                    && front_within_range(far)
                    && seen(now, far)
                    && (behind.is_none() || seen(now, near) && far.lane_u.abs() <= 1.35)
            })
        });
        splits[side] = split.map(|near| near.obj_id);
        let split =
            split.or_else(|| alongside.filter(|near| behind.is_none() && now <= near.split_until));
        let front = cars
            .iter()
            .copied()
            .filter(|o| {
                !o.rear
                    && front_within_range(o)
                    && split.is_none_or(|near| o.s >= near.s + TWO_CAR_GAP)
            })
            .min_by(|a, b| cost(now, a, a.s, front_slot).total_cmp(&cost(now, b, b.s, front_slot)));
        let rear = split.or(behind);
        slots[side + 1] = front.map(|o| slot(frame, o));
        slots[side + 3] = rear.map(|o| slot(frame, o));
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

fn slot(frame: &RoadFrame, o: &TrackedObject) -> EngineSlot {
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
    }
}
