use super::object::TrackedObject;
use super::road_frame::RoadFrame;
use super::tracker::CONFIRM;
use super::EngineSlot;

const SIDE_FRONT_LIMIT: f64 = 50.;
const SPLIT_REAR_MIN: f64 = -8.;
const SPLIT_REAR_MAX: f64 = 6.;
const TWO_CAR_GAP: f64 = 3.;
const PASS_BAND: f64 = 0.65;

fn lead(now: f64, obj: &TrackedObject) -> bool {
    obj.evidence.last_lead_t.is_some_and(|t| now - t <= 0.16) && obj.lane_u.abs() <= 0.82
}

fn seen(now: f64, obj: &TrackedObject) -> bool {
    obj.evidence.last_radar_t == Some(now)
        || obj.evidence.last_vision_t.is_some_and(|t| now - t <= 0.14)
}

pub(super) fn select(
    now: f64,
    frame: &RoadFrame,
    objects: &mut [TrackedObject],
) -> [Option<EngineSlot>; 5] {
    for o in objects.iter_mut() {
        update_lane(o);
    }
    let mut slots = [None; 5];
    let mut front = Vec::new();
    let mut sides: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    for o in objects.iter() {
        let has_lead = lead(now, o);
        let qualified = o.motion_hits >= 3 || o.vision_hits >= 3 || has_lead;
        if o.existence_log_odds < CONFIRM || !qualified || (o.radar_hits < 3 && o.vision_hits < 3) {
            continue;
        }
        if o.s > 0. && (o.lane == 0 || lead(now, o)) && o.s <= 100. {
            front.push(o);
        } else if o.lane.abs() == 1 && (o.rear || o.s <= SIDE_FRONT_LIMIT) {
            sides[usize::from(o.lane > 0)].push(o);
        }
    }
    slots[0] = front
        .into_iter()
        .min_by(|a, b| {
            (lead(now, b), seen(now, b))
                .cmp(&(lead(now, a), seen(now, a)))
                .then_with(|| a.s.total_cmp(&b.s))
        })
        .map(|o| slot(frame, o));
    let front_within_range = |o: &TrackedObject| {
        o.s <= SIDE_FRONT_LIMIT && frame.cartesian(o.s, o.d).0 <= SIDE_FRONT_LIMIT
    };
    let mut splits = [None; 2];
    for (side, cars) in sides.iter().enumerate() {
        let alongside = cars
            .iter()
            .copied()
            .filter(|o| {
                // Treat the alongside threshold as an interval: a car whose range uncertainty still
                // overlaps it can leave room for a separate measured car farther ahead.
                o.s >= SPLIT_REAR_MIN && o.s - o.var_s.sqrt().min(0.7) <= SPLIT_REAR_MAX
            })
            .min_by(|a, b| {
                seen(now, b)
                    .cmp(&seen(now, a))
                    .then_with(|| a.s.abs().total_cmp(&b.s.abs()))
            });
        let split = alongside.filter(|near| {
            cars.iter().any(|far| {
                far.s >= near.s + TWO_CAR_GAP
                    && far.s > 0.
                    && front_within_range(far)
                    && (seen(now, far) || far.existence_p() >= 0.92)
            })
        });
        splits[side] = split.map(|near| near.obj_id);
        let split = split.or_else(|| alongside.filter(|near| now <= near.split_until));
        let front = cars
            .iter()
            .copied()
            .filter(|o| {
                !o.rear
                    && front_within_range(o)
                    && split.is_none_or(|near| o.s >= near.s + TWO_CAR_GAP)
            })
            .min_by(|a, b| {
                seen(now, b)
                    .cmp(&seen(now, a))
                    .then_with(|| a.s.total_cmp(&b.s))
            });
        let rear = split.or_else(|| {
            cars.iter().copied().filter(|o| o.rear).max_by(|a, b| {
                seen(now, a)
                    .cmp(&seen(now, b))
                    .then_with(|| a.s.total_cmp(&b.s))
            })
        });
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
