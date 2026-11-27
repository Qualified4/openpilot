use super::object::{TrackedObject, REAR_LIMIT, STATIONARY_VISION_WINDOW};
use super::road_frame::RoadFrame;
use super::tracker::CONFIRM;
use super::EngineSlot;

const SIDE_FRONT_LIMIT: f64 = 50.;
const SPLIT_REAR_MIN: f64 = -8.;
const SPLIT_REAR_MAX: f64 = 6.;
const SPLIT_RELEASE: f64 = 7.5;
pub(super) const ALONGSIDE_SPEED: f64 = 2.;
const SPLIT_RELEASE_SPEED: f64 = 2.75;
const FRONT_PASS_RADAR_GAP: f64 = 0.50;
const TWO_CAR_GAP: f64 = 4.;
pub(super) const PASS_BAND: f64 = 0.65;
const SIDE_CONFIRM: f64 = 1.35;
const HOLD: f64 = 0.5;
const FF_HANDOFF_OFFSET: f64 = 0.35;
const FF_HANDOFF_TIME: f64 = 1.05;
const SIDE_RANGE_HOLD: f64 = 0.85;
const SIDE_RANGE_ESTABLISHED: f64 = 0.15;
const SIDE_RANGE_MIN_STREAK: u32 = 15;
const SIDE_RANGE_RECOVERY: f64 = 0.66;

#[derive(Clone, Copy, Debug)]
pub(super) struct SideRange {
    time: f64,
    since: f64,
    lane_u: f64,
    speed: f64,
}

fn same_radar(now: f64, o: &TrackedObject, old: EngineSlot) -> bool {
    old.radar_seen && o.evidence.last_radar_t == Some(now)
        && o.evidence.last_track_id == old.last_track_id
}

fn side_range_hold(
    now: f64,
    o: &TrackedObject,
    old: Option<EngineSlot>,
) -> Option<SideRange> {
    let old = old.filter(|old| old.obj_id == o.obj_id && same_radar(now, o, *old))?;
    old.side_range.filter(|proof| {
        now - proof.time <= SIDE_RANGE_HOLD
            && (o.lane_u - proof.lane_u).abs() <= 0.24
            && (o.v_s - proof.speed).abs() <= 1.4
    })
}


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
    blind: [bool; 2],
    primary: Option<u64>,
    same_lane_ahead: impl Fn(EngineSlot, &TrackedObject) -> bool,
) -> [Option<EngineSlot>; 5] {
    let occupied = previous.map(|slot| slot.map(|o| o.obj_id));
    let mut slots = [None; 5];
    let mut front = Vec::new();
    let mut ff_handoffs = [None; 3];
    let mut sides: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    let mut close: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    let mut camera_queue: [Vec<&TrackedObject>; 2] = [Vec::new(), Vec::new()];
    let mut passing = [None; 2];
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
        // A confirmed previous FF can be closer than the current side vehicle. Remember
        // the chosen side across brief inward excursions while the lane index catches up.
        let extended_handoff = if o.moved && o.lane == 0 && !o.rear && offset < 0.85 {
            if ff_incumbent && offset >= 0.33 && previous[0].is_some_and(|old| {
                old.lane_u * o.lane_u > 0. && same_radar(now, o, old)
            }) {
                Some(now + FF_HANDOFF_TIME)
            } else if offset >= 0.23 && side_incumbent {
                // Keep a measured handoff at the marking while the filtered lane catches up.
                previous[handoff_side + 1]
                    .filter(|old| same_radar(now, o, *old))
                    .and_then(|old| old.ff_handoff_until)
                    .filter(|until| now <= *until || offset >= 0.48)
            } else {
                None
            }
        } else {
            None
        };
        let legacy_handoff = ff_candidate
            && !o.rear
            && o.lane == 0
            && (FF_HANDOFF_OFFSET..1.5).contains(&offset)
            && (leaving_ff || previous_lane == 0 && side_incumbent
                && previous[handoff_side + 1].is_some_and(|old| old.ff_handoff_until.is_none()));
        let ff_handoff = legacy_handoff || extended_handoff.is_some();
        if (!ff_handoff && (ff_candidate || o.lane.abs() != 1)) || o.s > SIDE_FRONT_LIMIT {
            continue;
        }
        let side = usize::from(o.lane > 0);
        let near = o.s <= SPLIT_REAR_MAX + o.var_s.sqrt().min(0.7);
        // A confirmed front vehicle can finish a short, unobserved pass at the origin.
        let side_pass = previous[side + 1].is_some_and(|old| old.obj_id == o.obj_id && old.s >= 0.)
            && near && o.s >= -PASS_BAND && o.v_s < -ALONGSIDE_SPEED
            && now <= o.handoff_until
            && (observed || o.radar_age(now) + o.s.max(0.) / -o.v_s <= FRONT_PASS_RADAR_GAP);
        if side_pass && o.existence_log_odds >= HOLD {
            passing[side] = Some(o);
        }
        let coasting_rear = o.rear_coast(now, occupied[side + 3] == Some(o.obj_id), blind[side]);
        let handoff = o.rear && (REAR_LIMIT..=0.).contains(&o.s) && now <= o.handoff_until;
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
        let side_range = (o.lane.abs() == 1 && !o.rear)
            .then(|| side_range_hold(now, o, previous[side + 1])).flatten();
        let measured_vehicle = vision(now, o)
            || confirmed_stopped
            // Near handoff was confirmed when measured. Current ego speed cannot
            // reclassify a rear coast that has no new relative-speed measurement.
            || coasting_rear
            || o.moved
                && (o.range_consistent
                    || o.same_track_streak >= SIDE_RANGE_MIN_STREAK
                        && side_range.is_some_and(|p| p.time - p.since >= SIDE_RANGE_ESTABLISHED)
                    || near_continuity || handoff || extended_handoff.is_some())
                && (!stationary || o.verified_travel >= 3.)
            || speed <= 5. && o.motion_hits >= 3;
        if ff_handoff {
            let threshold = if side_incumbent { HOLD } else { SIDE_CONFIRM };
            if measured_vehicle
                && o.existence_log_odds >= threshold
                && (observed || o.radar_age(now) <= hold_age)
            {
                let previous_slot = if ff_incumbent { 0 } else { handoff_side + 1 };
                ff_handoffs[previous_slot] = Some((o, extended_handoff));
            }
            continue;
        }
        let threshold = if incumbent { HOLD } else { SIDE_CONFIRM };
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
        if o.s >= REAR_LIMIT && measured_vehicle
            && o.existence_log_odds >= threshold
            && (observed || incumbent && (handoff || coasting_rear || !o.rear && o.radar_age(now) <= hold_age))
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
    let ff = slots[0];
    let front_within_range = |o: &TrackedObject| {
        o.s <= SIDE_FRONT_LIMIT && o.cartesian(frame).0 <= SIDE_FRONT_LIMIT
            && !ff.is_some_and(|lead| same_lane_ahead(lead, o))
    };
    let mut splits = [None; 2];
    let mut crossed = [None; 2];
    for (side, cars) in sides.iter().enumerate() {
        let front_slot = occupied[side + 1];
        let rear_slot = occupied[side + 3];
        let pass = passing[side].filter(|o| front_within_range(o));
        let crossing = pass.filter(|o| o.s <= 0.);
        let behind = cars.iter().copied().filter(|o| o.rear
            || (o.s <= 0. || !seen(now, o)) && o.rear_coast(now, rear_slot == Some(o.obj_id), blind[side]))
            .chain(crossing)
            .min_by(|a, b| {
                seen(now, b).cmp(&seen(now, a)).then_with(|| {
                    cost(now, a, a.s.abs(), rear_slot).total_cmp(&cost(now, b, b.s.abs(), rear_slot))
                })
            });
        let front_cost = |o: &TrackedObject| cost(now, o, o.s, front_slot);
        let front_candidates = || {
            cars.iter()
                .copied()
                .filter(|o| !o.rear && behind.is_none_or(|rear| rear.obj_id != o.obj_id)
                    && front_within_range(o))
        };
        let established_side = front_candidates().any(|o| {
            side_range_hold(now, o, previous[side + 1])
                .is_some_and(|proof| proof.time - proof.since >= SIDE_RANGE_ESTABLISHED)
        });
        let front_default = front_candidates()
            .filter(|o| {
                // A recovering range check must persist before it can displace a
                // continuously measured incumbent. Retain normal admission to an empty slot.
                !established_side || front_slot == Some(o.obj_id) || vision(now, o)
                    || previous[side + 3].is_some_and(|old| old.obj_id == o.obj_id && same_radar(now, o, old))
                    || o.range_recovered_t.is_none_or(|t| now - t >= SIDE_RANGE_RECOVERY)
            })
            .min_by(|a, b| front_cost(a).total_cmp(&front_cost(b)));
        let alongside = front_candidates()
            .chain(close[side].iter().copied())
            .filter(|o| {
                let (range, speed) = if rear_slot == Some(o.obj_id) && now <= o.split_until {
                    (SPLIT_RELEASE, SPLIT_RELEASE_SPEED)
                } else {
                    (SPLIT_REAR_MAX, ALONGSIDE_SPEED)
                };
                pass.is_none()
                    && o.v_s.abs() <= speed
                    && o.s >= SPLIT_REAR_MIN
                    && o.s - o.var_s.sqrt().min(0.7) <= range
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
            // Choose the closest measured front when the pair first forms. Apply
            // incumbent preference once both slots belong to the same ongoing split.
            let far_incumbent = front_slot.filter(|_| rear_slot == Some(near.obj_id));
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
                .min_by(|a, b| {
                    cost(now, a, a.s, far_incumbent).total_cmp(&cost(now, b, b.s, far_incumbent))
                })?;
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
            (pass.or(front_default).filter(|_| crossing.is_none()), behind)
        };
        let handoff = [ff_handoffs[0], ff_handoffs[side + 1]]
            .into_iter()
            .flatten()
            .filter(|(o, until)| {
                pass.is_none() && usize::from(o.lane_u > 0.) == side
                    && slots[0].is_some_and(|ff| ff.obj_id != o.obj_id)
                    && front_within_range(o)
                    && (front.is_none() || until.is_some() && rear.is_none()
                        && front.is_some_and(|car| front_cost(o) < front_cost(car)))
            })
            .min_by(|(a, _), (b, _)| front_cost(a).total_cmp(&front_cost(b)));
        slots[side + 1] = if let Some((o, until)) = handoff {
            let mut selected = slot(frame, o, now);
            selected.ff_handoff_until = until;
            Some(selected)
        } else {
            front.map(|o| {
                let mut selected = slot(frame, o, now);
                selected.side_range = if o.moved && o.range_consistent && selected.radar_seen {
                    Some(SideRange {
                        time: now,
                        since: side_range_hold(now, o, previous[side + 1])
                            .map_or(now, |proof| proof.since),
                        lane_u: o.lane_u,
                        speed: o.v_s,
                    })
                } else {
                    side_range_hold(now, o, previous[side + 1])
                };
                selected
            })
        };
        let rear = rear.or_else(|| cars.iter().copied().find(|o| {
            !seen(now, o) && o.rear_coast(now, rear_slot == Some(o.obj_id), blind[side])
                && slots[side + 1].is_none_or(|chosen| chosen.obj_id != o.obj_id)
        }));
        slots[side + 3] = rear.map(|o| slot(frame, o, now));
        crossed[side] = crossing
            .filter(|o| rear.is_some_and(|r| r.obj_id == o.obj_id))
            .map(|o| o.obj_id);
    }
    for o in objects {
        if splits.contains(&Some(o.obj_id)) {
            o.split_until = now + 0.18;
        }
        if crossed.contains(&Some(o.obj_id)) {
            o.rear = true;
        }
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
    let (x, y) = o.cartesian(frame);
    EngineSlot {
        obj_id: o.obj_id,
        s: o.s,
        d: o.d,
        x,
        y,
        display_d: o.d,
        rear_animated: false,
        v_s: o.v_s,
        kind: o.kind,
        existence_p: o.existence_p(),
        last_track_id: o.evidence.last_track_id,
        lane_u: o.lane_u,
        camera_seen: vision(now, o),
        radar_seen: o.evidence.last_radar_t == Some(now),
        side_range: None,
        ff_handoff_until: None,
    }
}
