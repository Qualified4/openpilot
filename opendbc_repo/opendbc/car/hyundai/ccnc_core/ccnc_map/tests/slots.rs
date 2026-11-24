use ccnc_map::rear::{WorldInput, WorldRear};
use ccnc_map::slots::{keep_radar, Assigner, Slot, VISION_CONFIRMS, VISION_NONE, VISION_ONLY};
use ccnc_map::vehicles::{Flags, Tracker, Vehicle};
use ccnc_map::world::{World, RADAR_LAG};

mod support;
use support::{ego_path, model};

const FF: usize = 0;
const LF: usize = 1;
const RF: usize = 2;

fn road() -> World {
    let mut world = World::default();
    world.model(0., &model(0., 0., 0.));
    world
}

fn vehicle(world: &World, vid: u32, now: f64, x: f64, r: f64) -> Vehicle {
    Vehicle {
        vid,
        wx: x as f32,
        wy: (-3.5 * r) as f32,
        vx: 20.,
        vy: 0.,
        off: 0.,
        u: (world.u_ego() + r) as f32,
        u_slot: (world.u_ego() + r) as f32,
        ids: Default::default(),
        hist: Default::default(),
        link: ccnc_map::vehicles::Link::Sure,
        kind: None,
        seen_radar: now,
        seen_vision: f64::NEG_INFINITY,
        born: 0.,
        flags: Flags {
            proven: true,
            ..Flags::default()
        },
        radar_id: Some(vid),
        oncoming: false,
        oncoming_since: f64::NAN,
        u_mapped: f64::NEG_INFINITY,
    }
}

fn slide(world_y: f64, start: usize, end: usize) {
    let mut world = World::new(RADAR_LAG, true);
    let mut tracker = Tracker::default();
    let mut assigner = Assigner::default();
    let mut transitions = Vec::new();
    let mut previous: Option<f32> = None;
    for step in 0..1200 {
        let t = f64::from(step) * 0.01;
        let (ego_y, heading, yaw) = ego_path(t);
        world.live_pose(t, yaw);
        world.car_state(t, 25., false);
        if step % 5 != 0 {
            continue;
        }
        world.model(t, &model(ego_y, heading, 1.));
        world.radar(t);
        let (sensor_y, sensor_heading, _) = ego_path(t - RADAR_LAG);
        let (sin, cos) = sensor_heading.sin_cos();
        let dx = 30. + 25. * RADAR_LAG;
        let dy = world_y - sensor_y;
        tracker.update(
            t,
            &[(1, cos * dx + sin * dy, -sin * dx + cos * dy, 0.)],
            &world,
        );
        let mut cars = tracker.vehicles().iter().map(|car| {
            let mut qualified = vehicle(&world, car.vid, t, f64::from(car.wx), 0.);
            qualified.wy = car.wy;
            qualified.u = car.u;
            qualified.u_slot = car.u_slot;
            qualified
        });
        let slots = assigner.update(t, cars.next().as_slice(), &world, None);
        if t <= 1. {
            continue;
        }
        let mut shown = slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.map(|s| (i, s.lateral)));
        let (index, lateral) = shown.next().expect("one car stays visible");
        assert!(shown.next().is_none());
        if let Some(last) = previous {
            assert!((lateral - last).abs() <= 0.2, "t={t}: {last} → {lateral}");
        }
        previous = Some(lateral);
        if transitions.last() != Some(&index) {
            transitions.push(index);
        }
    }
    assert_eq!(transitions, [start, end]);
}

#[test]
fn right_car_slides_to_ff_on_ego_right_change() {
    slide(-3.5, RF, FF);
}

#[test]
fn ff_slides_to_lf_on_ego_right_change() {
    slide(0., FF, LF);
}

#[test]
fn slot_distance_is_the_radar_distance_at_100_kph() {
    let speed = 100. / 3.6;
    let mut world = World::new(RADAR_LAG, true);
    let mut tracker = Tracker::default();
    let mut assigner = Assigner::default();
    let mut checked = 0;
    for step in 0..400 {
        let t = f64::from(step) * 0.01;
        world.live_pose(t, 0.);
        world.car_state(t, speed, false);
        if step % 5 != 0 {
            continue;
        }
        world.model(t, &model(0., 0., 1.));
        world.radar(t);
        // A car holding the same speed 30 m ahead: the radar reports dRel 30 m, vRel 0.
        tracker.update(t, &[(1, 30., 0., 0.)], &world);
        let cars: Vec<_> = tracker
            .vehicles()
            .iter()
            .map(|car| {
                let mut qualified = vehicle(&world, car.vid, t, f64::from(car.wx), 0.);
                qualified.wy = car.wy;
                qualified.u = car.u;
                qualified.u_slot = car.u_slot;
                qualified
            })
            .collect();
        let slots = assigner.update(t, &cars, &world, None);
        if t < 1. {
            continue;
        }
        let ff = slots[FF].expect("FF");
        assert!((ff.distance - 30.).abs() <= 0.3, "t={t}: FF {} m", ff.distance);
        checked += 1;
    }
    assert!(checked > 50);
}

#[test]
fn straddling_car_belongs_to_the_lane_containing_its_center() {
    let world = road();
    let slots = Assigner::default().update(0., &[vehicle(&world, 1, 0., 40., 0.4)], &world, None);
    assert_eq!(
        slots.map(|s| s.map(|s| s.vid)),
        [Some(1), None, None, None, None]
    );
}

#[test]
fn slot_changes_as_soon_as_the_center_clears_the_boundary_margin() {
    let world = road();
    let mut assigner = Assigner::default();
    for (t, r, expected) in [
        (0., 0., FF),
        (0.05, 0.51, FF),
        (0.10, 0.53, RF),
        (0.15, 0.49, RF),
        (0.20, 0.47, FF),
    ] {
        let slots = assigner.update(t, &[vehicle(&world, 1, t, 40., r)], &world, None);
        assert_eq!(
            slots.iter().position(Option::is_some),
            Some(expected),
            "t={t}"
        );
    }
}

#[test]
fn a_returning_closer_vid_can_take_the_slot_immediately() {
    let world = road();
    let mut assigner = Assigner::default();
    for (t, x, expected) in [(0., 20., 1), (0.05, 40., 2), (0.10, 20., 1)] {
        let cars = [vehicle(&world, 1, t, x, 0.), vehicle(&world, 2, t, 30., 0.)];
        let slots = assigner.update(t, &cars, &world, None);
        assert_eq!(slots[FF].unwrap().vid, expected, "t={t}");
    }
}

#[test]
fn lead_one_wins_a_close_ff_competition() {
    let world = road();
    let mut assigner = Assigner::default();
    let cars = [
        vehicle(&world, 1, 0., 40., 0.),
        vehicle(&world, 2, 0., 40.5, -0.4),
    ];
    assert_eq!(assigner.update(0., &cars, &world, None)[FF].unwrap().vid, 1);
    let cars = [
        vehicle(&world, 1, 0.05, 40., 0.),
        vehicle(&world, 2, 0.05, 40.5, -0.4),
    ];
    let slots = assigner.update(0.05, &cars, &world, Some((40.5, 1.4)));
    assert_eq!(
        slots.map(|s| s.map(|s| s.vid)),
        [Some(2), None, None, None, None]
    );
}

#[test]
fn lead_one_respects_lane_and_range() {
    let world = road();
    for (distance, r) in [(40., 0.), (20.5, 1.)] {
        let cars = [
            vehicle(&world, 1, 0., 20., 0.),
            vehicle(&world, 2, 0., distance, r),
        ];
        let slots = Assigner::default().update(0., &cars, &world, Some((distance, -3.5 * r)));
        assert_eq!(slots[FF].unwrap().vid, 1);
    }
}

#[test]
fn a_new_radar_frame_replaces_an_unseen_owner() {
    let mut world = road();
    let mut assigner = Assigner::default();
    world.radar(0.);
    let old = vehicle(&world, 1, 0., 20., 0.);
    let slots = assigner.update(0.02, std::slice::from_ref(&old), &world, None);
    assert_eq!(slots[FF].unwrap().vid, 1);

    world.radar(0.05);
    let cars = [old, vehicle(&world, 2, 0.05, 30., 0.)];
    let slots = assigner.update(0.07, &cars, &world, None);
    assert_eq!(slots[FF].unwrap().vid, 2);
}

/// A car 12 m ahead leaves ego's lane to the right at 1.5 m/s, its first step a 0.4 m jump the tracker takes for a
/// reflection, while ego drives `ego_off` lanes right of its lane center. Returns, per radar frame from the start of
/// the move, the drawn slot, `r` from `u` and `r` from `u_slot`.
fn near_cut_out(ego_off: f64) -> Vec<(usize, f64, f64)> {
    let mut world = World::new(RADAR_LAG, false);
    world.blinker(true);
    let mut tracker = Tracker::default();
    let mut assigner = Assigner::default();
    let mut frames = Vec::new();
    let mut last_u: Option<f32> = None;
    for step in 0..80 {
        let t = f64::from(step) * 0.05;
        world.live_pose(t, 0.);
        world.car_state(t, 20., false);
        world.model(t, &model(-3.5 * ego_off, 0., 0.));
        world.radar(t);
        let y = if t < 1. { 0. } else { -0.4 - 1.5 * (t - 1.) };
        tracker.update(t, &[(1, 12., y, 0.)], &world);
        let car = &tracker.vehicles()[0];
        if let Some(last) = last_u {
            assert!((car.u - last).abs() <= 0.1 + 1e-6, "u keeps its 0.1 lane step at t={t}");
        }
        last_u = Some(car.u);
        let mut shown = vehicle(&world, car.vid, t, f64::from(car.wx), 0.);
        (shown.wy, shown.off, shown.u, shown.u_slot) = (car.wy, car.off, car.u, car.u_slot);
        let slots = assigner.update(t, &[shown], &world, None);
        let (index, slot) = slots.iter().enumerate().find_map(|(i, s)| s.map(|s| (i, s))).expect("the car is drawn");
        let u_ego = world.u_ego();
        let r = f64::from(car.u) - u_ego;
        assert!((f64::from(slot.lane_coord_rel) - r).abs() < 1e-6, "lane_coord_rel stays from u");
        if t >= 1. {
            frames.push((index, r, f64::from(car.u_slot) - u_ego));
        }
    }
    assert!(frames[0].1 < 0.2, "the reflection jump stays in off");
    frames
}

fn switch_frame(frames: &[(usize, f64, f64)]) -> usize {
    assert_eq!(frames[0].0, FF);
    let switch = frames.iter().position(|f| f.0 == RF).expect("the car reaches RF");
    assert!(frames[switch..].iter().all(|f| f.0 == RF), "one switch");
    switch
}

#[test]
fn a_near_cut_out_changes_slot_from_u_slot_before_u_crosses() {
    // The blinker alone (on in every frame here) is no lane change.
    let frames = near_cut_out(0.);
    let switch = switch_frame(&frames);
    let (_, r, r_slot) = frames[switch];
    assert!(r_slot >= 0.52 - 1e-9 && frames[switch - 1].2 < 0.52, "switches as u_slot clears the margin");
    assert!(r < 0.5, "u still has the car in ego's lane: {r}");
    let late = frames.iter().position(|f| f.1 >= 0.52).expect("u crosses later");
    assert!(late >= switch + 5, "at least 0.25 s earlier: {switch} vs {late}");
}

#[test]
fn an_ego_lane_change_keeps_the_slot_on_u() {
    let early = switch_frame(&near_cut_out(0.));
    let frames = near_cut_out(0.3);
    let switch = switch_frame(&frames);
    assert!(frames[switch].1 >= 0.52 - 1e-9 && frames[switch - 1].1 < 0.52, "switches as u clears the margin");
    assert!(frames[switch - 1].2 >= 0.52, "not on u_slot");
    assert!(switch > early);
}

#[test]
fn ego_changing_follows_the_lane_center_with_a_3_s_hold() {
    let mut world = World::new(RADAR_LAG, false);
    let mut changing = Vec::new();
    for step in 0..200 {
        let t = f64::from(step) * 0.05;
        world.live_pose(t, 0.);
        world.car_state(t, 20., false);
        // Ego 0.3 lane off its center from 2 s to 3 s, back on it after.
        let y = if (2. ..3.).contains(&t) { -0.3 * 3.5 } else { 0. };
        world.model(t, &model(y, 0., 0.));
        world.radar(t);
        changing.push((t, world.ego_changing()));
    }
    let at = |t: f64| changing.iter().find(|(s, _)| (s - t).abs() < 1e-6).unwrap().1;
    assert!(!at(1.9));
    assert!(at(2.5) && at(5.5));
    assert!(!at(6.5));
    world.blinker(true);
    assert!(!world.ego_changing(), "the blinker alone is no lane change");
}

#[test]
fn u_step_limited_to_0_1_lane() {
    let mut world = World::default();
    let mut narrow = model(0., 0., 0.);
    for line in &mut narrow.lane_lines {
        for y in &mut line.y {
            *y *= 2. / 3.5;
        }
    }
    world.model(0., &narrow);
    let mut tracker = Tracker::default();
    let before = tracker.update(0., &[(1, 40., 0., 0.)], &world)[0].u;
    // A larger coordinate innovation that is within the same-ID door.
    let car = &tracker.update(0.25, &[(1, 40., -1.49, 0.)], &world)[0];
    assert_eq!(car.vid, 1);
    assert!((car.u - before - 0.1).abs() < 1e-6);
}

#[test]
fn unmapped_near_car_uses_ego_lane_lateral() {
    let world = road();
    let mut near = vehicle(&world, 1, 0., 4.9, 0.);
    near.wy = -3.5;
    (near.u, near.u_slot) = (f32::NAN, f32::NAN);
    let slots = Assigner::default().update(0., &[near], &world, None);
    let shown = slots[RF].unwrap();
    assert!((shown.lane_coord_rel - 1.).abs() < 1e-6);
    assert!((shown.lateral + 3.5).abs() < 1e-6);
}

#[test]
fn mapped_car_keeps_the_same_lateral_across_5m() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut previous = None;
    for (t, x) in [(0., 5.1), (0.05, 4.9)] {
        let mut car = vehicle(&world, 1, t, x, 1.);
        car.wy = -1.8;
        let slots = assigner.update(t, &[car], &world, None);
        let lateral = slots[RF].unwrap().lateral;
        assert_eq!(lateral, previous.unwrap_or(lateral));
        previous = Some(lateral);
    }
}

#[test]
fn backlash_0_0025_per_m() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut y = 0.;
    let draw = |assigner: &mut Assigner, i: u32, y: f64| -> Slot {
        let t = f64::from(i) * 0.05;
        assigner.update(t, &[vehicle(&world, 1, t, 40., -y / 3.5)], &world, None)[FF].unwrap()
    };
    for i in 0..80 {
        let raw = if (i / 4) % 2 == 0 { 0. } else { 0.15 };
        y = f64::from(draw(&mut assigner, i, raw).lateral);
        assert!(y >= -1e-6 && y <= 0.05 + 1e-6);
    }
    assert!((y - 0.05).abs() < 1e-6);
    for i in 80..120 {
        let raw = 0.15 + f64::from(i - 79) * 0.025;
        let next = f64::from(draw(&mut assigner, i, raw).lateral);
        assert!((next - y - 0.025).abs() < 1e-6);
        y = next;
    }
}

#[test]
fn rear_slots_use_r_and_active_bsd_uses_the_existing_drawing() {
    let mut world = road();
    let mut assigner = Assigner::default();
    let mut rear = WorldRear::default();
    for (now, blind) in [(0., [false; 2]), (0.05, [true, false])] {
        world.radar(now);
        let cars = [
            vehicle(&world, 1, now, -3., -1.),
            vehicle(&world, 2, now, -3., 1.),
        ];
        let mut slots = assigner.update(now, &cars, &world, None);
        rear.finish(
            &world,
            WorldInput {
                now,
                changing: false,
                can_valid: true,
                blind,
                vehicles: &cars,
            },
            &mut slots,
        )
        .unwrap();
        let left = slots[3].unwrap();
        let right = slots[4].unwrap();
        assert_eq!(right.vid, 2);
        assert!(right.distance < 0. && right.lateral < 0.);
        assert!(left.distance < 0. && left.lateral > 0.);
        assert_eq!(left.vid, if blind[0] { 0 } else { 1 });
    }
}

#[test]
fn a_confirmed_truck_and_bus_keep_their_kind_in_the_rear_slots_after_the_tracker_forgets_them() {
    use ccnc_map::vehicles::Kind;
    use ccnc_map::vision::kind::{KIND_BUS, KIND_TRUCK};
    let mut world = road();
    let mut assigner = Assigner::default();
    let mut rear = WorldRear::default();
    let kind = |class, confirmed_at| Some(Kind { class, conf: 0.9, confirmed_at, frames: 2, inherited: false });
    // Ego 10 m/s, two cars 2 m/s slower pass from 6 m ahead: a confirmed truck on the left, a confirmed bus on the
    // right. The radar loses them 3 m behind with the BSD on; the tracker forgets them 2 s (RETAIN) later.
    let lost = 4.5;
    let mut checked = 0;
    for i in 0..=((lost + 9.5) * 20.) as i32 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, 10., false);
        if i % 20 == 0 {
            world.model(now, &model(0., 0., 0.));
        }
        world.radar(now);
        let seen = now.min(lost);
        let mut cars = vec![
            vehicle(&world, 1, seen, 8. * seen + 6., -1.),
            vehicle(&world, 2, seen, 8. * seen + 6., 1.),
        ];
        cars[0].kind = kind(KIND_TRUCK, 0.);
        cars[1].kind = kind(KIND_BUS, 0.);
        if now - lost > 2. {
            cars.clear();
        }
        let blind = [now >= lost - 1.; 2];
        let mut slots = assigner.update(now, &cars, &world, None);
        rear.finish(&world, WorldInput { now, changing: false, can_valid: true, blind, vehicles: &cars }, &mut slots).unwrap();
        let after = now - lost;
        if [2.05, 5., 9.].iter().any(|t| (after - t).abs() < 1e-6) {
            assert!(cars.is_empty(), "the tracker has forgotten them at {after:.2} s");
            let (left, right) = (slots[3].unwrap(), slots[4].unwrap());
            assert_eq!((left.vid, left.kind, left.kind_confirmed), (1, KIND_TRUCK, true), "LR at {after:.2} s");
            assert_eq!((right.vid, right.kind, right.kind_confirmed), (2, KIND_BUS, true), "RR at {after:.2} s");
            checked += 1;
        }
    }
    assert_eq!(checked, 3);
}

#[test]
fn an_ambiguous_inherited_truck_keeps_its_kind_in_the_rear_slots_including_after_2s() {
    use ccnc_map::vehicles::{Kind, Link};
    use ccnc_map::vision::kind::{KIND_BUS, KIND_TRUCK};
    let mut world = road();
    let mut assigner = Assigner::default();
    let mut rear = WorldRear::default();
    let lost = 4.5;
    let mut checked_tracked = 0;
    let mut checked_forgotten = 0;
    for i in 0..=((lost + 9.5) * 20.) as i32 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, 10., false);
        if i % 20 == 0 {
            world.model(now, &model(0., 0., 0.));
        }
        world.radar(now);
        let seen = now.min(lost);
        let mut left_car = vehicle(&world, 1, seen, 8. * seen + 6., -1.);
        left_car.link = Link::Ambiguous;
        left_car.kind = Some(Kind { class: KIND_TRUCK, conf: 0.9, confirmed_at: 0., frames: 2, inherited: true });
        let mut right_car = vehicle(&world, 2, seen, 8. * seen + 6., 1.);
        right_car.link = Link::Ambiguous;
        right_car.kind = Some(Kind { class: KIND_BUS, conf: 0.9, confirmed_at: 0., frames: 2, inherited: false });
        let cars = if now - lost > 2. { vec![] } else { vec![left_car, right_car] };
        let blind = [now >= lost - 1.; 2];
        let mut slots = assigner.update(now, &cars, &world, None);
        rear.finish(&world, WorldInput { now, changing: false, can_valid: true, blind, vehicles: &cars }, &mut slots).unwrap();
        let after = now - lost;
        if (after - 0.5).abs() < 1e-6 {
            assert!(!cars.is_empty());
            let (left, right) = (slots[3].unwrap(), slots[4].unwrap());
            assert_eq!((left.vid, left.kind, left.kind_confirmed), (1, KIND_TRUCK, true), "inherited ambiguous truck kept in LR");
            assert_eq!((right.vid, right.kind, right.kind_confirmed), (2, 0, false), "non-inherited ambiguous bus stays hidden in RR");
            checked_tracked += 1;
        }
        if [2.05, 5., 9.].iter().any(|t| (after - t).abs() < 1e-6) {
            assert!(cars.is_empty(), "tracker has forgotten them at {after:.2} s");
            let (left, right) = (slots[3].unwrap(), slots[4].unwrap());
            assert_eq!((left.vid, left.kind, left.kind_confirmed), (1, KIND_TRUCK, true), "inherited ambiguous truck kept in LR after 2 s");
            assert_eq!((right.vid, right.kind, right.kind_confirmed), (2, 0, false), "non-inherited ambiguous bus not saved in RR");
            checked_forgotten += 1;
        }
    }
    assert_eq!((checked_tracked, checked_forgotten), (1, 3));
}

#[test]
fn an_unconfirmed_kind_is_not_kept_after_the_tracker_forgets_the_car() {
    use ccnc_map::vehicles::Kind;
    use ccnc_map::vision::kind::KIND_TRUCK;
    let mut world = road();
    let mut assigner = Assigner::default();
    let mut rear = WorldRear::default();
    let lost = 4.5;
    for i in 0..=((lost + 3.) * 20.) as i32 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, 10., false);
        if i % 20 == 0 {
            world.model(now, &model(0., 0., 0.));
        }
        world.radar(now);
        let seen = now.min(lost);
        let mut car = vehicle(&world, 1, seen, 8. * seen + 6., -1.);
        car.kind = Some(Kind { class: KIND_TRUCK, conf: 0.9, confirmed_at: f64::INFINITY, frames: 1, inherited: false });
        let cars: Vec<_> = if now - lost > 2. { vec![] } else { vec![car] };
        let mut slots = assigner.update(now, &cars, &world, None);
        let blind = [now >= lost - 1., false];
        rear.finish(&world, WorldInput { now, changing: false, can_valid: true, blind, vehicles: &cars }, &mut slots).unwrap();
        if let Some(left) = slots[3].filter(|_| now >= lost) {
            let shown = if cars.is_empty() { (0, false) } else { (KIND_TRUCK, false) };
            assert_eq!((left.vid, left.kind, left.kind_confirmed), (1, shown.0, shown.1), "at {:.2} s", now - lost);
        }
    }
}

/// Ego stopped; a car waits in the left lane 5 m ahead and the radar loses it at 1 s.
fn stop_memory(blind: [bool; 2], remember: bool, next: impl Fn(f64) -> Vec<Vehicle>) -> Vec<(f64, Option<Slot>)> {
    let mut world = road();
    let mut assigner = Assigner::default();
    let mut out = Vec::new();
    for i in 0..=80 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, 0., false);
        world.radar(now);
        let mut waiting = vehicle(&world, 1, now.min(1.), 5., -1.);
        waiting.vx = 0.;
        let mut cars = vec![waiting];
        if now > 1. {
            cars.extend(next(now));
        }
        let mut slots = assigner.update(now, &cars, &world, None);
        if now <= 1. {
            assert_eq!(slots[LF].map(|s| s.vid), Some(1), "measured at {now:.2}");
        } else {
            assigner.hold_stopped(now, &cars, &world, blind, |car| remember && car.vid == 1, &mut slots);
            out.push((now, slots[LF]));
        }
    }
    out
}

#[test]
fn a_stopped_ego_keeps_a_waiting_side_car_after_the_radar_loses_it() {
    let kept = stop_memory([false; 2], true, |_| vec![]);
    assert!(kept.iter().all(|(now, s)| s.is_some_and(|s| s.vid == 1 && (s.distance - 5.).abs() < 1e-3)
        && *now > 1.), "{kept:?}");
    // Not when the radar map would not remember it, or with the BSD on that side.
    assert!(stop_memory([false; 2], false, |_| vec![]).iter().all(|(_, s)| s.is_none()));
    assert!(stop_memory([true, false], true, |_| vec![]).iter().all(|(_, s)| s.is_none()));
}

#[test]
fn another_car_in_the_slot_ends_the_stop_memory() {
    let shown = stop_memory([false; 2], true, |now| {
        if now >= 2. { vec![vehicle(&road(), 2, now, 8., -1.)] } else { vec![] }
    });
    for (now, s) in shown {
        assert_eq!(s.map(|s| s.vid), Some(if now >= 2. { 2 } else { 1 }), "at {now:.2}");
    }
}

#[test]
fn a_far_car_passing_through_the_slot_leaves_the_waiting_car_remembered() {
    let shown = stop_memory([false; 2], true, |now| {
        if now < 1.5 { vec![vehicle(&road(), 2, now, 70., -1.)] } else { vec![] }
    });
    for (now, s) in shown {
        assert_eq!(s.map(|s| s.vid), Some(if now < 1.5 { 2 } else { 1 }), "at {now:.2}");
    }
}

#[test]
fn a_remembered_distance_follows_ego_travel_after_the_last_measurement() {
    // Ego brakes from 3 m/s to a stop as the radar loses the waiting car.
    let mut world = World::new(RADAR_LAG, true);
    world.model(0., &model(0., 0., 0.));
    let mut assigner = Assigner::default();
    let mut last_drawn = None;
    for i in 0..=60 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, if now < 1. { 3. } else { 0. }, false);
        world.radar(now);
        let mut waiting = vehicle(&world, 1, now.min(1.), 8., -1.);
        waiting.vx = 0.;
        let cars = [waiting];
        let mut slots = assigner.update(now, &cars, &world, None);
        if now <= 1. {
            last_drawn = slots[LF].map(|s| s.distance);
            continue;
        }
        let travel = world.pose.at(now).s - world.pose.at(1.).s;
        assigner.hold_stopped(now, &cars, &world, [false; 2], |_| true, &mut slots);
        let kept = slots[LF].expect("remembered").distance;
        assert!((f64::from(kept) - (f64::from(last_drawn.unwrap()) - travel)).abs() < 0.05, "at {now:.2}: {kept}");
    }
}

#[test]
fn a_remembered_car_stays_when_the_camera_would_replace_it() {
    let radar_car = Slot {
        vid: 7,
        distance: 6.,
        lateral: 3.5,
        detect: 1,
        lane_coord_rel: -1.,
        v_rel: 0.,
        kind: 0,
        kind_confirmed: false,
        kind_wanted: false,
        vision: VISION_NONE,
    };
    let camera_car = Slot { vid: 0xFFFF_FFF1, distance: 3., vision: VISION_ONLY, ..radar_car };
    let mut radar = [None; 5];
    radar[LF] = Some(radar_car);
    radar[RF] = Some(Slot { lateral: -3.5, ..radar_car });
    // The camera swaps both sides; only the remembered LF stays the radar's, now confirmed.
    let mut slots = radar;
    slots[LF] = Some(camera_car);
    slots[RF] = Some(Slot { lateral: -3.5, ..camera_car });
    keep_radar([false, true, false, false, false], &radar, &mut slots);
    assert_eq!(slots[LF], Some(Slot { vision: VISION_CONFIRMS, ..radar_car }));
    assert_eq!(slots[RF].map(|s| s.vision), Some(VISION_ONLY));
    // A camera confirmation, or no camera answer, leaves the slot as it is.
    let mut slots = radar;
    slots[LF] = Some(Slot { vision: VISION_CONFIRMS, ..radar_car });
    keep_radar([false, true, true, false, false], &radar, &mut slots);
    assert_eq!(slots[LF].map(|s| (s.vid, s.vision)), Some((7, VISION_CONFIRMS)));
    assert_eq!(slots[RF], radar[RF]);
}

const LR: usize = 3;
const RR: usize = 4;

/// Cars in one side lane (r -1 left, 1 right) at these distances, vids 1, 2, ..., drawn and then split.
fn side_pair(assigner: &mut Assigner, world: &World, now: f64, xs: &[f64], r: f64) -> [Option<Slot>; 5] {
    let cars: Vec<_> = xs.iter().enumerate().map(|(i, &x)| vehicle(world, i as u32 + 1, now, x, r)).collect();
    let mut slots = assigner.update(now, &cars, world, None);
    assigner.split_alongside(now, &cars, &mut slots);
    slots
}

fn left_pair(assigner: &mut Assigner, world: &World, now: f64, xs: &[f64]) -> [Option<Slot>; 5] {
    side_pair(assigner, world, now, xs, -1.)
}

#[test]
fn a_car_alongside_with_another_ahead_is_drawn_in_the_rear_slot() {
    let world = road();
    let slots = left_pair(&mut Assigner::default(), &world, 0., &[3., 35.]);
    assert_eq!(slots[LR].map(|s| (s.vid, s.distance)), Some((1, 0.)));
    assert_eq!(slots[LF].map(|s| (s.vid, s.distance)), Some((2, 35.)));
    // The rear drawing keeps the car's observed lateral.
    let alone = Assigner::default().update(0., &[vehicle(&world, 1, 0., 3., -1.)], &world, None);
    assert_eq!(slots[LR].unwrap().lateral, alone[LF].unwrap().lateral);
}

#[test]
fn a_single_car_alongside_stays_in_the_front_slot() {
    let world = road();
    let slots = left_pair(&mut Assigner::default(), &world, 0., &[3.]);
    assert_eq!(slots[LF].map(|s| (s.vid, s.distance)), Some((1, 3.)));
    assert_eq!(slots[LR], None);
}

#[test]
fn the_side_pair_switches_once_each_way() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut switches = Vec::new();
    let mut previous = None;
    // The near car comes from 5.5 m to 4.5 m and back out to 6.5 m, with 0.3 m of noise.
    let path = (0..=40).map(|i| 5.5 - 0.025 * f64::from(i)).chain((0..=80).map(|i| 4.5 + 0.025 * f64::from(i)));
    for (i, x) in path.enumerate() {
        let now = i as f64 * 0.05;
        let noise = if i % 2 == 0 { 0.3 } else { -0.3 };
        let slots = left_pair(&mut assigner, &world, now, &[x + noise, 35.]);
        let alongside = slots[LR].is_some_and(|s| s.vid == 1);
        assert_eq!(slots[LF].unwrap().vid, if alongside { 2 } else { 1 }, "at {now:.2}");
        if previous.is_some_and(|p| p != alongside) {
            switches.push((now, alongside));
        }
        previous = Some(alongside);
    }
    assert_eq!(switches.iter().map(|s| s.1).collect::<Vec<_>>(), [true, false], "{switches:?}");
}

#[test]
fn the_car_alongside_replaces_a_rear_handover_drawing() {
    let world = road();
    let mut assigner = Assigner::default();
    let cars = [vehicle(&world, 1, 0., 3., -1.), vehicle(&world, 2, 0., 35., -1.)];
    let mut slots = assigner.update(0., &cars, &world, None);
    slots[LR] = Some(Slot { vid: 9, distance: -8., ..slots[LF].unwrap() });
    assert_eq!(assigner.split_alongside(0., &cars, &mut slots), [true, false]);
    assert_eq!(slots[LR].map(|s| s.vid), Some(1));
}

#[test]
fn a_split_side_stays_split_while_the_next_car_drops_out_briefly() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut front = Vec::new();
    for i in 0..40 {
        let now = f64::from(i) * 0.05;
        // The car 35 m ahead drops out for 0.3 s from 0.5 s.
        let gone = (10..16).contains(&i);
        let slots = left_pair(&mut assigner, &world, now, if gone { &[3.] } else { &[3., 35.] });
        assert_eq!(slots[LR].map(|s| (s.vid, s.distance)), Some((1, 0.)), "at {now:.2}");
        let lf = slots[LF].map(|s| s.vid);
        if front.last() != Some(&lf) {
            front.push(lf);
        }
    }
    assert_eq!(front, [Some(2), None, Some(2)]);
}

#[test]
fn a_split_side_without_its_next_car_for_over_1_s_returns_the_near_car_to_the_front_once() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut drawn = Vec::new();
    for i in 0..60 {
        let now = f64::from(i) * 0.05;
        // The car 35 m ahead is gone from 0.5 s on, more than 1.2 s.
        let slots = left_pair(&mut assigner, &world, now, if i < 10 { &[3., 35.] } else { &[3.] });
        let state = (slots[LF].map(|s| (s.vid, s.distance)), slots[LR].map(|s| (s.vid, s.distance)));
        if drawn.last().is_none_or(|(_, last)| *last != state) {
            drawn.push((now, state));
        }
    }
    let states: Vec<_> = drawn.iter().map(|d| d.1).collect();
    assert_eq!(
        states,
        [(Some((2, 35.)), Some((1, 0.))), (None, Some((1, 0.))), (Some((1, 3.)), None)],
        "{drawn:?}"
    );
    // Held 1 s from the frame the next car went missing.
    assert!((drawn[2].0 - 1.55).abs() < 1e-9, "{drawn:?}");
}

#[test]
fn a_car_alongside_on_the_right_with_another_ahead_is_drawn_in_rr() {
    let world = road();
    let slots = side_pair(&mut Assigner::default(), &world, 0., &[3., 35.], 1.);
    assert_eq!(slots[RR].map(|s| (s.vid, s.distance)), Some((1, 0.)));
    assert_eq!(slots[RF].map(|s| (s.vid, s.distance)), Some((2, 35.)));
    assert_eq!((slots[LF], slots[LR]), (None, None));
}

/// Left lane: the near car at 3 m, unreported (radar coverage ends alongside) while `lost` holds, and the next at 35 m.
fn near_lost(assigner: &mut Assigner, world: &World, now: f64, lost: bool) -> [Option<Slot>; 5] {
    let near = vehicle(world, 1, if lost { now - 1. } else { now }, 3., -1.);
    let cars = [near, vehicle(world, 2, now, 35., -1.)];
    let mut slots = assigner.update(now, &cars, world, None);
    assigner.split_alongside(now, &cars, &mut slots);
    slots
}

#[test]
fn a_split_near_car_stays_alongside_while_its_reports_drop_out_briefly() {
    let world = road();
    let mut assigner = Assigner::default();
    for i in 0..40 {
        let now = f64::from(i) * 0.05;
        // The near car's reports stop for 0.3 s from 0.5 s; the tracker keeps it.
        let slots = near_lost(&mut assigner, &world, now, (10..16).contains(&i));
        assert_eq!(slots[LR].map(|s| (s.vid, s.distance)), Some((1, 0.)), "at {now:.2}");
        assert_eq!(slots[LF].map(|s| s.vid), Some(2), "at {now:.2}");
    }
}

#[test]
fn a_split_near_car_unreported_for_1_2_s_stays_alongside() {
    let world = road();
    let mut assigner = Assigner::default();
    for i in 0..50 {
        let now = f64::from(i) * 0.05;
        // Unreported for 1.2 s from 0.5 s.
        let slots = near_lost(&mut assigner, &world, now, (10..34).contains(&i));
        assert_eq!(slots[LR].map(|s| (s.vid, s.distance)), Some((1, 0.)), "at {now:.2}");
        assert_eq!(slots[LF].map(|s| s.vid), Some(2), "at {now:.2}");
    }
}

#[test]
fn a_split_near_car_unreported_for_over_2_s_leaves_the_rear_slot_once() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut rear = Vec::new();
    for i in 0..60 {
        let now = f64::from(i) * 0.05;
        // Unreported from 0.5 s on, 2.3 s and more.
        let slots = near_lost(&mut assigner, &world, now, i >= 10);
        let lr = slots[LR].map(|s| s.vid);
        if rear.last().is_none_or(|(_, last)| *last != lr) {
            rear.push((now, lr));
        }
        assert_eq!(slots[LF].map(|s| s.vid), Some(2), "at {now:.2}");
    }
    assert_eq!(rear.iter().map(|r| r.1).collect::<Vec<_>>(), [Some(1), None], "{rear:?}");
    assert!((rear[1].0 - 2.55).abs() < 1e-9, "{rear:?}");
    // Gone from the tracker, it is not drawn at all.
    let mut assigner = Assigner::default();
    near_lost(&mut assigner, &world, 0., false);
    let cars = [vehicle(&world, 2, 0.05, 35., -1.)];
    let mut slots = assigner.update(0.05, &cars, &world, None);
    assigner.split_alongside(0.05, &cars, &mut slots);
    assert_eq!(slots[LR], None);
}

#[test]
fn a_split_near_car_falling_behind_goes_on_in_the_rear_slot_as_the_same_car() {
    let world = road();
    let mut assigner = Assigner::default();
    let mut drawn = Vec::new();
    for (i, x) in [2., 1., 0.2, -0.5, -1., -2.].into_iter().enumerate() {
        let now = i as f64 * 0.05;
        let slots = left_pair(&mut assigner, &world, now, &[x, 35.]);
        drawn.push(slots[LR].map(|s| (s.vid, s.distance)));
    }
    assert_eq!(drawn, [Some((1, 0.)), Some((1, 0.)), Some((1, 0.)), Some((1, -0.5)), Some((1, -1.)), Some((1, -2.))]);
}

/// Ego waits while one radar car in the left lane moves along `x(t)` at ground speed `v(t)` and stops. Returns
/// the left slot per frame and whether the tracker marked the car oncoming.
fn left_lane_car(x: impl Fn(f64) -> f64, v: impl Fn(f64) -> f64) -> (Vec<(f64, Option<u32>)>, bool) {
    let mut world = road();
    let mut tracker = Tracker::default();
    let mut assigner = Assigner::default();
    let mut shown = Vec::new();
    for i in 0..=100 {
        let now = f64::from(i) * 0.05;
        world.car_state(now, 0., false);
        world.radar(now);
        tracker.update(now, &[(7, x(now), 3.5, v(now))], &world);
        let cars: Vec<_> = tracker
            .vehicles()
            .iter()
            .map(|car| Vehicle {
                wx: car.wx,
                wy: car.wy,
                vx: car.vx,
                vy: car.vy,
                oncoming: car.oncoming,
                ..vehicle(&world, car.vid, now, f64::from(car.wx), -1.)
            })
            .collect();
        let mut slots = assigner.update(now, &cars, &world, None);
        assigner.hold_stopped(now, &cars, &world, [false; 2], |_| true, &mut slots);
        shown.push((now, slots[LF].map(|s| s.vid)));
    }
    (shown, tracker.vehicles().iter().any(|car| car.oncoming))
}

#[test]
fn an_oncoming_car_that_stops_beside_ego_stays_out_of_the_side_slot() {
    // 25 m ahead at -5 m/s, braking at 1.25 m/s² to a stop 5 m ahead at 4 s, then standing (the old -1 km/h rule
    // let it in from 3.8 s on).
    let x = |t: f64| { let t = t.min(4.); 25. - 5. * t + 0.625 * t * t };
    let v = |t: f64| (-5. + 1.25 * t).min(0.);
    let (shown, oncoming) = left_lane_car(x, v);
    assert!(oncoming);
    assert!(shown.iter().all(|(_, vid)| vid.is_none()), "{shown:?}");
}

#[test]
fn a_car_of_our_direction_that_stops_beside_ego_is_drawn() {
    // 8 m ahead pulling away at 2 m/s, braking to a stop at 9 m.
    let x = |t: f64| 8. + (2. * t - t * t).max(0.).min(1.);
    let v = |t: f64| (2. - 2. * t).max(0.);
    let (shown, oncoming) = left_lane_car(x, v);
    assert!(!oncoming);
    assert!(shown[10..].iter().all(|&(_, vid)| vid.is_some()), "{shown:?}");
}

#[test]
fn a_side_car_near_its_outer_line_stays_in_the_side_slot() {
    let world = road();
    let mut assigner = Assigner::default();
    // r -1.6 is past the outer line of LF (-1.5) but not 0.35 lane into the lane beyond: still LF.
    let steps = [(-1.0, true), (-1.6, true), (-1.64, true), (-1.0, true), (-1.7, false), (-1.6, false), (-1.45, true)];
    for (i, (r, drawn)) in steps.into_iter().enumerate() {
        let now = i as f64 * 0.05;
        let slots = assigner.update(now, &[vehicle(&world, 4, now, 10., r)], &world, None);
        assert_eq!(slots[LF].is_some(), drawn, "r {r}");
    }
}

/// One radar car 3.5 m left at `x(t)`, moving over ground at `v` m/s, while ego waits: whether it has a lane
/// coordinate per frame and whether the lane map could place it at that frame.
fn far_car(x: impl Fn(f64) -> f64, v: f64) -> Vec<(f64, bool, bool)> {
    let mut world = road();
    let mut tracker = Tracker::default();
    (0..=40)
        .map(|i| {
            let now = f64::from(i) * 0.05;
            world.car_state(now, 0., false);
            world.radar(now);
            let car = tracker.update(now, &[(9, x(now), 3.5, v)], &world)[0].u;
            let mapped = world.vehicle_lane_coord(now, x(now), (x(now), 3.5)).is_some();
            (now, car.is_finite(), mapped)
        })
        .collect()
}

#[test]
fn a_car_the_lane_map_misses_briefly_keeps_its_lane_coordinate() {
    // Out past the mapped 80 m for 0.3 s and back.
    let frames = far_car(|t| if (0.5..0.8).contains(&t) { 81. } else { 79. }, 3.);
    assert!(frames.iter().any(|f| !f.2), "the map misses it");
    assert!(frames.iter().all(|f| f.1), "{frames:?}");
}

#[test]
fn a_standing_object_the_lane_map_misses_loses_its_lane_coordinate_at_once() {
    let frames = far_car(|t| if (0.5..0.8).contains(&t) { 81. } else { 79. }, 0.);
    assert!(frames.iter().all(|f| f.1 == f.2), "{frames:?}");
}

#[test]
fn a_car_the_lane_map_misses_for_over_half_a_second_loses_its_lane_coordinate() {
    let frames = far_car(|t| if t >= 0.5 { 81. } else { 79. }, 3.);
    let missed = frames.iter().position(|f| !f.2).unwrap();
    let lost = frames.iter().position(|f| !f.1).unwrap();
    assert!((frames[lost].0 - frames[missed - 1].0 - 0.55).abs() < 1e-6, "{frames:?}");
}
