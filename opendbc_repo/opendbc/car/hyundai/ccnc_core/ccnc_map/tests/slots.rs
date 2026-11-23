use ccnc_map::rear::{WorldInput, WorldRear};
use ccnc_map::slots::{Assigner, Slot};
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
    near.u = f32::NAN;
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
    let kind = |class, confirmed_at| Some(Kind { class, conf: 0.9, confirmed_at, frames: 2 });
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
        car.kind = Some(Kind { class: KIND_TRUCK, conf: 0.9, confirmed_at: f64::INFINITY, frames: 1 });
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
