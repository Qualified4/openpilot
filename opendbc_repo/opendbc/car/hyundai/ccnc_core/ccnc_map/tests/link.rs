use ccnc_map::vehicles::{Kind, Link, LinkGates, Tracker, MAX_VEHICLES};
use ccnc_map::world::World;

fn tracker() -> Tracker {
    Tracker::new(LinkGates::BODY)
}

fn point(id: u32, x: f64, y: f64) -> (u32, f64, f64, f64) {
    (id, x, y, 0.)
}

#[test]
fn handoff_along_the_body_keeps_vid() {
    let world = World::default();
    let mut tracker = tracker();
    let vid = tracker.update(1., &[point(10, 20., 0.)], &world)[0].vid;
    tracker.update(1.05, &[], &world);
    let cars = tracker.update(1.3, &[point(20, 24.5, 0.1)], &world);
    assert_eq!(cars.len(), 1);
    assert_eq!((cars[0].vid, cars[0].link), (vid, Link::Sure));
    assert_eq!(tracker.vid_at(1., 10), Some(vid));
    assert_eq!(tracker.vid_at(1.3, 10), None);
    assert_eq!(tracker.vid_at(1.3, 20), Some(vid));
}

#[test]
fn ambiguous_handoff_starts_new_vid() {
    let world = World::default();
    let mut tracker = tracker();
    tracker.update(1., &[point(10, 20., -0.4), point(11, 20., 0.6)], &world);
    tracker.update(1.3, &[point(20, 20., 0.)], &world);
    let vid = tracker.vid_at(1.3, 20).unwrap();
    assert_ne!(Some(vid), tracker.vid_at(1., 10));
    assert_ne!(Some(vid), tracker.vid_at(1., 11));
    assert_eq!(tracker.vehicle_mut(vid).unwrap().link, Link::Ambiguous);
}

#[test]
fn covered_point_is_not_linked() {
    let world = World::default();
    let mut tracker = tracker();
    tracker.update(1., &[point(10, 20., 0.), point(11, 20., 0.6)], &world);
    tracker.update(1.3, &[point(20, 20., 0.), point(11, 20., 0.6)], &world);
    assert_ne!(tracker.vid_at(1., 10), tracker.vid_at(1.3, 20));
    assert_eq!(tracker.vid_at(1., 11), tracker.vid_at(1.3, 11));
}

#[test]
fn same_id_lateral_jump_splits() {
    let world = World::default();
    let mut tracker = tracker();
    tracker.update(1., &[point(10, 20., 0.), point(11, 20., 1.6)], &world);
    tracker.update(1.05, &[point(10, 20., 1.6)], &world);
    assert_ne!(tracker.vid_at(1., 10), tracker.vid_at(1.05, 10));
    assert_ne!(tracker.vid_at(1., 11), tracker.vid_at(1.05, 10));
}

#[test]
fn gap_over_0_8s_needs_vision() {
    let world = World::default();
    for new_id in [10, 20] {
        let mut tracker = tracker();
        tracker.update(1., &[point(10, 20., 0.)], &world);
        tracker.update(1.9, &[point(new_id, 20., 0.)], &world);
        assert_ne!(tracker.vid_at(1., 10), tracker.vid_at(1.9, new_id));
    }
}

#[test]
fn vehicle_deleted_after_2s() {
    let world = World::default();
    let mut tracker = tracker();
    tracker.update(1., &[point(10, 20., 0.)], &world);
    assert_eq!(tracker.update(3., &[], &world).len(), 1);
    assert!(tracker.update(3.01, &[], &world).is_empty());
    assert_eq!(tracker.vid_at(1., 10), None);
}

#[test]
fn a_kept_vehicle_outlives_2s() {
    let world = World::default();
    let mut tracker = tracker();
    let kept = tracker.update(1., &[point(10, 5., 3.5)], &world)[0].vid;
    tracker.update(1.05, &[point(11, 30., 0.)], &world);
    tracker.keep([Some(kept), None]);
    let cars = tracker.update(4., &[], &world);
    assert_eq!(cars.iter().map(|car| car.vid).collect::<Vec<_>>(), [kept]);
    tracker.keep([None; 2]);
    assert!(tracker.update(4.05, &[], &world).is_empty());
}

#[test]
fn ambiguous_link_does_not_carry_kind() {
    let world = World::default();
    let mut tracker = tracker();
    let cars = tracker.update(1., &[point(10, 20., -0.4), point(11, 20., 0.6)], &world);
    let old = [cars[0].vid, cars[1].vid];
    for vid in old {
        tracker.vehicle_mut(vid).unwrap().kind = Some(Kind {
            class: 2,
            conf: 0.9,
            confirmed_at: 1.,
            frames: 2,
            inherited: false,
        });
    }
    tracker.update(1.3, &[point(20, 20., 0.)], &world);
    let vid = tracker.vid_at(1.3, 20).unwrap();
    let car = tracker.vehicle_mut(vid).unwrap();
    assert_eq!(car.link, Link::Ambiguous);
    assert!(car.kind.is_none());
}

#[test]
fn vehicles_capped_at_128() {
    let world = World::default();
    let mut tracker = tracker();
    let storage = tracker.vehicles().as_ptr();
    let points: Vec<_> = (1..=200).map(|id| point(id, 20., 0.)).collect();
    assert_eq!(tracker.update(1., &points, &world).len(), MAX_VEHICLES);
    assert_eq!(tracker.update(1.05, &points, &world).len(), MAX_VEHICLES);
    assert_eq!(tracker.vehicles().as_ptr(), storage);
}

#[test]
fn vehicle_flags_follow_the_current_map_report() {
    use ccnc_map::radar::{MapObjectData, MapState};

    let world = World::default();
    let mut tracker = tracker();
    let mut map = MapState::default();
    let mut object = MapObjectData::measured(1, 10, 20., 0., 0., 0., 1.);
    object.report = Some((1., 20., 0.));
    object.admitted = true;
    object.motion_confirmed = true;
    object.radar_moving = true;
    map.objects.insert(object.key, object);

    let cars = map.update_vehicles(&mut tracker, 1., &[point(10, 20., 0.)], &world, 2.4);
    assert!(cars[0].flags.motion_confirmed && cars[0].flags.proven && cars[0].flags.admitted);
    let vid = cars[0].vid;

    let cars = map.update_vehicles(&mut tracker, 1.05, &[point(10, 20., 0.)], &world, 2.4);
    assert_eq!(cars[0].vid, vid);
    assert!(!cars[0].flags.admitted && !cars[0].flags.proven);
}
