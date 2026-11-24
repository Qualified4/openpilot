use ccnc_map::world::{
    LaneChange, LaneMap, Pose, World, RADAR_LAG, STATUS_FROZEN, STATUS_NO_LIVE_POSE,
};

mod support;
use support::{ego_path, model};

fn car_r(w: &World, t: f64, x: f64, y: f64) -> f64 {
    let (ego_y, heading, _) = ego_path(t - RADAR_LAG);
    let dx = x + 25. * RADAR_LAG;
    let dy = y - ego_y;
    let (sin, cos) = heading.sin_cos();
    let (wx, wy) = w.radar_point_world(t, cos * dx + sin * dy, -sin * dx + cos * dy);
    w.lanes.coord(w.pose.current().s + x, (wx, wy)).unwrap() - w.u_ego()
}

/// What the map carries to 50 m of a 0.7 m shift beyond 25 m (`vehicle_lane_uses_fresh_camera_beyond_mapped_boundaries`):
/// the 21 m bin moves 0.0049 m (0.07 m at 21 m, blended at 0.07) and the far fit extends it. Measured, in lanes.
const MAP_TILT: f64 = 0.006537;

#[test]
fn vehicle_lane_uses_fresh_camera_beyond_mapped_boundaries() {
    let mut world = World::default();
    world.model(0., &model(0., 0., 0.));

    let mut distant_shift = model(0., 0., 0.);
    for line in &mut distant_shift.lane_lines {
        for (&x, y) in line.x.iter().zip(&mut line.y) {
            if x > 25. {
                *y -= 0.7;
            }
        }
    }
    world.model(0.05, &distant_shift);
    let far = (50., 0.);
    let near = (15., 0.);
    let mapped_far = world.lanes.coord(50., far);
    let updated_far = world.vehicle_lane_coord(0.05, 50., far);
    // The far point takes the camera's coordinate, 1.5 + 0.7 m / 3.5 m. The last mapped bin is sampled at its centre,
    // 21 m, inside the model segment 20-30 m that the shift tilts, and the map extends that little tilt to 50 m.
    assert!((updated_far.unwrap() - 1.7).abs() < 1e-4, "{updated_far:?}");
    assert!((mapped_far.unwrap() - 1.5 - MAP_TILT).abs() < 1e-4, "{mapped_far:?}");
    assert_eq!(
        world.vehicle_lane_coord(0.05, 15., near),
        world.lanes.coord(15., near)
    );
    assert_eq!(world.vehicle_lane_coord(0.31, 50., far), mapped_far);
}

fn update(world: &mut World, step: u32, lost_after: f64) -> Option<(f64, [f64; 2])> {
    let t = f64::from(step) * 0.01;
    let (y, heading, yaw) = ego_path(t);
    if t < lost_after && step % 5 == 0 {
        world.live_pose(t, yaw);
    }
    world.car_state(t, 25., false);
    if step % 5 != 0 {
        return None;
    }
    world.model(t, &model(y, heading, 1.0));
    world.radar(t);
    (t > 1.).then(|| (t, [car_r(world, t, 30., -3.5), car_r(world, t, 45., 0.)]))
}

#[test]
fn ego_lane_change_moves_cars_continuously() {
    let mut world = World::new(RADAR_LAG, true);
    let mut best = [f64::INFINITY; 2];
    let mut previous: Option<[f64; 2]> = None;
    let mut middle = [None; 2];
    for step in 0..1200 {
        let Some((t, r)) = update(&mut world, step, f64::INFINITY) else {
            continue;
        };
        if previous.is_none() {
            assert!((r[0] - 1.).abs() < 0.08 && r[1].abs() < 0.08, "{r:?}");
        }
        for i in 0..2 {
            if let Some(prev) = previous {
                assert!(
                    (r[i] - prev[i]).abs() * 3.5 <= 0.5,
                    "t={t}, {prev:?} -> {r:?}"
                );
            }
            assert!((r[i] - best[i]) * 3.5 < 0.18, "t={t}, {best:?} -> {r:?}");
            best[i] = best[i].min(r[i]);
            if r[i] < 0.5 - i as f64 {
                middle[i].get_or_insert(t);
            }
        }
        previous = Some(r);
    }
    let r = previous.unwrap();
    assert!(r[0].abs() < 0.08 && (r[1] + 1.).abs() < 0.08, "{r:?}");
    for t in middle {
        assert!((t.unwrap() - 5.).abs() <= 0.3, "{t:?}");
    }
    assert!(world.lanes.renumbers() > 0);
}

#[test]
fn renumbered_model_line_keeps_world_boundary() {
    let mut lanes = LaneMap::default();
    let pose = Pose::default();
    let first = model(0., 0., 0.);
    lanes.ingest(0., &first, &pose, false);
    let mut shifted = model(-3.5, 0., 0.);
    for line in &mut shifted.lane_lines {
        for y in &mut line.y {
            *y += 3.5;
        }
    }
    let u = lanes.coord(10., (10., -1.75)).unwrap();
    lanes.ingest(0.05, &shifted, &pose, false);
    assert_eq!(lanes.renumbers(), 1);
    assert!((lanes.coord(10., (10., -1.75)).unwrap() - u).abs() < 1e-9);
    assert!((lanes.width_at(10.) - 3.5).abs() < 1e-9);
}

#[test]
fn far_points_wait_one_second() {
    let mut world = World::default();
    let m = model(0., 0., 0.);
    world.lanes.ingest(0., &m, &world.pose, false);
    assert_eq!(world.lanes.observations(1, 30.), 0);
    world.lanes.ingest(0.99, &m, &world.pose, false);
    assert_eq!(world.lanes.observations(1, 30.), 0);
    world.lanes.ingest(1., &m, &world.pose, false);
    assert_eq!(world.lanes.observations(1, 30.), 1);
}

#[test]
fn far_points_dropped_when_freeze_starts() {
    let mut lanes = LaneMap::default();
    let pose = Pose::default();
    let m = model(0., 0., 0.);
    lanes.ingest(0., &m, &pose, false);
    lanes.ingest(0.6, &m, &pose, true);
    lanes.ingest(1.01, &m, &pose, false);
    assert_eq!(lanes.observations(1, 30.), 0);
}

#[test]
fn freeze_starts_on_motion_not_blinker() {
    let mut world = World::new(RADAR_LAG, true);
    let m = model(0., 0., 0.);
    world.blinker(true);
    for step in 0..20 {
        let t = f64::from(step) * 0.05;
        world.live_pose(t, 0.);
        world.car_state(t, 20., false);
        world.model(t, &m);
        world.radar(t);
        assert_eq!(world.status() & STATUS_FROZEN, 0);
    }
    let mut change = LaneChange::default();
    assert!(!change.step(1., 1.5));
    assert!(change.step(1.2, 1.81));
}

#[test]
fn freeze_ends_within_8s() {
    let mut change = LaneChange::default();
    assert!(change.step(2., 1.85));
    assert!(change.step(9.99, 1.85));
    assert!(!change.step(10., 1.85));
}

#[test]
fn radar_point_uses_pose_at_lag() {
    let mut world = World::default();
    for i in 0..31 {
        let t = f64::from(i) * 0.01;
        world.live_pose(t, 0.2);
        world.car_state(t, 10., false);
    }
    let stamp = 0.297;
    let (x, y) = world.radar_point_world(stamp, 30., 2.);
    let (px, py) = world.pose.at(stamp - RADAR_LAG).to_world(30., 2.);
    assert!((x - px).abs() < 1e-9 && (y - py).abs() < 1e-9);
    assert!((world.pose.at(0.097).s - 0.97).abs() < 1e-9);
}

#[test]
fn no_live_pose_disables_freeze() {
    let mut world = World::new(RADAR_LAG, true);
    for step in 0..720 {
        if let Some((t, _)) = update(&mut world, step, 1.) {
            if t >= 1.5 {
                assert_eq!(
                    world.status() & (STATUS_FROZEN | STATUS_NO_LIVE_POSE),
                    STATUS_NO_LIVE_POSE
                );
            }
        }
    }
}

#[test]
fn live_pose_lost_during_freeze_ends_freeze_without_jump() {
    let mut world = World::new(RADAR_LAG, true);
    let mut previous: Option<[f64; 2]> = None;
    let mut froze = false;
    for step in 0..900 {
        let Some((t, r)) = update(&mut world, step, 4.7) else {
            continue;
        };
        if t < 5.2 && world.status() & STATUS_FROZEN != 0 {
            froze = true;
        }
        if t >= 5.2 {
            assert_eq!(
                world.status() & (STATUS_FROZEN | STATUS_NO_LIVE_POSE),
                STATUS_NO_LIVE_POSE
            );
        }
        if let Some(p) = previous {
            for i in 0..2 {
                assert!(
                    (r[i] - p[i]).abs() * 3.5 <= 0.5,
                    "t={t}, {p:?} -> {r:?}, u={}, pose={:?}",
                    world.u_ego(),
                    world.pose.current()
                );
            }
        }
        previous = Some(r);
    }
    assert!(froze);
}

#[test]
fn reverse_or_fast_turn_resets_map() {
    for (t, yaw, reverse) in [(0.01, 0., true), (0.01, 1.01, false), (0.251, 0., false)] {
        let mut world = World::default();
        world.live_pose(0., 0.);
        world.car_state(0., 10., false);
        world.model(0., &model(0., 0., 0.));
        assert!(world.u_ego().is_finite());
        world.live_pose(t, yaw);
        world.car_state(t, 10., reverse);
        assert_eq!(world.generation(), 1);
        assert_eq!(world.lanes.observations(1, 0.), 0);
        assert!(world.u_ego().is_nan());
    }
}

#[test]
fn radar_before_pose_history_uses_oldest_pose() {
    let mut world = World::default();
    world.car_state(5., 12., false);
    world.car_state(5.1, 12., false);
    world.car_state(5.05, 30., false);
    assert_eq!(world.radar_point_world(5.1, 30., 2.), (30., 2.));
    assert_eq!(world.radar_point_world(4.9, 30., 2.), (30., 2.));
    assert_eq!(world.pose.current().t, 5.1);
    assert_eq!(world.generation(), 0);
}

/// From frame 20 the camera frames stop for `blind` frames while ego runs at 250 m/s, so it leaves the mapped 80 m
/// after about 0.3 s and the lane map cannot place it until the camera is back. (t, u_ego, lane_held) per 50 ms frame.
fn lane_map_miss(blind: u32) -> Vec<(f64, f64, bool)> {
    let mut world = World::default();
    (0..60u32)
        .map(|frame| {
            let t = f64::from(frame) * 0.05;
            let dark = (20..20 + blind).contains(&frame);
            world.live_pose(t, 0.);
            world.car_state(t, if dark { 250. } else { 10. }, false);
            if !dark {
                world.model(t, &model(0., 0., 0.));
            }
            world.radar(t);
            (t, world.u_ego(), world.lane_held())
        })
        .collect()
}

#[test]
fn ego_keeps_its_lane_coordinate_over_a_short_lane_map_miss() {
    let short = lane_map_miss(9);
    let misses: Vec<_> = short.iter().filter(|r| r.2).collect();
    assert!(!misses.is_empty() && misses.len() < 10, "{short:?}");
    assert!(short.iter().all(|&(_, u, _)| u.is_finite()), "never unknown: {short:?}");
    assert!(misses.iter().all(|&&(_, u, _)| u == short[19].1), "the last mapped coordinate");
    assert!(!short[29].2 && !short.last().unwrap().2, "mapped again once the camera is back");
}

#[test]
fn a_lane_map_miss_longer_than_half_a_second_leaves_ego_unknown() {
    let long = lane_map_miss(30);
    let unknown = long.iter().position(|r| !r.1.is_finite()).unwrap();
    let mapped = long[..unknown].iter().rposition(|r| !r.2).unwrap();
    assert!((long[unknown].0 - long[mapped].0 - 0.55).abs() < 1e-6, "{long:?}");
    assert!(long[unknown..50].iter().all(|r| !r.1.is_finite()), "{long:?}");
    assert!(long[51..].iter().all(|r| r.1.is_finite()), "{long:?}");
}

/// Straight road, lanes at ±1.75 and ±5.25 m, a left road edge at `edge` m (left positive). modelV2 is right-positive.
fn road_model(edge: f64) -> ccnc_map::input::model_v2::Model {
    let mut m = model(0., 0., 0.);
    let grid: Vec<f32> = m.lane_lines[0].x.clone();
    m.road_edges = vec![
        ccnc_map::input::model_v2::Line { x: grid.clone(), y: vec![-edge as f32; grid.len()] },
        ccnc_map::input::model_v2::Line { x: grid.clone(), y: vec![9.; grid.len()] },
    ];
    m
}

/// Ego slows from 2 m/s to a stop and waits while modelV2 moves the left road edge 1 m out (5.5 -> 6.5 m, the model
/// wobbling as it did in 11-24 0000047c segment 5). A car waits 8.2 m ahead at 2.65 m left, 3 m inside that edge:
/// its lane coordinate per camera frame of the wait.
fn car_inside_the_edge_while_ego_waits() -> Vec<Option<f64>> {
    let mut world = World::default();
    let mut out = Vec::new();
    let mut stopped = None;
    for step in 0..2000u32 {
        let t = f64::from(step) * 0.01;
        let v = (2. * (1. - (t - 8.) / 2.)).clamp(0., 2.);
        world.live_pose(t, 0.);
        world.car_state(t, v, false);
        if step % 5 != 0 {
            continue;
        }
        let wait = stopped.map_or(0., |since| t - since);
        if v == 0. && stopped.is_none() {
            stopped = Some(t);
        }
        world.model(t, &road_model(5.5 + (wait / 3.).min(1.)));
        world.radar(t);
        if stopped.is_some() {
            let ego = world.pose.current();
            out.push(world.lanes.coord(ego.s + 8.2, ego.to_world(8.2, 2.65)));
        }
    }
    out
}

#[test]
fn a_car_inside_the_road_edge_keeps_its_lane_coordinate_while_ego_waits() {
    let coords = car_inside_the_edge_while_ego_waits();
    assert!(coords.len() > 100);
    assert!(coords.iter().all(|u| u.is_some_and(|u| (0.6..0.9).contains(&u))), "{coords:?}");
}

#[test]
fn every_near_bin_takes_the_lines_while_ego_waits() {
    let mut world = World::default();
    let mut seen = Vec::new();
    for step in 0..3600u32 {
        let t = f64::from(step) * 0.01;
        world.live_pose(t, 0.);
        world.car_state(t, if t < 3. { 5. } else { 0. }, false);
        if step % 5 == 0 {
            world.model(t, &model(0., 0., 0.));
            world.radar(t);
        }
        if step == 400 || step == 3599 {
            let s = world.pose.current().s;
            // Every bin 0-20 m ahead, at the grid points (0/5/10/15/20 m) or between them.
            seen.push((0..10).map(|k| world.lanes.observations(1, s + 1. + 2. * f64::from(k))).collect::<Vec<_>>());
        }
    }
    // 32 s at a stop, 20 camera frames a second.
    assert!(seen[1].iter().zip(&seen[0]).all(|(after, before)| after - before >= 600), "{seen:?}");
}

#[test]
fn a_yaw_rate_bias_does_not_turn_the_map_while_ego_waits() {
    let mut world = World::default();
    let mut heading = Vec::new();
    for step in 0..4000u32 {
        let t = f64::from(step) * 0.01;
        // livePose keeps 0.0014 rad/s through the stop, as in 11-24 0000047c segment 5.
        world.live_pose(t, 0.0014);
        world.car_state(t, if t < 5. { 5. } else { 0. }, false);
        if step % 5 == 0 {
            world.model(t, &model(0., 0., 0.));
            world.radar(t);
        }
        if step == 600 || step == 3999 {
            heading.push(world.pose.current().heading);
        }
    }
    assert!((heading[1] - heading[0]).abs() < 1e-9, "{heading:?}");
    // On the move the same yaw rate turns ego: 5 s at 0.0014 rad/s.
    assert!((heading[0] - 0.007).abs() < 1e-3, "{heading:?}");
}
