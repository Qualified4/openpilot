use ccnc_map::input::model_v2::{Line, Model};
use ccnc_map::world::{
    LaneChange, LaneMap, Pose, World, RADAR_LAG, STATUS_FROZEN, STATUS_NO_LIVE_POSE,
};

const GRID: [f32; 10] = [0., 5., 10., 15., 20., 30., 40., 50., 60., 80.];
const BOUNDARIES: [f64; 6] = [8.75, 5.25, 1.75, -1.75, -5.25, -8.75];

fn model(y: f64, heading: f64, far_lag: f64) -> Model {
    let right = BOUNDARIES.iter().position(|&b| b < y).unwrap();
    let lane_lines = BOUNDARIES[right - 2..right + 2]
        .iter()
        .map(|&boundary| Line {
            x: GRID.to_vec(),
            y: GRID
                .iter()
                .map(|&x| {
                    let x = f64::from(x);
                    -(boundary - y - x * heading.tan()
                        + far_lag * heading.tan() * (x - 20.).max(0.)) as f32
                })
                .collect(),
        })
        .collect();
    Model {
        lane_lines,
        lane_line_probs: vec![0.9; 4],
        road_edges: Vec::new(),
    }
}

fn ego_path(t: f64) -> (f64, f64, f64) {
    let phase = (t - 3.) / 4.;
    let angle = std::f64::consts::PI * phase.clamp(0., 1.);
    let y = -1.75 * (1. - angle.cos());
    let speed = -1.75 * std::f64::consts::PI / 4.;
    let vy = speed * angle.sin();
    let ay = if (0. ..1.).contains(&phase) {
        speed * std::f64::consts::PI / 4. * angle.cos()
    } else {
        0.
    };
    (y, (vy / 25.).atan(), ay / 25. / (1. + (vy / 25.).powi(2)))
}

fn car_r(w: &World, t: f64, x: f64, y: f64) -> f64 {
    let (ego_y, heading, _) = ego_path(t - RADAR_LAG);
    let dx = x + 25. * RADAR_LAG;
    let dy = y - ego_y;
    let (sin, cos) = heading.sin_cos();
    let (wx, wy) = w.radar_point_world(t, cos * dx + sin * dy, -sin * dx + cos * dy);
    w.lanes.coord(w.pose.current().s + x, (wx, wy)).unwrap() - w.u_ego()
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
