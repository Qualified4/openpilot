use super::*;

const STEP: f64 = 0.05;

fn radar(id: u64, x: f32, y: f32, speed: f32) -> RawPoint {
    RawPoint {
        track_id: id,
        d_rel: x,
        y_rel: -y,
        v_rel: speed,
        a_rel: 0.,
        yv_rel: 0.,
        measured: true,
        v_lead: 0.,
        a_lead: 0.,
        j_lead: 0.,
        source: 0,
        track_state: 2,
        motion_state: 2,
    }
}

fn input(tick: u32, radar: &[RawPoint]) -> EngineInput<'_> {
    EngineInput {
        time: f64::from(tick) * STEP,
        speed: 0.,
        yaw: None,
        road: None,
        radar,
        lead: None,
        vision: &[],
        blind: [false; 2],
    }
}

fn confirm<'a>(engine: &'a mut MapEngine, points: &[RawPoint]) -> EngineOutput<'a> {
    for tick in 0..7 {
        engine.step(input(tick, points));
    }
    engine.step(input(7, points))
}

#[test]
fn curved_position_keeps_car_ahead_with_low_confidence_straight_boundaries() {
    use crate::input::model_v2::Line;

    let mut engine = MapEngine::default();
    let sample = [
        (0., 0.),
        (5., 0.5),
        (10., 2.),
        (15., 4.5),
        (20., 8.),
        (30., 18.),
    ];
    let model = Model {
        lane_lines: [-5.25, -1.75, 1.75, 5.25]
            .map(|y| Line {
                x: vec![0., 30.],
                y: vec![y, y],
            })
            .into(),
        lane_line_probs: vec![0.1; 4],
        road_edges: Vec::new(),
    };
    for tick in 0..8 {
        let point = [radar(40, 15., 4.5, 0.)];
        let mut entry = input(tick, &point);
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &sample,
            lines: Some(&model),
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: false,
        });
        let out = engine.step(entry);
        if tick >= 3 {
            let ff = out.slots[0].unwrap();
            assert!((15.7..16.).contains(&ff.s) && ff.d.abs() < 0.05);
            assert!(out.slots[1..].iter().all(Option::is_none));
        }
    }
}

#[test]
fn trusted_lane_pair_corrects_width_and_small_centre_offset() {
    use crate::input::model_v2::Line;
    let model = Model {
        lane_lines: [-6., -1.7, 2.5, 6.]
            .map(|y| Line {
                x: vec![0., 30.],
                y: vec![y, y],
            })
            .into(),
        lane_line_probs: vec![0.99; 4],
        road_edges: Vec::new(),
    };
    let frame = RoadFrame::new(
        &[(0., 0.), (30., 0.)],
        Some(&model),
        PoseAt::default(),
        PoseAt::default(),
        std::iter::empty(),
        false,
    );
    assert!((frame.lane_u(12., 4.6) - 1.).abs() < 1e-6);
}

#[test]
fn stopping_path_extrapolates_from_its_last_forward_segment() {
    let frame = RoadFrame::new(
        &[(0., 0.), (6., 0.), (12., 0.), (11.5, 0.), (11., 0.)],
        None,
        PoseAt::default(),
        PoseAt::default(),
        std::iter::empty(),
        false,
    );
    assert_eq!(frame.project(35., 1.), (35., 1.));
    assert_eq!(frame.cartesian(35., 1.), (35., 1.));
}

#[test]
fn refreshed_road_preserves_ego_compensated_cartesian_position() {
    let past = PoseAt::default();
    let ego = PoseAt {
        x: 0.4,
        y: 0.08,
        heading: 0.015,
        ..past
    };
    let old = RoadFrame::new(&[], None, past, past, std::iter::empty(), false);
    let new = RoadFrame::new(
        &[(0., 0.), (5., 0.), (10., 4.)],
        None,
        ego,
        ego,
        std::iter::empty(),
        false,
    );
    let mut tracker = Tracker::default();
    tracker.radar(0., 0., &old, past, &[radar(40, 23., -2., 0.)]);
    tracker.predict(0.05, 0., &old, &new, past, ego);
    let object = &tracker.objects[0];
    let (x, y) = object.cartesian(&new);
    let (ex, ey) = ego.to_ego(23., 2.);
    assert!((x - ex).hypot(y + ey) < 1e-9);
    assert!((object.d + 2.).abs() <= 0.40);
    assert!(new.beyond_forecast(object.s));
    assert!(new.cartesian(object.s + 30., 0.).1 < 8.);
}

#[test]
fn corner_rejects_same_lane_far_car_and_preserves_a_real_side_car() {
    let path = [(0., 0.), (15., 1.131), (35., 1.131)];
    let curve_y = (1. - (1. - (0.01_f64 * 28.).powi(2)).sqrt()) / 0.01;
    let model = Model {
        lane_lines: [-5., -2., 2.8]
            .map(|y| crate::input::model_v2::Line { x: vec![0., 50.], y: vec![y; 2] })
            .into(),
        lane_line_probs: vec![0., 0., 0.62],
        road_edges: Vec::new(),
    };
    for (yaw, offset, lines, expected_side) in [
        (-0.12, 0., None, None),
        (-0.12, 2.3, None, Some(41)),
        (0., 0., None, Some(41)),
        (-0.12, 0., Some(&model), Some(41)),
    ] {
        let mut engine = MapEngine::default();
        let points = [radar(40, 15., 1.131, 0.), radar(41, 28., (curve_y + offset) as f32, 0.)];
        for tick in 0..24 {
            let mut entry = input(tick, &points);
            entry.yaw = Some(yaw);
            entry.speed = 12.;
            entry.road = Some(RoadObservation {
                time: entry.time,
                path: &path,
                lines,
                road_edge_stds: [f32::INFINITY; 2],
                changing_lanes: false,
            });
            let out = engine.step(entry);
            if tick >= 8 {
                let far = out.objects.iter().find(|o| o.evidence.last_track_id == Some(41)).unwrap();
                assert!(far.lane_u > 0.59);
                assert_eq!(out.slots[0].unwrap().last_track_id, Some(40));
                assert_eq!(out.slots[2].and_then(|slot| slot.last_track_id), expected_side);
            }
        }
    }
}

#[test]
fn left_curve_draws_the_lead_relative_to_the_road() {
    let mut engine = MapEngine::default();
    let path = [(0., 0.), (10., -0.4), (20., -1.6), (30., -3.5), (40., -6.2)];
    let points = [radar(42, 26.55, -3.75, 0.)];
    let mut model = Model {
        lane_lines: [-5.05, -1.55, 1.95, 5.45]
            .map(|offset| crate::input::model_v2::Line {
                x: path.iter().map(|(x, _)| *x as f32).collect(),
                y: path.iter().map(|(_, y)| (y + offset) as f32).collect(),
            })
            .into(),
        lane_line_probs: vec![0.; 4],
        road_edges: Vec::new(),
    };
    for tick in 0..8 {
        model.lane_line_probs.fill(if tick % 2 == 0 { 0.96 } else { 0.1 });
        let mut entry = input(tick, &points);
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &path,
            lines: Some(&model),
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: false,
        });
        let out = engine.step(entry);
        if tick >= 3 {
            let ff = out.slots[0].unwrap();
            assert_eq!(ff.last_track_id, Some(42));
            assert!((ff.y + 3.75).abs() < 1e-6);
            assert!((ff.display_d + 1.).abs() < 0.15);
            assert!(out.slots[1].is_none());
        }
    }
}

#[test]
fn brief_forecast_jump_keeps_the_measured_side_vehicle() {
    let mut engine = MapEngine::default();
    let points = [radar(49, 30., -3.5, 0.)];
    let good = [(0., 0.), (50., 0.)];
    let jump = [(0., 0.), (12., 3.), (24., 8.), (50., 18.)];
    let mut incumbent = None;
    for tick in 0..11 {
        let mut entry = input(tick, &points);
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: if (8..10).contains(&tick) { &jump } else { &good },
            lines: None,
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: false,
        });
        let out = engine.step(entry);
        if tick >= 7 {
            let lf = out.slots[1].unwrap();
            assert_eq!(lf.last_track_id, Some(49));
            assert_eq!(lf.obj_id, *incumbent.get_or_insert(lf.obj_id));
            assert!((lf.display_d + 3.5).abs() < 0.05);
        }
    }
}

#[test]
fn one_tick_range_spike_and_radar_id_change_keep_the_same_object() {
    let mut engine = MapEngine::default();
    let start = confirm(&mut engine, &[radar(40, 11.4, 0., 0.)]).slots[0].unwrap();
    let spike = engine.step(input(8, &[radar(40, 13.25, 0., 0.)])).slots[0].unwrap();
    let after = engine.step(input(9, &[radar(52, 10.8, 0., 0.)])).slots[0].unwrap();
    assert_eq!([spike.obj_id, after.obj_id], [start.obj_id; 2]);
    assert_eq!(after.last_track_id, Some(52));
    assert!((spike.s - start.s).abs().max((after.s - spike.s).abs()) <= 0.6);
}

#[test]
fn two_observed_vehicles_get_distinct_slots_over_an_unverified_coast() {
    for (side, sign) in [(0, -1.), (1, 1.)] {
        let mut engine = MapEngine::default();
        let mut out = EngineOutput::default();
        for tick in 0..13 {
            let x = 2.4 - 0.35 * tick as f32;
            let points = if tick < 6 {
                vec![radar(50, x, 3.5 * sign, -7.)]
            } else {
                vec![
                    radar(40, 5.5, 3.5 * sign, 0.),
                    radar(41, 20., 3.5 * sign, 0.),
                ]
            };
            let mut entry = input(tick, &points);
            entry.speed = 7.;
            out = engine.step(entry);
        }
        assert_eq!(out.slots[side + 1].unwrap().last_track_id, Some(41));
        assert_eq!(out.slots[side + 3].unwrap().last_track_id, Some(40));
        assert!((out.slots[side + 3].unwrap().s - 5.5).abs() < 0.05);
    }
}

#[test]
fn moving_side_vehicles_need_doppler_consistent_travel() {
    let mut engine = MapEngine::default();
    for tick in 0..35 {
        // The right echo stays at the same distance despite high closing/opening evidence.
        let points = [radar(41, 21., 3.5, 13.), radar(42, 24., -3.5, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 13.;
        let out = engine.step(entry);
        assert!(out.slots[2].is_none());
        if tick >= 6 {
            assert_eq!(out.slots[1].unwrap().last_track_id, Some(42));
        }
    }
}

#[test]
fn established_vehicle_can_stop_but_a_stationary_oblique_echo_cannot_promote() {
    let mut engine = MapEngine::default();
    let mut x = 24.;
    for tick in 0_u32..37 {
        let relative = -0.6 * tick.saturating_sub(13).min(17) as f32;
        x += relative * STEP as f32;
        let fixed_x = 24. - tick as f32 * 10. * STEP as f32;
        let radial = |x: f32, v: f32| v * x / x.hypot(3.5);
        let points = [
            radar(41, fixed_x, 3.5, radial(fixed_x, -10.)),
            radar(42, x, -3.5, radial(x, relative)),
        ];
        let mut entry = input(tick, &points);
        entry.speed = 10.;
        let out = engine.step(entry);
        assert!(out.slots[2].is_none());
        if tick >= 32 {
            assert_eq!(out.slots[1].unwrap().last_track_id, Some(42));
        }
    }
}

#[test]
fn observed_pass_holds_rear_when_a_front_echo_is_at_the_lane_edge() {
    let mut engine = MapEngine::default();
    for tick in 0..82 {
        let x = 13. - 6. * STEP as f32 * tick as f32;
        let mut points = vec![];
        if x > 1. {
            points.push(radar(40, x, 3.5, -6.));
        }
        if tick >= 49 {
            points.extend([radar(41, 5.8, 3.5, 0.), radar(42, 25., 5.1, 0.)]);
        }
        let mut entry = input(tick, &points);
        entry.speed = 10.;
        let out = engine.step(entry);
        if (56..=71).contains(&tick) {
            let rear = out.slots[4].unwrap();
            assert!((-10. ..0.).contains(&rear.s));
            assert_eq!(rear.last_track_id, Some(40));
        } else if tick == 77 {
            assert_eq!(out.slots[4].unwrap().last_track_id, Some(41));
        }
    }
}

#[test]
fn close_lead_stays_in_ff_through_path_transition_then_moves_to_side() {
    let mut engine = MapEngine::default();
    for tick in 0_u32..26 {
        let y = tick.saturating_sub(9).min(6) as f32 * 0.45;
        let points = [radar(40, 10., y, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 10.;
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &[(0., 0.), (40., 0.)],
            lines: None,
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: (10..16).contains(&tick),
        });
        let out = engine.step(entry);
        if (3..21).contains(&tick) {
            assert_eq!(out.slots[0].unwrap().last_track_id, Some(40));
        }
        if tick >= 23 {
            assert!(out.slots[0].is_none());
            assert_eq!(out.slots[2].unwrap().last_track_id, Some(40));
        }
    }
}

#[test]
fn front_to_rear_passes_through_zero_with_one_id_and_no_teleport() {
    let mut engine = MapEngine::default();
    let mut previous: Option<EngineSlot> = None;
    let (mut front, mut rear) = (false, false);
    for tick in 0..20 {
        let x = 2.8 - tick as f32 * 0.25;
        let points: Vec<_> = (x > 0.1)
            .then(|| radar(40, x, 3.5, -5.))
            .into_iter()
            .collect();
        let mut frame = input(tick, &points);
        frame.speed = 5.;
        frame.blind = [false, true];
        let out = engine.step(frame);
        let real_rear = out.slots[4].filter(|s| s.obj_id > 0);
        front |= out.slots[2].is_some();
        rear |= real_rear.is_some();
        if let Some(s) = out.slots[2].or(real_rear) {
            if let Some(before) = previous {
                assert_eq!(s.obj_id, before.obj_id);
                assert!((s.s - before.s).abs() < 0.6);
            }
            previous = Some(s);
        }
    }
    assert!(front && rear);
}

#[test]
fn fast_pass_crosses_the_origin_before_the_next_car_appears() {
    for (rear, y) in [(3, -3.5), (4, 3.5)] {
        let mut engine = MapEngine::default();
        let front = rear - 2;
        let mut last_front = None;
        let mut crossing = None;
        let mut last_rear = 0.;
        for tick in 0..130 {
            let x = 12. - tick as f32 * 0.175;
            let mut points = vec![radar(41, 20., y, 0.)];
            if x >= 1.2 {
                points.push(radar(40, x, y, -3.5));
            }
            let mut frame = input(tick, &points);
            frame.speed = 12.;
            let out = engine.step(frame);
            if let Some(first) = crossing {
                assert_eq!(out.slots[front].unwrap().last_track_id, Some(41));
                if let Some(slot) = out.slots[rear] {
                    assert_eq!(slot.last_track_id, Some(40));
                    assert!((slot.x - last_rear).abs() <= 0.25);
                    assert!(slot.x <= last_rear);
                    last_rear = slot.x;
                } else {
                    assert!(tick > first + 50);
                }
            } else if let Some(slot) = out.slots[rear] {
                let previous: EngineSlot = last_front.unwrap();
                assert!(previous.x <= 0.25);
                assert!(slot.s <= 0.);
                assert_eq!(slot.x, 0.);
                assert_eq!(slot.obj_id, previous.obj_id);
                assert!(out.slots[front].is_none());
                crossing = Some(tick);
            } else if x <= 6. {
                let slot = out.slots[front].unwrap();
                assert_eq!(slot.last_track_id, Some(40));
                last_front = Some(slot);
            }
        }
        assert!(crossing.is_some() && last_rear == -10.);
    }
}

#[test]
fn adjacent_pass_keeps_its_settled_lateral_anchor_through_rear_coasting() {
    for (rear, center) in [(3, -3.5_f32), (4, 3.5)] {
        let front = rear - 2;
        let mut engine = MapEngine::default();
        let mut seen_front = None;
        let mut rear_ticks = 0;
        let mut last_x = 0.;
        for tick in 0..125 {
            let x = 12. - tick as f32 * 0.175;
            let y = if x >= 4.7 { center } else { 2.1 * center.signum() };
            let point = [radar(40, x, y, -3.5)];
            let mut frame = input(tick, if x >= 1.2 { &point } else { &[] });
            frame.speed = 12.;
            let out = engine.step(frame);
            if let Some(slot) = out.slots[front] {
                assert_eq!(slot.display_d, f64::from(center));
                seen_front = Some(slot.obj_id);
            }
            if let Some(slot) = out.slots[rear] {
                assert_eq!(Some(slot.obj_id), seen_front);
                assert_eq!(slot.display_d, f64::from(center));
                assert!((slot.y - f64::from(center)).abs() > 0.5);
                assert!(!slot.radar_seen && slot.x <= last_x);
                last_x = slot.x;
                rear_ticks += 1;
            }
        }
        assert!(rear_ticks > 45 && last_x < -8.);
    }
}

#[test]
fn alongside_pair_splits_at_six_metres() {
    let mut engine = MapEngine::default();
    for tick in 0..40 {
        let x = 6.5 - tick as f32 * 0.025;
        let points = [radar(40, x, -3.5, -0.5), radar(41, 20., -3.5, 0.)];
        let mut frame = input(tick, &points);
        frame.speed = 12.;
        let out = engine.step(frame);
        if x <= 6. {
            let rear = out.slots[3].unwrap();
            assert_eq!(rear.last_track_id, Some(40));
            assert_eq!((-rear.x).max(0.), 0.);
            assert_eq!(out.slots[1].unwrap().last_track_id, Some(41));
        }
    }
}

#[test]
fn rear_pass_converges_to_bsd_and_drifts_away_when_it_clears() {
    let mut engine = MapEngine::default();
    let mut last_x = 0.;
    let mut vid = None;
    for tick in 0..170 {
        let x = 12. - tick as f32 * 0.175;
        let points = [radar(40, x, 3.5, -3.5)];
        let mut frame = input(tick, if x >= 1.2 { &points } else { &[] });
        frame.speed = 12.;
        frame.blind[1] = (82..155).contains(&tick);
        let out = engine.step(frame);
        if (70..=169).contains(&tick) {
            let rear = out.slots[4].unwrap();
            assert_eq!(rear.obj_id, *vid.get_or_insert(rear.obj_id));
            assert!((-0.31..=0.).contains(&(rear.x - last_x)));
            if tick == 154 {
                assert_eq!(rear.x, -10.);
            }
            if tick == 169 {
                assert_eq!(rear.x, -12.5);
            }
            last_x = rear.x;
        }
    }
    let mut frame = input(170, &[]);
    frame.speed = 12.;
    assert!(engine.step(frame).slots[4].is_none());
}

#[test]
fn confirmed_rear_coast_survives_ego_speed_crossing_the_low_speed_threshold() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(40, 4., 3.5, 0.)]);
    let mut rear = engine.displayed[2].unwrap();
    rear.s = -5.;
    let previous = [None, None, None, None, Some(rear)];
    let frame = engine.previous.as_ref().unwrap();
    let object = &mut engine.tracker.objects[0];
    object.s = -5.2;
    object.v_s = -4.3;
    object.moved = false; // Confirmed by the low-speed motion evidence.
    object.handoff_until = 1.5;

    let mut locks = [None; 2];
    let slots = selector::select(1.65, 5.2, false, false, frame, &mut engine.tracker.objects,
                                 previous, &mut locks, [false; 2], None, |_, _| false, |_, _| false);
    assert_eq!(slots[4].unwrap().obj_id, rear.obj_id);
    let expired = selector::select(2.01, 5.2, false, false, frame, &mut engine.tracker.objects,
                                   slots, &mut locks, [false; 2], None, |_, _| false, |_, _| false);
    assert!(expired[4].is_none());
}

#[test]
fn bsd_overtake_draws_one_real_vid_from_rear_through_origin_to_front() {
    let mut engine = MapEngine::default();
    let mut samples = Vec::new();
    for tick in 0_u32..26 {
        let x = 3.5 + tick.saturating_sub(4) as f32 * 0.1;
        let points = [radar(40, x, 3.5, 2.)];
        let mut entry = input(tick, if tick >= 4 { &points } else { &[] });
        entry.speed = 8.;
        entry.blind = [false, true];
        let out = engine.step(entry);
        assert!(out.blind[1]);
        let real_rear = out.slots[4].filter(|s| s.obj_id > 0);
        assert!(out.slots[2].is_none() || real_rear.is_none());
        if let Some(slot) = out.slots[2].or(real_rear) {
            let object = out.objects.iter().find(|o| o.obj_id == slot.obj_id).unwrap();
            assert_eq!(slot.last_track_id, Some(40));
            assert!(object.s >= 3.4 && (slot.s - object.s).abs() < 1e-9);
            samples.push((slot.obj_id, slot.x));
        }
    }
    assert_eq!(samples.first().unwrap().1, -10.);
    assert!(samples.iter().any(|s| s.1 == 0.));
    assert!(samples.last().unwrap().1 > 5.);
    assert!(samples.windows(2).all(|pair| pair[0].0 == pair[1].0 && pair[0].1 <= pair[1].1));
}

#[test]
fn prior_front_vehicle_does_not_consume_bsd_warning_on_return() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(40, 4., 3.5, 0.)]);
    let old = engine.displayed[2].unwrap();
    let mut display = rear_display::SideDisplay::default();
    let front = [None, None, Some(old), None, None];
    let mut shown = front;
    display.update(1., 1, true, front, &mut shown);
    let rear = [None, None, None, None, Some(EngineSlot { s: -0.8, x: -0.8, ..old })];
    shown = rear;
    display.update(1.1, 1, true, front, &mut shown);
    display.update(1.2, 1, true, rear, &mut [None; 5]);
    let mut returned = [None, None, Some(EngineSlot { v_s: 2., ..old }), None, None];
    display.update(1.3, 1, true, [None; 5], &mut returned);
    assert_eq!(returned[2].unwrap().x, old.x);
    assert!(returned[4].is_none_or(|s| s.obj_id == 0));
}

#[test]
fn remaining_measured_rear_vid_joins_front_at_exact_origin() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(40, 4., 3.5, 0.)]);
    let old = engine.displayed[2].unwrap();
    let rear = [None, None, None, None, Some(EngineSlot { s: -0.2, x: -0.2, ..old })];
    let selected = [None, None, Some(EngineSlot { s: 0.8, x: 0.8, v_s: 5., ..old }), None, None];
    let mut display = rear_display::SideDisplay::default();
    for (now, previous, expect_front) in [(1., rear, None), (1.05, selected, Some(0.)),
                                           (1.30, selected, Some(0.8))] {
        let mut slots = selected;
        display.update(now, 1, false, previous, &mut slots);
        assert_eq!(slots[2].map(|slot| slot.x), expect_front);
        assert_eq!(slots[2].or(slots[4]).unwrap().obj_id, old.obj_id);
    }
}

#[test]
fn distant_new_vehicle_without_bsd_appears_directly_in_front() {
    let mut engine = MapEngine::default();
    for tick in 0..8 {
        let points = [radar(40, 15. + tick as f32 * 0.1, 3.5, 2.)];
        let mut entry = input(tick, &points);
        entry.speed = 8.;
        let out = engine.step(entry);
        assert!(out.slots[4].is_none());
        if tick >= 6 {
            assert!(out.slots[2].unwrap().x >= 15.);
        }
    }
}

#[test]
fn visible_object_decays_faster_than_blind_or_occluded_object() {
    let mut engine = MapEngine::default();
    let near = radar(40, 4., 3.5, 0.);
    let hidden = radar(42, 17., 0., 0.);
    let visible = radar(43, 20., -3.5, 0.);
    confirm(&mut engine, &[near, radar(41, 9., 0., 0.), hidden, visible]);
    for tick in 8..85 {
        let out = engine.step(input(tick, &[radar(41, 9., 0., 0.)]));
        let has = |id| {
            out.objects
                .iter()
                .any(|o| o.evidence.last_track_id == Some(id))
        };
        if tick == 14 {
            assert!(has(40) && has(42) && !has(43));
        }
        if tick == 84 {
            assert!(!has(40) && !has(42));
        }
    }
}

#[test]
fn bsd_only_supports_an_existing_object_and_switch_off_reduces_probability() {
    let mut engine = MapEngine::default();
    let mut empty = input(0, &[]);
    empty.blind = [true; 2];
    assert!(engine.step(empty).objects.is_empty());
    let mut engine = MapEngine::default();
    let mut bsd = MapEngine::default();
    for tick in 0..25 {
        let x = 3. - 0.25 * tick as f32;
        let points: Vec<_> = (x > 0.1)
            .then(|| radar(40, x, 3.5, -5.))
            .into_iter()
            .collect();
        let mut entry = input(tick, &points);
        entry.speed = 5.;
        let no_bsd = engine.step(entry);
        let mut entry = input(tick, &points);
        entry.speed = 5.;
        entry.blind = [false, (..24).contains(&tick)];
        let with_bsd = bsd.step(entry);
        if tick == 23 {
            let real = with_bsd.slots[4].unwrap();
            assert_eq!(real.last_track_id, Some(40));
            assert!(real.existence_p > no_bsd.objects[0].existence_p());
            assert!(real.s < -1.);
        }
        if tick == 24 {
            assert!(with_bsd.objects[0].existence_log_odds < 4. - 12. * 0.30 * STEP);
        }
    }
}

#[test]
fn side_front_never_exceeds_fifty_metres() {
    let mut engine = MapEngine::default();
    let out = confirm(
        &mut engine,
        &[radar(40, 50.1, -3.5, 0.), radar(41, 49.9, 3.5, 0.)],
    );
    assert!(out.slots[1].is_none());
    assert!(out.slots[2].unwrap().s <= 50.);
}

#[test]
fn a_matched_lead_supplies_ff_evidence_but_cannot_create_a_ghost() {
    let mut engine = MapEngine::default();
    let existing = [radar(40, 10., 2.1, 0.)];
    for tick in 0..8 {
        let mut entry = input(tick, &existing);
        entry.lead = Some((
            entry.time,
            DisplayLead {
                motion: (10., -2.1, 0.),
                measured_track: Some(40),
                model_prob: 0.95,
            },
        ));
        let out = engine.step(entry);
        if tick >= 3 {
            assert_eq!(out.slots[0].unwrap().last_track_id, Some(40));
            assert!(out.slots[2].is_none());
        }
    }
    let mut entry = input(0, &[]);
    entry.lead = Some((
        0.,
        DisplayLead {
            motion: (10., 0., 0.),
            measured_track: Some(40),
            model_prob: 0.98,
        },
    ));
    assert!(MapEngine::default().step(entry).objects.is_empty());
}

#[test]
fn current_radar_agreement_outweighs_a_closer_road_frame_candidate() {
    let mut engine = MapEngine::default();
    let points = [radar(40, 7., 0., 0.), radar(41, 12., 3.5, 0.)];
    assert_eq!(confirm(&mut engine, &points).slots[0].unwrap().last_track_id, Some(40));

    let mut entry = input(8, &points);
    entry.lead = Some((entry.time, DisplayLead {
        motion: (12., -3.5, 0.),
        measured_track: Some(41),
        model_prob: 0.95,
    }));
    let out = engine.step(entry);
    assert_eq!(out.slots[0].unwrap().last_track_id, Some(41));
    assert!(out.slots[2].is_none());
}

#[test]
fn displaced_ff_moves_directly_to_the_matching_side_only_outside_the_center() {
    for (offset, side) in [(-1.4, Some(1)), (1.4, Some(2)), (-1., None), (1., None)] {
        let mut engine = MapEngine::default();
        let points = [radar(40, 14., offset, 0.), radar(41, 8., 0., 0.)];
        let old_ff = confirm(&mut engine, &points[..1]).slots[0].unwrap();
        assert_eq!(old_ff.last_track_id, Some(40));
        assert_eq!((0.35..0.60).contains(&old_ff.lane_u.abs()), side.is_some());

        for tick in 8..15 {
            let mut entry = input(tick, &points);
            entry.lead = Some((
                entry.time,
                DisplayLead {
                    motion: (8., 0., 0.),
                    measured_track: Some(41),
                    model_prob: 0.95,
                },
            ));
            let out = engine.step(entry);
            assert_eq!(out.slots[0].unwrap().last_track_id, Some(41));
            for (index, slot) in out.slots.iter().enumerate().skip(1) {
                assert_eq!(
                    slot.map(|o| o.obj_id),
                    (side == Some(index)).then_some(old_ff.obj_id),
                    "offset {offset}, tick {tick}, slot {index}",
                );
            }
        }
    }
}

#[test]
fn side_range_grace_requires_an_established_and_continuously_measured_car() {
    let mut engine = MapEngine::default();
    let point = [radar(40, 22., 3.5, 0.)];
    for tick in 0..30 {
        let mut entry = input(tick, &point);
        entry.speed = 12.;
        engine.step(entry);
    }
    let frame = engine.previous.as_ref().unwrap();
    let mut slots = engine.displayed;
    let vid = slots[2].unwrap().obj_id;
    let objects = &mut engine.tracker.objects;
    assert!(objects[0].moved && objects[0].range_consistent);
    let mut locks = [None; 2];
    for tick in 30..47 {
        let now = f64::from(tick) * STEP;
        objects[0].range_consistent = false;
        objects[0].evidence.last_radar_t = Some(now);
        slots = selector::select(now, 12., false, false, frame, objects, slots, &mut locks, [false; 2], None, |_, _| false, |_, _| false);
        assert_eq!(slots[2].map(|o| o.obj_id), (tick < 46).then_some(vid));
    }

    // A missing measurement or an ID change cannot qualify for a range waiver.
    for new_track_id in [None, Some(41)] {
        let now = 30. * STEP;
        objects[0].evidence.last_radar_t = new_track_id.map(|_| now);
        objects[0].evidence.last_track_id = new_track_id.or(Some(40));
        let slots = selector::select(now, 12., false, false, frame, objects, engine.displayed, &mut locks, [false; 2], None, |_, _| false, |_, _| false);
        assert!(slots[2].is_none());
    }
    objects[0].evidence.last_radar_t = Some(30. * STEP);
    objects[0].evidence.last_track_id = Some(40);
    objects[0].moved = false;
    let slots = selector::select(30. * STEP, 12., false, false, frame, objects, [None; 5], &mut locks, [false; 2], None, |_, _| false, |_, _| false);
    assert!(slots[2].is_none());
}

#[test]
fn the_displaced_ff_can_replace_a_farther_side_car_without_double_display() {
    let mut engine = MapEngine::default();
    let points = [radar(40, 14., 1.70, 0.), radar(41, 32., 3.5, 0.)];
    for tick in 0..23 {
        let mut entry = input(tick, &points);
        entry.speed = 12.;
        engine.step(entry);
    }
    let old = engine.displayed[0].unwrap();
    assert_eq!(old.last_track_id, Some(40));
    assert_eq!(engine.displayed[2].unwrap().last_track_id, Some(41));
    let [old_lead, side_car] = points;
    let next = [old_lead, side_car, radar(42, 24., 0., 0.)];
    for tick in 23..33 {
        let mut entry = input(tick, &next);
        entry.speed = 12.;
        entry.lead = Some((entry.time, DisplayLead {
            motion: (24., 0., 0.),
            measured_track: Some(42),
            model_prob: 0.95,
        }));
        let out = engine.step(entry);
        if out.slots[0].is_some_and(|o| o.last_track_id == Some(42)) {
            let side = out.slots[2].unwrap();
            assert_eq!(side.obj_id, old.obj_id);
            assert_eq!(side.display_d, old.display_d);
            assert_eq!(out.slots.iter().flatten().filter(|o| o.obj_id == old.obj_id).count(), 1);
        }
    }
    assert_eq!(engine.displayed[0].unwrap().last_track_id, Some(42));
    assert_eq!(engine.displayed[2].unwrap().obj_id, old.obj_id);
}

#[test]
fn vision_birth_requires_repeated_nearfield_detections_and_carries_kind() {
    let mut engine = MapEngine::default();
    for tick in 0..10 {
        let mut entry = input(tick, &[]);
        let observations = [VisionObservation {
            time: entry.time,
            x: 4.,
            y: 3.5,
            kind: VehicleKind::Heavy,
            confidence: 0.9,
            position: true,
        }];
        entry.vision = &observations;
        let out = engine.step(entry);
        if tick >= 3 {
            assert_eq!(out.slots[2].unwrap().kind, VehicleKind::Heavy);
        }
    }
}

#[test]
fn persistent_static_return_requires_independent_vehicle_evidence() {
    let mut engine = MapEngine::default();
    let mut reflector = radar(55, 15., 3.5, 0.);
    reflector.motion_state = 1;
    let points = [reflector];
    for tick in 0..30 {
        let out = engine.step(input(tick, &points));
        assert!(out.slots.iter().all(Option::is_none));
    }
}

#[test]
fn delayed_camera_confirms_a_stopped_side_vehicle_only_with_current_radar() {
    let mut engine = MapEngine::default();
    let mut reflector = radar(55, 12., 3.5, 0.);
    reflector.motion_state = 1;
    let points = [reflector];
    for tick in 0..250 {
        let mut entry = input(tick, if matches!(tick, 64 | 220) { &[] } else { &points });
        let time = match tick {
            30 | 40 => Some(0.6), // A repeat of the same capture is one vote.
            60 => Some(2.),
            _ => None,
        };
        let observations = time.map(|time| VisionObservation {
            time,
            x: 12.,
            y: 3.5,
            kind: VehicleKind::Car,
            confidence: 0.9,
            position: true,
        });
        entry.vision = observations.as_slice();
        let out = engine.step(entry);
        let expected = (60..=200).contains(&tick) && tick != 64;
        assert_eq!(out.slots[2].is_some(), expected, "tick {tick}");
    }
}

#[test]
fn confirmed_stopped_vehicle_outlives_camera_window_only_with_unbroken_radar() {
    for invalid_frame in [false, true] {
        let mut engine = MapEngine::default();
        let mut stopped = radar(55, 12., 3.5, 0.);
        stopped.motion_state = 1;
        let points = [stopped];
        for tick in 0..260 {
            if tick == 220 && invalid_frame {
                engine.break_radar_continuity();
                continue;
            }
            let mut entry = input(tick, if tick == 220 { &[] } else { &points });
            let observations = match tick {
                30 => Some(0.6),
                60 => Some(2.),
                _ => None,
            }
            .map(|time| VisionObservation {
                time,
                x: 12.,
                y: 3.5,
                kind: VehicleKind::Car,
                confidence: 0.9,
                position: true,
            });
            entry.vision = observations.as_slice();
            let out = engine.step(entry);
            assert_eq!(out.slots[2].is_some(), (60..220).contains(&tick), "tick {tick}");
        }
    }
}

#[test]
fn a_delayed_box_beyond_the_outer_line_does_not_promote_a_stationary_echo() {
    use crate::input::model_v2::Line;

    for (edge, expected) in [(-4., false), (-5., true)] {
        let mut engine = MapEngine::default();
        let mut reflector = radar(40, 18., -4.5, 0.);
        reflector.motion_state = 1;
        let points = [reflector];
        for tick in 0..29 {
            engine.step(input(tick, &points));
        }
        let obs = [VisionObservation {
            time: 0.8,
            x: 18.,
            y: -4.5,
            kind: VehicleKind::Car,
            confidence: 0.9,
            position: true,
        }];
        engine.tracker.objects[0].front_box_t = Some(obs[0].time);
        let model = Model {
            lane_lines: vec![Line { x: vec![0., 30.], y: vec![edge; 2] }],
            lane_line_probs: vec![0.9],
            road_edges: Vec::new(),
        };
        let mut entry = input(29, &points);
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &[],
            lines: Some(&model),
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: false,
        });
        entry.vision = &obs;
        assert_eq!(engine.step(entry).slots[1].is_some(), expected);
    }
}

#[test]
fn a_confirmed_near_car_stays_in_front_when_it_cannot_take_the_rear_slot() {
    let mut engine = MapEngine::default();
    let out = confirm(
        &mut engine,
        &[radar(1, 7.5, -3.5, 0.), radar(2, 17., -3.5, 0.)],
    );
    assert_eq!(out.slots[1].unwrap().last_track_id, Some(1));
    assert!(out.slots[3].is_none());
}

#[test]
fn steady_near_return_at_a_stop_can_split_only_with_a_distinct_confirmed_car() {
    let mut engine = MapEngine::default();
    let mut near = radar(1, 4.5, 3.5, 0.);
    near.motion_state = 0;
    let out = confirm(&mut engine, std::slice::from_ref(&near));
    assert!(out.slots[2].is_none() && out.slots[4].is_none());
    let points = [near, radar(2, 12., 3.5, 0.)];
    for tick in 8..15 {
        let out = engine.step(input(tick, &points));
        if tick >= 11 {
            assert_eq!(out.slots[2].unwrap().last_track_id, Some(2));
            assert_eq!(out.slots[4].unwrap().last_track_id, Some(1));
        }
    }
}

fn front_kind(
    cam: &FrontCam,
    captured: u32,
    range: f64,
    lateral: f64,
    kind: u8,
    height: f64,
) -> KindResult {
    use crate::vision::kind::KindView;

    let (u, v) = cam.radar_ground(range, -lateral).unwrap();
    let scale = 1141.5 / (range + det_rules::CAM2B);
    KindResult {
        time_ns: u64::from(captured) * 50_000_000,
        view: KindView::Front,
        xyxy: [u - scale, v - height * scale, u + scale, v],
        kind,
        prob: 0.98,
    }
}

#[test]
fn delayed_tall_classifier_answer_follows_capture_object_into_rear() {
    use crate::vision::kind::KIND_BUS;

    let mut engine = MapEngine::default();
    let cam = FrontCam::new([0.; 3], 1.22).unwrap();
    let mut identity = None;
    for tick in 0..120 {
        let x = 31. - 6. * STEP * f64::from(tick);
        if tick == 58 {
            let capture = 20;
            let answer = front_kind(&cam, capture, 25., 3.5, KIND_BUS, 3.);
            // The answer arrives 1.9 seconds later, by which time the range has changed by 11.4 m.
            engine.observe_front_kinds(u64::from(tick) * 50_000_000, &cam, &[answer]);
        }
        let points = if x > 0.1 {
            [
                Some(radar(1, x as f32, 3.5, -6.)),
                Some(radar(2, 16., -3.5, 0.)),
            ]
        } else {
            [None, Some(radar(2, 16., -3.5, 0.))]
        };
        let points: Vec<_> = points.into_iter().flatten().collect();
        let mut entry = input(tick, &points);
        entry.speed = 10.;
        entry.blind[1] = x <= 0.1;
        let out = engine.step(entry);
        if tick == 57 {
            let rf = out.slots[2].unwrap();
            identity = Some(rf.obj_id);
            assert_eq!(rf.kind, VehicleKind::Unknown);
        }
        if tick == 58 || (106..110).contains(&tick) {
            let slot = out.slots[if tick == 58 { 2 } else { 4 }].unwrap();
            assert_eq!(Some(slot.obj_id), identity);
            assert_eq!(slot.kind, VehicleKind::Heavy);
            assert_ne!(out.slots[1].unwrap().kind, VehicleKind::Heavy);
        }
    }
}

#[test]
fn heavy_ignores_single_car_vote_then_returns_after_four_agreeing_answers() {
    use crate::vision::kind::{KIND_BUS, KIND_CAR};

    let mut engine = MapEngine::default();
    let cam = FrontCam::new([0.; 3], 1.22).unwrap();
    let points = [radar(1, 20., 3.5, 0.)];
    for tick in 0..125 {
        let answer = match tick {
            10 | 50 => Some((KIND_BUS, 3.)),
            30 | 70 | 87 | 104 | 121 => Some((KIND_CAR, 1.5)),
            _ => None,
        };
        if let Some((kind, height)) = answer {
            let answer = front_kind(&cam, tick - 1, 20., 3.5, kind, height);
            engine.observe_front_kinds(u64::from(tick) * 50_000_000, &cam, &[answer]);
        }
        let out = engine.step(input(tick, &points));
        let expected = match tick {
            10..=120 => Some(VehicleKind::Heavy),
            121.. => Some(VehicleKind::Car),
            _ => None,
        };
        if let Some(kind) = expected {
            assert_eq!(out.slots[2].unwrap().kind, kind, "tick {tick}");
        }
    }
}

#[test]
fn a_tall_box_does_not_classify_a_second_track_within_truck_length() {
    use crate::vision::kind::KIND_BUS;

    let mut engine = MapEngine::default();
    let cam = FrontCam::new([0.; 3], 1.22).unwrap();
    let points = [radar(1, 20., 3.5, 0.), radar(2, 30., 3.5, 0.)];
    confirm(&mut engine, &points);
    let answer = front_kind(&cam, 7, 20., 3.5, KIND_BUS, 3.);
    engine.observe_front_kinds(400_000_000, &cam, &[answer]);
    let out = engine.step(input(8, &points));
    assert_eq!(out.slots[2].unwrap().kind, VehicleKind::Heavy);
    let farther = out
        .objects
        .iter()
        .find(|o| o.evidence.last_track_id == Some(2))
        .unwrap();
    assert_eq!(farther.kind, VehicleKind::Unknown);
}

#[test]
fn stopping_road_frame_preserves_previous_curvature_and_lanes() {
    use crate::input::model_v2::Line;

    let curved = [(0., 0.), (5., 0.5), (10., 2.), (15., 4.5), (20., 8.), (30., 18.)];
    let make_lines = || {
        [-5.5, -1.9, 2.1, 5.8]
            .map(|y| Line { x: vec![0., 30.], y: vec![y, y] })
            .into()
    };
    let model = Model {
        lane_lines: make_lines(),
        lane_line_probs: vec![0.95; 4],
        road_edges: Vec::new(),
    };
    let pose = PoseAt::default();
    let prev = RoadFrame::new(&curved, Some(&model), pose, pose, std::iter::empty(), false);
    // When stopping, modelV2.position collapses below 1 m and lane probabilities drop.
    let collapsed = [(0., 0.), (0.4, 0.02)];
    let low_prob = Model {
        lane_lines: make_lines(),
        lane_line_probs: vec![0.4; 4],
        road_edges: Vec::new(),
    };
    let next = RoadFrame::new_with_previous(
        &collapsed,
        Some(&low_prob),
        pose,
        pose,
        std::iter::empty(),
        false,
        Some((&prev, pose)),
    );
    // Point on the previous curve at (15, 4.5) must stay near d = 0 instead of snapping to (50, 0).
    let (_, d_curved) = next.project(15., 4.5);
    assert!(d_curved.abs() < 0.35, "expected continuous curve extension, got d={d_curved}");
    // Carried-forward lane width (4.0 m) & offset (+0.1 m) must remain stable instead of jumping to 3.5 m.
    let prev_u = prev.lane_u(10., 4.1);
    let next_u = next.lane_u(10., 4.1);
    assert!((prev_u - next_u).abs() < 0.05, "lane_u jumped: {prev_u} -> {next_u}");
}

#[test]
fn pedestrian_requires_n2_confirmation_and_displays_dedicated_kind_without_creating_ghosts() {
    let mut engine = MapEngine::default();
    // Low-speed roadside walking radar track at (12 m, +3.2 m) with ground speed ~1.2 m/s.
    let mut ped_radar = radar(77, 12., 3.2, 1.2);
    ped_radar.motion_state = 2;
    let pts = [ped_radar];
    // Establish radar track first (no pedestrian observations yet -> Unknown).
    for tick in 0..8 {
        let mut entry = input(tick, &pts);
        entry.speed = 0.0;
        let out = engine.step(entry);
        if let Some(rf) = out.slots[2] {
            assert_eq!(rf.kind, VehicleKind::Unknown);
        }
    }
    // Single pedestrian observation (N = 1) must not confirm Pedestrian yet.
    engine.observe_pedestrian(PedestrianObservation {
        time: 8. * STEP,
        x: 12.1,
        y: 3.1,
        confidence: 0.85,
    });
    let mut entry = input(8, &pts);
    entry.speed = 0.0;
    let out = engine.step(entry);
    assert_eq!(out.slots[2].expect("walking radar track shown in RF").kind, VehicleKind::Unknown);

    // Second pedestrian observation (N = 2) confirms Pedestrian.
    engine.observe_pedestrian(PedestrianObservation {
        time: 9. * STEP,
        x: 12.1,
        y: 3.1,
        confidence: 0.85,
    });
    let mut entry = input(9, &pts);
    entry.speed = 0.0;
    let out = engine.step(entry);
    assert_eq!(out.slots[2].expect("pedestrian shown in RF").kind, VehicleKind::Pedestrian);

    // Pedestrian observation alone (without radar) must never create a displayed slot.
    let mut empty_engine = MapEngine::default();
    for tick in 0..6 {
        empty_engine.observe_pedestrian(PedestrianObservation {
            time: f64::from(tick) * STEP,
            x: 8.,
            y: 3.0,
            confidence: 0.9,
        });
        let out = empty_engine.step(input(tick, &[]));
        assert!(out.slots.iter().all(Option::is_none));
    }

    // Stationary roadside radar clutter (motion_state = 0, not moving) must never be promoted
    // into a new displayed slot by pedestrian observations (preserves 482 G4 new_shown = 0).
    let mut clutter_engine = MapEngine::default();
    let mut bollard = radar(88, 12., 3.2, 0.0);
    bollard.motion_state = 0;
    let bollard_pts = [bollard];
    for tick in 0..10 {
        clutter_engine.observe_pedestrian(PedestrianObservation {
            time: f64::from(tick) * STEP,
            x: 12.0,
            y: 3.2,
            confidence: 0.85,
        });
        let mut entry = input(tick, &bollard_pts);
        entry.speed = 0.0;
        let out = clutter_engine.step(entry);
        assert!(out.slots[2].is_none(), "stationary bollard must not become a new shown slot at tick {tick}");
    }
}

#[test]
fn motorcycle_rider_priority_and_pedestrian_size_guard_in_front_frame() {
    use crate::pipeline::Pipeline;
    use crate::vision::{
        kind::{FrontCam, FOCAL, KIND_MOTORCYCLE, KIND_PEDESTRIAN},
        Box as VisionBox, FrontBox,
    };

    let calib = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.22];
    let cam = FrontCam::new([0., 0., 0.], 1.22).unwrap();
    let make_box = |d_rel: f64, y_rel: f64, w_m: f64, h_m: f64, class_id: u8, conf: f64| -> FrontBox {
        let (u, v) = cam.radar_ground(d_rel, y_rel).unwrap();
        let x = d_rel + det_rules::CAM2B;
        let w_px = w_m * FOCAL / x;
        let h_px = h_m * FOCAL / x;
        FrontBox {
            detection: VisionBox {
                xyxy: [u - w_px / 2., v - h_px, u + w_px / 2., v],
                confidence: conf,
                class_id,
            },
            cut: false,
        }
    };

    // Case 1: Person box overlapping a motorcycle box -> rider priority selects Motorcycle, not Pedestrian.
    let mut pipe = Pipeline::new(1);
    let mut last_map = None;
    for tick in 0..10u64 {
        let t_ns = 1_000_000_000 + tick * 50_000_000;
        if tick >= 5 {
            let moto = make_box(12.0, 3.2, 0.8, 1.3, 3, 0.65);
            // Rider sitting on motorcycle: overlaps upper portion of motorcycle box.
            let rider = make_box(12.0, 3.2, 0.6, 1.7, 0, 0.85);
            pipe.front_frame(t_ns, &calib, &[moto, rider]);
        }
        let mut moto_pt = radar(51, 12.0, -3.2, 1.5);
        moto_pt.motion_state = 2;
        last_map = Some(pipe.step_for_test(t_ns, 0.0, &[moto_pt]));
    }
    let lf = last_map.unwrap().slots[1].expect("LF motorcycle shown");
    assert_eq!(lf.kind, KIND_MOTORCYCLE, "rider overlapping motorcycle must emit KIND_MOTORCYCLE");
    assert!(lf.kind_confirmed);

    // Case 2: Bollard/pole failing physical size guard (0.12 m wide, 0.6 m tall) must not confirm Pedestrian.
    let mut pipe_small = Pipeline::new(1);
    let mut last_small = None;
    for tick in 0..10u64 {
        let t_ns = 1_000_000_000 + tick * 50_000_000;
        if tick >= 5 {
            let bollard_box = make_box(12.0, -3.2, 0.12, 0.6, 0, 0.85);
            pipe_small.front_frame(t_ns, &calib, &[bollard_box]);
        }
        let mut pt = radar(52, 12.0, 3.2, 1.2);
        pt.motion_state = 2;
        last_small = Some(pipe_small.step_for_test(t_ns, 0.0, &[pt]));
    }
    let rf_small = last_small.unwrap().slots[2].expect("RF radar track shown");
    assert_eq!(rf_small.kind, 0, "out-of-size-guard box must not confirm Pedestrian");

    // Case 3: Valid person box (0.6 m wide, 1.7 m tall) for N >= 2 frames confirms KIND_PEDESTRIAN.
    let mut pipe_ped = Pipeline::new(1);
    let mut last_ped = None;
    for tick in 0..10u64 {
        let t_ns = 1_000_000_000 + tick * 50_000_000;
        if tick >= 7 {
            let ped_box = make_box(12.0, -3.2, 0.6, 1.7, 0, 0.80);
            pipe_ped.front_frame(t_ns, &calib, &[ped_box]);
        }
        let mut pt = radar(53, 12.0, 3.2, 1.2);
        pt.motion_state = 2;
        last_ped = Some(pipe_ped.step_for_test(t_ns, 0.0, &[pt]));
    }
    let rf_ped = last_ped.unwrap().slots[2].expect("RF pedestrian shown");
    assert_eq!(rf_ped.kind, KIND_PEDESTRIAN, "valid N>=2 person box must emit KIND_PEDESTRIAN");
    assert!(rf_ped.kind_confirmed);
}

#[test]
fn rear_handoff_preserves_heavy_motorcycle_and_pedestrian_kinds() {
    use super::rear_display::SideDisplay;

    for kind in [VehicleKind::Heavy, VehicleKind::Motorcycle, VehicleKind::Pedestrian] {
        let mut disp = SideDisplay::default();
        let base_front = EngineSlot {
            obj_id: 99,
            s: 1.5,
            d: -3.2,
            x: 1.5,
            y: -3.2,
            display_d: -3.2,
            rear_animated: false,
            v_s: -4.0,
            kind,
            existence_p: 0.99,
            last_track_id: Some(99),
            lane_u: -1.0,
            camera_seen: false,
            radar_seen: true,
            side_range: None,
            ff_handoff_until: None,
        };
        let prev = [None, Some(base_front), None, None, None];
        let rear_cross = EngineSlot { s: -0.5, x: -0.5, ..base_front };
        let mut slots = [None, None, None, Some(rear_cross), None];
        disp.update(10.0, 0, false, prev, &mut slots);
        let lr = slots[3].expect("LR rear pass slot active");
        assert_eq!(lr.kind, kind, "LR crossing must preserve {kind:?}");

        // Coasting in LR without new radar must still preserve kind.
        let mut coast_slots = [None; 5];
        disp.update(10.5, 0, false, slots, &mut coast_slots);
        let lr_coast = coast_slots[3].expect("LR rear coast active");
        assert_eq!(lr_coast.kind, kind, "LR coasting must preserve {kind:?}");
    }
}

#[test]
fn motorcycle_keeps_separate_track_and_does_not_displace_settled_adjacent_car() {
    let mut engine = MapEngine::default();
    // Establish a moving passenger car in RF at (15 m, +3.5 m).
    let car_pts = [radar(10, 15., 3.5, 2.0)];
    for tick in 0..20 {
        let mut entry = input(tick, &car_pts);
        entry.speed = 6.0;
        engine.step(entry);
    }
    // A lane-splitting motorcycle appears straddling the right lane boundary at (10 m, +1.8 m, u ~ 0.51).
    let both_pts = [radar(10, 15., 3.5, 2.0), radar(11, 10., 1.8, 2.5)];
    let mut car_kept = true;
    for tick in 20..26 {
        let t = f64::from(tick) * STEP;
        let obs = [VisionObservation {
            time: t,
            x: 10.,
            y: 1.8,
            kind: VehicleKind::Motorcycle,
            confidence: 0.8,
            position: true,
        }];
        let mut entry = input(tick, &both_pts);
        entry.speed = 6.0;
        entry.vision = &obs;
        let out = engine.step(entry);
        if out.slots[2].is_some_and(|s| s.last_track_id != Some(10)) {
            car_kept = false;
        }
    }
    assert!(car_kept, "lane-splitting motorcycle must not displace established RF car");
    let moto_obj = engine
        .objects()
        .iter()
        .find(|o| o.evidence.last_track_id == Some(11))
        .unwrap();
    assert_eq!(moto_obj.kind, VehicleKind::Motorcycle);
}

#[test]
fn dynamic_pitch_compensation_restores_ground_range_under_braking_dive() {
    let level = FrontCam::new([0., 0., 0.], 1.22).unwrap();
    // Under braking dive (-1.15 deg = -0.02 rad nose down), a point at 20 m projects higher in the image.
    let dive = -0.02_f64;
    let tilted = FrontCam::new([0., -dive, 0.], 1.22).unwrap();
    let (u, v_tilted) = tilted.radar_ground(20.0 - det_rules::CAM2B, 0.0).unwrap();
    let (x_uncomp, _) = level.ground_from_pixel(u, v_tilted).unwrap();
    let (x_comp, _) = tilted.ground_from_pixel(u, v_tilted).unwrap();
    assert!(x_uncomp > 25.0, "uncompensated braking dive inflates ground range: {x_uncomp}");
    assert!((x_comp - 20.0).abs() < 1e-6, "compensated range must match 20.0 m: {x_comp}");

    // Verify WCam::ground_with_pitch matches exact pitch rotation.
    let calib_level = det_rules::Calib {
        rpy: [0., -0.01, 0.004],
        wide_from_device: [0., -0.01, -0.02],
        height: 1.22,
    };
    let calib_tilted = det_rules::Calib {
        rpy: [0., -0.01 - dive, 0.004],
        wide_from_device: [0., -0.01, -0.02],
        height: 1.22,
    };
    let w_level = det_rules::geom::WCam::new(&calib_level, det_rules::geom::LENS);
    let w_tilted = det_rules::geom::WCam::new(&calib_tilted, det_rules::geom::LENS);
    let ray = w_level.ray(672., 430.);
    let g_exact = w_tilted.ground(w_tilted.ray(672., 430.)).unwrap();
    let g_comp = w_level.ground_with_pitch(ray, dive).unwrap();
    assert!((g_exact[0] - g_comp[0]).abs() < 0.05 && (g_exact[1] - g_comp[1]).abs() < 0.05);
}

#[test]
fn side_kinds_confirms_tall_side_bus_and_rejects_short_side_van() {
    use crate::vision::{kind::{KindResult, KindView, KIND_BUS}, Side};
    use det_rules::{Cand, View};

    let mut engine = MapEngine::default();
    let cam = FrontCam::new([0.; 3], 1.22).unwrap();
    let points = [radar(5, 4.0, 3.5, 0.)];
    confirm(&mut engine, &points);
    // Tall bus box in SideR (height 180 px at range ~6.5 m -> ~3.8 m tall, >= 48 px and >= 2.05 m).
    let mut cand = Cand::at(View::SideR, 4.0, 3.5);
    cand.bbox = [100., 80., 300., 260.];
    let bus_ans = |t_ns| KindResult {
        time_ns: t_ns,
        view: KindView::Side(Side::Right),
        xyxy: cand.bbox,
        kind: KIND_BUS,
        prob: 0.96,
    };
    engine.observe_kinds(400_000_000, &cam, &[bus_ans(350_000_000)], |_, _| Some(vec![cand]));
    let out = engine.step(input(8, &points));
    assert_eq!(out.slots[2].unwrap().kind, VehicleKind::Heavy);

    // Short van box in SideR (height 75 px at range ~6.5 m -> ~1.60 m < MIN_TRUCK_HEIGHT_M=2.05 m).
    let mut van_engine = MapEngine::default();
    confirm(&mut van_engine, &points);
    let mut van_cand = Cand::at(View::SideR, 4.0, 3.5);
    van_cand.bbox = [100., 185., 260., 260.];
    let van_ans = KindResult {
        time_ns: 350_000_000,
        view: KindView::Side(Side::Right),
        xyxy: van_cand.bbox,
        kind: KIND_BUS,
        prob: 0.98,
    };
    van_engine.observe_kinds(400_000_000, &cam, &[van_ans], |_, _| Some(vec![van_cand]));
    let out = van_engine.step(input(8, &points));
    assert_ne!(out.slots[2].unwrap().kind, VehicleKind::Heavy, "low side van must not confirm as Heavy");
}

#[test]
fn driving_blind_spot_vision_track_survives_cruise_cadence_with_comoving_init() {
    let mut engine = MapEngine::default();
    let speed = 18.0;
    // Cruise wide cadence: 500 ms total -> 1.0 s per side (every 20 ticks at 20 Hz), with 75 ms latency.
    for tick in 0..70 {
        let mut entry = input(tick, &[]);
        entry.speed = speed;
        let obs = (tick % 20 == 0).then(|| {
            let age = 0.075;
            VisionObservation {
                time: entry.time - age,
                // Static ground-contact warp shifts captured x by -speed * age; co-moving v_s = 0 restores s = 4.0.
                x: 4.0 - speed * age,
                y: 3.5,
                kind: VehicleKind::Car,
                confidence: 0.85,
                position: true,
            }
        });
        entry.vision = obs.as_slice();
        let out = engine.step(entry);
        let slot2 = out.slots[2];
        if tick == 0 {
            let obj = engine.objects().iter().find(|o| (o.d - 3.5).abs() < 0.5).expect("blind-spot track created");
            assert!(obj.v_s.abs() < 0.5, "driving vision birth must initialize v_s near 0, got {}", obj.v_s);
        }
        if tick == 19 {
            assert!(!engine.objects().is_empty(), "blind-spot track must survive 1.0 s cruise gap");
        }
        if tick >= 40 {
            assert!(slot2.is_some(), "confirmed blind-spot car must stay visible at tick {tick}");
        }
    }
}

#[test]
fn driving_parked_roadside_radar_not_promoted_by_side_vision() {
    let mut engine = MapEngine::default();
    let speed = 12.0;
    // Stationary roadside parked car approaching at -speed with radar + matching wide vision.
    for tick in 0..15 {
        let x = 14.0 - speed * (tick as f64) * STEP;
        let mut parked = radar(77, x as f32, 3.5, -(speed as f32));
        parked.motion_state = 1;
        let points = [parked];
        let mut entry = input(tick, &points);
        entry.speed = speed;
        let obs = (tick % 4 == 0 && x > 0.5).then_some(VisionObservation {
            time: entry.time,
            x,
            y: 3.5,
            kind: VehicleKind::Car,
            confidence: 0.9,
            position: true,
        });
        entry.vision = obs.as_slice();
        let out = engine.step(entry);
        assert!(out.slots[2].is_none(), "parked roadside car at tick {tick} must not be promoted by side vision");
    }
}

#[test]
fn side_raw_yolox_bus_or_truck_does_not_bypass_min_truck_height() {
    use crate::pipeline::Pipeline;
    use crate::vision::{Box as VisionBox, Calib, Frame, FrameView, Geometry};

    let calib = Calib { rpy: [0.; 3], wide_from_device: [0.; 3], height: 1.22 };
    let mut pipe = Pipeline::new(1);
    pipe.set_vision_geometry(Geometry::build(&calib));
    let frame = Frame {
        time_ns: 1_000_000_000,
        view: FrameView::Side(crate::vision::Side::Right),
        calibration: [0., 0., 0., 0., 0., 0., 1.22],
        // Large SideR box with raw YOLOX class 5 (bus) and 7 (truck).
        boxes: vec![
            VisionBox { xyxy: [180., 185., 340., 260.], confidence: 0.92, class_id: 5 },
            VisionBox { xyxy: [180., 185., 340., 260.], confidence: 0.92, class_id: 7 },
        ],
    };
    pipe.vision(1_050_000_000, &frame).unwrap();
    assert!(pipe.objects().iter().all(|o| o.kind != VehicleKind::Heavy));
}

#[test]
fn side_vehicle_far_band_50_to_70m_gates_moving_width_and_curve() {
    // Moving side vehicle at 60 m in adjacent lane (y=3.5 m, u=1.0) is displayed on straight road.
    let mut engine = MapEngine::default();
    for tick in 0..30 {
        let points = [radar(81, 60., 3.5, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 25.;
        let out = engine.step(entry);
        if tick == 25 {
            let rf = out.slots[2].expect("moving 60 m side car should be shown");
            assert!((rf.x - 60.).abs() < 1.0, "slot.x must not be clamped to 50 m, got {}", rf.x);
        }
    }
    // Sharp curve (R = 300 m < 400 m -> yaw = speed / 300) shrinks side limit back to 50 m.
    let mut curve_engine = MapEngine::default();
    for tick in 0..30 {
        let points = [radar(82, 60., 3.5, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 25.;
        entry.yaw = Some(25. / 300.);
        let out = curve_engine.step(entry);
        if tick >= 20 {
            assert!(out.slots[2].is_none(), "60 m side car must be hidden on R<400m curve");
        }
    }
    // Too wide (y=4.8 m > 4.4 m) in 50..70 m band is rejected.
    let mut wide_engine = MapEngine::default();
    for tick in 0..30 {
        let points = [radar(83, 60., 4.8, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 25.;
        let out = wide_engine.step(entry);
        if tick >= 20 {
            assert!(out.slots[2].is_none(), "60 m car outside 4.4 m width gate must be hidden");
        }
    }
}

#[test]
fn stopped_side_vehicle_fast_mnn_vision_holds_without_flicker() {
    let mut engine = MapEngine::default();
    // Stopped ego + stopped adjacent radar car at 5.5 m, fast MNN vision (age = 0.10 s) every 0.60 s.
    for tick in 0..80 {
        let mut car = radar(91, 5.5, 3.5, 0.);
        car.motion_state = 1;
        let points = [car];
        let mut entry = input(tick, &points);
        entry.speed = 0.;
        let obs = (tick >= 8 && tick % 12 == 0).then_some(VisionObservation {
            time: entry.time - 0.10,
            x: 5.5,
            y: 3.5,
            kind: VehicleKind::Car,
            confidence: 0.85,
            position: true,
        });
        entry.vision = obs.as_slice();
        let out = engine.step(entry);
        // After two vision frames (tick 24), RF must stay continuously ON with zero off-flickers.
        if tick >= 25 {
            assert!(out.slots[2].is_some(), "stopped side car flickered off at tick {tick}");
        }
    }
}

#[test]
fn highway_median_mirror_ghost_and_oblique_guardrail_rejected() {
    use crate::input::model_v2::Line;
    // Case 3-A: FF truck at 43 m + LF radar-only mirror ghost at 44 m, y=-4.3 m outside left roadEdge (-2.8 m).
    let mut engine = MapEngine::default();
    let model = Model {
        lane_lines: Vec::new(),
        lane_line_probs: Vec::new(),
        road_edges: vec![
            Line { x: vec![0., 10., 50., 80.], y: vec![-2.8, -2.8, -2.8, -2.8] },
            Line { x: vec![0., 10., 50., 80.], y: vec![10.0, 10.0, 10.0, 10.0] },
        ],
    };
    for tick in 0..30 {
        let points = [radar(101, 43., 0., 0.), radar(102, 44., -4.3, 0.)];
        let mut entry = input(tick, &points);
        entry.speed = 28.;
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &[(0., 0.), (60., 0.)],
            lines: Some(&model),
            road_edge_stds: [0.25, 1.2],
            changing_lanes: false,
        });
        let out = engine.step(entry);
        if tick >= 10 {
            assert!(out.slots[0].is_some(), "FF lead must remain visible");
            assert!(out.slots[1].is_none(), "mirror ghost outside roadEdge must be rejected at tick {tick}");
        }
    }
    // Case 3-B: Oblique stationary guardrail at x=6.0 m, y=-3.7 m, vEgo=22.4, vRel=-20.9 (vAbs=+1.5 m/s).
    let mut rail_engine = MapEngine::default();
    for tick in 0..10 {
        let x = 7.3 - 0.5 * (tick as f32);
        let mut rail = radar(103, x, -3.7, -20.9);
        rail.motion_state = 1;
        let points = [rail];
        let mut entry = input(tick, &points);
        entry.speed = 22.4;
        let out = rail_engine.step(entry);
        assert!(out.slots[1].is_none(), "oblique guardrail must be rejected at tick {tick}");
    }
}

#[test]
fn bsd_only_animates_approach_and_retreat_smoothly() {
    let mut engine = MapEngine::default();
    let mut on_xs = Vec::new();
    for tick in 0..60 {
        let mut entry = input(tick, &[]);
        entry.speed = 15.;
        entry.blind = [true, false];
        let out = engine.step(entry);
        let lr = out.slots[3].expect("BSD-only must emit animated LR slot");
        assert_eq!(lr.obj_id, 0);
        assert!(lr.rear_animated);
        on_xs.push(lr.x);
    }
    assert!((on_xs[0] - (-20.)).abs() < 1e-6, "BSD ON must start at -20 m (16 m cluster), got {}", on_xs[0]);
    assert!((*on_xs.last().unwrap() - (-10.)).abs() < 0.1, "BSD ON must settle at -10 m (8 m cluster), got {}", on_xs.last().unwrap());
    assert!(on_xs.windows(2).all(|w| w[1] >= w[0] - 1e-9), "BSD approach must be monotonic");

    let mut off_xs = Vec::new();
    let mut cleared = false;
    for tick in 60..240 {
        let mut entry = input(tick, &[]);
        entry.speed = 15.;
        entry.blind = [false, false];
        let out = engine.step(entry);
        if let Some(lr) = out.slots[3] {
            off_xs.push(lr.x);
        } else {
            cleared = true;
            break;
        }
    }
    assert!(cleared, "BSD OFF must clear after 3s hold + retreating to -20 m");
    // First 3.0 s (60 ticks at 20 Hz) after BSD OFF must hold at -10 m without retreating.
    assert!(off_xs.len() >= 70, "BSD OFF must hold 3s then retreat smoothly, got {} ticks", off_xs.len());
    assert!(
        off_xs[..60].iter().all(|&x| (x - (-10.)).abs() < 0.1),
        "BSD OFF must hold at -10 m for 3.0 s without retreating, min={}",
        off_xs[..60].iter().copied().fold(f64::INFINITY, f64::min),
    );
    assert!(off_xs[60] < -10.05, "BSD OFF must begin retreating after 3.0 s hold, got {}", off_xs[60]);
    assert!(off_xs.windows(2).all(|w| w[1] <= w[0] + 1e-9), "BSD retreat must be monotonic");
}

#[test]
fn bsd_only_holds_three_seconds_and_joins_front_without_retreat() {
    // 1. End-to-end MapEngine: BSD ON (-20 -> -10 m), BSD OFF (hold at -10 m during B-pillar blind zone),
    //    then overtaking vehicle emerges in front -> FrontJoin (-10 -> 0 m -> LF) with zero backwards steps.
    let mut engine = MapEngine::default();
    let mut all_xs = Vec::new();
    let mut saw_origin = false;
    let mut saw_lf = false;
    for tick in 0_u32..140 {
        let points = if tick >= 80 {
            let x = 3.5 + (tick - 80) as f32 * 0.12;
            vec![radar(41, x, -3.5, 2.4)]
        } else {
            Vec::new()
        };
        let mut entry = input(tick, &points);
        entry.speed = 15.;
        entry.blind = [tick < 60, false];
        let out = engine.step(entry);
        let slot = out.slots[1].or(out.slots[3]).expect("overtaking car must stay visible through hold and handoff");
        all_xs.push(slot.x);
        if (60..80).contains(&tick) {
            assert!((slot.x - (-10.)).abs() < 0.1, "must hold at -10 m during BSD OFF, tick {tick} x={}", slot.x);
        }
        if slot.x == 0. {
            saw_origin = true;
        }
        if out.slots[1].is_some() {
            saw_lf = true;
        }
    }
    assert!(saw_origin, "FrontJoin must pass through exact origin 0 m");
    assert!(saw_lf, "overtaking car must hand off into LF");
    assert!(
        all_xs.windows(2).all(|w| w[1] >= w[0] - 1e-9),
        "BSD approach -> hold -> FrontJoin -> LF must never move backwards",
    );

    // 2. Direct SideDisplay::update with old_rear = Some(BsdOnly { obj_id: 0 }):
    //    from_bsd must still fire and start from the held BsdOnly x without retreating.
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(42, 3.5, -3.5, 2.5)]);
    let front_slot = engine.displayed[1].unwrap();
    let mut display = rear_display::SideDisplay::default();
    let mut prev = [None; 5];
    for i in 0..60 {
        let mut slots = [None; 5];
        display.update(i as f64 * STEP, 0, true, prev, &mut slots);
        prev = slots;
    }
    // BSD OFF for 1.5 s: held at -10 m with obj_id == 0 in prev[3].
    for i in 60..90 {
        let mut slots = [None; 5];
        display.update(i as f64 * STEP, 0, false, prev, &mut slots);
        assert_eq!(slots[3].unwrap().obj_id, 0);
        assert!((slots[3].unwrap().x - (-10.)).abs() < 0.1);
        prev = slots;
    }
    let mut join_xs = Vec::new();
    for i in 90..115 {
        let mut slots = [None, Some(EngineSlot { s: 3.5, x: 3.5, v_s: 2.5, ..front_slot }), None, None, None];
        display.update(i as f64 * STEP, 0, false, prev, &mut slots);
        let active = slots[1].or(slots[3]).expect("FrontJoin must stay visible");
        join_xs.push(active.x);
        prev = slots;
    }
    assert!((join_xs[0] - (-10.)).abs() < 0.1, "FrontJoin must start from held BsdOnly x=-10 m, got {}", join_xs[0]);
    assert!(join_xs.iter().any(|&x| x == 0.), "FrontJoin must pass through 0 m");
    assert!(join_xs.last().unwrap() > &3.0, "FrontJoin must complete into LF");
    assert!(join_xs.windows(2).all(|w| w[1] >= w[0] - 1e-9), "FrontJoin must advance monotonically");
}

#[test]
fn standstill_oncoming_radar_and_vision_are_rejected_without_breaking_stopped_side() {
    use crate::input::model_v2::Line;

    // 1. Oncoming vehicle at standstill (vRel = -2.4 m/s, motion_state = 2) with radar + vision
    //    must never appear in LF/LR, while a stopped adjacent RF car stays continuously ON.
    let mut engine = MapEngine::default();
    for tick in 0..60 {
        let oncoming_x = 14.0 - 0.12 * tick as f32;
        let mut oncoming = radar(111, oncoming_x, -3.5, -2.4);
        oncoming.motion_state = 2;
        let mut stopped_rf = radar(112, 5.5, 3.5, 0.0);
        stopped_rf.motion_state = 1;
        let points = [oncoming, stopped_rf];
        let mut entry = input(tick, &points);
        entry.speed = 0.;
        let vis = [
            VisionObservation {
                time: entry.time - 0.10,
                x: f64::from(oncoming_x),
                y: -3.5,
                kind: VehicleKind::Car,
                confidence: 0.9,
                position: true,
            },
            VisionObservation {
                time: entry.time - 0.10,
                x: 5.5,
                y: 3.5,
                kind: VehicleKind::Car,
                confidence: 0.85,
                position: true,
            },
        ];
        entry.vision = if tick >= 8 && tick % 12 == 0 { &vis } else { &[] };
        let out = engine.step(entry);
        assert!(out.slots[1].is_none(), "oncoming LF flashed at tick {tick}");
        assert!(out.slots[3].is_none(), "oncoming LR flashed at tick {tick}");
        if tick >= 25 {
            assert!(out.slots[2].is_some(), "stopped RF must stay ON at tick {tick}");
        }
    }

    // 2. Unconfirmed vision-only detections spaced 1.4 s apart (passing oncoming stream) must
    //    decay instead of stitching into vision_hits >= 3 and flashing LF.
    let mut stream_engine = MapEngine::default();
    for tick in 0..100 {
        let mut entry = input(tick, &[]);
        entry.speed = 0.;
        let obs = (tick == 5 || tick == 33 || tick == 61).then_some(VisionObservation {
            time: entry.time - 0.10,
            x: 3.8,
            y: -3.7,
            kind: VehicleKind::Car,
            confidence: 0.85,
            position: true,
        });
        entry.vision = obs.as_slice();
        let out = stream_engine.step(entry);
        assert!(out.slots[1].is_none(), "passing oncoming vision stream flashed LF at tick {tick}");
        assert!(out.slots[3].is_none(), "passing oncoming vision stream flashed LR at tick {tick}");
    }

    // 3. Repeated vision hits beyond a trusted outer lane line must not confirm a stationary echo.
    let mut outer_engine = MapEngine::default();
    let model = Model {
        lane_lines: vec![Line { x: vec![0., 30.], y: vec![-4.0; 2] }],
        lane_line_probs: vec![0.9],
        road_edges: Vec::new(),
    };
    for tick in 0..60 {
        let mut echo = radar(113, 12.0, -4.6, 0.0);
        echo.motion_state = 1;
        let points = [echo];
        let mut entry = input(tick, &points);
        entry.speed = 0.;
        entry.road = Some(RoadObservation {
            time: entry.time,
            path: &[],
            lines: Some(&model),
            road_edge_stds: [f32::INFINITY; 2],
            changing_lanes: false,
        });
        let obs = (tick == 12 || tick == 24 || tick == 36).then_some(VisionObservation {
            time: entry.time - 0.10,
            x: 12.0,
            y: -4.6,
            kind: VehicleKind::Car,
            confidence: 0.9,
            position: true,
        });
        entry.vision = obs.as_slice();
        let out = outer_engine.step(entry);
        // After the immediate 0.45 s vision window expires, stationary_vision_confirmed must stay false.
        if tick >= 50 {
            assert!(out.slots[1].is_none(), "echo beyond outer lane line stayed confirmed at tick {tick}");
        }
    }
}

#[test]
fn side_fr_min_gap_enforces_car_and_heavy_thresholds_smoothly() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(201, 4.0, 3.5, 0.)]);
    let base_front = engine.displayed[2].unwrap();
    let base_rear = EngineSlot { obj_id: base_front.obj_id + 1, s: 0., x: 0., v_s: 0., ..base_front };

    // 1. Both cars simultaneously present (R at 0 m -> d_r = 0, F at 3.5 m -> d_f = 2.8 m):
    //    F must be separated to d_f >= 6.5 m (x >= 8.125 m) while R stays at 0 m.
    let mut display = rear_display::SideDisplay::default();
    let mut prev = [None; 5];
    let mut slots = [
        None,
        None,
        Some(EngineSlot { s: 3.5, x: 3.5, kind: VehicleKind::Car, ..base_front }),
        None,
        Some(EngineSlot { s: 0., x: 0., kind: VehicleKind::Car, ..base_rear }),
    ];
    display.update(0.0, 1, false, prev, &mut slots);
    let rf = slots[2].unwrap();
    let rr = slots[4].unwrap();
    let gap_car = 0.8 * rf.x + 0.8 * (-rr.x).max(0.);
    assert!((gap_car - 6.5).abs() < 1e-6, "car+car display gap must be 6.5 m, got {gap_car}");
    assert_eq!(rr.x, 0., "rear car must keep 0 m alongside position");

    // As R recedes to -5.0 m (d_r = 4.0 m), F (physical x = 3.5 m, d_f = 2.8 m -> gap = 6.8 m >= 6.5 m)
    // smoothly glides back to its physical 3.5 m without any jump.
    prev = slots;
    let mut prev_x = rf.x;
    for i in 1..=60 {
        let mut s = [
            None,
            None,
            Some(EngineSlot { s: 3.5, x: 3.5, kind: VehicleKind::Car, ..base_front }),
            None,
            Some(EngineSlot { s: -5.0, x: -5.0, kind: VehicleKind::Car, ..base_rear }),
        ];
        display.update(i as f64 * STEP, 1, false, prev, &mut s);
        let cur_x = s[2].unwrap().x;
        assert!((cur_x - prev_x).abs() <= 0.35, "receding R must not cause front jump: {prev_x} -> {cur_x}");
        let gap = 0.8 * cur_x + 0.8 * (-s[4].unwrap().x).max(0.);
        assert!(gap >= 6.5 - 1e-6, "gap must stay >= 6.5 m while R recedes, got {gap}");
        prev_x = cur_x;
        prev = s;
    }
    assert!((prev_x - 3.5).abs() < 0.05, "F must converge to physical 3.5 m once d_r=4.0 m gives gap >= 6.5 m, got {prev_x}");

    // 2. Heavy vehicle in either F or R requires 8.5 m display gap (x >= 10.625 m when d_r = 0).
    for (f_kind, r_kind) in [
        (VehicleKind::Heavy, VehicleKind::Car),
        (VehicleKind::Car, VehicleKind::Heavy),
        (VehicleKind::Heavy, VehicleKind::Heavy),
    ] {
        let mut d = rear_display::SideDisplay::default();
        let mut s = [
            None,
            None,
            Some(EngineSlot { s: 4.0, x: 4.0, kind: f_kind, ..base_front }),
            None,
            Some(EngineSlot { s: 0., x: 0., kind: r_kind, ..base_rear }),
        ];
        d.update(0.0, 1, false, [None; 5], &mut s);
        let gap = 0.8 * s[2].unwrap().x + 0.8 * (-s[4].unwrap().x).max(0.);
        assert!((gap - 8.5).abs() < 1e-6, "heavy pair ({f_kind:?}, {r_kind:?}) gap must be 8.5 m, got {gap}");
    }

    // 3. No 1-tick jump when R suddenly appears next to an already-displayed F, or suddenly disappears.
    let mut smooth = rear_display::SideDisplay::default();
    let mut prev = [None; 5];
    // F alone at x = 3.5 m (d_f = 2.8 m) for 3 ticks.
    for i in 0..3 {
        let mut s = [None, None, Some(EngineSlot { s: 3.5, x: 3.5, kind: VehicleKind::Car, ..base_front }), None, None];
        smooth.update(i as f64 * STEP, 1, false, prev, &mut s);
        assert!((s[2].unwrap().x - 3.5).abs() < 1e-9);
        prev = s;
    }
    // R suddenly appears at x = 0 m: F must ramp smoothly (<= 1.0 m/tick map = 0.8 m/tick display) to 8.125 m.
    let mut last_x = 3.5;
    for i in 3..15 {
        let mut s = [
            None,
            None,
            Some(EngineSlot { s: 3.5, x: 3.5, kind: VehicleKind::Car, ..base_front }),
            None,
            Some(EngineSlot { s: 0., x: 0., kind: VehicleKind::Car, ..base_rear }),
        ];
        smooth.update(i as f64 * STEP, 1, false, prev, &mut s);
        let x = s[2].unwrap().x;
        let step_disp = 0.8 * (x - last_x).abs();
        assert!(step_disp <= 0.8001, "R appearance caused 1-tick spike: {step_disp} m display at tick {i}");
        assert!(x >= last_x - 1e-9, "F must ramp monotonically upward");
        last_x = x;
        prev = s;
    }
    assert!((0.8 * last_x - 6.5).abs() < 1e-6, "F must reach 6.5 m display, got {}", 0.8 * last_x);

    // R suddenly disappears: F must glide smoothly back to 3.5 m without 1-tick drop.
    for i in 15..65 {
        let mut s = [None, None, Some(EngineSlot { s: 3.5, x: 3.5, kind: VehicleKind::Car, ..base_front }), None, None];
        smooth.update(i as f64 * STEP, 1, false, prev, &mut s);
        let x = s[2].unwrap().x;
        let step_disp = 0.8 * (x - last_x).abs();
        assert!(step_disp <= 0.25, "R disappearance caused 1-tick drop: {step_disp} m display at tick {i}");
        assert!(x <= last_x + 1e-9, "F must return monotonically");
        last_x = x;
        prev = s;
    }
    assert!((last_x - 3.5).abs() < 0.05, "F must settle back to physical 3.5 m, got {last_x}");

    // 4. Single-vehicle origin crossing (RearPass & FrontJoin) must NOT be interfered with.
    let mut single = rear_display::SideDisplay::default();
    let mut prev = [None; 5];
    let mut s0 = [None, None, Some(EngineSlot { s: 0.2, x: 0.2, v_s: -3.0, ..base_front }), None, None];
    single.update(0.0, 1, false, prev, &mut s0);
    assert!((s0[2].unwrap().x - 0.2).abs() < 1e-9);
    prev = s0;
    let mut s1 = [None, None, None, None, Some(EngineSlot { s: -0.1, x: -0.1, v_s: -3.0, ..base_front })];
    single.update(STEP, 1, false, prev, &mut s1);
    assert!(s1[2].is_none());
    assert_eq!(s1[4].unwrap().x, 0.0, "single-car RearPass origin tick must remain exact 0.0 m");

    // 5. Fast-passing F approaching 0 m while R is already receding behind ego:
    //    F keeps its physical trajectory to 0 m (no handoff jump) while R opens the gap rearward.
    let mut passing = rear_display::SideDisplay::default();
    let mut prev = [None; 5];
    let mut last_rx = -0.2;
    for i in 0..10 {
        let fx = 3.2 - 0.32 * i as f64;
        let rx = -0.2 - 0.30 * i as f64;
        let mut s = [
            None,
            None,
            Some(EngineSlot { s: fx, x: fx, v_s: -6.4, kind: VehicleKind::Car, ..base_front }),
            None,
            Some(EngineSlot { s: rx, x: rx, v_s: -6.0, kind: VehicleKind::Car, ..base_rear }),
        ];
        passing.update(i as f64 * STEP, 1, false, prev, &mut s);
        assert!((s[2].unwrap().x - fx).abs() < 1e-9, "fast-passing F must not be held away from 0 m");
        let cur_rx = s[4].unwrap().x;
        assert!(cur_rx <= rx + 1e-9, "receding R must open gap rearward");
        if i > 0 {
            assert!(0.8 * (last_rx - cur_rx).abs() <= 0.85, "R rearward ramp must stay smooth");
        }
        last_rx = cur_rx;
        prev = s;
    }
}

#[test]
fn bsd_hold_hands_off_near_overtake_despite_distant_lf_occupant() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(210, 42.0, -3.5, 1.0)]);
    let far_front = engine.displayed[1].unwrap();

    for cand_s in [4.7_f64, 7.5_f64] {
        let mut display = rear_display::SideDisplay::default();
        let mut prev = [None; 5];

        // 1. BSD ON (0..3.0 s) while a distant car occupies LF at 42 m.
        for i in 0..60 {
            let mut slots = [
                None,
                Some(EngineSlot { s: 42.0, x: 42.0, v_s: 1.0, ..far_front }),
                None,
                None,
                None,
            ];
            display.update(i as f64 * STEP, 0, true, prev, &mut slots);
            prev = slots;
        }

        // 2. BSD OFF for +1.5 s (ticks 60..90, inside 3.0 s hold) with distant LF still at 42 m.
        for i in 60..90 {
            let mut slots = [
                None,
                Some(EngineSlot { s: 42.0, x: 42.0, v_s: 1.0, ..far_front }),
                None,
                None,
                None,
            ];
            display.update(i as f64 * STEP, 0, false, prev, &mut slots);
            assert_eq!(slots[3].unwrap().obj_id, 0);
            assert!((slots[3].unwrap().x - (-10.)).abs() < 0.1);
            prev = slots;
        }

        // 3. Near overtaking vehicle (s = 4.7 m or 7.5 m) supersedes distant LF during hold:
        //    BsdOnly must clear immediately and hand off 1 vehicle (-10 m -> 0 m -> LF).
        let near_vid = far_front.obj_id + 100;
        let mut join_xs = Vec::new();
        let mut saw_origin = false;
        let mut saw_lf = false;
        for i in 90..120 {
            let x = cand_s + (i - 90) as f64 * 0.15;
            let cand = EngineSlot {
                obj_id: near_vid,
                s: x,
                x,
                v_s: 3.0,
                radar_seen: true,
                ..far_front
            };
            let mut slots = [None, Some(cand), None, None, None];
            display.update(i as f64 * STEP, 0, false, prev, &mut slots);
            // Zero duplicate ticks (never both LF and LR at once) and BsdOnly immediately gone.
            assert!(
                slots[1].is_none() || slots[3].is_none(),
                "cand_s={cand_s} tick {i}: duplicate LF+LR during handoff",
            );
            assert!(
                slots[3].is_none_or(|r| r.obj_id != 0),
                "cand_s={cand_s} tick {i}: BsdOnly must clear immediately on handoff",
            );
            let active = slots[1].or(slots[3]).expect("single vehicle must stay visible");
            assert_eq!(active.obj_id, near_vid);
            if active.x == 0. {
                saw_origin = true;
            }
            if slots[1].is_some() {
                saw_lf = true;
            }
            join_xs.push(active.x);
            prev = slots;
        }
        assert!(
            (join_xs[0] - (-10.)).abs() < 0.1,
            "cand_s={cand_s}: FrontJoin must start from held 8 m (-10 m map), got {}",
            join_xs[0],
        );
        assert!(saw_origin, "cand_s={cand_s}: FrontJoin must cross exact 0 m");
        assert!(saw_lf, "cand_s={cand_s}: FrontJoin must complete into LF");
        assert!(
            join_xs.windows(2).all(|w| w[1] >= w[0] - 1e-9),
            "cand_s={cand_s}: handoff must advance monotonically without retreat",
        );
    }
}

#[test]
fn bsd_retreating_after_hold_does_not_jump_forward_on_new_lf() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(211, 4.5, -3.5, 3.0)]);
    let cand_base = engine.displayed[1].unwrap();

    let mut display = rear_display::SideDisplay::default();
    let mut prev = [None; 5];

    // 1. BSD ON (0..3.0 s): BsdOnly reaches -10 m (8 m display).
    for i in 0..60 {
        let mut slots = [None; 5];
        display.update(i as f64 * STEP, 0, true, prev, &mut slots);
        prev = slots;
    }

    // 2. BSD OFF for +3.5 s (ticks 60..130): 3.0 s hold finishes and LR starts retreating behind -10 m.
    for i in 60..130 {
        let mut slots = [None; 5];
        display.update(i as f64 * STEP, 0, false, prev, &mut slots);
        prev = slots;
    }
    let retreat_start_x = prev[3].expect("LR must still be retreating at +3.5 s").x;
    assert!(
        retreat_start_x < -10.5,
        "LR must already be retreating behind -10 m at +3.5 s, got {retreat_start_x}",
    );

    // 3. New close LF appears while LR is retreating: from_bsd must NOT fire.
    //    LF appears directly in front; LR keeps retreating monotonically until it clears.
    let mut rear_xs = vec![retreat_start_x];
    let mut rear_cleared = false;
    for i in 130..240 {
        let x = 4.5 + (i - 130) as f64 * 0.15;
        let cand = EngineSlot {
            obj_id: cand_base.obj_id + 100,
            s: x,
            x,
            v_s: 3.0,
            radar_seen: true,
            ..cand_base
        };
        let mut slots = [None, Some(cand), None, None, None];
        display.update(i as f64 * STEP, 0, false, prev, &mut slots);
        let lf = slots[1].expect("LF must appear directly without being pulled into FrontJoin");
        assert_eq!(lf.obj_id, cand.obj_id);
        assert!(lf.x >= x - 1e-6, "LF must not be pulled back to 0 m: {} vs {x}", lf.x);
        if let Some(lr) = slots[3] {
            assert_eq!(lr.obj_id, 0, "LR must remain retreating BsdOnly, not FrontJoin");
            assert!(
                lr.x <= *rear_xs.last().unwrap() + 1e-9,
                "retreating LR must never jump forward at tick {i}: {} -> {}",
                rear_xs.last().unwrap(),
                lr.x,
            );
            rear_xs.push(lr.x);
        } else {
            rear_cleared = true;
        }
        prev = slots;
    }
    assert!(rear_cleared, "retreating LR must finish retreating and clear");
}

#[test]
fn consecutive_bsd_overtakes_under_continuous_blind_both_hand_off() {
    let mut engine = MapEngine::default();
    confirm(&mut engine, &[radar(212, 5.0, -3.5, 3.0)]);
    let base = engine.displayed[1].unwrap();
    let mut display = rear_display::SideDisplay::default();
    let mut prev = [None; 5];

    // Car 1 enters BSD (ticks 0..40) and hands off while blind stays continuously true for Car 2.
    for i in 0..40 {
        let mut slots = [None; 5];
        display.update(i as f64 * STEP, 0, true, prev, &mut slots);
        prev = slots;
    }
    let car1_vid = base.obj_id + 10;
    for i in 40..65 {
        let x = 5.0 + (i - 40) as f64 * 0.25;
        let car1 = EngineSlot { obj_id: car1_vid, s: x, x, v_s: 5.0, radar_seen: true, ..base };
        let mut slots = [None, Some(car1), None, None, None];
        display.update(i as f64 * STEP, 0, true, prev, &mut slots);
        prev = slots;
    }
    // Car 1 is now far ahead (> 20 m); blind stayed true continuously for Car 2, then turns OFF.
    for i in 65..125 {
        let x = 22.0 + (i - 65) as f64 * 0.2;
        let car1 = EngineSlot { obj_id: car1_vid, s: x, x, v_s: 4.0, radar_seen: true, ..base };
        let mut slots = [None, Some(car1), None, None, None];
        display.update(i as f64 * STEP, 0, i < 105, prev, &mut slots);
        prev = slots;
    }
    assert_eq!(prev[3].unwrap().obj_id, 0, "Car 2 BsdOnly must be holding at -10 m");

    // Car 2 emerges at s = 4.9 m during 3 s hold: must hand off cleanly instead of being dropped.
    let car2_vid = base.obj_id + 20;
    let mut saw_rear_join = false;
    let mut saw_origin = false;
    for i in 125..150 {
        let x = 4.9 + (i - 125) as f64 * 0.15;
        let car2 = EngineSlot { obj_id: car2_vid, s: x, x, v_s: 3.0, radar_seen: true, ..base };
        let mut slots = [None, Some(car2), None, None, None];
        display.update(i as f64 * STEP, 0, false, prev, &mut slots);
        assert!(slots[1].is_none() || slots[3].is_none(), "Car 2 handoff must have 0 duplicate ticks");
        if let Some(r) = slots[3] {
            assert_eq!(r.obj_id, car2_vid);
            saw_rear_join = true;
        }
        if slots[1].is_some_and(|f| f.x == 0.) {
            saw_origin = true;
        }
        prev = slots;
    }
    assert!(saw_rear_join && saw_origin, "Car 2 must complete BsdOnly -> FrontJoin -> 0 m -> LF");
}
