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

fn confirm(engine: &mut MapEngine, points: &[RawPoint]) -> EngineOutput {
    let mut out = EngineOutput::default();
    for tick in 0..8 {
        out = engine.step(input(tick, points));
    }
    out
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
    for tick in 0..62 {
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
        if (56..61).contains(&tick) {
            let rear = out.slots[4].unwrap();
            assert!(rear.s < 0.);
            assert_eq!(rear.last_track_id, Some(40));
        } else if tick == 61 {
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
        front |= out.slots[2].is_some();
        rear |= out.slots[4].is_some();
        if let Some(s) = out.slots[2].or(out.slots[4]) {
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
                confirmed_track: Some(40),
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
            confirmed_track: Some(40),
        },
    ));
    assert!(MapEngine::default().step(entry).objects.is_empty());
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
