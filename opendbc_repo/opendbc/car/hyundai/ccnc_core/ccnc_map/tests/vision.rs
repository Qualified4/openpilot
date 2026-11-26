//! The wide camera's side cars on the map slots (`ccnc_map::vision`).

use ccnc_map::slots::{Slot, VISION_CONFIRMS, VISION_NONE, VISION_ONLY};
use ccnc_map::vision::{Box, Context, Frame, FrameView, Geometry, Side, Skipped, SlotState, Stepped, Track, Vision, HOLD_S, LATE_S, RAW_HOLD_S, SIDE_LANE_PLAY, SWAP_HOLD_S, VISION_VID};
use ccnc_map::pos::common::LANE_WIDTH;
use ccnc_map::world::PoseAt;
use det_rules::rules::SWAP_MARGIN;

const FIXTURE: &str = include_str!("fixtures/wide_rules_c1.txt");

/// Slot (F or R), on, action, and the candidate's x, yc, cut and view.
type Expect = (String, bool, String, Option<(f64, f64, bool, String)>);

struct FixtureFrame {
    t_ns: u64,
    v: f64,
    guard: bool,
    lprob: Option<Vec<f64>>,
    slots: [[SlotState; 2]; 2],
    tracks: Vec<Track>,
    boxes: Vec<(String, Box)>,
    expect: Vec<Expect>,
}

struct Scene {
    index: usize,
    side: Side,
    calibration: [f32; 7],
    frames: Vec<FixtureFrame>,
}

fn num(s: &str) -> f64 {
    s.parse().unwrap()
}

fn scenes() -> Vec<Scene> {
    let mut out: Vec<Scene> = Vec::new();
    for line in FIXTURE.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "scene" => out.push(Scene {
                index: f[1].parse().unwrap(),
                side: if f[3] == "L" { Side::Left } else { Side::Right },
                calibration: std::array::from_fn(|k| num(f[4 + k]) as f32),
                frames: Vec::new(),
            }),
            "frame" => {
                let (lprob, rest) = if f[4] == "-" { (None, &f[5..]) } else { (Some(f[4..8].iter().map(|s| num(s)).collect()), &f[8..]) };
                let slot = |k: usize| SlotState { detect: num(rest[2 * k]) > 0., dist: num(rest[2 * k + 1]) };
                out.last_mut().unwrap().frames.push(FixtureFrame {
                    t_ns: f[1].parse().unwrap(),
                    v: num(f[2]),
                    guard: f[3] == "1",
                    lprob,
                    slots: [[slot(0), slot(1)], [slot(2), slot(3)]],
                    tracks: Vec::new(),
                    boxes: Vec::new(),
                    expect: Vec::new(),
                });
            }
            "track" => out.last_mut().unwrap().frames.last_mut().unwrap().tracks.push(Track {
                id: num(f[1]) as i64,
                d_rel: num(f[2]),
                y_rel: num(f[3]),
                v_rel: num(f[4]),
            }),
            "box" => out.last_mut().unwrap().frames.last_mut().unwrap().boxes.push((
                f[1].to_string(),
                Box { xyxy: std::array::from_fn(|k| num(f[2 + k])), confidence: num(f[6]), class_id: f[7].parse().unwrap() },
            )),
            "expect" => {
                let cand = (f[4] != "-").then(|| (num(f[4]), num(f[5]), f[6] == "1", f[7].to_string()));
                out.last_mut().unwrap().frames.last_mut().unwrap().expect.push((f[1].to_string(), f[2] == "1", f[3].to_string(), cand));
            }
            other => panic!("unknown fixture line {other}"),
        }
    }
    out
}

fn frame_of(scene: &Scene, fr: &FixtureFrame) -> Frame {
    let view = if scene.side == Side::Left { "vL" } else { "vR" };
    Frame {
        time_ns: fr.t_ns,
        view: FrameView::Side(scene.side),
        calibration: scene.calibration,
        boxes: fr.boxes.iter().filter(|(v, _)| v == view).map(|(_, b)| *b).collect(),
    }
}

fn raw_frame_of(scene: &Scene, fr: &FixtureFrame, time_ns: u64) -> Frame {
    Frame {
        time_ns,
        view: FrameView::Raw,
        calibration: scene.calibration,
        boxes: fr.boxes.iter().filter(|(v, _)| v == "raw").map(|(_, b)| *b).collect(),
    }
}

fn step_raw(vision: &mut Vision, frame: &Frame, arrival: f64, fr: &FixtureFrame) -> Result<Option<Stepped>, Skipped> {
    let context = Context { v_ego: fr.v, tracks: &fr.tracks, lprob: fr.lprob.as_deref(), slots: fr.slots, guard: fr.guard };
    match vision.frame(frame, arrival, &context, PoseAt::default()) {
        Err(Skipped::NeedsGeometry(calib)) => {
            vision.set_geometry(Geometry::build(&calib));
            vision.frame(frame, arrival, &context, PoseAt::default())
        }
        other => other,
    }
}

fn step(vision: &mut Vision, frame: &Frame, arrival: f64, fr: &FixtureFrame) -> Result<Stepped, Skipped> {
    step_raw(vision, frame, arrival, fr).map(|opt| opt.expect("side frame returns Stepped"))
}

fn seconds(t_ns: u64) -> f64 {
    t_ns as f64 * 1e-9
}

#[test]
fn wide_rules_match_pipeline_fixtures() {
    let scenes = scenes();
    assert_eq!(scenes.len(), 58);
    let (mut frames, mut values, mut on) = (0, 0, 0);
    let mut mismatches = Vec::new();
    for scene in &scenes {
        let mut vision = Vision::default();
        for fr in &scene.frames {
            let frame = frame_of(scene, fr);
            let stepped = step(&mut vision, &frame, seconds(fr.t_ns) + 0.5, fr).unwrap();
            frames += 1;
            for (slot, want_on, want_act, want_cand) in &fr.expect {
                let out = if slot == "F" { &stepped.out.front } else { &stepped.out.rear };
                let act = out.action.map_or("-", |a| a.name());
                let cand = out.cand.map(|i| {
                    let c = &stepped.cands[i];
                    (c.x, c.yc, c.cut, c.view.name().to_string())
                });
                values += 4;
                on += usize::from(*want_on && slot == "F");
                if out.display_on != *want_on || act != want_act || &cand != want_cand {
                    mismatches.push(format!(
                        "scene {} t {} {slot}: on {} act {act} cand {cand:?}, pipeline.py on {want_on} act {want_act} cand {want_cand:?}",
                        scene.index, fr.t_ns, out.display_on
                    ));
                }
            }
        }
    }
    assert_eq!(frames, 503);
    assert!(on > 100, "the fixture should show cars, {on} front slots on");
    assert!(mismatches.is_empty(), "{} of {values} values differ:\n{}", mismatches.len(), mismatches.join("\n"));
}

/// The first frame of a fixture scene where its front slot is on, and the vision state right after it.
fn first_on(index: usize) -> (Vision, Scene, usize, f64, f64) {
    let scene = scenes().into_iter().find(|s| s.index == index).unwrap();
    let mut vision = Vision::default();
    for (k, fr) in scene.frames.iter().enumerate() {
        let stepped = step(&mut vision, &frame_of(&scene, fr), seconds(fr.t_ns) + 0.5, fr).unwrap();
        if stepped.out.front.display_on && stepped.out.front.action.is_some() {
            let c = &stepped.cands[stepped.out.front.cand.unwrap()];
            let (x, yc) = (c.x, c.yc);
            return (vision, scene, k, x, yc);
        }
    }
    panic!("scene {index} never shows its front car");
}

fn on_scene() -> usize {
    // The first scene whose front slot fills an empty radar slot.
    let scenes = scenes();
    scenes
        .iter()
        .find(|s| s.frames.iter().any(|f| f.expect.iter().any(|e| e.0 == "F" && e.2 == "fill")))
        .unwrap()
        .index
}

fn radar_slot(distance: f32, lateral: f32) -> Slot {
    Slot { vid: 7, distance, lateral, detect: 1, lane_coord_rel: 0., v_rel: 1., kind: 0, kind_confirmed: false, kind_wanted: false, vision: VISION_NONE }
}

#[test]
fn late_result_over_1_5s_dropped() {
    let (mut vision, scene, k, _, _) = first_on(on_scene());
    let fr = &scene.frames[k + 1.min(scene.frames.len() - 1 - k)];
    let frame = frame_of(&scene, fr);
    let t = seconds(fr.t_ns);
    assert!(matches!(step(&mut vision, &frame, t + LATE_S + 0.01, fr), Err(Skipped::Late)));
    assert!(matches!(step(&mut vision, &frame, t - 0.01, fr), Err(Skipped::Late)), "a result before its frame");
    assert!(step(&mut vision, &frame, t + LATE_S, fr).is_ok());
}

#[test]
fn vision_cars_leave_two_seconds_after_the_last_result() {
    let (mut vision, scene, k, _, _) = first_on(on_scene());
    let arrival = seconds(scene.frames[k].t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    for (now, shown) in [(arrival, true), (arrival + HOLD_S, true), (arrival + HOLD_S + 0.01, false)] {
        let mut slots = [None; 5];
        vision.apply(now, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
        assert_eq!(slots[index].is_some(), shown, "{now}");
        if shown {
            assert_eq!(slots[index].unwrap().vision, VISION_ONLY);
            assert_eq!(slots[index].unwrap().vid, VISION_VID[index - 1]);
        }
    }
    // Moving off or reversing ends it at once; a standing car's slightly negative speed does not.
    for (v_ego, shown) in [(-0.0011, true), (3.01, false), (-0.3, false)] {
        let mut slots = [None; 5];
        vision.apply(arrival, v_ego, PoseAt::default(), [false; 5], [false; 2], &mut slots);
        assert_eq!(slots[index].is_some(), shown, "{v_ego}");
    }
}

/// A camera car on fresh state, the time its result is held at, its slot and its place (x, y left positive).
fn camera_car() -> (Vision, f64, usize, f32, f32) {
    let (vision, scene, k, x, yc) = first_on(on_scene());
    (vision, seconds(scene.frames[k].t_ns) + 0.5, 1 + scene.side as usize, x as f32, -yc as f32)
}

fn run(vision: &mut Vision, now: f64, radar: [Option<Slot>; 5]) -> [Option<Slot>; 5] {
    let mut slots = radar;
    vision.apply(now, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
    slots
}

#[test]
fn vision_only_car_merges_with_radar() {
    let (mut vision, now, index, x, y) = camera_car();
    // The radar draws the same car in FF, inside the gate (6 m along, 1 m across): it stays the radar's.
    let mut ff = [None; 5];
    ff[0] = Some(radar_slot(x + 5.9, y - 0.9 * y.signum()));
    let merged = run(&mut vision, now, ff);
    assert_eq!(merged[0], ff[0]);
    assert!(merged[index].is_none());
    // Outside the gate it is another car: the camera fills the side slot.
    for (dx, dy) in [(6.1, 0.), (0., 1.1)] {
        let mut other = [None; 5];
        other[0] = Some(radar_slot(x + dx, y - dy * y.signum()));
        let filled = run(&mut vision, now, other);
        assert_eq!(filled[index].map(|s| s.vision), Some(VISION_ONLY), "{dx} {dy}");
        assert_eq!(filled[0], other[0]);
    }
}

#[test]
fn a_radar_car_in_the_same_slot_within_a_car_length_is_the_camera_car() {
    // 3.1 m farther (past the swap margin) but within 4 m along and 1.5 m across: the same car, the radar's stays.
    for (dx, dy) in [(3.1, 0.), (3.9, 1.4)] {
        let (mut vision, now, index, x, y) = camera_car();
        let mut same = [None; 5];
        same[index] = Some(radar_slot(x + dx, y - dy * y.signum()));
        for dt in [0., 0.3, 0.6, 1.0] {
            assert_eq!(run(&mut vision, now + dt, same)[index].map(|s| (s.vid, s.vision)), Some((7, VISION_CONFIRMS)), "{dx} {dy} {dt}");
        }
    }
}

#[test]
fn the_slot_swaps_to_a_closer_camera_car_only_once_the_rules_held_the_swap() {
    let (mut vision, now, index, x, y) = camera_car();
    let camera = (VISION_VID[index - 1], VISION_ONLY);
    let radar = (7, VISION_CONFIRMS);
    let mut far = [None; 5];
    far[index] = Some(radar_slot(x + 4.1, y));
    let mut near = [None; 5];
    near[index] = Some(radar_slot(x + 1.0, y));
    let shown = |vision: &mut Vision, now: f64, slots| run(vision, now, slots)[index].map(|s| (s.vid, s.vision));
    // A farther radar car: kept until the swap held SWAP_HOLD_S.
    assert_eq!(shown(&mut vision, now, far), Some(radar));
    assert_eq!(shown(&mut vision, now + SWAP_HOLD_S - 0.05, far), Some(radar));
    assert_eq!(shown(&mut vision, now + SWAP_HOLD_S, far), Some(camera));
    // Once shown, the camera car stays while the radar car is no farther, or the same car (0000047f standing)...
    let mut closer = [None; 5];
    closer[index] = Some(radar_slot(x - SWAP_MARGIN as f32 - 0.1, y));
    for dt in [0.05, 0.6, 1.2] {
        assert_eq!(shown(&mut vision, now + SWAP_HOLD_S + dt, near), Some(camera), "{dt}");
    }
    // ...and goes back only to a radar car clearly closer, once that held, a frame in between not counting.
    let back = now + SWAP_HOLD_S + 1.3;
    assert_eq!(shown(&mut vision, back, closer), Some(camera));
    assert_eq!(shown(&mut vision, back + 0.05, near), Some(camera));
    assert_eq!(shown(&mut vision, back + 0.1, closer), Some(camera));
    assert_eq!(shown(&mut vision, back + 0.1 + SWAP_HOLD_S - 0.05, closer), Some(camera));
    assert_eq!(shown(&mut vision, back + 0.1 + SWAP_HOLD_S, closer), Some(radar));
    // An empty slot takes the camera car at once.
    assert_eq!(shown(&mut vision, back + 0.1 + SWAP_HOLD_S + 0.05, [None; 5]), Some(camera));
}

#[test]
fn a_protected_radar_car_stays_but_does_not_take_back_a_shown_camera_car() {
    let (mut vision, now, index, x, y) = camera_car();
    let camera = (VISION_VID[index - 1], VISION_ONLY);
    let mut far = [None; 5];
    far[index] = Some(radar_slot(x + 4.1, y));
    let shown = |vision: &mut Vision, now: f64, protect: bool| {
        let mut slots = far;
        let mut protected = [false; 5];
        protected[index] = protect;
        vision.apply(now, 0., PoseAt::default(), protected, [false; 2], &mut slots);
        slots[index].map(|s| (s.vid, s.vision))
    };
    // The stop memory's or a side pair's car is not taken over, however long the camera asks; the camera confirms it.
    for dt in [0., 0.6, 1.2] {
        assert_eq!(shown(&mut vision, now + dt, true), Some((7, VISION_CONFIRMS)), "{dt}");
    }
    // Once it is no longer protected the camera's hold starts over.
    assert_eq!(shown(&mut vision, now + 1.25, false), Some((7, VISION_CONFIRMS)));
    assert_eq!(shown(&mut vision, now + 1.25 + SWAP_HOLD_S - 0.05, false), Some((7, VISION_CONFIRMS)));
    assert_eq!(shown(&mut vision, now + 1.25 + SWAP_HOLD_S, false), Some(camera));
    // The protection coming back does not hand the slot back to the farther radar car.
    for dt in [0.05, 0.6, 1.2] {
        assert_eq!(shown(&mut vision, now + 1.25 + SWAP_HOLD_S + dt, true), Some(camera), "{dt}");
    }
}

#[test]
fn a_camera_only_car_is_drawn_near_the_next_lane_centre() {
    let (mut vision, now, index, _, _) = camera_car();
    let lateral = f64::from(run(&mut vision, now, [None; 5])[index].unwrap().lateral);
    let outward = if index % 2 == 1 { lateral } else { -lateral };
    assert!((LANE_WIDTH - SIDE_LANE_PLAY - 1e-6..=LANE_WIDTH + SIDE_LANE_PLAY + 1e-6).contains(&outward), "{lateral}");
}

#[test]
fn camera_car_follows_the_ego_motion_since_its_frame() {
    let (mut vision, scene, k, x, _) = first_on(on_scene());
    let now = seconds(scene.frames[k].t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    let mut slots = [None; 5];
    // 1 m forward since the frame: the car is 1 m closer.
    vision.apply(now, 0., PoseAt { x: 1., ..PoseAt::default() }, [false; 5], [false; 2], &mut slots);
    assert!((f64::from(slots[index].unwrap().distance) - (x - 1.)).abs() < 1e-5);
}

#[test]
fn raw_frame_combines_with_side_frame_and_expires_after_raw_hold() {
    let scene = scenes().into_iter().find(|s| s.index == 25).expect("scene 25");
    // Scene 25 has a front car in raw and a rear car in vR; feeding fresh raw boxes before vR enables RR.
    let mut vision = Vision::default();
    let mut rear_on = 0;
    for fr in &scene.frames {
        let raw_t = fr.t_ns.saturating_sub(600_000_000);
        let raw = raw_frame_of(&scene, fr, raw_t);
        if !raw.boxes.is_empty() {
            assert!(step_raw(&mut vision, &raw, seconds(raw_t) + 0.7, fr).unwrap().is_none());
        }
        let stepped = step(&mut vision, &frame_of(&scene, fr), seconds(fr.t_ns) + 0.6, fr).unwrap();
        if stepped.out.rear.display_on {
            rear_on += 1;
        }
    }
    if scene.frames.iter().any(|fr| fr.boxes.iter().any(|(v, _)| v == "raw")) {
        assert!(rear_on > 0, "fresh raw + side should enable rear slot on scene 25");
    }

    // Stale raw (> RAW_HOLD_S older than the side frame) is ignored, and an intervening raw extends hold past HOLD_S
    // without keeping the car alive indefinitely.
    let (mut vision, scene, k, _, _) = first_on(on_scene());
    let fr = &scene.frames[k];
    let arrival = seconds(fr.t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    let raw_t = fr.t_ns + 2_000_000_000;
    let raw_arrival = seconds(raw_t) + 1.0; // arrival + 2.5 s
    let raw = raw_frame_of(&scene, fr, raw_t);
    assert!(step_raw(&mut vision, &raw, raw_arrival, fr).unwrap().is_none());
    let mut slots = [None; 5];
    vision.apply(arrival + HOLD_S + 0.1, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
    assert!(slots[index].is_some(), "intervening raw bridges a slow 3-view cycle");
    let mut expired = [None; 5];
    vision.apply(raw_arrival + 1.55, 0., PoseAt::default(), [false; 5], [false; 2], &mut expired);
    assert!(expired[index].is_none(), "car still leaves promptly when visiond stops after raw");
    // Raw older than RAW_HOLD_S is not combined into a later side frame.
    let mut later = frame_of(&scene, fr);
    later.time_ns = raw_t + ((RAW_HOLD_S + 0.2) * 1e9) as u64;
    let stepped = step(&mut vision, &later, seconds(later.time_ns) + 0.5, fr).unwrap();
    assert!(stepped.cands.iter().all(|c| c.view != det_rules::View::Raw));
}

#[test]
fn slow_delayed_result_within_late_s_compensates_ego_motion_and_cuts_off_above_3ms() {
    // A little-core result arriving 2.5 s after capture (< LATE_S = 3.0 s) is accepted, placed in world coordinates
    // using the pose from frame capture time, and shifted by ego motion up to `now`; speeding above 3 m/s clears it.
    let (mut vision, scene, k, _, _) = first_on(on_scene());
    let fr = &scene.frames[(k + 1).min(scene.frames.len() - 1)];
    let frame = frame_of(&scene, fr);
    let t = seconds(fr.t_ns);
    let capture_pose = PoseAt { t, x: 10.0, y: 0.0, heading: 0.0, s: 10.0 };
    let context = Context { v_ego: fr.v, tracks: &fr.tracks, lprob: fr.lprob.as_deref(), slots: fr.slots, guard: fr.guard };
    let stepped = vision.frame(&frame, t + 2.5, &context, capture_pose).unwrap().unwrap();
    let cand = &stepped.cands[stepped.out.front.cand.unwrap()];
    let side_idx = 1 + scene.side as usize;

    // Ego crept forward 2.0 m between capture and display time: the camera car moves 2.0 m closer.
    let mut slots = [None; 5];
    vision.apply(t + 2.5, 1.0, PoseAt { t: t + 2.5, x: 12.0, y: 0.0, heading: 0.0, s: 12.0 }, [false; 5], [false; 2], &mut slots);
    assert_eq!(slots[side_idx].map(|s| s.vision), Some(VISION_ONLY));
    assert!((f64::from(slots[side_idx].unwrap().distance) - (cand.x - 2.0)).abs() < 1e-4);

    // As soon as ego exceeds 3.0 m/s, the held slot disappears immediately even well inside HOLD_S.
    let mut fast = [None; 5];
    vision.apply(t + 2.6, 3.05, PoseAt { t: t + 2.6, x: 12.3, y: 0.0, heading: 0.0, s: 12.3 }, [false; 5], [false; 2], &mut fast);
    assert!(fast[side_idx].is_none());
}

#[test]
fn starved_vision_over_1_5s_clears_camera_slot_and_keeps_radar_slots() {
    // Under SCHED_IDLE starvation, vision results arrive > LATE_S (1.5 s) after their frame: mapd drops them,
    // the camera-only side slot disappears quietly after HOLD_S, and radar slots (driving/display) stay intact.
    let (mut vision, scene, k, _, _) = first_on(on_scene());
    let fr = &scene.frames[k];
    let last_fresh_arrival = seconds(fr.t_ns) + 0.5;
    let side_idx = 1 + scene.side as usize;
    let other_idx = if side_idx == 1 { 2 } else { 1 };
    let mut radar_only = [None; 5];
    radar_only[0] = Some(radar_slot(22.0, 0.0));
    radar_only[other_idx] = Some(radar_slot(14.0, if other_idx == 1 { 3.2 } else { -3.2 }));

    // Initially the camera fills its side slot without disturbing radar slots FF or the opposite side.
    let mut slots = radar_only;
    vision.apply(last_fresh_arrival, 0., PoseAt::default(), [false; 5], [false; 2], &mut slots);
    assert_eq!(slots[side_idx].map(|s| s.vision), Some(VISION_ONLY));
    assert_eq!(slots[0], radar_only[0]);
    assert_eq!(slots[other_idx], radar_only[other_idx]);

    // Starved side and raw frames (> 1.5 s old) are dropped and do NOT refresh or extend hold.
    let next_fr = &scene.frames[(k + 1).min(scene.frames.len() - 1)];
    let starved_side = frame_of(&scene, next_fr);
    let starved_raw = raw_frame_of(&scene, next_fr, next_fr.t_ns);
    let late_arrival = seconds(next_fr.t_ns) + LATE_S + 0.15;
    assert_eq!(step_raw(&mut vision, &starved_side, late_arrival, next_fr).unwrap_err(), Skipped::Late);
    assert_eq!(step_raw(&mut vision, &starved_raw, late_arrival + 0.1, next_fr).unwrap_err(), Skipped::Late);

    // Once HOLD_S has passed since the last fresh arrival, the camera slot vanishes silently while radar stays exact.
    let mut after_starve = radar_only;
    vision.apply(last_fresh_arrival + HOLD_S + 0.05, 0., PoseAt::default(), [false; 5], [false; 2], &mut after_starve);
    assert!(after_starve[side_idx].is_none(), "starved camera slot must disappear silently");
    assert_eq!(after_starve[0], radar_only[0]);
    assert_eq!(after_starve[other_idx], radar_only[other_idx]);

    // Recovery: a subsequent fresh frame (< 1.5 s) is accepted again without error.
    let fresh_arrival = seconds(next_fr.t_ns) + 0.4;
    assert!(step_raw(&mut vision, &starved_side, fresh_arrival, next_fr).is_ok());
}
