//! The wide camera's side cars on the map slots (`ccnc_map::vision`).

use ccnc_map::slots::{Slot, VISION_CONFIRMS, VISION_NONE, VISION_ONLY};
use ccnc_map::vision::{Box, Context, Frame, Geometry, Side, Skipped, SlotState, Stepped, Track, Vision, HOLD_S, LATE_S, VISION_VID};
use ccnc_map::world::PoseAt;

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
        side: scene.side,
        calibration: scene.calibration,
        boxes: fr.boxes.iter().filter(|(v, _)| v == view).map(|(_, b)| *b).collect(),
    }
}

fn step(vision: &mut Vision, frame: &Frame, arrival: f64, fr: &FixtureFrame) -> Result<Stepped, Skipped> {
    let context = Context { v_ego: fr.v, tracks: &fr.tracks, lprob: fr.lprob.as_deref(), slots: fr.slots, guard: fr.guard };
    match vision.frame(frame, arrival, &context, PoseAt::default()) {
        Err(Skipped::NeedsGeometry(calib)) => {
            vision.set_geometry(Geometry::build(&calib));
            vision.frame(frame, arrival, &context, PoseAt::default())
        }
        other => other,
    }
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
    Slot { vid: 7, distance, lateral, detect: 1, lane_coord_rel: 0., v_rel: 0., kind: 0, kind_confirmed: false, vision: VISION_NONE }
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
    let (vision, scene, k, _, _) = first_on(on_scene());
    let arrival = seconds(scene.frames[k].t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    for (now, shown) in [(arrival, true), (arrival + HOLD_S, true), (arrival + HOLD_S + 0.01, false)] {
        let mut slots = [None; 5];
        vision.apply(now, 0., PoseAt::default(), &mut slots);
        assert_eq!(slots[index].is_some(), shown, "{now}");
        if shown {
            assert_eq!(slots[index].unwrap().vision, VISION_ONLY);
            assert_eq!(slots[index].unwrap().vid, VISION_VID[index - 1]);
        }
    }
    // Moving off or reversing ends it at once; a standing car's slightly negative speed does not.
    for (v_ego, shown) in [(-0.0011, true), (3.01, false), (-0.3, false)] {
        let mut slots = [None; 5];
        vision.apply(arrival, v_ego, PoseAt::default(), &mut slots);
        assert_eq!(slots[index].is_some(), shown, "{v_ego}");
    }
}

#[test]
fn vision_only_car_merges_with_radar() {
    let (vision, scene, k, x, yc) = first_on(on_scene());
    let now = seconds(scene.frames[k].t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    let (x, y) = (x as f32, -yc as f32);
    let run = |radar: [Option<Slot>; 5]| {
        let mut slots = radar;
        vision.apply(now, 0., PoseAt::default(), &mut slots);
        slots
    };
    // The radar draws the same car in FF, inside the gate (6 m along, 1 m across): it stays the radar's.
    let mut ff = [None; 5];
    ff[0] = Some(radar_slot(x + 5.9, y - 0.9 * y.signum()));
    let merged = run(ff);
    assert_eq!(merged[0], ff[0]);
    assert!(merged[index].is_none());
    // Outside the gate it is another car: the camera fills the side slot.
    for (dx, dy) in [(6.1, 0.), (0., 1.1)] {
        let mut other = [None; 5];
        other[0] = Some(radar_slot(x + dx, y - dy * y.signum()));
        let filled = run(other);
        assert_eq!(filled[index].map(|s| s.vision), Some(VISION_ONLY), "{dx} {dy}");
        assert_eq!(filled[0], other[0]);
    }
    // In the same slot the rule decides: a radar car less than 3 m farther is kept and marked, a farther one is
    // replaced by the closer camera car.
    let mut near = [None; 5];
    near[index] = Some(radar_slot(x + 2.9, y));
    assert_eq!(run(near)[index].map(|s| (s.vid, s.vision)), Some((7, VISION_CONFIRMS)));
    let mut far = [None; 5];
    far[index] = Some(radar_slot(x + 3.1, y));
    assert_eq!(run(far)[index].map(|s| (s.vid, s.vision)), Some((VISION_VID[index - 1], VISION_ONLY)));
}

#[test]
fn camera_car_follows_the_ego_motion_since_its_frame() {
    let (vision, scene, k, x, _) = first_on(on_scene());
    let now = seconds(scene.frames[k].t_ns) + 0.5;
    let index = 1 + scene.side as usize;
    let mut slots = [None; 5];
    // 1 m forward since the frame: the car is 1 m closer.
    vision.apply(now, 0., PoseAt { x: 1., ..PoseAt::default() }, &mut slots);
    assert!((f64::from(slots[index].unwrap().distance) - (x - 1.)).abs() < 1e-5);
}
