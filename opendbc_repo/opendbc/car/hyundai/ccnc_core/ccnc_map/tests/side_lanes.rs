//! The radar map's lane rules for the map process's LF and RF (`Lanes::side_blocked`).

use std::borrow::Cow;

use ccnc_map::radar::lanes::{Lanes, Line};

const LEFT: usize = 0;
const RIGHT: usize = 1;

/// Straight lines at these y (left outer, ego left, ego right, right outer, left edge, right edge); None = no line.
fn lanes(y: [Option<f64>; 6]) -> Lanes<'static> {
    let lines = y.map(|y| Line {
        lane: y.map(|_| (0.9, 5.)),
        geometry: y.map(|y| Cow::Owned(vec![(-10., y), (100., y)])),
    });
    Lanes::new(lines, 0, true, true, (2.3, 3.5), false)
}

fn blocked(l: &Lanes, y: f64, moving: bool, walls: &[f64]) -> [bool; 2] {
    l.side_blocked(20., y, moving, walls)
}

#[test]
fn a_side_slot_needs_an_adjacent_lane() {
    // A 1.25 m strip past ego's right line is no lane (1.25 − 0.25 ≤ 1.8); a 3.5 m lane is.
    let narrow = lanes([Some(5.25), Some(1.75), Some(-1.75), Some(-3.0), None, None]);
    assert!(blocked(&narrow, -2.4, true, &[])[RIGHT]);
    let wide = lanes([Some(5.25), Some(1.75), Some(-1.75), Some(-5.25), None, None]);
    assert_eq!(blocked(&wide, -3.5, true, &[]), [false; 2]);
    assert_eq!(blocked(&wide, 3.5, false, &[]), [false; 2]);
    // No outer line and no road edge on that side: no adjacent lane is known.
    let ego_only = lanes([None, Some(1.75), Some(-1.75), None, None, None]);
    assert_eq!(blocked(&ego_only, -3.5, true, &[]), [false, true]);
    assert_eq!(blocked(&ego_only, 3.5, true, &[]), [true, false]);
}

#[test]
fn a_side_slot_ends_0_95_m_inside_the_road_edge() {
    // Road edge 2.25 m past ego's right line: room 2.25 − 0.25 > 1.8, but 1.75 m past the line is beyond 2.25 − 0.95.
    let edge = lanes([None, Some(1.75), Some(-1.75), None, None, Some(-4.0)]);
    assert!(blocked(&edge, -3.5, true, &[])[RIGHT]);
    assert!(!blocked(&edge, -2.9, true, &[])[RIGHT]);
    let far_edge = lanes([None, Some(1.75), Some(-1.75), None, None, Some(-6.0)]);
    assert!(!blocked(&far_edge, -3.5, false, &[])[RIGHT]);
}

#[test]
fn a_side_slot_ends_at_a_wall_line() {
    let road = lanes([Some(5.25), Some(1.75), Some(-1.75), Some(-5.25), None, None]);
    assert!(blocked(&road, -3.5, true, &[-3.0])[RIGHT]);
    assert!(!blocked(&road, -2.8, true, &[-3.0])[RIGHT]);
    // A wall on the other side does not limit this one.
    assert!(!blocked(&road, -3.5, true, &[3.0])[RIGHT]);
}

#[test]
fn no_ego_line_means_no_side_slot_and_ego_lane_objects_are_left_to_the_map() {
    let none = lanes([None; 6]);
    assert_eq!(blocked(&none, 3.5, true, &[]), [true; 2]);
    // Inside ego's lines the world lane coordinate decides, even with no adjacent lane.
    let ego_only = lanes([None, Some(1.75), Some(-1.75), None, None, None]);
    assert_eq!(blocked(&ego_only, -1.0, true, &[]), [false; 2]);
}

#[test]
fn side_cars_stay_allowed_through_an_ego_lane_change_to_the_right() {
    // Ego moves right by d over a 3.5 m road of three lanes. The model relabels the lines once ego crosses its
    // right line: the right car (world y −3.5) becomes FF and the car ahead (world y 0) becomes LF.
    for step in 0..=35 {
        let d = f64::from(step) * 0.1;
        let crossed = d > 1.75;
        let at = |world: f64| Some(world + d);
        let l = if crossed {
            lanes([at(1.75), at(-1.75), at(-5.25), at(-8.75), at(5.25), at(-8.75)])
        } else {
            lanes([at(5.25), at(1.75), at(-1.75), at(-5.25), at(5.25), at(-8.75)])
        };
        let (right_car, ahead) = (-3.5 + d, d);
        assert!(!blocked(&l, right_car, true, &[])[RIGHT], "d={d}: right car");
        assert!(!blocked(&l, ahead, true, &[])[LEFT], "d={d}: car ahead");
    }
}
