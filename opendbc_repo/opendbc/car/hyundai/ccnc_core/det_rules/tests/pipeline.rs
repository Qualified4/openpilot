//! The unit tests of det-ready-20261121/scripts/test_pipeline.py, one to one (the two scene regressions are in
//! det_rules_cli/tests, they need the scene data).
use det_rules::geom::Ground;
use det_rules::rules::{front_box, swap_decision, Act, ClassConfirm, RuleE, SplitGate, SwapCfg, VehicleClass, FRONT_IN, FRONT_W};
use det_rules::wide::*;
use det_rules::{round2, Lane, Mask, Track, View, WideDet, CAM2B};

fn det(bbox: [f64; 4], view: View) -> (WideDet, Ground) {
    let g = Ground { bl: Some([6.0, 3.0]), bc: Some([6.1, 3.5]), br: Some([6.2, 4.0]), gsize: (view != View::Raw).then_some([5.0, 4.0]), wbox: None };
    (WideDet { view, bbox, score: 0.8, class: 2 }, g)
}
fn tr(id: i64, d_rel: f64, y_rel: f64, v_rel: f64) -> Track {
    Track { id, d_rel, y_rel, v_rel }
}

// ---------------------------------------------------------------- WideBand
#[test]
fn band_shape() {
    let b = Mask::band();
    assert!((b.mean() - 0.355).abs() < 0.005); // Windows report: 35.5 % of the frame
    assert!(b.get(672, 700) && !b.get(672, 300));
}
#[test]
fn drop_comes_before_cut() {
    // box mostly inside the band AND touching it: must be dropped, not rescued as a cut box
    let (d, g) = det([500.0, 470.0, 800.0, 759.0], View::Raw);
    assert!(wide_candidate(&d, &g, &Mask::band(), true).is_none());
}
#[test]
fn cut_side_box_uses_size_distance_and_lateral() {
    let m = Mask::from_fn(512, 384, |_, y| y >= 300);
    let (d, mut g) = det([100.0, 150.0, 300.0, 302.0], View::SideR);
    g.gsize = Some([1.7, 4.2]);
    let c = wide_candidate(&d, &g, &m, true).unwrap();
    assert!(c.cut);
    assert!((c.x - (1.7 - CAM2B)).abs() < 0.005);
    assert!((c.yc - 4.2).abs() < 1e-7);
    assert_eq!(c.x_bottom, Some(round2(6.0 - CAM2B)));
}
#[test]
fn cut_raw_box_is_near03() {
    let (d, mut g) = det([1000.0, 300.0, 1200.0, 500.0], View::Raw);
    g = Ground { bl: Some([2.0, 3.0]), bc: Some([2.0, 3.5]), br: Some([2.0, 4.0]), ..g };
    let c = wide_candidate(&d, &g, &Mask::band(), true).unwrap();
    assert!(c.cut && c.near03);
    assert_eq!(c.x, 1.5);
}
#[test]
fn cut_box_skips_hood_filter() {
    let c = Cand { cut: true, bbox: [0.0, 0.0, 100.0, 100.0], ..Cand::at(View::SideR, 0.1, 3.0) };
    assert!(valid(&c));
    assert!(!valid(&Cand { cut: false, ..c }));
}
#[test]
fn pick_prefers_side_view() {
    let a = Cand { bbox: [0.0, 0.0, 10.0, 10.0], ..Cand::at(View::Raw, 2.0, 3.0) };
    let b = Cand { bbox: [0.0, 0.0, 10.0, 10.0], ..Cand::at(View::SideR, 4.0, 3.0) };
    assert!(std::ptr::eq(pick_side(&[&a, &b], Lane::Right).unwrap(), &b));
}

// ---------------------------------------------------------------- Rules
fn rc(x: f64) -> Cand {
    Cand { xs: Some(x), ..Cand::at(View::SideR, x, 3.0) }
}
#[test]
fn rule_e_needs_two_stable() {
    let mut r = RuleE::default();
    assert!(!r.step(Some(&rc(3.0)), 0.0, true, true, 0.0));
    assert!(r.step(Some(&rc(3.4)), 0.0, true, true, 0.0));
}
#[test]
fn rule_e_jump_resets_even_for_cut() {
    let mut r = RuleE::default();
    r.step(Some(&Cand { cut: true, ..rc(3.0) }), 0.0, true, true, 0.0);
    assert!(!r.step(Some(&Cand { cut: true, ..rc(5.0) }), 0.0, true, true, 0.0)); // Windows semantics: cut boxes keep the x-jump check
}
#[test]
fn rule_e_off_when_moving_or_guard() {
    let mut r = RuleE::default();
    r.step(Some(&rc(3.0)), 0.0, true, true, 0.0);
    assert!(!r.step(Some(&rc(3.0)), 3.5, true, true, 0.0));
    let mut r = RuleE::default();
    r.step(Some(&rc(3.0)), 0.0, false, true, 0.0);
    assert!(!r.step(Some(&rc(3.0)), 0.0, false, true, 0.0));
}
#[test]
fn consistency() {
    let c = Cand { xs: Some(5.0), ..Cand::at(View::SideR, 2.0, 3.0) };
    assert!(!consistent(&c));
    assert!(consistent(&Cand { cut: true, ..c }));
}
#[test]
fn swap() {
    let cfg = SwapCfg { min_run: 2, margin: 3.0, ..SwapCfg::default() };
    let c = Some((3.0, false));
    assert_eq!(swap_decision(true, c, false, 0.0, 2, &cfg), Some(Act::Fill));
    assert_eq!(swap_decision(true, c, true, 9.6, 2, &cfg), Some(Act::Swap));
    assert_eq!(swap_decision(true, c, true, 5.0, 2, &cfg), Some(Act::Keep));
    assert_eq!(swap_decision(false, c, true, 9.6, 2, &cfg), None);
    assert_eq!(swap_decision(true, Some((3.0, true)), true, 9.6, 2, &SwapCfg { allow_cut: false, ..cfg }), Some(Act::Keep));
}
#[test]
fn radar_oncoming_sign() {
    let c = Cand { xs: Some(4.4), ..Cand::at(View::SideR, 4.4, -2.8) };
    assert!(radar_oncoming_near(&[tr(37, 4.8, 2.9, -3.7)], &c, 0.0)); // coming toward a stopped ego
    assert!(!radar_oncoming_near(&[tr(51, 4.1, 2.8, 1.0)], &c, 2.0)); // moving along with us
    assert!(!radar_oncoming_near(&[tr(51, 30.0, 2.8, -9.0)], &c, 0.0)); // too far from the camera car
}
#[test]
fn no_road() {
    let c = Cand { xs: Some(0.5), ..Cand::at(View::SideR, 0.5, 3.8) };
    assert!(no_road_context(Some(&[0.01, 0.02, 0.03, 0.01]), &[], &c));
    assert!(!no_road_context(Some(&[0.3, 0.6, 0.7, 0.1]), &[], &c));
    assert!(!no_road_context(Some(&[0.01, 0.02, 0.03, 0.01]), &[tr(1, 2.5, -3.8, 0.0)], &c)); // radar sees something there
}

// ---------------------------------------------------------------- Front
#[test]
fn front_input_shape_and_cut() {
    // the 1344x760 -> 416x192 input itself is made by the runner; here its scale and the cut flag
    let s = FRONT_IN.0 as f64 / FRONT_W;
    assert_eq!(FRONT_IN, (416, 192));
    let b = front_box(&[10.0, 100.0, 50.0, 191.9], s, 2.0);
    assert!(b.cut);
    assert!(!(b.use_bottom_distance || b.may_confirm_class || b.camera_only_ok));
    assert!(!front_box(&[10.0, 100.0, 50.0, 150.0], s, 2.0).cut);
}
#[test]
fn class_confirm_r1() {
    use VehicleClass::*;
    let mut k = ClassConfirm::default();
    assert_eq!(k.observe(0.0, 10.0, 500.0, Truck, false, false), Car); // night: never confirms
    assert_eq!(k.observe(0.0, 10.0, 500.0, Truck, true, false), Car);
    assert_eq!(k.observe(0.5, 12.0, 500.0, Truck, true, false), Car); // < 1 s apart: ignored
    assert_eq!(k.observe(1.2, 12.0, 500.0, Truck, true, false), Truck); // 2 distinct views agree
    for t in [2.5, 3.8, 5.1] {
        assert_eq!(k.observe(t, 12.0 + t, 500.0, Car, true, false), Truck);
    }
    assert_eq!(k.observe(6.4, 20.0, 500.0, Car, true, false), Car); // flip needs 4
    assert_eq!(k.observe(8.0, 25.0, 500.0, Truck, true, true), Car); // cut box: no evidence
}

// ---------------------------------------------------------------- LrRr: two cars on one side -> LF/RF + LR/RR
fn lc(view: View, x: f64, bbox: [f64; 4], wbox: Option<[f64; 4]>, cut: bool, lane: Lane, yc: f64) -> Cand {
    Cand { bbox, wbox, cut, lane, ..Cand::at(view, x, yc) }
}
fn car(c: &Cand) -> bool {
    counts_as_car(c, 0.0, &[], None)
}
fn same(a: Option<&Cand>, b: &Cand) -> bool {
    a.is_some_and(|a| std::ptr::eq(a, b))
}
const R: Lane = Lane::Right;
const L: Lane = Lane::Left;
fn beside_r() -> Cand {
    lc(View::SideR, 0.08, [259.0, 121.0, 405.0, 226.0], Some([1180.0, 470.0, 1330.0, 480.0]), true, R, 3.0)
}
fn ahead_r() -> Cand {
    lc(View::Raw, 5.2, [820.0, 370.0, 1014.0, 465.0], None, false, R, 3.0)
}
#[test]
fn one_car_in_two_views_stays_single() {
    // scene 47 pattern: raw cut box (x = 1.5 placeholder) and the side-view box of the same car overlap in the raw frame
    let raw = lc(View::Raw, 1.5, [1097.0, 330.0, 1322.0, 509.0], None, true, R, 3.0);
    let side = lc(View::SideR, 4.7, [100.0, 120.0, 300.0, 220.0], Some([1110.0, 500.0, 1300.0, 505.0]), false, R, 3.0);
    let (f, r) = split_side(&[&raw, &side], R, car);
    assert!(same(f, &side) && r.is_none());
}
#[test]
fn two_cars_rear_is_outermost() {
    // scene 25 pattern: white car beside at the right image edge, dark car ahead further inward
    let (beside, ahead) = (beside_r(), ahead_r());
    let (f, r) = split_side(&[&ahead, &beside], R, car);
    assert!(same(f, &ahead) && same(r, &beside));
    let (f, r) = split_side(&[&beside, &ahead], R, car); // order of detections does not matter
    assert!(same(f, &ahead) && same(r, &beside));
}
#[test]
fn left_side_mirrors() {
    let beside = lc(View::SideL, 0.5, [10.0, 100.0, 300.0, 230.0], Some([10.0, 520.0, 400.0, 530.0]), true, L, -3.0);
    let ahead = lc(View::SideL, 5.9, [380.0, 130.0, 500.0, 215.0], Some([410.0, 460.0, 520.0, 462.0]), false, L, -3.0);
    let (f, r) = split_side(&[&ahead, &beside], L, car);
    assert!(same(f, &ahead) && same(r, &beside));
}
#[test]
fn not_clearly_behind_stays_single() {
    // two separate boxes, but the 'outer' one is not REAR_GAP behind the inner one -> no rear slot
    let a = lc(View::Raw, 4.0, [1100.0, 330.0, 1300.0, 500.0], None, false, R, 3.0);
    let b = lc(View::Raw, 4.5, [820.0, 370.0, 1014.0, 465.0], None, false, R, 3.0);
    let (f, r) = split_side(&[&a, &b], R, car);
    assert!(r.is_none() && same(f, pick_side(&[&a, &b], R).unwrap()));
}
#[test]
fn other_side_and_far_cars_ignored() {
    let beside = lc(View::SideR, 0.1, [259.0, 121.0, 405.0, 226.0], Some([1180.0, 470.0, 1330.0, 480.0]), true, R, 3.0);
    let far = lc(View::Raw, 9.0, [820.0, 370.0, 900.0, 420.0], None, false, R, 3.0); // beyond near() range
    let left = lc(View::Raw, 5.0, [100.0, 370.0, 300.0, 465.0], None, false, L, -3.0);
    let (f, r) = split_side(&[&beside, &far, &left], R, car);
    assert!(same(f, &beside) && r.is_none());
}
#[test]
fn scene47_bonnet_cut_box_is_not_the_front_car() {
    // scene 47 n=445 (night): true car beside = vL cut x 2.9; 'front' = vL cut box on the bonnet line x 4.8 (false detection)
    let beside = lc(View::SideL, 2.89, [10.0, 120.0, 250.0, 230.0], Some([20.0, 560.0, 330.0, 570.0]), true, L, -4.9);
    let bonnet = lc(View::SideL, 4.79, [300.0, 150.0, 420.0, 230.0], Some([420.0, 520.0, 560.0, 525.0]), true, L, -2.3);
    let (f, r) = split_side(&[&bonnet, &beside], L, car);
    assert!(r.is_none() && same(f, pick_side(&[&bonnet, &beside], L).unwrap())); // LF/RF rule for the cut box unchanged
    let uncut = Cand { cut: false, ..beside };
    assert!(split_side(&[&bonnet, &uncut], L, car).1.is_none()); // a cut box is never the front car
}
#[test]
fn split_counts_only_cars_passing_the_display_filters() {
    let (beside, ahead) = (beside_r(), ahead_r());
    let weak = Cand { score: 0.4, ..beside };
    assert!(split_side(&[&ahead, &weak], R, car).1.is_none()); // weak_cut rear is not a car
    let incons = Cand { xs: Some(9.0), ..ahead };
    assert!(split_side(&[&incons, &beside], R, car).1.is_none()); // inconsistent front is not a car
    let onc = |c: &Cand| counts_as_car(c, 0.0, &[tr(37, 7.0, -3.0, -4.0)], None); // oncoming track at the front car
    assert!(split_side(&[&ahead, &beside], R, onc).1.is_none());
}
#[test]
fn split_gate_needs_front_car_on_by_rule_e() {
    let beside = beside_r();
    let ahead = Cand { xs: Some(5.2), ..ahead_r() };
    let mut g = SplitGate::default();
    let sp = split_side(&[&ahead, &beside], R, car);
    let single = |o: (Option<&Cand>, Option<&Cand>)| same(o.0, &beside) && o.1.is_none();
    assert!(single(g.apply(sp, Some(&beside), 0.0, true, |_| true, 0.0))); // 1st frame: rule E not on yet
    let o = g.apply(sp, Some(&beside), 0.0, true, |_| true, 0.0); // 2nd stable frame: split
    assert!(same(o.0, &ahead) && same(o.1, &beside));
    assert!(single(g.apply(sp, Some(&beside), 0.0, true, |_| false, 0.0))); // front car vetoed by a filter
    assert!(single(g.apply((Some(&beside), None), Some(&beside), 0.0, true, |_| true, 0.0)));
}
#[test]
fn weak_cut_counts_for_two_cars_only() {
    // bonnet-line cut box with a low score: not a car for the two-car split, but the display does not veto it
    let weak = Cand { cut: true, score: 0.35, bbox: [10.0, 200.0, 120.0, 383.0], ..Cand::at(View::SideL, 1.2, -3.4) };
    assert_eq!(filter_hits(&weak, 0.0, &[], None, Filters::CAR).names(), ["weak_cut"]);
    assert!(!car(&weak));
    assert!(display_filters(&weak, 0.0, &[], None).is_empty());
    assert!(car(&Cand { score: 0.7, ..weak }));
    assert_eq!(Filters(Filters::CAR.0 & !Filters::DISPLAY.0), Filters::WEAK_CUT);
}
#[test]
fn rear_distance_and_swap() {
    assert_eq!(rear_distance(&Cand::at(View::SideR, 0.4, 3.0)), 0.0);
    assert_eq!(rear_distance(&Cand::at(View::SideR, -1.2, 3.0)), 1.2);
    let (cfg, rear) = (SwapCfg { min_run: 2, margin: 3.0, ..SwapCfg::default() }, Some((0.0, true)));
    assert_eq!(swap_decision(true, rear, true, 11.1, 2, &cfg), Some(Act::Swap)); // logged LR slot 11 m behind, camera beside
    assert_eq!(swap_decision(true, rear, false, 0.0, 2, &cfg), Some(Act::Fill));
    assert_eq!(swap_decision(true, rear, true, 2.0, 2, &cfg), Some(Act::Keep));
}

// ---------------------------------------------------------------- port-specific: Python number semantics
#[test]
fn round2_matches_python_round() {
    // expected values printed by Python 3.12 round(x, 2)
    for (x, want) in [(0.125, 0.12), (0.375, 0.38), (0.625, 0.62), (2.675, 2.67), (1.005, 1.0), (-0.001, -0.0), (4.4849999, 4.48), (-1.515, -1.51)] {
        assert_eq!(round2(x), want, "round2({x})");
    }
}
