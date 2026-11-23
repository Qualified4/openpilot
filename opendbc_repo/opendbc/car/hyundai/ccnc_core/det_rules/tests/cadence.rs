//! t9f: the 1 Hz rules read in time for slower per-side cadences (`Timing::jump_dt_max`, `Timing::rear_span_s`).
use det_rules::wide::Cand;
use det_rules::{DetRules, Lane, SideOutput, SlotState, Timing, View, WideFrame};

fn rc(x: f64, yc: f64) -> Cand {
    Cand { xs: Some(x), ..Cand::at(View::SideR, x, yc) }
}

#[test]
fn jump_limits_stop_growing_past_two_seconds() {
    // 3 s apart, 5 m along: within 2.0 m/s * 3 s before t9f, beyond the 4.0 m cap now
    let mut old = Timing::TIME_1HZ_T9E.rule_e();
    let mut new = Timing::TIME_1HZ.rule_e();
    for r in [&mut old, &mut new] {
        r.step(Some(&rc(2.0, 3.0)), 0.0, true, true, 0.0);
    }
    assert!(old.step(Some(&rc(7.0, 3.0)), 0.0, true, true, 3.0));
    assert!(!new.step(Some(&rc(7.0, 3.0)), 0.0, true, true, 3.0));
    // lateral: 3.0 m in 3 s passes 1.4 m/s * 3 s, not the 2.8 m cap
    let mut new = Timing::TIME_1HZ.rule_e();
    new.step(Some(&rc(2.0, 3.0)), 0.0, true, true, 0.0);
    assert!(!new.step(Some(&rc(2.0, 6.0)), 0.0, true, true, 3.0));
    // at and below 2 s nothing changes
    for dt in [0.5, 1.0, 1.06, 2.0] {
        for (dx, dy) in [(0.0, 0.0), (1.9 * dt, 0.0), (2.1 * dt, 0.0), (0.0, 1.3 * dt), (0.0, 1.5 * dt)] {
            let mut old = Timing::TIME_1HZ_T9E.rule_e();
            let mut new = Timing::TIME_1HZ.rule_e();
            let a = [old.step(Some(&rc(2.0, 3.0)), 0.0, true, true, 10.0), old.step(Some(&rc(2.0 + dx, 3.0 + dy)), 0.0, true, true, 10.0 + dt)];
            let b = [new.step(Some(&rc(2.0, 3.0)), 0.0, true, true, 10.0), new.step(Some(&rc(2.0 + dx, 3.0 + dy)), 0.0, true, true, 10.0 + dt)];
            assert_eq!(a, b, "dt {dt} dx {dx} dy {dy}");
        }
    }
}

/// scene 25 pattern (tests/pipeline.rs): white car beside at the right image edge (cut), dark car ahead in the raw view.
fn pair() -> [Cand; 2] {
    let beside = Cand {
        bbox: [259.0, 121.0, 405.0, 226.0],
        wbox: Some([1180.0, 470.0, 1330.0, 480.0]),
        cut: true,
        lane: Lane::Right,
        ..Cand::at(View::SideR, 0.08, 3.0)
    };
    let ahead = Cand { bbox: [820.0, 370.0, 1014.0, 465.0], lane: Lane::Right, xs: Some(5.2), ..Cand::at(View::Raw, 5.2, 3.0) };
    [ahead, beside]
}

fn right(rules: &mut DetRules, t: f64) -> SideOutput {
    let frame = WideFrame {
        t_ns: (t * 1e9) as u64,
        v_ego: 0.0,
        start_guard_ok: true,
        tracks: &[],
        lprob: None,
        slots: [[SlotState::default(); 2]; 2],
    };
    rules.wide(&frame, &pair()).sides[1]
}

fn rear_on(timing: Timing, period: f64) -> Vec<(bool, u32)> {
    let mut rules = DetRules::new(timing);
    (0..4).map(|k| right(&mut rules, 100.0 + period * f64::from(k))).map(|o| (o.rear.display_on, o.rear.confirm_run)).collect()
}

#[test]
fn rear_slot_on_after_two_seconds_of_the_pair() {
    // 1 Hz: the third frame (2 s) as before
    assert_eq!(rear_on(Timing::TIME_1HZ, 1.0), rear_on(Timing::TIME_1HZ_T9E, 1.0));
    assert_eq!(rear_on(Timing::TIME_1HZ, 1.0), [(false, 0), (false, 1), (true, 2), (true, 3)]);
    // 2 s per side: the second frame (2 s) instead of the third (4 s), with the pair's two frames as the confirm count
    assert_eq!(rear_on(Timing::TIME_1HZ_T9E, 2.0), [(false, 0), (false, 1), (true, 2), (true, 3)]);
    assert_eq!(rear_on(Timing::TIME_1HZ, 2.0), [(false, 0), (true, 2), (true, 2), (true, 3)]);
    // 1.4 s per side: 1.4 s < 2.0 s - jitter, the third frame as before
    assert_eq!(rear_on(Timing::TIME_1HZ, 1.4), rear_on(Timing::TIME_1HZ_T9E, 1.4));
    // never from one frame of the pair, never in FRAMES
    assert_eq!(rear_on(Timing::TIME_1HZ, 5.0)[0], (false, 0));
    assert_eq!(rear_on(Timing::FRAMES, 2.0), [(false, 0), (false, 1), (true, 2), (true, 3)]);
}

#[test]
fn rear_pair_restarts_with_the_split_proposal() {
    // a frame without the front car breaks the proposal: the pair count starts again, so 2 s later the rear is not on
    let mut rules = DetRules::new(Timing::TIME_1HZ);
    right(&mut rules, 0.0);
    let frame = WideFrame { t_ns: 2_000_000_000, v_ego: 0.0, start_guard_ok: true, tracks: &[], lprob: None, slots: [[SlotState::default(); 2]; 2] };
    let [_, beside] = pair();
    let o = rules.wide(&frame, std::slice::from_ref(&beside)).sides[1];
    assert!(!o.rear.display_on && o.rear.cand.is_none());
    let o = right(&mut rules, 4.0);
    assert!(!o.rear.display_on);
    let o = right(&mut rules, 6.0);
    assert!(o.rear.display_on && o.rear.confirm_run == 2);
}
