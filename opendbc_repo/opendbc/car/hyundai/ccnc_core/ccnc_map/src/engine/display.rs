//! Per-vehicle lateral animation in the road frame.

const STOP_SPEED: f64 = 0.28;
const SETTLE_TIME: f64 = 0.70;
const OFF_CENTER_SETTLE_TIME: f64 = 0.95;
const CENTER_TOLERANCE: f64 = 0.35;
const ADJACENT_INNER_MIN_D: f64 = 2.0;
const LOW_SPEED_HOLD: f64 = 1.2;
const TRAFFIC_STOP_EGO_SPEED: f64 = 1.5;

#[derive(Clone, Copy, Debug)]
enum Motion {
    Settled {
        anchor: f64,
        reference: f64,
        centered: bool,
    },
    Moving {
        direction: f64,
        started_at: f64,
        reverse_time: f64,
        last_advance: f64,
        quiet_d: f64,
        quiet_time: f64,
    },
}

pub(super) struct LateralDisplay {
    motion: Motion,
    d: f64,
    observed_d: f64,
    last_d: f64,
    velocity: f64,
    direction: f64,
    direction_time: f64,
    pub(super) time: f64,
}

fn anchor(d: f64, lane_center: f64) -> f64 {
    if (d - lane_center).abs() <= CENTER_TOLERANCE {
        lane_center
    } else {
        d
    }
}

impl LateralDisplay {
    pub(super) fn new(now: f64, d: f64, lane_center: f64, adjacent_inner_snap: bool) -> Self {
        let anchor = if adjacent_inner_snap
            && lane_center.abs() >= ADJACENT_INNER_MIN_D
            && d * lane_center > 0.
            && d.abs() >= ADJACENT_INNER_MIN_D
            && d.abs() <= lane_center.abs() + CENTER_TOLERANCE
        {
            lane_center
        } else {
            anchor(d, lane_center)
        };
        Self {
            motion: Motion::Settled {
                anchor,
                reference: d,
                centered: anchor == lane_center,
            },
            d: anchor,
            observed_d: d,
            last_d: d,
            velocity: 0.,
            direction: 0.,
            direction_time: 0.,
            time: now,
        }
    }

    pub(super) fn update(
        &mut self,
        now: f64,
        observation: Option<(f64, f64)>,
        lane_center: f64,
        range: f64,
        ego_speed: f64,
        target_speed: f64,
        is_adjacent: bool,
    ) -> f64 {
        let dt = now - self.time;
        self.time = now;
        if dt <= 0. {
            return self.d;
        }
        let Some((d, v_d)) = observation else {
            self.direction_time = 0.;
            if let Motion::Moving { quiet_time, reverse_time, .. } = &mut self.motion {
                *quiet_time = 0.;
                *reverse_time = 0.;
            }
            return self.d;
        };
        let raw_step = d - self.last_d;
        self.last_d = d;
        // Limit the influence of a path refresh on the velocity estimate.
        let delta = (d - self.observed_d) * (1. - (-dt / 0.10).exp());
        self.observed_d += delta;
        let velocity = 0.8 * (delta / dt).clamp(-4., 4.) + 0.2 * v_d;
        self.velocity += (velocity - self.velocity) * (1. - (-dt / 0.20).exp());
        let direction = self.velocity.signum();
        self.direction_time = if dt > 0.10 || self.velocity.abs() < STOP_SPEED {
            0.
        } else if direction == self.direction {
            self.direction_time + dt
        } else {
            dt
        };
        self.direction = direction;

        match &mut self.motion {
            Motion::Settled { anchor, reference, centered } => {
                let both_stopped = ego_speed < LOW_SPEED_HOLD && target_speed < LOW_SPEED_HOLD;
                let adjacent_in_lane_at_stop = is_adjacent
                    && *centered
                    && ego_speed < TRAFFIC_STOP_EGO_SPEED
                    && lane_center.abs() >= ADJACENT_INNER_MIN_D
                    && self.observed_d * lane_center > 0.
                    && self.observed_d.abs() >= ADJACENT_INNER_MIN_D
                    && self.observed_d.abs() <= lane_center.abs() + 1.2;
                let base_departure = if *centered { 0.43 } else { 0.29 };
                let mut departure = base_departure + 0.004 * range.abs().min(100.);
                if both_stopped {
                    departure = (departure * 3.0).max(1.35);
                } else if adjacent_in_lane_at_stop {
                    departure = departure.max(1.35);
                }
                let displacement = (self.observed_d - *anchor) * direction;
                let travel = (self.observed_d - *reference) * direction;
                let sustained = self.direction_time >= 0.15 && displacement >= departure;
                let large = self.direction_time >= 0.10 && displacement >= 2.5 * departure;
                let early = 0.27 + 0.0006 * range.abs().min(100.);
                let progressive = !both_stopped
                    && !adjacent_in_lane_at_stop
                    && self.direction_time >= 0.20
                    && displacement >= early
                    && travel >= early
                    && v_d * direction >= 0.15;
                if progressive || (sustained || large) && travel >= departure {
                    self.motion = Motion::Moving {
                        direction,
                        started_at: now,
                        reverse_time: 0.,
                        last_advance: now,
                        quiet_d: self.observed_d,
                        quiet_time: 0.,
                    };
                }
            }
            Motion::Moving {
                direction: moving,
                reverse_time,
                last_advance,
                quiet_d,
                quiet_time,
                ..
            } => {
                *reverse_time = if dt <= 0.10 && raw_step * *moving < -0.025 {
                    *reverse_time + dt
                } else {
                    0.
                };
                let quiet = 0.10 + 0.001 * range.abs().min(100.);
                if dt <= 0.10
                    && self.velocity.abs() < STOP_SPEED
                    && v_d.abs() < STOP_SPEED
                    && (self.observed_d - *quiet_d).abs() <= quiet
                {
                    *quiet_time += dt;
                    // Between lane centers, allow a longer lull before declaring a stop.
                    let settle_time = if (self.observed_d - lane_center).abs() <= CENTER_TOLERANCE {
                        SETTLE_TIME
                    } else {
                        OFF_CENTER_SETTLE_TIME
                    };
                    if *quiet_time >= settle_time {
                        let anchor = anchor(self.observed_d, lane_center);
                        self.motion = Motion::Settled {
                            anchor,
                            reference: self.observed_d,
                            centered: anchor == lane_center,
                        };
                    }
                } else {
                    *quiet_time = 0.;
                    *quiet_d = self.observed_d;
                    // A sustained opposing observation can resume a stalled animation
                    // before the filtered velocity, which still includes the old motion.
                    if *reverse_time >= 0.16 && now - *last_advance > 0.151
                        && (self.observed_d - self.d) * *moving < -0.18
                    {
                        *moving = -*moving;
                        *reverse_time = 0.;
                    }
                }
            }
        }
        self.d += match &mut self.motion {
            Motion::Settled { anchor, .. } => {
                (*anchor - self.d).clamp(-dt.min(0.075), dt.min(0.075))
            }
            Motion::Moving { direction, started_at, last_advance, .. } => {
                let direction = *direction;
                let lead = 0.07 + 0.04 * (self.velocity.abs() - 0.45).clamp(0., 1.);
                let target = self.observed_d + lead * (self.velocity * direction).max(0.) * direction;
                let step = ((target - self.d) * direction).max(0.) * (1. - (-dt / 0.06).exp());
                if step > 0. {
                    *last_advance = now;
                }
                let ramp = 0.45 + 0.55 * ((now - *started_at) / 0.10).min(1.);
                // One CAN lateral quantum per tick, including irregular radar intervals.
                direction * step.min(ramp * (2.4 * dt).min(0.10))
            }
        };
        self.d
    }

    pub(super) fn diagnostic(&self) -> (bool, f64) {
        match self.motion {
            Motion::Settled { anchor, .. } => (false, anchor),
            Motion::Moving { .. } => (true, self.d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Trace {
        frame: usize,
        display: LateralDisplay,
    }

    impl Trace {
        fn new(d: f64) -> Self {
            Self {
                frame: 0,
                display: LateralDisplay::new(0., d, 0., false),
            }
        }

        fn step(&mut self, d: f64, velocity: f64, x: f64) -> (bool, f64) {
            self.frame += 1;
            let shown = self.display.update(
                self.frame as f64 * 0.05,
                Some((d, velocity)),
                (d / 3.5).round() * 3.5,
                x,
                15.,
                15.,
                false,
            );
            let (moving, anchor) = self.display.diagnostic();
            if std::env::var_os("CCNC_STAGE7_TEST_TRACE").is_some() {
                println!(
                    "display,{},{d:.5},{shown:.5},{},{anchor:.5}",
                    self.frame,
                    u8::from(moving)
                );
            }
            (moving, shown)
        }

        fn move_right_and_stop_on_line(&mut self) {
            for i in 1..=17 {
                self.step(f64::from(i) * 0.1, 2., 20.);
            }
            assert!(self.display.diagnostic().0);
            for _ in 0..8 {
                assert!(self.step(1.7, 0., 20.).0);
            }
            for i in 0..35 {
                self.step(1.7 + if i % 2 == 0 { 0.2 } else { -0.2 }, 0., 20.);
            }
            let (moving, anchor) = self.display.diagnostic();
            assert!(!moving);
            assert!((anchor - 1.7).abs() < 0.08);
            assert_eq!(self.display.d, anchor);
        }
    }

    #[test]
    fn straight_radar_noise_never_changes_the_display() {
        for x in [8., 35., 85.] {
            let mut trace = Trace::new(0.16);
            for tick in 0..160 {
                let noise = if tick % 8 < 4 { 0.25 } else { -0.25 };
                assert_eq!(trace.step(0.16 + noise, 0., x), (false, 0.));
            }
        }
    }

    #[test]
    fn a_sustained_offset_without_lateral_velocity_stays_centered() {
        let mut trace = Trace::new(0.);
        for _ in 0..20 {
            assert_eq!(trace.step(0.45, 0., 20.), (false, 0.));
        }
    }

    #[test]
    fn lane_change_is_monotone_through_three_slow_ticks() {
        let mut trace = Trace::new(0.);
        let mut previous = 0.;
        let mut d = 0.;
        let mut started = false;
        for i in 0..32 {
            let pause = (14..17).contains(&i);
            if !pause {
                d += 0.12;
            }
            let (moving, shown) = trace.step(d, if pause { 0. } else { 2.4 }, 20.);
            assert!(!started || moving, "settled at tick {i}");
            if moving && !started {
                assert!(shown - previous <= 0.045001);
            }
            assert!((0. ..=0.100001).contains(&(shown - previous)));
            previous = shown;
            started |= moving;
        }
        assert!(started && previous > 2.6);
        for _ in 0..5 {
            let (moving, shown) = trace.step(d, 0., 20.);
            assert!(moving && (0. ..=0.100001).contains(&(shown - previous)));
            previous = shown;
        }
        assert!(previous > 3.1);
        for _ in 0..35 {
            trace.step(d, 0., 20.);
        }
        assert_eq!(trace.display.diagnostic(), (false, 3.5));
        assert_eq!(trace.display.d, 3.5);
    }

    #[test]
    fn line_straddling_stop_locks_the_observed_position() {
        let mut trace = Trace::new(0.);
        trace.move_right_and_stop_on_line();
        let fixed = trace.display.d;
        for i in 0..48 {
            let noise = if i % 2 == 0 { 0.2 } else { -0.2 };
            assert_eq!(trace.step(1.7 + noise, 0., 20.), (false, fixed));
        }
    }

    #[test]
    fn departure_after_a_line_stop_resumes_continuously() {
        let mut trace = Trace::new(0.);
        trace.move_right_and_stop_on_line();
        let mut started = false;
        let mut previous = trace.display.d;
        for i in 1..=18 {
            let (moving, shown) = trace.step(1.7 + f64::from(i) * 0.1, 2., 20.);
            assert!(!started || moving);
            assert!((-0.01..=0.100001).contains(&(shown - previous)));
            previous = shown;
            started |= moving;
        }
        assert!(started && previous > 3.);
        assert!(trace.step(3.5, 0., 20.).1 > 3.1);
    }

    #[test]
    fn moving_follows_a_sustained_reversal_without_waiting_for_a_stop() {
        let mut trace = Trace::new(0.);
        for tick in 1..=20 {
            trace.step(f64::from(tick) * 0.12, 2.4, 20.);
        }
        let mut previous = trace.display.d;
        let mut reversed = false;
        for tick in 1..=9 {
            let (moving, shown) = trace.step(2.4 - f64::from(tick) * 0.15, -3., 20.);
            assert!(moving);
            reversed |= shown < previous;
            previous = shown;
        }
        assert!(reversed);
    }

    #[test]
    fn adjacent_inner_corner_start_anchors_at_lane_center() {
        let mut display = LateralDisplay::new(0., -2.1, -3.5, true);
        assert_eq!(display.diagnostic(), (false, -3.5));
        for tick in 1..=30 {
            let d = -2.1 - f64::from(tick) * 0.04;
            let shown = display.update(f64::from(tick) * 0.05, Some((d, -0.4)), -3.5, 8., 0., 2., true);
            assert_eq!(shown, -3.5);
        }
        // A line-straddling vehicle (< 2.0 m) is not snapped to lane center.
        let straddle = LateralDisplay::new(0., -1.7, -3.5, true);
        assert_eq!(straddle.diagnostic(), (false, -1.7));
    }

    #[test]
    fn stopped_vehicle_ignores_radar_reflection_drift() {
        let mut display = LateralDisplay::new(0., -2.7, -3.5, true);
        for tick in 1..=40 {
            let d = -2.4 - 0.55 * (f64::from(tick) * 0.15).sin().abs();
            let shown = display.update(f64::from(tick) * 0.05, Some((d, -0.5)), -3.5, 11., 0.1, 0.1, true);
            assert_eq!(shown, -3.5);
            assert!(!display.diagnostic().0);
        }
    }
}
