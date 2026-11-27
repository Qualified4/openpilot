//! Per-vehicle lateral animation in the road frame.

const STOP_SPEED: f64 = 0.28;
const SETTLE_TIME: f64 = 0.70;
const OFF_CENTER_SETTLE_TIME: f64 = 0.95;
const CENTER_TOLERANCE: f64 = 0.35;

#[derive(Clone, Copy, Debug)]
enum Motion {
    Settled {
        anchor: f64,
        reference: f64,
    },
    Moving {
        direction: f64,
        quiet_d: f64,
        quiet_time: f64,
    },
}

pub(super) struct LateralDisplay {
    motion: Motion,
    d: f64,
    observed_d: f64,
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
    pub(super) fn new(now: f64, d: f64, lane_center: f64) -> Self {
        let anchor = anchor(d, lane_center);
        Self {
            motion: Motion::Settled {
                anchor,
                reference: d,
            },
            d: anchor,
            observed_d: d,
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
    ) -> f64 {
        let dt = now - self.time;
        self.time = now;
        if dt <= 0. {
            return self.d;
        }
        let Some((d, v_d)) = observation else {
            self.direction_time = 0.;
            if let Motion::Moving { quiet_time, .. } = &mut self.motion {
                *quiet_time = 0.;
            }
            return self.d;
        };
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

        let departure = 0.29 + 0.004 * range.abs().min(100.);
        match &mut self.motion {
            Motion::Settled { anchor, reference } => {
                let displacement = (self.observed_d - *anchor) * direction;
                let travel = (self.observed_d - *reference) * direction;
                let sustained = self.direction_time >= 0.15 && displacement >= departure;
                let large = self.direction_time >= 0.10 && displacement >= 2.5 * departure;
                if (sustained || large) && travel >= departure {
                    self.motion = Motion::Moving {
                        direction,
                        quiet_d: self.observed_d,
                        quiet_time: 0.,
                    };
                }
            }
            Motion::Moving {
                direction: moving,
                quiet_d,
                quiet_time,
            } => {
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
                        self.motion = Motion::Settled {
                            anchor: anchor(self.observed_d, lane_center),
                            reference: self.observed_d,
                        };
                    }
                } else {
                    *quiet_time = 0.;
                    *quiet_d = self.observed_d;
                    // Allow deliberate reversals while suppressing small corrections mid-manoeuvre.
                    if direction != *moving
                        && self.direction_time >= 0.18
                        && (self.observed_d - self.d) * direction > 0.24
                    {
                        *moving = direction;
                    }
                }
            }
        }
        self.d += match self.motion {
            Motion::Settled { anchor, .. } => {
                (anchor - self.d).clamp(-dt.min(0.075), dt.min(0.075))
            }
            Motion::Moving { direction, .. } => {
                let target = self.observed_d + 0.07 * self.velocity;
                let step = ((target - self.d) * direction).max(0.) * (1. - (-dt / 0.10).exp());
                direction * step.min(7.5 * dt).min(0.37)
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
                display: LateralDisplay::new(0., d, 0.),
            }
        }

        fn step(&mut self, d: f64, velocity: f64, x: f64) -> (bool, f64) {
            self.frame += 1;
            let shown = self.display.update(
                self.frame as f64 * 0.05,
                Some((d, velocity)),
                (d / 3.5).round() * 3.5,
                x,
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
            assert!((0. ..=0.37).contains(&(shown - previous)));
            previous = shown;
            started |= moving;
        }
        assert!(started && previous > 3.1);
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
            assert!((-0.01..=0.37).contains(&(shown - previous)));
            previous = shown;
            started |= moving;
        }
        assert!(started && previous > 3.1);
    }
}
