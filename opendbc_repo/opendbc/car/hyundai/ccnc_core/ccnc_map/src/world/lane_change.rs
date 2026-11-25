use super::STATUS_FROZEN;

#[derive(Default)]
pub struct LaneChange {
    started: Option<f64>,
    settled: Option<f64>,
    last: Option<(f64, f64)>,
    rate: f64,
}

impl LaneChange {
    pub fn step(&mut self, t: f64, u: f64) -> bool {
        if !u.is_finite() || !t.is_finite() {
            *self = Self::default();
            return false;
        }
        if let Some((last, prior)) = self.last {
            let dt = t - last;
            if dt <= 0. {
                return self.frozen();
            }
            if dt > 1e-3 && dt < 0.5 {
                self.rate += (dt / 0.3).min(1.) * ((u - prior) / dt - self.rate);
            }
        }
        self.last = Some((t, u));
        let center = u - (u.floor() + 0.5);
        if let Some(start) = self.started {
            if center.abs() < 0.15 && self.rate.abs() < 0.08 {
                self.settled.get_or_insert(t);
            } else {
                self.settled = None;
            }
            if t - start >= 8. || self.settled.is_some_and(|since| t - since >= 0.5) {
                self.started = None;
                self.settled = None;
            }
        } else if center.abs() > 0.3 || (center.abs() > 0.08 && self.rate * center.signum() > 0.1) {
            self.started = Some(t);
        }
        self.frozen()
    }

    pub fn status(&self) -> u8 {
        if self.frozen() {
            STATUS_FROZEN
        } else {
            0
        }
    }

    pub(super) fn frozen(&self) -> bool {
        self.started.is_some()
    }
}

// Ego lane coordinate and lateral rate in units of lane width.
const BIAS_TAU_S: f64 = 0.15;
const BIAS_PROB_LO: f64 = 0.35;
const BIAS_PROB_HI: f64 = 0.55;
const RENUMBER_BIAS_HOLD_S: f64 = 0.25;

// Critically damped 2-state KF (eliminates velocity-momentum overshoot on smooth lane changes)
const VEL_DECAY_TAU_S: f64 = 0.90;
const Q_U: f64 = 0.020;
const Q_UDOT: f64 = 0.30;
const R_BASE: f64 = 0.0008;
const R_LOW_PROB_MULT: f64 = 2.0;
const RENUMBER_P_BOOST: f64 = 0.015;
const HUBER_THRESH_U: f64 = 0.06;
const MAX_UDOT: f64 = 1.2;

const MAX_CURV_INV_M: f64 = 0.045;
const MAX_YAW_RAD_S: f64 = 0.25;
const TURN_HOLD_S: f64 = 0.60;
const RELOCK_AFTER_LOSS_S: f64 = 0.80;
const INTERIOR_MARGIN_U: f64 = 0.20;
const INTERIOR_DWELL_S: f64 = 0.15;

const H_COMMIT: f64 = 0.08;
const V_MIN_COMMIT: f64 = 0.04;
const T_CONFIRM_S: f64 = 0.14;
const H_REN_COMMIT: f64 = 0.05;
const T_REN_CONFIRM_S: f64 = 0.08;
const REN_WINDOW_S: f64 = 0.65;
const H_DEEP: f64 = 0.18;
const T_DEEP_S: f64 = 0.25;

const COOLDOWN_S: f64 = 1.00;
const H_REVERSE_COOLDOWN: f64 = 0.24;
const T_REVERSE_COOLDOWN_S: f64 = 0.30;

const APPROACH_ENTER_DIST_U: f64 = 0.25;
const APPROACH_EXIT_DIST_U: f64 = 0.32;
const APPROACH_ENTER_VEL: f64 = 0.05;
const APPROACH_KEEP_VEL: f64 = -0.05;
const APPROACH_GRACE_S: f64 = 0.35;

#[derive(Clone, Copy, Debug)]
pub struct EgoLaneInput {
    pub t: f64,
    pub map_u: Option<f64>,
    pub raw_bias: f64,
    pub min_inner_prob: f64,
    pub yaw: Option<f64>,
    pub speed: f64,
    pub renumbers: u32,
    pub map_unknown: bool,
}

pub struct EgoLaneFilter {
    t_last: Option<f64>,
    last_ren: Option<u32>,
    last_ren_t: f64,
    init: bool,
    u: f64,
    udot: f64,
    p00: f64,
    p01: f64,
    p11: f64,
    bias_lp: f64,
    turn_guard_until: f64,
    relock_until: f64,
    was_map_unknown: bool,
    needs_interior_lock: bool,
    interior_lane: Option<i32>,
    interior_since: Option<f64>,
    committed_lane: Option<i32>,
    pending_dir: i32,
    pending_since: Option<f64>,
    deep_since: Option<f64>,
    last_commit: Option<(f64, i32, i32)>,
    last_commit_dir: i32,
    approaching_dir: i8,
    approaching_until: f64,
}

impl Default for EgoLaneFilter {
    fn default() -> Self {
        Self {
            t_last: None,
            last_ren: None,
            last_ren_t: f64::NEG_INFINITY,
            init: false,
            u: f64::NAN,
            udot: 0.,
            p00: 0.05,
            p01: 0.,
            p11: 0.25,
            bias_lp: 0.,
            turn_guard_until: f64::NEG_INFINITY,
            relock_until: f64::NEG_INFINITY,
            was_map_unknown: false,
            needs_interior_lock: false,
            interior_lane: None,
            interior_since: None,
            committed_lane: None,
            pending_dir: 0,
            pending_since: None,
            deep_since: None,
            last_commit: None,
            last_commit_dir: 0,
            approaching_dir: 0,
            approaching_until: f64::NEG_INFINITY,
        }
    }
}

impl EgoLaneFilter {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn committed_lane(&self) -> Option<i32> {
        self.committed_lane
    }

    pub fn last_commit(&self) -> Option<(f64, i32, i32)> {
        self.last_commit
    }

    pub fn approaching(&self) -> bool {
        self.approaching_dir != 0
    }

    pub fn step(&mut self, inp: EgoLaneInput) -> f64 {
        let t = inp.t;
        if !t.is_finite() {
            return self.u;
        }
        let dt = match self.t_last {
            Some(last) if t > last => (t - last).min(0.5),
            Some(_) => return self.u,
            None => 0.05,
        };
        self.t_last = Some(t);

        let ren_happened = self.last_ren.is_some_and(|prev| prev != inp.renumbers);
        if ren_happened {
            self.last_ren_t = t;
        }
        self.last_ren = Some(inp.renumbers);

        // Turn / intersection guard
        let speed = if inp.speed.is_finite() { inp.speed.max(0.) } else { 0. };
        let in_sharp_turn = inp.yaw.filter(|y| y.is_finite()).is_some_and(|yaw| {
            let ay = yaw.abs();
            let curv = ay / speed.max(0.5);
            ay > MAX_YAW_RAD_S || curv > MAX_CURV_INV_M || (speed < 4.5 && ay > 0.15)
        });
        if in_sharp_turn {
            self.turn_guard_until = self.turn_guard_until.max(t + TURN_HOLD_S);
            self.committed_lane = None;
            self.needs_interior_lock = true;
            self.interior_lane = None;
            self.interior_since = None;
        }

        if inp.map_unknown {
            self.was_map_unknown = true;
            self.init = false;
            self.u = f64::NAN;
            self.udot = 0.;
            self.bias_lp = 0.;
            self.pending_dir = 0;
            self.pending_since = None;
            self.deep_since = None;
            self.committed_lane = None;
            self.needs_interior_lock = true;
            self.interior_lane = None;
            self.interior_since = None;
            self.approaching_dir = 0;
            self.approaching_until = f64::NEG_INFINITY;
            return self.u;
        }

        if self.was_map_unknown {
            self.was_map_unknown = false;
            self.relock_until = self.relock_until.max(t + RELOCK_AFTER_LOSS_S);
        }

        // Confidence-faded LPF camera bias
        let prob_w = if inp.min_inner_prob.is_finite() {
            ((inp.min_inner_prob - BIAS_PROB_LO) / (BIAS_PROB_HI - BIAS_PROB_LO)).clamp(0., 1.)
        } else {
            0.
        };
        let raw_bias = if inp.raw_bias.is_finite() { inp.raw_bias } else { 0. };
        let eff_bias = raw_bias * prob_w;
        let mut alpha_b = (dt / BIAS_TAU_S).min(1.);
        if t - self.last_ren_t <= RENUMBER_BIAS_HOLD_S {
            alpha_b *= 0.30;
        }
        self.bias_lp += alpha_b * (eff_bias - self.bias_lp);

        let map_u = inp.map_u.filter(|u| u.is_finite());
        let map_fresh = map_u.is_some();
        let z = map_u.map(|u| u + self.bias_lp);

        if !self.init {
            if let Some(z0) = z {
                self.u = z0;
                self.udot = 0.;
                self.p00 = 0.02;
                self.p01 = 0.;
                self.p11 = 0.15;
                self.init = true;
                if self.committed_lane.is_none()
                    && !self.needs_interior_lock
                    && t >= self.relock_until
                    && t >= self.turn_guard_until
                {
                    self.committed_lane = Some(self.u.floor() as i32);
                }
            } else {
                self.approaching_dir = 0;
                return self.u;
            }
        } else {
            // KF Predict
            let decay = (-dt / VEL_DECAY_TAU_S).exp();
            let u_pred = self.u + self.udot * dt;
            let udot_pred = self.udot * decay;
            let mut p00 = self.p00 + 2. * dt * self.p01 + dt * dt * self.p11 + Q_U * dt;
            let p01 = decay * (self.p01 + dt * self.p11);
            let p11 = decay * decay * self.p11 + Q_UDOT * dt;

            if ren_happened {
                p00 += RENUMBER_P_BOOST;
            }

            // KF Update with rate-invariant R and IRLS linear Huber gate
            if let Some(z_val) = z {
                let innov = z_val - u_pred;
                let rate_scale = 0.05 / dt.clamp(0.01, 0.10);
                let mut r = R_BASE * rate_scale;
                if !inp.min_inner_prob.is_finite() || inp.min_inner_prob < 0.25 {
                    r *= R_LOW_PROB_MULT;
                }
                if innov.abs() > HUBER_THRESH_U {
                    r *= innov.abs() / HUBER_THRESH_U;
                }
                let s_innov = p00 + r;
                let k0 = p00 / s_innov;
                let k1 = p01 / s_innov;
                self.u = u_pred + k0 * innov;
                self.udot = (udot_pred + k1 * innov).clamp(-MAX_UDOT, MAX_UDOT);
                self.p00 = (1. - k0) * p00;
                self.p01 = (1. - k0) * p01;
                self.p11 = (p11 - k1 * p01).max(1e-6);
            } else {
                self.u = u_pred;
                self.udot = udot_pred * 0.5;
                self.p00 = p00;
                self.p01 = p01;
                self.p11 = p11;
            }
        }

        // Guards
        if t < self.turn_guard_until
            || t < self.relock_until
            || !map_fresh
        {
            self.pending_dir = 0;
            self.pending_since = None;
            self.deep_since = None;
            self.approaching_dir = 0;
            return self.u;
        }

        // Initial / post-turn / post-loss interior lock
        let Some(lane) = self.committed_lane else {
            let fl = self.u.floor() as i32;
            let frac = self.u - f64::from(fl);
            if !self.needs_interior_lock {
                self.committed_lane = Some(fl);
            } else if (INTERIOR_MARGIN_U..=(1. - INTERIOR_MARGIN_U)).contains(&frac) {
                if self.interior_lane != Some(fl) {
                    self.interior_lane = Some(fl);
                    self.interior_since = Some(t);
                }
                if self.interior_since.is_some_and(|s| t - s >= INTERIOR_DWELL_S) {
                    self.committed_lane = Some(fl);
                    self.needs_interior_lock = false;
                    self.interior_lane = None;
                    self.interior_since = None;
                }
            } else {
                self.interior_lane = None;
                self.interior_since = None;
            }
            self.approaching_dir = 0;
            return self.u;
        };

        // Latched approach indicator with grace window so a brief pre-renumber bin plunge does not drop handoff
        let left_dist = self.u - f64::from(lane);
        let right_dist = f64::from(lane + 1) - self.u;
        self.approaching_dir = match self.approaching_dir {
            -1 if left_dist <= APPROACH_EXIT_DIST_U && (-self.udot >= APPROACH_KEEP_VEL || t <= self.approaching_until) => {
                if -self.udot >= APPROACH_KEEP_VEL {
                    self.approaching_until = t + APPROACH_GRACE_S;
                }
                -1
            }
            1 if right_dist <= APPROACH_EXIT_DIST_U && (self.udot >= APPROACH_KEEP_VEL || t <= self.approaching_until) => {
                if self.udot >= APPROACH_KEEP_VEL {
                    self.approaching_until = t + APPROACH_GRACE_S;
                }
                1
            }
            _ => {
                if left_dist <= APPROACH_ENTER_DIST_U && -self.udot >= APPROACH_ENTER_VEL {
                    self.approaching_until = t + APPROACH_GRACE_S;
                    -1
                } else if right_dist <= APPROACH_ENTER_DIST_U && self.udot >= APPROACH_ENTER_VEL {
                    self.approaching_until = t + APPROACH_GRACE_S;
                    1
                } else {
                    self.approaching_until = f64::NEG_INFINITY;
                    0
                }
            }
        };

        // Commit FSM
        let (d, depth) = if self.u < f64::from(lane) {
            (-1, f64::from(lane) - self.u)
        } else if self.u >= f64::from(lane + 1) {
            (1, self.u - f64::from(lane + 1))
        } else {
            self.pending_dir = 0;
            self.pending_since = None;
            self.deep_since = None;
            return self.u;
        };

        if d != self.pending_dir {
            self.pending_dir = d;
            self.pending_since = None;
            self.deep_since = None;
        }

        let in_cooldown_reverse = self
            .last_commit
            .is_some_and(|(ct, _, _)| t - ct < COOLDOWN_S)
            && d == -self.last_commit_dir;
        let ren_recent = t - self.last_ren_t <= REN_WINDOW_S;
        let vel_aligned = f64::from(d) * self.udot >= V_MIN_COMMIT;

        let mut commit = false;
        if in_cooldown_reverse {
            if depth >= H_REVERSE_COOLDOWN && vel_aligned {
                let since = *self.pending_since.get_or_insert(t);
                if t - since >= T_REVERSE_COOLDOWN_S {
                    commit = true;
                }
            } else {
                self.pending_since = None;
            }
        } else {
            if ren_recent && depth >= H_REN_COMMIT && f64::from(d) * self.udot >= 0. {
                let since = *self.pending_since.get_or_insert(t);
                if t - since >= T_REN_CONFIRM_S {
                    commit = true;
                }
            } else if depth >= H_COMMIT && vel_aligned {
                let since = *self.pending_since.get_or_insert(t);
                if t - since >= T_CONFIRM_S {
                    commit = true;
                }
            } else {
                self.pending_since = None;
            }

            if depth >= H_DEEP {
                let since = *self.deep_since.get_or_insert(t);
                if !commit && t - since >= T_DEEP_S {
                    commit = true;
                }
            } else {
                self.deep_since = None;
            }
        }

        if commit {
            let new_lane = lane + d;
            self.committed_lane = Some(new_lane);
            self.last_commit = Some((t, lane, new_lane));
            self.last_commit_dir = d;
            self.pending_dir = 0;
            self.pending_since = None;
            self.deep_since = None;
            self.approaching_dir = 0;
            self.approaching_until = f64::NEG_INFINITY;
        }

        self.u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp(t: f64, map_u: f64, bias: f64, prob: f64, yaw: f64, speed: f64, ren: u32) -> EgoLaneInput {
        EgoLaneInput {
            t,
            map_u: Some(map_u),
            raw_bias: bias,
            min_inner_prob: prob,
            yaw: Some(yaw),
            speed,
            renumbers: ren,
            map_unknown: false,
        }
    }

    #[test]
    fn abort_and_single_tick_spike_do_not_commit() {
        let mut f = EgoLaneFilter::default();
        let mut commits = 0;
        let mut prev = None;
        for i in 0..120 {
            let t = f64::from(i) * 0.05;
            let u = if t < 1. {
                0.5
            } else if t < 2.2 {
                0.5 + 0.55 * (t - 1.) / 1.2
            } else if t < 3.4 {
                1.05 - 0.55 * (t - 2.2) / 1.2
            } else {
                0.5
            };
            f.step(inp(t, u, 0., 0.8, 0., 15., 0));
            if f.last_commit() != prev {
                commits += 1;
                prev = f.last_commit();
            }
        }
        assert_eq!(commits, 0);

        let mut f2 = EgoLaneFilter::default();
        let mut commits2 = 0;
        let mut prev2 = None;
        for i in 0..80 {
            let t = f64::from(i) * 0.05;
            let u = if i == 40 { 0.80 } else { 1.15 };
            f2.step(inp(t, u, 0., 0.8, 0., 15., 0));
            if f2.last_commit() != prev2 {
                commits2 += 1;
                prev2 = f2.last_commit();
            }
        }
        assert_eq!(commits2, 0);
    }

    #[test]
    fn real_lane_change_with_bias_bounce_commits_once() {
        let mut f = EgoLaneFilter::default();
        let mut commits = Vec::new();
        let mut prev = None;
        for i in 0..140 {
            let t = f64::from(i) * 0.05;
            let (u, bias, ren) = if t < 1. {
                (0.5, 0., 0)
            } else if t < 2.5 {
                (0.5 + 0.4 * (t - 1.), 0., u32::from(t >= 2.25))
            } else if t < 2.7 {
                (1.05, -0.10, 1)
            } else {
                ((1.1 + 0.25 * (t - 2.7)).min(1.5), 0., 1)
            };
            f.step(inp(t, u, bias, 0.7, 0., 15., ren));
            if f.last_commit() != prev {
                commits.push(f.last_commit().unwrap());
                prev = f.last_commit();
            }
        }
        assert_eq!(commits.len(), 1, "{commits:?}");
        assert_eq!((commits[0].1, commits[0].2), (0, 1));
    }

    #[test]
    fn sharp_intersection_turn_stays_uncommitted_and_relocks_in_interior() {
        let mut f = EgoLaneFilter::default();
        let mut commits = 0;
        let mut prev = None;
        for i in 0..160 {
            let t = f64::from(i) * 0.05;
            let (u, yaw, speed) = if t < 2.0 {
                (0.5 - 0.35 * t, 0.55, 3.5)
            } else if t < 3.5 {
                (-0.05 + 0.08 * (t - 2.0), 0.05, 8.0)
            } else {
                (0.35, 0.01, 10.0)
            };
            f.step(inp(t, u, 0., 0.1, yaw, speed, 0));
            if f.last_commit() != prev {
                commits += 1;
                prev = f.last_commit();
            }
        }
        assert_eq!(commits, 0);
        assert_eq!(f.committed_lane(), Some(0));
    }
}
