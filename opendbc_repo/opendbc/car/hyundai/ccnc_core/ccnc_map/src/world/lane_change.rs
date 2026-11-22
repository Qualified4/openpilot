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
