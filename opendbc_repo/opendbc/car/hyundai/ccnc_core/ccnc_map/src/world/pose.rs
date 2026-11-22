use std::collections::VecDeque;

// carState is 100 Hz. Leave room for the samples bracketing the two-second history.
const HISTORY_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Default)]
pub struct PoseAt {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub s: f64,
}

impl PoseAt {
    pub fn to_world(self, x: f64, y: f64) -> (f64, f64) {
        let (sin, cos) = self.heading.sin_cos();
        (self.x + cos * x - sin * y, self.y + sin * x + cos * y)
    }

    pub fn to_ego(self, x: f64, y: f64) -> (f64, f64) {
        let (sin, cos) = self.heading.sin_cos();
        let (dx, dy) = (x - self.x, y - self.y);
        (cos * dx + sin * dy, -sin * dx + cos * dy)
    }
}

pub struct Pose {
    history: VecDeque<PoseAt>,
    speed: f64,
    lane_yaw: f64,
    lane_heading: Option<(f64, f64)>,
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY_CAPACITY),
            speed: 0.,
            lane_yaw: 0.,
            lane_heading: None,
        }
    }
}

impl Pose {
    pub fn advance(&mut self, t: f64, v_ego: f64, yaw_live: Option<f64>) {
        if !t.is_finite() || !v_ego.is_finite() || self.stamp().is_some_and(|last| t <= last) {
            return;
        }
        let mut next = self.current();
        if let Some(last) = self.stamp() {
            let dt = (t - last).min(0.25);
            let distance = 0.5 * (self.speed + v_ego.max(0.)) * dt;
            let rotation = yaw_live
                .filter(|yaw| yaw.is_finite())
                .unwrap_or(self.lane_yaw)
                * dt;
            let (sin, cos) = (next.heading + 0.5 * rotation).sin_cos();
            next.x += distance * cos;
            next.y += distance * sin;
            next.heading += rotation;
            next.s += distance;
        }
        self.speed = v_ego.max(0.);
        next.t = t;
        if self.history.len() == HISTORY_CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(next);
        while self.history.len() > 1 && t - self.history[1].t >= 2. {
            self.history.pop_front();
        }
    }

    pub fn at(&self, t: f64) -> PoseAt {
        let i = self.history.partition_point(|p| p.t < t);
        if i == 0 || i == self.history.len() {
            return self
                .history
                .get(i)
                .copied()
                .unwrap_or_else(|| self.current());
        }
        let (a, b) = (self.history[i - 1], self.history[i]);
        let f = (t - a.t) / (b.t - a.t);
        let interp = |x, y| x + f * (y - x);
        PoseAt {
            t,
            x: interp(a.x, b.x),
            y: interp(a.y, b.y),
            heading: interp(a.heading, b.heading),
            s: interp(a.s, b.s),
        }
    }

    pub fn current(&self) -> PoseAt {
        self.history.back().copied().unwrap_or_default()
    }

    pub(crate) fn speed(&self) -> f64 {
        self.speed
    }

    pub fn to_world(&self, x: f64, y: f64) -> (f64, f64) {
        self.current().to_world(x, y)
    }

    pub(super) fn stamp(&self) -> Option<f64> {
        self.history.back().map(|p| p.t)
    }

    pub(super) fn observe_lane_heading(&mut self, t: f64, heading: f64, curvature: f64) {
        if let Some((last, angle)) = self.lane_heading {
            let dt = t - last;
            if dt <= 0. {
                return;
            }
            if dt > 0.01 && dt < 0.3 {
                let rate = self.speed * curvature - (heading - angle) / dt;
                self.lane_yaw += 0.3 * (rate.clamp(-0.5, 0.5) - self.lane_yaw);
            }
        }
        self.lane_heading = Some((t, heading));
    }

    pub(super) fn clear(&mut self) {
        self.history.clear();
        self.speed = 0.;
        self.lane_yaw = 0.;
        self.lane_heading = None;
    }
}
