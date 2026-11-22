use crate::world::PoseAt;

use super::Vehicle;

const AMBIGUITY: f64 = 0.3;
const COVER_LAT: f64 = 0.8;
const TIME_EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub struct LinkGates {
    lon: f64,
    speed: f64,
    gap: f64,
}

impl LinkGates {
    pub const BODY: Self = Self {
        lon: 6.,
        speed: 3.,
        gap: 0.8,
    };
    pub const SHORTER: Self = Self {
        lon: 4.5,
        ..Self::BODY
    };
    pub const CONSERVATIVE: Self = Self {
        lon: 2.5,
        speed: 1.5,
        gap: 0.55,
    };

    pub(super) fn same_id(dt: f64, (dx, dy, dv): (f64, f64, f64)) -> bool {
        dt <= Self::BODY.gap + TIME_EPS && dx.abs() <= 6. && dy.abs() <= 1.5 && dv.abs() <= 2.5
    }

    fn score(self, (dx, dy, dv): (f64, f64, f64)) -> Option<f64> {
        let (x, y, v) = (dx.abs(), dy.abs(), dv.abs());
        (x <= self.lon && y <= 1. && v <= self.speed).then(|| x / self.lon + y + v / self.speed)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Sample {
    pub id: u32,
    pub wx: f64,
    pub wy: f64,
    pub speed: f64,
}

impl Sample {
    pub fn from_point(
        (id, x, y, v_rel): (u32, f64, f64, f64),
        pose: PoseAt,
        speed: f64,
    ) -> Option<Self> {
        let (wx, wy) = pose.to_world(x, y);
        let speed = v_rel + speed;
        [wx, wy, speed]
            .iter()
            .all(|v| (*v as f32).is_finite())
            .then_some(Self { id, wx, wy, speed })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Axes {
    pub cos: f64,
    pub sin: f64,
}

impl Axes {
    pub fn along(self, x: f64, y: f64) -> f64 {
        self.cos * x + self.sin * y
    }

    pub fn across(self, x: f64, y: f64) -> f64 {
        -self.sin * x + self.cos * y
    }

    pub fn error(self, car: &Vehicle, sample: Sample, dt: f64) -> (f64, f64, f64) {
        let (vx, vy) = (car.vx as f64, car.vy as f64);
        let dx = sample.wx - (car.wx as f64 + vx * dt);
        let dy = sample.wy - (car.wy as f64 + vy * dt);
        (
            self.along(dx, dy),
            self.across(dx, dy),
            sample.speed - self.along(vx, vy),
        )
    }
}

#[derive(Clone, Copy)]
pub(super) enum Decision {
    Start,
    Ambiguous,
    Handoff(usize),
}

pub(super) fn handoff(
    gates: LinkGates,
    axes: Axes,
    now: f64,
    sample: Sample,
    cars: &[Vehicle],
) -> Decision {
    let mut best = None;
    let mut best_score = f64::INFINITY;
    let mut next_score = f64::INFINITY;
    let mut covered = false;
    for (i, car) in cars
        .iter()
        .enumerate()
        .filter(|(_, car)| car.radar_id.is_some())
    {
        let dt = now - car.seen_radar;
        let error = axes.error(car, sample, dt);
        let Some(score) = gates.score(error) else {
            continue;
        };
        if dt == 0. {
            covered |= error.1.abs() <= COVER_LAT;
        } else if (0. ..=gates.gap + TIME_EPS).contains(&dt) {
            if score < best_score {
                best = Some(i);
                next_score = best_score;
                best_score = score;
            } else {
                next_score = next_score.min(score);
            }
        }
    }
    match best {
        None => Decision::Start,
        Some(_) if covered || next_score - best_score < AMBIGUITY => Decision::Ambiguous,
        Some(i) => Decision::Handoff(i),
    }
}
