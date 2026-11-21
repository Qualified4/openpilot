//! Display B (DESIGN 6.7, 11): each vehicle is drawn on its lane centre or on the line it straddles. A state changes
//! only after the estimated centre has stayed across a line for CROSS_HOLD (another lane) or within the straddle band
//! for BAND_HOLD (entering or leaving a straddle); the drawn value then moves to the new place at ANIM_SPEED and
//! turns round where it is when the state changes again, never jumping.

use std::collections::BTreeMap;

use super::common::{Common, FrameOut, LaneView, Vid};

pub const CROSS_HOLD: f64 = 2.0;
pub const BAND_HOLD: f64 = 1.0;
pub const ANIM_SPEED: f64 = 1.2;
/// A place whose target moves this much in one update was renumbered (ego changed lanes, the lane view now counts
/// from another lane): it becomes the place nearest its last target instead of jumping a lane.
pub const REMAP: f64 = 1.2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Place {
    Center(i64),
    Straddle(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vehicle {
    pub place: Place,
    pub value: f64,
    pub moving: bool,
    /// Since when the centre has been in another lane than a Center place, or in or out of the band.
    pub away: Option<(i64, f64)>,
    pub band: Option<(usize, f64)>,
    pub out: Option<f64>,
    pub now: f64,
    pub goal: f64,
}

fn target(view: &LaneView, place: Place) -> f64 {
    match place {
        Place::Center(lane) => view.center(lane),
        Place::Straddle(i) => view.boundaries[i],
    }
}

fn since<T: PartialEq + Copy>(memory: Option<(T, f64)>, value: T, now: f64) -> Option<(T, f64)> {
    match memory {
        Some((v, t)) if v == value => Some((v, t)),
        _ => Some((value, now)),
    }
}

impl Vehicle {
    fn new(view: &LaneView, y: f64, now: f64) -> Self {
        let place = match view.straddled(y) {
            Some(i) => Place::Straddle(i),
            None => Place::Center(view.lane(y)),
        };
        let goal = target(view, place);
        Vehicle { place, value: goal, moving: false, away: None, band: None, out: None, now, goal }
    }

    fn step(&mut self, view: &LaneView, y: f64, now: f64) {
        let dt = (now - self.now).max(0.);
        self.now = now;
        if (target(view, self.place) - self.goal).abs() > REMAP {
            let last = self.goal;
            let nearest = |n: usize, at: &dyn Fn(usize) -> f64| {
                (1..n).fold(0, |best, i| if (at(i) - last).abs() < (at(best) - last).abs() { i } else { best })
            };
            self.place = match self.place {
                Place::Center(_) => Place::Center(nearest(5, &|i| view.center(i as i64 - 2)) as i64 - 2),
                Place::Straddle(_) => Place::Straddle(nearest(4, &|i| view.boundaries[i])),
            };
            self.away = None;
            self.band = None;
            self.out = None;
        }
        let lane = view.lane(y);
        let band = view.straddled(y);
        let mut next = None;
        match self.place {
            Place::Center(current) => {
                self.out = None;
                self.away = if lane != current { since(self.away, lane, now) } else { None };
                self.band = band.and_then(|i| since(self.band, i, now));
                if let Some((l, t)) = self.away {
                    if now - t >= CROSS_HOLD - 1e-9 {
                        next = Some(Place::Center(l));
                    }
                }
                if next.is_none() {
                    if let Some((i, t)) = self.band {
                        if now - t >= BAND_HOLD - 1e-9 {
                            next = Some(Place::Straddle(i));
                        }
                    }
                }
            }
            Place::Straddle(current) => {
                self.away = None;
                self.band = band.filter(|&i| i != current).and_then(|i| since(self.band, i, now));
                self.out = if band.is_none() { Some(self.out.unwrap_or(now)) } else { None };
                if let Some(t) = self.out {
                    if now - t >= BAND_HOLD - 1e-9 {
                        next = Some(Place::Center(lane));
                    }
                }
                if let Some((i, t)) = self.band {
                    if now - t >= BAND_HOLD - 1e-9 {
                        next = Some(Place::Straddle(i));
                    }
                }
            }
        }
        if let Some(place) = next {
            self.place = place;
            self.moving = true;
            self.away = None;
            self.band = None;
            self.out = None;
        }
        let goal = target(view, self.place);
        if self.moving {
            let step = ANIM_SPEED * dt;
            let gap = goal - self.value;
            if gap.abs() <= step {
                self.value = goal;
                self.moving = false;
            } else {
                self.value += if gap > 0. { step } else { -step };
            }
        } else {
            self.value = goal;
        }
        self.goal = goal;
    }
}

#[derive(Clone, Debug, Default)]
pub struct DisplayB {
    pub vehicles: BTreeMap<Vid, Vehicle>,
}

impl DisplayB {
    /// Every tracked vehicle advances, drawn or not, so a vehicle keeps its place across a slot change.
    pub fn draw(&mut self, common: &Common, out: &FrameOut, now: f64) -> [Option<f64>; 3] {
        self.vehicles.retain(|vid, _| common.tracks.contains_key(vid));
        for (vid, track) in &common.tracks {
            let view = LaneView::new(&out.objs[vid].lines);
            match self.vehicles.get_mut(vid) {
                Some(v) => v.step(&view, track.y, now),
                None => {
                    self.vehicles.insert(*vid, Vehicle::new(&view, track.y, now));
                }
            }
        }
        out.assigned.map(|vid| vid.map(|vid| self.vehicles[&vid].value))
    }
}
