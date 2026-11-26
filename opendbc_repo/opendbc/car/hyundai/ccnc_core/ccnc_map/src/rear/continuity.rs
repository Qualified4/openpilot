//! Short extrapolation across the side radar's gap around ego.

use crate::slots::Slot;
use crate::vehicles::Vehicle;

const MAX_AGE: f64 = 2.;
const MAX_BSD_AGE: f64 = 4.;
const MAX_STEP_TIME: f64 = 0.2;
const REJOIN_TIME: f64 = 0.25;

#[derive(Clone, Copy)]
pub(super) struct Alongside {
    measured: Slot,
    radar_id: Option<u32>,
    seen: f64,
    time: f64,
    x: f64,
    y: f64,
    v_rel: f64,
    ego: f64,
    rejoining: bool,
    rear: bool,
    crossing_since: Option<f64>,
}

impl Alongside {
    pub(super) fn vid(self) -> u32 {
        self.measured.vid
    }

    fn new(now: f64, ego: f64, measured: Slot, car: &Vehicle) -> Self {
        Self {
            measured,
            radar_id: car.radar_id,
            seen: now,
            time: now,
            x: f64::from(measured.distance),
            y: f64::from(measured.lateral),
            v_rel: f64::from(measured.v_rel),
            ego,
            rejoining: false,
            rear: measured.distance < 0.,
            crossing_since: None,
        }
    }

    pub(super) fn update(
        state: &mut Option<Self>,
        side: usize,
        now: f64,
        ego: f64,
        blind: bool,
        remembered: bool,
        selected: &[Option<Slot>; 5],
        cars: &[Vehicle],
        slots: &mut [Option<Slot>; 5],
    ) -> Option<usize> {
        let (front, rear) = (side + 1, side + 3);
        let measured = [selected[front], selected[rear]]
            .into_iter()
            .flatten()
            .filter_map(|slot| {
                cars.iter()
                    .find(|car| {
                        car.vid == slot.vid && car.radar_id.is_some() && car.seen_radar == now
                    })
                    .map(|car| (slot, car))
            });
        let nearest = measured
            .clone()
            .filter(|(s, _)| s.distance.abs() <= 6.)
            .min_by(|(a, _), (b, _)| a.distance.abs().total_cmp(&b.distance.abs()));

        let Some(mut motion) = state
            .take()
            .filter(|m| (0. ..=MAX_STEP_TIME).contains(&(now - m.time)))
        else {
            *state = nearest.map(|(s, car)| Self::new(now, ego, s, car));
            return None;
        };
        let observation = measured.clone().find(|(s, _)| s.vid == motion.measured.vid);
        let vid = motion.measured.vid;
        let live = cars
            .iter()
            .find(|car| car.vid == vid && car.seen_radar == now);
        // A new approaching front measurement needs its own bridge even if another car is already retreating.
        if motion.rear && observation.is_none() {
            if let Some((s, car)) = nearest.filter(|(s, _)| s.distance >= 0. && s.v_rel < -0.5) {
                *state = Some(Self::new(now, ego, s, car));
                return None;
            }
        }
        // Leave the handoff to another slot when the same vid is selected elsewhere,
        // and retire it if the tracker reused its radar id or it is now known to be an obstacle.
        if slots
            .iter()
            .enumerate()
            .any(|(i, slot)| i != front && i != rear && slot.is_some_and(|s| s.vid == vid))
            || live.is_some_and(|car| car.flags.structure || car.oncoming)
            || observation.is_none()
                && motion.radar_id.is_some()
                && cars.iter().any(|car| {
                    car.vid != vid && car.seen_radar == now && car.radar_id == motion.radar_id
                })
        {
            *state = nearest.map(|(s, car)| Self::new(now, ego, s, car));
            return None;
        }

        let dt = now - motion.time;
        let measured_age = now - motion.seen;
        let v_rel = (motion.v_rel + motion.ego - ego).clamp(-35., 25.);
        let predicted = motion.x + v_rel * dt;
        motion.time = now;
        if let Some((s, car)) = observation {
            let x = f64::from(s.distance);
            let y = f64::from(s.lateral);
            let crossed = (x >= 0.) != (motion.x >= 0.);
            motion.rejoining |= measured_age > dt + 0.01 || crossed;
            if motion.rejoining {
                // On reacquisition, decay the displacement from the prediction over several radar ticks.
                let weight = (dt / REJOIN_TIME).clamp(0., 1.);
                motion.x = predicted + (weight * (x - predicted)).clamp(-4. * dt, 4. * dt);
                motion.y += (weight * (y - motion.y)).clamp(-2.5 * dt, 2.5 * dt);
                if (x - motion.x).abs() < 0.10 && (y - motion.y).abs() < 0.10 {
                    motion.rejoining = false;
                }
            } else {
                motion.x = x;
                motion.y = y;
            }
            let slope = (x - f64::from(motion.measured.distance)) / measured_age.max(0.025);
            if measured_age <= MAX_STEP_TIME && slope.abs() <= 35. {
                motion.v_rel = 0.65 * v_rel + 0.35 * slope;
            } else {
                motion.v_rel = f64::from(s.v_rel);
            }
            motion.measured = s;
            motion.radar_id = car.radar_id;
            motion.seen = now;
        } else {
            let rear_supported = slots[rear].is_some_and(|s| s.vid == vid);
            // Before the zero crossing, coast only if the last motion was toward or through the radar gap.
            let approaching = (motion.measured.distance >= 0. && v_rel < -0.5)
                || (motion.measured.distance < 0. && v_rel > 0.5);
            let age_limit = if blind { MAX_BSD_AGE } else { MAX_AGE };
            if measured_age > age_limit
                || !(-20. ..=8.).contains(&predicted)
                || !(approaching || rear_supported || blind)
                || (motion.measured.distance > 5. && !rear_supported && !blind)
            {
                *state = nearest.map(|(s, car)| Self::new(now, ego, s, car));
                return None;
            }
            motion.x = predicted;
        }
        motion.ego = ego;
        if observation.is_some() && motion.x.abs() > 6. && !motion.rejoining {
            *state = nearest.map(|(s, car)| Self::new(now, ego, s, car));
            return None;
        }
        if (motion.x < 0.) != motion.rear {
            let since = *motion.crossing_since.get_or_insert(now);
            if motion.x.abs() >= 0.25 || v_rel.abs() >= 0.8 || now - since >= 0.20 {
                motion.rear = motion.x < 0.;
                motion.crossing_since = None;
            }
        } else {
            motion.crossing_since = None;
        }
        *state = Some(motion);

        if observation.is_some()
            && !motion.rejoining
            && slots.iter().any(|s| s.is_some_and(|s| s.vid == vid))
        {
            return None;
        }
        let split = motion.measured.distance >= 0.
            && slots[rear].is_some_and(|s| s.vid == vid && s.distance == 0.)
            && slots[front].is_none_or(|s| s.vid != vid);
        let owner = if motion.rear || split { rear } else { front };
        // A different measured rear car and a remembered stopped front car keep their own slots.
        if remembered && owner == front
            || slots[owner].is_some_and(|s| {
                s.vid != vid
                    && ((owner == front && (observation.is_some() || measured_age > 0.35))
                        || selected[rear].is_some_and(|measured| measured.vid == s.vid)
                        || (owner == rear
                            && observation.is_none()
                            && s.vid != 0
                            && (s.distance > -7. || v_rel > -4.)))
            })
        {
            return None;
        }
        // Clear only copies of this vehicle. An unassociated BSD drawing remains available on the other slot.
        for index in [front, rear] {
            if slots[index].is_some_and(|s| s.vid == vid) {
                slots[index] = None;
            }
        }
        if owner == front
            && motion.rejoining
            && slots[rear].is_some_and(|s| s.vid == 0)
            && motion.x <= 1.
        {
            slots[rear] = None;
        }
        slots[owner] = Some(Slot {
            distance: if split {
                motion.x.min(0.) as f32
            } else {
                motion.x as f32
            },
            lateral: motion.y as f32,
            v_rel: if observation.is_none() {
                v_rel as f32
            } else {
                motion.measured.v_rel
            },
            ..motion.measured
        });
        (observation.is_none() || motion.rejoining).then_some(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicles::Tracker;
    use crate::world::World;

    #[test]
    fn measured_vehicle_crosses_zero_once_in_each_direction_through_bsd() {
        for direction in [-1., 1.] {
            let mut world = World::default();
            let mut tracker = Tracker::default();
            let mut state = None;
            let mut last_x = -direction * 0.6;
            let mut indices = Vec::new();
            for i in 0..12 {
                let now = 1. + f64::from(i) * 0.05;
                let x = direction * (f64::from(i) * 0.1 - 0.6);
                let observing = i < 3 || i >= 8;
                world.car_state(now, 10., false);
                let point = (54, x, 3.5, direction * 2.);
                tracker.update(now, observing.then_some(point).as_slice(), &world);
                let car = &tracker.vehicles()[0];
                let slot = Slot {
                    vid: car.vid,
                    distance: x as f32,
                    lateral: if i >= 8 { 3.8 } else { 3.5 },
                    detect: 2,
                    lane_coord_rel: -1.,
                    v_rel: direction as f32 * 2.,
                    kind: 0,
                    kind_confirmed: false,
                    kind_wanted: false,
                    vision: 0,
                };
                let mut measured = [None; 5];
                let measured_index = if x < 0. { 3 } else { 1 };
                if observing {
                    measured[measured_index] = Some(slot);
                }
                let mut drawn = measured;
                if !observing || i == 8 {
                    drawn[3] = Some(Slot {
                        vid: 0,
                        distance: -10.,
                        ..slot
                    });
                }
                Alongside::update(
                    &mut state,
                    0,
                    now,
                    10.,
                    !observing || i == 8,
                    false,
                    &measured,
                    tracker.vehicles(),
                    &mut drawn,
                );
                let positions: Vec<_> = drawn
                    .iter()
                    .enumerate()
                    .filter_map(|(j, s)| s.filter(|s| s.vid == car.vid).map(|s| (j, s)))
                    .collect();
                assert_eq!(positions.len(), 1, "{i} {drawn:?}");
                let (index, s) = positions[0];
                if indices.last() != Some(&index) {
                    indices.push(index);
                }
                assert!((f64::from(s.distance) - last_x).abs() <= 0.12, "{i} {s:?}");
                if i == 8 {
                    assert!((3.5..3.8).contains(&s.lateral), "{s:?}");
                }
                last_x = f64::from(s.distance);
            }
            assert_eq!(indices, if direction < 0. { [1, 3] } else { [3, 1] });
        }
    }
}
