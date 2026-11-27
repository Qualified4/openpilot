use ccnc_map::{
    cmap,
    cmap::MapOut,
    pipeline::STALE_NS,
    slots::Slot,
    vision::kind::{KIND_BUS, KIND_TRUCK},
};
use pyo3::{intern, prelude::*, types::PyDict};

use crate::{
    guard,
    vehicle_display::{FRONT_FIELDS, REAR_FIELDS},
};

/// After falling back to mode 0, the map has to be healthy this long before card draws from it again.
const RECOVERY_NS: u64 = 2_000_000_000;
/// While drawing from the map, a message that is invalid, late or missing for at most this long since the last valid
/// one keeps that one's slots instead of falling back.
const GRACE_NS: u64 = 250_000_000;
// Card publishes the display every 50 ms. Limit a step to 16 m/s on that clock.
const REAR_MAX_STEP: f64 = 0.8;
const REAR_ALPHA: f64 = 0.05 / (0.20 + 0.05);

fn valid_map(bytes: &[u8]) -> Option<MapOut> {
    let map = cmap::decode(bytes)?;
    (map.valid
        && map.ego_lane_coord.is_finite()
        && map
            .slots
            .iter()
            .flatten()
            .all(|slot| slot.distance.is_finite() && slot.lateral.is_finite() && slot.detect <= 31))
    .then_some(map)
}

#[pyclass]
#[derive(Default)]
pub(crate) struct MapLink {
    /// The last message read and what it held: card hands the same message over at every tick until a new one comes.
    decoded: Option<(Vec<u8>, Option<MapOut>)>,
    last: Option<(u64, u32)>,
    healthy_since: Option<u64>,
    recovering: bool,
    /// The last valid map card drew from, and its message time.
    held: Option<(u64, MapOut)>,
    rear_distance: [Option<f64>; 2],
    #[pyo3(get)]
    using_process: bool,
}

#[pymethods]
impl MapLink {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (now_ns, msg, msg_mono_ns))]
    fn use_process(&mut self, now_ns: u64, msg: Option<&[u8]>, msg_mono_ns: Option<u64>) -> bool {
        self.select(now_ns, msg, msg_mono_ns).is_some()
    }
}

impl MapLink {
    fn read(&mut self, msg: &[u8]) -> Option<MapOut> {
        if self.decoded.as_ref().is_none_or(|(bytes, _)| bytes.as_slice() != msg) {
            self.decoded = Some((msg.to_vec(), valid_map(msg)));
        }
        self.decoded.as_ref().and_then(|(_, map)| *map)
    }

    fn select(&mut self, now: u64, msg: Option<&[u8]>, stamp: Option<u64>) -> Option<MapOut> {
        let fresh = |time| now.checked_sub(time).is_some_and(|age| age <= STALE_NS);
        let map = msg.and_then(|msg| self.read(msg));
        let candidate = stamp.filter(|stamp| fresh(*stamp)).zip(map);
        let Some((stamp, map)) = candidate.filter(|(stamp, map)| {
            self.last.is_none_or(|(last, _)| *stamp >= last)
                && map.radar_log_mono_time <= *stamp
                && fresh(map.radar_log_mono_time)
        }) else {
            if let Some((_, map)) = self
                .held
                .filter(|(stamp, _)| self.using_process && now.saturating_sub(*stamp) <= GRACE_NS)
            {
                return Some(map);
            }
            self.held = None;
            self.last = None;
            self.healthy_since = None;
            self.recovering = true;
            self.using_process = false;
            self.rear_distance = [None; 2];
            return None;
        };

        if let Some((last, generation)) = self.last {
            if stamp - last > STALE_NS {
                self.recovering = true;
                self.healthy_since = None;
            }
            if map.generation != generation {
                self.healthy_since = None;
            }
        }
        self.last = Some((stamp, map.generation));
        if self.recovering {
            let since = *self.healthy_since.get_or_insert(now);
            self.recovering = now.saturating_sub(since) < RECOVERY_NS;
        }
        self.using_process = !self.recovering;
        self.held = self.using_process.then_some((stamp, map));
        if !self.using_process {
            self.rear_distance = [None; 2];
        }
        self.using_process.then_some(map)
    }

    fn draw_rear_distances(&mut self, map: &MapOut) -> [f64; 2] {
        std::array::from_fn(|side| {
            let (target, _, code) = slot_values(map, side + 3);
            let animated = map.status & cmap::STATUS_REAR_ANIMATED[side] != 0;
            let shown = self.rear_distance[side].map_or(target, |previous| {
                // Mapd has already interpolated an origin crossing or a confirmed rear pass.
                let alpha = if animated { 1. } else { REAR_ALPHA };
                previous + (alpha * (target - previous)).clamp(-REAR_MAX_STEP, REAR_MAX_STEP)
            });
            self.rear_distance[side] = (code != 0).then_some(shown);
            if code == 0 { 0. } else { shown }
        })
    }
}

pub(crate) fn from_card(
    cs: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    now: u64,
    draw: bool,
) -> PyResult<Option<(MapOut, [f64; 2])>> {
    let py = cs.py();
    let attr = intern!(py, "_rust_map_link");
    let holder = match state.getattr_opt(attr)? {
        Some(value) => value.cast_into::<MapLink>()?,
        None => {
            let holder = Bound::new(py, MapLink::new())?;
            state.setattr(attr, &holder)?;
            holder
        }
    };
    let input = cs.getattr_opt(intern!(py, "ccnc_map_bytes"))?;
    let (bytes, stamp) = input
        .as_ref()
        .and_then(|value| value.extract::<(Option<&[u8]>, Option<u64>)>().ok())
        .unwrap_or_default();
    let mut link = holder.try_borrow_mut()?;
    let was_using = link.using_process;
    let output = link.select(now, bytes, stamp);
    if link.using_process != was_using {
        let message = if link.using_process {
            "CCNC display using ccnc_mapd"
        } else {
            "CCNC display using mode 0 fallback"
        };
        PyModule::import(py, intern!(py, "opendbc.car.carlog"))?
            .getattr(intern!(py, "carlog"))?
            .call_method1(intern!(py, "info"), (message,))?;
    }
    Ok(output.map(|map| {
        let distances = if draw { link.draw_rear_distances(&map) } else { [0.; 2] };
        (map, distances)
    }))
}

/// DETECT for a slot: the map's 1 (moving away) / 2 (closing or slow) become the grey / white car (3 / 4), or the
/// grey / white truck (5 / 6) once the map has confirmed a truck or bus there (the cluster has no bus shape).
/// Anything else goes as it is.
fn detect_code(slot: &Slot) -> u8 {
    match slot.detect {
        detect @ (1 | 2) if slot.kind_confirmed && matches!(slot.kind, KIND_TRUCK | KIND_BUS) => detect + 4,
        detect @ (1 | 2) => detect + 2,
        detect => detect,
    }
}

fn slot_position(index: usize, slot: &Slot) -> (f64, f64) {
    // Display A supplies lateral position. The OEM reads FF as signed even though the DBC does not.
    let y = f64::from(slot.lateral);
    let lateral = match index {
        0 => (-y).clamp(-6.4, 6.3),
        1 | 3 => y.clamp(0., 12.7),
        _ => (-y).clamp(0., 12.7),
    };
    let x = f64::from(slot.distance);
    let distance = if index < 3 {
        (0.8 * x).clamp(0., 204.7)
    } else {
        (0.8 * (-x).max(0.)).min(25.5)
    };
    (distance, lateral)
}

fn slot_values(map: &MapOut, index: usize) -> (f64, f64, u8) {
    if let Some(slot) = map.slots[index] {
        let (x, y) = slot_position(index, &slot);
        return (x, y, detect_code(&slot));
    }
    let warning = match index {
        3 => map.status & cmap::STATUS_BSD_LEFT != 0,
        4 => map.status & cmap::STATUS_BSD_RIGHT != 0,
        _ => false,
    };
    // The 8 m BSD warning uses the white passenger car shape.
    if warning { (8., 2.8, 4) } else { (0., 0., 0) }
}

pub(crate) fn apply_slots(
    values: &Bound<'_, PyDict>, map: &MapOut, rear_distances: Option<[f64; 2]>,
) -> PyResult<()> {
    for (index, [distance, lateral, detect]) in FRONT_FIELDS.into_iter().chain(REAR_FIELDS).enumerate() {
        let (x, y, code) = slot_values(map, index);
        let x = rear_distances.and_then(|rear| index.checked_sub(3).map(|side| rear[side])).unwrap_or(x);
        values.set_item(distance, x)?;
        values.set_item(lateral, y)?;
        values.set_item(detect, code)?;
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn apply_cmap(values_162: &Bound<'_, PyDict>, msg: &[u8]) -> PyResult<()> {
    guard(|| {
        if let Some(map) = valid_map(msg) {
            apply_slots(values_162, &map, None)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccnc_map::vision::kind::KIND_CAR;

    fn slot(detect: u8, kind: u8, kind_confirmed: bool) -> Slot {
        Slot { vid: 4, distance: 12., lateral: 0.5, detect, lane_coord_rel: 0., v_rel: 0., kind, kind_confirmed, kind_wanted: false, vision: 0 }
    }

    const TICK: u64 = 50_000_000;

    #[test]
    fn map_positions_saturate_at_can_field_limits() {
        let mut cell = slot(2, KIND_CAR, true);
        cell.distance = 300.;
        for (value, expected) in [(6.4, 6.3), (-6.51, -6.4), (-9.85, -6.4)] {
            cell.lateral = -value;
            assert_eq!(slot_position(0, &cell), (204.7, expected));
        }
        cell.lateral = 15.;
        assert_eq!(slot_position(1, &cell), (204.7, 12.7));
        assert_eq!(slot_position(2, &cell), (204.7, 0.));
        assert_eq!(slot_position(3, &cell), (0., 12.7));
        cell.lateral = -15.;
        cell.distance = -300.;
        assert_eq!(slot_position(2, &cell), (0., 12.7));
        assert_eq!(slot_position(4, &cell), (25.5, 12.7));
    }

    #[test]
    fn rear_slot_crosses_the_origin_before_showing_rear_distance() {
        let mut cell = slot(2, KIND_CAR, true);
        for (x, expected) in [(6.4, 0.), (0.2, 0.), (0., 0.), (-5., 4.), (-10., 8.)] {
            cell.distance = x;
            for index in [3, 4] {
                assert_eq!(slot_position(index, &cell).0, expected);
            }
        }
    }

    #[test]
    fn rear_distance_converges_between_bsd_and_a_split_car_on_both_sides() {
        for (side, warning) in [cmap::STATUS_BSD_LEFT, cmap::STATUS_BSD_RIGHT].into_iter().enumerate() {
            let mut link = MapLink::new();
            let mut map = MapOut { status: warning, ..valid_map(&message(TICK, true)).unwrap() };
            assert_eq!(link.draw_rear_distances(&map)[side], 8.);

            let mut car = slot(2, KIND_CAR, true);
            car.distance = 4.; // The nearer vehicle of a pair displays at 0 m.
            map.slots[side + 3] = Some(car);
            let mut previous = 8.;
            for _ in 0..40 {
                let x = link.draw_rear_distances(&map)[side];
                assert!(x < previous && previous - x <= REAR_MAX_STEP + 1e-9);
                previous = x;
            }
            assert!(previous < 0.01);

            map.slots[side + 3] = None;
            let rising = link.draw_rear_distances(&map)[side];
            assert!(rising > previous && rising - previous <= REAR_MAX_STEP + 1e-9);
            map.status = 0;
            assert_eq!(link.draw_rear_distances(&map)[side], 0.);
            map.status = warning;
            assert_eq!(link.draw_rear_distances(&map)[side], 8.);
        }
    }

    fn message(stamp: u64, valid: bool) -> Vec<u8> {
        let slots = [Some(slot(2, KIND_CAR, true)), None, None, None, None];
        cmap::encode(&MapOut { radar_log_mono_time: stamp, generation: 1, status: 0, ego_lane_coord: 0.5, slots, valid })
    }

    /// card's link after `healthy` map messages a tick apart, then `invalid` invalid ones, then valid ones again:
    /// whether it drew from the map at each later tick.
    fn after_invalid(healthy: u64, invalid: u64, then: u64) -> Vec<bool> {
        let mut link = MapLink::new();
        let mut drawn = Vec::new();
        for i in 0..healthy + invalid + then {
            let stamp = 1_000_000_000 + i * TICK;
            let used = link.select(stamp + 1_000_000, Some(&message(stamp, i < healthy || i >= healthy + invalid)), Some(stamp));
            if i >= healthy {
                drawn.push(used.is_some_and(|map| map.slots[0].is_some()));
            }
        }
        drawn
    }

    #[test]
    fn a_short_invalid_stretch_keeps_the_last_valid_slots() {
        let healthy = RECOVERY_NS / TICK + 2;
        for invalid in 1..=4 {
            assert!(after_invalid(healthy, invalid, 20).iter().all(|&drawn| drawn), "{invalid} invalid messages");
        }
    }

    #[test]
    fn a_longer_invalid_stretch_falls_back_until_the_map_is_healthy_again() {
        let healthy = RECOVERY_NS / TICK + 2;
        let ticks = RECOVERY_NS / TICK;
        let drawn = after_invalid(healthy, 6, ticks + 5);
        // Four invalid messages are within 0.25 s of the last valid one (card reads each 1 ms after it).
        assert!(drawn[..4].iter().all(|&d| d), "{drawn:?}");
        assert!(drawn[4..6 + ticks as usize].iter().all(|&d| !d), "mode 0 until healthy for RECOVERY_NS: {drawn:?}");
        assert!(drawn[6 + ticks as usize..].iter().all(|&d| d), "{drawn:?}");
    }

    #[test]
    fn a_confirmed_truck_or_bus_is_drawn_as_a_truck() {
        for detect in [1, 2] {
            // No kind, a kind that is not confirmed, or a car: the car shape, as before kinds.
            for (kind, confirmed) in [(0, false), (0, true), (KIND_CAR, true), (KIND_TRUCK, false), (KIND_BUS, false)] {
                assert_eq!(detect_code(&slot(detect, kind, confirmed)), detect + 2, "{detect} {kind} {confirmed}");
            }
            assert_eq!(detect_code(&slot(detect, KIND_TRUCK, true)), detect + 4);
            assert_eq!(detect_code(&slot(detect, KIND_BUS, true)), detect + 4);
        }
        for detect in [0, 3, 7, 31] {
            assert_eq!(detect_code(&slot(detect, KIND_TRUCK, true)), detect, "other values go as they are");
        }
    }
}
