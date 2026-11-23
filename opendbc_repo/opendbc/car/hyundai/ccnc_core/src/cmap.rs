use ccnc_map::{cmap, cmap::MapOut, pipeline::STALE_NS};
use pyo3::{intern, prelude::*, types::PyDict};

use crate::{
    guard,
    vehicle_display::{FRONT_FIELDS, REAR_FIELDS},
};

const RECOVERY_NS: u64 = 5_000_000_000;

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
            self.last = None;
            self.healthy_since = None;
            self.recovering = true;
            self.using_process = false;
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
        self.using_process.then_some(map)
    }
}

pub(crate) fn from_card(
    cs: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    now: u64,
) -> PyResult<Option<MapOut>> {
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
    Ok(output)
}

/// Every slot is rewritten from the message, so a new generation leaves nothing of the old one behind.
pub(crate) fn apply_slots(values: &Bound<'_, PyDict>, map: &MapOut) -> PyResult<()> {
    for (index, ([distance, lateral, detect], slot)) in FRONT_FIELDS
        .into_iter()
        .chain(REAR_FIELDS)
        .zip(map.slots)
        .enumerate()
    {
        let (x, y, code) = slot.map_or((0., 0., 0), |slot| {
            // The map has already computed Display A's lateral position.
            let signed_y = f64::from(slot.lateral);
            let y = match index {
                0 => -signed_y,
                1 | 3 => signed_y.clamp(0., 12.7),
                _ => (-signed_y).clamp(0., 12.7),
            };
            let (x, max) = if index < 3 {
                (f64::from(slot.distance), 204.7)
            } else {
                (f64::from(slot.distance).abs(), 25.5)
            };
            let code = if matches!(slot.detect, 1 | 2) {
                slot.detect + 2
            } else {
                slot.detect
            };
            ((0.8 * x).clamp(0., max), y, code)
        });
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
            apply_slots(values_162, &map)?;
        }
        Ok(())
    })
}
