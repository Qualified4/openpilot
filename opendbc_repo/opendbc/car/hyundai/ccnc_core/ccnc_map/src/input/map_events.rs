//! Minimal readers for the state inputs to the CCNC map.

use super::live_tracks::{data_bytes, Message, Struct, EVENT_DISCRIMINANT_OFFSET};

pub const CAR_STATE: u16 = 21;
pub const LIVE_POSE: u16 = 127;
pub const RADAR_STATE: u16 = 12;

#[derive(Clone, Copy, Debug, Default)]
pub struct CarState {
    pub speed: f64,
    pub accel: f64,
    pub reverse: bool,
    pub blinker: bool,
    pub blind: [bool; 2],
    pub can_valid: bool,
    pub standstill: bool,
}

fn flag(data: &[u8], bit: usize) -> bool {
    data.get(bit / 8)
        .is_some_and(|byte| byte & (1 << (bit % 8)) != 0)
}

fn f32_at(data: &[u8], offset: usize) -> f64 {
    f64::from(f32::from_le_bytes(data_bytes(data, 4 * offset)))
}

pub struct Event<'a> {
    message: Message<'a>,
    root: Struct<'a>,
    pub stamp: u64,
    pub which: u16,
    pub valid: bool,
}

impl<'a> Event<'a> {
    pub fn read(bytes: &'a [u8]) -> Option<Self> {
        let message = Message::new(bytes)?;
        let root = message.read_struct(0, 0)?;
        // Event.valid defaults to true on the wire.
        let valid = !flag(root.data, 80);
        let stamp = u64::from_le_bytes(data_bytes(root.data, 0));
        let which = u16::from_le_bytes(data_bytes(root.data, EVENT_DISCRIMINANT_OFFSET * 2));
        Some(Self {
            message,
            root,
            stamp,
            which,
            valid,
        })
    }

    fn payload(&self) -> Option<Struct<'a>> {
        self.message.struct_pointer(&self.root, 0)
    }

    pub fn car_state(&self) -> Option<CarState> {
        let car = self.payload()?;
        let data = car.data;
        Some(CarState {
            speed: f32_at(data, 0),
            accel: f32_at(data, 7),
            reverse: u16::from_le_bytes(data_bytes(data, 5 * 2)) == 4,
            blinker: flag(data, 69) || flag(data, 70),
            blind: [flag(data, 352), flag(data, 353)],
            can_valid: flag(data, 74),
            standstill: flag(data, 67),
        })
    }

    pub fn yaw(&self) -> Option<f64> {
        let rate = self.message.struct_pointer(&self.payload()?, 3)?;
        let yaw = -f32_at(rate.data, 2);
        (flag(rate.data, 192) && yaw.is_finite()).then_some(yaw)
    }

    pub fn lead(&self) -> Option<(f64, f64)> {
        let lead = self.message.struct_pointer(&self.payload()?, 1)?;
        let (x, y) = (f32_at(lead.data, 0), f32_at(lead.data, 1));
        (flag(lead.data, 321) && x.is_finite() && y.is_finite()).then_some((x, y))
    }

    /// The lead card drew in FF before the map (`_display_lead`, 93c5c6af): the nearer of leadOne and leadTwo with
    /// status on, dRel above 0 and dRel, yRel, vRel finite, leadOne on a tie. (dRel, yRel, vRel).
    pub fn display_lead(&self) -> Option<(f64, f64, f64)> {
        let payload = self.payload()?;
        let lead = |pointer| {
            let lead = self.message.struct_pointer(&payload, pointer)?;
            let (x, y, v) = (f32_at(lead.data, 0), f32_at(lead.data, 1), f32_at(lead.data, 2));
            (flag(lead.data, 321) && x > 0. && [x, y, v].iter().all(|v| v.is_finite())).then_some((x, y, v))
        };
        match (lead(1), lead(2)) {
            (Some(one), Some(two)) => Some(if two.0 < one.0 { two } else { one }),
            (one, two) => one.or(two),
        }
    }
}
