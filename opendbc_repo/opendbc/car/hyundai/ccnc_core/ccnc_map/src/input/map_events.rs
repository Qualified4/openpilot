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

#[derive(Clone, Copy, Debug)]
pub struct DisplayLead {
    pub motion: (f64, f64, f64),
    pub(crate) measured_track: Option<u32>,
    pub(crate) model_prob: f64,
}

impl DisplayLead {
    fn read(data: &[u8]) -> Option<Self> {
        let (x, y, v) = (f32_at(data, 0), f32_at(data, 1), f32_at(data, 2));
        if !flag(data, 321) || x <= 0. || ![x, y, v].iter().all(|v| v.is_finite()) {
            return None;
        }
        // RadarState.LeadData.radarTrackId defaults to -1, so the Cap'n Proto field is XOR encoded.
        let id = !i32::from_le_bytes(data_bytes(data, 13 * 4));
        let track = (id > 0 && flag(data, 322)).then_some(id as u32);
        let probability = f32_at(data, 12);
        Some(Self {
            motion: (x, y, v),
            measured_track: track,
            model_prob: probability,
        })
    }
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

    /// Pitch angle from LivePose.orientationNED.y in rad (nose-up positive).
    pub fn pitch(&self) -> Option<f64> {
        let orient = self.message.struct_pointer(&self.payload()?, 0)?;
        let pitch = f32_at(orient.data, 1);
        (flag(orient.data, 192) && pitch.is_finite()).then_some(pitch)
    }


    pub fn primary_display_lead(&self) -> Option<DisplayLead> {
        let payload = self.payload()?;
        DisplayLead::read(self.message.struct_pointer(&payload, 1)?.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_positive_measured_radar_id_identifies_the_lead() {
        let mut data = [0; 56];
        data[..4].copy_from_slice(&4.15f32.to_le_bytes());
        for (id, radar, probability, expected) in [
            (57, true, 0.99f32, Some(57)),
            (0, true, 0.99, None),
            (57, false, 0.99, None),
            (57, true, 0.5, Some(57)),
            (-1, true, 0.99, None),
        ] {
            data[40] = 0b10 | ((radar as u8) << 2);
            data[48..52].copy_from_slice(&probability.to_le_bytes());
            data[52..56].copy_from_slice(&(!id as i32).to_le_bytes());
            assert_eq!(DisplayLead::read(&data).unwrap().measured_track, expected);
        }
    }
}
