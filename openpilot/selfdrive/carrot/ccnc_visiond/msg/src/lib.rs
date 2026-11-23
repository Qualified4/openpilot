//! CcncVision as a cereal Event (`customReserved4`), written and read without a compiled schema.
//!
//! `ccnc_visiond` encodes with this crate and the reader side decodes with it, so both use the offsets below.
//! They are the positions Cap'n Proto assigns to `CustomReserved4` in `openpilot/cereal/custom.capnp`.

mod reader;

pub use reader::{Message, Struct};

pub const SERVICE: &str = "customReserved4";

/// `Event` union member numbers (the discriminant value at 16-bit offset 4 of the data section).
pub const EVENT_LIVE_CALIBRATION: u16 = 18;
pub const EVENT_CAR_STATE: u16 = 21;
pub const EVENT_CCNC_MAP: u16 = 108;
pub const EVENT_CCNC_VISION: u16 = 109;
/// `LiveCalibrationData.Status.calibrated`.
pub const CAL_STATUS_CALIBRATED: u16 = 1;
// CcncMap (customReserved3) layout, matching ccnc_map::cmap::schema().
const MAP_VALID_BIT: usize = 104;
const MAP_SLOT_PRESENT_BIT: usize = 113;
const MAP_SLOT_VISION_BYTE: usize = 15;

const EVENT_WORDS: usize = 2;
const EVENT_DISCRIMINANT_BYTE: usize = 8;
const EVENT_INVALID_BIT: usize = 80;
const VISION_WORDS: usize = 7;
const BOX_WORDS: usize = 3;
// Byte offsets inside the data section, except bits where named so.
const FRAME_ID: usize = 0;
const VIEW: usize = 4;
const STATUS: usize = 6;
const NIGHT_CORRECTED_BIT: usize = 40;
const TIMESTAMP_SOF: usize = 8;
const PROCESSING_MS: usize = 16;
/// rpyCalib roll, pitch, yaw, then wideFromDeviceEuler roll, pitch, yaw, then height: seven f32 from here.
const CALIBRATION: usize = 20;
const FRAME_MONO_TIME: usize = 48;
// LiveCalibrationData: calStatus (enum, u16) in the data section, the rest are List(Float32) pointers.
const CAL_STATUS: usize = 2;
const CAL_RPY_CALIB: usize = 4;
const CAL_WIDE_FROM_DEVICE_EULER: usize = 6;
const CAL_HEIGHT: usize = 7;
const BOX_XYXY: usize = 0;
const BOX_CONFIDENCE: usize = 16;
const BOX_CLASS_ID: usize = 20;
const BOX_CUT_BIT: usize = 168;

/// The image a box is measured in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum View {
    /// The whole road camera frame. Reserved for the front detector.
    Full = 0,
    /// The road camera frame cropped above the bonnet. Reserved for the front detector.
    CropFront = 1,
    /// Wide camera, virtual view to the left (512x384).
    SideL = 2,
    /// Wide camera, virtual view to the right (512x384).
    SideR = 3,
}

impl View {
    fn from_wire(value: u8) -> Option<Self> {
        [Self::Full, Self::CropFront, Self::SideL, Self::SideR]
            .into_iter()
            .find(|view| *view as u8 == value)
    }
}

/// What a message reports. Only `Frame` carries a frame and its boxes; the others come about once a second while
/// nothing is processed, so a reader can tell a resting process from a stopped one. No message at all means the
/// process is not running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Frame = 0,
    /// Speed above the side view limit, reversing, or no recent carState.
    Resting = 1,
    NotCalibrated = 2,
    NoCamera = 3,
    /// Stopped on an error until the process restarts (the reason is in the `.halted` file).
    Halted = 4,
    /// The wide camera is not the 1344x760 one the lens constants are for.
    UnsupportedCamera = 5,
}

impl Status {
    fn from_wire(value: u8) -> Option<Self> {
        [Self::Frame, Self::Resting, Self::NotCalibrated, Self::NoCamera, Self::Halted, Self::UnsupportedCamera]
            .into_iter()
            .find(|status| *status as u8 == value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionBox {
    /// x1, y1, x2, y2 in pixels of the view.
    pub xyxy: [f32; 4],
    pub confidence: f32,
    /// Detector class index (COCO).
    pub class_id: u8,
    /// The box bottom is the crop line of a cropped view. Side views are not cropped.
    pub cut: bool,
}

/// The liveCalibration values a side view is built from (radians, meters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub rpy_calib: [f32; 3],
    pub wide_from_device_euler: [f32; 3],
    pub height: f32,
}

impl Calibration {
    /// Marks a frame whose view did not come from a calibration (fixed maps of a replay check).
    pub const UNKNOWN: Calibration = Calibration { rpy_calib: [f32::NAN; 3], wide_from_device_euler: [f32::NAN; 3], height: f32::NAN };

    fn values(&self) -> [f32; 7] {
        let [a, b, c] = self.rpy_calib;
        let [d, e, f] = self.wide_from_device_euler;
        [a, b, c, d, e, f, self.height]
    }

    fn from_values(v: [f32; 7]) -> Self {
        Calibration { rpy_calib: [v[0], v[1], v[2]], wide_from_device_euler: [v[3], v[4], v[5]], height: v[6] }
    }
}

/// A liveCalibration message as sent: the lists are not checked here.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveCalibration {
    pub cal_status: u16,
    pub rpy_calib: Vec<f32>,
    pub wide_from_device_euler: Vec<f32>,
    pub height: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionOut {
    pub status: Status,
    pub frame_id: u32,
    /// Start of frame on the camera's clock (CLOCK_BOOTTIME).
    pub timestamp_sof: u64,
    /// The same instant on CLOCK_MONOTONIC, the clock of `logMonoTime` here and of carState and liveTracks.
    pub frame_mono_time: u64,
    pub view: View,
    /// Frame copy to publish, in milliseconds.
    pub processing_ms: f32,
    pub night_corrected: bool,
    /// The angles the frame's side view was built from, and the latest valid height.
    pub calibration: Calibration,
    pub boxes: Vec<VisionBox>,
}

fn struct_pointer(data_words: usize, pointers: usize) -> u64 {
    ((data_words as u64) << 32) | ((pointers as u64) << 48)
}

fn put<const N: usize>(bytes: &mut [u8], at: usize, value: [u8; N]) {
    bytes[at..at + N].copy_from_slice(&value);
}

pub fn encode(out: &VisionOut, log_mono_time: u64) -> Vec<u8> {
    // One segment: root pointer, Event, CcncVision, then the inline-composite box list.
    const EVENT: usize = 2;
    const VISION: usize = EVENT + EVENT_WORDS + 1;
    const LIST: usize = VISION + VISION_WORDS + 1;
    let list_words = out.boxes.len() * BOX_WORDS;
    let words = LIST + 1 + list_words;
    let mut bytes = vec![0; words * 8];
    put(&mut bytes, 4, ((words - 1) as u32).to_le_bytes());
    put(&mut bytes, 8, struct_pointer(EVENT_WORDS, 1).to_le_bytes());

    let event = &mut bytes[EVENT * 8..VISION * 8];
    put(event, 0, log_mono_time.to_le_bytes());
    put(event, EVENT_DISCRIMINANT_BYTE, EVENT_CCNC_VISION.to_le_bytes());
    put(event, EVENT_WORDS * 8, struct_pointer(VISION_WORDS, 1).to_le_bytes());

    let vision = &mut bytes[VISION * 8..LIST * 8];
    put(vision, FRAME_ID, out.frame_id.to_le_bytes());
    vision[VIEW] = out.view as u8;
    vision[STATUS] = out.status as u8;
    put(vision, FRAME_MONO_TIME, out.frame_mono_time.to_le_bytes());
    vision[NIGHT_CORRECTED_BIT / 8] |= u8::from(out.night_corrected) << (NIGHT_CORRECTED_BIT % 8);
    put(vision, TIMESTAMP_SOF, out.timestamp_sof.to_le_bytes());
    put(vision, PROCESSING_MS, out.processing_ms.to_le_bytes());
    for (k, value) in out.calibration.values().iter().enumerate() {
        put(vision, CALIBRATION + 4 * k, value.to_le_bytes());
    }
    let list_pointer = 1u64 | (7 << 32) | ((list_words as u64) << 35);
    put(vision, VISION_WORDS * 8, list_pointer.to_le_bytes());

    let tag = struct_pointer(BOX_WORDS, 0) | ((out.boxes.len() as u64) << 2);
    put(&mut bytes, LIST * 8, tag.to_le_bytes());
    for (i, vision_box) in out.boxes.iter().enumerate() {
        let start = (LIST + 1 + i * BOX_WORDS) * 8;
        let dst = &mut bytes[start..start + BOX_WORDS * 8];
        for (k, value) in vision_box.xyxy.iter().enumerate() {
            put(dst, BOX_XYXY + 4 * k, value.to_le_bytes());
        }
        put(dst, BOX_CONFIDENCE, vision_box.confidence.to_le_bytes());
        dst[BOX_CLASS_ID] = vision_box.class_id;
        dst[BOX_CUT_BIT / 8] |= u8::from(vision_box.cut) << (BOX_CUT_BIT % 8);
    }
    bytes
}

/// The Event's `logMonoTime` (CLOCK_MONOTONIC) and CcncVision. None for another Event member, an invalid Event or a malformed message.
pub fn decode(bytes: &[u8]) -> Option<(u64, VisionOut)> {
    let message = Message::new(bytes)?;
    let event = message.root()?;
    if event.u16_at(EVENT_DISCRIMINANT_BYTE) != EVENT_CCNC_VISION || event.bit(EVENT_INVALID_BIT) {
        return None;
    }
    let vision = message.struct_pointer(&event, 0)?;
    let boxes = message
        .struct_list(&vision, 0)?
        .iter()
        .map(|b| VisionBox {
            xyxy: std::array::from_fn(|k| b.f32_at(BOX_XYXY + 4 * k)),
            confidence: b.f32_at(BOX_CONFIDENCE),
            class_id: b.u8_at(BOX_CLASS_ID),
            cut: b.bit(BOX_CUT_BIT),
        })
        .collect();
    Some((
        event.u64_at(0),
        VisionOut {
            status: Status::from_wire(vision.u8_at(STATUS))?,
            frame_id: vision.u32_at(FRAME_ID),
            timestamp_sof: vision.u64_at(TIMESTAMP_SOF),
            frame_mono_time: vision.u64_at(FRAME_MONO_TIME),
            view: View::from_wire(vision.u8_at(VIEW))?,
            processing_ms: vision.f32_at(PROCESSING_MS),
            night_corrected: vision.bit(NIGHT_CORRECTED_BIT),
            calibration: Calibration::from_values(std::array::from_fn(|k| vision.f32_at(CALIBRATION + 4 * k))),
            boxes,
        },
    ))
}

/// `carState.vEgo` of a valid carState Event.
pub fn car_state_speed(bytes: &[u8]) -> Option<f32> {
    let message = Message::new(bytes)?;
    let event = message.root()?;
    if event.u16_at(EVENT_DISCRIMINANT_BYTE) != EVENT_CAR_STATE || event.bit(EVENT_INVALID_BIT) {
        return None;
    }
    Some(message.struct_pointer(&event, 0)?.f32_at(0))
}

/// `liveCalibration` of a valid liveCalibration Event.
pub fn live_calibration(bytes: &[u8]) -> Option<LiveCalibration> {
    let message = Message::new(bytes)?;
    let event = message.root()?;
    if event.u16_at(EVENT_DISCRIMINANT_BYTE) != EVENT_LIVE_CALIBRATION || event.bit(EVENT_INVALID_BIT) {
        return None;
    }
    let calibration = message.struct_pointer(&event, 0)?;
    Some(LiveCalibration {
        cal_status: calibration.u16_at(CAL_STATUS),
        rpy_calib: message.f32_list(&calibration, CAL_RPY_CALIB)?,
        wide_from_device_euler: message.f32_list(&calibration, CAL_WIDE_FROM_DEVICE_EULER)?,
        height: message.f32_list(&calibration, CAL_HEIGHT)?,
    })
}

/// Whether a valid CcncMap (`customReserved3`) Event currently has a camera-backed front side car (LF or RF:
/// `present && vision != 0`). Used by `ccnc_visiond` to insert full-frame raw inference only while a front side
/// car is shown, enabling LR/RR two-car splits without slowing side-view detection when no side car is shown.
pub fn map_front_vision_on(bytes: &[u8]) -> Option<bool> {
    let message = Message::new(bytes)?;
    let event = message.root()?;
    if event.u16_at(EVENT_DISCRIMINANT_BYTE) != EVENT_CCNC_MAP || event.bit(EVENT_INVALID_BIT) {
        return None;
    }
    let map = message.struct_pointer(&event, 0)?;
    if !map.bit(MAP_VALID_BIT) {
        return Some(false);
    }
    let slots = message.struct_list(&map, 0)?;
    let on = [1usize, 2].into_iter().any(|i| {
        slots.get(i).is_some_and(|s| s.bit(MAP_SLOT_PRESENT_BIT) && s.u8_at(MAP_SLOT_VISION_BYTE) != 0)
    });
    Some(on)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(boxes: usize) -> VisionOut {
        VisionOut {
            status: Status::Frame,
            frame_id: 0x8123_4567,
            timestamp_sof: 0xfedc_ba98_7654_3210,
            frame_mono_time: 0x0123_4567_89ab_cdef,
            view: View::SideR,
            processing_ms: 612.25,
            night_corrected: true,
            calibration: Calibration { rpy_calib: [1e-5, -0.0106, 0.004], wide_from_device_euler: [-6.5e-7, -0.0098, -0.018], height: 1.2228 },
            boxes: (0..boxes)
                .map(|i| VisionBox {
                    xyxy: [0.5 + i as f32, 12.3, 511.9, 384.0],
                    confidence: 0.9871,
                    class_id: [2, 7, 79][i % 3],
                    cut: i % 2 == 1,
                })
                .collect(),
        }
    }

    #[test]
    fn decode_returns_what_encode_was_given() {
        for boxes in [0, 1, 2, 37] {
            let out = sample(boxes);
            let bytes = encode(&out, 123_456_789);
            assert_eq!(bytes.len() % 8, 0);
            assert_eq!(decode(&bytes), Some((123_456_789, out)));
        }
        let day = VisionOut { view: View::SideL, night_corrected: false, ..sample(1) };
        assert_eq!(decode(&encode(&day, 0)), Some((0, day)));
        let fixed = VisionOut { calibration: Calibration::UNKNOWN, ..sample(1) };
        let (_, back) = decode(&encode(&fixed, 0)).unwrap();
        assert!(back.calibration.values().iter().all(|v| v.is_nan()));
        for status in [Status::Resting, Status::NotCalibrated, Status::NoCamera, Status::Halted, Status::UnsupportedCamera] {
            let quiet = VisionOut { status, frame_id: 0, timestamp_sof: 0, frame_mono_time: 0, ..sample(0) };
            assert_eq!(decode(&encode(&quiet, 5)), Some((5, quiet)));
        }
    }

    #[test]
    fn fields_sit_where_pycapnp_puts_them() {
        // Offsets printed by pycapnp for CustomReserved4 (data 7 words): status u8 at byte 6, frameMonoTime u64 at
        // word 6, timestampSof u64 at word 1.
        let bytes = encode(&sample(0), 0);
        let vision = &bytes[(2 + EVENT_WORDS + 1) * 8..];
        assert_eq!(VISION_WORDS, 7);
        assert_eq!(u64::from_le_bytes(vision[48..56].try_into().unwrap()), 0x0123_4567_89ab_cdef);
        assert_eq!(u64::from_le_bytes(vision[8..16].try_into().unwrap()), 0xfedc_ba98_7654_3210);
        let halted = encode(&VisionOut { status: Status::Halted, ..sample(0) }, 0);
        assert_eq!(halted[(2 + EVENT_WORDS + 1) * 8 + 6], 4);
    }

    #[test]
    fn decode_rejects_other_or_damaged_messages() {
        let bytes = encode(&sample(2), 1);
        for cut in [0, 7, 8, bytes.len() - 8, bytes.len() - 1] {
            assert_eq!(decode(&bytes[..cut]), None, "truncated to {cut}");
        }
        let mut other = bytes.clone();
        other[16 + EVENT_DISCRIMINANT_BYTE] = 108;
        assert_eq!(decode(&other), None);
        let mut invalid = bytes.clone();
        invalid[16 + EVENT_INVALID_BIT / 8] |= 1 << (EVENT_INVALID_BIT % 8);
        assert_eq!(decode(&invalid), None);
        let mut view = bytes.clone();
        view[(2 + EVENT_WORDS + 1) * 8 + VIEW] = 4;
        assert_eq!(decode(&view), None);
        let mut status = bytes.clone();
        status[(2 + EVENT_WORDS + 1) * 8 + STATUS] = 6;
        assert_eq!(decode(&status), None);
        let mut list = bytes;
        // The list claims more words than the segment holds.
        list[(2 + EVENT_WORDS + 1 + VISION_WORDS) * 8 + 4] = 0xff;
        assert_eq!(decode(&list), None);
    }

    fn car_state_event(discriminant: u16, valid: bool, speed: f32) -> Vec<u8> {
        // Root pointer, Event (2 data words, 1 pointer), CarState with one data word.
        let mut bytes = vec![0; 8 * 6];
        put(&mut bytes, 4, 5u32.to_le_bytes());
        put(&mut bytes, 8, struct_pointer(EVENT_WORDS, 1).to_le_bytes());
        put(&mut bytes, 16 + EVENT_DISCRIMINANT_BYTE, discriminant.to_le_bytes());
        bytes[16 + EVENT_INVALID_BIT / 8] = u8::from(!valid) << (EVENT_INVALID_BIT % 8);
        put(&mut bytes, 32, struct_pointer(1, 0).to_le_bytes());
        put(&mut bytes, 40, speed.to_le_bytes());
        bytes
    }

    /// Root pointer, Event, LiveCalibrationData (2 data words, 8 pointers), then each list: a pointer to a
    /// List(Float32) is element size 4 with the element count.
    fn live_calibration_event(valid: bool, status: u16, lists: [&[f32]; 3]) -> Vec<u8> {
        let list_words: Vec<usize> = lists.iter().map(|l| l.len().div_ceil(2)).collect();
        let words = 1 + 3 + 10 + list_words.iter().sum::<usize>();
        let mut bytes = vec![0; 8 * (words + 1)];
        put(&mut bytes, 4, (words as u32).to_le_bytes());
        put(&mut bytes, 8, struct_pointer(EVENT_WORDS, 1).to_le_bytes());
        put(&mut bytes, 16 + EVENT_DISCRIMINANT_BYTE, EVENT_LIVE_CALIBRATION.to_le_bytes());
        bytes[16 + EVENT_INVALID_BIT / 8] = u8::from(!valid) << (EVENT_INVALID_BIT % 8);
        put(&mut bytes, 32, struct_pointer(2, 8).to_le_bytes());
        put(&mut bytes, 40 + CAL_STATUS, status.to_le_bytes());
        // Word indices count the 8-byte segment table as word 0: the struct ends at word 14, the lists follow.
        let mut at = 15;
        for (list, pointer) in lists.iter().zip([CAL_RPY_CALIB, CAL_WIDE_FROM_DEVICE_EULER, CAL_HEIGHT]) {
            let pointer_word = 7 + pointer;
            let offset = (at - pointer_word - 1) as u64;
            put(&mut bytes, 8 * pointer_word, (1 | (offset << 2) | (4 << 32) | ((list.len() as u64) << 35)).to_le_bytes());
            for (k, value) in list.iter().enumerate() {
                put(&mut bytes, 8 * at + 4 * k, value.to_le_bytes());
            }
            at += list.len().div_ceil(2);
        }
        bytes
    }

    #[test]
    fn live_calibration_reads_the_lists() {
        let rpy: &[f32] = &[1e-5, -0.0106, 0.004];
        let wide: &[f32] = &[-6.5e-7, -0.0098, -0.018];
        let bytes = live_calibration_event(true, CAL_STATUS_CALIBRATED, [rpy, wide, &[1.2228]]);
        let want = LiveCalibration { cal_status: 1, rpy_calib: rpy.to_vec(), wide_from_device_euler: wide.to_vec(), height: vec![1.2228] };
        assert_eq!(live_calibration(&bytes), Some(want));
        assert_eq!(live_calibration(&live_calibration_event(true, 3, [&[], wide, &[]])).unwrap().rpy_calib, Vec::<f32>::new());
        assert_eq!(live_calibration(&live_calibration_event(false, 1, [rpy, wide, &[1.0]])), None);
        assert_eq!(live_calibration(&encode(&sample(1), 0)), None);
        let truncated = &bytes[..bytes.len() - 8];
        assert_eq!(live_calibration(truncated), None);
    }

    #[test]
    fn car_state_speed_needs_a_valid_car_state() {
        assert_eq!(car_state_speed(&car_state_event(EVENT_CAR_STATE, true, 2.5)), Some(2.5));
        assert_eq!(car_state_speed(&car_state_event(EVENT_CAR_STATE, false, 2.5)), None);
        assert_eq!(car_state_speed(&car_state_event(EVENT_CCNC_VISION, true, 2.5)), None);
        assert_eq!(car_state_speed(&[]), None);
    }

    fn ccnc_map_event(event_valid: bool, map_valid: bool, slots_cfg: [(bool, u8); 5]) -> Vec<u8> {
        // Root pointer, Event (2w, 1p), CcncMap (3w, 1p), inline-composite list of 5 Slots (3w, 0p).
        const MAP_START: usize = 5;
        const LIST_START: usize = 9;
        const WORDS: usize = LIST_START + 1 + 5 * 3;
        let mut bytes = vec![0u8; WORDS * 8];
        put(&mut bytes, 4, ((WORDS - 1) as u32).to_le_bytes());
        put(&mut bytes, 8, struct_pointer(EVENT_WORDS, 1).to_le_bytes());
        put(&mut bytes, 16 + EVENT_DISCRIMINANT_BYTE, EVENT_CCNC_MAP.to_le_bytes());
        bytes[16 + EVENT_INVALID_BIT / 8] = u8::from(!event_valid) << (EVENT_INVALID_BIT % 8);
        put(&mut bytes, 16 + EVENT_WORDS * 8, struct_pointer(3, 1).to_le_bytes());
        if map_valid {
            bytes[MAP_START * 8 + MAP_VALID_BIT / 8] |= 1 << (MAP_VALID_BIT % 8);
        }
        let list_ptr = 1u64 | (7 << 32) | ((5 * 3) as u64) << 35;
        put(&mut bytes, (MAP_START + 3) * 8, list_ptr.to_le_bytes());
        let list_tag = struct_pointer(3, 0) | (5u64 << 2);
        put(&mut bytes, LIST_START * 8, list_tag.to_le_bytes());
        for (i, (present, vision)) in slots_cfg.into_iter().enumerate() {
            let start = (LIST_START + 1 + i * 3) * 8;
            if present {
                bytes[start + MAP_SLOT_PRESENT_BIT / 8] |= 1 << (MAP_SLOT_PRESENT_BIT % 8);
            }
            bytes[start + MAP_SLOT_VISION_BYTE] = vision;
        }
        bytes
    }

    #[test]
    fn map_front_vision_reads_lf_and_rf_vision_slots() {
        let none = [(false, 0); 5];
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, true, none)), Some(false));
        // Radar-only LF/RF (vision == 0) does not trigger raw.
        let radar_lf = [(false, 0), (true, 0), (true, 0), (true, 1), (true, 2)];
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, true, radar_lf)), Some(false));
        // Camera-only LF (vision == 1) or camera-confirmed RF (vision == 2) triggers raw.
        let cam_lf = [(false, 0), (true, 1), (false, 0), (false, 0), (false, 0)];
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, true, cam_lf)), Some(true));
        let cam_rf = [(false, 0), (false, 0), (true, 2), (false, 0), (false, 0)];
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, true, cam_rf)), Some(true));
        // Invalid map or non-present slot does not trigger raw.
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, false, cam_lf)), Some(false));
        let absent_lf = [(false, 0), (false, 1), (false, 0), (false, 0), (false, 0)];
        assert_eq!(map_front_vision_on(&ccnc_map_event(true, true, absent_lf)), Some(false));
        // Invalid Event or wrong discriminant returns None.
        assert_eq!(map_front_vision_on(&ccnc_map_event(false, true, cam_lf)), None);
        assert_eq!(map_front_vision_on(&encode(&sample(0), 1)), None);
    }
}
