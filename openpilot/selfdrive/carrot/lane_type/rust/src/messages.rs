//! The cereal messages marking_daemon.py subscribes to, decoded from `log.Event` bytes (only the fields it reads).
//!
//! Positions: data fields in units of their size (bits for Bool), pointers by index; `fixtures/schema_offsets.json`
//! (tools/schema_offsets.py, from the pycapnp schema) is compared with these in the tests.

use crate::capnp::{Message, Struct};

pub const EVENT_DISCRIMINANT: usize = 4;
pub const EVENT_LOG_MONO_TIME: usize = 0;
pub const EVENT_VALID: (usize, bool) = (80, true);
pub const EVENT_UNION_POINTER: usize = 0;
pub const EVENT_MODEL_V2: u16 = 73;
pub const EVENT_CAR_STATE: u16 = 21;
pub const EVENT_LIVE_CALIBRATION: u16 = 18;
pub const EVENT_DEVICE_STATE: u16 = 5;
pub const EVENT_ROAD_CAMERA_STATE: u16 = 1;
pub const EVENT_RADAR_STATE: u16 = 12;

pub const MODEL_FRAME_ID: usize = 0; // UInt32
pub const MODEL_TIMESTAMP_EOF: usize = 2; // UInt64
pub const MODEL_LANE_LINES: usize = 4;
pub const MODEL_LANE_LINE_PROBS: usize = 5;
pub const MODEL_META: usize = 8;
pub const META_LANE_CHANGE_STATE: usize = 9; // Enum
pub const XYZT_X: usize = 0;
pub const XYZT_Y: usize = 1;
pub const XYZT_Z: usize = 2;
pub const CAR_V_EGO: usize = 0; // Float32
pub const CAR_LEFT_BLINKER: usize = 69; // Bool
pub const CAR_RIGHT_BLINKER: usize = 70;
pub const CALIB_STATUS: usize = 1; // Enum
pub const CALIB_RPY: usize = 4;
pub const DEVICE_STARTED: usize = 192; // Bool
pub const DEVICE_TYPE: usize = 41; // Enum
pub const FRAME_SENSOR: usize = 40; // Enum
pub const RADAR_LEAD_ONE: usize = 1;
pub const LEAD_D_REL: usize = 0; // Float32
pub const LEAD_STATUS: usize = 321; // Bool

pub const LANE_CHANGE_STATES: [&str; 4] = ["off", "preLaneChange", "laneChangeStarting", "laneChangeFinishing"];
pub const CAL_STATUSES: [&str; 4] = ["uncalibrated", "calibrated", "invalid", "recalibrating"];
pub const DEVICE_TYPES: [&str; 8] = ["unknown", "neo", "chffrAndroid", "chffrIos", "tici", "pc", "tizi", "mici"];
pub const IMAGE_SENSORS: [&str; 4] = ["unknown", "ar0231", "ox03c10", "os04c10"];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Line {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelV2 {
    pub frame_id: u32,
    pub timestamp_eof: u64,
    pub lane_lines: Vec<Line>,
    pub lane_line_probs: Vec<f32>,
    pub lane_change_state: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarState {
    pub v_ego: f32,
    pub left_blinker: bool,
    pub right_blinker: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    pub cal_status: u16,
    pub rpy_calib: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceState {
    pub started: bool,
    pub device_type: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RadarState {
    pub lead_status: bool,
    pub lead_d_rel: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    ModelV2(ModelV2),
    CarState(CarState),
    Calibration(Calibration),
    DeviceState(DeviceState),
    RoadCameraState { sensor: u16 },
    RadarState(RadarState),
    Other(u16),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub log_mono_time: u64,
    pub valid: bool,
    pub payload: Payload,
}

fn enum_in(value: u16, names: &[&str]) -> Option<u16> {
    ((value as usize) < names.len()).then_some(value)
}

fn model_v2(m: &Message, s: &Struct) -> Option<ModelV2> {
    let lane_lines = m
        .struct_list(s, MODEL_LANE_LINES)?
        .iter()
        .map(|l| Some(Line { x: m.f32_list(l, XYZT_X)?, y: m.f32_list(l, XYZT_Y)?, z: m.f32_list(l, XYZT_Z)? }))
        .collect::<Option<Vec<_>>>()?;
    let meta = m.struct_at(s, MODEL_META)?;
    Some(ModelV2 {
        frame_id: s.u32(MODEL_FRAME_ID, 0),
        timestamp_eof: s.u64(MODEL_TIMESTAMP_EOF, 0),
        lane_lines,
        lane_line_probs: m.f32_list(s, MODEL_LANE_LINE_PROBS)?,
        lane_change_state: enum_in(meta.u16(META_LANE_CHANGE_STATE, 0), &LANE_CHANGE_STATES)?,
    })
}

/// One `log.Event`, or None when the bytes do not read as one (or an enum is outside the schema's names).
pub fn parse(bytes: &[u8]) -> Option<Event> {
    let m = Message::new(bytes)?;
    let event = m.root()?;
    let which = event.u16(EVENT_DISCRIMINANT, 0);
    let inner = || m.struct_at(&event, EVENT_UNION_POINTER);
    let payload = match which {
        EVENT_MODEL_V2 => Payload::ModelV2(model_v2(&m, &inner()?)?),
        EVENT_CAR_STATE => {
            let s = inner()?;
            Payload::CarState(CarState {
                v_ego: s.f32(CAR_V_EGO),
                left_blinker: s.bool(CAR_LEFT_BLINKER, false),
                right_blinker: s.bool(CAR_RIGHT_BLINKER, false),
            })
        }
        EVENT_LIVE_CALIBRATION => {
            let s = inner()?;
            Payload::Calibration(Calibration {
                cal_status: enum_in(s.u16(CALIB_STATUS, 0), &CAL_STATUSES)?,
                rpy_calib: m.f32_list(&s, CALIB_RPY)?,
            })
        }
        EVENT_DEVICE_STATE => {
            let s = inner()?;
            Payload::DeviceState(DeviceState {
                started: s.bool(DEVICE_STARTED, false),
                device_type: enum_in(s.u16(DEVICE_TYPE, 0), &DEVICE_TYPES)?,
            })
        }
        EVENT_ROAD_CAMERA_STATE => {
            let s = inner()?;
            Payload::RoadCameraState { sensor: enum_in(s.u16(FRAME_SENSOR, 0), &IMAGE_SENSORS)? }
        }
        EVENT_RADAR_STATE => {
            let s = inner()?;
            let lead = m.struct_at(&s, RADAR_LEAD_ONE)?;
            Payload::RadarState(RadarState { lead_status: lead.bool(LEAD_STATUS, false), lead_d_rel: lead.f32(LEAD_D_REL) })
        }
        other => Payload::Other(other),
    };
    Some(Event { log_mono_time: event.u64(EVENT_LOG_MONO_TIME, 0), valid: event.bool(EVENT_VALID.0, EVENT_VALID.1), payload })
}

/// Build the framed bytes of an Event that carries only `customReservedRawData2` (Data = List(UInt8)) for publishing.
/// Single segment, no far pointers; `valid` is written true. The schema position is in `fixtures/schema_offsets.json`
/// (`Event.customReservedRawData2`).
pub fn raw_data_event(discriminant: u16, log_mono_time: u64, payload: &[u8]) -> Vec<u8> {
    let data_words = 2usize;
    let pointer_words = 1usize;
    let list_words = payload.len().div_ceil(8);
    let segment_words = 1 + data_words + pointer_words + list_words;
    let mut out = Vec::with_capacity(8 + segment_words * 8);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(segment_words as u32).to_le_bytes());
    // Root pointer: struct at offset 0, data/pointer sizes.
    let root = ((data_words as u64) << 32) | ((pointer_words as u64) << 48);
    out.extend_from_slice(&root.to_le_bytes());
    let mut data = [0u8; 16];
    data[0..8].copy_from_slice(&log_mono_time.to_le_bytes());
    data[8..10].copy_from_slice(&discriminant.to_le_bytes());
    // valid (bit 80) defaults to true and is stored XORed: 0 = true.
    out.extend_from_slice(&data);
    // List pointer: offset 0 (the list follows), element size 2 (byte), count.
    let list = 1u64 | (2u64 << 32) | ((payload.len() as u64) << 35);
    out.extend_from_slice(&list.to_le_bytes());
    out.extend_from_slice(payload);
    out.resize(8 + segment_words * 8, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn schema() -> Value {
        serde_json::from_str(include_str!("../fixtures/schema_offsets.json")).unwrap()
    }

    fn field<'a>(s: &'a Value, st: &str, f: &str) -> &'a Value {
        &s["structs"][st]["fields"][f]
    }

    #[test]
    fn constants_match_the_schema() {
        let s = schema();
        let off = |st: &str, f: &str| field(&s, st, f)["offset"].as_u64().unwrap() as usize;
        let disc = |f: &str| field(&s, "Event", f)["discriminant"].as_u64().unwrap() as u16;
        assert_eq!(s["structs"]["Event"]["discriminantOffset"].as_u64().unwrap() as usize, EVENT_DISCRIMINANT);
        assert_eq!(off("Event", "logMonoTime"), EVENT_LOG_MONO_TIME);
        assert_eq!(off("Event", "valid"), EVENT_VALID.0);
        assert_eq!(field(&s, "Event", "valid")["default"].as_bool().unwrap(), EVENT_VALID.1);
        for (name, value) in [("modelV2", EVENT_MODEL_V2), ("carState", EVENT_CAR_STATE), ("liveCalibration", EVENT_LIVE_CALIBRATION),
                              ("deviceState", EVENT_DEVICE_STATE), ("roadCameraState", EVENT_ROAD_CAMERA_STATE),
                              ("radarState", EVENT_RADAR_STATE), ("customReservedRawData2", crate::daemon::EVENT_CUSTOM_RAW_2)] {
            assert_eq!(disc(name), value, "{name}");
            assert_eq!(off("Event", name), EVENT_UNION_POINTER, "{name}");
        }
        assert_eq!(field(&s, "Event", "customReservedRawData2")["kind"], "data");
        assert_eq!(s["structs"]["Event"]["dataWords"].as_u64().unwrap(), 2);
        assert_eq!(s["structs"]["Event"]["pointers"].as_u64().unwrap(), 1);
        for (st, f, want) in [("ModelDataV2", "frameId", MODEL_FRAME_ID), ("ModelDataV2", "timestampEof", MODEL_TIMESTAMP_EOF),
                              ("ModelDataV2", "laneLines", MODEL_LANE_LINES), ("ModelDataV2", "laneLineProbs", MODEL_LANE_LINE_PROBS),
                              ("ModelDataV2", "meta", MODEL_META), ("ModelDataV2.MetaData", "laneChangeState", META_LANE_CHANGE_STATE),
                              ("XYZTData", "x", XYZT_X), ("XYZTData", "y", XYZT_Y), ("XYZTData", "z", XYZT_Z),
                              ("CarState", "vEgo", CAR_V_EGO), ("CarState", "leftBlinker", CAR_LEFT_BLINKER),
                              ("CarState", "rightBlinker", CAR_RIGHT_BLINKER), ("LiveCalibrationData", "calStatus", CALIB_STATUS),
                              ("LiveCalibrationData", "rpyCalib", CALIB_RPY), ("DeviceState", "started", DEVICE_STARTED),
                              ("DeviceState", "deviceType", DEVICE_TYPE), ("FrameData", "sensor", FRAME_SENSOR),
                              ("RadarState", "leadOne", RADAR_LEAD_ONE), ("RadarState.LeadData", "dRel", LEAD_D_REL),
                              ("RadarState.LeadData", "status", LEAD_STATUS)] {
            assert_eq!(off(st, f), want, "{st}.{f}");
            if let Some(d) = field(&s, st, f).get("default") {
                assert!(d == &Value::from(0) || d == &Value::from(false) || d == &Value::from(0.0), "{st}.{f} default {d}");
            }
        }
        let names = |e: &str| s["enums"][e].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(names("LaneChangeState"), LANE_CHANGE_STATES);
        assert_eq!(names("CalStatus"), CAL_STATUSES);
        assert_eq!(names("DeviceType"), DEVICE_TYPES);
        assert_eq!(names("ImageSensor"), IMAGE_SENSORS);
    }

    /// Real rlog events (fixtures/events.bin, tools/make_event_fixture.py) against their pycapnp values.
    #[test]
    fn real_events_read_as_pycapnp() {
        let bytes = include_bytes!("../fixtures/events.bin");
        let want: Vec<Value> = serde_json::from_str(include_str!("../fixtures/events.json")).unwrap();
        let mut at = 0;
        let mut n = 0;
        let lists = |v: &Value| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>();
        while at < bytes.len() {
            let len = crate::capnp::framed_len(&bytes[at..]).unwrap();
            let e = parse(&bytes[at..at + len]).expect("event");
            let w = &want[n];
            assert_eq!(e.log_mono_time, w["logMonoTime"].as_u64().unwrap());
            assert_eq!(e.valid, w["valid"].as_bool().unwrap());
            match (&e.payload, w["which"].as_str().unwrap()) {
                (Payload::ModelV2(m), "modelV2") => {
                    assert_eq!(m.frame_id as u64, w["frameId"].as_u64().unwrap());
                    assert_eq!(m.timestamp_eof, w["timestampEof"].as_u64().unwrap());
                    assert_eq!(m.lane_line_probs, lists(&w["laneLineProbs"]));
                    assert_eq!(LANE_CHANGE_STATES[m.lane_change_state as usize], w["laneChangeState"].as_str().unwrap());
                    for (i, l) in m.lane_lines.iter().enumerate() {
                        assert_eq!(l.x, lists(&w["laneLines"][i]["x"]));
                        assert_eq!(l.y, lists(&w["laneLines"][i]["y"]));
                        assert_eq!(l.z, lists(&w["laneLines"][i]["z"]));
                    }
                }
                (Payload::CarState(c), "carState") => {
                    assert_eq!(c.v_ego, w["vEgo"].as_f64().unwrap() as f32);
                    assert_eq!(c.left_blinker, w["leftBlinker"].as_bool().unwrap());
                    assert_eq!(c.right_blinker, w["rightBlinker"].as_bool().unwrap());
                }
                (Payload::Calibration(c), "liveCalibration") => {
                    assert_eq!(CAL_STATUSES[c.cal_status as usize], w["calStatus"].as_str().unwrap());
                    assert_eq!(c.rpy_calib, lists(&w["rpyCalib"]));
                }
                (Payload::DeviceState(d), "deviceState") => {
                    assert_eq!(d.started, w["started"].as_bool().unwrap());
                    assert_eq!(DEVICE_TYPES[d.device_type as usize], w["deviceType"].as_str().unwrap());
                }
                (Payload::RoadCameraState { sensor }, "roadCameraState") => {
                    assert_eq!(IMAGE_SENSORS[*sensor as usize], w["sensor"].as_str().unwrap());
                }
                (Payload::RadarState(r), "radarState") => {
                    assert_eq!(r.lead_status, w["leadStatus"].as_bool().unwrap());
                    assert_eq!(r.lead_d_rel, w["leadDRel"].as_f64().unwrap() as f32);
                }
                (p, which) => panic!("{which}: {p:?}"),
            }
            at += len;
            n += 1;
        }
        assert_eq!(n, want.len());
        assert!(n >= 6);
    }

    #[test]
    fn raw_data_event_reads_back() {
        let bytes = raw_data_event(crate::daemon::EVENT_CUSTOM_RAW_2, 123, b"{\"a\":1}");
        let m = Message::new(&bytes).unwrap();
        let root = m.root().unwrap();
        assert_eq!(root.u64(0, 0), 123);
        assert_eq!(root.u16(EVENT_DISCRIMINANT, 0), crate::daemon::EVENT_CUSTOM_RAW_2);
        assert!(root.bool(EVENT_VALID.0, EVENT_VALID.1));
        assert_eq!(framed_len_ok(&bytes), bytes.len());
    }

    fn framed_len_ok(b: &[u8]) -> usize {
        crate::capnp::framed_len(b).unwrap()
    }
}
