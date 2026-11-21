//! marking_daemon.main's per-modelV2 step without the IPC: the SubMaster's latest messages, the frame buffer, the
//! odometer, stale-input / error handling, DisplayPolicy and the published JSON packet. The IPC loop (src/ipc.rs,
//! bin/stripnet_daemon.rs) and the offline replay (bin/stripnet_replay.rs) both drive this.

use std::collections::VecDeque;

use serde_json::{json, Map, Value};

use crate::display::{Confirmed, DisplayPolicy};
use crate::messages::{CarState, Calibration, DeviceState, Event, ModelV2, Payload, RadarState, DEVICE_TYPES, IMAGE_SENSORS};
use crate::recogniser::{Frame, Recogniser};
use crate::result::SideResult;
use crate::strip::{LineF64, Nv12};

pub const EVENT_CUSTOM_RAW_2: u16 = 153;
pub const PACKET_TYPE: &str = "ccncLaneMarking";
pub const PACKET_VERSION: u64 = 1;
pub const FRAME_BUFFER: usize = 3;
pub const MAX_FRAME_OFFSET_NS: i64 = 50_000_000;
pub const MAX_HOLD_NS: i64 = 500_000_000;
pub const LEAD_MAX_AGE_NS: i64 = 200_000_000;
/// cereal SubMaster: a service is alive while its last message is newer than 10 periods (carState 100 Hz).
pub const CAR_STATE_ALIVE_NS: i64 = 100_000_000;
pub const COUNTERS: [&str; 10] = ["processed", "exact", "nearest", "frame_mismatch", "stale_inputs", "errors", "drop_hold_before",
                                  "drop_frame_id_before", "drop_hold_after", "drop_frame_id_after"];

/// common.transformations.camera DEVICE_CAMERAS fcam: (width, height, focal length).
pub fn road_camera(device: &str, sensor: &str) -> Option<(usize, usize, f64)> {
    const AR_OX: (usize, usize, f64) = (1928, 1208, 2648.);
    const OS: (usize, usize, f64) = (2688 / 2, 1520 / 2, 1522. * 3. / 4.);
    match (device, sensor) {
        ("neo", "unknown") => Some((1164, 874, 910.)),
        ("tici", "unknown") | ("unknown", "ar0231") | ("unknown", "ox03c10") | ("pc", "unknown") => Some(AR_OX),
        ("tici" | "tizi" | "mici", "ar0231" | "ox03c10") => Some(AR_OX),
        ("tici" | "tizi" | "mici", "os04c10") => Some(OS),
        _ => None,
    }
}

/// K scaled to the buffer size (marking_daemon.main: k[0] *= width/camera.width, k[1] *= height/camera.height).
pub fn intrinsics(camera: (usize, usize, f64), width: usize, height: usize) -> [[f64; 3]; 3] {
    let (cw, ch, f) = camera;
    let mut k = [[f, 0., cw as f64 / 2.], [0., f, ch as f64 / 2.], [0., 0., 1.]];
    let (sx, sy) = (width as f64 / cw as f64, height as f64 / ch as f64);
    for v in k[0].iter_mut() {
        *v *= sx;
    }
    for v in k[1].iter_mut() {
        *v *= sy;
    }
    k
}

/// One buffered camera frame: its own frameId / EOF captured at receive time, and the receive time.
pub struct Buffered<B> {
    pub frame_id: u32,
    pub eof: u64,
    pub buf: B,
    pub recv_ns: i64,
}

pub struct FrameBuffer<B> {
    pub frames: VecDeque<Buffered<B>>,
}

impl<B> Default for FrameBuffer<B> {
    fn default() -> Self {
        FrameBuffer { frames: VecDeque::with_capacity(FRAME_BUFFER + 1) }
    }
}

impl<B> FrameBuffer<B> {
    pub fn push(&mut self, f: Buffered<B>) {
        self.frames.push_back(f);
        while self.frames.len() > FRAME_BUFFER {
            self.frames.pop_front();
        }
    }

    /// FrameBuffer.select: the exact frame first, else the nearest EOF within MAX_FRAME_OFFSET_NS.
    pub fn select(&self, frame_id: u32, eof: u64) -> Option<(&Buffered<B>, &'static str)> {
        if let Some(f) = self.frames.iter().find(|f| f.frame_id == frame_id && f.eof == eof) {
            return Some((f, "exact"));
        }
        let mut best: Option<&Buffered<B>> = None;
        for f in &self.frames {
            if best.is_none_or(|b| (f.eof as i64 - eof as i64).abs() < (b.eof as i64 - eof as i64).abs()) {
                best = Some(f);
            }
        }
        best.filter(|f| (f.eof as i64 - eof as i64).abs() <= MAX_FRAME_OFFSET_NS).map(|f| (f, "nearest"))
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

/// buffer_drop_reason: 'hold' past MAX_HOLD_NS, 'frame_id' when the slot was republished, else None.
pub fn buffer_drop_reason(frame_id: u32, buf_frame_id: u32, recv_ns: i64, now_ns: i64) -> Option<&'static str> {
    if now_ns - recv_ns >= MAX_HOLD_NS {
        Some("hold")
    } else if buf_frame_id != frame_id {
        Some("frame_id")
    } else {
        None
    }
}

#[derive(Default)]
pub struct Latest {
    pub calibration: Option<(Calibration, bool)>,
    pub car: Option<(CarState, i64)>,
    pub device: Option<DeviceState>,
    pub sensor: Option<u16>,
    pub radar: Option<(RadarState, i64)>,
}

impl Latest {
    /// SubMaster.update for one non-model event (log_mono: the message's logMonoTime).
    pub fn take(&mut self, e: &Event) {
        let mono = e.log_mono_time as i64;
        match &e.payload {
            Payload::Calibration(c) => self.calibration = Some((c.clone(), e.valid)),
            Payload::CarState(c) => self.car = Some((*c, mono)),
            Payload::DeviceState(d) => self.device = Some(*d),
            Payload::RoadCameraState { sensor } => self.sensor = Some(*sensor),
            Payload::RadarState(r) => self.radar = Some((*r, mono)),
            _ => {}
        }
    }
}

pub fn lead_d_rel(radar: Option<&(RadarState, i64)>, stamp: i64) -> Option<f64> {
    let (r, mono) = radar?;
    if *mono == 0 || (stamp - mono).abs() > LEAD_MAX_AGE_NS {
        return None;
    }
    r.lead_status.then_some(r.lead_d_rel as f64)
}

pub fn line_of(line: &crate::messages::Line, shift: f64) -> LineF64 {
    LineF64 {
        x: line.x.iter().map(|&v| v as f64 - shift).collect(),
        y: line.y.iter().map(|&v| v as f64).collect(),
        z: line.z.iter().map(|&v| v as f64).collect(),
    }
}

pub struct ModelInfo {
    pub name: String,
    pub version: Value,
    pub sha256: String,
}

/// What one modelV2 led to.
pub enum Step {
    Published { packet: Value, raw: [SideResult; 2], frame: Frame },
    Mismatch,
    Stale,
    Error(String),
    Dropped(&'static str),
}

pub struct Daemon {
    pub latest: Latest,
    pub recogniser: Recogniser,
    pub policy: DisplayPolicy,
    pub counters: [u64; 10],
    pub odometer: f64,
    pub last_stamp: Option<i64>,
    pub info: ModelInfo,
}

fn count(counters: &mut [u64; 10], name: &str) {
    counters[COUNTERS.iter().position(|c| *c == name).expect("counter")] += 1;
}

impl Daemon {
    pub fn new(recogniser: Recogniser) -> Self {
        let m = &recogniser.model;
        let info = ModelInfo { name: m.name.clone(), version: m.version.clone(), sha256: m.sha256[..12].to_string() };
        Daemon { latest: Latest::default(), recogniser, policy: DisplayPolicy::default(), counters: [0; 10], odometer: 0., last_stamp: None, info }
    }

    /// The main loop's body for one new modelV2 (`model_mono`: its logMonoTime, for the carState alive check).
    /// `picked`: FrameBuffer.select's result (frame id, EOF, NV12 view, 'exact'/'nearest', buffer id at read-back).
    /// `clock`: monotonic ns now (called for the drop checks and the publish time).
    pub fn step(&mut self, model: &ModelV2, model_mono: i64, picked: Option<(u32, u64, &Nv12, &'static str, i64)>,
                mut clock: impl FnMut() -> i64, mut drop_check: impl FnMut(i64) -> Option<&'static str>) -> Step {
        let Some((frame_id, frame_eof, image, kind, _recv)) = picked else {
            count(&mut self.counters, "frame_mismatch");
            return Step::Mismatch;
        };
        count(&mut self.counters, kind);
        let calibrated = self.latest.calibration.as_ref().filter(|(c, valid)| *valid && c.cal_status == 1 && c.rpy_calib.len() == 3);
        let car = self.latest.car.filter(|(_, mono)| model_mono - mono < CAR_STATE_ALIVE_NS);
        let started = self.latest.device.is_some_and(|d| d.started);
        let (Some((calibration, _)), Some((car, _)), true, true) = (calibrated, car, model.lane_lines.len() >= 3, started) else {
            count(&mut self.counters, "stale_inputs");
            self.policy.reset();
            return Step::Stale;
        };
        let _ = frame_id;
        if let Some(d) = drop_check(clock()) {
            count(&mut self.counters, if d == "hold" { "drop_hold_before" } else { "drop_frame_id_before" });
            return Step::Dropped(d);
        }
        let start = clock();
        let stamp = frame_eof as i64;
        let offset_s = (frame_eof as i64 - model.timestamp_eof as i64) as f64 / 1e9;
        let v_ego = car.v_ego as f64;
        if let Some(last) = self.last_stamp {
            if 0 < stamp - last && stamp - last < 2_000_000_000 {
                self.odometer += 0f64.max(v_ego) * (stamp - last) as f64 / 1e9;
            }
        }
        self.last_stamp = Some(stamp);
        let device = DEVICE_TYPES[self.latest.device.map_or(0, |d| d.device_type) as usize];
        let sensor = IMAGE_SENSORS[self.latest.sensor.unwrap_or(0) as usize];
        let rpy = [calibration.rpy_calib[0] as f64, calibration.rpy_calib[1] as f64, calibration.rpy_calib[2] as f64];
        let result = (|| {
            let camera = road_camera(device, sensor).ok_or_else(|| anyhow::anyhow!("('{device}', '{sensor}')"))?;
            let k = intrinsics(camera, image.width, image.height);
            let shift = 0f64.max(v_ego) * offset_s;
            let frame = Frame {
                lines: [line_of(&model.lane_lines[1], shift), line_of(&model.lane_lines[2], shift)],
                probs: [model.lane_line_probs.get(1).copied().ok_or_else(|| anyhow::anyhow!("laneLineProbs"))? as f64,
                        model.lane_line_probs.get(2).copied().ok_or_else(|| anyhow::anyhow!("laneLineProbs"))? as f64],
                rpy,
                k,
                stamp,
                odometer: self.odometer,
                blinker: car.left_blinker || car.right_blinker,
                lcs: model.lane_change_state,
                v_ego,
                lead_d_rel: lead_d_rel(self.latest.radar.as_ref(), stamp),
            };
            let raw = self.recogniser.process(image, &frame)?;
            anyhow::Ok((frame, raw))
        })();
        let (frame, raw) = match result {
            Ok(v) => v,
            Err(e) => {
                count(&mut self.counters, "errors");
                self.policy.reset();
                return Step::Error(e.to_string());
            }
        };
        if let Some(d) = drop_check(clock()) {
            count(&mut self.counters, if d == "hold" { "drop_hold_after" } else { "drop_frame_id_after" });
            self.recogniser.reset();
            return Step::Dropped(d);
        }
        let now = clock();
        let confirmed = self.policy.confirm(stamp, &raw, now);
        self.recogniser.observe_confirmation(&confirmed, &frame);
        count(&mut self.counters, "processed");
        let t = self.recogniser.timing;
        let timing = json!({
            "processingMs": (now - start) as f64 / 1e6, "cameraToPublishMs": (now - stamp) as f64 / 1e6,
            "frameOffsetMs": offset_s * 1e3, "frameMatch": kind, "extractMs": t.extract_ms, "inferMs": t.infer_ms,
            "engine": "rust",
        });
        let packet = self.packet(model.frame_id, stamp, &raw, &confirmed, now, timing);
        Step::Published { packet, raw, frame }
    }

    /// marking_daemon.packet (mode 2).
    pub fn packet(&self, frame_id: u32, camera_ns: i64, raw: &[SideResult; 2], confirmed: &[Confirmed; 2], now: i64, timing: Value) -> Value {
        let counters: Map<String, Value> = COUNTERS.iter().zip(self.counters).map(|(k, v)| (k.to_string(), json!(v))).collect();
        let mut data = json!({
            "type": PACKET_TYPE, "version": PACKET_VERSION, "mode": "stripnet", "frameId": frame_id, "cameraMonoTimeNanos": camera_ns,
            "publishMonoTimeNanos": now, "timing": timing, "counters": counters,
            "model": {"name": self.info.name, "version": self.info.version, "sha256": self.info.sha256, "v4": true},
        });
        for (side, name) in ["left", "right"].iter().enumerate() {
            let r = &raw[side];
            let mut rawj = json!({"type": r.kind, "color": r.color, "typeConfidence": r.type_confidence,
                                  "colorConfidence": r.color_confidence, "reason": r.reason});
            if let Some(x) = &r.extras {
                let m = rawj.as_object_mut().unwrap();
                m.insert("typeClass".into(), json!(x.type_class));
                m.insert("colorClass".into(), json!(x.color_class));
                m.insert("double".into(), json!(x.double));
                m.insert("blue".into(), json!(x.blue));
                m.insert("pUnknown".into(), json!(x.p_unknown));
            }
            let mut entry = json!({"raw": rawj});
            for (head, h) in [("type", confirmed[side].kind), ("color", confirmed[side].color)] {
                let e = entry.as_object_mut().unwrap();
                e.insert(head.into(), json!(h.map_or("unknown", |h| h.value)));
                e.insert(format!("{head}AgeNs"), h.map_or(Value::Null, |h| json!(0i64.max(now - h.stamp))));
            }
            data.as_object_mut().unwrap().insert(name.to_string(), entry);
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::Head;
    use crate::model::StripModel;
    use crate::result::filter_input;

    /// A packet as published (JSON in a framed customReservedRawData2 event) for tools/check_packet.py
    /// (pycapnp + marking.parse) to read: target/packet_event.bin.
    #[test]
    fn packet_event_for_python() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = StripModel::load(&dir.join("../assets"), "lane_strip_v4").unwrap();
        let mut d = Daemon::new(Recogniser::new(model));
        d.counters[0] = 7;
        let stamp = 1_000_000_000i64;
        let mut left = filter_input("solid", "yellow", 0.97, 0.99, stamp, false, "strip_model");
        left.extras = Some(crate::result::Extras { type_class: "solid", color_class: "yellow", double: false, blue: false, p_unknown: 1e-6 });
        let right = filter_input("unknown", "unknown", 0., 0., stamp, false, "lane_prob_dip");
        let confirmed = [
            Confirmed { kind: Some(Head { value: "solid", stamp, confidence: 0.97 }), color: Some(Head { value: "yellow", stamp: stamp - 50_000_000, confidence: 0.99 }) },
            Confirmed::default(),
        ];
        let p = d.packet(1234, stamp, &[left, right], &confirmed, stamp + 40_000_000, json!({"processingMs": 1.5}));
        assert_eq!(p["left"]["typeAgeNs"], 40_000_000);
        assert_eq!(p["left"]["colorAgeNs"], 90_000_000);
        assert_eq!(p["right"]["type"], "unknown");
        assert!(p["right"]["typeAgeNs"].is_null());
        let event = crate::messages::raw_data_event(EVENT_CUSTOM_RAW_2, 5, &serde_json::to_vec(&p).unwrap());
        std::fs::write(dir.join("target/packet_event.bin"), event).unwrap();
    }

    #[test]
    fn frame_buffer_selects_exact_then_nearest() {
        let mut b: FrameBuffer<()> = FrameBuffer::default();
        for (i, eof) in [(10, 1_000), (11, 51_000_000), (12, 101_000_000), (13, 151_000_000)] {
            b.push(Buffered { frame_id: i, eof, buf: (), recv_ns: 0 });
        }
        assert_eq!(b.frames.len(), 3);
        assert_eq!(b.select(12, 101_000_000).map(|(f, k)| (f.frame_id, k)), Some((12, "exact")));
        assert_eq!(b.select(99, 140_000_000).map(|(f, k)| (f.frame_id, k)), Some((13, "nearest")));
        assert!(b.select(99, 1_000).is_none());
        assert_eq!(buffer_drop_reason(5, 5, 0, MAX_HOLD_NS), Some("hold"));
        assert_eq!(buffer_drop_reason(5, 6, 0, 1), Some("frame_id"));
        assert_eq!(buffer_drop_reason(5, 5, 0, 1), None);
    }

    #[test]
    fn camera_table() {
        assert_eq!(road_camera("mici", "os04c10"), Some((1344, 760, 1141.5)));
        assert_eq!(road_camera("tici", "ar0231"), Some((1928, 1208, 2648.)));
        assert_eq!(road_camera("mici", "unknown"), None);
        let k = intrinsics((1344, 760, 1141.5), 1344, 760);
        assert_eq!(k[0], [1141.5, 0., 672.]);
    }
}
