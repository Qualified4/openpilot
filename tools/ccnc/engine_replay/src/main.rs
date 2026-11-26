//! Replay exact daemon input messages at liveTracks cadence without publishing any output.
//! Emits a segment header and one tab-separated record per radar tick for offline comparison.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

use ccnc_map::engine::{
    model_changing_lanes, EngineInput, EngineOutput, MapEngine, RoadObservation, VehicleKind,
    VisionObservation,
};
use ccnc_map::input::{
    framing,
    live_tracks::{self, EVENT_LIVE_TRACKS},
    map_events::{CarState, DisplayLead, Event, CAR_STATE, LIVE_POSE, RADAR_STATE},
    model_v2::{self, Model, EVENT_MODEL_V2},
};
use ccnc_map::vision::kind::{FrontCam, KindResult, KindView};
use ccnc_vision_msg::{Calibration, Status, View, VisionOut, EVENT_CCNC_VISION};
use det_rules::{Calib, WideDet, WideGeometry, CAM2B};

struct Road {
    time: f64,
    path: Vec<(f64, f64)>,
    lines: Option<Model>,
    changing: bool,
}

#[derive(Default)]
struct Camera {
    wide: Option<(Calibration, WideGeometry)>,
}

impl Camera {
    fn observations(
        &mut self,
        published: u64,
        message: VisionOut,
        engine: &mut MapEngine,
        out: &mut Vec<VisionObservation>,
    ) -> usize {
        let calibration = message.calibration;
        let rpy = calibration.rpy_calib.map(f64::from);
        let height = f64::from(calibration.height);
        if !(0.5..=3.).contains(&height) {
            return 0;
        }
        let front = FrontCam::new(rpy, height);
        let mut kind_count = 0;
        if let Some(front) = &front {
            let kinds: Vec<_> = message
                .kinds
                .iter()
                .filter(|k| k.view == View::CropFront)
                .map(|k| KindResult {
                    time_ns: k.frame_mono_time,
                    view: KindView::Front,
                    xyxy: k.xyxy.map(f64::from),
                    kind: k.kind,
                    prob: f64::from(k.prob),
                })
                .collect();
            kind_count = kinds.len();
            if !kinds.is_empty() {
                engine.observe_front_kinds(published, front, &kinds);
            }
            if message.status == Status::Frame && message.view == View::CropFront {
                let boxes: Vec<_> = message
                    .boxes
                    .iter()
                    .filter(|b| {
                        b.confidence >= 0.55 && det_rules::VEH.contains(&i32::from(b.class_id))
                    })
                    .map(|b| b.xyxy.map(f64::from))
                    .collect();
                engine.observe_front_boxes(message.frame_mono_time, front, &boxes);
            }
        }
        if message.status != Status::Frame {
            return kind_count;
        }
        let time = secs(message.frame_mono_time);
        if let Some(view) = wide_view(message.view) {
            let boxes: Vec<_> = message
                .boxes
                .iter()
                .map(|b| WideDet {
                    view,
                    bbox: b.xyxy.map(f64::from),
                    score: f64::from(b.confidence),
                    class: i32::from(b.class_id),
                })
                .collect();
            for c in self.project_wide(calibration, &boxes) {
                let kind = match c.class {
                    2 => VehicleKind::Car,
                    5 | 7 => VehicleKind::Heavy,
                    _ => VehicleKind::Unknown,
                };
                out.push(VisionObservation {
                    time,
                    x: c.x,
                    y: c.yc,
                    kind,
                    confidence: c.score,
                    position: true,
                });
            }
        } else {
            for b in message
                .boxes
                .iter()
                .filter(|b| !b.cut && det_rules::VEH.contains(&i32::from(b.class_id)))
            {
                if let Some((x, y)) = front_ground(front.as_ref(), b.xyxy.map(f64::from)) {
                    out.push(VisionObservation {
                        time,
                        x,
                        y,
                        kind: VehicleKind::Unknown,
                        confidence: f64::from(b.confidence),
                        position: true,
                    });
                }
            }
        }
        kind_count
    }

    fn project_wide(&mut self, cal: Calibration, detections: &[WideDet]) -> Vec<det_rules::Cand> {
        let rpy = cal.rpy_calib.map(f64::from);
        let wide = cal.wide_from_device_euler.map(f64::from);
        let height = f64::from(cal.height);
        if !rpy.iter().chain(&wide).all(|v| v.is_finite()) {
            return Vec::new();
        }
        if self.wide.as_ref().is_none_or(|(old, _)| {
            old.rpy_calib != cal.rpy_calib
                || old.wide_from_device_euler != cal.wide_from_device_euler
        }) {
            self.wide = Some((
                cal,
                WideGeometry::new(&Calib {
                    rpy,
                    wide_from_device: wide,
                    height,
                }),
            ));
        }
        let (_, geometry) = self.wide.as_mut().expect("calibration installed");
        geometry.cam.h = height;
        geometry
            .candidates(detections)
            .into_iter()
            .filter(|c| !c.near03)
            .collect()
    }
}

fn front_ground(camera: Option<&FrontCam>, [x1, _, x2, bottom]: [f64; 4]) -> Option<(f64, f64)> {
    let (x, y) = camera?.ground_from_pixel((x1 + x2) / 2., bottom)?;
    (CAM2B..=130.).contains(&x).then_some((x - CAM2B, y))
}

fn wide_view(view: View) -> Option<det_rules::View> {
    match view {
        View::Full => Some(det_rules::View::Raw),
        View::SideL => Some(det_rules::View::SideL),
        View::SideR => Some(det_rules::View::SideR),
        View::CropFront => None,
    }
}

fn secs(stamp: u64) -> f64 {
    stamp as f64 * 1e-9
}

fn write_tick(
    out: &mut impl Write,
    stamp: u64,
    valid: bool,
    output: EngineOutput,
) -> io::Result<()> {
    write!(out, "{stamp}\t{}", u8::from(valid))?;
    for slot in output.slots {
        match slot {
            Some(s) => write!(
                out,
                "\t{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:?}",
                s.obj_id,
                s.last_track_id.unwrap_or(0),
                s.x,
                s.y,
                s.s,
                s.v_s,
                s.existence_p,
                s.kind
            )?,
            None => write!(out, "\t-")?,
        }
    }
    write!(out, "\t")?;
    for o in output.objects {
        write!(
            out,
            "{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{},{},{};",
            o.obj_id,
            o.evidence.last_track_id.unwrap_or(0),
            o.s,
            o.d,
            o.lane_u,
            o.v_s,
            o.existence_p(),
            u8::from(o.evidence.last_radar_t == Some(secs(stamp))),
            u8::from(
                o.evidence
                    .last_vision_t
                    .is_some_and(|t| secs(stamp) - t <= 0.14)
            ),
            u8::from(
                o.evidence
                    .last_lead_t
                    .is_some_and(|t| secs(stamp) - t <= 0.16)
            )
        )?;
    }
    writeln!(out)
}

fn replay(path: &Path, out: &mut impl Write) -> io::Result<()> {
    let mut engine = MapEngine::default();
    let mut camera = Camera::default();
    let (mut car, mut yaw, mut lead): (
        Option<(u64, CarState)>,
        Option<(u64, f64)>,
        Option<(f64, DisplayLead)>,
    ) = (None, None, None);
    let mut road = None;
    let mut vision = Vec::new();
    let mut camera_inputs = [0; 2];
    let mut reader = BufReader::new(File::open(path)?);
    let mut bytes = Vec::new();
    writeln!(
        out,
        "# {}",
        path.file_stem().unwrap_or_default().to_string_lossy()
    )?;
    while framing::next(&mut reader, &mut bytes)? {
        let Some(event) = Event::read(&bytes) else {
            return Err(io::Error::other("invalid cereal Event"));
        };
        let stamp = event.stamp;
        match event.which {
            CAR_STATE => {
                car = event
                    .car_state()
                    .filter(|c| event.valid && c.can_valid && !c.reverse && c.speed.is_finite())
                    .map(|c| (stamp, c))
            }
            LIVE_POSE if event.valid => {
                if let Some(rate) = event.yaw() {
                    yaw = Some((stamp, rate));
                }
            }
            EVENT_MODEL_V2 if event.valid => {
                if let Some(((x, y), _)) = model_v2::parse_path(&bytes) {
                    road = Some(Road {
                        time: secs(stamp),
                        path: x.into_iter().zip(y).collect(),
                        lines: model_v2::parse(&bytes),
                        changing: model_changing_lanes(&bytes),
                    });
                }
            }
            RADAR_STATE => {
                lead = event
                    .display_leads()
                    .1
                    .filter(|_| event.valid)
                    .map(|p| (secs(stamp), p))
            }
            EVENT_CCNC_VISION if event.valid => {
                if let Some((_, message)) = ccnc_vision_msg::decode(&bytes) {
                    camera_inputs[1] +=
                        camera.observations(stamp, message, &mut engine, &mut vision);
                }
            }
            EVENT_LIVE_TRACKS => {
                let Some((_, car)) = car.filter(|(t, _)| stamp.abs_diff(*t) <= 250_000_000) else {
                    write_tick(out, stamp, false, EngineOutput::default())?;
                    vision.clear();
                    continue;
                };
                let Some(radar) = event.valid.then(|| live_tracks::parse(&bytes)).flatten() else {
                    write_tick(out, stamp, false, EngineOutput::default())?;
                    vision.clear();
                    continue;
                };
                camera_inputs[0] += vision.len();
                let result = engine.step(EngineInput {
                    time: secs(stamp),
                    speed: car.speed,
                    yaw: yaw
                        .filter(|(t, _)| stamp.abs_diff(*t) <= 500_000_000)
                        .map(|(_, y)| y),
                    road: road.as_ref().map(|r| RoadObservation {
                        time: r.time,
                        path: &r.path,
                        lines: r.lines.as_ref(),
                        changing_lanes: r.changing,
                    }),
                    radar: &radar,
                    lead,
                    vision: &vision,
                    blind: car.blind,
                });
                write_tick(out, stamp, true, result)?;
                vision.clear();
            }
            _ => {}
        }
    }
    eprintln!(
        "{}: {} projected camera positions, {} front kind answers",
        path.file_stem().unwrap_or_default().to_string_lossy(),
        camera_inputs[0],
        camera_inputs[1]
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let mut paths = std::env::args_os().skip(1).peekable();
    if paths.peek().is_none() {
        return Err(io::Error::other(
            "usage: engine_replay CACHED_EVENTS.capnp ...",
        ));
    }
    let mut out = BufWriter::new(io::stdout().lock());
    for path in paths {
        replay(Path::new(&path), &mut out)?;
    }
    out.flush()
}
