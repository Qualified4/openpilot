//! Evaluate the new world geometry on the existing clean-map text exports.
//!
//! eval INPUT_DIR OUTPUT_DIR [--no-freeze] [--reference TRACE_DIR] [--gate body|shorter|conservative]
//! By default, use the vehicle layer. A reference trace supplies the prototype's
//! identities and reflection offsets to isolate the geometry as in Task 3.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use ccnc_map::input::model_v2::{Line, Model};
use ccnc_map::vehicles::{LinkGates, Tracker};
use ccnc_map::world::{near_heading, World, MODEL_GRID, RADAR_LAG, STATUS_FROZEN};

type Reference = BTreeMap<i64, BTreeMap<u32, (u32, f64)>>;

fn main() -> io::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "eval INPUT_DIR OUTPUT_DIR [--no-freeze] [--reference TRACE_DIR] [--gate body|shorter|conservative]",
        )
    };
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut freeze = true;
    let mut reference = None;
    let mut gates = None;
    while let Some(arg) = args.next() {
        if arg == "--no-freeze" {
            freeze = false;
        } else if arg == "--reference" {
            reference = Some(PathBuf::from(args.next().ok_or_else(usage)?));
        } else if arg == "--gate" {
            gates = Some(match args.next().as_deref().and_then(|s| s.to_str()) {
                Some("body") => LinkGates::BODY,
                Some("shorter") => LinkGates::SHORTER,
                Some("conservative") => LinkGates::CONSERVATIVE,
                _ => return Err(usage()),
            });
        } else {
            return Err(usage());
        }
    }
    fs::create_dir_all(&output)?;
    let mut files: Vec<_> = fs::read_dir(input)?.collect::<Result<_, _>>()?;
    files.sort_unstable_by_key(|entry| entry.file_name());
    for entry in files {
        let path = entry.path();
        if path.extension().is_some_and(|s| s == "txt") {
            let name = path.file_stem().unwrap();
            let bindings = reference
                .as_ref()
                .map(|dir| dir.join(name).with_extension("trace"))
                .filter(|trace| trace.is_file())
                .map(|p| load_reference(&p))
                .transpose()?;
            run(
                &path,
                &output.join(name).with_extension("trace"),
                freeze,
                bindings.as_ref(),
                gates,
            )?;
        }
    }
    Ok(())
}

fn fields(line: &str) -> Vec<&str> {
    line.split_ascii_whitespace().collect()
}

fn number(p: &[&str], i: usize) -> f64 {
    p.get(i).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN)
}

fn time_key(t: f64) -> i64 {
    (t * 10000.).round() as i64
}

fn load_reference(path: &Path) -> io::Result<Reference> {
    let mut rows = Reference::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let p = fields(&line);
        if p.first() == Some(&"V") {
            let rid = number(&p, 3);
            if rid > 0. {
                rows.entry(time_key(number(&p, 1)))
                    .or_default()
                    .insert(rid as u32, (number(&p, 2) as u32, number(&p, 7)));
            }
        }
    }
    Ok(rows)
}

#[derive(Default)]
struct CamRef {
    last: Option<(f64, f64)>,
    u: f64,
}

impl CamRef {
    fn observe(&mut self, t: f64, left: f64, right: f64) {
        if !(1.8..=5.5).contains(&(left - right)) {
            return;
        }
        let relative = -0.5 + left / (left - right);
        if let Some((last, old)) = self.last {
            let delta = relative
                - if t - last <= 0.3 {
                    old
                } else {
                    self.u - self.u.floor() - 0.5
                };
            self.u += delta - delta.round();
        } else {
            self.u = relative;
        }
        self.last = Some((t, relative));
    }
}

struct RadarReport {
    stamp: f64,
    vid: u32,
    x: f64,
    y: f64,
    v_rel: f64,
}

impl RadarReport {
    fn end(&self, events: &mut impl Write, id: u32) -> io::Result<()> {
        writeln!(
            events,
            "E {:.4} {id} {} {:.2} {:.3} {:.3}",
            self.stamp, self.vid, self.x, self.y, self.v_rel
        )
    }
}

struct Replay<'a> {
    world: World,
    reference: Option<&'a Reference>,
    camera: CamRef,
    live_pose: bool,
    speed: f64,
    heading: f64,
    lead: i64,
    blinker: bool,
    t: f64,
    points: Vec<(u32, f64, f64, f64)>,
    radar_frame: bool,
    tracker: Tracker,
    reports: BTreeMap<u32, RadarReport>,
    coord: BTreeMap<u32, (f64, f64, u32)>,
}

impl Replay<'_> {
    fn flush(&mut self, out: &mut impl Write, events: &mut impl Write) -> io::Result<()> {
        if !std::mem::take(&mut self.radar_frame) {
            return Ok(());
        }
        if self.reference.is_some() && self.points.is_empty() {
            return Ok(());
        }
        let generation = self.world.generation();
        self.world.radar(self.t);
        if self.world.generation() != generation {
            self.coord.clear();
        }
        let u = self.world.u_ego();
        let cam = self.camera.last.map_or(f64::NAN, |_| self.camera.u);
        writeln!(
            out,
            "G {:.4} {:.4} {} {} {:.3} {:.4} {} {} {:.4} {}",
            self.t,
            u,
            (self.world.status() & STATUS_FROZEN != 0) as u8,
            self.lead,
            self.speed,
            cam,
            self.world.lanes.renumbers(),
            self.live_pose as u8,
            self.heading,
            self.blinker as u8
        )?;
        let ego = self.world.pose.at(self.t);
        if self.reference.is_none() {
            return self.flush_vehicles(out, events, u);
        }
        let reference = self.reference.and_then(|r| r.get(&time_key(self.t)));
        self.coord.retain(|_, (t, _, _)| self.t - *t <= 2.);
        for (rid, dx, dy, _) in self.points.drain(..) {
            let Some((vid, off)) = reference.and_then(|r| r.get(&rid).copied()) else {
                continue;
            };
            let off = if off.is_finite() { off } else { 0. };
            let (wx, wy) = self.world.radar_point_world(self.t, dx, dy);
            let (x, y) = ego.to_ego(wx, wy);
            let Some(road) =
                self.world
                    .vehicle_lane_coord(self.t, ego.s + x, ego.to_world(x, y - off))
            else {
                continue;
            };
            let estimate = self.coord.entry(vid).or_insert((self.t, road, 0));
            estimate.1 = if estimate.2 >= 3 {
                estimate.1 + (0.3 * (road - estimate.1)).clamp(-0.1, 0.1)
            } else {
                road
            };
            estimate.0 = self.t;
            estimate.2 += 1;
            let r = estimate.1 - u;
            if (-5. ..=80.).contains(&x) && r.abs() < 2.5 {
                writeln!(
                    out,
                    "V {:.4} {vid} {rid} {x:.2} {y:.3} {r:.4} {off:.3}",
                    self.t
                )?;
            }
        }
        Ok(())
    }

    fn flush_vehicles(
        &mut self,
        out: &mut impl Write,
        events: &mut impl Write,
        ego_u: f64,
    ) -> io::Result<()> {
        let mut prior = std::mem::take(&mut self.reports);
        let cars = self.tracker.update(self.t, &self.points, &self.world);
        let ego = self.world.pose.at(self.t);
        for car in cars.iter().filter(|car| car.seen_radar == self.t) {
            let Some(rid) = car.radar_id else { continue };
            let Some(&(_, dx, dy, v_rel)) = self.points.iter().find(|p| p.0 == rid) else {
                continue;
            };
            let vid = car.vid;
            let previous = prior.remove(&rid);
            if previous.as_ref().is_none_or(|p| p.vid != vid) {
                if let Some(p) = previous {
                    p.end(events, rid)?;
                }
                let mut ids = car.ids.iter().rev();
                let from = if ids.next().is_some_and(|(t, _)| t == self.t) {
                    ids.next().map_or(-1, |(_, old)| i64::from(old))
                } else {
                    -1
                };
                writeln!(
                    events,
                    "B {:.4} {rid} {vid} {from} {dx:.2} {dy:.3} {v_rel:.3}",
                    self.t
                )?;
            }
            self.reports.insert(
                rid,
                RadarReport {
                    stamp: self.t,
                    vid,
                    x: dx,
                    y: dy,
                    v_rel,
                },
            );
            let (x, y) = ego.to_ego(f64::from(car.wx), f64::from(car.wy));
            let r = f64::from(car.u) - ego_u;
            if (-5. ..=80.).contains(&x) && r.abs() < 2.5 {
                let off = car.off;
                writeln!(
                    out,
                    "V {:.4} {vid} {rid} {x:.2} {y:.3} {r:.4} {off:.3}",
                    self.t
                )?;
            }
        }
        for (rid, p) in prior {
            p.end(events, rid)?;
        }
        self.points.clear();
        Ok(())
    }
}

fn run(
    input: &Path,
    output: &Path,
    freeze: bool,
    reference: Option<&Reference>,
    gates: Option<LinkGates>,
) -> io::Result<()> {
    let text = fs::read_to_string(input)?;
    // The existing C export includes interpolated yaw. Y records retain the
    // original livePose times in newer exports and allow dropouts in replay.
    let explicit_pose = text.lines().any(|s| s.starts_with("Y "));
    let live_pose = explicit_pose
        || text.lines().filter(|s| s.starts_with("C ")).any(|s| {
            let yaw = number(&fields(s), 3);
            yaw.is_finite() && yaw != 0.
        });
    let mut out = BufWriter::new(File::create(output)?);
    let mut events = BufWriter::new(File::create(output.with_extension("events"))?);
    let mut replay = Replay {
        world: World::new(RADAR_LAG, freeze),
        reference,
        camera: CamRef::default(),
        live_pose,
        speed: 0.,
        heading: 0.,
        lead: -1,
        blinker: false,
        t: f64::NAN,
        points: Vec::new(),
        radar_frame: false,
        tracker: gates.map(Tracker::new).unwrap_or_default(),
        reports: BTreeMap::new(),
        coord: BTreeMap::new(),
    };
    for line in text.lines() {
        let p = fields(line);
        let t = number(&p, 1);
        if !t.is_finite() {
            continue;
        }
        if t != replay.t {
            replay.flush(&mut out, &mut events)?;
            replay.t = t;
        }
        match p[0] {
            "C" => {
                let generation = replay.world.generation();
                if live_pose && !explicit_pose {
                    replay.world.live_pose(t, number(&p, 3));
                }
                replay.speed = number(&p, 2);
                replay
                    .world
                    .car_state(t, replay.speed, p.get(4) == Some(&"1"));
                if replay.world.generation() != generation {
                    replay.coord.clear();
                }
            }
            "Y" => replay.world.live_pose(t, number(&p, 2)),
            "L" => {
                let model = Model {
                    lane_lines: (0..4)
                        .map(|i| Line {
                            x: MODEL_GRID.to_vec(),
                            y: (0..10)
                                .map(|k| -number(&p, 6 + i * 10 + k) as f32)
                                .collect(),
                        })
                        .collect(),
                    lane_line_probs: (2..6).map(|i| number(&p, i) as f32).collect(),
                    road_edges: Vec::new(),
                };
                replay.camera.observe(t, number(&p, 16), number(&p, 26));
                if let Some((heading, _)) = near_heading(&model) {
                    replay.heading = heading;
                }
                replay.world.model(t, &model);
            }
            "T" => {
                replay.radar_frame = true;
                let (rid, x, y, speed) =
                    (number(&p, 2), number(&p, 3), number(&p, 4), number(&p, 5));
                if number(&p, 9) == 0. && rid > 0. && x > 0. && x <= 200. && y.abs() <= 15. {
                    replay.points.push((rid as u32, x, y, speed));
                }
            }
            "S" => {
                replay.blinker = number(&p, 2) > 0.5 || number(&p, 3) > 0.5;
                replay.world.blinker(replay.blinker);
            }
            "D" => {
                replay.lead = if number(&p, 2) > 0.5 {
                    number(&p, 6) as i64
                } else {
                    -1
                }
            }
            _ => {}
        }
    }
    replay.flush(&mut out, &mut events)?;
    for (rid, report) in replay.reports {
        report.end(&mut events, rid)?;
    }
    events.flush()?;
    out.flush()
}
