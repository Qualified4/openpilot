use std::io::{Cursor, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use ccnc_map::{
    cmap::{decode, encode, schema, MapOut},
    input::framing,
    pipeline::{Pipeline, STALE_NS},
    slots::Slot,
};

fn fixture(mode: &str) -> Command {
    let mut cmd = Command::new(std::env::var_os("PYO3_PYTHON").unwrap_or_else(|| "python3".into()));
    cmd.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cmap_fixture.py"))
        .arg(mode);
    cmd
}

fn from_python(mode: &str) -> Vec<u8> {
    let out = fixture(mode).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn cmap_schema_matches_capnp() {
    let values = String::from_utf8(from_python("schema")).unwrap();
    let expected: Vec<usize> = values
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let s = schema();
    let actual: Vec<usize> = [
        s.event_discriminant.into(),
        s.event_pointer,
        s.slots_pointer,
    ]
    .into_iter()
    .chain(s.map_fields)
    .chain(s.slot_fields)
    .chain(s.data_words)
    .chain(s.pointer_words)
    .collect();
    assert_eq!(actual, expected);
}

#[test]
fn cmap_roundtrip_matches_pycapnp() {
    let mut slots = [None; 5];
    slots[0] = Some(Slot {
        vid: 9,
        distance: 0.,
        lateral: 0.,
        detect: 0,
        lane_coord_rel: 0.,
        v_rel: 0.,
        kind: 0,
        kind_confirmed: false,
        kind_wanted: true,
        vision: 0,
    });
    slots[3] = Some(Slot {
        vid: 42,
        distance: -2.5,
        lateral: 3.25,
        detect: 6,
        lane_coord_rel: -0.75,
        v_rel: -1.5,
        kind: 3,
        kind_confirmed: true,
        kind_wanted: false,
        vision: 1,
    });
    slots[4] = Some(Slot {
        vid: 0,
        distance: 0.,
        lateral: -2.75,
        detect: 6,
        lane_coord_rel: 0.,
        v_rel: 0.,
        kind: 0,
        kind_confirmed: false,
        kind_wanted: false,
        vision: 0,
    });
    let expected = MapOut {
        radar_log_mono_time: 123456789012,
        generation: 17,
        status: 11,
        ego_lane_coord: -1.25,
        slots,
        valid: true,
    };
    assert_eq!(decode(&from_python("encode")), Some(expected));
    let bytes = encode(&expected);
    assert_eq!(decode(&bytes), Some(expected));
    assert_eq!(decode(&bytes[..bytes.len() - 8]), None);
    let invalid = MapOut {
        valid: false,
        ..expected
    };
    assert_eq!(decode(&encode(&invalid)), Some(invalid));

    let mut python = fixture("decode").stdin(Stdio::piped()).spawn().unwrap();
    python.stdin.take().unwrap().write_all(&bytes).unwrap();
    assert!(python.wait().unwrap().success());
}

fn pipeline_fixture() -> (Pipeline, Cursor<Vec<u8>>, MapOut) {
    let mut pipe = Pipeline::new(723);
    let mut input = Cursor::new(from_python("events"));
    let mut frame = Vec::new();
    loop {
        assert!(framing::next(&mut input, &mut frame).unwrap());
        if let Some(map) = pipe.ingest(&frame).unwrap() {
            assert!(map.valid);
            return (pipe, input, map);
        }
    }
}

#[test]
fn stale_inputs_mark_invalid() {
    let (mut pipe, _, before) = pipeline_fixture();
    let stale = before.radar_log_mono_time + STALE_NS + 1;
    assert_eq!(pipe.expire(stale - 1), None);
    let after = pipe.expire(stale).unwrap();
    assert!(!after.valid);
    assert_eq!(after.radar_log_mono_time, before.radar_log_mono_time);
    assert_eq!(after.slots, [None; 5]);
    assert_eq!(after.generation, before.generation + 1);
    assert_eq!(pipe.expire(stale + STALE_NS), None);
}

#[test]
fn a_blind_spot_warning_round_trips_without_creating_a_car() {
    let (_, _, map) = pipeline_fixture();
    let wire = decode(&encode(&map)).unwrap();
    assert_eq!(wire.status & (ccnc_map::cmap::STATUS_BSD_LEFT | ccnc_map::cmap::STATUS_BSD_RIGHT),
               ccnc_map::cmap::STATUS_BSD_LEFT);
    assert_eq!(wire.slots, [None; 5]);
}

#[test]
fn generation_increments_on_reset() {
    let (mut pipe, mut input, before) = pipeline_fixture();
    let mut frame = Vec::new();
    assert!(framing::next(&mut input, &mut frame).unwrap());
    assert_eq!(pipe.ingest(&frame).unwrap(), None);
    assert!(framing::next(&mut input, &mut frame).unwrap());
    let after = pipe.ingest(&frame).unwrap().unwrap();
    assert!(!after.valid);
    assert_eq!(after.generation, before.generation + 1);
}
