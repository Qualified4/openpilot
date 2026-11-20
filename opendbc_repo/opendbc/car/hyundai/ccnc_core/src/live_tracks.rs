//! `liveTracks.points` read from the bytes of their `log.Event` message, without a compiled schema.
//!
//! Only the positions this reads are kept here, as constants; `tests/test_ccnc_core_live_tracks.py` compares them with
//! the schema pycapnp loads (`openpilot/cereal/log.capnp`, `opendbc/car/car.capnp`), so a schema change fails that test
//! and is fixed here when it is merged. Every pointer and offset is bounds-checked. A message this does not read
//! exactly as pycapnp would (another union member, a capability or list form it does not take, an enum value the
//! schema does not name, bytes that do not frame a message) gives None, and the caller reads the Python objects.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};

use crate::guard;

/// `Event`: the union discriminant (in 16-bit units of the data section) and the liveTracks member.
pub const EVENT_DISCRIMINANT_OFFSET: usize = 4;
pub const EVENT_LIVE_TRACKS: u16 = 129;
pub const EVENT_LIVE_TRACKS_POINTER: usize = 0;
/// `RadarData.points`: its pointer index.
pub const RADAR_DATA_POINTS_POINTER: usize = 1;
/// `RadarPoint` fields: offsets in units of the field's size (bits for Bool), and default bits (XORed on read).
pub const TRACK_ID: (usize, u64) = (0, 0); // UInt64
pub const D_REL: (usize, u32) = (2, 0); // Float32
pub const Y_REL: (usize, u32) = (3, 0);
pub const V_REL: (usize, u32) = (4, 0);
pub const A_REL: (usize, u32) = (5, 0);
pub const YV_REL: (usize, u32) = (6, 0);
pub const MEASURED: (usize, bool) = (224, false); // Bool
pub const V_LEAD: (usize, u32) = (8, 0);
pub const A_LEAD: (usize, u32) = (9, 0);
pub const J_LEAD: (usize, u32) = (10, 0);
pub const RADAR_SOURCE: (usize, u16) = (15, 0); // Enum RadarSource
pub const TRACK_STATE: (usize, u8) = (29, 0); // UInt8
pub const MOTION_STATE: (usize, u8) = (44, 0); // UInt8
/// `RadarPoint.RadarSource` names by value: `str()` of the pycapnp enum.
pub const RADAR_SOURCES: [&str; 5] = ["frontRadar", "scc", "corner235", "corner180", "corner430"];

/// The segment count capnp's FlatArrayMessageReader takes (it rejects 512 or more).
const MAX_SEGMENTS: usize = 511;

/// One RadarPoint as pycapnp reads it.
pub struct RawPoint {
    pub track_id: u64,
    pub d_rel: f32,
    pub y_rel: f32,
    pub v_rel: f32,
    pub a_rel: f32,
    pub yv_rel: f32,
    pub measured: bool,
    pub v_lead: f32,
    pub a_lead: f32,
    pub j_lead: f32,
    pub source: u16, // A RADAR_SOURCES index.
    pub track_state: u8,
    pub motion_state: u8,
}

struct Message<'a> {
    segments: Vec<&'a [u8]>,
}

/// A struct's sections: data bytes and the segment/word where its pointers start.
struct Struct<'a> {
    data: &'a [u8],
    segment: usize,
    pointers: usize,
    pointer_count: usize,
}

impl Struct<'_> {
    /// The struct a null pointer reads as: all fields at their defaults.
    fn empty() -> Self {
        Struct { data: &[], segment: 0, pointers: 0, pointer_count: 0 }
    }
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at.checked_add(4)?)?.try_into().ok()?))
}

/// Fields past the end of the data section (a message from an older schema) read as their defaults.
fn data_bytes<const N: usize>(data: &[u8], at: usize) -> [u8; N] {
    data.get(at..at + N).and_then(|b| b.try_into().ok()).unwrap_or([0; N])
}

impl<'a> Message<'a> {
    /// The framing pycapnp's `to_bytes()` writes and messaging sends: segment count - 1, each segment's size in
    /// words, padding to a word, the segments. The bytes must hold exactly that.
    fn new(bytes: &'a [u8]) -> Option<Self> {
        let count = (u32_at(bytes, 0)? as usize).checked_add(1)?;
        if count > MAX_SEGMENTS {
            return None;
        }
        let header = (4 + 4 * count).div_ceil(8) * 8;
        let mut segments = Vec::with_capacity(count);
        let mut start = header;
        for i in 0..count {
            let words = u32_at(bytes, 4 + 4 * i)? as usize;
            let end = start.checked_add(words.checked_mul(8)?)?;
            segments.push(bytes.get(start..end)?);
            start = end;
        }
        (start == bytes.len()).then_some(Message { segments })
    }

    fn word(&self, segment: usize, index: usize) -> Option<u64> {
        let at = index.checked_mul(8)?;
        let bytes = self.segments.get(segment)?.get(at..at + 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }

    fn words(&self, segment: usize) -> usize {
        self.segments[segment].len() / 8
    }

    /// The pointer at (segment, index), past far pointers: None for null, else the segment and word where the object
    /// starts and the struct or list pointer word that describes it.
    fn resolve(&self, segment: usize, index: usize) -> Option<Option<(usize, usize, u64)>> {
        let pointer = self.word(segment, index)?;
        if pointer == 0 {
            return Some(None);
        }
        let near = |segment: usize, index: usize, pointer: u64| -> Option<(usize, usize, u64)> {
            // Offset: signed 30 bits, in words from the end of the pointer.
            let offset = (pointer as u32 as i32 >> 2) as i64;
            let start = usize::try_from(index as i64 + 1 + offset).ok()?;
            Some((segment, start, pointer))
        };
        match pointer & 3 {
            0 | 1 => near(segment, index, pointer).map(Some),
            2 => {
                let target = (pointer >> 32) as usize;
                let pad = ((pointer as u32) >> 3) as usize;
                if pointer & 4 == 0 {
                    // The landing pad is an ordinary pointer in the target segment.
                    let landing = self.word(target, pad)?;
                    if landing & 3 > 1 || landing == 0 {
                        return None;
                    }
                    near(target, pad, landing).map(Some)
                } else {
                    // Two words: a far pointer to the object's start, then the tag that describes it.
                    let far = self.word(target, pad)?;
                    let tag = self.word(target, pad.checked_add(1)?)?;
                    if far & 7 != 2 || tag & 3 > 1 {
                        return None;
                    }
                    let segment = (far >> 32) as usize;
                    if segment >= self.segments.len() {
                        return None;
                    }
                    Some(Some((segment, ((far as u32) >> 3) as usize, tag)))
                }
            }
            _ => None, // A capability.
        }
    }

    fn read_struct(&self, segment: usize, index: usize) -> Option<Struct<'a>> {
        let Some((segment, start, tag)) = self.resolve(segment, index)? else {
            return Some(Struct::empty());
        };
        if tag & 3 != 0 {
            return None;
        }
        let data_words = ((tag >> 32) & 0xffff) as usize;
        let pointer_count = (tag >> 48) as usize;
        let data_end = start.checked_add(data_words)?;
        if data_end.checked_add(pointer_count)? > self.words(segment) {
            return None;
        }
        Some(Struct {
            data: &self.segments[segment][start * 8..data_end * 8],
            segment,
            pointers: data_end,
            pointer_count,
        })
    }

    /// Pointer `index` of a struct; one past its pointer section reads as null.
    fn struct_pointer(&self, s: &Struct<'a>, index: usize) -> Option<Struct<'a>> {
        if index >= s.pointer_count {
            return Some(Struct::empty());
        }
        self.read_struct(s.segment, s.pointers + index)
    }

    /// A list of structs in the inline composite form pycapnp writes; elements whose sections are smaller or larger
    /// than this schema's (another schema version) read as such. Other list forms give None.
    fn struct_list(&self, s: &Struct<'a>, index: usize) -> Option<Vec<&'a [u8]>> {
        if index >= s.pointer_count {
            return Some(Vec::new());
        }
        let Some((segment, start, tag)) = self.resolve(s.segment, s.pointers + index)? else {
            return Some(Vec::new());
        };
        if tag & 3 != 1 || (tag >> 32) & 7 != 7 {
            return None;
        }
        let words = (tag >> 35) as usize;
        let element_tag = self.word(segment, start)?;
        if start.checked_add(1)?.checked_add(words)? > self.words(segment) || element_tag & 3 != 0 {
            return None;
        }
        let count = ((element_tag as u32) >> 2) as usize;
        let data_words = ((element_tag >> 32) & 0xffff) as usize;
        let step = data_words + (element_tag >> 48) as usize;
        if count.checked_mul(step)? > words || (step == 0 && count > 0) {
            return None;
        }
        let seg = self.segments[segment];
        Some((0..count).map(|i| {
            let at = (start + 1 + i * step) * 8;
            &seg[at..at + data_words * 8]
        }).collect())
    }
}

fn f32_field(data: &[u8], (offset, default): (usize, u32)) -> f32 {
    f32::from_bits(u32::from_le_bytes(data_bytes(data, offset * 4)) ^ default)
}

fn point(data: &[u8]) -> Option<RawPoint> {
    let source = u16::from_le_bytes(data_bytes(data, RADAR_SOURCE.0 * 2)) ^ RADAR_SOURCE.1;
    if source as usize >= RADAR_SOURCES.len() {
        return None; // pycapnp raises on an enum value the schema does not name.
    }
    let (bit, default) = MEASURED;
    let measured = (data.get(bit / 8).copied().unwrap_or(0) >> (bit % 8) & 1 != 0) ^ default;
    Some(RawPoint {
        track_id: u64::from_le_bytes(data_bytes(data, TRACK_ID.0 * 8)) ^ TRACK_ID.1,
        d_rel: f32_field(data, D_REL),
        y_rel: f32_field(data, Y_REL),
        v_rel: f32_field(data, V_REL),
        a_rel: f32_field(data, A_REL),
        yv_rel: f32_field(data, YV_REL),
        measured,
        v_lead: f32_field(data, V_LEAD),
        a_lead: f32_field(data, A_LEAD),
        j_lead: f32_field(data, J_LEAD),
        source,
        track_state: data.get(TRACK_STATE.0).copied().unwrap_or(0) ^ TRACK_STATE.1,
        motion_state: data.get(MOTION_STATE.0).copied().unwrap_or(0) ^ MOTION_STATE.1,
    })
}

/// The points of a liveTracks Event message, or None where this does not read it exactly as pycapnp.
pub fn parse(bytes: &[u8]) -> Option<Vec<RawPoint>> {
    let message = Message::new(bytes)?;
    let event = message.read_struct(0, 0)?;
    if u16::from_le_bytes(data_bytes(event.data, EVENT_DISCRIMINANT_OFFSET * 2)) != EVENT_LIVE_TRACKS {
        return None;
    }
    let radar = message.struct_pointer(&event, EVENT_LIVE_TRACKS_POINTER)?;
    message.struct_list(&radar, RADAR_DATA_POINTS_POINTER)?.into_iter().map(point).collect()
}

/// `hyundaicanfd_ccnc_points.snapshot` of the message in `raw`: a `tracks` object whose `points` list holds a `point`
/// object per RadarPoint with the 13 fields as pycapnp gives them. None where `parse` gives None.
#[pyfunction]
pub fn live_tracks_copy<'py>(
    py: Python<'py>,
    raw: &[u8],
    point: &Bound<'py, PyAny>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    guard(|| {
        let Some(parsed) = parse(raw) else { return Ok(None) };
        let points = PyList::empty(py);
        for p in &parsed {
            let out = point.call0()?;
            let float = |v: f32| PyFloat::new(py, v as f64);
            out.setattr(intern!(py, "trackId"), PyInt::new(py, p.track_id))?;
            out.setattr(intern!(py, "dRel"), float(p.d_rel))?;
            out.setattr(intern!(py, "yRel"), float(p.y_rel))?;
            out.setattr(intern!(py, "vRel"), float(p.v_rel))?;
            out.setattr(intern!(py, "aRel"), float(p.a_rel))?;
            out.setattr(intern!(py, "yvRel"), float(p.yv_rel))?;
            out.setattr(intern!(py, "measured"), PyBool::new(py, p.measured))?;
            out.setattr(intern!(py, "vLead"), float(p.v_lead))?;
            out.setattr(intern!(py, "aLead"), float(p.a_lead))?;
            out.setattr(intern!(py, "jLead"), float(p.j_lead))?;
            out.setattr(intern!(py, "trackState"), PyInt::new(py, p.track_state))?;
            out.setattr(intern!(py, "motionState"), PyInt::new(py, p.motion_state))?;
            out.setattr(intern!(py, "radarSource"), PyString::intern(py, RADAR_SOURCES[p.source as usize]))?;
            points.append(out)?;
        }
        let out = tracks.call0()?;
        out.setattr(intern!(py, "points"), points)?;
        Ok(Some(out))
    })
}

/// For the parser test only: each point as a tuple in RadarPoint field order, as `live_tracks_copy` gives its fields,
/// or None where `parse` gives None.
#[pyfunction]
pub fn _parse_for_test<'py>(py: Python<'py>, raw: &[u8]) -> PyResult<Option<Bound<'py, PyList>>> {
    guard(|| {
        let Some(parsed) = parse(raw) else { return Ok(None) };
        let out = PyList::empty(py);
        for p in &parsed {
            let float = |v: f32| PyFloat::new(py, v as f64).into_any();
            let fields = [
                PyInt::new(py, p.track_id).into_any(), float(p.d_rel), float(p.y_rel), float(p.v_rel), float(p.a_rel),
                float(p.yv_rel), PyBool::new(py, p.measured).to_owned().into_any(), float(p.v_lead), float(p.a_lead),
                float(p.j_lead), PyString::intern(py, RADAR_SOURCES[p.source as usize]).into_any(),
                PyInt::new(py, p.track_state).into_any(), PyInt::new(py, p.motion_state).into_any(),
            ];
            out.append(pyo3::types::PyTuple::new(py, fields)?)?;
        }
        Ok(Some(out))
    })
}

/// The constants above, by schema name: compared with the schema pycapnp loads, by the test and when card starts.
#[pyfunction]
pub fn live_tracks_schema(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("event_discriminant_offset", EVENT_DISCRIMINANT_OFFSET)?;
    d.set_item("event_live_tracks", EVENT_LIVE_TRACKS)?;
    d.set_item("event_live_tracks_pointer", EVENT_LIVE_TRACKS_POINTER)?;
    d.set_item("radar_data_points_pointer", RADAR_DATA_POINTS_POINTER)?;
    let fields = pyo3::types::PyDict::new(py);
    fields.set_item("trackId", ("uint64", TRACK_ID.0, TRACK_ID.1))?;
    for (name, (offset, default)) in [("dRel", D_REL), ("yRel", Y_REL), ("vRel", V_REL), ("aRel", A_REL), ("yvRel", YV_REL),
                                      ("vLead", V_LEAD), ("aLead", A_LEAD), ("jLead", J_LEAD)] {
        fields.set_item(name, ("float32", offset, default))?;
    }
    fields.set_item("measured", ("bool", MEASURED.0, MEASURED.1))?;
    fields.set_item("radarSource", ("enum", RADAR_SOURCE.0, RADAR_SOURCE.1))?;
    fields.set_item("trackState", ("uint8", TRACK_STATE.0, TRACK_STATE.1))?;
    fields.set_item("motionState", ("uint8", MOTION_STATE.0, MOTION_STATE.1))?;
    d.set_item("radar_point_fields", fields)?;
    d.set_item("radar_sources", RADAR_SOURCES.to_vec())?;
    Ok(d)
}
