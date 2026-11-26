//! The lane lines of a `modelV2` `log.Event` message read from its bytes.
//!
//! As for liveTracks (`live_tracks.rs`), the positions read are constants here; `tests/test_ccnc_core_model_v2.py`
//! and card's start (`hyundaicanfd_ccnc_points.model_schema_mismatch`) compare them with the schema pycapnp loads.
//! A message this does not read exactly as pycapnp would gives None, and the map reads the Cap'n Proto objects.

use super::live_tracks::{data_bytes, Message, Struct, EVENT_DISCRIMINANT_OFFSET};

/// `Event`: the modelV2 member of the union.
pub const EVENT_MODEL_V2: u16 = 73;
pub const EVENT_MODEL_V2_POINTER: usize = 0;
/// `ModelDataV2` pointer indices.
pub const POSITION_POINTER: usize = 0; // XYZTData
const META_POINTER: usize = 8; // ModelDataV2.MetaData
const LANE_CHANGE_STATE_OFFSET: usize = 9 * 2; // LaneChangeState enum (UInt16)
const ROAD_EDGE_STDS_POINTER: usize = 10; // List(Float32)
pub const LANE_LINES_POINTER: usize = 4; // List(XYZTData)
pub const LANE_LINE_PROBS_POINTER: usize = 5; // List(Float32)
pub const ROAD_EDGES_POINTER: usize = 6; // List(XYZTData)
/// `XYZTData` pointer indices, all List(Float32).
pub const XYZT_X_POINTER: usize = 0;
pub const XYZT_Y_POINTER: usize = 1;
pub const XYZT_Y_STD_POINTER: usize = 5;

/// Floats and list elements read from one message at most: overlapping pointers could otherwise make a small message
/// read as a huge one (messaging reads without a traversal limit; this gives None instead).
const READ_LIMIT: usize = 1 << 24;

pub struct Line {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

pub struct Model {
    pub lane_lines: Vec<Line>,
    pub lane_line_probs: Vec<f32>,
    pub road_edges: Vec<Line>,
}

fn model_root<'a>(message: &Message<'a>) -> Option<Struct<'a>> {
    let event = message.read_struct(0, 0)?;
    if u16::from_le_bytes(data_bytes(event.data, EVENT_DISCRIMINANT_OFFSET * 2)) != EVENT_MODEL_V2 {
        return None;
    }
    message.struct_pointer(&event, EVENT_MODEL_V2_POINTER)
}

/// laneLines, laneLineProbs and roadEdges of a modelV2 Event message, or None where this does not read them exactly
/// as pycapnp.
pub fn parse(bytes: &[u8]) -> Option<Model> {
    let message = Message::new(bytes)?;
    let model = model_root(&message)?;
    let mut read = 0usize;
    let mut lines = |index: usize| -> Option<Vec<Line>> {
        let elements = message.struct_elements(&model, index)?;
        read += elements.len();
        let mut out = Vec::with_capacity(elements.len());
        for e in &elements {
            let line = Line { x: message.f32_list(e, XYZT_X_POINTER)?, y: message.f32_list(e, XYZT_Y_POINTER)? };
            read += line.x.len() + line.y.len();
            if read > READ_LIMIT {
                return None;
            }
            out.push(line);
        }
        Some(out)
    };
    let lane_lines = lines(LANE_LINES_POINTER)?;
    let road_edges = lines(ROAD_EDGES_POINTER)?;
    let lane_line_probs = message.f32_list(&model, LANE_LINE_PROBS_POINTER)?;
    Some(Model { lane_lines, lane_line_probs, road_edges })
}

/// position x, y and yStd of a modelV2 Event message as f64, or None where this does not read them exactly as pycapnp.
pub fn parse_path(bytes: &[u8]) -> Option<((Vec<f64>, Vec<f64>), Vec<f64>)> {
    let message = Message::new(bytes)?;
    let model = model_root(&message)?;
    let position = message.struct_pointer(&model, POSITION_POINTER)?;
    let read = |index| {
        Some(message.f32_list(&position, index)?.into_iter().map(f64::from).collect())
    };
    Some((
        (read(XYZT_X_POINTER)?, read(XYZT_Y_POINTER)?),
        read(XYZT_Y_STD_POINTER)?,
    ))
}

/// Vehicle path and confidence used by the world map.
pub(crate) fn parse_road_path(bytes: &[u8]) -> Option<(crate::display::ModelPath, bool, [f32; 2])> {
    let message = Message::new(bytes)?;
    let model = model_root(&message)?;
    let changing = message.struct_pointer(&model, META_POINTER)
        .is_none_or(|meta| u16::from_le_bytes(data_bytes(meta.data, LANE_CHANGE_STATE_OFFSET)) != 0);
    let position = message.struct_pointer(&model, POSITION_POINTER)?;
    let xs = message.f32_list(&position, XYZT_X_POINTER)?;
    let mut ys = message.f32_list(&position, XYZT_Y_POINTER)?;
    // modelV2 uses right-positive lateral. The radar pose and display use left-positive lateral.
    for y in &mut ys {
        *y = -*y;
    }
    let path = crate::display::ModelPath::new(xs, ys)?
        .with_deviations(message.f32_list(&position, XYZT_Y_STD_POINTER).unwrap_or_default());
    let edges = message.f32_list(&model, ROAD_EDGE_STDS_POINTER).unwrap_or_default();
    let edge_stds = [0, 1].map(|i| edges.get(i).copied().unwrap_or(f32::INFINITY));
    Some((path, changing, edge_stds))
}
