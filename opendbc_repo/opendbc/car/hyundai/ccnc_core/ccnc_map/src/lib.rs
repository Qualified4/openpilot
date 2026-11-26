//! The CCNC map's computation without Python: the radar map state and its observation stages, the position display
//! and the message byte readers. The `ccnc_core`
//! extension and the `ccnc_mapd` process both call it.

pub mod cmap;
pub mod display;
pub mod engine;
pub mod pipeline;

pub mod pos {
    pub mod common;
    pub mod display_a;
    pub mod display_b;

    pub use self::common::{Common, FrameIn, FrameOut, LaneView, Obj, Track, Vid};
}

pub mod input {
    pub mod framing;
    pub mod live_tracks;
    pub mod map_events;
    pub mod model_v2;
}

pub mod pymath;
pub mod radar;

pub mod rear;
pub mod slots;
pub mod vehicles;
pub mod vision;
pub mod world;
