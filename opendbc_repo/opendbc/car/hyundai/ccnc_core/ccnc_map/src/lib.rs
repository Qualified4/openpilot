//! The CCNC map's computation without Python: the position display and the message byte readers. The `ccnc_core`
//! extension and the `ccnc_mapd` process both call it.

pub mod pos {
    pub mod common;
    pub mod display_a;
    pub mod display_b;

    pub use self::common::{Common, FrameIn, FrameOut, LaneView, Obj, Track, Vid};
}

pub mod input {
    pub mod live_tracks;
    pub mod model_v2;
}
