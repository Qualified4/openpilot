//! The radar map without Python: its state (`MapState`) and the stages that change it. The `ccnc_core` extension
//! keeps a `MapState` for each `CcncRadarMap` whose update runs in Rust; the stages not moved here yet (walls, bands,
//! selection, lanes, rear slots) still read the rest of the map from its Python attributes.

pub mod associate;
pub mod consts;
pub mod lanes;
pub mod object;
pub mod observe;
pub mod state;

pub use self::object::{MapObjectData, Ring};
pub use self::state::{Error, Frame, MapState, Pose, Range, Result};
