//! CCNC lane marking recogniser (StripNet) — the processing path of
//! `openpilot/selfdrive/carrot/lane_type/marking_daemon.py` in Rust.
//!
//! Pure Rust: strip extraction (`strip`), tract inference (`model`), decide4, the strip filter, DisplayPolicy, the
//! packet (`daemon`) and a schema-less cereal reader (`capnp`, `messages`). Only the openpilot IPC (msgq, VisionIPC)
//! goes through FFI, behind the `ipc` feature (`ipc`). Inference can also run on onnxruntime (feature `ort`, `ort_engine`). For comparing extraction speed, the remap can also go to
//! KleidiCV's C API (feature `kleidicv`) or OpenCV's cv::remap with the KleidiCV HAL (feature `opencv`) — `native`.

pub mod capnp;
pub mod daemon;
pub mod display;
pub mod filter;
pub mod messages;
pub mod model;
pub mod native;
pub mod recogniser;
pub mod result;
pub mod strip;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "ort")]
pub mod ort_engine;
