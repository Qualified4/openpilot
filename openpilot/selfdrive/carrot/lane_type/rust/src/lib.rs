//! CCNC lane marking recogniser (StripNet) — the processing path of
//! `openpilot/selfdrive/carrot/lane_type/marking_daemon.py` in Rust.
//!
//! Pure Rust: strip extraction (`strip`), tract inference (`model`), decide4, the strip filter, DisplayPolicy, the
//! packet (`daemon`) and a schema-less cereal reader (`capnp`, `messages`). Only the openpilot IPC (msgq, VisionIPC)
//! goes through FFI, behind the `ipc` feature (`ipc`).

pub mod capnp;
pub mod daemon;
pub mod display;
pub mod filter;
pub mod messages;
pub mod model;
pub mod recogniser;
pub mod result;
pub mod strip;
#[cfg(feature = "ipc")]
pub mod ipc;
