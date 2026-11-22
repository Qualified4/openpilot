//! What the observation and association stages ask of ego's lanes (`_lane`, `_centre`, `_trail`'s outer line,
//! `_body_clear`, `_edge_room`, `_band_line`). The lanes themselves are still kept by the extension's map object; it
//! answers through this trait.

use super::state::Result;

pub trait LaneQuery {
    /// `_lane(x, y, moving=moving, settled=settled)`: (slot, interior, distance past the line).
    fn lane(&mut self, x: f64, y: f64, moving: bool, settled: bool) -> Result<(Option<i64>, bool, f64)>;
    /// `_centre(x)`: ego's lane centre at x.
    fn centre(&mut self, x: f64) -> Result<Option<f64>>;
    /// The outer line of the side y is on, at x (`_settle`).
    fn outer(&mut self, x: f64, y: f64) -> Result<Option<f64>>;
    fn body_clear(&mut self, slot: i64, x: f64, y: f64, clearance: f64, far_side: f64) -> Result<bool>;
    fn edge_room(&mut self, slot: i64, x: f64, room: f64) -> Result<bool>;
    fn band_line(&mut self, x: f64, y: f64, odometer: f64, bin: f64, memory: f64) -> Result<bool>;
    /// Whether the map has any lane geometry (`bool(self.geometry)`).
    fn has_geometry(&mut self) -> Result<bool>;
}
