//! The `CcncRadarMap` constants the Rust loops use, read from the map object once per call (tests change instance and
//! class constants, and trial values can be tuned without a rebuild).

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

fn attr_f64(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

fn attr_bool(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    obj.getattr(name)?.is_truthy()
}

pub use ccnc_map::radar::consts::Consts;

/// The constants of a map object (or of the `CcncRadarMap` class).
pub fn read(m: &Bound<'_, PyAny>) -> PyResult<Consts> {
    let py = m.py();
    let k = |name: &Bound<'_, PyString>| attr_f64(m, name);
    Ok(Consts {
        min_x: k(intern!(py, "MIN_X"))?,
        max_x: k(intern!(py, "MAX_X"))?,
        max_y: k(intern!(py, "MAX_Y"))?,
        keep_x: k(intern!(py, "KEEP_X"))?,
        max_side_x: k(intern!(py, "MAX_SIDE_X"))?,
        far_y: k(intern!(py, "FAR_Y"))?,
        far_x: k(intern!(py, "FAR_X"))?,
        max_objects: k(intern!(py, "MAX_OBJECTS"))?,
        max_walls: k(intern!(py, "MAX_WALLS"))?,
        match_distance: k(intern!(py, "MATCH_DISTANCE"))?,
        max_display_uncertainty: k(intern!(py, "MAX_DISPLAY_UNCERTAINTY"))?,
        memory_min_x: k(intern!(py, "MEMORY_MIN_X"))?,
        memory_max_x: k(intern!(py, "MEMORY_MAX_X"))?,
        waiting_admission: attr_bool(m, intern!(py, "WAITING_ADMISSION"))?,
        waiting_tracked: k(intern!(py, "WAITING_TRACKED"))?,
        memory_tracked: k(intern!(py, "MEMORY_TRACKED"))?,
        crossing_travel: k(intern!(py, "CROSSING_TRAVEL"))?,
        body_clearance: k(intern!(py, "BODY_CLEARANCE"))?,
        body_far_side: k(intern!(py, "BODY_FAR_SIDE"))?,
        waiting_edge_room: k(intern!(py, "WAITING_EDGE_ROOM"))?,
        prediction_hold: k(intern!(py, "PREDICTION_HOLD"))?,
        memory_moving_hold: k(intern!(py, "MEMORY_MOVING_HOLD"))?,
        report_delay_bound: k(intern!(py, "REPORT_DELAY_BOUND"))?,
        doppler_speed: k(intern!(py, "DOPPLER_SPEED"))?,
        doppler_samples: k(intern!(py, "DOPPLER_SAMPLES"))?,
        doppler_min_x: k(intern!(py, "DOPPLER_MIN_X"))?,
        doppler_near_y: k(intern!(py, "DOPPLER_NEAR_Y"))?,
        release_gap: m.getattr(intern!(py, "RELEASE_GAP"))?.extract()?,
        release_lateral: k(intern!(py, "RELEASE_LATERAL"))?,
        release_speed: k(intern!(py, "RELEASE_SPEED"))?,
        release_advance: k(intern!(py, "RELEASE_ADVANCE"))?,
        id_cycle: k(intern!(py, "ID_CYCLE"))?,
        doppler_window: k(intern!(py, "DOPPLER_WINDOW"))?,
        displacement_evidence: attr_bool(m, intern!(py, "DISPLACEMENT_EVIDENCE"))?,
        resting_speed: k(intern!(py, "RESTING_SPEED"))?,
        outer_narrow: k(intern!(py, "OUTER_NARROW"))?,
        outer_lateral: k(intern!(py, "OUTER_LATERAL"))?,
        outer_time: k(intern!(py, "OUTER_TIME"))?,
        outer_far: k(intern!(py, "OUTER_FAR"))?,
        doppler_tracked: k(intern!(py, "DOPPLER_TRACKED"))?,
        doppler_yaw: k(intern!(py, "DOPPLER_YAW"))?,
        resting_travel: k(intern!(py, "RESTING_TRAVEL"))?,
        resting_net: k(intern!(py, "RESTING_NET"))?,
        dense_row_points: k(intern!(py, "DENSE_ROW_POINTS"))?,
        dense_row_span: k(intern!(py, "DENSE_ROW_SPAN"))?,
        wall_max_gap: k(intern!(py, "WALL_MAX_GAP"))?,
        band_memory: k(intern!(py, "BAND_MEMORY"))?,
        band_bin: k(intern!(py, "BAND_BIN"))?,
        band_fold: k(intern!(py, "BAND_FOLD"))?,
        band_prediction: k(intern!(py, "BAND_PREDICTION"))?,
        band_gap: k(intern!(py, "BAND_GAP"))?,
        wall_bend: k(intern!(py, "WALL_BEND"))?,
        wall_continue: k(intern!(py, "WALL_CONTINUE"))?,
        lead_beyond: k(intern!(py, "LEAD_BEYOND"))?,
        lead_margin: k(intern!(py, "LEAD_MARGIN"))?,
        lead_curve: k(intern!(py, "LEAD_CURVE"))?,
        lane_return_time: k(intern!(py, "LANE_RETURN_TIME"))?,
        trail_step: k(intern!(py, "TRAIL_STEP"))?,
        trail_near: k(intern!(py, "TRAIL_NEAR"))?,
        trail_far: k(intern!(py, "TRAIL_FAR"))?,
        trail_room: k(intern!(py, "TRAIL_ROOM"))?,
        lead_side_x: k(intern!(py, "LEAD_SIDE_X"))?,
        lead_side_hold_x: k(intern!(py, "LEAD_SIDE_HOLD_X"))?,
        lead_side_past: k(intern!(py, "LEAD_SIDE_PAST"))?,
        bridge_hold: k(intern!(py, "BRIDGE_HOLD"))?,
        bridge_lane: k(intern!(py, "BRIDGE_LANE"))?,
        bridge_jump: k(intern!(py, "BRIDGE_JUMP"))?,
        bridge_jump_time: k(intern!(py, "BRIDGE_JUMP_TIME"))?,
        bridge_reconnect: k(intern!(py, "BRIDGE_RECONNECT"))?,
        bridge_drift_x: k(intern!(py, "BRIDGE_DRIFT_X"))?,
        stable_speed: k(intern!(py, "STABLE_SPEED"))?,
        stable_window: k(intern!(py, "STABLE_WINDOW"))?,
        stable_time: k(intern!(py, "STABLE_TIME"))?,
        report_step: k(intern!(py, "REPORT_STEP"))?,
        ff_far_x: k(intern!(py, "FF_FAR_X"))?,
        ff_far_hold: k(intern!(py, "FF_FAR_HOLD"))?,
        switch_settle: k(intern!(py, "SWITCH_SETTLE"))?,
        switch_hold: k(intern!(py, "SWITCH_HOLD"))?,
        switch_margin_x: k(intern!(py, "SWITCH_MARGIN_X"))?,
        switch_margin_c: k(intern!(py, "SWITCH_MARGIN_C"))?,
        switch_grace: k(intern!(py, "SWITCH_GRACE"))?,
        switch_depart: k(intern!(py, "SWITCH_DEPART"))?,
    })
}
