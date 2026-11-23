//! Which liveCalibration the side views use, and when they are built again.

use ccnc_vision_msg::{Calibration, LiveCalibration, CAL_STATUS_CALIBRATED};

/// Larger angles are not a calibration of a mounted device (calibrationd itself rejects pitch and yaw beyond
/// about 0.17 and 0.16 rad), so such a message is ignored.
pub const MAX_ANGLE: f32 = 0.3;
/// The views are built again when an angle moved this much from the one they were built with: 2 mrad moves a
/// point 6 m to the side by about 1 cm. Calibration drifts slowly, so smaller changes only cost a rebuild.
pub const REBUILD_RAD: f32 = 0.002;

/// Why a liveCalibration message was not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    Lengths,
    NotFinite,
    AngleRange,
    Height,
}

/// The values of a calibrated message, or why they cannot be used.
fn values(message: &LiveCalibration) -> Result<Calibration, Rejected> {
    let (Ok(rpy_calib), Ok(wide_from_device_euler), &[height]) = (
        <[f32; 3]>::try_from(&message.rpy_calib[..]),
        <[f32; 3]>::try_from(&message.wide_from_device_euler[..]),
        &message.height[..],
    ) else {
        return Err(Rejected::Lengths);
    };
    let angles = rpy_calib.iter().chain(&wide_from_device_euler);
    if !angles.clone().chain([&height]).all(|v| v.is_finite()) {
        return Err(Rejected::NotFinite);
    }
    if angles.clone().any(|a| a.abs() > MAX_ANGLE) {
        return Err(Rejected::AngleRange);
    }
    if height <= 0.0 {
        return Err(Rejected::Height);
    }
    Ok(Calibration { rpy_calib, wide_from_device_euler, height })
}

/// The latest usable calibration. Nothing is processed before the first one, nor while calibrationd says it is
/// not calibrated. A message with unusable values is dropped and the earlier values stay; values that stop
/// coming stay too (calibration changes slowly).
#[derive(Default)]
pub struct Calib {
    latest: Option<Calibration>,
    calibrated: bool,
}

impl Calib {
    pub fn update(&mut self, message: &LiveCalibration) -> Result<(), Rejected> {
        if message.cal_status != CAL_STATUS_CALIBRATED {
            self.calibrated = false;
            return Ok(());
        }
        let calibration = values(message)?;
        self.latest = Some(calibration);
        self.calibrated = true;
        Ok(())
    }

    /// The calibration to build the views with now, if any.
    pub fn usable(&self) -> Option<&Calibration> {
        self.latest.as_ref().filter(|_| self.calibrated)
    }
}

/// Whether views built with `built` are too far from `now`. Height is not part of the views.
pub fn needs_rebuild(built: &Calibration, now: &Calibration) -> bool {
    let a = built.rpy_calib.iter().chain(&built.wide_from_device_euler);
    let b = now.rpy_calib.iter().chain(&now.wide_from_device_euler);
    a.zip(b).any(|(x, y)| (x - y).abs() >= REBUILD_RAD)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(status: u16, rpy: &[f32], wide: &[f32], height: &[f32]) -> LiveCalibration {
        LiveCalibration { cal_status: status, rpy_calib: rpy.to_vec(), wide_from_device_euler: wide.to_vec(), height: height.to_vec() }
    }

    const RPY: [f32; 3] = [0.0, -0.0106, 0.004];
    const WIDE: [f32; 3] = [0.0, -0.0098, -0.018];

    #[test]
    fn nothing_before_a_calibrated_message() {
        let mut calib = Calib::default();
        assert!(calib.usable().is_none());
        calib.update(&message(0, &RPY, &WIDE, &[1.22])).unwrap();
        assert!(calib.usable().is_none(), "uncalibrated");
        calib.update(&message(1, &RPY, &WIDE, &[1.22])).unwrap();
        assert_eq!(calib.usable(), Some(&Calibration { rpy_calib: RPY, wide_from_device_euler: WIDE, height: 1.22 }));
    }

    #[test]
    fn not_calibrated_stops_and_calibrated_resumes() {
        let mut calib = Calib::default();
        calib.update(&message(1, &RPY, &WIDE, &[1.22])).unwrap();
        for status in [0, 2, 3, 7] {
            calib.update(&message(status, &RPY, &WIDE, &[1.22])).unwrap();
            assert!(calib.usable().is_none(), "status {status}");
            calib.update(&message(1, &RPY, &WIDE, &[1.22])).unwrap();
            assert!(calib.usable().is_some());
        }
    }

    #[test]
    fn bad_values_keep_the_earlier_ones() {
        let mut calib = Calib::default();
        let bad = [
            (message(1, &RPY, &WIDE, &[]), Rejected::Lengths),
            (message(1, &RPY[..2], &WIDE, &[1.22]), Rejected::Lengths),
            (message(1, &[f32::NAN, 0.0, 0.0], &WIDE, &[1.22]), Rejected::NotFinite),
            (message(1, &RPY, &WIDE, &[f32::INFINITY]), Rejected::NotFinite),
            (message(1, &[0.0, 0.31, 0.0], &WIDE, &[1.22]), Rejected::AngleRange),
            (message(1, &RPY, &[0.0, 0.0, -0.31], &[1.22]), Rejected::AngleRange),
            (message(1, &RPY, &WIDE, &[0.0]), Rejected::Height),
            (message(1, &RPY, &WIDE, &[-1.0]), Rejected::Height),
        ];
        for (m, why) in &bad {
            assert_eq!(calib.update(m), Err(*why));
            assert!(calib.usable().is_none(), "never a usable one yet");
        }
        calib.update(&message(1, &RPY, &WIDE, &[1.22])).unwrap();
        let good = *calib.usable().unwrap();
        for (m, why) in &bad {
            assert_eq!(calib.update(m), Err(*why));
            assert_eq!(calib.usable(), Some(&good));
        }
        assert!(calib.update(&message(1, &[0.0, 0.3, 0.0], &WIDE, &[0.01])).is_ok(), "0.3 rad is still in range");
    }

    #[test]
    fn rebuilds_from_two_milliradians_on_any_angle() {
        let built = Calibration { rpy_calib: RPY, wide_from_device_euler: WIDE, height: 1.22 };
        assert!(!needs_rebuild(&built, &built));
        assert!(!needs_rebuild(&built, &Calibration { height: 1.5, ..built }));
        for k in 0..6 {
            for (step, rebuild) in [(0.0019f32, false), (0.0021, true), (-0.0021, true), (0.003, true)] {
                let mut now = built;
                let angle = if k < 3 { &mut now.rpy_calib[k] } else { &mut now.wide_from_device_euler[k - 3] };
                *angle += step;
                assert_eq!(needs_rebuild(&built, &now), rebuild, "angle {k} step {step}");
            }
        }
    }
}
