//! A side view frame through det_rules: boxes to candidates with the frame's calibration, then the side's rules.

use super::{Context, Frame, Side};
use det_rules::{Calib, Cand, DetRules, Mask, SideOutput, View, WideDet, WideFrame, WideGeometry};

/// det_rules' side views, band masks and camera for one set of calibration angles.
pub struct Geometry {
    angles: [f64; 6],
    wide: WideGeometry,
}

fn angles(calib: &Calib) -> [f64; 6] {
    let [a, b, c] = calib.rpy;
    let [d, e, f] = calib.wide_from_device;
    [a, b, c, d, e, f]
}

impl Geometry {
    /// Tens of milliseconds: `ccnc_mapd` builds it off the map's thread. Only side view boxes come, so the raw
    /// frame's band mask (4 MB of sums) is dropped once the side view masks are made from it.
    pub fn build(calib: &Calib) -> Self {
        let mut wide = WideGeometry::new(calib);
        wide.mask_raw = Mask::from_fn(0, 0, |_, _| false);
        Self {
            angles: angles(calib),
            wide,
        }
    }

    /// The height takes no part in the views and masks: only the angles must match.
    pub(super) fn serves(&self, calib: &Calib) -> bool {
        self.angles == angles(calib)
    }

    pub(super) fn step(
        &mut self,
        rules: &mut DetRules,
        frame: &Frame,
        calib: &Calib,
        context: &Context,
    ) -> (Vec<Cand>, SideOutput) {
        self.wide.cam.h = calib.height;
        let view = match frame.side {
            Side::Left => View::SideL,
            Side::Right => View::SideR,
        };
        let detections: Vec<WideDet> = frame
            .boxes
            .iter()
            .map(|b| WideDet {
                view,
                bbox: b.xyxy,
                score: b.confidence,
                class: i32::from(b.class_id),
            })
            .collect();
        let cands = self.wide.candidates(&detections);
        let input = WideFrame {
            t_ns: frame.time_ns,
            v_ego: context.v_ego,
            start_guard_ok: context.guard,
            tracks: context.tracks,
            lprob: context.lprob,
            slots: context.slots,
        };
        let out = rules.wide(&input, &cands).sides[frame.side as usize];
        (cands, out)
    }
}
