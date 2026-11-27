//! A wide camera frame through det_rules: boxes to candidates with the frame's calibration, then the side's rules.

use super::{Context, Frame, FrameView, Side};
use det_rules::{Calib, Cand, DetRules, SideOutput, View, WideDet, WideFrame, WideGeometry};

fn frame_detections(frame: &Frame) -> Vec<WideDet> {
    let view = match frame.view {
        FrameView::Raw => View::Raw,
        FrameView::Side(Side::Left) => View::SideL,
        FrameView::Side(Side::Right) => View::SideR,
    };
    frame
        .boxes
        .iter()
        .map(|b| WideDet {
            view,
            bbox: b.xyxy,
            score: b.confidence,
            class: i32::from(b.class_id),
        })
        .collect()
}

/// det_rules' raw and side views, band masks and camera for one set of calibration angles.
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
    /// Tens of milliseconds: `ccnc_mapd` builds it off the map's thread. Keeps both raw and side view band masks
    /// so conditional full-frame raw detections can be combined with side-view detections for LR/RR splitting.
    pub fn build(calib: &Calib) -> Self {
        let wide = WideGeometry::new(calib);
        Self {
            angles: angles(calib),
            wide,
        }
    }

    /// The height takes no part in the views and masks: only the angles must match.
    pub(crate) fn serves(&self, calib: &Calib) -> bool {
        self.angles == angles(calib)
    }

    pub(super) fn raw_candidates(&mut self, frame: &Frame, calib: &Calib) -> Vec<Cand> {
        self.candidates(frame, calib)
    }

    pub(crate) fn candidates(&mut self, frame: &Frame, calib: &Calib) -> Vec<Cand> {
        self.candidates_with_pitch(frame, calib, 0.0)
    }

    pub(crate) fn candidates_with_pitch(&mut self, frame: &Frame, calib: &Calib, delta_pitch: f64) -> Vec<Cand> {
        self.wide.cam.h = calib.height;
        let detections = frame_detections(frame);
        self.wide.candidates_with_pitch(&detections, delta_pitch)
    }

    pub(crate) fn person_points(&mut self, frame: &Frame, calib: &Calib, delta_pitch: f64) -> Vec<(f64, f64, f64)> {
        self.wide.cam.h = calib.height;
        let detections = frame_detections(frame);
        self.wide.person_points(&detections, delta_pitch)
    }

    pub(super) fn step(
        &mut self,
        rules: &mut DetRules,
        frame: &Frame,
        side: Side,
        calib: &Calib,
        context: &Context,
        raw_cands: &[Cand],
    ) -> (Vec<Cand>, SideOutput) {
        let mut cands = self.candidates(frame, calib);
        if !raw_cands.is_empty() {
            cands.extend_from_slice(raw_cands);
        }
        let input = WideFrame {
            t_ns: frame.time_ns,
            v_ego: context.v_ego,
            start_guard_ok: context.guard,
            tracks: context.tracks,
            lprob: context.lprob,
            slots: context.slots,
        };
        let out = rules.wide(&input, &cands).sides[side as usize];
        (cands, out)
    }
}
