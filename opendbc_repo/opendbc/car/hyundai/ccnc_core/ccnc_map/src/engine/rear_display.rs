use super::{
    object::{VehicleKind, REAR_LIMIT},
    selector::{ALONGSIDE_SPEED, PASS_BAND, SIDE_FRONT_LIMIT},
    EngineSlot,
};

const BSD_GRACE: f64 = 3.;
// The cluster renders 0.8 times the signed map range. Start at the 8 m warning.
const DISPLAY_SCALE: f64 = 0.8;
const REAR_DISP_MAX: f64 = 25.5;
const BSD_START_X: f64 = -10.;
// Sixteen metres on the cluster (0.8 * 20) for BSD-only approach/retreat animation.
const BSD_FAR_X: f64 = -20.;
// Ten metres on the cluster after BSD releases, using the same 0.8 conversion.
const BSD_EXIT_X: f64 = -12.5;
const REAR_PASS_HOLD: f64 = 3.;
const BSD_APPROACH_TIME: f64 = 0.6;
const BSD_APPROACH_SPEED: f64 = 6.;
// Match Python SideFlow approach (tau=0.4s, 6.0 m/s display = 7.5 m/s map),
// release hold (3.0s at 8 m), and exit (tau=0.65s, 4.0 m/s display = 5.0 m/s map).
const BSD_ONLY_HOLD: f64 = 3.;
const BSD_ONLY_ENTER_TAU: f64 = 0.4;
const BSD_ONLY_ENTER_SPEED: f64 = 7.5;
const BSD_ONLY_EXIT_TAU: f64 = 0.65;
const BSD_ONLY_EXIT_SPEED: f64 = 5.0;
const TO_ORIGIN_SPEED: f64 = 20.;
const JOIN_DURATION: f64 = 0.20;
// Minimum cluster display separation (d_f + d_r) when both F and R are shown on the same side.
const FR_MIN_GAP_CAR_DISP: f64 = 6.5;
const FR_MIN_GAP_HEAVY_DISP: f64 = 8.5;
// Smoothly ramp gap offsets without 1-tick CAN spikes (front <= 0.8 m/tick disp, rear <= 0.52 m/tick disp).
const FR_GAP_UP_SPEED: f64 = 20.;
const FR_REAR_GAP_UP_SPEED: f64 = 13.;
const FR_GAP_DOWN_SPEED: f64 = 6.;
const FR_GAP_DOWN_TAU: f64 = 0.25;

#[derive(Default)]
pub(super) struct SideDisplay {
    blind: bool,
    warning_until: Option<f64>,
    recent_front: Option<(u64, f64)>,
    join: Option<FrontJoin>,
    pass: Option<RearPass>,
    bsd_only: Option<BsdOnly>,
    front_gap: Option<GapOffset>,
    rear_gap: Option<GapOffset>,
}

#[derive(Clone, Copy)]
struct GapOffset {
    vid: u64,
    offset: f64,
    time: f64,
}

struct BsdOnly {
    x: f64,
    time: f64,
    off_since: Option<f64>,
}

struct RearPass {
    slot: EngineSlot,
    time: f64,
    until: f64,
    exit_x: f64,
}

struct FrontJoin {
    vid: u64,
    start: f64,
    origin_at: f64,
    start_x: f64,
    start_d: f64,
    from_bsd: bool,
    joined_at: Option<f64>,
}

impl SideDisplay {
    pub(super) fn update(
        &mut self,
        now: f64,
        side: usize,
        blind: bool,
        previous: [Option<EngineSlot>; 5],
        slots: &mut [Option<EngineSlot>; 5],
    ) {
        let front = side + 1;
        let rear = side + 3;
        if blind && !self.blind {
            self.warning_until = Some(now + BSD_GRACE);
        } else if blind {
            self.warning_until = self.warning_until.map(|_| now + BSD_GRACE);
        }
        self.blind = blind;
        let reassigned = self.pass.as_ref().is_some_and(|pass| {
            slots.iter().enumerate().any(|(i, slot)| {
                i != rear && slot.is_some_and(|slot| slot.obj_id == pass.slot.obj_id)
            })
        });
        if reassigned {
            self.pass = None;
        }
        self.update_rear_pass(now, blind, previous[front], previous[rear], &mut slots[rear]);
        self.update_front_join(now, side, previous[front], previous[rear], slots);
        self.update_bsd_only(now, side, blind, &mut slots[rear]);
        self.apply_fr_min_gap(now, side, slots);
    }

    fn ramp_gap_offset(
        state: &mut Option<GapOffset>,
        now: f64,
        vid: u64,
        target: f64,
        up_speed: f64,
    ) -> f64 {
        let offset = match *state {
            Some(prev) if prev.vid == vid => {
                let dt = (now - prev.time).clamp(0., 0.25);
                if target >= prev.offset {
                    (prev.offset + up_speed * dt).min(target)
                } else if prev.offset - target <= 0.02 {
                    target
                } else {
                    let step = (prev.offset - target) * (1. - (-dt / FR_GAP_DOWN_TAU).exp());
                    (prev.offset - step.min(FR_GAP_DOWN_SPEED * dt)).max(target)
                }
            }
            _ => target,
        };
        *state = Some(GapOffset { vid, offset, time: now });
        offset
    }

    fn apply_fr_min_gap(&mut self, now: f64, side: usize, slots: &mut [Option<EngineSlot>; 5]) {
        let front_idx = side + 1;
        let rear_idx = side + 3;
        let (target_front, target_rear) = match (slots[front_idx], slots[rear_idx]) {
            (Some(front), Some(rear)) if rear.obj_id != front.obj_id => {
                let min_gap_disp = if front.kind == VehicleKind::Heavy || rear.kind == VehicleKind::Heavy {
                    FR_MIN_GAP_HEAVY_DISP
                } else {
                    FR_MIN_GAP_CAR_DISP
                };
                let d_f = (DISPLAY_SCALE * front.x).max(0.);
                let d_r = (DISPLAY_SCALE * (-rear.x).max(0.)).min(REAR_DISP_MAX);
                if rear.x < 0. && front.v_s < -ALONGSIDE_SPEED {
                    // Front vehicle is actively passing ego toward the 0 m origin while rear is
                    // already receding behind ego: open the gap rearward so front crosses 0 m cleanly.
                    let min_d_r = (min_gap_disp - d_f).clamp(0., REAR_DISP_MAX);
                    let target_r = ((min_d_r - d_r).max(0.)) / DISPLAY_SCALE;
                    (0., target_r)
                } else {
                    let min_x_f = ((min_gap_disp - d_r).max(0.) / DISPLAY_SCALE).min(SIDE_FRONT_LIMIT);
                    ((min_x_f - front.x).max(0.), 0.)
                }
            }
            _ => (0., 0.),
        };
        if let Some(front) = &mut slots[front_idx] {
            let offset = Self::ramp_gap_offset(&mut self.front_gap, now, front.obj_id, target_front, FR_GAP_UP_SPEED);
            if offset > 0. {
                front.x = (front.x + offset).min(SIDE_FRONT_LIMIT);
            }
        } else {
            self.front_gap = None;
        }
        if let Some(rear) = &mut slots[rear_idx] {
            let offset = Self::ramp_gap_offset(&mut self.rear_gap, now, rear.obj_id, target_rear, FR_REAR_GAP_UP_SPEED);
            if offset > 0. {
                rear.x = (rear.x - offset).max(-REAR_DISP_MAX / DISPLAY_SCALE);
            }
        } else {
            self.rear_gap = None;
        }
    }

    fn update_front_join(
        &mut self,
        now: f64,
        side: usize,
        old_front: Option<EngineSlot>,
        old_rear: Option<EngineSlot>,
        slots: &mut [Option<EngineSlot>; 5],
    ) {
        let front = side + 1;
        let rear = side + 3;
        let Some(mut candidate) = slots[front].filter(|_| slots[rear].is_none()) else {
            self.join = None;
            if let Some(slot) = slots[front] {
                self.recent_front = Some((slot.obj_id, now));
            }
            return;
        };
        if self.join.as_ref().is_some_and(|join| join.vid != candidate.obj_id || candidate.s <= 0.) {
            self.join = None;
        }
        if self.join.is_none()
            && old_front.is_none_or(|old| old.obj_id != candidate.obj_id && old.s > candidate.s + 10.0)
            && (candidate.radar_seen || candidate.camera_seen)
            && candidate.v_s > 0.
        {
            let measured_rear = old_rear.filter(|old| {
                old.obj_id == candidate.obj_id && old.s < 0. && candidate.s > PASS_BAND
            });
            let bsd_active_or_hold = self.blind
                || self.bsd_only.as_ref().is_some_and(|b| {
                    b.off_since.is_none_or(|t| now - t < BSD_ONLY_HOLD)
                        && b.x >= BSD_START_X - 0.05
                });
            let from_bsd = old_rear.is_none_or(|old| old.obj_id == 0)
                && bsd_active_or_hold
                && self.warning_until.is_some_and(|until| now <= until)
                && (0. ..=10.).contains(&candidate.s) && candidate.s > 0.
                && self.recent_front.is_none_or(|(vid, t)| vid != candidate.obj_id || now - t > BSD_GRACE);
            if measured_rear.is_some() || from_bsd {
                let start_x = measured_rear.map_or_else(
                    || self.bsd_only.as_ref().map_or(BSD_START_X, |b| b.x.max(BSD_START_X)),
                    |old| old.x.min(0.),
                );
                let start_d = measured_rear.map_or(if side == 0 { -2.8 } else { 2.8 }, |old| old.display_d);
                // The previous physical rear was drawn one tick ago. The first front tick can
                // land directly on the origin for a measured crossing close to zero.
                let origin_at = now - start_x / TO_ORIGIN_SPEED;
                self.pass = None;
                self.bsd_only = None;
                self.join = Some(FrontJoin {
                    vid: candidate.obj_id, start: now, origin_at, start_x, start_d, from_bsd,
                    joined_at: None,
                });
                self.warning_until = None;
            }
        }
        self.recent_front = Some((candidate.obj_id, now));
        let Some(join) = &mut self.join else { return };
        if now < join.origin_at {
            let progress = (now - join.start) / (join.origin_at - join.start);
            candidate.x = join.start_x * (1. - progress);
            candidate.display_d = join.start_d + progress * (candidate.display_d - join.start_d);
            candidate.rear_animated = true;
            if join.from_bsd {
                candidate.v_s = 0.; // Keep the BSD car white on its way to the origin.
                candidate.kind = super::VehicleKind::Car;
            }
            slots[rear] = Some(candidate);
            slots[front] = None;
        } else {
            // Publish an exact zero on the first front tick, even if the radar tick overshot.
            let joined_at = *join.joined_at.get_or_insert(now);
            let progress = ((now - joined_at) / JOIN_DURATION).min(1.);
            candidate.x *= progress;
            slots[front] = Some(candidate);
            if progress == 1. {
                self.join = None;
            }
        }
    }

    fn update_bsd_only(
        &mut self,
        now: f64,
        side: usize,
        blind: bool,
        rear: &mut Option<EngineSlot>,
    ) {
        if rear.is_some() {
            self.bsd_only = None;
            return;
        }
        if blind {
            self.warning_until = Some(now + BSD_GRACE);
            match &mut self.bsd_only {
                None => {
                    self.bsd_only = Some(BsdOnly { x: BSD_FAR_X, time: now, off_since: None });
                }
                Some(bsd) => {
                    bsd.off_since = None;
                    let dt = (now - bsd.time).clamp(0., 0.25);
                    bsd.time = now;
                    let error = BSD_START_X - bsd.x;
                    let step = error * (1. - (-dt / BSD_ONLY_ENTER_TAU).exp());
                    bsd.x = if error.abs() <= 0.05 {
                        BSD_START_X
                    } else {
                        bsd.x + step.clamp(-BSD_ONLY_ENTER_SPEED * dt, BSD_ONLY_ENTER_SPEED * dt)
                    };
                }
            }
        } else if let Some(bsd) = &mut self.bsd_only {
            let off_since = *bsd.off_since.get_or_insert(now);
            let dt = (now - bsd.time).clamp(0., 0.25);
            bsd.time = now;
            if now - off_since >= BSD_ONLY_HOLD {
                let error = BSD_FAR_X - bsd.x;
                let step = error * (1. - (-dt / BSD_ONLY_EXIT_TAU).exp());
                bsd.x = (bsd.x + step.clamp(-BSD_ONLY_EXIT_SPEED * dt, BSD_ONLY_EXIT_SPEED * dt))
                    .max(BSD_FAR_X);
                if bsd.x <= BSD_FAR_X + 0.05 {
                    self.bsd_only = None;
                }
            }
        }
        if let Some(bsd) = &self.bsd_only {
            let lat = if side == 0 { -2.8 } else { 2.8 };
            *rear = Some(EngineSlot {
                obj_id: 0,
                s: bsd.x,
                d: lat,
                x: bsd.x,
                y: lat,
                display_d: lat,
                rear_animated: true,
                v_s: 0.,
                kind: super::VehicleKind::Car,
                existence_p: 1.,
                last_track_id: None,
                lane_u: if side == 0 { -1. } else { 1. },
                camera_seen: false,
                radar_seen: false,
                side_range: None,
                ff_handoff_until: None,
            });
        }
    }

    fn update_rear_pass(
        &mut self,
        now: f64,
        blind: bool,
        old_front: Option<EngineSlot>,
        old_rear: Option<EngineSlot>,
        rear: &mut Option<EngineSlot>,
    ) {
        // Unified origin crossing (+d -> 0 -> -d): accept both LF/RF -> LR/RR crossings
        // and split-rear vehicles receding past the origin.
        if let Some(mut slot) = rear.filter(|slot| {
            let crossed_from_front = old_front.is_some_and(|old| old.obj_id == slot.obj_id && old.s >= 0.);
            let crossed_from_split = self.pass.is_none()
                && old_rear.is_some_and(|old| old.obj_id == slot.obj_id && old.s >= 0.);
            slot.s <= 0. && slot.v_s < -ALONGSIDE_SPEED && (crossed_from_front || crossed_from_split)
        }) {
            slot.x = 0.;
            slot.rear_animated = true;
            self.pass = Some(RearPass {
                slot, time: now, until: now + REAR_PASS_HOLD, exit_x: REAR_LIMIT,
            });
            *rear = Some(slot);
            return;
        }
        let Some(pass) = &mut self.pass else { return };
        // Never carry a previous vehicle's rear trajectory onto a different VID.
        if rear.is_some_and(|slot| {
            slot.obj_id != pass.slot.obj_id
                || (slot.radar_seen || slot.camera_seen) && slot.v_s >= 0.
        }) {
            self.pass = None;
            return;
        }
        let dt = now - pass.time;
        pass.time = now;
        if let Some(slot) = rear.filter(|slot| !blind && (slot.radar_seen || slot.camera_seen)) {
            pass.slot = slot;
            pass.until = now + REAR_PASS_HOLD;
            return;
        }
        if let Some(slot) = rear.filter(|slot| slot.obj_id == pass.slot.obj_id) {
            pass.slot = EngineSlot { x: pass.slot.x, ..slot };
            *rear = None;
        }
        if blind {
            pass.until = now + REAR_PASS_HOLD;
            pass.exit_x = BSD_EXIT_X;
            let error = BSD_START_X - pass.slot.x;
            let step = error * (1. - (-dt / BSD_APPROACH_TIME).exp());
            pass.slot.x = if error.abs() <= 0.05 {
                BSD_START_X
            } else {
                pass.slot.x + step.clamp(-BSD_APPROACH_SPEED * dt, BSD_APPROACH_SPEED * dt)
            };
        } else if now > pass.until || pass.slot.x <= pass.exit_x {
            self.pass = None;
            return;
        } else {
            pass.slot.x = (pass.slot.x + pass.slot.v_s * dt).max(pass.exit_x);
        }
        pass.slot.radar_seen = false;
        pass.slot.camera_seen = false;
        pass.slot.rear_animated = true;
        *rear = Some(pass.slot);
    }
}
