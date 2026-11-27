use super::{selector::PASS_BAND, EngineSlot};

const BSD_GRACE: f64 = 2.;
// The cluster renders 0.8 times the signed map range. Start at the 8 m warning.
const BSD_START_X: f64 = -10.;
const TO_ORIGIN_SPEED: f64 = 25.;
const JOIN_DURATION: f64 = 0.20;

#[derive(Default)]
pub(super) struct SideDisplay {
    blind: bool,
    warning_until: Option<f64>,
    recent_front: Option<(u64, f64)>,
    join: Option<FrontJoin>,
}

struct FrontJoin {
    vid: u64,
    start: f64,
    origin_at: f64,
    start_x: f64,
    start_y: f64,
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
        if self.join.is_none() && previous[front].is_none() && (candidate.radar_seen || candidate.camera_seen)
            && candidate.v_s > 0.
        {
            let measured_rear = previous[rear].filter(|old| {
                old.obj_id == candidate.obj_id && old.s < 0. && candidate.s > PASS_BAND
            });
            let from_bsd = previous[rear].is_none()
                && self.warning_until.is_some_and(|until| now <= until)
                && (0. ..=6.).contains(&candidate.s) && candidate.s > 0.
                && self.recent_front.is_none_or(|(vid, t)| vid != candidate.obj_id || now - t > BSD_GRACE);
            if measured_rear.is_some() || from_bsd {
                let start_x = measured_rear.map_or(BSD_START_X, |old| old.x.min(0.));
                let start_y = measured_rear.map_or(if side == 0 { -2.8 } else { 2.8 }, |old| old.y);
                // The previous physical rear was drawn one tick ago. The first front tick can
                // land directly on the origin for a measured crossing close to zero.
                let origin_at = now - start_x / TO_ORIGIN_SPEED;
                self.join = Some(FrontJoin {
                    vid: candidate.obj_id, start: now, origin_at, start_x, start_y, from_bsd,
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
            candidate.y = join.start_y + progress * (candidate.display_d - join.start_y);
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
}
