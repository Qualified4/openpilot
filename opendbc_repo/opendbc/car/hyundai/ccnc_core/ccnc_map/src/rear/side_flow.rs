//! Continuous front/rear drawing. A drawing retains its physical motion's
//! serial when it crosses the origin, even after that motion is retired.

use crate::pymath::{py_max, py_min};
use crate::radar::{Error, Result};

use super::{divide, finite_integer, Identity, Number, Phase as MotionPhase, SideSlots};

#[derive(Clone, Debug)]
pub struct Consts {
    pub hold: f64,
    pub end: f64,
    pub gap: f64,
    pub space_distance: f64,
    pub front_limit: f64,
    pub fast_speed: f64,
    pub fast_tau: f64,
    pub pass_speed: f64,
    pub exit_speed: f64,
    pub exit_tau: f64,
    pub lateral_speed: f64,
    pub lateral_tau: f64,
}

impl Default for Consts {
    fn default() -> Self {
        Self {
            hold: 3.,
            end: 16.,
            gap: 6.,
            space_distance: 8.,
            front_limit: 204.7,
            fast_speed: 12.,
            fast_tau: 0.2,
            pass_speed: 3.,
            exit_speed: 4.,
            exit_tau: 0.65,
            lateral_speed: 2.,
            lateral_tau: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Motion {
    pub serial: u64,
    pub key: Identity,
    pub aliases: Vec<Identity>,
    pub phase: MotionPhase,
    pub signed: f64,
    pub speed: f64,
    pub distance: f64,
    pub force_at: f64,
    pub deadline: f64,
    pub max_speed: f64,
    pub end_reason: Option<&'static str>,
}

impl Motion {
    fn owns(&self, key: Option<Key>) -> bool {
        match key {
            Some(Key::Radar(key)) => self.key == key || self.aliases.contains(&key),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Key {
    Radar(Identity),
    Display(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Front,
    Bsd,
    Exit,
    JoinFront,
    ToFront,
    ToRear,
    Virtual,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::Bsd => "bsd",
            Self::Exit => "exit",
            Self::JoinFront => "join_front",
            Self::ToFront => "to_front",
            Self::ToRear => "to_rear",
            Self::Virtual => "virtual",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub key: Key,
    pub x: Number,
    pub y: Number,
    pub detect: Number,
}

#[derive(Clone, Debug)]
pub struct Drawing {
    pub key: Option<Key>,
    pub x: Number,
    pub y: Number,
    pub detect: Number,
    pub phase: Phase,
    pub display_pass: Option<bool>,
    pub target: Option<Candidate>,
    pub motion: Option<Motion>,
    pub model_distance: Option<f64>,
    pub spacing: Option<bool>,
}

impl Drawing {
    fn new(key: Option<Key>, x: Number, y: Number, detect: Number, phase: Phase) -> Self {
        Self {
            key,
            x,
            y,
            detect,
            phase,
            display_pass: None,
            target: None,
            motion: None,
            model_distance: None,
            spacing: None,
        }
    }

    fn front(candidate: Candidate) -> Self {
        Self::new(
            Some(candidate.key),
            candidate.x,
            candidate.y,
            candidate.detect,
            Phase::Front,
        )
    }

    fn lateral(&mut self, k: &Consts, y: Number, dt: f64) -> Result<()> {
        self.y = moved(self.y, y, dt, k.lateral_speed, k.lateral_tau)?;
        Ok(())
    }

    fn advance(&mut self, k: &Consts, x: Number, y: Number, dt: f64, speed: f64, tau: f64) -> Result<()> {
        self.x = moved(self.x, x, dt, speed, tau)?;
        self.lateral(k, y, dt)
    }
}

fn moved(value: Number, target: Number, dt: f64, speed: f64, tau: f64) -> Result<Number> {
    let error = target.value() - value.value();
    let limit = speed * dt;
    if error.abs() <= py_min(0.04, limit) {
        return Ok(target);
    }
    let exponent = divide(-dt, tau)?;
    let change = exponent.exp_m1();
    if change.is_infinite() && exponent.is_finite() {
        return Err(Error::Overflow("math range error"));
    }
    let step = py_min(error.abs() * (-change), limit);
    Ok(Number::Float(value.value() + step.copysign(error)))
}

fn gap(front: Number, rear: Number) -> Result<f64> {
    let (front, rear) = (front.value() * 10., -rear.value() * 10.);
    finite_integer(front)?;
    finite_integer(rear)?;
    Ok(front.floor() + rear.floor())
}

#[derive(Clone, Debug, Default)]
pub struct Flow {
    pub front: Option<Drawing>,
    pub rear: Option<Drawing>,
    pub bsd: bool,
    pub bsd_off: Option<f64>,
    pub after_pass: bool,
    pub events: Vec<&'static str>,
}

impl Flow {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render(
        &mut self,
        k: &Consts,
        bsd_join_limit: f64,
        slots: &mut SideSlots,
        side: usize,
        now: f64,
        dt: f64,
        key: Option<Identity>,
        elsewhere: [Option<Identity>; 2],
        motion: Option<&Motion>,
        retired: Option<&Motion>,
        bsd: bool,
        changing: bool,
        lateral: f64,
    ) -> Result<()> {
        let (x, y, detect) = (slots.distance(side)?, slots.lateral(side)?, slots.detect(side)?);
        let candidate = (detect.value() != 0.
            && x.value().is_finite()
            && y.value().is_finite()
            && (0. ..=k.front_limit).contains(&x.value()))
        .then_some(Candidate {
            key: key.map_or(Key::Display(side), Key::Radar),
            x,
            y,
            detect,
        });
        let (front, rear) = self.update(
            k,
            bsd_join_limit,
            now,
            dt,
            candidate,
            bsd,
            Number::Float(lateral),
            elsewhere,
            motion,
            retired,
            changing,
        )?;
        slots.front.detect = Some(front.map_or(Number::Integer(0), |f| f.detect));
        slots.rear.detect = Some(rear.map_or(Number::Integer(0), |r| r.detect));
        if let Some(front) = front {
            slots.front.distance = Some(front.x);
            slots.front.lateral = Some(front.y);
        }
        if let Some(rear) = rear {
            slots.rear.distance = Some(Number::Float(-rear.x.value()));
            slots.rear.lateral = Some(rear.y);
        }
        slots.reset_distance = Some(rear.map_or(16., |rear| -rear.x.value()));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        k: &Consts,
        bsd_join_limit: f64,
        now: f64,
        dt: f64,
        candidate: Option<Candidate>,
        bsd: bool,
        lateral: Number,
        elsewhere: [Option<Identity>; 2],
        motion: Option<&Motion>,
        retired: Option<&Motion>,
        changing: bool,
    ) -> Result<(Option<&Drawing>, Option<&Drawing>)> {
        self.events.clear();
        if self.bsd && !bsd {
            self.bsd_off = Some(now);
        } else if bsd {
            self.bsd_off = None;
        }
        self.bsd = bsd;
        if changing && !bsd {
            self.bsd_off = None;
        }
        if self.front.is_none() && self.rear.is_none() && candidate.is_none() && !bsd {
            return Ok((None, None));
        }
        let eligible_bsd = bsd || self.bsd_off.is_some_and(|off| now - off < k.hold);

        for drawing in [&mut self.front, &mut self.rear] {
            let Some(obj) = drawing else { continue };
            if let Some(owner) = &mut obj.motion {
                if let Some(latest) = [motion, retired]
                    .into_iter()
                    .flatten()
                    .find(|m| m.serial == owner.serial)
                {
                    *owner = latest.clone();
                }
            }
            let owner = obj.motion.as_ref().or_else(|| motion.filter(|m| m.owns(obj.key)));
            if elsewhere
                .into_iter()
                .flatten()
                .any(|key| Some(Key::Radar(key)) == obj.key || owner.is_some_and(|m| m.owns(Some(Key::Radar(key)))))
            {
                *drawing = None;
            }
        }
        let rear_start = self.rear.as_ref().map(|rear| rear.x.value());
        let f_owned = self
            .front
            .as_ref()
            .is_some_and(|f| motion.is_some_and(|m| m.owns(f.key)));

        if let Some(front) = &mut self.front {
            let same = candidate.is_some_and(|candidate| Some(candidate.key) == front.key);
            if front.phase == Phase::ToRear && same {
                front.phase = Phase::JoinFront;
            } else if front.phase == Phase::ToRear && !f_owned && !eligible_bsd && !front.display_pass.unwrap_or(false)
            {
                self.front = None;
            } else if front.phase != Phase::ToRear && !same {
                if front.x.value() <= 5. {
                    front.phase = Phase::ToRear;
                    front.display_pass = Some(!f_owned);
                } else {
                    self.front = candidate
                        .filter(|c| c.x.value() >= k.gap && !self.rear.as_ref().is_some_and(|r| Some(c.key) == r.key))
                        .map(Drawing::front);
                }
            }
        }

        let spawned = self.rear.is_none() && bsd;
        if spawned {
            self.rear = Some(Drawing::new(
                None,
                Number::Float(-k.end),
                lateral,
                Number::Integer(2),
                Phase::Bsd,
            ));
        }
        if let Some(rear) = &mut self.rear {
            if let Some(candidate) = candidate.filter(|_| self.front.is_none()) {
                if rear.phase != Phase::ToFront
                    && (Some(candidate.key) == rear.key
                        || candidate.x.value() <= bsd_join_limit && eligible_bsd && !self.after_pass)
                {
                    rear.phase = Phase::ToFront;
                    rear.key = Some(candidate.key);
                    rear.target = Some(candidate);
                    self.events.push("rear_to_front_start");
                }
            }
            if rear.phase == Phase::ToFront {
                if let Some(candidate) = candidate.filter(|candidate| Some(candidate.key) == rear.key) {
                    rear.target = Some(candidate);
                }
                let target = rear.target.expect("a returning drawing has a target").x;
                let min_x = k.fast_speed * k.fast_tau;
                let target = if min_x > target.value() {
                    Number::Float(min_x)
                } else {
                    target
                };
                rear.advance(k, target, rear.y, dt, k.fast_speed, k.fast_tau)?;
                let x = py_min(0., rear.x.value());
                if x == 0. {
                    rear.x = Number::Float(x);
                    rear.phase = Phase::JoinFront;
                    self.front = self.rear.take();
                    self.after_pass = false;
                    self.events.push("rear_to_front");
                    return Ok((self.front.as_ref(), None));
                }
            } else if let Some(event) =
                update_rear(k, rear, motion, lateral, now, dt, eligible_bsd, bsd, changing, spawned)?
            {
                self.rear = None;
                self.events.push(event);
            }
        }

        let mut crossed = false;
        if let Some(front) = self.front.as_mut().filter(|f| f.phase == Phase::ToRear) {
            let physical = motion.filter(|_| f_owned);
            let (target, speed) = match physical {
                Some(motion) if motion.phase == MotionPhase::Coast => (py_max(0., motion.signed), 8.),
                Some(motion) => (0., motion.speed),
                None => (0., k.pass_speed),
            };
            let position = front.x.value();
            let limit = position - py_min(8., speed) * dt;
            let target_or_position = py_min(position, target);
            let next = py_max(target_or_position, limit);
            if target < position || limit > target_or_position {
                front.x = Number::Float(next);
            }
            if target == 0. && next < 1e-9 {
                front.x = Number::Float(0.);
            }
            if front.x.value() == 0. {
                if self.rear.is_some() {
                    self.events.push("rear_display_replaced");
                }
                front.phase = Phase::Virtual;
                front.model_distance = Some(physical.map_or(0., |m| m.distance));
                front.motion = physical.cloned();
                self.rear = self.front.take();
                self.after_pass = true;
                self.events.push("front_to_rear");
                crossed = true;
            }
        }

        if let Some(rear) = self.rear.as_mut().filter(|r| r.phase != Phase::ToFront) {
            if self.front.as_ref().is_none_or(|f| f.phase != Phase::ToRear)
                && candidate.is_some_and(|c| c.x.value() < k.gap)
            {
                rear.spacing = Some(true);
            }
            let x = rear.x.value();
            if rear.spacing.unwrap_or(false) && x > -k.space_distance && !crossed {
                let limit = rear_start.map_or(x, |start| py_max(-k.end, start - 8. * dt));
                let adjusted = py_max(
                    limit,
                    moved(rear.x, Number::Float(-k.space_distance), dt, 8., 0.25)?.value(),
                );
                if adjusted < x {
                    rear.x = Number::Float(adjusted);
                    self.events.push("rear_spacing");
                }
            }
            if rear.x.value() <= -k.space_distance {
                rear.spacing = Some(false);
            }
        }

        if let Some(candidate) = candidate.filter(|_| self.front.is_none()) {
            let clear = match &self.rear {
                Some(rear) => gap(candidate.x, rear.x)? >= k.gap * 10. && rear.phase != Phase::ToFront,
                None => true,
            };
            if clear {
                self.front = Some(Drawing::front(candidate));
                self.after_pass = false;
                self.events.push("front_admitted");
            }
        }

        if let Some(front) = self.front.as_mut().filter(|f| f.phase != Phase::ToRear) {
            if let Some(target) = candidate.filter(|c| Some(c.key) == front.key).or(front.target) {
                if front.phase == Phase::JoinFront {
                    front.advance(k, target.x, target.y, dt, k.fast_speed, k.fast_tau)?;
                    if front.x.value() == target.x.value() && front.y.value() == target.y.value() {
                        front.phase = Phase::Front;
                    }
                } else {
                    let (old, x) = (front.x.value(), target.x.value());
                    let (low, high) = (old - 20. * dt, old + 20. * dt);
                    front.x = if low > x || high < x {
                        Number::Float(py_min(py_max(x, low), high))
                    } else {
                        target.x
                    };
                    front.y = target.y;
                    front.detect = target.detect;
                }
            }
            if let Some(rear) = self.rear.as_ref().filter(|r| r.phase != Phase::ToFront) {
                if gap(front.x, rear.x)? < k.gap * 10. {
                    return Ok((None, Some(rear)));
                }
            }
        }
        Ok((self.front.as_ref(), self.rear.as_ref()))
    }
}

#[allow(clippy::too_many_arguments)]
fn update_rear(
    k: &Consts,
    rear: &mut Drawing,
    motion: Option<&Motion>,
    lateral: Number,
    now: f64,
    dt: f64,
    eligible_bsd: bool,
    bsd: bool,
    changing: bool,
    spawned: bool,
) -> Result<Option<&'static str>> {
    let mut delta = 0.;
    if let Some(tracked) = &rear.motion {
        if motion.is_some_and(|m| tracked.serial == m.serial) {
            delta = py_max(0., tracked.distance - rear.model_distance.unwrap_or(0.));
            rear.model_distance = Some(tracked.distance);
        } else if eligible_bsd {
            rear.motion = None;
        } else if matches!(
            tracked.end_reason,
            Some("distance_end" | "timeout" | "rear_replaced" | "lane_change")
        ) {
            rear.phase = Phase::Exit;
        } else {
            return Ok(Some("rear_cancelled"));
        }
    }
    if changing && !eligible_bsd && rear.motion.is_none() && rear.phase == Phase::Virtual {
        rear.phase = Phase::Exit;
    }
    let passing = rear.motion.is_none() && rear.phase == Phase::Virtual;
    if let Some(physical) = rear.motion.as_ref().filter(|_| !eligible_bsd) {
        if rear.phase == Phase::Exit {
            delta = physical.max_speed * dt;
        }
        let x = rear.x.value();
        if now >= physical.force_at {
            let required = divide((k.end + x) * dt, py_max(dt, physical.deadline - now + dt))?;
            delta = py_max(delta, py_min(required, physical.max_speed * dt));
        }
        rear.x = Number::Float(py_max(-k.end, x - delta));
        rear.lateral(k, lateral, dt)?;
    } else {
        let target = if bsd || (passing && eligible_bsd) {
            Number::Float(-8.)
        } else if eligible_bsd {
            rear.x
        } else {
            Number::Float(-k.end)
        };
        let (mut speed, tau) = if bsd { (6., 0.4) } else { (k.exit_speed, k.exit_tau) };
        if passing {
            speed = k.pass_speed;
        }
        rear.advance(k, target, lateral, if spawned { 0. } else { dt }, speed, tau)?;
    }
    Ok((rear.x.value() == -k.end && !eligible_bsd).then_some("rear_exit"))
}
