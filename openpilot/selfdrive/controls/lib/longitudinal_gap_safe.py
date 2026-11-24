"""Safety-distance following gap: the gap level (1-4) picks how hard the lead may brake.

d_min = REACTION * v + v^2 / (2 * EGO_MAX_DECEL) - v_lead^2 / (2 * lead_decel) + stop_distance

d_min is the gap from which ego, braking at its own limit after REACTION, still stops behind a
lead that brakes at `lead_decel` (LEAD_DECELS, one per level). It is used for a lead with status in
ACC mode at a known gap level; a missing lead (the MPC's fake far lead), blended (e2e) mode and an
unknown level keep the TF computation. Where such a lead is the binding MPC obstacle it replaces
the TF distance: tF -> REACTION, comfort_brake -> EGO_MAX_DECEL, the lead's stopped equivalence
uses lead_decel instead of 2.5 and the danger factor is 1.0, so the danger zone starts at d_min.
Cruise, traffic-stop and Tesla-stop obstacles keep the TF values. Because the two kinds subtract
different distances, the MPC compares obstacles by their targets (target_offsets()), not by raw value.

The comfort target (cost reference only) is
  d_min + comfort_time(level) * v + closing,
  closing = CLOSING_TIME * (v - v_lead)+ + (v^2 - v_lead^2)+ * (1 / (2 * comfort_brake) - 1 / (2 * EGO_MAX_DECEL)):
closing on a slower or stopping lead, ego plans to shed the speed difference at comfort_brake, not at
its limit, and keeps CLOSING_TIME per m/s of closing speed, so it starts braking earlier. At equal
speed closing is 0, so the steady following gap is d_min + comfort_time * v. comfort_time is COMFORT_TIMES
per level at every speed; a level change ramps it with the lead decel.
gap_reserve() is the gap above the steady target; the planner uses it to pull in less of the lead-decel
preview. Taking it off the closing terms as well was replayed and dropped: departures that open the gap
re-closed it (gap falling > 2 m after its peak: 15 -> 20 % of 20 departures) and level 2 steady-following
brake entries rose 1.95 -> 2.16 per minute.
The whole comfort margin fades out between the COMFORT_FADE_SPEEDS (ego speed), so the car stops at
d_min, i.e. at the stop distance, behind a stopped lead. While ego brakes hard (COMFORT_FADE_ACCELS)
the margin stays full.

A lead that cuts in closer than the steady comfort target at about ego speed or faster (closing at most
CUTIN_CLOSING_MAX) and is not braking gets a cut-in credit equal to the shortfall beyond the comfort
cushion (comfort_time * v); the credit then shrinks at CUTIN_RECOVER_RATE, so the gap is reopened
gradually instead of braking for it at once. The credit moves the lead obstacle, and with it the
danger floor d_min, back, but never below the TF danger floor (LEAD_DANGER_FACTOR x TF distance -
v_lead^2 / (2 x COMFORT_BRAKE)), and lowers the cost reference to max(TF following target, comfort
target - credit): with a credit the plan is never looser than the TF one.
It ends at once if that lead brakes, and
a track id change of what looks like the same car (distance within SAME_LEAD_DISTANCE of the last
lead), a lead seen again within CUTIN_HOLD_EXPIRE after a dropout or the lead found after a lane change
get none. Only radar track ids (>= 0) identify the same car. Lane changes keep their own credit.
"""

import math
from typing import NamedTuple

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN

# The planner never requests more braking than ACCEL_MIN = -4.0 (opendbc/car/interfaces.py;
# Hyundai CarControllerParams.ACCEL_MIN is also -4.0), so that is ego's assumed maximum.
EGO_MAX_DECEL = -ACCEL_MIN
# s x v above d_min for the steady comfort target per level (1-4), at every speed; a level change blends them
# with the lead decel. Owner's choice C (work/drive-20261125/REPORT_GAP47D_KO.md): level 1 at about the old
# TF level 1 gap (equal speed, stop distance 3.7 m: 30/50/80/100 km/h 9.6/14.1/21.8/27.4 m), level 4 as before
# (16.0/29.0/55.7/78.3 m). Level 1 keeps less margin for a lead braking harder than assumed than 0a704cf3
# (synthetic F50 -6: 1.51 -> -3.90 m), still 0.45-0.85 m more than the old TF level 1 in all 9 hard stops.
COMFORT_TIMES = (0.05, 0.20, 0.30, 0.35)
DANGER_FACTOR = 1.0        # d_min is already the minimum, so no 0.8 shrink
LEAD_DECEL_INV_RATE = 0.025  # 1/(m/s^2) per s: a level change that lengthens the gap (4.25 -> 8) takes 4.4 s
LEAD_DECELS = (4.25, 5.0, 6.5, 8.0)  # m/s^2 assumed lead braking for gap levels 1-4
REACTION = 0.6             # s from the lead's braking until ego's braking starts
GAP_LEVELS = len(LEAD_DECELS)
CLOSING_TIME = 1.0         # s x closing speed (v - v_lead)+ added to the comfort target
CUTIN_RECOVER_RATE = 1.5   # m/s the cut-in credit shrinks by
CUTIN_CLOSING_MAX = 1.5    # m/s: only a cut-in at about ego speed (or faster) gets the credit
CUTIN_LEAD_ACCEL_MIN = -0.5  # m/s^2: a braking cut-in gets none, and braking later cancels it
SAME_LEAD_DISTANCE = 1.5   # m: a new track id this close to the last lead's distance is the same car
# m/s: the comfort margin is 0 at or below the first ego speed and full from the second. A margin linear
# in v (comfort time + closing time, ~1.35 s) kept the plan that much behind the stop distance at the
# low speed where the car stops: replay of 14 stops behind a stopped lead, median 4.36 m for 3.7 m
# (3.89 m with this fade; the rest is d_min's reaction term at the stop speed).
COMFORT_FADE_SPEEDS = (0.5, 3.0)
# m/s^2: no fade while ego brakes at or below the first accel, full fade from the second. A hard stop
# ending inside the stop distance keeps the full margin (synthetic F30 -4: 3.11 m, 3.01 m with the
# fade alone); the stops above all ended at -0.6 to -1.0 below 3 m/s.
COMFORT_FADE_ACCELS = (-2.0, -1.5)
# m/s^2 lead accel: no gap reserve at or below the first, full reserve from the second.
RESERVE_LEAD_ACCELS = (-1.0, -0.6)
CUTIN_HOLD_EXPIRE = 2.0    # s without a lead after which the last lead is forgotten
# The same lead ending its braking (aLeadK back at or above CUTIN_LEAD_ACCEL_MIN) with ego inside the steady
# target gets the shortfall as credit too, recovered at POST_BRAKE_RECOVER_RATE (ego about 1.8 km/h below the
# lead) instead of braking on until the gap is back. Replay 479 10:35: the lead settled at 31 km/h 4.7 m inside
# the target and ego braked on to 26 km/h. Only with ego and lead at or above POST_BRAKE_SPEED_MIN (m/s):
# near a stop the credit would move the stop closer.
POST_BRAKE_RECOVER_RATE = 0.5
POST_BRAKE_SPEED_MIN = 5.0
# The same lead starting to brake no harder than BRAKE_HOLD_ACCEL_MIN (m/s^2, aLeadK) with ego inside the steady
# target: that shortfall is held as credit while it brakes, so ego answers the lead's braking and does not also
# win the old shortfall back during it (479 10:35: ego -1.7 for a lead at -1.4, ending 4 km/h below it).
# Harder lead braking drops the credit at once and gives no post-brake credit for that braking either (a lead
# easing off from -3 may brake again: synthetic F50 -3 / release / -6 lost 0.3 m); below POST_BRAKE_SPEED_MIN
# it recovers at CUTIN_RECOVER_RATE.
BRAKE_HOLD_ACCEL_MIN = -2.0
# s the same lead must have been followed for the hold and the post-brake credit; a cut-in braking within it, or
# with cut-in credit left, ends the credit at once as before (synthetic cut-in then -4 at levels 2-4: -0.7 m).
BRAKE_HOLD_LEAD_AGE = 5.0
# s the braking must stay no harder than BRAKE_HOLD_ACCEL_MIN before the hold starts: a lead whose aLeadK is
# still ramping down to -4 was held for 0.3 s (synthetic cut-in then -4 at level 2: -0.11 m).
BRAKE_HOLD_CONFIRM = 0.5


class LeadGap(NamedTuple):
  """MPC lead-node parameters."""
  t_follow: float
  comfort_brake: float
  danger_factor: float
  inv_lead_decel: float
  comfort_time: float       # at the current ego speed (cut-in cushion, TF equivalents)
  closing_factor: float = 0.0
  closing_time: float = 0.0
  fade_weight: float = 1.0  # how much of the low-speed comfort fade applies (COMFORT_FADE_ACCELS)


class GapSafeState:
  """Lengthening the gap ramps in (like the TF ramp); shortening it applies at once."""

  def __init__(self):
    self.inv_lead_decel = None

  @property
  def active(self):
    return self.inv_lead_decel is not None

  def update(self, *, enabled, lead_decel, snap, dt):
    if not enabled or not math.isfinite(lead_decel) or not math.isfinite(dt) or dt <= 0.0:
      self.__init__()
      return
    target = 1.0 / float(lead_decel)
    if snap or self.inv_lead_decel is None or target >= self.inv_lead_decel:
      self.inv_lead_decel = target
    else:
      self.inv_lead_decel = max(target, self.inv_lead_decel - LEAD_DECEL_INV_RATE * dt)

  def lead_gap(self, *, comfort_brake, v_ego=0.0, a_ego=0.0):
    return LeadGap(
      t_follow=REACTION,
      comfort_brake=EGO_MAX_DECEL,
      danger_factor=DANGER_FACTOR,
      inv_lead_decel=self.inv_lead_decel,
      comfort_time=level_comfort_time(self.inv_lead_decel),
      closing_factor=max(0.0, 1.0 / (2.0 * comfort_brake) - 1.0 / (2.0 * EGO_MAX_DECEL)),
      closing_time=CLOSING_TIME,
      fade_weight=float(np.interp(a_ego, COMFORT_FADE_ACCELS, (0.0, 1.0))),
    )


def level_comfort_time(inv_lead_decel):
  """Comfort time for a (possibly ramping) lead decel: levels blend like the decel."""
  level = float(np.clip(1.0 / inv_lead_decel, LEAD_DECELS[0], LEAD_DECELS[-1]))
  return float(np.interp(level, LEAD_DECELS, COMFORT_TIMES))


def comfort_times(v_ego, gap: LeadGap):
  return gap.comfort_time


def stopped_equivalence(v_lead, gap: LeadGap):
  return v_lead**2 * gap.inv_lead_decel / 2.0


def safe_distance(v_ego, gap: LeadGap, stop_distance):
  return v_ego**2 / (2.0 * gap.comfort_brake) + gap.t_follow * v_ego + stop_distance


def min_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Danger-zone floor (d_min when fully blended)."""
  return gap.danger_factor * safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap)


def comfort_margin(v_ego, v_lead, gap: LeadGap):
  """Comfort target above d_min: comfort time plus shedding the closing speed at comfort_brake.

  It fades out between the COMFORT_FADE_SPEEDS, so near standstill the target is d_min alone, unless
  ego brakes hard (gap.fade_weight)."""
  v_ego = np.maximum(v_ego, 0.0)
  v_lead = np.maximum(v_lead, 0.0)
  closing = gap.closing_factor * np.maximum(0.0, v_ego**2 - v_lead**2) + gap.closing_time * np.maximum(0.0, v_ego - v_lead)
  margin = comfort_times(v_ego, gap) * v_ego + closing
  return margin * fade(v_ego, gap)


def fade(v_ego, gap: LeadGap):
  return 1.0 - gap.fade_weight * (1.0 - np.interp(v_ego, COMFORT_FADE_SPEEDS, (0.0, 1.0)))


def steady_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Comfort target without the closing terms: d_min + comfort time (faded)."""
  v = max(0.0, float(v_ego))
  return float(min_distance(v, v_lead, gap, stop_distance) + comfort_times(v, gap) * v * fade(v, gap))


def gap_reserve(*, distance, v_ego, v_lead, lead_accel, gap: LeadGap, stop_distance):
  """Gap above the steady comfort target (m); 0 while the lead brakes harder than RESERVE_LEAD_ACCELS."""
  values = (distance, v_ego, v_lead, lead_accel)
  if not all(map(math.isfinite, values)):
    return 0.0
  reserve = max(0.0, float(distance) - steady_distance(v_ego, v_lead, gap, stop_distance))
  return reserve * float(np.interp(lead_accel, RESERVE_LEAD_ACCELS, (0.0, 1.0)))


def desired_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Comfort target, the same quantity desired_follow_distance() gives in TF mode."""
  return safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap) + comfort_margin(v_ego, v_lead, gap)


def equivalent_t_follow(v_ego, gap: LeadGap):
  """(steady-following target - stop distance) / v, for code that reads a gap as TF seconds."""
  v = max(0.0, float(v_ego))
  return max(0.0, gap.t_follow + float(comfort_times(v, gap)) + v * (1.0 / (2.0 * gap.comfort_brake) - gap.inv_lead_decel / 2.0))


def target_offset(v_ego, gap: LeadGap, t_follow, comfort_brake, stop_distance):
  """How much more a lead obstacle under `gap` subtracts than a TF obstacle, at ego speeds v_ego.

  obstacle - target_offset() puts a gap-safe lead obstacle on the TF obstacles' scale, so the
  smallest value is the obstacle with the closest target."""
  v = np.maximum(v_ego, 0.0)
  return safe_distance(v, gap, stop_distance) - (v**2 / (2.0 * comfort_brake) + t_follow * v + stop_distance)


class CutinCredit:
  """Credit (m) for a lead inside the steady target: a cut-in that is not chased, or the shortfall held through
  and after a followed lead's moderate braking (relaxed). It is recovered gradually."""

  def __init__(self):
    self.track_id = None
    self.distance = None
    self.credit = 0.0
    self.hold_time = 0.0
    self.rate = CUTIN_RECOVER_RATE
    self.braking_seen = False
    self.brake_min = 0.0  # lowest lead accel of the braking that braking_seen marks
    self.relaxed = False  # credit from the brake hold / post-brake (not a cut-in): bounded by d_min, not the TF target
    self.age = 0.0  # s this lead has been followed
    self.brake_time = 0.0  # s since the braking that braking_seen marks started
    self.hold_pending = False  # that braking may be held once it lasted BRAKE_HOLD_CONFIRM

  def reset(self):
    self.__init__()
    return 0.0

  def hold(self, dt=0.0):
    """No credit now (no lead, lane change, not engaged); the last lead is kept for CUTIN_HOLD_EXPIRE,
    so it gets none on return within that time."""
    self.credit = 0.0
    self.hold_time += max(0.0, float(dt)) if math.isfinite(dt) else 0.0
    if self.hold_time > CUTIN_HOLD_EXPIRE:
      self.track_id, self.distance = None, None
    return 0.0

  def update(self, *, enabled, track_id, distance, desired, closing_speed, lead_accel, cushion, dt, grant=True,
             v_ego=0.0, v_lead=0.0):
    """desired: the steady comfort target (no closing terms)."""
    if not enabled or not math.isfinite(distance) or not math.isfinite(desired):
      return self.reset()
    braking = not lead_accel >= CUTIN_LEAD_ACCEL_MIN
    same_lead = (track_id is not None and track_id >= 0 and track_id == self.track_id) or (
      self.distance is not None and abs(float(distance) - self.distance) < SAME_LEAD_DISTANCE)
    self.track_id, self.distance = track_id, float(distance)
    self.hold_time = 0.0  # the lead is seen
    if not grant:
      self.credit = 0.0
      self.braking_seen = False
      return 0.0
    self.age = self.age + max(0.0, float(dt)) if same_lead else 0.0
    if not same_lead:
      self.braking_seen = braking
      self.brake_min = min(0.0, float(lead_accel))
      self.rate = CUTIN_RECOVER_RATE
      # A new lead: credit the shortfall only for a non-braking lead at about ego speed or faster;
      # a closing or braking cut-in gets the full response at once.
      fits = closing_speed <= CUTIN_CLOSING_MAX and not braking
      # The comfort cushion stays as the initial gap error, so ego still eases off at once.
      shortfall = float(desired) - float(distance) - max(0.0, float(cushion))
      self.credit = max(0.0, shortfall) if fits else 0.0
      self.relaxed = False
      return self.credit
    if braking:
      fast = min(float(v_ego), float(v_lead)) >= POST_BRAKE_SPEED_MIN
      self.brake_min = min(self.brake_min if self.braking_seen else 0.0, float(lead_accel))
      if self.brake_min < BRAKE_HOLD_ACCEL_MIN:
        self.credit = 0.0
      elif not self.braking_seen:  # braking starts: full response first, the hold may follow
        cutin_left = self.credit > 0.0 and not self.relaxed
        self.hold_pending = fast and closing_speed <= CUTIN_CLOSING_MAX and self.age >= BRAKE_HOLD_LEAD_AGE and not cutin_left
        self.credit, self.relaxed, self.brake_time = 0.0, False, 0.0
      elif not fast:
        self.credit = max(0.0, self.credit - CUTIN_RECOVER_RATE * dt)
      if self.braking_seen:
        self.brake_time += max(0.0, float(dt))
        if self.hold_pending and self.brake_time >= BRAKE_HOLD_CONFIRM - 1e-6 and self.brake_min >= BRAKE_HOLD_ACCEL_MIN:
          self.hold_pending = False
          if fast and closing_speed <= CUTIN_CLOSING_MAX:
            self.credit, self.relaxed = max(self.credit, float(desired) - float(distance), 0.0), True
      self.braking_seen = True
      return self.credit
    if self.braking_seen and closing_speed <= CUTIN_CLOSING_MAX:
      self.braking_seen = False
      if (min(float(v_ego), float(v_lead)) >= POST_BRAKE_SPEED_MIN and self.brake_min >= BRAKE_HOLD_ACCEL_MIN
          and self.age >= BRAKE_HOLD_LEAD_AGE):
        shortfall = max(0.0, float(desired) - float(distance))
        if shortfall > self.credit:
          self.credit, self.rate, self.relaxed = shortfall, POST_BRAKE_RECOVER_RATE, True
        return self.credit
    self.credit = max(0.0, self.credit - self.rate * dt)
    return self.credit
