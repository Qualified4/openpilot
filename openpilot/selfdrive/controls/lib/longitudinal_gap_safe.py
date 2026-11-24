"""Safety-distance following gap: the gap level (1-4) picks how hard the lead may brake.

d_min = REACTION * v + v^2 / (2 * EGO_MAX_DECEL) - v_lead^2 / (2 * lead_decel) + stop_distance

d_min is the gap from which ego, braking at its own limit after REACTION, still stops behind a
lead that brakes at `lead_decel` (LEAD_DECELS, one per level). Where a lead is the binding MPC
obstacle it replaces the TF distance: tF -> REACTION, comfort_brake -> EGO_MAX_DECEL, the lead's
stopped equivalence uses lead_decel instead of 2.5 and the danger factor is 1.0, so the danger
zone starts at d_min. Cruise, traffic-stop and Tesla-stop obstacles keep the TF values.

The comfort target (cost reference only) is
  d_min + COMFORT_TIME * v + CLOSING_TIME * (v - v_lead)+
        + (v^2 - v_lead^2)+ * (1 / (2 * comfort_brake) - 1 / (2 * EGO_MAX_DECEL)):
closing on a slower or stopping lead, ego plans to shed the speed difference at comfort_brake, not at
its limit, and keeps CLOSING_TIME per m/s of closing speed, so it starts braking earlier. At equal
speed both closing terms are 0, so the steady following gap is d_min + COMFORT_TIME * v.

A lead that cuts in closer than the comfort target at about ego speed or faster (closing at most
CUTIN_CLOSING_MAX) and is not braking gets an obstacle credit equal to the shortfall beyond the comfort
cushion (COMFORT_TIME * v); the credit then shrinks at CUTIN_RECOVER_RATE, so the gap is reopened
gradually instead of braking for it at once. The credit ends at once if that lead brakes. Lane changes
keep their own credit.
"""

import math
from typing import NamedTuple

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN

# The planner never requests more braking than ACCEL_MIN = -4.0 (opendbc/car/interfaces.py;
# Hyundai CarControllerParams.ACCEL_MIN is also -4.0), so that is ego's assumed maximum.
EGO_MAX_DECEL = -ACCEL_MIN
COMFORT_TIME = 0.35        # s x v above d_min for the comfort target (cushion above the danger zone)
DANGER_FACTOR = 1.0        # d_min is already the minimum, so no 0.8 shrink
ENABLE_TIME = 3.0          # s to blend in when longitudinal control engages
LEAD_DECEL_INV_RATE = 0.025  # 1/(m/s^2) per s: a level change that lengthens the gap (5 -> 8) takes 3 s
LEAD_DECELS = (5.0, 6.0, 7.0, 8.0)  # m/s^2 assumed lead braking for gap levels 1-4
REACTION = 0.6             # s from the lead's braking until ego's braking starts
GAP_LEVELS = len(LEAD_DECELS)
CLOSING_TIME = 1.0         # s x closing speed (v - v_lead)+ added to the comfort target
CUTIN_RECOVER_RATE = 1.5   # m/s the cut-in credit shrinks by
CUTIN_CLOSING_MAX = 1.5    # m/s: only a cut-in at about ego speed (or faster) gets the credit
CUTIN_LEAD_ACCEL_MIN = -0.5  # m/s^2: a braking cut-in gets none, and braking later cancels it


class LeadGap(NamedTuple):
  """MPC lead-node parameters, blended from the TF values by the state weight."""
  t_follow: float
  comfort_brake: float
  danger_factor: float
  inv_lead_decel: float
  comfort_time: float
  weight: float
  closing_factor: float = 0.0
  closing_time: float = 0.0


class GapSafeState:
  """Lengthening the gap ramps in (like the TF ramp); shortening it applies at once."""

  def __init__(self):
    self.weight = 0.0
    self.inv_lead_decel = None

  def update(self, *, enabled, lead_decel, snap, dt):
    if not enabled or not math.isfinite(lead_decel) or not math.isfinite(dt) or dt <= 0.0:
      self.__init__()
      return
    target = 1.0 / float(lead_decel)
    if snap or self.inv_lead_decel is None or target >= self.inv_lead_decel:
      self.inv_lead_decel = target
    else:
      self.inv_lead_decel = max(target, self.inv_lead_decel - LEAD_DECEL_INV_RATE * dt)
    self.weight = 1.0 if snap else min(1.0, self.weight + dt / ENABLE_TIME)

  def lead_gap(self, *, extra_time, t_follow, comfort_brake, danger_factor, inv_base_lead_decel):
    w = self.weight
    return LeadGap(
      t_follow=(1.0 - w) * t_follow + w * REACTION,
      comfort_brake=1.0 / ((1.0 - w) / comfort_brake + w / EGO_MAX_DECEL),
      danger_factor=(1.0 - w) * danger_factor + w * DANGER_FACTOR,
      inv_lead_decel=(1.0 - w) * inv_base_lead_decel + w * self.inv_lead_decel,
      comfort_time=w * (COMFORT_TIME + max(0.0, extra_time)),
      weight=w,
      closing_factor=w * max(0.0, 1.0 / (2.0 * comfort_brake) - 1.0 / (2.0 * EGO_MAX_DECEL)),
      closing_time=w * CLOSING_TIME,
    )


def stopped_equivalence(v_lead, gap: LeadGap):
  return v_lead**2 * gap.inv_lead_decel / 2.0


def safe_distance(v_ego, gap: LeadGap, stop_distance):
  return v_ego**2 / (2.0 * gap.comfort_brake) + gap.t_follow * v_ego + stop_distance


def min_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Danger-zone floor (d_min when fully blended)."""
  return gap.danger_factor * safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap)


def comfort_margin(v_ego, v_lead, gap: LeadGap):
  """Comfort target above d_min: comfort time plus shedding the closing speed at comfort_brake."""
  v_ego = np.maximum(v_ego, 0.0)
  v_lead = np.maximum(v_lead, 0.0)
  closing = np.maximum(0.0, v_ego**2 - v_lead**2)
  return gap.comfort_time * v_ego + gap.closing_factor * closing + gap.closing_time * np.maximum(0.0, v_ego - v_lead)


def desired_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Comfort target, the same quantity desired_follow_distance() gives in TF mode."""
  return safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap) + comfort_margin(v_ego, v_lead, gap)


def equivalent_t_follow(v_ego, gap: LeadGap):
  """(steady-following target - stop distance) / v, for code that reads a gap as TF seconds."""
  v = max(0.0, float(v_ego))
  return max(0.0, gap.t_follow + gap.comfort_time + v * (1.0 / (2.0 * gap.comfort_brake) - gap.inv_lead_decel / 2.0))


class CutinCredit:
  """Obstacle credit (m) for a lead that cut in inside the comfort target; it only shrinks."""

  def __init__(self):
    self.track_id = None
    self.credit = 0.0

  def reset(self):
    self.__init__()
    return 0.0

  def update(self, *, enabled, track_id, distance, desired, closing_speed, lead_accel, cushion, dt):
    if not enabled or not math.isfinite(distance) or not math.isfinite(desired):
      return self.reset()
    braking = not lead_accel >= CUTIN_LEAD_ACCEL_MIN
    if track_id != self.track_id:
      # A new lead: credit the shortfall only for a non-braking lead at about ego speed or faster;
      # a closing or braking cut-in gets the full response at once.
      self.track_id = track_id
      fits = closing_speed <= CUTIN_CLOSING_MAX and not braking
      # The comfort cushion stays as the initial gap error, so ego still eases off at once.
      shortfall = float(desired) - float(distance) - max(0.0, float(cushion))
      self.credit = max(0.0, shortfall) if fits else 0.0
      return self.credit
    self.credit = 0.0 if braking else max(0.0, self.credit - CUTIN_RECOVER_RATE * dt)
    return self.credit
