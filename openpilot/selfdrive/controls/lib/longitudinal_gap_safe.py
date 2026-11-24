"""Safety-distance following gap (GapSafeDistance): the gap level picks how hard the lead may brake.

d_min = reaction * v + v^2 / (2 * EGO_MAX_DECEL) - v_lead^2 / (2 * lead_decel) + stop_distance

d_min is the gap from which ego, braking at its own limit after `reaction`, still stops behind a
lead that brakes at `lead_decel`. Where a lead is the binding MPC obstacle it replaces the TF
distance: tF -> reaction, comfort_brake -> EGO_MAX_DECEL, the lead's stopped equivalence uses
lead_decel instead of 2.5 and the danger factor is 1.0, so the danger zone starts at d_min.
The comfort target is d_min + comfort_time * v, applied on the cost reference only.
Cruise, traffic-stop and Tesla-stop obstacles keep the TF values.
"""

import math
from typing import NamedTuple

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN

# The planner never requests more braking than ACCEL_MIN = -4.0 (opendbc/car/interfaces.py;
# Hyundai CarControllerParams.ACCEL_MIN is also -4.0), so that is ego's assumed maximum.
EGO_MAX_DECEL = -ACCEL_MIN
COMFORT_TIME = 0.15        # s x v above d_min for the comfort target
DANGER_FACTOR = 1.0        # d_min is already the minimum, so no 0.8 shrink
ENABLE_TIME = 3.0          # s to blend in after the setting turns on while engaged
LEAD_DECEL_INV_RATE = 0.025  # 1/(m/s^2) per s: a level change that lengthens the gap (5 -> 8) takes 3 s
LEAD_DECEL_LIMITS = (4.0, 10.0)
REACTION_LIMITS = (0.3, 1.2)
GAP_LEVELS = 4             # GapSafeLeadDecel1-4, one per LongitudinalPersonality


class LeadGap(NamedTuple):
  """MPC lead-node parameters, blended from the TF values by the state weight."""
  t_follow: float
  comfort_brake: float
  danger_factor: float
  inv_lead_decel: float
  comfort_time: float
  weight: float


class GapSafeState:
  """Lengthening the gap ramps in (like the TF ramp); shortening it applies at once."""

  def __init__(self):
    self.weight = 0.0
    self.inv_lead_decel = None

  def update(self, *, enabled, lead_decel, snap, dt):
    if not enabled or not math.isfinite(lead_decel) or not math.isfinite(dt) or dt <= 0.0:
      self.__init__()
      return
    target = 1.0 / float(np.clip(lead_decel, *LEAD_DECEL_LIMITS))
    if snap or self.inv_lead_decel is None or target >= self.inv_lead_decel:
      self.inv_lead_decel = target
    else:
      self.inv_lead_decel = max(target, self.inv_lead_decel - LEAD_DECEL_INV_RATE * dt)
    self.weight = 1.0 if snap else min(1.0, self.weight + dt / ENABLE_TIME)

  def lead_gap(self, *, reaction, extra_time, t_follow, comfort_brake, danger_factor, inv_base_lead_decel):
    w = self.weight
    reaction = float(np.clip(reaction, *REACTION_LIMITS))
    return LeadGap(
      t_follow=(1.0 - w) * t_follow + w * reaction,
      comfort_brake=1.0 / ((1.0 - w) / comfort_brake + w / EGO_MAX_DECEL),
      danger_factor=(1.0 - w) * danger_factor + w * DANGER_FACTOR,
      inv_lead_decel=(1.0 - w) * inv_base_lead_decel + w * self.inv_lead_decel,
      comfort_time=w * (COMFORT_TIME + max(0.0, extra_time)),
      weight=w,
    )


def stopped_equivalence(v_lead, gap: LeadGap):
  return v_lead**2 * gap.inv_lead_decel / 2.0


def safe_distance(v_ego, gap: LeadGap, stop_distance):
  return v_ego**2 / (2.0 * gap.comfort_brake) + gap.t_follow * v_ego + stop_distance


def min_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Danger-zone floor (d_min when fully blended)."""
  return gap.danger_factor * safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap)


def desired_distance(v_ego, v_lead, gap: LeadGap, stop_distance):
  """Comfort target, the same quantity desired_follow_distance() gives in TF mode."""
  return safe_distance(v_ego, gap, stop_distance) - stopped_equivalence(v_lead, gap) + gap.comfort_time * np.maximum(v_ego, 0.0)


def equivalent_t_follow(v_ego, gap: LeadGap):
  """(steady-following target - stop distance) / v, for code that reads a gap as TF seconds."""
  v = max(0.0, float(v_ego))
  return max(0.0, gap.t_follow + gap.comfort_time + v * (1.0 / (2.0 * gap.comfort_brake) - gap.inv_lead_decel / 2.0))
