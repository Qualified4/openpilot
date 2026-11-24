"""Lead brake release (LeadBrakeReleaseEase, off by default).

When a closely followed lead is seen to ease off its brake, the MPC stops
extrapolating the lead's deceleration until it stops (B1g), and while the lead
eases and ego still brakes clearly harder than the lead, the accel change and
jerk costs are lowered so the plan can come out of the deceleration (B4b).

Detection, release and cost values are the item-147 replay prototype's. A
renewed braking of the lead (jLead below REBRAKE_JERK) restarts the braking
episode, so the next cycle uses the original prediction again, and right after
a release it keeps the eased costs briefly so the brake comes back quickly.
Nothing is released while ego still closes on the lead (MAX_CLOSING_SPEED,
COST_EASE_MAX_CLOSING_SPEED); without that, a lead braking again at -6 after
letting go cost 2.2 m of minimum distance in replay.
"""
import math

BRAKING_ACCEL = -0.3      # m/s^2, the lead counts as braking below this
RELEASE_JERK = 1.0        # m/s^3, jLead above this shows the brake easing
RELEASE_RISE = 0.5        # m/s^2, or aLeadK this far above the episode's minimum
RELEASE_TAU = 1.5         # minimum aLeadTau while easing
RELEASE_PROJECTION_S = 0.5  # predicted accel = aLeadK + this * jLead (capped at 0)
RELEASE_MAX_TIME_GAP = 1.2  # s, close following only
REBRAKE_JERK = -1.0       # m/s^3, jLead below this is renewed braking
MIN_GAP = 3.0             # m
REBRAKE_WINDOW_S = 2.0    # renewed braking this soon after an easing re-applies the brake quickly
REBRAKE_BOOST_S = 1.0     # for this long, with the eased costs
# Guards added over the prototype: releasing while ego still closes on the lead
# costs distance if the lead brakes again (replay: lead lets go, -6 after 0.5 s).
MAX_CLOSING_SPEED = 0.0   # m/s, the prediction is released only once ego is no faster than the lead
COST_EASE_MAX_CLOSING_SPEED = 0.5  # m/s, the costs are eased only while ego closes slower than this
MIN_MODEL_PROB = 0.5

COST_EASE_LEAD_ACCEL = -0.5       # lead easing: aLeadK above this,
COST_EASE_LEAD_ACCEL_JERK = -1.5  # or jLead > RELEASE_JERK with aLeadK above this
COST_EASE_EGO_ACCEL = -1.0        # ego plan brakes harder than this
COST_EASE_EGO_EXCESS = 0.5        # and this much harder than the lead
A_CHANGE_COST_FACTOR = 0.02
JERK_COST_FACTOR = 0.1


def _lead_usable(lead) -> bool:
  if not (lead.status and lead.radar and lead.radarTrackId >= 0 and lead.modelProb >= MIN_MODEL_PROB):
    return False
  values = (lead.dRel, lead.vLead, lead.vRel, lead.aLeadK, lead.aLeadTau, lead.jLead)
  return all(math.isfinite(v) for v in values) and lead.dRel >= MIN_GAP


class LeadBrakeReleaseState:
  def __init__(self):
    self.reset()

  def reset(self):
    self.track_id = None
    self.a_min = 0.0
    self.easing = False
    self.since_easing = math.inf
    self.rebrake_boost = 0.0

  def prediction(self, lead, a_lead: float, a_lead_tau: float, dt: float) -> tuple[float, float]:
    """Lead accel and decay used for the MPC extrapolation."""
    if not _lead_usable(lead):
      self.reset()
      return a_lead, a_lead_tau
    if lead.radarTrackId != self.track_id:
      self.reset()
      self.track_id = lead.radarTrackId
    self.since_easing = 0.0 if self.easing else self.since_easing + dt
    self.rebrake_boost = max(0.0, self.rebrake_boost - dt)
    self.easing = False
    v_ego = lead.vLead - lead.vRel
    if lead.dRel / max(v_ego, 1.0) > RELEASE_MAX_TIME_GAP:
      return a_lead, a_lead_tau

    a, j = lead.aLeadK, lead.jLead
    rebraking = j < REBRAKE_JERK
    if rebraking and self.since_easing <= REBRAKE_WINDOW_S:
      self.rebrake_boost = REBRAKE_BOOST_S
    if a >= BRAKING_ACCEL:
      self.a_min = 0.0
      return a_lead, a_lead_tau
    self.a_min = a if rebraking else min(self.a_min, a)
    self.easing = (j > RELEASE_JERK or a > self.a_min + RELEASE_RISE) and lead.vRel > -MAX_CLOSING_SPEED
    if not self.easing:
      return a_lead, a_lead_tau
    return min(0.0, a + max(0.0, j) * RELEASE_PROJECTION_S), max(a_lead_tau, RELEASE_TAU)


def release_cost_ease(lead, a_ego: float, predicted_danger_margin: float) -> bool:
  """Lower the accel change and jerk costs while the lead's braking eases."""
  if not _lead_usable(lead) or predicted_danger_margin <= 0.0:
    return False
  a, j = lead.aLeadK, lead.jLead
  easing = a > COST_EASE_LEAD_ACCEL or (j > RELEASE_JERK and a > COST_EASE_LEAD_ACCEL_JERK)
  easing = easing and lead.vRel > -COST_EASE_MAX_CLOSING_SPEED
  return easing and a_ego < COST_EASE_EGO_ACCEL and a_ego < a - COST_EASE_EGO_EXCESS
