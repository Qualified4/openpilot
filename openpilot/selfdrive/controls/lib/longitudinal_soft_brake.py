"""Spread a moderate lead-following deceleration over the MPC comfort trajectory."""

import math

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_gap_safe import steady_distance


SOFT_LEAD_DECEL_LIMIT = -2.5  # m/s²
SOFT_LEAD_DECEL_ONSET = -0.8  # m/s²
SOFT_LEAD_DECEL_END = -0.3  # m/s²
SOFT_BRAKE_ACCEL = -1.5  # m/s², a cost reference only
HARD_EGO_DECEL = -2.0  # m/s², leave an already-established strong brake to the normal path
SOFT_BRAKE_TRACK_TIME = 4.0  # s of continuous radar tracking
SOFT_BRAKE_MIN_TIME_GAP = 1.8


class SoftBrakeState:
  def __init__(self):
    self.track_id = None
    self.time_gap = math.inf
    self.blocked = False
    self.braking = False

  def update(self, track_id, *, reset, lead_accel, ego_accel, closing, time_gap):
    if reset:
      self.__init__()
      return
    if track_id is None:
      # The vision-only handover frames can precede a new radar lead.
      # Retain the last measured gap so a close reveal stays inhibited.
      return
    # A hard brake stays in the normal path through the end of the episode.
    # A lead revealed from a close gap also needs the normal response.
    new_lead = track_id != self.track_id
    if new_lead:
      self.braking = False
    # An approach which started far back keeps its normal response after the gap closes.
    far_onset = not self.braking and lead_accel <= SOFT_LEAD_DECEL_ONSET and time_gap >= 3.25
    if lead_accel <= SOFT_LEAD_DECEL_ONSET:
      self.braking = True
    elif lead_accel >= SOFT_LEAD_DECEL_END:
      self.braking = False
    if (far_onset or lead_accel <= SOFT_LEAD_DECEL_LIMIT or ego_accel <= HARD_EGO_DECEL
        or (new_lead and (ego_accel <= SOFT_BRAKE_ACCEL or self.time_gap < SOFT_BRAKE_MIN_TIME_GAP))):
      self.blocked = True
    elif new_lead or (lead_accel >= SOFT_LEAD_DECEL_END and closing <= 0.2 and time_gap >= 2.1):
      self.blocked = False
    self.track_id = track_id
    self.time_gap = time_gap


def soft_brake_weight(*, distance, ego_speed, lead_speed, lead_accel, lead_end_speed, planned_decel, gap, stop_distance):
  """Use the spare distance above the steady target, with room for current closing speed."""
  if (not all(map(math.isfinite, (distance, ego_speed, lead_speed, lead_accel, lead_end_speed, planned_decel, stop_distance)))
      or lead_end_speed >= 1.0 or ego_speed <= 0.0):
    return 0.0
  closing = ego_speed - lead_speed
  if not SOFT_LEAD_DECEL_LIMIT < lead_accel < SOFT_LEAD_DECEL_END or closing <= 0.2 or lead_speed <= 3.0:
    return 0.0
  reserve = distance - steady_distance(ego_speed, lead_speed, gap, stop_distance) - 0.35 * closing
  return float(
    np.interp(reserve, (1.0, 3.0), (0.0, 1.0))
    * np.interp(lead_accel, (SOFT_LEAD_DECEL_LIMIT, -2.3, SOFT_LEAD_DECEL_ONSET, SOFT_LEAD_DECEL_END), (0.0, 1.0, 1.0, 0.0))
    * np.interp(lead_speed, (3.0, 4.5), (0.0, 1.0))
    * np.interp(closing, (0.2, 1.4), (0.0, 1.0))
    * np.interp(ego_speed, (14.5, 15.5), (1.0, 0.0))
    * np.interp(distance / ego_speed, (SOFT_BRAKE_MIN_TIME_GAP, 2.2, 2.85, 3.05), (0.0, 1.0, 1.0, 0.0))
    * np.interp(lead_speed / ego_speed, (0.4, 0.52), (0.0, 1.0))
    * np.interp(planned_decel, (1.25, 1.6, 2.65, 3.1), (0.0, 1.0, 1.0, 0.0))
  )


def soft_brake_reference_delta(times, previous_accels, *, weight, closing, danger_margin):
  """Advance the start of braking, and soften its depth only with predicted clearance."""
  preview_time = np.interp(closing, (1.0, 3.5), (1.5, 0.17))
  preview = np.interp(times + preview_time, times, previous_accels)
  delta = np.clip(preview, SOFT_BRAKE_ACCEL, 0.0) - previous_accels
  reserve = np.interp(danger_margin, (0.0, 1.5), (0.0, 1.0)) if math.isfinite(danger_margin) else 0.0
  return weight * (np.minimum(delta, 0.0) + reserve * np.maximum(delta, 0.0))
