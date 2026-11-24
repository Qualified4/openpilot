"""Fast approach to a stopped or slow lead seen from far: brake early and softly.

Without this the plan holds the set speed after a stopped lead is detected at
78-95 m and starts braking around 40 m, peaking at -1.2 to -1.3 m/s^2 where a
constant -0.41 to -0.45 from detection would have stopped at the same place
(item 147-4, drives 11-09..11-24). While active, the cost reference for the
lead follows a constant-decel stopping curve frozen at detection, and the gap
cost weight rises up to x5. Obstacles, the danger constraint and accel limits
are unchanged, so a lead that brakes harder still gets the full braking.
"""

import math

import numpy as np


# Entry: the lead is slow (or stops within the horizon), ego closes fast, and it is still far.
APPROACH_LEAD_SPEED_MAX = 15.0 / 3.6
APPROACH_LEAD_END_SPEED_MAX = 1.0
APPROACH_CLOSING_MIN = 15.0 / 3.6
APPROACH_TIME_GAP_MIN = 1.5
# Hold on the same track until ego no longer closes or the lead speeds up again.
APPROACH_HOLD_CLOSING_MIN = 1.0
APPROACH_HOLD_LEAD_SPEED_MARGIN = 2.0
APPROACH_RAMP_TIME = 1.0
# Decel frozen at entry: 1.1 x the decel that stops ego at the stop distance behind the lead.
APPROACH_DECEL_GAIN = 1.1
APPROACH_DECEL_MIN = 0.5
APPROACH_DECEL_MAX = 2.0
# Gap cost weight: x(1 + 4 * ramp).
APPROACH_OBSTACLE_COST_GAIN = 4.0
# Same stopped-equivalence decel as the MPC lead obstacle (long_mpc.COMFORT_BRAKE).
LEAD_STOP_DECEL = 2.5


def approach_decel(*, distance, lead_speed, v_ego, stop_distance):
  stop_room = max(distance + lead_speed ** 2 / (2 * LEAD_STOP_DECEL) - stop_distance, 1.0)
  required = v_ego ** 2 / (2 * stop_room)
  return float(np.clip(APPROACH_DECEL_GAIN * required, APPROACH_DECEL_MIN, APPROACH_DECEL_MAX))


class ApproachState:
  def __init__(self):
    self.reset()

  def reset(self):
    self.track_id = None
    self.active = False
    self.decel = APPROACH_DECEL_MAX
    self.weight = 0.0
    return self.weight

  def update(self, *, enabled, lead_status, track_id, distance, lead_speed, lead_end_speed,
             v_ego, stop_distance, dt):
    values = (distance, lead_speed, lead_end_speed, v_ego, stop_distance, dt)
    if not enabled or not lead_status or not all(map(math.isfinite, values)) or distance <= 0.0:
      return self.reset()
    closing = v_ego - lead_speed
    if self.active and track_id == self.track_id:
      self.active = bool(closing > APPROACH_HOLD_CLOSING_MIN and (
        lead_speed <= APPROACH_LEAD_SPEED_MAX + APPROACH_HOLD_LEAD_SPEED_MARGIN
        or lead_end_speed <= APPROACH_LEAD_END_SPEED_MAX))
    else:
      slow = lead_speed <= APPROACH_LEAD_SPEED_MAX or lead_end_speed <= APPROACH_LEAD_END_SPEED_MAX
      self.track_id = track_id
      self.active = bool(slow and closing >= APPROACH_CLOSING_MIN
                         and distance / max(v_ego, 1.0) >= APPROACH_TIME_GAP_MIN)
      self.decel = (approach_decel(distance=distance, lead_speed=lead_speed, v_ego=v_ego, stop_distance=stop_distance)
                    if self.active else APPROACH_DECEL_MAX)
    self.weight = min(1.0, self.weight + dt / APPROACH_RAMP_TIME) if self.active else 0.0
    return self.weight

  def obstacle_cost_factor(self):
    return 1.0 + APPROACH_OBSTACLE_COST_GAIN * self.weight

  def reference_margins(self, *, ego_speeds, following_distances, stop_distance):
    """Extra following distance (m) per node; reference only, never an obstacle."""
    if self.weight <= 0.0:
      return np.zeros_like(ego_speeds)
    target = stop_distance + np.maximum(ego_speeds, 0.0) ** 2 / (2 * self.decel)
    return self.weight * np.maximum(0.0, target - following_distances)
