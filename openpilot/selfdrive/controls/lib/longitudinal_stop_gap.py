"""Stopping behind a lead: weight the standstill gap enough to reach it.

The MPC gap residual is divided by (v_ego + 10), so near standstill a 1 m gap
error costs about as much as a small jerk. Behind a stopping lead the plan then
settles about 0.9 m short of the configured stop distance from 4-5 s before the
stop (2026-11-11 0000045e t=101.8 s: planned and recorded 2.6-2.7 m for 3.6 m;
closed-loop replay of four stops from >10 km/h: 0.7-2.2 m short). This raises
only the gap-cost weight. Obstacles, danger limits and accel limits are unchanged.
"""

import math

import numpy as np


STOP_GAP_OBSTACLE_COST_FACTOR = 20.0
# Lead speed predicted at the end of the MPC horizon: the lead stops in the plan.
STOP_GAP_LEAD_END_SPEED = 1.0
# Full factor below the first ego speed, none above the second.
STOP_GAP_EGO_SPEEDS = (6.0, 12.0)
# Entry ramp time from factor 1 to full, and smooth decay time on exit (Stage 2 of #253).
STOP_GAP_RISE_TIME = 1.0
STOP_GAP_FALL_TIME = 0.60
# Only the final approach: a heavier gap cost while still far behind the
# configured distance would pull the ego toward the stopped lead (replay
# 11-11 t=89.9 s: +0.31 -> +0.81 m/s^2 at 75 m) and soften approach braking.
STOP_GAP_MAX_EXCESS = 2.0
# One-way anti-creep deadband (P1): when already at low speed inside the natural settle tolerance
# (0 < gap_excess <= 0.90 m) and closing the remaining gap would force releasing brake-lamp pressure,
# drop the 20x pull back to 1.0 so the MPC never releases brakes just to creep 50-80 cm forward.
STOP_GAP_ANTI_CREEP_V_MAX = 1.8
STOP_GAP_ANTI_CREEP_EXCESS_MAX = 0.90
STOP_GAP_ANTI_CREEP_MARGIN = 0.10


def stop_gap_target_factor(*, lead_status: bool, lead_end_speed: float, lead_sets_stop: bool, v_ego: float,
                           gap_excess: float, a_lamp: float = -0.45) -> float:
  """gap_excess: lead distance minus the configured following distance now."""
  if (not lead_status or not lead_sets_stop or not all(map(math.isfinite, (lead_end_speed, v_ego, gap_excess)))
      or lead_end_speed >= STOP_GAP_LEAD_END_SPEED or gap_excess > STOP_GAP_MAX_EXCESS):
    return 1.0
  if (0.0 < gap_excess <= STOP_GAP_ANTI_CREEP_EXCESS_MAX
      and 0.0 <= v_ego <= STOP_GAP_ANTI_CREEP_V_MAX
      and math.isfinite(a_lamp) and a_lamp < -0.1):
    dx_lamp = (v_ego * v_ego) / (2.0 * abs(a_lamp))
    if dx_lamp < gap_excess - STOP_GAP_ANTI_CREEP_MARGIN:
      return 1.0
  weight = float(np.interp(v_ego, STOP_GAP_EGO_SPEEDS, (1.0, 0.0)))
  return 1.0 + (STOP_GAP_OBSTACLE_COST_FACTOR - 1.0) * weight


class StopGapWeightState:
  def __init__(self):
    self.factor = 1.0

  def update(self, *, enabled: bool, lead_status: bool, lead_end_speed: float, lead_sets_stop: bool,
             v_ego: float, gap_excess: float, dt: float, a_lamp: float = -0.45) -> float:
    if not enabled or not math.isfinite(dt) or dt <= 0.0:
      self.factor = 1.0
      return self.factor
    target = stop_gap_target_factor(lead_status=lead_status, lead_end_speed=lead_end_speed,
                                    lead_sets_stop=lead_sets_stop, v_ego=v_ego, gap_excess=gap_excess,
                                    a_lamp=a_lamp)
    if target > self.factor:
      step = (STOP_GAP_OBSTACLE_COST_FACTOR - 1.0) * dt / STOP_GAP_RISE_TIME
      self.factor = min(target, self.factor + step)
    elif target < self.factor:
      step = (STOP_GAP_OBSTACLE_COST_FACTOR - 1.0) * dt / STOP_GAP_FALL_TIME
      self.factor = max(target, self.factor - step)
    return self.factor
