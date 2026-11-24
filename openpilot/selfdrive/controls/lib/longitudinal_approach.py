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
# A lead that clearly pulls away (and is not predicted to stop) is no approach: no entry, no hold.
# Replay 47c 10:44: a lead starting from 0.8 km/h at 124 m was held for 5.1 s up to 22 km/h,
# planning up to 1.17 m/s^2 more braking than without the approach.
APPROACH_LEAD_ACCEL_MAX = 0.3
# Once held, that release applies only while the lead is still this far: close to the stop, the hold
# itself ends when ego stops closing. Replay 47c 14:30: a stopped lead crept to 0.7 m/s 9.9 m ahead
# (aLeadK > 0.3 for 1.1 s) and the release at 4.1 m/s moved the stop 3.68 -> 3.53 m; 10:44 released at 98 m.
APPROACH_PULL_HOLD_DISTANCE_MIN = 30.0
# Hold on the same track until ego no longer closes or the lead speeds up again.
APPROACH_HOLD_CLOSING_MIN = 1.0
APPROACH_HOLD_LEAD_SPEED_MARGIN = 2.0
APPROACH_RAMP_TIME = 1.0
# Same-track exit fades out; lead change, lead loss, driver input or control off drop it at once.
APPROACH_RELEASE_TIME = 0.5
# Decel frozen at entry: 1.1 x the decel that stops ego at the stop distance behind the lead.
# No entry when that exceeds the maximum: the stop is no longer soft, and a curve capped at 2.0
# only moved the stop closer (replay 446 5:38, entry at 26 m: stop 3.31 -> 3.10 m).
APPROACH_DECEL_GAIN = 1.1
APPROACH_DECEL_MIN = 0.5
APPROACH_DECEL_MAX = 2.0
# Gap cost weight: x(1 + 4 * ramp).
APPROACH_OBSTACLE_COST_GAIN = 4.0
# Stopped-equivalence decel of the TF lead obstacle (long_mpc.COMFORT_BRAKE). A safety-distance lead
# obstacle uses its level's lead decel (5-8 m/s^2); the caller passes that one.
LEAD_STOP_DECEL = 2.5
# A lead whose speed rose this much within APPROACH_LEAD_RISE_TIME pulls away, even when the
# decaying accel extrapolation still ends below APPROACH_LEAD_END_SPEED_MAX. Replay 47c 10:44: a lead
# starting at 124 m (aLeadK still 0.00, then 0.43) stayed "stops" for 1.45 s, -1.48 m/s^2 of request.
APPROACH_LEAD_RISE = 0.3
APPROACH_LEAD_RISE_TIME = 1.0
# While held, the closing terms of the comfort target give way to the approach curve: share removed at
# full weight, from the second planned ego speed (m/s) up, none at or below the first. They kept the gap
# at 20 km/h near 20 m behind a stopped lead (15 m before the safety-distance gap). Below ~11 km/h they
# stay: without them the plan stopped at -0.74 m/s^2 over the last 2 s instead of -0.42.
APPROACH_CLOSING_RELIEF = 1.0
APPROACH_CLOSING_RELIEF_SPEEDS = (3.0, 6.0)


def approach_decel_request(*, distance, lead_speed, v_ego, stop_distance, lead_stop_decel=LEAD_STOP_DECEL):
  stop_room = max(distance + lead_speed ** 2 / (2 * lead_stop_decel) - stop_distance, 1.0)
  return APPROACH_DECEL_GAIN * v_ego ** 2 / (2 * stop_room)


def approach_decel(*, distance, lead_speed, v_ego, stop_distance, lead_stop_decel=LEAD_STOP_DECEL):
  request = approach_decel_request(distance=distance, lead_speed=lead_speed, v_ego=v_ego, stop_distance=stop_distance,
                                   lead_stop_decel=lead_stop_decel)
  return float(np.clip(request, APPROACH_DECEL_MIN, APPROACH_DECEL_MAX))


class ApproachState:
  def __init__(self):
    self.reset()

  def reset(self):
    self.track_id = None
    self.active = False
    self.decel = APPROACH_DECEL_MAX
    self.weight = 0.0
    self.lead_speeds = []  # (age s, speed) on the current track, newest last
    return self.weight

  def _lead_rising(self, track_id, lead_speed, dt):
    if track_id != self.track_id:
      self.lead_speeds = []
    self.lead_speeds = [(age + dt, v) for age, v in self.lead_speeds if age + dt <= APPROACH_LEAD_RISE_TIME]
    self.lead_speeds.append((0.0, lead_speed))
    return lead_speed - min(v for _, v in self.lead_speeds) >= APPROACH_LEAD_RISE

  def update(self, *, enabled, lead_status, track_id, distance, lead_speed, lead_end_speed, lead_accel,
             v_ego, stop_distance, dt, lead_stop_decel=LEAD_STOP_DECEL):
    values = (distance, lead_speed, lead_end_speed, lead_accel, v_ego, stop_distance, dt, lead_stop_decel)
    if not enabled or not lead_status or not all(map(math.isfinite, values)) or distance <= 0.0 or lead_stop_decel <= 0.0:
      return self.reset()
    closing = v_ego - lead_speed
    rising = self._lead_rising(track_id, lead_speed, dt) and lead_accel > 0.0
    stops = lead_end_speed <= APPROACH_LEAD_END_SPEED_MAX and not rising
    not_pulling_away = stops or (lead_accel <= APPROACH_LEAD_ACCEL_MAX and not rising)
    same_track = track_id == self.track_id
    if self.active and same_track:
      self.active = bool(closing > APPROACH_HOLD_CLOSING_MIN
                         and (not_pulling_away or distance < APPROACH_PULL_HOLD_DISTANCE_MIN) and (
        lead_speed <= APPROACH_LEAD_SPEED_MAX + APPROACH_HOLD_LEAD_SPEED_MARGIN or stops))
    else:
      if not same_track:
        self.weight = 0.0
      self.track_id = track_id
      slow = lead_speed <= APPROACH_LEAD_SPEED_MAX or stops
      request = approach_decel_request(distance=distance, lead_speed=lead_speed, v_ego=v_ego, stop_distance=stop_distance,
                                       lead_stop_decel=lead_stop_decel)
      self.active = bool(slow and not_pulling_away and closing >= APPROACH_CLOSING_MIN
                         and distance / max(v_ego, 1.0) >= APPROACH_TIME_GAP_MIN and request <= APPROACH_DECEL_MAX)
      if self.active:
        self.decel = float(max(request, APPROACH_DECEL_MIN))
    if self.active:
      self.weight = min(1.0, self.weight + dt / APPROACH_RAMP_TIME)
    else:
      weight = self.weight - dt / APPROACH_RELEASE_TIME
      self.weight = weight if weight > 1e-6 else 0.0   # no float residue left on
    return self.weight

  def obstacle_cost_factor(self):
    return 1.0 + APPROACH_OBSTACLE_COST_GAIN * self.weight

  def closing_scale(self, ego_speeds):
    """Share of the comfort target's closing terms kept per node while the approach curve leads."""
    return 1.0 - APPROACH_CLOSING_RELIEF * self.weight * np.interp(ego_speeds, APPROACH_CLOSING_RELIEF_SPEEDS, (0.0, 1.0))

  def reference_margins(self, *, ego_speeds, following_distances, stop_distance):
    """Extra following distance (m) per node; reference only, never an obstacle."""
    if self.weight <= 0.0:
      return np.zeros_like(ego_speeds)
    target = stop_distance + np.maximum(ego_speeds, 0.0) ** 2 / (2 * self.decel)
    return self.weight * np.maximum(0.0, target - following_distances)
