"""CCNC parking-lot lane hide (display only): hide both cluster lane lines in unstructured lots.

Entry evidence is *distance driven slowly with no lane lines and uncertain road edges*, not time:
  both laneLineProbs[1,2] < PROB_ON, both roadEdgeStds[0,1] > EDGE_STD_ON, MIN_SPEED <= vEgo <= MAX_SPEED.
Standing still neither adds nor resets evidence, so a red-light queue (lanes often invisible, edges uncertain
while stopped) never enters; driving with lane lines visible resets it. Entry needs ENTRY_M of that travel.
Log evidence and threshold history: docs/ccnc_parking_lane_hide_20261019.md.
Lane change guard: while one blinker is on (hazards excluded, as in the CCNC animation), the lane change state is not off or the model desire is a lane change,
and for GRACE_S afterwards, evidence is cleared and cannot build. A confirmed hide is NOT released by a blinker.
Release (hysteresis): lane lines visible (max prob >= PROB_OFF) for EXIT_M of travel, or vEgo >= EXIT_SPEED
for EXIT_SPEED_S, or no model for NO_MODEL_S.
No lane-change permission, DesireHelper, steering or modeld input is read for anything but this display state.
"""
PROB_ON, EDGE_STD_ON = .2, 1.5
MIN_SPEED, MAX_SPEED = .5, 4.5          # m/s
# DaBin 10-27: 10 -> 5 m. 42b/22 47.6 s felt late (203 m driven, longest continuous qualifying run 28.8 m, 22 resets).
ENTRY_M = 5.
PROB_OFF, EXIT_M = .5, 10.
EXIT_SPEED, EXIT_SPEED_S = 24. / 3.6, 1.
GRACE_S, NO_MODEL_S, MAX_DT = 3., 1., .2
HIDE_CODE = 14  # hides the lane line on the CCNC cluster, confirmed on the car for both LEFT and RIGHT


class ParkingLaneHide:
  def __init__(self):
    self.reset()

  def reset(self):
    self.hidden, self.evidence_m, self.exit_m, self.fast_s = False, 0., 0., 0.
    self.last_t, self.guard_until, self.no_model_s = None, -1., 0.
    self.entered_at = None

  def update(self, t, v_ego, md, left_blinker, right_blinker, desire=0):
    """t: monotonic seconds. Returns True while both lane lines should be hidden."""
    dt = 0. if self.last_t is None else min(max(t - self.last_t, 0.), MAX_DT)
    self.last_t = t
    probs, stds = getattr(md, 'laneLineProbs', ()), getattr(md, 'roadEdgeStds', ())
    if md is None or len(probs) < 3 or len(stds) < 2:
      self.no_model_s += dt
      if self.no_model_s >= NO_MODEL_S:
        self.hidden, self.evidence_m = False, 0.
      return self.hidden
    self.no_model_s = 0.
    p_max, std_min = max(probs[1], probs[2]), min(stds[0], stds[1])
    # Like the CCNC lane-change animation (update_lanes: leftBlinker != rightBlinker): hazard lights, often used
    # while manoeuvring in a lot, are not a lane change.
    changing = (bool(left_blinker) != bool(right_blinker) or desire in (3, 4) or
                str(getattr(getattr(md, 'meta', None), 'laneChangeState', 'off')) != 'off')
    if changing:
      self.guard_until = t + GRACE_S
    moving = v_ego >= MIN_SPEED
    if not self.hidden:
      if t < self.guard_until:
        self.evidence_m = 0.
      elif moving:
        lot = p_max < PROB_ON and std_min > EDGE_STD_ON and v_ego <= MAX_SPEED
        self.evidence_m = self.evidence_m + v_ego*dt if lot else 0.
        if self.evidence_m >= ENTRY_M:
          self.hidden, self.entered_at, self.exit_m, self.fast_s = True, t, 0., 0.
      return self.hidden
    # Hidden: only road evidence releases it (a blinker alone does not).
    self.fast_s = self.fast_s + dt if v_ego >= EXIT_SPEED else 0.
    if moving:
      self.exit_m = self.exit_m + v_ego*dt if p_max >= PROB_OFF else 0.
    if self.exit_m >= EXIT_M or self.fast_s >= EXIT_SPEED_S:
      self.hidden, self.evidence_m = False, 0.
    return self.hidden


def apply(values, hidden):
  """Highest display priority: hide both lane lines (code 14) while in a parking lot."""
  if hidden:
    values["LANELINE_LEFT"] = HIDE_CODE
    values["LANELINE_RIGHT"] = HIDE_CODE
  return values
