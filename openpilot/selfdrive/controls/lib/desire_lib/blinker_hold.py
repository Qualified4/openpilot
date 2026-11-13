"""CanfdBlinkerHoldChange / CanfdBlinkerHoldWait: keep the lamp of the driver's own lane change request lit.

DesireHelper decides here and publishes the side in modelV2.meta.laneChangeBlinkerHold; the Hyundai CAN-FD
carcontroller lights it through SPAS2 (aprk_blinkers, same path and send mode as CanfdBlinkerKick).

- wait (CanfdBlinkerHoldWait): the driver asked, but the change waits in preLaneChange (BSD, side object, lines...).
  The body controller stops after its one-touch flashes (3, ~2.4 s) and the request would drop with the lamp; we keep it
  lit, at most WAIT_MAX_S. The held lamp is read as the driver still asking: that is what keeps the request alive.
- change (CanfdBlinkerHoldChange): the change runs (laneChangeStarting). Held until the car centre is 60 % into the new
  lane (same rule as the HDA2 ADRV hold, hyundaicanfd._blink_hold_change), laneChangeFinishing, or CHANGE_MAX_S.

Only for a request the driver's own lamp made (driver_enabled when the hold starts): never for changes started by the
model, remote, app, voice or navigation (that is CanfdBlinkerKickModel), and never with the lever latched.

Lamp feedback (EV4 10-26: a lamp held through the change never went off; measurements 31: a kick lamp started a change).
carState.left/rightBlinker are the lamp, and on the Sonata 2024 nothing else: its PT bus has no BLINKER_STALKS (the 0x3C1
on bus 1 is a different 24-byte frame; blinkerLever and the stalk counters stay 0 in the 11-12 logs), so our lamp and the
driver's cannot be told apart. So every release (except a latched lever, which is the driver's) owns the side through
BoundaryIndicator.own() with a short isolation: DesireHelper ignores that side's lamp until it has been dark for 1 s
(the body controller runs on ~1.2 s after the command stops), so the tail can neither start a second change nor re-arm
the wait. If the lamp is still lit MASK_MAX_S after the release it is not our tail but the driver's own lamp: the mask is
dropped. That is the only time the driver's lamp is ignored, from the release until dark for 1 s (at most MASK_MAX_S).
"""
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_lib.constants import (
  BLINKER_NONE, BLINKER_LEFT, BLINKER_RIGHT, LANE_CHANGE_TIME_MAX, LaneChangeState,
)

# DesireHelper itself gives up a running change after LANE_CHANGE_TIME_MAX (10 s): a hold never outlives it.
CHANGE_MAX_S = LANE_CHANGE_TIME_MAX
# A one-touch request flashes ~2.4 s, LaneChangeLeverWait waits 5 flashes (4 s) and an App request 10 s. 15 s lets a car
# in the blind spot pass at a 10-20 km/h speed difference (~40 m), then the request is dropped like a lever one.
WAIT_MAX_S = 15.0
# own(): the mask itself also waits for the lamp to be dark 1 s (BoundaryIndicator.update), so 1 s isolation is enough.
RELEASE_ISOLATE_S = 1.0
# The body controller runs on ~1.2 s after the command stops (EV4 10-26) and carState holds a lamp 0.5 s more: a lamp
# still lit 5 s after the release is the driver's own.
MASK_MAX_S = 5.0

# 60 % release: same rule and numbers as hyundaicanfd._blink_hold_change (6cfeeb999, d8ba6e75a)
CROSS_NEAR_M = 1.0
CROSS_JUMP_M = 1.5
RELEASE_PROGRESS = 0.6

LEVER_LATCHED = 2


def across(model, left: bool, track: dict) -> bool:
  """True once the car centre is RELEASE_PROGRESS of a lane width into the new lane (model lane lines)."""
  lines = getattr(model, "laneLines", None)
  try:
    if lines is None or len(lines) < 3 or not len(lines[1].y) or not len(lines[2].y):
      return False
    target, other = (1, 2) if left else (2, 1)
    y_target, y_other = abs(float(lines[target].y[0])), abs(float(lines[other].y[0]))
  except (AttributeError, IndexError, TypeError, ValueError):
    return False
  y_min = y_target if track.get("y_min") is None else min(track["y_min"], y_target)
  track["y_min"] = y_min
  if not track.get("crossed") and y_min < CROSS_NEAR_M and y_target - y_min > CROSS_JUMP_M:
    track["crossed"] = True
  return bool(track.get("crossed")) and y_other >= (RELEASE_PROGRESS - 0.5) * (y_target + y_other)


class BlinkerHold:
  def __init__(self):
    self.side = BLINKER_NONE   # side held now
    self.phase = None          # "wait" or "change"
    self.timer = 0.0
    self.done = False          # this request had its hold: none again until the FSM is off
    self.track = {}
    self.reason = ""           # why the last hold ended (log / tests)
    self.mask = BLINKER_NONE   # side we masked on release and have not seen cleared yet
    self.mask_time = 0.0

  def update(self, now, indicator, *, hold_change, hold_wait, state, driver_enabled, driver_st, blinker_state,
             carstate, model) -> int:
    """After DesireHelper's FSM (state is this frame's). Returns the side to hold."""
    lamps = int(bool(carstate.leftBlinker)) | (int(bool(carstate.rightBlinker)) << 1)
    lever = getattr(carstate, "blinkerLever", 0)
    drive = str(getattr(carstate, "gearShifter", "drive")) == "drive"
    self._expire_mask(now, lamps, indicator)
    if state == LaneChangeState.off:
      self.done = False

    if self.side != BLINKER_NONE:
      reason = self._end_reason(hold_change, hold_wait, state, driver_st, blinker_state, lamps, lever, drive, model)
      if reason:
        self._release(now, indicator, reason)
      else:
        self.timer += DT_MDL

    if (self.side == BLINKER_NONE and not self.done and driver_enabled and blinker_state in (BLINKER_LEFT, BLINKER_RIGHT) and
        driver_st == blinker_state and lever != LEVER_LATCHED and drive and lamps != 3):
      if hold_wait and state == LaneChangeState.preLaneChange:
        self.side, self.phase, self.timer = blinker_state, "wait", 0.0
      elif hold_change and state == LaneChangeState.laneChangeStarting:
        self.side, self.phase, self.timer, self.track = blinker_state, "change", 0.0, {}
    return self.side

  def _end_reason(self, hold_change, hold_wait, state, driver_st, blinker_state, lamps, lever, drive, model):
    if not drive:
      return "gear"
    if lamps == 3:
      return "hazard"
    if lever == LEVER_LATCHED:
      return "latched"
    if driver_st not in (BLINKER_NONE, self.side):
      return "opposite"
    if state == LaneChangeState.off:
      return "cancel"
    if self.phase == "wait":
      if state == LaneChangeState.laneChangeStarting:
        if not hold_change:
          return "started"
        self.phase, self.timer, self.track = "change", 0.0, {}
        return ""
      if not hold_wait:
        return "setting"
      if blinker_state != self.side:
        return "redirect"
      return "timeout" if self.timer >= WAIT_MAX_S else ""
    if not hold_change:
      return "setting"
    if state != LaneChangeState.laneChangeStarting:
      return "finished"
    if self.timer >= CHANGE_MAX_S:
      return "timeout"
    return "across" if across(model, self.side == BLINKER_LEFT, self.track) else ""

  def _release(self, now, indicator, reason):
    if reason != "latched":
      # our lamp's tail must not read back as the driver's request
      self.mask = self.side & ~indicator.blocked
      self.mask_time = now
      indicator.own(now, self.side, isolate=RELEASE_ISOLATE_S)
    self.side, self.phase, self.done, self.reason = BLINKER_NONE, None, True, reason

  def _expire_mask(self, now, lamps, indicator):
    if not self.mask:
      return
    if not indicator.blocked & self.mask:
      self.mask = BLINKER_NONE  # cleared by BoundaryIndicator: dark for 1 s
    elif now - self.mask_time >= MASK_MAX_S and lamps & self.mask:
      indicator.blocked &= ~self.mask  # still lit long after our command: the driver's own lamp
      self.mask = BLINKER_NONE
