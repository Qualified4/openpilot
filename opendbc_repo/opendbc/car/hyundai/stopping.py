"""Hyundai CAN FD: zero aReq with StopReq, without automatic release/retry.

Thresholds below are experimental, not OEM acceptance conditions. This controller
cannot guarantee stopping; ECU response must be measured on the vehicle.
"""
from dataclasses import dataclass
from enum import StrEnum
import math


DT = 0.02  # SCC_CONTROL is transmitted at 50 Hz
ENTRY_SPEED = 0.7  # m/s; retain approach braking above the low-speed stop region
STOP_SPEED = 0.05
MOVING_SPEED = 0.10
STOP_CONFIRM_TIME = 0.2
APPROACH_ACCEL = -0.45  # matches empirically-verified 0x412 brake-lamp hold floor (-0.45 m/s^2)
GRAVITY = 9.81  # m/s^2
MAX_UPHILL_SLOPE_COMP = 0.80  # m/s^2 (~8.2% grade cap matching unified_long.py)
STOP_LOWER_BAND = 0.20  # fixed during active ACC, including before StopReq; never copied from stock
PRECHARGE_MIN_SPEED = 0.15  # m/s; below this speed, enter StopReq immediately (covers queue_crawl)
PRECHARGE_HYD_THRESH = -0.44  # m/s^2; hydraulic decel required before non-emergency StopReq (0x412 lamp)
PRECHARGE_MAX_TIME = 0.26  # s (13 frames at 50 Hz; at 3.0 m/s^3 covers up to +3.5% uphill) max pre-charge dwell
PRECHARGE_CLOSE_LEAD_DIST = 2.80  # m; bypass pre-charge dwell for close-lead safety


def uphill_gravity_comp(pitch_rad: float = 0.0) -> float:
  """Uphill gravity deceleration component g*sin(pitch) capped at MAX_UPHILL_SLOPE_COMP."""
  if not math.isfinite(pitch_rad) or pitch_rad <= 0.0:
    return 0.0
  return min(MAX_UPHILL_SLOPE_COMP, GRAVITY * math.sin(pitch_rad))


def approach_accel_floor(pitch_rad: float = 0.0) -> float:
  """Slope-compensated approach deceleration floor keeping uphill hydraulic braking <= APPROACH_ACCEL."""
  return APPROACH_ACCEL - uphill_gravity_comp(pitch_rad)


class StopPhase(StrEnum):
  idle = "idle"
  approach = "approach"
  request = "request"
  held = "held"


@dataclass(frozen=True)
class StopCommand:
  stop_req: int
  raw: float
  value: float
  lower: float


class CanfdStopping:
  def __init__(self):
    # Packet history survives episode resets; it follows the final StopReq,
    # including approach and interlock overrides.
    self.last_scc_stop_req = 0
    self.last_scc_jerk_upper = 1.0
    self.reset()

  def limit_scc_jerk_upper(self, stop_req: int, jerk_upper: float) -> float:
    """Separate an Upper increase from the first StopReq frame, not its onset."""
    if stop_req == 1 and self.last_scc_stop_req != 1:
      jerk_upper = min(jerk_upper, self.last_scc_jerk_upper)
    self.last_scc_stop_req = stop_req
    self.last_scc_jerk_upper = jerk_upper
    return jerk_upper

  def reset(self):
    self.stop_object_active = False
    self.phase = StopPhase.idle
    self.reason = "inactive"
    self.stopped_time = 0.0
    self.precharge_time = 0.0

  def enter(self, phase: StopPhase, reason: str):
    self.phase = phase
    self.reason = reason

  def update(self, *, active: bool, requested: bool, speed: float, held: bool,
             accel: float, previous_value: float, jerk_u: float, jerk_l: float,
             pitch_rad: float = 0.0, soft_hold: bool = False, close_lead: bool = False) -> StopCommand | None:
    # Caller validates sensor values and applies pedal/CAN/hold interlocks.
    if not active or not requested:
      self.reset()
      return None

    a_hyd_prev = previous_value + uphill_gravity_comp(pitch_rad)
    needs_precharge = (
      PRECHARGE_MIN_SPEED < speed <= ENTRY_SPEED
      and not soft_hold
      and not close_lead
      and a_hyd_prev > PRECHARGE_HYD_THRESH
    )

    if self.phase == StopPhase.idle:
      if speed > ENTRY_SPEED:
        self.enter(StopPhase.approach, "stop_requested")
      elif needs_precharge:
        self.precharge_time = DT
        self.enter(StopPhase.approach, "precharge_guard")
      else:
        self.enter(StopPhase.request, "stop_requested")
    elif self.phase == StopPhase.approach and speed <= ENTRY_SPEED:
      if needs_precharge and self.precharge_time < PRECHARGE_MAX_TIME - 1e-9:
        self.precharge_time += DT
        self.reason = "precharge_guard"
      else:
        self.enter(StopPhase.request, "precharge_complete" if self.reason == "precharge_guard" else "entry_speed")

    self.stopped_time = self.stopped_time + DT if speed <= STOP_SPEED else 0.0
    stopped = (held and speed <= MOVING_SPEED) or self.stopped_time >= STOP_CONFIRM_TIME
    if stopped:
      if self.phase != StopPhase.held:
        self.enter(StopPhase.held, "stop_observed")
    elif self.phase == StopPhase.held:
      # Motion changes diagnostics only. Never release StopReq to retry, even
      # when speed rises above the initial entry threshold after a request.
      self.enter(StopPhase.request, "motion_after_hold")

    if self.phase in (StopPhase.request, StopPhase.held):
      # Match the observed Ioniq stock entry: both aReq fields become zero in
      # the FIRST StopReq frame, bypassing the ordinary aReqValue jerk ramp.
      return StopCommand(1, 0.0, 0.0, STOP_LOWER_BAND)

    # Preserve the existing approach deceleration (with uphill slope compensation) until low-speed entry.
    raw = min(accel, approach_accel_floor(pitch_rad))
    value = min(0.0, max(previous_value - jerk_l * DT, min(raw, previous_value + jerk_u * DT)))
    return StopCommand(0, raw, value, STOP_LOWER_BAND)
