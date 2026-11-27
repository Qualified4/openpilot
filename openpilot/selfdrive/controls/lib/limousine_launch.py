"""Limousine Launch: S-curve initial acceleration ramp for shock-free departures and low-speed re-acceleration.

Empirical basis (routes 00000486--0b5f8c8e99 & 00000480--6fd921a4c9--7, Hyundai Sonata 2024 CAN-FD):
- Standstill launches (n=7): on StopReq 1->0 (stopping -> pid), ccAccel jumps +0.70..+1.28 m/s^2
  in a single tick, then winds up to +0.85..+1.74 m/s^2 during the ~0.65-0.85 s hydraulic brake-hold
  release deadtime, causing a slingshot kick (measured positive jerk 5.24..12.40 m/s^3, mean 9.72 m/s^3).
- Low-speed decel->accel re-accelerations (n=12): aReqValue ramps at 2.0..4.1 m/s^3 (up to 14.66 m/s^3),
  whipping suspension pitch from braking to driving torque (measured positive jerk up to 10.80 m/s^3).

Design invariants:
- Never increases acceleration peak above upstream a_cmd (a_out <= a_cmd for all a_cmd > 0).
- Immediately bypasses and resets on any braking (a_cmd <= 0), should_stop, FCW, or driver pedal input.
- Strictly disjoint from LimousineStopController (operates only on a_cmd > 0; stop operates only on a_cmd < -0.42).
- Uphill gravity compensation (+g*sin(pitch), capped at 0.80 m/s^2) prevents rollback and preserves
  consistent net launch acceleration on grades.
- Zero new user Params toggles.
"""
import math

from openpilot.selfdrive.controls.lib.limousine_stop import GRAVITY, MAX_UPHILL_SLOPE_COMP, smoothstep

# Standstill launch regime (StopReq 1->0 / stopping -> starting|pid)
V_STANDSTILL_TRIGGER = 0.35  # m/s: speed threshold for standstill launch mode
V_BREAKAWAY = 0.04  # m/s: first wheel-motion threshold (zero artificial delay once moving)
BREAKAWAY_ACCEL_FLAT = 0.70  # m/s^2: crisp breakaway ceiling that still eliminates slingshot kick
BREAKAWAY_RAMP_TIME = 0.38  # s: short S-curve ramp from 0 -> breakaway ceiling while stationary

# Post-breakaway S-curve launch ramp (short 0.45s ramp: shaves 0.1-0.2s kick, minimal Δt loss)
LAUNCH_RAMP_TIME = 0.45  # s: short S-curve ramp duration after breakaway (0.35-0.55s target)
LAUNCH_JERK_MIN = 1.40  # m/s^3: initial soft jerk right after breakaway
LAUNCH_JERK_PEAK = 3.60  # m/s^3: rapid S-curve completion jerk so target accel/speed Δt <= ~0.25s
UPHILL_JERK_BONUS_GAIN = 0.50  # (m/s^3) per (m/s^2) of uphill gravity component

# Low-speed decel -> accel re-acceleration regime (rolling stop / congestion)
V_REACCEL_MAX = 12.0  # m/s (~43 km/h): city/congestion speed ceiling for re-accel shaping
DECEL_MEMORY_ACCEL = -0.10  # m/s^2: recent braking threshold that arms re-accel shaping
DECEL_MEMORY_TIME = 0.60  # s: memory window after braking before crossing into positive accel
REACCEL_RAMP_TIME = 0.45  # s: short S-curve ramp duration for decel -> accel transition
REACCEL_JERK_MIN = 0.85  # m/s^3: initial soft jerk right after crossing a=0
REACCEL_JERK_PEAK = 2.40  # m/s^3: controlled peak jerk to shave initial kick with <=0.15s Δt


def uphill_gravity_comp(pitch_rad: float = 0.0) -> float:
  """Positive uphill gravity acceleration component g*sin(pitch) in [0, MAX_UPHILL_SLOPE_COMP]."""
  if not math.isfinite(pitch_rad) or pitch_rad <= 0.0:
    return 0.0
  return min(MAX_UPHILL_SLOPE_COMP, GRAVITY * math.sin(pitch_rad))


def breakaway_accel_ceiling(pitch_rad: float = 0.0) -> float:
  """Maximum stationary breakaway acceleration (m/s^2) to release brake hold without slingshot."""
  return BREAKAWAY_ACCEL_FLAT + uphill_gravity_comp(pitch_rad)


def launch_jerk_limit(elapsed_s: float, *, mode: str, pitch_rad: float = 0.0) -> float:
  """C1-smooth S-curve positive jerk ceiling (m/s^3) as a function of elapsed ramp time."""
  uphill_bonus = UPHILL_JERK_BONUS_GAIN * uphill_gravity_comp(pitch_rad)
  if mode == "standstill":
    s = smoothstep(elapsed_s / LAUNCH_RAMP_TIME)
    return LAUNCH_JERK_MIN + s * (LAUNCH_JERK_PEAK - LAUNCH_JERK_MIN) + uphill_bonus
  s = smoothstep(elapsed_s / REACCEL_RAMP_TIME)
  return REACCEL_JERK_MIN + s * (REACCEL_JERK_PEAK - REACCEL_JERK_MIN) + uphill_bonus


class LimousineLaunchController:
  """Stateful S-curve acceleration shaper for standstill departures and low-speed re-accelerations."""

  def __init__(self):
    self.active = False
    self.mode = "idle"  # "idle" | "standstill" | "reaccel"
    self.shaped_jerk: float | None = None
    self._last_accel = 0.0
    self._stationary_t = 0.0
    self._ramp_t = 0.0
    self._broken_away = False
    self._was_stopping = False
    self._decel_timer = 0.0

  def reset(self):
    self.active = False
    self.mode = "idle"
    self.shaped_jerk = None
    self._last_accel = 0.0
    self._stationary_t = 0.0
    self._ramp_t = 0.0
    self._broken_away = False
    self._was_stopping = False
    self._decel_timer = 0.0

  def notify_stopping(self):
    """Record that LongControl is in the stopping state so the next release arms standstill launch."""
    self.active = False
    self.mode = "idle"
    self.shaped_jerk = None
    self._last_accel = 0.0
    self._stationary_t = 0.0
    self._ramp_t = 0.0
    self._broken_away = False
    self._was_stopping = True
    self._decel_timer = DECEL_MEMORY_TIME

  def _disengage(self):
    self.active = False
    self.mode = "idle"
    self.shaped_jerk = None
    self._stationary_t = 0.0
    self._ramp_t = 0.0
    self._broken_away = False

  def update(
    self,
    *,
    a_cmd: float,
    previous_accel: float,
    v_ego: float,
    dt: float,
    enabled: bool,
    should_stop: bool = False,
    fcw: bool = False,
    cruise_standstill: bool = False,
    in_starting_state: bool = False,
    pitch_rad: float = 0.0,
  ) -> float:
    """Return possibly S-curve-shaped positive longitudinal acceleration command."""
    if (
      not enabled
      or fcw
      or not math.isfinite(a_cmd)
      or not math.isfinite(previous_accel)
      or not math.isfinite(v_ego)
      or dt <= 0.0
    ):
      self.reset()
      return a_cmd

    if should_stop or cruise_standstill:
      if v_ego <= V_STANDSTILL_TRIGGER or cruise_standstill:
        self.notify_stopping()
      else:
        self._disengage()
        self._decel_timer = DECEL_MEMORY_TIME
      return a_cmd

    # Immediate bypass on braking or zero accel: never delay deceleration.
    if a_cmd <= 0.0:
      self._disengage()
      if a_cmd <= DECEL_MEMORY_ACCEL or previous_accel <= DECEL_MEMORY_ACCEL:
        self._decel_timer = DECEL_MEMORY_TIME
      else:
        self._decel_timer = max(0.0, self._decel_timer - dt)
      self._last_accel = a_cmd
      return a_cmd

    # Positive acceleration requested (a_cmd > 0.0).
    if not self.active:
      if self._was_stopping or in_starting_state or (v_ego <= V_STANDSTILL_TRIGGER and previous_accel <= 0.0):
        self.active = True
        self.mode = "standstill"
        self._last_accel = max(0.0, previous_accel)
        self._stationary_t = 0.0
        self._ramp_t = 0.0
        self._broken_away = v_ego > V_BREAKAWAY
        self._was_stopping = False
        self._decel_timer = 0.0
      elif v_ego <= V_REACCEL_MAX and (self._decel_timer > 0.0 or previous_accel <= DECEL_MEMORY_ACCEL):
        self.active = True
        self.mode = "reaccel"
        self._last_accel = max(0.0, previous_accel)
        self._stationary_t = 0.0
        self._ramp_t = 0.0
        self._broken_away = True
        self._was_stopping = False
        self._decel_timer = 0.0
      else:
        self._decel_timer = max(0.0, self._decel_timer - dt)
        self._last_accel = a_cmd
        return a_cmd

    # Active S-curve shaping.
    if self.mode == "standstill" and not self._broken_away:
      if v_ego > V_BREAKAWAY:
        self._broken_away = True
      else:
        self._stationary_t += dt
        s = smoothstep(self._stationary_t / BREAKAWAY_RAMP_TIME)
        a_ceil = s * breakaway_accel_ceiling(pitch_rad)
        # Never exceed upstream a_cmd; smoothly ramp from 0 to breakaway ceiling.
        a_out = min(a_cmd, a_ceil)
        self.shaped_jerk = LAUNCH_JERK_MIN + UPHILL_JERK_BONUS_GAIN * uphill_gravity_comp(pitch_rad)
        self._last_accel = a_out
        return a_out

    # Post-breakaway standstill ramp or low-speed re-acceleration ramp.
    self._ramp_t += dt
    ramp_duration = LAUNCH_RAMP_TIME if self.mode == "standstill" else REACCEL_RAMP_TIME
    j_max = launch_jerk_limit(self._ramp_t, mode=self.mode, pitch_rad=pitch_rad)
    a_upper = self._last_accel + j_max * dt
    a_out = min(a_cmd, a_upper)

    # Disengage cleanly once ramp completes and output has caught up to upstream command,
    # or once cruising speed exceeds the low-speed shaping window.
    if (self._ramp_t >= ramp_duration and a_out >= a_cmd - 1e-3) or v_ego > V_REACCEL_MAX:
      self._disengage()
      self._last_accel = a_cmd
      return a_cmd

    self.shaped_jerk = j_max
    self._last_accel = a_out
    return a_out
