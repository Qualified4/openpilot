"""Limousine Stop: low-speed deceleration relief curve with brake-lamp hold floor and collision safety guard.

Empirical basis (route 00000486--0b5f8c8e99, Hyundai Sonata 2024 CAN-FD):
- Physical brake lamp (0x412 byte 5 0x0c / TCS.BrakeLight) holds ON at 100.0%
  (1,016/1,016 low-speed frames, 0 OFF transitions) when aReqValue <= -0.40 m/s^2.
- First OFF transition occurs at aReqValue = -0.35 m/s^2.
- Holding a minimum deceleration floor of -0.42 m/s^2 keeps the brake lamp lit
  while smoothly unloading suspension compression before the 0.7 m/s StopReq=1 handover.
"""
import math

# Empirically-verified minimum hydraulic deceleration floor to keep physical brake lights ON.
BRAKE_LAMP_HOLD_ACCEL = -0.42  # m/s^2 (100% hold <= -0.40 m/s^2 + 0.02 margin on flat ground)
GRAVITY = 9.81  # m/s^2
MAX_UPHILL_SLOPE_COMP = 0.80  # m/s^2 (~8.2% grade cap on uphill hydraulic floor shift)

# Low-speed ease-out curve speed boundaries.
V_EASE_START = 3.5  # m/s (~12.6 km/h): begin easing strong braking
V_EASE_END = 0.7  # m/s (~2.5 km/h): matches CAN-FD StopReq ENTRY_SPEED
V_MIN_ACTIVE = 0.5  # m/s: below this, StopReq=1 / standstill hold owns final stop

# Deceleration envelope upper magnitude at V_EASE_START.
MAX_APPROACH_DECEL = 1.8  # m/s^2

# Smooth brake-release jerk cap to avoid unloading shock.
MAX_RELEASE_JERK = 1.2  # m/s^3

# Stop-distance tolerance & hard collision-avoidance safety floor.
DEFAULT_STOP_DISTANCE = 4.0  # m
STOP_DISTANCE_TOLERANCE = 1.0  # m: allowed forward tolerance during limousine stop
MIN_SAFE_STOP_DISTANCE = 2.5  # m: hard minimum clearance from lead vehicle
SAFETY_OVERRIDE_MARGIN = 0.3  # m: immediate bypass if d_rel <= MIN_SAFE_STOP_DISTANCE + margin

# Do not soften emergency/hard braking or fast-moving leads.
EMERGENCY_DECEL_THRESHOLD = -2.5  # m/s^2
LEAD_STOPPING_SPEED = 1.5  # m/s


def smoothstep(x: float) -> float:
  """C1-continuous cubic Hermite smoothstep on [0, 1] with zero endpoint slope."""
  t = min(1.0, max(0.0, float(x)))
  return t * t * (3.0 - 2.0 * t)


def brake_lamp_hold_floor(pitch_rad: float = 0.0) -> float:
  """Slope-compensated minimum net deceleration (m/s^2) that keeps hydraulic braking <= -0.42 m/s^2.

  On an uphill (pitch_rad > 0), gravity already supplies -g*sin(pitch) of net deceleration,
  so the ESC/IBS ECU reduces hydraulic pressure by g*sin(pitch). Shifting the floor downward
  by g*sin(pitch) keeps hydraulic pressure at -0.42 m/s^2 (preserving 0x412 brake lights and
  preventing zero-pressure rollback/lash). Downhills (pitch_rad < 0) retain -0.42 m/s^2.
  """
  if not math.isfinite(pitch_rad) or pitch_rad <= 0.0:
    return BRAKE_LAMP_HOLD_ACCEL
  uphill_g = min(MAX_UPHILL_SLOPE_COMP, GRAVITY * math.sin(pitch_rad))
  return BRAKE_LAMP_HOLD_ACCEL - uphill_g


def limousine_decel_envelope(v_ego: float, pitch_rad: float = 0.0) -> float:
  """Speed-dependent signed deceleration envelope (m/s^2) tapering to slope-compensated brake-lamp floor."""
  floor = brake_lamp_hold_floor(pitch_rad)
  max_decel = max(MAX_APPROACH_DECEL, abs(floor))
  if v_ego <= V_EASE_END:
    return floor
  if v_ego >= V_EASE_START:
    return -max_decel
  s = (v_ego - V_EASE_END) / (V_EASE_START - V_EASE_END)
  ease = smoothstep(s)
  return floor - ease * (max_decel - abs(floor))


class LimousineStopController:
  """Stateful jerk-limited low-speed deceleration relief for shock-free stopping.

  Invariants:
  - Never strengthens braking beyond upstream a_cmd (non-increasing decel peak).
  - Never touches positive/zero acceleration commands (non-increasing accel peak).
  - Never relaxes weaker than BRAKE_LAMP_HOLD_ACCEL (-0.42 m/s^2) when active.
  - Immediately yields to upstream a_cmd if collision safety floor is threatened.
  """

  def __init__(self):
    self.active = False
    self.safety_override = False
    self._last_accel = 0.0

  def reset(self):
    self.active = False
    self.safety_override = False
    self._last_accel = 0.0

  def update(
    self,
    *,
    a_cmd: float,
    previous_accel: float,
    v_ego: float,
    dt: float,
    enabled: bool,
    should_stop: bool,
    in_stopping_state: bool,
    fcw: bool = False,
    v_target_now: float = 0.0,
    has_lead: bool = False,
    d_rel: float = 100.0,
    v_lead: float = 0.0,
    stop_distance: float = DEFAULT_STOP_DISTANCE,
    pitch_rad: float = 0.0,
  ) -> float:
    """Return possibly-softened longitudinal acceleration command."""
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

    lamp_floor = brake_lamp_hold_floor(pitch_rad)

    # Never touch acceleration or braking already gentler than the slope-compensated brake-lamp floor.
    if a_cmd >= lamp_floor or v_ego > V_EASE_START or v_ego <= V_MIN_ACTIVE:
      self.reset()
      return a_cmd

    # Never weaken emergency/hard braking.
    if a_cmd < EMERGENCY_DECEL_THRESHOLD:
      self.active = False
      self.safety_override = True
      self._last_accel = a_cmd
      return a_cmd

    # Require genuine stopping context.
    stopping_context = (
      should_stop
      or in_stopping_state
      or (has_lead and math.isfinite(v_lead) and v_lead <= LEAD_STOPPING_SPEED)
      or (math.isfinite(v_target_now) and v_target_now <= 0.5)
    )
    if not stopping_context:
      self.reset()
      return a_cmd

    a_curve = limousine_decel_envelope(v_ego, pitch_rad)
    a_target = a_curve

    if has_lead:
      if not math.isfinite(d_rel) or not math.isfinite(v_lead):
        self.active = False
        self.safety_override = True
        self._last_accel = a_cmd
        return a_cmd

      v_closing = max(0.0, v_ego - max(0.0, v_lead))
      d_hard = d_rel - MIN_SAFE_STOP_DISTANCE

      # Hard collision-safety floor: immediately yield to full upstream braking.
      if d_rel <= MIN_SAFE_STOP_DISTANCE + SAFETY_OVERRIDE_MARGIN or d_hard <= 0.2:
        self.active = False
        self.safety_override = True
        self._last_accel = a_cmd
        return a_cmd

      a_req_hard = -(v_closing * v_closing) / (2.0 * d_hard)
      if a_req_hard <= a_cmd:
        self.active = False
        self.safety_override = True
        self._last_accel = a_cmd
        return a_cmd

      # Tolerance-aware kinematic limit: allow stopping up to STOP_DISTANCE_TOLERANCE
      # closer than nominal stop_distance, never closer than MIN_SAFE_STOP_DISTANCE.
      nominal_stop = stop_distance if (math.isfinite(stop_distance) and stop_distance > 0.0) else DEFAULT_STOP_DISTANCE
      tol_stop_dist = max(MIN_SAFE_STOP_DISTANCE, nominal_stop - STOP_DISTANCE_TOLERANCE)
      d_tol = d_rel - tol_stop_dist
      if d_tol <= 0.2:
        a_req_tol = a_req_hard
      else:
        a_req_tol = -(v_closing * v_closing) / (2.0 * d_tol)

      # Never relax weaker than what kinematics require to stop within tolerance.
      a_target = min(a_curve, a_req_tol)

    self.safety_override = False

    # Enforce invariants: never weaker than slope-compensated brake-lamp floor, never stronger than a_cmd.
    a_target = min(lamp_floor, a_target)
    if a_target < a_cmd - 1e-6 or (not self.active and a_target <= a_cmd):
      self.active = False
      self._last_accel = a_cmd
      return a_cmd

    base_prev = self._last_accel if self.active else min(previous_accel, a_cmd)
    # Limit only upward release jerk; allow immediate strengthening if needed.
    a_out = min(a_target, base_prev + MAX_RELEASE_JERK * dt)
    a_out = max(a_cmd, min(lamp_floor, a_out))

    self.active = True
    self._last_accel = a_out
    return a_out
