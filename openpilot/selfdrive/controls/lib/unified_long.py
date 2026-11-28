"""Unified continuous longitudinal controller (Stage 1 of #253).

Replaces the fragmented LongControl stopping ramp, LimousineStop post-clipper,
and LimousineLaunch post-shaper with a single continuous (a_ref, j_ref)
4-quadrant jerk integrator and first-class physical acceleration envelope.

Design invariants (DESIGN_KO.md):
- Pure IEEE-754 f64 scalar state (no_std / Rust-portable, zero dynamic allocations).
- Zero new user Params and 100% backward-compatible Cereal Actuators fields.
- 0-tick unconditional safety/emergency bypass (FCW, <-2.5 m/s^2, d_rel <= 2.8 m,
  kinematic 2.5 m hard stop guard, driver brake/gas pedals, inactive).
- Kinematic stop-pocket hydraulic pre-charge latch preserving uphill brake-lamp
  hydraulic deceleration (a_hyd <= -0.45 m/s^2) until StopReq=1 or clear departure.
- Zero mode-switch step jumps across decel -> relief -> hold -> breakaway -> S-curve.
- Highway / cruise-coasting pass-through (v_ego >= 14.0 m/s: 0.0% change).
"""
from dataclasses import dataclass
import math


# Physical & safety constants (shared 1:1 with Rust Stage 3 specification)
GRAVITY = 9.81
MAX_UPHILL_SLOPE_COMP = 0.80        # m/s^2 (~8.2% grade cap)
A_HYD_LAMP_FLOOR = -0.45            # m/s^2 flat brake-lamp hydraulic hold floor (0x412)
EMERGENCY_DECEL_THRESHOLD = -2.5    # m/s^2 immediate 0-tick emergency bypass
MIN_SAFE_STOP_DISTANCE = 2.50       # m immutable minimum lead stop clearance
CLOSE_LEAD_OVERRIDE_DISTANCE = 2.80 # m immediate 0-tick close-lead bypass
DEFAULT_STOP_DISTANCE = 4.00        # m default configured lead stop distance
STOP_TOLERANCE_MARGIN = 1.00        # m allowed ease-out tolerance above 2.5 m hard floor
LEAD_EMERGENCY_DECEL = 4.00         # m/s^2 equivalent stopping decel for moving lead

# Stop ease-out & kinematic stop-pocket pre-charge constants
V_EASE_START = 3.50                 # m/s upper speed for low-speed decel relief
V_EASE_END = 0.70                   # m/s lower speed where ease curve reaches a_lamp
V_EASE_MIN = 0.50                   # m/s low-speed handover boundary into final hold
EASE_PEAK_DECEL = 1.80              # m/s^2 anchor decel at V_EASE_START
POCKET_V_MAX = 1.60                 # m/s max speed to arm stop-pocket pre-charge latch
POCKET_HOLD_V_MAX = 2.40            # m/s max speed to preserve already-pressurized brake-lamp hold in pocket
POCKET_MARGIN_ON = 1.20             # m kinematic pocket entry margin above dx_lamp
POCKET_MARGIN_OFF = 2.20            # m kinematic pocket release hysteresis margin
POCKET_LEAD_DEPART_V = 0.85         # m/s min lead speed for relative pull-away departure
POCKET_LEAD_CLEAR_DEPART_V = 1.35   # m/s clear lead departure speed releasing pre-charge unconditionally
POCKET_LEAD_PULLAWAY_DV = 0.25      # m/s lead-ego speed excess confirming lead is actually pulling away
POCKET_DEPART_ACCEL = 0.08          # m/s^2 positive accel demand releasing pre-charge
POCKET_ARM_MARGIN = 0.05            # m/s^2 tolerance near a_lamp to arm pre-charge

# Hold latch & standstill breakaway constants
HOLD_CLOSE_LEAD_DIST = 4.00         # m close-lead stopping handover threshold
HOLD_FLICKER_DECEL = -0.05          # m/s^2 braking demand threshold guarding against 1-tick should_stop flicker
HOLD_FLICKER_V_TARGET = 0.20        # m/s low target speed guarding against 1-tick should_stop flicker
JERK_PRECHARGE_DOWN = 1.20          # m/s^3 smooth pre-charge downward jerk while above brake-lamp floor
V_BREAKAWAY = 0.04                  # m/s wheel motion detection threshold
BREAKAWAY_ACCEL_FLAT = 0.70         # m/s^2 flat stationary breakaway accel cap
BREAKAWAY_RELEASE_BASE = 0.65       # m/s^2 steep-uphill wheel-motion breakaway handoff base
BREAKAWAY_RELEASE_UPHILL_GAIN = 0.80  # steep-uphill gravity share for breakaway preload cap
BREAKAWAY_RAMP_TIME = 0.38          # s smoothstep ramp duration to breakaway cap

# 4-quadrant continuous jerk bounds (m/s^3)
JERK_DECEL_RELIEF_LOW = 1.20        # m/s^3 low-speed stop relief upward jerk (<= 3.5 m/s)
JERK_DECEL_RELIEF_HIGH = 3.50       # m/s^3 mid-speed relief upward jerk (at 6.5 m/s)
JERK_DECEL_RELIEF_V_HIGH = 6.50     # m/s upper speed for decel relief shaping
JERK_UNWIND_TO_POS = 2.50           # m/s^3 decel-to-accel brake unwind rate toward a=0
JERK_REACCEL_MIN = 0.85             # m/s^3 post-zero backlash absorption initial jerk
JERK_REACCEL_PEAK = 2.40            # m/s^3 post-zero reaccel peak jerk
JERK_LAUNCH_DELTA_MIN = 0.55        # m/s^3 standstill launch addition (0.85 + 0.55 = 1.40)
JERK_LAUNCH_DELTA_PEAK = 1.20       # m/s^3 standstill launch addition (2.40 + 1.20 = 3.60)
JERK_UPHILL_GAIN = 0.50             # s^-1 uphill jerk compensation gain * g_up
SCURVE_RAMP_TIME = 0.45             # s S-curve jerk ramp duration
LAUNCH_V_ZERO_LOW = 0.20            # m/s zero-crossing speed for full launch weight
LAUNCH_V_ZERO_SPAN = 0.60           # m/s transition span from launch to rolling reaccel
CITY_V_START = 10.0                 # m/s start blending S-curve out toward highway
CITY_V_END = 14.0                   # m/s full highway pass-through (0% shaping)
JERK_MAX_PHYS = 5.0                 # m/s^3 physical CAN-FD upper jerk limit
REACCEL_ARM_DECEL = -0.10           # m/s^2 braking depth that arms zero-crossing S-curve


def clamp(x: float, lo: float, hi: float) -> float:
  if x < lo:
    return lo
  if x > hi:
    return hi
  return x


def smoothstep(u: float) -> float:
  t = clamp(u, 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def uphill_gravity_comp(pitch_rad: float) -> float:
  """Uphill gravity deceleration component g*sin(pitch) capped at MAX_UPHILL_SLOPE_COMP."""
  if not math.isfinite(pitch_rad) or pitch_rad <= 0.0:
    return 0.0
  return min(MAX_UPHILL_SLOPE_COMP, GRAVITY * math.sin(pitch_rad))


def brake_lamp_floor(pitch_rad: float = 0.0) -> float:
  """Slope-compensated deceleration floor maintaining a_hyd <= A_HYD_LAMP_FLOOR (-0.45 m/s^2)."""
  return A_HYD_LAMP_FLOOR - uphill_gravity_comp(pitch_rad)


def compute_stop_margins(
  *,
  has_lead: bool,
  d_rel: float,
  v_lead: float,
  is_e2e_stop: bool,
  plan_stop_dist: float | None,
  d_stop_cfg: float = DEFAULT_STOP_DISTANCE,
) -> tuple[float | None, float | None, float | None]:
  """Compute unified physical stop clearances (d_hard, d_tol, d_rem) in meters."""
  if has_lead:
    if not (math.isfinite(d_rel) and math.isfinite(v_lead)):
      return None, None, None
    v_l = max(0.0, v_lead)
    d_lead_eq = d_rel + (v_l * v_l) / (2.0 * LEAD_EMERGENCY_DECEL)
    d_cfg = d_stop_cfg if (math.isfinite(d_stop_cfg) and d_stop_cfg >= MIN_SAFE_STOP_DISTANCE) else DEFAULT_STOP_DISTANCE
    d_hard = d_lead_eq - MIN_SAFE_STOP_DISTANCE
    d_tol_target = max(MIN_SAFE_STOP_DISTANCE, d_cfg - STOP_TOLERANCE_MARGIN)
    d_tol = d_lead_eq - d_tol_target
    d_rem = d_lead_eq - d_cfg
    return d_hard, d_tol, d_rem

  if is_e2e_stop and plan_stop_dist is not None and math.isfinite(plan_stop_dist):
    d = max(0.0, plan_stop_dist)
    return d, d, d

  return None, None, None


@dataclass
class UnifiedLongState:
  """Fixed-size pure scalar state for UnifiedLongController."""
  a_ref: float = 0.0
  j_ref: float = 0.0
  tau_pos: float = 0.0
  tau_stat: float = 0.0
  v_zero_cross: float = 0.0
  hold_latched: bool = False
  precharge_latched: bool = False
  envelope_limited: bool = False
  launch_armed: bool = False
  initialized: bool = False
  # Diagnostic / telemetry flags (read-only)
  stop_relief_active: bool = False
  launch_shaping_active: bool = False
  safety_override: bool = False
  last_j_u_shaped: float | None = None

  def reset(self) -> None:
    self.a_ref = 0.0
    self.j_ref = 0.0
    self.tau_pos = 0.0
    self.tau_stat = 0.0
    self.v_zero_cross = 0.0
    self.hold_latched = False
    self.precharge_latched = False
    self.envelope_limited = False
    self.launch_armed = False
    self.initialized = False
    self.stop_relief_active = False
    self.launch_shaping_active = False
    self.safety_override = False
    self.last_j_u_shaped = None


@dataclass(frozen=True)
class UnifiedLongInput:
  """Pure scalar inputs for a single 100 Hz UnifiedLongController tick."""
  active: bool
  dt: float
  v_ego: float
  a_ego: float
  a_des: float
  a_target: float
  v_target_now: float
  j_target_now: float
  accel_min: float
  accel_max: float
  should_stop: bool
  soft_hold: bool
  brake_pressed: bool
  gas_pressed: bool
  cruise_standstill: bool
  fcw: bool
  has_lead: bool
  d_rel: float
  v_lead: float
  pitch_rad: float
  is_e2e_stop: bool
  plan_stop_dist: float | None
  stopping_accel_cfg: float
  cp_stop_accel: float
  cp_stopping_decel_rate: float
  cp_starting_state: bool
  cp_v_starting: float
  cp_start_accel: float = 0.0
  d_stop_cfg: float = DEFAULT_STOP_DISTANCE
  prev_scc_areq_val: float = 0.0


@dataclass(frozen=True)
class UnifiedLongOutput:
  """Output of a single UnifiedLongController tick."""
  a_cmd: float
  j_cmd: float
  j_u_shaped: float | None
  mode: str  # "off" | "stopping" | "starting" | "pid"
  safety_override: bool
  precharge_latched: bool
  hold_latched: bool
  stop_relief_active: bool
  launch_shaping_active: bool


class UnifiedLongController:
  """Single continuous 100 Hz longitudinal trajectory & physical envelope engine."""

  def __init__(self) -> None:
    self.state = UnifiedLongState()

  def reset(self) -> None:
    self.state.reset()

  def sync_external_state(self, *, a_ref: float, in_stopping: bool = False) -> None:
    """Sync state when tests or external callers seed LongControl state directly."""
    if math.isfinite(a_ref):
      self.state.a_ref = float(a_ref)
      self.state.initialized = True
    if in_stopping:
      self.state.hold_latched = True
      self.state.launch_armed = True
      self.state.tau_pos = 0.0

  def update(self, inp: UnifiedLongInput) -> UnifiedLongOutput:
    st = self.state

    if not inp.active:
      st.reset()
      return UnifiedLongOutput(
        a_cmd=0.0, j_cmd=0.0, j_u_shaped=None, mode="off",
        safety_override=False, precharge_latched=False, hold_latched=False,
        stop_relief_active=False, launch_shaping_active=False,
      )

    # Finite input gate (NaN/inf defense)
    if not (math.isfinite(inp.dt) and inp.dt > 0.0
            and math.isfinite(inp.v_ego) and math.isfinite(inp.a_ego)
            and math.isfinite(inp.a_des) and math.isfinite(inp.accel_min)
            and math.isfinite(inp.accel_max)):
      a_safe = clamp(inp.a_des if math.isfinite(inp.a_des) else 0.0, -3.5, 2.0)
      st.a_ref = a_safe
      st.j_ref = 0.0
      st.precharge_latched = False
      st.stop_relief_active = False
      st.launch_shaping_active = False
      st.safety_override = True
      st.last_j_u_shaped = None
      return UnifiedLongOutput(
        a_cmd=a_safe, j_cmd=0.0, j_u_shaped=None, mode="pid",
        safety_override=True, precharge_latched=False, hold_latched=st.hold_latched,
        stop_relief_active=False, launch_shaping_active=False,
      )

    a_target = inp.a_target if math.isfinite(inp.a_target) else inp.a_des
    j_target_now = inp.j_target_now if math.isfinite(inp.j_target_now) else 0.0
    v_target_now = inp.v_target_now if math.isfinite(inp.v_target_now) else inp.v_ego
    v_ego = max(0.0, inp.v_ego)
    dt = inp.dt

    if not st.initialized:
      st.a_ref = clamp(inp.a_des, inp.accel_min, inp.accel_max)
      st.initialized = True

    # Driver pedal override: 0-tick bypass of all shaping
    if inp.brake_pressed or inp.gas_pressed:
      st.precharge_latched = False
      st.launch_armed = False
      st.tau_pos = 0.0
      st.tau_stat = 0.0
      st.stop_relief_active = False
      st.launch_shaping_active = False
      st.safety_override = False
      st.envelope_limited = False
      st.last_j_u_shaped = None
      if inp.soft_hold:
        a_out = clamp(inp.cp_stop_accel, inp.accel_min, inp.accel_max)
        st.hold_latched = True
      else:
        a_out = clamp(inp.a_des, inp.accel_min, inp.accel_max)
        if inp.brake_pressed and (inp.should_stop or st.hold_latched):
          st.hold_latched = True
        elif inp.gas_pressed:
          st.hold_latched = False
      st.j_ref = (a_out - st.a_ref) / dt
      st.a_ref = a_out
      mode = "stopping" if st.hold_latched else "pid"
      return UnifiedLongOutput(
        a_cmd=a_out, j_cmd=j_target_now, j_u_shaped=None, mode=mode,
        safety_override=False, precharge_latched=False, hold_latched=st.hold_latched,
        stop_relief_active=False, launch_shaping_active=False,
      )

    g_up = uphill_gravity_comp(inp.pitch_rad)
    a_lamp = A_HYD_LAMP_FLOOR - g_up

    # 1. Unified physical stop clearances (d_hard, d_tol, d_rem)
    d_hard, d_tol, d_rem = compute_stop_margins(
      has_lead=inp.has_lead,
      d_rel=inp.d_rel,
      v_lead=inp.v_lead,
      is_e2e_stop=inp.is_e2e_stop,
      plan_stop_dist=inp.plan_stop_dist,
      d_stop_cfg=inp.d_stop_cfg,
    )
    a_req_hard = -(v_ego * v_ego) / (2.0 * max(d_hard, 0.05)) if d_hard is not None else -1e9
    a_req_tol = -(v_ego * v_ego) / (2.0 * max(d_tol, 0.05)) if d_tol is not None else -1e9

    # 2. Priority #0: 0-tick unconditional safety / emergency bypass
    a_demand = min(inp.a_des, a_target) if inp.should_stop else inp.a_des
    emerg = (
      inp.fcw
      or inp.a_des < EMERGENCY_DECEL_THRESHOLD
      or a_target < EMERGENCY_DECEL_THRESHOLD
      or (a_demand < 0.0 and v_ego > 0.05 and (
        (inp.has_lead and math.isfinite(inp.d_rel) and inp.d_rel <= CLOSE_LEAD_OVERRIDE_DISTANCE)
        or (v_ego <= V_EASE_START and d_hard is not None and (d_hard <= 0.20 or a_req_hard <= min(a_demand, a_lamp)))
      ))
    )
    if emerg:
      st.precharge_latched = False
      st.launch_armed = False
      st.tau_pos = 0.0
      st.tau_stat = 0.0
      st.stop_relief_active = False
      st.launch_shaping_active = False
      st.safety_override = True
      st.envelope_limited = False
      st.last_j_u_shaped = None
      a_out = clamp(a_demand, inp.accel_min, inp.accel_max)
      da_emg = (a_out - st.a_ref) / dt
      j_out = min(j_target_now, da_emg) if a_out < st.a_ref else j_target_now
      st.j_ref = da_emg
      st.a_ref = a_out
      if inp.soft_hold or (inp.should_stop and (inp.a_ego > inp.stopping_accel_cfg or (inp.has_lead and inp.d_rel < HOLD_CLOSE_LEAD_DIST))):
        st.hold_latched = True
      elif not inp.should_stop and not inp.cruise_standstill:
        st.hold_latched = False
      mode = "stopping" if st.hold_latched else "pid"
      return UnifiedLongOutput(
        a_cmd=a_out, j_cmd=j_out, j_u_shaped=None, mode=mode,
        safety_override=True, precharge_latched=False, hold_latched=st.hold_latched,
        stop_relief_active=False, launch_shaping_active=False,
      )

    st.safety_override = False

    # 3. First-class physical acceleration envelope [a_env_min, a_env_max]
    a_env_min = inp.accel_min
    a_env_max = inp.accel_max
    dx_lamp = (v_ego * v_ego) / (2.0 * abs(a_lamp))
    stop_ctx = (
      inp.should_stop
      or inp.is_e2e_stop
      or (inp.has_lead and math.isfinite(inp.v_lead) and inp.v_lead <= 1.5)
      or v_target_now <= 0.5
    )

    # (a) Kinematic stop-pocket hydraulic pre-charge latch (decel-aware lead departure check)
    lead_pulling_away = (
      inp.has_lead
      and math.isfinite(inp.v_lead)
      and (inp.v_lead > POCKET_LEAD_CLEAR_DEPART_V
           or (inp.v_lead > POCKET_LEAD_DEPART_V and inp.v_lead > v_ego + POCKET_LEAD_PULLAWAY_DV))
    )
    lead_departing = (
      lead_pulling_away
      or (not inp.should_stop and inp.a_des > POCKET_DEPART_ACCEL)
    )
    in_pocket = (d_rem is not None and d_rem <= dx_lamp + POCKET_MARGIN_ON) or inp.should_stop
    out_of_pocket = (d_rem is not None and d_rem > dx_lamp + POCKET_MARGIN_OFF) and not inp.should_stop
    if lead_departing or out_of_pocket or inp.soft_hold or inp.cruise_standstill:
      st.precharge_latched = False
    elif stop_ctx and in_pocket and (
      (v_ego <= POCKET_HOLD_V_MAX and st.a_ref <= a_lamp + 1e-6)
      or (v_ego <= POCKET_V_MAX and inp.a_des <= a_lamp + 1e-6 and v_ego > V_EASE_MIN)
    ):
      st.precharge_latched = True

    # (b) Hold latch transition & flicker-invariant StopReq 1 -> 0 synchronization
    hold_flicker_guard = (
      inp.a_des <= HOLD_FLICKER_DECEL
      and a_target <= -0.05
      and v_target_now <= HOLD_FLICKER_V_TARGET
      and not lead_pulling_away
    )
    hold_depart_ok = (
      not inp.should_stop
      and not inp.cruise_standstill
      and not inp.soft_hold
      and not hold_flicker_guard
    )
    if st.hold_latched and hold_depart_ok:
      st.hold_latched = False
      # Match downstream CAN-FD StopReq=1 -> 0 output (aReqValue == 0.0) with zero step jump.
      st.a_ref = max(0.0, inp.prev_scc_areq_val if math.isfinite(inp.prev_scc_areq_val) else 0.0)
      st.tau_stat = 0.0
      st.tau_pos = 0.0
      st.v_zero_cross = v_ego
      st.launch_armed = True
    elif not st.hold_latched:
      close_lead = inp.has_lead and math.isfinite(inp.d_rel) and inp.d_rel < HOLD_CLOSE_LEAD_DIST
      if inp.soft_hold or inp.cruise_standstill or (
        inp.should_stop and ((inp.a_ego + g_up) > inp.stopping_accel_cfg or close_lead)
      ):
        st.hold_latched = True

    eff_should_stop = inp.should_stop or (st.hold_latched and not inp.soft_hold)
    hold_cap = min(inp.stopping_accel_cfg, a_lamp) if v_ego <= V_EASE_MIN else a_lamp

    # (c) Base target acceleration: soft_hold / starting / continuous closed-loop PID
    if inp.soft_hold:
      a_goal = inp.cp_stop_accel
      a_env_max = min(a_env_max, inp.cp_stop_accel)
      st.stop_relief_active = False
    else:
      a_des_eff = inp.a_des
      if (not st.hold_latched and inp.cp_starting_state and v_ego <= inp.cp_v_starting
          and inp.cp_start_accel > 0.0 and a_des_eff > 0.0):
        # Preserve non-increasing peak relative to startAccel when startingState is enabled.
        a_des_eff = min(a_des_eff, inp.cp_start_accel) if inp.cp_start_accel < a_des_eff else inp.cp_start_accel

      # Low-speed stop ease-out (3.5 -> 0.7 m/s, active down to 0.5 m/s)
      a_goal = a_des_eff
      st.stop_relief_active = False
      if stop_ctx and V_EASE_MIN < v_ego <= V_EASE_START:
        # When already in hold_latched with strong prior braking (e.g. a_ref = -1.4),
        # ease-out applies to min(a_des_eff, a_ref) so stopping decel unwinds smoothly
        # toward a_lamp instead of holding -1.4 all the way to standstill.
        a_base = min(a_des_eff, st.a_ref) if st.hold_latched else a_des_eff
        if a_base < a_lamp:
          ease_u = smoothstep((v_ego - V_EASE_END) / (V_EASE_START - V_EASE_END))
          a_ease = a_lamp - ease_u * (max(EASE_PEAK_DECEL, abs(a_lamp)) - abs(a_lamp))
          a_relief = min(a_lamp, a_ease, a_req_tol)
          a_goal = max(a_base, a_relief)
          st.stop_relief_active = (a_goal > a_base + 1e-6) or (st.a_ref < a_goal - 1e-6)

      if st.precharge_latched:
        a_env_max = min(a_env_max, a_lamp)

      # Final low-speed hold settle & precharge cap (v_ego <= 0.70 m/s & eff_should_stop)
      low_speed_settle = eff_should_stop and v_ego <= V_EASE_END
      if low_speed_settle:
        a_env_max = min(a_env_max, hold_cap)
        if v_ego <= V_EASE_MIN and st.hold_latched:
          if st.a_ref <= hold_cap:
            # One-way hold: keep existing deeper decel without snapping upward, and don't ramp further down.
            a_goal = st.a_ref
          else:
            a_goal = min(a_goal, hold_cap)

      # Standstill breakaway envelope (v_ego <= 0.04 m/s while departing)
      if not st.hold_latched and a_des_eff > 0.0 and v_ego <= V_BREAKAWAY:
        st.tau_stat += dt
        breakaway_cap = (
          min(BREAKAWAY_ACCEL_FLAT + g_up, BREAKAWAY_RELEASE_BASE + BREAKAWAY_RELEASE_UPHILL_GAIN * g_up)
          if g_up > 0.25 else (BREAKAWAY_ACCEL_FLAT + g_up)
        )
        a_breakaway = smoothstep(st.tau_stat / BREAKAWAY_RAMP_TIME) * breakaway_cap
        a_env_max = min(a_env_max, a_breakaway)
      else:
        if st.tau_stat > 0.0 and v_ego > V_BREAKAWAY and not st.hold_latched and a_des_eff > 0.0:
          st.a_ref = min(st.a_ref, max(BREAKAWAY_ACCEL_FLAT, BREAKAWAY_RELEASE_BASE + BREAKAWAY_RELEASE_UPHILL_GAIN * g_up))
        st.tau_stat = 0.0

    a_goal = clamp(a_goal, a_env_min, a_env_max)

    # 4. Arm / disarm zero-crossing launch S-curve cleanly
    if st.hold_latched or st.a_ref <= REACCEL_ARM_DECEL:
      st.launch_armed = True
      st.tau_pos = 0.0
      st.v_zero_cross = v_ego
    elif v_ego >= CITY_V_END:
      st.launch_armed = False

    # 5. 4-Quadrant Continuous Jerk Integrator
    da = a_goal - st.a_ref
    j_dn = math.inf
    j_up = math.inf
    j_u_shaped: float | None = None
    j_d_shaped: float | None = None
    st.launch_shaping_active = False

    if da < -1e-9:
      # Quadrant 1/4 downward: immediate tracking for safety, with smooth 2-stage low-speed hold settle:
      # gentle pre-charge (1.2 m/s^3) while above a_lamp, then soft settle (0.8 m/s^3) at/below a_lamp.
      # Do not emit negative j_d_shaped (which CAN-FD multiplies by 4x to JerkLowerLimit=5.0).
      if (eff_should_stop and v_ego <= V_EASE_END and not inp.soft_hold
          and st.hold_latched and st.a_ref > hold_cap):
        if st.a_ref > a_lamp + 1e-6:
          j_dn = max(JERK_PRECHARGE_DOWN, inp.cp_stopping_decel_rate)
        else:
          j_dn = max(0.1, inp.cp_stopping_decel_rate)
      else:
        j_dn = math.inf

    elif da > 1e-9:
      if st.a_ref < -1e-9:
        # Quadrant 2 upward (still in braking region, a_ref < 0)
        if a_goal <= 0.0:
          # Deceleration relief toward a_lamp (suspend pitch recovery)
          if st.stop_relief_active or (stop_ctx and v_ego < JERK_DECEL_RELIEF_V_HIGH):
            u_v = smoothstep((v_ego - V_EASE_START) / (JERK_DECEL_RELIEF_V_HIGH - V_EASE_START))
            j_up = JERK_DECEL_RELIEF_LOW + (JERK_DECEL_RELIEF_HIGH - JERK_DECEL_RELIEF_LOW) * u_v
            j_u_shaped = j_up
            if a_goal <= a_lamp + 1e-6:
              st.stop_relief_active = True
        else:
          # Braking -> positive acceleration transition: unwind caliper pressure cleanly to a=0
          if v_ego < CITY_V_END:
            j_up = JERK_UNWIND_TO_POS
            j_u_shaped = JERK_UNWIND_TO_POS
            st.tau_pos = 0.0
            st.v_zero_cross = v_ego
            st.launch_shaping_active = True
      else:
        # Quadrant 3 upward (a_ref >= 0 -> positive drive acceleration)
        if st.launch_armed and v_ego < CITY_V_END:
          if v_ego > V_BREAKAWAY:
            st.tau_pos += dt
          eta_launch = 1.0 - smoothstep((st.v_zero_cross - LAUNCH_V_ZERO_LOW) / LAUNCH_V_ZERO_SPAN)
          eta_city = 1.0 - smoothstep((v_ego - CITY_V_START) / (CITY_V_END - CITY_V_START))
          j0 = JERK_REACCEL_MIN + JERK_LAUNCH_DELTA_MIN * eta_launch + JERK_UPHILL_GAIN * g_up
          j_peak = JERK_REACCEL_PEAK + JERK_LAUNCH_DELTA_PEAK * eta_launch + JERK_UPHILL_GAIN * g_up
          j_scurve = j0 + smoothstep(st.tau_pos / SCURVE_RAMP_TIME) * (j_peak - j0)
          j_up = eta_city * j_scurve + (1.0 - eta_city) * JERK_MAX_PHYS
          # Emit 1:1 shaped jerk once wheels are rolling (v_ego > V_BREAKAWAY):
          # - When actively rate-limiting a_ref (da >= j_up * dt) or planner jerk saturated (<= 0),
          #   emit exact j_up so CAN-FD tracks aReqRaw 1:1 without the 1.0 m/s^3 bottleneck.
          # - When already caught up to a_goal with positive planner jerk, cap by min(j_up, j_target_now)
          #   so we never inflate jerk_u above upstream intent.
          if v_ego > V_BREAKAWAY:
            rate_limited = da >= (j_up * dt - 1e-9)
            if rate_limited or j_target_now <= 0.0:
              j_u_shaped = j_up
            else:
              j_u_shaped = min(j_up, j_target_now)
          st.launch_shaping_active = True
    else:
      # Steady state (da ~= 0): advance tau_pos if moving under positive accel so S-curve disarms cleanly.
      if st.launch_armed and st.a_ref >= 0.0 and a_goal > 0.0:
        if v_ego > V_BREAKAWAY:
          st.tau_pos += dt
          if st.tau_pos >= SCURVE_RAMP_TIME:
            st.launch_armed = False
          else:
            st.launch_shaping_active = True
        else:
          # Stationary at breakaway cap: shaping active, but j_u_shaped stays None (no premature inflation).
          st.launch_shaping_active = True

    max_up = j_up * dt if math.isfinite(j_up) else math.inf
    max_dn = j_dn * dt if math.isfinite(j_dn) else math.inf
    step = clamp(da, -max_dn, max_up)

    # Clean zero-crossing landing when transitioning from braking (< 0) to positive drive (> 0)
    if st.a_ref < -1e-9 and a_goal > 0.0 and v_ego < CITY_V_END and (st.a_ref + step) > 0.0:
      a_next = 0.0
      st.tau_pos = 0.0
      st.v_zero_cross = v_ego
    elif step == da:
      # Exact pass-through when unclipped (zero ULP drift for highway PID & coasting).
      a_next = a_goal
    else:
      a_next = st.a_ref + step

    a_next = clamp(a_next, inp.accel_min, inp.accel_max)
    j_ref = (a_next - st.a_ref) / dt

    if st.launch_armed and a_next >= a_goal - 1e-4 and st.tau_pos >= SCURVE_RAMP_TIME:
      st.launch_armed = False

    st.envelope_limited = abs(a_next - inp.a_des) > 1e-3
    st.a_ref = a_next
    st.j_ref = j_ref
    st.last_j_u_shaped = j_u_shaped

    # Output jerk contract:
    # - When actively shaping upward (j_u_shaped is not None), pass 1:1 shaped jerk bound.
    # - When rapidly pre-charging downward above a_lamp (j_d_shaped is not None), pass 1:1 pre-charge jerk.
    # - Otherwise pass planner j_target_now untouched (0.0% change in highway/coasting).
    if j_u_shaped is not None:
      j_out = j_u_shaped
    elif j_d_shaped is not None:
      j_out = j_d_shaped
    else:
      j_out = j_target_now

    if st.hold_latched:
      mode = "stopping"
    elif inp.cp_starting_state and v_ego <= inp.cp_v_starting and a_next > 0.0:
      mode = "starting"
    else:
      mode = "pid"

    return UnifiedLongOutput(
      a_cmd=a_next,
      j_cmd=j_out,
      j_u_shaped=j_u_shaped,
      mode=mode,
      safety_override=False,
      precharge_latched=st.precharge_latched,
      hold_latched=st.hold_latched,
      stop_relief_active=st.stop_relief_active,
      launch_shaping_active=st.launch_shaping_active,
    )
