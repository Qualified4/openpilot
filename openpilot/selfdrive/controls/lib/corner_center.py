"""Corner centering for the laneless target curvature (LatCornerCenterBlend).

In a corner the driving model's path can sit inside the lane center. When the
lane lines are trusted, plannerd moves the planned path part of the way back
toward the lane center and publishes the matching curvature difference Δκ in
lateralPlan. controlsd adds Δκ to modelV2.action.desiredCurvature only in the
laneless branch, after checking that the plan is fresh and valid.

plannerd, 20 Hz (CornerCenter.update):
  gate = CornerGate conditions, 0..1
  Δy_raw(x) = lane_center(x) - modelV2.position.y(x), interpolated on path x
  keep only the outward direction, minus a 3 cm deadband, |Δy| <= 0.4 m, faded to 0 over 40..50 m
  Δy_tgt(x) = gate * blend * Δy_raw(x) * taper(x), taper = u²(3-2u), u = clip(x / max(v*T_L, L_MIN))
  Δy <- Δy + RATE * (Δy_tgt - Δy)       (90 % of a step in about 2 s)
  Δκ_tgt = gcfp(yaw(P0 + Δy)) - gcfp(yaw(P0)), gcfp = get_curvature_from_plan at modeld's lat_action_t
  Δκ moves toward Δκ_tgt by at most J_RAMP / max(v, 8)² per second (J_DRIVER while the driver steers
  and for DRIVER_REST_S after), so turning on, fading out and every condition that clears Δy reach
  the output as a ramp. Only latActive off and a non-finite result zero Δκ at once.
  |Δκ| <= DK_LAT_ACC_MAX / max(v, 8)²
Only path points up to the first one beyond USE_X are read; farther points may run backward
while the model plans a stop, and Δy is 0 there.

controlsd, 100 Hz (CornerCenterOutput.update): the same ramp from the last value it added, toward the
plan's Δκ, or toward 0 for a stale or invalid plan and while the driver steers.

Coordinates: modelV2 and lane-line y are positive to the right, sgn = +1 in a right-hand corner.
radarState.leadsLeft holds dPath > 0 (left), leadsRight dPath < 0.
"""
import math

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan

TRAJECTORY_SIZE = 33
DT = 0.05
BLEND_MAX = 50  # LatCornerCenterBlend upper bound, percent

# Correction shape
T_L = 2.5             # taper length, seconds of travel
L_MIN = 15.0          # taper length floor, m
RATE = 0.056          # per-frame approach of Δy to its target: 90 % in 2 s
DEADBAND = 0.03       # m
DY_MAX = 0.4          # m
VIEW = (40.0, 50.0)   # Δy fades to 0 over this x range, m
SNAP = 0.0005         # residual Δy below this with a zero target is dropped, m
DK_LAT_ACC_MAX = 0.3  # |Δκ| <= this / max(v, 8)²
J_RAMP = 0.5          # |dΔκ/dt| <= this / max(v, 8)², m/s³ of lateral jerk
J_DRIVER = 2.0        # the same while the driver steers and for DRIVER_REST_S after: 0.15 s from the bound to 0
USE_X = 55.0          # m, path points read: up to and including the first one beyond this
INVARIANT_K = 2.5     # require L > INVARIANT_K * v * lat_action_t, else Δκ = 0
STARTUP_S = 0.5
PLAN_MAX_AGE_S = 0.2  # controlsd ignores an older lateralPlan

# Conditions
CORNER_ON, CORNER_OFF = 0.30, 0.25
CORNER_W = (0.30, 0.60)
KAPPA_FLOOR = 0.00055
MEAS_ON, MEAS_OFF = 0.20, 0.15
STD_RAMP = (0.20, 0.40)
PROB_ON, PROB_OFF = 0.6, 0.5
WIDTH_RANGE = (2.5, 4.5)
MERGE_MAX_DW = 0.5
MERGE_MAX_RATE = 0.3
FOLLOW_MAX_DEV = 0.8
FOLLOW_MARGIN = 0.3
INWARD_MIN = 0.03
LANE_CHANGE_REST_S = 1.0
DRIVER_REST_S = 0.5
BSD_HOLD_S = 0.5
SIDE_DREL = (-5.0, 30.0)
SIDE_MIN_VLEAD = 2.0
SPEED_KPH = (27.0, 33.0)
ON_FRAMES = 3
LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT = 3, 4  # log.Desire

# lateralPlan.cornerCenterBlock bits: conditions that kept the gate closed and reasons Δy was zeroed.
BLOCK_REASONS = (
  'input', 'prob', 'std', 'width', 'merge', 'corner', 'kappa', 'meas', 'follow', 'inward',
  'lanechange', 'driver', 'side', 'bsd', 'speed', 'startup',
  'latActive', 'laneMode', 'mpc', 'residualInward', 'nan', 'invariant',
)
BLOCK_BITS = {name: 1 << i for i, name in enumerate(BLOCK_REASONS)}


def block_mask(reasons):
  mask = 0
  for r in reasons:
    mask |= BLOCK_BITS[r]
  return mask


def block_names(mask):
  return [name for name, bit in BLOCK_BITS.items() if mask & bit]


def _interp(xq, xs, ys):
  return np.interp(xq, xs, ys, left=np.nan, right=np.nan)


def normal_width(lx, ly, rx, ry, x):
  l = _interp(x, lx, ly)
  r = _interp(x, rx, ry)
  x1 = max(x - 1.0, 0.0)
  c2 = (_interp(x + 1.0, lx, ly) + _interp(x + 1.0, rx, ry)) / 2
  c1 = (_interp(x1, lx, ly) + _interp(x1, rx, ry)) / 2
  yaw = np.arctan2(c2 - c1, x + 1.0 - x1)
  return float((r - l) * np.cos(yaw))


def quad_curv(xs, ys):
  m = np.isfinite(ys)
  if m.sum() < 4:
    return np.nan
  return 2 * np.polyfit(xs[m], ys[m], 2)[0]


def model_lat_acc(md):
  """Model lateral acceleration over 0.3..1.0 s and 1.0..1.8 s: the larger one if both share a sign, else 0."""
  tt = np.asarray(md.position.t, float)
  vel = np.hypot(np.asarray(md.velocity.x, float), np.asarray(md.velocity.y, float))
  a = vel * np.asarray(md.orientationRate.z, float)
  a1 = float(np.mean(a[(tt >= 0.3) & (tt <= 1.0)]))
  a2 = float(np.mean(a[(tt >= 1.0) & (tt <= 1.8)]))
  if np.sign(a1) == np.sign(a2):
    return a1 if abs(a1) > abs(a2) else a2
  return 0.0


def _finite(values):
  return bool(np.all(np.isfinite(np.asarray(values, float))))


def used_points(px):
  """Number of leading path points corner centering reads: up to and including the first one beyond USE_X."""
  beyond = np.flatnonzero(np.asarray(px, float) > USE_X)
  return int(beyond[0]) + 1 if len(beyond) else len(px)


def ramp(prev, target, v, j, dt):
  """prev moved toward target by at most j / max(v, 8)² * dt, then held within ±DK_LAT_ACC_MAX / max(v, 8)²."""
  v2 = max(v, 8.0) ** 2
  step = j / v2 * dt
  lim = DK_LAT_ACC_MAX / v2
  return float(np.clip(np.clip(target, prev - step, prev + step), -lim, lim))


class CornerGate:
  """Decides whether, and how strongly, corner centering may act this frame."""

  def __init__(self):
    self.t_start = None
    self.lc_rest_until = -1.0
    self.drv_rest_until = -1.0
    self.bsd_l_until = -1.0
    self.bsd_r_until = -1.0
    self.on_count = 0
    self.w_filtered = None
    self.prob_ok = False
    self.corner_ok = False
    self.meas_ok = False
    self.n_used = TRAJECTORY_SIZE

  def restart(self, t):
    self.t_start = t
    self.on_count = 0

  def update(self, t, md, cs, rs, atc_active, measured_curvature):
    """Returns (gate, sgn, fails)."""
    if self.t_start is None:
      self.t_start = t
    fails = []
    v = max(float(cs.vEgo), 0.1)
    ok_in = len(md.laneLines) >= 4 and len(md.position.x) == TRAJECTORY_SIZE and len(md.laneLineProbs) >= 4 and len(md.laneLineStds) >= 4
    if ok_in:
      self.n_used = used_points(md.position.x)
      px = np.asarray(md.position.x, float)[:self.n_used]
      py = np.asarray(md.position.y, float)[:self.n_used]
      lx, ly = np.asarray(md.laneLines[1].x, float), np.asarray(md.laneLines[1].y, float)
      rx, ry = np.asarray(md.laneLines[2].x, float), np.asarray(md.laneLines[2].y, float)
      ok_in = (_finite(px) and _finite(py) and _finite(lx) and _finite(ly) and _finite(rx) and _finite(ry)
               and _finite(list(md.laneLineProbs)[:4]) and _finite(list(md.laneLineStds)[:4])
               and bool(np.all(np.diff(px) > 0)) and px[-1] >= 25.0)
    if not ok_in:
      self.on_count = 0
      return 0.0, 0.0, ['input']

    # Raw lane-line probabilities with hysteresis; continuous standard-deviation fade
    pl, pr = float(md.laneLineProbs[1]), float(md.laneLineProbs[2])
    if self.prob_ok:
      self.prob_ok = pl >= PROB_OFF and pr >= PROB_OFF
    else:
      self.prob_ok = pl >= PROB_ON and pr >= PROB_ON
    if not self.prob_ok:
      fails.append('prob')
    std_w = float(np.interp(max(float(md.laneLineStds[1]), float(md.laneLineStds[2])), STD_RAMP, [1.0, 0.0]))
    if std_w <= 0.0:
      fails.append('std')

    # Normal lane width; merges and splits
    widths = [normal_width(lx, ly, rx, ry, x) for x in (0.0, min(20.0, 1.2 * v), min(35.0, 2.0 * v))]
    wm = widths[1]
    if not (WIDTH_RANGE[0] <= wm <= WIDTH_RANGE[1]):
      fails.append('width')
    if self.w_filtered is None:
      self.w_filtered = wm
    w_prev = self.w_filtered
    self.w_filtered += DT / (0.4 + DT) * (wm - self.w_filtered)
    w_rate = (self.w_filtered - w_prev) / DT
    if not (_finite(widths) and max(widths) - min(widths) <= MERGE_MAX_DW and abs(w_rate) <= MERGE_MAX_RATE):
      fails.append('merge')

    # Corner: model lateral acceleration with hysteresis, curvature floor, measured lateral acceleration in the same direction
    a = model_lat_acc(md)
    sgn = 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
    self.corner_ok = abs(a) >= CORNER_OFF if self.corner_ok else abs(a) > CORNER_ON
    corner_w = float(np.interp(abs(a), CORNER_W, [0.0, 1.0]))
    meas = v ** 2 * float(measured_curvature) * sgn
    self.meas_ok = meas >= MEAS_OFF if self.meas_ok else meas >= MEAS_ON
    if not self.corner_ok or sgn == 0:
      fails.append('corner')
    if abs(a) / v ** 2 < KAPPA_FLOOR:
      fails.append('kappa')
    if not self.meas_ok:
      fails.append('meas')

    # The model is following the lane (10..25 m)
    xs = np.arange(10.0, 25.01, 2.5)
    my = _interp(xs, px, py)
    ll = _interp(xs, lx, ly)
    rr = _interp(xs, rx, ry)
    dev = my - (ll + rr) / 2
    inside = bool(np.all((my >= ll + FOLLOW_MARGIN) & (my <= rr - FOLLOW_MARGIN)))
    xs_k = np.arange(0.0, 40.01, 2.5)
    k_lane = quad_curv(xs_k, (_interp(xs_k, lx, ly) + _interp(xs_k, rx, ry)) / 2)
    k_model = quad_curv(xs_k, _interp(xs_k, px, py))
    sign_ok = not (abs(k_lane) > 3e-4 and np.sign(k_lane) != np.sign(k_model))
    if not (float(np.nanmax(np.abs(dev))) <= FOLLOW_MAX_DEV and inside and sign_ok):
      fails.append('follow')

    # Inside bias (10..25 m mean)
    if not float(np.nanmean(dev)) * sgn > INWARD_MIN:
      fails.append('inward')

    # Lane change, then a rest
    ds = md.meta.desireState
    lc_now = bool(md.meta.desire.raw != 0 or md.meta.laneChangeState.raw != 0 or
                  (len(ds) > 4 and (ds[LANE_CHANGE_LEFT] >= 0.1 or ds[LANE_CHANGE_RIGHT] >= 0.1)) or
                  atc_active or cs.leftBlinker or cs.rightBlinker)
    if lc_now:
      self.lc_rest_until = t + LANE_CHANGE_REST_S
    if lc_now or t < self.lc_rest_until:
      fails.append('lanechange')
    # Driver steering, then a rest
    if cs.steeringPressed:
      self.drv_rest_until = t + DRIVER_REST_S
    if cs.steeringPressed or t < self.drv_rest_until:
      fails.append('driver')

    # Side traffic on the side the correction moves toward (corner outside)
    if cs.leftBlindspot:
      self.bsd_l_until = t + BSD_HOLD_S
    if cs.rightBlindspot:
      self.bsd_r_until = t + BSD_HOLD_S
    if sgn > 0:
      bsd, leads = t < self.bsd_l_until, rs.leadsLeft
    elif sgn < 0:
      bsd, leads = t < self.bsd_r_until, rs.leadsRight
    else:
      bsd, leads = False, []
    if bsd:
      fails.append('bsd')
    if any(SIDE_DREL[0] <= x.dRel <= SIDE_DREL[1] and x.vLead > SIDE_MIN_VLEAD for x in leads):
      fails.append('side')

    speed_w = float(np.interp(v * 3.6, SPEED_KPH, [0.0, 1.0]))
    if speed_w <= 0:
      fails.append('speed')
    if t - self.t_start < STARTUP_S:
      fails.append('startup')

    self.on_count = self.on_count + 1 if not fails else 0
    gate = corner_w * speed_w * std_w if self.on_count >= ON_FRAMES else 0.0
    return gate, sgn, fails


class CornerCenter:
  def __init__(self, blend_pct=0):
    self.gate = CornerGate()
    self.blend = 0.0
    self.set_blend(blend_pct)
    self.custom_delay = 0.0
    self.base_smooth = 0.0
    self.lat_delay_dynamic = None
    self.dy = np.zeros(TRAJECTORY_SIZE)
    self.curvature = 0.0
    self.gate_value = 0.0
    self.block = 0
    self.driver_until = -1.0

  def set_blend(self, blend_pct):
    self.blend = float(np.clip(blend_pct, 0, BLEND_MAX)) * 0.01

  def set_delay_params(self, steer_actuator_delay, lat_smooth_sec):
    """SteerActuatorDelay and LatSmoothSec params (x0.01 s), as modeld reads them."""
    self.custom_delay = float(steer_actuator_delay) * 0.01
    self.base_smooth = float(lat_smooth_sec) * 0.01

  @property
  def active(self):
    return self.curvature != 0.0

  def lat_action_t(self, md, live_delay):
    """modeld's lat_action_t for this frame: the previous frame's dynamic delay + 0.075 s."""
    if self.lat_delay_dynamic is None:
      self.lat_delay_dynamic = self.base_smooth
    lat_t = self.lat_delay_dynamic + 0.05 + 0.025
    ystd = float(md.position.yStd[10]) if len(md.position.yStd) > 10 else 0.0
    smooth_dyn = 0.0 if self.base_smooth <= 0 else float(np.clip(
      self.base_smooth + np.interp(ystd, [0.15, 0.25], [0.0, self.base_smooth * 2]), 0.0, 0.60))
    if self.custom_delay > 0.0:
      self.lat_delay_dynamic = self.custom_delay + smooth_dyn
    elif live_delay is not None:
      self.lat_delay_dynamic = float(live_delay) + smooth_dyn
    return lat_t

  def update(self, t, md, cs, rs, lat_active, atc_active, measured_curvature, live_delay,
             path_xyz, v_plan, t_idxs, v_ego, mpc_ok, lane_mode):
    """One plannerd frame. path_xyz is the path sent to the lateral MPC (P0). Returns Δκ, 1/m."""
    gate, sgn, fails = self.gate.update(t, md, cs, rs, atc_active, measured_curvature)
    lat_t = self.lat_action_t(md, live_delay)
    v = max(float(cs.vEgo), 0.1)
    if cs.steeringPressed:
      self.driver_until = t + DRIVER_REST_S

    zero = None
    if 'input' in fails:
      zero = 'input'
    elif cs.steeringPressed:
      zero = 'driver'
    elif not lat_active:
      zero = 'latActive'
    elif lane_mode:
      zero = 'laneMode'
    elif 'lanechange' in fails and (md.meta.desire.raw != 0 or md.meta.laneChangeState.raw != 0 or cs.leftBlinker or cs.rightBlinker):
      zero = 'lanechange'
    elif not mpc_ok:
      zero = 'mpc'
      self.gate.restart(t)

    nan = False
    if zero is not None:
      self.dy[:] = 0.0
    else:
      n = self.gate.n_used
      px = np.asarray(md.position.x, float)[:n]
      center = (np.interp(px, np.asarray(md.laneLines[1].x, float), np.asarray(md.laneLines[1].y, float)) +
                np.interp(px, np.asarray(md.laneLines[2].x, float), np.asarray(md.laneLines[2].y, float))) / 2
      raw = center - np.asarray(md.position.y, float)[:n]
      out = np.maximum(-raw * sgn - DEADBAND, 0.0)
      dy_raw = -sgn * np.minimum(out, DY_MAX) * np.interp(px, VIEW, [1.0, 0.0])
      u = np.clip(px / max(v * T_L, L_MIN), 0.0, 1.0)
      tgt = np.zeros(TRAJECTORY_SIZE)
      tgt[:n] = gate * self.blend * dy_raw * (u * u * (3 - 2 * u))
      self.dy += RATE * (tgt - self.dy)
      if not np.any(tgt != 0.0) and float(np.max(np.abs(self.dy))) < SNAP:
        self.dy[:] = 0.0
      near = (px >= 10.0) & (px <= 25.0)
      if sgn != 0 and float(np.mean(self.dy[:n][near])) * sgn > 1e-6:
        self.dy[:] = 0.0
        zero = 'residualInward'
      if not _finite(self.dy):
        self.dy[:] = 0.0
        zero = 'nan'
        nan = True

    dk = 0.0
    if np.any(self.dy != 0.0):
      if not max(v * T_L, L_MIN) > INVARIANT_K * v * lat_t:
        zero = zero or 'invariant'
      else:
        from openpilot.selfdrive.controls.lib.lateral_planner import yaw_from_path_no_scipy
        p1 = np.copy(path_xyz)
        p1[:, 1] += self.dy
        yaw0, yr0 = yaw_from_path_no_scipy(path_xyz, v_plan, smooth_window=5, clip_rate=2.0, align_first_yaw=None)
        yaw1, yr1 = yaw_from_path_no_scipy(p1, v_plan, smooth_window=5, clip_rate=2.0, align_first_yaw=None)
        dk = float(get_curvature_from_plan(yaw1, yr1, t_idxs, v_ego, lat_t) - get_curvature_from_plan(yaw0, yr0, t_idxs, v_ego, lat_t))
        if not math.isfinite(dk):
          dk = 0.0
          self.dy[:] = 0.0
          zero = zero or 'nan'
          nan = True

    if nan or not lat_active:
      self.curvature = 0.0
    else:
      self.curvature = ramp(self.curvature, dk, v, J_DRIVER if t < self.driver_until else J_RAMP, DT)
    self.gate_value = float(gate)
    self.block = block_mask(fails + ([zero] if zero else []))
    return self.curvature


def corner_center_curvature(sm, v_ego, steering_pressed):
  """controlsd side: lateralPlan.cornerCenterCurvature if the plan is seen, valid, fresh and finite, else 0."""
  lp = sm['lateralPlan']
  dk = float(lp.cornerCenterCurvature)
  if not (sm.seen['lateralPlan'] and sm.valid['lateralPlan'] and (sm.frame - sm.recv_frame['lateralPlan']) * DT_CTRL <= PLAN_MAX_AGE_S + 1e-9
          and lp.modelMonoTime > 0 and lp.mpcSolutionValid and math.isfinite(dk) and not steering_pressed):
    return 0.0
  lim = DK_LAT_ACC_MAX / max(v_ego, 8.0) ** 2
  return float(np.clip(dk, -lim, lim))


class CornerCenterOutput:
  """controlsd side: the Δκ added to the laneless target curvature, ramped from the value added last."""

  def __init__(self):
    self.curvature = 0.0
    self.driver_frames = 0

  def reset(self):
    self.curvature = 0.0
    self.driver_frames = 0

  def update(self, sm, v_ego, steering_pressed):
    if steering_pressed:
      self.driver_frames = round(DRIVER_REST_S / DT_CTRL)
    elif self.driver_frames > 0:
      self.driver_frames -= 1
    driver = self.driver_frames > 0
    target = 0.0 if driver else corner_center_curvature(sm, v_ego, False)
    self.curvature = ramp(self.curvature, target, v_ego, J_DRIVER if driver else J_RAMP, DT_CTRL)
    return self.curvature
