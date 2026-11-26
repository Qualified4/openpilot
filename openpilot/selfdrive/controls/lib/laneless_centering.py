"""Lane centering for the laneless target curvature (`LanelessCentering`, 0 = off, 100 = the gains below).

In the laneless branch controlsd follows modelV2.action.desiredCurvature. The
driving model's path often sits off the lane centre: on drive 0000047d it held
the car +8 cm right on average on straights (|offset| 16.5 cm, 47 % of the time
over 15 cm) and 0.1-0.4 m inside weak corners, while the controls tracked the
model's curvature closely. This adds a small, bounded lateral acceleration on
top of the model's, at 100 Hz in controlsd:

* P: the pure-pursuit acceleration that moves the model's path point at the
  lookahead L (2 s of travel, 15..50 m) onto the lane centre at L,
  2 (centre - path) v² / L². Path and centre bend together in a curve, so only
  their offset counts.
* I: the car's own offset from the lane centre 5 m ahead (less the bend κ·5²/2
  its own curvature puts there), integrated at 0.02 m/s² per m·s, capped at
  ±0.08 m/s², leaking over 2 s while off.

At 100 % the sum is capped at 0.15 m/s² on a straight road, plus 0.15 per m/s² of the
road's lateral acceleration (|κ_model|·v², up to 3 m/s²). The target is
smoothed over 0.5 s and may change at most 0.4 m/s² per second; the output
ramps in over 1 s and out over 0.5 s.

The setting scales the pursuit, the integral gain and limit, the cap and the
rate limit together up to 200 % (0.30 m/s² on a straight road); 0 switches it
off, ramping out over 0.5 s with the integral cleared. The default here is 100
(the Rust branch defaults to 0). controlsd rereads it once a second.

The formula and constants are those of LanelessCentering on the
ccnc-hda1-rust branch (rust/crates/controlsd/src/centering.rs, c4726d9b). On
top of its conditions (30 km/h, both ego lines at probability 0.5, 2.6..4.3 m
wide at 5 m, radius 80 m or 50 m with lines at 0.6, offset limits, a 0.7 m
lane-centre jump holds it off 1 s) it keeps the safety conditions of the
corner centering it replaces:

* with traffic beside the car, return toward the lane centre only when radar
  shows enough clearance there and the adjacent vehicles are not approaching
  laterally. A BSD warning (held 0.5 s) without nearby radar geometry still
  blocks that direction. The traffic limit also applies after smoothing;
* none during a lane change, turn desire, automatic turn or turn signal and
  for 1 s after; none while the driver steers and for 0.5 s after;
* the lane-line standard deviation fades it out over 0.2..0.4 m;
* a lane whose width changes along the view (more than 0.5 m over 0..35 m) or
  over time (more than 0.3 m/s) is a merge or split: off.

Coordinates: modelV2 and lane-line y are positive to the right, as is the
curvature, so a positive acceleration turns right and pushes the car right.
radarState.leadsLeft holds dPath > 0 (left), leadsRight dPath < 0.
"""
import math

import numpy as np

DT = 0.01
DT_MDL = 0.05
MIN_SPEED = 30.0 / 3.6
LOOKAHEAD_S = 2.0
LOOKAHEAD_MIN, LOOKAHEAD_MAX = 15.0, 50.0
CAR_POINT = 5.0
INTEGRAL_GAIN = 0.02   # m/s² of integral per metre of car offset per second
INTEGRAL_MAX = 0.08
INTEGRAL_LEAK_S = 2.0
ACCEL_MAX = 0.15       # m/s² on a straight road
CURVE_CAP_GAIN = 0.15  # m/s² of cap per m/s² of the road's lateral acceleration
CURVE_ACCEL_MAX = 3.0
CURVE_FULL = 1.5       # road lateral acceleration where the curve allowances stop growing
RAMP_IN_S, RAMP_OUT_S = 1.0, 0.5
LINE_PROB_MIN = 0.5
CONFIDENT_PROB = 0.6
WIDTH_MIN, WIDTH_MAX = 2.6, 4.3
CAR_OFFSET_MAX, CAR_OFFSET_MAX_CONFIDENT = 0.8, 1.5
PATH_OFFSET_MAX, PATH_OFFSET_MAX_CURVE, PATH_OFFSET_MAX_CONFIDENT = 0.6, 0.9, 2.0
CURVATURE_MAX, CURVATURE_MAX_CONFIDENT = 1.0 / 80.0, 1.0 / 50.0
LANE_JUMP = 0.7
LANE_JUMP_HOLD_S = 1.0
ACCEL_TAU_S = 0.5
ACCEL_RATE_MAX = 0.4   # m/s² per second
STRENGTH_MAX = 2.0     # LanelessCentering 200 %: every gain and limit above at most doubles

STD_RAMP = (0.20, 0.40)
MERGE_MAX_DW = 0.5
MERGE_MAX_RATE = 0.3
LANE_CHANGE_REST_S = 1.0
DRIVER_REST_S = 0.5
BSD_HOLD_S = 0.5
SIDE_DREL = (-5.0, 10.0)  # m: level with the car or just ahead
SIDE_MIN_VLEAD = 2.0
SIDE_CLEARANCE_M = 2.60  # 1.00 m ego + 1.25 m target half-widths, plus 0.35 m
BSD_MAX_DREL = 7.0      # CarState's forward corner-radar BSD range
TURN_DESIRE_MIN = 0.2       # desireState turn/lane-change probabilities, summed
LANE_CHANGE_DESIRE_MIN = 0.1
LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT = 3, 4  # log.Desire

# controlsState.lanelessCenteringBlock bits: why the correction is off or ramping out this step.
BLOCK_REASONS = (
  'input', 'speed', 'curvature', 'prob', 'width', 'offset', 'jump', 'std', 'merge',
  'lanechange', 'driver', 'side', 'bsd', 'off',
)
BLOCK_BITS = {name: 1 << i for i, name in enumerate(BLOCK_REASONS)}


def block_names(mask):
  return [name for name, bit in BLOCK_BITS.items() if mask & bit]


def lookahead(v):
  return float(np.clip(v * LOOKAHEAD_S, LOOKAHEAD_MIN, LOOKAHEAD_MAX))


def _at(xs, ys, x):
  """y at x along increasing xs; None outside the data or for bad input."""
  if len(xs) < 2 or len(xs) != len(ys) or not (xs[0] <= x <= xs[-1]):
    return None
  y = float(np.interp(x, xs, ys))
  return y if math.isfinite(y) else None


def _normal_width(lx, ly, rx, ry, x):
  """Lane width at x, measured across the lane's heading there."""
  l, r = _at(lx, ly, x), _at(rx, ry, x)
  x1 = max(x - 1.0, 0.0)
  ends = [_at(lx, ly, x + 1.0), _at(rx, ry, x + 1.0), _at(lx, ly, x1), _at(rx, ry, x1)]
  if l is None or r is None or None in ends:
    return None
  yaw = math.atan2((ends[0] + ends[1]) / 2 - (ends[2] + ends[3]) / 2, x + 1.0 - x1)
  return (r - l) * math.cos(yaw)


def _lane_center(lane, x):
  lx, ly, rx, ry = lane[:4]
  left, right = _at(lx, ly, x), _at(rx, ry, x)
  return (left + right) / 2 if left is not None and right is not None else None


def _side_clearance(lane, lead, side):
  """Signed, lane-normal distance from the lane centre to a radar return."""
  # Radar yRel/vLat point left, whereas model y points right. dPath is
  # relative to the model path and is used by radard to classify the side.
  if not (lead.status and lead.radar and math.isfinite(lead.yRel) and lead.yRel != 0.0):
    return None
  x = max(lead.dRel, 0.0)
  center, ahead = _lane_center(lane, x), _lane_center(lane, x + 1.0)
  if center is None or ahead is None:
    return None
  slope = ahead - center
  # Behind the car, extend the lane tangent and allow for unobserved bend.
  rear = min(lead.dRel, 0.0)
  distance = side * (-lead.yRel - center - rear * slope) / math.hypot(1.0, slope)
  return distance - 0.5 * CURVATURE_MAX_CONFIDENT * rear ** 2


def _side_limit(lane, offsets, leads, bsd, radar_valid, side, accel_per_m):
  traffic = [lead for lead in leads if SIDE_DREL[0] <= lead.dRel <= SIDE_DREL[1] and lead.vLead > SIDE_MIN_VLEAD]
  reasons = (['side'] if traffic else []) + (['bsd'] if bsd else [])
  if not reasons:
    return math.inf, reasons
  if offsets is None or not radar_valid or not traffic or (bsd and not any(lead.dRel <= BSD_MAX_DREL for lead in traffic)):
    return 0.0, reasons
  for lead in traffic:
    # Zero lateral velocity is a parallel track. Any inward motion or
    # unavailable velocity keeps the direction blocked.
    if not (math.isfinite(lead.vLat) and -side * lead.vLat >= 0.0):
      return 0.0, reasons
    clearance = _side_clearance(lane, lead, side)
    if clearance is None or clearance < SIDE_CLEARANCE_M:
      return 0.0, reasons
  center = _lane_center(lane, 0.0)
  if center is None:
    return 0.0, reasons
  # The car now, the 5 m bend-corrected car position, and the lookahead
  # model path must all have room before the centre in the pushed direction.
  return max(0.0, min(side * center, side * offsets[0], side * offsets[1])) * accel_per_m, reasons


def atc_active(atc_type):
  """carrotMan.atcType of an automatic turn or lane change under way (a prepare phase is not)."""
  return not (atc_type in ('', 'none', 'straight') or atc_type.endswith(' prepare'))


def strength_from_param(percent):
  """LanelessCentering (%) as the gain scale, 0..STRENGTH_MAX."""
  return min(max(float(percent), 0.0) / 100.0, STRENGTH_MAX)


class LanelessCentering:
  def __init__(self, strength=1.0):
    self.strength = strength  # LanelessCentering / 100, set by controlsd; reset() keeps it
    self.reset()

  def reset(self):
    self.blend = 0.0
    self.integral = 0.0
    self.accel = 0.0       # smoothed, rate-limited target before the blend, m/s²
    self.accel_out = 0.0   # blend * accel: what was added this step, m/s²
    self.block = 0
    self.last_car = None
    self.hold = 0.0
    self.lc_rest = 0.0
    self.driver_rest = 0.0
    self.bsd_left = 0.0
    self.bsd_right = 0.0
    self.frame_id = None
    self.w_filtered = None
    self.merge = False

  def _lane(self, md):
    """Ego lane lines and path as arrays, or None when modelV2 lacks them."""
    if len(md.laneLines) < 4 or len(md.laneLineProbs) < 4 or len(md.laneLineStds) < 4 or len(md.position.x) < 2:
      return None
    lane = (np.asarray(md.laneLines[1].x, float), np.asarray(md.laneLines[1].y, float),
            np.asarray(md.laneLines[2].x, float), np.asarray(md.laneLines[2].y, float),
            np.asarray(md.position.x, float), np.asarray(md.position.y, float))
    if not all(np.all(np.isfinite(a)) for a in lane) or not all(np.all(np.diff(a) > 0) for a in lane[0::2]):
      return None
    return lane

  def _update_merge(self, md, lane, v):
    """Once per model frame: a width that changes along the view or over time is a merge or split."""
    frame_id = (md.frameId, len(md.position.x))
    if frame_id == self.frame_id:
      return
    self.frame_id = frame_id
    lx, ly, rx, ry = lane[:4]
    widths = [_normal_width(lx, ly, rx, ry, x) for x in (0.0, min(20.0, 1.2 * v), min(35.0, 2.0 * v))]
    if None in widths:
      self.merge = True
      return
    wm = widths[1]
    if self.w_filtered is None:
      self.w_filtered = wm
    w_prev = self.w_filtered
    self.w_filtered += DT_MDL / (0.4 + DT_MDL) * (wm - self.w_filtered)
    w_rate = (self.w_filtered - w_prev) / DT_MDL
    self.merge = max(widths) - min(widths) > MERGE_MAX_DW or abs(w_rate) > MERGE_MAX_RATE

  def _offsets(self, md, lane, v, kappa, fails):
    """The car's offset from the lane centre at 5 m and the path's at the lookahead (+ = centre to the right)."""
    pl, pr = float(md.laneLineProbs[1]), float(md.laneLineProbs[2])
    sure = pl >= CONFIDENT_PROB and pr >= CONFIDENT_PROB
    if v < MIN_SPEED:
      fails.append('speed')
    if abs(kappa) > (CURVATURE_MAX_CONFIDENT if sure else CURVATURE_MAX):
      fails.append('curvature')
    if pl < LINE_PROB_MIN or pr < LINE_PROB_MIN:
      fails.append('prob')
    lx, ly, rx, ry, px, py = lane
    left5, right5 = _at(lx, ly, CAR_POINT), _at(rx, ry, CAR_POINT)
    L = lookahead(v)
    left_l, right_l, path_l = _at(lx, ly, L), _at(rx, ry, L), _at(px, py, L)
    if None in (left5, right5, left_l, right_l, path_l):
      fails.append('input')
      return None
    car = (left5 + right5) / 2 - kappa * CAR_POINT ** 2 / 2
    path = (left_l + right_l) / 2 - path_l
    curve = min(min(abs(kappa) * v * v, CURVE_ACCEL_MAX) / CURVE_FULL, 1.0)
    if sure:
      path_max = PATH_OFFSET_MAX + (PATH_OFFSET_MAX_CONFIDENT - PATH_OFFSET_MAX) * curve
      car_max = CAR_OFFSET_MAX + (CAR_OFFSET_MAX_CONFIDENT - CAR_OFFSET_MAX) * curve
    else:
      path_max = PATH_OFFSET_MAX + (PATH_OFFSET_MAX_CURVE - PATH_OFFSET_MAX) * curve
      car_max = CAR_OFFSET_MAX
    if not (WIDTH_MIN <= right5 - left5 <= WIDTH_MAX):
      fails.append('width')
    if abs(car) > car_max or abs(path) > path_max:
      fails.append('offset')
    return car, path

  def update(self, md, cs, rs, atc_type, v_ego, *, radar_valid=True):
    """One controlsd step in the laneless branch. Returns the curvature to add, 1/m (+ = right)."""
    v = float(v_ego)
    kappa = float(md.action.desiredCurvature)
    scale = min(self.strength, STRENGTH_MAX)
    fails = [] if scale > 0.0 else ['off']

    # Rests and holds that run whether or not the correction acts
    ds = list(md.meta.desireState)
    lc_now = bool(md.meta.desire.raw != 0 or md.meta.laneChangeState.raw != 0 or cs.leftBlinker or cs.rightBlinker or
                  atc_active(atc_type) or
                  (len(ds) > LANE_CHANGE_RIGHT and (sum(ds[1:5]) >= TURN_DESIRE_MIN or
                                                    ds[LANE_CHANGE_LEFT] >= LANE_CHANGE_DESIRE_MIN or
                                                    ds[LANE_CHANGE_RIGHT] >= LANE_CHANGE_DESIRE_MIN)))
    self.lc_rest = LANE_CHANGE_REST_S if lc_now else max(self.lc_rest - DT, 0.0)
    if lc_now or self.lc_rest > 0.0:
      fails.append('lanechange')
    self.driver_rest = DRIVER_REST_S if cs.steeringPressed else max(self.driver_rest - DT, 0.0)
    if cs.steeringPressed or self.driver_rest > 0.0:
      fails.append('driver')
    self.bsd_left = BSD_HOLD_S if cs.leftBlindspot else max(self.bsd_left - DT, 0.0)
    self.bsd_right = BSD_HOLD_S if cs.rightBlindspot else max(self.bsd_right - DT, 0.0)

    lane = self._lane(md)
    offsets = None
    std_w = 0.0
    if lane is None:
      fails.append('input')
      self.last_car = None
    else:
      offsets = self._offsets(md, lane, v, kappa, fails)
      car = offsets[0] if offsets is not None else None
      if car is not None and self.last_car is not None and abs(car - self.last_car) > LANE_JUMP:
        self.hold = LANE_JUMP_HOLD_S
      self.last_car = car
      if self.hold > 0.0:
        fails.append('jump')
      self._update_merge(md, lane, v)
      if self.merge:
        fails.append('merge')
      std_w = float(np.interp(max(float(md.laneLineStds[1]), float(md.laneLineStds[2])), STD_RAMP, [1.0, 0.0]))
      if std_w <= 0.0:
        fails.append('std')
    self.hold = max(self.hold - DT, 0.0)

    accel_per_m = 2.0 * v * v / lookahead(v) ** 2
    safe_offsets = offsets if not fails else None
    left, left_reasons = _side_limit(lane, safe_offsets, rs.leadsLeft, self.bsd_left > 0.0, radar_valid, -1.0, accel_per_m)
    right, right_reasons = _side_limit(lane, safe_offsets, rs.leadsRight, self.bsd_right > 0.0, radar_valid, 1.0, accel_per_m)
    lo, hi = -left, right

    limits = []
    if fails or offsets is None:
      self.blend = max(self.blend - DT / RAMP_OUT_S, 0.0)
      self.integral = self.integral * math.exp(-DT / INTEGRAL_LEAK_S) if scale > 0.0 else 0.0
    else:
      car, path = offsets
      i_max = INTEGRAL_MAX * scale
      self.integral = float(np.clip(np.clip(self.integral + INTEGRAL_GAIN * scale * car * DT, -i_max, i_max), lo, hi))
      cap = (ACCEL_MAX + CURVE_CAP_GAIN * min(abs(kappa) * v * v, CURVE_ACCEL_MAX)) * scale
      wanted = float(np.clip(scale * accel_per_m * path + self.integral, -cap, cap)) * std_w
      target = float(np.clip(wanted, lo, hi))
      if wanted < lo:
        limits += left_reasons
      if wanted > hi:
        limits += right_reasons
      self.blend = min(self.blend + DT / RAMP_IN_S, 1.0)
      # The 0.5 s filter and rate limit keep small model-frame changes from
      # shaking the wheel (2–3 Hz on the Rust branch before them).
      step = ACCEL_RATE_MAX * scale * DT
      self.accel += float(np.clip((target - self.accel) * DT / ACCEL_TAU_S, -step, step))

    # A new car or a shrinking centre envelope must also limit the
    # previously smoothed acceleration, including during other fades.
    if self.accel < lo:
      limits += left_reasons
    if self.accel > hi:
      limits += right_reasons
    self.accel = min(max(self.accel, lo), hi)
    self.block = 0
    for name in fails + limits:
      self.block |= BLOCK_BITS[name]
    if self.blend == 0.0:
      self.accel = 0.0
    self.accel_out = self.blend * self.accel
    if not math.isfinite(self.accel_out):
      self.reset()
      return 0.0
    return self.accel_out / max(v, MIN_SPEED) ** 2
