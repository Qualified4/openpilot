"""Optional custom CCNC presentation only; baseline CAN handling stays in hyundaicanfd."""
import math
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from opendbc.car import structs
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.params import Params


def _ccnc_valid_boundary(x, y):
  if len(x) < 2 or len(x) != len(y):
    return False
  xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
  return bool(np.isfinite(xs).all() and np.isfinite(ys).all() and (xs[1:] > xs[:-1]).all())

def _ccnc_side_lane_center(md, side):
  inner_idx = 1 if side == 0 else 2
  outer_idx = 0 if side == 0 else 3

  if md is None or len(md.laneLines) < 4 or len(md.laneLineProbs) < 4:
    return None

  if md.laneLineProbs[outer_idx] < 0.6:
    return None

  inner = md.laneLines[inner_idx]
  outer = md.laneLines[outer_idx]

  if len(inner.x) < 2 or len(outer.x) < 2:
    return None

  x = 20.0

  if not (inner.x[0] <= x <= inner.x[-1]
          and outer.x[0] <= x <= outer.x[-1]):
    return None

  inner_y = float(np.interp(x, inner.x, inner.y))
  outer_y = float(np.interp(x, outer.x, outer.y))

  width = abs(outer_y - inner_y)

  if not 2.3 <= width <= 4.8:
    return None

  return abs(inner.y[0]) + width * 0.5

class _CcncRadarDisplayTracker:
  """Observation continuity and lane-free fallback for the CCNC display only."""
  def __init__(self):
    self.approaching = False
    self.approach_pending = {}
    self.approach_holds = {}
    self.stop_last_frame = None
    self.stop_distance = 0.0
    self.saved_widths = [None, None]
    self.stopped = False
    self.stop_pending = {}
    self.stop_holds = {}
    self._saved_side = [None, None]
    self.live = None
    self.last_frame = -1
    self.tracks = {}
    self._points = ()
    self.selected = (None, None, None)
    self.crossing = {}
    self.positions = {}
    self.lateral_grace = {}
    self.recent_selected = {}
    self.recent_front = {}
    self.lane_ready = True
    self._model = None
    self._model_stamp = None
    self._inner_data = None
    self._lane_probs = None
    self._side_width_cache = {}
    self._lane_data = None
    self._projection_live = None
    self._projection = None
    self._path_data = None
    self._path_live = self._path_points = None

  def update_stop(self, speed_kph, frame, acceleration_kph=0.0):
    previous = self.stop_last_frame
    gap = previous is None or not 0 <= frame - previous <= 15
    stopped = math.isfinite(speed_kph) and abs(speed_kph) <= 0.3
    approaching = (not self.stopped and math.isfinite(speed_kph) and math.isfinite(acceleration_kph)
                   and 0.3 < speed_kph <= 10.0 and acceleration_kph < -0.1)
    if gap or self.live is None or not (stopped or approaching):
      self.approach_pending.clear()
      self.approach_holds.clear()
    self.approaching = approaching
    if gap:
      self.saved_widths = [None, None]
      self._lane_data = self._projection = None
    else:
      self.stop_distance += abs(speed_kph) / 3.6 * (frame - previous) * 0.01 if math.isfinite(speed_kph) else 10.0
    for side, saved in enumerate(self.saved_widths):
      if saved is not None and self.stop_distance - saved[0] >= 10.0:
        self.saved_widths[side] = None
        self._lane_data = self._projection = None
        self._side_width_cache.clear()
    if gap or not stopped or self.live is None:
      self.stop_pending.clear()
      self.stop_holds.clear()
    if stopped != self.stopped:
      self._lane_data = self._projection = None
      self._side_width_cache.clear()
    self.stopped = stopped
    self.stop_last_frame = frame

  def approaching_display(self, values, leads, frame):
    # A stationary target lost just before ego stops is carried in odometry coordinates.
    for side, lead in enumerate(leads):
      prefix = 'LF' if side == 0 else 'RF'
      keys = (prefix + '_DETECT', prefix + '_DETECT_DISTANCE', prefix + '_DETECT_LATERAL')
      if self.approaching and lead is not None:
        pending = self.approach_pending.get(side)
        world_x = lead.dRel + self.stop_distance
        if abs(lead.vLead) > (3.0 if side in self.approach_holds else 2.0) / 3.6:
          self.approach_pending.pop(side, None)
        else:
          if (pending is None or pending[0] != lead.trackId
              or abs(world_x - pending[2]) > 1.0 or abs(lead.yRel - pending[3]) > 0.75):
            pending = (lead.trackId, frame, world_x, lead.yRel)
            self.approach_pending[side] = pending
          if frame - pending[1] >= 50 and self.stable(lead.trackId, 50):
            self.approach_holds[side] = (lead.trackId, world_x, lead.yRel,
                                        tuple(values[k] for k in keys), frame, self.stop_distance)
      else:
        self.approach_pending.pop(side, None)
      held = self.approach_holds.get(side)
      if held is None:
        continue
      track_id, world_x, y, display, last_seen, last_distance = held
      x = world_x - self.stop_distance  # Keep negative distance internally.
      invalid = x < -1.0 or frame - last_seen > 300 or self.stop_distance - last_distance > 5.0
      for point in self.live.points:
        if (str(point.radarSource) == 'frontRadar' and math.isfinite(point.dRel)
            and math.isfinite(point.yRel) and abs(point.dRel - x) <= 2.0 and abs(point.yRel - y) <= 0.8):
          if (not math.isfinite(point.vLead) or abs(point.vLead) > 3.0 / 3.6
              or abs(point.dRel - x) > 1.0 or abs(point.yRel - y) > 0.75
              or point.trackId in (self.selected[0], self.selected[2 - side])):
            invalid = True
            break
      if invalid:
        self.approach_holds.pop(side, None)
        continue
      display = (display[0], max(0.0, x) * 0.8, display[2])
      if self.stopped:
        old = self.stop_holds.get(side)
        if old is None or x < old[1]:
          self.stop_holds[side] = (track_id, x, y, display)
        self.approach_holds.pop(side, None)
      elif lead is None:
        values.update(zip(keys, display))

  def stopped_display(self, values, leads, frame):
    # Display memory only: never feed synthetic tracks back into selection/history.
    if self.live is None:
      return
    if self.approaching or self.approach_holds:
      self.approaching_display(values, leads, frame)
    if not self.stopped:
      return
    for side, lead in enumerate(leads):
      prefix = 'LF' if side == 0 else 'RF'
      keys = (prefix + '_DETECT', prefix + '_DETECT_DISTANCE', prefix + '_DETECT_LATERAL')
      held = self.stop_holds.get(side)
      if held is not None and lead is not None:
        for point, px, py, vr, v in self._points:
          if (abs(px - held[1]) <= 2.0 and abs(py - held[2]) <= 0.8
              and (abs(v) > 2.0 / 3.6 or abs(vr) > 2.0 / 3.6)):
            self.stop_holds.pop(side, None)
            held = None
            break
      if lead is not None:
        # A current selected candidate always supersedes a stale display.
        pending = self.stop_pending.get(side)
        stationary = abs(lead.vLead) <= 2.0 / 3.6 and abs(lead.vRel) <= 2.0 / 3.6
        if not stationary:
          if held is not None and abs(lead.dRel - held[1]) <= 2.0 and abs(lead.yRel - held[2]) <= 0.8:
            self.stop_holds.pop(side, None)
          self.stop_pending.pop(side, None)
          continue
        if (pending is None or pending[0] != lead.trackId
            or abs(lead.dRel - pending[2]) > 1.0 or abs(lead.yRel - pending[3]) > 0.75):
          pending = (lead.trackId, frame, lead.dRel, lead.yRel)
          self.stop_pending[side] = pending
        if (frame - pending[1] >= 50 and self.stable(lead.trackId, 50)
            and (held is None or lead.dRel <= held[1] + 1.0)):
          self.stop_holds[side] = (lead.trackId, lead.dRel, lead.yRel, tuple(values[k] for k in keys))
        continue
      self.stop_pending.pop(side, None)
      if held is None:
        continue
      track_id, x, y, display = held
      invalid = False
      for point, px, py, vr, v in self._points:
        nearby = abs(px - x) <= 2.0 and abs(py - y) <= 0.8
        if nearby:
          # Motion, ID reuse, or a current selection in another slot cancels the ghost.
          if (abs(v) > 2.0 / 3.6 or abs(vr) > 2.0 / 3.6
              or abs(px - x) > 1.0 or abs(py - y) > 0.75
              or point.trackId in self.selected):
            invalid = True
            break
      if invalid:
        self.stop_holds.pop(side, None)
      else:
        values.update(zip(keys, display))

  def _update_model(self, md):
    # SubMaster model readers are immutable between publications; retain the reader itself.
    stamp = getattr(md, "timestampEof", None) if md is not None else None
    if md is self._model and stamp is not None and stamp == self._model_stamp:
      return
    self._side_width_cache.clear()
    self._model, self._model_stamp = md, stamp
    self._inner_data = self._lane_probs = self._lane_data = self._projection = self._path_data = None
    self._projection_live = None
    self._path_live = self._path_points = None

  def lane_probabilities(self, md):
    self._update_model(md)
    if self._lane_probs is None:
      left = right = 0.0
      if md is not None and len(md.laneLineProbs) >= 3 and len(md.laneLines) >= 3:
        self._inner_data = tuple((np.asarray(line.x, dtype=np.float64), np.asarray(line.y, dtype=np.float64))
                                 for line in (md.laneLines[1], md.laneLines[2]))
        if _ccnc_valid_boundary(*self._inner_data[0]):
          left = md.laneLineProbs[1]
        if _ccnc_valid_boundary(*self._inner_data[1]):
          right = md.laneLineProbs[2]
      self._lane_probs = left, right
    return self._lane_probs

  def lane_projection(self, live):
    if self._projection is not None and live is self._projection_live:
      return self._projection
    if self._lane_data is None:
      md = self._model
      lines, edges = md.laneLines, md.roadEdges
      data = list(self._inner_data)
      for valid, line in ((md.laneLineProbs[0] > 0.1, lines[0]), (md.laneLineProbs[3] > 0.1, lines[3]),
                          (True, edges[0]), (True, edges[1])):
        data.append((np.asarray(line.x, dtype=np.float64), np.asarray(line.y, dtype=np.float64)) if valid else None)
      flags = tuple(item is not None and _ccnc_valid_boundary(*item) for item in data[2:])
      self._saved_side = [None, None]
      for side in (0, 1):
        if flags[side] and self._lane_probs[side] >= 0.1:
          self.saved_widths[side] = (self.stop_distance, data[side], data[2 + side],
                                     data[4 + side] if flags[2 + side] else None)
        elif self.stopped and self.saved_widths[side] is not None:
          self._saved_side[side] = self.saved_widths[side]
      self._lane_data = data, flags
    data, flags = self._lane_data
    distances = np.fromiter((p[1] for p in self._points), dtype=np.float64, count=len(self._points))
    projected = [np.interp(distances, *data[0]), np.interp(distances, *data[1])]
    self._projection_distances = distances
    self._projection_inner = projected
    self._side_projection = [None, None]
    self._projection_live = live
    self._projection = float(data[0][1][0]), float(data[1][1][0]), tuple(zip(projected[0].tolist(), projected[1].tolist()))
    return self._projection

  def side_projection(self, side):
    cached = self._side_projection[side]
    if cached is not None:
      return cached
    data, flags = self._lane_data
    distances = self._projection_distances
    saved = self._saved_side[side]
    projected = []
    for offset in (0, 2):
      curve = data[2 + side + offset]
      saved_curve = saved[2 + offset // 2] if saved is not None else None
      if saved_curve is not None:
        inner = saved[1]
        xs, ys = saved_curve
        vals = self._projection_inner[side] + np.interp(distances, xs, ys) - np.interp(distances, *inner)
        vals[(distances < max(xs[0], inner[0][0])) | (distances > min(xs[-1], inner[0][-1]))] = np.nan
        projected.append(vals.tolist())
      elif flags[side + offset]:
        xs, ys = curve
        vals = np.interp(distances, xs, ys)
        vals[(distances < xs[0]) | (distances > xs[-1])] = np.nan
        projected.append(vals.tolist())
      else:
        projected.append([math.nan] * len(distances))
    cached = tuple(zip(*projected))
    self._side_projection[side] = cached
    return cached

  def side_entry_width(self, lead, side):
    # Existing continuous targets may stop without becoming new detections.
    if (lead.vLead * CV.MS_TO_KPH > 2.0 or self._model.laneLineProbs[3 if side else 0] > 0.1
        or self.recently_selected(lead.trackId)):
      return True
    key = (lead.trackId, side)
    cached = self._side_width_cache.get(key)
    if cached is not None:
      return cached
    data, flags = self._lane_data
    saved = self._saved_side[side]
    if saved is not None:
      _, inner, outer, edge = saved
      sign = 1.0 if side else -1.0
      valid = True
      for x in (max(0.0, lead.dRel - 20.0), lead.dRel + 20.0):
        if not inner[0][0] <= x <= inner[0][-1]:
          valid = False
          break
        iy = float(np.interp(x, *inner))
        for curve in (outer, edge):
          if curve is not None and (not curve[0][0] <= x <= curve[0][-1]
              or sign * (float(np.interp(x, *curve)) - iy) - 0.25 <= 1.8):
            valid = False
            break
        if not valid:
          break
      self._side_width_cache[key] = valid
      return valid
    valid = flags[2 + side]
    if valid:
      inner_x, inner_y = data[side]
      edge_x, edge_y = data[4 + side]
      sign = 1.0 if side else -1.0
      # The target-distance width has already passed; inspect only two extra points.
      for x in (max(0.0, lead.dRel - 20.0), lead.dRel + 20.0):
        if (not (inner_x[0] <= x <= inner_x[-1] and edge_x[0] <= x <= edge_x[-1])
            or sign * (float(np.interp(x, edge_x, edge_y)) - float(np.interp(x, inner_x, inner_y))) - 0.25 <= 1.8):
          valid = False
          break
    self._side_width_cache[key] = valid
    return valid

  def observe(self, live, frame):
    if frame < self.last_frame or frame - self.last_frame > 15:
      self.live = None
      self.lane_ready = True
    if frame < self.last_frame or frame - self.last_frame > 15 or live is None:
      self.crossing.clear()
      self.tracks.clear()
      self._points = ()
      self.selected = (None, None, None)
      self.positions.clear()
      self.lateral_grace.clear()
      self.recent_selected.clear()
      self.recent_front.clear()
    self.last_frame = frame
    if live is self.live:
      return
    self._side_width_cache.clear()
    self.live = live  # card keeps the same RadarData object until the next radar update.
    self._path_points = self._projection = None
    current = {}
    points = []
    if live is not None:
      for p in live.points:
        dRel, yRel, vRel, vLead = p.dRel, p.yRel, p.vRel, p.vLead
        if (str(p.radarSource) != "frontRadar" or dRel < 1
            or not (math.isfinite(dRel) and math.isfinite(yRel) and math.isfinite(vRel) and math.isfinite(vLead))):
          continue
        points.append((p, dRel, yRel, vRel, vLead))
        start, count = frame, 1
        previous = self.tracks.get(p.trackId)
        if previous is not None:
          first, last, n, _, (old_d, old_y, old_v) = previous
          dt = (frame - last) * 0.01
          if (0 < frame - last <= 15
              and abs(p.dRel - old_d - old_v * dt) <= 1.0 + 2.0 * dt):
            dy = abs(yRel - old_y)
            if dy <= 0.5 + 5.0 * dt:
              start, count = first, n + 1
            elif (p.trackId in self.selected[1:] and n >= 3 and last - first >= 15
                  and dy <= 0.75 + 5.0 * dt and abs(vRel - old_v) <= 1.0
                  and frame - self.lateral_grace.get(p.trackId, -1000) >= 30):
              # One small lateral discontinuity may keep an established side target.
              # Reset coordinate smoothing, not its selection eligibility.
              start, count = first, n + 1
              self.lateral_grace[p.trackId] = frame
              self.positions.pop(p.trackId, None)
        current[p.trackId] = (start, frame, count, p, (p.dRel, p.yRel, p.vRel))
    self.lateral_grace = {key: stamp for key, stamp in self.lateral_grace.items()
                          if key in current and current[key][0] <= stamp}
    self.tracks = current
    self._points = tuple(points)  # Snapshot values before point objects are reused by the next RadarData.
    for history in (self.recent_selected, self.recent_front):
      for key, (birth, selected_frame) in list(history.items()):
        if key not in current or current[key][0] != birth or frame - selected_frame > 30:
          del history[key]
    self.positions = {key: value for key, value in self.positions.items()
                      if key in current and value[0] == current[key][0]}

  def stable(self, track_id, frames=15):
    entry = self.tracks.get(track_id)
    return entry is not None and entry[2] >= 3 and entry[1] - entry[0] >= frames

  @staticmethod
  def speed_thresholds(v, a):
    if v != v:
      return (-100 if a < -3 else v), v, v
    front = (-100 if a < -3 or v <= 30 else 10.0 * (v - 30.0) - 100.0 if v < 40
             else (20.0 / 60.0) * (v - 40.0) if v < 100 else 20.0)
    side = (2.0 if v <= 0 else (8.0 / 30.0) * v + 2.0 if v < 30
            else (10.0 / 70.0) * (v - 30.0) + 10.0 if v < 100 else 20.0)
    low = -1.0 if v <= 10 else (11.0 / 30.0) * (v - 10.0) - 1.0 if v < 40 else 10.0
    return front, side, low

  def lane_available(self, probability, frame):
    if probability < 0.1:
      self.lane_ready = False
    elif probability >= 0.3:
      # Reliable geometry decides the slot immediately; smooth only the position.
      self.lane_ready = True
    return self.lane_ready

  def recently_selected(self, track_id, front_only=False):
    history = self.recent_front if front_only else self.recent_selected
    previous = history.get(track_id)
    current = self.tracks.get(track_id)
    return (previous is not None and current is not None and previous[0] == current[0]
            and 0 <= self.last_frame - previous[1] <= 30)

  def path_lead(self, md, minimum_speed, require_vision=False, front_only=False):
    if md is None:
      return None, 0.0
    self._update_model(md)
    if self._path_data is None:
      xs = np.asarray(md.position.x, dtype=np.float64)
      ys = np.asarray(md.position.y, dtype=np.float64)
      self._path_data = False
      if len(xs) == len(ys):
        stop = next((i for i in range(1, len(xs)) if xs[i] <= xs[i - 1]), len(xs))
        xs, ys = xs[:stop], ys[:stop]
        if _ccnc_valid_boundary(xs, ys):
          vision = None
          if len(md.leadsV3):
            lead = md.leadsV3[0]
            if len(lead.x) and len(lead.y) and len(lead.v):
              x, y, v = lead.x[0], lead.y[0], lead.v[0]
              if math.isfinite(x) and math.isfinite(y) and math.isfinite(v):
                vision = lead.prob, x - 1.52, y, v
          self._path_data = xs, ys, max(1.0, xs[0]), min(80.0, xs[-1]), vision
    if self._path_data is False:
      return None, 0.0
    xs, ys, minimum_distance, maximum_distance, vision = self._path_data
    if self._path_points is None or self._path_live is not self.live:
      entries = tuple(self.tracks.items())
      distances = np.fromiter((entry[3].dRel for _, entry in entries), dtype=np.float64, count=len(entries))
      aligned_values = np.interp(distances, xs, ys) - ys[0]
      self._path_points = tuple((track_id, entry, float(entry[3].yRel + aligned))
                                for (track_id, entry), aligned in zip(entries, aligned_values))
      self._path_live = self.live
    best, best_y, score = None, 0.0, math.inf
    recent = self.recently_selected
    for track_id, entry, aligned in self._path_points:
      first, last, count, p, _ = entry
      if count < 3 or last - first < 15:
        continue
      dRel, yRel = p.dRel, p.yRel
      if not minimum_distance <= dRel <= maximum_distance or p.vLead * CV.MS_TO_KPH <= minimum_speed:
        continue
      distance = dRel * dRel + yRel * yRel
      if distance >= score or (front_only and not recent(track_id, front_only=True)):
        continue
      retained = recent(track_id)
      vision_match = strong_vision_match = False
      if vision is not None:
        prob, vx, vy, vv = vision
        dx, dy, dv = abs(dRel - vx), abs(yRel + vy), abs(p.vLead - vv)
        vision_match = (prob >= (0.3 if retained else 0.5)
                        and dx <= max(8.0 if retained else 6.0, dRel * 0.2) and dy <= 1.5 and dv <= 5.0)
        strong_vision_match = (prob >= 0.8 and dRel <= 30.0
                               and dx <= max(3.0, dRel * 0.2) and dy <= 2.0 and dv <= 3.0)
        vision_match = vision_match or strong_vision_match
      if require_vision and not vision_match:
        continue
      corridor = 2.0 if strong_vision_match else 1.8 if retained else 1.2
      if not ((abs(aligned) <= corridor and (retained or vision_match))
              or (retained and vision_match and abs(aligned) <= 4.5)):
        continue
      best, best_y, score = p, aligned, distance
    return best, best_y

  def bridge_crossing(self, leads, lateral, probability, is_left, frame, thresholds):
    # Only previously lane-validated side targets may cross a short lane dropout.
    chosen = {p.trackId for p in leads if p is not None}
    for track_id, saved in list(self.crossing.items()):
      birth, stamp, distance, data, reference = saved
      entry = self.tracks.get(track_id)
      if (entry is None or entry[0] != birth or not 0 <= frame - stamp <= 200
          or self.stop_distance - distance > 30.0):
        del self.crossing[track_id]
        continue
      if probability >= 0.3:
        del self.crossing[track_id]
        continue
      if track_id in chosen:
        continue
      p = entry[3]
      if not 1 <= p.dRel <= 80 or not self.stable(track_id):
        del self.crossing[track_id]
        continue
      left, right = (-float(np.interp(p.dRel, *curve)) for curve in data[:2])
      y = p.yRel
      slot = 1 if y > left else 2 if y < right else 0
      curve = data[reference]
      aligned = y + float(np.interp(p.dRel, *curve)) - curve[1][0]
      speed = p.vLead * CV.MS_TO_KPH
      if (right >= left or abs(aligned) > 5.4
          or not (speed > thresholds[0] if slot == 0 else
                  speed > thresholds[1] or (p.dRel < 30 and speed > thresholds[2]))):
        del self.crossing[track_id]
        continue
      # Preserve the last validated road-edge clearance, including during FF entry.
      edge_left, edge_right = (-float(np.interp(p.dRel, *curve)) for curve in data[2:])
      margin = 0.25 if slot == 0 else 0.95
      if not edge_right + margin < y < edge_left - margin or leads[slot] is not None:
        del self.crossing[track_id]
        continue
      leads[slot], lateral[slot] = p, aligned
    if probability >= 0.3 and self._lane_data is not None:
      data, flags = self._lane_data
      if flags[2] and flags[3]:
        for p in leads[1:]:
          if p is not None and self.stable(p.trackId):
            self.crossing[p.trackId] = (self.tracks[p.trackId][0], frame, self.stop_distance,
                                        (data[0], data[1], data[4], data[5]), 0 if is_left else 1)
    return leads, lateral

  def filter_position(self, point, aligned_y, reference, frame):
    """Keep a physical track's filters across slots, before CAN sign/deadband mapping."""
    track_id = point.trackId
    observation = self.tracks.get(track_id)
    birth = observation[0] if observation is not None else frame
    previous = self.positions.get(track_id)
    if previous is None or previous[0] != birth or not 0 <= frame - previous[1] <= 15:
      distance = _CcncRadarPositionFilter(point.dRel, True)
      lateral = _CcncRadarPositionFilter(aligned_y, False)
      bias = 0.0
    else:
      _, last, old_reference, old_raw, old_input, bias, distance, lateral = previous
      if reference != old_reference:
        # Cancel the reference change, while retaining the radar's actual lateral motion.
        bias = old_input + (point.yRel - old_raw) - aligned_y
      if bias:
        step = (frame - last) * 0.03
        bias = math.copysign(max(0.0, abs(bias) - step), bias)
    filtered_input = aligned_y + bias
    self.positions[track_id] = (birth, frame, reference, point.yRel, filtered_input, bias, distance, lateral)
    return distance.apply(point.dRel), lateral.apply(filtered_input)

  def finish(self, ff, lf, rf, ff_y, lane_mode, frame):
    ids = (ff.trackId if ff is not None else None, lf.trackId if lf is not None else None, rf.trackId if rf is not None else None)
    old = self.selected
    changed = ids[0] != old[0], ids[1] != old[1], ids[2] != old[2]
    self.selected = ids
    for p in (ff, lf, rf):
      if p is not None and p.trackId in self.tracks:
        self.recent_selected[p.trackId] = (self.tracks[p.trackId][0], frame)
    if ff is not None and ff.trackId in self.tracks:
      self.recent_front[ff.trackId] = (self.tracks[ff.trackId][0], frame)
    return ff_y, changed

class _CcncRadarLaneSelector:
  """Keep a usable reference lane until the other is clearly better for 0.3s."""
  def __init__(self):
    self.is_left = None
    self.challenger_since = None
    self.last_frame = None

  def update(self, left_prob, right_prob, frame):
    # frame is the 100Hz CarController counter, not the model frameId.
    if self.last_frame is not None and (frame < self.last_frame or frame - self.last_frame > 100):
      self.is_left = None
      self.challenger_since = None
    self.last_frame = frame
    left_prob = left_prob if math.isfinite(left_prob) else 0.0
    right_prob = right_prob if math.isfinite(right_prob) else 0.0
    if max(left_prob, right_prob) < 0.1:
      self.is_left = None
      self.challenger_since = None
      return False
    preferred_left = left_prob > right_prob
    current_prob = left_prob if self.is_left else right_prob
    if self.is_left is None or current_prob < 0.1:
      self.is_left = preferred_left
      self.challenger_since = None
    elif preferred_left != self.is_left and abs(left_prob - right_prob) >= 0.1:
      if self.challenger_since is None:
        self.challenger_since = frame
      elif frame - self.challenger_since >= 30:
        self.is_left = preferred_left
        self.challenger_since = None
    else:
      self.challenger_since = None
    return self.is_left

class LaneHighlightStateMachine:
  def __init__(self):
    self.state = 0  # 현재 하이라이트 상태

  def update(self, accel_kph, drive_mode, v_ego_kph):
    # 상태별 전이 로직
    if self.state == 4:
      if accel_kph < -1.0 and v_ego_kph > 1.0:
        return self.state # 상태 유지
    elif self.state == 5:
      if drive_mode != 4 and accel_kph > 5.0:
        return self.state # 상태 유지

    # 진입 로직 (우선순위 순서)
    if accel_kph < -10.0:
      self.state = 4  # 급제동 (최우선)
    elif drive_mode == 4 or accel_kph > 10.0:
      self.state = 5  # 급가속/고속
    elif drive_mode < 3:
      self.state = 3  # 연비/안전
    else:
      self.state = 0  # 기본 상태 (회색 혹은 꺼짐)

    return self.state

class ThresholdTracker:
  def __init__(self, bounds, states):
    """
    :param bounds: (상한선, 하한선) 튜플
    :param states: (상한 이탈 시 상태, 하한 이탈 시 상태) 튜플
    :param initial_state: 객체 생성 시점의 초기 상태
    """
    self._upper_bound, self._lower_bound = bounds
    self._state_high, self._state_low = states
    self._current_state = self._state_high

  def apply(self, value):
    """
    입력값에 따라 상태를 업데이트하고 반환
    """
    if value > self._upper_bound:
        self._current_state = self._state_high
    elif value < self._lower_bound:
        self._current_state = self._state_low

    # 두 기준값 사이(박스권)에 있을 때는 기존 상태를 유지
    return self._current_state

class NoiseFilter:
  """
  고정/스텝/가변 알파를 지원하는 Median + LPF 통합 필터.
  - alpha_range: 필수 입력 (None 허용 안 함).
  - error_range: 선택 입력 (None일 경우 고정 알파 모드).
  """
  def __init__(self, median_buffer_size, lowpass_default, alpha_range, error_range=None):
    self._default_value = lowpass_default
    self._filtered_value = lowpass_default
    self._buffer = deque([lowpass_default] * median_buffer_size, maxlen=median_buffer_size)

    def normalize_range(r, default_val):
      # 리스트/튜플에서 첫 번째와 마지막 값 추출 (1개일 때도 r[0]==r[-1]로 안전)
      if isinstance(r, (int, float)): return [float(r), float(r)]
      if isinstance(r, (list, tuple)) and len(r) >= 1:
        return [float(r[0]), float(r[-1])]
      return [default_val, default_val]

    # 1. 알파 설정 (필수 입력값 정문화)
    norm_alpha = normalize_range(alpha_range, 1.0)
    self._a_min, self._a_max = [np.clip(v, 0.001, 1.0) for v in norm_alpha]

    # 2. 에러 범위 및 전략 할당
    if error_range is not None:
      self._err_min, self._err_max = normalize_range(error_range, 0.0)

      # 전략 선택: 경계값이 같으면 Step(리셋), 다르면 Adaptive(보간)
      if self._err_min == self._err_max:
        self.apply = self._apply_step
      else:
        self.apply = self._apply_adaptive
    else:
      # error_range가 None이면 리셋 로직 없이 고정 알파 필터로 동작
      self._alpha = self._a_min
      self.apply = self._apply_fixed

  def reset(self, new_value=None):
    """값을 초기화하고 버퍼를 완전히 비웁니다."""
    self._filtered_value = new_value if new_value is not None else self._default_value
    self._buffer.clear()
    return self._filtered_value

  def fill(self, value):
    """
    현재 필터의 버퍼를 특정 값으로 가득 채우고 필터 출력값도 동기화합니다.
    reset과 달리 내부 설정값(default_value 등)은 유지하며 데이터 흐름만 강제 수정합니다.
    """
    self._filtered_value = value
    self._buffer.extend([self._filtered_value] * self._buffer.maxlen)

    return self._filtered_value

  def update_alpha(self, new_alpha):
    """
    고정 알파 모드에서 알파 값을 업데이트합니다.
    """
    if self.apply == self._apply_fixed:
      self._alpha = np.clip(new_alpha, 0.001, 1.0)

  def reset_alpha(self):
    """
    고정 알파 모드에서 알파 값을 초기값으로 되돌립니다.
    """
    if self.apply == self._apply_fixed:
      self._alpha = self._a_min

  def _get_median(self):
    buf_len = len(self._buffer)
    if buf_len == 0:
        return self._filtered_value # 버퍼가 비어있으면 현재 필터값 반환
    # 꽉 차지 않은 상태(초기 진입 시)에도 현재 데이터 개수 기준으로 중간값 산출
    return sorted(self._buffer)[buf_len // 2]

  def _check_hard_reset(self, target):
    step_err = abs(target - self._filtered_value)
    if step_err > self._err_max:
      self.reset(target)  # 버퍼 비우고 target으로 즉시 리셋
      return True

    return False

  def _apply_fixed(self, target):
    self._buffer.append(target)
    med = self._get_median()
    self._filtered_value = (self._alpha * med) + ((1.0 - self._alpha) * self._filtered_value)
    return self._filtered_value

  def _apply_step(self, target):
    if self._check_hard_reset(target):
      return self._filtered_value

    self._buffer.append(target)
    med = self._get_median()

    self._filtered_value = (self._a_min * med) + ((1.0 - self._a_min) * self._filtered_value)
    return self._filtered_value

  def _apply_adaptive(self, target):
    if self._check_hard_reset(target):
      return self._filtered_value

    self._buffer.append(target)
    med = self._get_median()

    track_err = abs(med - self._filtered_value)

    # 오차 정도에 따라 알파값 보간
    alpha = float(np.interp(track_err, [self._err_min, self._err_max], [self._a_min, self._a_max]))
    self._filtered_value = (alpha * med) + ((1.0 - alpha) * self._filtered_value)

    return self._filtered_value

  @property
  def value(self):
    return self._filtered_value

class _CcncRadarPositionFilter(NoiseFilter):
  def __init__(self, value, distance):
    self._default_value = self._filtered_value = value
    self._buffer = deque((value, value, value), maxlen=3)
    self._a_min = 0.3
    self._a_max = 0.9 if distance else 0.3
    self._err_min = 1.0 if distance else 0.6
    self._err_max = 4.0 if distance else 0.6
    self.apply = self._apply_adaptive if distance else self._apply_step

  def _apply_adaptive(self, target):
    if self._check_hard_reset(target):
      return self._filtered_value
    self._buffer.append(target)
    med = self._get_median()
    error = abs(med - self._filtered_value)
    alpha = 0.3 if error <= 1.0 else 0.9 if error >= 4.0 else ((0.9 - 0.3) / 3.0) * (error - 1.0) + 0.3
    self._filtered_value = alpha * med + (1.0 - alpha) * self._filtered_value
    return self._filtered_value

def ease_in_interp(x, x_range, y_range, power=2):
  # x를 0~1 사이 비율로 변환
  t = (x - x_range[0]) / (x_range[1] - x_range[0])
  t = max(0, min(1, t)) # 범위 제한

  # Ease-in 적용
  eased_t = t ** power

  # 결과값 매핑
  return y_range[0] + (y_range[1] - y_range[0]) * eased_t

def apply_curved_deadband(value, center, radius, degree=2):
    # 1. 입력의 부호(+1 또는 -1) 추출
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)

    # 2. 중심(center)으로부터의 거리 계산
    diff = abs_val - center

    # 3. 데드밴드 영향 범위를 벗어나면 원본 값 그대로 반환
    # (center 기준 radius 이상 멀어지거나, center 안쪽으로 radius 이상 들어간 경우)
    if diff >= radius or diff <= -radius:
        return value

    # 4. 곡선 보간 비율 t 계산 (0 ~ 1 범위로 정규화)
    # diff = 0 (center 위치)일 때 t = 0 (완전 감쇠)
    # diff = radius 또는 -radius 일 때 t = 1 (원본 유지)
    t = abs(diff) / radius

    # 5. 곡선 가중치 적용
    weight = t ** degree

    # 6. center 위치로 당겨주는 보간 수행 후, 원본 부호 복원
    smoothed_abs = (1.0 - weight) * center + (weight * abs_val)

    return sign * smoothed_abs

def update_speed_limit(values, CS, cruise_enabled):
  if cruise_enabled:
    if CS.out.vCruiseCluster > values["vSetDis"]:
      if state.sla_active_time < 1:
        state.sla_active_time = time.monotonic()
      values["SETSPEED"] = 2
      values["SETSPEED_HUD"] = 2
      elapsed = time.monotonic() - state.sla_active_time
      values["SLA_ICON"] = 2 if (elapsed % 3.5) < 2.0 else 0
    else:
      state.sla_active_time = 0
      if CS.ccnc_0x162 is not None and values["SLA_ICON"] > 0:
        if CS.ccnc_0x162["SPEEDLIMIT"] > CS.out.vCruiseCluster:
          values["SLA_ICON"] = 3
        elif CS.ccnc_0x162["SPEEDLIMIT"] < CS.out.vCruiseCluster:
          values["SLA_ICON"] = 4
        else:
          values["SLA_ICON"] = 0
  else:
    state.sla_active_time = 0

def update_lfa_icon(values, CS, lat_enabled, lat_active, hdp_active):
  if lat_enabled:
    if lat_active:
      if CS.out.steeringPressed:
        # 횡컨 활성 중 운전자 조향 개입
        values["LFA_ICON"] = 3  # WHITE
      else:
        # 능동 제어 중 (HDP: 청색, 일반: 녹색)
        values["LFA_ICON"] = 5 if hdp_active else 2
    else:
      # 횡컨 켜짐 + 모델 미제어 (대기 상태)
      values["LFA_ICON"] = 1  # GRAY
  else:
    # 횡컨 OFF -> 아이콘 숨김
    values["LFA_ICON"] = 0

def update_lanes(values, CS, md, v_ego_kph, a_ego_kph, desire, lat_active, lat_enabled, lane_color=True, model_lanes=True):
  # 주행 기어에서만 가속도·드라이브 모드에 따른 차로 색 변경
  if lane_color and CS.out.gearShifter == structs.CarState.GearShifter.drive:
    now = time.monotonic()
    if now >= state.drive_mode_refresh:
      try:
        state.drive_mode = Params().get_int("MyDrivingMode")
      except Exception:
        state.drive_mode = 3
      state.drive_mode_refresh = now + 1.0
    drive_mode = state.drive_mode

    # 속도에 비례해 하이라이트 길이 동적으로 조절
    values["LANE_HIGHLIGHT_DISTANCE"] = int(ease_in_interp(v_ego_kph, [0, 80], [3, 60], power=1.5))
    values["LANE_HIGHLIGHT"] = state.drive_lane_color.update(a_ego_kph, drive_mode, v_ego_kph)

  # 차선 변경 판단
  is_auto_lane_changing = desire in (3, 4)
  is_blinking = CS.out.leftBlinker != CS.out.rightBlinker
  is_currently_lane_changing = model_lanes and (is_auto_lane_changing or (is_blinking and v_ego_kph > 20.0))

  if model_lanes:
    try:
      if lat_enabled:
        # 스칼라 np.interp 오버헤드 제거 (선형 보간 수식 직접 계산: 20~100 kph -> 30~80 m)
        max_lookahead_x = 30.0 + min(max((v_ego_kph - 20.0) / 80.0, 0.0), 1.0) * 50.0

        # 객체 속성 접근 오버헤드 캐싱 (루프 내 다중 점근 방지)
        pos = md.position
        pos_x, pos_y, pos_y_std = pos.x, pos.y, pos.yStd

        trust_threshold = 0.8
        max_y_abs = 0.0
        peak_idx = 0
        start_search_idx = 0
        start_found = not is_currently_lane_changing
        min_calc_dist = 20.0 if is_currently_lane_changing else 0.0

        for i in range(1, len(pos_x)):
          x = pos_x[i]

          if not start_found and x >= min_calc_dist:
            start_search_idx = i
            start_found = True

          if pos_y_std[i] > trust_threshold or x > max_lookahead_x:
            break

          y_abs = abs(pos_y[i])
          if y_abs > max_y_abs:
            max_y_abs = y_abs
            peak_idx = i

        if start_search_idx != peak_idx and pos_x[peak_idx] >= (20.0 + min_calc_dist):
          x_dist = pos_x[peak_idx]
          y_diff = pos_y[peak_idx] - pos_y[start_search_idx]
          # 곡률 공식: (2y / x^2) * 1800 -> (3600 * y) / x^2
          max_curve_val = (3600.0 * y_diff) / (x_dist * x_dist)
        else:
          max_curve_val = 0.0

        curvature = round(state.lane_curv.apply(-max_curve_val))
      else:
        curvature = round(CS.out.steeringAngleDeg / 3)

    except:
      # 모델 데이터 예외 발생 시 핸들 각도 기반 백업
      curvature = round(CS.out.steeringAngleDeg / 3)
      values["LFA_ICON"] = 5

    values["LANELINE_CURVATURE"] = min(abs(curvature), 15) + (-1 if curvature < 0 else 0)
    values["LANELINE_CURVATURE_DIRECTION"] = 1 if curvature < 0 else 0

  if not model_lanes:
    return

  try:
    # 차선 위치 갱신: 항시 적용
    l_prob = md.laneLineProbs[1]
    r_prob = md.laneLineProbs[2]

    # --- 차선 변경 상태 관리 및 알파값 조정 ---
    if is_currently_lane_changing != state._is_lane_change_active:
      if is_currently_lane_changing:
        state.l_lane_f.update_alpha(0.6)
        state.r_lane_f.update_alpha(0.6)
      else:
        state.l_lane_f.reset_alpha()
        state.r_lane_f.reset_alpha()
      state._is_lane_change_active = is_currently_lane_changing

    leftlaneraw = abs(md.laneLines[1].y[0])
    rightlaneraw = abs(md.laneLines[2].y[0])

    l_valid = l_prob > 0.3 or is_auto_lane_changing or is_blinking
    r_valid = r_prob > 0.3 or is_auto_lane_changing or is_blinking

    if not l_valid and not r_valid:
      leftlaneraw = rightlaneraw = 1.5
    elif not l_valid:
      leftlaneraw = state.last_known_lane_width - rightlaneraw
    elif not r_valid:
      rightlaneraw = state.last_known_lane_width - leftlaneraw

    if is_currently_lane_changing:
      is_moving_left = CS.out.leftBlinker or desire == 3
      # 위상 변화 시 차선 강조 변경
      if not state.draw_center:

        lane_raw = leftlaneraw if is_moving_left else rightlaneraw

        is_phase_shifted = lane_raw < 0.1 or (lane_raw - state.lane_phase_min) > 0.3
        state.lane_phase_min = min(state.lane_phase_min, lane_raw)

        if is_phase_shifted:
          state.draw_center = state.hold_lane = True
          state.lane_phase_min = 6.0
          state.hold_lane_escape_count = 0

          prev_l_val = state.l_lane_f.value
          prev_r_val = state.r_lane_f.value
          if is_moving_left:
            state.r_lane_f.reset(prev_l_val)
          else:
            state.l_lane_f.reset(prev_r_val)

      # RNN 보간 방지
      if state.hold_lane:
        swapped_lane_position = rightlaneraw if is_moving_left else leftlaneraw

        # 줄어드는 최솟값을 지속적으로 갱신
        state.lane_phase_min = min(state.lane_phase_min, swapped_lane_position)

        # 최솟값 대비 0.1m 이상 반등하면 작아지다 커지는 위상으로 판단
        if swapped_lane_position - state.lane_phase_min > 0.1:
          state.hold_lane_escape_count += 1
          if state.hold_lane_escape_count >= 2:
            state.hold_lane = False
        else:
          state.hold_lane_escape_count = 0

        holding_factor = state.hold_lane_escape_count * 0.1
        if is_moving_left:
          current_l_target = state.l_lane_f.reset(state.last_known_lane_width - holding_factor)
          current_r_target = state.r_lane_f.reset(holding_factor)
        else:
          current_l_target = state.l_lane_f.reset(holding_factor)
          current_r_target = state.r_lane_f.reset(state.last_known_lane_width - holding_factor)
      elif state.draw_center:
        MAX_STEP = 0.15  # 한 루프(프레임)당 최대 허용 변화량 (m단위, 부드러움 조절용)
        prev_l = state.l_lane_f.value
        prev_r = state.r_lane_f.value
        # 실제 값과 이전 값의 차이를 MAX_STEP 이내로 제한 (클리핑)
        bounded_l = prev_l + min(max(leftlaneraw - prev_l, -MAX_STEP), MAX_STEP)
        bounded_r = prev_r + min(max(rightlaneraw - prev_r, -MAX_STEP), MAX_STEP)
        current_l_target = state.l_lane_f.apply(bounded_l)
        current_r_target = state.r_lane_f.apply(bounded_r)
      else:
        current_l_target = state.l_lane_f.apply(leftlaneraw)
        current_r_target = state.r_lane_f.apply(rightlaneraw)

      # LCA 중에는 차로 강조
      if is_auto_lane_changing:
        if state.draw_center:
          values["LANE_HIGHLIGHT"] = 1
          values["LANE_HIGHLIGHT_DISTANCE"] = 60
        else:
          values["LANE_LEFT" if desire == 3 else "LANE_RIGHT"] = 1
      elif abs(current_l_target - current_r_target) < state.last_known_lane_width / 5:
        state.draw_center = state.hold_lane = False
        state.hold_lane_escape_count = 0
        state.lane_phase_min = 10.0
    else:
      state.draw_center = state.hold_lane = False
      state.hold_lane_escape_count = 0
      state.lane_phase_min = 10.0
      current_l_target = state.l_lane_f.apply(leftlaneraw)
      current_r_target = state.r_lane_f.apply(rightlaneraw)

      lane_width = current_l_target + current_r_target
      if 2 < lane_width < 4.6:
        state.last_known_lane_width = lane_width # 마지막 차선 폭을 기억해둠

    if model_lanes:
      values["LANELINE_LEFT_POSITION"] = int(round(min(max(current_l_target, 0.0), 3.0) * 10.0))
      values["LANELINE_RIGHT_POSITION"] = int(round(min(max(current_r_target, 0.0), 3.0) * 10.0))

    # 차선 변경 아이콘
    if model_lanes and lat_enabled:
      values["LCA_LEFT_ICON"] = 1 if CS.out.leftBlindspot else 4 if CS.out.rightBlinker or not md.meta.laneChangeAvailableLeft else 2
      values["LCA_RIGHT_ICON"] = 1 if CS.out.rightBlindspot else 4 if CS.out.leftBlinker or not md.meta.laneChangeAvailableRight else 2
  except:
    # Only show startup status while the required model lane data is absent.
    # Calculation errors with populated lane data keep the original alert.
    if (md is None or len(md.laneLineProbs) < 3 or len(md.laneLines) < 3 or
        len(md.laneLines[1].y) == 0 or len(md.laneLines[2].y) == 0):
      values["ALERTS_5"] = 19  # ACTIVATING_HIGHWAY_DRIVING_PILOT_SYSTEM
    else:
      if model_lanes:
        values["LANELINE_LEFT_POSITION"] = 30
        values["LANELINE_RIGHT_POSITION"] = 30
      if model_lanes:
        values["LANE_HIGHLIGHT"] = 3
        values["LANE_HIGHLIGHT_DISTANCE"] = 60
        values["LANE_LEFT"] = 1
        values["LANE_RIGHT"] = 1
        values["LKA_ICON"] = 1

def update_vehicles(values, CS, md, frame, v_ego_kph, a_ego_kph, model_lanes=True):
  # --- liveTracks 원본 레이더를 이용한 전방 차량 감지 ---
  try:
    ff_lead = lf_lead = rf_lead = None
    ff_yRel = lf_yRel = rf_yRel = 0
    boundary_front = None
    boundary_front_y = 0.0

    display_tracker = state.radar_display_tracker
    display_tracker.observe(CS.live_tracks, frame)
    display_tracker.update_stop(v_ego_kph, frame, a_ego_kph)
    left_prob, right_prob = display_tracker.lane_probabilities(md)
    selected_lane_is_left = state.radar_lane_selector.update(left_prob, right_prob, frame)
    selected_lane_prob = left_prob if selected_lane_is_left else right_prob

    # LF/RF deadband 중심은 차량 존재 여부와 무관하게 인접 차선 geometry로 계속 갱신합니다.
    lf_center = _ccnc_side_lane_center(md, 0)
    rf_center = _ccnc_side_lane_center(md, 1)

    if lf_center is not None:
      state.lf_center.apply(lf_center)

    if rf_center is not None:
      state.rf_center.apply(rf_center)

    # 차선이 유효하면 기존 분류를 사용하고, FF는 차선 소실 시 경로/영상으로 보완합니다.
    lane_mode = selected_lane_prob >= 0.1
    ff_lane_mode = display_tracker.lane_available(selected_lane_prob, frame)
    ff_uses_lane = True
    min_front_lead_speed, min_side_lead_speed, lowspeed_side_lead_speed = display_tracker.speed_thresholds(v_ego_kph, a_ego_kph)
    if CS.live_tracks is not None and lane_mode:
      left_y0, right_y0, projected = display_tracker.lane_projection(CS.live_tracks)
      selected_lane_y0 = left_y0 if selected_lane_is_left else right_y0
      interp = np.interp
      ms_to_kph = CV.MS_TO_KPH
      # 여러 차로의 후보를 허용하되, 보정 후 횡거리 5.4m 밖의 측면 점은 선택하지 않습니다.
      max_side_lateral = 5.4
      max_side_distance = 80.0
      ff_min_dist = lf_min_dist = rf_min_dist = math.inf
      left_projection = right_projection = None

      for point_index, ((lead, dRel, yRel, vRel, vLead), (left_y, right_y)) in enumerate(zip(display_tracker._points, projected)):
        velocity = vLead * ms_to_kph
        # FF와 저속 측면 예외까지 모두 탈락하는 점은 보간 전에 제외합니다.
        if not (velocity > min_front_lead_speed or velocity > min_side_lead_speed
                or (dRel < 30 and velocity > lowspeed_side_lead_speed)):
          continue

        # 차선 보간값과 시작점의 차이로 도로의 휘어짐만 보정합니다.
        lane_y_at_drel = left_y if selected_lane_is_left else right_y
        road_aligned_yRel = yRel + (lane_y_at_drel - selected_lane_y0)

        # 분류는 원본 레이더 좌표(좌측+)와 같은 좌표계의 차선 경계로 판단합니다.
        left_inner_bound, right_inner_bound = -left_y, -right_y
        if (not math.isfinite(left_inner_bound) or not math.isfinite(right_inner_bound)
            or right_inner_bound >= left_inner_bound):
          continue

        # 거리 순위만 필요하므로 원본 좌표의 제곱거리로 비교합니다.
        dist_score = dRel * dRel + yRel * yRel

        if (lead.trackId == display_tracker.selected[0]
            and display_tracker.recently_selected(lead.trackId, front_only=True)
            and velocity > min_front_lead_speed
            and right_inner_bound - 0.35 <= yRel <= left_inner_bound + 0.35):
          boundary_front, boundary_front_y = lead, road_aligned_yRel

        # 2. [전방 주행 차선] - 외곽선/도로경계선 interp 4회 전부 생략
        if right_inner_bound <= yRel <= left_inner_bound:
          if dist_score < ff_min_dist:
            if velocity > min_front_lead_speed:
              ff_min_dist, ff_lead, ff_yRel = dist_score, lead, road_aligned_yRel

        # 3. [왼쪽 차선 차량] - 좌측 외곽/도로경계선만 지연 계산 (우측 2회 interp 생략)
        elif left_inner_bound < yRel:
          if (dRel <= max_side_distance and display_tracker.stable(lead.trackId)
              and abs(road_aligned_yRel) <= max_side_lateral and dist_score < lf_min_dist):

            # 속도 조건을 통과한 모든 후보에 외곽 차선/도로 경계 검사를 적용합니다.
            if (velocity > min_side_lead_speed
                or (dRel < 30 and velocity > lowspeed_side_lead_speed and display_tracker.stable(lead.trackId, 50))):
              if left_projection is None:
                left_projection = display_tracker.side_projection(0)
              outer_left_y, edge_left_y = left_projection[point_index]
              # Expand the outer lane for moving candidates, keeping road-edge clearance.
              lane_margin = 0.75 if velocity > min_side_lead_speed and display_tracker.stable(lead.trackId, 30) else -0.25
              if lead.trackId == display_tracker.selected[1]:
                lane_margin += 0.3
              left_width_bound = math.inf
              left_effective_bound = math.inf
              if math.isfinite(outer_left_y):
                outer_bound = -outer_left_y
                left_width_bound = outer_bound
                left_effective_bound = outer_bound + lane_margin
              if math.isfinite(edge_left_y):
                edge_bound = -edge_left_y
                left_width_bound = min(left_width_bound, edge_bound)
                left_effective_bound = min(left_effective_bound, edge_bound - 0.95)

              if left_width_bound != math.inf:
                # Preserve the original width check before expanding candidate acceptance.
                left_width_bound = left_width_bound - 0.25
                if (yRel < left_effective_bound and left_width_bound - left_inner_bound > 1.8
                    and display_tracker.side_entry_width(lead, 0)):
                  lf_min_dist, lf_lead, lf_yRel = dist_score, lead, road_aligned_yRel

        # 4. [오른쪽 차선 차량] - 우측 외곽/도로경계선만 지연 계산 (좌측 2회 interp 생략)
        elif yRel < right_inner_bound:
          if (dRel <= max_side_distance and display_tracker.stable(lead.trackId)
              and abs(road_aligned_yRel) <= max_side_lateral and dist_score < rf_min_dist):

            # 속도 조건을 통과한 모든 후보에 외곽 차선/도로 경계 검사를 적용합니다.
            if (velocity > min_side_lead_speed
                or (dRel < 30 and velocity > lowspeed_side_lead_speed and display_tracker.stable(lead.trackId, 50))):
              if right_projection is None:
                right_projection = display_tracker.side_projection(1)
              outer_right_y, edge_right_y = right_projection[point_index]
              # Expand the outer lane for moving candidates, keeping road-edge clearance.
              lane_margin = 0.75 if velocity > min_side_lead_speed and display_tracker.stable(lead.trackId, 30) else -0.25
              if lead.trackId == display_tracker.selected[2]:
                lane_margin += 0.3
              right_width_bound = -math.inf
              right_effective_bound = -math.inf
              if math.isfinite(outer_right_y):
                outer_bound = -outer_right_y
                right_width_bound = outer_bound
                right_effective_bound = outer_bound - lane_margin
              if math.isfinite(edge_right_y):
                edge_bound = -edge_right_y
                right_width_bound = max(right_width_bound, edge_bound)
                right_effective_bound = max(right_effective_bound, edge_bound + 0.95)

              if right_width_bound != -math.inf:
                # Preserve the original width check before expanding candidate acceptance.
                right_width_bound = right_width_bound + 0.25
                if (yRel > right_effective_bound and right_inner_bound - right_width_bound > 1.8
                    and display_tracker.side_entry_width(lead, 1)):
                  rf_min_dist, rf_lead, rf_yRel = dist_score, lead, road_aligned_yRel

    if CS.live_tracks is not None and (not ff_lane_mode or selected_lane_prob < 0.5):
      minimum_speed = min_front_lead_speed
      # During uncertain lane recovery, only a vision-matched recent FF may
      # displace a lane candidate. A failed fallback preserves a recently displayed lane candidate.
      fallback_lead, fallback_y = display_tracker.path_lead(
        md, minimum_speed, require_vision=ff_lane_mode,
        front_only=ff_lane_mode and ff_lead is not None)
      if fallback_lead is not None and (
          ff_lead is None or not ff_lane_mode
          or fallback_lead.dRel ** 2 + fallback_lead.yRel ** 2 <= ff_lead.dRel ** 2 + ff_lead.yRel ** 2):
        ff_lead, ff_yRel = fallback_lead, fallback_y
        ff_uses_lane = False
      elif (not ff_lane_mode and ff_lead is not None
            and not display_tracker.recently_selected(ff_lead.trackId, front_only=True)):
        # Weak lanes alone must not introduce a new unconfirmed front target.
        ff_lead = None

    # Only bridge an empty slot; a real side transition or new FF wins immediately.
    if (ff_lead is None and boundary_front is not None
        and (lf_lead is None or lf_lead.trackId != boundary_front.trackId)
        and (rf_lead is None or rf_lead.trackId != boundary_front.trackId)):
      ff_lead, ff_yRel = boundary_front, boundary_front_y
      ff_uses_lane = True

    normal_ff = ff_lead
    (ff_lead, lf_lead, rf_lead), (ff_yRel, lf_yRel, rf_yRel) = display_tracker.bridge_crossing(
      [ff_lead, lf_lead, rf_lead], [ff_yRel, lf_yRel, rf_yRel], selected_lane_prob,
      selected_lane_is_left, frame, (min_front_lead_speed, min_side_lead_speed, lowspeed_side_lead_speed))
    if ff_lead is not normal_ff:
      ff_uses_lane = True

    # A retained FF must not also occupy a side slot during lane recovery.
    if ff_lead is not None:
      if lf_lead is not None and lf_lead.trackId == ff_lead.trackId:
        lf_lead = None
      if rf_lead is not None and rf_lead.trackId == ff_lead.trackId:
        rf_lead = None
    # Shared physical coordinates and filter history follow the ID across LF/FF/RF.
    filtered_positions = []
    for point, aligned, is_lane in ((ff_lead, ff_yRel, ff_uses_lane),
                                    (lf_lead, lf_yRel, True), (rf_lead, rf_yRel, True)):
      reference = ("lane", selected_lane_is_left) if is_lane else ("path", False)
      filtered_positions.append(display_tracker.filter_position(point, aligned, reference, frame)
                                if point is not None else (0.0, 0.0))
    ff_yRel = filtered_positions[0][1]
    lf_yRel = min(max(float(filtered_positions[1][1]), -5.4), 5.4)
    rf_yRel = min(max(float(filtered_positions[2][1]), -5.4), 5.4)
    ff_yRel, changed_tracks = display_tracker.finish(ff_lead, lf_lead, rf_lead, ff_yRel, ff_lane_mode, frame)

    # 전방(FF) 차량 정보 업데이트
    if ff_lead:
      values["FF_DISTANCE"] = filtered_positions[0][0] * 0.8
      values["FF_LATERAL"] = apply_curved_deadband(-ff_yRel, 0, 0.7, 1)
      values["FF_DETECT"] = 2 if ff_lead.vLead < 3 else state.ff_detect.apply(ff_lead.vRel)
    else:
      values["FF_DETECT"] = 0 # 순정 디텍션 제거
    # LF/RF 횡거리는 4m로 압축하지 않고 CAN 신호 범위(7-bit unsigned, 0.1m)만 제한합니다.
    # 전방 좌측(LF) 차량 정보 업데이트
    if lf_lead:
      values["LF_DETECT_DISTANCE"] = filtered_positions[1][0] * 0.8
      values["LF_DETECT_LATERAL"] = min(max(float(apply_curved_deadband(lf_yRel, state.lf_center.value, 0.9, 2)), 0.0), 12.7)
      values["LF_DETECT"] = state.lf_detect.apply(lf_lead.vRel)
    else:
      values["LF_DETECT"] = 0
    # 전방 우측(RF) 차량 정보 업데이트
    if rf_lead:
      values["RF_DETECT_DISTANCE"] = filtered_positions[2][0] * 0.8
      values["RF_DETECT_LATERAL"] = min(max(float(apply_curved_deadband(-rf_yRel, state.rf_center.value, 0.9, 2)), 0.0), 12.7)
      values["RF_DETECT"] = state.rf_detect.apply(rf_lead.vRel)
    else:
      values["RF_DETECT"] = 0

    display_tracker.stopped_display(values, (lf_lead, rf_lead), frame)

    center_lane_offset = (state.r_lane_f.value - state.l_lane_f.value) / 2 if model_lanes else 0.0

    # --- 후측방은 BSD 경고 시 고정 위치에 두부 출력. HDA1은 후측방 레이더 정보가 안채워져서 옴 ---
    BSD_LATERAL_FIXED = 2.8
    if CS.out.leftBlindspot:
      values["LR_DETECT_DISTANCE"] = state.lr_distance.apply(8)
      values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
      values["LR_DETECT"] = 2
    elif state.lr_distance.value < 15:
      values["LR_DETECT_DISTANCE"] = state.lr_distance.apply(16)
      values["LR_DETECT_LATERAL"] = BSD_LATERAL_FIXED - center_lane_offset
      values["LR_DETECT"] = 1

    if CS.out.rightBlindspot:
      values["RR_DETECT_DISTANCE"] = state.rr_distance.apply(8) # 8m
      values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
      values["RR_DETECT"] = 2
    elif state.rr_distance.value < 15:
      values["RR_DETECT_DISTANCE"] = state.rr_distance.apply(16)
      values["RR_DETECT_LATERAL"] = BSD_LATERAL_FIXED + center_lane_offset
      values["RR_DETECT"] = 1

  except:
    values = CS.ccnc_0x162.copy()
    values["FF_DISTANCE"] = 24
    values["FF_DETECT"] = 7
    values["LF_DETECT_DISTANCE"] = 12
    values["LF_DETECT_LATERAL"] = 1.5
    values["LF_DETECT"] = 4
    values["RF_DETECT_DISTANCE"] = 12
    values["RF_DETECT_LATERAL"] = 1.5
    values["RF_DETECT"] = 9
    values["LR_DETECT_DISTANCE"] = 1
    values["LR_DETECT_LATERAL"] = 3
    values["LR_DETECT"] = 6
    values["RR_DETECT_DISTANCE"] = 1
    values["RR_DETECT_LATERAL"] = 3
    values["RR_DETECT"] = 11
  return values

state = SimpleNamespace()
_options = None


def configure(lane_color, model_lanes, radar_vehicles):
  global _options
  options = (lane_color, model_lanes, radar_vehicles)
  if options == _options:
    return
  if _options is None or lane_color != _options[0]:
    state.drive_lane_color = LaneHighlightStateMachine()
    state.drive_mode = 3
    state.drive_mode_refresh = -math.inf
  if _options is None or model_lanes != _options[1]:
    reset_lanes()
  if _options is None or radar_vehicles != _options[2]:
    reset_vehicles()
  _options = options


def reset_lanes():
  state.sla_active_time = 0
  state.lane_curv = NoiseFilter(3, 0, alpha_range=0.5)
  state._is_lane_change_active = False
  state.draw_center = state.hold_lane = False
  state.hold_lane_escape_count = 0
  state.lane_phase_min = 10.0
  state.last_known_lane_width = 3.0
  state.l_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2)
  state.r_lane_f = NoiseFilter(3, 1.5, alpha_range=0.2)

  # LF/RF 표시 중심
  state.lf_center = NoiseFilter(3, 3.0, alpha_range=0.03)
  state.rf_center = NoiseFilter(3, 3.0, alpha_range=0.03)


def reset_vehicles():
  state.radar_lane_selector = _CcncRadarLaneSelector()
  state.radar_display_tracker = _CcncRadarDisplayTracker()
  state.ff_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))
  state.lf_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))
  state.rf_detect = ThresholdTracker(bounds=(2, -1), states=(1, 2))
  state.lr_distance = NoiseFilter(1, 15, alpha_range=0.05)
  state.rr_distance = NoiseFilter(1, 15, alpha_range=0.05)


def reset():
  global _options
  _options = None
  state.__dict__.clear()
  state.sla_active_time = 0
  configure(False, False, False)


reset()
