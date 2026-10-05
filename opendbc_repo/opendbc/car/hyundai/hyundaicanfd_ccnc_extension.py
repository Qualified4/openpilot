"""CCNC display extension; baseline CAN handling stays in hyundaicanfd.

Lane/color effects are also used by HDA2. Front-radar vehicle display is HDA1-only.
"""
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

def _ccnc_side_lane_center(md, side, curve=None, min_probability=0.6):
  inner_idx = 1 if side == 0 else 2
  outer_idx = 0 if side == 0 else 3

  if md is None or len(md.laneLines) < 4 or len(md.laneLineProbs) < 4:
    return None

  if not all(math.isfinite(md.laneLineProbs[i]) and md.laneLineProbs[i] >= min_probability
             for i in (inner_idx, outer_idx)):
    return None

  if curve is None:
    inner, outer = md.laneLines[inner_idx], md.laneLines[outer_idx]
    inner_xs, inner_ys, outer_xs, outer_ys = inner.x, inner.y, outer.x, outer.y
    valid = _ccnc_valid_boundary(inner_xs, inner_ys) and _ccnc_valid_boundary(outer_xs, outer_ys)
  else:
    (inner_xs, inner_ys, inner_valid), (outer_xs, outer_ys, outer_valid) = curve(md, inner_idx), curve(md, outer_idx)
    valid = inner_valid and outer_valid
  if not valid:
    return None

  x = 20.0

  if not (inner_xs[0] <= x <= inner_xs[-1]
          and outer_xs[0] <= x <= outer_xs[-1]):
    return None

  inner_y = float(np.interp(x, inner_xs, inner_ys))
  outer_y = float(np.interp(x, outer_xs, outer_ys))

  # Model y is positive to the right; reversed boundaries are not a valid lane.
  width = (inner_y - outer_y) if side == 0 else (outer_y - inner_y)

  if not 2.3 <= width <= 4.8:
    return None

  center = abs(float(inner_ys[0])) + width * 0.5
  return center if math.isfinite(center) else None

class _CcncTemporalTracks:
  """Bounded display-only observations; never change the published radar data."""
  HOLD = 30  # 300 ms since the last accepted position, not since the last prediction.
  RECONNECT = 35

  def __init__(self):
    self.entries = {}
    self.raw = self.view = None
    self.stamp = -1000
    self.next_id = 1 << 32  # Avoid alias collisions when a radar slot is reused.
    self.selection = [None, None, None]
    self.reconnections = {}

  @staticmethod
  def copy(point, track_id, frame):
    return SimpleNamespace(trackId=track_id, radarSource='frontRadar', dRel=point.dRel, yRel=point.yRel,
                           vRel=point.vRel, vLead=point.vLead, yvRel=getattr(point, 'yvRel', 0.0),
                           measured=getattr(point, 'measured', True), trackState=getattr(point, 'trackState', 0),
                           ccnc_source_id=point.trackId, ccnc_stamp=frame, ccnc_fresh=True, ccnc_continuous=False)

  def new(self, point, frame):
    track_id = point.trackId
    if track_id in self.entries:
      track_id, self.next_id = self.next_id, self.next_id + 1
    p = self.copy(point, track_id, frame)
    entry = dict(point=p, source=point.trackId, frame=frame, good=frame, selected=None, confirmed=False,
                 vy=0.0, pending=[deque(maxlen=4) for _ in range(3)], lateral=deque(maxlen=4))
    self.entries[track_id] = entry
    return entry

  @staticmethod
  def coherent(samples, axis):
    if len(samples) < 3 or samples[-1][0] - samples[0][0] < 10:
      return False
    steps = [b[1] - a[1] for a, b in zip(samples, list(samples)[1:])]
    rates = (4.0, 15.0, 10.0)
    return (all(0 < b[0] - a[0] <= 15
                and abs(b[1] - a[1] - (a[2] * (b[0] - a[0]) * .01 if axis == 0 else 0.0))
                <= 0.15 + rates[axis] * (b[0] - a[0]) * .01
                for a, b in zip(samples, list(samples)[1:]))
            and (sum(abs(s) for s in steps) < .2 or abs(sum(steps)) >= .8 * sum(abs(s) for s in steps)))

  def update(self, live, frame, selected, observations):
    if live is None:
      self.entries.clear()
      self.reconnections.clear()
      self.selection = [None, None, None]
      self.raw = self.view = None
      self.stamp = frame
      return None
    if live is self.raw and 0 <= frame - self.stamp <= 15:
      return self.view
    if not 0 < frame - self.stamp <= 15:
      self.entries.clear()
      self.reconnections.clear()
      self.selection = [None, None, None]
      if live is self.raw:  # Re-reading stale data cannot create a new observation.
        self.view = None
        return None
    self.raw, self.stamp = live, frame
    self.entries = {key: e for key, e in self.entries.items() if frame - e['good'] <= self.RECONNECT}
    for key, e in self.entries.items():
      previous = observations.get(key)
      e['confirmed'] |= previous is not None and previous[2] >= 3 and previous[1] - previous[0] >= 15
      if key in selected and e['confirmed']:
        e['selected'] = frame
    raw = {}
    for q in live.points:
      if str(q.radarSource) != 'frontRadar':
        continue
      dRel, yRel, vRel, vLead = q.dRel, q.yRel, q.vRel, q.vLead
      if dRel >= 1 and math.isfinite(dRel) and math.isfinite(yRel) and math.isfinite(vRel) and math.isfinite(vLead):
        track_id = q.trackId
        raw[track_id] = SimpleNamespace(trackId=track_id, dRel=dRel, yRel=yRel, vRel=vRel, vLead=vLead,
                                        yvRel=getattr(q, 'yvRel', 0.0), measured=getattr(q, 'measured', True),
                                        trackState=getattr(q, 'trackState', 0))
    by_source = {e['source']: key for key, e in self.entries.items() if e['source'] is not None}
    matches = {source: by_source[source] for source in raw if source in by_source}
    # Only reconnect absent sources, with a unique candidate in both directions.
    # ponytail: small radar candidate set; ambiguous matches wait instead of global assignment.
    candidates = {}
    for source, p in raw.items():
      if source in matches:
        continue
      options = []
      for key, e in self.entries.items():
        if e['source'] in raw or not e['confirmed'] or e['selected'] is None:
          continue
        dt = (frame - e['frame']) * .01
        old = e['point']
        if (frame - e['good'] <= self.RECONNECT and frame - e['selected'] <= self.RECONNECT
            and abs(p.dRel - old.dRel - old.vRel * dt) <= 1.0 + 2.0 * dt
            and abs(p.yRel - old.yRel - e['vy'] * dt) <= .5 + 2.0 * dt and abs(p.vRel - old.vRel) <= 2.0):
          options.append(key)
      candidates[source] = options
    reconnecting, reconnected, pending_reconnections = set(), set(), {}
    for source, options in candidates.items():
      if len(options) == 1 and sum(options[0] in others for others in candidates.values()) == 1:
        key = options[0]
        pending = self.reconnections.get(source)
        if pending is not None and pending[0] == key and 0 < frame - pending[1] <= 15:
          matches[source] = key
          self.entries[key]['source'] = source
          reconnected.add(source)
        else:
          # One-frame ghosts cannot acquire an established display identity.
          pending_reconnections[source] = (key, frame)
          reconnecting.add(source)
    self.reconnections = pending_reconnections
    points, used = [], set()
    for source, p in raw.items():
      if source in reconnecting:
        continue
      key = matches.get(source)
      e = self.entries.get(key)
      if e is None:
        e = self.new(p, frame)
        points.append(e['point'])
        used.add(e['point'].trackId)
        continue
      old = e['point']
      dt = (frame - e['frame']) * .01
      predicted = (old.dRel + old.vRel * dt, old.yRel + e['vy'] * dt, old.vRel)
      observed = (p.dRel, p.yRel, p.vRel)
      limits = (.5 + 2.0 * dt, .35 + 2.0 * dt, 1.0 + 5.0 * dt)
      accepted, values = [], []
      for axis, (value, estimate, limit) in enumerate(zip(observed, predicted, limits)):
        pending = e['pending'][axis]
        valid = abs(value - estimate) <= limit or (source in reconnected and axis < 2)
        if valid:
          pending.clear()
        else:
          pending.append((frame, value, p.vRel))
          valid = self.coherent(pending, axis)
        accepted.append(valid)
        values.append(value if valid else estimate)
      # A sustained large relocation is a new identity, never a correction of the old car.
      relocated = abs(p.dRel - predicted[0]) > 3.0 or abs(p.yRel - predicted[1]) > 3.0
      if not e['confirmed'] or (relocated and all(accepted[:2])):
        continuous = not relocated and all(accepted[:2])
        if relocated:
          e['source'] = None
          e = self.new(p, frame)
          old = e['point']
        result = self.copy(p, old.trackId, frame)
        result.ccnc_continuous = continuous
        e['good'], e['vy'] = frame, 0.0
      else:
        result = self.copy(p, old.trackId, frame)
        result.dRel, result.yRel, result.vRel = values
        result.vLead += result.vRel - p.vRel
        result.ccnc_fresh = all(accepted[:2])
        result.ccnc_continuous = True
        result.ccnc_stamp = frame if result.ccnc_fresh else old.ccnc_stamp
        result.measured = result.measured and result.ccnc_fresh
        if result.ccnc_fresh:
          e['good'] = frame
        # Pending lateral evidence must not drive the accepted coast velocity.
        if accepted[1]:
          if accepted[0] and accepted[2]:
            e['lateral'].append((frame, p.yRel))
          else:
            e['lateral'].clear()
          history = e['lateral']
          e['vy'] = 0.0
          if self.coherent(history, 1):
            span = (history[-1][0] - history[0][0]) * .01
            net = history[-1][1] - history[0][1]
            steps = [b[1] - a[1] for a, b in zip(history, list(history)[1:])]
            e['vy'] = net / span if max(abs(s) for s in steps) <= .7 * abs(net) else 0.0
        if not result.ccnc_fresh and frame - e['good'] > self.HOLD:
          e['source'] = None
          e = self.new(p, frame)
          result = e['point']
      e['point'], e['frame'] = result, frame
      points.append(result)
      used.add(result.trackId)
    for key, e in self.entries.items():
      if key in used or not e['confirmed'] or e['selected'] is None or frame - e['good'] > self.HOLD:
        continue
      old = e['point']
      dt = (frame - e['frame']) * .01
      predicted = SimpleNamespace(**vars(old))
      predicted.dRel += old.vRel * dt
      predicted.yRel += e['vy'] * dt
      predicted.ccnc_fresh = predicted.measured = False
      predicted.ccnc_continuous = True
      e['point'], e['frame'] = predicted, frame
      if predicted.dRel >= 1:
        points.append(predicted)
    self.view = SimpleNamespace(points=points)
    return self.view

  def select(self, leads, lateral, references, frame, bypass=False, rejected=()):
    if bypass:
      self.selection = [None, None, None]
      return leads, lateral, references
    chosen = {p.trackId for p in leads if p is not None}
    available = {p.trackId for p in self.view.points} if self.view is not None else set()
    for slot, point in enumerate(leads):
      saved = self.selection[slot]
      old_id = saved['id'] if saved else None
      e = self.entries.get(old_id)
      old = e['point'] if e else None
      retain = (old_id in available and saved is not None and e is not None and e['confirmed'] and frame - saved['qualified'] <= self.HOLD
                and frame - e['good'] <= self.HOLD and old_id not in rejected and old_id not in chosen
                and 1 <= old.dRel <= (160 if slot == 0 else 80)
                and (slot == 0 or (old.yRel + saved['offset']) * (1 if slot == 1 else -1) > 0))
      if retain and (point is None or point.dRel >= old.dRel - 2.0):
        if point is not None:
          stamp = point.ccnc_stamp
          if saved.get('pending') != point.trackId:
            saved.update(pending=point.trackId, start=stamp, stamp=stamp, count=1)
          elif stamp != saved['stamp']:
            saved.update(stamp=stamp, count=saved['count'] + 1)
          retain = stamp - saved['start'] < 15 or saved['count'] < 3
        if retain:
          leads[slot], lateral[slot], references[slot] = old, old.yRel + saved['offset'], saved['reference']
          chosen.add(old_id)
          continue  # A held selection cannot renew its own qualification deadline.
      if point is None:
        self.selection[slot] = None
      elif point.ccnc_fresh:
        self.selection[slot] = dict(id=point.trackId, qualified=frame, offset=lateral[slot] - point.yRel,
                                    reference=references[slot])
    return leads, lateral, references


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
    self.stop_motion = {}
    self._saved_side = [None, None]
    self.live = None
    self.last_frame = -1
    self.tracks = {}
    self._points = ()
    self.selected = (None, None, None)
    self.crossing = {}
    self.positions = {}
    self.position_correction = None
    self.temporal = None
    self.lateral_grace = {}
    self.recent_selected = {}
    self.recent_front = {}
    self.lane_sides = {}
    self.boundary_admission = {}
    self.boundary_rejected = set()
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
    self._display_path = None
    self._path_live = self._path_points = None
    self._curves = {}

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
    if gap or not stopped or self.live is None:
      self.stop_motion.clear()
    if stopped and self.live is not None:
      self.update_stop_motion()

  def update_stop_motion(self):
    # Only fresh, continuous radar positions while ego is stopped can prove crossing.
    self.stop_motion = {key: value for key, value in self.stop_motion.items()
                        if key in self.tracks and value['birth'] == self.tracks[key][0]}
    for track_id, (birth, stamp, _, p, _) in self.tracks.items():
      # Freshness metadata exists only on the temporal tracker's display copies.
      if self.temporal is not None and not getattr(p, 'ccnc_fresh', True):
        continue
      history = self.stop_motion.get(track_id)
      if history is None:
        history = dict(birth=birth, samples=deque(), blocked=False, settled=(stamp, p.dRel, p.yRel))
        self.stop_motion[track_id] = history
      samples = history['samples']
      if samples and samples[-1][0] == stamp:
        continue
      samples.append((stamp, p.yRel))
      while samples and stamp - samples[0][0] > 100:
        samples.popleft()
      if len(samples) >= 4 and stamp - samples[0][0] >= 50:
        steps = [b[1] - a[1] for a, b in zip(samples, list(samples)[1:])]
        net = samples[-1][1] - samples[0][1]
        travel = sum(abs(step) for step in steps)
        directed = sum(step * (1 if net > 0 else -1) >= 0.04 for step in steps)
        if (abs(net) >= 0.75 and abs(net) >= 0.8 * travel and directed >= 3
            and max(abs(step) for step in steps) <= 0.6 * abs(net)):
          history['blocked'] = True
          history['settled'] = (stamp, p.dRel, p.yRel)
      start, x, y = history['settled']
      if (abs(p.dRel - x) > 0.3 or abs(p.yRel - y) > 0.2
          or abs(p.vLead) > 2.0 / 3.6 or abs(p.vRel) > 2.0 / 3.6):
        history['settled'] = (stamp, p.dRel, p.yRel)
      elif history['blocked'] and stamp - start >= 200:
        history['blocked'] = False
        samples.clear()
        samples.append((stamp, p.yRel))
      if history['blocked']:
        for saved in (self.stop_holds, self.stop_pending, self.approach_holds, self.approach_pending):
          for side, entry in list(saved.items()):
            if entry[0] == track_id and (saved is self.stop_pending or saved is self.approach_pending or entry[-1] == birth):
              del saved[side]

  def approaching_display(self, values, leads, frame):
    # A stationary target lost just before ego stops is carried in odometry coordinates.
    for side, lead in enumerate(leads):
      prefix = 'LF' if side == 0 else 'RF'
      keys = (prefix + '_DETECT', prefix + '_DETECT_DISTANCE', prefix + '_DETECT_LATERAL')
      if self.approaching and lead is not None and (self.temporal is None or getattr(lead, 'ccnc_fresh', True)):
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
                                        tuple(values[k] for k in keys), frame, self.stop_distance, self.tracks[lead.trackId][0])
      else:
        self.approach_pending.pop(side, None)
      held = self.approach_holds.get(side)
      if held is None:
        continue
      track_id, world_x, y, display, last_seen, last_distance, birth = held
      x = world_x - self.stop_distance  # Keep negative distance internally.
      invalid = x < -1.0 or frame - last_seen > 300 or self.stop_distance - last_distance > 5.0
      for point in self.live.points:
        if (point.trackId == track_id and self.tracks.get(track_id, (None,))[0] == birth
            and str(point.radarSource) == 'frontRadar'
            and (abs(point.dRel - x) > 1.0 or abs(point.yRel - y) > 0.75
                 or abs(getattr(point, 'yvRel', 0.0)) > 2.0 / 3.6)):
          invalid = True
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
          self.stop_holds[side] = (track_id, x, y, display, birth)
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
      if held is not None:
        track_id, x, y, display, birth = held
        invalid = False
        for point, px, py, vr, v in self._points:
          same = point.trackId == track_id and self.tracks[point.trackId][0] == birth
          nearby = abs(px - x) <= 2.0 and abs(py - y) <= 0.8
          if not (same or nearby):
            continue
          moving = abs(v) > 2.0 / 3.6 or abs(vr) > 2.0 / 3.6
          other_slot = point.trackId in (self.selected[0], self.selected[2 - side])
          if moving or other_slot or (same and abs(px - x) > 1.0):
            invalid = True
            break
        if invalid:
          self.stop_holds.pop(side, None)
          held = None
      if lead is not None and (self.temporal is None or getattr(lead, 'ccnc_fresh', True)):
        pending = self.stop_pending.get(side)
        stationary = (abs(lead.vLead) <= 2.0 / 3.6 and abs(lead.vRel) <= 2.0 / 3.6
                      and not self.stop_motion.get(lead.trackId, {}).get('blocked', False))
        drifted = (pending is not None and pending[0] == lead.trackId
                   and (abs(lead.dRel - pending[2]) > 1.0 or abs(lead.yRel - pending[3]) > 0.75))
        if not stationary:
          self.stop_pending.pop(side, None)
          continue
        if pending is None or pending[0] != lead.trackId or drifted:
          pending = (lead.trackId, frame, lead.dRel, lead.yRel)
          self.stop_pending[side] = pending
        if (frame - pending[1] >= 50 and self.stable(lead.trackId, 50)
            and (held is None or lead.dRel <= held[1] + 1.0)):
          birth = self.tracks[lead.trackId][0]
          self.stop_holds[side] = (lead.trackId, lead.dRel, lead.yRel,
                                   tuple(values[k] for k in keys), birth)
      else:
        self.stop_pending.pop(side, None)
        if held is not None and lead is None:
          values.update(zip(keys, held[3]))

  def _update_model(self, md):
    # SubMaster model readers are immutable between publications; retain the reader itself.
    stamp = getattr(md, "timestampEof", None) if md is not None else None
    if md is self._model and stamp is not None and stamp == self._model_stamp:
      return
    self._side_width_cache.clear()
    self._model, self._model_stamp = md, stamp
    self._inner_data = self._lane_probs = self._lane_data = self._projection = self._path_data = None
    self._display_path = None
    self._projection_live = None
    self._path_live = self._path_points = None
    self._curves = {}

  def _curve(self, md, index):
    # Lane lines 0-3, road edges 4-5: one conversion and validation per model message.
    cached = self._curves.get(index) if md is self._model else None
    if cached is None:
      line = md.laneLines[index] if index < 4 else md.roadEdges[index - 4]
      xs, ys = np.asarray(line.x, dtype=np.float64), np.asarray(line.y, dtype=np.float64)
      cached = xs, ys, _ccnc_valid_boundary(xs, ys)
      if md is self._model:
        self._curves[index] = cached
    return cached

  def lane_probabilities(self, md):
    self._update_model(md)
    if self._lane_probs is None:
      left = right = 0.0
      if md is not None and len(md.laneLineProbs) >= 3 and len(md.laneLines) >= 3:
        inner = self._curve(md, 1), self._curve(md, 2)
        self._inner_data = inner[0][:2], inner[1][:2]
        if inner[0][2]:
          left = md.laneLineProbs[1]
        if inner[1][2]:
          right = md.laneLineProbs[2]
      self._lane_probs = left, right
    return self._lane_probs

  def lane_projection(self, live):
    if self._projection is not None and live is self._projection_live:
      return self._projection
    if self._lane_data is None:
      md = self._model
      data = list(self._inner_data)
      flags = []
      outer_valid = [md.laneLineProbs[i] >= self.position_correction.MIN_LANE_PROBABILITY
                     if self.position_correction is not None else md.laneLineProbs[i] > 0.1 for i in (0, 3)]
      for use, index in ((outer_valid[0], 0), (outer_valid[1], 3), (True, 4), (True, 5)):
        xs, ys, valid = self._curve(md, index) if use else (None, None, False)
        data.append((xs, ys) if use else None)
        flags.append(valid)
      flags = tuple(flags)
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
    if all(math.isfinite(prob) and prob >= 0.6 for prob in self._lane_probs):
      for (p, d, y, _, _), left_y, right_y in zip(self._points, *projected):
        if self.temporal is not None and not getattr(p, 'ccnc_fresh', True):
          continue
        if not (max(data[0][0][0], data[1][0][0]) <= d <= min(data[0][0][-1], data[1][0][-1])
                and right_y > left_y):
          continue
        side = 1 if y > -left_y + 0.35 else -1 if y < -right_y - 0.35 else 0
        if side:
          self.lane_sides[p.trackId] = (self.tracks[p.trackId][0], self.last_frame, self.stop_distance, side)
        else:
          self.lane_sides.pop(p.trackId, None)
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
    if self.temporal is not None:
      live = self.temporal.update(live, frame, self.selected, self.tracks)
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
      self.lane_sides.clear()
      self.boundary_admission.clear()
      self.boundary_rejected.clear()
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
        track_id = p.trackId
        previous = self.tracks.get(track_id)
        if self.temporal is not None:
          if not p.ccnc_continuous:
            previous = None
          elif previous is not None:
            first, last, n = previous[:3]
            current[track_id] = (first, p.ccnc_stamp, n + int(p.ccnc_fresh), p, (dRel, yRel, vRel))
            continue
        if previous is not None:
          first, last, n, _, (old_d, old_y, old_v) = previous
          dt = (frame - last) * 0.01
          if (0 < frame - last <= 15
              and abs(dRel - old_d - old_v * dt) <= 1.0 + 2.0 * dt):
            dy = abs(yRel - old_y)
            if dy <= 0.5 + 5.0 * dt:
              start, count = first, n + 1
            elif (track_id in self.selected[1:] and n >= 3 and last - first >= 15
                  and dy <= 0.75 + 5.0 * dt and abs(vRel - old_v) <= 1.0
                  and frame - self.lateral_grace.get(track_id, -1000) >= 30):
              # One small lateral discontinuity may keep an established side target.
              # Reset coordinate smoothing, not its selection eligibility.
              start, count = first, n + 1
              self.lateral_grace[track_id] = frame
              self.positions.pop(track_id, None)
        current[track_id] = (start, frame, count, p, (dRel, yRel, vRel))
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
    self.lane_sides = {key: value for key, value in self.lane_sides.items()
                       if key in current and value[0] == current[key][0]
                       and 0 <= frame - value[1] <= 100 and self.stop_distance - value[2] <= 15.0}

  def update_boundary_admission(self, md):
    """Confirm sustained radar motion before exempting a new detection."""
    self.boundary_admission = {key: value for key, value in self.boundary_admission.items()
                               if key in self.tracks and value['birth'] == self.tracks[key][0]}
    curves = None
    for track_id, (birth, stamp, _, p, _) in self.tracks.items():
      if self.temporal is not None and not getattr(p, 'ccnc_fresh', True):
        continue
      admission = self.boundary_admission.get(track_id)
      if admission is None:
        admission = dict(birth=birth, frame=-1, status='pending', samples=0, near=0,
                         moving=None)
        self.boundary_admission[track_id] = admission
      if admission['status'] == 'allowed':
        continue
      if admission['frame'] == stamp:
        continue
      admission['frame'] = stamp
      lateral_speed = getattr(p, 'yvRel', 0.0)
      stationary = abs(p.vLead) <= 2.0 / CV.MS_TO_KPH and abs(lateral_speed) <= 2.0 / CV.MS_TO_KPH
      # Compensate longitudinal travel only; radar lateral position/speed are unmodified.
      position = (self.stop_distance + p.dRel, p.yRel)
      vx, vy = p.vLead, lateral_speed
      speed = math.hypot(vx, vy)
      if not math.isfinite(speed) or speed <= 2.0 / CV.MS_TO_KPH:
        admission['moving'] = None
      else:
        direction = (vx / speed, vy / speed)
        moving = admission['moving']
        if moving is None or sum(a * b for a, b in zip(direction, moving[2])) < 0.5:
          admission['moving'] = (stamp, position, direction)
        elif (stamp - moving[0] >= (50 if admission['status'] == 'blocked' else 15)
              and sum((a - b) * d for a, b, d in zip(position, moving[1], moving[2])) >= 0.5):
          admission['status'] = 'allowed'
      if admission['status'] == 'pending' and stationary:
        if curves is None:
          curves = []
          if md is not None:
            for index, probability in zip(range(min(4, len(md.laneLines))), md.laneLineProbs):
              xs, ys, valid = self._curve(md, index)
              if valid:
                curves.append((xs, ys, probability))
        boundaries = [(abs(p.yRel + float(np.interp(p.dRel, xs, ys))), probability)
                      for xs, ys, probability in curves if xs[0] <= p.dRel <= xs[-1]]
        gap, probability = min(boundaries, default=(math.inf, 0.0))
        near = stationary and probability > 0.1 and gap <= 0.3
        # Missing/weak nearest geometry is not evidence that the point is interior.
        if stationary and (near or probability >= 0.3):
          admission['samples'] += 1
          admission['near'] += int(near)
          if admission['samples'] >= 7 and stamp - birth >= 30:
            admission['status'] = ('blocked' if admission['near'] >= 3
                                   and admission['near'] * 2 >= admission['samples'] else 'allowed')
    self.boundary_rejected = {key for key, value in self.boundary_admission.items()
                              if value['status'] == 'blocked' or (value['status'] == 'pending' and value['near'])}
    # Unknown geometry alone must not suppress a stationary vehicle or its display memory.
    for history in (self.stop_holds, self.stop_pending, self.approach_holds, self.approach_pending):
      for side, saved in list(history.items()):
        if saved[0] in self.boundary_rejected:
          if ((history is self.approach_holds or history is self.stop_holds)
              and saved[-1] != self.tracks[saved[0]][0]):
            continue  # A reused radar slot cannot revoke an older physical display memory.
          del history[side]

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

  def front_admitted(self, point, md):
    # Weak lane geometry cannot introduce a stationary FF solely from radar position.
    if abs(point.vLead) > 2.0 / CV.MS_TO_KPH or self.recently_selected(point.trackId):
      return True
    if max(self.lane_probabilities(md)) >= 0.5:
      return True
    if md is None or not len(md.leadsV3):
      return False
    lead = md.leadsV3[0]
    if not (len(lead.x) and len(lead.y) and len(lead.v)):
      return False
    x, y, v = lead.x[0], lead.y[0], lead.v[0]
    if not all(math.isfinite(a) for a in (lead.prob, x, y, v)):
      return False
    dx, dy, dv = abs(point.dRel - (x - 1.52)), abs(point.yRel + y), abs(point.vLead - v)
    return ((lead.prob >= 0.5 and dx <= max(6.0, point.dRel * 0.2) and dy <= 1.5 and dv <= 5.0)
            or (lead.prob >= 0.8 and point.dRel <= 30.0
                and dx <= max(3.0, point.dRel * 0.2) and dy <= 2.0 and dv <= 3.0))

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
      if track_id in self.boundary_rejected:
        continue
      first, last, count, p, _ = entry
      if not self.front_admitted(p, md):
        continue
      if count < 3 or last - first < 15:
        continue
      dRel, yRel = p.dRel, p.yRel
      if not minimum_distance <= dRel <= maximum_distance or p.vLead * CV.MS_TO_KPH <= minimum_speed:
        continue
      distance = dRel * dRel + yRel * yRel
      if distance >= score or (front_only and not recent(track_id, front_only=True)):
        continue
      retained = recent(track_id, front_only=True)
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
      # A recent side identity alone is not evidence of a cut-in during lane loss.
      # Require both present path agreement and vision before promoting it to FF.
      side = self.lane_sides.get(track_id)
      was_side = (side is not None and side[0] == first and 0 <= self.last_frame - side[1] <= 100
                  and self.stop_distance - side[2] <= 15.0)
      if (was_side or (recent(track_id) and not retained)) and not (
          vision_match and abs(aligned) <= 1.2):
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
      if (track_id in self.boundary_rejected or entry is None or entry[0] != birth or not 0 <= frame - stamp <= 200
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
      if slot == 0 and not self.front_admitted(p, self._model):
        del self.crossing[track_id]
        continue
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
      geometry_blend = False
    else:
      _, last, old_reference, old_raw, old_input, bias, geometry_blend, distance, lateral = previous
      dt = (frame - last) * 0.01
      lateral.dt = dt
      # At long range, repeated small geometry changes also need a bounded rate.
      # Raw radar motion remains outside this correction-only limiter.
      far_lane = reference[0] == 'lane' and point.dRel > 60.0
      rate = 3.0 - 1.5 * min(1.0, max(0.0, (point.dRel - 60.0) / 60.0)) if far_lane else 3.0
      step = dt * rate
      correction_change = (aligned_y - point.yRel) - (old_input - old_raw)
      if far_lane and reference == old_reference:
        raw_change = point.yRel - old_raw
        geometry_change = correction_change + bias  # Compare unblended geometry targets.
        if raw_change * geometry_change < 0.0:
          # Preserve the portion of radar motion cancelled by the road geometry.
          # Only the remaining geometry change needs the ordinary rate limit.
          step += min(abs(raw_change), abs(geometry_change))
      if reference != old_reference:
        bias = old_input + (point.yRel - old_raw) - aligned_y
        geometry_blend = False
      elif geometry_blend or (not bias and abs(correction_change) > (step if far_lane else 0.6)):
        # Blend abrupt road-geometry changes separately from actual radar motion.
        # Rebase an active blend so a transient model spike does not leave a tail.
        bias = old_input + (point.yRel - old_raw) - aligned_y
        geometry_blend = True
      if bias:
        bias = math.copysign(max(0.0, abs(bias) - step), bias)
      if not bias:
        geometry_blend = False
    filtered_input = aligned_y + bias
    self.positions[track_id] = (birth, frame, reference, point.yRel, filtered_input, bias, geometry_blend, distance, lateral)
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

  def align_display_position(self, point, aligned_y, reference, changing=False):
    """Use one road center for display correction, independently of lane selection."""
    if reference[0] != 'lane' or self._model is None:
      return aligned_y, reference
    curves = []
    for index, probability in zip((1, 2), self._lane_probs):
      xs, ys, valid = self._curve(self._model, index)
      if (valid and math.isfinite(probability) and probability >= self.position_correction.MIN_LANE_PROBABILITY
          and xs[0] <= 0 <= point.dRel <= xs[-1]):
        curves.append((index, xs, ys))
    if not curves:
      return aligned_y, reference
    samples = [np.interp((0.0, min(20.0, point.dRel), point.dRel), xs, ys) for _, xs, ys in curves]
    if len(curves) == 2:
      widths = samples[1] - samples[0]
      if not all(2.3 <= width <= 4.8 for width in widths):
        return aligned_y, reference
    correction = sum(float(y[2] - y[0]) for y in samples) / len(curves)
    # Far lane polynomials can remain confident while disagreeing with the path.
    # Do not substitute a maneuver trajectory or extrapolate uncertain position.
    pos = getattr(self._model, 'position', None)
    if not changing and point.dRel > 40.0 and pos is not None:
      if self._display_path is None:
        xs, ys = np.asarray(pos.x), np.asarray(pos.y)
        std = np.asarray(getattr(pos, 'yStd', ()))
        self._display_path = xs, ys, std, (_ccnc_valid_boundary(xs, ys) and len(std) == len(xs) and np.isfinite(std).all())
      xs, ys, std, valid = self._display_path
      if (valid
          and xs[0] <= 0 <= point.dRel <= xs[-1]
          and np.max(std[xs <= point.dRel]) <= .8):
        path = float(np.interp(point.dRel, xs, ys) - np.interp(0.0, xs, ys))
        previous = self.positions.get(point.trackId)
        using_path = previous is not None and previous[2] == ('lane', 'position')
        if abs(path - correction) > (.35 if using_path else .75):
          return point.yRel + path, ('lane', 'position')
    return point.yRel + correction, ('lane', 'center' if len(curves) == 2 else curves[0][0] == 1)

  def correct_position(self, point, aligned_y, filtered_y, slot, reference, frame):
    correction = self.position_correction
    if correction is None:
      return filtered_y
    position = self.positions.get(point.trackId)
    if position is not None and position[1] == frame:
      aligned_y = position[4]  # Use the same continuous geometry as the display filter.
    bounds = None
    if reference[0] == 'lane' and self._lane_data is not None:
      data, flags = self._lane_data
      indices = ((0, 1), (2, 0), (1, 3))[slot]
      saved = correction.tracks.get(point.trackId)
      retained = (saved is not None and saved['locked'] and saved['slot'] == slot
                  and 0 <= frame - saved['frame'] <= 15
                  and self.tracks.get(point.trackId, (None,))[0] == saved['birth'])
      reliable = (all(math.isfinite(p) and p >= correction.MIN_LANE_PROBABILITY for p in self._lane_probs) if slot == 0
                  else correction.center_valid[slot - 1] and flags[slot - 1])
      # Confidence hysteresis retains only a continuous, previously qualified lock.
      if retained and self._model is not None:
        line_indices = ((1, 2), (0, 1), (2, 3))[slot]
        reliable = all(math.isfinite(self._model.laneLineProbs[i])
                       and self._model.laneLineProbs[i] >= correction.MIN_LANE_PROBABILITY for i in line_indices)
      curves = [data[i] for i in indices]
      model = self._model
      valid = (model is not None and self._curve(model, 1)[2], model is not None and self._curve(model, 2)[2]) + flags[:2]
      if all(valid[i] and curves[n][0][0] <= point.dRel <= curves[n][0][-1] for n, i in enumerate(indices)):
        bounds = tuple(-float(np.interp(point.dRel, *c)) + aligned_y - point.yRel for c in curves)
    else:
      reliable = False
    return correction.apply(point, aligned_y, filtered_y, slot, reference, bounds,
                            self.tracks.get(point.trackId), frame, reliable)


class _CcncVehiclePositionCorrection:
  """Display-only center lock; fresh radar motion releases it across slot changes."""
  MIN_LANE_PROBABILITY = 0.1
  CENTER_TAU = 1.0
  CENTER_RATE = 0.3
  FOLLOW_RATE = 1.5
  MOVE_TIME = 60  # Control frames (10 ms); samples are distinct radar publications.
  MOVE_SPEED = 0.3
  MOVE_DISTANCE = 0.2
  SETTLE_TIME = 100
  SETTLE_SPEED = 0.1
  SETTLE_RANGE = 0.2
  GEOMETRY_HOLD = 50  # A short confidence dip may retain an existing center lock.
  BOUNDARY_RELEASE_MARGIN = 0.35
  BOUNDARY_RETURN_MARGIN = 0.55
  FAST_TIME = 25  # At least 250 ms of distinct, coherent radar observations.
  FAST_BOUNDARY_MARGIN = 0.6
  ROAD_BOUNDARY_RELEASE_MARGIN = .15
  ROAD_BOUNDARY_RETURN_MARGIN = .35
  ROAD_SETTLE_RANGE = .35
  ROAD_MOVE_DISTANCE = .35
  ROAD_FAST_DISTANCE = .8
  SLOT_TIME = 20  # Display-only debounce, with at least three fresh observations.
  SLOT_FIELDS = (('FF_DISTANCE', 'FF_LATERAL', 'FF_DETECT'),
                 ('LF_DETECT_DISTANCE', 'LF_DETECT_LATERAL', 'LF_DETECT'),
                 ('RF_DETECT_DISTANCE', 'RF_DETECT_LATERAL', 'RF_DETECT'))

  def __init__(self, left=3.0, right=3.0):
    self.centers = [left, -right]
    self.center_valid = [False, False]
    self.center_frame = None
    self.tracks = {}
    self.display_slots = {}
    self.display_selected = (None, None, None)

  @staticmethod
  def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

  def stabilize_slots(self, values, selected, observations, frame, bypass=False):
    """Debounce an object's vacant-slot crossing without changing raw selection."""
    previous_display = self.display_selected
    self.display_selected = selected
    if bypass:
      self.display_slots.clear()
      # Stopped/approach memory owns its slots. Only a current crossing car
      # may stay in an otherwise empty FF while its physical sign catches up.
      for slot in (1, 2):
        track_id = selected[slot]
        saved = self.tracks.get(track_id)
        observation = observations.get(track_id)
        if (track_id is not None and previous_display[0] == track_id and selected[0] is None
            and saved is not None and observation is not None and saved['birth'] == observation[0]
            and frame - saved['frame'] <= 15 and saved['output'] * (1 if slot == 1 else -1) < 0):
          distance, _, detect = self.SLOT_FIELDS[slot]
          if values['FF_DETECT'] == 0 and values[detect] != 0:
            values['FF_DISTANCE'], values['FF_LATERAL'], values['FF_DETECT'] = values[distance], -saved['output'], values[detect]
            values[detect] = 0
            destinations = list(selected)
            destinations[0], destinations[slot] = track_id, None
            self.display_selected = tuple(destinations)
      return
    snapshots = [tuple(values[key] for key in fields) for fields in self.SLOT_FIELDS]
    active = {}
    destinations = list(selected)
    for slot, track_id in enumerate(selected):
      observation = observations.get(track_id)
      if observation is None:
        continue
      birth, stamp = observation[:2]
      saved = self.display_slots.get(track_id)
      if saved is None or saved['birth'] != birth or not 0 <= frame - saved['frame'] <= 15:
        saved = dict(birth=birth, slot=slot, pending=None, stamp=stamp, count=0, start=stamp)
      old = saved['slot']
      distance, lateral, detect = snapshots[slot]
      common_y = -lateral if slot in (0, 2) else lateral
      correction = self.tracks.get(track_id)
      # Side CAN coordinates are unsigned. Preserve the physical coordinate
      # while a vacant FF still carries a crossing car on the opposite side.
      if correction and 'output' in correction:
        common_y = min(max(correction['output'], -5.4), 5.4) if slot else correction['output']
      aligned_y = correction['samples'][-1][2] if correction and correction['samples'] else common_y
      representable = old == 0 or (old == 1 and min(common_y, aligned_y) >= 0) or (old == 2 and max(common_y, aligned_y) <= 0)
      if old == slot or selected[old] is not None or not representable:
        saved['slot'], saved['pending'] = slot, None
      else:
        if saved['pending'] != slot:
          saved.update(pending=slot, start=stamp, count=1, stamp=stamp)
        elif stamp != saved['stamp']:
          saved['stamp'] = stamp
          saved['count'] += 1
        destination_ready = slot == 0 or common_y * (1 if slot == 1 else -1) >= 0
        if destination_ready and stamp - saved['start'] >= self.SLOT_TIME and saved['count'] >= 3:
          saved['slot'], saved['pending'] = slot, None
      destination = saved['slot']
      if destination != slot:
        destinations[slot], destinations[destination] = None, track_id
        values[self.SLOT_FIELDS[slot][2]] = 0
        mapped_y = -common_y if destination in (0, 2) else common_y
        for key, value in zip(self.SLOT_FIELDS[destination], (distance, mapped_y, detect)):
          values[key] = value
      saved['frame'] = frame
      active[track_id] = saved
    self.display_slots = active
    self.display_selected = tuple(destinations)

  def update_centers(self, left, right, frame, tracks):
    dt = 0.0 if self.center_frame is None else max(0.0, min(0.15, (frame - self.center_frame) * 0.01))
    self.center_frame = frame
    for side, target in enumerate((left, -right if right is not None else None)):
      self.center_valid[side] = target is not None and math.isfinite(target)
      if self.center_valid[side]:
        step = (target - self.centers[side]) * -math.expm1(-dt / self.CENTER_TAU)
        limit = dt * self.CENTER_RATE
        self.centers[side] += min(max(step, -limit), limit)
    self.tracks = {key: saved for key, saved in self.tracks.items()
                   if key in tracks and tracks[key][0] == saved['birth']}

  @staticmethod
  def _ongoing_motion(samples, stamp, nets):
    recent = [s for s in samples if stamp - s[0] <= 20]
    if len(recent) < 3 or stamp - recent[0][0] < 10:
      return False
    span = (stamp - recent[0][0]) * .01
    return all((recent[-1][i] - recent[0][i]) * net > 0
               and abs(recent[-1][i] - recent[0][i]) >= max(.04, .15 * span)
               for i, net in zip((1, 2), nets))

  def apply(self, point, aligned_y, filtered_y, slot, reference, bounds, observation, frame, reliable=True):
    if observation is None:
      return filtered_y
    birth, stamp = observation[:2]
    center = 0.0 if slot == 0 else self.centers[slot - 1]
    # A boundary-straddling or lane-free target must retain its measured position.
    clearance = (min(bounds[0] - aligned_y, aligned_y - bounds[1], bounds[0] - center, center - bounds[1])
                 if bounds is not None and all(math.isfinite(v) for v in (*bounds, aligned_y, center)) else -math.inf)
    common_road = reference == ('lane', 'center')
    release_margin = self.ROAD_BOUNDARY_RELEASE_MARGIN if common_road else self.BOUNDARY_RELEASE_MARGIN
    return_margin = self.ROAD_BOUNDARY_RETURN_MARGIN if common_road else self.BOUNDARY_RETURN_MARGIN
    inside = clearance >= release_margin
    saved = self.tracks.get(point.trackId)
    if saved is None or saved['birth'] != birth or not 0 <= frame - saved['frame'] <= 15:
      saved = dict(birth=birth, frame=frame, stamp=None, samples=deque(), settling=deque(),
                   locked=inside and reliable, following=not (inside and reliable), valid_frame=-1000,
                   output=filtered_y, slot=slot, reference=reference, slot_frame=-1000, boundary_follow=not inside,
                   fast=deque(maxlen=8), fast_speed=0.0)
      self.tracks[point.trackId] = saved
    bounds_valid = bounds is not None and all(math.isfinite(v) for v in bounds) and bounds[0] > bounds[1]
    center_valid = bounds_valid and bounds[1] <= center <= bounds[0]
    if center_valid and not inside:
      saved['boundary_follow'] = True
    elif clearance >= return_margin:
      saved['boundary_follow'] = False
    inside = inside and not saved['boundary_follow']
    samples = saved['samples']
    if slot != saved['slot'] or reference != saved['reference']:
      same_road = (reference in (('lane', 'center'), ('lane', 'position'))
                   and saved['reference'] in (('lane', 'center'), ('lane', 'position')))
      saved['slot_frame'] = frame if slot != saved['slot'] and reference == saved['reference'] else -1000
      # Keep the physical output, but never compare different geometry references.
      samples.clear()
      if reference != saved['reference']:
        saved['fast'].clear()
        saved['fast_speed'] = 0.0
      saved['settling'].clear()
      if slot != saved['slot'] or not same_road:
        saved['valid_frame'] = -1000
      if slot != saved['slot']:
        saved['locked'] = False
    saved['slot'], saved['reference'] = slot, reference
    fresh = getattr(point, 'ccnc_fresh', True)
    if inside and reliable and fresh:
      saved['valid_frame'] = frame
    # Temporary uncertainty changes the output mode without proving actual motion.
    geometry_hold = (saved['locked'] and not center_valid
                     and 0 <= frame - saved['valid_frame'] <= self.GEOMETRY_HOLD)
    forced_follow = (not inside and not geometry_hold) or (not reliable and frame - saved['valid_frame'] > self.GEOMETRY_HOLD)
    saved['following'] = not saved['locked'] or forced_follow
    if stamp != saved['stamp'] and fresh:
      saved['stamp'] = stamp
      fast = saved['fast']
      if fast:
        last, raw, aligned = fast[-1]
        elapsed = (stamp - last) * .01
        geometry = (aligned_y - point.yRel) - (aligned - raw)
        if not 0 < elapsed <= .15 or abs(geometry) > .04 + 3 * elapsed:
          fast.clear()
      fast.append((stamp, point.yRel, aligned_y))
      continuing_fast = saved['fast_speed'] > 0.0
      saved['fast_speed'] = 0.0
      if len(fast) >= 3 and stamp - fast[0][0] >= self.FAST_TIME:
        span = (stamp - fast[0][0]) * .01
        nets = [fast[-1][i] - fast[0][i] for i in (1, 2)]
        steps = [[b[i] - a[i] for a, b in zip(fast, list(fast)[1:])] for i in (1, 2)]
        boundary_valid = center_valid
        approaching_boundary = (boundary_valid
                                and (not bounds[1] <= aligned_y <= bounds[0]
                                     or abs((bounds[0] if nets[1] > 0 else bounds[1]) - aligned_y)
                                     <= self.FAST_BOUNDARY_MARGIN))
        recent_crossing = 0 <= frame - saved['slot_frame'] <= 30
        crossing = boundary_valid and not bounds[1] <= aligned_y <= bounds[0]
        minimum_fast_distance = (self.ROAD_FAST_DISTANCE if common_road and not (crossing or recent_crossing or continuing_fast)
                                 else .4)
        if (boundary_valid and reliable and self._ongoing_motion(fast, stamp, nets)
            and (approaching_boundary or continuing_fast or recent_crossing) and nets[0] * nets[1] > 0 and all(
            .8 <= abs(net) / span <= 15 and abs(net) >= minimum_fast_distance and abs(net) >= .8 * sum(abs(s) for s in changes)
            and max(abs(s) for s in changes) <= .7 * abs(net) for net, changes in zip(nets, steps))):
          saved['fast_speed'] = abs(nets[1]) / span
          saved['locked'], saved['following'] = False, True
      if samples:
        last, raw, aligned = samples[-1]
        dt = (stamp - last) * 0.01
        raw_step = point.yRel - raw
        geometry_step = (aligned_y - point.yRel) - (aligned - raw)
        # Exclude isolated jumps before a median/filter can turn them into a ramp.
        if not 0 < dt <= 0.15:
          saved['settling'].clear()
        if (not 0 < dt <= 0.15 or abs(raw_step) > 0.04 + 2.0 * dt
            or abs(geometry_step) > 0.04 + 3.0 * dt):
          samples.clear()
      samples.append((stamp, point.yRel, aligned_y))
      while len(samples) > 1 and stamp - samples[1][0] >= self.SETTLE_TIME:
        samples.popleft()
      moving = [sample for sample in samples if stamp - sample[0] <= 70]
      # Fast evidence has already unlocked following in this publication.
      if saved['fast_speed'] == 0.0 and len(moving) >= 5 and stamp - moving[0][0] >= self.MOVE_TIME:
        span = (stamp - moving[0][0]) * 0.01
        nets = [moving[-1][i] - moving[0][i] for i in (1, 2)]
        steps = [[b[i] - a[i] for a, b in zip(moving, moving[1:])] for i in (1, 2)]
        # Both raw and road-aligned motion must agree; curves alone cannot unlock.
        if (center_valid and reliable and self._ongoing_motion(moving, stamp, nets) and nets[0] * nets[1] > 0.0 and all(
            abs(net) >= max(self.ROAD_MOVE_DISTANCE if common_road else self.MOVE_DISTANCE, self.MOVE_SPEED * span)
            and abs(net) >= 0.8 * sum(abs(step) for step in changes)
            and max(abs(step) for step in changes) <= 0.5 * abs(net)
            for net, changes in zip(nets, steps))):
          saved['following'] = True
          saved['locked'] = False
      # Curves move raw yRel even when a vehicle keeps its lane. Isolated spikes
      # must not erase all evidence that its road-relative position has settled.
      settling = saved['settling']
      settling.append((stamp, point.yRel, aligned_y, filtered_y, inside))
      while len(settling) > 1 and stamp - settling[1][0] >= self.SETTLE_TIME:
        settling.popleft()
      if (saved['following'] and inside and reliable and len(settling) >= 5
          and stamp - settling[0][0] >= self.SETTLE_TIME):
        span = (stamp - settling[0][0]) * 0.01
        evidence = list(settling)
        values = [s[3] for s in evidence]
        ordered = sorted(values)
        trim = max(1, len(values) // 10)
        edge = max(2, len(values) // 3)
        start, end = (float(self._median(v)) for v in (values[:edge], values[-edge:]))
        trends = [float(self._median([s[i] for s in evidence[-edge:]]))
                  - float(self._median([s[i] for s in evidence[:edge]])) for i in (1, 2)]
        moving_trend = trends[0] * trends[1] > 0.0 and min(abs(v) for v in trends) > self.SETTLE_SPEED * span
        # Opposing trends suggest geometry drift; they cannot prove lane keeping.
        geometry_drift = trends[0] * trends[1] < 0.0 and min(abs(v) for v in trends) > self.SETTLE_SPEED * span
        settled = (ordered[-trim - 1] - ordered[trim] <= (self.ROAD_SETTLE_RANGE if common_road else self.SETTLE_RANGE)
                   and abs(end - start) <= self.SETTLE_SPEED * span)
        if ((settled or geometry_drift) and not moving_trend
            and sum(s[4] for s in evidence) >= 0.8 * len(evidence)):
          saved['following'] = False
          saved['locked'] = True
    dt = max(0.0, min(0.15, (frame - saved['frame']) * 0.01))
    saved['frame'] = frame
    target = filtered_y if saved['following'] else center
    rate = self.FOLLOW_RATE if saved['following'] else self.CENTER_RATE
    if saved['following'] and saved['fast_speed'] > 0 and fresh:
      # Keep ordinary smoothing until a physical boundary is actually crossed.
      if bounds is not None and (not bounds[1] <= aligned_y <= bounds[0] or 0 <= frame - saved['slot_frame'] <= 30):
        target = aligned_y
      rate = min(15.0, max(rate, 1.4 * saved['fast_speed'] + 1.0))
    saved['output'] += min(max(target - saved['output'], -rate * dt), rate * dt)
    return saved['output']

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
    self._err_max = 4.0 if distance else 0.6
    self.dt = 0.05
    self._following_step = False
    self.apply = self._apply_adaptive if distance else self._apply_step

  def _apply_step(self, target):
    # Track changes initialize immediately. A sustained step on the same track
    # catches up quickly without a single-frame reset or changing the deadband.
    self._buffer.append(target)
    med = self._get_median()
    error = med - self._filtered_value
    self._following_step = self._following_step or abs(error) > self._err_max
    if self._following_step:
      step = max(0.0, self.dt) * 5.0
      self._filtered_value += math.copysign(min(abs(error), step), error)
      if abs(error) <= step:
        self._following_step = False
    else:
      self._filtered_value += self._a_min * error
    return self._filtered_value

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

class _CcncLaneGeometry:
  """Display geometry only; prediction is bounded and never identifies radar objects."""
  _fit_coordinates = np.linspace(-1., 1., 5)
  _near_distances = np.linspace(0., 30., 5)

  def __init__(self):
    self.last_time = None
    self.model = self.stamp = None
    self.fresh = False
    self.dt = 0.05
    self.curvature = 0.0
    self.lane_alpha = .2
    self.valid_time = -math.inf
    self.target = None
    self.motion = deque(maxlen=8)
    self.direction = None
    self.hold_start = None
    self.hold_speed = 0.0
    self.progress = 0.0

  @staticmethod
  def fit_curve(xs, ys, start, end, with_residual=True):
    step = (end - start) * .25
    distances = (start, start + step, start + 2*step, start + 3*step, end)
    y = np.interp(distances, xs, ys)
    return _CcncLaneGeometry.fit_samples(y, end - start, with_residual)

  @staticmethod
  def fit_samples(y, span, with_residual=False):
    half = span * .5
    quadratic = (4*y[0] - 2*y[1] - 4*y[2] - 2*y[3] + 4*y[4]) / 7.0
    slope = (-y[0] - .5*y[1] + .5*y[3] + y[4]) / (2.5 * half)
    curvature = float(-3600 * quadratic / (half**2 * (1 + slope*slope)**1.5))
    if not with_residual:
      return curvature, None
    t = _CcncLaneGeometry._fit_coordinates
    fitted = np.mean(y) - quadratic*.5 + slope*half*t + quadratic*t*t
    return curvature, float(np.max(np.abs(y - fitted)))

  @staticmethod
  def road_curve(md, speed, changing=False):
    if md is None:
      return None
    pos = md.position
    xs, ys = np.asarray(pos.x, dtype=np.float64), np.asarray(pos.y, dtype=np.float64)
    stds = np.asarray(pos.yStd, dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or stds.ndim != 1 or len(ys) != len(xs) or len(stds) != len(xs) or not np.all(np.isfinite(stds)):
      return None
    # Keep the original position uncertainty limit, independent of lane probability.
    uncertain = np.flatnonzero(stds > .8)
    trusted = int(uncertain[0]) if len(uncertain) else len(xs)
    xs, ys = xs[:trusted], ys[:trusted]
    # Distant untrusted points can fold in x on a tight bend. Validate only
    # the prefix used for curvature; invalid trusted geometry still rejects.
    if not _ccnc_valid_boundary(xs, ys):
      return None
    available = xs[-1]
    end = 30.0 + min(max((speed - 20.0) / 80.0, 0.0), 1.0) * 50.0
    # Look beyond the near lateral transition, without trusting low-probability
    # lane polynomials to describe the distant road. This is not exact removal
    # of an arbitrary lane-change trajectory.
    end = min(end + (speed / 3.6 * 2.0 if changing else 0.0), available)
    start = max(20.0, end * .5) if changing else max(0.0, xs[0])
    if end - start < 20.0:
      return None
    fit = _CcncLaneGeometry.fit_curve
    target = min(max(fit(xs, ys, start, end, with_residual=False)[0], -15.0), 15.0)
    if not changing or xs[0] > 0:
      return target
    # A consistent quadratic bend is not maneuver evidence just because nearby
    # lanes are wrong. Preserve it even when those lanes incorrectly look straight.
    if fit(xs, ys, 0, end)[1] <= .02:
      return target
    # Near geometry can expose a maneuver bend in the path without requiring
    # distant lane polynomials (or lane probabilities) to describe the road.
    if len(md.laneLines) < 3:
      return target
    curves, samples = [], []
    for index in (1, 2):
      line = md.laneLines[index]
      lx, ly = np.asarray(getattr(line, 'x', ())), np.asarray(line.y)
      if not _ccnc_valid_boundary(lx, ly) or lx[0] > 0 or lx[-1] < 30:
        return target
      y = np.interp(_CcncLaneGeometry._near_distances, lx, ly)
      samples.append(y)
      curves.append(_CcncLaneGeometry.fit_samples(y, 30.)[0])
    widths = samples[1] - samples[0]
    if not np.all((widths >= 2.3) & (widths <= 4.8)) or abs(curves[1] - curves[0]) > 3:
      return target
    local = sum(curves) * .5
    if abs(fit(xs, ys, 0, 30, with_residual=False)[0] - local) > .5:
      # Compare the same near-road interval. When the path differs, use that
      # road bend directly instead of clipping a different, farther interval.
      return min(max(local, -15.0), 15.0)
    return target

  def update(self, md, speed, now, changing=False, compute_curve=True):
    elapsed = None if self.last_time is None else now - self.last_time
    if elapsed is None or not 0 <= elapsed <= .15:
      self.__init__()
    self.dt = min(.15, max(0.0, elapsed)) if elapsed is not None and elapsed >= 0 else .05
    self.last_time = now
    stamp = getattr(md, 'timestampEof', None) if md is not None else None
    self.fresh = md is not None and (stamp != self.stamp if stamp is not None and stamp > 0 else md is not self.model)
    self.model, self.stamp = md, stamp
    if not compute_curve:
      return 0
    if self.fresh:
      try:
        self.target = self.road_curve(md, speed, changing)
      except (AttributeError, IndexError, TypeError, ValueError):
        # This model was consumed, but cannot renew the previous target.
        self.target = None
      if self.target is not None:
        self.valid_time = now
    if md is None or now - self.valid_time > .35:
      self.target = None
    target = self.curvature if self.target is None and now - self.valid_time <= .35 else self.target or 0.0
    step = (target - self.curvature) * -math.expm1(-self.dt / .25)
    self.curvature += min(max(step, -10.0 * self.dt), 10.0 * self.dt)
    return round(self.curvature)

  def observe_motion(self, md, left, active, holding, now, speed=0.0):
    if not active or self.direction != left or md is None:
      self.motion.clear()
      self.hold_start = None
      self.hold_speed = self.progress = 0.0
    self.direction = left if active else None
    if not active or holding or not self.fresh:
      return
    lines = md.laneLines
    if len(lines) < 3 or not len(lines[1].y) or not len(lines[2].y):
      self.motion.clear()
      return
    sign = 1 if left else -1
    if not 2.3 <= lines[2].y[0] - lines[1].y[0] <= 4.8:
      self.motion.clear()
      return
    velocities = []
    for index in (1, 2):
      line = lines[index]
      xs, ys = np.asarray(line.x), np.asarray(line.y)
      if not _ccnc_valid_boundary(xs, ys) or xs[0] > 0 or xs[-1] < 5:
        self.motion.clear()
        return
      y = np.interp([0., 2.5, 5.], xs, ys)
      # Near-road tangent removes constant offsets and quadratic road bending.
      # A parallel index sweep cannot inject its y[0] jump into this velocity.
      velocities.append(sign * speed / 3.6 * (-3*y[0] + 4*y[1] - y[2]) / 5.)
    self.motion.append((now, *velocities))

  def start_hold(self, now):
    self.hold_start, self.hold_speed, self.progress = now, 0.0, 0.0
    # Use pre-trigger near-road heading, not the derivative of relabelled offsets.
    # Distorted lane shapes remain ambiguous, so continuation is still bounded.
    samples = list(self.motion)
    if len(samples) < 4 or now - samples[-1][0] > .15 or samples[-1][0] - samples[0][0] < .15:
      return
    if any(not 0 < b[0] - a[0] <= .15 for a, b in zip(samples, samples[1:])):
      return
    speeds = [float(np.median([s[i] for s in samples])) for i in (1, 2)]
    if any(max(s[i] for s in samples) - min(s[i] for s in samples) > .6 for i in (1, 2)):
      return
    if all(.1 <= s <= 1.5 for s in speeds) and abs(speeds[0] - speeds[1]) <= .4:
      self.hold_speed = min(speeds)

  def hold_progress(self, now):
    if self.hold_start is not None:
      # Velocity fades to zero, never extrapolates indefinitely through a cancellation.
      t = min(.6, max(0.0, now - self.hold_start))
      self.progress = min(.45, self.hold_speed * (t - t*t / 1.2))
    return self.progress


def _ccnc_basic_curve(md, speed, changing):
  # Original 6334db9d peak-displacement curvature, including its search limits.
  end = 30.0 + min(max((speed - 20.0) / 80.0, 0.0), 1.0) * 50.0
  pos = md.position
  xs, ys, stds = pos.x, pos.y, pos.yStd
  maximum, peak, start = 0.0, 0, 0
  start_found = not changing
  minimum = 20.0 if changing else 0.0
  for i in range(1, len(xs)):
    x = xs[i]
    if not start_found and x >= minimum:
      start, start_found = i, True
    if stds[i] > .8 or x > end:
      break
    if abs(ys[i]) > maximum:
      maximum, peak = abs(ys[i]), i
  if start != peak and xs[peak] >= 20.0 + minimum:
    return -3600.0 * (ys[peak] - ys[start]) / (xs[peak] * xs[peak])
  return 0.0


def update_lanes(values, CS, md, v_ego_kph, a_ego_kph, desire, lat_enabled, lane_color=True, model_lanes=True, frame=None):
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

  if not model_lanes:
    return

  now = time.monotonic() if frame is None else frame * .01
  geometry = state.lane_geometry
  if geometry.last_time is not None and not 0 <= now - geometry.last_time <= .15:
    state.draw_center = state.hold_lane = False
    state.hold_lane_escape_count = 0
    state.lane_phase_min = 10.0
    state.l_lane_f.reset_alpha()
    state.r_lane_f.reset_alpha()
  try:
    precise = state.lane_curve_mode == 2
    curvature = geometry.update(md, v_ego_kph, now, is_currently_lane_changing, compute_curve=precise)
    if lat_enabled and not precise:
      curvature = round(state.lane_curv.apply(_ccnc_basic_curve(md, v_ego_kph, is_currently_lane_changing)))
    elif not lat_enabled:
      curvature = round(CS.out.steeringAngleDeg / 3)

  except:
    # 모델 데이터 예외 발생 시 핸들 각도 기반 백업
    curvature = round(CS.out.steeringAngleDeg / 3)
    values["LFA_ICON"] = 5

  values["LANELINE_CURVATURE"] = min(abs(curvature), 15) + (-1 if curvature < 0 else 0)
  values["LANELINE_CURVATURE_DIRECTION"] = 1 if curvature < 0 else 0

  try:
    # 차선 위치 갱신: 항시 적용
    l_prob = md.laneLineProbs[1]
    r_prob = md.laneLineProbs[2]

    # --- 차선 변경 상태 관리 및 알파값 조정 ---
    target_alpha = .6 if is_currently_lane_changing else .2
    next_alpha = geometry.lane_alpha + min(max(target_alpha - geometry.lane_alpha, -2*geometry.dt), 2*geometry.dt)
    if next_alpha != geometry.lane_alpha:
      state.l_lane_f.update_alpha(next_alpha)
      state.r_lane_f.update_alpha(next_alpha)
      geometry.lane_alpha = next_alpha
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
      is_moving_left = desire == 3 if is_auto_lane_changing else CS.out.leftBlinker
      if geometry.direction is not None and geometry.direction != is_moving_left:
        geometry.observe_motion(md, is_moving_left, False, False, now)
        state.draw_center = state.hold_lane = False
        state.lane_phase_min = 10.0
        state.hold_lane_escape_count = 0
      # 위상 변화 시 차선 강조 변경
      if not state.draw_center:

        lane_raw = leftlaneraw if is_moving_left else rightlaneraw

        # A rebound far from the vehicle is recognition jitter, not evidence of
        # crossing a boundary (e.g. a turn signal before a U-turn).
        is_phase_shifted = lane_raw < 0.1 or (state.lane_phase_min < .5 and lane_raw - state.lane_phase_min > .3)
        state.lane_phase_min = min(state.lane_phase_min, lane_raw)

        if is_phase_shifted:
          geometry.start_hold(now)
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
          state.hold_lane_escape_count += int(geometry.fresh)
          if state.hold_lane_escape_count >= 2:
            state.hold_lane = False
        else:
          state.hold_lane_escape_count = 0

        # Preserve the original release steps when no prior motion qualifies;
        # a qualified continuation may already be farther along.
        holding_factor = max(geometry.hold_progress(now), state.hold_lane_escape_count * .1)
        if is_moving_left:
          current_l_target = state.l_lane_f.reset(state.last_known_lane_width - holding_factor)
          current_r_target = state.r_lane_f.reset(holding_factor)
        else:
          current_l_target = state.l_lane_f.reset(holding_factor)
          current_r_target = state.r_lane_f.reset(state.last_known_lane_width - holding_factor)
      elif state.draw_center:
        MAX_STEP = 3.0 * geometry.dt
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
      geometry.observe_motion(md, is_moving_left, True, state.draw_center, now, v_ego_kph)
    else:
      geometry.observe_motion(md, False, False, False, now)
      state.draw_center = state.hold_lane = False
      state.hold_lane_escape_count = 0
      state.lane_phase_min = 10.0
      current_l_target = state.l_lane_f.apply(leftlaneraw)
      current_r_target = state.r_lane_f.apply(rightlaneraw)

      lane_width = current_l_target + current_r_target
      if 2 < lane_width < 4.6:
        state.last_known_lane_width = lane_width # 마지막 차선 폭을 기억해둠

    values["LANELINE_LEFT_POSITION"] = int(round(min(max(current_l_target, 0.0), 3.0) * 10.0))
    values["LANELINE_RIGHT_POSITION"] = int(round(min(max(current_r_target, 0.0), 3.0) * 10.0))

    # 차선 변경 아이콘
    if lat_enabled:
      values["LCA_LEFT_ICON"] = 1 if CS.out.leftBlindspot else 4 if CS.out.rightBlinker or not md.meta.laneChangeAvailableLeft else 2
      values["LCA_RIGHT_ICON"] = 1 if CS.out.rightBlindspot else 4 if CS.out.leftBlinker or not md.meta.laneChangeAvailableRight else 2
  except:
    # Only show startup status while the required model lane data is absent.
    # Calculation errors with populated lane data keep the original alert.
    if (md is None or len(md.laneLineProbs) < 3 or len(md.laneLines) < 3 or
        len(md.laneLines[1].y) == 0 or len(md.laneLines[2].y) == 0):
      values["ALERTS_5"] = 19  # ACTIVATING_HIGHWAY_DRIVING_PILOT_SYSTEM
    else:
      values["LANELINE_LEFT_POSITION"] = 30
      values["LANELINE_RIGHT_POSITION"] = 30
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
    display_tracker.update_boundary_admission(md)
    selected_lane_is_left = state.radar_lane_selector.update(left_prob, right_prob, frame)
    selected_lane_prob = left_prob if selected_lane_is_left else right_prob

    # LF/RF deadband 중심은 차량 존재 여부와 무관하게 인접 차선 geometry로 계속 갱신합니다.
    center_probability = (display_tracker.position_correction.MIN_LANE_PROBABILITY
                          if display_tracker.position_correction is not None else 0.6)
    lf_center = _ccnc_side_lane_center(md, 0, display_tracker._curve, center_probability)
    rf_center = _ccnc_side_lane_center(md, 1, display_tracker._curve, center_probability)

    if display_tracker.position_correction is not None:
      display_tracker.position_correction.update_centers(lf_center, rf_center, frame, display_tracker.tracks)

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
      ms_to_kph = CV.MS_TO_KPH
      # 여러 차로의 후보를 허용하되, 보정 후 횡거리 5.4m 밖의 측면 점은 선택하지 않습니다.
      max_side_lateral = 5.4
      max_side_distance = 80.0
      ff_min_dist = lf_min_dist = rf_min_dist = math.inf
      left_projection = right_projection = None

      for point_index, ((lead, dRel, yRel, vRel, vLead), (left_y, right_y)) in enumerate(zip(display_tracker._points, projected)):
        if lead.trackId in display_tracker.boundary_rejected:
          continue
        if display_tracker.temporal is not None and not display_tracker.stable(lead.trackId):
          continue
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
            and display_tracker.front_admitted(lead, md)
            and right_inner_bound - 0.35 <= yRel <= left_inner_bound + 0.35):
          boundary_front, boundary_front_y = lead, road_aligned_yRel

        # 2. [전방 주행 차선] - 외곽선/도로경계선 interp 4회 전부 생략
        if right_inner_bound <= yRel <= left_inner_bound:
          if dist_score < ff_min_dist:
            if velocity > min_front_lead_speed and display_tracker.front_admitted(lead, md):
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

    references = [('lane', selected_lane_is_left) if ff_uses_lane else ('path', False),
                  ('lane', selected_lane_is_left), ('lane', selected_lane_is_left)]
    if display_tracker.temporal is not None:
      (ff_lead, lf_lead, rf_lead), (ff_yRel, lf_yRel, rf_yRel), references = display_tracker.temporal.select(
        [ff_lead, lf_lead, rf_lead], [ff_yRel, lf_yRel, rf_yRel], references, frame,
        bypass=display_tracker.stopped or display_tracker.approaching or display_tracker.live is None,
        rejected=display_tracker.boundary_rejected)

    # A retained FF must not also occupy a side slot during lane recovery.
    if ff_lead is not None:
      if lf_lead is not None and lf_lead.trackId == ff_lead.trackId:
        lf_lead = None
      if rf_lead is not None and rf_lead.trackId == ff_lead.trackId:
        rf_lead = None
    # Shared physical coordinates and filter history follow the ID across LF/FF/RF.
    corrected = display_tracker.position_correction is not None
    filtered_positions = []
    for slot, (point, aligned, is_lane) in enumerate(((ff_lead, ff_yRel, ff_uses_lane),
                                                    (lf_lead, lf_yRel, True), (rf_lead, rf_yRel, True))):
      reference = references[slot]
      if corrected and point is not None:
        changing = (CS.out.leftBlinker or CS.out.rightBlinker
                    or (md is not None and str(getattr(md.meta, 'laneChangeState', None)) != 'off'))
        aligned, reference = display_tracker.align_display_position(point, aligned, reference, changing)
      distance, lateral = display_tracker.filter_position(point, aligned, reference, frame) if point is not None else (0.0, 0.0)
      if corrected and point is not None:
        lateral = display_tracker.correct_position(point, aligned, lateral, slot, reference, frame)
      filtered_positions.append((distance, lateral))
    ff_yRel = filtered_positions[0][1]
    lf_yRel = min(max(float(filtered_positions[1][1]), -5.4), 5.4)
    rf_yRel = min(max(float(filtered_positions[2][1]), -5.4), 5.4)
    ff_yRel, _ = display_tracker.finish(ff_lead, lf_lead, rf_lead, ff_yRel, ff_lane_mode, frame)

    # 전방(FF) 차량 정보 업데이트
    if ff_lead:
      values["FF_DISTANCE"] = filtered_positions[0][0] * 0.8
      values["FF_LATERAL"] = -ff_yRel if corrected else apply_curved_deadband(-ff_yRel, 0, 0.7, 1)
      values["FF_DETECT"] = 2 if ff_lead.vLead < 3 else state.ff_detect.apply(ff_lead.vRel)
    else:
      values["FF_DETECT"] = 0 # 순정 디텍션 제거
    # LF/RF 횡거리는 4m로 압축하지 않고 CAN 신호 범위(7-bit unsigned, 0.1m)만 제한합니다.
    # 전방 좌측(LF) 차량 정보 업데이트
    if lf_lead:
      values["LF_DETECT_DISTANCE"] = filtered_positions[1][0] * 0.8
      values["LF_DETECT_LATERAL"] = min(max(float(lf_yRel if corrected else apply_curved_deadband(lf_yRel, state.lf_center.value, 0.9, 2)), 0.0), 12.7)
      values["LF_DETECT"] = state.lf_detect.apply(lf_lead.vRel)
    else:
      values["LF_DETECT"] = 0
    # 전방 우측(RF) 차량 정보 업데이트
    if rf_lead:
      values["RF_DETECT_DISTANCE"] = filtered_positions[2][0] * 0.8
      values["RF_DETECT_LATERAL"] = min(max(float(-rf_yRel if corrected else apply_curved_deadband(-rf_yRel, state.rf_center.value, 0.9, 2)), 0.0), 12.7)
      values["RF_DETECT"] = state.rf_detect.apply(rf_lead.vRel)
    else:
      values["RF_DETECT"] = 0

    display_tracker.stopped_display(values, (lf_lead, rf_lead), frame)
    if corrected:
      display_tracker.position_correction.stabilize_slots(
        values, display_tracker.selected, display_tracker.tracks, frame,
        bypass=display_tracker.stopped or display_tracker.approaching or bool(display_tracker.approach_holds))

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


def configure(lane_color, model_lanes, radar_vehicles, position_correction=False, model_lane_mode=2):
  global _options
  options = (lane_color, model_lanes, radar_vehicles, position_correction, model_lane_mode)
  if options == _options:
    return
  if _options is None or lane_color != _options[0]:
    state.drive_lane_color = LaneHighlightStateMachine()
    state.drive_mode = 3
    state.drive_mode_refresh = -math.inf
  if _options is None or model_lanes != _options[1] or model_lane_mode != _options[4]:
    reset_lanes()
  state.lane_curve_mode = model_lane_mode
  if _options is None or radar_vehicles != _options[2]:
    reset_vehicles()
  if _options is None or radar_vehicles != _options[2] or position_correction != _options[3]:
    state.radar_display_tracker.position_correction = (
      _CcncVehiclePositionCorrection(state.lf_center.value, state.rf_center.value)
      if radar_vehicles and position_correction else None)
    state.radar_display_tracker.temporal = _CcncTemporalTracks() if radar_vehicles and position_correction else None
  _options = options


def reset_lanes():
  state.lane_geometry = _CcncLaneGeometry()
  state.lane_curv = NoiseFilter(3, 0, alpha_range=0.5)
  state.sla_active_time = 0
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
  configure(False, False, False)


reset()
