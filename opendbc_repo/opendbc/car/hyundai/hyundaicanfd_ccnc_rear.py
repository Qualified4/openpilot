"""Opt-in display memory only. No radar-map writes or control consumers.

Distances are CCNC display metres (the existing front projection is 0.8).
Missing rear observations never become radar measurements or new identities.
"""
from collections import deque
import math


class RearHandover:
  SCALE = .8
  END_DISTANCE = 16.
  CONFIRM_TIME = .3
  FRONT_COAST_TIME = 1.
  # Trial display-retirement bounds, evaluated by replay, not physical rear motion.
  EXIT_DELAY = .8
  EXIT_ACCEL = 1.2
  EXIT_MAX_SPEED = 8.
  MAX_TIME = 6.

  def __init__(self):
    self.fronts = [None, None]
    self.rears = [None, None]
    self.consumed = [deque(maxlen=16), deque(maxlen=16)]
    self.context = self.last = self.raw = None
    self.raw_stamp = -math.inf
    self.events = []

  @property
  def sides(self):
    return [rear or front for front, rear in zip(self.fronts, self.rears, strict=True)]

  def _drop(self, memories, side, reason, values, filters, *, bsd=False):
    saved = memories[side]
    if saved is not None:
      self.events.append({'side': side, 'event': reason, 'source': saved['source'], 'phase': saved['phase'],
                          'distance': saved['distance'], 'age': self.last-saved['seen']})
      if saved['phase'] == 'rear' and not bsd:
        values[('LR', 'RR')[side]+'_DETECT'] = 0
        filters[side].reset(15.)
    memories[side] = None

  @staticmethod
  def _sample(point, stamp):
    if point is None or str(point.radarSource) != 'frontRadar' or not getattr(point, 'measured', False):
      return None
    numbers = point.dRel, point.yRel, point.vRel, point.vLead
    if not all(math.isfinite(v) for v in numbers) or getattr(point, 'trackState', 0) != 2:
      return None
    return {'x': float(point.dRel), 'y': float(point.yRel), 'vrel': float(point.vRel), 'vlead': float(point.vLead),
            'vehicle': getattr(point, 'motionState', 0) == 2, 'stamp': stamp}

  def update(self, values, CS, md, frame, selected, filters, context, *, proven=(True, True)):
    """selected: (source id, identity epoch) for FF/LF/RF; inputs stay untouched."""
    now = frame*.01
    dt = now-self.last if self.last is not None else 0.
    self.events = []
    blind = bool(CS.out.leftBlindspot), bool(CS.out.rightBlindspot)
    meta = getattr(md, 'meta', None)
    changing = (getattr(CS.out, 'leftBlinker', False) or getattr(CS.out, 'rightBlinker', False)
                or str(getattr(meta, 'desire', '')) in ('laneChangeLeft', 'laneChangeRight')
                or str(getattr(meta, 'laneChangeState', 'off')) != 'off')
    probs = list(getattr(md, 'laneLineProbs', ()))  # pycapnp list readers reject slicing
    blocked = ('clock_or_context' if self.context != context or not 0 < dt <= .2 else
               'reverse' if str(getattr(CS.out, 'gearShifter', '')) == 'reverse' else
               'lane_change' if changing else
               'confidence_lost' if md is None or len(probs) < 3 or not all(math.isfinite(p) and p >= .1 for p in probs[1:3])
               or not getattr(CS.out, 'canValid', True) else None)
    self.context, self.last = context, now
    if CS.live_tracks is not self.raw:
      self.raw, self.raw_stamp = CS.live_tracks, now
    points = {int(p.trackId): p for p in getattr(self.raw, 'points', ())} if now-self.raw_stamp <= .15 else {}
    for side, (front, rear) in enumerate((('LF', 'LR'), ('RF', 'RR'))):
      key = selected[side+1]
      if blind[side] or blocked:
        for memories in (self.fronts, self.rears):
          self._drop(memories, side, 'bsd' if blind[side] else blocked, values, filters, bsd=blind[side])
        continue  # The existing 8/16 BSD policy already ran, exactly once.
      for memories in (self.fronts, self.rears):
        saved = memories[side]
        if saved and saved['key'] in (selected[0], selected[2-side]):
          self._drop(memories, side, 'changed_lane_slot', values, filters)
      if key is not None and key in self.consumed[side]:
        saved = self.rears[side]
        point = self._sample(points.get(key[0]), self.raw_stamp)
        if saved is not None and key == saved['key'] and point is not None and point['x'] >= 1. and point['vrel'] > .3:
          # A new measured forward observation wins; virtual memory never
          # becomes a radar observation or another forward handoff candidate.
          self._drop(self.rears, side, 'measured_front_reacquired', values, filters)
        elif (saved is not None and key == saved['key']) or values[front+'_DETECT_DISTANCE'] <= self.SCALE:
          values[front+'_DETECT'] = 0
        key = None  # Never seed another pass from any consumed identity epoch.
      self._observe_front(side, front, values, filters, key, points, proven[side], now)
      saved = self.rears[side]
      if saved is None:
        continue
      if saved['start'] < now:
        # Integrate a bounded ramp analytically, including its start/cap within
        # a step. Changing the send interval cannot create a position jump.
        ramp_start = max(saved['seen']+self.EXIT_DELAY, saved['start'])
        ramp0, ramp1 = max(0., now-dt-ramp_start), max(0., now-ramp_start)
        base = -saved['vrel']*self.SCALE
        cap = max(0., (self.EXIT_MAX_SPEED-base)/self.EXIT_ACCEL)

        def integral(t, cap_time=cap):
          q = min(t, cap_time)
          return .5*self.EXIT_ACCEL*q*q+self.EXIT_ACCEL*cap_time*max(0., t-cap_time)
        saved['distance'] += base*dt+integral(ramp1)-integral(ramp0)
        saved['speed'] = min(self.EXIT_MAX_SPEED, base+self.EXIT_ACCEL*ramp1)
        if saved['distance'] >= self.END_DISTANCE or now-saved['start'] >= self.MAX_TIME:
          self._drop(self.rears, side, 'distance_end' if saved['distance'] >= self.END_DISTANCE else 'timeout', values, filters)
          continue
      values[rear+'_DETECT_DISTANCE'] = saved['distance']
      values[rear+'_DETECT_LATERAL'] = saved['lateral']
      values[rear+'_DETECT'] = saved['detect']
      filters[side].reset(saved['distance'])  # BSD takes over from the actual drawn position.

  def _observe_front(self, side, front, values, filters, key, points, proven, now):
    saved = self.fronts[side]
    if saved and key is not None and key != saved['key']:
      self._drop(self.fronts, side, 'object_replaced', values, filters)
      saved = None  # This never discards an unrelated retiring rear vehicle.
    if key is not None and values[front+'_DETECT']:
      if saved is not None:
        saved.update(drawn_distance=float(values[front+'_DETECT_DISTANCE']), drawn_stamp=now,
                     drawn_lateral=float(values[front+'_DETECT_LATERAL']))
      point = self._sample(points.get(key[0]), self.raw_stamp)
      if not proven or point is None:
        if saved is not None and now-saved['seen'] <= .3+1e-6:
          return  # Existing front prediction is not another observation.
        self._drop(self.fronts, side, 'measurement_unqualified', values, filters)
        return
      if point['stamp'] <= (saved['seen'] if saved else -math.inf):
        return
      continuous = saved is not None
      if continuous:
        elapsed = point['stamp']-saved['seen']
        continuous = (0 < elapsed <= .2 and abs(point['x']-saved['x']-saved['vrel']*elapsed) <= .5+2*elapsed
                      and abs(point['y']-saved['y']) <= .25+elapsed
                      and abs(point['vrel']-saved['vrel']) <= 1.+2*elapsed)
        if not continuous:
          self._drop(self.fronts, side, 'identity_innovation', values, filters)
      if not continuous:
        saved = dict(key=key, source=key[0], first=point['stamp'], first_x=point['x'], hits=0, moving_hits=0,
                     phase='front', qualified=False)
        self.fronts[side] = saved
      saved.update(x=point['x'], y=point['y'], vrel=point['vrel'], seen=point['stamp'], update=now,
                   distance=float(values[front+'_DETECT_DISTANCE']), lateral=float(values[front+'_DETECT_LATERAL']),
                   drawn_distance=float(values[front+'_DETECT_DISTANCE']), drawn_stamp=now,
                   drawn_lateral=float(values[front+'_DETECT_LATERAL']), detect=int(values[front+'_DETECT']))
      saved['hits'] += 1
      saved['moving_hits'] += int(point['vehicle'])
      saved['qualified'] = (saved['hits'] >= 4 and saved['moving_hits'] >= 3 and now-saved['first'] >= self.CONFIRM_TIME
                            and saved['first_x']-point['x'] >= .25 and -.3 >= point['vrel'] >= -self.EXIT_MAX_SPEED/self.SCALE
                            and point['vlead'] >= 0.)
      return  # Measured front motion/position is left entirely unchanged.
    if saved is None:
      return
    if saved['drawn_stamp'] > saved['update']:
      # A displayed prediction anchors continuity without refreshing evidence.
      saved.update(distance=saved['drawn_distance'], update=saved['drawn_stamp'], lateral=saved['drawn_lateral'])
    point = self._sample(points.get(saved['source']), self.raw_stamp)
    if point is not None and point['x'] >= 1.:
      self._drop(self.fronts, side, 'selector_rejected', values, filters)
      return
    age = now-saved['seen']
    signed = saved['distance']+saved['vrel']*self.SCALE*(now-saved['update'])
    if (not saved['qualified'] or age > self.FRONT_COAST_TIME or saved['x'] > 6.
        or signed > -saved['vrel']*self.SCALE*max(0., self.FRONT_COAST_TIME-age)+.1):
      self._drop(self.fronts, side, 'no_qualified_zero_crossing', values, filters)
      return
    origin_distance = saved['distance']
    saved['distance'], saved['update'] = signed, now
    if signed > 0:
      values[front+'_DETECT_DISTANCE'] = signed
      values[front+'_DETECT_LATERAL'] = saved['lateral']
      values[front+'_DETECT'] = saved['detect']
      return
    # One rear slot per side: replace only at the new vehicle's qualified
    # crossing, never when it first appears in the front slot. No interpolation
    # between two different identities (old rear position -> new origin).
    old = self.rears[side]
    replaced_key = old['key'] if old is not None else None
    replaced_distance = old['distance'] if old is not None else None
    self._drop(self.rears, side, 'rear_replaced', values, filters)
    self.consumed[side].append(saved['key'])
    saved.update(phase='rear', distance=0., start=now, speed=-saved['vrel']*self.SCALE)
    self.rears[side], self.fronts[side] = saved, None
    filters[side].reset(0.)
    values[front+'_DETECT'] = 0
    self.events.append({'side': side, 'event': 'handoff', 'source': saved['source'], 'key': saved['key'],
                        'distance': 0., 'seen_age': age, 'speed': saved['speed'], 'lateral': saved['lateral'],
                        'origin_step': origin_distance, 'replaced_key': replaced_key, 'replaced_distance': replaced_distance})
