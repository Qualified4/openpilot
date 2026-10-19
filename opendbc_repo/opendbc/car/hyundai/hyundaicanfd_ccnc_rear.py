"""Opt-in display memory only. No radar-map writes or control consumers.

Distances are CCNC display metres (the existing front projection is 0.8).
Missing rear observations never become radar measurements or new identities.

Constraint doc section 3 item 2 ("trackState 3 is never first-class evidence") governs admitting new objects and
structure checks; it does not apply to extending an LF/RF car that already qualified as it leaves (DaBin, 10-19).

Phases of rears[side]: 'coast' starts the moment a qualified car leaves LF/RF (replaced by a new object, its report
lost, the map no longer proving it, or its selection ended) and already draws LR/RR at 0, so the car is never
missing from every slot (DaBin, 10-19: hand it to LR as soon as it will be handed over). While the extrapolated x has
not passed the origin LR/RR stays at 0, however slow the pass (bounded only by MAX_TIME); 'rear' then moves it
0 -> 16. The only pass exclusion is a car still 3 m or more ahead (reported or extrapolated); a stopped car
(vLead < 0) and a car leaving for the outer lane are not handed over either; BSD wins.

Unreported motion is extrapolated as the saved vRel scaled by the ego speed of each send over the ego speed when it
was saved, clamped to 0-1.5 (DaBin, 10-19): the gap closes in proportion to how far the ego car still travels. When
the ego car stops beside a car it stays put (LR/RR at 0 until MAX_TIME), and braking through a real pass still reaches
the origin. A car saved with the ego car below 0.5 m/s does not move at all.
"""
from collections import deque
import math


class RearHandover:
  SCALE = .8
  END_DISTANCE = 16.
  CONFIRM_TIME = .3
  FRONT_COAST_TIME = 1.
  COAST_MIN_X = -2.
  # Measured vRel noise is 1.2-1.6 m/s between 0.05 s reports (16-segment replay); 1+2*dt reset qualification.
  VREL_GATE = 2.
  # Alongside, the same radar id's yRel moves 0.3-0.5 m in one report (3fe/3 27.0): it must not reset a qualified car.
  QUALIFIED_Y_GATE = 1.
  # Trial display-retirement bounds, evaluated by replay, not physical rear motion.
  EXIT_DELAY = .8
  EXIT_ACCEL = 1.2
  EXIT_MAX_SPEED = 8.
  MAX_TIME = 6.
  # A different object closer than this (raw metres, y metres, m/s) to the coasting car is that car's new radar id.
  ALIAS_GAP = 2.
  ALIAS_Y = 1.
  ALIAS_VREL = 2.
  # Half a lane: a coasting car reported this far sideways from where it left LF/RF is leaving for the outer lane.
  LANE_HALF = 1.75
  # card passes liveTracks None when it is older than 0.15 s; a stale message is no news, not a lost report.
  TRACKS_FRESH = .15
  STALE_HOLD = .3
  # Ego speed history kept for extrapolation: the longest span is a coast with no report (1 s + MAX_TIME).
  EGO_SPAN = 8.
  # Extrapolation scale vEgo now / vEgo at the anchor: bounded, and none when the anchor was at a standstill.
  RATIO_MAX = 1.5
  ANCHOR_MIN_EGO = .5

  def __init__(self):
    self.fronts = [None, None]
    self.rears = [None, None]
    self.consumed = [deque(maxlen=16), deque(maxlen=16)]
    self.context = self.last = self.raw = None
    self.raw_stamp = -math.inf
    self.fresh = False
    self.unproven = [False, False]
    self.events = []
    self.v_ego = 0.
    self.ego_hist = deque()  # (send time, vEgo, dt) of every send

  @property
  def sides(self):
    return [rear or front for front, rear in zip(self.fronts, self.rears, strict=True)]

  def _drop(self, memories, side, reason, values, filters, *, bsd=False):
    saved = memories[side]
    if saved is not None:
      self.events.append({'side': side, 'event': reason, 'source': saved['source'], 'phase': saved['phase'],
                          'distance': saved['distance'], 'age': self.last-saved['seen'],
                          'qualified': saved.get('qualified'), 'hits': saved.get('hits')})
      if saved['phase'] in ('coast', 'rear') and not bsd:
        values[('LR', 'RR')[side]+'_DETECT'] = 0
        filters[side].reset(15.)
    memories[side] = None

  @staticmethod
  def _sample(point, stamp):
    # trackState/measured are not inputs: this radar turns an alongside car into trackState 3 (motionState 0) near
    # x 2.5 m while its reported x keeps following the car. Motion evidence (motionState 2) still gates qualification.
    if point is None or str(point.radarSource) != 'frontRadar':
      return None
    numbers = point.dRel, point.yRel, point.vRel, point.vLead
    if not all(math.isfinite(v) for v in numbers):
      return None
    return {'x': float(point.dRel), 'y': float(point.yRel), 'vrel': float(point.vRel), 'vlead': float(point.vLead),
            'vehicle': getattr(point, 'motionState', 0) == 2, 'stamp': stamp}

  def _ratio(self, saved, v_ego):
    anchor = saved['v_anchor']
    return 0. if anchor < self.ANCHOR_MIN_EGO else min(max(v_ego/anchor, 0.), self.RATIO_MAX)

  def _dx(self, saved, t0, t1):
    """Raw metres the car moves relative to the ego from t0 to t1 (send times): the saved vRel times each send's dt,
    scaled by that send's vEgo over the anchor's."""
    return saved['vrel']*sum(dt*self._ratio(saved, v) for t, v, dt in self.ego_hist if t0 < t <= t1)

  def _retreat(self, saved):
    # Display m/s the car falls behind at now, with the same scale; never back toward the origin.
    return min(self.EXIT_MAX_SPEED, max(0., -saved['vrel']*self._ratio(saved, self.v_ego)*self.SCALE))

  def _continuous(self, saved, point, limit):
    elapsed = point['stamp']-saved['seen']
    return (0 < elapsed <= limit and abs(point['x']-saved['x']-self._dx(saved, saved['seen'], point['stamp'])) <= .5+2*elapsed
            and abs(point['y']-saved['y']) <= (self.QUALIFIED_Y_GATE if saved['qualified'] else .25)+elapsed
            and abs(point['vrel']-saved['vrel']) <= self.VREL_GATE+2*elapsed)

  def _anchor_rate(self, saved):
    # Display metres per raw metre at the anchor, so the drawn distance reaches 0 when the extrapolated raw x does
    # (the origin pass). The map draws close to 1:1 near the origin, the selector about 0.8.
    x = saved['x']+self._dx(saved, saved['seen'], saved['update'])
    saved['rate'] = min(max(saved['signed']/x, self.SCALE), 1.5) if x > .1 else self.SCALE

  def _signed(self, saved, now):
    """Display-metre position the car would have in LF/RF; it reaches 0 when the extrapolated raw x does."""
    return saved['signed']+saved['rate']*self._dx(saved, saved['update'], now)

  def _reachable(self, saved, now):
    # No "reaches the origin within 1 s" bound (DaBin, 10-19): a slow pass waits at LR/RR 0 for the origin.
    age = now-saved['seen']
    return (saved['qualified'] and age <= self.FRONT_COAST_TIME
            and saved['x']+self._dx(saved, saved['seen'], now) >= self.COAST_MIN_X)

  def _start_coast(self, side, reason, values, filters, points, now):
    """A qualified car left LF/RF: draw it at LR/RR 0 now (the 'coast' event), not when it passes the origin."""
    saved = self.fronts[side]
    if saved['drawn_stamp'] > saved['update']:
      saved.update(distance=saved['drawn_distance'], update=saved['drawn_stamp'], lateral=saved['drawn_lateral'])
    saved['signed'] = saved['distance']
    self._anchor_rate(saved)
    point = self._sample(points.get(saved['source']), self.raw_stamp)
    if point is not None and point['stamp'] > saved['seen'] and not self._continuous(saved, point, self.FRONT_COAST_TIME):
      point = None  # The radar id now describes another object: no news about this car.
    x = point['x'] if point is not None else saved['x']+self._dx(saved, saved['seen'], now)
    vlead = point['vlead'] if point is not None else saved['vlead']
    # Was x >= 1 m. This radar keeps reporting a passing car at 1-3 m (trackState 3) while the map still draws or
    # has just replaced it, so 1 m discarded every real pass. A car 3 m ahead only lost its slot: it did not pass.
    # Qualification is kept once acquired (motion evidence); stopping (vLead < 0) is checked again here.
    rejected = (reason if not self._reachable(saved, now) else 'still_ahead' if x >= 3. else
                'stopped' if vlead < 0. else None)
    if rejected:
      self._drop(self.fronts, side, rejected, values, filters)
      return
    self.fronts[side] = None
    self._drop(self.rears, side, 'rear_replaced', values, filters)  # One rear slot per side: the newest car wins.
    self.consumed[side].append(saved['key'])
    saved.update(phase='coast', aliases=[], distance=0., shown=now, start=None, speed=0.,
                 coast_y=point['y'] if point is not None else saved['y'])
    self.rears[side] = saved
    filters[side].reset(0.)
    self.events.append({'side': side, 'event': 'coast', 'reason': reason, 'source': saved['source'], 'x': saved['x'],
                        'vrel': saved['vrel'], 'vlead': saved['vlead'], 'v_anchor': saved['v_anchor'], 'v_ego': self.v_ego,
                        'front_distance': saved['signed']})

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
    v_ego = float(getattr(CS.out, 'vEgo', 0.))
    self.v_ego = v_ego if math.isfinite(v_ego) else 0.
    if blocked == 'clock_or_context':
      self.ego_hist.clear()  # Every memory is dropped below; times before this are not comparable.
    self.ego_hist.append((now, self.v_ego, dt if blocked != 'clock_or_context' else 0.))
    while self.ego_hist[0][0] < now-self.EGO_SPAN:
      self.ego_hist.popleft()
    if CS.live_tracks is not self.raw:
      self.raw, self.raw_stamp = CS.live_tracks, now
    # Fresh: the latest message is current, so an id missing from it is a lost report. Stale (None or old): no news.
    self.fresh = self.raw is not None and now-self.raw_stamp <= self.TRACKS_FRESH
    points = {int(p.trackId): p for p in getattr(self.raw, 'points', ())} if self.fresh else {}
    for side, (front, rear) in enumerate((('LF', 'LR'), ('RF', 'RR'))):
      key = selected[side+1]
      # The map's proof (structure/crossing/superseded) is debounced by one send: a single False holds.
      once = not proven[side] and not self.unproven[side]
      self.unproven[side] = not proven[side]
      if blind[side] or blocked:
        for memories in (self.fronts, self.rears):
          self._drop(memories, side, 'bsd' if blind[side] else blocked, values, filters, bsd=blind[side])
        continue  # The existing 8/16 BSD policy already ran, exactly once.
      for memories in (self.fronts, self.rears):
        saved = memories[side]
        if saved and {saved['key'], *saved.get('aliases', ())} & {selected[0], selected[2-side]}:
          self._drop(memories, side, 'changed_lane_slot', values, filters)
      if key is not None and key in self.consumed[side]:
        saved = self.rears[side]
        point = self._sample(points.get(key[0]), self.raw_stamp)
        own = saved is not None and (key == saved['key'] or key in saved['aliases'])
        if own and point is not None and point['x'] >= 1. and point['vrel'] > .3:
          # A new measured forward observation wins; virtual memory never
          # becomes a radar observation or another forward handoff candidate.
          self._drop(self.rears, side, 'measured_front_reacquired', values, filters)
        elif own or values[front+'_DETECT_DISTANCE'] <= self.SCALE:
          values[front+'_DETECT'] = 0
        key = None  # Never seed another pass from any consumed identity epoch.
      self._observe_front(side, front, values, filters, key, points, proven[side] or once, once, now)
      self._coast(side, front, values, filters, key, points, now)
      saved = self.rears[side]
      if saved is None:
        continue
      if now-saved['shown'] >= self.MAX_TIME:
        self._drop(self.rears, side, 'timeout', values, filters)
        continue
      if saved['phase'] == 'rear' and saved['start'] < now:
        # Integrate a bounded ramp analytically, including its start/cap within
        # a step. Changing the send interval cannot create a position jump.
        ramp_start = max(saved['seen']+self.EXIT_DELAY, saved['start'])
        ramp0, ramp1 = max(0., now-dt-ramp_start), max(0., now-ramp_start)
        base = self._retreat(saved)
        cap = max(0., (self.EXIT_MAX_SPEED-base)/self.EXIT_ACCEL)

        def integral(t, cap_time=cap):
          q = min(t, cap_time)
          return .5*self.EXIT_ACCEL*q*q+self.EXIT_ACCEL*cap_time*max(0., t-cap_time)
        saved['distance'] += base*dt+integral(ramp1)-integral(ramp0)
        saved['speed'] = min(self.EXIT_MAX_SPEED, base+self.EXIT_ACCEL*ramp1)
        if saved['distance'] >= self.END_DISTANCE:
          self._drop(self.rears, side, 'distance_end', values, filters)
          continue
      values[rear+'_DETECT_DISTANCE'] = saved['distance']
      values[rear+'_DETECT_LATERAL'] = saved['lateral']
      values[rear+'_DETECT'] = saved['detect']
      filters[side].reset(saved['distance'])  # BSD takes over from the actual drawn position.

  def _observe_front(self, side, front, values, filters, key, points, proven, hold, now):
    saved = self.fronts[side]
    if saved and key is not None and key != saved['key']:
      if saved['qualified']:
        self._start_coast(side, 'object_replaced', values, filters, points, now)
      else:
        self._drop(self.fronts, side, 'object_replaced', values, filters)
      saved = self.fronts[side] = None  # This never discards an unrelated retiring rear vehicle.
    if key is not None and saved is None and self._alias(side, key, points, values, front, now):
      return
    if key is not None and values[front+'_DETECT']:
      if saved is not None:
        saved.update(drawn_distance=float(values[front+'_DETECT_DISTANCE']), drawn_stamp=now,
                     drawn_lateral=float(values[front+'_DETECT_LATERAL']))
      point = self._sample(points.get(key[0]), self.raw_stamp)
      if hold:
        return  # One unproven send: neither an observation nor a reason to leave LF/RF.
      if saved is not None and saved['qualified'] and (not proven or point is None):
        if proven and not self.fresh and now-saved['seen'] <= self.STALE_HOLD+1e-6:
          return  # liveTracks stale or None: keep LF/RF as drawn, as for any front prediction.
        self._start_coast(side, 'not_proven' if not proven else 'measurement_lost', values, filters, points, now)
        if self.rears[side] is saved:
          values[front+'_DETECT'] = 0  # Same send: the car leaves LF/RF exactly when LR/RR 0 appears.
        return
      if not proven or point is None:
        if saved is not None and now-saved['seen'] <= self.STALE_HOLD+1e-6:
          return  # Existing front prediction is not another observation.
        self._drop(self.fronts, side, 'measurement_unqualified', values, filters)
        return
      if point['stamp'] <= (saved['seen'] if saved else -math.inf):
        return
      # Was .2: a report back after the whole stale hold (0.3 s, one send late) is still this car.
      continuous = saved is not None and self._continuous(saved, point, self.STALE_HOLD+.05+1e-6)
      if saved is not None and not continuous:
        self._drop(self.fronts, side, 'identity_innovation', values, filters)
      if not continuous:
        saved = dict(key=key, source=key[0], first=point['stamp'], first_x=point['x'], hits=0, moving_hits=0,
                     phase='front', qualified=False)
        self.fronts[side] = saved
      saved.update(x=point['x'], y=point['y'], vrel=point['vrel'], vlead=point['vlead'], v_anchor=self.v_ego,
                   seen=point['stamp'], update=now,
                   distance=float(values[front+'_DETECT_DISTANCE']), lateral=float(values[front+'_DETECT_LATERAL']),
                   drawn_distance=float(values[front+'_DETECT_DISTANCE']), drawn_stamp=now,
                   drawn_lateral=float(values[front+'_DETECT_LATERAL']), detect=int(values[front+'_DETECT']))
      saved['hits'] += 1
      saved['moving_hits'] += int(point['vehicle'])
      # Once acquired, qualification survives later motionState/trackState changes (the radar drops both alongside).
      saved['qualified'] = saved['qualified'] or (
        saved['hits'] >= 4 and saved['moving_hits'] >= 3 and now-saved['first'] >= self.CONFIRM_TIME
        and saved['first_x']-point['x'] >= .25 and -.3 >= point['vrel'] >= -self.EXIT_MAX_SPEED/self.SCALE
        and point['vlead'] >= 0.)
      return  # Measured front motion/position is left entirely unchanged.
    if saved is None:
      return
    if saved['qualified']:
      self._start_coast(side, 'selection_ended', values, filters, points, now)
    else:
      self._drop(self.fronts, side, 'no_qualified_zero_crossing', values, filters)

  def _alias(self, side, key, points, values, front, now):
    """True when a newly selected object sits where the coasting car is: the radar re-identified it. It joins the
    car's consumed identities, so the same car is never drawn in LF/RF and LR/RR at once."""
    coast = self.rears[side]
    if coast is None or coast['phase'] != 'coast':
      return False
    point = self._sample(points.get(key[0]), self.raw_stamp)
    x = point['x'] if point is not None else float(values[front+'_DETECT_DISTANCE'])/self.SCALE
    if not abs(x-(coast['x']+self._dx(coast, coast['seen'], now))) < self.ALIAS_GAP:
      return False  # NaN included
    if point is not None and not (abs(point['y']-coast['y']) <= self.ALIAS_Y
                                  and abs(point['vrel']-coast['vrel']) <= self.ALIAS_VREL):
      return False  # e.g. another car overtaking from behind, or one in the next lane
    coast['aliases'].append(key)
    self.consumed[side].append(key)
    values[front+'_DETECT'] = 0
    return True

  def _coast(self, side, front, values, filters, key, points, now):
    """LR/RR is drawn at 0 until the extrapolated x passes the origin, then the rear phase moves it back."""
    saved = self.rears[side]
    if saved is None or saved['phase'] != 'coast':
      return
    point = self._sample(points.get(saved['source']), self.raw_stamp)
    if point is not None and point['stamp'] > saved['seen'] and self._continuous(saved, point, self.FRONT_COAST_TIME):
      if point['x'] >= 3.:
        self._drop(self.rears, side, 'still_ahead', values, filters)  # It is still well ahead: not passing.
        return
      if abs(point['y']-saved['coast_y']) > self.LANE_HALF:
        self._drop(self.rears, side, 'lane_departed', values, filters)  # Leaving for the outer lane: not alongside.
        return
      # Any later report of this car (trackState 3 included) re-anchors the extrapolation, keeping the drawn offset.
      predicted = saved['x']+self._dx(saved, saved['seen'], point['stamp'])
      saved.update(signed=self._signed(saved, now)+saved['rate']*(point['x']-predicted), update=now,
                   x=point['x'], y=point['y'], vrel=point['vrel'], vlead=point['vlead'], v_anchor=self.v_ego,
                   seen=point['stamp'])
      self._anchor_rate(saved)
    # A discontinuous report is another object under the reused radar id: it is ignored, not a reason to drop.
    if saved['x']+self._dx(saved, saved['seen'], now) >= 3.:
      self._drop(self.rears, side, 'still_ahead', values, filters)  # Extrapolated 3 m ahead (DaBin, 10-19).
      return
    age = now-saved['seen']
    signed = self._signed(saved, now)
    saved['signed'], saved['update'] = signed, now
    if signed > 0:
      return  # However long it takes; MAX_TIME (from the LR/RR 0 start) still bounds the whole display.
    # Origin passed: LR/RR starts moving back from 0.
    saved.update(phase='rear', start=now, speed=self._retreat(saved))
    self.events.append({'side': side, 'event': 'handoff', 'source': saved['source'], 'key': saved['key'],
                        'distance': 0., 'seen_age': age, 'speed': saved['speed'], 'lateral': saved['lateral'],
                        'coast_time': now-saved['shown'], 'origin': signed <= 0})
