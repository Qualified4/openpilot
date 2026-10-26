"""Opt-in display memory only. No radar-map writes or control consumers.

Distances are CCNC display metres (the existing front projection is 0.8).
Missing rear observations never become radar measurements or new identities.

Constraint doc section 3 item 2 ("trackState 3 is never first-class evidence") governs admitting new objects and
structure checks; it does not apply to extending an LF/RF car that already qualified as it leaves (DaBin, 10-19).

Phases of rears[side]: 'coast' starts the moment a qualified car leaves LF/RF (replaced by a new object, its report
lost, the map no longer proving it, or its selection ended) and already draws LR/RR at 0, so the car is never
missing from every slot (DaBin, 10-19: hand it to LR as soon as it will be handed over). While the extrapolated x has
not passed the origin LR/RR stays at 0, unless another LF/RF needs the 6 m drawing gap (bounded only by MAX_TIME);
'rear' then moves it out to 16. The only pass exclusion is a car still 3 m or more ahead (reported or extrapolated); a stopped car
(vLead < 0) and a car leaving for the outer lane are not handed over either; BSD wins.

vRel is the median of the same radar id's last reports (rain, 10-20): the raw value jumps 2-6 m/s between reports.

Unreported motion is extrapolated as the saved vRel scaled by the ego speed of each send over the ego speed when it
was saved, clamped to 0-1.5 (DaBin, 10-19): the gap closes in proportion to how far the ego car still travels. When
the ego car stops beside a car it stays put (LR/RR at 0 until MAX_TIME), and braking through a real pass still reaches
the origin. A car saved with the ego car below 0.5 m/s does not move at all.
"""
from collections import deque
import math
import statistics

from opendbc.car.hyundai.hyundaicanfd_ccnc_side_flow import SideFlow


class RearHandover:
  SCALE = .8
  END_DISTANCE = 16.
  MIN_FRONT_REAR_GAP = 6.  # Display metres: positive front distance + positive rear distance on the same side.
  CONFIRM_TIME = .3
  FRONT_COAST_TIME = 1.
  COAST_MIN_X = -2.
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
  # Rain (00000404): one car's vRel jumps 2-6 m/s between 0.05 s reports while x/y stay within 1 m. Identity,
  # qualification, extrapolation and retreat use the median of the last VREL_N reports of the same radar id, and the
  # identity gates rely on x/y: the vRel gate only rejects a different object (it was 2 m/s, which reset 26 qualified
  # cars near the origin in that drive).
  VREL_N = 7  # a count of reports (0.35 s at 20 Hz), not a time window
  X_GATE = 1.
  VREL_GATE = 8.
  # Not stopped (median vLead), relative to ego: a 0.7 m/s object passed at 21 m/s is a structure in Doppler noise (3dd/5 35.3 s);
  # a slow queue passed at 7-14 m/s (404/10 33-40 s, 1.5-2.5 m/s) is not. Without radar motion (motionState 2 under
  # 3 reports, map proof only) the object must also move at MOVING_VLEAD.
  MOVING_FRACTION = .15
  # Radar-moving objects (motionState 2 at least 3 times): below 10 m/s ego this stays at most vLead >= 0 as before,
  # so a queue car stopped beside a creeping ego still hands over (3fd/4 DaBin 10-19, 404/20 29.4 s review 10-20);
  # at 18.5-21 m/s it needs 1.3-1.7 m/s (structures 3dd/5, 3e1/6). Map proof only: also MOVING_VLEAD.
  MOVING_SLACK = 1.5
  MOVING_SLACK_MAP = .5
  MOVING_VLEAD = 1.
  MOVING_N = 40  # 40 reports: 2 s at 20 Hz; fewer or missing reports make it longer (a count, not a time window)
  # MAX_TIME no longer removes the car where it is drawn: it slides out to 16 at least this fast (display m/s),
  # bounded by MAX_TIME + EXIT_GRACE.
  EXIT_FORCED_SPEED = 4.
  EXIT_GRACE = 4.

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
    self.changing = False  # blinker/lane-change block of the last update; SideFlow retires its non-BSD R with it
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
    return (0 < elapsed <= limit and abs(point['x']-saved['x']-self._dx(saved, saved['seen'], point['stamp'])) <= self.X_GATE+2*elapsed
            and abs(point['y']-saved['y']) <= (self.QUALIFIED_Y_GATE if saved['qualified'] else .25)+elapsed
            and abs(point['vrel']-saved['vrel']) <= self.VREL_GATE+2*elapsed)

  def _median(self, saved, name, raw, n=None):
    # A plain list: memories hold no reader or container types besides lists (reader-lifetime test).
    hist = saved.setdefault(name, [])
    hist.append(raw)
    del hist[:-(n or self.VREL_N)]
    return statistics.median(hist)

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
    # Stopped: the last 3 reports (0.15 s), so a car that just stopped is caught; one noisy report is not enough.
    recent = list(saved.get('vlead_hist', []))
    if point is not None and point['stamp'] > saved['seen']:
      recent.append(point['vlead'])  # this send's report is not in the history yet (review 10-20 item 2)
    recent = recent[-3:]
    vlead = statistics.median(recent) if recent else saved['vlead']
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
    self.consumed[side].extend([saved['key'], *saved['aliases']])
    saved.update(phase='coast', distance=0., shown=now, start=None, speed=0.,
                 coast_y=point['y'] if point is not None else saved['y'])
    self.rears[side] = saved
    filters[side].reset(0.)
    self.events.append({'side': side, 'event': 'coast', 'reason': reason, 'source': saved['source'], 'x': saved['x'],
                        'vrel': saved['vrel'], 'vlead': saved['vlead'], 'v_anchor': saved['v_anchor'], 'v_ego': self.v_ego,
                        'front_distance': saved['signed']})

  def update(self, values, CS, md, frame, selected, filters, context, *, proven=(True, True), rear_laterals=None):
    """selected: (source id, identity epoch) for FF/LF/RF; inputs stay untouched."""
    now = frame*.01
    dt = now-self.last if self.last is not None else 0.
    self.events = []
    blind = bool(CS.out.leftBlindspot), bool(CS.out.rightBlindspot)
    meta = getattr(md, 'meta', None)
    changing = (getattr(CS.out, 'leftBlinker', False) or getattr(CS.out, 'rightBlinker', False)
                or str(getattr(meta, 'desire', '')) in ('laneChangeLeft', 'laneChangeRight')
                or str(getattr(meta, 'laneChangeState', 'off')) != 'off')
    self.changing = changing
    probs = list(getattr(md, 'laneLineProbs', ()))  # pycapnp list readers reject slicing
    blocked = ('clock_or_context' if self.context != context or not 0 < dt <= .2 else
               'reverse' if str(getattr(CS.out, 'gearShifter', '')) == 'reverse' else
               'lane_change' if changing else None)
    # Per side (was: either lane line below 0.1 blocked both sides, 10-20 segment 15: the right line's 0.097 near a
    # stop line dropped a left car 0.1 s before its handoff). A side whose own line is lost drops only its front
    # memory and acquires nothing new; a car already handed to LR/RR keeps coasting/retiring, bounded by MAX_TIME
    # and END_DISTANCE as always.
    lost = [md is None or len(probs) < 3 or not getattr(CS.out, 'canValid', True)
            or not (math.isfinite(probs[1+side]) and probs[1+side] >= .1) for side in (0, 1)]
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
      if lost[side]:
        self._drop(self.fronts, side, 'confidence_lost', values, filters)
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
      if not lost[side]:
        self._observe_front(side, front, values, filters, key, points, proven[side] or once, once, now)
      self._coast(side, front, values, filters, key, points, now)
      saved = self.rears[side]
      if saved is None:
        continue
      # A car still at LR/RR 0 (coast: not past the origin) is removed at MAX_TIME as before. One already moving back
      # slides on to 16 instead of vanishing mid-screen, bounded by MAX_TIME + EXIT_GRACE (review 10-20: kept
      # separate from the qualification changes).
      if now-saved['shown'] >= self.MAX_TIME+(self.EXIT_GRACE if saved['phase'] == 'rear' else 0.):
        self._drop(self.rears, side, 'timeout', values, filters)
        continue
      if saved['phase'] == 'rear' and now-saved['shown'] >= self.MAX_TIME and not saved.get('forced'):
        saved['forced'] = now
        self.events.append({'side': side, 'event': 'forced_exit', 'source': saved['source'], 'distance': saved['distance']})
      if saved['phase'] == 'rear' and saved['start'] < now:
        # Integrate a bounded ramp analytically, including its start/cap within
        # a step. Changing the send interval cannot create a position jump.
        ramp_start = max(saved['seen']+self.EXIT_DELAY, saved['start'])
        if saved.get('forced'):
          ramp_start = min(ramp_start, saved['forced'])
        ramp0, ramp1 = max(0., now-dt-ramp_start), max(0., now-ramp_start)
        base = self._retreat(saved)
        if saved.get('forced'):
          base = max(base, self.EXIT_FORCED_SPEED)
        cap = max(0., (self.EXIT_MAX_SPEED-base)/self.EXIT_ACCEL)

        def integral(t, cap_time=cap):
          q = min(t, cap_time)
          return .5*self.EXIT_ACCEL*q*q+self.EXIT_ACCEL*cap_time*max(0., t-cap_time)
        saved['distance'] += base*dt+integral(ramp1)-integral(ramp0)
        saved['speed'] = min(self.EXIT_MAX_SPEED, base+self.EXIT_ACCEL*ramp1)
        if saved['distance'] >= self.END_DISTANCE:
          self._drop(self.rears, side, 'distance_end', values, filters)
          continue
      front_distance = values[front+'_DETECT_DISTANCE']
      if values[front+'_DETECT'] and 0. <= front_distance < self.MIN_FRONT_REAR_GAP:
        # Move only the virtual rear drawing, never forward again; ceil protects the 0.1 m CAN packing boundary.
        saved['distance'] = max(saved['distance'], math.ceil(10*(self.MIN_FRONT_REAR_GAP-front_distance))/10)
      values[rear+'_DETECT_DISTANCE'] = saved['distance']
      values[rear+'_DETECT_LATERAL'] = saved['lateral']
      values[rear+'_DETECT'] = saved['detect']
      filters[side].reset(saved['distance'])  # BSD takes over from the actual drawn position.

  def _observe_front(self, side, front, values, filters, key, points, proven, hold, now):
    saved = self.fronts[side]
    # The map can re-key the same radar id (two map objects of one track, 404/14 38-40 s): the radar keeps the id while it
    # is the same object, so the memory continues under the new key; the old one stays an alias.
    # Only with a fresh raw report of that id continuous with the memory and a proven send: without one the new key
    # may be another object (a 20 m front car must not be merged into the passing one, review 10-20 item 1).
    rekey = None
    if saved and key is not None and key != saved['key'] and key[0] == saved['source'] and proven and not hold:
      rekey = self._sample(points.get(key[0]), self.raw_stamp)
    if (rekey is not None and rekey['stamp'] > saved['seen']
        and self._continuous(saved, rekey, self.STALE_HOLD+.05+1e-6)):
      if saved['key'] not in saved['aliases']:
        saved['aliases'].append(saved['key'])
      saved['aliases'] = [k for k in saved['aliases'] if k != key]  # The map re-keyed the same radar id: keep its history.
      saved['key'] = key
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
                     phase='front', qualified=False, aliases=[])
        self.fronts[side] = saved
      vrel, vlead = self._median(saved, 'vrel_hist', point['vrel']), self._median(saved, 'vlead_hist', point['vlead'])
      vlead_long = self._median(saved, 'vlead_long', point['vlead'], self.MOVING_N)
      saved.update(x=point['x'], y=point['y'], vrel=vrel, vrel_raw=point['vrel'], vlead=vlead, v_anchor=self.v_ego,
                   seen=point['stamp'], update=now,
                   distance=float(values[front+'_DETECT_DISTANCE']), lateral=float(values[front+'_DETECT_LATERAL']),
                   drawn_distance=float(values[front+'_DETECT_DISTANCE']), drawn_stamp=now,
                   drawn_lateral=float(values[front+'_DETECT_LATERAL']), detect=int(values[front+'_DETECT']))
      saved['hits'] += 1
      saved['radar_moving'] = saved.get('radar_moving', 0)+int(point['vehicle'])
      # Map path: the map's vehicle proof (moving or Doppler-confirmed, not structure/crossing) is motion evidence; the
      # radar reports a slow queue the ego passes as motionState 0 throughout (404/10 33-40 s).
      saved['moving_hits'] += int(point['vehicle'] or (self.context is not None and self.context[0] == 'map'))
      # Once acquired, qualification survives later motionState/trackState changes (the radar drops both alongside).
      saved['qualified'] = saved['qualified'] or (
        saved['hits'] >= 4 and saved['moving_hits'] >= 3 and now-saved['first'] >= self.CONFIRM_TIME
        and saved['first_x']-point['x'] >= .25 and -.3 >= vrel  # A fast pass is kept; its retreat is capped at 8.
        # vRel is radial: alongside, a structure's relative speed falls toward 0 and its vLead rises (3e1/6 43.2 s: 0.4
        # -> 2.6 m/s within 1 s at 18.5 m/s). Judge "moving" by the median over the whole recent history.
        and vlead_long >= (self.MOVING_FRACTION*self.v_ego-self.MOVING_SLACK if saved['radar_moving'] >= 3 else
                           max(self.MOVING_FRACTION*self.v_ego-self.MOVING_SLACK_MAP, self.MOVING_VLEAD)))
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
    """Wait for the extrapolated origin pass, then retire; a same-side front may have already spaced the drawing back."""
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
      vrel, vlead = self._median(saved, 'vrel_hist', point['vrel']), self._median(saved, 'vlead_hist', point['vlead'])
      saved.update(signed=self._signed(saved, now)+saved['rate']*(point['x']-predicted), update=now,
                   x=point['x'], y=point['y'], vrel=vrel, vrel_raw=point['vrel'], vlead=vlead, v_anchor=self.v_ego,
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
                        'distance': saved['distance'], 'seen_age': age, 'speed': saved['speed'], 'lateral': saved['lateral'],
                        'coast_time': now-saved['shown'], 'origin': signed <= 0})


class SmoothHandover(RearHandover):
  """Keep the existing evidence/identity machinery and animate only the final side-slot drawings."""
  MIN_FRONT_REAR_GAP = 0.  # The renderer eases R back for spacing; never jump the physical model back.
  FIELDS = (('LF_DETECT', 'LF_DETECT_DISTANCE', 'LF_DETECT_LATERAL', 'LR_DETECT', 'LR_DETECT_DISTANCE', 'LR_DETECT_LATERAL'),
            ('RF_DETECT', 'RF_DETECT_DISTANCE', 'RF_DETECT_LATERAL', 'RR_DETECT', 'RR_DETECT_DISTANCE', 'RR_DETECT_LATERAL'))

  def __init__(self):
    super().__init__()
    self.flows = [SideFlow(), SideFlow()]

  def update(self, values, CS, md, frame, selected, filters, context, *, proven=(True, True), rear_laterals=None):
    now = frame*.01
    dt = now-self.last if self.last is not None else 0.
    reset = self.context != context or not 0 < dt <= .2 or not getattr(CS.out, 'canValid', True)
    reverse = str(getattr(CS.out, 'gearShifter', '')) == 'reverse'
    previous_rears = tuple(self.rears)
    super().update(values, CS, md, frame, selected, filters, context, proven=proven)
    for side, saved in enumerate(previous_rears):
      if saved is not None and saved is not self.rears[side]:
        reasons = [e['event'] for e in self.events if e['side'] == side and e.get('source') == saved['source']
                   and e.get('phase') in ('coast', 'rear')]
        saved['end_reason'] = reasons[-1] if reasons else 'cancelled'
    if reset or reverse:
      self.flows = [SideFlow(), SideFlow()]
    if reverse or not getattr(CS.out, 'canValid', True):
      return
    dt = 0. if reset else dt
    rear_laterals = rear_laterals or (2.8, 2.8)
    for side, (fd, fx, fy, rd, rx, ry) in enumerate(self.FIELDS):
      x, y = values[fx], values[fy]
      key = selected[side+1]
      candidate = None
      if values[fd] and math.isfinite(x) and math.isfinite(y) and 0. <= x <= SideFlow.FRONT_LIMIT:
        candidate = {'key': key if key is not None else ('display', side), 'x': x, 'y': y, 'detect': values[fd]}
      motion = self.rears[side]
      if motion is not None:
        motion.update(force_at=motion['shown']+self.MAX_TIME, deadline=motion['shown']+self.MAX_TIME+self.EXIT_GRACE,
                      max_speed=self.EXIT_MAX_SPEED)
      f, r = self.flows[side].update(now, dt, candidate, bool(CS.out.leftBlindspot if side == 0 else CS.out.rightBlindspot),
                                   rear_laterals[side], elsewhere=(selected[0], selected[2-side]),
                                   motion=motion, lane_change=self.changing)
      values[fd] = f['detect'] if f is not None else 0
      values[rd] = r['detect'] if r is not None else 0
      if f is not None:
        values[fx], values[fy] = f['x'], f['y']
      if r is not None:
        values[rx], values[ry] = -r['x'], r['y']
      filters[side].reset(-r['x'] if r is not None else 16.)
      if self.flows[side].events:
        self.events.extend({'side': side, 'event': event} for event in self.flows[side].events)
