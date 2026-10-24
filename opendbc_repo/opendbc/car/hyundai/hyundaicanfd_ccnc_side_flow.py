"""Continuous display coordinates for one side. No radar/selection/control writes.

x is signed display metres: F >= 0, R <= 0. The existing selector/map still chooses and stabilises every real F.
BSD supplies presence only; its anonymous drawing never becomes a radar identity.
"""
import math


class SideFlow:
  HOLD = 3.
  END = 16.
  GAP = 6.
  SPACE_DISTANCE = 8.
  FRONT_LIMIT = 204.7  # Preserve the existing DBC range; the display usually shows only about 100 m.
  FAST_SPEED, FAST_TAU = 12., .2
  PASS_SPEED = 3.
  EXIT_SPEED, EXIT_TAU = 4., .65
  LATERAL_SPEED, LATERAL_TAU = 2., .5

  def __init__(self):
    self.front = self.rear = None
    self.bsd = False
    self.bsd_off = None
    self.after_pass = False
    self.events = []

  @staticmethod
  def _move(value, target, dt, speed, tau):
    error = target-value
    limit = speed*dt
    if abs(error) <= min(.04, limit):
      return target
    step = min(abs(error)*(-math.expm1(-dt/tau)), limit)
    return value+math.copysign(step, error)

  def _advance(self, obj, x, y, dt, speed, tau):
    obj['x'] = self._move(obj['x'], x, dt, speed, tau)
    obj['y'] = self._move(obj['y'], y, dt, self.LATERAL_SPEED, self.LATERAL_TAU)

  @staticmethod
  def _object(key, x, y, detect, phase):
    return {'key': key, 'x': x, 'y': y, 'detect': detect, 'phase': phase}

  def _new_bsd(self, y):
    self.rear = self._object(None, -self.END, y, 2, 'bsd')

  def update(self, now, dt, candidate, bsd, rear_lateral, *, elsewhere=(), motion=None):
    """candidate is the final, already stabilised real front drawing or None. Returns F/R drawings."""
    self.events = []
    if self.bsd and not bsd:
      self.bsd_off = now
    elif bsd:
      self.bsd_off = None
    self.bsd = bsd
    if self.front is None and self.rear is None and candidate is None and not bsd:
      return None, None
    eligible_bsd = bsd or (self.bsd_off is not None and now-self.bsd_off < self.HOLD)

    # The upstream lane-change tracker owns cross-slot identity movement. Never leave an old copy on this side.
    for name in ('front', 'rear'):
      obj = getattr(self, name)
      if obj is not None and obj['key'] is not None and obj['key'] in elsewhere:
        setattr(self, name, None)

    f, r = self.front, self.rear
    previous_rear, rear_start = r, r['x'] if r is not None else None

    f_owned = (f is not None and motion is not None
               and (f['key'] == motion['key'] or f['key'] in motion.get('aliases', ())))

    if f is not None:
      same = candidate is not None and candidate['key'] == f['key']
      if f['phase'] == 'to_rear' and same:
        f['phase'] = 'join_front'  # a short observation gap may be cancelled without resetting its position
      elif f['phase'] == 'to_rear' and not f_owned and not eligible_bsd:
        self.front = f = None  # Original coast cancellation/timeout wins; animation is not tracking evidence.
      elif f['phase'] != 'to_rear' and not same:
        if f['x'] <= 5. and (f_owned or eligible_bsd):
          f['phase'] = 'to_rear'
        elif f['x'] <= 5. or candidate is None or candidate['x'] < self.GAP or r is not None and candidate['key'] == r['key']:
          self.front = f = None
        else:
          # A replacement at distance changes the target, not the currently drawn position.
          f['key'], f['detect'], f['phase'] = candidate['key'], candidate['detect'], 'join_front'

    spawned = r is None and bsd
    if spawned:
      self._new_bsd(rear_lateral)
      r = self.rear

    # A new close F associated with a recent BSD travels through the origin; do not draw it twice.
    if (f is None and r is not None and r['phase'] != 'to_front' and candidate is not None
        and (candidate['key'] == r['key'] or candidate['x'] <= self.GAP and eligible_bsd and not self.after_pass)):
      r['phase'], r['key'] = 'to_front', candidate['key']
      r['target'] = dict(candidate)
      self.events.append('rear_to_front_start')

    if r is not None:
      if r['phase'] == 'to_front':
        if candidate is not None and candidate['key'] == r['key']:
          r['target'] = dict(candidate)
        # Zero is a slot boundary, not a stopping target. Brake only toward the actual front target.
        target = max(r['target']['x'], self.FAST_SPEED*self.FAST_TAU)
        self._advance(r, target, r['y'], dt, self.FAST_SPEED, self.FAST_TAU)
        r['x'] = min(0., r['x'])
        if r['x'] == 0.:
          f = self.front = r
          f['phase'] = 'join_front'
          self.rear = r = None
          self.after_pass = False
          self.events.append('rear_to_front')
          # The receiving F is rendered at the exact old R lateral and zero before it moves forward.
          return self.front, None
      else:
        physical = r.get('motion')
        delta = 0.
        if physical is not None:
          if physical is motion:
            # The original model owns vRel median, ego-speed scaling, origin wait, acceleration and caps.
            # Skip held progress instead of teleporting to the model's already advanced position.
            delta = max(0., physical['distance']-r['model_distance'])
            r['model_distance'] = physical['distance']
          elif eligible_bsd:
            r['motion'] = physical = None  # Apply the BSD pass speed immediately, including this transition frame.
          elif physical.get('end_reason') in ('distance_end', 'timeout', 'rear_replaced'):
            r['phase'] = 'exit'  # Tracking has retired; finish only the bounded drawing, without resurrecting it.
          else:
            self.rear = r = None  # Invalid/coast/identity/lane cancellation from the existing model.
        if r is None:
          self.events.append('rear_cancelled')
        else:
          held = r.get('hold_until', -math.inf) > now
          passing = physical is None and r['phase'] == 'virtual'
          target = (-8. if bsd or passing and eligible_bsd else r['x'] if held or eligible_bsd else -self.END)
          speed, tau = (6., .4) if bsd else (self.EXIT_SPEED, self.EXIT_TAU)
          if passing:
            speed = self.PASS_SPEED
          if physical is not None and not (bsd or held or eligible_bsd):
            if r['phase'] == 'exit':
              delta = physical['max_speed']*dt
            if now >= physical['force_at']:
              # Preserve the original bounded lifetime, including after the requested visual hold.
              required = (self.END+r['x'])*dt/max(dt, physical['deadline']-now+dt)
              delta = max(delta, min(required, physical['max_speed']*dt))
            r['x'] = max(-self.END, r['x']-delta)
            r['y'] = self._move(r['y'], rear_lateral, dt, self.LATERAL_SPEED, self.LATERAL_TAU)
          else:
            self._advance(r, target, rear_lateral, 0. if spawned else dt, speed, tau)
          if r['x'] == -self.END and not (bsd or held or eligible_bsd):
            self.rear = r = None
            self.events.append('rear_exit')

    f = self.front
    if f is not None and f['phase'] == 'to_rear':
      # Keep the last F lateral all the way through zero. R begins relaxing only after receiving it.
      target = max(0., motion['signed']) if f_owned and motion['phase'] == 'coast' else 0.
      speed = (8. if motion['phase'] == 'coast' else motion['speed']) if f_owned else self.PASS_SPEED
      f['x'] = max(min(f['x'], target), f['x']-min(8., speed)*dt)
      if target == 0. and f['x'] < 1e-9:
        f['x'] = 0.  # Floating-point residue must not add an extra zero-distance CAN frame.
      if f['x'] == 0.:
        # The newly passing F owns the slot immediately, even while an older R is holding or retiring.
        if r is not None:
          self.events.append('rear_display_replaced')
        self.rear = r = f
        physical = motion if f_owned else None
        r.update(phase='virtual', hold_until=now, motion=physical,
                 model_distance=physical['distance'] if physical is not None else 0.)
        self.front = f = None
        self.after_pass = True
        self.events.append('front_to_rear')

    f = self.front
    if r is not None and r['phase'] != 'to_front':
      if ((f is None or f['phase'] != 'to_rear') and candidate is not None and candidate['x'] < self.GAP):
        r['spacing'] = True
      if r.get('spacing') and r['x'] > -self.SPACE_DISTANCE and 'front_to_rear' not in self.events:
        # A close real F actively sends R to 8 m, even while F waits for the 6 m clearance.
        limit = max(-self.END, rear_start-8.*dt) if r is previous_rear else r['x']
        adjusted = max(limit, self._move(r['x'], -self.SPACE_DISTANCE, dt, 8., .25))
        if adjusted < r['x']:
          r['x'] = adjusted
          self.events.append('rear_spacing')
      if r['x'] <= -self.SPACE_DISTANCE:
        r['spacing'] = False

    if self.front is None and candidate is not None:
      # A BSD relay still owns R until it has reached the origin.
      clear = r is None or math.floor(candidate['x']*10)+math.floor(-r['x']*10) >= self.GAP*10
      if clear and (r is None or r['phase'] != 'to_front'):
        self.front = dict(candidate, phase='front')
        self.after_pass = False
        self.events.append('front_admitted')

    f = self.front
    if f is not None and f['phase'] != 'to_rear':
      target = candidate if candidate is not None and candidate['key'] == f['key'] else f.get('target')
      if target is not None:
        x = target['x']
        if f['phase'] == 'join_front':
          self._advance(f, x, target['y'], dt, self.FAST_SPEED, self.FAST_TAU)
          if f['x'] == x and f['y'] == target['y']:
            f['phase'] = 'front'
        else:
          # Retain the upstream anti-jitter/ego-lane-change trajectory; limit only discontinuous longitudinal jumps.
          f['x'] = min(max(x, f['x']-20.*dt), f['x']+20.*dt)
          f['y'], f['detect'] = target['y'], target['detect']

    if f is not None and f['phase'] != 'to_rear' and r is not None and r['phase'] != 'to_front':
      if math.floor(f['x']*10)+math.floor(-r['x']*10) < self.GAP*10:
        return None, r  # Keep the candidate/identity; avoid drawing an overlap during a sudden approach.
    return self.front, self.rear
