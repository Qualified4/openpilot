"""Display-only rear handoff, including BSD and the unobserved-motion limits."""
import copy
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N

import pytest

PATH = Path(__file__).resolve().parents[2]/'opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_rear.py'
spec = importlib.util.spec_from_file_location('rear_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RearHandover = module.RearHandover


class Filter:
  def __init__(self): self.value = 15.
  def apply(self, target):
    self.value += .05*(target-self.value)
    return self.value
  def reset(self, value): self.value = value


class Drive:
  def __init__(self, side=0):
    self.side = side
    self.policy = RearHandover()
    self.filters = Filter(), Filter()
    self.history, self.events = [], []
    self.cs = N(live_tracks=None, out=N(leftBlindspot=False, rightBlindspot=False, leftBlinker=False,
      rightBlinker=False, gearShifter='drive', canValid=True, vEgo=20.))  # vLead defaults to vEgo+vRel
    self.md = N(laneLineProbs=[1., 1., 1., 1.], meta=N(desire='none', laneChangeState='off'))

  def step(self, i, *, x=None, key=1, speed=-2., vehicle=True, bsd=False, show=True, track_state=2, measured=True,
           vlead=None, extra=(), drawn=None, y=None, tracks='new', proven=True, ff=None):
    front, rear = ('LF', 'LR') if self.side == 0 else ('RF', 'RR')
    setattr(self.cs.out, 'leftBlindspot' if self.side == 0 else 'rightBlindspot', bsd)
    y = (3. if self.side == 0 else -3.) if y is None else y
    point = N(trackId=key, dRel=x, yRel=y, vRel=speed, vLead=20.+speed if vlead is None else vlead,
              measured=measured, trackState=track_state, motionState=2 if vehicle else 0,
              radarSource='frontRadar') if x is not None else None
    if tracks == 'new':
      self.cs.live_tracks = N(points=([point] if point else [])+list(extra))
    elif tracks == 'none':  # card passes None when liveTracks is older than 0.15 s
      self.cs.live_tracks = None
    # tracks == 'same': card passes the same (frozen) reader again
    values = {}
    for f, r, filter_, blind in zip(('LF', 'RF'), ('LR', 'RR'), self.filters,
                                   (self.cs.out.leftBlindspot, self.cs.out.rightBlindspot), strict=True):
      values.update({f+'_DETECT': 0, f+'_DETECT_DISTANCE': 0., f+'_DETECT_LATERAL': 3.,
                     r+'_DETECT': 0, r+'_DETECT_DISTANCE': 0., r+'_DETECT_LATERAL': 2.8})
      if blind or filter_.value < 15:
        values.update({r+'_DETECT': 2, r+'_DETECT_DISTANCE': filter_.apply(8 if blind else 16)})  # as the 0x162 policy
    selected = [ff, None, None]
    if drawn is not None:  # the selector draws (key, display distance) regardless of radar reports
      selected[self.side+1] = (drawn[0], 0)
      values.update({front+'_DETECT': 2, front+'_DETECT_DISTANCE': drawn[1]})
    elif point is not None and show:
      selected[self.side+1] = (key, 0)
      values.update({front+'_DETECT': 2, front+'_DETECT_DISTANCE': .8*x})
    original = copy.deepcopy((self.cs, self.md, selected))
    self.policy.update(values, self.cs, self.md, i*5, selected, self.filters, 'same-context',
                       proven=(proven, True) if self.side == 0 else (True, proven))
    assert vars(original[0]) == vars(self.cs) and vars(original[1]) == vars(self.md)
    assert original[2] == selected
    self.history.append(dict(i=i, **values))
    self.events.extend(self.policy.events)
    return values, rear

  def seed(self, **kwargs):
    for i in range(15):
      self.step(i, x=2.6-.1*i, **kwargs)

  def handoff(self):
    self.seed()
    for i in range(15, 40):
      values, rear = self.step(i)
      if any(e['event'] == 'handoff' for e in self.policy.events):
        return i, values, rear
    raise AssertionError(self.events)


@pytest.mark.parametrize('side', [0, 1])
def test_zero_origin_handoff_is_monotonic_bounded_and_never_duplicates(side):
  d = Drive(side)
  i, values, rear = d.handoff()
  assert values[rear+'_DETECT_DISTANCE'] == 0 and d.filters[side].value == 0
  front = ('LF', 'RF')[side]
  assert values[front+'_DETECT'] == 0
  old_distance, old_speed = 0., d.policy.sides[side]['speed']
  for j in range(i+1, i+150):
    values, _ = d.step(j)
    state = d.policy.sides[side]
    if state is None:
      assert not values[rear+'_DETECT']
      break
    distance, speed = state['distance'], state['speed']
    assert old_distance <= distance < 16 and speed <= d.policy.EXIT_MAX_SPEED
    assert 0 <= speed-old_speed <= d.policy.EXIT_ACCEL*.05+1e-9
    assert distance-old_distance <= d.policy.EXIT_MAX_SPEED*.05+1e-9
    assert not (values[front+'_DETECT'] and values[rear+'_DETECT'])
    old_distance, old_speed = distance, speed
  else:
    raise AssertionError('did not retire')
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  assert d.events[-1]['event'] == 'distance_end'


def test_faster_real_vehicle_keeps_its_measured_forward_motion():
  d = Drive()
  for i in range(40):
    x = 2.+.15*i
    v, _ = d.step(i, x=x, speed=3.)
    assert v['LF_DETECT_DISTANCE'] == .8*x and v['LF_DETECT'] == 2
  assert not any(e['event'] == 'handoff' for e in d.events)


@pytest.mark.parametrize('failure', ['reflection', 'jump', 'too_far'])
def test_unqualified_or_unreachable_objects_never_handoff(failure):
  d = Drive()
  d.seed(vehicle=failure != 'reflection')
  if failure == 'jump':
    d.step(15, x=12.)
  elif failure == 'too_far':
    d.policy.sides[0]['x'] = 3.5  # last reported 3.5 m: extrapolated 3.3 m ahead when the slot ends
  for i in range(16, 70):
    v, _ = d.step(i)
    assert not v['LR_DETECT']
  assert not any(e['event'] == 'handoff' for e in d.events)


def test_bsd_preempts_memory_and_keeps_the_existing_filter_policy():
  d = Drive()
  i, values, rear = d.handoff()
  d.step(i+1)
  drawn = d.filters[0].value
  values, _ = d.step(i+2, bsd=True)
  assert d.policy.sides[0] is None
  assert values[rear+'_DETECT_DISTANCE'] == pytest.approx(drawn+.05*(8-drawn))
  assert values[rear+'_DETECT'] == 2
  drawn = d.filters[0].value
  values, _ = d.step(i+3, bsd=False)
  assert values[rear+'_DETECT_DISTANCE'] == pytest.approx(drawn+.05*(16-drawn))
  assert values[rear+'_DETECT'] == 2 and d.policy.sides[0] is None
  for j in range(i+4, i+10):
    d.step(j, x=.3, key=1)
    assert not d.history[-1]['LF_DETECT']
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


def test_bsd_already_active_does_not_start_another_track():
  d = Drive()
  d.seed(bsd=True)
  for i in range(15, 50):
    d.step(i, bsd=True)
    assert d.policy.sides[0] is None
  assert not d.events


@pytest.mark.parametrize('reason', ['reverse', 'lane_change', 'clock'])
def test_invalid_tracking_cancels_rear_without_leaving_a_gray_ghost(reason):
  d = Drive()
  i, _, rear = d.handoff()
  if reason == 'reverse':
    d.cs.out.gearShifter = 'reverse'
  if reason == 'lane_change':
    d.cs.out.leftBlinker = True
  if reason == 'confidence':
    d.md.laneLineProbs = [0.]*4
  if reason == 'CAN':
    d.cs.out.canValid = False
  step = i+20 if reason == 'clock' else i+1
  v, _ = d.step(step)
  assert not v[rear+'_DETECT']
  assert d.policy.sides[0] is None or d.policy.sides[0]['phase'] == 'front'
  assert d.filters[0].value == 15


def test_explicit_time_limit_and_velocity_cap():
  d = Drive()
  i, _, rear = d.handoff()
  d.policy.MAX_TIME, d.policy.EXIT_GRACE = .2, 1.
  drawn = []
  for j in range(i+1, i+40):
    v, _ = d.step(j)
    drawn.append(v[rear+'_DETECT_DISTANCE'] if v[rear+'_DETECT'] else None)
  # At MAX_TIME the car is not removed where it is drawn: it slides out (forced exit) and ends at 16 or the grace.
  assert any(e['event'] == 'forced_exit' for e in d.events)
  shown = [x for x in drawn if x is not None]
  assert shown == sorted(shown) and drawn[-1] is None
  assert any(e['event'] in ('distance_end', 'timeout') for e in d.events)
  d = Drive()
  d.policy.EXIT_MAX_SPEED = 2.
  i, _, _ = d.handoff()
  for j in range(i+1, i+80):
    d.step(j)
    if d.policy.sides[0]:
      assert d.policy.sides[0]['speed'] <= 2.


def test_reobserved_consumed_id_cannot_jump_back_into_front():
  d = Drive()
  i, _, rear = d.handoff()
  for j in range(i+1, i+8):
    v, _ = d.step(j, x=.5, key=1)
    assert not v['LF_DETECT'] and v[rear+'_DETECT']


def test_bsd_does_not_consume_a_still_measured_front_vehicle():
  d = Drive()
  d.seed()
  d.step(15, x=1.1, bsd=True)
  v, _ = d.step(16, x=1., bsd=False)
  assert v['LF_DETECT'] == 2 and not d.policy.consumed[0]


def test_new_measured_forward_motion_wins_without_reversing_virtual_memory():
  d = Drive()
  i, _, rear = d.handoff()
  v, _ = d.step(i+1, x=2., speed=3.)
  assert v['LF_DETECT'] == 2 and not v[rear+'_DETECT']
  assert d.policy.sides[0] is None
  for j in range(i+2, i+25):
    d.step(j)
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


def test_rereading_one_frozen_radar_message_cannot_confirm_a_vehicle():
  d = Drive()
  d.step(0, x=1.1)
  d.step(1, x=1.)
  for frame in range(10, 70, 5):
    values = dict(d.history[-1])
    d.policy.update(values, d.cs, d.md, frame, [None, (1, 0), None], d.filters, 'same-context')
    assert d.policy.sides[0] is None or d.policy.sides[0]['hits'] == 1
    assert not any(e['event'] == 'handoff' for e in d.policy.events)


@pytest.mark.parametrize('side', [0, 1])
def test_new_front_replaces_old_rear_the_moment_it_leaves_the_front_slot(side):
  d = Drive(side)
  i, _, rear = d.handoff()
  front = ('LF', 'RF')[side]
  old = d.policy.rears[side]
  last = 0.
  for n in range(10):
    v, _ = d.step(i+1+n, x=1.4-.1*n, key=2)
    assert d.policy.rears[side] is old  # the old car keeps LR/RR while the new one is still in LF/RF
    assert v[front+'_DETECT'] and v[rear+'_DETECT']
    assert v[rear+'_DETECT_DISTANCE'] >= last
    last = v[rear+'_DETECT_DISTANCE']
  v, _ = d.step(i+11)  # the new car's report and selection end at x 0.4
  assert d.policy.rears[side]['key'] == (2, 0)
  assert v[rear+'_DETECT_DISTANCE'] == 0 and v[rear+'_DETECT'] and not v[front+'_DETECT']
  assert d.filters[side].value == 0
  replaced, coast = [e for e in d.policy.events if e['event'] in ('rear_replaced', 'coast')]
  assert replaced['source'] == 1 and replaced['distance'] >= last and coast['source'] == 2
  assert (1, 0) in d.policy.consumed[side] and (2, 0) in d.policy.consumed[side]


def test_unqualified_new_front_cannot_replace_rear_and_bsd_discards_both_memories():
  d = Drive()
  i, _, _ = d.handoff()
  old = d.policy.rears[0]
  for n in range(10):
    d.step(i+1+n, x=1.2-.1*n, key=2, vehicle=False)
    assert d.policy.rears[0] is old
  drawn = d.filters[0].value
  v, _ = d.step(i+11, bsd=True, x=.1, key=2)
  assert d.policy.fronts == d.policy.rears == [None, None]
  assert v['LR_DETECT_DISTANCE'] == pytest.approx(drawn+.05*(8-drawn))
  v, _ = d.step(i+12)
  assert v['LR_DETECT'] == 2 and d.policy.rears[0] is None
  assert not any(e['event'] == 'rear_replaced' for e in d.events)


def test_slowest_qualified_vehicle_reaches_exit_before_the_six_second_limit():
  d = Drive()
  for i in range(60):
    d.step(i, x=1.-.015*i, speed=-.3)
  for i in range(60, 220):
    d.step(i)
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  assert d.events[-1]['event'] == 'distance_end'
  assert not any(e['event'] == 'timeout' for e in d.events)


class SliceRejectingList:
  """Fallback for a VM without pycapnp: indexes and iterates like a capnp list reader, rejects slices."""
  def __init__(self, values): self.values = list(values)
  def __len__(self): return len(self.values)
  def __iter__(self): return iter(self.values)
  def __getitem__(self, index):
    if not isinstance(index, int):
      raise TypeError('an integer is required')
    return self.values[index]


def capnp_model():
  try:
    from openpilot.cereal import log
  except ImportError:
    return N(laneLineProbs=SliceRejectingList([.9]*4), meta=N(desire='none', laneChangeState='off')), False
  builder = log.ModelDataV2.new_message()
  builder.laneLineProbs = [.9]*4
  return builder.as_reader(), True  # the car hands card a reader, which rejects laneLineProbs[1:3]


@pytest.mark.parametrize('side', [0, 1])
def test_capnp_lane_line_probs_reader_does_not_raise_and_tracks_the_front(side):
  md, _ = capnp_model()
  with pytest.raises(TypeError):
    md.laneLineProbs[1:3]  # the bug: slicing a list reader
  d = Drive(side)
  for i in range(8):
    point = N(trackId=1, dRel=2.6-.1*i, yRel=3. if side == 0 else -3., vRel=-2., vLead=18., measured=True,
              trackState=2, motionState=2, radarSource='frontRadar')
    d.cs.live_tracks = N(points=[point])
    selected = [None, None, None]
    selected[side+1] = (1, 0)
    values = {}
    for k, lateral in (('LF', 3.), ('RF', 3.), ('LR', 2.8), ('RR', 2.8)):
      values.update({k+'_DETECT': 0, k+'_DETECT_DISTANCE': 0., k+'_DETECT_LATERAL': lateral})
    values.update({('LF', 'RF')[side]+'_DETECT': 2, ('LF', 'RF')[side]+'_DETECT_DISTANCE': .8*point.dRel})
    d.policy.update(values, d.cs, md, i*5, selected, d.filters, 'same-context')
    assert not any(e['event'] in ('confidence_lost', 'clock_or_context') for e in d.policy.events if i)
  assert d.policy.fronts[side] is not None and d.policy.fronts[side]['source'] == 1


def test_qualification_ignores_track_state_and_measured_flags():
  """The radar marks an alongside car trackState 3 / unmeasured; motion evidence alone qualifies it."""
  d = Drive()
  for i in range(15):
    d.step(i, x=2.6-.1*i, track_state=3 if i % 2 else 2, measured=bool(i % 3))
  assert d.policy.sides[0]['qualified']
  for i in range(15, 40):
    d.step(i)
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


def test_qualified_car_turning_track_state_3_while_still_drawn_is_extended_to_rear():
  """3fe/3 18.5: trackState 2->3 and motionState 2->0 near 2.5 m while the map keeps drawing LF."""
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  assert d.policy.fronts[0]['qualified']
  shown = []
  for i in range(10, 60):
    x = 4.-.15*i  # trackState 3 reports follow the car to 1.6 m, then stop; the map holds LF near 1 m
    v, _ = d.step(i, x=x if x >= 1.6 else None, speed=-3., track_state=3, vehicle=False, drawn=(1, max(.8*x, .8)))
    shown.append((i, v['LF_DETECT'], v['LF_DETECT_DISTANCE'], v['LR_DETECT'], v['LR_DETECT_DISTANCE'],
                  any(e['event'] == 'handoff' for e in d.policy.events)))
  assert any(e['event'] == 'coast' and e['reason'] == 'measurement_lost' for e in d.events)
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  start = next(k for k, s in enumerate(shown) if s[3])
  assert shown[start][0] == 17 and shown[start-1][1] and not shown[start][1]  # no gap: LF -> LR 0 in one send
  front = [s[2] for s in shown[:start]]
  assert front == sorted(front, reverse=True)
  origin = next(k for k, s in enumerate(shown) if s[5])
  assert shown[origin][0] <= 28  # the extrapolated x passes the origin (~i 26.7), not after the map's 1 m hold
  assert all(s[3] and s[4] == 0 and not s[1] for s in shown[start:origin+1])  # LR waits at 0 until the origin
  rear = [s[4] for s in shown[origin:] if s[3]]
  assert rear == sorted(rear) and rear[-1] > 0


@pytest.mark.parametrize('side', [0, 1])
def test_replaced_qualified_car_goes_to_rear_while_a_different_new_front_is_drawn_immediately(side):
  d = Drive(side)
  front, rear = ('LF', 'LR') if side == 0 else ('RF', 'RR')
  for i in range(10):
    v, _ = d.step(i, x=4.-.15*i, speed=-3.)
    assert v[front+'_DETECT'] and not v[rear+'_DETECT']
  v, _ = d.step(10, x=13., key=2, speed=-2.)  # the slot jumps to the next car 13 m ahead
  assert v[front+'_DETECT'] == 2 and v[front+'_DETECT_DISTANCE'] == pytest.approx(.8*13.)
  assert v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0  # the old car is in LR/RR in the same send
  coast = next(e for e in d.events if e['event'] == 'coast')
  assert coast['source'] == 1 and coast['reason'] == 'object_replaced'
  for i in range(11, 30):
    v, _ = d.step(i, x=13.-.1*(i-10), key=2, speed=-2.)
    assert v[front+'_DETECT'] == 2 and v[rear+'_DETECT']
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  assert d.policy.fronts[side]['source'] == 2  # the new car is observed as its own front candidate


def test_replacement_by_the_same_car_under_a_new_radar_id_does_not_duplicate_it():
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  for i in range(10, 30):
    x = 4.-.15*i
    v, _ = d.step(i, x=max(x, .5), key=7, speed=-3., track_state=3, vehicle=False)
    assert not v['LF_DETECT'] and v['LR_DETECT']  # LR from the replacing send on, the new id never in LF
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  assert (7, 0) in d.policy.consumed[0]
  v, _ = d.step(30, x=.4, key=7, speed=-3., track_state=3, vehicle=False)
  assert not v['LF_DETECT'] and v['LR_DETECT']


@pytest.mark.parametrize('x,rejected', [(3.05, True), (2.9, False)])
def test_selector_rejection_needs_a_report_three_metres_ahead(x, rejected):
  d = Drive()
  for i in range(8):
    d.step(i, x=4.6-.2*i, speed=-4.)
  d.step(8, x=x, speed=-4., show=False)
  for i in range(9, 40):
    d.step(i)
  assert any(e['event'] == 'still_ahead' for e in d.events) == rejected
  assert (sum(e['event'] == 'handoff' for e in d.events) == 1) != rejected


def test_slow_pass_waits_at_rear_zero_until_the_origin_without_a_one_second_bound():
  """DaBin 10-19: the only pass exclusion is 3 m ahead. 2.9 m at -1 m/s takes 2.9 s to reach the origin."""
  d = Drive()
  for i in range(15):
    d.step(i, x=3.6-.05*i, speed=-1.)
  shown = []
  for i in range(15, 100):
    v, _ = d.step(i)
    shown.append((i, v['LR_DETECT'], v['LR_DETECT_DISTANCE'], any(e['event'] == 'handoff' for e in d.policy.events)))
  assert [e['reason'] for e in d.events if e['event'] == 'coast'] == ['selection_ended']
  origin = next(s for s in shown if s[3])
  assert 70 <= origin[0] <= 74  # ~2.9 s after the last report at i 14, well past the old 1 s bound
  assert all(s[1] and s[2] == 0 for s in shown if s[0] <= origin[0])
  assert not any(e['event'] == 'timeout' for e in d.events)


def test_alongside_car_still_times_out_six_seconds_after_rear_zero():
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  d.step(10)  # selection ends at ~2.5 m: LR 0
  start = d.policy.rears[0]['shown']
  for i in range(11, 200):
    v, _ = d.step(i, x=1., speed=0., show=False)  # it stays alongside, reported at 1 m
    if not v['LR_DETECT']:
      break
  assert [e['event'] for e in d.policy.events] == ['timeout']
  assert i*.05-start == pytest.approx(d.policy.MAX_TIME, abs=.051)
  assert not any(e['event'] == 'handoff' for e in d.events)


def test_ego_stopping_beside_a_stopped_car_keeps_rear_zero_until_the_timeout():
  """DaBin 10-19 (3fd/4 RR 56): the unreported car moves at the saved vRel scaled by vEgo now over vEgo when saved.
  Creeping at 0.5 m/s past a stopped car (vLead 0), the reports end 2.5 m ahead and the ego stops: x stays put."""
  d = Drive()
  d.cs.out.vEgo = .5
  for i in range(15):
    d.step(i, x=2.9-.025*i, speed=-.5, vlead=0.)
  assert d.policy.fronts[0]['qualified']
  shown = []
  for i in range(15, 200):
    d.cs.out.vEgo = max(0., .5-.1*(i-14))  # 2 m/s^2 to a stop in 0.25 s
    v, _ = d.step(i)
    shown.append((v['LR_DETECT'], v['LR_DETECT_DISTANCE']))
    if not v['LR_DETECT']:
      break
    assert d.policy.rears[0]['phase'] == 'coast' and d.policy.rears[0]['signed'] > 1.5
  assert [e['event'] for e in d.events] == ['coast', 'timeout']
  assert all(s == (2, 0.) for s in shown[:-1]) and len(shown)-1 == pytest.approx(d.policy.MAX_TIME/.05, abs=1)
  assert not any(e['event'] == 'handoff' for e in d.events)


def test_ego_braking_while_the_car_beside_keeps_going_still_hands_over():
  """Ego 7 m/s braking at 1 m/s^2, the car beside holding 5 m/s: x 1.2 m reaches the origin about 0.7 s later
  (vRel -2 alone would say 0.6 s), then LR/RR moves back."""
  d = Drive()
  d.cs.out.vEgo = 7.
  for i in range(15):
    d.step(i, x=2.6-.1*i, speed=-2., vlead=5.)
  d.cs.out.vEgo = 6.95
  d.step(15)  # selection ends: LR 0
  assert [e['event'] for e in d.policy.events] == ['coast']
  handoff, old = None, 0.
  for i in range(16, 200):
    d.cs.out.vEgo = max(3., 7.-.05*(i-14))
    v, _ = d.step(i)
    if any(e['event'] == 'handoff' for e in d.policy.events):
      handoff = i
    if not v['LR_DETECT']:
      break
    assert v['LR_DETECT_DISTANCE'] >= old
    old = v['LR_DETECT_DISTANCE']
  assert handoff is not None and .65 <= (handoff-14)*.05 <= .85
  assert old > 8.
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


def test_ego_braking_from_5_to_3_through_a_real_pass_still_hands_over():
  """DaBin 10-19 (3fe/3 LR 52): ego 5.1 -> 2.9 m/s at 1.5 m/s^2, the car beside at 3.35 m/s, reports end 1.2 m ahead.
  The saved vRel -1.75 scaled by vEgo/5.1 reaches the origin after 3.4 m of ego travel (~0.8 s). A fixed vLead 3.35
  (round 4) never gets there: the gap stops closing once the ego car is slower than 3.35."""
  d = Drive()
  d.cs.out.vEgo = 5.1
  for i in range(15):
    d.step(i, x=2.4-.0875*i, speed=-1.75, vlead=3.35)
  assert d.policy.fronts[0]['qualified']
  handoff, old, shown = None, 0., []
  for i in range(15, 200):
    d.cs.out.vEgo = max(2.9, 5.1-.075*(i-14))
    v, _ = d.step(i)
    shown.append((v['LR_DETECT'], v['LR_DETECT_DISTANCE']))
    if any(e['event'] == 'handoff' for e in d.policy.events):
      handoff = i
    if not v['LR_DETECT']:
      break
    assert v['LR_DETECT_DISTANCE'] >= old
    old = v['LR_DETECT_DISTANCE']
  assert handoff is not None and .7 <= (handoff-14)*.05 <= .9
  assert all(s == (2, 0.) for s in shown[:handoff-15])  # LR 0 from the slot's end until the origin
  assert sum(e['event'] == 'handoff' for e in d.events) == 1
  assert d.events[-1]['event'] == 'distance_end'


def test_extrapolated_x_three_metres_ahead_during_the_coast_withdraws_it():
  """DaBin 10-19: still_ahead applies to the extrapolated x too. The last report has the car 2.95 m ahead and falling
  back at 3 m/s; with no further report it is 3.1 m ahead one send later."""
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(10, x=2.5, speed=-3., show=False)
  assert v['LR_DETECT'] and any(e['event'] == 'coast' for e in d.policy.events)
  for i, (x, speed) in enumerate(((2.45, -1.), (2.5, 1.), (2.65, 3.), (2.8, 3.), (2.95, 3.)), 11):
    v, _ = d.step(i, x=x, speed=speed, show=False)
    assert v['LR_DETECT'] and v['LR_DETECT_DISTANCE'] == 0
  v, _ = d.step(16)  # no report: extrapolated 2.95 + 3 * 0.05 = 3.1 m
  assert not v['LR_DETECT'] and [e['event'] for e in d.policy.events] == ['still_ahead']
  for i in range(17, 40):
    v, _ = d.step(i)
    assert not v['LR_DETECT']
  assert not any(e['event'] == 'handoff' for e in d.events)


@pytest.mark.parametrize('vlead', [-.5, -3.])
def test_stationary_object_is_never_handed_off(vlead):
  d = Drive()
  for i in range(15):
    d.step(i, x=2.6-.1*i, vlead=vlead)
  assert not d.policy.fronts[0]['qualified']
  for i in range(15, 50):
    v, _ = d.step(i)
    assert not v['LR_DETECT']
  assert not any(e['event'] == 'handoff' for e in d.events)


def test_measured_vrel_noise_does_not_reset_qualification():
  d = Drive()
  for i in range(15):
    d.step(i, x=2.6-.1*i, speed=-2.+(1.5 if i % 2 else 0.))
  assert not any(e['event'] == 'identity_innovation' for e in d.events)
  assert d.policy.fronts[0]['qualified']


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('reason', ['object_replaced', 'measurement_lost', 'selection_ended'])
def test_no_gap_the_car_leaves_front_and_appears_at_rear_zero_in_the_same_send(side, reason):
  d = Drive(side)
  front, rear = ('LF', 'LR') if side == 0 else ('RF', 'RR')
  for i in range(10):
    v, _ = d.step(i, x=4.-.15*i, speed=-3.)
  assert v[front+'_DETECT'] and not v[rear+'_DETECT'] and d.policy.fronts[side]['qualified']
  if reason == 'object_replaced':
    v, _ = d.step(10, x=14., key=2, speed=-2.)
  elif reason == 'measurement_lost':
    v, _ = d.step(10, drawn=(1, 2.))  # still drawn, no report
  else:
    v, _ = d.step(10)
  coast = [e for e in d.policy.events if e['event'] == 'coast']
  assert len(coast) == 1 and coast[0]['reason'] == reason and coast[0]['source'] == 1
  assert v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0 and d.filters[side].value == 0
  assert v[front+'_DETECT'] == (2 if reason == 'object_replaced' else 0)  # only a different car stays in LF/RF


@pytest.mark.parametrize('reason', ['object_replaced', 'selection_ended'])
def test_slot_lost_three_metres_ahead_is_not_handed_over(reason):
  d = Drive()
  for i in range(8):
    d.step(i, x=5.6-.3*i, speed=-6.)  # 3.5 m at i 7: it could reach the origin within the 1 s window
  old = N(trackId=1, dRel=3.2, yRel=3., vRel=-6., vLead=14., measured=True, trackState=2, motionState=2,
          radarSource='frontRadar')  # the old car is still reported 3.2 m ahead
  if reason == 'object_replaced':
    v, _ = d.step(8, x=14., key=2, extra=[old])
  else:
    v, _ = d.step(8, x=3.2, speed=-6., show=False)
  assert not v['LR_DETECT']
  assert [e['event'] for e in d.events if e['event'] in ('coast', 'still_ahead')] == ['still_ahead']
  for i in range(9, 40):
    v, _ = d.step(i)
    assert not v['LR_DETECT']


def test_report_three_metres_ahead_during_the_coast_withdraws_it():
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(10, x=2.5, speed=-3., show=False)
  assert v['LR_DETECT'] and any(e['event'] == 'coast' for e in d.policy.events)
  for i, (x, speed) in enumerate(((2.45, -1.), (2.5, 1.), (2.65, 3.), (2.8, 3.), (2.95, 3.)), 11):
    v, _ = d.step(i, x=x, speed=speed, show=False)  # the car falls back ahead again: LR stays at 0 meanwhile
    assert v['LR_DETECT'] and v['LR_DETECT_DISTANCE'] == 0
  v, _ = d.step(16, x=3.1, speed=3., show=False)
  assert not v['LR_DETECT'] and any(e['event'] == 'still_ahead' for e in d.policy.events)


@pytest.mark.parametrize('side', [0, 1])
def test_bsd_takes_over_a_coast_at_zero(side):
  d = Drive(side)
  rear = ('LR', 'RR')[side]
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(10)
  assert v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0 and d.policy.rears[side]['phase'] == 'coast'
  v, _ = d.step(11, bsd=True)
  assert d.policy.rears[side] is None and v[rear+'_DETECT'] == 2
  assert v[rear+'_DETECT_DISTANCE'] == pytest.approx(.05*8)  # the BSD filter continues from the drawn 0
  assert any(e['event'] == 'bsd' for e in d.policy.events)
  for i in range(12, 40):
    d.step(i, bsd=True)
  assert not any(e['event'] == 'handoff' for e in d.events)


@pytest.mark.parametrize('out_kind', ['reader', 'builder'])
def test_capnp_event_readers_take_the_whole_coast_and_handoff_path(out_kind):
  """Every operation the rear module applies to md / CS.out / liveTracks, on real log.Event readers as card passes
  them (SubMaster keeps one reader per message): enum str() and ==, UInt8 ==, list(), len(), the identity check that
  ignores a re-read frozen message, and no reader kept in the policy's memories. Live CS.out is a structs.CarState
  capnp builder; the rlog replays pass a carState reader: both are covered."""
  try:
    from openpilot.cereal import log
  except ImportError:
    pytest.skip('pycapnp/cereal unavailable')

  def event(kind, fill):
    builder = log.Event.new_message()
    fill(builder.init(kind))
    with log.Event.from_bytes(builder.to_bytes()) as msg:  # messaging.log_from_bytes, which SubMaster uses
      return getattr(msg, kind)

  def car_state(b, reverse=False):
    b.gearShifter = 'reverse' if reverse else 'drive'
    b.canValid = True
    b.vEgo = 20.  # the track's vLead 18 = vEgo + vRel -2

  def model(b):
    b.laneLineProbs = [.9]*4
    b.meta.desire = 'none'
    b.meta.laneChangeState = 'off'

  def tracks(x):
    def fill(b):
      points = b.init('points', 1)
      p = points[0]
      p.trackId, p.dRel, p.yRel, p.vRel, p.vLead = 1, x, 3., -2., 18.
      p.measured, p.trackState, p.motionState, p.radarSource = True, 2, 2, 'frontRadar'
    return fill

  md, out = event('modelV2', model), event('carState', car_state)
  points = event('liveTracks', tracks(2.))
  # The operations themselves, on readers.
  with pytest.raises(TypeError):
    md.laneLineProbs[1:3]
  assert list(md.laneLineProbs)[1:3] == pytest.approx([.9, .9]) and len(list(md.laneLineProbs)) == 4
  assert str(md.meta.desire) in ('none',) and str(md.meta.laneChangeState) == 'off'
  assert str(out.gearShifter) == 'drive' and str(event('carState', lambda b: car_state(b, True)).gearShifter) == 'reverse'
  assert bool(out.leftBlindspot) is False and getattr(out, 'canValid', True) is True
  point = points.points[0]
  assert str(point.radarSource) == 'frontRadar' and point.motionState == 2 and int(point.trackId) == 1
  assert getattr(point, 'motionState', 0) == 2 and type(float(point.dRel)) is float
  assert len(points.points) == 1 and [int(p.trackId) for p in points.points] == [1]

  if out_kind == 'builder':
    from opendbc.car import structs
    out = structs.CarState()
    out.gearShifter, out.canValid, out.vEgo = structs.CarState.GearShifter.drive, True, 20.
    assert str(out.gearShifter) == 'drive' and bool(out.leftBlindspot) is False
  d = Drive()
  d.md = md
  d.cs = N(out=out, live_tracks=None)
  frozen_hits = []
  for i in range(30):
    x = 2.6-.1*i
    d.cs.live_tracks = event('liveTracks', tracks(x)) if i < 15 else None
    values = {}
    for k, lateral in (('LF', 3.), ('RF', 3.), ('LR', 2.8), ('RR', 2.8)):
      values.update({k+'_DETECT': 0, k+'_DETECT_DISTANCE': 0., k+'_DETECT_LATERAL': lateral})
    selected = [None, None, None]
    if i < 15:
      selected[1] = (1, 0)
      values.update(LF_DETECT=2, LF_DETECT_DISTANCE=.8*x)
    d.policy.update(values, d.cs, d.md, i*5, selected, d.filters, 'same-context')
    d.events.extend(d.policy.events)
    if i == 5:  # the same message object again (SubMaster between two radar frames) is not another observation
      hits = d.policy.fronts[0]['hits']
      d.policy.update(dict(values), d.cs, d.md, i*5+1, selected, d.filters, 'same-context')
      frozen_hits.append(d.policy.fronts[0]['hits'] == hits)
    for memory in (*d.policy.fronts, *d.policy.rears):
      for value in (memory or {}).values():
        assert isinstance(value, (bool, int, float, str, tuple, list, type(None))), value  # nothing refers to a reader
  assert frozen_hits == [True]
  assert [e['event'] for e in d.events if e['event'] in ('coast', 'handoff')] == ['coast', 'handoff']
  assert not any(e['event'] in ('confidence_lost', 'clock_or_context', 'reverse', 'lane_change') for e in d.events[1:])


def _gap_drive(gap, tracks):
  """vRel -3 car qualified from 5 m; liveTracks missing for the sends in gap while the map keeps drawing LF."""
  d = Drive()
  shown = []
  for i in range(60):
    x = 5.-.15*i
    if i in gap:
      v, _ = d.step(i, drawn=(1, .8*x), tracks=tracks)
    elif x >= 1.:
      v, _ = d.step(i, x=x, speed=-3.)
    else:
      v, _ = d.step(i)  # selection ends at ~0.9 m
    shown.append((i, v['LF_DETECT'], v['LR_DETECT'], v['LR_DETECT_DISTANCE'], [e['event'] for e in d.policy.events]))
  return d, shown


@pytest.mark.parametrize('tracks', ['none', 'same'])
@pytest.mark.parametrize('gap', [(18,), (18, 19), (18, 19, 20)])
def test_short_live_tracks_gap_keeps_the_front_and_does_not_hand_over_early(gap, tracks):
  """Review A: liveTracks None/stale for one to three sends (<= 0.3 s) is no news about the car, not a lost report."""
  d, shown = _gap_drive(set(gap), tracks)
  for i, lf, lr, _, events in shown[16:gap[-1]+2]:
    assert lf == 2 and not lr and not events, (i, lf, lr, events)
  coast = [e for e in d.events if e['event'] == 'coast']
  assert len(coast) == 1 and coast[0]['reason'] == 'selection_ended'
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


def test_live_tracks_stale_beyond_the_hold_coasts_as_measurement_lost():
  d, shown = _gap_drive(set(range(18, 40)), 'none')
  for i, lf, lr, _, events in shown[18:24]:  # last report at i 17: held for 0.3 s
    assert lf == 2 and not lr and not events, (i, events)
  i, lf, lr, distance, events = shown[24]
  assert not lf and lr and distance == 0 and events == ['coast']
  assert [e['reason'] for e in d.events if e['event'] == 'coast'] == ['measurement_lost']
  assert sum(e['event'] == 'handoff' for e in d.events) == 1


@pytest.mark.parametrize('side', [0, 1])
def test_single_unproven_send_is_debounced_and_two_coast_as_not_proven(side):
  front, rear = ('LF', 'LR') if side == 0 else ('RF', 'RR')
  d = Drive(side)
  for i in range(14):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(14, x=4.-.15*14, speed=-3., proven=False)  # one send: structure/crossing flicker
  assert v[front+'_DETECT'] and not v[rear+'_DETECT'] and not d.policy.events
  v, _ = d.step(15, x=4.-.15*15, speed=-3.)
  assert v[front+'_DETECT'] and not v[rear+'_DETECT'] and d.policy.fronts[side]['qualified']
  d.step(16, x=4.-.15*16, speed=-3., proven=False)
  v, _ = d.step(17, x=4.-.15*17, speed=-3., proven=False)  # two sends in a row: the map really lost it
  coast = [e for e in d.policy.events if e['event'] == 'coast']
  assert len(coast) == 1 and coast[0]['reason'] == 'not_proven'
  assert not v[front+'_DETECT'] and v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0


def test_coasting_car_leaving_for_the_outer_lane_is_dropped():
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  d.step(10)
  assert d.policy.rears[0]['phase'] == 'coast'
  for n, i in enumerate(range(11, 20), 1):
    v, _ = d.step(i, x=2.5-.15*n, speed=-3., y=3.+.3*n, show=False)  # drifts 0.3 m per report to the left
    if 3.+.3*n-3. > 1.75:
      break
    assert v['LR_DETECT'] and v['LR_DETECT_DISTANCE'] == 0
  assert not v['LR_DETECT'] and [e['event'] for e in d.policy.events] == ['lane_departed']
  for j in range(i+1, i+40):
    v, _ = d.step(j)
    assert not v['LR_DETECT']
  assert not any(e['event'] == 'handoff' for e in d.events)


def test_car_that_stopped_after_qualifying_is_not_handed_over():
  d = Drive()
  d.cs.out.vEgo = 4.  # in a queue: vLead 1 = vEgo 4 + vRel -3
  for i in range(12):
    d.step(i, x=4.-.15*i, speed=-3., vlead=1.)
  assert d.policy.fronts[0]['qualified']
  d.cs.out.vEgo = 2.
  for i in range(12, 15):
    d.step(i, x=4.-.15*i, speed=-3., vlead=-1.)  # it is now stopped/reversing (vLead < 0)
  for i in range(15, 50):
    v, _ = d.step(i)
    assert not v['LR_DETECT']
  assert [e['event'] for e in d.events] == ['stopped']


@pytest.mark.parametrize('other', ['overtaking_from_behind', 'next_lane', 'nan_distance'])
def test_a_different_car_near_the_coasting_one_is_not_taken_for_its_new_radar_id(other):
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  d.step(10)  # coasting at ~2.5 m, extrapolated ~2.35 m at i 11
  coast = d.policy.rears[0]
  if other == 'overtaking_from_behind':
    v, _ = d.step(11, x=2., key=9, speed=3., vlead=23.)
  elif other == 'next_lane':
    v, _ = d.step(11, x=2., key=9, speed=-3., y=4.5)
  else:
    v, _ = d.step(11, drawn=(9, math.nan))
  assert v['LF_DETECT'] == 2 and (9, 0) not in d.policy.consumed[0] and not coast['aliases']
  assert d.policy.rears[0] is coast and v['LR_DETECT'] and v['LR_DETECT_DISTANCE'] == 0


def test_alias_key_moving_to_the_front_lane_slot_ends_the_rear_drawing():
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(10, x=2.5, key=7, speed=-3.)  # the same car under a new radar id
  assert d.policy.rears[0]['aliases'] == [(7, 0)] and not v['LF_DETECT'] and v['LR_DETECT']
  v, _ = d.step(11, x=2.35, key=7, speed=-3., show=False, ff=(7, 0))  # the map now draws it in FF
  assert d.policy.rears[0] is None and not v['LR_DETECT']
  assert [e['event'] for e in d.policy.events] == ['changed_lane_slot']


def _coasting(side=0):
  d = Drive(side)
  rear = ('LR', 'RR')[side]
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  v, _ = d.step(10)
  assert v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0 and d.policy.rears[side]['phase'] == 'coast'
  return d, rear


@pytest.mark.parametrize('side', [0, 1])
def test_the_other_lines_low_prob_leaves_this_sides_front_and_rear_alone(side):
  """10-20 segment 15: the right line's 0.097 used to drop the left car 0.1 s before its handoff."""
  other = 2 if side == 0 else 1
  d = Drive(side)
  for i in range(8):
    d.step(i, x=4.-.15*i, speed=-3.)
  saved = d.policy.fronts[side]
  hits = saved['hits']
  d.md.laneLineProbs[other] = .09
  d.step(8, x=4.-.15*8, speed=-3.)
  assert d.policy.fronts[side] is saved and saved['hits'] == hits+1 and not d.policy.events
  d, rear = _coasting(side)
  d.md.laneLineProbs[other] = .09
  v, _ = d.step(11)
  assert v[rear+'_DETECT'] and v[rear+'_DETECT_DISTANCE'] == 0 and d.policy.rears[side]['phase'] == 'coast'
  assert not any(e['event'] == 'confidence_lost' for e in d.policy.events)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('cause', ['prob', 'CAN'])
def test_own_low_prob_drops_only_the_front_and_the_coast_still_hands_off(side, cause):
  d, rear = _coasting(side)
  if cause == 'prob':
    d.md.laneLineProbs[1+side] = .09
  else:
    d.cs.out.canValid = False
  handoff = None
  for i in range(11, 40):
    v, _ = d.step(i)
    if handoff is None:
      assert v[rear+'_DETECT'] and d.policy.rears[side] is not None
    if any(e['event'] == 'handoff' for e in d.policy.events):
      handoff = i
      break
  assert handoff is not None and not any(e['event'] == 'confidence_lost' for e in d.events)


def test_own_low_prob_drops_the_front_and_qualifies_nothing():
  d = Drive()
  for i in range(8):
    d.step(i, x=4.-.15*i, speed=-3.)
  assert d.policy.fronts[0]['qualified']
  d.md.laneLineProbs[1] = .09
  d.step(8, x=4.-.15*8, speed=-3.)
  assert d.policy.fronts[0] is None and [e['event'] for e in d.policy.events] == ['confidence_lost']
  for i in range(9, 30):
    d.step(i, x=max(4.-.15*i, .2), speed=-3.)
    assert d.policy.fronts[0] is None and d.policy.rears[0] is None
  assert not any(e['event'] in ('coast', 'handoff') for e in d.events)


def test_rain_vrel_noise_keeps_a_qualified_car_and_hands_it_over():
  """00000404 (rain): the same radar id's vRel jumps 2-6 m/s between reports while x/y follow the car."""
  d = Drive()
  noise = [0., 4., -3., 2.5, -4., 3., -2., 4.5]
  for i in range(14):
    d.step(i, x=3.4-.12*i, speed=-2.4+noise[i % len(noise)])
  assert d.policy.fronts[0]['qualified']
  assert not any(e['event'] == 'identity_innovation' for e in d.events)
  v, _ = d.step(14)  # it leaves LF
  assert v['LR_DETECT'] and any(e['event'] == 'coast' for e in d.events)


def test_object_passed_at_high_speed_with_little_speed_of_its_own_is_not_qualified():
  """3dd/5 35.3 s: at 21 m/s a 0.7 m/s object (Doppler noise on a structure) is not a moving car."""
  d = Drive()
  d.cs.out.vEgo = 21.
  for i in range(15):
    d.step(i, x=9.-.5*i, speed=-20.3, vlead=.7)
  assert not d.policy.fronts[0]['qualified']


def test_same_source_new_map_key_without_a_raw_report_does_not_hide_the_new_front_car():
  """Review 10-20 item 1: the map selects (1, 1) at 20 m while liveTracks has no report of id 1. The qualified (1, 0)
  car goes to LR as before; the new front car stays drawn and is not consumed."""
  d = Drive()
  d.seed()
  v = dict(d.history[-1])
  v.update(LF_DETECT=2, LF_DETECT_DISTANCE=20., LR_DETECT=0)
  d.cs.live_tracks = N(points=[])
  d.policy.update(v, d.cs, d.md, 75, [None, (1, 1), None], d.filters, 'same-context')
  assert v['LF_DETECT'] == 2 and v['LR_DETECT'] == 2
  assert (1, 1) not in d.policy.consumed[0] and d.policy.rears[0]['key'] == (1, 0)


def test_same_source_new_map_key_with_a_continuous_report_keeps_the_memory():
  """404/14 38-40 s: the map alternates two keys of one radar id; a continuous raw report keeps the qualification."""
  d = Drive()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  assert d.policy.fronts[0]['qualified']
  values, _ = d.step(10, x=4.-1.5, speed=-3., drawn=(1, .8*2.5))
  saved = d.policy.fronts[0]
  d.cs.live_tracks = N(points=[N(trackId=1, dRel=2.35, yRel=3., vRel=-3., vLead=17., measured=True, trackState=2,
                                 motionState=2, radarSource='frontRadar')])
  v = {k: values[k] for k in values}
  v.update(LF_DETECT=2, LF_DETECT_DISTANCE=.8*2.35, LR_DETECT=0)
  d.policy.update(v, d.cs, d.md, 55, [None, (1, 7), None], d.filters, 'same-context')
  assert d.policy.fronts[0] is saved and saved['key'] == (1, 7) and saved['qualified'] and v['LF_DETECT'] == 2


def test_stopped_check_includes_the_newest_report():
  """Review 10-20 item 2: vLead 1, then -1 twice; the second -1 arrives with the front replacement: stopped."""
  d = Drive()
  d.cs.out.vEgo = 4.
  d.seed(speed=-3., vlead=1.)
  assert d.policy.fronts[0]['qualified']
  d.cs.out.vEgo = 2.
  d.step(15, x=1.1, speed=-3., vlead=-1.)
  v, _ = d.step(16, x=1., speed=-3., vlead=-1., drawn=(2, 20.))
  assert not v['LR_DETECT'] and any(e['event'] == 'stopped' for e in d.policy.events)


def test_queue_car_stopped_beside_a_slow_ego_still_qualifies():
  """404/20 29.4 s (review 10-20 item 3): a car stopped in the next lane's queue (vLead ~0.1, radar motionState 2)
  passed at 7.5 m/s qualifies as before."""
  d = Drive()
  d.cs.out.vEgo = 7.5
  for i in range(15):
    d.step(i, x=6.-.35*i, speed=-7.4, vlead=.1)
  assert d.policy.fronts[0]['qualified']
