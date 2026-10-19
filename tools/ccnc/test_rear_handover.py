"""Display-only rear handoff, including BSD and the unobserved-motion limits."""
import copy
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
      rightBlinker=False, gearShifter='drive', canValid=True))
    self.md = N(laneLineProbs=[1., 1., 1., 1.], meta=N(desire='none', laneChangeState='off'))

  def step(self, i, *, x=None, key=1, speed=-2., vehicle=True, bsd=False, show=True):
    front, rear = ('LF', 'LR') if self.side == 0 else ('RF', 'RR')
    setattr(self.cs.out, 'leftBlindspot' if self.side == 0 else 'rightBlindspot', bsd)
    point = N(trackId=key, dRel=x, yRel=3. if self.side == 0 else -3., vRel=speed, vLead=20.+speed,
              measured=True, trackState=2, motionState=2 if vehicle else 1, radarSource='frontRadar') if x is not None else None
    self.cs.live_tracks = N(points=[point] if point else [])
    values = {}
    for f, r, filter_, blind in zip(('LF', 'RF'), ('LR', 'RR'), self.filters,
                                   (self.cs.out.leftBlindspot, self.cs.out.rightBlindspot), strict=True):
      values.update({f+'_DETECT': 0, f+'_DETECT_DISTANCE': 0., f+'_DETECT_LATERAL': 3.,
                     r+'_DETECT': 0, r+'_DETECT_DISTANCE': 0., r+'_DETECT_LATERAL': 2.8})
      if blind or filter_.value < 15:
        values.update({r+'_DETECT': 2 if blind else 1, r+'_DETECT_DISTANCE': filter_.apply(8 if blind else 16)})
    selected = [None, None, None]
    if point is not None and show:
      selected[self.side+1] = (key, 0)
      values.update({front+'_DETECT': 2, front+'_DETECT_DISTANCE': .8*x})
    original = copy.deepcopy((self.cs, self.md, selected))
    self.policy.update(values, self.cs, self.md, i*5, selected, self.filters, 'same-context')
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


@pytest.mark.parametrize('failure', ['reflection', 'replacement', 'jump', 'positive_rejected', 'too_far'])
def test_unqualified_or_replaced_objects_never_handoff(failure):
  d = Drive()
  d.seed(vehicle=failure != 'reflection')
  if failure == 'replacement':
    d.step(15, x=2., key=2)
  elif failure == 'jump':
    d.step(15, x=12.)
  elif failure == 'positive_rejected':
    d.step(15, x=1.2, show=False)
  elif failure == 'too_far':
    d.policy.sides[0]['distance'] = 5.
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
  assert values[rear+'_DETECT'] == 1 and d.policy.sides[0] is None
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


@pytest.mark.parametrize('reason', ['reverse', 'lane_change', 'confidence', 'clock', 'CAN'])
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
  d.policy.MAX_TIME = .2
  for j in range(i+1, i+8):
    v, _ = d.step(j)
  assert not v[rear+'_DETECT'] and any(e['event'] == 'timeout' for e in d.events)
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
def test_new_front_keeps_old_rear_until_its_own_qualified_crossing(side):
  d = Drive(side)
  i, _, rear = d.handoff()
  front = ('LF', 'RF')[side]
  old = d.policy.rears[side]
  last = 0.
  for n in range(14):
    v, _ = d.step(i+1+n, x=1.4-.1*n, key=2)
    assert d.policy.rears[side] is old
    assert v[front+'_DETECT'] and v[rear+'_DETECT']
    assert v[rear+'_DETECT_DISTANCE'] >= last
    last = v[rear+'_DETECT_DISTANCE']
  for j in range(i+15, i+25):
    v, _ = d.step(j)
    if any(e['event'] == 'handoff' for e in d.policy.events):
      break
  else:
    raise AssertionError('second vehicle never handed off')
  assert d.policy.rears[side]['key'] == (2, 0)
  assert v[rear+'_DETECT_DISTANCE'] == 0 and not v[front+'_DETECT']
  assert d.filters[side].value == 0
  replaced, handoff = d.policy.events
  assert replaced['event'] == 'rear_replaced' and replaced['source'] == 1
  assert handoff['replaced_key'] == (1, 0) and handoff['replaced_distance'] >= last
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
  assert v['LR_DETECT'] == 1 and d.policy.rears[0] is None
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
