"""Display trajectories: origin seams, BSD grace, replacement ordering and lateral continuity."""
import copy

import pytest

from opendbc.car.hyundai.hyundaicanfd_ccnc_side_flow import SideFlow
from opendbc.car.hyundai.hyundaicanfd_ccnc_rear import SmoothHandover
from tools.ccnc.test_rear_handover import Drive  # noqa: TID251


def car(key=1, x=4., y=4.):
  return {'key': (key, 0), 'x': x, 'y': y, 'detect': 2}


class Scene:
  def __init__(self):
    self.flow, self.now, self.rows = SideFlow(), -.05, []
    self.motion = None

  def step(self, candidate=None, bsd=False, elsewhere=()):
    self.now = round(self.now+.05, 8)
    if self.flow.front and (candidate is None or candidate['key'] != self.flow.front['key']) and self.motion is None:
      self.motion = {'key': self.flow.front['key'], 'phase': 'rear', 'distance': 0., 'signed': 0.,
                     'shown': self.now, 'force_at': self.now+6., 'deadline': self.now+10., 'max_speed': 8., 'speed': 2.}
    if self.motion:
      self.motion['distance'] += .1  # deterministic stand-in for the existing model's 2 m/s progress
    f, r = self.flow.update(self.now, .05, candidate, bsd, 2.8, elsewhere=elsewhere, motion=self.motion)
    self.rows.append(copy.deepcopy((self.now, f, r)))
    return f, r


@pytest.mark.parametrize('old_distance,new_distance', [(5.01, 6.), (20., 80.), (80., 6.)])
def test_different_far_front_replaces_position_immediately(old_distance, new_distance):
  flow = SideFlow()
  flow.update(0., .05, car(1, old_distance, 4.), False, 2.8)
  candidate = dict(car(2, new_distance, 3.), detect=3)
  f, r = flow.update(.05, .05, candidate, False, 2.8)
  assert f == dict(candidate, phase='front')
  assert r is None
  # Ordinary tracking still uses its existing distance jump limit after the replacement.
  candidate['x'] += 10.
  f, r = flow.update(.1, .05, candidate, False, 2.8)
  assert f['x'] == pytest.approx(new_distance+1.)
  assert r is None


@pytest.mark.parametrize('old_distance', [4., 5.])
def test_new_far_front_waits_for_old_front_zero_and_rear_receives_same_lateral(old_distance):
  s = Scene()
  s.step(car(x=old_distance))
  for _ in range(200):
    f, r = s.step(car(2, 24., 3.1))
    if r is not None:
      assert r['x'] == 0. and r['y'] == 4.
      assert f['key'] == (2, 0) and f['x'] == 24.
      assert 0. < s.rows[-2][1]['x'] <= .1+1e-9 and s.rows[-2][1]['y'] == 4.
      break
    assert f['key'] == (1, 0) and f['y'] == 4.
  else:
    pytest.fail('old F never handed over')
  _, r = s.step(car(2, 24., 3.1))
  assert 2.8 < r['y'] < 4. and 4.-r['y'] <= .1+1e-9


def test_new_far_front_is_admitted_when_the_origin_replaces_an_existing_bsd_rear():
  s = Scene()
  s.step(car())
  for _ in range(100):
    s.step(car(), bsd=True)
  previous = s.flow.rear
  for i in range(40):
    f, r = s.step(car(2, 24.), bsd=True)
    if 'front_to_rear' in s.flow.events:
      assert r['x'] == 0. and f['x'] == 24. and f['key'] == (2, 0)
      assert r is not previous and 'rear_display_replaced' in s.flow.events
      assert i == 39  # Scene's 2m/s model: exactly 4m / 2m/s, with no rear-slot delay.
      break
    assert r is previous and r['x'] == -8.
  else:
    pytest.fail('the existing BSD rear blocked the pass')


@pytest.mark.parametrize('old_phase', ['virtual', 'bsd', 'exit'])
@pytest.mark.parametrize('new_distance', [2., 24.])
def test_front_origin_immediately_replaces_any_occupied_rear_and_resets_its_state(old_phase, new_distance):
  flow = SideFlow()
  old_motion = None if old_phase == 'bsd' else {'key': (9, 0), 'phase': 'rear', 'distance': 8.,
    'force_at': 10., 'deadline': 14., 'max_speed': 8., 'end_reason': 'rear_replaced'}
  old = dict(car(9, -8., 2.8), phase=old_phase, hold_until=100., spacing=True,
             motion=old_motion, model_distance=8.)
  flow.rear = old
  flow.front = dict(car(1, .1, 4.), phase='to_rear')
  motion = {'key': (1, 0), 'phase': 'rear', 'distance': 0., 'signed': 0., 'speed': 2.,
            'force_at': 10., 'deadline': 14., 'max_speed': 8.}
  f, r = flow.update(1., .05, car(2, new_distance), False, 2.8, motion=motion)
  assert r is not old and r['key'] == (1, 0) and r['x'] == 0. and r['y'] == 4.
  assert r['motion'] is motion and r['model_distance'] == 0. and 'hold_until' not in r
  assert r.get('spacing', False) == (new_distance < 6.) and 'rear_display_replaced' in flow.events
  assert (f is not None) == (new_distance >= 6.)
  if f:
    assert f['x'] == new_distance and f['key'] == (2, 0)
  motion['distance'] = .1
  _, r = flow.update(1.05, .05, car(2, new_distance), False, 2.8, motion=motion)
  assert r['x'] < 0. and r['y'] < 4.  # Neither the old hold timer nor the old lateral survives.


@pytest.mark.parametrize('bsd_already_on', [False, True])
def test_new_front_handoff_goes_to_eight_when_bsd_is_on_or_arrives_after_the_pass(bsd_already_on):
  flow = SideFlow()
  flow.bsd, flow.bsd_off = bsd_already_on, None if bsd_already_on else -5.
  flow.rear = dict(car(9, -8. if bsd_already_on else -12., 2.8), phase='bsd')
  flow.front = dict(car(1, .1, 4.), phase='to_rear')
  motion = {'key': (1, 0), 'phase': 'rear', 'distance': 0., 'signed': 0., 'speed': 2.,
            'force_at': 10., 'deadline': 14., 'max_speed': 8.}
  _, r = flow.update(0., .05, car(2, 24.), bsd_already_on, 2.8, motion=None if bsd_already_on else motion)
  assert r['x'] == 0. and r['key'] == (1, 0) and r['y'] == 4.
  for i in range(1, 120):
    motion['distance'] += .4  # Must not pull the BSD drawing past 8m, even with fast relative motion.
    _, r = flow.update(i*.05, .05, car(2, 24.), True, 2.8)
    assert -8. <= r['x'] < 0. and r['motion'] is None
    if i == 1:
      assert r['x'] == pytest.approx(-.15)  # No one-frame 6m/s spike when BSD replaces the relative-motion model.
  assert r['x'] == -8.


def test_moving_front_crosses_origin_without_an_ease_out_or_a_second_zero_packet():
  s = Scene()
  s.step(car(x=1.))
  for _ in range(30):
    s.step(car(2, 24.))
  positions = [f['x'] if f and f['key'] == (1, 0) else r['x'] for _, f, r in s.rows]
  seam = next(i for i, x in enumerate(positions) if x == 0.)
  assert sum(x == 0. for x in positions) == 1
  assert positions[seam-2]-positions[seam-1] == pytest.approx(.1)
  assert 0. < positions[seam-1] <= .1+1e-9
  assert positions[seam+1] == pytest.approx(-.1)
  assert positions[seam+2] == pytest.approx(-.2)


@pytest.mark.parametrize('bsd', [False, True])
def test_bsd_front_handoff_keeps_moving_after_zero_during_presence_grace(bsd):
  flow = SideFlow()
  flow.front = dict(car(x=1.), phase='front')
  flow.bsd_off = 0.
  received = False
  for i in range(1, 30):
    f, r = flow.update(i*.05, .05, None, bsd, 2.8)
    if 'front_to_rear' in flow.events:
      assert f is None and r['x'] == 0. and r['y'] == 4.
      received = True
    elif received:
      assert r['x'] == pytest.approx(-.15) and r['y'] < 4.
      break
  else:
    pytest.fail('pass stopped at the origin')


def test_rear_to_front_brakes_at_the_front_target_instead_of_at_the_slot_boundary():
  s = Scene()
  for _ in range(100):
    s.step(bsd=True)
  positions = []
  for _ in range(60):
    f, r = s.step(car(x=3.), bsd=False)
    positions.append(f['x'] if f else r['x'])
    if f and f['x'] >= 2.:
      break
  seam = positions.index(0.)
  assert positions.count(0.) == 1
  assert positions[seam-1]-positions[seam-2] >= .5
  assert 0. < -positions[seam-1] <= .6
  assert positions[seam+1] >= .5
  assert positions[-1] <= 3.


@pytest.mark.parametrize('distance', [0., 2.04, 2.05, 2.06, 5.99, 6., 24.])
def test_replacement_waits_for_six_metre_gap_while_actively_sending_rear_to_eight(distance):
  s = Scene()
  s.step(car())
  for _ in range(300):
    f, r = s.step(car(2, distance))
    if f is not None and f['key'] == (2, 0):
      assert r is not None and f['x'] == distance
      assert distance >= 6. or int(distance*10)+int(-r['x']*10) >= 60
      if distance >= 6.:
        assert r['x'] == 0.
      break
  else:
    pytest.fail('replacement remained hidden')
  for _ in range(100):
    previous = r['x']
    f, r = s.step(car(2, distance))
    assert f['x'] == distance and previous-.4-1e-9 <= r['x'] <= previous
    if int(distance*10)+int(-r['x']*10) >= 60:
      break
  else:
    pytest.fail('rear did not make room for the visible front')


@pytest.mark.parametrize('existing_front', [False, True])
def test_close_front_sends_stationary_rear_to_eight_and_appears_at_six_metre_gap(existing_front):
  flow = SideFlow()
  motion = {'key': (1, 0), 'phase': 'rear', 'distance': 1., 'shown': 0., 'force_at': 100.,
            'deadline': 104., 'max_speed': 8.}
  flow.rear = dict(car(1, -1.), phase='virtual', motion=motion, model_distance=1., hold_until=0.)
  flow.after_pass = True
  if existing_front:
    flow.front = dict(car(2, 2.), phase='front')
  previous = -1.
  first_visible = None
  for i in range(50):
    f, r = flow.update(i*.05, .05, car(2, 2.), False, 2.8, motion=motion)
    if f:
      assert f['x'] == 2. and f['key'] == (2, 0) and 2.+int(-r['x']*10)/10 >= 6.
      first_visible = -r['x'] if first_visible is None else first_visible
    else:
      assert -r['x'] < 4.
    assert previous-.4-1e-9 <= r['x'] <= previous
    previous = r['x']
  assert first_visible is not None and 4. <= first_visible < 8.
  assert r['x'] == -8.
  f, r = flow.update(2.5, .05, car(2, 10.), False, 2.8, motion=motion)
  assert r['x'] == -8.  # shrinking the correction must not pull R back toward the origin


def test_bsd_off_holds_full_three_seconds_then_retires_without_step():
  s = Scene()
  for _ in range(100):
    s.step(bsd=True)
  _, old = s.step()
  x, off = old['x'], s.now
  for _ in range(59):
    _, r = s.step()
    assert r['x'] == x
  assert s.now-off < 3.
  _, r = s.step()
  assert x-.2-1e-9 <= r['x'] < x
  previous = r['x']
  for _ in range(200):
    _, r = s.step()
    if r is None:
      assert previous <= -15.96
      break
    assert previous-.2-1e-9 <= r['x'] <= previous
    previous = r['x']
  else:
    pytest.fail('BSD drawing did not retire')


@pytest.mark.parametrize('remain_on', [False, True])
@pytest.mark.parametrize('distance', [3., 6.])
def test_recent_bsd_slides_through_zero_into_front_and_reenters_from_sixteen(remain_on, distance):
  s = Scene()
  for _ in range(100):
    s.step(bsd=True)
  if not remain_on:
    for _ in range(20):
      s.step()
  prior = s.flow.rear['x']
  for _ in range(100):
    f, r = s.step(car(x=distance, y=4.2), bsd=remain_on)
    if f is not None:
      assert r is None and f['x'] == 0. and f['y'] == 2.8
      break
    assert prior <= r['x'] <= prior+.6+1e-9
    prior = r['x']
  else:
    pytest.fail('R never reached F')
  f, r = s.step(car(x=distance, y=4.2), bsd=remain_on)
  assert 0. < f['x'] <= .6+1e-9 and 2.8 < f['y'] <= 2.9+1e-9
  assert (r is not None) == remain_on
  if r:
    assert r['x'] == -16.
    _, r = s.step(car(x=3., y=4.2), bsd=True)
    assert -16. < r['x'] < -8.


def test_close_initial_front_does_not_wait_for_a_nonexistent_previous_handoff():
  f, r = Scene().step(car(x=1.))
  assert f['x'] == 1. and r is None


def test_empty_side_still_records_bsd_off_and_clears_events():
  flow = SideFlow()
  flow.bsd, flow.after_pass = True, True
  flow.events = ['rear_to_front']
  assert flow.update(5., .05, None, False, 2.8) == (None, None)
  assert flow.bsd is False and flow.bsd_off == 5. and flow.after_pass is True and flow.events == []


def test_reappearing_front_cancels_handoff_without_position_reset():
  s = Scene()
  s.step(car())
  for _ in range(5):
    s.step()
  old = s.flow.front['x']
  f, r = s.step(car())
  assert old < f['x'] <= old+.6 and r is None


def test_ego_lane_change_to_another_slot_removes_the_old_copy_immediately():
  s = Scene()
  s.step(car())
  f, r = s.step(elsewhere=((1, 0),))
  assert f is None and r is None


def test_same_vehicle_reacquired_after_handoff_returns_from_rear_without_duplicate():
  s = Scene()
  s.step(car())
  while s.flow.rear is None:
    s.step()
  for _ in range(100):
    f, r = s.step(car(x=24.))
    assert not (f and r and f['key'] == r['key'])
    if f:
      assert f['x'] == 0.
      break
  else:
    pytest.fail('reacquired vehicle never returned')


def test_rear_reacquisition_while_another_far_front_was_visible_never_duplicates():
  s = Scene()
  s.step(car())
  while s.flow.rear is None:
    s.step(car(2, 24.))
  f, r = s.step(car(1, 20.))
  assert r is None and f['phase'] == 'join_front' and f['key'] == (1, 0) and f['x'] == 0.


def test_virtual_rear_also_holds_three_seconds_after_bsd_falls():
  s = Scene()
  s.step(car())
  while s.flow.rear is None:
    s.step()
  for _ in range(100):
    s.step(bsd=True)
  _, r = s.step()
  x = r['x']
  for _ in range(59):
    _, r = s.step()
    assert r['x'] == x


@pytest.mark.parametrize('fault', ['reverse', 'can', 'clock', 'context'])
def test_production_resets_animation_on_invalid_input_or_context_change(fault):
  d = Drive()
  d.policy = SmoothHandover()
  for i in range(15):
    d.step(i, x=5.)
  if fault == 'reverse':
    d.cs.out.gearShifter = 'reverse'
  elif fault == 'can':
    d.cs.out.canValid = False
  elif fault == 'context':
    d.policy.context = 'changed'
  d.step(100 if fault == 'clock' else 15)
  assert all(f.front is None and f.rear is None for f in d.policy.flows)


@pytest.mark.parametrize('side', [0, 1])
def test_production_policy_receives_zero_before_admitting_twenty_four_metres(side):
  d = Drive(side)
  d.policy = SmoothHandover()
  front = ('LF', 'RF')[side]
  rear = ('LR', 'RR')[side]
  for i in range(15):
    d.step(i, x=2.6-.1*i, drawn=(1, 4.))
  for i in range(15, 200):
    values, _ = d.step(i, x=30., key=2, drawn=(2, 24.))
    if values[rear+'_DETECT']:
      assert values[rear+'_DETECT_DISTANCE'] == 0.
      assert values[rear+'_DETECT_LATERAL'] == 3.
      assert values[front+'_DETECT_DISTANCE'] == 24. and values[front+'_DETECT']
      break
    assert values[front+'_DETECT_DISTANCE'] <= 4.
  else:
    pytest.fail('production policy never handed over')


@pytest.mark.parametrize('speed', [-.5, -2.])
@pytest.mark.parametrize('ego', [5., 20.])
def test_production_rear_uses_original_relative_motion_instead_of_fixed_exit(speed, ego):
  d = Drive()
  d.policy = SmoothHandover()
  for i in range(15):
    d.step(i, x=1.8+(14-i)*(-speed)*.05, speed=speed, drawn=(1, .2))
  d.cs.out.vEgo = ego
  checked = 0
  deadline = None
  for i in range(15, 260):
    old = d.policy.flows[0].rear
    snapshot = dict(old) if old else None
    values, _ = d.step(i)
    r = d.policy.flows[0].rear
    now = i*.05
    physical = d.policy.rears[0]
    if physical is not None and deadline is None:
      deadline = physical['deadline']
    if r:
      assert deadline is not None and now <= deadline+1e-9
    if snapshot and r:
      assert 0. <= snapshot['x']-r['x'] <= 8.*.05+1e-9
    if (snapshot and r and r.get('motion') is snapshot.get('motion') and r.get('motion') is not None
        and r['phase'] == 'virtual' and now < r['motion']['force_at']):
      expected = r['motion']['distance']-snapshot['model_distance']
      assert values['LR_DETECT_DISTANCE']+snapshot['x'] == pytest.approx(expected, abs=1e-9)
      assert d.policy._retreat(r['motion']) == pytest.approx(-speed*ego/20.*.8)
      checked += 1
  assert checked >= 5
  assert d.policy.flows[0].rear is None


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('replacement', [None, (2, 24.)])
@pytest.mark.parametrize('evidence', ['none', 'unqualified'])
def test_near_display_passes_without_radar_qualification_or_bsd(side, replacement, evidence):
  d = Drive(side)
  d.policy = SmoothHandover()
  front, rear = ('LF', 'LR') if side == 0 else ('RF', 'RR')
  for i in range(2):
    d.step(i, x=4.8 if evidence == 'unqualified' else None, vehicle=False, drawn=(1, 3.8))
  assert d.policy.fronts[side] is None or not d.policy.fronts[side]['qualified']
  last_x = 3.8
  received = False
  for i in range(2, 260):
    values, _ = d.step(i, drawn=replacement)
    f, r = d.policy.flows[side].front, d.policy.flows[side].rear
    if not received and 'front_to_rear' not in d.policy.flows[side].events:
      assert f is not None and f['phase'] == 'to_rear' and f['y'] == 3.
      assert last_x-f['x'] == pytest.approx(.15)
      last_x = f['x']
    elif not received:
      assert r['x'] == 0. and r['y'] == 3. and r['motion'] is None
      assert last_x <= .15+1e-9
      assert bool(values[front+'_DETECT']) == (replacement is not None)
      received = True
    elif r is None:
      assert not values[rear+'_DETECT']
      break
    else:
      assert r['x'] < 0. and -r['x'] <= 16.
  else:
    pytest.fail('unqualified display did not finish its bounded rear exit')
  assert received


def test_distant_display_is_not_handed_over_without_a_near_pass():
  d = Drive()
  d.policy = SmoothHandover()
  for i in range(3):
    d.step(i, x=8., drawn=(1, 6.))
  for i in range(3, 70):
    values, _ = d.step(i)
    assert not values['LR_DETECT']
  assert not any(e['event'] == 'front_to_rear' for e in d.events)


def test_original_ego_stop_holds_the_front_coast_without_inventing_a_backward_pass():
  d = Drive()
  d.policy = SmoothHandover()
  d.seed()
  d.cs.out.vEgo = 0.
  for i in range(15, 110):
    values, _ = d.step(i)
    assert not values['LR_DETECT']
    assert values['LF_DETECT_DISTANCE'] == pytest.approx(.96)
  assert d.policy.rears[0]['phase'] == 'coast'


def test_original_coast_cancellation_removes_animation_when_car_leaves_outer_lane():
  d = Drive()
  d.policy = SmoothHandover()
  for i in range(10):
    d.step(i, x=4.-.15*i, speed=-3.)
  d.step(10)
  for n, i in enumerate(range(11, 20), 1):
    values, _ = d.step(i, x=2.5-.15*n, speed=-3., y=3.+.3*n, show=False)
    if d.policy.rears[0] is None:
      assert not values['LF_DETECT'] and not values['LR_DETECT']
      break
  else:
    pytest.fail('cancelled coast survived as a drawing')


def test_lane_change_retires_the_rear_through_bounded_exit_even_after_forced_retirement_started():
  """DaBin 10-26 (was: the blinker removed it mid-screen, 40f--9 19.47 s LR 12.2 m): the cancelled radar R finishes the
  same bounded exit as distance_end/timeout, at most EXIT_MAX_SPEED, and never comes back toward the origin."""
  d = Drive()
  d.policy = SmoothHandover()
  d.policy.MAX_TIME = .5
  for i in range(15):
    d.step(i, x=2.6-.1*i, drawn=(1, .05))
  for i in range(15, 45):
    d.step(i)
    r = d.policy.flows[0].rear
    if r and d.policy.rears[0] is r.get('motion') and i*.05 >= r['motion']['force_at']:
      break
  else:
    pytest.fail('forced-retirement setup did not have an active rear')
  d.cs.out.leftBlinker = True
  previous = -r['x']
  for j in range(i+1, i+60):
    values, _ = d.step(j)
    r = d.policy.flows[0].rear
    assert d.policy.rears[0] is None  # the original model still cancels its memory
    if r is None:
      assert not values['LR_DETECT'] and previous >= 15.6
      break
    assert values['LR_DETECT'] and r['phase'] == 'exit'
    assert previous <= values['LR_DETECT_DISTANCE'] <= previous+8.*.05+1e-9
    previous = values['LR_DETECT_DISTANCE']
  else:
    pytest.fail('lane-change exit did not finish')


def test_blinker_ends_bsd_grace_and_retires_the_rear_without_removing_it():
  """DaBin 10-26: a held BSD-off R retires to 16 m when the blinker comes on (it is neither held nor dropped)."""
  runs = []
  for blinker in (False, True):
    flow = SideFlow()
    flow.rear = {'key': None, 'x': -8., 'y': 2.8, 'detect': 2, 'phase': 'bsd'}
    flow.bsd = True
    flow.update(0., .05, None, False, 2.8)  # BSD falls: the 3 s grace starts
    xs = []
    for i in range(1, 40):
      _, r = flow.update(i*.05, .05, None, False, 2.8, lane_change=blinker and i >= 10)
      xs.append(r['x'] if r else None)
    runs.append(xs)
  held, retired = runs
  assert set(held) == {-8.}
  assert retired[:9] == [-8.]*9 and None not in retired
  assert all(b < a for a, b in zip(retired[9:], retired[10:], strict=False))
  assert all(a-b <= SideFlow.EXIT_SPEED*.05+1e-9 for a, b in zip(retired[8:], retired[9:], strict=False))


@pytest.mark.parametrize('phase', ['bsd', 'virtual', 'exit'])
def test_blinker_keeps_a_bsd_on_rear_at_eight_metres(phase):
  flow = SideFlow()
  flow.rear = {'key': (9, 0) if phase != 'bsd' else None, 'x': -8., 'y': 2.8, 'detect': 2, 'phase': phase,
               'motion': None, 'model_distance': 0.}
  for i in range(100):
    _, r = flow.update(i*.05, .05, None, True, 2.8, lane_change=True)
    assert r['x'] == -8.


def _lane_change_rear():
  motion = {'key': (9, 0), 'phase': 'rear', 'distance': 3., 'force_at': 10., 'deadline': 14., 'max_speed': 8.,
            'end_reason': 'lane_change'}
  flow = SideFlow()
  flow.rear = dict(car(9, -3., 2.8), phase='virtual', motion=motion, model_distance=3.)
  return flow


def test_lane_change_end_reason_exits_like_other_retired_tracking():
  _, r = _lane_change_rear().update(1., .05, None, False, 2.8, lane_change=True)
  assert r['phase'] == 'exit' and r['x'] == pytest.approx(-3.4)


def test_lane_change_still_removes_an_identity_moved_to_another_slot():
  flow = _lane_change_rear()
  assert flow.update(1., .05, None, False, 2.8, elsewhere=((9, 0),), lane_change=True) == (None, None)


@pytest.mark.parametrize('candidate', [None, 24.])
@pytest.mark.parametrize('start', [0., -8.])
def test_blinker_retires_a_display_only_rear_through_bounded_exit(candidate, start):
  """Review M2: with the blinker on and BSD off a display-only R (motion None) retires like a radar one: phase exit,
  at most EXIT_SPEED, through 16 m (never removed mid-screen)."""
  flow = SideFlow()
  flow.rear = dict(car(1, start, 2.8), phase='virtual', motion=None, model_distance=0.)
  previous = start
  for i in range(200):
    _, r = flow.update(i*.05, .05, car(2, candidate) if candidate else None, False, 2.8, lane_change=True)
    if r is None:
      assert previous <= -15.96
      break
    assert r['phase'] == 'exit' and 0. <= previous-r['x'] <= SideFlow.EXIT_SPEED*.05+1e-9
    previous = r['x']
  else:
    pytest.fail('display-only rear did not retire')


@pytest.mark.parametrize('side_moved', ['ff', 'other'])
@pytest.mark.parametrize('obj', ['rear', 'front'])
def test_lane_change_removes_a_copy_whose_confirmed_alias_moved_to_another_slot(side_moved, obj):
  """Review M3: the original model confirmed (7, 0) as an alias of the drawn (9, 0). When (7, 0) is selected as FF or
  on the other side during a blinker, the old copy goes at once instead of exiting next to the new slot."""
  motion = {'key': (9, 0), 'aliases': [(7, 0)], 'phase': 'rear', 'distance': 3., 'force_at': 10., 'deadline': 14.,
            'max_speed': 8., 'end_reason': 'lane_change', 'signed': 0., 'speed': 2.}
  elsewhere = ((7, 0), None) if side_moved == 'ff' else (None, (7, 0))
  flow = SideFlow()
  if obj == 'rear':
    flow.rear = dict(car(9, -3., 2.8), phase='virtual', motion=motion, model_distance=3.)
    current = None  # the original model already dropped its memory for lane_change
  else:
    flow.front = dict(car(9, 2., 3.), phase='front')
    current = dict(motion, phase='coast', end_reason=None)
  assert flow.update(1., .05, None, False, 2.8, elsewhere=elsewhere, motion=current,
                     lane_change=True) == (None, None)
