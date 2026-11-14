import time
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, STALK_QUEUE_GUARD_S
from openpilot.selfdrive.controls.lib.desire_lib.lever import LEVER_REQUEST_WAIT_S
from openpilot.selfdrive.controls.lib.desire_lib.constants import TurnDirection

LCS = log.LaneChangeState
LCD = log.LaneChangeDirection


def test_manual_turn_signal_remains_isolated_during_acceleration_with_atc_off():
  from openpilot.cereal import car

  for lateral_active in (False, True):
    harness = TestDesireHelperStalkRequest()
    harness.setup_method()
    helper = harness.helper
    helper.boundary_indicator_supported = True
    helper.manual_turn_indicator_enabled = True
    cs = car.CarState.new_message(canValid=True, gearShifter='drive', vEgo=5.)
    md = log.ModelDataV2.new_message()
    md.laneLines = [{'x': [0., 5., 10., 15.], 'y': [y] * 4} for y in (-5., -1.8, 1.8, 5.)]
    md.laneLineProbs = [0.] * 4
    md.meta.desireState = [1., 0., 0., 0., 0., 0., 0., 0.]
    saw_request = False
    with patch('openpilot.selfdrive.controls.lib.desire_helper.read_json', return_value={}):
      for i in range(90):
        now = 10 + i * .05
        md.frameId = i
        cs.steeringPressed = 25 <= i < 55
        cs.steeringTorque = 400. if cs.steeringPressed else 0.
        cs.steeringAngleDeg = min(100., max(0, i - 25) * 8.)
        cs.vEgo = 12. if i >= 55 else 5.
        cs.leftBlinker = i >= 50
        with patch('openpilot.selfdrive.controls.lib.desire_helper.time.monotonic', return_value=now):
          helper.update(cs, md, lateral_active, 0., harness.carrot_man, SimpleNamespace(),
                        indicator_valid=True, indicator_yaw_rate=-.2 if i >= 29 else 0.)
        saw_request |= helper.model_indicator.active is not None
        assert helper.lane_change_state == LCS.off
        assert helper.desire == log.Desire.none
    assert saw_request and helper.model_indicator.active.source == 'manual_turn'


class TestDesireHelperStalkRequest:
  """Turn-signal lever presses (carState.*BlinkerStalkCount) as explicit next-lane-change requests."""

  def setup_method(self):
    # These FSM tests must not read workstation/device settings or Bluetooth IO.
    with patch('openpilot.selfdrive.controls.lib.desire_helper.Params'), \
         patch('openpilot.selfdrive.controls.lib.desire_helper.CommandReader') as reader:
      reader.return_value.read.return_value = None
      self.helper = DesireHelper()
    self.helper._update_params_periodic = lambda: None
    self.helper._process_sides = lambda car, model, radar: None
    self.helper._check_desire_state = lambda model, car, maneuver: None
    self.helper.laneChangeNeedTorque = 0
    self.helper.laneChangeBsd = 0
    self.helper.laneLineCheck = 0
    for side in (self.helper.left, self.helper.right):
      side.lane_available = True
      side.edge_available = False
      side.dist_to_edge_far = 2.0
      side.lane_change_available_geom = True
      side.lane_change_available = True
      side.side_object_detected = False
      side.bsd_hold_counter = 0
      side.lane_line_info_mod = 0
      side.lane_line_info_edge_detect = False
      side.lane_change_available_released = False
      side.lane_available_trigger = False
      side.lane_appeared = False
      side.lane_exist_count.counter = 10
    self.carrot_man = SimpleNamespace(atcType="", carrotCmdIndex=0, carrotCmd="", carrotArg="")
    self.left_count = 0

  def update(self, left_blinker=True, lane_change_prob=0.5, press=False, counters=True, lever=0, v_kph=90.0, lines=None):
    if press:
      self.left_count = (self.left_count + 1) % 256
    car = SimpleNamespace(canValid=True, leftBlinker=left_blinker, rightBlinker=False, vEgo=v_kph / 3.6, aEgo=0.0,
                          trailerConnected=False, steeringTorque=0.0, steeringPressed=False, blinkerLever=lever)
    if counters:
      car.leftBlinkerStalkCount, car.rightBlinkerStalkCount = self.left_count, 0
    model = SimpleNamespace(laneLines=[SimpleNamespace(y=[y]) for y in (-5.0, *lines, 5.0)]) if lines else SimpleNamespace()
    self.helper.update(car, model, True, lane_change_prob, self.carrot_man, SimpleNamespace())
    return self.helper.lane_change_state

  def start_and_cross(self, counters=True):
    """off -> pre -> starting with the left blinker, then hold starting until the model says done."""
    self.update(left_blinker=False, counters=counters)
    assert self.update(press=counters, counters=counters) == LCS.preLaneChange
    assert self.update(counters=counters) == LCS.laneChangeStarting
    for _ in range(5):
      self.update(counters=counters)

  def finish(self, press_at_s=None, counters=True):
    """laneChangeStarting -> finishing -> end; optionally press the lever press_at_s into finishing."""
    for _ in range(int(0.6 / DT_MDL)):                      # ll_prob decays at 2/s; model prob already low
      if self.update(lane_change_prob=0.0, counters=counters) == LCS.laneChangeFinishing:
        break
    assert self.helper.lane_change_state == LCS.laneChangeFinishing
    t = DT_MDL                                                # the transition frame counts as the first
    while self.helper.lane_change_state == LCS.laneChangeFinishing:
      press = press_at_s is not None and abs(t - press_at_s) < DT_MDL / 2
      self.update(lane_change_prob=0.0, press=press, counters=counters)
      t += DT_MDL
    return self.helper.lane_change_state

  def test_blinker_held_through_the_change_still_needs_torque(self):
    self.start_and_cross()
    assert self.finish() == LCS.preLaneChange
    assert self.helper.next_lane_change
    assert self.update() == LCS.preLaneChange              # no torque: no second lane change

  def test_press_while_crossing_is_ignored(self):
    self.start_and_cross()
    self.update(press=True)                                 # re-lighting the lamp mid-change
    assert self.helper.lane_change_state == LCS.laneChangeStarting
    assert self.finish() == LCS.preLaneChange
    assert self.helper.next_lane_change
    assert self.update() == LCS.preLaneChange

  def test_press_during_finishing_queues_the_next_change_without_torque(self):
    self.start_and_cross()
    assert self.finish(press_at_s=STALK_QUEUE_GUARD_S + 0.2) == LCS.preLaneChange
    assert not self.helper.next_lane_change
    assert self.update() == LCS.laneChangeStarting          # one lane at a time: starts after the first ended

  # left change, 2026-10-05 drive: the left line closes in, the model relabels it, then the car settles in the new lane
  CROSSING = [(-1.3, 1.9), (-0.9, 2.2), (-0.5, 2.7), (-0.34, 2.84), (-0.41, 2.83), (-2.5, 0.47), (-2.94, 0.16)]

  def cross(self, press_after_s=None, press_before=False):
    """Still in laneChangeStarting: the car centre crosses the left line; optionally press before or after it."""
    for i, lines in enumerate(self.CROSSING):
      self.update(lines=lines, press=press_before and i == 1)
    t = DT_MDL
    for _ in range(int(1.2 / DT_MDL)):                      # the model keeps the change probability up after it
      press = press_after_s is not None and abs(t - press_after_s) < DT_MDL / 2
      self.update(lines=(-2.6, 0.6), press=press)
      t += DT_MDL
    assert self.helper.lane_change_state == LCS.laneChangeStarting

  def test_press_after_the_crossing_queues_the_next_change(self):
    self.start_and_cross()
    self.cross(press_after_s=0.1)                          # 2026-10-05 #1: pressed 0.1 s after the relabel
    assert self.helper.queued_lane_change == 1
    assert self.helper.blinker_hold_direction == LCD.left  # keep the lamp lit until the queued change starts
    assert self.finish() == LCS.preLaneChange
    assert not self.helper.next_lane_change
    assert self.update() == LCS.laneChangeStarting

  def test_queued_request_keeps_the_lamp_until_the_next_change_starts(self):
    self.start_and_cross()
    self.cross(press_after_s=0.5)
    side = self.helper.left
    side.bsd_hold_counter, side.lane_change_available = 40, False   # BSD when the next change becomes due
    assert self.finish() == LCS.preLaneChange
    assert self.helper.blinker_hold_direction == LCD.left
    for _ in range(int(2.0 / DT_MDL)):
      self.update()
    assert self.helper.blinker_hold_direction == LCD.left     # still waiting, lamp held
    side.bsd_hold_counter, side.lane_change_available = 0, True
    assert self.update() == LCS.laneChangeStarting
    assert self.helper.queued_lane_change == 0                # the queued change started
    self.update()
    assert self.helper.blinker_hold_direction == LCD.none

  def test_blocked_queued_request_is_dropped_after_five_flashes(self):
    self.start_and_cross()
    self.cross(press_after_s=0.5)
    side = self.helper.left
    side.bsd_hold_counter, side.lane_change_available = 40, False
    self.finish()
    for _ in range(int((LEVER_REQUEST_WAIT_S + 0.2) / DT_MDL)):
      self.update()
    assert self.helper.queued_lane_change == 0
    assert self.helper.blinker_hold_direction == LCD.none

  def test_press_before_the_crossing_is_ignored(self):
    self.start_and_cross()
    self.cross(press_before=True)
    assert self.helper.queued_lane_change == 0
    assert self.finish() == LCS.preLaneChange
    assert self.helper.next_lane_change

  def test_press_late_in_the_change_after_the_crossing_still_queues(self):
    self.start_and_cross()
    self.cross(press_after_s=1.1)
    assert self.helper.queued_lane_change == 1

  def test_press_right_after_the_crossing_is_still_the_same_change(self):
    self.start_and_cross()
    assert self.finish(press_at_s=STALK_QUEUE_GUARD_S / 2) == LCS.preLaneChange
    assert self.helper.next_lane_change

  def test_press_while_waiting_for_torque_releases_the_wait(self):
    self.start_and_cross()
    self.finish()
    assert self.helper.next_lane_change
    self.update(press=True)
    assert not self.helper.next_lane_change
    assert self.update() == LCS.laneChangeStarting

  def test_press_while_the_lamp_still_flashes_starts_a_request_from_off(self):
    self.update(left_blinker=True)                          # lamp already flashing: no lamp rising edge left
    self.helper.lane_change_state = LCS.off
    self.helper.prev_desire_enabled = True
    assert self.update() == LCS.off
    assert self.update(press=True) == LCS.preLaneChange

  def test_cars_without_lever_counters_keep_the_lamp_behaviour(self):
    self.start_and_cross(counters=False)
    assert self.finish(counters=False) == LCS.preLaneChange
    assert self.helper.next_lane_change
    assert self.update(counters=False) == LCS.preLaneChange


class TestBlinkerLatchedTurn:
  """BlinkerLatchedTurn (here 50 km/h): at or below it the turn gesture (one-touch, two lamp lightings, then latched)
  is a turn and any other lever input a lane change; above it every lever input is a lane change."""
  setup_method = TestDesireHelperStalkRequest.setup_method
  update = TestDesireHelperStalkRequest.update

  def lever_on(self, lever, v_kph=45.0, steps=1):
    self.update(left_blinker=False, v_kph=v_kph)
    for i in range(steps):
      state = self.update(press=(i == 0), lever=lever, v_kph=v_kph)
    return state

  def gesture(self):
    # LeverForcedTurn itself is covered by test_lever_forced_turn.py; here a latch counts as the finished gesture.
    self.helper.lever_forced_turn.update = lambda lever, lamp: lever == 2

  def test_off_keeps_the_classifier(self):
    self.lever_on(2, steps=3)
    assert self.helper.maneuver_type == "lane_change"       # 45 km/h, lanes visible: the classifier says lane change

  def test_plain_latch_is_a_lane_change_the_gesture_a_turn(self):
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.lever_on(2, steps=3)
    assert self.helper.maneuver_type == "lane_change"       # e.g. holding the signal for the next lane
    self.setup_method()
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.gesture()
    self.lever_on(2, steps=3)
    assert self.helper.maneuver_type == "turn"
    assert self.helper.lane_change_state == LCS.off
    assert self.helper.turn_direction == TurnDirection.turnLeft

  def test_latch_passing_the_one_touch_detent_does_not_start_a_lane_change(self):
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.gesture()
    assert self.lever_on(1, steps=3) == LCS.preLaneChange    # still in the detent: wait
    self.update(lever=2, v_kph=45.0)
    assert self.helper.maneuver_type == "turn"
    assert self.helper.lane_change_state == LCS.off

  def test_one_touch_is_a_lane_change_after_the_lever_returns(self):
    self.helper.blinkerLatchedTurn = 50 / 3.6
    assert self.lever_on(1, steps=2) == LCS.preLaneChange
    assert self.update(lever=0, v_kph=45.0) == LCS.laneChangeStarting
    assert self.helper.maneuver_type == "lane_change"

  def test_latched_lever_above_the_set_speed_is_a_lane_change(self):
    # At 80 km/h the classifier calls a navigation turn with a far road edge a turn; the lever setting overrides it.
    self.carrot_man.atcType = "turn left"
    self.helper.left.dist_to_edge_far = 5.0
    self.lever_on(2, v_kph=80.0, steps=3)
    assert self.helper.maneuver_type == "turn"               # off: the classifier's guess
    self.setup_method()
    self.carrot_man.atcType = "turn left"
    self.helper.left.dist_to_edge_far = 5.0
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.lever_on(2, v_kph=80.0, steps=3)
    assert self.helper.maneuver_type == "lane_change"
    assert self.helper.lane_change_state == LCS.laneChangeStarting

  def test_gesture_at_the_set_speed_is_a_turn(self):
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.gesture()
    self.lever_on(2, v_kph=50.0, steps=3)
    assert self.helper.maneuver_type == "turn"
    assert self.helper.lever_turn                           # cluster message: the lever made this turn

  def test_lever_turn_flag_only_for_a_lever_made_turn(self):
    # off: a navigation turn guessed by the classifier is a turn, but not the lever's
    self.carrot_man.atcType = "turn left"
    self.helper.left.dist_to_edge_far = 5.0
    self.lever_on(2, v_kph=80.0, steps=3)
    assert self.helper.maneuver_type == "turn" and not self.helper.lever_turn
    # on, one-touch at 45 km/h: a lane change, no message
    self.setup_method()
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.lever_on(1, steps=2)
    self.update(lever=0, v_kph=45.0)
    assert self.helper.maneuver_type == "lane_change" and not self.helper.lever_turn
    # the lever released: the message goes with the turn
    self.setup_method()
    self.helper.blinkerLatchedTurn = 50 / 3.6
    self.gesture()
    self.lever_on(2, steps=3)
    assert self.helper.lever_turn
    self.update(left_blinker=False, lever=0, v_kph=45.0)
    assert not self.helper.lever_turn


class TestLaneChangeLeverWait:
  """LaneChangeLeverWait: a blocked one-touch request holds the lamp for 5 flashes from the press, then gives up."""
  update = TestDesireHelperStalkRequest.update

  def setup_method(self):
    TestDesireHelperStalkRequest.setup_method(self)
    self.helper.laneChangeLeverWait = True
    self.block(True)

  def block(self, blocked):
    side = self.helper.left                                 # BSD on the left: the FSM waits for torque
    side.bsd_hold_counter = 40 if blocked else 0
    side.lane_change_available = not blocked

  def press(self, lever=0):
    self.update(left_blinker=False)
    assert self.update(press=True, lever=lever) == LCS.preLaneChange

  def wait(self, seconds, lever=0):
    for _ in range(round(seconds / DT_MDL)):
      self.update(lever=lever)
    return self.helper.lane_change_state

  def test_blocked_request_holds_the_lamp_then_gives_up(self):
    self.press()
    assert self.helper.blinker_hold_direction == LCD.left
    assert self.wait(LEVER_REQUEST_WAIT_S - 0.2) == LCS.preLaneChange
    assert self.helper.blinker_hold_direction == LCD.left
    assert self.wait(0.3) == LCS.off                        # 5 flashes after the press: dropped
    assert self.helper.blinker_hold_direction == LCD.none
    self.block(False)
    assert self.wait(1.0) == LCS.off                        # lamp still flashing: no restart from it
    self.update(left_blinker=False)                         # lamp dark: the next input is a new request
    assert self.update() == LCS.preLaneChange

  def test_clearing_within_the_wait_starts_the_change(self):
    self.press()
    self.wait(2.0)
    self.block(False)
    assert self.update() == LCS.laneChangeStarting
    self.update()
    assert self.helper.blinker_hold_direction == LCD.none  # the change itself holds the lamp from here

  def test_a_new_press_restarts_the_count(self):
    self.press()
    self.wait(3.0)
    self.update(press=True)
    assert self.wait(LEVER_REQUEST_WAIT_S - 0.5) == LCS.preLaneChange
    assert self.wait(0.6) == LCS.off

  def test_latched_lever_keeps_waiting(self):
    self.press(lever=2)
    assert self.wait(LEVER_REQUEST_WAIT_S + 1.0, lever=2) == LCS.preLaneChange
    assert self.helper.blinker_hold_direction == LCD.none

  def test_lane_change_delay_extends_the_wait(self):
    self.helper.laneChangeDelay = 5.0                       # longer than the 5 flashes: must not give up first
    self.block(False)
    self.press()
    assert self.wait(LEVER_REQUEST_WAIT_S + 0.5) == LCS.preLaneChange
    assert self.wait(1.0) == LCS.laneChangeStarting

  def test_off_keeps_waiting_on_the_lamp(self):
    self.helper.laneChangeLeverWait = False
    self.press()
    assert self.wait(LEVER_REQUEST_WAIT_S + 1.0) == LCS.preLaneChange
    assert self.helper.blinker_hold_direction == LCD.none


class TestAtcPhysicalIndicator:
  setup_method = TestDesireHelperStalkRequest.setup_method
  update = TestDesireHelperStalkRequest.update
  start_and_cross = TestDesireHelperStalkRequest.start_and_cross
  def test_turn_origin_persists_through_lamp_feedback_and_ends_with_desire(self):
    self.carrot_man.atcType = "turn left"
    self.helper.left.dist_to_edge_far = 8.0
    self.update(left_blinker=False, v_kph=10)
    self.update(left_blinker=False, v_kph=10)
    assert self.helper.desire == log.Desire.turnLeft
    assert self.helper.auto_turn_signal == LCD.left
    assert not self.helper.auto_turn_signal_lane_change
    self.update(left_blinker=True, v_kph=10)
    assert self.helper.auto_turn_signal == LCD.left
    self.helper.turn_disable_count = 20
    self.update(left_blinker=True, v_kph=10)
    assert self.helper.desire != log.Desire.turnLeft
    assert self.helper.auto_turn_signal == LCD.none

  def test_driver_lane_change_does_not_become_atc_indicator(self):
    self.start_and_cross()
    assert self.helper.auto_turn_signal == LCD.none

  def test_atc_waits_until_actual_start_and_holds_through_finishing(self):
    self.carrot_man.atcType = "atc right"
    self.update(left_blinker=False)
    self.update(left_blinker=False)
    assert self.helper.lane_change_state == LCS.preLaneChange
    for _ in range(5):
      self.update(left_blinker=False)
      assert self.helper.auto_turn_signal == LCD.none
    self.helper.right.lane_line_info_edge_detect = True
    self.update(left_blinker=False)
    assert self.helper.lane_change_state == LCS.laneChangeStarting
    assert self.helper.auto_turn_signal == LCD.right
    assert self.helper.auto_turn_signal_lane_change
    for _ in range(12):
      self.update(left_blinker=False,lane_change_prob=0.)
    assert self.helper.lane_change_state == LCS.laneChangeFinishing
    assert self.helper.auto_turn_signal == LCD.right
    for _ in range(24):
      self.update(left_blinker=False,lane_change_prob=0.)
    assert self.helper.auto_turn_signal == LCD.none


class TestKickModelOwnership:
  """CanfdBlinkerKickModel: a lane change the driver's lamp did not start owns its lamp side on the Starting frame
  (indicatorIsolationMask), so carcontroller.kick_pulse may light it without the lamp reading back as driver input."""
  def setup_method(self):
    TestDesireHelperStalkRequest.setup_method(self)
    self.cc = self.md = None
    self.seen = []  # (cc frame, laneChangeState the carcontroller saw)

  def step(self, cmd=None, left=False, right=False, prob=0.5, lateral=True):
    car = SimpleNamespace(canValid=True, leftBlinker=left, rightBlinker=right, vEgo=25., aEgo=0., trailerConnected=False,
                          steeringTorque=0., steeringPressed=False, blinkerLever=0, naviTunnel=False, naviRoundabout=False)
    self.helper.bluetooth_commands = SimpleNamespace(read=lambda allowed: cmd if allowed else None)
    self.helper.update(car, SimpleNamespace(), lateral, prob, self.carrot_man, SimpleNamespace())
    return self.helper.lane_change_state

  def test_remote_start_owns_its_side(self):
    for cmd, bit in (('laneLeft', 1), ('laneRight', 2)):
      self.setup_method()
      self.helper.kick_model = True
      assert self.step(cmd) == LCS.preLaneChange and self.helper.boundary_indicator.blocked == 0
      assert self.step() == LCS.laneChangeStarting and self.helper.boundary_indicator.blocked == bit

  def test_driver_lamp_start_does_not_own(self):
    self.helper.kick_model = True
    self.step()
    assert self.step(left=True) == LCS.preLaneChange
    assert self.step(left=True) == LCS.laneChangeStarting and self.helper.boundary_indicator.blocked == 0

  def test_lamp_already_on_owns_nothing(self):
    # kick_pulse never kicks over a lit lamp: owning it would only block the same-side lever for nothing
    for lamp, refreshed in ((True, False), (False, True)):
      self.setup_method()
      self.helper.kick_model = True
      bi = self.helper.boundary_indicator
      bi.own(time.monotonic() - 5., 1)  # an earlier change still owns a lit left lamp: not driver input
      until = bi.isolate_until
      assert self.step('laneLeft', left=lamp) == LCS.preLaneChange
      assert self.step(left=lamp) == LCS.laneChangeStarting and (bi.isolate_until > until) == refreshed

  def test_key_off_owns_nothing(self):
    self.step('laneLeft')
    assert self.step() == LCS.laneChangeStarting and self.helper.boundary_indicator.blocked == 0

  def test_needs_both_keys(self):
    for on, want in (((), False), (('CanfdBlinkerKick',), False), (('CanfdBlinkerKickModel',), False),
                     (('CanfdBlinkerKick', 'CanfdBlinkerKickModel'), True)):
      self.helper.params = SimpleNamespace(get_bool=lambda key, on=on: key in on, get_int=lambda key: 0, get_float=lambda key: 0.)
      self.helper.frame = 100
      DesireHelper._update_params_periodic(self.helper)
      assert self.helper.kick_model == want

  def run(self, frames, lamp=lambda j: False, dh=lambda k: {}, cc_model=lambda j: True):
    """DesireHelper at 20 Hz feeding its modelV2 meta (seen by the next 100 Hz frame) into CarController.kick_pulse.
    Every kick must carry its side's ownership bit in that same modelV2. -> [(cc frame, side)]"""
    from opendbc.car import structs
    from opendbc.car.hyundai.carcontroller import CarController, BLINKER_PULSE_FRAMES
    from opendbc.car.hyundai.lane_crossing import LaneCrossing
    cc = self.cc = self.cc or SimpleNamespace(
      blinker_control=True, blinker_kick_model=True, lane_crossing=LaneCrossing(), blinker_side=0, blinker_pulse_end=0,
      frame=0, yellow_until=[0, 0], lane_change_prev='off', model_pulse=False, params=SimpleNamespace(STEER_THRESHOLD=250))
    md, kicks, start = self.md, [], cc.frame
    for j in range(start, start + frames):
      new = None
      if j % 5 == 0:
        opts = dh(j // 5)
        self.helper.kick_model = opts.pop('kick_model', self.helper.kick_model)
        self.step(left=lamp(j), **opts)
        new = log.ModelDataV2.new_message()
        new.meta.laneChangeState = self.helper.lane_change_state
        new.meta.laneChangeDirection = self.helper.lane_change_direction
        new.meta.indicatorIsolationMask = self.helper.boundary_indicator.blocked
        new = new.as_reader()
      cc.blinker_kick_model = cc_model(j)
      before = cc.blinker_pulse_end
      self.seen.append((j, md and md.meta.laneChangeState))
      CarController.kick_pulse(cc, SimpleNamespace(modelV2=md, out=SimpleNamespace(
        gearShifter=structs.CarState.GearShifter.drive, leftBlinker=lamp(j), rightBlinker=False, steeringTorque=0.,
        naviTunnel=False, naviRoundabout=False, leftLaneLine=10, rightLaneLine=10, cruiseCancelState=False)))
      if cc.blinker_pulse_end != before and cc.blinker_pulse_end == j + BLINKER_PULSE_FRAMES:
        assert md.meta.indicatorIsolationMask & (cc.blinker_side - 2), (j, cc.blinker_side)
        kicks.append((j, cc.blinker_side))
      md = new or md
      cc.frame += 1
    self.md = md
    return kicks

  def test_combined_kick_implies_ownership(self):
    # 1) the driver's one-touch lamp ends within +-100 ms of the Starting edge: never owned, never kicked
    self.helper.kick_model = True
    self.run(100, lambda j: j >= 10)
    edge = next(j for j, state in self.seen if state == 'laneChangeStarting')  # first CC frame that sees it
    for offset in range(-10, 11):
      self.setup_method()
      self.helper.kick_model = True
      assert self.run(300, lambda j, off=edge + offset: 10 <= j < off) == [], offset
      assert self.helper.boundary_indicator.blocked == 0

    # 2) the key just turned on: carcontroller has it, DesireHelper reads it mid-change -> no kick for this change;
    #    the next remote change is owned and kicked once; carcontroller turning it on after an edge does not kick late
    self.setup_method()
    remote = lambda k, cmd: {'cmd': cmd} if k == 2 else {'prob': 0.} if k > 10 else {}
    assert self.run(60, dh=lambda k: dict(remote(k, 'laneLeft'), **({'kick_model': True} if k == 6 else {}))) == []
    assert self.run(200, dh=lambda k: {'prob': 0.}) == [] and self.helper.lane_change_state == LCS.off
    kicks = self.run(300, dh=lambda k: remote(k - 52, 'laneRight'))
    assert [side for _, side in kicks] == [4] and self.helper.boundary_indicator.blocked == 2
    self.setup_method()
    self.helper.kick_model = True
    assert self.run(300, dh=lambda k: remote(k, 'laneLeft'), cc_model=lambda j: j >= 30) == []

    # 3) a remote change cancelled (disengaged) right after its kick, then the lamp rising: the kick stops with the
    #    change and the lamp is the owned side, so no new change starts from it
    self.setup_method()
    self.helper.kick_model = True
    kicks = self.run(40, dh=lambda k: {'cmd': 'laneLeft'} if k == 2 else {})
    assert [side for _, side in kicks] == [3] and self.helper.lane_change_state == LCS.laneChangeStarting
    assert self.run(10, dh=lambda k: {'lateral': False}) == [] and self.helper.lane_change_state == LCS.off
    assert self.cc.blinker_pulse_end <= self.cc.frame  # cut
    assert self.run(200, lamp=lambda j: True) == [] and self.helper.lane_change_state == LCS.off


class TestBlinkerHold:
  """CanfdBlinkerHoldChange / CanfdBlinkerHoldWait (desire_lib/blinker_hold.py). The lamp is simulated as the Sonata reads
  it: carState.left/rightBlinker are the lamp (no lever signal), lit by the driver's one-touch for BCM_FLASH_S, and by our
  hold for as long as DesireHelper asks plus TAIL_S (body controller run-on + carState's 0.5 s lamp hold)."""
  BCM_FLASH_S = 2.4  # one-touch: 3 flashes
  TAIL_S = 1.7       # ~1.2 s body controller run-on after the command stops (EV4 10-26) + 0.5 s carState lamp hold

  def setup_method(self):
    TestDesireHelperStalkRequest.setup_method(self)
    self.t = 100.0
    self.driver = {1: -1., 2: -1.}  # lamp lit by the driver until
    self.ours, self.ours_until = 0, -1.  # lamp lit by our hold until
    self.starts, self.states, self.holds = 0, [], []

  def options(self, change=False, wait=False):
    self.helper.hold_change, self.helper.hold_wait = change, wait

  def tap(self, side=1):
    self.driver[side] = self.t + self.BCM_FLASH_S

  def lamps(self):
    lit = {side for side, until in self.driver.items() if self.t < until}
    if self.ours and self.t < self.ours_until:
      lit.add(self.ours)
    return 1 in lit, 2 in lit

  def step(self, prob=0.5, torque=0., lateral=True, available=True, lever=0, gear='drive', model=None, follow=True):
    for side in (self.helper.left, self.helper.right):
      side.lane_change_available = available
    left, right = self.lamps()
    car = SimpleNamespace(canValid=True, leftBlinker=left, rightBlinker=right, vEgo=25., aEgo=0., trailerConnected=False,
                          steeringTorque=torque, steeringPressed=torque != 0., blinkerLever=lever, gearShifter=gear)
    prev = self.helper.lane_change_state
    with patch('openpilot.selfdrive.controls.lib.desire_helper.time.monotonic', return_value=self.t):
      self.helper.update(car, model or SimpleNamespace(), lateral, prob, self.carrot_man, SimpleNamespace(), assistant_valid=True)
    state = self.helper.lane_change_state
    self.starts += state == LCS.laneChangeStarting and prev != LCS.laneChangeStarting
    hold = {LCD.left: 1, LCD.right: 2}.get(self.helper.blinker_hold_direction, 0)
    if hold and follow:
      self.ours, self.ours_until = hold, self.t + self.TAIL_S
    self.states.append(state)
    self.holds.append(hold)
    self.t += DT_MDL
    return state

  def run(self, seconds, **kw):
    for _ in range(round(seconds / DT_MDL)):
      self.step(**kw)
    return self.helper.lane_change_state

  def change(self, seconds=5.0, **kw):
    """one-touch left, the change runs `seconds` (model probability up), then ends"""
    self.tap(1)
    assert self.step(**kw) == LCS.preLaneChange
    self.run(seconds, **kw)
    return self.run(3.0, prob=0., **kw)

  def test_change_hold_keeps_the_lamp_to_the_end_and_starts_no_second_change(self):
    self.options(change=True)
    self.change(5.0)
    starting = [h for s, h in zip(self.states, self.holds) if s == LCS.laneChangeStarting]
    assert set(starting) == {1}  # the whole change, also after the body controller's 3 flashes (2.4 s)
    assert self.helper.blinker_hold.reason == 'finished' and self.helper.lane_change_state == LCS.off
    # the tail of our lamp: even with torque toward it, never a second change, never the wait again
    self.run(4.0, torque=100.)
    assert self.starts == 1 and self.helper.lane_change_state == LCS.off and not any(self.holds[-80:])
    # dark for 1 s: the side is free again and a new tap starts the next change
    assert not any(self.lamps()) and self.helper.boundary_indicator.blocked == 0
    self.change(1.0)
    assert self.starts == 2

  def test_without_the_mask_the_held_lamp_would_start_a_second_change(self):
    # control for the test above: the lamp tail is read as the driver's request if the release does not own the side
    self.options(change=True)
    self.helper.boundary_indicator.own = lambda *a, **kw: None
    self.change(5.0, torque=100.)
    assert self.starts == 2

  def test_options_off_hold_nothing(self):
    self.change(5.0)
    assert not any(self.holds) and self.helper.boundary_indicator.blocked == 0 and self.starts == 1
    # the body controller's 3 flashes end mid-change: the change still runs, only its lamp stops
    assert LCS.laneChangeStarting in self.states[round(3.0 / DT_MDL):]

  @staticmethod
  def lines(y1, y2):
    return SimpleNamespace(laneLines=[SimpleNamespace(y=[y]) for y in (y1 - 3.6, y1, y2, y2 + 3.6)])

  def left_change_to(self, end):
    """one-touch left, the car moves smoothly into the left lane (3.6 m wide) up to progress `end` -> [progress per frame]"""
    self.tap(1)
    self.step()
    p, seen = 0., []
    while p < end:
      d = p * 3.6
      y1, y2 = (-1.8 + d, 1.8 + d) if d < 1.8 else (d - 5.4, d - 1.8)  # the crossed line is relabelled
      self.step(model=self.lines(y1, y2))
      seen.append(p)
      p += 0.02
    return seen

  def test_change_hold_released_70_percent_into_the_new_lane(self):
    self.options(change=True)
    seen = self.left_change_to(1.0)
    released = next(p for p, h in zip(seen, self.holds[1:]) if h == 0)
    # 0.2 lane widths past the line, ACROSS_FRAMES (3) frames in a row: progress 0.70 .. 0.70 + 2 frames
    assert 0.70 - 1e-6 <= released <= 0.70 + 3 * 0.02, released
    assert self.helper.blinker_hold.reason == 'across'
    assert self.helper.lane_change_state == LCS.laneChangeStarting  # before Finishing
    # the tail of our lamp starts no second change
    self.run(3.0, prob=0., torque=100.)
    assert self.starts == 1 and self.helper.lane_change_state == LCS.off and not any(self.holds[-60:])

  def test_change_hold_not_released_by_the_relabel_frames(self):
    # 11-14 logs (00000468): laneLines[1..2].y[0] while the model relabels the lines; the old rule let go at the marked
    # in-between frame, right on the line
    for side, frames in (
        (2, [(-1.8, 1.8), (-2.5, 1.1), (-3.31, 0.54), (-1.04, 2.28), (0.02, 3.39), (-0.17, 3.09)]),  # seg 18 at 736.69
        (1, [(-1.8, 1.8), (-0.9, 2.7), (-0.24, 3.18), (-0.81, 2.62), (-2.72, 1.09), (-3.32, 0.18)])):  # seg 15 at 570.81
      self.setup_method()
      self.options(change=True)
      self.tap(side)
      self.step()
      for y1, y2 in frames + [frames[-1]] * 10:
        assert self.step(model=self.lines(y1, y2)) == LCS.laneChangeStarting
        assert self.holds[-1] == side, (side, y1, y2)
      assert self.helper.crossing.since >= 0.0  # the crossing itself was seen

  def test_change_hold_released_at_finishing_short_of_70_percent(self):
    self.options(change=True)
    self.left_change_to(0.62)  # across the line, short of 70 %
    assert self.holds[-1] == 1 and self.helper.crossing.since >= 0.0
    while self.step(prob=0., model=self.lines(-5.4 + 0.62 * 3.6, -1.8 + 0.62 * 3.6)) == LCS.laneChangeStarting:
      assert self.holds[-1] == 1
    # released on the very frame Finishing starts
    assert self.helper.lane_change_state == LCS.laneChangeFinishing and self.holds[-1] == 0
    assert self.helper.blinker_hold.reason == 'finished'
    self.run(3.0, prob=0., torque=100.)
    assert self.starts == 1 and self.helper.lane_change_state == LCS.off

  def test_wait_hold_keeps_the_request_until_the_side_clears(self):
    for change in (False, True):
      self.setup_method()
      self.options(change=change, wait=True)
      self.tap(1)
      assert self.run(6.0, available=False) == LCS.preLaneChange  # long after the 3 flashes
      assert set(self.holds[1:]) == {1}
      self.run(0.2)
      assert self.starts == 1
      if change:  # Change takes over from the start
        assert self.holds[-1] == 1 and self.helper.blinker_hold.phase == 'change'
      else:  # stops as the change starts
        assert self.holds[-1] == 0 and self.helper.blinker_hold.reason == 'started'
      self.run(3.0)
      self.run(4.0, prob=0., torque=100.)
      assert self.starts == 1 and self.helper.lane_change_state == LCS.off

  def test_without_wait_hold_the_request_drops_with_the_flashes(self):
    self.tap(1)
    self.run(6.0, available=False)
    assert self.helper.lane_change_state == LCS.off and not any(self.holds)

  def test_wait_hold_times_out_and_does_not_re_arm(self):
    self.options(wait=True)
    self.tap(1)
    self.run(20.0, available=False)
    held = sum(1 for h in self.holds if h) * DT_MDL
    assert abs(held - 15.0) <= 2 * DT_MDL and self.helper.blinker_hold.reason == 'timeout'
    assert self.helper.lane_change_state == LCS.off and self.starts == 0
    self.run(5.0)  # the side clears while the tail of our lamp is still lit: no change from it
    assert self.starts == 0 and not any(self.holds[-100:]) and self.helper.boundary_indicator.blocked == 0

  def test_change_hold_has_a_time_cap(self):
    self.options(change=True)
    self.tap(1)
    self.run(15.0)  # the model never ends the change
    held = sum(1 for h in self.holds if h) * DT_MDL
    assert held <= 10.0 + DT_MDL and self.helper.lane_change_state == LCS.off and self.starts == 1

  def test_every_cancel_ends_the_hold(self):
    cases = {
      'steering against it': (dict(torque=-100.), 'cancel', True),
      'disengaged': (dict(lateral=False), 'cancel', True),
      'out of D': (dict(gear='park'), 'gear', True),
      'lever latched': (dict(lever=2), 'latched', False),
    }
    for phase in ('wait', 'change'):
      for name, (kw, reason, masked) in cases.items():
        self.setup_method()
        self.options(change=True, wait=True)
        self.tap(1)
        self.run(3.0, available=phase == 'change')
        assert self.holds[-1] == 1, (phase, name)
        self.step(**kw)
        assert self.holds[-1] == 0 and self.helper.blinker_hold.reason == reason, (phase, name)
        assert bool(self.helper.boundary_indicator.blocked & 1) == masked, (phase, name)

  def test_driver_lever_while_held(self):
    # the other side tapped while waiting: our hold stops at once and the request turns to that side, as without it
    self.options(change=True, wait=True)
    self.tap(1)
    self.run(3.0, available=False)
    self.tap(2)
    self.step(available=False)
    assert self.holds[-1] == 0 and self.helper.blinker_hold.reason == 'hazard'
    self.run(3.0)
    assert self.starts == 1 and self.helper.lane_change_direction == LCD.right
    # hazard lamps while the change runs stop it too
    self.setup_method()
    self.options(change=True)
    self.tap(1)
    self.run(1.0)
    self.driver[2] = self.t + 1.0
    self.step()
    assert self.holds[-1] == 0 and self.helper.blinker_hold.reason == 'hazard'

  def test_model_and_remote_changes_are_not_held(self):
    self.options(change=True, wait=True)
    self.helper.bluetooth_commands = SimpleNamespace(read=lambda allowed: 'laneLeft' if allowed and self.t < 100.1 else None)
    self.run(4.0)
    assert self.starts == 1 and not any(self.holds)

  def test_settings_are_read_with_the_other_lane_change_keys(self):
    for on in ((), ('CanfdBlinkerHoldChange',), ('CanfdBlinkerHoldWait',), ('CanfdBlinkerHoldChange', 'CanfdBlinkerHoldWait')):
      self.helper.params = SimpleNamespace(get_bool=lambda key, on=on: key in on, get_int=lambda key: 0, get_float=lambda key: 0.)
      self.helper.frame = 100
      DesireHelper._update_params_periodic(self.helper)
      assert (self.helper.hold_change, self.helper.hold_wait) == ('CanfdBlinkerHoldChange' in on, 'CanfdBlinkerHoldWait' in on)

  def test_integrated_with_carcontroller(self):
    """DesireHelper at 20 Hz feeding modelV2.meta (seen by the next 100 Hz frame) into CarController.hold_side; the lamp
    follows what carcontroller sends, with the body controller's run-on. One-touch, 3 s blocked, a 4 s change."""
    from openpilot.cereal import log as cereal_log
    from opendbc.car import structs
    from opendbc.car.hyundai.carcontroller import CarController
    self.options(change=True, wait=True)
    cc = SimpleNamespace(blinker_hold_change=True, blinker_hold_wait=True, hold_refused=False, blinker_pulse_end=0, frame=0)
    md, sent = None, []
    self.tap(1)
    for j in range(int(14.0 / 0.01)):
      if j % 5 == 0:
        k = j // 5
        prob = 0.5 if k < round(7.0 / DT_MDL) else 0.
        self.step(available=k >= round(3.0 / DT_MDL), prob=prob, torque=100. if k > round(7.0 / DT_MDL) else 0., follow=False)
        self.t -= DT_MDL  # the lamp clock below advances at 100 Hz
        new = cereal_log.ModelDataV2.new_message()
        new.meta.laneChangeState = self.helper.lane_change_state
        new.meta.laneChangeDirection = self.helper.lane_change_direction
        new.meta.laneChangeBlinkerHold = self.helper.blinker_hold_direction
        new.meta.indicatorIsolationMask = self.helper.boundary_indicator.blocked
      left, right = self.lamps()
      side = CarController.hold_side(cc, SimpleNamespace(modelV2=md, out=SimpleNamespace(
        gearShifter=structs.CarState.GearShifter.drive, leftBlinker=left, rightBlinker=right)))
      sent.append((j, side, md and str(md.meta.laneChangeState)))
      if side:
        self.ours, self.ours_until = side - 2, self.t + self.TAIL_S
      md = new.as_reader() if j % 5 == 0 else md
      cc.frame += 1
      self.t += 0.01
    held = [(j, state) for j, side, state in sent if side]
    assert held and {side for _, side, _ in sent if side} == {3}
    assert held[0][0] <= 10  # from the first model frame after the tap: no gap after the 3 flashes
    assert {state for _, state in held} <= {'preLaneChange', 'laneChangeStarting', 'laneChangeFinishing'}
    assert self.starts == 1 and self.helper.lane_change_state == LCS.off
    assert not any(self.lamps()) and self.helper.boundary_indicator.blocked == 0
