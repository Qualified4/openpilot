import json
from types import SimpleNamespace
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper


def make_car_state(left_blinker=False):
  return SimpleNamespace(
    canValid=True,
    leftBlinker=left_blinker,
    rightBlinker=False,
    vEgo=20.0,
    aEgo=0.0,
    trailerConnected=False,
    steeringTorque=0.0,
    steeringPressed=False,
  )


class TestDesireHelperDriverIntent:
  def setup_method(self):
    self.helper = DesireHelper()
    self.helper._update_params_periodic = lambda: None
    self.helper._process_sides = lambda car, model, radar: None
    self.helper._check_desire_state = lambda model, car, maneuver: None
    self.helper.laneChangeNeedTorque = 0
    self.helper.laneChangeBsd = 0
    self.helper.laneLineCheck = 0

    side = self.helper.left
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

    self.carrot_man = SimpleNamespace(
      atcType="fork left",
      carrotCmdIndex=0,
      carrotCmd="",
      carrotArg="",
    )

  def update(self, left_blinker=False, lateral_active=True):
    self.helper.update(
      make_car_state(left_blinker),
      SimpleNamespace(),
      lateral_active,
      0.1,
      self.carrot_man,
      SimpleNamespace(),
    )

  def test_driver_blinker_rearms_lane_change_while_atc_desire_is_already_active(self):
    # The first ATC frame is intentionally ignored when its type changes.
    self.update()

    # ATC keeps the combined desire high while lateral control leaves the FSM off.
    self.update(lateral_active=False)
    assert self.helper.prev_desire_enabled
    assert self.helper.lane_change_state == log.LaneChangeState.off

    # A new physical blinker request must be treated as fresh driver intent even
    # though the combined ATC/driver desire has no rising edge.
    self.update(left_blinker=True)
    assert self.helper.lane_change_state == log.LaneChangeState.preLaneChange

    self.update(left_blinker=True)
    assert self.helper.lane_change_state == log.LaneChangeState.laneChangeStarting

  def test_atc_desire_alone_does_not_rearm_lane_change(self):
    self.update()
    self.update(lateral_active=False)

    self.update()

    assert self.helper.lane_change_state == log.LaneChangeState.off

  @pytest.mark.parametrize('block', ['inactive', 'can', 'speed', 'trailer'])
  def test_bluetooth_lane_request_rejected_by_existing_gates(self, block):
    state = make_car_state()
    state.canValid = block != 'can'
    state.vEgo = 0 if block == 'speed' else 20
    state.trailerConnected = block == 'trailer'
    self.carrot_man.atcType = ''
    seen = []
    self.helper.bluetooth_commands = SimpleNamespace(read=lambda allowed: seen.append(allowed))
    self.helper.update(state, SimpleNamespace(), block != 'inactive', .1, self.carrot_man, SimpleNamespace())
    assert seen == [False]
    assert self.helper.lane_change_state == log.LaneChangeState.off

  def test_bluetooth_lane_request_does_not_skip_blindspot(self):
    self.carrot_man.atcType = ''
    self.helper.laneChangeBsd = 1
    self.helper.left.bsd_hold_counter = 10
    self.helper.left.lane_change_available = False
    self.helper.bluetooth_commands = SimpleNamespace(read=lambda allowed: 'laneLeft' if allowed else None)
    self.update()
    self.update()
    assert self.helper.lane_change_state != log.LaneChangeState.laneChangeStarting


class TestYellowSolidLaneChangeBlock:
  """StripNet (CCNC lane type) yellow solid on a side makes that side wait like the camera's yellow line."""

  def setup_method(self):
    self.helper = DesireHelper()
    self.helper.laneLineCheck = 0
    self.helper.laneChangeBsd = 0
    self.helper.laneChangeYellowSolid = True
    for side in (self.helper.left, self.helper.right):
      side.update_lane_geometry = lambda *a, **kw: None
      side.update_lane_line_info = lambda *a, **kw: None
      side.update_obstacles = lambda *a, **kw: None
      side.update_triggers = lambda: None
      side.lane_available = True
      side.edge_available = False
      side.side_object_detected = False
      side.bsd_hold_counter = 0
      side.lane_line_info_raw = 10  # camera: white
    self.model = SimpleNamespace(laneLines=[None] * 4, laneLineProbs=[0.9] * 4, roadEdges=[None] * 2)
    self.car = SimpleNamespace(leftLaneLine=10, rightLaneLine=10, vEgo=20.0, leftBlindspot=False, rightBlindspot=False)
    self.radar = SimpleNamespace(leadLeft=None, leadRight=None, leadsLeft=(), leadsRight=())

  def available(self, left=None, right=None):
    import time
    from openpilot.selfdrive.carrot.lane_type import marking
    packet = {'type': marking.PACKET_TYPE, 'version': marking.PACKET_VERSION, 'mode': 'stripnet', 'frameId': 1}
    for name, value in (('left', left), ('right', right)):
      kind, color = value or (None, None)
      packet[name] = {'type': kind or 'unknown', 'typeAgeNs': 0 if kind else None,
                      'color': color or 'unknown', 'colorAgeNs': 0 if color else None}
    self.helper.lane_marking = marking.parse(json.dumps(packet), time.monotonic_ns())
    self.helper._process_sides(self.car, self.model, self.radar)
    return self.helper.lane_change_available_left, self.helper.lane_change_available_right

  def test_yellow_solid_blocks_only_that_side(self):
    assert self.available(left=('solid', 'yellow')) == (False, True)
    assert self.helper.yellow_solid_left and not self.helper.yellow_solid_right
    assert self.available(right=('solid', 'yellow')) == (True, False)

  @pytest.mark.parametrize('value', (('dashed', 'yellow'), ('solid', 'white'), (None, None), ('solid', None), (None, 'yellow')))
  def test_other_results_do_not_block(self, value):
    assert self.available(left=value, right=value) == (True, True)

  def test_toggle_off_or_no_packet_does_not_block(self):
    self.helper.laneChangeYellowSolid = False
    assert self.available(left=('solid', 'yellow'), right=('solid', 'yellow')) == (True, True)
    self.helper.laneChangeYellowSolid = True
    self.helper.lane_marking = None
    self.helper._process_sides(self.car, self.model, self.radar)
    assert (self.helper.lane_change_available_left, self.helper.lane_change_available_right) == (True, True)

  def test_expired_packet_does_not_block(self):
    from openpilot.selfdrive.carrot.lane_type import marking
    self.available(left=('solid', 'yellow'))
    self.helper.lane_marking = marking.parse(json.dumps({
      'type': marking.PACKET_TYPE, 'version': marking.PACKET_VERSION,
      'left': {'type': 'solid', 'typeAgeNs': 0, 'color': 'yellow', 'colorAgeNs': 0}}), 0)  # received long ago
    self.helper._process_sides(self.car, self.model, self.radar)
    assert self.helper.lane_change_available_left and not self.helper.yellow_solid_left

  def test_camera_yellow_still_blocks(self):
    self.helper.left.lane_line_info_raw = 20
    assert self.available() == (False, True)

  def test_param_needs_lane_type_detection(self):
    from openpilot.selfdrive.carrot.lane_type import marking
    values = {'LaneChangeYellowSolid': True, marking.PARAM: 0}
    self.helper.params = SimpleNamespace(get_int=lambda k: int(values.get(k, 0)), get_float=lambda k: 0.,
                                         get_bool=lambda k: bool(values.get(k, False)))
    self.helper.frame = 0
    self.helper._update_params_periodic()
    assert not self.helper.laneChangeYellowSolid
    values[marking.PARAM] = 1
    self.helper._update_params_periodic()
    assert self.helper.laneChangeYellowSolid
    values['LaneChangeYellowSolid'] = False
    self.helper._update_params_periodic()
    assert not self.helper.laneChangeYellowSolid
