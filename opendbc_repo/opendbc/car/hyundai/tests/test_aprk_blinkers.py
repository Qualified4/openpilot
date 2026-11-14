import json
import time
from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car import Bus, structs
from opendbc.car.hyundai import carcontroller, hyundaicanfd
from opendbc.car.hyundai.carcontroller import CarController
from opendbc.car.hyundai.values import CAR, HyundaiFlags

D, P = structs.CarState.GearShifter.drive, structs.CarState.GearShifter.park
EXT, QUIET, DEFAULT = carcontroller.APRK_EXT_SESSION, carcontroller.APRK_COM_DISABLE, carcontroller.APRK_DEFAULT_SESSION
TICK = 10_000_000  # one controls frame in CAN nanos


class Crossing:
  """LaneCrossing stand-in: reports `side` once, on the first allowed call after it is set."""
  def __init__(self):
    self.side = 0

  def update(self, md, t, allowed, torque=None, torque_min=250):
    side, self.side = (self.side, 0) if allowed else (0, self.side)
    return side


class Ecu:
  """Parking ECU stand-in: own SPAS frames every 20 ms; answers 10 03 one frame later if `answers`, stops its frames
  after 28 83 01 if `obeys`, resumes on 10 01."""
  def __init__(self, answers=True, obeys=True):
    self.answers, self.obeys = answers, obeys
    self.quiet, self.stock, self.ack = False, 0, 0

  def tick(self, now, sent):
    for addr, dat, _ in sent:
      if addr == carcontroller.APRK_ADDR:
        if dat == EXT and self.answers:
          self.ack = now
        elif dat == QUIET:
          self.quiet = self.obeys
        elif dat == DEFAULT:
          self.quiet = False
    if not self.quiet and now % (2 * TICK) == 0:
      self.stock = now


def make_cc(control=True, test=0, ecu=None, mode=1, model=True, hold_change=False, hold_wait=False):
  cc = SimpleNamespace(blinker_control=control, blinker_test=test, blinker_kick_mode=mode, aprk_state=0, aprk_frame=0, aprk_nanos=0, aprk_acked=False,
                       blinker_hold_change=hold_change, blinker_hold_wait=hold_wait, hold_refused=False, bvm_view=0,
                       lane_crossing=Crossing(), blinker_side=0, blinker_pulse_end=0, frame=0, yellow_until=[0, 0],
                       blinker_kick_model=model, lane_change_prev="off", model_pulse=False,
                       CAN=SimpleNamespace(ECAN=0, ACAN=1), packer=CANPacker('hyundai_canfd_generated'), params=SimpleNamespace(STEER_THRESHOLD=250),
                       aprk_buses=(), ecu=ecu or Ecu())
  cc.kick_pulse = lambda CS: CarController.kick_pulse(cc, CS)
  cc.hold_side = lambda CS: CarController.hold_side(cc, CS)
  return cc


def run(cc, frames, out):
  sent = []
  for _ in range(frames):
    now = (cc.frame + 1) * TICK
    CS = out()
    CS.can_nanos, CS.aprk_stock_nanos, CS.aprk_ack_nanos = now, cc.ecu.stock, cc.ecu.ack
    msgs = CarController.aprk_blinkers(cc, None, CS)
    sent += msgs
    cc.frame += 1
    cc.ecu.tick((cc.frame + 1) * TICK, msgs)
  return sent


def step(cc, frames, gear, kph, cross=0, lamp=False, tunnel=0, roundabout=False, left_line=10, right_line=10):
  cc.lane_crossing.side = cross or cc.lane_crossing.side
  return run(cc, frames, lambda: SimpleNamespace(out=SimpleNamespace(
    gearShifter=gear, vEgo=kph / 3.6, leftBlinker=lamp, rightBlinker=False, steeringTorque=0, naviTunnel=bool(tunnel),
    naviRoundabout=roundabout, leftLaneLine=left_line, rightLaneLine=right_line)))


def spas2(cc, sent):
  sig = cc.packer.dbc.name_to_msg['SPAS2'].sigs['BLINKER_CONTROL']
  return [get_raw_value(m[1], sig) for m in sent if m[0] == 0x16A]


def uds(sent):
  return [(m[1], m[2]) for m in sent if m[0] == carcontroller.APRK_ADDR]


def test_aprk_paused_only_around_a_crossing_kick():
  cc = make_cc()
  assert step(cc, 50, D, 60) == [] and cc.aprk_state == carcontroller.APRK_IDLE  # nobody crossing: ECU untouched
  sent = step(cc, 30, D, 5, cross=3)  # any speed in D
  assert uds(sent)[:4] == [(EXT, 0), (EXT, 1), (QUIET, 0), (QUIET, 1)] and cc.aprk_state == carcontroller.APRK_ACTIVE
  assert set(spas2(cc, step(cc, 10, D, 0))) == {3}
  sent = step(cc, carcontroller.BLINKER_PULSE_FRAMES, D, 60)
  assert uds(sent)[-2:] == [(DEFAULT, 0), (DEFAULT, 1)]
  assert cc.aprk_state == carcontroller.APRK_IDLE and step(cc, 100, D, 60) == []  # restored after the kick
  step(cc, 30, D, 60, cross=4)
  assert uds(step(cc, 1, P, 60)) == [(DEFAULT, 0), (DEFAULT, 1)]  # out of D: restored


def test_frames_only_after_the_ecu_answered_and_went_quiet():
  cc = make_cc()
  sent = step(cc, 30, D, 60, cross=3)
  first = next(i for i, m in enumerate(sent) if m[0] == 0x16A)
  assert any(m[0] == carcontroller.APRK_ADDR and m[1] == QUIET for m in sent[:first])
  # 10-30: the stock stream never stopped, every kick collided -> now no frame while it is on the bus
  assert cc.ecu.quiet


def test_no_answer_still_blinks_behind_the_stock_frames():
  cc = make_cc(ecu=Ecu(answers=False))  # 10-24..11-05: no 0x7B9 reply to ~180 requests, stock stream stays on
  sent = step(cc, carcontroller.APRK_REPLY_FRAMES, D, 60, cross=3)
  assert spas2(cc, sent) == [] and uds(sent) == [(EXT, 0), (EXT, 1)]  # the reply window: nothing yet
  sent = step(cc, carcontroller.BLINKER_PULSE_FRAMES, D, 60)
  # a full pulse after the window, next to the stock frames (panda sends each frame right behind a stock one)
  assert set(spas2(cc, sent)) == {3} and len(spas2(cc, sent)) == carcontroller.BLINKER_PULSE_FRAMES - 1  # none on the switch frame
  assert cc.aprk_state == carcontroller.APRK_ACTIVE and not cc.aprk_acked
  assert uds(step(cc, 1, D, 60)) == [(DEFAULT, 0), (DEFAULT, 1)] and cc.aprk_state == carcontroller.APRK_IDLE


def test_kick_mode_picks_the_send_pattern_without_an_ack():
  # 0: as before fcd251a2 - one try, then given back for this kick
  cc = make_cc(ecu=Ecu(answers=False), mode=0)
  sent = step(cc, carcontroller.BLINKER_PULSE_FRAMES, D, 60, cross=3)
  assert spas2(cc, sent) == [] and uds(sent) == [(EXT, 0), (EXT, 1), (DEFAULT, 0), (DEFAULT, 1)]
  assert cc.aprk_state == carcontroller.APRK_FAILED
  # 2: direct at a quarter rate; 3: direct at full rate (panda decides piggyback vs direct from the safety flag)
  for mode, frames in ((2, carcontroller.BLINKER_PULSE_FRAMES // 4), (3, carcontroller.BLINKER_PULSE_FRAMES - 1)):
    cc = make_cc(ecu=Ecu(answers=False), mode=mode)
    step(cc, carcontroller.APRK_REPLY_FRAMES, D, 60, cross=4)
    n = len(spas2(cc, step(cc, carcontroller.BLINKER_PULSE_FRAMES, D, 60)))
    assert abs(n - frames) <= 1 and cc.aprk_state == carcontroller.APRK_ACTIVE
  # an ECU that answered and went quiet is never rate-limited
  cc = make_cc(mode=2)
  step(cc, 30, D, 60, cross=3)
  assert len(spas2(cc, step(cc, 20, D, 60))) == 20


def test_ecu_that_keeps_sending_is_never_overwritten():
  cc = make_cc(ecu=Ecu(obeys=False))  # answers but its frames continue
  sent = step(cc, carcontroller.BLINKER_PULSE_FRAMES, D, 60, cross=4)
  assert spas2(cc, sent) == [] and uds(sent)[-2:] == [(DEFAULT, 0), (DEFAULT, 1)]


def test_stock_frames_coming_back_stop_the_kick_at_once():
  cc = make_cc()
  step(cc, 30, D, 60, cross=4)
  assert cc.aprk_state == carcontroller.APRK_ACTIVE
  cc.ecu.quiet = False  # the ECU resumes on its own
  cc.ecu.stock = (cc.frame + 1) * TICK
  sent = step(cc, 1, D, 60)
  assert spas2(cc, sent) == [] and uds(sent) == [(DEFAULT, 0), (DEFAULT, 1)]


def test_kick_is_short_and_follows_the_crossing_side():
  cc = make_cc()
  kick = spas2(cc, step(cc, carcontroller.BLINKER_PULSE_FRAMES + 30, D, 60, cross=4))
  assert kick and set(kick) == {4}
  assert spas2(cc, step(cc, 300, D, 60)) == []
  assert set(spas2(cc, step(cc, 30, D, 60, cross=3))) == {3}


def test_no_kick_while_the_lamp_is_on_or_setting_off():
  cc = make_cc()
  assert step(cc, 100, D, 60, cross=3, lamp=True) == []  # the driver already signalled
  cc = make_cc(control=False)
  assert step(cc, 100, D, 60, cross=3) == []


def test_menu_test_blinks_at_standstill_in_p():
  cc = make_cc(control=False, test=2)
  sent = step(cc, 30, P, 0)
  assert cc.aprk_state == carcontroller.APRK_ACTIVE and set(spas2(cc, sent)) == {4}
  cc.blinker_test = 0
  assert uds(step(cc, 1, P, 0)) == [(DEFAULT, 0), (DEFAULT, 1)] and cc.aprk_state == carcontroller.APRK_IDLE


def test_no_kick_in_a_tunnel():
  # CarState.naviTunnel (read with getattr; not produced in this fork): the crossing stays pending until out of it
  cc = make_cc()
  assert 3 not in spas2(cc, step(cc, 50, D, 80, cross=3, tunnel=1))
  assert 3 in spas2(cc, step(cc, 50, D, 80, tunnel=0))


def test_no_kick_across_a_yellow_line():
  # 10-30 seg 1: kicked left over a yellow dashed centre line (20); the color drops to 0 while crossing
  cc = make_cc()
  step(cc, 50, D, 30, left_line=20)
  assert 3 not in spas2(cc, step(cc, 50, D, 30, cross=3, left_line=0))   # yellow 0.5 s ago: no kick, crossing consumed
  assert 4 in spas2(cc, step(cc, 50, D, 30, cross=4, left_line=20))      # the white right line still kicks
  step(cc, 300, D, 30, left_line=0)
  assert 3 in spas2(cc, step(cc, 50, D, 30, cross=3, left_line=0))       # yellow over 2 s ago


# CCNC lane marking (card: marking.parse of customReservedRawData2 -> CS.ccnc_lane_marking): the Sonata 2024 has no stock
# 0x2A4/0x362, so leftLaneLine/rightLaneLine stay 0 and only the recogniser's colour can see a yellow line
def ccnc(left=None, right=None, age_ns=0, received_ns=None):
  from openpilot.selfdrive.carrot.lane_type import marking
  def side(value):
    kind, color = value or (None, None)
    return {'type': kind, 'typeAgeNs': age_ns, 'color': color, 'colorAgeNs': age_ns}
  payload = json.dumps({'type': marking.PACKET_TYPE, 'version': marking.PACKET_VERSION, 'mode': 'stripnet', 'frameId': 1,
                        'left': side(left), 'right': side(right)})
  return marking.parse(payload.encode(), time.monotonic_ns() if received_ns is None else received_ns)


def ccnc_step(cc, frames, lane_marking, cross=0):
  cc.lane_crossing.side = cross or cc.lane_crossing.side
  return run(cc, frames, lambda: SimpleNamespace(ccnc_lane_marking=lane_marking, out=SimpleNamespace(
    gearShifter=D, vEgo=20., leftBlinker=False, rightBlinker=False, steeringTorque=0, leftLaneLine=0, rightLaneLine=0)))


YELLOW_LEFT = (('solid', 'yellow'), ('dashed', 'white'))


@pytest.mark.parametrize("yellow", [('solid', 'yellow'), ('dashed', 'yellow')])
def test_no_kick_across_a_ccnc_yellow_line(yellow):
  for side, sides in ((3, dict(left=yellow, right=('dashed', 'white'))), (4, dict(left=('dashed', 'white'), right=yellow))):
    cc = make_cc()
    m = ccnc(**sides)
    assert m[('left', 'right')[side - 3]]['color'][0] == 'yellow'  # the real parsed shape: (value, expires_ns)
    assert side not in spas2(cc, ccnc_step(cc, 50, m, cross=side)) and cc.lane_crossing.side == 0  # consumed, no kick
    assert 7 - side in spas2(cc, ccnc_step(cc, 50, m, cross=7 - side))  # the other (white) side still kicks


def test_ccnc_yellow_holds_two_seconds():
  cc = make_cc()
  ccnc_step(cc, 50, ccnc(*YELLOW_LEFT))
  # the colour drops (packet gone / expired) while the line is crossed: still no kick within YELLOW_HOLD_FRAMES
  assert 3 not in spas2(cc, ccnc_step(cc, 50, None, cross=3))
  ccnc_step(cc, carcontroller.YELLOW_HOLD_FRAMES, None)
  assert 3 in spas2(cc, ccnc_step(cc, 50, None, cross=3))  # yellow over 2 s ago


@pytest.mark.parametrize("lane_marking", [
  None,
  'expired',  # a yellow packet received 1 s ago: past marking.MAX_AGE_NS, as the CCNC display reads it
  'too old',  # a head aged beyond MAX_AGE_NS at publish: parse drops it
  {'mode': 'stripnet', 'frameId': 1, 'left': {'type': None, 'color': None}, 'right': {'type': None, 'color': None}},
])
def test_missing_or_stale_ccnc_marking_is_as_before(lane_marking):
  if lane_marking == 'expired':
    lane_marking = ccnc(*YELLOW_LEFT, received_ns=time.monotonic_ns() - 1_000_000_000)
  elif lane_marking == 'too old':
    lane_marking = ccnc(*YELLOW_LEFT, age_ns=600_000_000)
    assert lane_marking['left']['color'] is None
  cc = make_cc()
  assert 3 in spas2(cc, ccnc_step(cc, 50, lane_marking, cross=3))
  cc = make_cc(control=False)  # setting off: nothing, with or without a marking
  assert ccnc_step(cc, 100, lane_marking, cross=3) == [] and ccnc_step(cc, 100, ccnc(*YELLOW_LEFT), cross=3) == []


def test_ccnc_white_lines_kick():
  for kind in ('solid', 'dashed'):
    cc = make_cc()
    m = ccnc(left=(kind, 'white'), right=(kind, 'white'))
    assert 3 in spas2(cc, ccnc_step(cc, 50, m, cross=3))
    ccnc_step(cc, 300, m)
    assert 4 in spas2(cc, ccnc_step(cc, 50, m, cross=4))


def test_model_lane_change_kick_ignores_a_ccnc_yellow():
  cc = make_cc()
  run(cc, 10, lambda: SimpleNamespace(modelV2=OFF, ccnc_lane_marking=ccnc(*YELLOW_LEFT), out=SimpleNamespace(
    gearShifter=D, vEgo=20., leftBlinker=False, rightBlinker=False, steeringTorque=0)))
  sent = run(cc, 30, lambda: SimpleNamespace(modelV2=LEFT, ccnc_lane_marking=ccnc(*YELLOW_LEFT), out=SimpleNamespace(
    gearShifter=D, vEgo=20., leftBlinker=False, rightBlinker=False, steeringTorque=0)))
  assert set(spas2(cc, sent)) == {3}


def test_no_kick_after_a_roundabout_sign():
  # CarState.naviRoundabout (read with getattr; not produced in this fork): same ban as a tunnel
  cc = make_cc()
  assert 4 not in spas2(cc, step(cc, 50, D, 30, cross=4, roundabout=True))
  assert 4 in spas2(cc, step(cc, 50, D, 30))


def test_missing_lane_line_and_navi_fields_do_not_break_the_kick():
  # a CarState without leftLaneLine/rightLaneLine/naviTunnel/naviRoundabout: no yellow guard, no tunnel ban
  cc = make_cc()
  bare = lambda: SimpleNamespace(out=SimpleNamespace(gearShifter=D, vEgo=20., leftBlinker=False, rightBlinker=False,
                                                     steeringTorque=0))
  assert run(cc, 30, bare) == []
  cc.lane_crossing.side = 3
  assert set(spas2(cc, run(cc, 30, bare))) == {3}


def test_carstate_sees_stock_frames_and_replies_without_touching_can_validity():
  from opendbc.can import CANParser
  from opendbc.car.hyundai.carstate import CarState
  cp, alt = CANParser('hyundai_canfd_generated', [], 0), CANParser('hyundai_canfd_generated', [], 1)
  cs = SimpleNamespace(aprk_watch=False, can_nanos=0, aprk_stock_nanos=0, aprk_ack_nanos=0)
  def feed(t, frames):
    for parser in (cp, alt):
      parser.update([(t, frames)])
    CarState._update_aprk(cs, cp, alt)
  feed(1 * TICK, [])
  # stock SPAS2 with a garbage checksum and jumping counter: still counts as "the ECU is sending"
  feed(2 * TICK, [(0x16A, bytes([1, 2, 7]) + bytes(29), 0)])
  feed(3 * TICK, [(0x16A, bytes([9, 9, 200]) + bytes(29), 0)])
  assert cs.aprk_stock_nanos == 3 * TICK and cs.aprk_ack_nanos == 0 and cp.can_valid
  feed(4 * TICK, [(0x7B9, bytes([0x06, 0x50, 0x03, 0, 0x32, 1, 0xF4, 0]), 1)])  # positive reply on ACAN
  assert cs.aprk_ack_nanos == 4 * TICK
  feed(5 * TICK, [(0x7B9, bytes([0x03, 0x7F, 0x10, 0x22, 0, 0, 0, 0]), 0)])  # a refusal is not an answer
  assert cs.aprk_ack_nanos == 4 * TICK
  # CanfdBvmView copies the car's own 0x400 from ECAN only; nothing before one arrived
  assert cs.svm_status_400 is None
  feed(6 * TICK, [(0x400, bytes.fromhex('0000007004000000'), 1)])
  assert cs.svm_status_400 is None
  feed(7 * TICK, [(0x400, bytes.fromhex('0000007000000000'), 0)])
  assert cs.svm_status_400 == bytes.fromhex('0000007000000000') and cp.can_valid


# CanfdBlinkerKickModel: a model lane change (remote, app, voice, navigation) kicks on its laneChangeStarting edge, only
# with DesireHelper's lamp ownership (indicatorIsolationMask, 1 left 2 right) in the same modelV2
def lc(state="off", direction="none", mask=0):
  return SimpleNamespace(meta=SimpleNamespace(laneChangeState=state, laneChangeDirection=direction, indicatorIsolationMask=mask))


OFF, LEFT, RIGHT = lc(), lc("laneChangeStarting", "left", 1), lc("laneChangeStarting", "right", 2)


def model_step(cc, frames, md, gear=D, lamp=False, cancel=False, tunnel=False, cross=0, left_line=10):
  cc.lane_crossing.side = cross or cc.lane_crossing.side
  return run(cc, frames, lambda: SimpleNamespace(modelV2=md, out=SimpleNamespace(
    gearShifter=gear, vEgo=20., leftBlinker=lamp, rightBlinker=False, steeringTorque=0, naviTunnel=tunnel,
    naviRoundabout=False, leftLaneLine=left_line, rightLaneLine=10, cruiseCancelState=cancel)))


def test_model_lane_change_start_kicks_with_the_mask():
  for md, side in ((LEFT, 3), (RIGHT, 4)):
    cc = make_cc()
    assert model_step(cc, 50, OFF) == []
    # the model decided: no yellow-line guard (that one is for the driver's own crossing)
    sent = model_step(cc, 30, md, left_line=20)
    assert set(spas2(cc, sent)) == {side} and cc.aprk_state == carcontroller.APRK_ACTIVE
    # one 1 s pulse for the edge, however long Starting lasts
    sent = model_step(cc, carcontroller.BLINKER_PULSE_FRAMES + 100, md)
    assert uds(sent)[-2:] == [(DEFAULT, 0), (DEFAULT, 1)] and cc.aprk_state == carcontroller.APRK_IDLE
    assert model_step(cc, 100, md) == []


def test_no_model_kick_without_ownership_or_outside_the_rules():
  cases = {
    "mask of the other side": dict(md=lc("laneChangeStarting", "left", 2)),
    "no mask": dict(md=lc("laneChangeStarting", "right", 0)),
    "no direction": dict(md=lc("laneChangeStarting", "none", 3)),
    "lamp on": dict(lamp=True),
    "cruise CANCEL latch": dict(cancel=True),
    "in P": dict(gear=P),
    "tunnel": dict(tunnel=True),
    "KickModel 0": dict(model=False),
    "Kick 0": dict(control=False),
  }
  for name, case in cases.items():
    cc = make_cc(control=case.pop("control", True), model=case.pop("model", True))
    model_step(cc, 10, OFF)
    assert model_step(cc, 50, case.pop("md", LEFT), **case) == [] and cc.blinker_pulse_end == 0, name


def test_model_kick_only_on_the_edge():
  cc = make_cc()
  assert set(spas2(cc, model_step(cc, 30, LEFT))) == {3} and cc.model_pulse  # Starting when first seen: one kick
  model_step(cc, carcontroller.BLINKER_PULSE_FRAMES + 50, LEFT)
  assert model_step(cc, 50, LEFT) == []  # repeated Starting frames: no second kick
  # modelV2 None (not alive / invalid) is no transition: Starting -> None -> Starting does not kick again
  assert model_step(cc, 20, None) == [] and cc.lane_change_prev == "laneChangeStarting"
  assert model_step(cc, 50, LEFT) == []
  # off -> None -> Starting still kicks once
  cc = make_cc()
  model_step(cc, 10, OFF)
  model_step(cc, 10, None)
  assert set(spas2(cc, model_step(cc, 30, LEFT))) == {3}


def test_model_kick_cut_when_the_change_is_cancelled():
  cc = make_cc()
  model_step(cc, 10, OFF)
  assert set(spas2(cc, model_step(cc, 30, LEFT))) == {3}
  sent = model_step(cc, 1, OFF)  # cancelled: the lamp stops with it
  assert spas2(cc, sent) == [] and uds(sent) == [(DEFAULT, 0), (DEFAULT, 1)]
  assert model_step(cc, 50, OFF) == []
  # a cut never ends the driver's own crossing kick
  cc = make_cc()
  model_step(cc, 10, OFF, cross=4)
  assert not cc.model_pulse and set(spas2(cc, model_step(cc, 30, OFF))) == {4}


def test_crossing_kick_waits_out_a_model_lane_change():
  # no ownership for the driver-crossing kick: it must not light up a change DesireHelper is running
  cc = make_cc(model=False)
  for state in ("laneChangeStarting", "laneChangeFinishing"):
    assert model_step(cc, 50, lc(state, "left", 0), cross=3) == []
  assert cc.lane_crossing.side == 3  # still pending in the stand-in
  assert set(spas2(cc, model_step(cc, 30, lc("preLaneChange", "left")))) == {3}
  cc = make_cc()
  assert set(spas2(cc, model_step(cc, 30, None, cross=4))) == {4}  # no modelV2: as before


# Real CarController.update on the Sonata 2024 CAN-FD camera-SCC setup: with CanfdBlinkerKick, CanfdBlinkerKickModel,
# CanfdBlinkerMenuTest, CanfdBlinkerHoldChange/Wait and CanfdBvmView at their defaults no parking-ECU UDS, SPAS or 0x400
# frame is ever produced - even with a stock 0x400 seen and DesireHelper asking for a hold.
APRK_IDS = {carcontroller.APRK_ADDR, 0x165, 0x16A}
BLINKER_IDS = APRK_IDS | {0x400}
SONATA_400 = bytes.fromhex('0000007000000000')  # 11-12 logs: byte 3 is always 0x70


def sonata_update(monkeypatch, values, frames, md=None):
  params = SimpleNamespace(get_int=lambda key: values.get(key, 0), get_bool=lambda key: bool(values.get(key, 0)),
                           get_float=lambda key: 0.)
  monkeypatch.setattr(carcontroller, "Params", lambda: params)
  for name in ("create_steering_messages_camera_scc", "create_steering_messages", "create_lfa_icon_non_camera_scc"):
    monkeypatch.setattr(hyundaicanfd, name, lambda *a, **kw: [])
  monkeypatch.setattr(carcontroller.CarController, "create_button_messages", lambda *a, **kw: [])
  cp = structs.CarParams(carFingerprint=CAR.HYUNDAI_SONATA_2024, flags=int(HyundaiFlags.CANFD), wheelbase=2.9,
                         steerRatio=14.0,
                         safetyConfigs=[structs.CarParams.SafetyConfig(safetyModel=structs.CarParams.SafetyModel.hyundaiCanfd)])
  controller = carcontroller.CarController({Bus.pt: "hyundai_canfd_generated"}, cp)
  controller.lane_crossing = Crossing()
  controller.lane_crossing.side = 3  # a crossing pending the whole time: still nothing without the setting
  cs = SimpleNamespace(lfahda_cluster=None, lfa=None, mdps=None, scc_control=None, modelV2=md, is_metric=True,
                       adrv_0x161=None, svm_status_400=SONATA_400)
  cs.out = structs.CarState(canValid=True, vEgoRaw=15, vEgo=15, gearShifter=D)
  cc = structs.CarControl(enabled=True, latActive=True, longActive=False).as_reader()
  sent = []
  for frame in range(frames):
    sent += controller.update(cc, cs, (frame + 1) * TICK)[1]
  return sent


def test_defaults_send_no_aprk_or_spas_frames(monkeypatch):
  for md in (None, model_hold("laneChangeStarting", "left"), model_hold("preLaneChange", "right")):
    sent = sonata_update(monkeypatch, {"CanfdBlinkerKickMode": 1}, 300, md)
    assert not [m for m in sent if m[0] in BLINKER_IDS]


@pytest.mark.parametrize("values,phase,want", [
  ({"CanfdBlinkerHoldChange": 1}, "change", carcontroller.APRK_ADDR),
  ({"CanfdBlinkerHoldWait": 1}, "wait", carcontroller.APRK_ADDR),
  ({"CanfdBvmView": 2}, None, 0x400),
])
def test_the_new_settings_reach_the_bus_in_the_same_setup(monkeypatch, values, phase, want):
  # control for the test above
  md = {"change": model_hold("laneChangeStarting", "left"), "wait": model_hold("preLaneChange", "right"), None: None}[phase]
  assert any(m[0] == want for m in sonata_update(monkeypatch, values, 300, md))


@pytest.mark.parametrize("values", [{"CanfdBlinkerKick": 1, "CanfdBlinkerKickMode": 1}, {"CanfdBlinkerMenuTest": 2}])
def test_the_same_setup_sends_them_when_enabled(monkeypatch, values):
  # control for the test above: the harness reaches aprk_blinkers once a setting is on
  sent = sonata_update(monkeypatch, values, 300)
  assert any(m[0] == carcontroller.APRK_ADDR for m in sent)


# CanfdBvmView: blind-spot camera on the cluster without the blinker - the stock 0x400 copied, only byte 4 changed
def bvm_cc(view, stock=SONATA_400):
  cc = make_cc(control=False)
  cc.bvm_view = view
  return cc, SimpleNamespace(svm_status_400=stock)


def test_bvm_view_copies_the_stock_frame_and_changes_byte4_only():
  for view, byte4 in ((1, 3), (2, 6)):
    cc, cs = bvm_cc(view)
    sent = []
    for cc.frame in range(100):
      sent += CarController.bvm_view_msg(cc, cs)
    assert [m[1] for m in sent] == [bytes([0, 0, 0, 0x70, byte4, 0, 0, 0])] * 5  # 5 Hz, never faster than the car
    assert {m[2] for m in sent} == {0}  # ECAN
    cc, cs = bvm_cc(view, bytes(8))  # Ioniq 5 PE shape (byte 3 = 0) is copied as is
    assert CarController.bvm_view_msg(cc, cs)[0][1] == bytes([0, 0, 0, 0, byte4, 0, 0, 0])


def test_bvm_view_needs_a_stock_frame_and_the_setting():
  for view, stock in ((1, None), (2, bytes(4)), (0, SONATA_400), (3, SONATA_400), (4, SONATA_400)):
    cc, cs = bvm_cc(view, stock)
    assert all(CarController.bvm_view_msg(cc, cs) == [] for cc.frame in range(100)), (view, stock)
  for bad in ('0000007100000000', '0100007000000000', '0000007000000001'):  # a shape panda would refuse: not sent
    cc, cs = bvm_cc(1, bytes.fromhex(bad))
    assert CarController.bvm_view_msg(cc, cs) == [], bad


# CanfdBlinkerHoldChange / CanfdBlinkerHoldWait: carcontroller follows DesireHelper's laneChangeBlinkerHold in the phase
# its setting covers, through the same APRK/SPAS2 path as the kick
def hold_md(state, hold, direction=None):
  return SimpleNamespace(meta=SimpleNamespace(laneChangeState=state, laneChangeDirection=direction or hold, indicatorIsolationMask=0,
                                              laneChangeBlinkerHold=hold))


def model_hold(state, hold):
  # a real modelV2 for CarController.update, which reads more of it than the hold
  from openpilot.cereal import log
  md = log.ModelDataV2.new_message()
  md.meta.laneChangeState, md.meta.laneChangeDirection, md.meta.laneChangeBlinkerHold = state, hold, hold
  return md.as_reader()


def hold_step(cc, frames, md, gear=D, left=False, right=False):
  return run(cc, frames, lambda: SimpleNamespace(modelV2=md, out=SimpleNamespace(
    gearShifter=gear, vEgo=20., leftBlinker=left, rightBlinker=right, steeringTorque=0, naviTunnel=False,
    naviRoundabout=False, leftLaneLine=10, rightLaneLine=10, cruiseCancelState=False)))


def test_hold_follows_desire_helper_without_the_kick():
  for hold, side in (("left", 3), ("right", 4)):
    cc = make_cc(control=False, hold_change=True)
    assert hold_step(cc, 50, hold_md("laneChangeStarting", "none")) == []
    sent = hold_step(cc, 100, hold_md("laneChangeStarting", hold), left=hold == "left", right=hold == "right")
    assert set(spas2(cc, sent)) == {side} and cc.aprk_state == carcontroller.APRK_ACTIVE
    assert set(spas2(cc, hold_step(cc, 30, hold_md("laneChangeFinishing", hold)))) == {side}
    sent = hold_step(cc, 1, hold_md("laneChangeFinishing", "none"))  # DesireHelper released: restored at once
    assert spas2(cc, sent) == [] and uds(sent) == [(DEFAULT, 0), (DEFAULT, 1)] and cc.aprk_state == carcontroller.APRK_IDLE


def test_hold_only_in_the_phase_of_its_setting():
  cases = (
    (dict(hold_change=True), "preLaneChange", False),
    (dict(hold_change=True), "laneChangeStarting", True),
    (dict(hold_wait=True), "preLaneChange", True),
    (dict(hold_wait=True), "laneChangeStarting", False),
    (dict(hold_wait=True), "laneChangeFinishing", False),
    (dict(hold_change=True, hold_wait=True), "preLaneChange", True),
    (dict(), "laneChangeStarting", False),
  )
  for opts, state, want in cases:
    cc = make_cc(control=False, **opts)
    assert bool(spas2(cc, hold_step(cc, 60, hold_md(state, "left")))) == want, (opts, state)


def test_hold_stops_out_of_d_with_hazards_or_without_a_model():
  for kw in (dict(gear=P), dict(left=True, right=True)):
    cc = make_cc(control=False, hold_change=True)
    hold_step(cc, 60, hold_md("laneChangeStarting", "left"))
    sent = hold_step(cc, 1, hold_md("laneChangeStarting", "left"), **kw)
    assert spas2(cc, sent) == [] and uds(sent) == [(DEFAULT, 0), (DEFAULT, 1)], kw
  cc = make_cc(control=False, hold_change=True)
  assert hold_step(cc, 60, None) == []


def test_no_hold_right_after_a_kick():
  # a kick lamp reads as the driver's blinker: DesireHelper may ask to hold it, but that lamp is ours
  cc = make_cc(hold_wait=True, model=False)
  assert set(spas2(cc, step(cc, 30, D, 60, cross=3))) == {3}
  hold_step(cc, carcontroller.HOLD_AFTER_KICK_FRAMES - 100, hold_md("preLaneChange", "left"), left=True)
  sent = hold_step(cc, 100, hold_md("preLaneChange", "left"), left=True)  # the kick is over, the request still there
  assert spas2(cc, sent) == [] and cc.hold_refused
  hold_step(cc, 120, hold_md("preLaneChange", "none"))  # refused until the request clears and the kick is 3 s old ...
  assert not cc.hold_refused
  assert set(spas2(cc, hold_step(cc, 60, hold_md("preLaneChange", "left"), left=True))) == {3}  # ... then held again


def test_a_new_modelv2_is_followed_on_its_own_frame():
  # the lane change fields of a modelV2 reader are read once (card hands the same reader until the next one): a new
  # reader must still count on the very frame it arrives, for the hold and for the model kick
  cc = make_cc(control=False, hold_change=True)
  off, held = model_hold("laneChangeStarting", "none"), model_hold("laneChangeStarting", "left")
  for md, side in ((off, 0), (held, 3), (held, 3), (off, 0), (held, 3)):
    assert CarController.hold_side(cc, SimpleNamespace(modelV2=md, out=SimpleNamespace(
      gearShifter=D, leftBlinker=False, rightBlinker=False))) == side
  cc = make_cc()
  model_step(cc, 10, model_hold("off", "none"))
  starting = model_hold("laneChangeStarting", "none")
  starting = starting.as_builder()
  starting.meta.laneChangeDirection, starting.meta.indicatorIsolationMask = "left", 1
  starting = starting.as_reader()
  cs = SimpleNamespace(modelV2=starting, out=SimpleNamespace(gearShifter=D, vEgo=20., leftBlinker=False, rightBlinker=False,
                                                              steeringTorque=0, cruiseCancelState=False))
  assert CarController.kick_pulse(cc, cs) and cc.blinker_side == 3 and cc.model_pulse  # the edge, on the first frame


def test_a_capnp_carstate_without_navi_fields_reads_them_as_false():
  # this fork's CarState has no naviTunnel / naviRoundabout: a real CarState kicks as a stand-in without them does
  assert "naviTunnel" not in carcontroller._CAR_STATE_FIELDS and "leftLaneLine" in carcontroller._CAR_STATE_FIELDS
  out = structs.CarState(gearShifter=D, leftLaneLine=20)
  assert carcontroller._out_flag(out, "naviTunnel") is False and carcontroller._out_flag(out, "naviRoundabout") is False
  assert carcontroller._out_flag(SimpleNamespace(naviTunnel=True), "naviTunnel") is True
  cc = make_cc()
  cc.lane_crossing.side = 4
  sent = run(cc, 30, lambda: SimpleNamespace(out=structs.CarState(gearShifter=D, vEgo=20.)))
  assert set(spas2(cc, sent)) == {4}


def test_hold_ending_leaves_a_menu_test_running():
  cc = make_cc(control=False, test=2, hold_wait=True)
  hold_step(cc, 60, hold_md("preLaneChange", "left"))
  sent = hold_step(cc, 20, hold_md("preLaneChange", "none"))
  assert set(spas2(cc, sent)) == {4} and uds(sent) == []
