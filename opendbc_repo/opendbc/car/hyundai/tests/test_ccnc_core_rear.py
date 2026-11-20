from collections import Counter, deque
import math
import random
import sys
from statistics import median
from struct import pack
from time import perf_counter_ns
from types import SimpleNamespace

import pytest

from opendbc.car.hyundai import ccnc_core_loader

try:
  import openpilot.common.params  # noqa: F401
except ImportError:
  # The ARM job builds the Rust extension without the Params extension.
  sys.modules['openpilot.common.params'] = SimpleNamespace(Params=None)

from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.hyundai import hyundaicanfd as main
from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ext
from opendbc.car.hyundai import hyundaicanfd_ccnc_rear as rear
from opendbc.car.hyundai import hyundaicanfd_ccnc_radar_map as radar_map
from opendbc.car.hyundai.tests.test_ccnc_extension import display  # noqa: F401
from opendbc.car.hyundai.values import HyundaiExtFlags, HyundaiFlags
from opendbc.car.interfaces import CarInterfaceBase
from openpilot.cereal import log


def new_filters():
  return tuple(ext.NoiseFilter(1, 15, alpha_range=.05) for _ in range(2))


def filter_state(filters):
  return tuple((filter_._filtered_value, filter_._buffer) for filter_ in filters)


def freeze(value, seen=None):
  """Record types, insertion order, exact floats and shared mutable objects."""
  if seen is None:
    seen = {}
  kind = type(value)
  if kind is float:
    return kind, pack('>d', value)
  if kind is tuple:
    return kind, tuple(freeze(item, seen) for item in value)
  if kind in (dict, list, deque) or hasattr(value, '__dict__'):
    if id(value) in seen:
      return kind, 'ref', seen[id(value)]
    seen[id(value)] = len(seen)
    if kind is dict:
      return kind, tuple((freeze(key, seen), freeze(item, seen)) for key, item in value.items())
    if kind in (list, deque):
      return kind, getattr(value, 'maxlen', None), tuple(freeze(item, seen) for item in value)
    # Visit __dict__ directly without tracking its identity as another state reference.
    return kind, tuple((key, freeze(item, seen)) for key, item in vars(value).items())
  return kind, value


def point(source, x, y, speed, ego, moving):
  return SimpleNamespace(trackId=source, dRel=x, yRel=y, vRel=speed, vLead=ego + speed, motionState=2 if moving else 0,
                         radarSource='frontRadar')


def frames(seed):
  rng = random.Random(seed)
  tracks = None
  for i in range(600):
    scene, step = divmod(i, 150)
    primary = (scene + seed) % 2
    context = ('map' if (scene + seed // 2) % 2 == 0 else 'selector', scene)
    frame = i * 5 + rng.randrange(2)
    ego = 10. + rng.uniform(-2., 3.) if step < 105 else rng.uniform(-1., 12.)
    blind = (45 <= step < 56, 67 <= step < 78)
    changing = 93 <= step <= 99
    out = SimpleNamespace(leftBlindspot=blind[0], rightBlindspot=blind[1], leftBlinker=changing and scene % 2 == 0,
                          rightBlinker=False, gearShifter='reverse' if step == 141 else 'drive',
                          canValid=step != 138, vEgo=ego)
    md = SimpleNamespace(laneLineProbs=[1., math.nan if step == 105 else .9, .95, 1.],
                         meta=SimpleNamespace(desire='laneChangeRight' if changing and scene % 2 else 'none', laneChangeState='off'))
    values = {'FF_DISTANCE': 24, 'FF_DETECT': 0}
    selected = [None, None, None]
    points = []
    proven = [True, True]

    for side, fields in enumerate(rear.SmoothHandover.FIELDS):
      fd, fx, fy, rd, rx, ry = fields
      source = 4 * scene + 2 * side + 1
      y = (3.5 if side == 0 else -3.5) + rng.uniform(-.05, .05)
      speed = -2.5 + rng.uniform(-.22, .22)
      x = 3.2 - .125 * step + rng.uniform(-.025, .025)
      if step < 38:
        points.append(point(source, x, y, speed, ego, step < 17 or context[0] == 'selector'))
      distance, detect, lateral = -0.0, 0, 3.
      if step < 22 or (side != primary and step < 33):
        # The map can keep a new epoch for the same raw ID through a continuous report.
        selected[side + 1] = (source, int(step >= 14))
        distance, detect, lateral = max(-0.0, .8 * x), 2, abs(y)
      elif side == primary and step < 84:
        replacement = source + 1
        new_x = (24. if scene % 2 == 0 else 5.8) - .085 * (step - 22)
        points.append(point(replacement, new_x, y, -1.7, ego, True))
        selected[side + 1] = (replacement, 0)
        distance, detect, lateral = max(0., new_x), 3, abs(y) + .3
      elif step == 88:
        # SideFlow can animate a display-only drawing with an integer coordinate.
        distance, detect, lateral = 2, 2, 3
      if 19 <= step < 21 and side != primary:
        proven[side] = False
      if step == 97 and side == primary:
        selected[0] = (source + 1, 0)
      values.update({fd: detect, fx: distance, fy: lateral, rd: 2 if blind[side] else 0,
                     rx: 8 if blind[side] else 16., ry: 2.8})
    for extra in range(seed % 4):
      points.append(point(100 + extra, 18. + extra * 4, (-1) ** extra * 7., .25, ego, True))
    if step == 8 and seed % 2:
      tracks = None
    elif step not in (9, 34):
      tracks = SimpleNamespace(points=points)
    yield (values, SimpleNamespace(out=out, live_tracks=tracks), md, frame, selected, context,
           tuple(proven), (3.0 + rng.uniform(-.25, .25), 2.6 + rng.uniform(-.25, .25)))


def update(policy, filters, row, *, core=None, values=None):
  original, cs, md, frame, selected, context, proven, laterals = row
  if values is None:
    values = original.copy()
  if core is not None:
    core._update_rear_for_test(policy, values, cs, md, frame, selected, filters, context, proven, laterals)
  else:
    for i, active in enumerate((cs.out.leftBlindspot, cs.out.rightBlindspot)):
      f = filters[i]
      if active or f.value < 15:
        _, _, _, detect, distance, lateral = rear.SmoothHandover.FIELDS[i]
        values[distance], values[lateral], values[detect] = f.apply(8 if active else 16), laterals[i], 2
    policy.update(values, cs, md, frame, selected, filters, context, proven=proven, rear_laterals=laterals)
  return values


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('policy', (rear.RearHandover, rear.SmoothHandover))
def test_rear_state_and_display_match(seed, policy):
  python, rust = policy(), policy()
  python_filters, rust_filters = new_filters(), new_filters()
  core = ccnc_core_loader.load()
  assert core is not None
  reached = Counter()
  previous_bsd = (False, False)
  for i, row in enumerate(frames(seed)):
    left = update(python, python_filters, row)
    right = update(rust, rust_filters, row, core=core)
    assert freeze(right) == freeze(left), f'seed {seed}, frame {i}, values'
    assert freeze((rust, filter_state(rust_filters))) == freeze((python, filter_state(python_filters))), f'seed {seed}, frame {i}, rear and flow state'

    reached.update(f'event:{event["event"]}' for event in python.events)
    reached.update(f'reason:{event["reason"]}' for event in python.events if 'reason' in event)
    reached.update(f'phase:{saved["phase"]}' for saved in python.rears if saved is not None)
    if isinstance(python, rear.SmoothHandover):
      reached.update(f'flow:{side.rear["phase"]}' for side in python.flows if side.rear is not None)
    out = row[1].out
    bsd = out.leftBlindspot, out.rightBlindspot
    reached['bsd_on'] += sum(new and not old for new, old in zip(bsd, previous_bsd, strict=True))
    reached['bsd_off'] += sum(old and not new for new, old in zip(bsd, previous_bsd, strict=True))
    previous_bsd = bsd
    reached['lane_change'] += bool(python.changing)
    reached['proven_false'] += not all(row[6])
    reached[row[5][0]] += 1

  for name in ('phase:coast', 'phase:rear', 'reason:object_replaced', 'reason:not_proven', 'event:handoff',
               'bsd_on', 'bsd_off', 'lane_change', 'proven_false', 'map', 'selector'):
    assert reached[name] > 0, (seed, name, reached)
  if policy is rear.SmoothHandover:
    assert reached['flow:virtual'] > 0
  print(policy.__name__, seed, sorted(reached.items()))


def test_missing_lateral_preserves_type_and_partial_state():
  core = ccnc_core_loader.load()
  assert core is not None
  rows = []
  for backend in (None, core):
    policy = rear.RearHandover()
    filters = new_filters()
    source = iter(frames(0))
    for _ in range(3):
      update(policy, filters, next(source), core=backend)
    row = next(source)
    values = row[0].copy()
    del values['LF_DETECT_LATERAL']
    with pytest.raises(KeyError) as error:
      update(policy, filters, row, core=backend, values=values)
    rows.append(freeze((error.value.args, values, policy, filter_state(filters))))
  assert rows[0] == rows[1]


def test_rear_forced_exit_matches():
  core = ccnc_core_loader.load()
  assert core is not None
  traces = []
  for backend in (None, core):
    policy = rear.RearHandover()
    policy.context, policy.last = ('map', 0), 0.
    policy.rears[0] = dict(key=(1, 0), source=1, phase='rear', shown=-6., start=-1., seen=-1., distance=15.8,
                           vrel=-.4, v_anchor=10., lateral=3.1, detect=2)
    filters = new_filters()
    trace = []
    for i, row in enumerate(frames(0)):
      if 1 <= i <= 3:
        row[4][:] = [None, None, None]
        for fd, *_ in rear.SmoothHandover.FIELDS:
          row[0][fd] = 0
        values = update(policy, filters, row, core=backend)
        trace.append(freeze((values, policy, filter_state(filters))))
        if i == 1:
          assert policy.events[0]['event'] == 'forced_exit'
        if i == 3:
          assert policy.rears == [None, None]
          break
    traces.append(trace)
  assert traces[0] == traces[1]


VEHICLE_FIELDS = ('FF_DISTANCE', 'FF_LATERAL', 'FF_DETECT', *(key for side in rear.SmoothHandover.FIELDS for key in side))


def replay_inputs():
  """Fresh, repeated and missing readers through the same message schemas as card."""
  inputs = []
  for tick in range(650):
    step = tick // 5
    scene = step % 65
    if tick % 5 == 0:
      model = log.Event.new_message()
      md = model.init('modelV2')
      md.timestampEof = (tick + 1) * 10_000_000
      md.orientationRate.z = [.035 if 33 <= scene < 38 else 0.]
      md.meta.desire = 'laneChangeLeft' if 40 <= scene < 48 else 'none'
      md.meta.laneChangeState = 'laneChangeFinishing' if 43 <= scene < 48 else 'off'
      md.meta.laneChangeAvailableLeft = md.meta.laneChangeAvailableRight = True
      md.meta.desireState = [0.] * 8
      xs = [0., 5., 15., 30., 60., 100., 170.]
      bend = [.0001 * x**2 for x in xs]
      shift = -.27 * (scene - 40) if 40 <= scene < 48 else 0.
      for line, base in zip(md.init('laneLines', 4), (-5.4, -1.8, 1.8, 5.4), strict=True):
        line.x, line.y = xs, [base - shift + y for y in bend]
      md.laneLineProbs = [.85, .08 if 29 <= scene < 33 else .92, .09 if 30 <= scene < 34 else .88, .87]
      for line, y in zip(md.init('roadEdges', 2), (-7., 7.), strict=True):
        line.x, line.y = xs, [y - shift + b for b in bend]
      md.position.x, md.position.y, md.position.yStd = xs, bend, [0.] * len(xs)
      lead = md.init('leadsV3', 1)[0]
      lead.prob, lead.x, lead.y, lead.v = .93, [24.52], [.2], [12.]
      model_data = model.as_reader().modelV2, model.to_bytes()

      track = log.Event.new_message()
      radar = track.init('liveTracks')
      candidates = [(1, 23., .2, -.2), (2, max(.3, 3.7 - .13 * scene), 3.5, -2.6), (3, 14., -3.4, .2)]
      if 35 <= scene < 42:
        candidates.extend((100 + j, 7. + 1.9 * j, 1.7, -.55) for j in range(5))
      if 18 <= scene < 24:
        candidates[1] = (17, *candidates[1][1:])
      if scene >= 29:
        candidates = [p for p in candidates if p[0] != 2]
      for target, (key, x, y, speed) in zip(radar.init('points', len(candidates)), candidates, strict=True):
        target.trackId, target.dRel, target.yRel, target.vRel, target.vLead = key, x, y, speed, 12. + speed
        target.radarSource, target.motionState, target.trackState, target.measured = 'frontRadar', 2, 2, True
      if scene % 11 != 8:
        radar_data = track.as_reader().liveTracks, track.to_bytes()
    inputs.append((tick, scene, model_data, radar_data))
  return inputs


def test_display_entry_returns_every_slot_in_one_call(display, monkeypatch, caplog):  # noqa: F811
  core = ccnc_core_loader.load()
  assert core is not None
  params, cs, send = display
  mode = 'extended'
  params['CcncExtendedDisplay'] = True
  rows = replay_inputs()
  control = structs.CarControl()
  dims = SimpleNamespace(ECAN=0, CAM=2)
  cp = SimpleNamespace(brand="hyundai", flags=HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC,
                       openpilotLongitudinalControl=True, extFlags=HyundaiExtFlags.CCNC_CLUSTER_RX)
  car_fields = CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162'].sigs

  def replay(native, patch):
    ext.reset()
    radar_map.reset_live()
    radar_map._live.failed = False
    lead_filter = main.DisplayLeadLateralFilter()
    main._stop_go_background.reset()
    for name in ('_display_options', '_radar_map', '_parking', '_camera_sync', '_lane_display'):
      patch.delattr(main.create_ccnc_messages, name, raising=False)

    lanes, draws, calls, times = [], [], [], []
    if native:
      entered = core.update_display

      def counted(cs, md, frame, inputs, *args):
        calls.append((frame, inputs is not None))
        rendered = entered(cs, md, frame, inputs, *args)
        if inputs is not None:
          assert set(VEHICLE_FIELDS) <= set(rendered[1])
        return rendered

      def unexpected(*args, **kwargs):
        pytest.fail('production display called a Python drawing or state method')

      patch.setattr(core, 'update_display', counted)
      for name in ('update_lanes', 'update_speed_limit', 'update_lfa_icon'):
        patch.setattr(ext, name, unexpected)
      for name in ('_display_lead', '_apply_driver_monitoring_alert', '_hide_replaced_adas_service_warning',
                   '_select_cluster_background', '_normalize_cluster_corner_objects', '_apply_cluster_lane_lines'):
        patch.setattr(main, name, unexpected)
      patch.setattr(main.ccnc_lane_marking.SideDisplay, 'update', unexpected)
      patch.setattr(main.ccnc_parking.ParkingLaneHide, 'update', unexpected)
      for owner, names in (
          (rear.SmoothHandover, ('update',)), (rear.SideFlow, ('update',)),
          (radar_map.CcncRadarMap, ('update', 'display_values')),
          (ext._CcncRadarDisplayTracker, ('observe', 'filter_position')),
          (ext._CcncVehiclePositionCorrection, ('update_ego_follow', 'update_centers')),
          (main.DisplayLeadLateralFilter, ('update',)),
      ):
        for name in names:
          patch.setattr(owner, name, unexpected)
    else:
      patch.setattr(main, '_CCNC_DISPLAY_RUST', None)
      patch.setattr(rear, 'RUST', None)
      # Keep the old map observation pipeline. For correction, compare with its original Python policy.
      patch.setattr(radar_map, 'CORRECT_RUST', None)

    class Capture:
      def __init__(self):
        self.packer = CANPacker('hyundai_canfd_generated')

      def make_can_msg(self, name, bus, values, **kwargs):
        if name == 'ADRV_0x161':
          lanes.append(dict(values))
        if name == 'CCNC_0x162':
          draws.append((frame, tuple(values[key] for key in VEHICLE_FIELDS)))
        return self.packer.make_can_msg(name, bus, values, **kwargs)

    packer = Capture()

    def dispatch(c, state, now_nanos):
      return main.create_ccnc_messages(cp, packer, dims, frame, SimpleNamespace(enabled=True, latActive=True),
                                       state, send.hud, 0., False, False, 0, False, 0, 0, lead_filter=lead_filter)

    ci = SimpleNamespace(CS=cs, CC=SimpleNamespace(update=dispatch))
    cs.ccnc_0x162 = dict.fromkeys(car_fields, 0)
    if mode == 'stock':
      for _, x, y, _, rx, ry in rear.SmoothHandover.FIELDS:
        cs.ccnc_0x162[x], cs.ccnc_0x162[y], cs.ccnc_0x162[rx], cs.ccnc_0x162[ry] = 9., 3., 7., 2.8

    traces = []
    for frame, scene, (md, model_bytes), (tracks, radar_bytes) in rows:
      cs.out.vEgo = 0. if 49 <= scene < 58 else 12.
      cs.out.aEgo = -.7 if 46 <= scene < 49 else 0.
      cs.out.leftBlinker, cs.out.rightBlinker = 40 <= scene < 48, False
      cs.out.leftBlindspot, cs.out.rightBlindspot = 21 <= scene < 27, 33 <= scene < 44
      cs.out.canValid = scene != 61
      cs.out.gearShifter = 'reverse' if scene == 62 else structs.CarState.GearShifter.drive
      radar = SimpleNamespace(
        leadOne=SimpleNamespace(status=scene != 58, dRel=18. + scene * .02, yRel=.2 + frame * .001,
                                vRel=-.15, radar=True, radarTrackId=7),
        leadTwo=SimpleNamespace(status=scene % 13 == 9, dRel=15., yRel=-.35, vRel=-.1, radar=True, radarTrackId=12),
      )
      start = perf_counter_ns()
      packed = CarInterfaceBase.apply(ci, control, frame * 10_000_000, md, radar, tracks,
                                      radar_bytes if frame < 400 else None, model_bytes if frame < 400 else None)
      elapsed = perf_counter_ns() - start
      if frame % 5 == 0:
        times.append(elapsed)
      traces.append((packed, pack('>d', lead_filter.value)))
      assert not getattr(ext.state, 'vehicle_display_failed', False)
      assert not getattr(ext.state.rear_handover, 'failed', False)
      assert not radar_map._live.failed
    assert len(draws) == 130
    if native:
      assert calls == [(frame, frame % 5 == 0) for frame in range(len(rows))]
    return traces, lanes, draws, times

  with monkeypatch.context() as patch:
    expected, expected_lanes, values, python_time = replay(False, patch)
  with monkeypatch.context() as patch:
    actual, actual_lanes, rendered, rust_time = replay(True, patch)

  assert len(actual_lanes) == len(expected_lanes) == 130
  for tick, (actual_fields, expected_fields) in enumerate(zip(actual_lanes, expected_lanes, strict=True)):
    assert actual_fields == expected_fields, (tick, {
      field: (actual_fields.get(field), expected_fields.get(field))
      for field in actual_fields.keys() | expected_fields.keys() if actual_fields.get(field) != expected_fields.get(field)
    })
  for frame, (a, b) in enumerate(zip(actual, expected, strict=True)):
    assert a == b, (mode, frame, a, b, caplog.text)
  for (frame, got), (_, want) in zip(rendered, values, strict=True):
    differences = {}
    for key, a, b in zip(VEHICLE_FIELDS, got, want, strict=True):
      # The typed transport returns Python floats. Compare the float64
      # signal bits whether the reference used float or a NumPy scalar.
      equal = a == b if key.endswith('_DETECT') else pack('>d', a) == pack('>d', b)
      if not equal:
        differences[key] = a, b
    assert not differences, (mode, frame, differences)
  for key in ('FF_DETECT', 'LF_DETECT', 'RF_DETECT', 'LR_DETECT', 'RR_DETECT'):
    assert any(row[VEHICLE_FIELDS.index(key)] for _, row in rendered), (mode, key)
  def us(samples):
    return round(median(samples) / 1_000, 3), round(sorted(samples)[int(.95 * len(samples))] / 1_000, 3)
  print(f'{mode}: 130 complete CAN frames, 650 filter steps, median/p95 µs {us(python_time)} → {us(rust_time)}')
