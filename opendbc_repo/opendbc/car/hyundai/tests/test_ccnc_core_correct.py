"""The Rust position correction of the radar map display (_correct_positions) gives the same bits as the Python
reference, through update_vehicles: every frame's 0x162 values, the display memory in _live (filters, aligned,
handover, ego_blend) and the whole state of the extension's _CcncVehiclePositionCorrection, filters' inner values
included. The radar map's own update runs on its Rust path in both runs; only the correction differs."""
from collections import Counter, deque
import math
import struct
import random
from types import MethodType, SimpleNamespace as N

import pytest

from opendbc.can import CANPacker
from opendbc.car.hyundai import ccnc_core_loader
from opendbc.car.hyundai import hyundaicanfd as main
from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ext
from opendbc.car.hyundai import hyundaicanfd_ccnc_radar_map as radar_map
from opendbc.car.hyundai.tests import test_ccnc_core_radar_map as scenes
from openpilot.cereal import log

SEEDS = 8
FRAMES = 600
SIGNALS = tuple(CANPacker('hyundai_canfd_generated').dbc.name_to_msg['CCNC_0x162'].sigs)


def bits(value):
  """Type-strict and bit-exact form of a value (NaN equal to NaN; 0.0 and -0.0 differ)."""
  if isinstance(value, float):
    return ('f', 'nan') if math.isnan(value) else ('f', struct.pack('<d', value))
  if isinstance(value, (bool, int, str, type(None))):
    return type(value).__name__, value
  if isinstance(value, dict):
    return 'd', tuple((bits(k), bits(v)) for k, v in value.items())
  if isinstance(value, (tuple, list, deque)):
    return type(value).__name__, getattr(value, 'maxlen', None), tuple(bits(v) for v in value)
  if isinstance(value, MethodType):  # A filter's bound apply: which method.
    return 'm', value.__func__.__qualname__
  if hasattr(value, '__dict__'):
    return 'o', type(value).__name__, bits(vars(value))
  raise TypeError(type(value))


def model(rng, frame, scenario):
  """The radar map test's model, with the fields the correction reads besides: the planned path, a stamp."""
  desire, change_state = scenario.meta(frame)
  md = scenes.model(rng, frame, desire=desire, state=change_state, lead=scenario.vision_lead, bend=scenario.bend,
                    shift=scenario.shift(frame), weak=scenario.weak_lines and 420 <= frame < 500)
  xs = [0., 10., 20., 40., 80., 120.]
  md.position = N(x=xs, y=[-(.02 * scenario.path_side(frame) * x) for x in xs])
  md.timestampEof = 1000 + frame
  if scenario.uncertain(frame):  # The selected lines below .5: the FF path reference.
    md.laneLineProbs = [p * .45 for p in md.laneLineProbs]
  return md


class Scenario(scenes.Scenario):
  def __init__(self, seed):
    super().__init__(seed)
    self.bend = .001 if seed % 2 else 0.
    self.vision_lead = None

  def path_side(self, frame):
    return 1. if 300 <= frame < 340 else 0.

  def uncertain(self, frame):
    return self.seed % 2 == 0 and 305 <= frame < 330

  def frame(self, frame, v_ego):
    points = super().frame(frame, v_ego)
    if 240 <= frame < 380:  # A car 8m ahead moving from ego's lane to the left one during ego's change (handover).
      y = 3.6 * min(max((frame - 310) / 15, 0.), 1.)
      points.append(dict(trackId=990, dRel=8., yRel=y, vRel=0., vLead=v_ego, measured=True, radarSource='frontRadar',
                         motionState=2, trackState=2))
    return points


def control_frame(frame):
  """The 10 ms control frame of a send: every 5th, with gaps. A 20-frame gap every 97 sends is past the 15 frames a
  filter keeps (the same object gets new filters); a 35-frame gap every 151 sends (away from ego's lane change) is past the .25 s the map bridges,
  so the map resets and its keys start again with new births (a remembered key with another birth)."""
  return 100000 + 5 * frame + 15 * (frame // 97) + 30 * ((frame + 75) // 151)


class Counted:
  """The Rust module, counting the correction calls that reach it."""

  def __init__(self, core):
    self.core, self.calls = core, 0

  def correct_positions(self, *args):
    self.calls += 1
    return self.core.correct_positions(*args)


def drive(seed, rust, frames=FRAMES, check=None):
  """Run update_lanes' correction part and update_vehicles on one seed from a fresh state; check(frame, bits) each
  frame. Returns reach counts (the alignment ones from the Python path only)."""
  main._stop_go_background.reset()
  ext.reset()
  ext.configure(False, True, True, True, model_lane_mode=2)
  radar_map.reset_live()
  radar_map._live.failed = False
  scenario = Scenario(seed)
  structures = scenes.Structures(seed)
  rng = random.Random(seed + 1000)
  counts = Counter()
  saved = radar_map.CORRECT_RUST, radar_map._road_alignment, radar_map._model_alignment, radar_map._correct_positions
  core = Counted(saved[0]) if rust else None
  radar_map.CORRECT_RUST = core
  if not rust:  # Counting wrappers: they also keep the Python path (replaced functions).
    def correct(*args):
      counts['corrections'] += 1
      return saved[3](*args)
    radar_map._correct_positions = correct

    def road(tracker, slot, x, y):
      out = saved[1](tracker, slot, x, y)
      counts['bounds_none'] += out[2] is None
      counts['bounds'] += out[2] is not None
      return out

    def model_alignment(*args):
      out = saved[2](*args)
      counts['model_target'] += out is not None
      counts['path_target'] += out is not None and args[1] == 0 and args[4] > 0.
      return out
    radar_map._road_alignment, radar_map._model_alignment = road, model_alignment
  previous_filters, previous_blend = {}, 0.
  try:
    for frame in range(frames):
      control = control_frame(frame)
      v_ego = scenes.speed_profile(scenario.kind, frame)
      points = scenario.frame(frame, v_ego)
      tracker = radar_map._live.tracker or radar_map.CcncRadarMap()
      posts = structures.points(tracker.pose) + scenes.approaching(seed, tracker.pose)
      live = scenes.namespace_live(points + posts)
      scenario.vision_lead = scenario.vision or next(((p['dRel'], p['yRel'], p['vLead']) for p in points
                                                      if abs(p['yRel']) < 1.5 and 0. < p['dRel'] < 80.
                                                      and all(math.isfinite(p[k]) for k in ('dRel', 'yRel', 'vLead'))), None)
      md = model(rng, frame, scenario)
      desire = {'laneChangeLeft': 3, 'laneChangeRight': 4}.get(md.meta.desire, 0)
      out = N(vEgo=v_ego, aEgo=0., gearShifter='drive', leftBlinker=300 <= frame < 330, rightBlinker=False,
              leftBlindspot=False, rightBlindspot=False)
      cs = N(out=out, live_tracks=live, radarState=None)
      values = dict.fromkeys(SIGNALS, 0)
      ext.update_lanes(values, cs, md, v_ego * 3.6, 0., desire, True, False, False, control)
      values = radar_map.update_vehicles(values, cs, md, control, v_ego * 3.6, 0., True, ext)
      assert not radar_map._live.failed, (seed, frame)
      correction = ext.state.radar_display_tracker.position_correction
      live_state = radar_map._live
      tracker = live_state.tracker
      if check is not None:
        check(frame, bits((values, live_state.filters, live_state.aligned, live_state.handover, live_state.ego_blend,
                           vars(correction))))
      blend = live_state.ego_blend[1]
      counts['blend_in'] += previous_blend == 0. < blend
      counts['blend_out'] += previous_blend > 0. == blend
      counts['blend'] += blend > 0.
      counts['handover'] += sum(carried[2] == control for carried in live_state.handover.values())
      for key, (born, _, far, _) in live_state.filters.items():
        old = previous_filters.get(key)
        if old is not None and old[2] is not far:
          counts['new_filter_born' if old[0] != born else 'new_filter_gap'] += 1
      counts['remembered'] += sum(bool(r) and values[f[2]] != 0 for r, f in zip(tracker.remembered, tracker.SLOT_FIELDS, strict=True))
      counts['holding'] += tracker.holding and any(values[f[2]] for f in tracker.SLOT_FIELDS)
      counts['drawn'] += sum(bool(values[f[2]]) for f in tracker.SLOT_FIELDS)
      counts['ego_follow'] += correction.ego_follow_active
      if frame % 50 == 25:
        # Another birth under a kept key: the map never draws a new object under an old key within 15 frames
        # (keys are unique per map; filters are kept for drawn keys only), so the stored birth is changed instead.
        live_state.filters = {key: (born + 1., *rest) for key, (born, *rest) in live_state.filters.items()}
      previous_filters, previous_blend = dict(live_state.filters), blend
  finally:
    radar_map.CORRECT_RUST, radar_map._road_alignment, radar_map._model_alignment, radar_map._correct_positions = saved
  counts['rust_calls'] = core.calls if rust else 0
  return counts


@pytest.fixture(scope='module', autouse=True)
def rust_core():
  core = ccnc_core_loader.load()
  assert core is not None and core.abi_version() == 7
  assert radar_map.CORRECT_RUST is core
  return core


TOTALS = {}
SHARED = ('blend_in', 'blend_out', 'blend', 'handover', 'new_filter_born', 'new_filter_gap', 'remembered', 'holding',
          'drawn', 'ego_follow')


@pytest.mark.parametrize('seed', range(SEEDS))
def test_rust_and_python_corrections_give_the_same_bits(seed):
  python = []
  TOTALS[seed] = drive(seed, rust=False, check=lambda frame, record: python.append(record))
  compared = []

  def check(frame, record):
    assert record == python[frame], (seed, frame)
    compared.append(frame)

  rust = drive(seed, rust=True, check=check)
  assert len(compared) == FRAMES and all(rust[k] == TOTALS[seed][k] for k in SHARED), (rust, TOTALS[seed])
  # Every send that drew a vehicle ran the Rust correction (none fell back to the Python one).
  assert rust['rust_calls'] == TOTALS[seed]['corrections'] > 0 and TOTALS[seed]['rust_calls'] == 0
  TOTALS[seed]['rust_calls'] = rust['rust_calls']


def test_scenarios_reach_the_corner_cases():
  for seed in range(SEEDS):
    if seed not in TOTALS:  # Run by another pytest-xdist worker.
      TOTALS[seed] = drive(seed, rust=False)
  total = sum(TOTALS.values(), Counter())
  print(dict(total))
  for name in ('blend_in', 'blend_out', 'handover', 'new_filter_born', 'new_filter_gap', 'remembered', 'holding',
               'bounds_none', 'bounds', 'model_target', 'path_target'):
    assert total[name] > 0, name


def test_replaced_alignment_keeps_the_python_correction(monkeypatch):
  assert radar_map._rust_correctable()
  monkeypatch.setattr(radar_map, '_road_alignment', lambda *args: radar_map._CORRECT_ORIGINALS[0](*args))
  assert not radar_map._rust_correctable()


def test_rust_error_returns_to_the_existing_selector(monkeypatch):
  # A Rust failure (here a filter class that cannot be made) takes update_vehicles' fallback, as a Python one does.
  calls = []
  monkeypatch.setattr(ext, 'update_vehicles', lambda values, *args: calls.append(1) or values)
  monkeypatch.setattr(radar_map.carlog, 'exception', lambda *args: None)
  for rust in (False, True):
    monkeypatch.setattr(ext, '_CcncRadarPositionFilter', None)
    with pytest.raises(AssertionError):
      drive(1, rust=rust, frames=200)
    assert radar_map._live.tracker is None and radar_map._live.failed
    monkeypatch.undo()
    monkeypatch.setattr(ext, 'update_vehicles', lambda values, *args: calls.append(1) or values)
    monkeypatch.setattr(radar_map.carlog, 'exception', lambda *args: None)
  assert len(calls) == 2


def test_model_inputs_read_capnp_like_the_namespace_models():
  md = log.ModelDataV2.new_message()
  for i, line in enumerate(md.init('laneLines', 4)):
    line.x, line.y = [0., 10., 20.], [1.5 * i, 1.5 * i + .25, 1.5 * i + 1.]
  md.laneLineProbs = [.25, .5, .75, 1.]
  md.position.x, md.position.y = [0., 5.], [0., .125]
  namespace = N(laneLines=[N(x=list(line.x), y=list(line.y)) for line in md.laneLines], laneLineProbs=list(md.laneLineProbs),
                position=N(x=list(md.position.x), y=list(md.position.y)))
  assert radar_map._model_inputs(md.as_reader()) == radar_map._model_inputs(namespace) == (
    [([0., 10., 20.], [1.5, 1.75, 2.5]), ([0., 10., 20.], [3., 3.25, 4.])], ([0., 5.], [0., .125]), [.5, .75])
  assert radar_map._model_inputs(N(laneLines=[N(x=[0.], y=[0.])])) == ([], None, [])
