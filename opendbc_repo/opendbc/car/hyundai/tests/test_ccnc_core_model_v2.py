"""The Rust reader of modelV2 message bytes (ccnc_core/src/model_v2.rs) against pycapnp: its schema constants against
the schema pycapnp loads, the lane lines, their probabilities and the road edges it reads against pycapnp reading the
same bytes, bit by bit, and bytes it cannot read exactly as pycapnp giving None (the map then reads the objects), never
a panic or a wrong value. The lane update itself is compared on the map in test_ccnc_core_radar_map.py."""
from collections import Counter
import math
import random
import struct

import capnp
import pytest

from opendbc.car.hyundai import ccnc_core_loader
from opendbc.car.hyundai import hyundaicanfd_ccnc_points as ccnc_points
from openpilot.cereal import log

F32_MAX = 3.4028234663852886e38
EDGE_FLOATS = (math.nan, -math.nan, math.inf, -math.inf, 0., -0., F32_MAX, -F32_MAX, 1.401298464324817e-45,
               -1.401298464324817e-45, 1.1754943508222875e-38, 1e39, -1e39, .1, 1.)
MESSAGES = 24_000
CHUNKS = 6
CORRUPTED = 6_000
REAL_X = [192. * (i / 32) ** 2 for i in range(33)]  # The model's 33 distances, 0 to 192m.


@pytest.fixture(scope='module')
def core():
  core = ccnc_core_loader.load()
  assert core is not None, 'the x86_64 test build must be in prebuilt/linux-x86_64'
  assert core.abi_version() == ccnc_core_loader.ABI_VERSION
  return core


def key(value):
  """Bit-exact form of a float (NaNs are equal to each other), and of lists and tuples of them."""
  if isinstance(value, float):
    return ('nan',) if math.isnan(value) else ('f', struct.pack('<d', value))
  if isinstance(value, (list, tuple)):
    return tuple(key(v) for v in value)
  raise TypeError(type(value))


def lines_of(model):
  def lines(ls):
    return [(list(line.x), list(line.y)) for line in ls]

  return lines(model.laneLines), list(model.laneLineProbs), lines(model.roadEdges)


def pycapnp_lines(raw):
  """What pycapnp reads from the bytes, as _parse_model_for_test gives it; 'other' for another union member, 'error'
  where pycapnp raises. The traversal is limited to a few times the message's words, as in the liveTracks test."""
  try:
    with log.Event.from_bytes(raw, traversal_limit_in_words=4 * len(raw) // 8 + 64) as event:
      if event.which() != 'modelV2':
        return 'other'
      return lines_of(event.modelV2)
  except Exception:
    return 'error'


def frame(segments):
  head = struct.pack('<I', len(segments) - 1) + b''.join(struct.pack('<I', len(s) // 8) for s in segments)
  return head + b'\0' * (-len(head) % 8) + b''.join(segments)


def random_float(rng):
  kind = rng.random()
  if kind < .1:
    return rng.choice(EDGE_FLOATS)
  if kind < .2:  # Any float32 bit pattern, NaN payloads included.
    return struct.unpack('<f', rng.getrandbits(32).to_bytes(4, 'little'))[0]
  return rng.uniform(-300., 300.)


def random_line(rng):
  if rng.random() < .3:  # Real size: 33 points along the model's distances.
    offset = rng.uniform(-6., 6.)
    line = {'x': REAL_X, 'y': [offset + rng.gauss(0., .02) * x for x in REAL_X]}
  else:
    n = rng.choice((0, 1, 2, rng.randrange(41), rng.randrange(41), 40))
    line = {'x': [random_float(rng) for _ in range(n)], 'y': [random_float(rng) for _ in range(n)]}
    if rng.random() < .1:  # x and y of different lengths.
      line['y'] = line['y'][:rng.randrange(n + 1)] if rng.random() < .5 else line['y'] + [random_float(rng)]
  for name in ('x', 'y'):
    if rng.random() < .05:
      del line[name]  # Never set: a null list.
  if rng.random() < .3:
    line['z'] = [0.] * 33
    line['t'] = REAL_X
  return line


def random_message(rng):
  """A real log.Event(modelV2) from pycapnp with the other fields the model sends, sometimes in many small segments;
  its bytes from to_bytes() or framed from to_segments()."""
  model = {}
  if rng.random() < .95:
    model['laneLines'] = [random_line(rng) for _ in range(rng.choice((4, 4, 0, 1, 2, 3, 5, 6)))]
  if rng.random() < .95:
    model['laneLineProbs'] = [random_float(rng) if rng.random() < .3 else rng.random() for _ in range(rng.choice((4, 4, 0, 2, 6)))]
  if rng.random() < .9:
    model['roadEdges'] = [random_line(rng) for _ in range(rng.choice((2, 2, 0, 1, 3)))]
  if rng.random() < .5:
    model['position'] = {'x': REAL_X, 'y': [0.] * 33}
    model['leadsV3'] = [{'prob': .9, 'x': [10.] * 6, 'y': [0.] * 6, 'v': [3.] * 6} for _ in range(3)]
    model['meta'] = {'desire': 'laneChangeLeft', 'laneChangeState': 'laneChangeStarting'}
    model['laneLineStds'] = [.1] * 4
    model['frameId'] = rng.getrandbits(32)
  if rng.random() < .2:
    builder = capnp._MallocMessageBuilder(rng.choice((1, 2, 3, 4, 8, 16, 64)))
    event = builder.init_root(log.Event)
  else:
    event = log.Event.new_message()
  event.logMonoTime = rng.getrandbits(63)
  event.valid = rng.random() < .9
  event.init('modelV2')
  event.modelV2 = model
  if rng.random() < .5:
    return event.to_bytes()
  raw = frame(event.to_segments())
  assert raw == event.to_bytes()
  return raw


def test_schema_constants_are_the_schema(core):
  """A schema change that moves what the reader reads fails here; the constants are then fixed in model_v2.rs. card
  runs the same comparison when it starts and passes no modelV2 bytes if it fails."""
  assert ccnc_points.model_schema_mismatch(log) is None


@pytest.mark.parametrize('change', ['discriminant', 'model_pointer', 'lane_lines', 'probs', 'edges', 'x', 'y', 'offset'])
def test_schema_mismatch_names_what_differs(core, monkeypatch, change):
  c = dict(core.model_v2_schema())
  name = {'discriminant': 'event_model_v2', 'model_pointer': 'event_model_v2_pointer', 'lane_lines': 'lane_lines_pointer',
          'probs': 'lane_line_probs_pointer', 'edges': 'road_edges_pointer', 'x': 'xyzt_x_pointer', 'y': 'xyzt_y_pointer',
          'offset': 'event_discriminant_offset'}[change]
  c[name] += 1

  class Reader:
    model_v2_schema = staticmethod(lambda: c)

  monkeypatch.setattr(ccnc_points, 'RUST', Reader)
  mismatch = ccnc_points.model_schema_mismatch(log)
  assert mismatch is not None
  print(mismatch)
  monkeypatch.setattr(ccnc_points, 'RUST', None)
  assert ccnc_points.model_schema_mismatch(log) == 'no Rust core'


@pytest.mark.parametrize('chunk', range(CHUNKS))
def test_reader_is_pycapnp(core, chunk):
  rng = random.Random(7300 + chunk)
  stats = Counter()
  for _ in range(MESSAGES // CHUNKS):
    raw = random_message(rng)
    want = pycapnp_lines(raw)
    got = core._parse_model_for_test(raw)
    assert isinstance(want, tuple) and got is not None
    assert key(tuple(got)) == key(want)
    lines, probs, edges = want
    stats['messages'] += 1
    stats['segments>1'] += struct.unpack_from('<I', raw)[0] > 0
    stats['lines0'] += not lines
    stats['lines>4'] += len(lines) > 4
    stats['no_edges'] += not edges
    stats['33'] += any(len(x) == 33 for x, _ in lines)
    stats['40'] += any(len(x) == 40 for x, _ in lines + edges)
    stats['nan'] += any(math.isnan(v) for x, y in lines + edges for v in x + y)
    stats['inf'] += any(math.isinf(v) for x, y in lines + edges for v in x + y)
    stats['-0'] += any(v == 0. and math.copysign(1., v) < 0. for x, y in lines + edges for v in x + y)
    stats['xy_lengths'] += any(len(x) != len(y) for x, y in lines + edges)
  print(f'\nchunk {chunk}:', dict(stats))
  for name, count in stats.items():
    assert count > 0, name


def word(value):
  return struct.pack('<Q', value & (2**64 - 1))


def struct_pointer(offset, data, pointers):
  return (offset << 2) & 0xffffffff | data << 32 | pointers << 48


def list_pointer(offset, size, count):
  return (offset << 2) & 0xffffffff | 1 | size << 32 | count << 35


def far_pointer(pad, segment, double=False):
  return 2 | int(double) << 2 | pad << 3 | segment << 32


def floats(values):
  data = b''.join(struct.pack('<f', v) for v in values)
  return data + b'\0' * (-len(data) % 8)


X, Y, PROBS = [0., 10., 20.5], [-1.75, -1.8, -0.], [.9, .05]


def crafted(*, model_pointers=17, xyzt_pointers=7, which=73, element_size=4, probs_size=4, double_far=None, lines=2,
            line_count=None, float_count=None, probs_count=None):
  """A modelV2 Event laid out by hand: root, Event (2 data words), ModelDataV2 (no data), then laneLines (inline
  composite XYZTData), their x/y lists and laneLineProbs; no road edges. `double_far` moves the pointer to the lane
  lines ('lines') or to the first line's x ('x') behind a double-far landing pad in a second segment."""
  words = [struct_pointer(0, 2, 1), 0, which, 0]  # root; Event data (discriminant at bytes 8-9); Event pointer
  model = len(words)
  words[3] = struct_pointer(model - 3 - 1, 0, model_pointers)
  words += [0] * model_pointers
  out = list(words)

  def append(chunk):
    start = len(out)
    out.extend(struct.unpack(f'<{len(chunk) // 8}Q', chunk))
    return start

  pointers = {}
  if model_pointers > 4:
    tag = append(word(struct_pointer(lines if line_count is None else line_count, 0, xyzt_pointers)))
    elements = append(b'\0' * 8 * xyzt_pointers * lines)
    out[model + 4] = list_pointer(tag - (model + 4) - 1, 7, xyzt_pointers * lines)
    for j in range(lines):
      for k, values in enumerate((X, Y)[:xyzt_pointers]):
        at = elements + j * xyzt_pointers + k
        count = len(values) if float_count is None else float_count
        start = append(floats(values))
        out[at] = list_pointer(start - at - 1, element_size, count)
        pointers[j, k] = at
    pointers['lines'] = model + 4
  if model_pointers > 5:
    start = append(floats(PROBS) if probs_size == 4 else b''.join(word(0) for _ in PROBS))
    out[model + 5] = list_pointer(start - (model + 5) - 1, probs_size, len(PROBS) if probs_count is None else probs_count)
  segment = b''.join(word(w) for w in out)
  if double_far is None:
    return frame([segment])
  at = pointers['lines'] if double_far == 'lines' else pointers[0, 0]
  original = out[at]
  offset = (original & 0xffffffff) >> 2
  target = at + 1 + (offset - (1 << 30) if offset >= 1 << 29 else offset)
  segment = bytearray(segment)
  segment[8 * at:8 * at + 8] = word(far_pointer(0, 1, double=True))
  description = original & ~0xfffffffc  # The same pointer with offset 0.
  return frame([bytes(segment), word(far_pointer(target, 0)) + word(description)])


@pytest.mark.parametrize('case, readable, expect', [
  (dict(), True, 2),
  (dict(double_far='lines'), True, 2),  # Double-far landing pads.
  (dict(double_far='x'), True, 2),
  (dict(xyzt_pointers=1), True, 2),  # Lines of an older schema: no y list, so y reads empty.
  (dict(model_pointers=5), True, 2),  # A model without the probabilities pointer: no probabilities.
  (dict(model_pointers=4), True, 0),  # Nor the lane lines.
  (dict(lines=0), True, 0),
  (dict(which=72), False, None),  # Another union member.
  (dict(element_size=5), False, None),  # x as 8-byte elements: not a List(Float32) pycapnp writes.
  (dict(probs_size=5), False, None),
  (dict(line_count=3), False, None),  # The tag counts more elements than the list holds.
  (dict(float_count=4), True, 2),  # A count that takes the list's padding: pycapnp reads it too.
  (dict(probs_count=3), False, None),  # The last list runs past the segment's end.
])
def test_crafted_layouts(core, case, readable, expect):
  raw = crafted(**case)
  got, want = core._parse_model_for_test(raw), pycapnp_lines(raw)
  if readable:
    assert isinstance(want, tuple) and got is not None and key(tuple(got)) == key(want)
    assert len(got[0]) == expect
  else:
    assert got is None


def test_unreadable_forms(core):
  good = crafted()
  assert key(tuple(core._parse_model_for_test(good))) == key(pycapnp_lines(good))
  assert core._parse_model_for_test(good + b'\0' * 8) is None  # Trailing bytes.
  assert core._parse_model_for_test(good[:-8]) is None
  assert core._parse_model_for_test(b'') is None and core._parse_model_for_test(b'\0' * 7) is None
  capability = bytearray(good)
  capability[8 + 8 * 3:8 + 8 * 4] = word(3)  # Event's pointer to the model as a capability.
  assert core._parse_model_for_test(bytes(capability)) is None
  other = log.Event.new_message()
  other.init('liveTracks')
  assert core._parse_model_for_test(other.to_bytes()) is None


def corrupt(rng, raw):
  kind = rng.choice(('truncate', 'flip', 'word', 'pointer', 'segment_table', 'splice'))
  data = bytearray(raw)
  if kind == 'truncate':
    del data[rng.randrange(len(data)):]
  elif kind == 'flip':
    for _ in range(rng.randrange(1, 8)):
      data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
  elif kind == 'word':
    at = rng.randrange(len(data) // 8) * 8
    data[at:at + 8] = rng.getrandbits(64).to_bytes(8, 'little')
  elif kind == 'pointer':
    at = rng.randrange(len(data) // 8) * 8
    data[at:at + 8] = word(rng.randrange(4) | rng.randrange(-64, 64) << 2 & 0xffffffff | rng.getrandbits(32) << 32)
  elif kind == 'segment_table':
    data[rng.randrange(min(len(data), 16))] = rng.randrange(256)
  else:
    other = random_message(rng)
    at = rng.randrange(len(data))
    data[at:] = other[at:]
  return kind, bytes(data)


def test_corrupted_bytes_give_none_or_pycapnp_values(core):
  rng = random.Random(79)
  bases = [random_message(rng) for _ in range(200)] + [crafted(double_far='x'), crafted(xyzt_pointers=1)]
  stats = Counter()
  for _ in range(CORRUPTED):
    kind, raw = corrupt(rng, rng.choice(bases))
    got, want = core._parse_model_for_test(raw), pycapnp_lines(raw)
    if got is None:
      stats[kind, 'none', 'pycapnp ' + ('read' if isinstance(want, tuple) else want)] += 1
    else:
      assert isinstance(want, tuple), (kind, raw.hex())
      assert key(tuple(got)) == key(want), (kind, raw.hex())
      stats[kind, 'read'] += 1
  print('\ncorrupted:', sorted(stats.items()))
  assert sum(stats.values()) == CORRUPTED and sum(v for k, v in stats.items() if k[1] == 'read') > 0
