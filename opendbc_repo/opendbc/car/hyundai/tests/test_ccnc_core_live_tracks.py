"""The Rust reader of liveTracks message bytes (ccnc_core/src/live_tracks.rs) against pycapnp: its schema constants
against the schema pycapnp loads, the points it reads against pycapnp reading the same bytes, field by field and bit by
bit, and bytes it cannot read exactly as pycapnp giving None (the caller then reads the objects), never a panic or a
wrong value. Also the display's two uses: the copy `hyundaicanfd_ccnc_points.snapshot` makes, and the map's points."""
from collections import Counter
import math
import random
import struct

import capnp
import pytest

from opendbc.car.hyundai import ccnc_core_loader
from opendbc.car.hyundai import hyundaicanfd_ccnc_points as ccnc_points
from openpilot.cereal import car, log

NO_TRAVERSAL_LIMIT = 2**64 - 1  # As openpilot.cereal.messaging.log_from_bytes reads.
NAMES = ('trackId', 'dRel', 'yRel', 'vRel', 'aRel', 'yvRel', 'measured', 'vLead', 'aLead', 'jLead', 'radarSource',
         'trackState', 'motionState')
FLOATS = ('dRel', 'yRel', 'vRel', 'aRel', 'yvRel', 'vLead', 'aLead', 'jLead')
SOURCES = ('frontRadar', 'scc', 'corner235', 'corner180', 'corner430')
F32_MAX = 3.4028234663852886e38
EDGE_FLOATS = (math.nan, -math.nan, math.inf, -math.inf, 0., -0., F32_MAX, -F32_MAX, 1.401298464324817e-45,
               -1.401298464324817e-45, 1.1754943508222875e-38, 1e39, -1e39, .1, 1.)
MESSAGES = 100_000
CHUNKS = 8
CORRUPTED = 12_000


@pytest.fixture(scope='module')
def core():
  core = ccnc_core_loader.load()
  assert core is not None, 'the x86_64 test build must be in prebuilt/linux-x86_64'
  assert core.abi_version() == 6
  return core


def key(value):
  """Type-strict, bit-exact form of a field value; NaNs are equal to each other."""
  if isinstance(value, float):
    return ('nan',) if math.isnan(value) else ('f', struct.pack('<d', value))
  return type(value).__name__, value


def pycapnp_points(raw):
  """The points pycapnp reads from the bytes, as field tuples; 'other' for another union member, 'error' where pycapnp
  raises. Unlike messaging, the traversal is limited to a few times the message's words: corrupted lists of zero-sized
  elements would otherwise have pycapnp make billions of points (the reader gives None for those)."""
  try:
    with log.Event.from_bytes(raw, traversal_limit_in_words=4 * len(raw) // 8 + 64) as event:
      if event.which() != 'liveTracks':
        return 'other'
      return [tuple(str(p.radarSource) if n == 'radarSource' else getattr(p, n) for n in NAMES) for p in event.liveTracks.points]
  except Exception:
    return 'error'


def keys(points):
  return [tuple(key(v) for v in p) for p in points]


def frame(segments):
  """Segments (bytes, whole words) in the framing to_bytes() writes."""
  head = struct.pack('<I', len(segments) - 1) + b''.join(struct.pack('<I', len(s) // 8) for s in segments)
  return head + b'\0' * (-len(head) % 8) + b''.join(segments)


def segments_of(raw):
  count = struct.unpack_from('<I', raw)[0] + 1
  sizes = struct.unpack_from(f'<{count}I', raw, 4)
  start = (4 + 4 * count + 7) // 8 * 8
  out = []
  for size in sizes:
    out.append(raw[start:start + 8 * size])
    start += 8 * size
  return out


def random_float(rng):
  kind = rng.random()
  if kind < .3:
    return rng.choice(EDGE_FLOATS)
  if kind < .6:  # Any float32 bit pattern, NaN payloads included.
    return struct.unpack('<f', rng.getrandbits(32).to_bytes(4, 'little'))[0]
  return rng.uniform(-300., 300.)


def random_point(rng):
  p = {}
  for name in NAMES:
    if rng.random() < .15:
      continue  # Left at its default.
    if name == 'trackId':
      p[name] = rng.choice((0, 2**64 - 1, 2**63, rng.getrandbits(64), rng.randrange(100000)))
    elif name in FLOATS:
      p[name] = random_float(rng)
    elif name == 'measured':
      p[name] = rng.random() < .5
    elif name == 'radarSource':
      p[name] = rng.choice(SOURCES)
    else:
      p[name] = rng.choice((0, 255, 1, 2, 3, rng.randrange(256)))
  return p


def random_message(rng):
  """A real log.Event(liveTracks) from pycapnp, sometimes in many small segments; its bytes from to_bytes() or framed
  from to_segments()."""
  count = rng.choice((0, rng.randrange(65), rng.randrange(65), rng.randrange(16), rng.randrange(8), 64))
  live = {'points': [random_point(rng) for _ in range(count)]}
  if rng.random() < .3:
    live['displayPoints'] = [random_point(rng) for _ in range(rng.randrange(4))]
  if rng.random() < .3:
    live['radarTrackFlipped'] = live['hasDisplayData'] = True
  if rng.random() < .15:
    builder = capnp._MallocMessageBuilder(rng.choice((1, 2, 3, 4, 8, 16, 64)))
    event = builder.init_root(log.Event)
  else:
    event = log.Event.new_message()
  event.logMonoTime = rng.getrandbits(63)
  event.valid = rng.random() < .9
  event.init('liveTracks')
  event.liveTracks = live
  if rng.random() < .5:
    return event.to_bytes()
  raw = frame(event.to_segments())
  assert raw == event.to_bytes()
  return raw


def test_schema_constants_are_the_schema(core):
  """A schema change that moves what the reader reads fails here; the constants are then fixed in live_tracks.rs.
  card runs the same comparison when it starts and passes no bytes if it fails."""
  assert ccnc_points.schema_mismatch(log, car) is None
  c = core.live_tracks_schema()
  assert set(c['radar_point_fields']) == set(NAMES)
  assert list(c['radar_sources']) == list(SOURCES)


@pytest.mark.parametrize('change', ['offset', 'kind', 'default', 'discriminant', 'pointer', 'source', 'field'])
def test_schema_mismatch_names_what_differs(core, monkeypatch, change):
  """The comparison against a reader that assumes something else: each kind of difference is reported."""
  c = core.live_tracks_schema()
  kind, offset, default = c['radar_point_fields']['dRel']
  if change == 'offset':
    c['radar_point_fields']['dRel'] = (kind, offset + 1, default)
  elif change == 'kind':
    c['radar_point_fields']['trackState'] = ('uint16', *c['radar_point_fields']['trackState'][1:])
  elif change == 'default':
    c['radar_point_fields']['dRel'] = (kind, offset, 1)
  elif change == 'discriminant':
    c['event_live_tracks'] += 1
  elif change == 'pointer':
    c['radar_data_points_pointer'] += 1
  elif change == 'source':
    c['radar_sources'] = list(c['radar_sources'])[::-1]
  else:
    c['radar_point_fields']['notInSchema'] = c['radar_point_fields'].pop('jLead')

  class Reader:
    live_tracks_schema = staticmethod(lambda: c)

  monkeypatch.setattr(ccnc_points, 'RUST', Reader)
  assert ccnc_points.schema_mismatch(log, car) is not None
  monkeypatch.setattr(ccnc_points, 'RUST', None)
  assert ccnc_points.schema_mismatch(log, car) == 'no Rust core'


@pytest.mark.parametrize('chunk', range(CHUNKS))
def test_reader_is_pycapnp(core, chunk):
  rng = random.Random(4100 + chunk)
  stats = Counter()
  for _ in range(MESSAGES // CHUNKS):
    raw = random_message(rng)
    want = pycapnp_points(raw)
    got = core._parse_for_test(raw)
    assert isinstance(want, list) and got is not None
    assert len(got) == len(want) and keys(got) == keys(want)
    stats['messages'] += 1
    stats['points'] += len(got)
    stats['segments>1'] += struct.unpack_from('<I', raw)[0] > 0
    stats['nan'] += any(isinstance(v, float) and math.isnan(v) for p in got for v in p)
    stats['empty'] += not got
    stats['64'] += len(got) == 64
  print(f'\nchunk {chunk}:', dict(stats))
  assert stats['segments>1'] > 0 and stats['empty'] > 0 and stats['64'] > 0 and stats['nan'] > 0


def word(value):
  return struct.pack('<Q', value & (2**64 - 1))


def struct_pointer(offset, data, pointers):
  return (offset << 2) & 0xffffffff | data << 32 | pointers << 48


def list_pointer(offset, size, count):
  return (offset << 2) & 0xffffffff | 1 | size << 32 | count << 35


def far_pointer(pad, segment, double=False):
  return 2 | int(double) << 2 | pad << 3 | segment << 32


def point_words(p, data_words):
  """A RadarPoint's data section from field values, at the schema's offsets, cut or padded to data_words."""
  data = bytearray(48)
  struct.pack_into('<Q', data, 0, p.get('trackId', 0))
  for i, name in zip((2, 3, 4, 5, 6, 8, 9, 10), FLOATS, strict=True):
    struct.pack_into('<f', data, 4 * i, p.get(name, 0.))
  data[28] = int(p.get('measured', False))
  struct.pack_into('<H', data, 30, p.get('source', 0))
  data[29], data[44] = p.get('trackState', 0), p.get('motionState', 0)
  return bytes(data[:8 * data_words]) + b'\0' * max(0, 8 * data_words - 48)


def crafted(points, *, data_words=6, pointer_words=0, radar_pointers=5, event_data=2, which=129, element_size=7,
            list_count=None, null_radar=False, null_list=False, double_far=None):
  """A liveTracks Event laid out by hand: root, Event, RadarData, then the points list. `double_far` moves the
  pointer to RadarData ('radar') or to the points ('points') behind a double-far landing pad in a second segment."""
  words = [struct_pointer(0, event_data, 1)]
  event = [0] * event_data
  if event_data >= 2:
    event[1] = which
  words += event
  event_pointer = len(words)
  words.append(0)
  radar = len(words)
  words += [0] + [0] * radar_pointers
  if not null_radar:
    words[event_pointer] = struct_pointer(radar - event_pointer - 1, 1, radar_pointers)
  points_pointer = radar + 1 + 1
  tag = len(words)
  element = data_words + pointer_words
  body = b''.join(point_words(p, data_words) + b'\0' * 8 * pointer_words for p in points)
  count = len(points) if list_count is None else list_count
  if not null_list and radar_pointers > 1:
    words[points_pointer] = list_pointer(tag - points_pointer - 1, element_size, len(points) * element)
  words.append(struct_pointer(count, data_words, pointer_words))
  segment = b''.join(word(w) for w in words) + body
  if double_far is None:
    return frame([segment])
  at = event_pointer if double_far == 'radar' else points_pointer
  target = radar if double_far == 'radar' else tag
  original = struct.unpack_from('<Q', segment, 8 * at)[0]
  segment = bytearray(segment)
  segment[8 * at:8 * at + 8] = word(far_pointer(0, 1, double=True))
  description = original & ~0xfffffffc  # The tag: the same pointer with offset 0.
  return frame([bytes(segment), word(far_pointer(target, 0)) + word(description)])


POINTS = [dict(trackId=7, dRel=12.5, yRel=-1.25, vRel=-3., vLead=11., measured=True, source=0, trackState=2, motionState=2),
          dict(trackId=2**64 - 1, dRel=-0., yRel=math.inf, measured=False, source=4, trackState=255)]


@pytest.mark.parametrize('points, case, readable', [
  (POINTS, dict(), True),
  (POINTS, dict(double_far='radar'), True),  # Double-far landing pads.
  (POINTS, dict(double_far='points'), True),
  (POINTS, dict(data_words=2), True),  # Elements of an older schema: the missing fields read as defaults.
  (POINTS, dict(data_words=8, pointer_words=2), True),  # Of a newer one: larger elements.
  ([], dict(), True),
  ([], dict(data_words=0), True),
  (POINTS, dict(radar_pointers=1), True),  # RadarData without the points pointer: no points.
  (POINTS, dict(null_list=True), True),
  (POINTS, dict(null_radar=True), True),
  (POINTS, dict(event_data=1), False),  # The discriminant past the data section: initData, not liveTracks.
  (POINTS, dict(which=128), False),  # Another union member.
  (POINTS, dict(element_size=5), False),  # Not an inline composite list.
  (POINTS, dict(list_count=3), False),  # The tag counts more elements than the list holds.
  (POINTS, dict(data_words=0), False),  # Zero-sized elements.
])
def test_crafted_layouts(core, points, case, readable):
  raw = crafted(points, **case)
  got, want = core._parse_for_test(raw), pycapnp_points(raw)
  if readable:
    assert isinstance(want, list) and got is not None and keys(got) == keys(want)
    assert len(got) == (len(points) if not {'radar_pointers', 'null_list', 'null_radar'} & set(case) else 0)
  else:
    assert got is None


def test_unreadable_forms(core):
  good = crafted(POINTS)
  assert keys(core._parse_for_test(good)) == keys(pycapnp_points(good))
  # Unknown enum value: pycapnp raises on str(); the reader gives None and the caller's object path raises the same.
  unknown = crafted([dict(POINTS[0], source=5)])
  assert core._parse_for_test(unknown) is None and pycapnp_points(unknown) == 'error'
  assert core._parse_for_test(good + b'\0' * 8) is None  # Trailing bytes (pycapnp reads them).
  assert core._parse_for_test(good[:-8]) is None
  assert core._parse_for_test(b'') is None and core._parse_for_test(b'\0' * 7) is None
  assert core._parse_for_test(struct.pack('<I', 511) + b'\0' * 2048) is None  # 512 segments
  capability = bytearray(good)
  capability[8 + 8 * 3:8 + 8 * 4] = word(3)  # Event's pointer to RadarData as a capability.
  assert core._parse_for_test(bytes(capability)) is None
  other = log.Event.new_message()
  other.init('carState')
  assert core._parse_for_test(other.to_bytes()) is None


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
  elif kind == 'pointer':  # A pointer-like word: another kind, offset or size in the same place.
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
  rng = random.Random(77)
  bases = [random_message(rng) for _ in range(200)] + [crafted(POINTS, double_far='points'), crafted(POINTS, data_words=8, pointer_words=2)]
  stats = Counter()
  for _ in range(CORRUPTED):
    kind, raw = corrupt(rng, rng.choice(bases))
    got, want = core._parse_for_test(raw), pycapnp_points(raw)
    if got is None:
      stats[kind, 'none', 'pycapnp ' + ('read' if isinstance(want, list) else want)] += 1
    else:
      # Values only where pycapnp reads the same bytes; that is every case the reader read here.
      assert isinstance(want, list), (kind, raw.hex())
      assert keys(got) == keys(want), (kind, raw.hex())
      stats[kind, 'read'] += 1
  print('\ncorrupted:', sorted(stats.items()))
  assert sum(stats.values()) == CORRUPTED and sum(v for k, v in stats.items() if k[1] == 'read') > 0


def test_snapshot_from_bytes_is_the_object_copy(core, monkeypatch):
  rng = random.Random(5)
  compared = 0
  for i in range(2000):
    raw = random_message(rng)
    with log.Event.from_bytes(raw, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as event:
      live = event.liveTracks
      monkeypatch.setattr(ccnc_points, '_last', (None, None))
      from_objects = ccnc_points.snapshot(live)
      monkeypatch.setattr(ccnc_points, '_last', (None, None))
      from_bytes = ccnc_points.snapshot(live, raw)
      assert from_bytes is not from_objects and type(from_bytes) is ccnc_points.LiveTracks
      assert ccnc_points.snapshot(live, raw) is from_bytes  # The same message gives the same copy.
      assert len(from_bytes.points) == len(from_objects.points)
      for a, b in zip(from_bytes.points, from_objects.points, strict=True):
        assert type(a) is ccnc_points.Point
        assert [key(getattr(a, n)) for n in NAMES] == [key(getattr(b, n)) for n in NAMES]
        compared += 1
      if i % 10 == 0:  # Without the module the objects are copied.
        monkeypatch.setattr(ccnc_points, '_last', (None, None))
        monkeypatch.setattr(ccnc_points, 'RUST', None)
        assert keys((getattr(p, n) for n in NAMES) for p in ccnc_points.snapshot(live, raw).points) == \
          keys((getattr(p, n) for n in NAMES) for p in from_objects.points)
        monkeypatch.setattr(ccnc_points, 'RUST', core)
  assert compared > 10000
  not_capnp = object()
  assert ccnc_points.snapshot(not_capnp, b'x') is not_capnp
  # Bytes the reader does not read: the Python copy.
  unknown = crafted([dict(POINTS[0], source=1)]) + b'\0' * 8
  with log.Event.from_bytes(unknown, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as event:
    monkeypatch.setattr(ccnc_points, '_last', (None, None))
    copy = ccnc_points.snapshot(event.liveTracks, unknown)
    assert [p.radarSource for p in copy.points] == ['scc']
