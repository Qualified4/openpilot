"""Plain copies of liveTracks points for the CCNC display.

Reading a Cap'n Proto field from Python costs far more than a plain attribute, and a lookup of a name the
schema does not have (the display code probes `ccnc_fresh` on every point) costs about twenty times that.
The display reads the same points many times per update, so each radar message is copied once. With the message's
bytes the Rust core makes the copy from them, without reading the Cap'n Proto objects. card also asks here whether the
Rust readers of liveTracks and modelV2 bytes match the loaded schema."""
import struct

from opendbc.car.hyundai import ccnc_core_loader

RUST = ccnc_core_loader.load()

FIELDS = ('trackId', 'dRel', 'yRel', 'vRel', 'aRel', 'yvRel', 'measured', 'vLead', 'aLead', 'jLead',
          'trackState', 'motionState')


class Point:
  __slots__ = FIELDS + ('radarSource',)


class LiveTracks:
  __slots__ = ('points',)


_last = (None, None)


def snapshot(live, raw=None):
  """The copy of a liveTracks message. The same message gives the same copy: the display tells a new radar
  update from a repeated one by object identity. Anything that is not a Cap'n Proto reader (replay and test
  inputs) is already cheap to read and is returned as it is. ``raw`` is the bytes of the Event message ``live``
  is from, or None."""
  global _last
  if not type(live).__module__.startswith('capnp'):
    return live
  if live is _last[0]:
    return _last[1]
  if raw is not None and RUST is not None:
    try:
      copy = RUST.live_tracks_copy(raw, Point, LiveTracks)
    except Exception:  # A Rust panic: the display's copy must not end card.
      copy = None
    if copy is not None:  # None: bytes the Rust reader does not read exactly as pycapnp; the objects are read.
      _last = (live, copy)
      return copy
  copy = LiveTracks()
  points = copy.points = []
  for p in live.points:
    point = Point()
    for name in FIELDS:
      setattr(point, name, getattr(p, name))
    point.radarSource = str(p.radarSource)
    points.append(point)
  _last = (live, copy)
  return copy


def schema_mismatch(log, car):
  """What the Rust reader of liveTracks bytes assumes about the schema that the loaded ``log``/``car`` schema does
  not say, or None. The reader holds the offsets it reads as constants; with a different schema it would read wrong
  points without any error, so card asks this once before it passes bytes."""
  if RUST is None:
    return 'no Rust core'
  c = RUST.live_tracks_schema()
  event = log.Event.schema
  member = event.fields['liveTracks'].proto
  points = car.RadarData.schema.fields['points'].proto
  element = points.slot.type.list.elementType if points.slot.type.which() == 'list' else None
  fields = car.RadarData.RadarPoint.schema.fields
  checks = [
    ('Event discriminant offset', event.node.struct.discriminantOffset, c['event_discriminant_offset']),
    ('Event.liveTracks discriminant', member.discriminantValue, c['event_live_tracks']),
    ('Event.liveTracks pointer', member.slot.offset, c['event_live_tracks_pointer']),
    ('Event.liveTracks type', member.slot.type.which() == 'struct' and member.slot.type.struct.typeId, car.RadarData.schema.node.id),
    ('RadarData.points pointer', points.slot.offset, c['radar_data_points_pointer']),
    ('RadarData.points element', element is not None and element.which() == 'struct' and element.struct.typeId,
     car.RadarData.RadarPoint.schema.node.id),
    ('RadarPoint fields read', set(c['radar_point_fields']), set(FIELDS) | {'radarSource'}),
    ('RadarSource names', [e.name for e in car.RadarData.RadarPoint.RadarSource.schema.node.enum.enumerants], list(c['radar_sources'])),
  ]
  for name, (kind, offset, default) in c['radar_point_fields'].items():
    if name not in fields:
      checks.append((f'RadarPoint.{name}', None, (kind, offset, default)))
      continue
    slot = fields[name].proto.slot
    value = getattr(slot.defaultValue, kind) if slot.type.which() == kind else None
    if kind == 'float32' and value is not None:
      value = struct.unpack('<I', struct.pack('<f', value))[0]
    enum = kind != 'enum' or slot.type.enum.typeId == car.RadarData.RadarPoint.RadarSource.schema.node.id
    checks.append((f'RadarPoint.{name}', (slot.type.which(), slot.offset, value, enum), (kind, offset, default, True)))
  return next((f'{what}: schema {found!r}, Rust reader {assumed!r}' for what, found, assumed in checks if found != assumed), None)


def model_schema_mismatch(log):
  """As schema_mismatch, for the Rust reader of modelV2 bytes (the map's lane update): what it assumes about the loaded
  ``log`` schema that the schema does not say, or None."""
  if RUST is None:
    return 'no Rust core'
  c = RUST.model_v2_schema()
  event = log.Event.schema
  member = event.fields['modelV2'].proto
  model = log.ModelDataV2.schema.fields
  xyzt = log.XYZTData.schema.fields

  def slot(fields, name):
    """A field's (pointer index, list element type), or None for a field the schema does not have."""
    if name not in fields or fields[name].proto.which() != 'slot':
      return None
    s = fields[name].proto.slot
    if s.type.which() != 'list':
      return s.offset, s.type.which()
    element = s.type.list.elementType
    return s.offset, element.which() if element.which() != 'struct' else element.struct.typeId

  xyzt_id = log.XYZTData.schema.node.id
  checks = [
    ('Event discriminant offset', event.node.struct.discriminantOffset, c['event_discriminant_offset']),
    ('Event.modelV2 discriminant', member.discriminantValue, c['event_model_v2']),
    ('Event.modelV2 pointer', member.slot.offset, c['event_model_v2_pointer']),
    ('Event.modelV2 type', member.slot.type.which() == 'struct' and member.slot.type.struct.typeId, log.ModelDataV2.schema.node.id),
    ('ModelDataV2.laneLines', slot(model, 'laneLines'), (c['lane_lines_pointer'], xyzt_id)),
    ('ModelDataV2.laneLineProbs', slot(model, 'laneLineProbs'), (c['lane_line_probs_pointer'], 'float32')),
    ('ModelDataV2.roadEdges', slot(model, 'roadEdges'), (c['road_edges_pointer'], xyzt_id)),
    ('XYZTData.x', slot(xyzt, 'x'), (c['xyzt_x_pointer'], 'float32')),
    ('XYZTData.y', slot(xyzt, 'y'), (c['xyzt_y_pointer'], 'float32')),
  ]
  return next((f'{what}: schema {found!r}, Rust reader {assumed!r}' for what, found, assumed in checks if found != assumed), None)
