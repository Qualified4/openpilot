"""Use the same compiled pycapnp schema as card for Rust wire tests."""
from pathlib import Path
import sys

root = next(p for p in Path(__file__).resolve().parents if (p / "openpilot/cereal").is_dir())
sys.path[:0] = [str(root), str(root / "opendbc_repo")]
from openpilot.cereal import custom, log

map_type = custom.CustomReserved3
slot_type = map_type.Slot


def offset(schema, name):
  return schema.fields[name].proto.slot.offset


def schema_values():
  event = log.Event.schema
  map_schema = map_type.schema
  slot = slot_type.schema
  yield event.fields["customReserved3"].proto.discriminantValue
  yield offset(event, "customReserved3")
  yield offset(map_schema, "slots")
  for schema, names in (
    (map_schema, "radarLogMonoTime generation status egoLaneCoord valid"),
    (slot, "vid distance lateral detect laneCoordRel vRel kind kindConfirmed present vision kindWanted"),
  ):
    yield from (offset(schema, name) for name in names.split())
  for name in ("dataWordCount", "pointerCount"):
    yield from (getattr(schema.node.struct, name) for schema in (event, map_schema, slot))


def sample():
  # One root word forces far pointers in the Python fixture.
  event = log.Event.new_message(num_first_segment_words=1, logMonoTime=123456789012, valid=True)
  msg = event.init("customReserved3")
  msg.radarLogMonoTime, msg.generation, msg.status, msg.egoLaneCoord, msg.valid = 123456789012, 17, 11, -1.25, True
  slots = msg.init("slots", 5)
  slot = slots[3]
  slot.vid, slot.distance, slot.lateral, slot.detect = 42, -2.5, 3.25, 6
  slot.laneCoordRel, slot.vRel, slot.kind, slot.kindConfirmed = -0.75, -1.5, 3, True
  slot.present, slot.vision = True, 1
  slots[0].present, slots[0].vid, slots[0].kindWanted = True, 9, True
  slots[4].present, slots[4].detect, slots[4].lateral = True, 6, -2.75
  return event


def check_wire():
  with log.Event.from_bytes(sys.stdin.buffer.read()) as event:
    expected = sample()
    assert event.which() == expected.which()
    assert event.to_dict() == expected.to_dict()


def events():
  # A car and model first, then valid tracks, reverse, and one more radar event.
  cs = log.Event.new_message(logMonoTime=1_000_000_000, valid=True)
  state = cs.init("carState")
  state.canValid, state.vEgo, state.gearShifter = True, 10.0, "drive"
  md = log.Event.new_message(logMonoTime=1_025_000_000, valid=True)
  model = md.init("modelV2")
  model.laneLineProbs = [1.0] * 4
  for line, y in zip(model.init("laneLines", 4), (-5.25, -1.75, 1.75, 5.25), strict=True):
    line.x = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    line.y = [y] * 6
  tracks = log.Event.new_message(logMonoTime=1_050_000_000, valid=True)
  tracks.init("liveTracks").init("points", 0)
  reverse = log.Event.new_message(logMonoTime=1_060_000_000, valid=True)
  state = reverse.init("carState")
  state.canValid, state.gearShifter = True, "reverse"
  next_tracks = log.Event.new_message(logMonoTime=1_100_000_000, valid=True)
  next_tracks.init("liveTracks").init("points", 0)
  for event in (cs, md, tracks, reverse, next_tracks):
    sys.stdout.buffer.write(event.to_bytes())


if __name__ == "__main__":
  match sys.argv[1]:
    case "schema":
      print(*schema_values())
    case "encode":
      sys.stdout.buffer.write(sample().to_bytes())
    case "decode":
      check_wire()
    case "events":
      events()
