"""Read target/packet_event.bin (written by `cargo test packet_event_for_python`) as card does: pycapnp Event, then marking.parse.

  python openpilot/selfdrive/carrot/lane_type/rust/tools/check_packet.py openpilot/selfdrive/carrot/lane_type/rust/target/packet_event.bin
"""
import sys

from openpilot.cereal import log
from openpilot.selfdrive.carrot.lane_type import marking

with log.Event.from_bytes(open(sys.argv[1], 'rb').read()) as e:
  assert e.which() == 'customReservedRawData2' and e.valid and e.logMonoTime == 5
  payload = bytes(e.customReservedRawData2)
parsed = marking.parse(payload, 2_000_000_000)
assert parsed['mode'] == 'stripnet' and parsed['frameId'] == 1234, parsed
assert parsed['left']['type'] == ('solid', 2_000_000_000 - 40_000_000 + marking.MAX_AGE_NS), parsed
assert parsed['left']['color'] == ('yellow', 2_000_000_000 - 90_000_000 + marking.MAX_AGE_NS), parsed
assert parsed['right'] == {'type': None, 'color': None}, parsed
print('ok', parsed)
