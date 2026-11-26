"""Replay-only event cache for ccnc_map's standalone engine_replay.

  python tools/ccnc/replay_engine.py --cache OUTPUT_DIR RLOG...
  cargo run --manifest-path tools/ccnc/engine_replay/Cargo.toml -- OUTPUT_DIR/*.capnp
"""
import argparse
from pathlib import Path

import capnp

import replay_full


# Include the recorded rc18 slots for exact stamp joins by the comparator.
# The CAN projection and unrelated services are not needed for this shadow run.
WANTED = ("carState", "livePose", "modelV2", "liveTracks", "radarState",
          "customReserved3", "customReserved4")


def schema():
  root = replay_full.ROOT
  cereal = root / "openpilot" / "cereal"
  capnp.remove_import_hook()
  return capnp.load(str(cereal / "log.capnp"), imports=[
    str(cereal), str(cereal / "include"), str(root / "opendbc_repo" / "opendbc" / "car"),
  ])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cache", type=Path, required=True)
  parser.add_argument("logs", type=Path, nargs="+")
  args = parser.parse_args()
  replay_full.CACHE = args.cache
  replay_full.WANTED = WANTED
  log = schema()
  for path in args.logs:
    route, seg = replay_full.segkey(path)
    events = replay_full.events(log, path)
    counts = {name: 0 for name in WANTED}
    for event in events:
      counts[event.which()] += 1
    print(f"{route}--{seg}", counts, flush=True)


if __name__ == "__main__":
  main()
