"""Re-export of the runtime scanner (openpilot/selfdrive/carrot/lane_type/profile_scanner.py).

The validated prototype moved into the runtime package unchanged so the on-device process does not import tools.
"""
from openpilot.selfdrive.carrot.lane_type.profile_scanner import *  # noqa: F403
from openpilot.selfdrive.carrot.lane_type.profile_scanner import _runs, _segment  # noqa: F401
