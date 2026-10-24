"""Optional CCNC display map loops, selected once at process startup."""
import os

loops = None
if os.environ.get("CCNC_RADAR_MAP_NATIVE", "1") != "0":
  try:
    from opendbc.car.hyundai import _ccnc_map_native
    if all(callable(getattr(_ccnc_map_native, name)) for name in ('to_ego', 'geometry', 'interpolate', 'update_lanes', 'lane_view')):
      loops = _ccnc_map_native
  except Exception:
    pass

BACKEND = "cython" if loops is not None else "python"


# The options backend is independent of the map switch, and tolerates an older binary.
options = None
if os.environ.get("CCNC_OPTIONS_NATIVE", "1") != "0":
  try:
    from opendbc.car.hyundai import _ccnc_map_native
    if all(callable(getattr(_ccnc_map_native, name)) for name in (
        'valid_boundary', 'side_lane_center', 'nearest_boundary', 'interp_np', 'update_ego_follow')):
      options = _ccnc_map_native
  except Exception:
    pass
