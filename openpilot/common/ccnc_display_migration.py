from pathlib import Path

from opendbc.car.hyundai.ccnc_capability import hda1_cluster_supported

OPTION = "CcncExtendedDisplay"
PENDING = "CcncExtendedDisplayMigrationPending"
LEGACY_VALUES = {
  "CcncLaneColor": (b"1",),
  "CcncModelLanes": (b"2",),
  "CcncLaneCurvatureSource": (b"3",),
  "CcncLaneTypeDetection": (b"1", b"2"),
  "CcncRadarVehicles": (b"1",),
  "CcncVehiclePositionCorrection": (b"1",),
  "CcncRadarMap": (b"1",),
}


def prepare_migration(params) -> bool:
  """Preserve a legacy match before clear_all removes unregistered Params."""
  if params.get(OPTION) is not None:
    return False
  if params.get_bool(PENDING):
    return True
  try:
    matches = all(Path(params.get_param_path(key)).read_bytes() in expected for key, expected in LEGACY_VALUES.items())
  except OSError:
    return False
  if matches:
    params.put_bool(PENDING, True)
  return matches


def finish_migration(params, cp) -> None:
  if params.get_bool(PENDING):
    if params.get(OPTION) is None:
      params.put_bool(OPTION, hda1_cluster_supported(cp))
    params.remove(PENDING)
