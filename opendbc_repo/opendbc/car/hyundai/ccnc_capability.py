from opendbc.car.hyundai.values import HyundaiExtFlags, HyundaiFlags


def hda1_cluster_supported(cp) -> bool:
  required = HyundaiFlags.CANFD | HyundaiFlags.CAMERA_SCC
  flags = getattr(cp, "flags", 0)
  return (
    getattr(cp, "brand", None) == "hyundai"
    and flags & required == required
    and not flags & HyundaiFlags.CANFD_HDA2
    and bool(getattr(cp, "openpilotLongitudinalControl", False))
    and bool(getattr(cp, "extFlags", 0) & HyundaiExtFlags.CCNC_CLUSTER_RX)
  )


def read_hda1_cluster_support(params) -> bool:
  """Use the current fingerprinted vehicle. A cached or missing vehicle is not enough."""
  from capnp import KjException
  from opendbc.car import structs

  try:
    raw = params.get("CarParams")
    if raw:
      with structs.CarParams.from_bytes(raw) as cp:
        return hda1_cluster_supported(cp)
  except (KjException, OSError, TypeError, ValueError):
    return False
  return False
