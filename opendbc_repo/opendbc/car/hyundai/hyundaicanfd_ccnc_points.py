"""Plain copies of liveTracks points for the CCNC display.

Reading a Cap'n Proto field from Python costs far more than a plain attribute, and a lookup of a name the
schema does not have (the display code probes `ccnc_fresh` on every point) costs about twenty times that.
The display reads the same points many times per update, so each radar message is copied once. With the message's
bytes the Rust core makes the copy from them, without reading the Cap'n Proto objects."""
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
