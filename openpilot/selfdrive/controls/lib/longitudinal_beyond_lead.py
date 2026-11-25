"""A slow car beyond leadOne, in ego's path, seen by the radar while leadOne moves aside.

Owner interventions 474_21 / 472_2 (work/drive-20261125/REPORT_INTERV_KO.md): the radar had the stopped or slow
car beyond the lead 2.4-2.8 s before the lead moved out of the way, and the planner only reacted when the lead
changed. While leadOne moves aside, BeyondLeadState offers that car as leadTwo (an ordinary second obstacle with
the safety-distance gap), so ego starts slowing before the lead is gone.

Guards against roadside clutter, all needed at once:
- in ego's path: |dPath| (radar point projected on the model path, as radard does) <= BEYOND_PATH_HALF_WIDTH,
- beyond leadOne by BEYOND_MIN_GAP and within BEYOND_MAX_DISTANCE,
- slow: absolute speed at most BEYOND_SPEED_MAX and BEYOND_SPEED_MARGIN below leadOne,
- moving now or before: a track never faster than BEYOND_MOVED_SPEED (a parked car, a pole) never counts,
- persistent: the same track id passing all of the above for BEYOND_PERSIST_S,
- leadOne moving aside: its |dPath| at least BEYOND_LEAD_ASIDE and growing by BEYOND_LEAD_ASIDE_RATE.
The obstacle and its gap are the existing leadTwo ones; nothing changes while any guard fails.
"""
import math
from types import SimpleNamespace

from openpilot.selfdrive.carrot.radar_motion.predictor import project_to_model_path

BEYOND_PATH_HALF_WIDTH = 1.0   # m from the model path
BEYOND_MIN_GAP = 5.0           # m beyond leadOne
BEYOND_MAX_DISTANCE = 80.0     # m
BEYOND_SPEED_MAX = 22.0 / 3.6  # m/s absolute
BEYOND_SPEED_MARGIN = 3.0      # m/s slower than leadOne
BEYOND_MOVED_SPEED = 2.0       # m/s the track must have reached at some point
BEYOND_PERSIST_S = 1.0         # s
BEYOND_LEAD_ASIDE = 0.5        # m path offset of leadOne
BEYOND_LEAD_ASIDE_RATE = 0.2   # m/s growth of that offset (over BEYOND_LEAD_ASIDE_WINDOW)
BEYOND_LEAD_ASIDE_WINDOW = 0.5  # s
BEYOND_TRACK_FORGET_S = 2.0    # s a track unseen is forgotten


def model_path(model):
  position = getattr(model, 'position', None)
  if position is None:
    return ()
  return tuple((float(x), float(y)) for x, y in zip(position.x, position.y, strict=False)
               if math.isfinite(float(x)) and math.isfinite(float(y)))


class BeyondLeadState:
  def __init__(self):
    self.tracks = {}       # track id -> dict(v_max, ok_time, last_seen)
    self.lead_hist = []    # (t, |dPath| of leadOne, track id)
    self.time = 0.0
    self.active_id = None

  def reset(self):
    self.__init__()

  def lead_aside(self, lead_one):
    if not lead_one.status:
      self.lead_hist.clear()
      return False
    track_id = int(getattr(lead_one, 'radarTrackId', -1))
    offset = abs(float(lead_one.dPath))
    if self.lead_hist and self.lead_hist[-1][2] != track_id:
      self.lead_hist.clear()
    self.lead_hist.append((self.time, offset, track_id))
    while self.lead_hist and self.time - self.lead_hist[0][0] > BEYOND_LEAD_ASIDE_WINDOW + 1e-6:
      self.lead_hist.pop(0)
    if self.time - self.lead_hist[0][0] < BEYOND_LEAD_ASIDE_WINDOW - 0.06:
      return False
    growth = (offset - self.lead_hist[0][1]) / BEYOND_LEAD_ASIDE_WINDOW
    return offset >= BEYOND_LEAD_ASIDE and growth >= BEYOND_LEAD_ASIDE_RATE

  def update(self, *, live_tracks, lead_one, v_ego, path, dt, enabled=True):
    """Returns a leadTwo-like lead for the car beyond leadOne, or None."""
    self.time += max(0.0, float(dt))
    if not enabled or live_tracks is None or not lead_one.status or len(path) < 2:
      self.active_id = None
      self.lead_hist.clear()
      return None
    aside = self.lead_aside(lead_one)
    lead_speed = float(lead_one.vLead)
    best = None
    for p in live_tracks.points:
      track_id = int(p.trackId)
      v_lead = float(getattr(p, 'vLead', float('nan')))
      v_abs = v_lead if math.isfinite(v_lead) else float(v_ego) + float(p.vRel)
      st = self.tracks.setdefault(track_id, {"v_max": v_abs, "ok_time": 0.0, "last_seen": self.time})
      st["v_max"] = max(st["v_max"], v_abs)
      st["last_seen"] = self.time
      distance = float(p.dRel)
      in_path = abs(project_to_model_path(path, distance, float(p.yRel)).d_path) <= BEYOND_PATH_HALF_WIDTH
      beyond = float(lead_one.dRel) + BEYOND_MIN_GAP <= distance <= BEYOND_MAX_DISTANCE
      slow = v_abs <= min(BEYOND_SPEED_MAX, lead_speed - BEYOND_SPEED_MARGIN)
      moved = st["v_max"] >= BEYOND_MOVED_SPEED
      if in_path and beyond and slow and moved:
        st["ok_time"] += max(0.0, float(dt))
      else:
        st["ok_time"] = 0.0
      if st["ok_time"] >= BEYOND_PERSIST_S - 1e-6 and (best is None or distance < best[0]):
        best = (distance, p, v_abs)
    for track_id in [k for k, s in self.tracks.items() if self.time - s["last_seen"] > BEYOND_TRACK_FORGET_S]:
      del self.tracks[track_id]
    if best is None or not aside:
      self.active_id = None
      return None
    distance, p, v_abs = best
    self.active_id = int(p.trackId)
    return SimpleNamespace(
      status=True, radar=True, radarTrackId=int(p.trackId), dRel=distance, yRel=float(p.yRel),
      vRel=float(p.vRel), vLead=max(0.0, v_abs), vLeadK=max(0.0, v_abs), aLeadK=0.0, aLeadTau=1.5, aRel=0.0,
      modelProb=0.0, dPath=0.0, vLat=0.0, jLead=0.0, score=0.0, cutOutTime=0.0, cutOutConfidence=0.0, fcw=False,
    )


def with_lead_two(radar_state, lead_two):
  """radarState stand-in with leadTwo replaced (leadOne and everything else as given)."""
  return SimpleNamespace(leadOne=radar_state.leadOne, leadTwo=lead_two)
