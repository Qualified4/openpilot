"""Score #191 on paired rc18 and shadow ticks. Requires pycapnp and matplotlib.

  python tools/ccnc/compare_engine_482.py CACHE_DIR TICKS_TSV OUTPUT_DIR
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from replay_engine import schema

SLOTS = ("FF", "LF", "RF", "LR", "RR")
ROUTE = "00000482--2b8fe366ed"

# Windows are in seconds since the first carState, using the recorded map publication time.
# The B3 windows bracket the two onset ticks. B4/B8 include the neighbouring samples
# whose publication times round to the endpoints in the written plan.
SCENES = {
  "B1": (7, [(3.38, 4.19)], (0, 1)),
  "B2": (11, [(4.50, 17.00)], (0, 2)),
  "B3": (12, [(38.14, 38.45), (52.59, 52.90)], (0, 1, 2)),
  "B4": (12, [(48.245, 48.356)], (0,)),
  "B5": (18, [(27.50, 33.90)], (1, 3)),
  "B6": (10, [(37.24, 41.85)], (2, 4)),
  "B7": (2, [(39.10, 41.40)], (2, 4)),
  "B8": (10, [(42.78, 42.90)], (4,)),
  "R1": (7, [(35.9, 48.0)], (2,)),
  "R2": (12, [(16.1, 21.5)], (2,)),
}


@dataclass(frozen=True)
class Draw:
  identity: int
  track: int
  x: float
  y: float
  s: float
  speed: float
  vision: bool = False


@dataclass(frozen=True)
class Radar:
  x: float
  y: float
  absolute_speed: float
  relative_speed: float


@dataclass(frozen=True)
class Evidence:
  track: int
  s: float
  radar: bool
  vision: bool
  lead: bool


@dataclass
class Tick:
  ns: int
  t: float
  speed: float
  old: tuple
  new: tuple
  raw: dict
  support: dict
  lead: tuple | None


def read_shadow(path):
  with path.open() as stream:
    return parse_shadow(stream)


def parse_shadow(lines):
  result = {}
  for line in lines:
    if line.startswith("# "):
      seg = int(line.rsplit("--", 1)[1])
      if seg in result:
        raise ValueError(f"repeated segment {seg}")
      ticks = result[seg] = {}
      continue
    stamp, valid, *fields = line.rstrip("\n").split("\t")
    slots = []
    for field in fields[:5]:
      if field == "-":
        slots.append(None)
        continue
      identity, track, x, y, s, speed, _, _ = field.split(",")
      slots.append(Draw(int(identity), int(track), float(x), float(y), float(s), float(speed)))
    objects = {}
    for obj in fields[5].split(";"):
      if obj:
        identity, track, s, _, _, _, _, radar, vision, lead = obj.split(",")
        objects[int(identity)] = Evidence(int(track), float(s), radar == "1", vision == "1", lead == "1")
    if int(stamp) in ticks:
      raise ValueError(f"repeated shadow stamp in segment {seg}")
    ticks[int(stamp)] = (valid == "1", tuple(slots), objects)
  return result


def read_segment(log, path, shadow):
  raw = {}
  recordings = {}
  counts = Counter()
  t0 = None
  speed = 0.
  lead = None
  for e in log.Event.read_multiple_bytes(path.read_bytes(), traversal_limit_in_words=2**50):
    which = e.which()
    counts[which] += 1
    ns = e.logMonoTime
    if which == "carState":
      t0 = ns if t0 is None else t0
      speed = e.carState.vEgo
    elif which == "radarState":
      p = e.radarState.leadOne
      lead = (ns, p.radarTrackId, p.dRel) if e.valid and p.status and p.radar and p.radarTrackId > 0 and p.modelProb >= 0.75 else None
    elif which == "liveTracks":
      points = {p.trackId: Radar(p.dRel, -p.yRel, p.vLead, p.vRel)
                for p in e.liveTracks.points if e.valid and p.measured and p.trackId > 0 and p.radarSource.raw == 0}
      raw[ns] = (speed, points, lead)
    elif which == "customReserved4":
      p = e.customReserved4
      counts["camera_frames"] += e.valid and p.status == 0
      counts["camera_boxes"] += len(p.boxes) if e.valid and p.status == 0 else 0
      counts["kind_answers"] += len(p.kinds) if e.valid else 0
    elif which == "customReserved3":
      m = e.customReserved3
      baseline = tuple(Draw(p.vid, 0, p.distance, p.lateral, p.distance, p.vRel, p.vision != 0)
                       if p.present else None for p in m.slots)
      if m.radarLogMonoTime in recordings:
        counts["repeated_map_stamp"] += 1
      recordings[m.radarLogMonoTime] = (ns, m.valid and e.valid, baseline)

  rows = []
  for ns, (pub, valid, old) in sorted(recordings.items()):
    if ns not in shadow or ns not in raw:
      counts["recording_without_radar_tick"] += 1
      continue
    new_valid, new, support = shadow[ns]
    if not valid or not new_valid:
      counts["recorded_invalid"] += not valid
      counts["engine_invalid"] += not new_valid
      counts["paired_invalid"] += 1
      counts["engine_shown_when_baseline_invalid"] += not valid and new_valid and any(new)
      continue
    if len(old) != 5:
      raise ValueError(f"invalid slot count for radar stamp {ns}")
    speed, points, lead = raw[ns]
    if lead is not None and ns - lead[0] > 180_000_000:
      lead = None
    rows.append(Tick(ns, (pub - t0) * 1e-9, speed, old, new, points, support, lead))
  counts["engine_ticks"] = len(shadow)
  counts["radar_without_recording"] = len(shadow.keys() - recordings.keys())
  counts["paired_valid"] = len(rows)
  if counts["engine_ticks"] != counts["liveTracks"]:
    raise ValueError(f"lost radar ticks in {path.name}")
  return rows, dict(counts)


def draws(rows, output, indices, identity=None):
  for row in rows:
    slots = getattr(row, output)
    yield row, tuple(slots[i] if identity is None or (slots[i] and slots[i].identity == identity) else None
                     for i in indices)


def continuity(rows, output, indices, identity=None):
  """Same ID/slot ≤150ms, or same-side F↔R ≤500ms, as in the original 482 analysis."""
  values = Counter({key: 0 for key in ("flickers", "same_id_jumps", "handoffs", "handoff_jumps", "bounces", "replacement_jumps", "handoff_residual_ge3m")})
  last_present = {}
  disappeared = {}
  last_id = {}
  last_handoff = {}
  last_distance = {}
  max_same = max_handoff = max_any = 0.
  for row, slots in draws(rows, output, indices, identity):
    for i, cur in zip(indices, slots):
      prev = last_present.get(i)
      if not cur and prev:
        disappeared[i] = row.ns
      if cur and not prev and i in disappeared and row.ns - disappeared[i] <= 1_000_000_000:
        values["flickers"] += 1
      last_present[i] = cur
      if not cur:
        last_distance.pop(i, None)
        continue
      if i in last_distance:
        dt = row.ns - last_distance[i]
        jump = abs(cur.x - prev.x)
        if 0 < dt <= 150_000_000:
          max_any = max(max_any, jump)
          if cur.identity and prev.identity == cur.identity:
            max_same = max(max_same, jump)
            values["same_id_jumps"] += jump >= 3.
          elif cur.identity != prev.identity:
            values["replacement_jumps"] += jump >= 3.
      last_distance[i] = row.ns
      before = last_id.get(cur.identity)
      if cur.identity and i != 0 and before and abs(before[1] - i) == 2 and before[1] != 0:
        t, _, old = before
        if 0 < row.ns - t <= 500_000_000:
          jump = abs(cur.x - old.x)
          values["handoffs"] += 1
          values["handoff_jumps"] += jump >= 3.
          expected = 0.5 * (cur.speed + old.speed) * (row.ns - t) * 1e-9
          values["handoff_residual_ge3m"] += abs(cur.x - old.x - expected) >= 3.
          max_handoff = max(max_handoff, jump)
          prev_handoff = last_handoff.get(cur.identity)
          values["bounces"] += prev_handoff is not None and row.ns - prev_handoff <= 1_000_000_000
          last_handoff[cur.identity] = row.ns
      last_id[cur.identity] = (row.ns, i, cur)
  return {**values, "max_same_id_jump_m": round(max_same, 3), "max_handoff_jump_m": round(max_handoff, 3),
          "max_adjacent_tick_distance_change_m": round(max_any, 3)}


def target(rows, output, identity):
  slots = Counter({slot: 0 for slot in SLOTS})
  samples = []
  ids = set()
  for row in rows:
    matches = [(i, slot) for i, slot in enumerate(getattr(row, output))
               if slot and (slot.identity if output == "old" else slot.track) == identity]
    slots.update(SLOTS[i] for i, _ in matches)
    if len(matches) == 1:
      ids.add(matches[0][1].identity)
    ffs = [slot for i, slot in matches if i == 0]
    if ffs:
      samples.append((round(row.t, 5), round(ffs[0].s, 3)))
  return {"slots": dict(slots), "missing_ff": len(rows) - slots["FF"],
          "missing_all_slots": len(rows) - sum(slots.values()), "object_ids": sorted(ids), "ff_range_samples": samples,
          "max_ff_step_m": round(max((abs(a[1] - b[1]) for a, b in zip(samples, samples[1:]) if b[0] - a[0] <= 0.15), default=0.), 3)}


def pair_ids(row, side):
  if row.speed <= 3.:
    return None
  side_points = sorted(((rid, p) for rid, p in row.raw.items()
                        if 1.8 <= p.y * side <= 5.2 and p.absolute_speed > 2. and 0.2 <= p.x <= 25.),
                       key=lambda point: point[1].x)
  if len(side_points) < 2:
    return None
  near_id, near = side_points[0]
  if near.x > 6.5:
    return None
  far = next(((rid, p) for rid, p in side_points[1:] if p.x >= max(7., near.x + 4.)), None)
  return (near_id, near.x, far[0], far[1].x) if far else None


def split_score(rows, side, output, fixed=None):
  front, rear = (1, 3) if side < 0 else (2, 4)
  result = Counter(ticks=0, far_matched=0, near_matched=0, both_matched=0, far_missing=0, near_missing=0)
  for row in rows:
    if fixed:
      near, far = (row.raw.get(i) for i in fixed)
      if near is None or far is None:
        continue
      pair = (fixed[0], near.x, fixed[1], far.x)
    else:
      pair = pair_ids(row, side)
      if pair is None:
        continue
    near_id, near_x, far_id, far_x = pair
    front_slot, rear_slot = (getattr(row, output)[i] for i in (front, rear))
    # The recorded slots have no raw track ID. Positive range distinguishes the
    # observed near object from an unrelated rear coasting picture.
    def match(slot, raw_id, x, tolerance):
      return bool(slot and abs(slot.x - x) <= tolerance and (output == "old" or slot.track == raw_id))
    found_far = match(front_slot, far_id, far_x, 3.5)
    found_near = match(rear_slot, near_id, near_x, 2.5)
    result.update({"ticks": 1, "far_matched": found_far, "near_matched": found_near, "both_matched": found_near and found_far,
                   "far_missing": not found_far, "near_missing": not found_near})
  return dict(result)


def observation_proxy(rows, index, output):
  """Use the same range/speed association for both drawings. rc18 has no object evidence flags."""
  counts = Counter(shown=0, no_match_or_camera_or_lead=0, stationary_radar_only=0)
  for row in rows:
    draw = getattr(row, output)[index]
    if draw is None or draw.identity == 0:  # A BSD symbol is not a vehicle object.
      continue
    counts["shown"] += 1
    radar_match = any(abs(draw.x - p.x) <= 2.0 and abs(draw.speed - p.relative_speed) <= 2.5
                      for p in row.raw.values())
    camera = draw.vision if output == "old" else row.support[draw.identity].vision
    lead_match = index == 0 and row.lead is not None and abs(draw.x - row.lead[2]) <= 2.5
    counts["no_match_or_camera_or_lead"] += not (radar_match or camera or lead_match)
    counts["stationary_radar_only"] += (index in (1, 2) and radar_match and not camera
                                       and row.speed > 5. and abs(row.speed + draw.speed) < 1.5)
  return dict(counts)


def segment_score(rows):
  result = {"rc18": {}, "engine": {}, "changes": {}}
  for index, name in enumerate(SLOTS):
    for label, output in (("rc18", "old"), ("engine", "new")):
      result[label][name] = {**continuity(rows, output, (index,)),
                             "observation_proxy": observation_proxy(rows, index, output),
                             "shown": sum(getattr(r, output)[index] is not None for r in rows)}
    c = Counter({key: 0 for key in ("newly_filled", "newly_empty", "new_coasted", "new_radar_only_stationary")})
    for r in rows:
      old, new = r.old[index], r.new[index]
      c["newly_filled"] += old is None and new is not None
      c["newly_empty"] += old is not None and new is None
      if new is not None and (old is None or abs(old.x - new.x) > 3.5):
        evidence = r.support[new.identity]
        c["new_coasted"] += not (evidence.radar or evidence.vision or evidence.lead)
        c["new_radar_only_stationary"] += (evidence.radar and not (evidence.vision or evidence.lead)
                                            and r.speed > 5. and abs(r.speed + new.speed) < 1.5)
    result["changes"][name] = dict(c)
  for label, output in (("rc18", "old"), ("engine", "new")):
    result[label]["sides"] = continuity(rows, output, (1, 2, 3, 4))

  lead_score = Counter({key: 0 for key in ("eligible", "engine_empty", "engine_other", "engine_correct")})
  examples = []
  for row in rows:
    if row.lead is None or row.old[0] is None:
      continue
    _, rid, dist = row.lead
    raw = row.raw.get(rid)
    if raw is None or abs(row.old[0].x - raw.x) > 2.0 or abs(dist - raw.x) > 2.5:
      continue
    lead_score["eligible"] += 1
    outcome = ("engine_empty" if row.new[0] is None else
               "engine_correct" if row.new[0].track == rid else "engine_other")
    lead_score[outcome] += 1
    if outcome != "engine_correct":
      examples.append([round(row.t, 3), rid, row.new[0].track if row.new[0] else None])
  result["lead_concordant_ff"] = dict(lead_score)
  result["lead_ff_difference_examples"] = examples
  return result


def scene_score(rows, scene, indices):
  stats = {label: continuity(rows, output, indices) for label, output in (("rc18", "old"), ("engine", "new"))}
  stats["ticks"] = len(rows)
  stats["occupancy"] = {
    label: {SLOTS[i]: sum(getattr(r, output)[i] is not None for r in rows) for i in indices}
    for label, output in (("rc18", "old"), ("engine", "new"))
  }
  if scene in ("B1", "B2", "B3", "B4"):
    vid, rid = {"B1": (1920, 34), "B2": (2702, 36), "B3": (5306, 40), "B4": (5306, 40)}[scene]
    stats["rc18_target"] = target(rows, "old", vid)
    stats["engine_target"] = target(rows, "new", rid)
    if scene == "B4":
      stats["engine_target"]["range_span_m"] = range_span(stats["engine_target"])
      stats["rc18_target"]["range_span_m"] = range_span(stats["rc18_target"])
  if scene in ("B5", "B6"):
    side, fixed = (-1, (41, 37)) if scene == "B5" else (1, None)
    stats["rc18_split"] = split_score(rows, side, "old", fixed)
    stats["engine_split"] = split_score(rows, side, "new", fixed)
  if scene in ("B7", "B8"):
    vid = 691 if scene == "B7" else 4513
    stats["rc18_witness"] = continuity(rows, "old", indices, vid)
    stats["timeline"] = [{"t": round(r.t, 4), **{
      label: [[s.identity, s.track, round(s.x, 3)] if (s := getattr(r, output)[i]) else None for i in indices]
      for label, output in (("rc18", "old"), ("engine", "new"))}} for r in rows]
  if scene == "B8":
    # The last measured return consistent with rc18 VID 4513 was track 43.
    # Keep its shadow trajectory even when the RR draw uses a different object.
    stats["engine_track_43"] = [
      [round(r.t, 4), identity, evidence.s]
      for r in rows for identity, evidence in r.support.items() if evidence.track == 43
    ]
  return stats


def range_span(target_stats):
  samples = [s for _, s in target_stats["ff_range_samples"]]
  return round(max(samples) - min(samples), 3) if samples else None


def summarize(segments):
  result = {}
  for label in ("rc18", "engine"):
    counters, proxy = Counter(), Counter()
    for seg in segments.values():
      for name in SLOTS:
        slot = seg[label][name]
        counters.update({key: slot[key] for key in ("shown", "flickers", "same_id_jumps", "replacement_jumps")})
        proxy.update(slot["observation_proxy"])
    result[label] = {**counters, "observation_proxy": dict(proxy)}
  for name in ("counts", "lead_concordant_ff"):
    counters = Counter()
    for seg in segments.values():
      counters.update(seg[name])
    result[name] = dict(counters)
  changes = Counter()
  for seg in segments.values():
    for slot in seg["changes"].values():
      changes.update(slot)
  result["changes"] = dict(changes)
  return result


def plot_scene(scene, seg, windows, samples, indices, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(len(indices), len(samples), figsize=(5.4 * len(samples) + 1.2, 1.9 * len(indices) + 1.4),
                           squeeze=False, layout="constrained")
  fig.suptitle(f"{scene}  |  Segment {seg}  |  Recorded rc18 and shadow engine", fontweight="bold")
  for col, (window, rows) in enumerate(zip(windows, samples, strict=True)):
    times = [r.t for r in rows]
    for row, i in enumerate(indices):
      ax = axes[row, col]
      series = [
        ("rc18", "#6d5f56", [r.old[i].x if r.old[i] else float("nan") for r in rows]),
        ("Engine", "#147d84", [r.new[i].x if r.new[i] else float("nan") for r in rows]),
      ]
      if scene == "B4":
        series.append(("Engine s", "#4d46a2", [r.new[i].s if r.new[i] else float("nan") for r in rows]))
      for label, color, distances in series:
        ax.plot(times, distances, ".-", color=color, markersize=3, linewidth=1.25, label=label)
      ax.set_xlim(*window)
      if i in (3, 4):
        ax.axhline(0., color="#a8adb1", linewidth=0.6)
      if not any(r.old[i] or r.new[i] for r in rows):
        ax.text(0.5, 0.5, "No slot output", ha="center", va="center", transform=ax.transAxes, color="#555")
        ax.set_yticks([])
      ax.set_ylabel(f"{SLOTS[i]} (m)")
      ax.ticklabel_format(axis="x", style="plain", useOffset=False)
      ax.grid(axis="y", alpha=0.23)
      if row == len(indices) - 1:
        ax.set_xlabel("Seconds from first carState")
      if row == 0 and col == 0:
        ax.legend(loc="best", frameon=False, ncol=len(series))
  fig.savefig(path, dpi=155)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("cache", type=Path)
  parser.add_argument("trace", type=Path)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  shadow = read_shadow(args.trace)
  log = schema()
  args.out.mkdir(parents=True, exist_ok=True)
  plots = args.out / "plots"
  plots.mkdir(exist_ok=True)

  replay = {}
  result = {"route": ROUTE, "rc18": "e4fe0dbf600cc53abfcebb8c921dad7742b58a33", "segments": {}, "scenes": {}}
  for seg in sorted(shadow):
    rows, counts = read_segment(log, args.cache / f"{ROUTE}--{seg}.capnp", shadow[seg])
    if any(seg == scene[0] for scene in SCENES.values()):
      replay[seg] = rows
    result["segments"][seg] = {"counts": counts, **segment_score(rows)}
  result["summary"] = summarize(result["segments"])
  for name, (seg, windows, indices) in SCENES.items():
    if seg not in replay:
      continue
    samples = [[row for row in replay[seg] if a <= row.t <= b] for a, b in windows]
    result["scenes"][name] = {"segment": seg, "windows": [
      {"start": a, "end": b, **scene_score(rows, name, indices)} for (a, b), rows in zip(windows, samples, strict=True)]}
    plot_scene(name, seg, windows, samples, indices, plots / f"{name}.png")
  (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()
