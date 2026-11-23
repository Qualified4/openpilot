"""Write fixtures/wide_rules_c1.txt: the 58 labeled wide scenes through the offline camera rules (item 68 pipeline.py).

The rules run as the device runs them: 1 Hz frames (`step5_lrrr.every(1.0)`), `pipeline.TIME_1HZ`, the logged 0x162 slot
0.5 s after the frame, and only the two side views (`ccnc_visiond` makes no raw-frame boxes). Radar points farther than
40 m or 15 m to the side are left out of the file; the expected rows are computed from the same pruned points.

Needs the det-ready scripts and caches (not in this tree). Run with the det-bench Python:
  python wide_rules_fixture.py [OUT]

Line format (whitespace separated, floats as Python repr):
  scene I KEY SIDE ROLL PITCH YAW WROLL WPITCH WYAW HEIGHT
  frame T_NS V GUARD LPROB0..3|- LF_DET LF_DIST LR_DET LR_DIST RF_DET RF_DIST RR_DET RR_DIST
  track ID DREL YREL VREL
  box VIEW X1 Y1 X2 Y2 SCORE CLASS
  expect SLOT ON ACT X YC CUT VIEW      (SLOT F or R, ACT -|fill|swap|keep, X YC CUT VIEW - without a candidate)
"""
import os
import sys

WORK = "/home/orca/.orca/bots/pilo/work"
sys.path.insert(0, f"{WORK}/det-ready-20261121/scripts")
import evaluate as E  # noqa: E402
import pipeline as P  # noqa: E402
import step5_lrrr as L  # noqa: E402

C1 = dict(tm=P.TIME_1HZ, sub=L.every(1.0), lat=0.5)
SLOTS = ("LF", "LR", "RF", "RR")


def side_only(R):
  out = {}
  for key, frames in R.items():
    out[key] = {}
    for n, fb in frames.items():
      fb = dict(fb)
      fb["dets"] = [d for d in fb["dets"] if d["m"] != "raw"]
      fb["tracks"] = [t for t in (fb.get("tracks") or []) if abs(t[1]) <= 40 and abs(t[2]) <= 15]
      out[key][n] = fb
  return out


def num(x):
  return repr(float(x))


def main(path):
  R, S = side_only(E.load_results()), E.scenes()
  lines = []
  for i, r in enumerate(S, 1):
    key, side = r["key"], r["side"]
    _, rpy, h, _, wfd = E.cache(key)["calib"][-1]
    lines.append(" ".join(["scene", str(i), key, side, *map(num, rpy), *map(num, wfd), num(h)]))
    F = R[key]
    ns_all = sorted([n for n in F if r["n0"] <= int(n) <= r["n1"]], key=int)
    ns = C1["sub"](ns_all, F)
    tall = [int(F[n]["t"]) for n in ns_all]
    rows = {row["n"]: row for row in L.run_scene(R, r, True, **C1)}
    assert list(rows) == ns
    for n in ns:
      fb = F[n]
      t = int(fb["t"])
      fa = F[ns_all[min(range(len(tall)), key=lambda k: abs(tall[k] - t - C1["lat"] * 1e9))]]
      s162 = fa["s162"] if isinstance(fa["s162"], dict) else {}
      slots = []
      for s in SLOTS:
        slots += [num(s162.get(f"{s}_DETECT", 0) or 0), num(s162.get(f"{s}_DETECT_DISTANCE", 0) or 0)]
      guard = not key.endswith("--0") or E.vmax_before(key, t) >= 5.5
      lp = fb.get("lprob")
      lprob = [num(p) for p in lp] if lp is not None else ["-"]
      lines.append(" ".join(["frame", str(t), num(fb["v"]), str(int(guard)), *lprob, *slots]))
      lines += [" ".join(["track", *map(num, tr[:4])]) for tr in fb["tracks"]]
      lines += [" ".join(["box", d["m"], *map(num, d["box"]), num(d["s"]), str(int(d["c"]))]) for d in fb["dets"]]
      for slot in ("F", "R"):
        o = rows[n][slot]
        c = o["c"]
        pos = ["-"] * 4 if c is None else [num(c["x"]), num(c["yc"]), str(int(c["cut"])), c["m"]]
        lines.append(" ".join(["expect", slot, str(int(o["on"])), o["act"] or "-", *pos]))
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
  print(path, len(S), "scenes", sum(1 for x in lines if x.startswith("frame")), "frames")


if __name__ == "__main__":
  here = os.path.dirname(os.path.abspath(__file__))
  main(sys.argv[1] if len(sys.argv) > 1 else f"{here}/fixtures/wide_rules_c1.txt")
