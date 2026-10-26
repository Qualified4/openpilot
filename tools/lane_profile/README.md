# Lane boundary profile scanner (offline prototype)

Separate from `openpilot/selfdrive/carrot/lane_type` and `tools/lane_type`; imports them read-only.
No runtime, display, CAN, ONNX or training change. Not wired to any process.

- `scanner.py` — re-exports the runtime scanner `openpilot/selfdrive/carrot/lane_type/profile_scanner.py` (moved unchanged). Per frame: row-aligned 4.5–20 m strip on the current modelV2 boundary (one row per
  integer source row, metric columns), per-row paint / road / unknown, 1-D accumulation indexed by
  `odometer + row distance` (0.25 m bins, small correlation correction). No SE(2)/3D pose, optical flow
  or registration.
  - dashed: a confirmed-road run ≥ 1 m between two dash-sized (≥ 1.5 m) paint groups; road bins must be
    seen ≥ 2 times after ≥ 1 m of travel. Occlusion, shadow bands, texture, weak/worn paint, transverse
    paint and missing paths are UNKNOWN, never a gap.
  - solid: the latest continuous paint run ≥ `solid_m` (12 m: train-route dash lengths, 98.4 % ≤ 12 m).
  - colour: near (≤ 12 m) paint chroma relative to road, `dCr − dCb` ≥ 4.5 yellow, ≤ 2 white (train routes).
- `labels.py` — interval ground truth schema (`ccnc-lane-interval-v1`); only `human_confirmed`/`ai_reviewed` count.
- `run_clips.py`, `evaluate_clips.py` — frozen 5-clip 20 Hz comparison with Pixel Ribbon results and the active ONNX.
- `synthetic.py` — the Pixel Ribbon reports' synthetic scenarios, compared with their stored results.
- `run_logs.py`, `analyze_logs.py` — every synced frame of every recorded segment; dash length distribution;
  reviewed-frame accuracy (train/validation/test) against the active ONNX.
- `auto_label.py` — automatic interval labels: auto-approval candidates (`auto_proposed`, training only, never
  ground truth), a short exception review queue (strong-evidence conflicts + random audits) and deferred intervals.
- `make_review.py` + `review_page.html` — exception review page: plays the original video (H.264 copy, frame
  index preserved), marks the boundary under review, timeline of scanner verdicts, split at a marking change,
  keyboard control, autosave/restore, import/export as `human_confirmed` interval labels.
- `serve_review.py` — localhost static server with HTTP Range (video seeking) for a review folder.

Run from the repository root with the project OpenCV on `PYTHONPATH`
(see `.analysis/lane-profile-20261018/REPRODUCE.ps1`).

Runtime integration (display only): `openpilot/selfdrive/carrot/lane_type/marking_daemon.py` runs the option's
recogniser (`CcncLaneTypeDetection` 1 Scanner / 2 StripNet), `strip_model.py` + `assets/lane_strip_v4.onnx` hold the
frozen StripNet, card reads the packet via `lane_type/marking.py`, and `opendbc/car/hyundai/ccnc_lane_marking.py`
maps it onto the CCNC lane code. `replay_marking.py` replays logs through that exact path.
