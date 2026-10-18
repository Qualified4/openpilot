# CCNC lane marking experiment

This is an experimental classifier and comparison workflow, not an enabled CAN
feature. `CcncLaneTypeDetection` defaults to OFF. ON starts an independent
`ccnc_lane_type` process; OFF imports no OpenCV engine and loads no model.
`modeld`, controls, Xiaoge and Hyundai CAN generation are unchanged.

The bundled FP32 model is a PC fine-tune of the official
[PP-LCNet-0.25x](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/ppcls/arch/backbone/legendary_models/pp_lcnet.py).
Its backbone weights are hash-pinned, with two new binary heads. Training uses
PyTorch only on the PC. Live inference uses the existing OpenCV DNN CPU backend,
one OpenCV thread, background CPU affinity and a 5Hz target. No new device
dependency is introduced. Architecture attribution and the Apache-2.0 license
are in `model.py` and `LICENSE.PaddleClas.txt`.

## Data and explicit review

Prepare a JSON list with `id`, `route`, `segment`, `split` (`train`,
`validation`, `test`), `log` and `video` for every segment. Assign entire routes
to a single split before sampling. This is stricter than segment-level splits
and prevents adjacent segments or the two sides of one frame leaking across
splits. Use full `fcamera` video where available; compressed `qcamera` is an
explicit domain limitation, not an equivalent full-resolution camera input.

From the repository root, with existing OpenCV, NumPy, Pillow, PyAV and capnp:

```sh
python -m tools.lane_type.dataset --sources sources.json --output private-data --times 5 15 25 35 45 55
```

The extractor joins encoded frame indices to camera EOF timestamps, modelV2
lane coordinates and calibrated camera intrinsics. A missing timestamp match,
invalid calibration or incomplete 6–46m geometry is rejected. `--include-missing`
retains unavailable geometry as unconfirmed observations for sequence analysis.
Both sides use the same 64x192 RGB strips in both classifiers. Conservative
silhouettes from existing `leadsV3` mask potential occlusions; they are not a
ground-truth visibility segmentation and cannot detect all unmodelled objects.

Review each ROI **and its full-frame context**, including the identity of the
model lane boundary. The generated `suggestion` is never a training label.
Only after visual verification, set `verified: true`, `type: dashed|solid`,
`color: white|yellow`, `label_source: visual_review`, `reviewer`, `review_note`
and `lighting: day|night`. Leave ambiguous, mixed-type, misprojected and hidden
boundaries unverified. Never infer "solid" just from a crop without a gap.
The pilot artifact uses single-assistant visual annotations, explicitly records
that fact, and has no independent human audit or accuracy approval.

Training refuses unreviewed truth, changed ROI hashes and route/pixel leakage.
Only reviewed, sufficiently observed, non-lane-change samples train the model.
The test partition is not used for checkpoint selection. Class imbalance, a
very small pilot dataset and lack of night labels remain material limitations.

## PC training, export and comparison

PC-only dependencies are PyTorch, ONNX and ONNX Runtime, in addition to the
extraction tools. psutil is used only for isolated PC benchmarks. They are not
added to the comma runtime. The official checkpoint URL and SHA-256 are pinned
in `model.py`; download it separately rather than executing a remote script.

```sh
python -m tools.lane_type.train --manifest private-data/manifest.json --pretrained PPLCNet_x0_25_pretrained.pdparams --output trained
python -m tools.lane_type.evaluate --manifest private-data/manifest.json --model-dir trained --output comparison.json
python -m tools.lane_type.evaluate --manifest private-data/manifest.json --model-dir trained --benchmark off --output pc-off.json
python -m tools.lane_type.evaluate --manifest private-data/manifest.json --model-dir trained --benchmark onnx --output pc-onnx.json
python -m tools.lane_type.evaluate --manifest private-data/manifest.json --model-dir trained --benchmark opencv --output pc-opencv.json
```

Run benchmark commands sequentially, each in a fresh process. These measure
classification of two precomputed ROIs at a 5Hz target, excluding common camera
reception/projection, and are **PC measurements**, not comma performance.
Precision/recall/F1 penalize abstentions as false negatives; zero ground-truth
support and unavailable scenario metrics remain null. Dense sequences must be
sampled at 5Hz before evaluating type transitions. A sparse sequence or no pair
of confirmed outputs cannot establish temporal stability.

OpenCV uses HSV paint masks, local road contrast and run lengths along the
same strip; it has no object detector or learned weights. Its scores are
heuristic evidence, while PP-LCNet softmax scores are also uncalibrated.

## Runtime, visibility and CCNC verification

The daemon samples shared NV12 planes directly into the small strips, without
copying or converting a full RGB camera frame. It skips inference during lane
changes or insufficient observations. A new result requires at least 32m of
usable strip and at most 15% conservative occlusion. Those thresholds are
experimental. Missing model/calibration/frame identity causes abstention.

Trusted head results may be held for at most 0.5s from their **original camera
timestamp**, independently for type and color. Insufficient observations do
not refresh that deadline. Turn signals, model lane changes, loss of tracking,
camera time discontinuities and boundary jumps clear the corresponding state.
Repeated stationary frames do not extend observed length. Spatial stitching
across moving frames is deferred until physical boundary identity and odometry
alignment can be validated; it could otherwise transfer paint to another lane.

The engine defaults to `onnx`. `CCNC_LANE_TYPE_ENGINE=opencv` selects the
comparison implementation when launching the daemon in a controlled bench
session. Use only one publisher at a time. Result packets are logged on
`customReservedRawData2` with side-specific confidence, visibility and timing.

`display.DisplayPolicy` takes a **copy** of baseline display values and never
publishes CAN. It proposes white-solid 2, orange-solid 4 for recognized yellow,
white-dashed 8 and hidden 1 only in the enabled Standard/Refined preview.
Both type and color must be confirmed independently; an expired head hides the
preview. Two fresh consistent observations, normally 0.2s apart at 5Hz, confirm
or switch a head. A trusted result expires within 0.5s of its original evidence;
pending or held results do not extend that deadline. Boundary identity changes
reset the display history before confirming a replacement.

Dashed candidates additionally require three paint runs and two internal gaps,
with no modelled occlusion. This mask evidence remains unvalidated: road shadows
produced False Dashed cases in the dense offline test. Existing departure,
blind-spot and lane-line warnings preserve the baseline code and blink phase,
including hidden; missing warning context blocks a new preview code. Recognized
yellow and warning orange have different reasons in the log. Actual Sonata hide
and dashed support is unverified; 10/14 have no verified DBC meaning and are not
selected as new codes. Production always uses its existing display.

Before any future live integration, require independent label review, a much
larger route-disjoint test including night/occlusion and marking transitions,
confidence calibration, a paired same-input ON/OFF replay on the actual comma,
and separate Sonata cluster encoding verification on an isolated bench.
The bundled manifest keeps accuracy/device validation and CAN transmission false.

See [the measured pilot report](../../docs/ccnc_lane_type_20261018.md).

The subsequent full-log/ROI audit is in
[the retraining and ROI report](../../docs/ccnc_lane_roi_retrain_20261018.md).
Use `retrain_data.py` for full-timeline exact-frame matching and feature-diverse
review proposals, `roi_quality.py` for range/paint/warp diagnostics,
`compare_rois.py` for frozen-model comparisons, and `evaluate_display.py` for
False Dashed, orange errors, confirmation coverage and dense transitions.
Route splits are locked before review; paired recordings share a scene identity.
Proposals are not ground truth. A dataset containing `ROI_REVIEW_HOLD.json` is
blocked by `train.py` until its documented ROI-quality issues are resolved.

Tests without a built driving runtime:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c tools/ccnc/pytest.ini --noconftest tools/lane_type/test_lane_type.py
```

The subsequent GPU comparison is **quarantined by a pixel-quality hold**.
See [the pixel audit and Korean comparison report](../../docs/ccnc_lane_pixel_audit_20261018.md).
`audit_sync.py` verifies exact encode/model joins against freshly decoded video;
`audit_pixels.py` saves source-coordinate correspondence, interactive inspection,
partial paint traces and codec-free counterexamples. `pixel_ribbon.py` preserves
ragged original pixels without recentering, resizing or padding; it deliberately
does not classify unverified paint/gaps or feed the runtime model.

`prepare_dual_data.py`, `native_data.py` and `admit_labels.py` preserve proposal
and review provenance. Unreviewed automatic labels were not admitted.
`train_compare.py` supports four CUDA models with separate colour/type inputs,
but rejects manifests beneath `ROI_REVIEW_HOLD.json` before CUDA/output setup.
All existing checkpoints and ONNX files stay intact. `temporal.py` is an offline
3-second distance-bin prototype, with unqualified curved-road alignment; the
production daemon does not import it. `report_comparison.py` regenerates the
frozen diagnostic tables without retraining or choosing models from test scores.

Use `.analysis/lane-model-comparison-20261018/REPRODUCE.ps1` for audit/test commands.
GPU training stays blocked until target-paint geometry, visibility and the new
input contract are independently qualified. No experimental model is installed.

`pixel_tracker.py` adds offline target-paint selection to Pixel Ribbon v2:
colour-separated metric path fits, double-boundary members, ambiguity rejection
and a three-second world-boundary map aligned using SE(2) ego motion. Unobserved
pixels never establish a gap; gaps require translated views and static-road flow.
`validate_pixel_tracking.py` reuses five reviewed HD clips and synthetic controls.
See [the target-paint tracking report](../../docs/ccnc_pixel_ribbon_tracking_20261018.md).
Real image-flow/ego alignment remains unqualified; training and promotion stay held.

The follow-up [white-dash and physical-alignment audit](../../docs/ccnc_pixel_ribbon_alignment_20261018.md)
keeps the same 80 source frames. `audit_pixel_alignment.py` audits source trace
gates, exact adjacent frames, calibrated gyro/sample-time integration, identifiable
paint endpoints and road corners. Road observations can retain their height under
full measured attitude instead of replacing it with each new model height.
Long-solid flow is no longer reported as physical accuracy. Real dashed confirmation
and broad physical registration remain unqualified; this code is offline only.
Use `.analysis/pixel-ribbon-alignment-20261018/REPRODUCE.ps1`; no training is launched.

The [hybrid cadence comparison](../../docs/ccnc_pixel_ribbon_hybrid_20261018.md)
adds `hybrid_tracker.py`: 5Hz precise target selection and 20Hz bounded native-pixel
tracking, with unique static-feature checks before spatial evidence admission.
`validate_hybrid_tracking.py` compares it with both precise cadences on the same
recorded NV12-plane inputs. `extract_all_lane_candidates.py` inventories all videos
under OpenpilotLogs, including the added HD folder, and writes unlabelled native
review candidates and explicit per-video exclusions. These candidates are not
training truth. All runtime, model, CAN and debug integration remains unchanged.
