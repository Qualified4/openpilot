# Lane classifiers

## Active StripNet asset

`lane_strip_v4.onnx` / `lane_strip_v4.json` is the single small UB s0 model
selected by the owner on 2026-11-08: 34,079 parameters, SHA256
`03e5f51537013160cb33d88f6694edda534baf7df49ee7ca563c8686e7d4b2ad`.
It replaces U s0 (`7fb3d57a6b5d`).

Manifest v4: raw YCbCr `N x 3 x 128 x 128`, 4.5–20 m row-aligned strip over
±1.0 m, right strip mirrored. The type head has dashed/solid/dbl_near/dbl_far/unknown;
the colour head is trained on white/yellow and pads blue with -10000. Blue paint
is labelled white. The 0.90 threshold, A8/4L filter and display policy are retained.

Training adds 116 manually confirmed white-dashed left-lane frames from route44e,
segment8 to the U data. Same small recipe/initializer, seed0 and40 epochs; epoch30
selected on unchanged validation data. Route454 remains withheld from training.
The new bookmark and recent-drive final output improve, with mixed regressions
on older scenes. Raw fine-type accuracy and final display accuracy are distinct.
See [the replacement record](../../../../../docs/lane_model_ub_s0_20261108.md).

The experimental graded latch, persistence and opposing-evidence policies are
not used. This deployment has no new device timing or actual road acceptance claim.

Missing or invalid assets log a warning and publish no lane type, retaining the
existing display. There is no fallback model or runtime model selector. Version
and model SHA are logged and included in `customReservedRawData2.model`.
Training tools: [tools/lane_profile](../../../../../tools/lane_profile/README.md).

## Legacy experimental PP-LCNet assets

The following describes `lane_type.onnx` / `manifest.json`, not the active
`CcncLaneTypeDetection` StripNet path.

Experimental FP32 PP-LCNet-0.25x lane classifier, 245,308 parameters.

Input: RGB/ImageNet-normalized `N x 3 x 192 x 64`, `N=1` or `2` live sides.
Outputs: `line_type` = dashed/solid and `color` = white/yellow probabilities.

`manifest.json` pins the exact ONNX/checkpoint/data provenance. The model was
actually fine-tuned on a small, single-assistant visually reviewed dataset.
It is not accuracy-approved, confidence-calibrated or comma-validated, and
cannot enable actual vehicle CAN output. It is unloaded while the option is OFF.

Training/reproduction: [tools/lane_type](../../../../../tools/lane_type/README.md).
Architecture/weight origin: official PaddleClas PP-LCNet-0.25x, Apache-2.0;
see [the license](../../../../../tools/lane_type/LICENSE.PaddleClas.txt).
