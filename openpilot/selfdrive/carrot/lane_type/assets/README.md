# Lane classifiers

## Active StripNet asset

`lane_strip_v4.onnx` / `lane_strip_v4.json` is the single small U s0 model
selected by the owner on 2026-11-08: 34,079 parameters, SHA256
`7fb3d57a6b5dba4ca1dbfa72b6b7b07c7bb2d5d745a3eecfc466dc2d3c71ff0b`.
It replaces N s0 (`f452fa69a522`); it is not the two-network comparison candidate.

Manifest v4: raw YCbCr `N x 3 x 128 x 128`, 4.5–20 m row-aligned strip over
±1.0 m, right strip mirrored. The type head has dashed/solid/dbl_near/dbl_far/unknown;
the colour head is trained on white/yellow and pads blue with -10000. Blue paint
is labelled white. The 0.90 threshold, A8/4L filter and display policy are retained.

Training adds the November 6–7 normal and bookmark rows to the N data, with one
whole drive excluded from training. The same seed-0/40-epoch recipe selects epoch
28 on the existing validation set. General display error and blue-paint colour
recognition improve in the recorded evaluation, with known regressions in 417/412
and more short display roundtrips. This is not road acceptance or calibrated confidence.
See [the replacement record](../../../../../docs/lane_model_u_s0_20261108.md).

Offroad comma timing for this exact model: OpenCV 2 threads, extract + infer
median 14.73 ms for both lane sides. This excludes the remaining vehicle pipeline
and does not establish onroad timing or accuracy acceptance.

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
