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

`lane_strip.onnx` (StripNet, 21,532 parameters, 93 KB) is the frozen strip model used by the
`CcncLaneTypeDetection` (StripNet) display mode with `LaneStripModel = 0` (current). Input: raw YCbCr `N x 3 x 128 x 96`
row-aligned 4.5-20 m strip; outputs `line_type` dashed/solid and `color` white/yellow.
`lane_strip.json` pins its hash, input contract and the 0.90 display threshold. Training and
evaluation: [tools/lane_profile](../../../../../tools/lane_profile/README.md). Not device-measured.

`lane_strip_v4.onnx` / `lane_strip_v4.json` (not shipped yet) are the `LaneStripModel = 1` (new model) slot, same
contract checks (`strip_model.StripModel`, manifest v4 outputs allowed). When they are missing or fail to load,
`marking_daemon` logs a warning and runs `lane_strip.*`; the active slot, manifest version and sha256 prefix are
logged and carried in each `customReservedRawData2` packet's `model` entry.
