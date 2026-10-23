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

`lane_strip_v4.onnx` / `lane_strip_v4.json` (StripNet2 + central colour pool, 33,775 parameters, 143 KB; 10-23 evening retrain
with the label fixes, the 10-23 drives and Road Viewer bookmarks (`today1023b` negsyn0.25 seed 1), sha256 `e470e03852c0`; previously `c621e1e41cab`) are the `LaneStripModel = 1` (new model) slot, same contract checks (`strip_model.StripModel`,
manifest v4: raw YCbCr `N x 3 x 128 x 128` over +-1.0 m, right strip mirrored, outputs `color` white/yellow/blue and
`type` dashed/solid/dbl_near/dbl_far/unknown logits). A stored `LaneStripModel = 2` (the removed hybrid option) runs slot 1. When the v4 files are missing or fail to
load, slot 1 logs a warning and runs `lane_strip.*`; the active slot, manifest version and sha256 prefix are
logged and carried in each `customReservedRawData2` packet's `model` entry. The manifest records the round-3 and 40f
replay results; neither model is device-validated.
