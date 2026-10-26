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

`lane_strip_v4.onnx` / `lane_strip_v4.json` (StripNet2 + central colour pool, 33,775 parameters, 143 KB; 10-24 retrain
adding the 10-24 rain drive, person-confirmed double labels only and synthetic doubles 0.05 (`rain1024b` D1 seed 2), sha256
`23ff1bc74942`; previously `e470e03852c0`) is the only StripNet model of the `CcncLaneTypeDetection` display mode
(`strip_model.StripModel`, manifest v4: raw YCbCr `N x 3 x 128 x 128` row-aligned 4.5-20 m strip over +-1.0 m, right strip
mirrored, outputs `color` white/yellow/blue and `type` dashed/solid/dbl_near/dbl_far/unknown logits; 0.90 display threshold).
Device cost, measured for this model (`claude/device1026`, offroad, OpenCV 2 threads): extract + infer median 14.7 ms, p95
15.6 ms per frame. The 10-26 4-type model (`decb4e8c966f`, autolabel1026 R1 large s2) was rolled back the same day
(dashed lines shown solid more often in the car; `claude/rollback1026`).
When its files are missing or fail the contract checks, a warning is logged and no lane type is published (the existing lane
display stays); there is no fallback model (the earlier v3 strip model and the model choice setting were removed 10-26). The
manifest version and sha256 prefix are logged and carried in each `customReservedRawData2` packet's `model` entry. The
manifest records the round-3 and 40f replay results and the device timing (`eval.device`). Training and evaluation:
[tools/lane_profile](../../../../../tools/lane_profile/README.md).
