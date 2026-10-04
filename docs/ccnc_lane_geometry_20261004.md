# CCNC lane-change display geometry (2026-10-04)

## Applied sequential runtime and curvature revision (2026-10-04)

Following the five-perspective review, the user authorized implementing and
comparing the stages in order. Both cereal-incompatible slices in `road_curve`
and `observe_motion` now use integer indices. An absent inner-lane pair retains
the Position estimate. A fresh model's expected data/calculation exception now
sets `target=None`, retains the existing bounded 0.35 s curvature hold/decay,
and cannot renew `valid_time` or leave a stale target marked as a new result.
The explicit disabled-lateral steering-based output remains unchanged.

Original, pre-fix, runtime-only and five estimator candidates were compared on
105 unique segments /124,896 ADRV_0x161 updates with actual cereal readers.
The pre-fix source raises 2,904 road-curve and 5,265 motion exceptions; both
counts are zero in the runtime-only and final variants. These are exception
invocations, not distinct faulty display frames. The runtime repair restores
motion history and removes recognition-data-present position backups caused
by slicing.

The final estimator keeps Position as its primary source and retains the
existing two-second farther search, uncertainty limit, quadratic safeguard,
time filter and curve cap. During a maneuver, when near Position and both
geometrically consistent inner lanes disagree by >0.5 code over the same
0–30 m interval, use the near road curve directly, capped at +/-15. The old
clamp of a farther estimate to local +/-one code is removed. Lane probabilities
do not gate this comparison. Normal non-maneuver calculation is unchanged;
the existing filter may carry a revised target briefly past maneuver end.

Removing the farther search is rejected: synthetic straight-road lane-change
residual grows from at most0.624 to5.591 codes in the tested speed/duration
cases. Removing the quadratic safeguard, or always using near lanes, is also
rejected because confidently bending Position with wrong straight lanes becomes
zero; always using lanes additionally loses a genuinely upcoming bend.

Across 2,201 model-reported start/finish observations in19 segments, mean code
disagreement with the near-lane model proxy decreases0.652 ->0.500 from the
runtime-only baseline. This proxy is not independent road ground truth. In
lat-enabled near-straight observations, outputs exceeding one code decrease
13 ->6; in near-curved observations, zero outputs decrease488 ->277. Outputs
must still be checked against physical road/cluster behavior rather than
optimizing this model-derived agreement alone.

Actual-object regressions cover builder and serialized reader, curve/motion
paths, failed fresh-model hold/expiry/recovery and same-interval maneuver
selection, in addition to existing synthetic geometry and option checks.
728 tests pass. Final-source replay matches the selected candidate on all
124,896 updates, including stock/default startup-field retention. ModelLanes
OFF ADRV_0x161 packets are byte-identical, and runtime-only/final output fields
other than the two curvature fields are identical on all those updates.
Three alternating-order desktop timing passes measure lane-update-only median
0.1083 ->0.2555 ms in the target automatic maneuver: restored calculation is
more expensive than the former exception shortcut. The estimator change after
runtime repair adds only0.0015 ms in that measurement. No device-CPU claim follows.
Private stages, source snapshots, samples, candidate rejection evidence and
comparisons are archived in
`.analysis/archive/2026-10-04/ccnc-geometry-stages/`. The review below documents
the pre-fix discovery; its statements about no production edits apply to that
earlier review stage.

## Real cereal object review correction (2026-10-04)

A subsequent curved-road lane-change review found that `road_curve()` uses
`md.laneLines[1:3]`, but cereal's `_DynamicListReader` does not support Python
slicing. The resulting TypeError is caught by `update_lanes()`, which emits
steering-angle-based curvature. Previous namespace/list-based replay and mock
tests did not exercise this runtime behavior. Their successful estimator output
must not be treated as validation of real-object execution.

Actual-object replay against original `6334db9d` confirms that nonzero curvature
alone does not establish correct road-curve retention: steering fallback can
weaken the late-maneuver display. An analysis-only indexed-access variant retains
nonzero model curvature, while the farther Position fit and local +/-one-code
guard can still produce a weaker curve than the original. The original estimator
also includes heading/maneuver effects, so matching its amplitude is not proof
of road accuracy. No production code is changed by this review. Fix list access
and add real cereal-object coverage before evaluating further estimator changes.

Private source snapshots, actual-object replay, road-video alignment, comparison
plots and the Korean report are retained in
`.analysis/archive/2026-10-04/ccnc-3c2-curvature/`. Raw captures stay outside Git.

## Follow-up correction after synchronized road-video review

The first lane-only implementation (29ad072d, merged in d5f14561) could turn a
real road bend into a straight display when lane probabilities dropped during
lane changes. Reduced curvature amplitude in the initial three-log replay did
not establish improved road accuracy. This revision restores the model driving
path as the curvature source and corrects the lane-position transition.

### Curvature

- Fit a quadratic to five equally spaced position samples. Constant lateral
  offset and linear heading cancel; the slope denominator retains geometric
  curvature scaling. Preserve the 1800 scale, sign and CAN encoding.
- Normal driving uses the existing speed-dependent 30–80 m horizon. During a
  lane change, extend it by two seconds of measured travel and fit its farther
  half, starting at least 20 m ahead. Clip the horizon to available trustworthy
  position data. This reduces near maneuver contributions; it does not exactly
  separate arbitrary lane-change curvature from road curvature.
- Preserve the original position yStd limit of 0.8. Lane probability and road
  edge uncertainty do not gate this estimator, including during lane changes.
- Keep the 0.25 s filter, maximum 10 codes/s, and bounded retention for missing,
  stale or unusable position data. Ordinary low lane confidence no longer
  triggers decay toward straight.

### Lane-position transition

- Evaluate model freshness even when lateral control is disabled. The previous
  implementation passed None to the geometry tracker in that mode, preventing
  fresh-model release evidence from accumulating although position inputs arrived.
  The disabled-lateral curvature still comes from the steering angle as before.
- Retain phase detection and the two-distinct-model release condition. Require
  a preceding boundary distance below 0.5 m for the rebound-trigger branch;
  a distance below 0.1 m still directly triggers. A distant recognition rebound
  alone must not reassign displayed lanes. Early physical index reassignment
  without a near-boundary observation remains ambiguous.
- Estimate pre-trigger lateral motion from the two inner lanes' near tangents
  multiplied by measured speed, rather than differences between relabelled
  lateral offsets. Use a quadratic tangent estimate at 0/2.5/5 m. A parallel
  sweep changes offsets without changing this heading estimate; deformed lane
  shapes can still corrupt it.
- Collect at most eight fresh observations with at least 150 ms of history.
  Require bounded observation gaps, median approaching speeds of 0.1–1.5 m/s,
  side-to-side agreement within 0.4 m/s, and within-window speed spread <=0.6 m/s.
  Reversed, stale or inconsistent evidence gives no continuation.
- Continue only inside the existing hold, fading speed to zero over 0.6 s and
  limiting displacement to 0.45 m. Never update the prediction with hold data.
  The revised bound covers the observed 0.30–0.55 s automatic holds; it is not
  permission to extrapolate indefinitely through recognition loss or cancellation.
- Preserve gain ramping, post-transition 3 m/s limiter, direction/gap resets,
  cancellation handling and CAN position limits. Saturation/quantization and
  insufficient history can still produce a visible pause.

### Same-input comparison

Input: 27 rlog/qcamera segment pairs supplied for the follow-up review. All were
recorded on d5f14561. Comparison variants are original 6334db9d, the committed
lane-only version, and this revision. There are no separate before/after vehicle
runs. Replay uses the latest valid model within 0.5 s, matching card's model
alive budget, and calls the display at recorded 0x161 send times. Packet-sampling
and event-order differences prevent claiming bit-exact reconstruction of the
recorded full vehicle session.

Eight model-reported automatic lane changes cover 850 display samples. A change
across consecutive segments19/20 is counted once. Seven representative videos
have 1,200 frames and 1,200 encode indices each; relative timing disagreement is
less than0.012 ms. Display values are compared with the synchronized road video,
not footage of the physical cluster or independently measured road curvature.

| Check | Committed lane-only version | Revision |
|---|---:|---:|
| Curved-road change, A/3 around35 s | curve0 | curve-6 |
| Same curve around36 s | curve0 | curve-5 |
| Curved-road change, B/16 around52 s | curve0 | curve-2 |
| Hold continuation in8 automatic changes | 0/8 | 7/8 |
| Qualified continuation displacement | 0 m | 0.162–0.279 m |
| B/24 before U-turn: hold around21.85 s | 5.95 s | 0.30 s |

A/3 still has no motion history at its trigger and retains a short hold. B/19
had a premature distant-boundary rebound transition; the revision delays it
until near the actual crossing and obtains 0.170 m of continuation. Normal
curve A/12 around57 s remains-13; around59 s it is-10 versus the committed-8
and original-10. These are display codes, not calibrated road-radius estimates.

### Validation and remaining limits

- 699 focused CCNC/settings/Wiki tests pass, covering malformed and stale path
  data, zero lane confidence, distant straight lane polynomials on a curved path,
  affine translation/heading,24 parallel sweep schedules, near-heading motion,
  cancellation/gaps, disabled-lateral release and distant recognition rebounds.
- Added smooth quintic lateral-maneuver checks on straight roads: residual raw
  curvature maxima are approximately0.624 codes for90 km/h over6 s,0.049 for60 km/h
  over4 s, and0 for50 km/h over3 s. Curved lane-change components therefore
  remain; the earlier claim of exact straight-road cancellation applies only
  to affine synthetic paths, not a full maneuver.
- On the independent4,155-frame input set, complete CAN output with model lanes
  OFF matches original 6334db9d in all8 combinations (33,240 comparisons).
  Vehicle-display packets match in all16 combinations (66,480 comparisons).
- B/24's model recognition error is not repaired. Its long display lock is
  corrected, and disabled-lateral steering-based curvature remains unchanged.
- No steering, radar classification, option defaults, trailer/warning precedence
  or Panda forwarding policy is changed. Normal model-lane curvature does change.
  Physical cluster animation, timing and road accuracy require vehicle validation.

Private route mappings, scripts, source snapshots, summaries and comparison
images are retained locally in `.analysis/archive/2026-10-04/ccnc-revision/`.
Original first-revision evidence remains in `ccnc-lanes/`; the follow-up diagnosis
is in `ccnc-road-review/`. Raw logs and videos are not committed or uploaded.

### Final refinement and direct comparison against original 6334db9d

Position remains the primary curvature source. During lane changes only, a
nearby-lane consistency check limits residual maneuver curvature. Fit position
and both inner lanes over 0–30 m; require valid increasing geometry, width
2.3–4.8 m at all five samples and lane curvature agreement within three codes.
Do not gate this check on lane probability. A full position quadratic fit with
maximum residual <=0.02 m bypasses the guard, preserving consistent true bends
even when nearby lanes incorrectly appear straight. Otherwise, when near
position differs from mean local lane curvature by >0.5 code, bound the far
estimate to local curvature +/- one code. For local magnitude <0.5 code, use
local directly. Invalid or inconsistent lane evidence retains position curvature.

This is a display heuristic, not exact maneuver/road separation. Incorrect
nearby lanes with a nonquadratic position path remain ambiguous. A genuinely
upcoming bend is retained when nearby position and lanes agree. Tests preserve
true quadratic road bends with wrong straight lanes and zero-confidence curved
maneuvers. The synthetic 90 km/h case can still quantize to one code.

Hold progress now preserves the original 0.1/0.2 m release-evidence steps as a
minimum, taking the greater of that progress and qualified prediction. Missing
motion history therefore no longer delays the original release in A/3.

| Check | Original | Final correction |
|---|---:|---:|
| A/3: first position movement after hold trigger | 0.256 s | 0.256 s |
| A/9: first position movement | 0.449 s | 0.103 s |
| A/18: first position movement | 0.496 s | 0.147 s |
| B/24 before U-turn: first position movement | 0.255 s | 0.153 s |
| B/24 hold duration | approximately0.30 s | approximately0.30 s |
| A/3 lane-change curvature range | -5..+1 | -5..0 |
| A/18 lane-change curvature range | -6..0 | -5..0 |
| A/19–20 lane-change curvature range | -1..+1 | 0..+1 |
| B/11 lane-change curvature range | 0..+5 | 0..+5 |
| A/12 normal curve around57/59 s | -13 / -10 | -13 / -10 |

A/3 release-position samples match the original around the transition. The
opposite +1 in A/18 and +2 peak in A/19–20 from the earlier follow-up are gone.
B/19's hold begins near the boundary instead of a premature distant recognition
rebound, but lasts approximately0.55 s versus original0.50 s. B/24's recognition
error is not repaired: the first revision's long lock is eliminated, restoring
the original short hold. Curvature amplitude is not independently validated.

Three paired desktop timing passes, including Python CAN packing, give median
total call time215 ->259 microseconds with model lanes alone and612 ->653
microseconds with model lanes plus radar vehicles. The correction costs about
0.04 ms per call; these are not device timings.

Final scripts, source snapshots, summaries, output rows and seven synchronized
comparison images are local in `.analysis/archive/2026-10-04/ccnc-refine/`.
Earlier snapshots remain unchanged in `ccnc-revision/`. No raw capture is
committed or uploaded. Physical cluster animation remains unvalidated.

### Road-video comparison: unresolved trusted-prefix rejection

Subsequent review compared original `6334db9d`, pre-sequential-fix `7f004d68`
and the current working tree against road-video frames. Of 105 unique replayed
segments, 102 have corresponding video; 25 videos and 29 selected windows were
visually compared, not all available videos in full. Seven overlaid comparison
clips were rendered and all 981 encoded frames decoded successfully. These
are replay outputs over road video, not recordings of the physical cluster.

Some lane-change windows restore the original bend direction or reduce false
curvature, but a sustained right bend still becomes straight in both the
pre-sequential and current versions while the original retains its bend.
`road_curve` validates monotonic x over the entire Position before truncating
it by uncertainty. In this case untrusted distant points fold in x while the
trusted approximately 35–40 m prefix remains valid and strongly curved. The
whole-path rejection loses that usable prefix; bounded hold then decays to zero.
This is an unresolved regression from the earlier geometry revision, not a
low-lane-probability event or a new direct-local lane-change regression.

A follow-up should validate the trusted prefix in the display-only curvature
path before rejecting its geometry, preserving uncertainty, minimum-span and
invalid-prefix rejection. Do not broadly relax shared boundary validation.
Compare a valid prefix with an untrusted folded tail against prefix-only input,
and retain rejection for genuinely invalid trusted geometry. No additional
production change was made during this video review. Video supports bend
direction and straightening observations, not exact curvature amplitude.

Private case details, synchronized clips, posters, diagnosis and scripts are in
`.analysis/archive/2026-10-04/ccnc-video-review/`. The current revision cannot yet
be described as uniformly better than the original; physical cluster behavior
and numerical road-curvature accuracy remain unvalidated.

### Trusted-prefix fix and full-log verification

The display curvature path now checks array dimensions/lengths and finite
uncertainty, truncates Position before its first yStd >0.8 sample, then validates
only that trusted prefix. Invalid trusted geometry still rejects. Shared lane
and radar boundary validation, minimum fit span, curvature caps and filtering
are unchanged. Regression checks cover folded/nonfinite untrusted tails in
normal and lane-change modes, prefix-only equivalence and rejection of a
nonincreasing trusted prefix. The complete CCNC check runner passes 732 tests.

Same-input actual-cereal replay of all 105 unique segments (124,896 display
updates) changes 336 curvature outputs across four segments, with zero runtime
errors, zero lane-position differences and zero curvature differences during
automatic laneChangeStarting/laneChangeFinishing observations. There are 90
enabled-control observations where previous zero becomes magnitude >1, and
none in the reverse direction. These counts measure output changes, not a
ground-truth accuracy percentage.

All 78 previously identified original-curved/current-zero observations in the
sustained tight bend recover to -15, matching original 6334db9d. A second road
bend previously weakened to -7 also recovers to the original -15. Seven updated
road-video comparison clips were rendered and all 981 frames decoded; selected
tight-bend frames visibly retain the road's right turn. Exact road curvature
amplitude and physical cluster appearance remain unvalidated.

Three alternating-order desktop lane-update timing passes give median
0.0639 ->0.0632 ms for the curved lane-change segment overall and
0.2347 ->0.2314 ms during its automatic change; the pre-U-turn segment gives
0.0592 ->0.0596 ms overall. These small differences do not establish a speed
improvement or target-device CPU cost.

Private source snapshots, replay outputs, scripts, tests, comparisons and
videos are in `.analysis/archive/2026-10-04/ccnc-trusted-prefix/`.

### Direct original-versus-current cost and direction audit

Three alternating-order same-input desktop passes now compare original
6334db9d directly with current production, rather than an intermediate revision.
The measurement covers update_lanes (curvature plus lane-position animation),
excluding CAN packing and radar vehicles; it cannot isolate curvature-only cost
or establish target-device whole-process CPU impact.

| Segment / observations | Original median ms | Current median ms | Increase |
|---|---:|---:|---:|
| Sustained right bend, all | 0.0236 | 0.0681 | 188.6% |
| Straight/change segment, all | 0.0228 | 0.0636 | 178.9% |
| Curved/change segment, all | 0.0224 | 0.0631 | 181.7% |
| Pre-U-turn segment, all | 0.0126 | 0.0613 | 386.5% |
| Straight/change, automatic-change observations | 0.02415 | 0.22515 | 832.3% |
| Curved/change, automatic-change observations | 0.02390 | 0.22770 | 852.7% |

Current geometry updates even when lateral control is disabled, maintaining
filter history; original skips model curvature then. This contributes to the
larger pre-U-turn overhead. Restored cereal-compatible lane geometry, multiple
five-point fits and near-lane consistency, and qualified animation motion all
contribute to the automatic-change cost. The latest trusted-prefix fix alone
did not create this overall increase.

An explicit selected-window road-video direction audit uses four visually
right-curved windows (461 enabled-control observations) and three approximately
straight windows (148 observations). Original/current correct curve direction
is 448/461 versus 461/461 (97.18% ->100%, +2.82 percentage points, +2.90% relative).
Original has six zero and seven opposite-direction outputs in these curve
windows, all in one lane-change window. Straight-window magnitude >1 outputs
are 21/148 versus 0/148 (14.19% ->0%). The one-code tolerance is explicit;
these selected problem windows are neither independent statistical trials nor
an unbiased full-route accuracy measure. They do not validate curvature size.
The tight-bend prefix fix restores original behavior, not a gain over original.

Across all 105 segments current differs from original on 17,918/86,010 enabled
observations and 1,549/2,201 automatic-change observations. Output difference
alone cannot classify improvements. Whole-log curve-detection improvement
remains unknown without independent road labels. The architecture adds input
validity boundaries, fitting intervals, maneuver/road ambiguity guards and
temporal state that the original simple peak-displacement loop lacked; fixes
must address these underlying boundaries rather than accumulate case-specific
exceptions. Private raw measurements/scripts are retained in
`.analysis/archive/2026-10-04/ccnc-original-cost/`. No further production change
was made during this audit.

### CcncModelLanes selector

The parameter is now an integer selector: 0 Off, 1 Basic (기본), 2 Refined
(정밀). Default remains 0. Existing saved ON bytes represent 1 and now select
Basic; there is no automatic migration to Refined. The existing roughly
one-second poll applies changes without reboot. Unsupported integer values
select Off. The internal explicit extended_ccnc=True compatibility override
continues to select Refined.

Basic restores original 6334db9d peak-displacement curvature and its original
three-sample median/0.5 low-pass filter. It skips refined road_curve computation
while updating the timestamps/freshness needed by the shared animation. Refined
retains the current trusted-prefix curvature, maneuver guard and temporal
filter. Both enabled modes retain current lane positions, hold/motion animation,
icons and speed indications. Switching modes resets lane history, including both
curvature filters, but leaves radar selection/history independent.

Actual-cereal replay across 105 segments/124,896 updates finds zero Basic
curvature differences against the original, zero Refined replay-row differences
against the pre-selector improved snapshot, and zero Basic/Refined differences
in positions, hold, draw, hold speed or progress. No runtime errors occur.
739 CCNC/catalog checks and 22 existing Web control/catalog/choice checks pass.
The catalog uses the existing named select control in Korean, English and
Chinese; localized guides describe the modes and saved-value behavior.

Previous boolean OFF/ON cost estimates are removed from this setting's
description because they do not describe the new Basic mode with shared current
animation. Per-mode total cost versus OFF has not been measured. Neither the
Refined name nor replay parity guarantees greater physical road accuracy.
Private scripts/snapshots/results are retained in
`.analysis/archive/2026-10-04/ccnc-model-selector/`. No commit or vehicle/UI
hardware validation is implied.

### Curvature computation cleanup after selector commit 5dc6bf5d

The cleanup retains the existing algorithms, guards, thresholds and temporal
state. It separates interpolation from fitting five samples, reuses each inner
lane's samples for both width and curvature, computes fit residual only for the
quadratic-maneuver bypass that actually consumes it, and reuses fixed sample
coordinates. Dynamic five-point positions are formed directly with the same
spacing/endpoints instead of allocating linspace for every fit. The impossible
two-lane-loop length check is removed. No new geometry or fallback rule is added.

Entire update_lanes output dictionaries, animation state and curvature targets
match before/after across all 105 segments/124,896 actual-reader updates in both
Basic and Refined. Basic still matches original 6334db9d curvature. There are
zero output mismatches/errors and zero maximum target difference. An additional
2,000 seeded noisy-path/interval fit comparisons are numerically identical;
739 existing CCNC checks pass without test changes.

Three alternating-order same-input desktop timing passes measure lane update
only, excluding CAN packing and radar vehicles:

| Segment / observations | Original median ms | Refined before ms | Refined after ms | Reduction |
|---|---:|---:|---:|---:|
| Sustained right bend, all | 0.0227 | 0.06495 | 0.0432 | 33.5% |
| Straight/change segment, all | 0.0228 | 0.0649 | 0.0432 | 33.4% |
| Curved/change segment, all | 0.0237 | 0.0743 | 0.0468 | 37.0% |
| Curved/change, automatic-change observations | 0.0253 | 0.2502 | 0.1314 | 47.5% |
| Straight/change, automatic-change observations | 0.0246 | 0.23205 | 0.1257 | 45.8% |
| Pre-U-turn segment, all | 0.0128 | 0.0626 | 0.0453 | 27.6% |

Refined still costs more than original: approximately 5.2x during the measured
curved change instead of 9.9x in these same passes. Basic with current common
animation costs 0.0245 ms overall and 0.0550 ms during that change; it is not the
original animation implementation. Small Basic before/after timing differences
are measurement variation; its curvature code is unchanged.

Disabled lateral control still advances Refined curvature history, preserving
its first re-enabled output. Skipping this history entirely would be a behavior
change and was not included. The cleanup nevertheless reduces the measured
pre-U-turn overall cost while retaining outputs. This is desktop computation
evidence, not target-device CPU or new curvature-accuracy validation. Named
selector descriptions retain no total-display OFF cost claim because this
benchmark is not that measurement. Private reproducible scripts, snapshots,
outputs and timings are in `.analysis/archive/2026-10-04/ccnc-refactor/`.

### Requested single-segment check on 2026-10-05

The explicitly requested 35f/1 log and corresponding road video were compared
against original, Basic, Refined before cleanup and Refined after cleanup.
All 1,201 updates have lateral enabled, with no automatic or blinker-based
lane-change observations. Basic curvature equals original throughout; Refined
before/after complete lane-output dictionaries are identical. Original and
Refined curve codes differ on 245 observations, which is not an improvement
count. Video at the curve exit shows Refined reducing bend earlier (about 22 s:
original -8 / Refined -5); the later right bend grows earlier/stronger (about
54 s: -1 / -3; 56 s: -9 / -11). Both keep the rightward direction. Exact amplitude
cannot be ranked from this road-video comparison alone.

Seven passes with reversed pass order and rotated per-input policy order give
mean lane-update times original 0.022325 ms, Basic 0.024828 ms, Refined before
0.064786 ms and Refined after 0.043659 ms. Cleanup reduces Refined mean 32.61%;
Basic is 11.21% above original and Refined after is 95.56% above original. These
are same-input desktop timings excluding CAN packing and radar vehicles, not
OFF-relative whole-display costs or device CPU. The first 20 updates are omitted
from timing summaries, while output comparison covers the whole segment.
Two comparison clips (18–26 s and 52–60 s) decode all 319 frames successfully.
This log cannot assess the cleanup's lane-change performance/accuracy because
it contains no change. Private evidence is in
`.analysis/archive/2026-10-05/ccnc-35f-1/`; no production changes were made.

### Large-Position curvature comparison and corrected interpretation on 2026-10-05

A newly supplied segment outside the previous 105-segment corpus has 1,199
display updates, all lateral-enabled, and no automatic or blinker-based lane
change. Near its end, road video shows a left-side entrance and the lead vehicle
moving left. Position departs
strongly left from the inner-lane corridor. The video/logs cannot separate the
entrance's influence from the lead vehicle's influence on model inference.

Original and Basic curvature remain identical. In the 56–60 s window their
maximum is +4 versus Refined +8. At about58.37 s Refined target is +9.86 and its
filtered display +8, while inner-lane estimates are +0.73/+0.50 with probabilities
0.886/0.886. At about58.01 s both inner-lane geometries are valid, their five
sample widths are approximately2.69–2.73 m and their mean curvature is +0.25,
but Position target is +8.49 and display +6. At59.02 s original/Basic display0
while Refined still displays+6 after its trusted span falls below20 m and the
existing hold/decay retains earlier curvature.

The initial review treated the lane/Position difference as an unnecessary bend.
The user reviewed the road video/log and considers the stronger Refined turn
appropriate. That earlier error classification is withdrawn. This segment is
a large-curvature preservation case, not a confirmed false-bend regression;
nearby lane disagreement alone cannot establish which road representation is
correct. Numerical curvature amplitude still lacks independent ground truth.

`road_curve` returns the Position target immediately outside lane changes, so
the near-lane check is not reached here. This explains the output difference,
but does not demonstrate that the stronger turn should be suppressed. Any
future road/maneuver discrimination must preserve this case together with
upcoming genuine bends and uncertain-lane fallback, rather than universally
replacing Position with near lanes. No production code was changed in this
validation. Aligned video, trusted/untrusted path plots and replay scripts are local in
`.analysis/archive/2026-10-05/ccnc-39d-7/` and should accompany the existing true
curve and lane-change cases in any follow-up evaluation.

### Selector cost descriptions measured against OFF on 2026-10-05

Current committed production (1b112e66) was benchmarked through the real
create_ccnc_messages caller and Python CANPacker, with independent extension,
caller-cache and packer state for Off/Basic/Refined. Lane color, radar vehicles
and position correction were OFF in all modes. Eight logs cover normal/large
curves, straight and curved lane changes, lateral-disabled driving and the new
large-Position case. A fixed CAMERA_SCC display fixture generates ADRV_0x161 and
CCNC_0x162; optional stock0x200/0x1ea inputs are absent. This is the same class
of total-display comparison as the earlier boolean measurement, not curvature
alone, device CPU or wire timing. Recorded model/carState/carControl readers
are retained; typed Params values/storage are substituted on desktop, so real
device storage I/O is not measured.

The experiment uses one warmup pass followed by seven measured passes. Mode
order is reversed between passes and rotated each input; every mode receives
the same inputs. There are8,952 replayed calls/pass, with the first20 calls of
each segment excluded from timing, leaving8,792 measured calls/mode/pass.
Report the median of the seven per-pass mean times:

| Mode | Mean total ms | Increase versus Off |
|---|---:|---:|
| Off | 0.167754 | 0% |
| Basic | 0.202221 | 20.55% |
| Refined | 0.233741 | 39.34% |

Korean/English/Chinese catalog descriptions now show approximately21%
(0.168 ->0.202 ms) for Basic and39% (0.168 ->0.234 ms) for Refined. The localized
guides carry the same values and measurement scope. Other three option
descriptions are unchanged. Raw timings, source snapshots, input hashes and
reproduction script are local in
`.analysis/archive/2026-10-05/ccnc-mode-total-cost/`. Existing CCNC/catalog checks
and Web catalog/choice checks verify the descriptions integrate normally.
