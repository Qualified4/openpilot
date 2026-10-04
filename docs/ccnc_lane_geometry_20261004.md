# CCNC lane-change display geometry (2026-10-04)

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
