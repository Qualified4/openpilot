# CCNC lane-change display geometry (2026-10-04)

`CcncModelLanes` now separates the displayed road bend from lateral motion in
the model driving path. The cluster accepts a single curvature code, not a
continuous trajectory. The previous peak-position calculation could interpret
lane-change translation and heading as a road bend.

## Implementation

- Estimate a quadratic from five equally spaced samples over the existing
  speed-dependent 30–80 m lookahead, shortened to available geometry (at least
  20 m). Remove the constant/linear components and account for slope in the
  geometric curvature denominator. Retain the previous 1800 display scale,
  sign convention and CAN encoding. This scale is not a calibrated cluster
  road-radius measurement.
- Prefer inner lanes, then outer lanes, then road edges. Require finite,
  increasing coordinates; lane probability >=0.6 or edge standard deviation
  <=0.3. Reject conflicting estimates differing by more than three display
  units. Reuse the existing per-model curve conversion/validation cache.
- Apply a 0.25 s time constant and 10 display units/s maximum change. Missing,
  weak or repeated stale geometry can retain curvature for 0.35 s from the
  last valid fresh observation, then fade toward straight. Therefore weak
  geometry on a real bend can temporarily underrepresent that bend. Do not
  claim exact cancellation of vehicle rotation on arbitrary road shapes.
- Retain the existing phase/hold/release decisions. Release evidence counts
  distinct model observations rather than repeated calls. Reset transition
  evidence on a direction change or a control-call gap greater than 150 ms.
- Before the hold trigger, collect at most eight observations of the two
  inner boundaries. Require a 150 ms history, coherent approaching-boundary
  movement, bounded individual steps, positive net speeds in 0.1–1.5 m/s and
  agreement within 0.4 m/s. The opposite boundary may contain bounded jitter.
  These conditions do not establish physical identity: an early common sweep
  can contaminate this history.
- Only inside the existing hold, integrate that prior speed with a linear
  decay to zero over 0.3 s. Cap predicted displacement at 0.25 m (the speed
  bound currently gives an effective maximum of 0.225 m). Never renew the
  prediction with observations from the hold. Without adequate history,
  retain position. This deliberately does not eliminate all pauses.
- Ramp lane filter alpha between 0.2 and 0.6 at 2/s and keep the existing
  post-transition position limiter at 3 m/s, expressed using elapsed control
  time instead of a fixed 0.15 m per call. Preserve filter values at cancellation.

The setting defaults, live polling, lane colors, warning/trailer precedence,
vehicle selection and control algorithms are unchanged. Road curvature now
uses lane geometry throughout active model-lane display, including ordinary
driving, to avoid switching estimators at lane-change entry/exit. Normal curve
appearance can consequently differ from the previous driving-path display.

## Verification and limitations

Baseline: `6334db9d`. The user-provided local log folder contains three
explicitly grouped lane-change segments with 3,600 valid model observations.
One segment contains left and right lane changes. During these maneuvers,
lane probabilities fall almost to zero and lane indices are reassigned over
multiple observations. Road edges, particularly in the left maneuver, are
too uncertain to use as a generally reliable physical-motion reference.

Same-input display replay (signed cluster curvature units):

| Maneuver | Previous range | New range |
|---|---:|---:|
| Left change | -1 to 3 | 0 to 1 |
| Right change | -1 to 1 | 0 |

The hold state still occupies 17 left-change observations and two right-change
observations. Left-change continuation reaches 0.102 m; right-change evidence
does not qualify. These are deliberately conservative results, not evidence
that physical display pauses have been eliminated. The two adjacent segments
also change curvature ranges (-1..8 to -1..2 and -4..6 to -4..5); no road-shape
ground truth is available to establish which normal-driving curve is more accurate.

- 686 focused CCNC/settings/Wiki tests pass, including synthetic straight-road
  translations, headings and 24 left/right sweep schedules (0–1 s duration,
  -0.3/0/+0.3 s onset), real quadratic bends, invalid geometry, stale input,
  bounded prediction, cancellation, filter-gain transition and control gaps.
- On the separate 4,155-frame input set, all eight combinations with model
  lanes OFF preserve complete CCNC CAN output (33,240 comparisons). All 16
  combinations preserve the vehicle-display packet (66,480 comparisons).
- Windows paired replay of the complete CCNC call, with Python CAN packing:
  model lanes add approximately 72–75 us with radar OFF and 42–53 us with
  radar ON versus the previous implementation in the measured runs. These
  are desktop costs, not measured ARM/device CPU effects.
- Korean/English guides and Korean/English/Chinese catalog descriptions are
  updated; current-catalog generated Wiki validation and user-doc checks pass.

Replay forces the display path enabled to compare the same model/car inputs;
it does not reconstruct vehicle response, independently measure road curvature,
or establish actual cluster appearance. Longer, early and common-mode sweeps
remain ambiguous. Do not widen prediction limits solely to remove a visible
pause. Physical-cluster validation remains necessary before claiming the
perceived smoothness is resolved.

Local scripts, route references, samples and results are retained under
`.analysis/archive/2026-10-04/ccnc-lanes/`; they are not committed route data.
