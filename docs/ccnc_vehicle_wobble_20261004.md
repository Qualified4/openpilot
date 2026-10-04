# CCNC vehicle-position wobble diagnosis (2026-10-04)

## Applied follow-up movement qualification (2026-10-04)

The user authorized the follow-up to the common-road correction below. Its
baseline is that exact corner revision, source SHA
`fd20293b3dcdbb10e278d6f3f9937ebd5f2eab8729198df5fba6bbb3b0c8aceb`.
Confidence remains 0.1; no control, published radar, or lane-animation change
is included in this follow-up.

Movement qualification now also requires continuing, same-direction raw and
road-aligned displacement over the most recent 100–200 ms. A short burst that
has already stopped cannot unlock centering later merely by filling the older
evidence window. Boundary and movement evidence use the same continuity-blended
road geometry as the display position filter, before radar-position smoothing.

Missing boundaries no longer immediately bypass the existing 0.5 s geometry
hold for an already qualified, locked object. Geometry whose boundaries do not
contain the expected slot center cannot qualify movement or latch boundary
following. A common-center/Position reference transition preserves the last
qualified timestamp, while still clearing movement and settling evidence.
Slot changes retain the timestamp reset. The hold expires; a valid actual
boundary crossing still follows immediately. New objects cannot inherit a hold.

All 105 unique segments / 124,892 updates were replayed against the preceding
corner revision. On 193,811 matching display-position observations, total
lateral travel decreases 1904.711 ->1792.395 m (5.90%), and back-and-forth
travel decreases 1043.995 ->935.716 m (10.37%). Model-derived non-maneuver corner
mean center error decreases 0.20877 ->0.19811 m (5.11%). Among 2,737 three-second
windows, 245 improve, 42 worsen and 2,450 are similar, using a 0.01 m tolerance.
Travel includes real movement; these are display metrics, not physical accuracy.

There is one display-slot timing difference: three packets in `3c2 /16`,
50.801–50.900 s, retain object 54 in RF rather than moving it to FF, about
0.15 s later through the existing display transition debounce. Targeted replay
confirms physical selection is identical on all 1,198 updates of that segment.
Those three observations are excluded from paired position totals. The only
non-lateral CAN-field differences are FF_DETECT, RF_DETECT and
RF_DETECT_DISTANCE in those packets (nine field comparisons).

The remaining difficult windows are not resolved uniformly. In `3c1 /23`,
27–30 s LF, range decreases 1.422 ->1.361 m but back-and-forth travel increases
1.544 ->1.722 m. In `39e /28`, 45–48 s FF, back-and-forth travel decreases
1.896 ->1.746 m but range increases 1.486 ->1.539 m. Geometry loss beyond the
bounded hold and large geometry changes still need investigation. Regression
windows also include `358 /7`, `3c1 /15`, `35f /1` and `358 /1`.

723 tests pass with desktop Params and ASCII schema-path substitution. Three
OFF replay cases /3,599 updates have byte-identical CAN. ON desktop medians on
those cases remain about 0.61–0.73 ms; this single-order measurement establishes
no speed gain. Actual device CPU, cluster appearance and vehicle-response
validation remain outstanding. Private replay sources, samples, plots and
Korean comparison report are archived under
`.analysis/archive/2026-10-04/ccnc-motion/`; raw logs remain outside the repo.

## Applied common-road correction and comparison

The user authorized implementing the common-road/boundary improvement. This
comparison starts from the immediately preceding 0.1 fast/boundary code
(source SHA b93f5b9b...), not the older 0.6/0.3 code.

Correction ON now uses the midpoint of both usable inner boundaries, requiring
probability >=0.1, origin/target coverage and widths 2.3–4.8 m at origin, 20 m
or target if nearer, and target. One usable boundary retains relative-bend
correction and the existing reference-transition blending. Physical target
selection still uses the original boundaries and selector.

Beyond 40 m, certain Position may replace the lane bend when the relative-bend
disagreement exceeds 0.75 m; retention uses 0.35 m. Require valid covered
Position and yStd <=0.8 throughout the target prefix. Any ego blinker or
non-off/unknown model lane-change state prevents this substitution. This does
not promote Position as an unconditional road-center estimate.

For a common midpoint reference, use boundary release/return clearances
0.15/0.35 m, ordinary movement evidence >=0.35 m, and newly qualified in-lane
fast evidence >=0.8 m. Actual crossing, recent slot crossing and continuing
qualified motion retain the 0.4 m fast threshold. Settling range is 0.35 m;
the 0.1 m/s trend limit remains. Single-boundary/path fallbacks retain legacy
boundary/settling thresholds. Confidence remains 0.1. OFF, control, radar
publication, lane animation/curvature and target selection are unchanged.

All 105 unique segments / 124,892 updates were replayed. Identical tracked
position observations total 193,814, spanning 1,697 continuous episodes. Total
displayed lateral travel decreases 2139.531 ->1905.011 m (10.96%), and
back-and-forth travel decreases 1233.473 ->1044.595 m (15.31%). Model-derived
corner observations without reported ego maneuver have mean absolute center
error 0.22858 ->0.20877 m (8.67%). FF center is zero; LF/RF use their respective
current adjacent-lane centers. These are display metrics, not physical accuracy.

Of 2,737 shared control-frame-aligned three-second windows, with a 0.01 m
tolerance, 534 improve, 304 worsen and 1,899 remain similar. This window alignment
differs from the earlier video-time-aligned audit. Local regressions remain,
including 3c1/23 LF near27 s and 39e/28 FF near45 s. 35f/1 12–18 s mean center
error improves 1.293 ->1.111 m but is still substantial; 356/11 57–60 s improves
only 1.331 ->1.297 m. Far 366/45 36–39 s mean error worsens 0.619 ->0.702 m.
Do not claim that sharp-corner correction is solved or force all targets to center.

717 focused display/vehicle/settings/Wiki tests pass using ASCII schema staging
and desktop Params substitution. All non-lateral/non-checksum 0x162 fields and
all display identities/presences match over the complete replay. Separate OFF
replay matches all 3,602 packets across three representative segments. New
synthetic tests cover asymmetric boundaries, single-boundary fallback, invalid
width, Position uncertainty/domain/maneuver rejection, short in-lane noise,
boundary recovery and rapid real-boundary crossing. Selection timing equality
does not establish physical cut-in response or lateral-display latency.

Desktop update_vehicles median increases about0.029–0.042 ms (6–9%) on three
representative segments, excluding parsing, lane updates and CAN packing;
physical C4 CPU timing remains unvalidated. Full source snapshots, measurements,
remaining regressions and the Korean comparison report are private under
`.analysis/archive/2026-10-04/ccnc-center/`. The final replay source differs only
in the synthetic missing-laneChangeState guard; existing cereal fields make it
equivalent, verified explicitly. This change is uncommitted.

## Expanded audit of all supplied folders

The user requested examining every log below OpenpilotLogs, including corner
centering. FF's center is zero; LF/RF must use their own contemporaneous adjacent
lane centers. Normalized signed display coordinates are compared with those
centers, not with zero for all slots.

There are 114 rlogs, including nine byte-identical same-name copies. All 105
unique segments from 13 routes were replayed (about 104 minutes, 124,892 display
updates). Each segment contains recorded CCNC 0x162 traffic. Replay explicitly
enables all four options in the current and preceding confidence-only 0.1 code;
it does not assert that every recorded session had those settings enabled.
This audit changes no production code.

Of 2,725 eligible non-overlapping three-second identity/slot-consistent windows:
172 meet boundary-following/back-and-forth criteria, 27 regress versus the prior
0.1 code, 23 need corner-center-offset review, and nine show distant Position/
inner-lane disagreement. Categories overlap: 211 distinct candidate windows in
77 segments. Thresholds and complete candidates are in the private audit index.
These are review candidates, not independently confirmed false object motion.

Priority examples, using route suffix / segment:

- 35f/1, 12–21 s: video shows following a front vehicle through a bend. FF
  center error is about 0.62–1.71 m during 12–18 s; 18–21 s back-and-forth grows
  0.173 ->1.518 m. Fast evidence, boundary release/reacquisition and an eventual
  lane-reference switch coexist. Do not attribute it solely to return hysteresis.
- 356/11, 57–60 s: bend and lane-reference switching; normalized FF display
  spans -1.886 to +1.568 m, with object distance 20.75–42.2 m.
- 366/46, 12–15 s: curved sound barrier, 142.95–147 m selected target, FF center
  error 0.793–2.195 m despite reliable geometry on all calls. The exact far
  radar object is not identified from video.
- 366/45, 36–39 s: tunnel exit, approximately 106 m target and FF display
  0.09–2.21 m. Fast evidence reaches 5.76 m/s; check geometric residuals before
  interpreting coherent raw/aligned movement as genuine crossing.
- 366/64, 9–12 s: nearly straight video; RF back-and-forth 0.392 ->1.350 m.
- 393/8, 18–21 s, and 38a/16, 51–54 s: Position versus mean inner lane differs
  about 7.83/7.10 m at the selected 47/45 m range, yet FF remains zero. Output
  disagreement alone does not establish which trajectory is physically correct.
- B/24, 0–3 s: FF average center error 0.034 m, RF 0.738 m with following
  throughout. This remains a useful pre-U-turn geometry regression location.

A separate stable-input corner comparison group has 34 windows: FF23/LF5/RF6,
with average absolute center errors 0.003/0.079/0.065 m. Its selection requires
reliable, stable near-center aligned input; this is not an all-corner accuracy
percentage. Model-derived curvature can label a nearly straight/intersection
approach as a corner. Road video was checked at 12 selected positions, not
manually across every candidate or for physical-cluster output. Distinguish
real side-vehicle movement, reference switching and distant geometry from the
remaining boundary-following regression.

Full private candidates, sources, sampled histories and inspected road contact
sheets are under `.analysis/archive/2026-10-04/ccnc-all/`.

## Full-log comparison

All 27 supplied segments (approximately 27 minutes, 32,397 display updates)
were compared using the same recorded inputs. The baselines are the correction
code before the confidence change (0.6/0.3), the preceding confidence-only 0.1
trial, and the current 0.1 fast/boundary improvement. The first baseline is not
the original code before the earlier lane-animation changes.

Follow identical observed track identities across slots and contiguous segments;
split observation gaps greater than 0.15 seconds. There are 56,853 matched
position observations and 302 continuous episodes with at least two samples.
Two singleton observations contribute no travel. Another 1,892 detected-slot
observations have no current identity and are counted for detection consistency,
but excluded from identity-based movement metrics. Ego lane changes are included.

| Display-coordinate metric, summed over episodes | Before confidence change | Prior 0.1 trial | Current 0.1 |
|---|---:|---:|---:|
| Total lateral travel (m) | 692.760 | 662.889 | 511.354 |
| Back-and-forth travel (m) | 513.899 | 479.289 | 339.650 |

Back-and-forth travel is summed absolute movement minus absolute start-to-end
displacement within each episode. Current code reduces it by 29.1% versus the
preceding 0.1 trial and 33.9% versus the pre-confidence-change baseline. Total
travel decreases 22.9% versus the preceding trial. All 27 segments have lower
total travel, although individual subintervals can worsen. Real maneuvers also
contribute to these metrics; they are not independently measured radar noise.

In 2,730 eligible non-overlapping one-second windows, using a 0.01 m difference
tolerance, 337 improve, 72 worsen and 2,321 remain similar versus the preceding
trial. Window-level summed back-and-forth decreases 37.4%; windows with neither
ego blinkers nor model starting/finishing state decrease 37.8%. This classifier
does not prove that surrounding vehicles remain in their lane or that ego motion
has already settled. Window totals differ from continuous-episode totals because
each window subtracts its own net displacement and excludes incomplete windows.

A remaining regression is A/8 at approximately 43.08–44.03 seconds, LF track34:
lateral range increases 0.030 ->0.619 m and back-and-forth travel 0.020 ->0.818 m.
The new boundary return margin keeps measured following active longer after
release, exposing radar/alignment variations that the previous centered mode
suppressed. Selection and identity are unchanged. This is evidence of a display
regression, not proof of the object's real motion. Boundary-following retention
still needs refinement; aggregate improvement does not resolve every case.

All variants have identical display presence and selection identity across every
slot/update, with 296 presence transitions and 194 detected identity transitions.
Physical cluster display, actual object motion and cut-in response remain
unvalidated. Private reproduction scripts, complete metrics and the 27-segment
comparison chart are in `.analysis/archive/2026-10-04/ccnc-full/`.

## Final follow/boundary improvement and confidence decision

The user authorized improving fast following and boundary-mode transitions,
then delegated the confidence decision. Keep confidence0.1 after comparing it
with restored0.6/0.3 under the same improved algorithm, rather than selecting
confidence from the earlier centering-mode increase alone.

- Fast evidence now spans at least250 ms rather than100 ms, remains coherent
  in raw and aligned coordinates, and must approach within0.6 m of a boundary
  or have crossed it. Already qualified motion can continue while coherent.
- Retain the ordinary lateral filter until the boundary is crossed. A same-
  reference slot crossing has a300 ms continuation window so motion beginning
  in a side slot can finish the250 ms evidence window after entering FF.
- Boundary release still follows immediately below0.35 m clearance. After
  release, require0.55 m clearance to restore centering. Invalid geometry still
  follows; this hysteresis does not hold a boundary-straddling object at center.
- Geometry timeout, temporal track policies, selection, control/radar publication,
  CPU placement and option defaults remain unchanged. No last-good-road snapshot
  feature was added as part of this change.

### Comparison with the preceding 0.1 trial

Same27 logs and32,397 updates. Both variants have correction ON and confidence
0.1. In379 eligible non-overlapping five-second windows with a non-null, observed
correction identity shared by both variants and no reported ego maneuver,
back-and-forth travel decreases225.156 ->143.085 m, about36.5%. With0.01 m
tolerance,94 windows improve,18 worsen and267 are similar. Earlier392-window
reports also admitted some retained displays without current identity; the final
comparison explicitly excludes those. These remain display metrics, not semantic
vehicle ground truth or measured physical motion.

| Window | Prior0.1 lateral range (m) | Improved0.1 range (m) | Prior -> improved back-and-forth (m) |
|---|---:|---:|---:|
| A/13, 31–36 s, right | 1.852 | 0.583 | 4.238 ->0.265 |
| A/7, 1–6 s, left | 1.124 | 0.607 | 4.272 ->1.377 |
| A/7, 19–24 s, left | 0.809 | 0.646 | 3.060 ->0.585 |
| A/5, 20–25 s, left | 0.781 | 1.323 | 1.594 ->0.248 |
| B/24, 1–6 s, right | 1.114 | 1.114 | 3.763 ->3.274 |

A/5 stops repeatedly switching modes and stays in measured following. Its
range grows because the earlier centered display returns to the measured target
in one direction; this is not an across-the-board reduction in movement. The
actual target's semantic identity remains unproved, as discussed below.

Detected correction-tracked centering samples increase75.55% ->81.48%; fast
calls decrease846 ->214. These are mode counts, not accuracy scores. Full
parallel correction-OFF CAN is identical on all32,397 updates. Display identity,
birth and presence match at every slot. Within recorded model lane-change
starting/finishing intervals,306 packets change lateral display values; unchanged
selection does not independently establish vehicle-response or cut-in latency.

### Is0.1 still justified after the algorithm change?

Replayed an offline restored0.6/0.3 variant with exactly the same fast/boundary
algorithm and original outer-boundary admission. On the same379 shared windows,
improved0.1 has7.1% less summed back-and-forth travel and7.6% less summed range
than improved0.6/0.3. There are73 better,56 worse and250 similar windows in
back-and-forth travel. Selection identity and presence remain identical. The
incremental benefit is modest and not universal: for example A/12 at55–60 s
has more back-and-forth travel at0.1. Keep0.1 for the measured aggregate benefit,
with these regressions explicit; the main improvement is the fast/boundary fix.

703 tests pass, including short coherent in-lane oscillations, boundary return
hysteresis, invalid/stale geometry, both directions of rapid side-to-front
crossing and inherited motion history. Fast confirmation now requires an extra
150 ms minimum; ordinary following remains available during confirmation. Tests
and replay do not prove physical cluster feel or surrounding-car classification.
Sources, threshold variants, all output histories and plots are local in
`.analysis/archive/2026-10-04/ccnc-follow/`. Changes remain uncommitted.

## Subsequent threshold trial

After this diagnosis, the user requested lowering position-correction lane
confidence to 0.1. New centering, reacquisition and retained centering now all
accept the relevant pair of lane probabilities at >=0.1. Side-center geometry
and outer-boundary availability use the same threshold when correction is ON;
correction OFF retains its original center and boundary thresholds. Geometry,
boundary clearance, uncertainty timeout and motion/release rules are unchanged.
699 tests pass, including exactly0.1 admission for all three slots and fallback
below0.1. The diagnosis below describes the earlier thresholds. The subsequent
trial comparison is recorded separately below; fast-motion following remains
a separate contributor.

### Same-input comparison of the 0.1 trial

Replayed all27 supplied log/video segments and32,397 display updates against
the frozen preceding source/output. Both compared variants have correction ON.
The trial changes lane-confidence thresholds only, retaining the fast branch.
The fraction of detected, correction-tracked slot samples in centering mode
increases from60.73% to75.55% (56,853 samples each). This is a mode fraction,
not a vehicle lane-keeping accuracy score. Fast-evidence calls remain846.

| Window / slot | Prior lateral range (m) | 0.1 trial range (m) | Prior -> trial centering |
|---|---:|---:|---:|
| A/7, 19–24 s, left | 0.887 | 0.809 | 0% ->46% |
| A/7, 1–6 s, left | 1.124 | 1.124 | 45% ->45% |
| A/13, 31–36 s, right | 1.852 | 1.852 | 41% ->41% |
| A/5, 20–25 s, left | 0.135 | 0.781 | 0% ->53% |
| B/24, 1–6 s, right | 1.114 | 1.114 | 0% ->13% |

A/7's 19–24 s back-and-forth travel decreases4.062 ->3.060 m, about25%,
while its range decreases only about9%. A/13 is unchanged because its previous
lane evidence was already adequate; fast following remains the cause.

The A/5 regression is a newly enabled center lock repeatedly alternating with
the unchanged0.35 m boundary-clearance gate. Reliable evidence now exists on
all100 calls, but boundary clearance passes on only53; follow/center mode
switches30 times in five seconds. The estimated center itself moves less than
0.01 m. This is a boundary-mode oscillation rather than a moving center target.
The old variant kept following continuously and stayed nearly steady.

On the user's subsequent object-identity question, A/5's recorded left-display
signal disappears around1.72–4.08 s. That early dropout is separate from the
20–25 s comparison target. Selection changes to a different radar source around
14.73 s; the later target is about7.8 m ahead and4.45 m left with near-zero
absolute speed when ego is stopped. Video shows stationary cars to the left,
so the point is compatible with a real stationary vehicle, but a radar ID alone
does not establish semantic vehicle identity or exclude a structure reflection.
Do not present this window as verified nonvehicle misdetection or as independently
proved lane-keeping of one physical car across the earlier source change.

In392 non-overlapping five-second windows sharing continuous detected identity,
birth and slot in both variants, without ego blinkers or active model-reported
lane changes, back-and-forth travel improves in74 windows and worsens in53
using a0.01 m difference tolerance;265 stay within that tolerance. Summed travel
decreases237.284 ->225.156 m, about5.1%. It is not a population accuracy score
or proof that all surrounding vehicles physically maintained their lanes.

Selection identity/birth and display presence match at every compared slot;
the parallel correction-OFF CAN output is identical on all32,397 updates.
Within model-reported starting/finishing intervals,48 vehicle-display packets
change. Preserving selection does not independently establish cut-in response
or correct displayed lateral motion. There is no physical cluster A/B footage.

Conclusion:0.1 improves weak-lane centering retention but is insufficient to
resolve the principal fast-following wobble and reveals boundary-mode chatter.
The user-requested trial remains uncommitted; no fast or boundary behavior has
been silently changed. Evidence is local in
`.analysis/archive/2026-10-04/ccnc-confidence/`.

Position correction can increase lateral display oscillation even with a
continuous vehicle identity and slot. The main reproduced contributor is the
fast-motion branch; weak lane evidence can also prevent a return to centering.
This investigation changes no production vehicle-display code.

## Evidence

Replayed 27 supplied lane-change-test rlog/video pairs, 32,397 display updates,
with current production code and position correction OFF/ON. Models use a
0.5-second freshness budget and liveTracks 0.15 seconds. State persists only
across contiguous segments. Rank five-second windows without ego blinkers or
model-reported active lane changes, requiring continuous ON identity/birth/slot.
This excludes reported ego maneuvers, not all actual surrounding-vehicle motion.

Below, A is the first supplied route; private route mappings remain local.
Values are lateral output max-minus-min in meters before CAN quantization.
The OFF comparisons use only detected samples matching the ON identity.

| Window / display slot | Correction OFF | Correction ON | Diagnostic ON without fast branch |
|---|---:|---:|---:|
| A/7, 1–6 s, left | 0.547 | 1.124 | 0.605 |
| A/7, 19–24 s, left | 0.212 | 0.887 | 0.667 |
| A/13, 31–36 s, right | 1.443 | 1.852 | 0.584 |

A/7 first window has 97 matching OFF samples versus 100 ON; the other two
have 100 for every variant. ON identity, birth and source do not change in any
of these windows. A/13 has reliable geometry and boundary clearance throughout,
so neither an ID reset nor weak-lane fallback explains its repeated unlocks.
Viewed synchronized road contact sheets show no obvious corresponding lane
change, but radar-reflection identity is not independently proved by video.

Recorded lateral CAN magnitude codes span 28–37 in A/7's 19–24 s window and
25–43 in A/13's 31–36 s window. Replay ON matches these individual lateral
codes on 94/100 and 71/100 samples, respectively, versus OFF 4/100 and 20/100.
These are selected-field checks, not complete recorded-session CAN equality.
The recorded lateral signal itself moves; this is not solely a visualization
artifact introduced by the analysis.

## Causes

1. `_CcncVehiclePositionCorrection.apply` accepts fast-motion evidence after
   at least three distinct observations spanning 0.1 s, with a 0.4 m minimum
   net movement and matching raw/road-aligned direction. Road-aligned position
   contains the same radar measurement, so agreement does not independently
   establish physical lane crossing. Short coherent radar fluctuations can
   qualify. In A/13, rapid estimates reach about 2.64 m/s; the branch repeatedly
   releases centering while geometry remains reliable and inside the lane.
2. Fast following targets `aligned_y`, bypassing the ordinary lateral filter,
   and raises the rate above the normal 1.5 m/s to `1.4 * fast_speed + 1.0`
   (capped at 15 m/s). This transmits brief fluctuations more directly. Removing
   only fast evidence in an offline variant materially reduces the oscillation;
   it does not remove all drift or prove that deleting the branch is suitable.
3. Low-confidence/reference changes keep vehicles in ordinary following.
   A/7's 19–24 s window stays inside the lane, but reliable geometry exists on
   only 25% of calls and it follows on all calls. New/recovered centering still
   requires reliable evidence and a one-second settling window. Inner-lane
   reference changes clear that evidence. Thus correction ON can follow noisy
   measurements for seconds even without a real lane change.
4. OFF uses the previous curved deadband; ON replaces it with center-lock/
   following logic. During following, that old deadband does not additionally
   suppress near-center movement. This explains why ON is not automatically
   less variable than OFF. In A/7's 19–24 s window, removing fast evidence cuts
   back-and-forth travel from 4.062 to 0.813 m, but range remains 0.667 m versus
   OFF 0.212 m: weak-lane following and slow movement remain separate issues.

## Implications and limits

Investigate requiring sustained motion toward a physical lane boundary for
rapid unlock and retaining filtered output until crossing is corroborated.
Evaluate established-vehicle centering retention and robust reacquisition
separately from new-vehicle admission. Simply deleting fast recovery or globally
strengthening smoothing may delay actual cut-ins; genuine crossing, dropout,
ID-reconnection and weak-lane cases must be compared before a production fix.

The diagnostic variant changes only fast-evidence eligibility in a local source
copy. Its benefit is causal evidence for display behavior, not vehicle-response
validation or a proposed release. No steering, radar publication, thresholds,
CPU placement or option defaults change. There is no physical cluster footage.
Private scripts, source snapshots, ranked windows, sampled histories and plots
are archived in `.analysis/archive/2026-10-04/ccnc-vehicle-wobble/`.
