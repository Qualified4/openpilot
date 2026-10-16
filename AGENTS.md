# Repository memory

- On 2026-10-16, Davin accepted FF-exit round 4's .5m vision-confirmed self-baseline release: 3e7/18 5.384 -> 4.830s; common43 departure p90 .962 -> .390s; FF step p95 .0171 -> .0180m. Slot/presence/active ego-change/cutin/departing-lead regressions are zero. Road-test residuals: five departures slower than round3 (conditional added-delay p90 .30s), 28 large return windows (eight overlap prior video measurement-error judgements), and one abandoned release at 3d8/35 (legacy behavior). Production/prototype outputs match all 175 segments/205449 sends, and 462+5 tests pass; no push/deploy/timing measurement. Evidence: .analysis/archive/2026-10-16/ccnc-ff-exit-r4 and the follow-up adoption archive.

- On 2026-10-16, FF self-baseline round 2 was rejected: the user chose raw .35m movement during threshold dwell. On 175 cached segments/205,449 sends, .5m releases at 5.129s, catches up to .142m error, but departure p90 stays 1.05s; FF step p95 .018m, 50 zero-centre return windows (39 >.3m, 5 sharp-curve windows). Existing 462 tests pass; 2 new tests pass and the stepped .7m/s test fails (.5s release delay vs .3s). Side/active/post ego-lane-change differences are zero; expanded +/- .5s windows retain 30 pre-change sends. Slot/presence/cutin/departure regressions are zero. Both raw-period interpretations and .4/.5/.6m fail acceptance; selected-family video 20 windows reviewed. Evidence .analysis/archive/2026-10-16/ccnc-ff-exit-r2; production unchanged, no commit/push/deploy/timing benchmark; see the experiment doc.

- On 2026-10-16, FF displacement-release prototypes at .4/.5/.6m were rejected on 175 cached segments (205,449 sends): .5m improved departure delay but worsened FF jitter and ego-lane-change displays; new 3e7/18 is cached in ccnc-replay-v7, 12e replay matches 62.83% of nine packed front-slot fields exactly; production unchanged, no commit/push/deploy; see docs/ccnc_radar_map_experiment_20261011.md and private .analysis/archive/2026-10-16/ccnc-ff-exit.

- On 2026-10-15, the user requested merging carrot-wip into ccnc-hda1 again.
  origin/carrot-wip 8e7eec04 (19 commits: eGPU recovery/runtime and MD v3,
  S3 Wi-Fi/registration/installer and retry lifecycle, in-lane radar braking,
  Tesla PR551 corrections, docs) is merged with current CCNC work preserved.
  Only AGENTS.md conflicted; both complete blocks are retained, CCNC first.
  The CCNC extension/map have no change against pre-merge b595f099.
  tools/ccnc/pytest_plugin.py BASELINE now follows the incoming revision;
  CCNC 928 (including Wiki) and map 161 checks pass. Incoming focused checks:
  modeld 320, cluster/radar 100, Tesla 50, browser tests 10 pass. Additional
  web-server/viewer and angle-saturation checks cannot run on this PC without
  aiohttp/setproctitle. Installer browser tests need LF working-copy text on
  Windows (their import-removal regex assumes LF); no Git blob change needed.
  No push, deployment, MCU safety build or vehicle validation performed.
  Private logs: .analysis/archive/2026-10-15/carrot-merge.

- On 2026-10-15 (CCNC round 14, committed, not pushed), baseline working-copy hashes
  extension b4499608db61 / map b231140f84fe: same-id longitudinal report jumps
  split near side vehicles and stale objects then stole reports back. Accepted
  association only for an already selected, motion-confirmed side vehicle while
  holding, within 15m, same id seen within ID_CYCLE, both positions strongly lane
  interior, forward innovation <=2m / lateral <=.35m / speed error <=6m/s;
  report forward speed >RESTING_SPEED, off during ego lane changes. Existing
  cross-id matching is unchanged, no new history inheritance. Actual-reader
  174 segments/206444 sends: continuous same-source key changes 25697 -> 25688,
  new objects among those 10511 -> 10509, resulting display reinitializations
  335 -> 331; >.6m initial offsets 218 -> 212, 3e6/4 13 -> 7. Presence additions/
  removals, own lane-change differences and late maneuvers 0. Map-only 130+27
  has no display/source differences; residual 0, c1/23 6/6, 19 bookmarks same.
  Seven waiting windows unchanged (3c8/20 is exactly 845/846 in both; the earlier
  100% was rounded). Broad continuity failed with false additions; a narrower
  predecessor lost 4 map-only sends near an outer boundary and was rejected.
  Entry-only curved drawing with immediate release on confirmed following kept
  internal tracking unchanged but delayed outward moves/departing leads and one
  legacy cut-in, so rejected. No new saw suppression. PC serial 4 logs/4794
  updates, warmup+3 reps: lanes+vehicles 1.1254 -> 1.1406ms (+1.35%, no CAN pack).
  Map tests 161 passed; existing suite 926 passed / same 2 Wiki catalog failures.
  Evidence .analysis/archive/2026-10-15/ccnc-r14.

- On 2026-10-15 the user requested applying the reviewed CCNC candidates:
  extension A2 preserves recently confirmed inward following through jump-hold
  entry only at 10-30m; below 10m the existing reflection hold stays. Confirmation
  comes from existing raw/aligned motion gates and expires after .3s or a slot/
  reference change, never from forced-follow alone. Map B prefers an eligible
  candidate within OUTER_FAR of the inner line when the outer line is unusable;
  it keeps the original choice if none qualifies, with valid outer lines, while
  holding or during ego lane changes. No unconditional ID-history inheritance.
  The first unrestricted A was rejected for near-field straight-car movement.
  Actual-reader replay: 174 segments/206,444 sends; 3d8/35 map threshold delay
  .753 -> .401s; 3e6/5 RF range 2.901 -> .846m, object swaps 9 -> 2.
  No display additions/removals, stopped/ego-lane-change changes; side selection
  differs on 173 sends/13 windows, some visually inconclusive. Repeated reflection
  wobble and 3e6/4's ID-change step remain. Claude's review fixes (named
  INWARD_KEEP_X/INWARD_KEEP_TIME, a skipped hold does not mark the track as
  jumped, outer-line lookup hoisted) changed no replayed send. Committed; not
  pushed or vehicle-validated.
  See docs/ccnc_radar_map_experiment_20261011.md and local evidence
  .analysis/archive/2026-10-15/ccnc-candidates.

- On 2026-10-14 (round 12e) the extension went 099be84d8c6f -> 5e38cabe8d7c: rule 2
  (no following of a slow outward creep) applies only to a track that has made
  an inward reflection jump, kept for the track's life; without one a slow
  outward move is followed as before (3e1/20 id48, a real exit, equals HEAD).
  Outer-lane moves back to HEAD (map late >2.5s 6, lag >1.2m 12); sawtooth map
  median .36 -> .41m, accepted. 12d (release after 2.5s / 1.2m) was rejected.
- On 2026-10-14 (round 12c) the extension went d473d0fb4bd7 ->
  099be84d8c6f: side-slot position correction (x<30m, off during ego lane
  change) holds the drawn lateral .3s after an inward reflection jump (>=.6m
  in .3s; ends on crossing ego's line or moving .2m further in .2s) and does
  not follow an outward creep <=.8m/s over >=1.5s unless the object was FF in
  the last 3s; sawtooth round trip median legacy .14 -> .11, map .45 -> .36m; accepted residuals: a
  hold 3s after a lead left (356/21) and <=.15m carried into own lane changes.
- On 2026-10-14 the CCNC replays run on os.cpu_count()-2 processes by default
  (replay_radar_map_fast, replay_live_path --jobs/--prepare, and the private
  probe/run harnesses), consecutive segments of a route in one job; outputs
  are byte-identical to the round 11 runs. Wall time is bound by the longest
  run (22 of 171 segments). Measure timing only in rounds that add a rule.
- On 2026-10-14 (round 11 B) the map module went acf245a5b309 -> 2fead7f97fd7:
  LEAD_SIDE_X 50 -> 60 except while holding (LEAD_SIDE_HOLD_X 50), since a
  50-60m side candidate displaced the stop memory of a car beside ego (15
  windows, 169 sends); legacy-only 50-80m side sends 7,524 -> 5,170, holding
  side swaps 0, 50-60m flicker LF 15 -> 61, RF 17 -> 32 left for a later round.
- On 2026-10-14 (round 11 C) the map module went 8e0a77855e73 -> acf245a5b309:
  far FF off-confirmation. 339 of 356 FF flickers beyond 50m were the lane slot
  leaving ego's lane for .1-.3s with the object measured; the displayed FF
  object beyond FF_FAR_X 50m stays FF for FF_FAR_HOLD .3s while measured and
  eligible, not during a lane change or a bridge, and a nearer FF candidate
  wins. The check sits before the MAX_SIDE_X side cut (after it, beyond 80m it
  never ran: 356 -> 279 only). Flickers beyond 50m 356 -> 142 (68 if applied
  during lane changes too, which also changes displays within 50m); within 50m
  116 unchanged. Tests 1050 -> 1055; the case C bridge test runs 60 steps.
- On 2026-10-14 (CCNC setting texts; NOT committed, no behaviour change) the
  five Ccnc* entries in carrot_settings.json got short titles and one-line
  descriptions ending in a compute-cost percentage; CcncModelLanes labels are
  0 끔 / 1 일반 (_ccnc_basic_curve) / 2 정밀 (compute_curve=precise), en
  Standard; CcncVehiclePositionCorrection and CcncRadarMap became detail
  children of CcncRadarVehicles (detail_section "radar"), so VEHICLE lists 20
  menu items. Children are not greyed out while the parent is off, and an
  existing favourite on a child no longer shows in Favourites (detail items are
  filtered there). Moved out of the descriptions (kept in docs/user settings
  pages): lane colour only in Drive, highlight longer at speed; radar vehicles
  stop memory only within 15 m radar range, side display kept up to 2 s or 30 m
  of ego travel when lanes vanish, rear icons are blind-spot alerts; position
  correction predicts up to 0.3 s after the last accepted position, reconnects
  a gap only with one candidate seen twice, tolerates a further lane-confidence
  drop for 0.5 s, confirms boundary slot changes for 0.2 s; radar map keeps a
  20 m map, joins close stationary points into walls, remembers waiting and
  beside vehicles while stopped, not crossing pedestrians or cyclists. Old PC
  costs (total display time): lane colour 0.185 -> 0.192 ms, radar vehicles
  0.185 -> 0.633 ms, correction 0.633 -> 0.785 ms; model lanes 0.168 -> 0.202
  / 0.234 ms is in docs/ccnc_lane_geometry_20261004.md. The map's +85% is the
  2026-10-14 legacy-vs-map selector timing (0.44 -> 0.81 ms), a different
  basis from the other four.
- On 2026-10-14 (legacy vs map overview; measurement only, no code change)
  the existing selector alone (CcncRadarMap off) and the map (8e0a77855e73,
  extension d473d0fb4bd7) ran side by side through the live entry order
  (update_lanes, then update_vehicles; one extension instance each) on 171
  segments, 200,653 sends. The harness needed modelV2 position, timestampEof
  and named laneChangeState for the legacy path fallback: side cache
  .analysis/cache/ccnc-meta-v3; legacy then matches the replay cache values[1]
  on 99.4-99.9% of sends. Actual sendcan matches neither closely (drive-time
  code). Shown by either: FF both 95.5%, legacy only 3.6%, map only 0.9%; LF
  legacy only 14.5% (58.6% of 50-80m, the 50m side limit); RF legacy only
  8.0%, map only 4.6%. Video, 80 windows: structure shown as a vehicle,
  legacy 1,413 sends vs map 0; real vehicles not shown, map 1,324 (613 beyond
  the side limit) vs legacy 2,378. Map FF flickers more beyond 50m (356 vs 91).
  PC per send: legacy 0.43-0.45ms, map 0.80-0.82ms mean. Section "기존 vs 지도
  종합" in the experiment doc; evidence in
  .analysis/archive/2026-10-14/ccnc-legacy-vs-map.
- On 2026-10-14 (bent walls; NOT committed) the map module went c2d313c7d0bf
  -> 8e0a77855e73: a fresh resting return within WALL_MAX_GAP of a wall end and
  within WALL_BEND 30 degrees of the end segment joins the wall, and the next
  one is compared with the new segment (3e3/15: a post row bending into a turn
  pocket; one post was admitted as waiting .5s before its part became a wall).
  Unlike new rows, the lane-interior protection does not apply to these joins:
  the pocket posts stand inside the lane the model draws (with it the scene was
  not fixed). 171 segments (3e3/15 added to ccnc-replay-v5 and meta-v2): only
  that LF (10 sends) disappears; 799 objects joined by bends, none admitted at
  the time; 15 degrees gives the same displays, 45 changes LF more. Consumers:
  bands, band-line hits and stop memory unchanged; wall members +6%. One
  synthetic test (two wall directions not joined) now switches WALL_BEND off.
  The waiting-admission pair hold tried before (round 9) was dropped: a car's
  two reflections are 0.8-3.9m apart, overlapping post spacing, and it hid 26
  real vehicles of 30 windows; recorded in the experiment doc.
- On 2026-10-14 (round 8c; NOT committed) the map module went 5e9d3ceb0d8e
  -> c2d313c7d0bf: a bridged FF object (cases A/B/C) is drawn at its stable
  y, the mean report y from ego's lane centre over its last .5s calm stretch
  (sideways < .3m/s over .15s after subtracting the .05m report step), kept
  while it moves and not updated while bridged (the merged reports would look
  stable); without one, the last in-lane report as before. Display only. 3e1/19
  and 3dd/11 return to the lane centre during the bridge (were +1.6/+1.4m);
  3e3/2 unchanged (deadband). 170 segments: only FF lateral changes, 72 sends
  in bridges, 75 within 3s after, 5 LF/RF in 366/29 (slot debounce), none
  elsewhere; bridge-end steps over .5m 2 -> 0. Tests 1045 -> 1047.
- On 2026-10-14 (FF bridge case C; NOT committed) the map module went
  21ec6cd78825 -> 5e9d3ceb0d8e: the FF bridge also starts when the FF object
  is measured into a side lane without a jump (the merge pushing the lead's
  track across the line over several cycles, 3e1/19 16.9s), within
  BRIDGE_DRIFT_X 50m and not for a structure; everything after the start is
  as case B. Stage 0: of 428 drift events (no jump) only 13 are within 30m
  with vision on the lead, 2 are merges (3e1/19 16.94, 358/0 40.6), 11 are
  ego lane changes. On the 77 departing leads vision lets go a median 1s
  before the map (later in 2 of 64), so no extra vision-movement end rule.
  Without the 50m limit the bridge started 58 times, mostly at 60-130m; without
  the side-slot condition it bridged a car-park pillar (366/0 30.0s). 170
  segments (live entry): case A/B 8 bridges / 66 sends unchanged, case C 8 /
  78 sends, far jump-and-return events 543 -> 537, lane-change windows 0,
  77 departures: 2 differ (3dd/11 13.28 is a merge, 366/29 earlier). Known:
  a slow merge's held lateral is already pushed toward the line (3e1/19 FF
  drawn at +1.6m). The fast map-only replay passes no lane_change, so its
  differences are larger than the live path's. Tests 1043 -> 1045 (round 8's
  "drift is never bridged" test replaced).
- On 2026-10-14 (FF bridge round; NOT committed) the map module went
  bc52b9c4e6e5 -> 82b2155583c3 -> 21ec6cd78825 (working copy, CRLF): a displayed FF object
  whose radar report is lost, or whose same-id report jumps >= .8m sideways
  within .15s (the radar merged the lead with a trailer beside it, 3e3/2
  52.3-54.7s), stays FF for up to BRIDGE_HOLD 2s while a leadsV3 entry
  (p >= .5) is at its distance (+-max(3m, 20%)) and within 1.2m of ego's lane
  centre. Distance follows vision with the offset fixed at the start; lateral
  is the last in-lane report relative to the lane centre. A measured object
  nearer or at the bridged distance takes FF; a farther one cannot. A new
  report within 1.5m in ego's lane reconnects to the same object (any id).
  Ends at once without vision, after 2s, or while ego changes lanes (model
  desire laneChangeLeft/Right or a turn signal, passed as update(...,
  lane_change=)). Gradual drift out of the lane never starts a bridge. Vision
  creates no object, gives no evidence and fills no side slot. SCC (trackId 0)
  is still ignored. New logs: 3e3/2 and 3e3/11 decoded to
  .analysis/cache/ccnc-replay-v5 (3e3/15 also in the folder, not used); model
  desire and turn signals for all 170 segments in .analysis/cache/ccnc-meta-v1;
  170-segment manifest in .analysis/archive/2026-10-14/ccnc-bridge.
  Live-entry replay of 170 segments: 66 bridged sends in 8 episodes, FF far
  jump-and-return events 549 -> 543, LF/RF unchanged, 0 bridged sends in 103
  lane-change windows, 77 departing-lead scenes unchanged. Review residual 0,
  c1/23 6/6, waiting 7/7, bookmarks 18/18. Tests 1033 -> 1041. Stage 3 (FF
  lateral jitter, measurement only) and the reliable-flag difference (map keeps
  lanes whose probability fell below .1, decaying by .9 per frame; legacy uses
  the current probabilities; both threshold .1) are in the experiment doc.
  Follow-up (module 21ec6cd78825): during a bridge the FF closing speed is the
  vision lead's leadsV3.v, else the speed at the last in-lane report; only the
  speed update_vehicles passes changes (extension untouched). 3e3/2 FF_DETECT
  stays 1; 170 segments differ from round 8 only in 107 detect values (3e3/2
  92 -> 1, 366/46 15 -> 2 from vision 19.6m/s at 58m). Vision v is kept in
  .analysis/cache/ccnc-meta-v2 (meta-v1 plus leadsV3 v). Tests 1041 -> 1043.
- On 2026-10-13 the user asked to commit the root CLAUDE.md after all (it had
  been kept out of every commit on the user's instruction until then). It is
  tracked from this commit on; the "never CLAUDE.md" notes in the entries
  below are historical. The audit entry below is committed as 61415e94.
- On 2026-10-13 (audit of the day's four rounds; documents only, NOT committed)
  nothing in the map or extension module changed (map bc52b9c4e6e5; the far
  limit round below is committed as 58ba313b). Old = fed331cb's module
  1ace50c16016 (what the car ran), new = HEAD, both through the live entry
  point (update_vehicles with position correction) on every logged segment:
  168 segments, 197,050 sends = Full130 (ccnc-replay-v2) + 27 road-test
  (ccnc-replay-v3) + the eleven 3dd segments newly decoded to ccnc-replay-v4
  (manifest .analysis/archive/2026-10-13/ccnc-audit/all-manifest.json).
  Totals: removed LF 6,190 (3.1%), RF 3,556 (1.8%), FF 171; added 6; swapped
  1,058. 70% of the removed sends are beyond 50m and 98% are side slots past
  the FF. By rule (leave-one-out variants): far limit 6,185 sends, C(2) 1,007
  (one bollard), A(1') 720, B' 569, A(1')+A(2) overlap 536, A rules + far
  limit overlap 1,036, A(2) 406, D 132, H 95, mixed 295. 153 windows of 20+
  sends are listed in the experiment doc with rule and video verdict. Video
  (top 30 by length and the 19 with sends inside 25m, 45 in all): 15 correct,
  22 undecided, 8 that look like real neighbours plus one seen earlier - all
  nine at 41-80m and due to the far limit (3d8/9, 3c1/12, 366/33, 3c1/16,
  356/12, 3c1/13, 384/12, 3c1/10, 358/11; 983 sends; not certain at that
  range). No real vehicle loss confirmed inside 25m (13 correct, 5 undecided,
  1 far). LEAD_SIDE_X 60 restores the 50-60m part (four windows almost fully)
  and brings back 22 sends of 3e0/4 1.5 and 30 of 3e0/1 54.7; 356/12 0.7-5.6s
  LF is mixed trail/far limit and no single number restores it. Today's
  bookmarks through the live path: B' four show only the adjacent car, D/H
  three and 3e1/3 gone, A six of seven at 0 (3e0/2 20.4 keeps the car behind,
  58 sends), 3e1/5 unchanged (not fixed), out-of-scope ones unchanged; the 18
  earlier scenes same or less. Stability: same object back within .5s LF 508
  -> 251, RF 643 -> 517, FF 652 -> 649; object counts identical (mean 21.3,
  33.4 while stopped, max 178, never 256). The live-only FF difference is
  TRACED: stabilize_slots keeps an object that changes slot drawn in its old
  slot for up to .2s only when that slot is vacant; at 366/29 20.27s a car
  straddling ego's left line 8m ahead alternates LF/FF, the old module had a
  59m object in LF so it went to FF at once, the far limit now empties LF and
  the car is drawn LF with FF value 0 for .05-.2s (40 of the 176 FF-differing
  sends have this shape; the other 136 are D/H removing speed-only FF). PC
  time unchanged (.58-.62 vs .59ms), no live-entry exception in any of 13
  module variants, 1,033 CCNC tests pass. Watch in the car: neighbours beyond
  50m behind a lead and their appearance at 50m, cars moving to and from the
  lane beyond, a car changing lane right ahead (FF empty ~.2s), second leads
  and real neighbours on curves, turns, stops beside centre-line bollards,
  cars riding wide in a last lane. Commit files when asked: the experiment
  doc and AGENTS.md only; never CLAUDE.md.
- On 2026-10-13 the user requested merging carrot-wip into ccnc-hda1 again, after
  the four road-test rounds were committed (1f3abaa2, 099ce103, ae21b0de,
  58ba313b - the "NOT committed" notes in the CCNC entries below are
  historical; the far-limit round is 58ba313b). origin/carrot-wip 09df1475 (14
  commits: signal preparation input validation, signal/planner/MPC/CAN hot-path
  performance, braking-onset following distance memory removed, onroad overlay
  layer reuse and its fixes, vNavi motorway signal suspension, driver
  monitoring pause in soft hold, startup recovery and active-turn indication,
  docs) was merged. One conflict, openpilot/selfdrive/ui/mici/onroad/
  hud_renderer.py: carrot-wip moved the wheel alpha filter update into
  _update_state, so its side was taken there; it also updates _wheel_y_filter
  in _update_state, which this branch had removed, so that one line was dropped
  (it would raise AttributeError). AGENTS.md did not conflict.
  tools/ccnc/pytest_plugin.py BASELINE follows the merged carrot-wip revision
  (carrot_settings.json changed). Map module bc52b9c4e6e5 and extension
  d473d0fb4bd7 unchanged; incoming hyundaicanfd.py and carcontroller.py changes
  merged without conflict. CCNC tests 1,033 passed (14 incoming cluster cases),
  wiki settings 27 passed. Incoming tests on this PC: those that import pass
  (among them test_carrot_hud_renderer 29); the rest stop on this PC's
  environment (capnp abort on the non-ASCII path, missing params_pyx, pyray,
  fcntl). Not pushed or deployed.
- On 2026-10-13 (fourth road-test round: far side slots behind a lead; NOT
  committed) map is bc52b9c4e6e5 (from 25ff38ec9160; the curve entry below WAS
  committed as ae21b0de). ONE rule: while an FF is displayed, a side candidate
  beyond 50m AND more than 30m past the FF gets no side slot (LEAD_SIDE_X,
  LEAD_SIDE_PAST); without an FF MAX_SIDE_X 80 as before. The FF+30m part was
  added because a bare limit also removed displays 5-25m past the FF (Full130
  866 at 50m), which the user required to be zero. Bookmarked object sends in
  +-3s, A round -> now: 3e0/4 1.5 48 -> 0, 3e0/1 54.7 31 -> 0, 3e0/2 50.0 21 ->
  0, 3e0/1 45.0 2 -> 0; unchanged 3e0/0 43.9 9 (within FF+30), 3e0/2 7.9 1,
  3e0/2 20.4 7 and the car behind it 35 -> 32 (43-54m). FF never changes
  standalone. A 60m limit leaves 22 sends of 3e0/4 (it approaches to 53m) and
  30 of 3e0/1 54.7, so 50 was taken; 50 removes 2,125 more Full130 sends than
  60, all at 50-60m. Standalone vs 25ff38ec9160: Full130 removed 4,444 (all
  class 2, all beyond 50m), swapped 243 (all to stop memory: the freed slot
  shows a remembered near vehicle), added 0; 27 segments removed 1,001. Video
  of five 50-58m scenes that 50 removes and 60 keeps: three look like real
  neighbours (358/11, 3d8/9, 366/33; not certain at that range), two
  undecided; the rest unverifiable. Near real neighbours (35f/1 16.3-17.3 RF,
  3e1/5 8.1-24.2 RF, 3e1/14 24.9-27.9 LF) unchanged. Boundary: 9 of 241
  dropped objects on Full130 were dropped more than once (0 of 117 on the 27);
  a +5m keep margin gives 5, not taken. The A round's live-entry FF difference
  of 6 sends is at 356/14 55.04-55.09s and 366/47 27.78-27.92s; this round
  changes live FF by 28 sends in seven places (standalone 0). At 366/29
  20.3-21.1s the FF object is the same and only the shown value differs where
  FF alternates among objects - presumably the existing stabilize_slots
  debounce reacting to the side slots; not confirmed. Regressions: residual 0,
  c1/23 6/6, seven waiting scenes 100%, ten earlier bookmarked segments same
  per bookmark, B'/D/H and 3e1/3 scenes unchanged, live entry no exception on
  130 + 27. 1,019 tests (4 new cases; two A-round tests moved their far car
  from 55-60m to 44-45m because the new limit hides it, still failing on the
  pre-A module). PC time unchanged. Open: real neighbours beyond 50m behind a
  lead are not shown; live FF differences unexplained; out of scope as before
  (lone resting return admitted, crawling neighbour, bus at a stop, objects
  piling up while stopped). Not validated in the car. Commit files when asked:
  the map module, tools/ccnc/test_radar_map.py, the experiment doc, AGENTS.md;
  never CLAUDE.md. Evidence: .analysis/archive/2026-10-13/ccnc-far.
- On 2026-10-13 (third road-test round: second lead shown LF/RF on curves; NOT
  committed) map is 25ff38ec9160 (from 6dd843a64690; the bollard entry below WAS
  committed as 099ce103). Rules, both only while an FF is displayed and for a
  candidate more than 5m beyond it (no side slot, not promoted to FF): (1') on
  a curve (ego lane centre at the candidate over 2m off ego's axis) a side slot
  needs the candidate 1.2m past ego's lane line; (2) each motion-confirmed
  object keeps a trail (one world point per 3m, 20 points) and gets no side
  slot when its trail points 5-25m ahead of ego number two or more and all lie
  within .3m of ego's lane lines. The user's seven bookmarks are NOT all gone:
  sends of the bookmarked object in +-3s, before -> now: 3e0/0 43.9 54 -> 9,
  3e0/1 45.0 27 -> 2, 3e0/1 54.7 56 -> 31, 3e0/2 7.9 56 -> 1, 3e0/2 20.4 79 ->
  7 (the car behind it then shows longer, 40 -> 74), 3e0/2 50.0 52 -> 21,
  3e0/4 1.5 48 -> 48. Those cars lie 1.5-1.9m (median) past the lane line, and
  3e0/4 is on a straight (lane centre .1-.8m) with no trail. FF never changes.
  Standalone vs 6dd843a64690: Full130 removed 1,622 (all class 2), swapped 151,
  added 0; 27 segments removed 909, swapped 134. Measured, not taken: (1)
  without the curve condition (margin 1.0/1.2/1.5) removes 2-5x more and on
  straights hides real neighbours (video: 3e1/5 8.1-24.2s RF 273 sends, 3e1/14
  24.9-27.9s LF 53); margin 1.5 still leaves 3 of 7; (3) slots from the model
  path clears none of the seven and changes FF on Full130 (reference only).
  Video of what the adopted rules remove: 356/7 49.9-54.6s RF correct (one
  lane each way), five others undecided, none confirmed a real neighbour; most
  removals unverified. Trail: 3.2-3.3 points per object; undecided (under two
  points in the stretch) for 53% (27) / 27% (130) of side displays 25m+ past
  the lead; 26-32% fall at 1.0-2.0m, where real neighbours also are, so a
  drifted second lead there is not caught. Regressions: residual 0, c1/23 6/6,
  seven waiting scenes 100%, ten earlier bookmarked segments same per bookmark,
  earlier road-test scenes (B'/D/H seven, 3e1/3) unchanged, live entry no
  exception on 130 + 27 (live FF differs by 6 sends on 130, cause not
  checked). 1,015 tests (7 new cases, none changed). PC time unchanged.
  Model paths for (3) are in .analysis/cache/ccnc-path-v1. Out of scope and
  open: lone resting return admitted (3e1/5 49.3, 3e1/4 6.0), crawling
  neighbour (3e1/6 19.8), bus at a stop (3e1/6 37.5), objects piling up while
  stopped. Not validated in the car. Commit files when asked: the map module,
  tools/ccnc/test_radar_map.py, the experiment doc, AGENTS.md; never CLAUDE.md.
  Evidence: .analysis/archive/2026-10-13/ccnc-curve.
- On 2026-10-13 (second road-test round: lone bollard admitted as a waiting
  vehicle; NOT committed) map is 6dd843a64690 (from b79648fe642b; the entry
  below WAS committed as 1f3abaa2). ONE source change, the user's variant (2):
  a resting return on the line of a band (+-BAND_BIN .25m from the band's
  offset, within BAND_MEMORY 40m of its members along the road) is not admitted
  as a waiting vehicle (_band_line). Standalone vs b79648fe642b: 27 segments
  1,008 sends removed, all one bollard (3e1/3 15.1s to 3e1/4 5.6s, 56 measured
  + 952 stop memory); Full130 output byte-identical, so residual 0, c1/23 6/6,
  seven waiting scenes, c1/7 and c2/24 queued cars 100%, 356/3 row 0% are
  unchanged; ten earlier bookmarked segments identical; live entry no
  exception on 130 + 27; admitted object-sends 7,298 -> 6,250 (27) and 31,537
  unchanged (130). 1,008 tests (3 new cases, none changed). PC time unchanged
  within repeat spread. Measured and NOT taken: (1) storing resting returns on
  a band line out to 40m does not fix 3e1/3 - the 11.5m gap to the next post is
  a road distance and stays beyond WALL_CONTINUE 7m however early the post is
  stored; it adds 3% objects, 8.3% of the far-stored objects on Full130 later
  had vehicle evidence (345 class 2), band-line match failure 18% (Full130) /
  35% (rainy 27) at +-.25m, displays +12 on Full130, and it removes 31 false
  sends at 3e1/4 6.1-7.6s that (2) leaves. (3) BAND_CONTINUE 12m also removes
  3e1/3 (reference only, as the user asked). (1)+(2) = sum of both. NOT FIXED:
  3e1/5 49.3s and 52.5-53.5s LF - no band or wall ever forms on that side
  there (two lone resting returns at 15-17m, 1.0-1.6m from the inner line), so
  no band-based rule can act; the object could not be identified on the dusk
  video. Remaining: that scene, 3e1/4 6.1-7.6s LF, and the cost of (2): a real
  car standing on a band's line within 40m of its end is not admitted (none in
  157 recorded segments). Not validated in the car. Commit files when asked:
  the map module, tools/ccnc/test_radar_map.py, the experiment doc, AGENTS.md;
  never CLAUDE.md. Evidence: .analysis/archive/2026-10-13/ccnc-bollard.
- On 2026-10-13 (first road-test round; NOT committed) the user reported 21 Road
  Viewer bookmarks from driving fed331cb in the car (27 segments, routes
  000003e0--056c8059b7 0-5 and 000003e1--0786defbd1 0-20, rain at dusk) and asked
  for three small fixes only. Map is b79648fe642b (from 1ace50c16016). The two
  CCNC entries of 2026-10-12 below marked "NOT committed" WERE committed as
  fed331cb. The 27 segments are decoded to .analysis/cache/ccnc-replay-v3
  (manifest .analysis/archive/2026-10-13/ccnc-road/road-manifest.json); Full130
  stays ccnc-replay-v2. Changes: (1) a moving vehicle measured outside its
  side's outer lane line for over 1s with sideways speed under .3m/s is
  "settled" and loses the +.75m outer margin, until it is measured inside the
  line again; kept margin where the outer line is under 2.3m from the inner one
  OR the return is under 3.5m from the inner line. Two departures from the
  request, both measured: the latch (deciding per report let the far-lane car
  back twice at 3e1/15 53.0s, sideways estimate -.3) and the 3.5m condition (as
  requested, 358/0 49.8-53.8s RF lost 65 sends of a real car in the last lane
  of a two-lane road, line drawn at 2.6-3.1m, return at 3.3m; 358/7 29 more).
  3.5 was the only value tried. (2) doppler_confirmed needs obj.tracked >= 3;
  samples are still collected and confirm once measured reports arrive. (3) no
  Doppler samples while |yaw rate| > .3rad/s (update() stores yaw_rate);
  variant (b), adding yaw * (3.5 + |y|) to the allowance, also removed both
  bookmarks but changed more at small yaw rates (Full130 106 removed + 60
  swapped vs 89 + 0) and was not taken. Numbers vs fed331cb, standalone:
  Full130 removed 248 (class 2 115, of which 86 beyond 30m; speed-only 133),
  added 0, swapped 323; 27 segments removed 29, added 0, swapped 196.
  Beyond-normal-outer-line side sends 381 -> 198 (27) and 1,646 -> 1,250
  (Full130). Speed-only displays with under 3 measured reports 20 -> 0 / 112 ->
  0, above .3rad/s 16 -> 0 / 64 -> 0; no class-2 loss from (2) or (3). Video
  (qcamera) of the larger Full130 changes: far-lane cars replaced by the
  adjacent one at 3d8/9 RF, 356/20, 356/21, 39e/19, 366/65 (correct);
  undecided 3d8/9 LF truck, 3c1/9, 3c2/23; no video 38a/10; (2)/(3) removals
  were a car park's parked cars and walls passed while turning, except a real
  parked SUV at 3c2/24 LF (10 sends). All seven in-scope bookmarks are gone in
  replay (3e1/13 only from 12.9s: at 12.4s the car still moves sideways at
  .5m/s). Residual 0, c1/23 6/6, seven waiting scenes 100%, the ten earlier
  bookmarked segments identical to fed331cb, live entry no exception on 130 +
  27. 1,005 tests (8 new cases; the synthetic point() helper now defaults to
  trackState=2 because 21 older tests confirmed vehicles by speed without a
  track state). PC time unchanged within repeat spread (.65/.72 vs .67/.70ms).
  Known costs: a car returning from the lane beyond is shown .75m later (132
  Full130 sends dropped after settling with sideways estimate >= .3, 123
  inward); a car riding 3.5m+ past the inner line in a last lane; parked cars
  while turning. Next round, untouched: second lead shown LF/RF on curves
  (3e0/0-4; model lanes bend too little beyond 30m), lone bollards admitted as
  waiting (3e1/3, 3e1/5; band storage beyond MAX_X 20), crawling neighbour not
  shown (3e1/6 19.8), bus at a stop (3e1/6 37.5). 3e1/1 35-45s missing 65m FF
  is an oncoming vehicle while waiting to U-turn: correct, not a loss. Not
  validated in the car. Commit files when asked: the map module,
  tools/ccnc/test_radar_map.py, tools/ccnc/replay_radar_map_fast.py, the
  experiment doc, AGENTS.md; never CLAUDE.md. Evidence:
  .analysis/archive/2026-10-13/ccnc-road.
- On 2026-10-13 the user requested merging carrot-wip into ccnc-hda1 again.
  origin/carrot-wip 2ec25da3 (four commits: green signal group release and small
  night red tracks, stopped signal veto released on one central red-green
  transition, two docs commits) was merged; only AGENTS.md conflicted and both
  sides' entries are kept, CCNC entries first. tools/ccnc/pytest_plugin.py
  BASELINE follows the merged carrot-wip revision (settings catalog unchanged).
  The previous merge 0e379e08 was pushed on 2026-10-12 at the user's request;
  this one is not pushed or deployed.
- On 2026-10-12 the user requested merging carrot-wip into ccnc-hda1 again, after
  the CCNC wall work was committed as fed331cb (so the "NOT committed" notes in
  the two CCNC entries below are historical). origin/carrot-wip 0f417955 (two
  commits: signal preparation history between inference arrivals, persistent
  night red lamp groups) was merged; only AGENTS.md conflicted and both sides'
  entries are kept, CCNC entries first. tools/ccnc/pytest_plugin.py BASELINE
  follows the merged carrot-wip revision. Not pushed or deployed.
- On 2026-10-12 (seventh follow-up: wall work wrap-up, commit preparation; NOT
  committed) map is 1ace50c16016 (from fb02d3a9f1bc; 9c0da097c29f at 12eda56d).
  ONE source change: a motion-confirmed object reporting over 1m/s is not classed
  stationary. The travel bound alone had classed 525 object-sends of class-2
  vehicles reporting >2m/s as resting; now 0, class-2 resting 41,059 -> 39,523
  (36,888 at 12eda56d), wall table identical, no display removed (5 added, 19
  swapped), 840 standalone detect values 2->1. The user's two variants were
  measured and not taken as given: (a) no stationary on motion-confirmed objects
  also strips standing vehicles (41,365 detect values, 64 class-2 sends lost);
  (b) |vLead| <= 1m/s for every object breaks wall forming (>=60km/h members
  44.0 -> 35.2%) because fixed returns report up to vEgo*(1-cos az). The
  approach-only side-slot rule STAYS: replacing it by last_motion from the
  reported speed leaves 18 of the 155 exposed fixed-return sends, and adds 1,975
  RF memory sends (38a/10-11) and loses 55 class-2 sends because last_motion
  also drives the memory conditions. Totals since 12eda56d on Full130: removed
  215 (206 approach-only crossing objects, not counted either way per the user),
  added 30 (28 a real waiting car), swapped 349, no class-2 loss, residual 0,
  c1/23 6/6, seven reviewed waiting scenes 100%, live entry no exception; all 18
  bookmarked false scenes gone (16/0.65 too), 23 kept. 997 tests (11 new cases,
  none changed). PC time mean .58-.60 -> .64-.66ms (+9-12%), p99 1.03-1.06 -> 1.14-1.17ms; without the band code the module times like 12eda56d, so the band histogram and prediction cost about .06-.07ms. Known issues now: approach-speed confirmation
  itself, coasted re-links, 356/2 FF swaps while turning, 358/0 fence, segment
  15 far FF toggling, MEMORY_MAX_X 15, band joining without effect, no confirmed
  display benefit of bands, no speed floor, no vehicle validation. Commit files:
  the map module, tools/ccnc/test_radar_map.py, replay_radar_map_fast.py,
  replay_live_path.py, the experiment doc, AGENTS.md; never CLAUDE.md.
- On 2026-10-12 (sixth follow-up: walls at speed; NOT committed) map is
  fb02d3a9f1bc (from 9c0da097c29f). Source changes: (1) stationary = net
  movement over the history window <= .15 * ego travel + .5m (was: all reports
  within .4m), since reports scatter .5-.7m above 30km/h; (2) an object
  confirmed only by approach-speed samples is not shown in a side slot
  (_approach_only, same condition as the memory exclusion): step 1 zeroes the
  map velocity of resting returns and exposed 154 sends of class-1 roadside
  returns that only the "opposing" check had hidden; (3) bands ADDED to the
  straight rows: resting returns at one offset (+-.25m) from ego's own lane
  line, collected over 40m driven, three posts <=3.5m apart over 4m, continued
  within 7m, folded after max(3 spacings, 10m) driven without a measured
  member; (4) a fresh return within .5m of a band's next expected post joins at
  its first report unless it has vehicle evidence or is lane-interior. The
  user's literal step 2 (bands replace rows, road edge/outer line reference, no
  spacing limit) was measured and rejected: real waiting cars lost (c1/7 18%,
  c2/24 4%), 356/3 bollard row shown as memory; row members keep their offset
  within .5m from the lane line in 94.5% (>=60km/h) but 63% at rest, road edge
  90%/36%. Step 4 (joining bands across <=12m) changed nothing (96 joins) and is
  NOT adopted. Full130: wall members among resting side returns >=60km/h 21.1 ->
  44.0%, curve 9.2 -> 31.8%; display change of steps 2-3 is 11 sends removed
  (356/3) and 31 swapped; step 1 removed 13, added 40 (28 a real waiting car by
  video); residual 0, c1/23 6/6, live entry no exception; bookmark 16/0.65 gone,
  23 kept. 995 tests (9 new cases, none changed). PC time mean +8-9%, p99
  1.19 -> 1.30ms. User decision the same day: crossing pedestrians/cyclists
  shown as vehicles while measured are acceptable and are not counted as loss or
  gain; their memory stays blocked (_crossing). Tools: replay_radar_map_fast.py
  --structure, replay_live_path.py rows carry structure counts. Open: steps 2-3
  have no confirmed display benefit; bookmarks 5/10/19 still have no wall; 525
  object-sends of slow class-2 vehicles classed resting; no vehicle validation.
  Private evidence: .analysis/archive/2026-10-12/ccnc-wall.
- On 2026-10-12 the user requested merging carrot-wip into ccnc-hda1 after the
  CCNC radar-map fixes were committed as 12eda56d (so the "NOT committed" notes
  in the entries below are historical). origin/carrot-wip 5e187ab9 (36 commits:
  signal assistance, Mountain Dew eGPU, Bosch MRRevo14F passive radar tracks) was
  merged; only AGENTS.md conflicted and both sides' entries are kept, CCNC entries
  first. Not pushed or deployed.
- On 2026-10-12 (fifth follow-up, commit preparation; NOT committed) map is
  9c0da097c29f. ONE source change: when a radar id creates a new object and
  another object was measured with that id within .15s, the track jumped beyond
  the association gate; if the new report has radar class 2 the object left
  behind is not a _stop_memory candidate (superseded). The unconditional form
  erased a 1.1s memory of a real waiting car (388/6 23.6-24.7, video: resting
  body, reflection point changed), hence the class-2 condition. Result: the
  17/35.2-37.0 fragment memory (35 sends) is gone, 17/35.5 keeps only 3 sends of
  a real passing car; Full130 no add/remove, 53 sends in 366/65 swap a passing
  car's fragment memory for the standing white car's; live entry no exception,
  c1/23 6/6; 986 tests (one new, two cases). Measured, NOT adopted: (a) the
  existing selector's speed_thresholds on non-vision candidates: Full130 removes
  1,518 sends, 1,355 of radar class-2 objects (slow or stopped vehicles in the
  next lane while ego drives), 10 segments 89/80. The user's 76 high-speed
  resting own-lane returns (ids; 55 with vision lead prob <.5) were never
  displayed by the current module and 7 sends by the as-driven module, already
  removed by the Doppler band change. (b) coasted re-link blocked after 2s
  unseen: 2,893 links cut (176 to class-2 objects), no add/remove, 110 swaps,
  gain is 7 sends in one scene. Summary since bbfeaa80 (00c01eab796e): seven
  rules (Doppler band, lane width, waiting tracked>=5, near Doppler exclusion,
  release on reappearance, approach-only not remembered, id moved on); bookmarks
  17/18 gone (only 16/0.65 barrier left), 23 recovered, 29 afterimage gone;
  Full130 map-only removed 202, added 357 (349 class 2), replaced 156, residual
  0, c1/23 6/6; PC time mean +2.3..+5.5%, p99 1.09-1.33 vs .97-1.32ms. Files:
  the map module, tools/ccnc/test_radar_map.py, new tools/ccnc/replay_live_path.py,
  the experiment doc, AGENTS.md. Root CLAUDE.md is the user's untracked local
  agent setting, not part of this work: never stage it (no git add -A). Open:
  16/0.65 barrier; approach-only confirmation and coasted re-links; FF Doppler
  swaps while turning; 358/0 fence; segment 15 far FF toggling; no high-speed
  floor in the map. No vehicle validation, no commit/push/deploy. Private
  evidence: ccnc-release-4 archive.
- On 2026-10-12 (fourth follow-up) map is 3cf97550948b. ONE source change, per the
  user: Doppler samples are counted as forward (above the fixed-return band) or
  approach (towards ego); an object with no forward sample, radar class 2, latched
  vision match (vision_seen) or waiting admission is not a _stop_memory candidate
  (a same-direction vehicle never moves towards ego). 28->29: the 929-send
  afterimage is gone, the real waiting minicar's memory shows in 28 instead;
  Full130 map-only zero display change, c1/19, c2/23, 366/65-66 memories and
  reviewed waiting scenes unchanged, residual 0, c1/23 6/6; live entry point no
  exception; ten bookmarked segments identical to the previous module (16/18
  gone, 23 recovered). 984 tests (one new, two cases, from the 28/29 numbers; no
  existing test changed). PC time vs the original module mean +0.9..+7.9%, p99
  1.07-1.41 vs 1.05-1.45ms, run-to-run noise of a few %. Measured, NOT adopted:
  (a) approach samples only from measured tracks: the 29 return stays unconfirmed
  (4 of 19 samples) but Full130 loses 58 sends, 10 while legacy shows, unreviewed;
  (b) no coasted report may re-link an object unseen >0.3s: blocks the 43.2s
  re-link but cuts 9,658 links (1,089 to class-2 objects), 170 sends change, 2
  class-2 sends lost, more fragmentation; (c) pass-through release (moving object
  within +-1.5m/1.0m of a remembered position): removes the afterimage alone, 59
  releases, 51 sends swap to another memory, 366 kept. 17/35.2-37.0 fragment
  memory cause found, fix only proposed: the Doppler object appeared with
  20d78a176912 and became visible with 9f35bfc73247; radar id 39 reported
  2.5-4m/s while its position stayed at 2.2-2.7m, then jumped 1.17m (same-id gate
  ~.89m) creating a second object, and the stale one passed the "alongside"
  memory condition for 5s. Proposal: an object whose radar id was measured in the
  previous cycle and now creates a new object elsewhere is not remembered. Still
  open: that fragment; approach-only confirmation itself (7 measured sends at
  28/43.2); coasted re-links; FF Doppler swaps while turning (356/2); the 358/0
  fence; segment 15 far FF toggling; 16/0.65 barrier. A leftover hung replay
  process from the first session ran 1.5h unnoticed: after a multiprocessing run
  check that no worker is left. No vehicle validation, no commit/push/deploy.
  Private evidence: ccnc-release-3 archive.
- On 2026-10-12 (third follow-up) the user had the memory-release rule put into
  the source with a lateral condition, and segment 29 examined. Map now
  9f35bfc73247. Rule (ego stopped/slowing): an object first measured 2-6m ahead
  of and within 1.0m laterally of a remembered object in that side lane at
  <=3m/s, with radar class 2 or the remembered object's radar id, whose gap
  then grows 1m, releases that memory (RELEASE_* trial constants; cancelled if
  the object is measured again). 17: release still at 29.6s; stale sends removed
  326 (17: 288, 18: 38), not 361, because a different 1.75s memory (35 sends,
  17/35.2-37.0, a split fragment of a passing car that got Doppler evidence at
  2.8m after the x<3/|y|<1.5 correction; inferred, not isolated) now shows there.
  366/65-66 memory windows identical. Full130: 13 releases (18 without lateral;
  the 5 dropped were 1.10-2.35m aside), zero display change; live entry point
  no exception, c1/23 6/6, slot counts unchanged. 982 tests (one new test, two
  cases, from the 17 numbers). Timing not remeasured. Segment 28->29: bookmark
  reproduced, 29 presence 100% equal to the car. The afterimage is NOT fixed and
  is a separate cause, source untouched: a return 16-20m ahead right 3.4m
  reported -0.9..-1.9m/s towards ego at 28/28.1-28.4 and was Doppler-confirmed
  (approach-speed evidence kept in the first follow-up), unseen 14s, re-linked
  at 43.2s to two coasted reports of the same id at 13.8m, then remembered
  (28: 180 sends, 29: 929 sends/46.9s). By video the spot is empty crosswalk;
  a white minicar waited behind it and left at 53-55s, cars then drove through
  the remembered position; what the 28s return was is not identifiable. Known
  open: that afterimage; the 17/35.2 fragment memory; FF Doppler swaps while
  turning (356/2 26.4-27.7, 17 sends); why the 358/0 fence got Doppler evidence;
  cause of segment 15 far FF toggling; 16/0.65 barrier vs BODY_CLEARANCE. No
  vehicle validation, no commit/push/deploy. Private evidence: ccnc-release-2.
- On 2026-10-12 (follow-up to the bookmark entry below) the user asked for a
  narrower memory-release trial and checks of the leftovers. Map now 20d78a176912.
  ONE source change, correcting the previous step: "no Doppler sample below x 3m"
  delayed real vehicles right beside ego (Full130 RF52/LF9 sends showed a vehicle
  12-46m ahead instead; three RF scenes video-checked: car, double-decker bus,
  car), reported before only as "replaced". The exclusion is now x<3m AND |y|<1.5m
  (DOPPLER_NEAR_Y; false speeds were within 1.2m of ego's axis, vehicles alongside
  2.0-2.6m): bookmarks unchanged, 61 sends restored. When reporting display
  changes, "replaced by another object" must be examined like removals. Release
  trial (scratch only, NOT in source): object first measured 2-6m ahead of the
  remembered position in that side lane at <=3m/s, radar class 2 or same radar
  id, gap then growing 1m. 17/18: memory released at 29.6s, 361 stale sends gone.
  366/65-66 memory kept (by video the remembered object is a white car ahead-right,
  not the bus; cars pass in the next lane, one first measured +2.2m at 2.7m/s
  and excluded only for lacking class 2, 1.5-1.9m further out laterally). Full130:
  18 objects released, zero display change, so the benefit rests on one scene.
  Video: of 28 removed-while-legacy-shows sends, 14 opposing-lane vehicles, 5 a
  real stopped car at 5.2m lateral (c1/15), 9 no vehicle; G's 8 non-class-2
  additions: 4 median fence (false), 2 bus (real), 2 nothing visible. Segment 15
  FF 95.9%: far (121-197m) class-2 vehicles toggling a few sends apart plus 3
  start-up sends; nothing within 30m. Live entry point on all130: no exception,
  c1/23 6/6. PC map time mean -0.1..+3.3%, p99 .98-1.16ms vs .97-1.14. 980 tests
  (one check added). Segment 28 rlog has arrived; 29 not yet done. No vehicle
  validation, no commit/push/deploy. Private evidence: ccnc-release archive.
- On 2026-10-12 the user gave 18 driver-bookmarked false CcncRadarMap displays and
  one missed vehicle from a real drive (3dd, 10 segments; the option was on) and a
  pre-tested fix list. New tools/ccnc/replay_live_path.py replays the live entry
  point per recorded 0x162 send and compares with the decoded sendcan and the
  existing selector; map presence matched the car 95.9-100% per segment/slot and
  all 18 scenes reproduced. Map now 567e45a7531e (was 00c01eab796e): (1) Doppler
  evidence is speed outside the fixed-return band [0, vEgo*(1-cos az)], i.e.
  abs(vLead - explained): fixed returns report less than the geometry and were
  confirmed as vehicles in .25s. The user's "positive difference only" gave the
  same 10-segment result but on Full130 removed 600 and swapped 760 displays of
  class-1 objects crossing in front of a stopped ego (395/7, 356/21; not video
  identified), so approach speed stays evidence. (2) _lane: a confirmed moving
  vehicle uses the wider of outer line/road edge. (3) waiting admission needs
  tracked >= 5 (WAITING_TRACKED). (4) no Doppler sample below x 3m
  (DOPPLER_MIN_X). Bookmarks: 16/18 gone, 23/20.86 recovered (14 -> 20 sends);
  10 segments lost FF47/LF98/RF10 displays, none with radar class 2 or vision,
  and gained LF12/RF11, all class 2. Full130 map-only: residual 0, c1/23 6/6,
  missed vehicle-evidence displays LF1708/RF1584/FF2972 -> 1633/1512/2972;
  removed 202 (no class-2 object; 28 while legacy shows that slot), added 357
  (349 class 2; 205 where legacy shows nothing), neither video reviewed; reviewed
  waiting scenes and the c1/19, c2/23 memories unchanged; hybrid identical.
  980 tests: two waiting tests now give trackState=2 (their input was a never
  measured return, the cause of 16/25.6), three recorded-scene checks added.
  Left: 17/35.5 is by video a REAL waiting car remembered ~30s after it drove
  off; a "moving vehicle within 5m ahead in that lane clears memory" trial also
  erased the real 366/66 waiting bus, not adopted, no 8s cap. 16/0.65 is an
  orange barrier at .83-.99m lane-line clearance while that real car was at
  .80-.94m, so BODY_CLEARANCE stays .9. tracked>=5 is marginal for 11/19.9
  (tracked 4). No timing, no vehicle validation, no commit/push/deploy. Private
  evidence: ccnc-bookmarks archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12 the user asked for a Carrot Web option that switches CCNC vehicle
  selection to the radar map: CcncRadarMap (default OFF, VEH_CANFD group, needs
  CcncRadarVehicles, not HDA2). OFF keeps the existing extension d473d0fb4bd7
  byte-identical; ON calls hyundaicanfd_ccnc_radar_map.update_vehicles (map-only
  selection, existing deadband/colour/BSD conventions, falls back to the existing
  selector on an exception). CcncVehiclePositionCorrection also applies on the
  map path, keyed by map object: centre lock, lateral following, slot debounce
  and the existing position filters; the temporal track checks are not used and
  remembered vehicles keep their last drawn position. Live replay on152695
  sends: no exception, c1/23 6/6, 706 slot-presence changes (FF<->side debounce),
  lateral step p99 .20-.23m -> .075m. Icon behaviour not vehicle-validated. Toggling resets
  both selectors. Without the radar's own motion class the map loses ~40-48% of
  vehicle-evidence side displays (22400/46986 LF), so RadarPoint.motionState @12
  (car.capnp) is now filled from VALID2 in hyundai/radar_interface.py: shared
  radar code, NOT committed, NAS replay image NOT rebuilt, and the radar tests
  cannot run on this PC (generated radar DBC missing). The controller has no
  message timestamps; arrival stamping changes a few hundred of 152695 sends.
  Live entry point replayed on all130 segments: no exception, c1/23 6/6; one
  borderline scene (38a/10-11) remembers a class-2 vehicle beside ego that the
  replay drops. Same day before this: memory rules added to the map (now
  96a0f38c31f1): objects that travelled >=1.5m sideways with less forward travel
  (395/7 pedestrians/cyclist on a crosswalk, remembered 24-30s) are neither
  waiting vehicles nor remembered; a never-moved neighbour is remembered only
  after 40 observations with trackState 2. Map-only memory/waiting displays
  absent in legacy 38 scenes/2817 sends -> 15/917; removed 2016 = 1781 false
  (video), 217 unverifiable, 18 real (bus .7s, SUV .2s). "Moved vehicles only"
  memory rejected: loses real occluded waiting cars (c1/19 31.5s, c2/23 9.1s).
  PROB and last-report state do not discriminate. 976 focused tests. PC timing
  unchanged (.51-.66ms). No vehicle validation, no commit/push/deploy.
- On 2026-10-12, user said to continue the remaining work. Replay tool gained
  --explain (why each legacy-selected point is not the map-only display).
  Offline map d1641d1970e4: (1) lane use matched to the live selector: lines
  from probability .1, side lane bounded by outer line and/or road edge with
  1.8m room, .75m outer margin for moving vehicles, straight 1.2m FF corridor
  with no lines, side slots to 80m; trusted-interior (.6) unchanged. (2) BUG
  fixed: the map reset itself whenever filtered vEgo read slightly negative at
  standstill, erasing the stopped lead every frame (~4900 FF misses); reset
  only below -.5m/s. (3) far vehicle candidates accepted to 40m lateral.
  (4) lane handling optimized with identical output on 5997 sends. Full130
  map-only: reviewed residual0, c1/23 6/6; vehicle-evidence legacy displays
  missed 22368 -> 5645 (~2.5%: LF1440/RF1400/FF2805; mostly lane slot
  disagreements); added radar-class-2 LF2042/RF2051/FF1014, no-evidence 10;
  map stop memory adds LF6296/RF2714 and misses 8402 legacy memory displays,
  not reconciled. Hybrid unchanged: residual1, no additions, 0 vehicle-evidence
  removals. Timing, run alone on5 segments x3: legacy .68-.92ms, map-only
  .69-.84ms mean (parity, -9..+9%), p99 worse (1.25-2.52 vs 1.19-1.62ms);
  hybrid adds ~.9-1.0ms to legacy. Map-only is not faster than legacy. PC
  timing only. 958 tests. Live extension d473d0fb4bd7 unchanged; no
  commit/push/deploy. Private evidence: ccnc-why archive.
- On 2026-10-12 the user instructed: whenever LF/RF removals or displays are
  measured, measure vehicle-or-not with them. All display-change counts are now
  split by vehicle evidence (radar class 2 / Doppler / vision / stop memory /
  never moved with ego moving or stopped); removed uses the legacy-selected
  point, added uses the candidate's object (replay rows carry `evidence`).
  Evidence is not video truth. Reviewed-scene residual counts only displays
  within15m of the structure: 12 "residual" sends in 395/6 were a class-2
  vehicle at ~59m taking the slot. Wall gap: 3.7/5/8m hide stop-memory displays
  (116/189/433 vs14) and 8m hides15 vehicle-evidence displays; user judged
  widening unsuitable, keep 3.5m. User said 30m was an arbitrary value and
  allowed a better one: structure-suspect side displays are 95% within20m and
  20/30/40m give the same hybrid result with fewer objects at20m (max145 vs
  256 cap), so mapping range is 20m (keep22m). Beyond it only radar class 2 or
  vision-matched returns are admitted to 200m, never remembered; single Doppler
  samples are not an admission reason (noise accumulated as evidence). Offline
  map 43ede572c82f, 958 tests. Full130 replay: hybrid residual1, c1/23 6/6, no
  additions, removes 0 vehicle-evidence displays, LF816/RF67/FF104 never-moved
  ego-moving, 15 stop-memory, 17 never-moved ego-stopped. Map-only residual0,
  c1/23 6/6, but removes 22368 (~10%) vehicle-evidence displays and most legacy
  stop-memory displays, and adds 14400 radar-class-2 displays legacy did not
  show; both causes unexamined. No reliable timing yet. Live extension
  d473d0fb4bd7 unchanged; no commit/push/deploy. Private evidence:
  ccnc-wall-range archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12, user approved offline-first use of the radar's own motion class
  (no shared radar code change yet). tools/ccnc/replay_radar_map.py reads raw
  0x3A5.. CAN with --radar-evidence; offline map 8e888df4801b takes vehicle
  evidence from radar class 2 (distance within1.5m) or >=5 samples of speed
  beyond geometry by .7+.25*|aEgo| m/s inside45deg. User asked whether tests
  were holding the work back: seven tests required position-only motion with
  zero reported speed, a synthetic input never seen in logs. Decided by data:
  with position-only evidence map-only reviewed29 residual17, without it 3
  (c1/23 RF6/6 both), so DISPLACEMENT_EVIDENCE is False and those tests now
  report speeds matching their motion; say which tests change and why whenever
  a test is edited, and prefer recorded-log results over synthetic invariants.
  958 focused tests. Full130 replay: legacy6 / hybrid1 / map-only3 residual;
  hybrid removes FF110/LF829/RF67 vs legacy, adds none, far kept. Map-only is
  first better than legacy on reviewed structures while keeping the known
  vehicle, but cannot yet replace it: no display beyond30m by design, misses
  ~10% (13047) of radar-class-2 legacy displays within30m (cause unexamined),
  different memory behaviour, never-moved stationary policy undecided, and
  FF683/LF3012/RF1548 displays absent in legacy are unreviewed. No timing
  measured. Live use needs radar_interface.py to export the class. Live
  extension d473d0fb4bd7 unchanged; no commit/push/deploy. Private evidence:
  ccnc-evidence archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12, after the user asked for new ideas instead of more map tuning,
  raw CAN study (analysis only, no source change) found unused radar fields in
  the 0x3A5.. front track message. Byte3 bits4..3 (DBC VALID2): 1 stationary
  (94.4% of Doppler-static), 2 forward-moving (96.6%) and LATCHED after the
  target stops (no 2->1 transition in12 segments; stopped-after-moving tracks
  1599 vs 9), 0 mostly oncoming/unknown. Never 2 on reviewed structure points;
  about76% of legacy side displays within30m have 2. c1/23 real right vehicle
  is 0 (slow starter) but has Doppler residual >.7m/s on26 frames. VALID/PROB:
  vehicles 97% state2/PROB>=64; near high-azimuth points incl. structure
  residuals and c1/23 frames are state0/1/3 or low PROB, i.e. radar-coasted
  tracks, consistent with the earlier position drift. Semantics are inferred
  statistically, not documented. Remaining ambiguity is only never-moved
  stationary returns (~10% of near side displays, 8154 sends, incl. all6
  reviewed residuals and real already-stopped neighbours): motion evidence
  cannot separate them; walls/geometry are needed only there. Scratch trial
  requiring field2 or >=5 Doppler samples for legacy motion admission: LF141/
  RF19/FF65 removed, FF9 added, residual6 unchanged, c1/23 6/6; FF changes
  unreviewed. Using the field live needs radar_interface.py to export it
  (shared radar code, user decision, NAS replay rule applies); not done.
  Private evidence: ccnc-doppler archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12 (continued), three more offline map changes (module now aa8de379b9cd):
  motion confirmation needs >=.4s, >=.3m and rate > .35+.25*|aEgo| (5cm report
  resolution; report delay measured .07s on 395/6 but .22s on 356/0, so only a
  bound, not a constant); wall gap 3.5m (356/1 posts measured 3.07m after
  compensation); a side lane with a missing outer line uses ego lane width only
  if the road edge leaves 1.8m (cause of map-only c1/23 loss: outer line prob 0).
  955 focused tests. Full130 replay: compensated hybrid reviewed29 residual1,
  c1/23 6/6, far kept, reappearing sends 85->17, but removals vs legacy grew to
  FF110/LF822/RF59 (two new FF windows are6-9-point stationary rows on a lane
  line; not video-reviewed). Map-only: residual 23->8, then 17 once c1/23 was
  recovered (6/6); added LF2729/RF2183 vs legacy unreviewed. Each map-only fix
  exposed another failure; user said the work felt circular and asked for new
  ideas. Do not keep tuning map thresholds without a new classifier idea.
  No timing measured, no commit/push/deploy, live extension unchanged. Private
  evidence: ccnc-map-next archive.
- On 2026-10-12, user listed the intended map rules and asked for a plan and
  sequential work. Review: most rules already existed in the offline map; the
  defect was its motion input. Added to hyundaicanfd_ccnc_radar_map.py only
  (now 7a9a33bc41ff): (1) geometry compensation: unconfirmed objects subtract
  the accumulated min(max(vLead,0), vEgo*(1-cos az)) from map position, history
  and speed; confirmed vehicles keep reported positions; raw-point matching uses
  reported coordinates. (2) stop memory, per user: only stopped or <=10km/h
  slowing, never while driving, only if that side slot is otherwise empty,
  unseen vehicle from -4m (drawn at distance 0; legacy limit is -1m) to +5m,
  not with same-side blind-spot display. -4/+5m are trial values, not log-fitted.
  951 focused tests pass. Full130-segment/152695-send replay, reviewed29 residual:
  legacy6, previous hybrid2 -> compensated hybrid1, integrated5 -> 4, map-only
  59 -> 23. c1/23 RF6/6 in hybrid/integrated, still0/6 map-only, so the map
  cannot replace legacy selection. Compensated hybrid vs legacy: FF71/LF524/RF47
  removed, none added, all58308 beyond30m kept. Versus previous hybrid: 85 sends
  REAPPEAR (mostly 356/1 3.48-6.64s LF), cause unexamined; newly removed side
  points checked had speed at or below geometry (db/48 max +.31), no video
  identity review. Integrated keeps 395/6 4 sends because unchanged legacy
  admission protects them. Stop memory appeared 858 times in13 segments
  map-only, not checked against video. No timing measured. Live extension
  d473d0fb4bd7 unchanged; no live hook, commit, push or deploy. Private evidence:
  ccnc-map-rules archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12, user asked Claude to continue from the handoff and first confirm
  why 395/6 structure returns gain motion admission. Diagnosis only; no source
  change. LF source33 is measured, one ID: reported vLead .34->2.18m/s tracks
  vEgo*(1-cos(azimuth)) (.37->2.12) as azimuth grows18->43deg, and reported
  distance drifts forward by the same integral (.70m). The speed, three-window
  and .5m displacement tests in update_boundary_admission count one geometry
  effect, not independent motion. All27 moving-ego reviewed structure points
  have vLead at or below this value; above ~45deg reports saturate lower.
  Scratch-only trial 09c0e4c87dcc (copy of d473d0fb4bd7) subtracts that term
  from forward speed before the motion test, thresholds unchanged. Full
  130-segment/152695-send replay: reviewed29 residual legacy6->6, previous
  hybrid2->2, integrated5->1; c1/23 RF6/6 kept. Versus unmodified legacy:
  no additions, LF40/RF19 removed, positions FF6/LF173/RF21 changed, 43 beyond30m.
  With integrated: FF4 at~49m removed and RF2 added in 358/11, unexplained.
  Removed displays are NOT all identity-reviewed (366/64, 356/16, 358/9,
  38a/16, cb/16). Slow real vehicles under~.7m/s are indistinguishable; yaw,
  timing and other radars untested. Not implemented in the new module; live
  extension and d32c0c386b09 unchanged; no commit/push/deploy. Private evidence:
  ccnc-admission-geometry archive; see docs/ccnc_radar_map_experiment_20261011.md.
- On 2026-10-12, user approved combining existing CCNC selection with the map
  after duplicate-work review. Final offline integrated candidate d32c0c386b09
  classifies before selection, carries blocked IDs to fallback/retention paths,
  preserves motion/vision/confirmed-memory protections and ordinary stop admission.
  Retain original simultaneous3point/4m rows: deleting them caused false LF.
  Old stationary memory admits new points only from15m, retaining15..30m overlap
  and old far histories; not all duplicate work or all range risks are removed.
  Share validated lane arrays/projections and timestamp transforms; keep hypot.
  Fresh aligned wall extensions use observed gap variation, never invented points.
  947 tests pass; final130-segment/152695-send replay exactly matches the detailed
  capture. Known c1/23 RF6/6 and all58308 displayed slots beyond30m are preserved.
  Versus legacy: FF2/LF593/RF3 removed, LF9 added; physical identities not all proven.
  Reviewed29 scenes residual6 legacy/2 previous hybrid/5 integrated: mixed result,
  NOT improvement-only. 395/6 false LF inherits original motion admission after
  selection changes; cb/16 RF remains unresolved. Five-case paired PC totals
  1.003..1.265ms, change-4.0..+0.2% vs previous hybrid, still above legacy alone.
  Keep new module offline; original live extension d473d0fb4bd7 remains unchanged.
  No commit/push/deploy. Private evidence: ccnc-map-fusion archive; see
  docs/ccnc_radar_map_experiment_20261011.md for limitations and source identities.

- On 2026-10-11, user requested faster near-field radar-map tracking and then
  explicitly continued wall linking. Final offline candidate b6584c8d9a06 admits
  0..30m, retains to32m/-5m/12m sideways, protects near15m points, widens only
  same-ID near association, caches transforms and keeps fixed wall anchors.
  One drifting member no longer revokes a whole wall; aligned stationary middle
  observations can join adjacent walls. User's distance-formula question was NOT
  permission to change arithmetic: squared-distance trial withdrawn; keep hypot.
  Offline hybrid reuses existing candidates, only hiding freshly matched wall
  members; retain uncertain identities, stop-only memory and >30m existing leads.
  Production extension d473d0fb4bd7 unchanged; no live hook, commit/push/deploy.
  929 focused tests pass. Full130-segment/152,695-send replay: reviewed29 scenes
  residual6 legacy/83 prior-map ->2; c1/23 RF6/6 preserved. Extra hides vs legacy:
  FF68/LF296/RF7; cb16 RF identity remains unresolved, not all371 proven correct.
  Max179 objects/no capacity evictions; short-gap same-radar-ID map-key changes
  6558->2668 are a continuity proxy, not certified physical identities.
  Paired PC map means .367-.493ms,40.7-45.7% below prior map; hybrid total remains
  1.044-1.333ms including existing selection, so not cheaper than legacy alone.
  Preserve private ccnc-map-near evidence. See docs/ccnc_radar_map_experiment_20261011.md.

- On 2026-10-11, user requested preserving the current CCNC code and creating
  hyundaicanfd_ccnc_radar_map.py for independent map-based detection. Added an
  offline candidate plus replay and 46 tests; original extension d473d0fb4bd7
  stays byte-identical and active. No controller import, switch, deployment,
  commit or push. 916 focused tests pass; fresh139-file inventory/9 duplicates
  yields130 unique segments and152,695 eligible sends, with2,203 excluded before
  input readiness. Candidate worsens residual slot displays in29 reviewed scenes
  from6 to83 (not all new representatives individually certified), and loses
  all6 known c1/23 RF frames at21.74-21.99s despite retained nearby vehicle points.
  Lane classification and permanent false motion evidence remain problems.
  Five-case repeated PC mean .703-.970ms is not vehicle CPU/accuracy validation.
  Keep candidate offline; preserve full/failed evidence and interactive maps
  privately. See docs/ccnc_radar_map_experiment_20261011.md.

- On 2026-10-11, after the authorized snapshot commit b06bef4b, tested a whole-row
  registration alternative b306997c7f7c. Keep it analysis-only: runtime/tests stay
  committed d473d0fb4bd7. Match measured rows to ego-compensated fixed anchors,
  unwrap common motion, require7 samples/300ms, cap6 rows/16 anchors/8 samples;
  immutable seed expiry3s/15m and current side geometry are required. Extra96
  row anchors are additional to the existing96 individual points. Reject the
  unbounded interim trial. Fresh139-file SHA256 inventory covers130 unique
  segments; each existing correction profile replays152,695 eligible sends.
  Correction used:10 LF removals, reviewed29 scenes6->5 residual sends/25->26
  hidden; unused:all output fields equal,6 residual/25 hidden. No FF/RF changes,
  additions, c1/23 right-vehicle loss or db48 regression. Inspect all4 change
  videos without claiming certified point/body identity. c1/21 unused still
  shows LF because row witness age160ms exceeds its150ms hold; used allows300ms.
  New473 display tests and runtime870 focused tests pass. Registration alone
  rejects coherent moving arrays; the original detector still suppresses the
  previous synthetic input, so whole-pipeline error remains. PC paired mean
  increment+.067-.176ms (+9.0-18.2%); maxima before10.154/after16.957ms have
  unresolved causes and are not vehicle timing. Limited gain and added cost
  do not justify applying this candidate. Keep private ccnc-alternative archive
  and uncommitted investigation notes; no further commit/push/deploy authorized.
  See docs/ccnc_front_radar_detection.md.

- On 2026-10-11, user explicitly authorized committing the current tested CCNC
  snapshot before trying another method. Commit runtime d473d0fb4bd7 with its
  existing tests/investigation notes; the failed 3-point/4m candidate stays private
  and inactive. This authorization does not include pushing or deploying, or
  committing subsequent alternative experiments without another request.

- On 2026-10-11, user requested validating map rows at the existing detector's
  3-point/4m minimum. Candidate 9f7ff5daeae2 remains analysis-only; runtime stays
  the uncommitted d473d0fb4bd7, not HEAD 588e087d. Fresh 139-file SHA256 inventory
  covers 130 unique segments; both fixed position-correction settings replay
  152,695 eligible sends each. Reviewed 29 bollard scenes stay 6 residual sends
  and 25 fully hidden scenes per setting. A separate divider scene gains 9 LF
  removals only with position correction; two crosswalk RF removals per setting
  have no certified object identity. No FF changes or LF/RF additions. Candidate
  map-only checks pass 463/fail 1: moving positions with stationary speed reports
  revisit old map cells and falsely establish a row. Combined original detector
  already suppresses that synthetic input on HEAD/before/candidate alike; do not
  claim a newly observed vehicle loss. Reject candidate, preserve runtime and
  its 870 passing focused tests. Jitter confirmation is active, but residual
  nearby candidates lack completed windows or exceed center variation/matching
  bounds. The 0.6s confirmation was chosen as three 200ms windows, not fitted to
  logs; do not portray it or the 15cm/10cm bounds as empirically established.
  Preserve private trial evidence in ccnc-map-three archive; no commit/deploy.
  See docs/ccnc_front_radar_detection.md for comparison scope and limitations.

- On 2026-10-11, the user requested sequential jitter-map implementation and
  suppression/compute validation. Keep the final d473d0fb4bd7 CCNC display trial:
  fix matching-prefix extension/4m float rounding, record measured candidates
  up to2.5m/s longitudinal/.6m/s lateral within1.5m of a qualified boundary,
  but require bounded center/spread and three completed200ms windows for relaxed
  static evidence. Preserve original strict anchors and existing confirmed
  motion/vision/stop/approach vehicle guards; keep96 points,3s/15m and freshness.
  New slow longitudinal motion needs consecutive advancing window centers.
  The first center trial regressed db48 (10ON/4OFF reappearances); reject and
  preserve it privately. Final 870 focused tests pass. Fresh SHA256 inventory
  covers139 rlogs,9 identical duplicates,130 unique segments; ON/OFF each replay
  152,695 eligible sends, excluding2,203 unready. Final adds23ON/24OFF LF
  suppressions and1OFF RF suppression vs restored combined source, with no
  added detections or FF changes; seven segments differ,123 are equal. Reviewed
  images support structures but do not certify per-point identity; c1/23's RF
  stays equal. In29 labelled windows each mode retains6 of196 sends (was10),
  25 episodes clear;358/3,395/6,c1/21 at5.48s still retain LF. Do not claim all
  bollards gone. Five-case three-repeat PC mean paired increment is.036-.167ms
  per display update (5.0-19.2%); not whole CPU or vehicle measurement. Mixed
  five-version timing has an11.13ms final outlier; standalone final max3.84ms,
  baseline max3.14ms, with GC enabled. Outlier cause unresolved, no hard latency
  guarantee. Keep changes local/uncommitted and CCNC-isolated; no deployment.
  See docs/ccnc_front_radar_detection.md and private ccnc-jitter-stages archive.

- On 2026-10-11, after the map-only replacement failed divider regressions,
  the user explicitly requested restoring the earlier combined detector and
  checking previously recorded bollard logs. Production source is restored
  byte-for-byte to the successful54eff4c6631b point-map snapshot: original
  structure detector plus the bounded stationary map, not map-only classification.
  Preserve the failed replacement privately, not as current runtime code.
  Original structure tests and map tests are restored;853 focused tests pass.
  Fresh contextual replay covers41 segments and29 previously labelled structure
  episodes. ON/OFF each hide186 of196 sends in those windows;24 episodes clear
  completely but10 LF sends remain across358/3,395/6,c1/21. Some retained sends
  select a different point; do not claim all bollards
  disappear. db48's original false LF sends are all suppressed again (17ON/11OFF).
  Reviewed outputs match the archived combined full-route replay.
  Follow-up diagnostics correct the stop-memory explanation: none of these20
  residual sends hit LF stop/approach-memory or vision exemptions; saved memory
  in c1/21 belongs to RF. 358/3 ID41 is outside the confirmed row and exceeds
  the map speed gate; ID60 is motion-admitted. 395/6 first row is only50ms old,
  later OFF ID33 is motion-admitted. c1/21 has only two near points at5.48s;
  at11.63s ID51 sits1.95m beyond the row, and relevant boundary gaps0.92-1.07m
  exceed the map's0.75m gate. Matching a three-point prefix stops row extension;
  the alternate three-point span is3.9999995m and fails the strict4m threshold.
  Preserve diagnosis privately; no additional detector change was requested.
  Keep changes uncommitted, branch-isolated and evidence private.

- On 2026-10-11, the user authorized replacing the separate CCNC structure
  detector with map classification; this supersedes preserving that detector
  below. Use one bounded ego-compensated point map and vehicle/structure/unknown
  output. Preserve motion/vision/stop-memory vehicle guards; unknown retains
  existing admission and display. Current mature points seed >=3 anchors over
  >=4m, including denser arrays needing more than three points. A new row from
  never-simultaneous memory requires four anchors over5.5m. Store classification
  on observed anchors so surviving points do not need neighbors to re-form a row.
  Raw vLead can report ~2m/s on divider scenes; repeated fixed-position support
  establishes geometry, with lateral-motion protection and no interpolation to
  unseen points. Boundary candidates retain the old1.5m range; keep96 points,
  3s/15m expiry and existing prediction hold. Early speed-only, tight-boundary,
  three-only-seed and row-rebuild trials exposed regressions; preserve evidence
  privately. Memory-only moving points falsely suppressed RF near a visible
  right vehicle in c1-23; restore <=0.6m/s longitudinal/lateral conditions for
  memory inference and retain observed-vs-remembered classification provenance.
  Never upgrade an unobserved extension to observed evidence. 858 focused tests
  pass, but full130-segment replay is partially FAILED: each mode152,695 eligible
  sends; compared with the combined implementation ON35/OFF23 extra LF removals
  and ON139/OFF116 side redisplays, in13/14 segments. FF equal, no extra RF removal;
  the visible-right-vehicle regression is fixed in both modes. db48's earlier
  ON17/OFF11 successful removals retain only10 in each mode; ON7/OFF1 reappear.
  Do not claim improvement-only or promote this uncommitted replacement candidate
  as deployment-ready. Local measured extra mean time ON0.009..0.066ms,
  OFF-0.044..0.010ms is display-only PC timing, not vehicle CPU. Keep branch
  isolation, failed evidence private and changes local;
  no commit/push/deployment is authorized by this request.

- On 2026-10-11, the user authorized a CCNC display-only stationary point map
  and full local-log validation. Keep individual measured points in a bounded
  ego-relative frame using existing speed integration and model yaw, independent
  of radar IDs. Cap 96 points; expire missing points after 3s/15m and outside
  -3..50m/+-6m. Four repeated static anchors may recognize a dense boundary row
  before simultaneous visibility, with one missing position allowed in extension.
  Preserve motion-confirmed/strong-vision/stop-memory exceptions and existing
  width admission, radar data, FF, control and original structure detector.
  Match classification to the existing 300ms temporal prediction hold; never
  renew measured memory from predictions or extend the display hold. This fixes
  a reviewed final redisplay after a 150ms-only trial. Avoid missing internal
  attribute queries on raw capnp points; profiling exposed costly exceptions.
  853 tests pass; 130 unique local segments, ON/OFF each 152,695 eligible sends:
  only reviewed divider LF changes 17/11 times, other 129 segments and FF/RF equal.
  Recorded-input equality is not object-truth or vehicle/CPU validation. Keep
  branch isolation and evidence private; no commit/push/deployment authorized.
  NAS shared-source bundle/fingerprint checked locally, but current wip-only
  publishing and lead replay do not deploy/recalculate this CCNC display map.
  See docs/ccnc_front_radar_detection.md and private 2026-10-11 point-map archive.
- On 2026-10-15, user requested review/integration of Tesla PR551 c61d998ff8.
  Reproduced inactive phase-lead CAN/feedback and saturation defects; Tesla's
  encoded zero375 also regranted controls through the shared accel helper.
  Port the nine PR files unchanged: measured inactive angle, final feedback,
  cooperative torque blending only above25km/h with normal low-speed steering,
  output-limit-aware saturation and Tesla-specific longitudinal permission.
  Both maintained branches:199 focused tests/128 subtests pass. Three inherited
  generic disabled-angle safety cases still fail identically before/after;
  do not claim full safety suite success. Native C uses a desktop packet/forward
  ABI bridge, not MCU/vehicle validation. Publish single-parent on both branches;
  no vehicle installation. See docs/tesla_pr551_review_20261015.md.

- On 2026-10-15, user requested historical attribution of memory >80%. Read925
  retained local logs (916 own-device with memory) plus on-device qlogs. Earliest
  observed >80% is Oct10/108e, but Oct11/1095 stays73--77% with HUD+signals;
  not one proven regression date. Current DM warmup/plateau and logger sawtooth
  match earlier runs. GC retry fix reduces Python retention, but native HUD ION
  buffers persist after process death (89.6MB), and new PID accumulates again.
  Do not call memory normal/fixed or recommend process cycling as its fix.
  Expected S3 transport log cleanup published both branches; fresh-manager
  reboot verifies concise timeout lines (HUD-only restart kept cached code).
  Reboot clears old native buffers but four retries retain6.6MB again; current
  81%/685MiB available is not leak clearance. Preboot model/DM1200 frames/60s,
  postboot1199 valid consecutive each at19.974Hz, no missing IDs/fallback;
  do not claim a strict >=20Hz postboot pass. S3 still times out. See
  docs/hud_retry_memory_20261015.md; raw evidence private.

- On 2026-10-15, user's C4 Git Pull was blocked by two untracked manually
  installed eGPU helpers identical to incoming blobs; back up only those files,
  preserve unrelated build/model artifacts. Old93160a93 HUD repeatedly lost S3
  before renderer ready. Prior qlogs reach99%; same HUD PID grows168.62->528.55
  MiB RSS, +181.89MiB PSS across segments5->27. Actual failed JPEG pipelines
  remain frozen after close/collect; unfreeze releases them. Pair HUD session
  freeze with finally unfreeze/threshold restore, retaining active-frame policy.
  27 focused tests and native isolated12-cycle reclamation pass. This fixes a
  demonstrated retention path, not every system/native allocation or reboot
  cause. User authorizes parked system stop/update; verify fresh P/zero/inactive.
  Publish both maintained branches, single-parent. See docs/hud_retry_memory_20261015.md.

- On 2026-10-15, user requested public S3 firmware/install guide after confirming
  the panel wakes and turns off. S3 1.1.0 confirms sync/brightness-off, retries
  failed shutdown and reasserts idle-off; C4 offroad dimming includes registered
  S3. N16R8 / powered hub / TURZX 1cbe:0092 only. Browser installer and NAS ZIP
  preserve NVS; never distribute device flash backups or pairing keys. UART
  --auto requires a unique bridge and verifies identity. Native three-phase
  150-frame final test passes with preserved pairing; browser-native flashing
  is distinct from UART validation. Publish both maintained branches, single-parent.
  See docs/s3_release_20261015.md and tools/esp32s3_display/INSTALL_KO.md.

- On 2026-10-15, user requested MD v3 PR39087 on carrot-wip ONLY unless already
  installed. Prior selection was8d736683; new checkpoint1365126a/12864, AMDd4c7a3fc,
  pinned tinygrad02e69816/runtimec7c58622. PR changes only the big PKL; no matching
  ONNX, internal/DM and Jetson remain unchanged. Preserve retry/fallback policies;
  source-verified new runtime retains USB no-op sleep and +/-1 link-error meanings,
  so extend only the exact hash allowlist.411 focused tests/13 Web tests pass,
  8 Windows skips; Linux CI403 plus3 native-runtime tests and HTTPS empty-cache
  install/offline reuse pass on8864280c9c.
  No vehicle install or native20Hz/driving claim. See docs/md_v3_20261015.md.

- On 2026-10-15, user requested explicit tmux error/retry visibility after
  repeated eGPU investigations. Add bounded stderr [eGPU] lifecycle events with
  PID/monotonic time, saved failure cause/classification, attempt/limit/backoff,
  waiting reason, probe-ready/handover, actual first live publication and stop
  reason. Preserve all retry/validation/model policies; never write diagnostic
  text to the worker stdout protocol.179 focused tests pass; no vehicle action.
  See docs/egpu_persistent_rejection_20261015.md.

- On 2026-10-15, identified the persistent Ioniq 5 PE block from actual NAS
  diagnostics: pinned90970e01 reports AMD failed with +1 during input upload.
  Its usb_reap latches LIBUSB_TRANSFER_ERROR=1; old classifier recognized only-1.
  Both branches now classify exactly +/-1 only for that verified runtime, with
  previous-boot/hash/artifact/smoke validation for old rejections.175 tests and
  the exact recorded traceback probe pass. The failing734634 predates Oct15
  timing changes. Device wall clock was stale; do not assert recorded21:05 as
  incident time. Physical USB root cause remains unknown; no vehicle install.
  This supersedes the earlier unknown reporter cause below; numerical migration
  is a separate correction. See docs/egpu_persistent_rejection_20261015.md.

- On 2026-10-15, a Mountain Dew report remained rejected after reboot. Current
  wip lacked nav's previous-boot numerical-rejection migration; four positive
  replay cases fail before the port and pass afterward. Port only that migration,
  preserving branch model/runtime and per-frame policies. Keep identity, fresh
  boot, artifact verification and new smoke checks; unknown/integrity failures
  remain blocked. Both web branches deny ineffective compile-restart on rejection
  and offer a bounded matching historical failure download. Original reporter
  failure/commit is still unknown; no vehicle action or USB root-cause claim.
  See docs/egpu_persistent_rejection_20261015.md.

- On 2026-10-15, Carnival225-28 on0e647cd2 has a visible moving-to-stop corner
  born in-lane, excluded from L1 and outside-history/45m CUT-IN L2. Prior Oct14
  fallback is inactive, not a demonstrated regression. Add measured braking L2
  only with current confident lanes/vision,350ms lane/braking/physical history,
  actual speed decline and integrated range-rate agreement; preserve front/SCC
  primary and ordinary CUT-IN. Broad cruising association was rejected. All
  102 logs/502 labels: prior499 rows identical in modes1/2/3; failures11/10/13,
  predecel0/0/1 and208 unverified remain. Three new cases pass. All122,090 mode1
  frames have unchanged L1; only6 Carnival and7 reviewed Azera L2 samples differ.
  Carnival first66.35m/video16.602s, only6 samples:1.45s before replay front or
  about1.3s before recorded front. Tighter obstacle input is not MPC/closed-loop
  braking validation; the later evidence gap remains.1230 Windows tests pass,
  vault36 pass/4 skips. Publish both maintained branches, single-parent, verify
  automatic NAS image/page/replay; no vehicle action. See
  docs/carnival_in_lane_braking_20261015.md. Recordings remain private.

- On 2026-10-15, the user requested allowing S3 Wi-Fi HUD up to10fps while eGPU
  is active. Its live rate reader inherited the wired USB5fps cap. Give only the
  Wi-Fi reader a10fps cap, preserve lower configured rates and direct USB policy.
  Parked HUD-only restart reports9.8-10.0Hz; model/DM each1200 consecutive valid
  frames in60s, no process restart or fallback.32 focused tests pass on both
  branches. This is host refresh, not optical panel FPS. See
  docs/s3_wifi_egpu_investigation_20261015.md.

- On 2026-10-15, the user required normal model output at20Hz and requested
  upstream NAV n16 afbf2e547e / AMD52205044 (checkpoint96244b1f/12800). It changes
  only the PKL; runtime d3f09c9/90970e01 and old Jetson ONNX are unchanged. Pin
  the separate NAS n16 package on carrot-nav; preserve wip's model selection.
  Shared eGPU fixes yield200us after1ms in this exact runtime's previously no-op
  USB sleep, retaining FIFO54/core7 after lower-priority/core trials failed20Hz.
  Full-response deadline150ms, bounded failed-worker kill/reaping plus ownership
  lock, fresh-frame fallback and startup-only internal inference warmup/reset
  replace the ~3s stale fallback. Stopped-worker probe reaches internal output
  in718ms with106ms capture age; this is not fault-time20Hz or root-cause proof.
  Final native300s has6000/6000 consecutive valid model/DM frames at20.0002Hz;
  no fallback, S3 connected. Later USB link counter remains1->1, not zero.
  Nav298/wip257 focused tests pass. User authorized parked restarts/install. See
  docs/egpu_n16_stability_20261015.md for final validation and remaining limits.

- On 2026-10-15, deeper direct-eGPU analysis confirms all three fault snapshots
  were USB4-1/5Gbps, active with autosuspend off. Outstanding output_read used
  ~95–99% main-thread CPU; worker FIFO54 and DM FIFO5 share CPU7, with DM CPU
  sharply reduced during stalls. Pinned90970e01 has no-op USB sleep and zero-time
  libusb polling; older AMDSignal yield tests do not cover it. Polling/starvation
  is a supported failure-path hypothesis, not the initial GPU/bridge root cause.
  Parent close can add2s; no runtime/model change or eGPU restart performed.
  Parked retry permission pending. S3-only TCP disconnect recovers in9.217s
  without process restart, model/DM601/600 valid updates in30s. Offroad Wi-Fi
  association differs from HUD output gating. See docs/s3_wifi_egpu_investigation_20261015.md.

- On 2026-10-15, S3 home Wi-Fi delay improved after an approved single-AP
  channel1->6/20MHz change: 35% loss/217ms mean to no loss/6.7ms after removing
  the temporary AP lock. C4 had no pairing file; saved the existing key privately.
  Defer S3 TCP open until renderer readiness to avoid its2s idle timeout; notice
  new registration during USB wait.30 focused tests pass; native parked HUD runs
  typically9-13Hz, model/DM600/601 valid updates in30s with original PIDs. Physical
  panel/road validation is separate. Three new Oct14/15 NAV-RL eGPU output-read
  timeouts cause~3s model gaps and~3s-old first fallback; underlying cause remains
  unresolved. No eGPU/model/control change. Publish host fixes to both maintained
  branches, single-parent. See docs/s3_wifi_egpu_investigation_20261015.md.

- On 2026-10-14, the upper/lower signal-group trial fixes supplied qualified
  observations but not actual departure: 43,090 historical planner inputs at
  nominal/+20ms and 20 recent segments show no output changes; Santa Fe encoded
  165-frame replay still has zero releases. Keep UpperSignalGroupTrial OFFLINE;
  production guard/tracker remain unchanged. Fresh neural comparison gives
  68/73 box-eligible images versus rules47, tiles70; tiny green cores10/37 or
  tiles19/37 remain weak. Neural also fails identity/release at recorded timing.
  No retraining or vehicle change. Next target is learned whole-housing/lamp
  recognition with encounter-separated labels, not another live rule relaxation.
  Publish offline tools/tests/findings to both maintained branches, single-parent.
  See docs/signal_upper_group_trial_20261014.md; raw evidence stays private.

- On 2026-10-14, Carnival 222-1 on 0245e76f shows late front arrival and a
  stopping-path tail rejecting visible stationary corner 1193. Add missing-L1
  measured L2 only with confident actual inner lanes, stopping context, stable
  corner history and current visual agreement. Do not relax ordinary path/L1
  gates. Replay selects 11 frames at 74.32m, about0.57s earlier, not closed-loop
  stopping validation. User also requested screen + one short sound for an
  unselected, vision-missing, stable front/corner stationary pair: TTC<=4.8s,
  range<=120m, active longitudinal only, 500ms confirmation and encounter/cooldown
  suppression. Existing100 logs/497 labels are identical in modes1/2/3; existing
  failures11/10/13 and208 unverified remain. All119,689 prior mode1 outputs match;
  no advisory fires, so real positive performance is unproven.1176 wip tests pass;
  two new Carnival cases pass all modes. Publish both maintained branches and
  verify automatic NAS replay deployment; no vehicle action. See
  docs/carnival_stopped_car_20261014.md; recordings stay private.

- On 2026-10-14, Elantra route311 segments3/10 confirms86->88% memory use.
  DPathRadarD's lifetime Cap'n Proto builder leaks replaced list/error storage:
  restore a fresh builder each update, preserving controller history and fields.
  532 tests pass;2,399 recorded-input frames serialize identically.72,000 real-
  payload updates grow old RSS23.02MiB, fixed0.086MiB then flat (accelerated PC,
  not an hour of vehicle operation). Actual radar RSS grows3.31MiB across~8min;
  loggerd grows36.86MiB and remains unexplained. Do not claim the overall88%
  issue solved. Publish both maintained branches, single-parent; no vehicle action.
  See docs/elantra_memory_20261014.md; recordings remain private.

- On 2026-10-14, the user requested recovering uncommitted completed work and
  an explanation before publication. The pending October 3 changes are NNFF
  startup-notice retry, a default-OFF search-only NNFF setting, and a common
  120 ms temporary-steering warning presentation delay. Publish separately to
  carrot-nav and carrot-wip with single-parent commits after current checks.
  The user explicitly requested deleting the failed parked-blinker experiment:
  remove its uncommitted safety hooks, helper and tests; retain only historical
  findings. Do not reinstall or revive that experiment. See
  docs/uncommitted_work_recovery_20261014.md. No vehicle deployment is included.

- On 2026-10-12, 10a5 on nav ac1da01591/mode2 reproduces a real green held by
  a tiny sole red anchor and zero moving preparation after small night lamps
  repeatedly lose IDs. Add a two-housing red-to-green release: each anchor needs
  independent red history and fresh 700ms green, with no forward red conflict.
  Night-only pending-red templates and unique >=0.9 image correspondence retain
  small red identity through bounded motion; do not relax green/daytime rules.
  512 tests pass; prior 43,090-frame guard summaries match at nominal/+20ms;
  3,347 prior image comparisons have unchanged fresh colors/green boxes. New
  replay releases the reviewed green hold; image replay requests preparation
  5.60s earlier for only0.9s, cap about-0.19m/s2. Not sustained braking or a full
  stop; no vehicle install/native timing/closed-loop claim. Publish to BOTH
  carrot-nav and carrot-wip, single-parent only. See docs/signal_a5_followup_20261012.md.

- On 2026-10-12, the user reaffirmed that corrections must be committed and
  pushed to ALL maintained carrot-* branches unless explicitly excepted. This
  supersedes the earlier wip-only signal scope. Current remote targets are
  carrot-wip and carrot-nav; do not recreate retired branches or modify other
  contributors' namespaced branches. Port both signal preparation cadence and
  night red-group corrections to nav, preserving its navigation/model/runtime
  choices. Keep every new commit single-parent, preserve unrelated local work,
  and distinguish branch publication from vehicle installation and road validation.

- On 2026-10-12, the user requested solving the failed night-red preparation.
  A separate night contract now requires 2–4 persistent compact red lamps with
  compatible road-direction support, NOT a bounded distance or stop-line target.
  Same-ID current-red templates bridge a direct core <=500ms without renewing
  it. Night capture freshness is350ms; missing pair evidence is inactive with
  <=350ms confirmation retention, and fresh forward green cancels. Only night
  geometry erodes road edges by0.3+2*std (std<=1m); measured housing policy stays.
  Mode2 alone actuates; keep CPU/DM protection and original ONNX. 492 tests pass.
  Nine-minute replay enters once6.10s before original stopping, for1.1s to target
  -0.27m/s2; +20/+100ms still enter, +250ms rejects. Prior4329-input regression
  adds no entries. Not closed-loop improvement or guaranteed stop: identity loss
  and applicability remain limitations. No vehicle deployment; read-only vehicle
  remains carrot-nav135020d1. Work remains carrot-wip, single-parent only. See
  docs/signal_night_group_20261012.md and private2026-10-12-night-geometry archive.

- On 2026-10-12, the latest 10a4 evening drive has 1,063 fresh-red observer
  outputs but zero moving preparation: every confirmed red box is a night core
  or template, not a measured housing. Preserve confirmation across repeated
  accepted samples ageing beyond 200ms, for at most 450ms from capture; output
  remains inactive, and fresh new evidence/context/gap/session checks still win.
  448 focused tests pass. Recorded-input replay of 10,276 planner samples and
  +20ms delay still gives zero entries, as does an offline night lamp-ray/height
  experiment: road confidence and broad/truncated range remain blocking. Do not
  connect the failed night candidate, pretend inferred width is metric housing
  size, or claim better stopping. No vehicle deployment/model changes. Work only
  on carrot-wip with single-parent commits; preserve the separate nav checkout.
  See docs/signal_night_prepare_20261012.md and private evidence archives.

- On 2026-10-12, SignalAssistMode adds the requested live0/1/2 selection on
  carrot-wip only:0 auxiliary OFF,1 display/alerts and stopped-red departure veto,
  2 also conditional red preparation toward-0.5m/s2. Default0; old production
  enable files cannot bypass it. Mode2 uses CPU housing evidence alongside either
  driving model; preserve DM protection. Mode changes reject pre-activation
  observations;0 clears the auxiliary veto without issuing go. Native rebuild
  after pull is required for the new Params key. The user requested all signal
  trial content on wip and SINGLE-PARENT commits going forward, superseding the
  trial-only correction. Leave old topology/ref alone; no parallel trial changes.
  Range audit finds6 observations of a real red outside3.5m, all with blinker on;
  close signals mostly fail boundary confidence, not proven lateral exclusion.
  See docs/signal_color_experiment_20261012.md for validation and limits.

- On 2026-10-12, after correcting branch mistakes, the user superseded the
  earlier trial-only instruction and explicitly requested ALL carrot-signal-shadow
  contents integrated into carrot-wip. Preserve existing wip radar/font fixes.
  Continue ONLY on carrot-wip with linear, single-parent commits; no merge
  commits or parallel signal-branch development. Leave historical topology alone.
  This includes the retrained Mountain Dew eGPU selection8d736683 and its pinned
  runtime, while keeping internal/DM models and Jetson selection unchanged.
  Add SignalAssistMode:0 fully OFF,1 display/alerts and red departure veto,
  2 additionally gentle red stop preparation. This supersedes old per-file enables.

- On 2026-10-12, the user clarified that traffic-light geometry must allow
  space on both sides of the ego lane. The opt-in stop preparation now expands
  each inner lane boundary by3.5m, clipped by reliable road edges with0.3m inward
  clearance. Check original source agreement before expansion. This provisional
  search region is not signal-to-lane assignment. The3348-observation screen
  restores lateral/height support for3 observations of one real red signal;
  only1 passes the unchanged full25-140m distance-interval requirement, and none
  passes all color/context/temporal gates. Wider margins also widen distance
  uncertainty; do not claim measured range or earlier braking. 193 focused tests
  pass. Default OFF, no vehicle deployment. See docs/signal_color_experiment_20261012.md.

- On 2026-10-12, the user requested edge/lane XYZ geometry for red preparation.
  Remove position input/range constraints; use confident inner lanes or road
  edges, intersect when both reliable, and never extrapolate. Correct the new
  runtime pose reader to actual XYZMeasurement fields (previous adapter failed
  closed). 187 tests pass, including a real Cap'n Proto pose. Full-rate-context
  screening of3,348 retained observations gives9 new usable lane corridors but
  zero qualified candidates/entries. Exact frames confirm real red lamps among
  lateral rejects: do not call these false-positive removals or claim improved
  braking. Signal mounting need not align with the ego lane. Keep OFF; no vehicle
  deployment. See docs/signal_color_experiment_20261012.md and private road-geometry
  archive for assumptions and negative results.

- On 2026-10-12, the user requested red pre-deceleration before model stopping.
  Add default-OFF per-device stop_prepare_enabled and state6 e2eStopPrepare:
  confirmed measured red housing, calibrated pose/road-edge/path geometry,
  no lead or driver override; final acceleration cap ramps0.25m/s3 to-0.5m/s2.
  Release on missing/changed evidence, preserve model-stop handover and stronger
  braking, and reuse trafficSignChanged audio once with cooldown. Remove only
  visible signal track numbers. Geometry ranges are assumed, not metric truth.
  184 desktop tests and1,200 default-OFF target comparisons pass. Screening3,348
  retained night/morning observations yields zero qualified entries: do not claim
  better stopping or activate from this result. No vehicle deployment. See
  docs/signal_color_experiment_20261012.md and private stop-prepare archive.

- On 2026-10-12, during the color experiment, the user prioritized avoiding
  false green over missed detections, accepted rejecting turn-time readings,
  and suggested pose. Offline rotation variants now reject green before tracker
  confirmation, immediately remove old confirmed green and preserve red history.
  Existing red start latch holds through unknown; old green never releases it.
  Steering+speed avoids treating a stationary -39.5-degree wheel as rotation.
  Preferred pose experiment uses calibrated livePose yaw, required validity,
  uncertainty and separate arrival/sample freshness; no permissive fallback.
  Four-way3348-frame replay: pose gates898 observations with zero fresh/confirmed
  green outputs, retains the stationary real green;108 tests pass. Counts are
  not accuracy and LED/reflection false greens remain. Both variants stay
  offline, no vehicle deployment. See docs/signal_color_experiment_20261012.md.

- On 2026-10-12, the user requested a signal correction experiment. Add
  default-OFF offline conservative_green HSV38-90 consistently across housing,
  day/night seeds and templates. Exact-frame paired replay of13 full videos and
  3348 observations reduces morning green candidates5608->581, not an accuracy
  metric. Selected development boxes remove61 replayed background greens;
  21 newly reviewed frames keep37 visible green lamps and reject29/36 backgrounds.
  LED text/reflections remain; one motion-blurred real night green is lost.
  Only one of two live red miscrops reproduces green in encoded pixels.
  Guard color-confirmation probe differs from live release and is not control
  validation. 84 tests pass;21 default-OFF image results match prior code.
  No deployment, model training or live enable; keep DM GPU/start-guard policy.
  See docs/signal_color_experiment_20261012.md and its private evidence archive.

- On 2026-10-11, user requested the latest signal camera observer with both
  internal and retrained eGPU models, parked switching and performance checks.
  Integrate the already deployed 42760 signal/display revisions into this trial;
  retain its newer comparison/assist-OFF policy and preserve 7f3bfcd7 model/runtime
  selection. The old October 10 assist-ON snapshot is historical, not the current
  requested configuration. New camera neural output remains display/log only.

- On 2026-10-11, the user selected upstream 6f4c0faf eGPU on this
  carrot-signal-shadow branch only, for testing before a combined promotion.
  Wip publication 03142df861 was reverted by b09ae3ddad; wip keeps Cinque v3.
  Use retrained Mountain Dew 8d736683/12864, AMD 256f2178, isolated tinygrad
  d3f09c9b and verified NAS package comma4-big-mdm-8d736683. Adapt byte-addressed
  buffers while retaining prior dtype-bearing buffers. Empty-cache HTTPS
  install and offline reuse pass; preserve startup delivery before build/manager.
  Public ONNX catalog now has old 870a4823 but no matching 8d736683. Jetson
  stays Cinque v2; internal/DM models and control/validity/warp policies stay
  unchanged. No physical GPU, first-boot or driving validation is implied.
  See docs/mdm_retrained_20261011.md. Keep the existing signal experiment scope.

- On 2026-10-12, the user explicitly corrected the branch scope: current
  signal experiments belong ONLY on carrot-signal-shadow, not carrot-wip.
  Remove the five experimental commits5a80605d61,3f537bc13f,6df4f6296f,
  df51eed7fb,4bc5307d0b from wip by a normal corrective commit; preserve the
  trial branch, prior deployed helper/DM fixes and unrelated work. Future
  signal experiments/settings/validation are trial-only until explicitly promoted.
  Wip runtime/tools are restored byte-for-byte to85d97b3160. No vehicle action.

- On 2026-10-12, the user requested collecting unseen night/morning logs and
  evaluating signal size/height/road-edge filtering, analysis first. All52
  completed segments/169 source files are locally SHA256-verified. Morning CPU
  observer emits6792 fresh green candidate samples,6011 from day_color_core;
  reviewed branches/sky pass hue38-105. Two still-red signals receive green
  through tiny/cropped housings and adjacent cyan sky. This is not just UI hold.
  In73 selected labels, conservative geometry removes8/62 background candidates
  and neither red miscrop; an exploratory narrower-hue pixel test removes61/62
  and both miscrops but is not a validated detector or night regression test.
  No runtime/settings/model change or deployment. Prioritize sky rejection and
  complete housing/lamp alignment; road edges are supporting evidence, not lane
  assignment. Keep DM GPU protection. See docs/signal_morning_false_green_20261012.md
  and private .analysis/archive/2026-10-12-day-signal/ for evidence and limits.

- On 2026-10-11, the next drive exposed signal QCOM GPU contention with DM:
  parked A/B/A gives DM 10.6-10.8 Hz with neural GPU versus 20 Hz with CPU;
  priority15 also fails. This supersedes eGPU-implies-QCOM eligibility below.
  Auto AND explicit QCOM now use CPU comparison whenever DM or Driver View
  needs the internal GPU; never disable DM or use camera failure as permission.
  Preserve DM policies and external/internal driving-model selections. Latest
  drive neural observations at 2.33-2.67 Hz reset identity on every >250 ms gap.
  Display revision4 extends UI-only memory to two seconds from capture, unique
  same-housing unknown-ID carry without renewing old color, dim/~ previous and
  immediate new color replacement. Ten-phase replay reduces visibility dropouts
  146-162 to14, not recognition accuracy. Stopped red guard stays650 ms, warning
  ON and moving assist OFF. 163 desktop tests pass, one Linux-only skip.
  See docs/signal_dm_flicker_20261011.md for native verification and limitations.

- On 2026-10-11, the user explicitly requested red automatic-departure blocking
  AND one warning sound. Add separate default-OFF start_guard_enabled, fresh
  observation transport and stopped-only veto for both observer backends. Keep
  old moving assist OFF. Red needs two samples/150 ms; green same red-anchored
  housing needs three/700 ms with no competing forward red. Unknown/stale and
  observer restart never grant green release. Stop-only freshness is 650 ms
  for measured GPU latency; old moving-assist 200 ms stays unchanged. Gas
  overrides with 10 s cooldown. Final shouldStop/vTarget/aTarget and ACC mode
  enforce the veto; audioRefuse is once per blocked stop's departure request.
  264 checks pass, one skip. 43,090-frame nominal/+20 ms replays block both
  1095 red departures; reviewed-red green releases are zero, but green holds
  remain. Forward ROI is not lane association and detection can still fail.
  New MDM stays trial-only; auto chooses QCOM neural with eGPU, lightweight
  CPU tracker with internal model. Both sustain ~20 Hz parked; internal+QCOM
  co-load falls to ~8 Hz and is rejected. See docs/signal_start_guard_20261011.md
  and docs/signal_gpu_dual_model_20261011.md for evidence and device status.

- On 2026-10-11, the user redirected the new upstream 6f4c0faf eGPU model
  to carrot-signal-shadow for testing before a combined promotion. Revert the
  just-pushed 03142df861 model/runtime change from carrot-wip; wip keeps Cinque
  v3. Retrained Mountain Dew 8d736683 and its d3f09c9 runtime belong only on
  carrot-signal-shadow until the user approves promotion. Its verified NAS
  package is comma4-big-mdm-8d736683. The public ONNX catalog now includes old
  870a4823, but not 8d736683; keep Jetson Cinque v2 and internal/DM models.

- On 2026-10-11, the user explicitly requested connecting the new neural
  signal detector to the vehicle, engine-off camera testing and CPU/QCOM GPU
  timing comparison. Add opt-in neural_enabled dispatch in signalcolord, a
  display/log-only VisionIPC worker, hash-pinned artifacts and explicit CPU/QCOM
  backends. Preserve original ONNX/x/v and assist OFF; no control transport.
  Default remains existing observer unless explicitly selected. QCOM uses the
  existing tinygrad runtime, fixed-shape JIT and synchronous readback; no hidden
  CPU fallback. 130 desktop tests pass, one skip. Vehicle offroad/noOutput and
  runtime inventory were read, but intermittent SSH timeouts interrupted model
  upload. GPU speed/parity, new live inference and forced camera/model co-load
  remain unvalidated and must not be called installed or activated. Last checked
  vehicle HEAD fc7588d67a contains only the earlier display-continuity deployment.
  Resume private scripts/data in .analysis/scratch/2026-10-11-signal-live/ after
  connection recovery. See docs/signal_neural_live_20261011.md.

- On 2026-10-11, the user requested holding prior signal display results and
  combining upper ROI, neural detection, fast tracking and high-resolution crop.
  New 26-segment data has 5,714 observations at2.71--4.99 Hz, median gap250 ms
  and result age183 ms. Display revision3 remembers candidates up to800 ms from
  capture, marks them dim/~ previous, immediately replaces new red/green and
  never renews old color from unknown/repeated data. Camera/session failures reset
  it. This supersedes display-only no-hold policy; control freshness stays200 ms.
  Offline pinned neural+tracking on381 new frames reduces PC mean work100 to33 ms;
  faster search after loss removes a150 ms green reacquisition delay in that case.
  Dark-light loss/arrows remain unresolved. ROI NV12 copying is implemented, but
  native full-resolution camerad access, dynamic path ROI and GPU inference are
  not connected. New neural reference remains offline, original ONNX/assist OFF
  preserved. 123 related tests pass, one Linux-only skip. Vehicle install guards
  offroad/inactive Park; see private deployment.json for live activation status.
  See docs/signal_continuity_20261011.md. Keep recordings private.

- On 2026-10-11, the user requested the best distant/fast/accurate signal
  recognition approach, including upper-path ROI and overlapping frames. Offline
  pretrained Autoware YOLOX-s on 73 selected boxes improves IoU>=0.5 coverage
  from rules 49 to whole-frame 63, upper ROI 68 and two tiles 70; this is not
  exhaustive precision or vehicle timing. A separate MobileNet classifier still
  misses arrow combinations and returns unknown on dark real-green frames.
  Five-frame averaging retains old red across a green transition even after
  alignment. Prefer native-pixel ROI, neural detection, tracking and fresh lamp
  evidence; dynamic ROI/full-resolution/vehicle integration are not implemented.
  Logs confirm OS04C10; code downsamples 2688x1520 to1344x760 and has a full-res
  placeholder, whose safe per-frame access remains unverified. Preserve original
  driving ONNX and assist OFF; no vehicle changes in this investigation.
  See docs/signal_detection_strategy_20261011.md; keep all raw evidence private.

- On 2026-10-11, the post-overlay 1096/1097 drive contains 22 collected segments.
  Helper observations run 3.23--4.93 Hz; median result age is 178 ms and
  2,345/5,345 gaps exceed the 250 ms color-confirmation limit. Track 1267
  keeps identity but never confirms 13 raw green observations after red.
  Display expiry and red-history qualification also cause hiding/unknown labels.
  User requested installation before departure: display-only revision 2 uses
  current color R?/G? for unconfirmed candidates, R/G for matching confirmation.
  Preserve freshness/lost-track gates, original ONNX and assist OFF; no memory
  hold or control timing change. 68 focused tests pass, one Linux-only skip.
  See docs/signal_display_drive_20261011.md; keep all recordings private.

- On 2026-10-11, the user requested signal candidate boxes on the USB
  camera image and vehicle installation. display_enabled opts into a separate
  bounded tmpfs display endpoint; comparison's experimental observer supplies
  R/G/? boxes and track IDs. Use the road camera's actual image rectangle,
  reject wide/replay/3D overlays, expired/lost tracks and reused camera buffers.
  Display age is <=350 ms with <=250 ms/five-frame camera skew; control freshness
  stays 200 ms. Installed 64151b60ff with revalidated/comparison/display ON and
  assist OFF, original ONNX unchanged, existing camera mode 2. This supersedes
  the prior vehicle-code-unchanged status only for observation/visualization;
  the candidate remains unapproved for control. Parked verification saw 501
  valid consecutive model frames at 20.01 Hz, 501 plans, 25 renderer updates,
  an open USB device and unchanged process PIDs/control transport. 425 desktop
  tests pass, one Linux-only skip. Physical panel/road accuracy remain distinct.
  See docs/signal_usb_display_20261011.md; keep captures and telemetry private.

- On 2026-10-11, accumulated signal replay tested 37 segments / 15,422
  images and 43,090 planner updates. The opt-in candidate combines housing/core
  proposals, image-supported 500 ms identity retention with unchanged 250 ms
  color continuity and 200 ms freshness, and explicit producer-red-history reuse.
  Five nominal false-start windows are blocked, but added-latency trials still
  fail acquisition; 1094-10's new green release lacks visually verified green.
  Eight isolated parked camera-timed sequences improve nominal holds/releases
  but 1093-6 loses acquisition with +20 ms. Earlier dusk green coverage regresses.
  Do not enable this candidate for control. revalidated_enabled is only an
  experimental opt-in name, default OFF, with separate assist_enabled required.
  Vehicle remains c115ce4, original ONNX, comparison ON and assist OFF; no vehicle
  code/settings changes. 360 tests and production/prototype parity pass, not
  perception or closed-loop validation. This is hand-written OpenCV, not neural
  retraining. See docs/signal_revalidation_20261011.md. Keep raw evidence private.

- On 2026-10-11, Ioniq 5 PE 1095 segments 65/66 show two red-light
  departures without gas while original x/v changes produce trafficState green.
  Both legacy and daytime shadow detectors recorded red; current c115ce4 vehicle
  uses original internal ONNX with assist OFF and comparison-only publication.
  Same-recorded-input planner replay blocks both departures with either observer,
  but delaying daytime arrivals by 20 ms loses both holds: fresh observations
  miss stop-time acquisition and consumer confirmation resets across gaps.
  Sequential dual-engine timing penalizes the second result. No confirmed green
  tracks in this drive; manual overrides prevent continuous green-release proof.
  Do not treat replay as vehicle-response validation or enable failed candidate.
  No vehicle/model/control changes in this investigation; CPL is not supported
  as the remedy by these images. See docs/signal_1095_red_departure_20261011.md.
  Keep raw logs and captures private.

- On 2026-10-11, the user authorized installing the new signal detector for
  live comparison recording, with the same pushed code on the vehicle. Add
  explicit per-device daytime_comparison_enabled: one camera copy, independent
  legacy/daytime trackers and separate signalTrackingShadow/DaytimeShadow logs.
  In comparison mode never publish either engine to the control transport;
  install with assist_enabled=0. Missing opt-in preserves legacy operation.
  Keep original driving ONNX/x/v, CPU duty/affinity and freshness limits. The
  failed daytime candidate is authorized for observation only, not braking/go.
  329 related desktop tests pass, including actual worker logging/no-publish.
  Use guarded Park/inactive installation and verify commit/model/log identity.
  See docs/signal_live_comparison_20261011.md; keep vehicle evidence private.

- On 2026-10-11, the user demanded actual corrections and validation of the
  remaining daytime signal miss and latency failures. A compact colored-core
  plus dark-bar trial fixes the reviewed morning underpass red-to-green case,
  but full 23-segment replay exposes a new retained stop request at green in
  1094-8 after selected-track loss. Do NOT promote this daytime detector into
  the live worker: SignalTracker(daytime_cores=True) is analysis-only; default
  construction preserves legacy proposals with equivalent compute optimizations.
  10,631 trial observations match the clean port; 459 final default/trial image
  comparisons and 324 related tests pass. Park-only camera-timed stored-image
  workload reduces >200ms results from12/150 to0/107, but this is the rejected
  daytime candidate's timing, not a guarantee for the default/live pipeline.
  Keep 200ms freshness, same-track release, original ONNX/x/v and opt-in policy.
  Vehicle production remains b80db2c97d; only an isolated test folder was used.
  Validation was performed and partially failed, not left unperformed. Preserve
  failed evidence privately; see docs/signal_day_latency_fix_20261011.md.

- On 2026-10-11, the user requested latest-drive signal transition training and
  validation using lamp brightness/color changes plus original ONNX x/v. Three
  offline logistic probes were actually fitted; x/v reduced reviewed red abstention
  but did not improve held-out-encounter green detection. Original driving ONNX
  and vehicle files/settings remain unchanged; learned weights are analysis-only.
  Fix the helper's 125 ms confirmation/cadence mismatch, count consecutive color
  samples, retain bounded identity after a late result, and transport same-ID
  producer green history (>=400 ms, >=3 samples) to avoid a second missed streak.
  Keep 200 ms final freshness, 250 ms camera-gap expiry, default OFF, original
  x/v and normal departure gates. Recorded-input replay releases both known
  night holds and reaches normal departure states; 318 tests pass. Extra delay
  still loses red acquisition, and daytime housing proposals merge into dark
  background. No device deployment or vehicle-response validation is implied.
  Keep footage, labels and weights private. See docs/signal_transition_training_20261011.md.

- On 2026-10-11, first post-assist night/morning drives confirm a helper
  regression: two visible green transitions remain hold_through_unknown until
  gas/inactive override. Native stopping-function reconstruction is green on
  69/13 overlapping held frames; not a full vehicle counterfactual. Tracker
  confirmation restarts above125ms while44.1%/51.2% of real gaps exceed it.
  Same103 observations yield0 green at recorded times vs24 at artificial100ms;
  this diagnoses timing, not a deployable fix. A208ms stale result also wipes
  red identity before green; morning visible-red housing proposals are missed.
  23 driving segments,161 source files/2.21GB SHA256 verified;27,380 valid model
  results. Existing internal ONNX hash unchanged, eGPU and Jetlink inactive.
  Vehicle stays b80db2c97d with assist ON; analysis changed no vehicle settings.
  Recommend disabling assist pending timing/identity/day-detector corrections
  and both green-release/red-false-start regression validation. Parked readiness
  and old replay were insufficient; do not claim the helper is road-validated.
  Keep raw evidence private. See docs/signal_assist_followup_20261011.md.

- On 2026-10-10, after successful parked installation, user requested bringing
  the signal experiment into carrot-wip. Imported signal camera observation,
  causal tracking, optional stop assistance, offline tools and investigation
  documents from e3dc031730. Keep per-device opt-in default OFF; the user's
  installed b80db2c97d vehicle has it ON. CANFD zero-aReq/no-retry is already
  present here as2608792bdd and must not be applied twice. Original driving
  ONNX/x/v unchanged; no failed trained model is promoted. Red hold survives
  unknown, same-track green only removes the constraint, gas overrides with
  10s cooldown. Moving red uses plausible original model distance and existing
  lead/turn/speed gates. This is not lane/arrow association or guaranteed stop.
  All300 focused observer/tracker/assist/CANFD tests pass (Windows Params stub,
  OpenCV4.13). Prior parked checks belong to the signal branch; no carrot-wip
  vehicle installation or closed-loop validation is implied by this promotion.
  Preserve unrelated local edits. See docs/signal_assist_trial_20261010.md.

- On 2026-10-10, RAV4 route 0d segments 5-9 on e7987ed confirmed an
  onroadEvents cadence regression: healthy event-change bursts exceed the 1 Hz
  tracker's upper bound and the continuous readiness gate interrupts steering.
  Check event receipt/validity/liveness without fixed-rate rejection; retain
  periodic-input checks and the separately requested startup-only latch in
  7ceee11642, which includes this correction. Lateral selection alone still
  needs active cruise or AlwaysLateral. All five minutes have cruise inactive;
  no Panda TX rejection or new CAN errors. The long segment 6-7 inactivity is
  not fully explained by event cadence; live setting changes are not captured.
  166 focused tests pass; recorded-input readiness matches all 11,201 examined
  decisions in segments 5/8 after warmup. No physical steering validation.
  Keep incident data local. See docs/rav4_steering_20261010.md.

- On 2026-10-10, the user clarified that the Sonata startup fix must apply
  only until FIRST readiness, never reblock during operation. This supersedes
  e7987ed0fd's continuous readiness checks and repeated PID/curvature resets.
  controlsd/card now latch readiness once per process lifetime; disengagement,
  reengagement and later bad inputs do not rearm it. Nominal geometry is startup-
  only; measured curvature seeds the first lateral activation only. Restore
  existing torque reset behavior. Hyundai camera-SCC waits only for its first
  steering TX template, preserving the bounded first CAN command. Keep existing
  runtime safety, limits, longitudinal policy and firmware unchanged. 241 focused
  tests, 5,969 recorded-input frames and 6,000 post-startup control frames pass;
  the latter exactly match pre-fix 5479d1279a outputs/PID/ratio with input health
  changes and reengagement. Replay is not vehicle-response validation. Keep raw
  incident data local. See docs/sonata_92_startup_steering_20261010.md.

- On 2026-10-09, after trying Mountain Dew 870a4823, the user requested
  returning carrot-wip's eGPU model to Cinque v3 (892fc3a1, AMD e758b96d,
  isolated tinygrad d5e17c93). This supersedes the MDM selection below, not
  the completed branch consolidation. Retain shared HUD/camera/startup/UI
  fixes, MDM format compatibility, internal/DM models and Jetson Cinque v2.
  Do not recreate retired branches. A user-facing model selector was discussed
  as a possible follow-up, not implemented. See docs/mdm2_20261008.md.

- On 2026-10-09, the user requested full integration of `carrot-mdm2` into
  `carrot-wip` and deletion of the local and remote `carrot-mdm2`/`carrot-mdm`
  branches. `carrot-wip` now selects Mountain Dew v1 checkpoint 870a4823,
  AMD model e20cde17, with the existing pinned runtime and NAS mdm2 package URL.
  This supersedes the branch-only restriction below; do not recreate the retired
  branches. Keep the newer wip HUD/camera recovery, DM notices and GV70 fixes.
  Internal/DM models, Jetson Cinque v2 and control/validity policies remain
  unchanged. Model compatibility CI follows carrot-wip. See docs/mdm2_20261008.md.

- On 2026-10-08, the user confirmed PR #39047 commit 4bfb534063 for `carrot-mdm2`,
  starting from carrot-wip ca8f553d1e. The previous bea3fd4 remains carrot-mdm.
  Pin the new e20cde17 AMD model and 870a4823/12864 metadata checkpoint on this
  branch only; upstream's commit subject says 1284 but the file says 12864.
  Keep the 9d0446a4 tinygrad runtime and MDM compatibility adapter. The matching
  870a4823 ONNX is absent from the public export catalog; Jetson stays Cinque v2.
  Do not label this a new official MDM v2 or claim Jetson MDM support. Preserve
  internal/DM models, control, validity and C3 warp policy. Include the inherited
  boot selected-model delivery gate and core4 USB cluster placement.
  See docs/mdm2_20261008.md for artifact identity and validation limits.

- On 2026-10-07, the user instructed not to commit until explicitly told to do
  so. This supersedes the 2026-10-05 interpretation that "작업해" or "진행해"
  automatically includes committing and pushing. Keep completed changes local
  and uncommitted unless the user explicitly authorizes a commit.

- On 2026-10-07, the user requested improving CCNC boundary-point selection.
  Keep motion-confirmed vehicles and existing admission; never blanket-reject
  an allowed vehicle for later boundary proximity. For stationary admissions,
  0.5s/7 fresh near-boundary observations may prefer a nearby eligible interior
  stationary side point, preserving the original candidate if none exists.
  This is display-only representative selection, not proven same-body grouping.
  369 display tests and 79-segment/91,510-sample replay pass; only one segment's
  RF representative changes, with all detection presence preserved. The separate
  intersection RF flashes remain unresolved. Keep evidence local and preserve
  the structure rollback, 15m memory limit and branch isolation. See
  docs/ccnc_front_radar_detection.md.
  The same-day full-local-log follow-up found 35 omitted segments: 123 files,
  9 identical duplicates, 114 unique segments. ON/OFF each replay 133,504
  eligible sends; 2,195 pre-input sends per mode are excluded. Only that same
  RF representative and its stop memory change; other 113 segments are equal.
  The early brief ID reversal predates this edit. No commit/push was made.

- On 2026-10-05, the user clarified that instructions such as "작업해" or
  "진행해" include committing and pushing the completed task unless explicitly
  instructed otherwise. Commit only the task's changes; preserve unrelated work.

- On 2026-10-05, the user requested an offline clock floor after pull/reboot,
  before building. AGNOS startup reads local HEAD's committer timestamp before
  dependency/Params/main builds; only an earlier clock advances to commit+1s.
  Log and verify correction; failures enter existing startup recovery. Preserve
  later clocks, NTP/GPS, file mtimes, caches and Cython/SCons behavior. 26 focused
  desktop tests pass; device clock setting and native builds remain unvalidated.
  This does not establish the cause or cure of intermittent native build errors.
  See docs/build_time_floor_20261005.md.

- On 2026-10-05, the user approved a distance range for holding extra following
  headroom. Define D as ego speed times base TF plus configured stop distance,
  excluding extra TF. Preserve existing hold at <=1.2D; smoothly reduce its
  envelope/rise rate and restore recovery strength through 1.2-1.5D; at >=1.5D
  recover at the existing level rate even for a stopped/opening lead. Use the
  same rule in live state and MPC prediction. Preserve new-lead entry, level-5
  bypass, physical obstacles, base TF, stopping logic/limits and the J20 trial.
  299 focused tests pass; recorded-input reference replay reduces one approach's
  extra margin from 1.916 to 0.142 m, not a vehicle stopping-time/clearance result.
  Closed-loop comfort and clearance remain unvalidated. Keep incident data local.
  See docs/longitudinal_gap_hold_band_20261005.md.

- On 2026-10-04, the user established the CCNC branch merge direction:
  always merge the latest `carrot-wip` changes into `ccnc-hda1`, never the
  reverse, when requesting this integration. Preserve the current CCNC work.
- On 2026-10-03, the user explicitly approved adding core7 to C3/C3X main UI
  onroad affinity: cores0,1,2,3,6,7 with SCHED_OTHER/nice19 for all UI threads.
  This supersedes the earlier core7 exclusion for this UI. C4 stays core6;
  offroad returns to cores0..3. Check each big core independently; preserve
  model/DM, control/camera, IRQ and USB cluster policies. An allowed mask is
  not a CPU quota or parallel rendering and may concentrate UI work on core7.
  Device affinity/FPS and model/DM impact remain unvalidated. See
  docs/camera_core5_trial.md.

- On 2026-10-03, the user requested a manual compatibility option for intermittent
  cluster warnings: HyundaiCanfdClusterDirectTx defaults OFF on every vehicle,
  including EV6. In CAN-FD CAMERA_SCC only, enabling it at startup selects the
  legacy direct host-TX path for 0x161/162/1e0/1ea/200 via Hyundai flag bit 27
  and Panda safetyParam 2048. Preserve other control FIFOs/reuse, allowlists,
  relay protection and the default RX-paced path. This explicitly permits
  independent cluster TX only when selected; no automatic RX timeout fallback,
  vehicle-specific default or live switching. Reboot and updated Panda firmware
  are required. 543 focused/settings/Wiki/firmware-identity tests and F4/H7
  builds pass; actual warning resolution and physical timing remain unvalidated.
  See docs/canfd_cluster_rx_forwarding.md. Keep incident data local.

- On 2026-10-03, Tucson `0000030c--adf522a321--4` showed unnecessary left
  steering while passing a transporter. Actual speed stayed near 104 km/h
  while the model velocity trajectory fell to about 36 km/h; lane MPC remained
  active because the old end/start 70% test missed whole-trajectory collapse.
  The user requested diagnosis through correction. LaneModelSpeedGuard now
  also requires model starting speed >=70% of measured speed, retaining the
  original future-deceleration gate and continuous one-second reacquisition.
  Preserve model paths/speeds, MPC tuning, actuator limits and angle handover.
  26 focused tests pass; same-input target replay reduces initial left peak
  about 79% with fallback around 6.33 s. Small-angle MPC reconstruction matches
  logged mode decisions and incident curvature closely; this is not native
  acados or vehicle-response validation. Shadow versus transporter influence
  on the original model output remains unresolved. See
  docs/tucson_30c_left_steering_20261003.md.

- On 2026-10-03, the user selected existing combined handover mode 3 as
  standard for Hyundai/Kia/Genesis angle control and removed the selector.
  CarController always uses mode 3 inside ANGLE_CONTROL; SteerHandoverMode
  registration, catalog/menu and runtime reads are removed. Saved values no
  longer affect behavior. Preserve the combined algorithm, thresholds, targets,
  angle/CAN limits, torque-control paths and touch/DM. Internal helper variants
  remain for comparisons; diagnostics still identify mode 3. 83 steering tests,
  45 settings tests, 25 Wiki tests and 6,000-frame old-mode-3/new CAN equality
  pass on desktop with Windows Params storage substituted. This promotion is
  not a new retry fix or vehicle-response validation. See
  docs/steering_handover_20260930.md.

- On 2026-10-01, Ioniq 5 PE `000005d3--54fb087f33--21` on eb58e6d7
  confirmed live handover mode 3. The reporter clarified first-curve inside
  positioning and insufficient turning at the street end after 35 s; this
  supersedes interpreting the complaint as vibration/catching. First-curve
  authority is full and angle tracking close; path/target and unwind timing
  need investigation, without declaring the driving model at fault. First
  0–10 s also has roughly 10 Hz original-MDPS
  column-torque oscillation with ceiling 250 and no pressed detection. In the
  tight left turn after 35 s, command reaches the existing 175-degree limit,
  then repeated torque-based override drops authority to 25; at 39.125 s actual
  wheel 51.7 versus target/command 147.1 gives 95.4 degrees error. Mode 3 stays
  waiting during those drops. Exit capture/recover then hits the raw-force veto;
  Panda TX echoes confirm 44-to-25 in 7.1 ms, followed by bounded return until
  42.79 s. No CAN/steering/model fault or decoded warning-vibration request.
  Torque/contact signals do not prove driver intent; the source of full-authority
  oscillation and physical remedy remain unresolved. Analysis only; do not infer
  that faster handover alone fixes it. See docs/ioniq5_5d3_steering_20261001.md.

- On 2026-10-02, after the native CPU experiment and Ioniq 5 PE before/after
  logs, the user explicitly approved promotion to `carrot-wip` and deletion of
  the remote `carrot-native-cpu` branch. Keep the tested Cython radar statistics/
  path projection and CAN extraction/packing kernels, Python comparison/fallback,
  strict floating-point build flags and backend timing diagnostics. Preserve
  radar algorithms, history, thresholds, validity, counters and CPU placement.
  The discussed trajectory prefilter is deferred and must not be included.
  Ioniq 5 logs confirm native activation and core5 mean 77.7 -> 70.9%, but input
  workload differs; same-input replay matches all 2,400 radar frames with 30-31%
  lower PC compute time. EV9 overloaded mode-3 native vehicle behavior is still
  unvalidated. Maintain native x86/ARM CI and the shared NAS replay build.
  See docs/native_cpu_experiment_20261002.md.

- On 2026-10-01, Casper EV `00001e75--ace5ac2325--9` confirmed SCC-only mode 0
  with every SCC lateral measurement zero. The user requested always using the
  measured SCC object in SCC-only modes and ignoring unreliable SCC lateral
  position. Modes 0/-1 now use SCC longitudinal range/speed directly, then the
  existing probability-qualified vision lead without an extra dPath gate when
  SCC is absent. All SCC matching excludes its lateral coordinate; it cannot
  establish geometric path occupancy. Mode 2 retains low-speed longitudinal
  vision matching and independently corroborated SCC L2; mode 1 excludes SCC
  and mode 3 retains front-first/SCC fallback. Web replay restores the recorded
  source policy instead of forcing mode 2. Original mode-0 replay loses L1 on
  30/1,199 frames with SCC and strong vision; corrected replay loses none and
  uses vision on all 45 SCC-absent frames. Replay is not vehicle-response
  validation. See docs/scc_longitudinal_lead_20261001.md.

- On 2026-09-30, the user approved the handover revision and then explicitly
  selected torque-ceiling-only rapid recovery: keep target angles and existing
  angle limits unchanged, raise the ceiling faster for small error and slower
  for large error, with no angle-error entry gate. This supersedes the earlier
  captured-angle/offset-blending design proposal below. Mode 1 combines effort
  and error levels/trends with tolerance, paused increases and gradual error-only
  withdrawal; strong renewed force still yields quickly. Modes 2/3 use limited
  early capture then low-force confirmation and continuous error-dependent rise.
  Active experimental transitions own the total ceiling, so legacy max() cannot
  bypass their rate; preserve independent legacy history, mode 0 and live polling.
  88 focused tests and 6,000-frame mode-0 CAN/angle equivalence pass. Recorded-input
  schedules are not vehicle response or steering-feel validation. See
  docs/steering_handover_20260930.md for constants, replay and limitations.

- On 2026-09-30, the handover follow-up review uses both angle-error and driver-
  effort trends for mode 1, with tolerance and paused/gradual withdrawal for
  ambiguous error growth, retaining fast yield for strong renewed driver effort.
  For rapid release, onset angle error is an initial transition condition, not
  a delayed small-error permission gate. Offline prototypes avoid ff7's short
  46-to-25 withdrawal and schedule earlier limited offers, but unchanged-input
  replay cannot establish vehicle response. The existing max(legacy, extra)
  ceiling defeats bounded recovery; any implementation must coordinate total
  authority and reference transition while preserving mode 0 and actuator limits.
  This review changes no production code. See
  docs/steering_handover_ff6_ff7_20260930.md for evidence and unresolved cases.

- On 2026-09-30, Ioniq 5 PE ff6 segments 2/3 and ff7 segment 1 on b8a8a532
  confirmed live handover mode 2→3 but delayed recovery after torque release.
  Low-force confirmation can let angle error exceed the two-degree fast-recovery
  gate; other releases miss arming/deadline conditions. No rapid recovery is
  reconstructed in these windows. Legacy repeated-override ramps reach three
  seconds; ff7's last release waits 520 ms then ramps for three seconds. One
  short convergence offer is withdrawn 46→25 in about 12 ms on error growth
  despite decreasing force. This may explain a tactile discontinuity but is not
  proof of the user's exact felt moment. Touch-release edges arrive later and
  are not demonstrated to be a faster cue. No controller change was requested
  or made during this analysis. See docs/steering_handover_ff6_ff7_20260930.md.

- On 2026-09-30, the user requested original-RX-paced forwarding of Hyundai
  CAN-FD CAMERA_SCC cluster 0x161/162/1e0/1ea/200 from bus2 to bus0. Consume
  allowed host copies into independent latest-value caches; use each stock RX
  counter and recompute CRC, including byte-2 8-bit COUNTER for 8-byte 0x200.
  No independent send without RX. Missing/expired host (150 ms) returns stock;
  invalid original frames pass unchanged and invalidate the cache. Preserve
  allowlists, relay protection, non-camera paths and existing control FIFO/reuse.
  Latest-value sampling supports differing rates but can coalesce transient
  displays and delay changes until next RX; freshness bounds host arrival only.
  348 tests, 36,120-frame replay, 285,594 unchanged control comparisons and
  F4/H7 builds pass. Wire timing, vehicle warning resolution and display/chime
  behavior remain unvalidated. Requires updated Panda firmware. See
  docs/canfd_cluster_rx_forwarding.md.

- On 2026-09-30, the user requested live SteerHandoverMode for Hyundai/Kia/Genesis
  angle control: 0 preserves legacy/default, 1 offers bounded recovery using
  continuous driver effort and angle-error trends, 2 confirms abrupt force release
  before faster recovery, and 3 combines them with release priority and no summed
  gains. Poll every 0.5 seconds; only actual mode changes reset experimental history.
  Keep legacy recovery state independent, steeringPressed boolean, torque-control
  vehicles, touch/DM and angle/CAN limits unchanged. Effort is unbounded above 2;
  the offer ceiling 80 is not physical torque or a proven tactile notification.
  Reversal, rising force, error and invalidity withdraw added authority. Mode 0
  matches the prior controller in a 6,000-frame input replay; synthetic/CAN tests
  do not establish closed-loop driving or driver consent. A prolonged zero crossing
  remains ambiguous. See docs/steering_handover_20260930.md for tests and limits.

- On 2026-09-29, the user selected IMU-based suspected-impact detection at 1.5g
  horizontal acceleration, with a visible/audible warning, ten seconds to cancel
  by touching anywhere, then OpenpilotEnabledToggle=false and manager DoReboot.
  Compensate gravity and mounting angle using fresh valid pose/calibration;
  require two fresh samples within 30ms. aEgo is supporting context only.
  Unseen/frozen UI cancels the transition; preserve takeover alert precedence.
  Block control including AlwaysLateral during reboot, preserve normal volume,
  and use the existing bounded reboot sound helper. Saved OFF persists until
  manually enabled; reboot interrupts recording. No incident file protection or
  upload is implied. The 1.5g threshold, drop/rough-road rejection, physical
  display/audio and actual vehicle reboot remain unvalidated. See
  docs/impact_dashcam_20260929.md.

- On 2026-09-29, the user expanded the Carrot Web auto-update reboot sound request
  to ordinary reboots. Use the stdlib-parent common/reboot.py helper for hardware,
  manager, main Web tools and startup recovery: existing prompt.wav once before
  reboot, separate audio child, four-second timeout, saved volume/mute respected.
  Audio failure must not block reboot. Keep update eligibility and recovery policy
  unchanged. Raw OS/factory-reset/standalone-recovery-web commands are not hooked.
  Desktop tests do not establish physical speaker/reboot behavior. See
  docs/reboot_sound_20260929.md.

- On 2026-09-29, the user requested one onroad readiness sound at the first
  engageable state (no NO_ENTRY event), preferring an existing sound. Use
  prompt.wav once after 0.5 seconds of initialized, non-passive, onroad, healthy
  CAN/service readiness and after current alerts finish. systemReady is sound-only
  and lowest priority; preserve warning precedence and normal user/ambient volume.
  The latch lasts for selfdrived's onroad process lifetime. Desktop tests do not
  establish vehicle speaker/timing validation. See docs/system_ready_sound_20260929.md.

- On 2026-09-29, Ioniq 5 PE C4 ff1 segments 0/2 on 250f14ed showed startup
  DM inference/model readiness delay and a separate Jetlink 97.28 ms roundtrip
  causing one model input skip and transient downstream invalidity. Expected
  process PIDs and all camera frame-ID sequences remain continuous; this is
  not evidence of a process crash or sensor capture loss. The user requested
  hiding an absent Jetson: a fresh waiting report is now quiet before modeld's
  first report, with READY restored on healthy connection. Preserve fresh model
  errors, active-session conflicts and stale-link errors. Physical display and
  the underlying isolated latency remain unvalidated. See
  docs/jetlink_ff1_investigation_20260929.md.

- On 2026-09-29, after two GV70 camera-side warning recurrences with unknown
  cause, the user authorized blocking the observed stock-cluster popup and
  requested checking its sound. Scope suppression to GENESIS_GV70_1ST_GEN
  camera-SCC, lateral-only control, HDA_InfoPUDis=3 with the observed camera
  FCA_SYSWARN=1/VALUE63=15 signature and no decoded MDPS/SCC fault or separate
  popup/sound request. Modify only the outgoing cluster copy; retain raw CAN,
  camera state, actual control and other fault/hands-off alerts. Both logs have
  HDA_LFA_WrnSnd=0 and openpilot alertSound=none; popup-associated chime is an
  inference, not confirmed audio causality. Replay removes all four observed
  popup frames; physical display/sound suppression remains unvalidated.
  See docs/canfd_feedback_counters.md. This supersedes the earlier recommendation
  to leave this popup unchanged pending root-cause diagnosis.

- On 2026-09-29, the user clarified that DM's 20-second standard hold starts
  only when surrounding moving traffic appears after an absence. Additional
  vehicles during occupancy do not extend it. Camera monitoring during the hold
  uses stock timing/detection/inputs, expires prior grace and suspends experimental
  resets; camera-unavailable timing stays 15/30/45. Then experimental criteria
  resume, but occupied surroundings cannot earn the empty-road bonus. Retain
  accumulated warnings/lockout and the two-second observation dropout retention.
  No forced warning for attentive drivers. See docs/dm_traffic_hold_20260929.md.

- On 2026-09-29, the user approved the C4 DM inset immediately right of D:
  84x84 at (382,144), leaving 10px before the right strip. VISION moves above it;
  confidence-dot travel returns to full height. C3 placement is unchanged.
  DM event stage1 is visual-only; stage2 (first audible) has final PCM gain
  >=0.7, and stage3 (final) always uses 1.0 regardless of user/ambient volume.
  Match event identity and sound together so navigation sharing the WAV retains
  normal volume. Desktop PCM/UI tests and synthetic rendering do not establish
  physical-device loudness or readability. See docs/dm_onroad_preview_20260928.md.

- On 2026-09-28, the user requested live DriverMonitoringMode changes. Poll
  typed Params every 0.5 seconds in the existing DM dispatcher; ignore the retired
  CARROT_DM_MODE startup latch. Preserve elapsed awareness, calibration, traffic
  hold, warning counts and lockout. A real mode change ends previous interaction
  grace and the forward-attention streak; an unchanged read must preserve them.
  Shorter budgets may immediately trigger warnings; toggling is never attention
  or a lockout reset. See docs/driver_monitoring_dm2.md for desktop validation.

- On 2026-09-28, the user superseded the Ioniq 5 PE-only touch restriction:
  Hyundai/Kia/Genesis CAN-FD uses original ECAN STEER_TOUCH_2AF by received
  profile, without a vehicle-name whitelist. Require the named DBC/address/size,
  existing layout/checksum/status/counter and freshness checks. Discover late
  arrivals with optional registration only after reception; do not add missing-
  hardware CAN faults or populate/modify ADAS TX caches. Address 0x2AF alone
  is insufficient. All 37 configured CAN-FD platforms pass synthetic parser
  tests; physical evidence remains Ioniq 5 PE only. See docs/driver_monitoring_dm2.md.

- On 2026-09-28, the user authorized clearing DM lockout after confirmed parking:
  valid/fresh Park, raw zero speed, standstill and disengaged/inactive status for
  one continuous second, in both modes with or without camera. Filtered speed
  may have only <0.01 m/s settling residue. Speed-only or engage OFF/ON resets
  are excluded. Keep stock policy.py unchanged; selfdrived persists fresh DM
  lock/release transitions so a cleared saved flag cannot relock on DM restart.
  Desktop tests do not validate actual parking. See docs/driver_monitoring_dm2.md.

- On 2026-09-28, the user requested a manual DM switch triggered by three
  distinct physical vehicle CANCEL presses,
  each separated by a release, within three seconds. Held/repeated packets, BT
  CANCEL and automatic-control CANCEL echoes do not count. Another received
  non-CANCEL button event, invalid/stale state, input-stream gap or timeout resets
  progress. On 2026-09-29, the user removed all gear and speed restrictions:
  every gear and standstill are eligible, and gear or speed changes alone do not
  reset progress. Ambiguous stock-ACC speed-button echoes conservatively reset
  progress because they cannot be distinguished from a physical intervening
  press. The
  cancel-echo filter correlates carControl requests rather than confirmed CAN
  transmission, so a physical CANCEL overlapping that 150 ms window may be
  conservatively ignored and must be pressed again. Hyundai/Kia/Genesis
  openpilot-long bypasses this filter because its controller does not transmit
  CANCEL buttons from that path even though the internal request level can stay
  high; stock-long and other platforms retain the filter. Its paired-release
  suppression expires after 0.5 seconds; an interleaved physical press/release
  can therefore require one additional press without causing a false disable.
  The gesture turns DM off only for the current ignition session. The later
  2026-09-29 revision also retains DriverMonitoringEnabled as a persistent,
  default-on switch exposed only through Carrot Web search. It is intended for
  absent or failed DM cameras; recommend leaving it on, and explain that turning
  DM off may violate applicable laws or driving requirements without claiming
  universal illegality. The next ignition-on or manager/device restart clears
  only DriverMonitoringSessionDisabled; a saved Web OFF stays off until the
  user manually enables DriverMonitoringEnabled again. File/QR backups and
  file/QR/profile restore paths exclude DriverMonitoringEnabled, including
  values in older backups; old backup downloads are filtered too. Resetting all
  settings may restore the default ON value. Persistent OFF must be selected
  locally on each device. Disabled DM
  stops the model during normal onroad operation and gates alerts, monitoring
  force deceleration and lockout while retaining a neutral state heartbeat.
  Driver View may run the model only for face preview while enforcement remains
  neutral. Keep DriverMonitoringMode and CarrotVisionEnabled independent;
  DisableDM remains migration-only. Desktop tests do not establish vehicle
  validation. See docs/driver_monitoring_dm2.md and both localized DM guides.

- On 2026-09-28, the user authorized automatic Git update/reboot after failed
  builds or manager startup, waiting through network loss. The launcher owns a
  standalone recovery display and releases its build lock before recovery Git.
  Retry after 30 seconds; automatic reboot requires a newly applied commit, so
  the same broken revision cannot reboot-loop. Keep the manual Git pull/reboot
  button, current branch/upstream, dirty-file protection and shared repo lock.
  No hard reset, normal onroad update action or AGNOS-policy change is implied.
  Graphics failure has a stdlib-only update fallback. Desktop tests/renders do
  not validate physical C3/C4 touch or device reboot. See docs/startup_recovery.md.

- On 2026-09-28, the user requested original Ioniq 5 PE wheel touch in DM,
  explicitly preserving existing ADAS transmission. ECAN 0x2AF raw bytes now
  feed separate CarState.steeringTouch; torque-based steeringPressed and TX
  remain unchanged. Six historical segments verify the receive layout/checksum
  and counter, not physical-contact ground truth or current vehicle validation.
  Accept the lowest reported touch level 1; raw TOUCH1/2 ranges overlap and must
  not become an unvalidated baseline-plus-one threshold. No-camera modes accept
  fresh held contact; camera mode 1 accepts only release-to-contact edges, never
  indefinite grace from holding or reconnecting. Camera mode 0 stays stock.
  Unknown, stale, malformed or frozen-counter data grants no touch credit;
  terminal alerts remain. Scope this empirical profile to Ioniq 5 PE until other
  vehicles are verified. See docs/driver_monitoring_dm2.md.

- On 2026-09-28, the user revised DriverMonitoringMode after the initial DM2
  implementation. Mode 0 keeps stock camera behavior, but unavailable-camera
  interaction timing is now 15/30/45 seconds. Mode 1 uses the same interaction
  timing, doubled only on a verified empty straight road. New moving traffic
  removes the empty-road bonus for 20 seconds. Camera mode 1 uses 2x stock vision
  timing, 4x on a verified empty road, and 20% head-pose tolerance relaxation.
  The user explicitly selected a full interaction grace: fresh control/BT input
  resets monitoring and defers camera warnings for 45/90 seconds before its
  warning clock starts. This supersedes the earlier two-second credit and
  protected-distraction debt restriction; detection thresholds remain unchanged,
  but sleep/eye/phone warnings are also delayed. Confident forward attention for
  two seconds resets the camera clock without renewing interaction grace.
  Terminal alerts and lockout remain; no input or context change clears them.
  Camera absence AND failure automatically use interaction monitoring, with
  recovery preserving progress; do not add a manual camera-installation setting.
  Stock policy/dmonitoringd files stay unchanged. DisableDM is migration-only;
  CarrotVisionEnabled is independent. These are requested experimental timing
  choices, not statutory limits or device/driving validation. See
  docs/driver_monitoring_dm2.md and both localized DM guides.

- On 2026-09-28, the user requested ordinary Git storage wherever possible to
  eliminate this branch's Git LFS bandwidth dependency. All seven remaining
  LFS pointers were converted to byte-identical Git blobs; bundled models and
  the legacy updater are below GitHub's per-file limit. Do not reintroduce LFS
  tracking or setup pulls. Existing NAS model delivery stays unchanged, and
  historical refs are not rewritten. See docs/lfs_to_git_20260928.md.

- On 2026-09-28, the user approved C3/C3X main UI onroad affinity cores0,1,2,3,6
  with SCHED_OTHER/nice19, superseding core6-only for tici/tizi. C4/mici stays
  core6. Apply to all UI threads; offroad returns to little cores, and onroad
  C3 keeps nice19 during big-core unavailability. Cluster/core7, camera/control/
  model/radar and IRQ policies are unchanged. Casper logs on a3278c04 measured
  UI15.54/14.68Hz with camera20Hz and substantial UI runnable wait; this is
  pre-change evidence, not validation of the new mask. Affinity does not pin
  one whole frame or guarantee little-first placement. See docs/camera_core5_trial.md.

- On 2026-09-28, the user requested a single Windows installation ZIP and a
  minimal Korean guide: extract, run 01, run 02, insert the finished card.
  Follow-up requires bilingual stage introductions, approximate durations,
  exact response instructions and brief safety guidance; brevity must not
  remove backup/write-in-progress cautions or Jetson shutdown and power
  disconnection before card insertion. Label the link "설치파일 받기".
  Present Korean first with English underneath on a separate, visually secondary
  line. Use clear stage headings, spacing and a styled offline HTML guide; never
  interleave Korean and English with slash-separated sentences.
  Keep hashes, portable dependencies, USB-C patching and readback automatic;
  do not restore manual Python/Etcher/hash/hotfix steps to the default guide.
  The package prepares a patched file before writing, preserves the published
  base image/runtime/model and confirms the selected USB card before erasing.
  PC preparation and disk-guard tests do not establish physical-card writing
  or first-boot validation. See docs/jetson_windows_installer_20260928.md.

- On 2026-09-27, the user requested full integration of `carrot-jetlink` into
  `carrot-wip` and Korean-first public installation/release instructions. The
  complete Jetlink history through b9950442ca is merged; do not treat it as an
  independently maintained vehicle feature branch or recreate older experiments.
  Keep the existing internal model, AMD Cinque v3 selection, AGNOS and validity
  policies unchanged. Jetson uses its separately pinned Cinque v2 contract and
  signed f2b22dc host release; merging vehicle code does not promote a new host
  runtime/model or justify another image rebuild. Public host sources remain in
  ajouatom/carrot-jetson and images on NAS. PC offline SD patch first-boot and
  integrated vehicle driving/C3 checks remain distinct from prior parked C4
  trials. See docs/jetson_wip_integration_20260927.md and the linked Korean guide.

- On 2026-09-24, the user requested AGNOS updates without per-update approval:
  automatically download/install, wait and retry transient network failures,
  then reboot and continue normal startup. Both startup UIs now start the
  updater without consulting saved confirmation. Keep Wi-Fi setup accessible
  during retries, prevent duplicate workers, and retain image verification,
  inactive-slot installation and fatal-error handling. This does not change
  the required OS image/version or authorize weakening startup compatibility.
  See docs/cinque_v3_integration_20260919.md for behavior and validation limits.

- On 2026-09-24, the user requested cleanup of accumulated root `.tmp_*`
  analysis work. Local archives and an index are under
  `.analysis/archive/2026-09-24/`; they are private, ignored working data,
  not Git-tracked documentation or a remote backup. Use
  `.analysis/scratch/<date>-<task>/` for new temporary analysis, captures,
  dependency installs and Wiki staging instead of new root `.tmp_*` paths.
  At task completion, retain useful findings/reproduction evidence with an
  index in `.analysis/archive/`, then remove reproducible caches and scratch.
  Keep durable conclusions in the relevant tracked investigation document.
  Archived scripts may contain old relative paths; restore their original
  layout in scratch and adjust paths before running them. For radar lead
  validation, pass `--cache-dir .analysis/scratch/radar-validation-cache`
  explicitly to avoid the tool's legacy root cache default. Never include
  local captures, settings snapshots or credentials in commits.

- On 2026-09-24, parked C4 exposure A/B/A on original bt1 reproduced isolated
  driver-camera gaps of 95.671/95.653 ms when switching OS04C10 exposure
  2298 -> 2309. CSID hardware timestamps also gap by 95.677/95.656 ms;
  both other streams stay near 50 ms. Four historical isolated wide gaps have
  the same near-limit exposure-byte crossing and three-frame command offset.
  Grouping exposure/gain writes with manual delayed group-0 launch completed
  600 s / 12,000 frames per camera without a >75 ms gap (maximum57.112 ms).
  A separate brightness trial verified commands affect actual image statistics.
  Camera-only A/B/A had stale model IPC and does not establish model validity;
  a separate 600 s normal-AE/DM/eGPU run after reboot had 12,000 camera/model/
  pose/DM messages each, no invalidity or model skips, camera max57.540 ms and
  accelerometer/gyro max ages34.973/35.796 ms. DisableDM restored to2. Parked C4
  validation does not establish loaded driving or C3 behavior.
  No thresholds, exposure limits, priorities or model behavior are weakened.
  This hazard predates September19; the recent frequency change and BT causality
  remain unproved. f8d--3 is a distinct low-exposure IFE error-signalled fence,
  not a wait timeout, and is not shown fixed by grouping. Radar work did not
  spike before either incident type. The bt2 OS trial remains separate and has
  not been installed. See docs/os04c10_exposure_investigation.md.

- On 2026-09-23, Ioniq 5 C4 `00000594--abf5912e57--7` on 1fbfe331 reproduced
  a 102.044 ms wide-camera BOOT_TS gap with consecutive raw-derived frame and
  request IDs, one skipped model input and about 304 ms invalid pose inputs.
  BOOT_TS is sampled in kernel SOF handling, not an independent sensor clock;
  do not claim a physical sensor or UI cause from this log. Separately, startup
  expected a 25 ms driver offset although bundled Panda still drives all FSIN
  channels in phase from TIM1. Driver staggered_sof is now false, retaining
  the strict startup tolerance and all runtime validity/scheduling policies.
  Passive SOF/receive timing logs are bounded to one per second per camera.
  The startup fix is not a demonstrated fix for the later driving gap; C3/C4
  target validation remains required. See docs/camera_sof_gap_20260923.md.

- On 2026-09-23, Group1 video/CAN comparisons showed reversed front lateral
  coordinates on one Tucson and one Sportage, but normal left/right on a
  Staria; another Sportage was inconclusive. Do not infer upside-down mounting
  or automatically invert a whole model/group. The user approved RadarTrackFlip:
  default normal, manually invert frontRadar yRel/yvRel per vehicle at the next
  onroad start. Preserve SCC/corner/vision and scheduling. liveTracks records
  radarTrackFlipped; replay must avoid double inversion and preserve recorded
  leads. NAS recorded/normal/flipped choices are analysis-only. See
  docs/radar_track_flip.md for offline verification and vehicle-validation limits.

- On 2026-09-23, the user approved the parked display-placement candidate:
  onroad main UI core6 and USB cluster core7, both SCHED_OTHER/nice19;
  offroad both return to cores0..3 before big-core power saving. This supersedes
  the UI-little placement below. Keep camera/control/model/radar priorities and
  placement unchanged. Apply the display policy to all workers, including the
  software encoder child. USB render/encode/controller rate is fixed at10 FPS,
  or5 while UsbGpuActive; removed ClusterHudLiveFps/ClusterHudCoreMode and legacy
  FPS/core environment overrides must not restore custom placement/rates.
  DM-enabled parked UI6/cluster7 measured core2/6/7 means61.7/77.6/85.2%, UI19.77Hz,
  road/wide max48.73/49.02ms, no camera/model/DM gaps or pose/CAN invalidity.
  Core7 still briefly reached100%; C3, loaded driving and actual ignition-off
  hotplug remain unvalidated. Isolated device syscalls verified nice19 and
  core6/7-to-little restoration for threads and a child under comma credentials.
  See docs/camera_core5_trial.md for comparisons and limits.

- On 2026-09-22, the user reported near-idle core6 and busy cores4/5 after
  camera reservation and requested balanced placement. Whole-core /proc/stat
  and deviceState, including background work, confirmed core4 about91%.
  Earlier selected-process CPU sums did not establish total core headroom.
  New parked C4 A/B/A trials place controlsd/selfdrived together on core6
  FIFO53 with camerad SCHED_OTHER. Keep planner/radarcan core4 FIFO51,
  card core5 FIFO53, radard core5 FIFO51, model/DM core7, and UI little/normal.
  This supersedes the camera-exclusive and control-core4 placements below.
  Both-controls trial core4/5/6/7 means were59/63/54/48%; camera road/wide max
  rose to47.261/49.151ms while planner work max fell to8.613ms and radar input
  age max to13.360ms. No model/pose/CAN failure occurred. Core4 still briefly
  reached100%; parked timing is not proof of loaded driving or C3 behavior.
  Follow-up temporarily enabled DM:90s yielded1804 valid DM frames with no
  DM/driving-model skips or pose/CAN failures, whole-core means61/71/46/69%,
  road/wide max51.370/52.838ms. DisableDM was restored to2 afterwards.
  Preserve priorities and validity thresholds. See docs/camera_core5_trial.md.

- On 2026-09-22, after three parked C4 grouping comparisons, the user approved
  leaving camerad/camera IRQ on core6, moving card to core5 FIFO53 with radard
  FIFO51, and moving planner to core4 FIFO51 with radarcan below the unchanged
  controlsd/selfdrived FIFO53. This supersedes the card/planner placements in
  earlier entries. Camera/UI remain SCHED_OTHER; model/DM stay on core7.
  The selected 60-second trial reduced road/wide maximum ages to 36.889/37.335ms
  but increased planner maximum work from about 8ms to 19ms and radar input
  maximum age from about 15ms to 25ms. No CAN/pose/model-gap failure occurred.
  DM was disabled; loaded driving, DM-enabled behavior and C3 are unvalidated.
  Core6 is reserved by application placement, not free of all kernel/IRQ work.
  Preserve pose limits and radar semantics. See docs/camera_core5_trial.md.

- On 2026-09-22, parked Ioniq 5 C4 follow-up reproduced a 91.961ms road-camera
  delay, model input frame gap and invalid odometry/pose inputs with ftrace off;
  IMU ages stayed below 34ms. Live boot args isolate only cores6..7. Kernel
  tracing on core5 verified ready-camera scheduling delays behind normal
  proclogd/kswapd work and FIFO planner/radard. The high-reclaim trace also had
  diagnostic tmpfs overhead; do not treat its frequency as an unperturbed result.
  The camera/IRQ core5 trial below is rolled back to core6, retaining UI on
  cores0..3 and camera/UI SCHED_OTHER. A short 5/6/5 parked comparison reduced
  core6's observed tail/runqueue wait but added about 3ms mean camera age.
  A nice=-10 trial did not materially improve mean/p99; keep nice0. Preserve
  other process placements and pose limits. This is a measured mitigation, not
  proof of a driving fix, C3 benefit or the cause of earlier SOF/IFE faults.
  See docs/camera_core5_trial.md for evidence and limitations.

- On 2026-09-22, PV5 follow-up `0000022a--99e06b0cc0--17` disproved
  interpreting A-CAN 0x380 bit 6 falling as camera passage: roughly 4.7 s
  notification pulses ended while MapSource=2 and a 30 km/h camera still
  had 139 m of tracked distance. Do not use that edge or byte value 0x04 as
  proof of passage. Retain a matched PV5 camera only while fresh messages
  confirm the same MapSource=2 enforcement limit. End/change/invalidity or
  more than one second of message loss clears current and queued cameras;
  cached profiles cannot reinsert them. Queued previews alone never authorize
  PV5 camera control. Consume a completed camera's unchanged map warning;
  a new warning without a new matched profile uses a separate virtual distance.
  PV5 does not decode current-route/position 0x4B9/0x4B4: do not claim that
  generic route-reset code detects its departure. If stock enforcement remains
  unchanged after a turn, this cancellation cannot detect that turn independently.
  Follow-up replay reconstructs initial targets from logged distances because
  the preceding segment is unavailable; current cancellation drops those old
  previews and uses the warning's virtual distance at the reported pulse end.
  Cancellation/recovery/completion tests are synthetic, not vehicle validation.
  Earlier bit-transition tests did not establish physical passage semantics.

- On 2026-09-21, after Ioniq 5 C4 `00000f90--96d7dcd525--4` reproduced a
  101 ms wide-camera SOF gap, the user authorized a CPU-placement trial:
  main UI uses cores0..3 with SCHED_OTHER (core0 bootstrap), camerad and its
  camera IRQ targets move from core6 to core5. card remains core6 FIFO53;
  planner/radard remain core5 FIFO51 and camera keeps normal scheduling.
  Preserve the UI's verified SCHED_OTHER contract and pose validity limits.
  This supersedes the camera/UI placements described in older observations,
  not radar isolation or cluster affinity. No C3/C4 vehicle benefit is yet
  validated; do not claim same-core contention caused the camera fault.
  See docs/camera_core5_trial.md for scope, trade-offs and validation.

- On 2026-09-21, ID.4 replay showed that adding CP.radarDelay (0.8 s) to
  distance alignment could switch the selected lead to a farther CAN object.
  The user approved zero extra distance projection for VW MEB. Use the shared
  radar_motion/timing.py policy in runtime and NAS replay; preserve measured
  camera/publication skew. Do not also zero CP.radarDelay: its ego-history
  compensation and velocity/acceleration effects have not been recalibrated.
  Other platforms retain their existing delay. See
  docs/meb_radar_distance_alignment.md for scope and regression evidence.

- On 2026-09-21, K9 C4 logs reproduced locationd timing-check invalidity from
  repeated IMU timestamps over 100 ms old. Historical captures first showed
  these failures after the September 19 update, despite unchanged HUD 10 FPS,
  cores 1..4 and FIFO 10. The user authorized normal SCHED_OTHER scheduling for
  cluster autorun/render workers so sensord FIFO 1 and other realtime work
  take precedence. Keep legacy ClusterHudPriority/environment overrides from
  restoring FIFO; retain FPS and core selection. This is a contention mitigation,
  not a proved fix for the OS/runtime regression. Do not weaken pose validity
  thresholds or claim vehicle validation from desktop tests. Official 521db4c
  changes initial gyro-bias covariance, not the observed sensor timestamp delays.

- On 2026-09-21 the user authorized radar optimization and preprocessing
  isolation to reduce card/camerad contention on core6, with mandatory radar
  regression validation. RadarInterface/liveTracks now belong to radarcan on
  core4 FIFO51 (below controlsd/selfdrived FIFO53); card remains core6 FIFO53,
  model-driven radard/planner core5. Preserve carState.radarInput batch metadata
  and non-conflated CAN/ego joining: using an arbitrary latest ego sample breaks
  delay/filter cadence. Keep planner's existing fast liveTracks path during this
  first isolation step. See docs/radar_process_isolation.md for equivalence,
  corpus failures and limits. C3/C4 device timing/camera improvements are NOT
  yet validated. Never present same-core contention as a proved IFE root cause
  or desktop speedup as a vehicle result. Radar changes also require NAS replay
  deployment and actual result verification below.

- On 2026-09-20, EV9 `3eef70e8fb92485c` (tizi/C3 family) reproduced Cinque v3
  dropped-frame odometry invalidity even with `xiaoge_data` stopped. Raw-image
  upload averaged 24.75 ms and model execution 50.69 ms; C4 `07b62e389ed26c81`
  used the same artifact at 13.94 ms upload / 39.73 ms execution. Old TG code
  warped on QCOM before transferring model-sized images, whereas generic v3
  uploads full NV12 images before AMD warp. The user approved C3-only QCOM
  pre-upload warp while retaining the existing C4 path. See
  `docs/c3_preupload_warp.md` for implementation, validation limits and evidence.
  EV9 segment `000002c9--15d447d91b--0` on 9a349b60 failed the QCOM/AMD pixel
  comparison and fell back to AMD (7,471,616 USB bytes, about 24.8 ms upload).
  At that stage the optimization was NOT confirmed active. Diagnose the
  per-probe mismatch details before changing warp math or acceptance criteria.
  Follow-up `000002ca--50469cb155--0` on 820f82ea found 16 repeat-stable
  projective-only mismatches; all eight logged samples reconstruct as adjacent
  source pixels at half-pixel rounding boundaries. Validation now checks each
  mismatch against correct-camera/plane NV12 source values within 0.00025
  source pixels of a rounding boundary. Do not replace this with a percentage
  or intensity tolerance. On 2026-09-21, EV9 `000002cc--03d0a44f7d--10`
  on 2723a8eb confirmed QCOM active: 393,728 USB bytes, 5.36 ms upload,
  34.98 ms mean model execution. Four remaining warnings matched complete
  camera streams with 12.6-13.2 ms SOF skew: Carrot's strict 10 ms pairing
  discarded four main frames. Current pairing allows at most 20 ms skew;
  metadata replay retains all 1,200 EV9 pairs while preserving real Ioniq
  phase-slip/IFE-loss gaps. This is not on-device validation of the pairing fix.
  Official v3 also publishes invalid odometry after a real main-frame gap;
  do not describe this policy as a Carrot-only regression. Official pairing
  logs >10 ms skew but proceeds; Carrot still bounds large/stale pairs.
  Preserve official model input/outputs and recurrent state; never hide overload
  by weakening pose validity. Evaluate model/runtime updates per device family;
  do not assume C4 validation covers C3, or automatically freeze all C3 models.
  Keep World Model experiments local and apply its separate validation rules.

- As of 2026-09-20, the user requested deletion of the remote `carrot-worldmodel`
  branch to prevent others from installing an unfinished experiment. Keep this
  experiment local only; do not recreate or push its remote branch unless the
  user explicitly authorizes publication again. Continue applying common
  `carrot-wip` changes locally while preserving World Model-specific artifacts
  and runtime work. Only `carrot-wip` must be pushed for shared changes; this
  exception does not restore any retired branch. World Model has passed isolated
  synthetic inference, but vehicle control integration remains unvalidated.

- As of 2026-09-19, the user requests full integration of `carrot-cinque_v3` into
  `carrot-wip`, including the pinned Cinque v3 eGPU model/runtime, AGNOS
  `19.8-carrot-bt1`, and Bluetooth remote features. This supersedes the earlier
  Cinque v2/OS separation below. Keep the internal-GPU driving model unchanged;
  driver monitoring uses official Super Leicht (#38942). After successful
  integration the user explicitly retired `carrot-cinque_v3`; `carrot-wip` is
  the sole maintained top-level `carrot-*` branch. Do not recreate v3 or push
  changes to its detached worktree. Its complete history is merged into wip.

- Whenever radar detection or lead-selection code changes, update the NAS Carrot Routes
  radar replay service in the same task. The `Carrot Routes image` GitHub workflow builds
  committed shared code using `tools/carrot_route_vault/build_bundle.py`; the NAS scheduled
  updater pulls, validates and deploys the tested image to the existing
  `/volume1/docker/carrot-route-vault` project. Use this automatic path for routine updates;
  do not manually copy sources or rebuild on the NAS. Verify the workflow and NAS updater
  report the intended commit, and verify an actual upload-result page and
  its recalculated radar data before reporting completion. This includes replay adapters and
  shared detection dependencies such as `radar_motion`, `cluster`, cut-in helpers, and required
  cereal/DBC compatibility changes. Keep the server's code fingerprint/cache invalidation and
  deployed `SOURCE_COMMIT` current; a vehicle-branch push alone does not complete this work.
- For long-running work, treat user questions, status checks, clarifications, and added in-scope
  requests as interruptions to answer while continuing the active work. Stop an active process or
  abandon the task only when the user explicitly asks to stop, cancel, pause, or replace it.
- Apply model selection/artifacts, model and branch names, required model compatibility changes,
  and branch-specific features (such as YOLO2) only to their relevant branches. Preserve these
  differences when synchronizing shared code; do not spread an experiment to other branches.
  The user will explicitly identify new feature experiments and their target branches.
- As of 2026-09-13, `carrot-wip` is the sole maintained top-level `carrot-*` branch.
  It incorporates the complete `carrot-cinque_v2` history. Its former Cinque v2
  selection was superseded by the 2026-09-19 integration above.
  Commit and push common changes, including radar processing and Carrot Web, to `carrot-wip`;
  verify it matches `origin/carrot-wip` with no unpushed commits before completion.
  Do not recreate retired branches or synchronize changes to their archive tags or detached
  worktrees. Retired local branch tips are preserved under `archive/2026-09-13/<branch>`.
  Namespaced contributor branches such as `thftgr/carrot-*` are outside this consolidation.
- Create new model or feature experiment branches from current `carrot-wip` only when the user
  explicitly requests them. Keep their model selections, generated display assets, compatibility
  changes and dedicated features scoped to those experiments; agree their maintenance scope
  with the user instead of automatically restoring the retired multi-branch synchronization rule.
- The 2026-09-17 exception maintaining `carrot-cinque_v3` separately ended on
  2026-09-19 after its complete integration and the user's explicit deletion request.
- On this Windows workstation, vehicle tmux session captures are stored under
  `\\DS1821P\openpilot\<branch>`. When tmux is mentioned, search the directory for the known
  branch for a vehicle folder whose name ends with the exact dongle ID. If the branch is unknown,
  search `\\DS1821P\openpilot` across branch directories for the exact dongle ID first. Do not
  start by looking for a local Windows or WSL tmux installation.
- Big-model ONNX files and manifests are hosted on the user's NAS under
  `\\DS1821P\openpilot\models\<model-directory>`. Vehicles download the same files through
  `https://upload.shind0.synology.me/models/<model-directory>/`. For a new big-model branch,
  create a distinct model directory, place the verified ONNX and `manifest.json` there, and point
  the branch at that NAS manifest. Do not use GitHub LFS as the vehicle download source.
- Navigation deceleration behavior for the `origin/thftgr/navi-stream` branch is documented in
  `docs/carrot_navi_7713_7714_deceleration.md`.
- The 7714-only control comparison between `origin/carrot-wip` and `origin/thftgr/navi-stream` is
  documented in `docs/carrot_navi_7714_branch_comparison.md`.
- When changing the UDP 7713 legacy navigation path, TCP 7714 Carrot Navi v2 path, shared
  `CarrotServ` speed selection, or the on-road/cluster speed-source UI, update that document and
  the focused tests together.
- Both `origin/carrot-wip` and `origin/thftgr/navi-stream` evaluate a new 7714 speed item before
  assigning the current `lane_current.road_category`. A present lane item without that key becomes
  category 0. This can suppress primary SDI type 22 (`xSpdType=-1`) even though raw 7714 UI shows it;
  7713 assigns `roadcate` before `_update_sdi()` and does not have this ordering failure.
- The `origin/thftgr/navi-stream` cluster road-camera/map flicker analysis is documented in
  `docs/cluster_road_camera_map_flicker_analysis.md`. The hardware H.264 map and TICI road camera
  both use `samplerExternalOES` on texture unit 0; each external image must be rebound immediately
  before every draw, not only when its source frame changes.

# Vehicle settings snapshots

- On this Windows workstation, uploaded vehicle settings are stored under
  `W:\<branch>\<car-fingerprint> <dongle-id>\toggles-YYYYMMDD-HHMMSS.json`.
- To find a vehicle's most recent settings, first search all of `W:\` for directories whose names
  end with the exact dongle ID. Gather `toggles-*.json` from every matching directory and select the
  file with the newest timestamp encoded in its filename.
- A dongle can appear under several branch or fingerprint directories. For incident analysis,
  narrow the matches using the branch and car fingerprint from the route/upload metadata, then
  inspect the newest snapshot at or before the incident time and compare it with the newest later
  snapshot.
- Treat the JSON values as raw Params values; for example, `StoppingAccel` is stored in hundredths
  of m/s^2.

# Vehicle route log lookup

- On this Windows workstation, vehicle route logs are stored under `\\DS1821P\openpilot\routes`. Whenever
  the user mentions an `rlog`, a route log, or a vehicle log, start by searching that share for a
  vehicle directory whose name ends with the exact dongle ID or device ID. The vehicle directory
  name identifies the car fingerprint; its child directories identify the route/segment numbers.
  Do not start by searching the repository, tmux captures, or another route root.

- A `Carrot Dashcam Upload` result is always uploaded under `\\DS1821P\openpilot\routes`. When the user
  provides a `Carrot Dashcam Upload` block, resolve that local route first and do not start from
  the remote result link, the repository, tmux captures, or another route root. Build the exact
  preferred log path as
  `\\DS1821P\openpilot\routes\<Car name> <DongleId>\<Result segment>\rlog.zst`.
- Decode presentation escaping before building the path: Markdown `\_` is `_`, URL `%20` is a
  space, and the result link's final directory name is the route segment.
- If the exact path is absent, search `\\DS1821P\openpilot\routes` for a vehicle directory ending
  in the exact dongle ID and then the exact result segment. Prefer `rlog.zst`; use `qlog.zst` only
  when the full log is unavailable.
- Treat Upload Time, Branch, and Commit as incident-analysis metadata, not as path components.

# Vehicle rlog decoding

- For full rlog analysis, use the full OpenPilot cereal schema. Prefer
  `openpilot.tools.lib.logreader.LogReader`; do not use
  `opendbc.car.logreader.LogReader(..., only_union_types=True)` to determine which services are
  present. The opendbc reader loads the reduced `opendbc/car/rlog.capnp` schema and silently drops
  services unknown to it, which can make `carState`, `carrotMan`, `navInstruction`,
  `navInstructionCarrot`, `controlsState`, and `carControl` appear absent.
- On Windows, if importing `openpilot.tools.lib.logreader` fails because of platform-only
  dependencies such as `fcntl`, decompress the `.zst` file with `zstandard` and parse it directly
  with `openpilot.cereal.log.Event.read_multiple_bytes`. Use `opendbc.car.logreader` only for an
  intentionally CAN-only inspection.
- Before reporting that a service is missing, count message types with the full cereal schema and
  inspect at least one expected service value. When calculating segment-relative time, use the
  first message of the analyzed service (for example `can` or `carState`) rather than `initData`,
  whose boot-time timestamp can make later segments appear cumulatively longer.

# User documentation policy

- Do not create or edit files under `docs/user/ko/` or `docs/user/en/` unless the user explicitly
  requests user-documentation work. User-visible code changes alone do not authorize guide edits.
- The user has explicitly requested that every user-visible setting addition, removal, or behavior
  change update the relevant Korean and English user guides in the same change. Treat setting work
  as user-documentation work, keep the catalog summary and detailed guide synchronized with the
  implementation, and run the user-docs validator.
- Also keep setting-level explanations in the generated GitHub Wiki workflow. Keep explanations
  specific to web-only features in the localized UI instead of duplicating web internals into
  `docs/user/`.
- `docs/user/docs_map.json` and `tools/docs/check_user_docs.py` are validation aids, not instructions
  to generate documentation. For an ordinary code pull request without explicitly requested docs,
  record a concrete `Docs-Not-Needed: <reason>` in the PR body when the workflow requires it.
  For direct pushes, put the reason in each affected commit message; it only exempts that
  commit. Settings-related rules remain required; other mapped changes produce review advice
  and do not authorize unsolicited guide edits. Settings behavior changes still require the
  relevant Korean/English guides and Wiki explanations, even when outside mapped paths.
- Do not place private, internal-only, credential-bearing, or non-public feature documentation in
  `docs/user/` or link it from the public Wiki.

# Settings Wiki authoring

- On 2026-10-08, the user explicitly gave standing approval to commit and push
  task-related Wiki updates to `https://github.com/ajouatom/openpilot.wiki.git`,
  including its default `master` branch. Do not ask for separate Wiki-push
  confirmation again for requested work. Continue following the authoring and
  validation rules below and preserve unrelated Wiki content.

- Before editing generated settings Wiki content or its generator, read
  `tools/docs/wiki_settings/AUTHORING_GUIDE.md` completely.
- In an existing generated Wiki setting page, edit only the matching `CARROT:MANUAL` region.
  Preserve every `CARROT:*` marker and never hand-edit `CARROT:AUTO` content.
- Verify behavior against the current `carrot-wip` implementation instead of inferring it from the
  parameter name. Run the Wiki validator and focused generator tests after editing.
- Generated setting pages carry the same authoring-guide URL in a hidden `CARROT:AUTHORING` marker
  so an agent working from the Wiki checkout alone can discover the canonical rules.
