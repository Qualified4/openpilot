import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_preview import (
  DRIVING_MODE_ECO,
  DRIVING_MODE_HIGH,
  DRIVING_MODE_NORMAL,
  DRIVING_MODE_SAFE,
  LEAD_ACCEL_RESPONSE_TUNING,
  OPENING_A_CHANGE_COST_FACTOR,
  OPENING_COAST_ACCEL_MAX,
  OPENING_JERK_COST_FACTOR,
  OPENING_PREDICTION_S,
  LeadAccelResponseState,
  apply_preview_target,
  clip_action_time,
  clip_preview_offset,
  get_lead_accel_mpc_request,
  get_lead_preview_request,
  lead_accel_response_allowed,
  lead_accel_response_source_allowed,
  lead_gap_opening_while_coasting,
  rate_limit_preview,
)


def preview_request(mode, a_lead, *, a_ego=0.0, lead_status=True):
  return get_lead_preview_request(
    mode,
    lead_status=lead_status,
    a_lead=a_lead,
    a_ego=a_ego,
  )


def mpc_request(level, *, source='lead0', enabled=True, lead_status=True,
                a_lead=1.1, a_ego=0.0, v_rel=0.0, gap_margin=3.0,
                speed_error=20.0):
  return get_lead_accel_mpc_request(
    level,
    enabled=enabled,
    source=source,
    lead_status=lead_status,
    a_lead=a_lead,
    a_ego=a_ego,
    v_rel=v_rel,
    gap_margin=gap_margin,
    speed_error=speed_error,
  )


@pytest.fixture
def planner_response_gate():
  # Evaluate the production activation condition without loading the native MPC
  # solver. This catches a gap/personality restriction reintroduced at the caller.
  planner_path = Path(__file__).resolve().parents[1] / "lib" / "longitudinal_planner.py"
  tree = ast.parse(planner_path.read_text(encoding="utf-8"))
  condition = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "lead_accel_response_enabled"
                           for target in node.targets))
  code = compile(ast.Expression(condition), str(planner_path), "eval")

  def enabled(personality, level=3, *, reset_state=False, gas_pressed=False,
              force_slow_decel=False, accel_max=1.6, should_stop=False):
    return eval(code, {
      "sm": {"selfdriveState": SimpleNamespace(personality=personality),
             "carState": SimpleNamespace(gasPressed=gas_pressed)},
      "log": SimpleNamespace(LongitudinalPersonality=SimpleNamespace(aggressive="aggressive")),
      "carrot": SimpleNamespace(leadAccelResponse=level, lane_change_active=False),
      "reset_state": reset_state,
      "force_slow_decel": force_slow_decel,
      "accel_limits_turns": [-2.0, accel_max],
      "self": SimpleNamespace(output_should_stop=should_stop),
    })

  return enabled


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed", "moreRelaxed"])
@pytest.mark.parametrize("level", range(1, 6))
def test_planner_enables_response_at_every_following_gap(planner_response_gate, personality, level):
  assert planner_response_gate(personality, level)


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed", "moreRelaxed"])
@pytest.mark.parametrize("blocked", [
  {"level": 0}, {"reset_state": True}, {"gas_pressed": True},
  {"force_slow_decel": True}, {"accel_max": 0.0}, {"accel_max": -0.5}, {"should_stop": True},
])
def test_every_gap_preserves_planner_response_inhibits(planner_response_gate, personality, blocked):
  assert not planner_response_gate(personality, **blocked)


@pytest.mark.parametrize("mode", [
  DRIVING_MODE_SAFE,
  DRIVING_MODE_ECO,
  DRIVING_MODE_NORMAL,
  DRIVING_MODE_HIGH,
])
def test_all_modes_keep_same_lead_deceleration_preview(mode):
  result = preview_request(mode, -0.5)
  assert result.active
  assert result.lead_accel_signal == pytest.approx(-0.4)
  assert result.offset_s == pytest.approx(0.4)


@pytest.mark.parametrize("mode", [
  DRIVING_MODE_SAFE,
  DRIVING_MODE_ECO,
  DRIVING_MODE_NORMAL,
  DRIVING_MODE_HIGH,
])
def test_positive_lead_acceleration_is_not_post_mpc_preview(mode):
  result = preview_request(mode, 1.1)
  assert result.active
  assert result.lead_accel_signal == pytest.approx(1.0)
  assert result.offset_s == 0.0


def test_preview_uses_relative_acceleration_and_releases_when_matched():
  assert preview_request(DRIVING_MODE_NORMAL, 0.0, a_ego=0.8).lead_accel_signal == pytest.approx(-0.7)
  matched = preview_request(DRIVING_MODE_NORMAL, -0.5, a_ego=-0.5)
  assert matched.lead_accel_signal == 0.0
  assert matched.offset_s == 0.0


def test_preview_is_disabled_for_invalid_or_missing_lead():
  assert not preview_request(DRIVING_MODE_SAFE, -1.0, lead_status=False).active
  assert not preview_request(DRIVING_MODE_SAFE, float('nan')).active
  assert not preview_request(DRIVING_MODE_SAFE, -1.0, a_ego=float('nan')).active


def test_response_costs_progress_from_gentle_to_maximum():
  tunings = [LEAD_ACCEL_RESPONSE_TUNING[level] for level in range(1, 6)]
  assert [t.a_change_cost_factor for t in tunings] == sorted(
    (t.a_change_cost_factor for t in tunings), reverse=True,
  )
  assert [t.jerk_cost_factor for t in tunings] == sorted(
    (t.jerk_cost_factor for t in tunings), reverse=True,
  )


@pytest.mark.parametrize(("level", "a_change_factor", "jerk_factor"), [
  (1, 0.95, 0.95),
  (2, 0.85, 0.85),
  (3, 0.65, 0.70),
  (4, 0.18, 0.35),
  (5, 0.05, 0.15),
])
def test_active_response_returns_mpc_cost_factors(level, a_change_factor, jerk_factor):
  result = mpc_request(level)
  assert result.active
  assert result.level == level
  assert result.a_change_cost_factor == pytest.approx(a_change_factor)
  assert result.jerk_cost_factor == pytest.approx(jerk_factor)


def test_cruise_source_response_starts_at_balanced_level():
  assert not lead_accel_response_source_allowed(2, 'cruise')
  assert lead_accel_response_source_allowed(3, 'cruise')
  assert lead_accel_response_source_allowed(4, 'cruise')
  assert lead_accel_response_source_allowed(5, 'cruise')
  assert lead_accel_response_source_allowed(1, 'lead0')
  assert not lead_accel_response_source_allowed(5, 'e2e')


@pytest.mark.parametrize("level", [3, 4, 5])
def test_cruise_source_uses_positive_raw_lead_acceleration(level):
  result = mpc_request(
    level,
    source='cruise',
    a_lead=0.6,
    a_ego=0.6,
    v_rel=0.4,
    gap_margin=1.0,
  )
  assert result.active


def test_gentle_levels_cannot_enable_cruise_source_response():
  assert not mpc_request(2, source='cruise', v_rel=0.4).active


@pytest.mark.parametrize("level", [3, 4, 5])
def test_cruise_response_releases_when_lead_acceleration_ends_or_gap_will_close(level):
  # a_ego 0.5: an ego still below OPENING_COAST_ACCEL_MAX would see an opening gap.
  assert not mpc_request(level, source='cruise', a_lead=0.1, a_ego=0.5, v_rel=0.4).active
  assert not mpc_request(
    level, source='cruise', a_lead=0.6, a_ego=2.0, v_rel=0.0,
  ).active


@pytest.mark.parametrize("level", range(1, 6))
def test_every_level_releases_when_raw_lead_acceleration_ends(level):
  # v_rel 0.5 with a_ego -0.5 is now an opening gap (released below); keep the
  # original intent with a gap that is not opening, or an ego that already accelerates.
  assert not mpc_request(level, a_lead=0.1, a_ego=-0.5, v_rel=0.3).active
  assert not mpc_request(level, a_lead=0.1, a_ego=0.5, v_rel=0.5).active


def test_cruise_response_requires_set_speed_headroom():
  assert not mpc_request(5, source='cruise', speed_error=1.0 / 3.6).active
  assert mpc_request(5, source='cruise', speed_error=1.01 / 3.6).active


def test_acceleration_response_requires_gap_and_relative_speed_margin():
  assert not lead_accel_response_allowed(1, v_rel=0.0, gap_margin=0.0,
                                         lead_accel_signal=1.0, a_lead=1.0)
  assert lead_accel_response_allowed(3, v_rel=0.0, gap_margin=0.01,
                                     lead_accel_signal=1.0, a_lead=1.0)
  assert not lead_accel_response_allowed(3, v_rel=0.0, gap_margin=-0.01,
                                         lead_accel_signal=1.0, a_lead=1.0)
  assert not lead_accel_response_allowed(5, v_rel=-0.21, gap_margin=1.0,
                                         lead_accel_signal=1.0, a_lead=1.0)
  assert lead_accel_response_allowed(5, v_rel=-0.20, gap_margin=1.0,
                                     lead_accel_signal=0.4, a_lead=1.0)


@pytest.mark.parametrize("level", range(1, 6))
def test_every_level_returns_to_normal_cost_at_configured_tf(level):
  assert mpc_request(level, v_rel=0.2, gap_margin=0.01).active
  assert not mpc_request(level, v_rel=0.2, gap_margin=0.0).active
  assert not mpc_request(level, v_rel=0.2, gap_margin=-0.01).active


def test_level_five_requires_positive_raw_lead_acceleration():
  assert not mpc_request(5, a_lead=0.0, a_ego=-0.5, v_rel=0.5).active
  assert not mpc_request(5, a_lead=-0.7, a_ego=-2.1, v_rel=0.6).active


# 2026-11-11 0000045e seg 4 +45 s: braking at -2.1 m/s^2 while the lead only
# braked at -0.7 (aLeadK); the gap opened at +0.6..+1.5 m/s but the response
# stayed off until aLeadK > 0.1, 2.5 s later, and we lost a further 9 km/h.
@pytest.mark.parametrize("level", range(1, 5))
@pytest.mark.parametrize("source", ['lead0', 'cruise'])
def test_lead_opening_while_we_brake_harder_enables_response(level, source):
  if source == 'cruise' and level < 3:
    pytest.skip("gentle levels never use the cruise source")
  result = mpc_request(level, source=source, a_lead=-0.69, a_ego=-2.12, v_rel=0.64, gap_margin=1.5)
  assert result.active and result.opening
  tuning = LEAD_ACCEL_RESPONSE_TUNING[level]
  # The opening release is stronger than the level's positive-lead factors.
  assert result.a_change_cost_factor < tuning.a_change_cost_factor or result.strength < 1.0
  assert result.a_change_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_A_CHANGE_COST_FACTOR))
  assert result.jerk_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_JERK_COST_FACTOR))


@pytest.mark.parametrize("level", range(1, 5))
def test_lead_opening_release_survives_the_entry_ramp(level):
  state = LeadAccelResponseState()
  request = mpc_request(level, source='lead0', a_lead=-0.69, a_ego=-2.12, v_rel=0.64, gap_margin=3.0)
  for _ in range(40):
    result = state.update(request, .05, 10)
  assert result.a_change_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_A_CHANGE_COST_FACTOR))
  assert result.jerk_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_JERK_COST_FACTOR))
  # A positive-lead response at the same level keeps the level's own factors.
  normal = mpc_request(level, a_lead=1.1, a_ego=0.0, v_rel=0.2, gap_margin=3.0)
  assert not normal.opening
  assert normal.a_change_cost_factor == pytest.approx(1.0 - normal.strength * (1.0 - LEAD_ACCEL_RESPONSE_TUNING[level].a_change_cost_factor))


@pytest.mark.parametrize("blocked", [
  {"a_ego": -0.4},                  # not braking: normal cost, as before
  {"v_rel": 0.2},                   # gap not opening yet
  {"a_lead": -2.5},                 # lead brakes harder than we do
  {"gap_margin": -1.0},             # still inside the TF even 1.5 s ahead (0.0 now opens: predicted +0.96 m)
])
def test_lead_opening_response_keeps_existing_safety_gates(blocked):
  kwargs = dict(source='cruise', a_lead=-0.69, a_ego=-2.12, v_rel=0.64, gap_margin=1.5)
  kwargs.update(blocked)
  assert not mpc_request(3, **kwargs).active


def test_inactive_request_restores_default_mpc_cost_factors():
  for result in (
    mpc_request(5, enabled=False),
    mpc_request(5, lead_status=False),
    mpc_request(0),
    mpc_request(5, source='e2e'),
  ):
    assert not result.active
    assert result.a_change_cost_factor == 1.0
    assert result.jerk_cost_factor == 1.0


def test_preview_rate_and_action_time_are_bounded():
  assert rate_limit_preview(0.60, 0.0) == pytest.approx(0.08)
  assert rate_limit_preview(0.0, 0.20) == pytest.approx(0.17)
  assert rate_limit_preview(-0.10, 0.0) == pytest.approx(-0.03)
  assert clip_action_time(2.40, 1.0) == pytest.approx(2.50)
  assert clip_action_time(0.20, -1.0) == pytest.approx(0.05)


def test_zero_preview_preserves_configured_actuator_delay():
  assert clip_action_time(2.05, 0.0) == pytest.approx(2.05)


def test_negative_preview_cannot_cross_zero_actuator_delay():
  long_actuator_delay = 0.15
  base_action_t = long_actuator_delay + 0.05
  effective_preview = clip_preview_offset(base_action_t, -1.0)
  assert effective_preview == pytest.approx(-long_actuator_delay)


def test_preview_only_removes_acceleration_with_bounded_prebraking():
  assert apply_preview_target(0.20, 0.40, DRIVING_MODE_SAFE) == pytest.approx(0.20)
  assert apply_preview_target(0.20, -0.20, DRIVING_MODE_SAFE) == pytest.approx(-0.05)
  assert apply_preview_target(0.08, -0.20, DRIVING_MODE_ECO) == pytest.approx(-0.04)
  assert apply_preview_target(-0.50, -1.0, DRIVING_MODE_NORMAL) == pytest.approx(-0.58)


@pytest.mark.parametrize("mode", [
  DRIVING_MODE_SAFE,
  DRIVING_MODE_ECO,
  DRIVING_MODE_NORMAL,
  DRIVING_MODE_HIGH,
])
def test_preview_never_increases_mpc_acceleration_output(mode):
  assert apply_preview_target(0.20, 1.20, mode) == pytest.approx(0.20)
  assert apply_preview_target(-0.50, 0.50, mode) == pytest.approx(-0.50)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_small_lead_changes_and_small_gap_margin_fade_boost(level):
  # v_rel 0.3 (not above OPENING_V_REL) keeps the coasting-opening release out of this test.
  full = mpc_request(level, a_lead=1., v_rel=.3)
  quiet = mpc_request(level, a_lead=.101, v_rel=.3)
  near = mpc_request(level, a_lead=1., v_rel=.3, gap_margin=.001)
  assert full.active and quiet.active and near.active
  assert full.a_change_cost_factor < quiet.a_change_cost_factor < 1.
  assert full.a_change_cost_factor < near.a_change_cost_factor < 1.
  assert quiet.a_change_cost_factor > .99
  assert near.a_change_cost_factor > .99


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_response_entry_ramps_and_track_change_restarts(level):
  state = LeadAccelResponseState()
  request = mpc_request(level)
  first = state.update(request, .05, 10)
  assert first.a_change_cost_factor > request.a_change_cost_factor
  for _ in range(20):
    full = state.update(request, .05, 10)
  assert full.a_change_cost_factor == pytest.approx(request.a_change_cost_factor)
  assert state.update(request, .05, 11).strength == pytest.approx(first.strength)


@pytest.mark.parametrize("level", range(1, 6))
@pytest.mark.parametrize("blocked", [{"a_lead": -3.}, {"gap_margin": 0.}, {"v_rel": -3.},
                                     {"lead_status": False}, {"enabled": False}, {"a_lead": float('nan')}])
def test_braking_closing_and_invalid_input_release_boost_immediately(level, blocked):
  state = LeadAccelResponseState()
  for _ in range(20):
    state.update(mpc_request(level), .05, 10)
  result = state.update(mpc_request(level, **blocked), .05, 10)
  assert not result.active
  assert result.a_change_cost_factor == 1.
  assert result.jerk_cost_factor == 1.
  assert state.strength == 0.


def test_maximum_response_has_no_fade_or_entry_delay():
  state = LeadAccelResponseState()
  request = mpc_request(5, a_lead=.101, v_rel=.1, gap_margin=.001)
  assert request.active
  assert request.a_change_cost_factor == .05
  assert request.jerk_cost_factor == .15
  assert state.update(request, .05, 10) is request
  assert state.update(request, .05, 11) is request


def test_mild_response_does_not_latch_strong_cost_near_a_deadband():
  state = LeadAccelResponseState()
  factors = [state.update(mpc_request(3, a_lead=a, a_ego=.1, v_rel=.2), .05, 10).a_change_cost_factor
             for a in [.19, .21] * 50]
  assert min(factors) > .98
  assert max(factors) == 1.


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_response_ramp_uses_elapsed_time_across_planner_rates(level):
  strengths = []
  for dt in [.01, .025, .05]:
    state = LeadAccelResponseState()
    for _ in range(round(.1 / dt)):
      result = state.update(mpc_request(level), dt, 10)
    strengths.append(result.strength)
  assert strengths == pytest.approx([strengths[0]] * 3)


# 2026-11-11/12 closed-loop replay: the gap kept opening (vRel +0.4..0.7 m/s)
# after the hard braking ended, the ego still at -0.2..-0.45 m/s^2 and the lead
# coasting (aLeadK -0.3..0). No release applied and a > +0.2 came 2-4 s later.
@pytest.mark.parametrize("level", range(1, 5))
@pytest.mark.parametrize("source", ['lead0', 'cruise'])
def test_gap_opening_while_still_decelerating_releases(level, source):
  if source == 'cruise' and level < 3:
    pytest.skip("gentle levels never use the cruise source")
  for a_ego in (-0.45, -0.2, -0.05):
    result = mpc_request(level, source=source, a_lead=-0.3, a_ego=a_ego, v_rel=0.6, gap_margin=2.0)
    assert result.active and result.opening
    assert result.a_change_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_A_CHANGE_COST_FACTOR))
    assert result.jerk_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_JERK_COST_FACTOR))


@pytest.mark.parametrize("level", range(1, 5))
def test_opening_gap_uses_predicted_margin(level):
  # Still 0.5 m short of the configured TF, but +0.6 m/s opens it within 1.5 s.
  assert mpc_request(level, a_lead=-0.3, a_ego=-0.3, v_rel=0.6, gap_margin=-0.5).active
  assert mpc_request(level, a_lead=-0.69, a_ego=-2.12, v_rel=0.64, gap_margin=-0.5).active
  # Short gap that the opening speed cannot recover within the prediction time.
  short = -0.6 * OPENING_PREDICTION_S - 0.01
  assert not mpc_request(level, a_lead=-0.3, a_ego=-0.3, v_rel=0.6, gap_margin=short).active
  # A positive-lead response without an opening gap keeps the current-margin gate.
  assert not mpc_request(level, a_lead=1.1, a_ego=0.0, v_rel=0.2, gap_margin=-0.01).active


@pytest.mark.parametrize("blocked", [
  {"a_ego": 0.3},                   # past the coasting acceleration limit
  {"v_rel": 0.3},                   # gap not opening
  {"a_lead": -0.5},                 # lead brakes: never release
  {"a_lead": -1.2, "a_ego": -0.3},  # lead brakes harder than we do
])
@pytest.mark.parametrize("level", range(1, 5))
def test_coasting_opening_release_gates(level, blocked):
  kwargs = dict(source='lead0', a_lead=-0.3, a_ego=-0.3, v_rel=0.6, gap_margin=2.0)
  kwargs.update(blocked)
  assert not mpc_request(level, **kwargs).active


def test_level_five_does_not_use_coasting_opening():
  assert not mpc_request(5, a_lead=-0.3, a_ego=-0.3, v_rel=0.6, gap_margin=2.0).active
  assert not mpc_request(5, a_lead=1.1, a_ego=0.0, v_rel=0.6, gap_margin=-0.5).active


def test_coasting_opening_covers_mild_positive_acceleration():
  assert OPENING_COAST_ACCEL_MAX == pytest.approx(0.3)
  assert lead_gap_opening_while_coasting(a_ego=0.2, v_rel=0.6, a_lead=-0.3)
  assert not lead_gap_opening_while_coasting(a_ego=0.3, v_rel=0.6, a_lead=-0.3)


# 2026-11-11/12 closed-loop replay: ending the release at aEgo 0 put the level's
# jerk cost back on the 0 -> +0.2 rise; the release now continues to +0.3.
def test_opening_release_continues_while_accelerating_mildly():
  result = mpc_request(4, a_lead=0.0, a_ego=0.1, v_rel=0.6, gap_margin=3.0)
  assert result.active and result.opening
  assert result.a_change_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_A_CHANGE_COST_FACTOR))
  assert result.jerk_cost_factor == pytest.approx(1.0 - result.strength * (1.0 - OPENING_JERK_COST_FACTOR))
  assert not mpc_request(5, a_lead=0.0, a_ego=0.1, v_rel=0.6, gap_margin=3.0).opening
