import pytest

from openpilot.selfdrive.controls.lib.longitudinal_stop_gap import (
  STOP_GAP_EGO_SPEEDS,
  STOP_GAP_MAX_EXCESS,
  STOP_GAP_OBSTACLE_COST_FACTOR,
  STOP_GAP_RISE_TIME,
  StopGapWeightState,
  stop_gap_target_factor,
)


def target(**kw):
  args = dict(lead_status=True, lead_end_speed=0.0, lead_sets_stop=True, v_ego=3.0, gap_excess=0.5)
  args.update(kw)
  return stop_gap_target_factor(**args)


def test_stopping_lead_at_low_speed_gets_full_gap_weight():
  assert target() == pytest.approx(STOP_GAP_OBSTACLE_COST_FACTOR)
  assert target(v_ego=STOP_GAP_EGO_SPEEDS[0]) == pytest.approx(STOP_GAP_OBSTACLE_COST_FACTOR)
  mid = target(v_ego=sum(STOP_GAP_EGO_SPEEDS) / 2)
  assert 1.0 < mid < STOP_GAP_OBSTACLE_COST_FACTOR


@pytest.mark.parametrize("kw", [
  {"lead_status": False},
  {"lead_end_speed": 1.0},                 # lead keeps moving in the plan
  {"lead_sets_stop": False},               # cruise or traffic stop sets the end
  {"v_ego": STOP_GAP_EGO_SPEEDS[1]},       # following at speed: costs unchanged
  {"gap_excess": STOP_GAP_MAX_EXCESS + 0.01},  # still far behind: never pull toward the lead
  {"gap_excess": float('inf')},
  {"v_ego": float('nan')},
  {"lead_end_speed": float('nan')},
])
def test_other_cases_keep_normal_gap_weight(kw):
  assert target(**kw) == 1.0


def test_entry_ramps_and_exit_is_immediate():
  state = StopGapWeightState()
  args = dict(enabled=True, lead_status=True, lead_end_speed=0.0, lead_sets_stop=True, v_ego=3.0, gap_excess=0.0, dt=0.05)
  first = state.update(**args)
  assert 1.0 < first < STOP_GAP_OBSTACLE_COST_FACTOR
  for _ in range(round(STOP_GAP_RISE_TIME / 0.05)):
    factor = state.update(**args)
  assert factor == pytest.approx(STOP_GAP_OBSTACLE_COST_FACTOR)
  assert state.update(**{**args, 'lead_end_speed': 2.0}) == 1.0
  assert state.update(**{**args, 'enabled': False}) == 1.0
  state.update(**args)
  assert state.update(**{**args, 'gap_excess': 10.0}) == 1.0
