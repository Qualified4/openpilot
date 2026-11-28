import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  A_CHANGE_COST,
  get_a_change_cost,
  low_speed_lead_creep_suppress,
)


def test_active_accel_change_cost_is_fixed():
  assert get_a_change_cost(True, 0.0) == A_CHANGE_COST
  assert get_a_change_cost(True, 500.0) == A_CHANGE_COST


def test_starting_cost_is_preserved_only_without_previous_accel_constraint():
  assert get_a_change_cost(False, 10.0) == 10.0


def test_lead_response_scales_cost_inside_mpc():
  assert get_a_change_cost(True, 500.0, 0.40) == 80.0
  assert get_a_change_cost(True, 500.0, 0.18) == 36.0
  assert get_a_change_cost(True, 500.0, 0.05) == 10.0


def test_response_factor_cannot_raise_or_invert_cost():
  assert get_a_change_cost(True, 10.0, 2.0) == A_CHANGE_COST
  assert get_a_change_cost(True, 10.0, -1.0) == 0.0


def test_low_speed_lead_creep_suppress_488_seg80_and_launch_safety():
  # 488 seg 80 creeping lead: v_ego=1.1 m/s, d_rel=5.65 m (d_stop_cfg=3.5), v_lead=1.42 m/s, a_lead=+0.45 -> full suppression.
  assert low_speed_lead_creep_suppress(1.1, 5.65, 1.42, 0.45, d_stop_cfg=3.5) == pytest.approx(1.0)

  # Standstill launch (486 / 480-7): v_ego <= 0.35 m/s -> 0.0 suppression (zero launch delay!).
  assert low_speed_lead_creep_suppress(0.0, 4.0, 1.2, 1.5, d_stop_cfg=3.5) == 0.0
  assert low_speed_lead_creep_suppress(0.35, 4.5, 1.0, 0.8, d_stop_cfg=3.5) == 0.0

  # Genuine lead departure: d_rel >= d_stop_cfg + 3.6 m OR v_lead >= 2.4 m/s -> 0.0 suppression.
  assert low_speed_lead_creep_suppress(1.1, 7.1, 1.35, 0.45, d_stop_cfg=3.5) == 0.0
  assert low_speed_lead_creep_suppress(1.1, 5.65, 2.4, 0.45, d_stop_cfg=3.5) == 0.0

  # Faster ego speed (>= 2.6 m/s) or braking lead (a_lead <= 0) -> 0.0 suppression.
  assert low_speed_lead_creep_suppress(2.6, 5.65, 1.4, 0.45, d_stop_cfg=3.5) == 0.0
  assert low_speed_lead_creep_suppress(1.1, 5.65, 1.35, -0.3, d_stop_cfg=3.5) == 0.0
  assert low_speed_lead_creep_suppress(1.1, 5.65, 1.35, 0.0, d_stop_cfg=3.5) == 0.0
