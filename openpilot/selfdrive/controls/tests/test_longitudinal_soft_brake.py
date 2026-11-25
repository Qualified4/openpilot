import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_gap_safe import GapSafeState
from openpilot.selfdrive.controls.lib.longitudinal_soft_brake import SoftBrakeState, soft_brake_reference_delta, soft_brake_weight


@pytest.mark.parametrize("overrides", [
  {"distance": 14.0},
  {"distance": 14.3, "ego_speed": 8.0},
  {"ego_speed": 16.0},
  {"lead_speed": 2.9},
  {"lead_accel": -2.5},
  {"lead_accel": 0.1},
  {"lead_end_speed": 1.0},
  {"planned_decel": 1.0},
  {"planned_decel": 3.5},
  {"distance": float("nan")},
])
def test_soft_brake_requires_a_moving_lead_and_comfort_reserve(overrides):
  gap = GapSafeState()
  gap.update(enabled=True, lead_decel=4.25, snap=True, dt=0.05)
  args = {
    "distance": 27.0, "ego_speed": 10.0, "lead_speed": 7.0, "lead_accel": -1.3, "lead_end_speed": 0.0,
    "planned_decel": 1.9, "gap": gap.lead_gap(comfort_brake=2.5), "stop_distance": 3.7,
  }
  assert soft_brake_weight(**args) == pytest.approx(1.0)
  assert soft_brake_weight(**(args | overrides)) == 0.0


def test_no_braking_relief_from_the_reference_when_predicted_clearance_is_gone():
  times = np.arange(6) * 0.5
  previous = np.array([0.1, -0.4, -1.4, -2.4, -2.5, -2.0])
  full = soft_brake_reference_delta(times, previous, weight=1.0, closing=3.0, danger_margin=1.5)
  close = soft_brake_reference_delta(times, previous, weight=1.0, closing=3.0, danger_margin=0.0)
  assert full[0] < 0.0
  assert max(full) > 0.0
  np.testing.assert_array_equal(close, np.minimum(full, 0.0))


def test_hard_braking_episode_and_lead_replacement_keep_normal_response():
  state = SoftBrakeState()
  state.update(7, reset=False, lead_accel=-1.0, ego_accel=-0.4, closing=2.0, time_gap=2.5)
  assert not state.blocked
  state.update(7, reset=False, lead_accel=-2.5, ego_accel=-0.4, closing=2.0, time_gap=2.5)
  state.update(7, reset=False, lead_accel=-1.0, ego_accel=-2.0, closing=2.0, time_gap=2.5)
  assert state.blocked
  state.update(8, reset=False, lead_accel=-0.5, ego_accel=-2.0, closing=3.0, time_gap=2.5)
  state.update(8, reset=False, lead_accel=-0.2, ego_accel=-0.5, closing=2.0, time_gap=2.5)
  assert state.blocked
  state.update(8, reset=False, lead_accel=0.0, ego_accel=-0.1, closing=0.1, time_gap=2.5)
  assert not state.blocked
  state.update(8, reset=False, lead_accel=-1.1, ego_accel=-2.0, closing=3.0, time_gap=2.5)
  state.update(8, reset=False, lead_accel=-1.0, ego_accel=-1.5, closing=2.0, time_gap=2.5)
  assert state.blocked
  state.update(None, reset=True, lead_accel=0.0, ego_accel=0.0, closing=0.0, time_gap=float("inf"))
  assert not state.blocked


def test_lead_revealed_after_a_close_gap_gets_normal_braking():
  state = SoftBrakeState()
  state.update(7, reset=False, lead_accel=0.3, ego_accel=0.0, closing=0.1, time_gap=1.0)
  state.update(None, reset=False, lead_accel=0.0, ego_accel=0.0, closing=0.1, time_gap=float("inf"))
  state.update(8, reset=False, lead_accel=-0.5, ego_accel=0.0, closing=2.0, time_gap=2.5)
  state.update(8, reset=False, lead_accel=-1.1, ego_accel=0.0, closing=2.0, time_gap=2.5)
  assert state.blocked


def test_braking_that_started_far_back_stays_on_the_normal_path():
  state = SoftBrakeState()
  state.update(7, reset=False, lead_accel=0.0, ego_accel=0.0, closing=0.1, time_gap=3.5)
  state.update(7, reset=False, lead_accel=-1.1, ego_accel=0.0, closing=1.5, time_gap=3.5)
  state.update(7, reset=False, lead_accel=-1.1, ego_accel=-1.0, closing=2.0, time_gap=2.8)
  assert state.blocked
  state.update(7, reset=False, lead_accel=0.0, ego_accel=-0.4, closing=0.1, time_gap=1.9)
  assert state.blocked
  state.update(7, reset=False, lead_accel=0.0, ego_accel=0.0, closing=0.1, time_gap=2.2)
  assert not state.blocked
