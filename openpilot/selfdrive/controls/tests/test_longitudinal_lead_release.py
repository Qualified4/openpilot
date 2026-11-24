import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.radar_motion.lead_dynamics import LeadAccelTau
from openpilot.selfdrive.controls.lib import longitudinal_lead_release as release
from openpilot.selfdrive.controls.lib.longitudinal_lead_release import LeadBrakeReleaseState, release_cost_ease
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, T_IDXS

DT = 0.05


def make_lead(**kw):
  lead = {'status': True, 'radar': True, 'radarTrackId': 7, 'modelProb': 1.0, 'dRel': 10.0, 'vLead': 10.0, 'vRel': 0.5,
          'aLeadK': -2.0, 'aLeadTau': 0.05, 'jLead': 0.0}
  lead.update(kw)
  return SimpleNamespace(**lead)


class SyntheticLead:
  """True lead accel profile; aLeadK through a 0.25 s lag, jLead its derivative, aLeadTau from the real class."""
  def __init__(self, x, v, profile):
    self.x, self.v, self.a, self.ak, self.jk = x, v, 0.0, 0.0, 0.0
    self.profile = profile
    self.tau = LeadAccelTau()

  def step(self, t):
    self.a = self.profile(t) if self.v > 0.0 else 0.0
    v_new = max(0.0, self.v + self.a * DT)
    self.x += (self.v + v_new) / 2 * DT
    self.v = v_new
    ak = self.ak + (self.a - self.ak) * DT / 0.25
    self.jk, self.ak = (ak - self.ak) / DT, ak
    return self.tau.update(self.ak, self.jk, t)


def brake_profile(start, accel, hold, rebrake=None, rebrake_accel=None):
  def profile(t):
    tb = t - start
    if tb < 0.0:
      return 0.0
    a = max(accel, -10.0 * tb)
    if tb > hold:
      a = min(0.0, accel + 10.0 * (tb - hold))
    if rebrake is not None and tb > rebrake:
      a = max(rebrake_accel, -10.0 * (tb - rebrake))
    return a
  return profile


def simulate(enabled, profile, v0=50 / 3.6, duration=12.0, track_id=7):
  carrot = SimpleNamespace(leadAccelResponse=0, jerk_factor=1.0, comfort_brake=2.4, stop_distance=3.7, mode='acc',
                           v_cruise=v0 + 3.0, stop_dist=1000.0, trafficStopDistanceAdjust=0.0, lane_change_active=False,
                           get_T_FOLLOW=lambda *args, **kwargs: 0.57)
  gap = 0.57 * v0 + 3.7 + v0 ** 2 / 4.8 - v0 ** 2 / 5.0
  lead = SyntheticLead(gap, v0, profile)
  mpc = LongitudinalMpc()
  x, v, a, a_plan, cmd = 0.0, v0, 0.0, 0.0, []
  rows = []
  zeros = [np.zeros(len(T_IDXS)) for _ in range(4)]
  for i in range(int(duration / DT)):
    t = i * DT
    tau = lead.step(t)
    rs = SimpleNamespace(
      leadOne=make_lead(radarTrackId=track_id, dRel=lead.x - x, vLead=lead.v, vRel=lead.v - v, aLeadK=lead.ak,
                        aLeadTau=tau, jLead=lead.jk),
      leadTwo=make_lead(status=False, radar=False, radarTrackId=-1))
    mpc.set_accel_limits(-1.2, 2.0)
    mpc.set_cur_state(v, a_plan)
    mpc.update(carrot, False, rs, carrot.v_cruise, *zeros, lead_brake_release_enabled=enabled)
    # like the planner: the MPC state follows the plan, the command looks ahead by the actuator delay
    a_plan = float(np.interp(DT, T_IDXS, mpc.a_solution))
    a_target = float(np.interp(0.3, T_IDXS, mpc.a_solution))
    cmd.append(a_target)
    # ego: 0.1 s delay then 0.25 s lag
    a_cmd = cmd[-3] if len(cmd) >= 3 else 0.0
    a += (a_cmd - a) * DT / 0.25
    v_new = max(0.0, v + a * DT)
    x += (v + v_new) / 2 * DT
    v = v_new
    rows.append((t, lead.x - x, lead.v, v, a_target, mpc.lead_release_states[0].easing, mpc.lead_release_cost_eased))
  return np.array(rows, dtype=float), mpc


RELEASE = brake_profile(2.0, -3.0, 1.3)


def under_speed(rows):
  lead_v, ego_v = rows[:, 2], rows[:, 3]
  moving = lead_v > 2.0
  return float(np.max(lead_v[moving] - ego_v[moving]))


def test_disabled_never_enters_the_release_branch():
  off, mpc = simulate(False, RELEASE)
  assert not off[:, 5].any() and not off[:, 6].any()
  assert all(state.track_id is None for state in mpc.lead_release_states)
  on, _ = simulate(True, RELEASE)
  assert on[:, 6].any()
  assert not np.array_equal(off[:, 4], on[:, 4])


def test_disabled_prediction_is_the_raw_lead_prediction():
  mpc = LongitudinalMpc()
  mpc.set_cur_state(10.0, -2.0)
  lead = make_lead(aLeadK=-1.5, jLead=3.0, aLeadTau=0.01)
  xv, _ = mpc.process_lead(lead)
  assert np.array_equal(xv, mpc.extrapolate_lead(lead.dRel, lead.vLead, lead.aLeadK, lead.aLeadTau))


def test_release_reduces_under_speed_after_lead_lets_go():
  off, _ = simulate(False, RELEASE)
  on, _ = simulate(True, RELEASE)
  assert under_speed(on) < under_speed(off) - 0.5
  assert on[:, 1].min() > 2.0


@pytest.mark.parametrize('rebrake_after', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('rebrake_accel', [-3.0, -6.0])
def test_rebraking_returns_to_original_prediction(rebrake_after, rebrake_accel):
  # lead lets go (ramp to 0 takes 0.3 s) and brakes again shortly after
  profile = brake_profile(2.0, -3.0, 1.3, rebrake=1.3 + 0.3 + rebrake_after, rebrake_accel=rebrake_accel)
  off, _ = simulate(False, profile)
  on, _ = simulate(True, profile)
  t_rebrake = 2.0 + 1.3 + 0.3 + rebrake_after
  # once renewed braking shows in the measurement the original prediction is back
  late = on[on[:, 0] > t_rebrake + 0.4]
  assert not late[:, 5].any()
  assert on[:, 1].min() >= off[:, 1].min() - 0.05


def test_rebrake_jerk_restarts_the_episode():
  state = LeadBrakeReleaseState()
  assert state.prediction(make_lead(aLeadK=-3.0, jLead=-5.0), -3.0, 0.01, DT) == (-3.0, 0.01)
  a, tau = state.prediction(make_lead(aLeadK=-2.0, jLead=2.0), -2.0, 0.01, DT)
  assert state.easing and tau == release.RELEASE_TAU and a == pytest.approx(-1.0)
  assert state.prediction(make_lead(aLeadK=-2.2, jLead=-2.0), -2.2, 0.01, DT) == (-2.2, 0.01)
  assert not state.easing
  # steady braking at the new level is not an easing
  assert state.prediction(make_lead(aLeadK=-2.2, jLead=0.0), -2.2, 0.01, DT) == (-2.2, 0.01)


@pytest.mark.parametrize('change', [
  {'radarTrackId': 8},
  {'aLeadK': math.nan}, {'jLead': math.nan}, {'dRel': math.inf}, {'aLeadTau': math.nan},
  {'status': False}, {'radar': False}, {'radarTrackId': -1}, {'modelProb': 0.2}, {'dRel': 2.0},
])
def test_lead_change_or_invalid_values_reset_the_state(change):
  state = LeadBrakeReleaseState()
  state.prediction(make_lead(aLeadK=-3.0, jLead=-5.0), -3.0, 0.01, DT)
  state.prediction(make_lead(aLeadK=-2.4, jLead=0.0), -2.4, 0.01, DT)
  assert state.easing and state.a_min == -3.0
  lead = make_lead(**{'aLeadK': -2.4, 'jLead': 0.0, **change})
  a, tau = state.prediction(lead, lead.aLeadK, lead.aLeadTau, DT)
  assert not state.easing
  assert (a, tau) == (lead.aLeadK, lead.aLeadTau) or (math.isnan(a) or math.isnan(tau))
  if 'radarTrackId' in change and change['radarTrackId'] >= 0:
    assert state.track_id == 8 and state.a_min == -2.4
  else:
    assert state.track_id is None and state.a_min == 0.0


def test_far_lead_keeps_the_original_prediction():
  state = LeadBrakeReleaseState()
  lead = make_lead(dRel=30.0, vLead=10.0, vRel=0.5, aLeadK=-2.0, jLead=3.0)
  assert state.prediction(lead, -2.0, 0.01, DT) == (-2.0, 0.01)


def test_no_release_while_ego_closes_on_the_lead():
  state = LeadBrakeReleaseState()
  state.prediction(make_lead(aLeadK=-3.0, jLead=-5.0), -3.0, 0.01, DT)
  assert state.prediction(make_lead(aLeadK=-2.0, jLead=2.0, vRel=-0.5), -2.0, 0.01, DT) == (-2.0, 0.01)
  assert not release_cost_ease(make_lead(aLeadK=-0.4, jLead=0.0, vRel=-1.0), -2.0, 1.0)
  assert release_cost_ease(make_lead(aLeadK=-0.4, jLead=0.0, vRel=-0.4), -2.0, 1.0)


def test_rebrake_after_release_keeps_eased_costs_briefly():
  state = LeadBrakeReleaseState()
  state.prediction(make_lead(aLeadK=-3.0, jLead=-5.0), -3.0, 0.01, DT)
  state.prediction(make_lead(aLeadK=-2.0, jLead=2.0), -2.0, 0.01, DT)
  assert state.easing and state.rebrake_boost == 0.0
  state.prediction(make_lead(aLeadK=-2.2, jLead=-3.0), -2.2, 0.01, DT)
  assert not state.easing and state.rebrake_boost == release.REBRAKE_BOOST_S
  for _ in range(int(release.REBRAKE_BOOST_S / DT) + 1):
    state.prediction(make_lead(aLeadK=-2.2, jLead=0.0), -2.2, 0.01, DT)
  assert state.rebrake_boost == 0.0


def test_cost_ease_needs_easing_lead_harder_ego_braking_and_margin():
  lead = make_lead(aLeadK=-0.4, jLead=0.0)
  assert release_cost_ease(lead, -2.0, 1.0)
  assert not release_cost_ease(lead, -0.8, 1.0)
  assert not release_cost_ease(make_lead(aLeadK=-1.2, jLead=0.0), -1.5, 1.0)
  assert release_cost_ease(make_lead(aLeadK=-1.2, jLead=1.5), -2.0, 1.0)
  assert not release_cost_ease(lead, -2.0, -0.1)
  assert not release_cost_ease(make_lead(aLeadK=math.nan), -2.0, 1.0)


def test_mpc_reset_states_on_long_control_off_and_driver_off():
  _, mpc = simulate(True, RELEASE, duration=4.0)
  assert mpc.lead_release_states[0].track_id == 7
  carrot = SimpleNamespace(leadAccelResponse=0, jerk_factor=1.0, comfort_brake=2.4, stop_distance=3.7, mode='acc',
                           v_cruise=15.0, stop_dist=1000.0, trafficStopDistanceAdjust=0.0, lane_change_active=False,
                           get_T_FOLLOW=lambda *args, **kwargs: 0.57)
  rs = SimpleNamespace(leadOne=make_lead(), leadTwo=make_lead(status=False, radar=False, radarTrackId=-1))
  zeros = [np.zeros(len(T_IDXS)) for _ in range(4)]
  mpc.update(carrot, True, rs, 15.0, *zeros, lead_brake_release_enabled=True)
  assert mpc.lead_release_states[0].track_id is None and not mpc.lead_release_cost_eased
