import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import car, custom, log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib import corner_center as cc_mod
from openpilot.selfdrive.controls.lib.corner_center import CornerCenter, CornerCenterOutput, block_names, corner_center_curvature

T = np.array(ModelConstants.T_IDXS)
X = np.array(ModelConstants.X_IDXS)
DT = 0.05


def scenario(k=0.002, bias=0.3, v=20.0, width=3.5, probs=(0.9, 0.9, 0.9, 0.9), stds=(0.05, 0.05, 0.05, 0.05),
             lane_slope=0.0):
  """Constant-curvature corner (k > 0 turns right, y positive right); the model path sits `bias` m right of the lane center."""
  md = log.ModelDataV2.new_message()
  px = v * T
  pb = bias * np.clip(px / 10.0, 0.0, 1.0)
  md.position.x = px.tolist()
  md.position.y = (0.5 * k * px ** 2 + pb).tolist()
  md.position.z = [0.0] * 33
  md.position.t = T.tolist()
  md.position.yStd = [0.1] * 33
  md.velocity.x = [v] * 33
  md.velocity.y = [0.0] * 33
  md.velocity.z = [0.0] * 33
  md.orientation.x = [0.0] * 33
  md.orientation.z = (k * px).tolist()
  md.orientationRate.x = [0.0] * 33
  md.orientationRate.z = [v * k] * 33
  md.acceleration.x = [0.0] * 33
  md.acceleration.y = [0.0] * 33
  center = 0.5 * k * X ** 2
  half = width / 2 + 0.5 * lane_slope * X
  lanes = md.init('laneLines', 4)
  for i, off in enumerate((-half - width, -half, half, half + width)):
    lanes[i].x = X.tolist()
    lanes[i].y = (center + off).tolist()
    lanes[i].z = [0.0] * 33
    lanes[i].t = T.tolist()
  md.laneLineProbs = list(probs)
  md.laneLineStds = list(stds)
  edges = md.init('roadEdges', 2)
  for i, off in enumerate((-half - 2 * width, half + 2 * width)):
    edges[i].x = X.tolist()
    edges[i].y = (center + off).tolist()
    edges[i].z = [0.0] * 33
    edges[i].t = T.tolist()
  md.roadEdgeStds = [0.3, 0.3]
  md.meta.desireState = [1.0] + [0.0] * 7
  cs = car.CarState.new_message(vEgo=v)
  rs = log.RadarState.new_message()
  return {'md': md, 'cs': cs, 'rs': rs, 'meas': k, 'lat_active': True, 'atc': False}


def mirror(s):
  """Left-right mirror of a scenario: y and yaw negate, left and right swap."""
  o = s['md']
  m = o.as_reader().as_builder()
  for name in ('position', 'velocity', 'acceleration'):
    getattr(m, name).y = [-y for y in getattr(o, name).y]
  for name in ('orientation', 'orientationRate'):
    getattr(m, name).z = [-z for z in getattr(o, name).z]
  for i in range(4):
    src = o.laneLines[3 - i]
    m.laneLines[i].x = list(src.x)
    m.laneLines[i].y = [-y for y in src.y]
  m.laneLineProbs = list(o.laneLineProbs)[::-1]
  m.laneLineStds = list(o.laneLineStds)[::-1]
  cs = s['cs'].as_reader().as_builder()
  cs.leftBlinker, cs.rightBlinker = s['cs'].rightBlinker, s['cs'].leftBlinker
  cs.leftBlindspot, cs.rightBlindspot = s['cs'].rightBlindspot, s['cs'].leftBlindspot
  rs = log.RadarState.new_message()
  flip = lambda leads: [dict(x.to_dict(), dPath=-x.dPath, yRel=-x.yRel) for x in leads]  # noqa: E731
  rs.leadsLeft, rs.leadsRight = flip(s['rs'].leadsRight), flip(s['rs'].leadsLeft)
  return dict(s, md=m, cs=cs, rs=rs, meas=-s['meas'])


def step(c, s, t, mpc_ok=True, lane_mode=False):
  md = s['md']
  path = np.column_stack([md.position.x, md.position.y, md.position.z])
  v = float(md.velocity.x[0])
  return c.update(t, md, s['cs'], s['rs'], s['lat_active'], s['atc'], s['meas'], 0.15,
                  path, np.full(33, v), T, v, mpc_ok, lane_mode)


def step_bound(v, j=cc_mod.J_RAMP, dt=DT):
  return j / max(v, 8.0) ** 2 * dt * (1 + 1e-9)


def assert_rate_limited(dks, v=20.0, j=cc_mod.J_RAMP):
  assert np.all(np.abs(np.diff([0.0] + list(dks))) <= step_bound(v, j))


def run(s, frames=40, blend=50, c=None, t0=0.0):
  c = c or CornerCenter(blend)
  c.set_delay_params(0, 13)
  out = []
  for i in range(frames):
    dk = step(c, s, t0 + i * DT)
    out.append((dk, c.block, c.gate_value))
  return c, out


def test_base_corner_activates_outward_and_ramps():
  _, out = run(scenario())
  dks = [dk for dk, _, _ in out]
  # startup 0.5 s (10 frames), then 3 frames before the gate opens
  first = next(i for i, dk in enumerate(dks) if dk != 0.0)
  assert first == 12
  assert all(dk < 0.0 for dk in dks[first:])  # right corner, inside bias right → correction to the left
  assert abs(dks[first + 1]) > abs(dks[first])  # no step to the full value
  assert_rate_limited(dks)  # turning on stays within J_RAMP
  assert out[-1][1] == 0 and out[-1][2] == pytest.approx(1.0)
  assert max(abs(d) for d in dks) <= cc_mod.DK_LAT_ACC_MAX / 20.0 ** 2


def test_rate_reaches_ninety_percent_in_two_seconds():
  assert 1 - (1 - cc_mod.RATE) ** round(2.0 / DT) == pytest.approx(0.9, abs=0.01)


def _far_points(kind):
  s = scenario()
  px = np.array(s['md'].position.x)
  first = int(np.flatnonzero(px > cc_mod.USE_X)[0])
  tail = px[first + 1:]
  if kind == 'stalled':      # stopping plan: x stops growing beyond 60 m
    px[first + 1:] = np.maximum(px[first], 60.0) - 0.01 * np.arange(len(tail))
  elif kind == 'backward':   # runs back inside 55 m after the first point beyond it
    px[first + 1:] = 45.0 - 0.1 * np.arange(len(tail))
  elif kind == 'nan':
    px[first + 1:] = math.nan
  s['md'].position.x = px.tolist()
  return s


@pytest.mark.parametrize('kind', ['stalled', 'backward', 'nan'])
def test_far_path_points_are_not_read(kind):
  _, ref = run(scenario(), 80)
  _, out = run(_far_points(kind), 80)
  assert ref[-1][0] != 0.0
  assert all('input' not in block_names(m) for _, m, _ in out)
  assert [dk for dk, _, _ in out] == pytest.approx([dk for dk, _, _ in ref], rel=1e-6, abs=1e-12)


def test_non_monotonic_path_inside_use_range_still_fails_input():
  s = scenario()
  px = list(s['md'].position.x)
  px[10] = px[9] - 0.1  # x ≈ 2 m
  s['md'].position.x = px
  _, out = run(s)
  assert all(dk == 0.0 for dk, _, _ in out) and 'input' in block_names(out[-1][1])


def test_one_input_failure_frame_ramps_within_bound():
  s = scenario()
  c, out = run(s, 80)
  assert out[-1][0] != 0.0
  good = list(s['md'].position.x)
  bad = list(good)
  bad[12] = bad[11]  # not increasing at about 5 m
  dks = [out[-1][0]]
  for i in range(60):
    s['md'].position.x = bad if i == 0 else good
    dks.append(step(c, s, (80 + i) * DT))
    if i == 0:
      assert 'input' in block_names(c.block) and not c.dy.any() and dks[-1] != 0.0
  assert np.all(np.abs(np.diff(dks)) <= step_bound(20.0))


def test_mirror_symmetry_is_exact():
  s = scenario(probs=(0.9, 0.8, 0.95, 0.9), stds=(0.05, 0.12, 0.25, 0.05))
  s['rs'].leadsRight = [log.RadarState.LeadData.new_message(dRel=10, vLead=20, dPath=-3.5, yRel=-3.5)]  # inside side: no block
  _, a = run(s)
  _, b = run(mirror(s))
  assert any(dk != 0.0 for dk, _, _ in a)
  for (dka, ma, ga), (dkb, mb, gb) in zip(a, b, strict=True):
    assert dkb == -dka and ma == mb and ga == gb


def test_zero_offset_gives_zero_curvature():
  _, out = run(scenario(bias=0.0))
  assert all(dk == 0.0 for dk, _, _ in out)
  assert 'inward' in block_names(out[-1][1])
  # G1 with Δy = 0 is exactly 0: force the gate open and a zero target
  c = CornerCenter(50)
  s = scenario(bias=0.0)
  c.gate.update = lambda *a: (1.0, 1.0, [])
  for i in range(5):
    assert step(c, s, i * DT) == 0.0


def _blocked(s, reason, frames=40):
  _, out = run(s, frames)
  assert all(dk == 0.0 for dk, _, _ in out), reason
  assert reason in block_names(out[-1][1])


@pytest.mark.parametrize('reason,kw', [
  ('prob', {'probs': (0.9, 0.4, 0.9, 0.9)}),
  ('std', {'stds': (0.05, 0.45, 0.05, 0.05)}),
  ('width', {'width': 2.2}),
  ('merge', {'lane_slope': 0.04}),
  ('follow', {'bias': 0.9}),
  ('corner', {'k': 0.0005}),
  ('kappa', {'k': 0.00045, 'v': 33.0}),   # 0.49 m/s² but κ below the 0.00055 floor (105 km/h gentle bend)
  ('speed', {'v': 7.0, 'k': 0.012}),
])
def test_condition_blocks(reason, kw):
  _blocked(scenario(**kw), reason)


def test_std_fades_continuously():
  _, a = run(scenario())
  _, b = run(scenario(stds=(0.05, 0.30, 0.05, 0.05)))
  assert b[-1][2] == pytest.approx(0.5)
  assert abs(b[-1][0]) < abs(a[-1][0])


def test_measured_lateral_acceleration_must_match_direction():
  s = scenario()
  s['meas'] = 0.0
  _blocked(s, 'meas')
  s['meas'] = -0.002
  _blocked(s, 'meas')


@pytest.mark.parametrize('field', ['leftBlinker', 'steeringPressed'])
def test_lane_change_and_driver_ramp_out_then_rest(field):
  driver = field == 'steeringPressed'
  rest, reason, j = (cc_mod.DRIVER_REST_S, 'driver', cc_mod.J_DRIVER) if driver else (cc_mod.LANE_CHANGE_REST_S, 'lanechange', cc_mod.J_RAMP)
  s = scenario()
  c, out = run(s, 80)
  start = out[-1][0]
  assert start != 0.0
  t = 80 * DT
  setattr(s['cs'], field, True)
  dks = [start, step(c, s, t)]
  assert not c.dy.any() and abs(dks[-1]) == pytest.approx(abs(start) - step_bound(20.0, j), rel=1e-6)
  setattr(s['cs'], field, False)
  n_rest = round(rest / DT)
  for i in range(1, n_rest):
    dks.append(step(c, s, t + i * DT))
    assert reason in block_names(c.block)
  assert np.all(np.abs(np.diff(dks)) <= step_bound(20.0, j))
  assert all(abs(b) <= abs(a) for a, b in zip(dks, dks[1:], strict=False))
  if driver:
    n0 = math.ceil(abs(start) / step_bound(20.0, j))
    assert n0 < len(dks) and all(dk == 0.0 for dk in dks[n0:])  # at 0 after the J_DRIVER ramp, within the rest
    assert abs(start) > 0.5 * cc_mod.DK_LAT_ACC_MAX / 20.0 ** 2
  assert dks[-1] == 0.0
  dks = [step(c, s, t + i * DT) for i in range(n_rest, n_rest + 5)]
  first = next(i for i, dk in enumerate(dks) if dk != 0.0)
  assert first in (2, 3)  # rest over (float boundary), then the 3-frame on rule
  assert abs(dks[-1]) <= 3 * step_bound(20.0)


def test_atc_and_desire_block():
  s = scenario()
  s['atc'] = True
  _blocked(s, 'lanechange')
  s = scenario()
  s['md'].meta.desireState = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
  _blocked(s, 'lanechange')


def test_startup_and_three_frames():
  _, out = run(scenario(), frames=13)
  assert [dk != 0.0 for dk, _, _ in out] == [False] * 12 + [True]
  assert all('startup' in block_names(m) for _, m, _ in out[:10])


@pytest.mark.parametrize('k', [0.002, -0.002])
def test_side_radar_uses_corner_outside_by_dpath(k):
  # radarState.leadsLeft holds dPath > 0. A right corner (k > 0) corrects to the left, so only leadsLeft blocks it.
  outside, inside = ('leadsLeft', 'leadsRight') if k > 0 else ('leadsRight', 'leadsLeft')
  sign = 1.0 if outside == 'leadsLeft' else -1.0
  moving = log.RadarState.LeadData.new_message(dRel=8, vLead=20, dPath=3.5 * sign, yRel=-3.5 * sign)
  s = scenario(k=k, bias=0.3 if k > 0 else -0.3)
  setattr(s['rs'], inside, [log.RadarState.LeadData.new_message(dRel=8, vLead=20, dPath=-3.5 * sign, yRel=3.5 * sign)])
  _, out = run(s)
  assert out[-1][0] != 0.0
  setattr(s['rs'], outside, [moving])
  _blocked(s, 'side')
  stopped = log.RadarState.LeadData.new_message(dRel=8, vLead=0.5, dPath=3.5 * sign)
  far = log.RadarState.LeadData.new_message(dRel=40, vLead=20, dPath=3.5 * sign)
  setattr(s['rs'], outside, [stopped, far])
  _, out = run(s)
  assert out[-1][0] != 0.0


@pytest.mark.parametrize('kind', ['side', 'bsd'])
def test_side_traffic_appearing_clears_dy_and_ramps_out(kind):
  s = scenario()
  c, out = run(s, 80)
  start = out[-1][0]
  assert start != 0.0
  if kind == 'side':
    s['rs'].leadsLeft = [log.RadarState.LeadData.new_message(dRel=8, vLead=20, dPath=3.5, yRel=-3.5)]
  else:
    s['cs'].leftBlindspot = True
  dks = [start]
  for i in range(round(0.65 / DT)):
    dks.append(step(c, s, (80 + i) * DT))
    assert kind in block_names(c.block) and not c.dy.any()
  assert np.all(np.abs(np.diff(dks)) <= step_bound(20.0))
  assert dks[1] != 0.0 and dks[-1] == 0.0  # ramp, not a step; at 0 within 0.65 s


def test_bsd_on_outside_blocks_and_holds():
  s = scenario()
  s['cs'].rightBlindspot = True  # inside side
  _, out = run(s)
  assert out[-1][0] != 0.0
  s = scenario()
  s['cs'].leftBlindspot = True
  _blocked(s, 'bsd')


@pytest.mark.parametrize('target', ['laneY', 'laneStd', 'laneProb', 'positionY'])
def test_nan_input_zeroes_that_frame(target):
  s = scenario()
  c, out = run(s)
  assert out[-1][0] != 0.0
  md = s['md']
  saved = {'laneY': list(md.laneLines[1].y), 'laneStd': list(md.laneLineStds), 'laneProb': list(md.laneLineProbs),
           'positionY': list(md.position.y)}[target]
  bad = list(saved)
  bad[5 if target in ('laneY', 'positionY') else 1] = math.nan
  if target == 'laneY':
    md.laneLines[1].y = bad
  elif target == 'laneStd':
    md.laneLineStds = bad
  elif target == 'laneProb':
    md.laneLineProbs = bad
  else:
    md.position.y = bad
  dk = step(c, s, 40 * DT)
  assert not c.dy.any() and 'input' in block_names(c.block)
  assert dk != 0.0 and abs(out[-1][0] - dk) <= step_bound(20.0)


def test_non_finite_result_zeroes_at_once(monkeypatch):
  s = scenario()
  c, out = run(s)
  assert out[-1][0] != 0.0
  monkeypatch.setattr(cc_mod, 'get_curvature_from_plan', lambda *a: math.nan)
  assert step(c, s, 40 * DT) == 0.0 and not c.dy.any() and 'nan' in block_names(c.block)


def test_mpc_failure_and_lane_mode_ramp_out_lat_inactive_zeroes_at_once():
  s = scenario()
  c, out = run(s)
  dks = [out[-1][0], step(c, s, 2.0, mpc_ok=False)]
  assert 'mpc' in block_names(c.block) and not c.dy.any() and dks[-1] != 0.0
  # an MPC reset restarts the 0.5 s startup hold; the output ramps to 0 meanwhile
  dks += [step(c, s, 2.0 + i * DT) for i in range(1, 10)]
  assert 'startup' in block_names(c.block)
  assert np.all(np.abs(np.diff(dks)) <= step_bound(20.0))
  assert all(abs(b) <= abs(a) for a, b in zip(dks, dks[1:], strict=False))
  c, out = run(s)
  dk = step(c, s, 2.0, lane_mode=True)
  assert 'laneMode' in block_names(c.block) and abs(out[-1][0] - dk) <= step_bound(20.0) and dk != 0.0
  s['lat_active'] = False
  assert step(c, s, 2.05) == 0.0 and 'latActive' in block_names(c.block)


def test_residual_inward_correction_is_dropped():
  s = scenario()
  c, _ = run(s)
  prev = c.curvature
  c.dy[:] = -c.dy  # residual from a previous corner now pointing to this corner's inside
  dk = step(c, s, 2.0)
  assert 'residualInward' in block_names(c.block) and not c.dy.any()
  assert abs(prev - dk) <= step_bound(20.0) and abs(dk) < abs(prev)


def test_released_gate_decays_without_step():
  s = scenario()
  c, out = run(s)
  s['md'].laneLineProbs = [0.9, 0.3, 0.9, 0.9]
  dks = [step(c, s, 2.0 + i * DT) for i in range(150)]
  assert abs(dks[0]) < abs(out[-1][0]) and abs(dks[0]) > 0.7 * abs(out[-1][0])
  assert np.all(np.abs(np.diff([out[-1][0]] + dks)) <= step_bound(20.0))
  assert all(abs(b) <= abs(a) for a, b in zip(dks, dks[1:], strict=False))
  assert dks[-1] == 0.0 and not c.dy.any()


def test_invariant_violation_gives_zero():
  s = scenario()
  c = CornerCenter(50)
  c.set_delay_params(300, 13)  # lat_action_t ≈ 3.2 s: L = v·2.5 s is not > 2.5·v·lat_action_t
  for i in range(40):
    assert step(c, s, i * DT) == 0.0
  assert c.dy.any() and 'invariant' in block_names(c.block)
  # a violation after Δκ was nonzero keeps Δy and ramps the output to 0
  c, out = run(s)
  assert out[-1][0] != 0.0
  c.set_delay_params(300, 13)
  dks = [out[-1][0]] + [step(c, s, (40 + i) * DT) for i in range(20)]
  assert 'invariant' in block_names(c.block) and c.dy.any() and dks[-1] == 0.0
  assert np.all(np.abs(np.diff(dks)) <= step_bound(20.0))


def test_blend_is_clipped_and_scales():
  _, a = run(scenario(), 120, blend=25)
  _, b = run(scenario(), 120, blend=80)
  _, c = run(scenario(), 120, blend=50)
  assert b[-1][0] == c[-1][0]
  assert abs(a[-1][0]) == pytest.approx(abs(c[-1][0]) / 2, rel=0.05)


# ---- controlsd side ----

class SM(dict):
  pass


def make_sm(dk=-1e-4, age_frames=2, seen=True, valid=True, **kw):
  fields = {'modelMonoTime': 1, 'mpcSolutionValid': True, 'cornerCenterCurvature': dk, 'cornerCenterActive': True} | kw
  sm = SM(lateralPlan=log.LateralPlan.new_message(**fields))
  sm.seen = {'lateralPlan': seen}
  sm.valid = {'lateralPlan': valid}
  sm.frame = 100
  sm.recv_frame = {'lateralPlan': 100 - age_frames}
  return sm


def test_controls_helper_accepts_only_fresh_valid_plans():
  assert corner_center_curvature(make_sm(), 20.0, False) == pytest.approx(-1e-4)
  assert corner_center_curvature(make_sm(age_frames=20), 20.0, False) == pytest.approx(-1e-4)
  for sm, pressed in [(make_sm(age_frames=21), False), (make_sm(seen=False), False), (make_sm(valid=False), False),
                      (make_sm(modelMonoTime=0), False), (make_sm(mpcSolutionValid=False), False),
                      (make_sm(dk=math.nan), False), (make_sm(dk=math.inf), False), (make_sm(), True)]:
    assert corner_center_curvature(sm, 20.0, pressed) == 0.0
  lim = cc_mod.DK_LAT_ACC_MAX / 20.0 ** 2
  assert corner_center_curvature(make_sm(dk=0.01), 20.0, False) == pytest.approx(lim)


def test_controls_output_ramps_from_the_last_value():
  out = CornerCenterOutput()
  v = 20.0
  lim = cc_mod.DK_LAT_ACC_MAX / v ** 2
  fresh = make_sm(dk=-lim)
  dks = [out.update(fresh, v, False) for _ in range(100)]
  assert dks[-1] == pytest.approx(-lim)
  assert np.all(np.abs(np.diff([0.0] + dks)) <= step_bound(v, dt=cc_mod.DT_CTRL))
  # a stale plan (and any other rejected plan) ramps the output to 0 at J_RAMP instead of a step
  for stale in (make_sm(dk=-lim, age_frames=21), make_sm(dk=-lim, mpcSolutionValid=False), make_sm(dk=math.nan)):
    out.curvature = -lim
    dks = [-lim] + [out.update(stale, v, False) for _ in range(70)]
    assert np.all(np.abs(np.diff(dks)) <= step_bound(v, dt=cc_mod.DT_CTRL))
    assert dks[1] != 0.0 and dks[-1] == 0.0
  # driver steering: J_DRIVER to 0, held at 0 during the press and 0.5 s after, then J_RAMP back
  start = -lim / 2
  out.curvature = start
  dks = [start] + [out.update(fresh, v, True) for _ in range(5)]
  dks += [out.update(fresh, v, False) for _ in range(round(cc_mod.DRIVER_REST_S / cc_mod.DT_CTRL) - 1)]
  assert np.all(np.abs(np.diff(dks)) <= step_bound(v, cc_mod.J_DRIVER, cc_mod.DT_CTRL))
  n0 = math.ceil(abs(start) / step_bound(v, cc_mod.J_DRIVER, cc_mod.DT_CTRL))
  assert n0 + 1 < len(dks) and all(dk == 0.0 for dk in dks[n0 + 1:])  # +1: float residual of an exact multiple
  back = [out.update(fresh, v, False) for _ in range(3)]
  assert back[0] != 0.0 and np.all(np.abs(np.diff([0.0] + back)) <= step_bound(v, dt=cc_mod.DT_CTRL))
  out.reset()
  assert out.curvature == 0.0 and out.driver_frames == 0


def _controls(lane_full=False):
  from openpilot.selfdrive.controls.tests.test_lateral_startup_control import control_fixture
  control, step_fn = control_fixture()
  control.lateral_startup.ready = True
  sm = control.sm
  sm['selfdriveState'].enabled = sm['selfdriveState'].active = True
  sm['modelV2'].action.desiredCurvature = 0.002
  sm.recv_frame['lateralPlan'] = 1
  if lane_full:
    lp = sm['lateralPlan']
    lp.useLaneLines = True
    lp.psis = [0.] * 17
    lp.curvatures = [0.002] * 17
    lp.distances = list(range(17))
    sm['carrotMan'].vTurnSpeed = 300
  return control, step_fn


def _curvatures(control, step_fn, n=30):
  out = []
  for _ in range(n):
    cc, _ = step_fn()
    out.append(cc.actuators.curvature)
  return out


def test_controls_laneless_branch_adds_curvature_and_ignores_inactive_plans():
  base_c, base_step = _controls()
  base = _curvatures(base_c, base_step, 150)
  c, s = _controls()
  lp = c.sm['lateralPlan']
  lp.modelMonoTime = 1
  lp.mpcSolutionValid = True
  lp.cornerCenterCurvature = -2e-4
  lp.cornerCenterActive = False   # parameter 0: the field is never set, the helper is never called
  c.sm.frame = 1
  assert _curvatures(c, s) == base[:30]
  lp.cornerCenterActive = True
  out = _curvatures(c, s, 60)
  assert out[-1] == pytest.approx(base[89] - 2e-4, abs=2e-6)
  lp.mpcSolutionValid = False
  out = _curvatures(c, s, 60)
  assert out[-1] == pytest.approx(base[149], abs=2e-6)
  assert c.corner_center.curvature == 0.0
  # the planner dropping its correction (cornerCenterActive false) also ramps out from the last value
  lp.mpcSolutionValid = True
  _curvatures(c, s, 60)
  assert c.corner_center.curvature == pytest.approx(-2e-4)
  lp.cornerCenterActive = False
  lp.cornerCenterCurvature = 0.0
  _curvatures(c, s, 1)
  assert -2e-4 < c.corner_center.curvature < 0.0
  _curvatures(c, s, 60)
  assert c.corner_center.curvature == 0.0


@pytest.mark.parametrize('meb', [False, True])
def test_controls_lane_mode_and_meb_branches_unchanged(meb):
  a, sa = _controls(lane_full=True)
  b, sb = _controls(lane_full=True)
  for ctl in (a, b):
    ctl.is_vw_meb = meb
    ctl.calibrated_pose = None
    ctl.meb_curvature_pid = NS(reset=lambda: None, update=lambda *x, **kw: 0.)
  lp = b.sm['lateralPlan']
  lp.modelMonoTime = 1
  lp.mpcSolutionValid = True
  lp.cornerCenterCurvature = -2e-4
  lp.cornerCenterActive = True
  assert _curvatures(a, sa) == _curvatures(b, sb)


# ---- plannerd wiring ----

class FakeParams:
  values = {}

  def __init__(self, *a, **k):
    pass

  def get_int(self, key, *a, **k):
    return int(FakeParams.values.get(key, 0))

  def get_float(self, key, *a, **k):
    return float(FakeParams.values.get(key, 0))

  def get_bool(self, key, *a, **k):
    return bool(FakeParams.values.get(key, False))


class PlannerSM(dict):
  pass


def planner_sm(s):
  sm = PlannerSM(modelV2=s['md'], carState=s['cs'], radarState=s['rs'],
                 controlsState=log.ControlsState.new_message(curvature=s['meas']),
                 carControl=car.CarControl.new_message(latActive=True), liveParameters=log.LiveParametersData.new_message(),
                 carrotMan=custom.CarrotMan.new_message(), liveDelay=log.LiveDelayData.new_message(lateralDelay=0.15))
  sm.seen = dict.fromkeys(sm, True)
  sm.valid = dict.fromkeys(sm, True)
  sm.alive = dict.fromkeys(sm, True)
  sm.logMonoTime = dict.fromkeys(sm, 0)
  sm.all_checks = lambda service_list=None: True
  return sm


def make_planner(monkeypatch, blend):
  import openpilot.selfdrive.controls.lib.lateral_planner as lpmod
  FakeParams.values = {'LatCornerCenterBlend': blend, 'LatMpcPathCost': 100, 'LatMpcMotionCost': 7, 'LatMpcAccelCost': 120,
                       'LatMpcJerkCost': 4, 'LatMpcSteeringRateCost': 7, 'LatSmoothSec': 13}
  monkeypatch.setattr(lpmod, 'Params', FakeParams)
  cp = car.CarParams.new_message(brand='hyundai', wheelbase=2.84, centerToFront=1.136, mass=1600,
                                 tireStiffnessFront=100000, tireStiffnessRear=100000)
  return lpmod.LateralPlanner(cp.as_reader())


def drive_planner(planner, frames=40):
  s = scenario()
  sm = planner_sm(s)
  pm = NS(sent=[], send=lambda name, msg: pm.sent.append(msg))
  carrot = NS(atc_active=False)
  for i in range(frames):
    sm.logMonoTime['modelV2'] = int((1.0 + i * DT) * 1e9)
    planner.update(sm, carrot)
    planner.publish(sm, pm, carrot)
  return [m.lateralPlan for m in pm.sent]


def test_planner_parameter_zero_never_calls_corner_center(monkeypatch):
  monkeypatch.setattr(CornerCenter, 'update', lambda *a, **k: pytest.fail('called with LatCornerCenterBlend=0'))
  planner = make_planner(monkeypatch, 0)
  plans = drive_planner(planner)
  assert planner.corner_center is None
  assert all(p.cornerCenterCurvature == 0.0 and not p.cornerCenterActive and p.cornerCenterBlock == 0 for p in plans)


def test_planner_parameter_fifty_publishes_correction(monkeypatch):
  planner = make_planner(monkeypatch, 50)
  plans = drive_planner(planner)
  last = plans[-1]
  assert last.cornerCenterActive and last.cornerCenterCurvature < 0.0
  assert last.cornerCenterGate == pytest.approx(1.0) and last.cornerCenterBlock == 0
  # switching the parameter to 0 drops the state at the next parameter read
  FakeParams.values['LatCornerCenterBlend'] = 0
  plans = drive_planner(planner, 12)
  assert planner.corner_center is None and not plans[-1].cornerCenterActive


def test_planner_disabled_for_vw_meb(monkeypatch):
  import openpilot.selfdrive.controls.lib.lateral_planner as lpmod
  planner = make_planner(monkeypatch, 50)
  monkeypatch.setattr(lpmod, 'is_volkswagen_meb', lambda cp: True)
  meb = lpmod.LateralPlanner(car.CarParams.new_message(brand='volkswagen', wheelbase=2.77, centerToFront=1.2, mass=2100,
                                                       tireStiffnessFront=100000, tireStiffnessRear=100000).as_reader())
  drive_planner(meb, 15)
  assert planner.corner_center_supported and not meb.corner_center_supported and meb.corner_center is None
