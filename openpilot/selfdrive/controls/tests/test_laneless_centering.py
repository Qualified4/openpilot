import numpy as np
import pytest

from openpilot.cereal import car, log
from openpilot.selfdrive.controls.lib import laneless_centering as lc
from openpilot.selfdrive.controls.lib.laneless_centering import LanelessCentering, block_names

XS = 192.0 * (np.arange(33) / 32.0) ** 2  # modelV2's x grid


def model(centre=0.0, path=0.0, kappa=0.0, width=3.5, prob=0.9, std=0.1, inside=None, widen=0.0, frame=0):
  """A lane whose centre is `centre` m off the car (+ right) on a road of curvature `kappa`, and the model's path:
  `path` m off on a straight, or `inside` m to the curve's inside (reached by 10 m). `widen` m of extra width at 35 m."""
  md = log.ModelDataV2.new_message()
  md.frameId = frame
  bend = kappa * XS * XS / 2
  c = centre + bend
  w = width + widen * np.clip(XS / 35.0, 0.0, 1.0)
  lines = md.init('laneLines', 4)
  for i, y in enumerate((c - 1.5 * w, c - w / 2, c + w / 2, c + 1.5 * w)):
    lines[i].x = XS.tolist()
    lines[i].y = y.tolist()
  md.laneLineProbs = [0.5, prob, prob, 0.5]
  md.laneLineStds = [0.5, std, std, 0.5]
  py = (c + inside * np.sign(kappa) * np.minimum(XS / 10.0, 1.0)) if inside is not None else np.full(33, path) + bend
  md.position.x = XS.tolist()
  md.position.y = py.tolist()
  md.action.desiredCurvature = kappa
  md.meta.desireState = [1.0, 0, 0, 0, 0, 0, 0]
  return md


def state(**kw):
  return car.CarState.new_message(**kw)


def radar(left=(), right=()):
  rs = log.RadarState.new_message()
  for name, leads in (('leadsLeft', left), ('leadsRight', right)):
    out = rs.init(name, len(leads))
    for o, (d, v) in zip(out, leads, strict=True):
      o.dRel, o.vLead = d, v
  return rs


def run(c, md, steps, v=70 / 3.6, cs=None, rs=None, atc=''):
  cs = cs or state()
  rs = rs or radar()
  k = 0.0
  for i in range(steps):
    md.frameId = md.frameId + (1 if i % 5 == 0 else 0)
    k = c.update(md, cs, rs, atc, v)
  return k


def test_steers_toward_the_centre_when_the_model_holds_off_it():
  # the Rust test: centre 0.19 m left of the car and the path, 70 km/h: 2·0.19·v²/L² ≈ 0.095 m/s² plus a little integral
  c = LanelessCentering()
  k = run(c, model(centre=-0.19), 150)
  assert k < 0.0
  assert 0.09 <= -c.accel < 0.12
  c = LanelessCentering()
  assert run(c, model(centre=0.19), 150) > 0.0


def test_the_correction_never_exceeds_its_cap_and_bad_lines_get_nothing():
  c = LanelessCentering()
  v = 50 / 3.6
  k = run(c, model(centre=-0.55), 3000, v=v)
  assert abs(c.accel) <= lc.ACCEL_MAX + 1e-12 and abs(k * v * v) <= lc.ACCEL_MAX + 1e-12
  # 0.8 m off on a straight is taken for lane lines gone wrong; a centred car and path get nothing
  for md in (model(centre=-0.8), model()):
    c = LanelessCentering()
    assert run(c, md, 300) == 0.0


def test_a_curve_the_model_cuts_gets_more_than_the_straight_cap_toward_the_outside():
  v = 50 / 3.6
  for kappa in (1 / 150, -1 / 150):
    c = LanelessCentering()
    k = run(c, model(kappa=kappa, inside=0.5), 300, v=v)
    assert k * kappa < 0.0
    assert lc.ACCEL_MAX < abs(c.accel) <= lc.ACCEL_MAX + lc.CURVE_CAP_GAIN * abs(kappa) * v * v + 1e-12


def test_a_curve_driven_on_its_centre_gets_next_to_nothing():
  c = LanelessCentering()
  run(c, model(kappa=1 / 150, inside=0.0), 300, v=50 / 3.6)
  assert abs(c.accel) < lc.ACCEL_MAX / 20


def test_a_jittering_path_moves_the_correction_no_faster_than_the_rate_limit():
  rng = np.random.default_rng(1)
  c = LanelessCentering()
  accels = []
  for frame in range(400):
    md = model(kappa=1 / 120, inside=0.4 + rng.uniform(-0.12, 0.12), frame=frame)
    for _ in range(5):
      c.update(md, state(), radar(), '', 46 / 3.6)
      accels.append(c.accel)
  assert np.max(np.abs(np.diff(accels[1000:]))) / lc.DT <= lc.ACCEL_RATE_MAX + 1e-9


def test_ramps_in_over_a_second_and_out_over_half_a_second():
  c = LanelessCentering()
  md = model(centre=-0.19)
  first = run(c, md, 1)
  full = run(c, md, 100)
  assert abs(first) < 0.05 * abs(full)
  half = run(c, md, 25, cs=state(steeringPressed=True))
  assert 0.0 < abs(half) < 0.6 * abs(full)
  assert run(c, md, 30, cs=state(steeringPressed=True)) == 0.0


@pytest.mark.parametrize('side', ['bsd', 'radar'])
def test_never_pushes_toward_traffic_in_the_adjacent_lane(side):
  # centre to the left: the correction pushes left, so traffic on the left stops it and traffic on the right does not
  left = {'cs': state(leftBlindspot=True)} if side == 'bsd' else {'rs': radar(left=[(10.0, 20.0)])}
  right = {'cs': state(rightBlindspot=True)} if side == 'bsd' else {'rs': radar(right=[(10.0, 20.0)])}
  c = LanelessCentering()
  assert run(c, model(centre=-0.19), 300, **left) == 0.0
  assert ('bsd' if side == 'bsd' else 'side') in block_names(c.block)
  c = LanelessCentering()
  assert run(c, model(centre=-0.19), 300, **right) < 0.0
  # acting already: traffic appearing on the pushed side ramps it out within 0.5 s
  c = LanelessCentering()
  run(c, model(centre=-0.19), 300)
  assert run(c, model(centre=-0.19), 50, **left) == 0.0


def test_a_slow_or_far_radar_vehicle_does_not_count():
  c = LanelessCentering()
  assert run(c, model(centre=-0.19), 300, rs=radar(left=[(10.0, 1.0), (40.0, 20.0)])) < 0.0


def test_bsd_is_held_half_a_second():
  c = LanelessCentering()
  run(c, model(centre=-0.19), 1, cs=state(leftBlindspot=True))
  run(c, model(centre=-0.19), 45)
  assert 'bsd' in block_names(c.block)
  run(c, model(centre=-0.19), 10)
  assert 'bsd' not in block_names(c.block)


@pytest.mark.parametrize('cause,rest', [('blinker', lc.LANE_CHANGE_REST_S), ('driver', lc.DRIVER_REST_S),
                                        ('atc', lc.LANE_CHANGE_REST_S), ('desire', lc.LANE_CHANGE_REST_S)])
def test_rests_after_a_lane_change_or_driver_steering(cause, rest):
  md = model(centre=-0.19)
  busy = {'cs': state(leftBlinker=True)} if cause == 'blinker' else {'cs': state(steeringPressed=True)} if cause == 'driver' \
    else {'atc': 'turn left'} if cause == 'atc' else {}
  c = LanelessCentering()
  run(c, md, 300)
  if cause == 'desire':
    md.meta.desireState = [0.7, 0.3, 0, 0, 0, 0, 0]
  run(c, md, 10, **busy)
  md.meta.desireState = [1.0, 0, 0, 0, 0, 0, 0]
  assert run(c, md, int(rest / lc.DT) - 5) == 0.0
  assert block_names(c.block) == ['lanechange' if cause != 'driver' else 'driver']
  run(c, md, 10)
  assert c.block == 0 and c.blend > 0.0


def test_an_atc_prepare_phase_does_not_stop_it():
  c = LanelessCentering()
  assert run(c, model(centre=-0.19), 150, atc='turn left prepare') < 0.0


def test_lane_line_uncertainty_fades_it_out():
  full, half = LanelessCentering(), LanelessCentering()
  run(full, model(centre=-0.19, std=0.1), 1000)
  run(half, model(centre=-0.19, std=0.3), 1000)
  assert half.accel == pytest.approx(0.5 * full.accel, rel=0.1)
  c = LanelessCentering()
  assert run(c, model(centre=-0.19, std=0.45), 300) == 0.0
  assert 'std' in block_names(c.block)


def test_a_merge_or_split_turns_it_off():
  c = LanelessCentering()
  assert run(c, model(centre=-0.19, widen=1.0), 300) == 0.0
  assert 'merge' in block_names(c.block)


@pytest.mark.parametrize('why,kw,v', [
  ('speed', {}, 25 / 3.6),
  ('curvature', {'kappa': 0.03, 'inside': 0.0}, 50 / 3.6),
  ('prob', {'prob': 0.3}, 70 / 3.6),
  ('width', {'width': 4.6}, 70 / 3.6),
])
def test_does_nothing_when_slow_curved_or_unsure_of_the_lane(why, kw, v):
  c = LanelessCentering()
  assert run(c, model(centre=-0.19, **kw), 300, v=v) == 0.0
  assert why in block_names(c.block)


def test_a_lane_the_model_swaps_for_its_neighbour_holds_it_off():
  c = LanelessCentering()
  v = 55 / 3.6
  run(c, model(kappa=1 / 150, inside=0.4), 300, v=v)
  assert abs(c.accel) > 0.05
  run(c, model(kappa=1 / 150, inside=0.4, centre=-0.8), 1, v=v)
  assert 'jump' in block_names(c.block)


def test_missing_model_lines_get_nothing():
  c = LanelessCentering()
  assert run(c, log.ModelDataV2.new_message(), 10) == 0.0
  assert block_names(c.block) == ['input']
