"""CcncPosDisplayMode 1 (display A) and 2 (display B): the common estimate and slot assignment, both displays, and
the Rust core against the Python reference on the same frames."""
import math
import random

import pytest

from opendbc.car.hyundai import ccnc_core_loader
from opendbc.car.hyundai import hyundaicanfd_ccnc_pos_display as ref

CORE = ccnc_core_loader.load()
LINES = (5.25, 1.75, -1.75, -5.25)
DT = 0.05


def obj(vid, x, y, now, lines=LINES, report=True):
  return (vid, x, y, now, (now, x, y) if report else None, lines)


def frame(now, objs, selected, yaw=None, lead=None, heading=None):
  return (now, yaw, tuple(selected), lead, list(objs), heading)


def run_both(mode, frames):
  """The Rust output, after checking it equals the Python reference's exactly."""
  python = [(list(a), list(d)) for a, d in ref.run(mode, frames)]
  rust = [(list(a), list(d)) for a, d in CORE._pos_run_for_test(mode, frames)]
  assert rust == python
  return rust


def single(ys, x=30., yaw=None, mode=1, selected=(1, None, None), lead=None, lines=LINES, heading=None):
  frames = []
  for i, y in enumerate(ys):
    now = i * DT
    h = heading(now) if callable(heading) else heading
    frames.append(frame(now, [obj(1, x, y, now, lines)], selected, yaw(now) if callable(yaw) else yaw, lead, h))
  return frames, run_both(mode, frames)


def drawn_of(out, vid=1):
  values = []
  for assigned, drawn in out:
    values.append(next((d for a, d in zip(assigned, drawn, strict=True) if a == vid), None))
  return values


def test_jump_is_absorbed():
  """A 0.5 m saw tooth (reflection point switching for 0.2 s) moves the drawn centre less than 0.1 m."""
  ys = [0.] * 40 + [0.5] * 4 + [0.] * 60
  _, out = single(ys)
  assert max(abs(v) for v in drawn_of(out)) < 0.1


def test_jump_that_stays_is_followed_slowly():
  """A step that stays is reached gradually (1.5-2 s), not at once."""
  ys = [0.] * 20 + [0.6] * 120
  _, out = single(ys)
  values = drawn_of(out)
  assert values[21] < 0.1
  assert 0.25 < values[20 + 35] < 0.45  # about one time constant
  assert abs(values[-1] - 0.6) < 0.05 + ref.FF_PLAY_PER_M * 30.


def test_steady_move_is_followed():
  """0.6 m/s sideways for 3 s (0.05 m radar steps): within 0.3 m of the car 2 s into the move and at its end."""
  true = [0.] * 20 + [0.6 * DT * i for i in range(61)] + [1.8] * 20
  ys = [round(y / 0.05) * 0.05 for y in true]
  _, out = single(ys, x=20.)
  values = drawn_of(out)
  assert abs(values[20 + 40] - true[20 + 40]) < 0.3
  assert abs(values[20 + 60] - true[20 + 60]) < 0.3
  assert abs(values[-1] - 1.8) < 0.1


def test_ego_rotation_is_not_a_jump():
  """A parked car 60 m ahead while ego turns at 0.12 rad/s moves 0.36 m per step in the ego frame, more than the jump
  threshold. With the livePose yaw rate the drawn centre stays on it (fixed once the rotation is taken out); without it
  the centre lags as if every step were a reflection jump."""
  x, rate = 60., 0.12
  ys = [0.5 - x * rate * DT * i for i in range(40)]
  _, out = single(ys, x=x, yaw=rate, lines=(None,) * 4)
  values = drawn_of(out)
  play = ref.FF_PLAY_PER_M * x
  assert max(abs(v - y) for v, y in zip(values, ys, strict=True)) < 0.05 + play
  fixed = [v + x * rate * DT * i for i, v in enumerate(values)]
  assert max(fixed) - min(fixed) < 0.05 + play
  _, out = single(ys, x=x, yaw=None, lines=(None,) * 4)
  assert max(abs(v - y) for v, y in zip(drawn_of(out), ys, strict=True)) > 1.


def test_a_takes_the_road_heading_out():
  """Ego turns 0.03 rad towards a new lane (0.03 rad/s for 1 s) before it moves sideways: a car 40 m ahead in its own
  lane moves 1.2 m right in the ego frame. A removes the angular correction beyond the soft deadband.
  The low-pass holds its input over frames where heading is unavailable."""
  x, rate = 40., 0.03
  turn = lambda now: min(max(now - 1., 0.), 1.)
  th = -rate
  ys = [x * th * turn(i * DT) for i in range(70)]
  yaw = lambda now: rate if 1. < now <= 2. else 0.
  heading = lambda now: None if 2.5 < now < 2.8 else th * turn(now)
  _, out = single(ys, x=x, yaw=yaw, lines=(None,) * 4, heading=heading)
  play = ref.FF_PLAY_PER_M * x
  values = drawn_of(out)
  assert values[-1] == pytest.approx(-x * ref.HEADING_DEADBAND, abs=play + 0.01)
  _, out = single(ys, x=x, yaw=yaw, lines=(None,) * 4)
  assert drawn_of(out)[-1] == pytest.approx(x * th, abs=play + 0.01)


def test_a_ignores_small_heading_changes():
  _, out = single([0.] * 40, heading=lambda now: ref.HEADING_DEADBAND * math.sin(5. * now))
  assert drawn_of(out) == [0.] * 40


@pytest.mark.parametrize("slot, offset, play_per_m", [
  ((1, None, None), 0., ref.FF_PLAY_PER_M),
  ((None, 1, None), 3.5, ref.PLAY_PER_M),
])
def test_a_backlash_hides_small_back_and_forth(slot, offset, play_per_m):
  """Both slots allow steady motion through at the same speed with their own backlash."""
  x = 40.
  play = play_per_m * x
  wobble = [0.15 if (i // 4) % 2 else 0. for i in range(80)]
  move = [0.15 + 0.5 * DT * i for i in range(1, 41)]
  ys = [y + offset for y in wobble + move]
  _, out = single(ys, x=x, selected=slot)
  values = drawn_of(out)
  assert max(values[:80]) - min(values[:80]) <= 0.15 - play + 1e-9
  slope = (values[-1] - values[-11]) / (10 * DT)
  assert abs(slope - 0.5) < 0.05
  assert abs(values[-1] - ys[-1]) <= play + 0.1  # the play plus the estimate's own lag at 0.5 m/s


def test_b_lane_change_after_hold_at_constant_speed():
  """B: a car that crosses into the left lane and stays there 2 s is then drawn moving to that lane's centre at
  ANIM_SPEED, one constant step per update, without stopping or going back. Its slot follows the same vid FF -> LF."""
  ys = [0.] * 20 + [min(3.5, 0.15 * i) for i in range(1, 30)] + [3.5] * 120
  frames, out = single(ys, mode=2)
  values = drawn_of(out)
  cross = next(i for i, y in enumerate(ys) if y > 1.75)
  hold = round(ref.CROSS_HOLD / DT)
  assert all(v == 0. for v in values[:cross + hold - 1])
  moving = values[cross + hold - 1:]
  steps = [b - a for a, b in zip(moving, moving[1:], strict=False)]
  assert all(s >= 0. for s in steps)
  ramp = [s for s in steps if s > 0.]
  assert ramp[:-1] == pytest.approx([ref.ANIM_SPEED * DT] * (len(ramp) - 1))
  assert moving[-1] == pytest.approx(3.5)
  assert out[0][0][0] == 1 and out[-1][0][1] == 1 and out[-1][0][0] is None


def test_b_straddle_needs_band_hold():
  """B: a car that stays on the left line is drawn on the line after BAND_HOLD, not before."""
  ys = [0.] * 20 + [min(1.6, 0.1 * i) for i in range(1, 17)] + [1.6] * 60
  _, out = single(ys, mode=2)
  values = drawn_of(out)
  enter = next(i for i, y in enumerate(ys) if y >= 1.35)
  band = round(ref.BAND_HOLD / DT)
  assert all(v == 0. for v in values[:enter + band - 2])
  assert values[-1] == pytest.approx(1.75)


def test_straddling_car_goes_to_side_slot():
  """A car the selector put in FF but whose centre lies on ego's left line is shown as LF."""
  _, out = single([1.6] * 10)
  assert out[-1][0] == [None, 1, None]
  _, out = single([-1.9] * 10)
  assert out[-1][0] == [None, None, 1]


def test_lead_one_stays_ff():
  """The car openpilot follows (radarState leadOne) is FF even while it straddles, also when the selector has it as
  a side vehicle."""
  _, out = single([1.6] * 10, lead=(30.5, 1.5))
  assert out[-1][0] == [1, None, None]
  _, out = single([1.6] * 10, selected=(None, 1, None), lead=(30., 1.6))
  assert out[-1][0] == [1, None, None]


def test_ping_pong_is_blocked_but_not_a_lane_change():
  """Two vids that the selector swaps in FF every 0.25 s: the slot does not go back to the vid it just left sooner
  than PINGPONG_HOLD. A vid moving to another slot is never held back (test_b_lane_change...)."""
  frames = []
  for i in range(80):
    now = i * DT
    choice = 1 if (i // 5) % 2 == 0 else 2
    frames.append(frame(now, [obj(1, 30., 0., now), obj(2, 32., 0.2, now)], (choice, None, None)))
  out = run_both(1, frames)
  ff = [a[0] for a, _ in out]
  switches = [i for i in range(1, len(ff)) if ff[i] != ff[i - 1]]
  selector = [i for i in range(1, len(frames)) if frames[i][2][0] != frames[i - 1][2][0]]
  assert len(switches) < len(selector)
  for a, b in zip(switches, switches[1:], strict=False):
    assert (b - a) * DT >= ref.PINGPONG_HOLD - 1e-9


@pytest.mark.parametrize("mode", [1, 2])
def test_rust_matches_python_on_random_frames(mode):
  rng = random.Random(7 + mode)
  frames, now = [], 0.
  cars = {vid: [rng.uniform(5, 80), rng.uniform(-6, 6), rng.uniform(-1, 1)] for vid in range(1, 7)}
  for _ in range(3000):
    now += rng.choice((0.05, 0.05, 0.05, 0.1, 0.0))
    objs = []
    for vid, c in cars.items():
      c[1] += c[2] * 0.05 + (rng.choice((0.5, -0.5, 1.2)) if rng.random() < 0.03 else 0.)
      if rng.random() < 0.02:
        c[2] = rng.uniform(-1, 1)
      if rng.random() < 0.9:
        lines = tuple(None if rng.random() < 0.1 else v + rng.uniform(-0.3, 0.3) for v in LINES)
        stamp = now - rng.choice((0., 0.02, 0.06))
        objs.append((vid, c[0], c[1], stamp, (stamp, c[0], c[1]) if rng.random() < 0.9 else None, lines))
    keys = [o[0] for o in objs]
    selected = [rng.choice(keys + [None]) if keys else None for _ in range(3)]
    selected = [s if s not in selected[:i] else None for i, s in enumerate(selected)]
    yaw = None if rng.random() < 0.1 else rng.uniform(-0.2, 0.2)
    lead = (rng.uniform(5, 80), rng.uniform(-2, 2)) if rng.random() < 0.5 else None
    if objs and rng.random() < 0.3:
      o = rng.choice(objs)
      lead = (o[1] + rng.uniform(-1, 1), o[2])
    heading = None if rng.random() < 0.2 else rng.uniform(-0.05, 0.05)
    frames.append(frame(now, objs, selected, yaw, lead, heading))
  out = run_both(mode, frames)
  assert any(a != [None, None, None] for a, _ in out)
  assert all(d is None or math.isfinite(d) for _, ds in out for d in ds)


@pytest.mark.parametrize("mode", [1, 2])
def test_ego_lane_change_moves_cars_without_a_step(mode):
  """Ego moves one lane to the right at 1 m/s. In the ego frame the cars and lines move left; when ego's centre
  crosses the right line the lane view counts from the new lane (the map's lane offset, or the model renumbering its
  lines). The car on the right (2) moves on into FF and the car ahead (1) into LF, each drawn value changing by no
  more than the cars' own motion plus ANIM_SPEED per update: no step, no gap."""
  frames, ys = [], {1: 0., 2: -3.5}
  for i in range(140):
    now = i * DT
    shift = min(3.5, max(0., (now - 1.) * 1.))
    k = 1 if shift > 1.75 else 0  # lanes counted from ego's new lane once its centre crossed
    lines = tuple(v + shift - k * 3.5 for v in LINES)
    objs = [obj(vid, 30. + vid, y + shift, now, lines) for vid, y in ys.items()]
    view = ref.LaneView(lines)
    sel = [None, None, None]
    for vid, x, y, *_ in objs:
      sel[{0: 0, -1: 1, 1: 2}.get(view.lane(y), 0)] = vid
    frames.append(frame(now, objs, sel))
  out = run_both(mode, frames)
  for vid in (1, 2):
    values = drawn_of(out, vid)
    assert None not in values
    steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    assert max(steps) <= 1. * DT + ref.ANIM_SPEED * DT + 1e-6
  assert out[-1][0][0] == 2 and out[-1][0][1] == 1
  play = ref.FF_PLAY_PER_M * 32. if mode == 1 else 0.
  assert drawn_of(out, 2)[-1] == pytest.approx(0., abs=0.05 + play)
