import pytest

from openpilot.selfdrive.carrot.lane_type.display import DisplayPolicy


def tick(policy, stamp, kind='solid', color='white', *, now=None, pattern=False, baseline=None, warnings=None, **extra):
  result = {'type': kind, 'color': color, 'typeConfidence': .99, 'colorConfidence': .99,
    'typeMonoTimeNanos': stamp, 'colorMonoTimeNanos': stamp,
    'dashedPatternObserved': pattern, 'dashedPatternMonoTimeNanos': stamp, **extra}
  return policy.update(baseline or {}, {'type': 'ccncLaneType', 'version': 1, 'cameraMonoTimeNanos': stamp, 'left': result},
                       enabled=True, model_lanes=2, now_ns=stamp if now is None else now, warnings=warnings)


def test_default_white_needs_a_type_and_dashed_needs_real_repeating_pattern():
  p = DisplayPolicy()
  for t in (0, 200_000_000, 400_000_000):
    assert tick(p, t, kind='unknown')['values']['LANELINE_LEFT'] == 1
  for t in (600_000_000, 800_000_000):
    assert tick(p, t, kind='dashed')['values']['LANELINE_LEFT'] == 1
  assert tick(p, 1_000_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 1
  assert tick(p, 1_200_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 8


def test_type_hysteresis_does_not_extend_expiry_and_tracking_reset_prevents_lane_transfer():
  p = DisplayPolicy()
  tick(p, 0)
  assert tick(p, 200_000_000)['values']['LANELINE_LEFT'] == 2
  assert tick(p, 400_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 2
  assert tick(p, 600_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 8
  assert tick(p, 600_000_000, kind='unknown', color='unknown', now=1_100_000_000)['values']['LANELINE_LEFT'] == 8
  assert tick(p, 600_000_000, kind='unknown', color='unknown', now=1_100_000_001)['values']['LANELINE_LEFT'] == 1
  # Expired evidence cannot combine with one fresh observation after a gap.
  assert tick(p, 1_200_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 1
  assert tick(p, 1_400_000_000, kind='dashed', pattern=True)['values']['LANELINE_LEFT'] == 8
  assert tick(p, 1_600_000_000, kind='dashed', pattern=True, trackingReset=True)['values']['LANELINE_LEFT'] == 1


@pytest.mark.parametrize('warning', ['laneDeparture', 'blindSpot', 'laneLineWarning', 'unknown'])
@pytest.mark.parametrize('existing', [1, 4, 14])
def test_existing_warning_and_blink_phase_are_immutable(warning, existing):
  p = DisplayPolicy()
  baseline = {'LANELINE_LEFT': existing, 'COUNTER': 77}
  for t in (0, 200_000_000):
    r = tick(p, t, color='yellow', baseline=baseline, warnings={'left': {warning: True}})
    assert all(r['values'][key] == value for key, value in baseline.items())
    assert r['values']['LANELINE_RIGHT'] == 1
    assert r['sides']['left']['semantic'] == 'existing_warning'
  assert baseline == {'LANELINE_LEFT': existing, 'COUNTER': 77}


def test_recognised_yellow_and_warning_orange_have_distinct_reasons_and_color_expiry():
  p = DisplayPolicy()
  tick(p, 0, color='yellow')
  r = tick(p, 200_000_000, color='yellow')
  assert r['values']['LANELINE_LEFT'] == 4 and r['sides']['left']['reason'] == 'recognised_yellow_boundary'
  # Type refresh alone cannot prolong or replace an expired colour.
  tick(p, 400_000_000, color='unknown')
  assert tick(p, 800_000_000, color='unknown')['values']['LANELINE_LEFT'] == 1
  # No baseline available: do not invent an existing warning's code or blink phase.
  r = tick(p, 1_000_000_000, warnings={'left': {'blindSpot': True}})
  assert r['sides']['left']['candidateCode'] is None and 'LANELINE_LEFT' not in r['values']


def test_off_mode_and_other_side_do_not_reuse_display_state():
  p = DisplayPolicy()
  tick(p, 0)
  r = tick(p, 200_000_000)
  assert r['values']['LANELINE_LEFT'] == 2 and r['values']['LANELINE_RIGHT'] == 1
  baseline = {'LANELINE_LEFT': 6}
  assert p.update(baseline, {}, enabled=False, model_lanes=2, now_ns=200_000_000)['values'] == baseline
  assert tick(p, 400_000_000)['values']['LANELINE_LEFT'] == 1
