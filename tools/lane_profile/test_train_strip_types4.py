"""4-type mode helpers (fourclass1026): unknown_weight / cap_dbl take the class count from the weight vector; the 5-type results
are unchanged; dataset_io.type4_id masks the 'unknown' abstention label."""
import pytest

torch = pytest.importorskip('torch')

from tools.lane_profile import dataset_io, train_strip as ts  # noqa: E402


def data5():
  T = torch.tensor([0]*50 + [1]*30 + [2]*3 + [3]*2 + [4]*15)
  return T, torch.ones(len(T))


def test_type4_id_and_display_family():
  assert [dataset_io.type4_id(t) for t in dataset_io.TYPES] == [0, 1, 2, 3, -100]
  assert dataset_io.TYPES4 == ts.TYPES4 == ('dashed', 'solid', 'dbl_near', 'dbl_far')
  assert [dataset_io.DISPLAY_TYPE[t] for t in dataset_io.TYPES4] == ['dashed', 'solid', 'dashed', 'solid']
  assert dataset_io.DISPLAY_COLOR['blue'] == 'white'


def test_5type_weights_unchanged():
  T, W = data5()
  wt = ts.class_weights(T, 5, (0, 1))
  S = torch.stack([W[T == i].sum() for i in range(5)])
  rest = sum(float(wt[i]*S[i]) for i in range(4))
  assert float(ts.unknown_weight(T, W, wt, .10)[4]) == pytest.approx(.1/.9*rest/float(S[4]))
  capped, scale = ts.cap_dbl(T, W, wt, .05, .10)
  m = capped*S
  assert scale < 1 and float((m[2] + m[3])/m.sum()) == pytest.approx(.05) and float(m[4]/m.sum()) == pytest.approx(.10)


def test_4type_weights_have_no_unknown():
  T, W = data5()
  T4 = torch.where(T == 4, torch.full_like(T, -100), T)
  wt = ts.class_weights(T4, 4, (0, 1), ignore=-100)
  assert len(wt) == 4 and torch.equal(ts.unknown_weight(T4, W, wt, .10), wt)  # nothing to solve
  S = torch.stack([W[T4 == i].sum() for i in range(4)])
  capped, scale = ts.cap_dbl(T4, W, wt, .05, 0., S)
  m = capped*S
  assert len(capped) == 4 and scale < 1 and float((m[2] + m[3])/m.sum()) == pytest.approx(.05)
  assert torch.equal(capped[:2], wt[:2])
