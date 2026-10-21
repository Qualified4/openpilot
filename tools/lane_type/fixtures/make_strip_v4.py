"""Builds the tiny synthetic StripNet v4 ONNX fixtures for test_strip_model_v4.py (run once with a torch venv, e.g.
~/t112/bin/python tools/lane_type/fixtures/make_strip_v4.py). The 'model' reads its logits straight from the strip:
type = channel 0, row 0, the last 5 columns; color = channel 1, row 0, the last 3 columns (the ego side of a
left strip, so a mirrored right strip shows its first columns there). No weights, so the files are a few hundred bytes."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


class Probe(torch.nn.Module):
  def __init__(self, n_type=5):
    super().__init__()
    self.n_type = n_type

  def forward(self, x):
    return x[:, 1, 0, 93:96], x[:, 0, 0, 96 - self.n_type:96]  # constant slices: the outputs keep static class dims


def export(path, n_type, names):
  torch.onnx.export(Probe(n_type), torch.zeros(1, 3, 128, 96), str(path), input_names=['strip_ycbcr'], output_names=names,
                    dynamic_axes={'strip_ycbcr': {0: 'N'}, names[0]: {0: 'N'}, names[1]: {0: 'N'}}, opset_version=11)


if __name__ == '__main__':
  export(HERE/'strip_v4_probe.onnx', 5, ['color', 'type'])
  export(HERE/'strip_v4_probe_type4.onnx', 4, ['color', 'type'])  # wrong type dim: the manifest check must refuse it
  print('written', *sorted(p.name for p in HERE.glob('*.onnx')), file=sys.stderr)
