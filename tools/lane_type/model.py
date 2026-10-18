# Copyright (c) 2021 PaddlePaddle Authors. Licensed under the Apache License, Version 2.0.
# Architecture adapted to PyTorch from PaddleClas pp_lcnet.py; see README.md.
"""PC-only PP-LCNet-0.25x fine-tuning; torch is never imported on the comma."""
import hashlib
import pickle

import numpy as np
import torch
from torch import nn

PRETRAINED_URL = 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams'
PRETRAINED_SHA256 = '916a60d5db8669709550bd799e0ca95ab55c1fbd3154cf762dc48112d0f47afe'


def channels(value, scale=.25):
  result = max(8, int(value*scale+4)//8*8)
  return result+8 if result < .9*value*scale else result


class ConvBN(nn.Sequential):
  def __init__(self, inputs, outputs, kernel, stride=1, groups=1):
    super().__init__()
    self.add_module('conv', nn.Conv2d(inputs, outputs, kernel, stride, kernel//2, groups=groups, bias=False))
    self.add_module('bn', nn.BatchNorm2d(outputs))
    self.add_module('act', nn.Hardswish())


class SE(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.conv1 = nn.Conv2d(size, size//4, 1)
    self.conv2 = nn.Conv2d(size//4, size, 1)

  def forward(self, x):
    gate = x.mean((2, 3), keepdim=True)
    return x*torch.nn.functional.hardsigmoid(self.conv2(torch.relu(self.conv1(gate))))


class Block(nn.Module):
  def __init__(self, kernel, inputs, outputs, stride, se=False, scale=.25):
    super().__init__()
    inputs, outputs = channels(inputs, scale), channels(outputs, scale)
    self.dw_conv = ConvBN(inputs, inputs, kernel, stride, inputs)
    self.se = SE(inputs) if se else nn.Identity()
    self.pw_conv = ConvBN(inputs, outputs, 1)

  def forward(self, x):
    return self.pw_conv(self.se(self.dw_conv(x)))


class LaneTypeModel(nn.Module):
  def __init__(self, scale=.25):
    super().__init__()
    if scale not in (.25, .5):
      raise ValueError('only verified LCNet scales .25/.5 supported')
    self.scale = scale
    self.conv1 = ConvBN(3, channels(16, scale), 3, 2)
    stages = [
      [(3, 16, 32, 1)],
      [(3, 32, 64, 2), (3, 64, 64, 1)],
      [(3, 64, 128, 2), (3, 128, 128, 1)],
      [(3, 128, 256, 2)]+[(5, 256, 256, 1)]*5,
      [(5, 256, 512, 2, True), (5, 512, 512, 1, True)],
    ]
    for index, blocks in enumerate(stages, 2):
      self.add_module('blocks'+str(index), nn.Sequential(*(Block(*b, scale=scale) for b in blocks)))
    self.last_conv = nn.Conv2d(channels(512, scale), 1280, 1, bias=False)
    self.type_head = nn.Linear(1280, 2)
    self.color_head = nn.Linear(1280, 2)

  def features(self, x):
    x = self.conv1(x)
    for name in ('blocks2', 'blocks3', 'blocks4', 'blocks5', 'blocks6'):
      x = getattr(self, name)(x)
    return torch.nn.functional.hardswish(self.last_conv(x.mean((2, 3), keepdim=True))).flatten(1)

  def forward(self, x):
    x = self.features(x)
    return self.type_head(x), self.color_head(x)

  def load_pretrained(self, path):
    expected = PRETRAINED_SHA256 if self.scale == .25 else '3d76fa2b975dad4f9227a19ed5bae614c2455208c46618fb1a0eb487b0a69648'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
      raise ValueError('official pretrained checkpoint checksum mismatch')
    # Only load the hash-pinned official Paddle NumPy parameter dictionary.
    weights = pickle.loads(path.read_bytes())
    state = self.state_dict()
    loaded = []
    for key, target in state.items():
      if key.startswith(('type_head.', 'color_head.')) or key.endswith('num_batches_tracked'):
        continue
      source = key.replace('running_mean', '_mean').replace('running_var', '_variance')
      value = np.asarray(weights[source])
      if tuple(value.shape) != tuple(target.shape):
        raise ValueError('pretrained tensor shape mismatch: '+key)
      state[key] = torch.from_numpy(value.copy())
      loaded.append(key)
    self.load_state_dict(state, strict=True)
    return loaded


class ExportModel(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x):
    line_type, color = self.model(x)
    return line_type.softmax(1), color.softmax(1)
