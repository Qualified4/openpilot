"""PC-only four-backbone comparison; separate inputs, shared pretrained body."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from tools.lane_type.model import LaneTypeModel  # noqa: TID251 - offline repository tooling

NAMES = ('pplcnet_025', 'pplcnet_050', 'mobilenet_v3_small', 'shufflenet_v2_050')


class DualInputModel(nn.Module):
  def __init__(self, name, pretrained_root=None):
    super().__init__()
    if name not in NAMES:
      raise ValueError('unknown architecture')
    self.name = name
    loaded = []
    if name.startswith('pplcnet'):
      self.body = LaneTypeModel(.25 if name.endswith('025') else .5)
      if pretrained_root:
        record = json.loads((Path(pretrained_root)/'sources.json').read_text())[name]
        loaded = self.body.load_pretrained(Path(record['path']))
      # Avoid dead single-input heads in parameter/size accounting.
      self.body.type_head = nn.Identity()
      self.body.color_head = nn.Identity()
      size = 1280
    else:
      from torchvision.models import mobilenet_v3_small, shufflenet_v2_x0_5
      self.body = mobilenet_v3_small(weights=None) if name == 'mobilenet_v3_small' else shufflenet_v2_x0_5(weights=None)
      if pretrained_root:
        record = json.loads((Path(pretrained_root)/'sources.json').read_text())[name]
        path = Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
          raise ValueError('official weight checksum changed')
        state = torch.load(path, weights_only=True, map_location='cpu')
        self.body.load_state_dict(state, strict=True)
        loaded = list(state)
      if name == 'mobilenet_v3_small':
        self.body.classifier[-1] = nn.Identity()
        size = 1024
      else:
        self.body.fc = nn.Identity()
        size = 1024
    self.type_head = nn.Linear(size, 2)
    self.color_head = nn.Linear(size, 2)
    self.pretrained_tensors = len(loaded)

  def feature(self, image):
    return self.body.features(image) if self.name.startswith('pplcnet') else self.body(image)

  def forward(self, type_image, color_image):
    # The colour head never sees the long type crop. Both passes use the same
    # backbone parameters; ONNX stores shared weights once, inference does two passes.
    return self.type_head(self.feature(type_image)), self.color_head(self.feature(color_image))


class ExportDual(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, type_image, color_image):
    kind, color = self.model(type_image, color_image)
    return kind.softmax(1), color.softmax(1)
