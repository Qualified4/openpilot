"""Fine-tune official PP-LCNet-0.25x on visually reviewed route-disjoint data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn

from tools.lane_type.dataset import validate_rows  # noqa: TID251 - repository-level tooling
from tools.lane_type.model import LaneTypeModel, ExportModel, PRETRAINED_SHA256, PRETRAINED_URL  # noqa: TID251
from openpilot.selfdrive.carrot.lane_type.roi import input_blob
from openpilot.selfdrive.carrot.lane_type.result import TYPE_CLASSES, COLOR_CLASSES


def reviewed_data(manifest, quality=None):
  hold = manifest.parent/'ROI_REVIEW_HOLD.json'
  if hold.exists():
    raise ValueError('ROI review hold: resolve the recorded projection/paint-visibility audit before training')
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  validate_rows(rows, manifest.parent)
  selected = [r for r in rows if r.get('verified') is True and r['observation']['sufficient'] and not r['lane_change']]
  if quality:
    selected = [r for r in selected if r['video_source'].split('.')[0] == quality]
  splits = {name:[r for r in selected if r['split'] == name] for name in ('train', 'validation', 'test')}
  if any(len(v) < 4 for v in splits.values()):
    raise ValueError('need at least four reviewed usable ROIs in each route-disjoint split')
  for field, classes in [('type', TYPE_CLASSES), ('color', COLOR_CLASSES)]:
    if {r[field] for r in splits['train']} != set(classes):
      raise ValueError('training split is missing a reviewed '+field+' class')
  return splits


class ReviewedStrips(torch.utils.data.Dataset):
  def __init__(self, rows, root, augment=False):
    self.rows = rows
    self.images = [np.asarray(Image.open(root/r['image']).convert('RGB')).copy() for r in rows]
    self.augment = augment

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, index):
    image = self.images[index]
    if self.augment:
      image = np.clip(image.astype(np.float32)*np.random.uniform(.8, 1.2), 0, 255).astype(np.uint8)  # noqa: NPY002 - reproduce bundled pilot RNG
      if random.random() < .5:
        image = image[:, ::-1].copy()
    row = self.rows[index]
    return torch.from_numpy(input_blob([image])[0]), TYPE_CLASSES.index(row['type']), COLOR_CLASSES.index(row['color'])


def train(manifest, pretrained, output, epochs=30, seed=42, quality=None):
  torch.set_num_threads(4)
  random.seed(seed)
  np.random.seed(seed)  # noqa: NPY002 - reproduce bundled pilot RNG
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  splits = reviewed_data(manifest, quality)
  loaders = {name:torch.utils.data.DataLoader(ReviewedStrips(rows, manifest.parent, name == 'train'),
                                            batch_size=16, shuffle=name == 'train', num_workers=0)
             for name, rows in splits.items() if name != 'test'}
  model = LaneTypeModel()
  loaded = model.load_pretrained(pretrained)
  heads = list(model.type_head.parameters())+list(model.color_head.parameters())
  head_ids = {id(p) for p in heads}
  optimizer = torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in head_ids], 'lr':1e-4},
                                {'params':heads, 'lr':1e-3}], weight_decay=1e-4)
  losses = []
  for field, classes in [('type', TYPE_CLASSES), ('color', COLOR_CLASSES)]:
    counts = Counter(r[field] for r in splits['train'])
    weights = torch.tensor([len(splits['train'])/(len(classes)*counts[c]) for c in classes])
    losses.append(nn.CrossEntropyLoss(weight=weights))
  best_loss = float('inf')
  best = None
  history = []
  for epoch in range(epochs):
    values = {}
    for split, loader in loaders.items():
      model.train(split == 'train')
      # Small route-disjoint pilot batches cannot estimate stable BN moments.
      for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
          module.eval()
      total_loss, count = 0.0, 0
      for x, target_type, target_color in loader:
        with torch.set_grad_enabled(split == 'train'):
          line_type, color = model(x)
          loss = losses[0](line_type, target_type)+losses[1](color, target_color)
          if split == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach())*len(x)
        count += len(x)
      values[split] = total_loss/count
    history.append(dict(epoch=epoch+1, **values))
    if values['validation'] < best_loss:
      best_loss = values['validation']
      best = {k:v.detach().clone() for k, v in model.state_dict().items()}
    print(history[-1], flush=True)
  model.load_state_dict(best)
  model.eval()
  output.mkdir(parents=True, exist_ok=True)
  torch.save(best, output/'checkpoint.pt')
  wrapper = ExportModel(model).eval()
  example = torch.zeros(2, 3, 192, 64)
  path = output/'lane_type.onnx'
  torch.onnx.export(wrapper, example, str(path), input_names=['roi'], output_names=['line_type', 'color'],
                    dynamic_axes={'roi':{0:'sides'}, 'line_type':{0:'sides'}, 'color':{0:'sides'}},
                    opset_version=13, dynamo=False)
  import onnx
  import onnxruntime as ort
  onnx.checker.check_model(onnx.load(str(path)))
  sample = input_blob([np.asarray(Image.open(manifest.parent/r['image']).convert('RGB')) for r in splits['train'][:2]])
  session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
  actual = session.run(None, {'roi':sample})
  with torch.no_grad():
    expected = wrapper(torch.from_numpy(sample))
  parity = max(float(np.max(np.abs(a-b.numpy()))) for a, b in zip(actual, expected, strict=True))
  if parity > 1e-4:
    raise ValueError('ONNX export parity failed')
  metadata = {'version': 1, 'architecture': 'PP-LCNet-0.25x', 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'input_shape': [3, 192, 64], 'type_classes': list(TYPE_CLASSES), 'color_classes': list(COLOR_CLASSES),
                  'pretrained_url': PRETRAINED_URL, 'pretrained_sha256': PRETRAINED_SHA256, 'pretrained_tensors': len(loaded),
                  'dataset_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'seed': seed, 'epochs': epochs,
                  'best_epoch': min(history, key=lambda r:r['validation'])['epoch'],
                  'reviewed_samples': {k:len(v) for k, v in splits.items()},
                  'training_quality': quality or 'mixed',
                  'route_counts': {k:len({r['route'] for r in v}) for k, v in splits.items()},
                  'label_counts': {split:{field:dict(Counter(r[field] for r in rows)) for field in ('type', 'color')}
                                for split, rows in splits.items()},
                  'trainable_parameters': sum(p.numel() for p in model.parameters()), 'onnx_parity_max_abs': parity,
                  'batch_norm': 'frozen_pretrained_running_statistics',
                  'accuracyValidated': False, 'deviceValidated': False, 'canTransmitEnabled': False,
                  'confidenceCalibrated': False, 'status': 'experimental_small_visually_reviewed_dataset'}
  (output/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
  (output/'training-history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
  return metadata


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--pretrained', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--quality', choices=['fcamera', 'qcamera'], help='Source ablation; evaluation uses the common held-out manifest')
  args = parser.parse_args()
  print(json.dumps(train(args.manifest, args.pretrained, args.output, args.epochs, quality=args.quality), indent=2))
