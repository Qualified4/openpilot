"""CUDA-only fair comparison. Test images/labels are never loaded in training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from tools.lane_type.multi_model import DualInputModel, ExportDual, NAMES  # noqa: TID251 - offline repository tooling
from tools.lane_type.native_data import read_image  # noqa: TID251 - offline repository tooling
from openpilot.selfdrive.carrot.lane_type.roi import input_blob


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path):
  rows = json.loads(path.read_text(encoding='utf-8'))
  splits, scenes, pixels = {}, {}, {}
  arrays = {}
  for r in rows:
    if r['split'] not in ('train', 'validation', 'test'):
      raise ValueError('bad split')
    for table, key in [(splits, r['route']), (scenes, r['scene_id'])]:
      if table.setdefault(key, r['split']) != r['split']:
        raise ValueError('route/scene leakage')
    if r['verified'] is not True or not r.get('reviewer') or r['label_source'] not in ('visual_review', 'auto_individually_reviewed'):
      raise ValueError('unapproved label')
    if r['type'] not in ('dashed', 'solid') or r['color'] not in ('white', 'yellow'):
      raise ValueError('unknown truth')
    for head in ('type', 'color'):
      h = r['heads'][head]
      if pixels.setdefault(h['image_sha256'], r['split']) != r['split']:
        raise ValueError('pixel leakage')
      if not 4.5 <= h['near'] < h['far'] <= (35.01 if head == 'type' else 25.01) or h['paddingUsed']:
        raise ValueError('invalid effective ROI')
      actual_sha = hashlib.sha256(read_image(path.parent, h, arrays).tobytes()).hexdigest() if 'array_file' in h else sha(path.parent/h['image'])
      if actual_sha != h['image_sha256']:
        raise ValueError('ROI pixels changed')
  return rows


class Images(torch.utils.data.Dataset):
  def __init__(self, rows, root, augment):
    self.rows, self.augment = rows, augment
    arrays = {}
    self.images = [(read_image(root, r['heads']['type'], arrays), read_image(root, r['heads']['color'], arrays)) for r in rows]

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, index):
    images = self.images[index]
    if self.augment:
      gain = np.random.uniform(.85, 1.15)  # noqa: NPY002 - reproduce the recorded comparison RNG
      flip = random.random() < .5
      images = tuple(np.clip((im[:, ::-1] if flip else im).astype(float)*gain, 0, 255).astype(np.uint8) for im in images)
    r = self.rows[index]
    return *(torch.from_numpy(input_blob([im])[0]) for im in images), ('dashed', 'solid').index(r['type']), ('white', 'yellow').index(r['color'])


def train(manifest, pretrained, output, name, epochs=40, seed=42):
  if any((parent/'ROI_REVIEW_HOLD.json').exists() for parent in manifest.resolve().parents):
    raise RuntimeError('ROI pixel audit hold: training is blocked; preserve existing experiments')
  if (output/'checkpoint.pt').exists() or (output/'manifest.json').exists():
    raise FileExistsError('refusing to overwrite an existing experiment')
  if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
    raise RuntimeError('this comparison explicitly requires the RTX 4090; no silent CPU fallback')
  output.mkdir(parents=True, exist_ok=True)
  random.seed(seed)
  np.random.seed(seed)  # noqa: NPY002 - reproduce the recorded comparison RNG
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(4)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  all_rows = load_manifest(manifest)
  # Test files are checksum-validated above, but no test image is decoded and
  # no test label is used for any training, checkpoint or threshold choice.
  rows = {s: [r for r in all_rows if r['split'] == s] for s in ('train', 'validation')}
  if any(len(v) < 8 for v in rows.values()):
    raise ValueError('insufficient reviewed train/validation data')
  loaders = {s: torch.utils.data.DataLoader(Images(v, manifest.parent, s == 'train'), batch_size=32,
              shuffle=s == 'train', num_workers=0) for s, v in rows.items()}
  model = DualInputModel(name, pretrained).cuda()
  heads = list(model.type_head.parameters())+list(model.color_head.parameters())
  ids = {id(p) for p in heads}
  optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if id(p) not in ids], 'lr': 1e-4},
                                {'params': heads, 'lr': 1e-3}], weight_decay=1e-4)
  losses = []
  for head, classes in [('type', ('dashed', 'solid')), ('color', ('white', 'yellow'))]:
    c = Counter(r[head] for r in rows['train'])
    if any(c[label] == 0 for label in classes):
      raise ValueError('missing training class')
    weight = torch.tensor([len(rows['train'])/(2*c[label]) for label in classes], device='cuda')
    losses.append(nn.CrossEntropyLoss(weight=weight))
  start = time.perf_counter()
  history, best, best_loss, best_epoch = [], None, float('inf'), None
  torch.cuda.reset_peak_memory_stats()
  gradient_check = None
  initial_weight = model.type_head.weight.detach().clone()
  for epoch in range(epochs):
    record = {'epoch': epoch+1}
    for split, loader in loaders.items():
      model.train(split == 'train')
      for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
          module.eval()
      total, count = 0., 0
      for a, b, kind, color in loader:
        a, b, kind, color = (x.to('cuda') for x in (a, b, kind, color))
        with torch.set_grad_enabled(split == 'train'):
          yt, yc = model(a, b)
          loss = losses[0](yt, kind)+losses[1](yc, color)
          if split == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_check is None:
              gradient_check = {'model': str(next(model.parameters()).device), 'inputs': [str(a.device), str(b.device)],
                'targets': [str(kind.device), str(color.device)], 'loss_weights': [str(x.weight.device) for x in losses],
                'gradient_devices': sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None}),
                'finite_gradients': all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)}
            optimizer.step()
        total += loss.detach().item()*len(a)
        count += len(a)
      record[split+'_loss'] = total/count
    if record['validation_loss'] < best_loss:
      best_loss, best_epoch = record['validation_loss'], epoch+1
      best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history.append(record)
    print(name, epoch+1, record, flush=True)
  torch.cuda.synchronize()
  elapsed = time.perf_counter()-start
  gradient_check['head_max_change'] = float((model.type_head.weight.detach()-initial_weight).abs().max())
  allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
  model.cpu().eval()
  model.load_state_dict(best)
  torch.save({'state_dict': best, 'architecture': name, 'seed': seed, 'best_epoch': best_epoch,
              'dataset_sha256': sha(manifest)}, output/'checkpoint.pt')
  (output/'training.json').write_text(json.dumps({'history': history, 'training_seconds': elapsed,
    'best_epoch': best_epoch, 'cuda_evidence': gradient_check, 'max_allocated_bytes': allocated,
    'max_reserved_bytes': reserved}, indent=2)+'\n', encoding='utf-8')
  shapes = [loaders['train'].dataset.images[0][j].shape[:2] for j in (0, 1)]
  dummy = tuple(torch.zeros(2, 3, *shape) for shape in shapes)
  exported = ExportDual(model).eval()
  torch.onnx.export(exported, dummy, str(output/'lane_type.onnx'), input_names=['type_image', 'color_image'],
                    output_names=['type_prob', 'color_prob'], opset_version=13, do_constant_folding=True)
  import onnxruntime as ort
  session = ort.InferenceSession(str(output/'lane_type.onnx'), providers=['CPUExecutionProvider'])
  rng = np.random.default_rng(seed)
  a, b = [rng.normal(size=(2, 3, *shape)).astype(np.float32) for shape in shapes]
  with torch.no_grad():
    expected = [v.numpy() for v in exported(torch.from_numpy(a), torch.from_numpy(b))]
  actual = session.run(None, {'type_image': a, 'color_image': b})
  error = max(float(np.max(np.abs(x-y))) for x, y in zip(actual, expected, strict=True))
  if error > 1e-4:
    raise ValueError('ONNX numerical mismatch')
  info = {'architecture': name, 'seed': seed, 'input_contract': 'separate_type_color_RGB_ImageNet', 'input_shapes': [list(x.shape) for x in dummy],
    'train_samples': len(rows['train']), 'validation_samples': len(rows['validation']), 'test_used_for_selection': False,
    'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'epochs': epochs, 'history': history,
    'parameters': sum(p.numel() for p in model.parameters()), 'onnx_bytes': (output/'lane_type.onnx').stat().st_size,
    'training_seconds': elapsed, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
    'max_allocated_bytes': allocated, 'max_reserved_bytes': reserved, 'cuda_evidence': gradient_check,
    'pretrained_tensors': model.pretrained_tensors, 'onnx_torch_max_abs_error': error,
    'dataset_sha256': sha(manifest), 'onnx_sha256': sha(output/'lane_type.onnx'), 'accuracyValidated': False,
    'deviceValidated': False, 'canTransmitEnabled': False, 'adoption': 'experimental_offline_only'}
  (output/'manifest.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
  print('DONE', name, elapsed, 'seconds', flush=True)
  return info


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--pretrained', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--model', choices=NAMES, required=True)
  parser.add_argument('--epochs', type=int, default=40)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  train(args.manifest, args.pretrained, args.output, args.model, args.epochs, args.seed)
