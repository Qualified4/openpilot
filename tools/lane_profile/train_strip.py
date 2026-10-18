"""Train a lightweight lane type/colour classifier on the 4.5-20 m row-aligned strip (PC/GPU only).

Input: YCbCr uint8 strip 128 x 96 (rows far->near, metric columns) from build_dataset.py. Labels: interval
labels (human > scanner auto > scanner pool > AI review) with per-origin weights. Classes: type dashed/solid,
colour white/yellow. Yellow dashed has no training data and is NOT a learned class (dashed+yellow is never
claimed; downstream must treat it as unknown).

Model selection and the abstention threshold use the validation routes only. The held-out human test set is
evaluated once, after selection, by evaluate(); its result never feeds back into training.
The exported ONNX takes the raw strip (N,3,128,96 float in 0..255) and contains normalisation and softmax.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.lane_profile.dataset_io import COLORS, TYPES, load_set  # noqa: TID251


class Normalize(nn.Module):
  """Raw YCbCr strip -> network input. Chroma is centred and amplified (lane colour is a small offset)."""
  def __init__(self, rgb_imagenet=False):
    super().__init__()
    self.rgb = rgb_imagenet

  def forward(self, x):
    if not self.rgb:
      return torch.cat(((x[:, :1] - 110.)/50., (x[:, 1:] - 128.)/16.), 1)
    y, cb, cr = x[:, :1], x[:, 1:2] - 128., x[:, 2:] - 128.
    rgb = torch.cat((y + 1.402*cr, y - .344136*cb - .714136*cr, y + 1.772*cb), 1)/255.
    mean = torch.tensor([.485, .456, .406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=x.device).view(1, 3, 1, 1)
    return (rgb - mean)/std


def dw(cin, cout, stride):
  return nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                       nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class StripNet(nn.Module):
  """~40k-parameter depthwise CNN. Keeps a coarse along-road profile (8 distance cells) so the head
  can see paint/gap alternation, while pooling across the lateral axis."""
  def __init__(self, width=1.):
    super().__init__()
    c = [int(round(v*width)) for v in (16, 24, 32, 48, 64)]
    self.stem = nn.Sequential(nn.Conv2d(3, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
    self.body = nn.Sequential(dw(c[0], c[1], 2), dw(c[1], c[2], 2), dw(c[2], c[3], 2), dw(c[3], c[4], 1))
    self.profile = nn.Conv2d(c[4], 16, 1)
    self.fc = nn.Sequential(nn.Linear(16*8 + c[4], 64), nn.ReLU(inplace=True), nn.Dropout(.2))
    self.type_head, self.color_head = nn.Linear(64, 2), nn.Linear(64, 2)

  def forward(self, x):
    f = self.body(self.stem(x))                     # N, C, 8, 6
    prof = self.profile(f).mean(3).flatten(1)        # along-road profile, lateral pooled
    z = self.fc(torch.cat((prof, f.mean((2, 3))), 1))
    return self.type_head(z), self.color_head(z)


MBV3_SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'


class MobileNetHeads(nn.Module):
  """torchvision MobileNetV3-Small backbone + its 1024-d hidden layer, with type/colour heads."""
  def __init__(self):
    super().__init__()
    from torchvision.models import mobilenet_v3_small
    base = mobilenet_v3_small(weights=None)
    self.features, self.hidden = base.features, base.classifier[:3]  # Linear 576->1024, Hardswish, Dropout
    self.type_head, self.color_head = nn.Linear(1024, 2), nn.Linear(1024, 2)

  def load_pretrained(self, path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MBV3_SHA256:
      raise ValueError('MobileNetV3-Small checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    own = {k: v for k, v in state.items() if k.startswith('features.')}
    own.update({'hidden.0.weight': state['classifier.0.weight'], 'hidden.0.bias': state['classifier.0.bias']})
    missing, unexpected = self.load_state_dict(own, strict=False)
    if unexpected or any(not k.startswith(('type_head', 'color_head')) for k in missing):
      raise ValueError(f'pretrained mismatch: {missing} {unexpected}')
    return len(own)

  def forward(self, x):
    z = self.hidden(self.features(x).mean((2, 3)))
    return self.type_head(z), self.color_head(z)


class Net(nn.Module):
  def __init__(self, arch):
    super().__init__()
    self.arch = arch
    self.norm = Normalize(rgb_imagenet=arch in ('pplcnet025', 'pplcnet050', 'mobilenetv3s'))
    if arch == 'stripnet':
      self.model = StripNet()
    elif arch in ('pplcnet025', 'pplcnet050'):
      from tools.lane_type.model import LaneTypeModel  # noqa: TID251
      self.model = LaneTypeModel(.25 if arch == 'pplcnet025' else .5)
    elif arch == 'mobilenetv3s':
      self.model = MobileNetHeads()
    else:
      raise ValueError(arch)

  def forward(self, x):
    return self.model(self.norm(x))


class Export(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net

  def forward(self, x):
    t, c = self.net(x)
    return t.softmax(1), c.softmax(1)


def augment(x, gen):
  """x: float N,3,H,W raw YCbCr. Photometric + geometric jitter that keeps the label valid."""
  n = x.shape[0]
  flip = torch.rand(n, device=x.device, generator=gen) < .5
  x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)            # left/right boundary symmetry
  gain = 1 + (torch.rand(n, 1, 1, 1, device=x.device, generator=gen) - .5)*.6
  bias = (torch.rand(n, 1, 1, 1, device=x.device, generator=gen) - .5)*40
  y = x[:, :1]*gain + bias
  csc = 1 + (torch.rand(n, 1, 1, 1, device=x.device, generator=gen) - .5)*.3
  ch = (x[:, 1:] - 128)*csc + 128 + (torch.rand(n, 2, 1, 1, device=x.device, generator=gen) - .5)*4
  x = torch.cat((y, ch), 1)
  # transverse shadow band on some samples (luma only)
  band = torch.rand(n, device=x.device, generator=gen) < .3
  if band.any():
    h = x.shape[2]
    start = (torch.rand(n, device=x.device, generator=gen)*h).long()
    length = (torch.rand(n, device=x.device, generator=gen)*h*.3).long() + 4
    rows = torch.arange(h, device=x.device).view(1, h)
    m = (rows >= start.view(-1, 1)) & (rows < (start + length).view(-1, 1)) & band.view(-1, 1)
    shade = .45 + .3*torch.rand(n, 1, device=x.device, generator=gen)
    factor = torch.where(m, shade, torch.ones_like(shade)).view(n, 1, h, 1)
    x = torch.cat((x[:, :1]*factor, x[:, 1:]), 1)
  shift = int(torch.randint(-6, 7, (1,), generator=gen, device=x.device))
  if shift:
    x = torch.roll(x, shift, 2)
  return (x + torch.randn(x.shape, device=x.device, generator=gen)*2).clamp(0, 255)


BLUR = torch.tensor([.25, .5, .25])


def augment_plus(x, gen):
  """augment() plus luma contrast/gamma, mild blur and noise, small lateral shift.

  Bounded so labels survive: luma-only photometric changes (chroma = colour evidence untouched beyond
  augment()'s +-15% scale), blur <= one 3-tap pass (~0.7px, a dash gap is tens of rows), lateral shift
  <= 4 columns (6cm; the stripe stays inside the 1.5m strip), extra luma-only noise sigma <= 3 levels."""
  x = augment(x, gen)
  n, dev = x.shape[0], x.device
  r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
  y = x[:, :1]
  mean = y.mean((2, 3), keepdim=True)
  y = (y - mean)*(.8 + .4*r(n, 1, 1, 1)) + mean                       # contrast 0.8-1.2
  gamma = torch.exp((r(n, 1, 1, 1) - .5)*2*np.log(1.25))               # gamma 0.8-1.25
  y = 255*(y.clamp(0, 255)/255)**gamma
  x = torch.cat((y, x[:, 1:]), 1)
  blur = r(n) < .3
  if blur.any():
    k = BLUR.to(dev)
    kern = (k.view(3, 1)*k.view(1, 3)).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    xb = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), kern, groups=3)
    x = torch.where(blur.view(-1, 1, 1, 1), xb, x)
  shift = int(torch.randint(-4, 5, (1,), generator=gen, device=dev))
  if shift:
    x = F.pad(x, (max(shift, 0), max(-shift, 0), 0, 0), mode='replicate')[..., max(-shift, 0):max(-shift, 0) + 96]
  noise = torch.randn((n, 1) + x.shape[2:], device=dev, generator=gen)*3*r(n, 1, 1, 1)  # luma only
  return torch.cat((x[:, :1] + noise, x[:, 1:]), 1).clamp(0, 255)


def tensors(rows, x, device):
  xt = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
  t = torch.tensor([TYPES.index(r['type']) for r in rows], device=device)
  c = torch.tensor([COLORS.index(r['color']) for r in rows], device=device)
  w = torch.tensor([r['weight'] for r in rows], dtype=torch.float32, device=device)
  return xt, t, c, w


@torch.no_grad()
def predict(net, xt, batch=1024):
  net.eval()
  out_t, out_c = [], []
  for i in range(0, len(xt), batch):
    t, c = net(xt[i:i + batch])
    out_t.append(t.softmax(1))
    out_c.append(c.softmax(1))
  return torch.cat(out_t).cpu().numpy(), torch.cat(out_c).cpu().numpy()


def metrics(pt, pc, rows, threshold=0.):
  res = {}
  for head, p, classes in (('type', pt, TYPES), ('color', pc, COLORS)):
    truth = np.array([classes.index(r[head]) for r in rows])
    pred, conf = p.argmax(1), p.max(1)
    decided = conf >= threshold
    per = {}
    for k, name in enumerate(classes):
      sel = truth == k
      per[name] = {'support': int(sel.sum()), 'recall_raw': float((pred[sel] == k).mean()) if sel.any() else None,
                   'recall_decided_over_all': float(((pred == k) & decided)[sel].mean()) if sel.any() else None,
                   'precision_decided': float((truth[(pred == k) & decided] == k).mean()) if ((pred == k) & decided).any() else None}
    res[head] = {'accuracy_raw': float((pred == truth).mean()), 'decided_rate': float(decided.mean()),
                 'accuracy_decided': float((pred[decided] == truth[decided]).mean()) if decided.any() else None,
                 'macro_recall_raw': float(np.mean([v['recall_raw'] for v in per.values() if v['recall_raw'] is not None])),
                 'classes': per}
  solid = np.array([r['type'] == 'solid' for r in rows])
  res['false_dashed_raw'] = f"{int(((pt.argmax(1) == 0) & solid).sum())}/{int(solid.sum())}"
  res['false_dashed_decided'] = f"{int(((pt.argmax(1) == 0) & (pt.max(1) >= threshold) & solid).sum())}/{int(solid.sum())}"
  return res


def train(arch, seed, train_rows, xtr, val_rows, xva, device, epochs, pretrained, aug='base'):
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  net = Net(arch).to(device)
  if arch in ('pplcnet025', 'pplcnet050'):
    net.model.load_pretrained(pretrained[arch])
  elif arch == 'mobilenetv3s':
    net.model.load_pretrained(pretrained['mobilenetv3s'])
  augment_fn = augment_plus if aug == 'plus' else augment
  X, T, C, W = tensors(train_rows, xtr, device)
  Xv, _, _, _ = tensors(val_rows, xva, device)
  # class-balanced weights from TRAIN only
  wt = torch.tensor([len(T)/(2*max(1, int((T == k).sum()))) for k in range(2)], device=device)
  wc = torch.tensor([len(C)/(2*max(1, int((C == k).sum()))) for k in range(2)], device=device)
  lr = 2e-3 if arch == 'stripnet' else 1e-3
  opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
  steps = epochs*math.ceil(len(X)/256)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.15)
  best, history = None, []
  start = time.monotonic()
  for epoch in range(epochs):
    net.train()
    order = torch.randperm(len(X), device=device, generator=gen)
    total = 0.
    for i in range(0, len(X), 256):
      idx = order[i:i + 256]
      lt, lc = net(augment_fn(X[idx], gen))
      loss = ((F.cross_entropy(lt, T[idx], weight=wt, reduction='none') + F.cross_entropy(lc, C[idx], weight=wc, reduction='none'))*W[idx]).sum()/W[idx].sum()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      sched.step()
      total += float(loss)*len(idx)
    pt, pc = predict(net, Xv)
    m = metrics(pt, pc, val_rows)
    score = (m['type']['macro_recall_raw'] + m['color']['macro_recall_raw'])/2
    history.append({'epoch': epoch, 'train_loss': total/len(X), 'val_type_macro_recall': m['type']['macro_recall_raw'],
                    'val_color_macro_recall': m['color']['macro_recall_raw'], 'val_score': score})
    if best is None or score > best[0]:
      best = (score, epoch, {k: v.detach().clone() for k, v in net.state_dict().items()})
  net.load_state_dict(best[2])
  return net, {'best_epoch': best[1], 'best_val_score': best[0], 'history': history, 'gpu_seconds': time.monotonic() - start,
               'params': sum(p.numel() for p in net.parameters())}


def choose_threshold(pt, pc, rows, target=.99):
  """Smallest confidence threshold whose validation decided-accuracy reaches ``target`` for BOTH heads."""
  for th in np.arange(.5, 1.0, .01):
    m = metrics(pt, pc, rows, th)
    if all((m[h]['accuracy_decided'] or 0) >= target for h in ('type', 'color')):
      return float(round(th, 2))
  return .99


def export(net, path):
  net = net.cpu().eval()
  wrapper = Export(net).eval()  # export restores the module's previous mode; keep it eval for the parity check
  dummy = torch.full((2, 3, 128, 96), 100.)
  torch.onnx.export(wrapper, dummy, str(path), input_names=['strip_ycbcr'], output_names=['line_type', 'color'],
                    dynamic_axes={'strip_ycbcr': {0: 'n'}, 'line_type': {0: 'n'}, 'color': {0: 'n'}}, opset_version=14)
  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
  x = (torch.rand(8, 3, 128, 96)*255)
  ref = [o.detach().numpy() for o in wrapper(x)]
  got = sess.run(None, {'strip_ycbcr': x.numpy()})
  return float(max(np.abs(a - b).max() for a, b in zip(ref, got, strict=True)))


def main(a):
  device = 'cuda'
  out = Path(a.output)
  out.mkdir(parents=True, exist_ok=True)
  rows, x = load_set(a.data)
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  assert not {rows[i]['route'] for i in tr} & {rows[i]['route'] for i in va}
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  report = {'data': str(a.data), 'train_frames': len(tr), 'validation_frames': len(va), 'runs': []}
  best = None
  for arch in a.arch:
    for seed in a.seeds:
      net, info = train(arch, seed, train_rows, x[tr], val_rows, x[va], device, a.epochs,
                        {'pplcnet025': Path(a.pretrained), 'mobilenetv3s': Path(a.mbv3)}, a.aug)
      pt, pc = predict(net, tensors(val_rows, x[va], device)[0])
      run = {'arch': arch, 'seed': seed, 'aug': a.aug, **info, 'validation': metrics(pt, pc, val_rows)}
      report['runs'].append(run)
      print(arch, seed, 'best epoch', info['best_epoch'], 'val score', round(info['best_val_score'], 4), 'params', info['params'], flush=True)
      if best is None or info['best_val_score'] > best[0]:
        best = (info['best_val_score'], arch, seed, net, pt, pc)
  _, arch, seed, net, pt, pc = best
  threshold = choose_threshold(pt, pc, val_rows)
  human_val = [i for i, r in enumerate(val_rows) if r['origin'] == 'human']
  report['selected'] = {'arch': arch, 'seed': seed, 'aug': a.aug, 'threshold': threshold, 'selection': 'validation macro recall (type+colour)/2',
                        'validation_at_threshold': metrics(pt, pc, val_rows, threshold),
                        'validation_human_only_at_threshold': metrics(pt[human_val], pc[human_val], [val_rows[i] for i in human_val], threshold)}
  torch.save(net.state_dict(), out/'model.pt')
  onnx_path = out/'lane_strip.onnx'
  report['selected']['onnx_parity_max_abs'] = export(net, onnx_path)
  report['selected']['onnx_bytes'] = onnx_path.stat().st_size
  report['selected']['onnx_sha256'] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
  report['selected']['params'] = sum(p.numel() for p in net.parameters())
  (out/'train-report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
  (out/'model-contract.json').write_text(json.dumps({'input': 'strip_ycbcr float32 N x 3 x 128 x 96, raw 0..255 (Y, Cb, Cr), rows far(20m)->near(4.5m)',
    'strip': 'tools/lane_profile/scanner.py strip_geometry+sample, rows INTER_AREA resized to 128', 'outputs': {'line_type': TYPES, 'color': COLORS},
    'threshold': threshold, 'yellow_dashed': 'not a learned class: dashed+yellow must be reported as unknown',
    'arch': arch, 'aug': a.aug, 'trained_on': 'train routes only; selected on validation routes; test never used'}, indent=1), encoding='utf-8')
  print(json.dumps(report['selected'], indent=1)[:3000])


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--arch', nargs='+', default=['stripnet', 'pplcnet025'])
  p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
  p.add_argument('--epochs', type=int, default=40)
  p.add_argument('--aug', choices=('base', 'plus'), default='base')
  p.add_argument('--mbv3', default='.analysis/lane-model-comparison-20261018/pretrained/mobilenet_v3_small-047dcff4.pth')
  p.add_argument('--pretrained', default='.analysis/lane-model-comparison-20261018/pretrained/PPLCNet_x0_25_pretrained.pdparams')
  main(p.parse_args())
