"""Train a lightweight lane type/colour classifier on the 4.5-20 m row-aligned strip (PC/GPU only).

Input: YCbCr uint8 strip 128 x 96 (rows far->near, metric columns, +-0.75 m) from build_dataset.py, or 128 x 128 (+-1.0 m,
same 1.56 cm/col: lane-retrain-w100-20261021); the width is taken from the dataset and recorded in model-contract.json 'strip'. Labels: interval
labels (human > scanner auto > scanner pool > AI review) with per-origin weights. Classes: type dashed/solid,
colour white/yellow. Yellow dashed has no training data and is NOT a learned class (dashed+yellow is never
claimed; downstream must treat it as unknown).

Model selection and the abstention threshold use the validation routes only. The held-out human test set is
evaluated once, after selection, by evaluate(); its result never feeds back into training.
The exported ONNX takes the raw strip (N,3,128,W float in 0..255) and contains normalisation and softmax.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.lane_profile.dataset_io import COLORS_V1 as COLORS, TYPES_V1 as TYPES, check_labels, load_set  # noqa: TID251  (2-class heads)
from tools.lane_profile import dataset_io  # noqa: TID251  (StripNet2: the extended vocabulary)


NORMS = ('fixed', 'road', 'road-y')
ROAD_ROWS, ROAD_COLS = 64, 16  # road reference: near half (rows 64-127 = ~4.5-10 m) x outer 16 columns each side (0.5-0.75 m; 0.75-1.0 m on a 128-column strip)
COL_M = 1.5/96  # strip column pitch (m): 96 columns over +-0.75 m, 128 over +-1.0 m


class Normalize(nn.Module):
  """Raw YCbCr strip -> network input. Chroma is centred and amplified (lane colour is a small offset).

  mode 'fixed' (default, unchanged): Y (Y-110)/50, Cb/Cr (C-128)/16.
  mode 'road': Cb/Cr relative to the strip's own road, (C - road C)/8, road C = mean over rows ROAD_ROWS: and the outer
  ROAD_COLS columns on both sides (the boundary runs down the centre columns); Y as 'fixed'.
  mode 'road-y': as 'road' and Y relative to the strip mean, (Y - mean Y)/50.
  Only slicing / mean / sub / div: exported inside the ONNX, so training and on-device inference use the same op."""
  def __init__(self, rgb_imagenet=False, mode='fixed'):
    super().__init__()
    if mode not in NORMS or (rgb_imagenet and mode != 'fixed'):
      raise ValueError(f'normalisation {mode!r}')
    self.rgb = rgb_imagenet
    self.mode = mode

  def forward(self, x):
    if not self.rgb:
      if self.mode == 'fixed':
        return torch.cat(((x[:, :1] - 110.)/50., (x[:, 1:] - 128.)/16.), 1)
      road = torch.cat((x[:, 1:, ROAD_ROWS:, :ROAD_COLS], x[:, 1:, ROAD_ROWS:, -ROAD_COLS:]), 3).mean((2, 3), keepdim=True)
      y = x[:, :1] - (x[:, :1].mean((2, 3), keepdim=True) if self.mode == 'road-y' else 110.)
      return torch.cat((y/50., (x[:, 1:] - road)/8.), 1)
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
    f = self.body(self.stem(x))                     # N, C, 8, 6 (8, 8 for a 128-column strip)
    prof = self.profile(f).mean(3).flatten(1)        # along-road profile, lateral pooled
    z = self.fc(torch.cat((prof, f.mean((2, 3))), 1))
    return self.type_head(z), self.color_head(z)


ARCH2 = ('stripnet2', 'stripnet2-nolat')  # 2026-10-21 wide strip round (manifest v4): colour 3 + type 5 heads


class StripNet2(nn.Module):
  """StripNet body for the +-1.0 m / 128-column strip with the last stride-2 block strided (2, 1) only: 8 along-road x 16
  lateral cells (12.5 cm each). Head input = along-road profile (16 x 8, lateral mean) + lateral profile (16 x 16, along-road
  mean: where across the strip the paint is, i.e. near / far for a double line) + global mean. Outputs (colour, type) logits:
  colour [white, yellow, blue], type [dashed, solid, dbl_near, dbl_far, unknown] (dataset_io.COLORS / TYPES).
  lateral=False ('stripnet2-nolat' ablation): no lateral profile."""
  def __init__(self, width=1., lateral=True):
    super().__init__()
    c = [int(round(v*width)) for v in (16, 24, 32, 48, 64)]
    self.lateral = lateral
    self.stem = nn.Sequential(nn.Conv2d(3, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
    self.body = nn.Sequential(dw(c[0], c[1], 2), dw(c[1], c[2], 2), dw(c[2], c[3], (2, 1)), dw(c[3], c[4], 1))
    self.profile = nn.Conv2d(c[4], 16, 1)
    self.fc = nn.Sequential(nn.Linear(16*8 + (16*16 if lateral else 0) + c[4], 64), nn.ReLU(inplace=True), nn.Dropout(.2))
    self.color_head, self.type_head = nn.Linear(64, len(dataset_io.COLORS)), nn.Linear(64, len(dataset_io.TYPES))

  def forward(self, x):
    f = self.body(self.stem(x))                     # N, C, 8, 16 for a 128 x 128 strip
    p = self.profile(f)
    feats = [p.mean(3).flatten(1)]                   # along-road profile (lateral pooled)
    if self.lateral:
      feats.append(p.mean(2).flatten(1))             # lateral profile (along-road pooled)
    z = self.fc(torch.cat(feats + [f.mean((2, 3))], 1))
    return self.type_head(z), self.color_head(z)     # (type, colour) like Net / StripNet; Export2 reorders to (color, type)


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
  def __init__(self, arch, norm='fixed'):
    super().__init__()
    self.arch = arch
    self.norm = Normalize(rgb_imagenet=arch in ('pplcnet025', 'pplcnet050', 'mobilenetv3s'), mode=norm)
    if arch == 'stripnet':
      self.model = StripNet()
    elif arch in ARCH2:
      self.model = StripNet2(lateral=arch != 'stripnet2-nolat')
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


LAST_FLIP = [None]


def augment(x, gen):
  """x: float N,3,H,W raw YCbCr. Photometric + geometric jitter that keeps the label valid."""
  n = x.shape[0]
  flip = torch.rand(n, device=x.device, generator=gen) < .5
  x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)            # left/right boundary symmetry
  LAST_FLIP[0] = flip  # StripNet2: dbl_near <-> dbl_far on the flipped samples (read by train2; draws unchanged)
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
    w = x.shape[3]
    x = F.pad(x, (max(shift, 0), max(-shift, 0), 0, 0), mode='replicate')[..., max(-shift, 0):max(-shift, 0) + w]
  noise = torch.randn((n, 1) + x.shape[2:], device=dev, generator=gen)*3*r(n, 1, 1, 1)  # luma only
  return torch.cat((x[:, :1] + noise, x[:, 1:]), 1).clamp(0, 255)


RAIN_P = {'desaturate': .5, 'darken': .5, 'blue': .3, 'red_blob': .3}  # per-sample probabilities (independent)
RAIN_P_LITE = {**RAIN_P, 'blue': .15, 'red_blob': .15}  # plus-rain-lite: blue cast / red ellipse halved


def augment_plus_rain(x, gen, p=None):
  """augment_plus() (incl. augment()'s 50 % left/right flip) plus rain/night photometrics; labels unchanged.

  desaturate: Cb/Cr pulled toward 128 by 0.3-1.0x (wet night paint is nearly grey); darken: Y x0.4-1.0 + luma noise
  sigma 0-4; blue cast: Cb +2..+8 (wet road at night); red reflection: one soft ellipse (rows 4-20, cols 3-12 radius,
  anywhere in the strip) with Cr +20..+50 at its centre (tail/brake light and its reflection; colour label unchanged:
  the boundary under it keeps its own colour). Every draw is made for every sample so the stream is seed-deterministic.
  p: probabilities (default RAIN_P; RAIN_P_LITE for plus-rain-lite). The draw order does not depend on p."""
  p = RAIN_P if p is None else p
  x = augment_plus(x, gen)
  n, dev = x.shape[0], x.device
  r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
  y, cb, cr = x[:, :1], x[:, 1:2], x[:, 2:]
  m, k = (r(n) < p['desaturate']).view(-1, 1, 1, 1), .3 + .7*r(n, 1, 1, 1)
  cb, cr = torch.where(m, (cb - 128)*k + 128, cb), torch.where(m, (cr - 128)*k + 128, cr)
  m, g, sd = (r(n) < p['darken']).view(-1, 1, 1, 1), .4 + .6*r(n, 1, 1, 1), 4*r(n, 1, 1, 1)
  y = torch.where(m, y*g + torch.randn(y.shape, device=dev, generator=gen)*sd, y)
  m, b = (r(n) < p['blue']).view(-1, 1, 1, 1), 2 + 6*r(n, 1, 1, 1)
  cb = torch.where(m, cb + b, cb)
  m = (r(n) < p['red_blob']).view(-1, 1, 1, 1)
  h, w = x.shape[2], x.shape[3]
  cy, cx, ry, rx, amp = r(n, 1, 1, 1)*h, r(n, 1, 1, 1)*w, 4 + 16*r(n, 1, 1, 1), 3 + 9*r(n, 1, 1, 1), 20 + 30*r(n, 1, 1, 1)
  rows = torch.arange(h, device=dev, dtype=x.dtype).view(1, 1, h, 1)
  cols = torch.arange(w, device=dev, dtype=x.dtype).view(1, 1, 1, w)
  blob = (1 - ((rows - cy)/ry)**2 - ((cols - cx)/rx)**2).clamp(0, 1)
  cr = cr + blob*amp*m
  return torch.cat((y, cb, cr), 1).clamp(0, 255)


def augment_plus_rain_lite(x, gen):
  return augment_plus_rain(x, gen, RAIN_P_LITE)


MORE_P = {'occlude': .3, 'drop_blur': .3, 'low_contrast': .3}  # plus-rain-more extras, per-sample probabilities (independent)
DROP_SIGMAS = (1., 2., 3.)  # drop blur radius 2-6 px = 2 sigma, quantised to sigma 1/2/3 (one blurred copy per sigma per batch)


def _gauss(x, sigma):
  k = int(2*round(3*sigma) + 1)
  g = torch.exp(-(torch.arange(k, device=x.device, dtype=x.dtype) - k//2)**2/(2*sigma**2))
  g = g/g.sum()
  c = x.shape[1]
  x = F.conv2d(F.pad(x, (k//2, k//2, 0, 0), mode='replicate'), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
  return F.conv2d(F.pad(x, (0, 0, k//2, k//2), mode='replicate'), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def augment_plus_rain_more(x, gen, p=None):
  """augment_plus_rain() plus rain-drop / spray effects; labels unchanged. Applied twice per sample per epoch (AUG_REPEAT).

  low contrast (30 %): Y = mean + (Y - mean)*0.5-1.0 (spray haze); drop blur (30 %): 2-4 soft discs (radius 6-14 px) where the
  strip is replaced by a Gaussian blur of radius 2-6 px (sigma 1/2/3); occlusion (30 %): one horizontal band (height 8-24 px,
  full width) or one disc (radius 6-16 px), 50/50, filled with the road colour = per-channel median of the bottom 16 rows
  (wiper / drop / spray covering part of the boundary; the rest of the strip still carries the label).
  Every draw is made for every sample so the stream is seed-deterministic."""
  x = augment_plus_rain(x, gen, p)
  n, dev = x.shape[0], x.device
  h, w = x.shape[2], x.shape[3]
  r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
  rows = torch.arange(h, device=dev, dtype=x.dtype).view(1, 1, h, 1)
  cols = torch.arange(w, device=dev, dtype=x.dtype).view(1, 1, 1, w)
  # (c) contrast reduction
  m, k = (r(n) < MORE_P['low_contrast']).view(-1, 1, 1, 1), .5 + .5*r(n, 1, 1, 1)
  y = x[:, :1]
  mean = y.mean((2, 3), keepdim=True)
  x = torch.cat((torch.where(m, (y - mean)*k + mean, y), x[:, 1:]), 1)
  # (b) local drop blur
  m = (r(n) < MORE_P['drop_blur']).view(-1, 1, 1, 1)
  count = 2 + (r(n)*3).long().clamp(max=2)                       # 2-4 discs
  level = (r(n)*len(DROP_SIGMAS)).long().clamp(max=len(DROP_SIGMAS) - 1)  # radius 2-6 px -> sigma 1/2/3
  cy, cx, rad = r(n, 4, 1, 1)*h, r(n, 4, 1, 1)*w, 6 + 8*r(n, 4, 1, 1)
  d = ((rows.view(1, 1, h, 1) - cy.view(n, 4, 1, 1))**2 + (cols.view(1, 1, 1, w) - cx.view(n, 4, 1, 1))**2).sqrt()
  on = (torch.arange(4, device=dev).view(1, 4) < count.view(-1, 1)).view(n, 4, 1, 1)
  alpha = ((1.5 - d/rad.view(n, 4, 1, 1)).clamp(0, 1)*on).amax(1, keepdim=True)*m  # soft edge, 1 inside 0.5 r
  if bool(m.any()):
    xb = torch.stack([_gauss(x, s) for s in DROP_SIGMAS])[level, torch.arange(n, device=dev)]
    x = x*(1 - alpha) + xb*alpha
  # (a) occlusion with the road colour
  m = (r(n) < MORE_P['occlude']).view(-1, 1, 1, 1)
  band = (r(n) < .5).view(-1, 1, 1, 1)
  start, length = r(n, 1, 1, 1)*h, 8 + 16*r(n, 1, 1, 1)
  oy, ox, orad = r(n, 1, 1, 1)*h, r(n, 1, 1, 1)*w, 6 + 10*r(n, 1, 1, 1)
  band_m = (rows >= start) & (rows < start + length) & (cols >= 0)
  disc_m = ((rows - oy)**2 + (cols - ox)**2) <= orad**2
  cover = torch.where(band, band_m, disc_m) & m
  road = x[:, :, -16:, :].flatten(2).median(2).values.view(n, 3, 1, 1)
  x = torch.where(cover, road, x)
  return x.clamp(0, 255)


RAIN_P_NORED = {**RAIN_P, 'red_blob': 0.}  # plus-rain-more-nored: no red ellipse (d-aug-more read a stopped car's tail light as yellow)


def augment_plus_rain_more_nored(x, gen):
  """augment_plus_rain_more() with the red-ellipse draw disabled (same draw order, so only that effect changes)."""
  return augment_plus_rain_more(x, gen, p=RAIN_P_NORED)


AUGMENTS = {'base': augment, 'plus': augment_plus, 'plus-rain': augment_plus_rain, 'plus-rain-lite': augment_plus_rain_lite,
            'plus-rain-more': augment_plus_rain_more, 'plus-rain-more-nored': augment_plus_rain_more_nored}
AUGMENTS['plus-rain-more-nored-occ1'] = augment_plus_rain_more_nored  # 2026-10-21 (407 round): same draws incl. the road-colour occlusion, AUG_REPEAT 1
AUG_REPEAT = {'plus-rain-more': 2, 'plus-rain-more-nored': 2}  # passes over the training set per epoch (each pass: new order, new augmentation draws)


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


def train(arch, seed, train_rows, xtr, val_rows, xva, device, epochs, pretrained, aug='base', norm='fixed', lr_scale=1.):
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  net = Net(arch, norm).to(device)
  if arch in ('pplcnet025', 'pplcnet050'):
    net.model.load_pretrained(pretrained[arch])
  elif arch == 'mobilenetv3s':
    net.model.load_pretrained(pretrained['mobilenetv3s'])
  augment_fn = AUGMENTS[aug]
  X, T, C, W = tensors(train_rows, xtr, device)
  Xv, _, _, _ = tensors(val_rows, xva, device)
  # class-balanced weights from TRAIN only
  wt = torch.tensor([len(T)/(2*max(1, int((T == k).sum()))) for k in range(2)], device=device)
  wc = torch.tensor([len(C)/(2*max(1, int((C == k).sum()))) for k in range(2)], device=device)
  lr = (2e-3 if arch == 'stripnet' else 1e-3)*lr_scale  # lr_scale: OneCycle max lr factor (e-long: 0.5)
  opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
  reps = AUG_REPEAT.get(aug, 1)
  steps = epochs*reps*math.ceil(len(X)/256)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.15)
  best, history = None, []
  start = time.monotonic()
  for epoch in range(epochs):
    net.train()
    order = torch.cat([torch.randperm(len(X), device=device, generator=gen) for _ in range(reps)])
    total = 0.
    for i in range(0, len(order), 256):
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
    history.append({'epoch': epoch, 'train_loss': total/len(order), 'val_type_macro_recall': m['type']['macro_recall_raw'],
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


def export(net, path, cols=96, rows=128):
  net = net.cpu().eval()
  wrapper = Export(net).eval()  # export restores the module's previous mode; keep it eval for the parity check
  dummy = torch.full((2, 3, rows, cols), 100.)
  torch.onnx.export(wrapper, dummy, str(path), input_names=['strip_ycbcr'], output_names=['line_type', 'color'],
                    dynamic_axes={'strip_ycbcr': {0: 'n'}, 'line_type': {0: 'n'}, 'color': {0: 'n'}}, opset_version=14)
  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
  x = (torch.rand(8, 3, rows, cols)*255)
  ref = [o.detach().numpy() for o in wrapper(x)]
  got = sess.run(None, {'strip_ycbcr': x.numpy()})
  return float(max(np.abs(a - b).max() for a, b in zip(ref, got, strict=True)))


def main(a):
  device = a.device
  out = Path(a.output)
  out.mkdir(parents=True, exist_ok=True)
  rows, x = load_set(a.data)
  check_labels(rows, TYPES, COLORS)  # the extended vocabulary (dbl_near/dbl_far/unknown, blue) is not a 2-class head label
  srows, scols = x.shape[1:3]
  strip = {'half_m': round(scols*COL_M/2, 6), 'cols': int(scols), 'rows': int(srows)}
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  if a.limit:  # smoke test: evenly spaced subsets
    tr = [tr[j] for j in np.linspace(0, len(tr) - 1, min(a.limit, len(tr))).astype(int)]
    va = [va[j] for j in np.linspace(0, len(va) - 1, min(a.limit, len(va))).astype(int)]
  assert not {rows[i]['route'] for i in tr} & {rows[i]['route'] for i in va}
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  report = {'data': str(a.data), 'train_frames': len(tr), 'validation_frames': len(va), 'strip': strip, 'limit': a.limit, 'runs': []}
  best = None
  for arch in a.arch:
    for seed in a.seeds:
      net, info = train(arch, seed, train_rows, x[tr], val_rows, x[va], device, a.epochs,
                        {'pplcnet025': Path(a.pretrained), 'mobilenetv3s': Path(a.mbv3)}, a.aug, a.norm)
      pt, pc = predict(net, tensors(val_rows, x[va], device)[0])
      run = {'arch': arch, 'seed': seed, 'aug': a.aug, 'norm': a.norm, **info, 'validation': metrics(pt, pc, val_rows)}
      report['runs'].append(run)
      print(arch, seed, 'best epoch', info['best_epoch'], 'val score', round(info['best_val_score'], 4), 'params', info['params'], flush=True)
      if best is None or info['best_val_score'] > best[0]:
        best = (info['best_val_score'], arch, seed, net, pt, pc)
  _, arch, seed, net, pt, pc = best
  threshold = choose_threshold(pt, pc, val_rows)
  human_val = [i for i, r in enumerate(val_rows) if r['origin'] == 'human']
  report['selected'] = {'arch': arch, 'seed': seed, 'aug': a.aug, 'norm': a.norm, 'threshold': threshold,
                        'selection': 'validation macro recall (type+colour)/2',
                        'validation_at_threshold': metrics(pt, pc, val_rows, threshold),
                        'validation_human_only_at_threshold': metrics(pt[human_val], pc[human_val], [val_rows[i] for i in human_val], threshold)}
  torch.save(net.state_dict(), out/'model.pt')
  onnx_path = out/'lane_strip.onnx'
  report['selected']['onnx_parity_max_abs'] = export(net, onnx_path, scols, srows)
  report['selected']['onnx_bytes'] = onnx_path.stat().st_size
  report['selected']['onnx_sha256'] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
  report['selected']['params'] = sum(p.numel() for p in net.parameters())
  (out/'train-report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
  (out/'model-contract.json').write_text(json.dumps({
    'input': f'strip_ycbcr float32 N x 3 x {srows} x {scols}, raw 0..255 (Y, Cb, Cr), rows far(20m)->near(4.5m)',
    'strip_geometry': strip,  # -> lane_strip.json 'strip' (absent = 0.75 / 96 / 128)
    'strip': 'tools/lane_profile/scanner.py strip_geometry+sample, rows INTER_AREA resized to 128', 'outputs': {'line_type': TYPES, 'color': COLORS},
    'threshold': threshold, 'yellow_dashed': 'not a learned class: dashed+yellow must be reported as unknown',
    'arch': arch, 'aug': a.aug, 'norm': a.norm, 'trained_on': 'train routes only; selected on validation routes; test never used'}, indent=1), encoding='utf-8')
  print(json.dumps(report['selected'], indent=1)[:3000])


# ---------------------------------------------------------------------------------------------------------------------------
# StripNet2 (manifest v4 draft, 2026-10-21 wide strip round). Labels: dataset_io.TYPES / COLORS (extended vocabulary);
# 2-class sets map unchanged (dashed/solid, white/yellow). Right strips mirrored in the loader (load_set mirror_right) so the
# ego side is the high column index; augment()'s random flip swaps dbl_near <-> dbl_far. type 'unknown': colour loss masked.
T2, C2 = dataset_io.TYPES, dataset_io.COLORS
DASHED, SOLID, DBL_NEAR, DBL_FAR, UNKNOWN = range(5)
WHITE, YELLOW, BLUE = range(3)
RARE_CAP = 3.  # rare-class (dbl_near / dbl_far / unknown, blue) loss weight <= 3 x the mean weight of the common classes
SYNTH = {'shift': (17, 38), 'shift_yellow': (9, 24), 'period_m': (8., 18.), 'duty': (.35, .55), 'lum': .15,
         'min_peak': 15., 'width': (3, 20), 'margin': 2, 'min_gap': 4}
# (dCb, dCr) blue paint - white paint relative to the road, measured on dataset-dbl-w100
# (blue +8.7..9.8 / -14.1..-14.3, white -2.6..0.9 / -2.7..0.7)
BLUE_SHIFT = (10., -14.)
NEAR_M, FAR_M = 4.5, 20.


def row_distance(h, device):
  """metres of each strip row (row 0 = 20 m, last = 4.5 m; rows are image rows, i.e. ~linear in 1/distance)."""
  r = torch.arange(h, device=device, dtype=torch.float32)/(h - 1)
  return 1/(1/FAR_M + r*(1/NEAR_M - 1/FAR_M))


def synth_double(x, t, c, w, gen, p):
  """Synthetic double line from a confidently labelled (weight >= 0.7) single-line strip (raw YCbCr, mirrored: high column =
  near). The stripe columns come from the luma column profile (columns above half the peak, within 12 of the argmax); the
  stripe band (+2 columns margin, per-pixel paint alpha = (Y - row median)/peak, so the road under it is not copied) is copied
  17-38 columns (27-60 cm) toward near (+) or far (-). White solid -> the copy gets metric dash gaps (period 8-18 m, duty
  0.35-0.55) -> dbl_near if the copy is on the near side else dbl_far; white dashed -> the copy is filled solid (column paint
  colour along every row, luma following the road brightness) -> the dashed original is far (dbl_far) if the copy is near, else
  dbl_near. Yellow solid -> a solid copy 9-24 columns away (yellow solid+solid; label stays solid). Copy luma x0.85-1.15,
  width jitter (dilate / erode 1 column / none), 50 % lateral 3-tap alpha blur. Discarded (sample unchanged) when there is no
  clear stripe, the copy would leave the strip or land on existing paint. Every draw is made for every sample (seeded stream).
  -> (x, t) with the changed samples' type label."""
  n, _, h, wd = x.shape
  dev = x.device
  r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
  pick = r(n) < p
  side = (r(n) < .5).long()*-2 + 1                              # -1 far / +1 near
  lo, hi = SYNTH['shift']
  d_white = lo + (r(n)*(hi - lo + 1)).long().clamp(max=hi - lo)
  lo, hi = SYNTH['shift_yellow']
  d_yel = lo + (r(n)*(hi - lo + 1)).long().clamp(max=hi - lo)
  period = SYNTH['period_m'][0] + (SYNTH['period_m'][1] - SYNTH['period_m'][0])*r(n)
  duty = SYNTH['duty'][0] + (SYNTH['duty'][1] - SYNTH['duty'][0])*r(n)
  phase = r(n)*period
  lum = 1 + (r(n) - .5)*2*SYNTH['lum']
  wop = (r(n)*3).long().clamp(max=2)
  blur = r(n) < .5
  elig = pick & (w >= .7) & (((t == DASHED) | (t == SOLID)) & (c == WHITE) | ((t == SOLID) & (c == YELLOW)))
  if not bool(elig.any()):
    return x, t
  idx = elig.nonzero().squeeze(1)
  k = len(idx)
  xs = x[idx]
  Y = xs[:, 0]
  base = Y.median(2, keepdim=True).values                      # k, h, 1 road luma per row
  dy = Y - base
  prof = dy.mean(1)                                            # k, wd
  pmax, am = prof.max(1)
  cols = torch.arange(wd, device=dev)
  near_am = (cols[None] - am[:, None]).abs() <= 12
  m = (prof > .5*pmax[:, None]) & near_am
  c0 = torch.where(m, cols[None], wd).amin(1)
  c1 = torch.where(m, cols[None], -1).amax(1)
  peak = dy.gather(2, am.view(k, 1, 1).expand(k, h, 1)).squeeze(2).quantile(.9, dim=1)   # paint level (dashed: paint rows)
  width = c1 - c0 + 1
  yellow = c[idx] == YELLOW
  d = torch.where(yellow, d_yel[idx], d_white[idx])
  s = side[idx]
  shift = s*d
  mg = SYNTH['margin']
  b0, b1 = c0 - mg + shift, c1 + mg + shift
  ok = (peak >= SYNTH['min_peak']) & (width >= SYNTH['width'][0]) & (width <= SYNTH['width'][1]) & (b0 >= 0) & (b1 <= wd - 1) & \
       (d - width >= SYNTH['min_gap'])
  dst_band = (cols[None] >= b0[:, None]) & (cols[None] <= b1[:, None])
  ok &= torch.where(dst_band, prof, torch.full_like(prof, -1e3)).amax(1) < .3*pmax           # lands on bare road
  src = cols[None] - shift[:, None]                            # k, wd: source column of each destination column
  inband = (src >= (c0 - mg)[:, None]) & (src <= (c1 + mg)[:, None])
  srcc = src.clamp(0, wd - 1)
  alpha = (dy/peak.clamp(min=1).view(k, 1, 1)).clamp(0, 1)
  alpha = alpha.gather(2, srcc[:, None, :].expand(k, h, wd))*inband[:, None, :]
  val = xs.gather(3, srcc[:, None, None, :].expand(k, 3, h, wd))
  dist = row_distance(h, dev)
  on = (((dist[None] + phase[idx, None]) % period[idx, None]) < duty[idx, None]*period[idx, None]).float()   # k, h
  src_solid = t[idx] == SOLID
  to_dashed = src_solid & ~yellow
  to_solid = ~src_solid
  # dashed -> solid: per-column paint colour / level from the paint rows, every row (luma follows the road brightness)
  aw = alpha.sum(1, keepdim=True).clamp(min=1e-3)              # k, 1, wd
  colval = (val*alpha[:, None]).sum(2, keepdim=True)/aw[:, None]      # k, 3, 1, wd
  colalpha = (alpha**2).sum(1, keepdim=True)/aw                 # k, 1, wd
  ratio = (base/base.mean(1, keepdim=True).clamp(min=1)).clamp(.5, 1.5)   # k, h, 1
  solid_val = colval.expand(k, 3, h, wd).clone()
  solid_val[:, 0] = solid_val[:, 0]*ratio
  val = torch.where(to_solid.view(k, 1, 1, 1), solid_val, val)
  alpha = torch.where(to_solid.view(k, 1, 1), colalpha.expand(k, h, wd), alpha)
  alpha = torch.where(to_dashed.view(k, 1, 1), alpha*on[:, :, None], alpha)
  dil, ero = torch.maximum(alpha, torch.maximum(alpha.roll(1, 2), alpha.roll(-1, 2))), torch.minimum(alpha, alpha.roll(1, 2))
  alpha = torch.where((wop[idx] == 1).view(k, 1, 1), dil, torch.where((wop[idx] == 2).view(k, 1, 1), ero, alpha))
  ab = F.pad(alpha[:, None], (1, 1, 0, 0), mode='replicate')
  ab = (ab[..., :-2]*.25 + ab[..., 1:-1]*.5 + ab[..., 2:]*.25)[:, 0]
  alpha = torch.where(blur[idx].view(k, 1, 1), ab, alpha)*dst_band[:, None, :]
  val = torch.cat((val[:, :1]*lum[idx].view(k, 1, 1, 1), val[:, 1:]), 1)
  out = (xs + alpha[:, None]*(val - xs)).clamp(0, 255)
  # label: the dashed element on the near (high column) side -> dbl_near
  near_copy = s > 0
  full = lambda v: torch.full_like(t[idx], v)  # noqa: E731
  nt = torch.where(to_dashed, torch.where(near_copy, full(DBL_NEAR), full(DBL_FAR)),
                   torch.where(to_solid, torch.where(near_copy, full(DBL_FAR), full(DBL_NEAR)), t[idx]))
  x, t = x.clone(), t.clone()
  sel = idx[ok]
  x[sel], t[sel] = out[ok], nt[ok]
  return x, t


def blue_hue(x, t, c, gen, p):
  """Blue bus-lane paint from a white-line strip: paint alpha (Y - row median)/p90 peak, Cb/Cr += alpha * BLUE_SHIFT * 0.6-1.4,
  paint luma contrast x0.8-1.0 (blue paint is darker than white); colour label -> blue. Eligible: white, type not unknown.
  Every draw is made for every sample. -> (x, c)."""
  n, _, h, wd = x.shape
  dev = x.device
  r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
  pick = r(n) < p
  scale, dark = .6 + .8*r(n), .8 + .2*r(n)
  elig = pick & (c == WHITE) & (t != UNKNOWN)
  if not bool(elig.any()):
    return x, c
  idx = elig.nonzero().squeeze(1)
  k = len(idx)
  xs = x[idx]
  Y = xs[:, 0]
  base = Y.median(2, keepdim=True).values
  dy = Y - base
  peak = dy.flatten(1).quantile(.98, dim=1).clamp(min=10).view(k, 1, 1)
  alpha = (dy/peak).clamp(0, 1)
  sc = scale[idx].view(k, 1, 1)
  y = base + dy*torch.where(alpha > .3, dark[idx].view(k, 1, 1), torch.ones_like(alpha))
  cb = xs[:, 1] + alpha*BLUE_SHIFT[0]*sc
  cr = xs[:, 2] + alpha*BLUE_SHIFT[1]*sc
  x, c = x.clone(), c.clone()
  x[idx] = torch.stack((y, cb, cr), 1).clamp(0, 255)
  c[idx] = BLUE
  return x, c


class Export2(nn.Module):
  """StripNet2 ONNX graph (manifest v4 contract): raw strip -> LOGITS (color (N,3) [white, yellow, blue], type (N,5) [dashed, solid,
  dbl_near, dbl_far, unknown]); the consumer applies the softmax (strip_model.StripModel.decide4). Normalisation is inside."""
  def __init__(self, net):
    super().__init__()
    self.net = net

  def forward(self, x):
    t, c = self.net(x)
    return c, t


def export2(net, path, cols=128, rows=128):
  net = net.cpu().eval()
  wrapper = Export2(net).eval()
  torch.onnx.export(wrapper, torch.full((2, 3, rows, cols), 100.), str(path), input_names=['strip_ycbcr'], output_names=['color', 'type'],
                    dynamic_axes={'strip_ycbcr': {0: 'n'}, 'color': {0: 'n'}, 'type': {0: 'n'}}, opset_version=14)
  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
  assert [o.name for o in sess.get_outputs()] == ['color', 'type']
  x = torch.rand(8, 3, rows, cols)*255
  ref = [o.detach().numpy() for o in wrapper(x)]
  got = sess.run(None, {'strip_ycbcr': x.numpy()})
  return float(max(np.abs(a - b).max() for a, b in zip(ref, got, strict=True)))


def tensors2(rows, x, device):
  xt = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
  t = torch.tensor([T2.index(r['type']) for r in rows], device=device)
  c = torch.tensor([C2.index(r['color']) for r in rows], device=device)
  w = torch.tensor([r['weight'] for r in rows], dtype=torch.float32, device=device)
  return xt, t, c, w


def class_weights(y, k, common, ignore=None):
  """balanced N/(K n) over the classes present; rare classes (not in common) capped at RARE_CAP x the common classes' mean."""
  y = y[y != ignore] if ignore is not None else y
  cnt = torch.stack([(y == i).sum() for i in range(k)]).float()
  present = cnt > 0
  w = torch.where(present, len(y)/(int(present.sum())*cnt.clamp(min=1)), torch.ones_like(cnt))
  cap = RARE_CAP*w[list(common)].mean()
  rare = torch.tensor([i not in common for i in range(k)], device=y.device)
  return torch.where(rare, torch.minimum(w, cap), w)


def f1s(truth, pred, k):
  """per-class F1 (None without support) and macro F1 over the classes with support."""
  out = []
  for i in range(k):
    tp, fp, fn = int(((pred == i) & (truth == i)).sum()), int(((pred == i) & (truth != i)).sum()), int(((pred != i) & (truth == i)).sum())
    out.append(None if tp + fn == 0 else 2*tp/max(1, 2*tp + fp + fn))
  vals = [v for v in out if v is not None]
  return out, (float(np.mean(vals)) if vals else 0.)


def metrics2(pt, pc, t, c):
  """pt / pc: softmax (N,5) / (N,3) numpy; t / c: label ids (c = -100 on unknown type)."""
  ptd, pcd = pt.argmax(1), pc.argmax(1)
  cm = c >= 0
  tf, tmac = f1s(t, ptd, len(T2))
  cf, cmac = f1s(c[cm], pcd[cm], len(C2))
  conf_t = np.zeros((len(T2), len(T2)), int)
  np.add.at(conf_t, (t, ptd), 1)
  conf_c = np.zeros((len(C2), len(C2)), int)
  np.add.at(conf_c, (c[cm], pcd[cm]), 1)
  return {'type_acc': float((ptd == t).mean()), 'color_acc': float((pcd[cm] == c[cm]).mean()) if cm.any() else None,
          'type_macro_f1': tmac, 'color_macro_f1': cmac, 'type_f1': dict(zip(T2, tf, strict=True)), 'color_f1': dict(zip(C2, cf, strict=True)),
          'type_confusion': conf_t.tolist(), 'color_confusion': conf_c.tolist(), 'score': (tmac + cmac)/2}


@torch.no_grad()
def predict2(net, xt, batch=1024):
  net.eval()
  pt, pc = [], []
  for i in range(0, len(xt), batch):
    t, c = net(xt[i:i + batch])
    pt.append(t.softmax(1))
    pc.append(c.softmax(1))
  return torch.cat(pt).cpu().numpy(), torch.cat(pc).cpu().numpy()


def plot_history(hist, path):
  """history.png: loss (top) and macro-F1 (bottom) per epoch; matplotlib when available, else a plain numpy-drawn PNG."""
  ep = [h['epoch'] for h in hist]
  series = {'train_loss': 'tab:blue', 'val_loss': 'tab:orange'}, {'val_type_macro_f1': 'tab:green', 'val_color_macro_f1': 'tab:red',
                                                                   'train_type_acc': 'tab:olive', 'val_type_acc': 'tab:cyan'}
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for a, ser in zip(ax, series, strict=True):
      for k, col in ser.items():
        a.plot(ep, [h[k] for h in hist], color=col, label=k)
      a.legend(fontsize=7)
      a.grid(alpha=.3)
    ax[1].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return
  except ImportError:
    pass
  import zlib, struct  # noqa: E401
  W, H = 640, 480
  img = np.full((H, W, 3), 255, np.uint8)
  pal = {'tab:blue': (31, 119, 180), 'tab:orange': (255, 127, 14), 'tab:green': (44, 160, 44), 'tab:red': (214, 39, 40),
         'tab:olive': (188, 189, 34), 'tab:cyan': (23, 190, 207)}
  for panel, ser in enumerate(series):
    y0, y1 = panel*H//2 + 10, (panel + 1)*H//2 - 10
    vals = [v for k in ser for v in (h[k] for h in hist)]
    lo, hi = min(vals), max(vals) + 1e-9
    img[y1, 10:W - 10] = 0
    for k, col in ser.items():
      xs = np.linspace(10, W - 10, len(hist))
      ys = [y1 - (h[k] - lo)/(hi - lo)*(y1 - y0) for h in hist]
      for (xa, ya), (xb, yb) in zip(zip(xs, ys, strict=True), list(zip(xs, ys, strict=True))[1:], strict=False):
        for f in np.linspace(0, 1, 50):
          img[int(ya + (yb - ya)*f), int(xa + (xb - xa)*f)] = pal[col]
  raw = b''.join(b'\x00' + img[i].tobytes() for i in range(H))
  chunk = lambda tag, data: struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)  # noqa: E731
  Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) +
                         chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def unknown_weight(T, W, wt, share):
  """type weight of 'unknown' that gives it ``share`` of the effective (class weight x sample weight) training mass:
  w_u = share/(1 - share) * sum_{k != u} w_k S_k / S_u, S_k = summed sample weights of class k (before augmentation)."""
  S = torch.stack([W[T == i].sum() for i in range(len(T2))])
  if float(S[UNKNOWN]) <= 0:
    return wt
  rest = sum(float(wt[i]*S[i]) for i in range(len(T2)) if i != UNKNOWN)
  wt = wt.clone()
  wt[UNKNOWN] = share/(1 - share)*rest/float(S[UNKNOWN])
  return wt


def effective_share(T, W, wt):
  S = torch.stack([W[T == i].sum() for i in range(len(T2))])*wt
  return dict(zip(T2, [round(float(v), 4) for v in S/S.sum()], strict=True))


def train2(arch, seed, train_rows, xtr, val_rows, xva, device, epochs, aug='plus-rain-more-nored', norm='fixed', lr_scale=1.,
           synth_p=.12, blue_p=.05, out=None, unknown_share=None):
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  net = Net(arch, norm).to(device)
  augment_fn = AUGMENTS[aug]
  X, T, C, W = tensors2(train_rows, xtr, device)
  C = torch.where(T == UNKNOWN, torch.full_like(C, -100), C)      # unknown type: colour loss masked
  Xv, Tv, Cv, Wv = tensors2(val_rows, xva, device)
  Cv = torch.where(Tv == UNKNOWN, torch.full_like(Cv, -100), Cv)
  wt = class_weights(T, len(T2), (DASHED, SOLID))
  share0 = effective_share(T, W, wt)
  if unknown_share:  # round 2: unknown class weight set for a target effective share (default None: round-1 balanced/capped weight)
    wt = unknown_weight(T, W, wt, unknown_share)
  wc = class_weights(C, len(C2), (WHITE, YELLOW), ignore=-100)
  lr = 2e-3*lr_scale
  opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
  reps = AUG_REPEAT.get(aug, 1)
  steps = epochs*reps*math.ceil(len(X)/256)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.15)
  swap = torch.tensor([DASHED, SOLID, DBL_FAR, DBL_NEAR, UNKNOWN], device=device)
  best, history, counts = None, [], torch.zeros(len(T2), device=device)
  start = time.monotonic()
  for epoch in range(epochs):
    net.train()
    order = torch.cat([torch.randperm(len(X), device=device, generator=gen) for _ in range(reps)])
    tot = n_t = ok_t = n_c = ok_c = 0.
    for i in range(0, len(order), 256):
      idx = order[i:i + 256]
      xb, tb, cb = X[idx], T[idx], C[idx]
      if synth_p > 0:
        xb, tb = synth_double(xb, tb, cb, W[idx], gen, synth_p)
      if blue_p > 0:
        xb, cb = blue_hue(xb, tb, cb, gen, blue_p)
      xb = augment_fn(xb, gen)
      tb = torch.where(LAST_FLIP[0], swap[tb], tb)
      counts += torch.bincount(tb, minlength=len(T2)).float()
      lt, lc = net(xb)
      lw = W[idx]
      loss = ((F.cross_entropy(lt, tb, weight=wt, reduction='none') +
               F.cross_entropy(lc, cb, weight=wc, reduction='none', ignore_index=-100))*lw).sum()/lw.sum()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      sched.step()
      tot += float(loss)*len(idx)
      n_t += len(idx)
      ok_t += float((lt.argmax(1) == tb).sum())
      cm = cb >= 0
      n_c += float(cm.sum())
      ok_c += float(((lc.argmax(1) == cb) & cm).sum())
    pt, pc = predict2(net, Xv)
    with torch.no_grad():
      ptt, pct = torch.from_numpy(pt).to(device), torch.from_numpy(pc).to(device)
      vl = ((F.nll_loss(ptt.clamp(min=1e-8).log(), Tv, weight=wt, reduction='none') +
             F.nll_loss(pct.clamp(min=1e-8).log(), Cv, weight=wc, reduction='none', ignore_index=-100))*Wv).sum()/Wv.sum()
    m = metrics2(pt, pc, Tv.cpu().numpy(), Cv.cpu().numpy())
    history.append({'epoch': epoch, 'train_loss': tot/n_t, 'train_type_acc': ok_t/n_t, 'train_color_acc': ok_c/max(1, n_c),
                    'val_loss': float(vl), 'val_type_acc': m['type_acc'], 'val_color_acc': m['color_acc'],
                    'val_type_macro_f1': m['type_macro_f1'], 'val_color_macro_f1': m['color_macro_f1'], 'val_score': m['score'],
                    **{f'val_f1_{k}': v for k, v in m['type_f1'].items()}, **{f'val_f1_{k}': v for k, v in m['color_f1'].items()}})
    if best is None or m['score'] > best[0]:
      best = (m['score'], epoch, {k: v.detach().clone() for k, v in net.state_dict().items()})
  net.load_state_dict(best[2])
  if out is not None:
    keys = list(history[0])
    lines = [','.join(keys)] + [','.join('' if h[k] is None else f'{h[k]:.6g}' if isinstance(h[k], float) else str(h[k]) for k in keys) for h in history]
    Path(out/f'history-{arch}-{seed}.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot_history(history, out/f'history-{arch}-{seed}.png')
  seen = counts/counts.sum()
  return net, {'best_epoch': best[1], 'best_val_score': best[0], 'history': history, 'gpu_seconds': time.monotonic() - start,
               'params': sum(p.numel() for p in net.parameters()), 'type_weights': wt.tolist(), 'color_weights': wc.tolist(),
               'train_type_share_after_aug': dict(zip(T2, [round(float(v), 4) for v in seen], strict=True)),
               'effective_type_share_balanced': share0, 'effective_type_share': effective_share(T, W, wt), 'unknown_share_target': unknown_share,
               'blue_shift': list(BLUE_SHIFT)}


def main2(a):
  """StripNet2 runs: every --arch x --seeds -> <output>/runs/<arch>-<seed>.pt, history-<arch>-<seed>.csv/.png,
  onnx/<arch>-seed<k>.onnx (outputs color, type), runs.json."""
  import hashlib as _h
  out = Path(a.output)
  (out/'runs').mkdir(parents=True, exist_ok=True)
  (out/'onnx').mkdir(exist_ok=True)
  rows, x = load_set(a.data, mirror_right=True)
  srows, scols = x.shape[1:3]
  strip = {'half_m': round(scols*COL_M/2, 6), 'cols': int(scols), 'rows': int(srows)}
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  if a.limit:  # smoke test: evenly spaced subsets
    tr = [tr[j] for j in np.linspace(0, len(tr) - 1, min(a.limit, len(tr))).astype(int)]
    va = [va[j] for j in np.linspace(0, len(va) - 1, min(a.limit, len(va))).astype(int)]
  key = lambda r: (r['route'], r['segment'])  # noqa: E731  (route 407 / dbl: segment-level split)
  assert not {key(rows[i]) for i in tr} & {key(rows[i]) for i in va if not rows[i].get('holdout')}  # held-out unknown slice may share a segment
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  count = lambda rs: {f"{t}/{c}": n for (t, c), n in sorted(Counter((r['type'], r['color']) for r in rs).items())}  # noqa: E731
  global BLUE_SHIFT
  BLUE_SHIFT = (BLUE_SHIFT[0]*a.blue_strength, BLUE_SHIFT[1]*a.blue_strength)  # round 2: x1.5 (r2/blue_measure.json)
  report = {'data': str(a.data), 'strip': strip, 'mirror_right': True, 'aug': a.aug, 'synth_double': a.synth_double, 'blue_hue': a.blue_hue,
            'blue_strength': a.blue_strength, 'blue_shift': list(BLUE_SHIFT), 'unknown_share': a.unknown_share,
            'train_frames': len(tr), 'validation_frames': len(va), 'train_counts': count(train_rows), 'validation_counts': count(val_rows),
            'limit': a.limit, 'epochs': a.epochs, 'runs': []}
  print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)
  xv = tensors2(val_rows, x[va], a.device)[0]
  for arch in a.arch:
    for seed in a.seeds:
      net, info = train2(arch, seed, train_rows, x[tr], val_rows, x[va], a.device, a.epochs, a.aug, a.norm, a.lr_scale,
                         a.synth_double, a.blue_hue, out, a.unknown_share)
      pt, pc = predict2(net, xv)
      ck = out/'runs'/f'{arch}-{seed}.pt'
      torch.save(net.state_dict(), ck)
      np.savez(out/'runs'/f'{arch}-{seed}-val.npz', pt=pt, pc=pc)
      onnx = out/'onnx'/(f'seed{seed}.onnx' if len(a.arch) == 1 else f'{arch}-seed{seed}.onnx')
      parity = export2(net, onnx, scols, srows)
      net.to(a.device)
      run = {'arch': arch, 'seed': seed, **{k: v for k, v in info.items() if k != 'history'}, 'checkpoint': ck.name, 'onnx': onnx.name,
             'onnx_parity': parity, 'onnx_sha256': _h.sha256(onnx.read_bytes()).hexdigest(), 'onnx_bytes': onnx.stat().st_size,
             'val_macro_score': info['best_val_score'], 'history': info['history']}
      report['runs'].append(run)
      print(arch, seed, 'best epoch', info['best_epoch'], 'val score', round(info['best_val_score'], 4), 'params', info['params'],
            'parity', parity, 'seconds', round(info['gpu_seconds'], 1), flush=True)
      (out/'runs.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
  (out/'model-contract.json').write_text(json.dumps({
    'input': f'strip_ycbcr float32 N x 3 x {srows} x {scols}, raw 0..255 (Y, Cb, Cr), rows far(20m)->near(4.5m); RIGHT strips flipped left/right',
    'strip_geometry': strip, 'mirror_right': True, 'outputs': {'color': list(C2), 'type': list(T2)},
    'arch': a.arch, 'aug': a.aug, 'synth_double': a.synth_double, 'blue_hue': a.blue_hue, 'norm': a.norm}, indent=1), encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--arch', nargs='+', default=['stripnet', 'pplcnet025'])
  p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
  p.add_argument('--epochs', type=int, default=40)
  p.add_argument('--aug', choices=tuple(AUGMENTS), default='base')
  p.add_argument('--norm', choices=NORMS, default='fixed', help='input normalisation inside the model/ONNX (Normalize)')
  p.add_argument('--mbv3', default='.analysis/lane-model-comparison-20261018/pretrained/mobilenet_v3_small-047dcff4.pth')
  p.add_argument('--pretrained', default='.analysis/lane-model-comparison-20261018/pretrained/PPLCNet_x0_25_pretrained.pdparams')
  p.add_argument('--device', default='cuda', help="torch device (default cuda; 'cpu' for a smoke test)")
  p.add_argument('--limit', type=int, default=0, help='smoke test: at most N evenly spaced train and N validation frames')
  p.add_argument('--synth-double', type=float, default=.12, help='StripNet2: per-sample probability of the synth-double augmentation (0: off)')
  p.add_argument('--blue-hue', type=float, default=.05, help='StripNet2: per-sample probability of the blue-hue augmentation (0: off)')
  p.add_argument('--lr-scale', type=float, default=1., help='StripNet2: OneCycle max lr factor')
  p.add_argument('--blue-strength', type=float, default=1., help='StripNet2: BLUE_SHIFT factor of the blue-hue augmentation (round 2: 1.5)')
  p.add_argument('--unknown-share', type=float, default=None,
                 help="StripNet2: set the 'unknown' type weight for this effective training share (class x sample weight), e.g. 0.10")
  a = p.parse_args()
  if any(x in ARCH2 for x in a.arch):
    if not all(x in ARCH2 for x in a.arch):
      raise SystemExit(f'--arch: StripNet2 variants {ARCH2} cannot be mixed with 2-class archs')
    main2(a)
  else:
    main(a)
