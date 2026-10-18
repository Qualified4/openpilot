"""Three-architecture comparison on the 4.5-20 m YCbCr strip (GPU venv). PC-only, no runtime change.

StripNet (scratch), PP-LCNet-0.25 and MobileNetV3-Small (ImageNet weights; YCbCr->RGB + ImageNet
normalisation inside the network), each with augment 'base' and 'plus', seeds 0/1/2, same split/weights.

Selection rule, fixed BEFORE any training and applied to VALIDATION only:
  operating point = lowest confidence threshold (one for both heads) on the validation set with
    False Dashed (decided dashed on solid truth) <= 0.5% of solid frames, and decided accuracy >= 98.5%
    for type and for colour;
  run score = mean of type/colour decided rates at that point (0 if no threshold qualifies);
  config score = mean run score over the three seeds; the best config's best-seed run is selected.
The selection is written (and hashed) before the held-out human test set is scored, report-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.lane_profile.dataset_io import COLORS, TYPES, load_set  # noqa: TID251
from tools.lane_profile.train_strip import Export, Net, predict, tensors, train  # noqa: TID251

ARCHS = ('stripnet', 'pplcnet025', 'mobilenetv3s')
AUGS = ('base', 'plus')
SEEDS = (0, 1, 2)
FD_MAX, ACC_MIN = .005, .985
THRESHOLDS = np.round(np.arange(.5, .9951, .005), 3)
REPORT_THRESHOLDS = (.5, .8, .9, .95, .99)
SCAN_T, SCAN_C = ('unknown', 'dashed', 'solid'), ('unknown', 'white', 'yellow')


def decisions(pt, pc, th):
  t = np.where(pt.max(1) >= th, pt.argmax(1), -1)
  c = np.where(pc.max(1) >= th, pc.argmax(1), -1)
  yd = (t == 0) & (c == 1)  # dashed+yellow is not a learned class -> unknown
  t[yd], c[yd] = -1, -1
  return t, c


def score(t, c, rows):
  tt = np.array([TYPES.index(r['type']) for r in rows])
  tc = np.array([COLORS.index(r['color']) for r in rows])
  out = {'frames': len(rows)}
  for head, p, truth, classes in (('type', t, tt, TYPES), ('color', c, tc, COLORS)):
    d = p >= 0
    out[head] = {'decided_rate': float(d.mean()) if len(d) else None,
                 'accuracy_decided': float((p[d] == truth[d]).mean()) if d.any() else None,
                 'recall': {n: (float((p[truth == k] == k).mean()) if (truth == k).any() else None) for k, n in enumerate(classes)},
                 'support': {n: int((truth == k).sum()) for k, n in enumerate(classes)}}
  solid = tt == 1
  out['false_dashed'] = int(((t == 0) & solid).sum())
  out['solid_frames'] = int(solid.sum())
  out['false_dashed_rate'] = out['false_dashed']/max(1, out['solid_frames'])
  return out


def operating_point(pt, pc, rows):
  for th in THRESHOLDS:
    s = score(*decisions(pt, pc, th), rows)
    if (s['false_dashed_rate'] <= FD_MAX and (s['type']['accuracy_decided'] or 0) >= ACC_MIN and
        (s['color']['accuracy_decided'] or 0) >= ACC_MIN):
      return float(th), (s['type']['decided_rate'] + s['color']['decided_rate'])/2, s
  return None, 0., None


def by_quality(pt, pc, rows, th):
  out = {}
  for q in ('fcamera', 'qcamera'):
    idx = [i for i, r in enumerate(rows) if r['quality'] == q]
    if idx:
      out[q] = {'operating': score(*decisions(pt[idx], pc[idx], th), [rows[i] for i in idx]) if th is not None else None,
                'argmax': score(*decisions(pt[idx], pc[idx], 0.), [rows[i] for i in idx])}
  return out


def evaluate_probs(pt, pc, rows, th):
  return {'argmax': score(*decisions(pt, pc, 0.), rows),
          'thresholds': {str(t): score(*decisions(pt, pc, t), rows) for t in REPORT_THRESHOLDS},
          'operating': score(*decisions(pt, pc, th), rows) if th is not None else None,
          'by_quality': by_quality(pt, pc, rows, th)}


def scanner_score(rows):
  t = np.array([{'dashed': 0, 'solid': 1}.get(SCAN_T[r['scanner'][0]], -1) for r in rows])
  c = np.array([{'white': 0, 'yellow': 1}.get(SCAN_C[r['scanner'][1]], -1) for r in rows])
  return {'all': score(t, c, rows), 'by_quality': {q: score(t[[i for i, r in enumerate(rows) if r['quality'] == q]],
          c[[i for i, r in enumerate(rows) if r['quality'] == q]], [r for r in rows if r['quality'] == q])
          for q in ('fcamera', 'qcamera') if any(r['quality'] == q for r in rows)}}


def main(a):
  device = 'cuda'
  out = Path(a.output)
  out.mkdir(parents=True, exist_ok=True)
  rows, x = load_set(a.data)
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  xv = tensors(val_rows, x[va], device)[0]
  pretrained = {'pplcnet025': Path(a.pplcnet), 'mobilenetv3s': Path(a.mbv3)}
  runs = []
  for arch in ARCHS:
    for aug in AUGS:
      for seed in SEEDS:
        net, info = train(arch, seed, train_rows, x[tr], val_rows, x[va], device, a.epochs, pretrained, aug)
        pt, pc = predict(net, xv)
        th, op, _ = operating_point(pt, pc, val_rows)
        path = out/'runs'/f'{arch}-{aug}-{seed}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
        np.savez(out/'runs'/f'{arch}-{aug}-{seed}-val.npz', pt=pt, pc=pc)
        runs.append({'arch': arch, 'aug': aug, 'seed': seed, 'best_epoch': info['best_epoch'], 'params': info['params'],
                     'gpu_seconds': info['gpu_seconds'], 'val_macro_score': info['best_val_score'], 'operating_threshold': th,
                     'operating_score': op, 'validation': evaluate_probs(pt, pc, val_rows, th), 'checkpoint': path.name})
        print(arch, aug, seed, 'epoch', info['best_epoch'], 'op th', th, 'op score', round(op, 4), flush=True)
  configs = {}
  for arch in ARCHS:
    for aug in AUGS:
      rs = [r for r in runs if (r['arch'], r['aug']) == (arch, aug)]
      configs[f'{arch}/{aug}'] = {'operating_score_mean': float(np.mean([r['operating_score'] for r in rs])),
                                  'operating_score_std': float(np.std([r['operating_score'] for r in rs])),
                                  'val_macro_mean': float(np.mean([r['val_macro_score'] for r in rs])),
                                  'val_macro_std': float(np.std([r['val_macro_score'] for r in rs])),
                                  'best_seed': max(rs, key=lambda r: (r['operating_score'], r['val_macro_score']))['seed']}
  best = max(configs, key=lambda k: (configs[k]['operating_score_mean'], configs[k]['val_macro_mean']))
  selection = {'rule': __doc__.split('Selection rule')[1].split('The selection')[0].strip(), 'selected_config': best,
               'selected_seed': configs[best]['best_seed'], 'configs': configs}
  (out/'selection.json').write_text(json.dumps(selection, indent=1), encoding='utf-8')
  (out/'selection.sha256').write_text(hashlib.sha256((out/'selection.json').read_bytes()).hexdigest(), encoding='utf-8')
  # Export the best seed of every config (for size/latency) -- selection already frozen above.
  for key, cfg in configs.items():
    arch, aug = key.split('/')
    net = Net(arch)
    net.load_state_dict(torch.load(out/'runs'/f"{arch}-{aug}-{cfg['best_seed']}.pt", map_location='cpu'))
    wrapper = Export(net).eval()
    onnx = out/'onnx'/f'{arch}-{aug}.onnx'
    onnx.parent.mkdir(exist_ok=True)
    torch.onnx.export(wrapper, torch.full((2, 3, 128, 96), 100.), str(onnx), input_names=['strip_ycbcr'], output_names=['line_type', 'color'],
                      dynamic_axes={'strip_ycbcr': {0: 'n'}, 'line_type': {0: 'n'}, 'color': {0: 'n'}}, opset_version=14)
    import onnxruntime as ort
    probe = torch.rand(8, 3, 128, 96)*255
    ref = [o.detach().numpy() for o in wrapper(probe)]
    got = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None, {'strip_ycbcr': probe.numpy()})
    cfg.update(onnx=onnx.name, onnx_bytes=onnx.stat().st_size, onnx_parity=float(max(np.abs(p - q).max() for p, q in zip(ref, got, strict=True))))
  # Held-out human test: report only, after the selection file exists.
  test_rows, xt = load_set(a.test)
  xtt = tensors(test_rows, xt, device)[0]
  for r in runs:
    net = Net(r['arch']).to(device)
    net.load_state_dict(torch.load(out/'runs'/r['checkpoint'], map_location=device))
    pt, pc = predict(net, xtt)
    r['test_human'] = evaluate_probs(pt, pc, test_rows, r['operating_threshold'])
  baseline = Net('stripnet').to(device)
  baseline.load_state_dict(torch.load(a.baseline, map_location=device))
  pt, pc = predict(baseline, xv)
  th, op, _ = operating_point(pt, pc, val_rows)
  bt = predict(baseline, xtt)
  reference = {'stripnet_frozen_baseline': {'operating_threshold': th, 'operating_score': op, 'validation': evaluate_probs(pt, pc, val_rows, th),
                                            'test_human': evaluate_probs(*bt, test_rows, th)},
               'scanner_frame': {'validation': scanner_score(val_rows), 'test_human': scanner_score(test_rows)}}
  (out/'runs.json').write_text(json.dumps({'runs': runs, 'configs': configs, 'reference': reference}, indent=1), encoding='utf-8')
  print(json.dumps(selection, indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--test', required=True)
  p.add_argument('--baseline', required=True, help='frozen StripNet model.pt')
  p.add_argument('--output', required=True)
  p.add_argument('--epochs', type=int, default=40)
  p.add_argument('--pplcnet', default='.analysis/lane-model-comparison-20261018/pretrained/PPLCNet_x0_25_pretrained.pdparams')
  p.add_argument('--mbv3', default='.analysis/lane-model-comparison-20261018/pretrained/mobilenet_v3_small-047dcff4.pth')
  main(p.parse_args())
