"""PP-LCNet-0.5 added to the three-architecture comparison (GPU venv). PC-only.

Exactly the compare_models.py conditions: same dataset/split/per-origin weights, ImageNet weights (hash-pinned
PPLCNet_x0_5), 'plus' augmentation, 40 epochs, seeds 0/1/2, the same validation selection rule and metrics
(compare_models.operating_point / evaluate_probs), the best seed exported to ONNX, the human test scored once
afterwards (report only). Training history is kept for the learning curve.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.lane_profile.compare_models import evaluate_probs, operating_point  # noqa: TID251
from tools.lane_profile.dataset_io import load_set  # noqa: TID251
from tools.lane_profile.train_strip import AUGMENTS, Export, Net, export, predict, tensors, train  # noqa: TID251

ARCH, AUG, NORM, SEEDS = 'pplcnet050', 'plus', 'fixed', (0, 1, 2)  # --arch/--aug/--norm override (default: the PP-LCNet-0.5 run)


def main(a):
  global ARCH, AUG, NORM
  ARCH, AUG, NORM = a.arch, a.aug, a.norm
  tag = f'{ARCH}-{AUG}' + ('' if NORM == 'fixed' else f'-{NORM}')  # file names unchanged for fixed
  device = a.device
  out = Path(a.output)
  (out/'runs').mkdir(parents=True, exist_ok=True)
  rows, x = load_set(a.data)
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  if a.limit:  # smoke test only: a seeded random subset
    rng = np.random.default_rng(0)
    tr = sorted(rng.choice(tr, min(len(tr), a.limit), replace=False).tolist())
    va = sorted(rng.choice(va, min(len(va), max(1, a.limit//4)), replace=False).tolist())
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  xv = tensors(val_rows, x[va], device)[0]
  runs = []
  for seed in a.seeds:
    net, info = train(ARCH, seed, train_rows, x[tr], val_rows, x[va], device, a.epochs, {ARCH: Path(a.pretrained)}, AUG, NORM, a.lr_scale)
    pt, pc = predict(net, xv)
    th, op, _ = operating_point(pt, pc, val_rows)
    path = out/'runs'/f'{tag}-{seed}.pt'
    torch.save(net.state_dict(), path)
    np.savez(out/'runs'/f'{tag}-{seed}-val.npz', pt=pt, pc=pc)
    runs.append({'arch': ARCH, 'aug': AUG, 'norm': NORM, 'lr_scale': a.lr_scale, 'seed': seed, 'best_epoch': info['best_epoch'], 'params': info['params'],
                 'gpu_seconds': info['gpu_seconds'], 'val_macro_score': info['best_val_score'], 'operating_threshold': th,
                 'operating_score': op, 'validation': evaluate_probs(pt, pc, val_rows, th), 'checkpoint': path.name,
                 'history': info['history']})
    print(ARCH, AUG, seed, 'epoch', info['best_epoch'], 'val', round(info['best_val_score'], 4), 'op th', th, flush=True)
  cfg = {'operating_score_mean': float(np.mean([r['operating_score'] for r in runs])),
         'operating_score_std': float(np.std([r['operating_score'] for r in runs])),
         'val_macro_mean': float(np.mean([r['val_macro_score'] for r in runs])),
         'val_macro_std': float(np.std([r['val_macro_score'] for r in runs])),
         'best_seed': max(runs, key=lambda r: (r['operating_score'], r['val_macro_score']))['seed']}
  net = Net(ARCH, NORM)
  net.load_state_dict(torch.load(out/'runs'/f"{tag}-{cfg['best_seed']}.pt", map_location='cpu'))
  wrapper = Export(net).eval()
  onnx = out/'onnx'/f'{tag}.onnx'
  onnx.parent.mkdir(exist_ok=True)
  torch.onnx.export(wrapper, torch.full((2, 3, 128, 96), 100.), str(onnx), input_names=['strip_ycbcr'], output_names=['line_type', 'color'],
                    dynamic_axes={'strip_ycbcr': {0: 'n'}, 'line_type': {0: 'n'}, 'color': {0: 'n'}}, opset_version=14)
  import onnxruntime as ort
  probe = torch.rand(8, 3, 128, 96)*255
  ref = [o.detach().numpy() for o in wrapper(probe)]
  got = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None, {'strip_ycbcr': probe.numpy()})
  cfg.update(onnx=onnx.name, onnx_bytes=onnx.stat().st_size, onnx_parity=float(max(np.abs(p - q).max() for p, q in zip(ref, got, strict=True))))
  if a.export_seeds:  # every seed as onnx/seed<k>.onnx (stage-3 seed selection evaluates all of them)
    cfg['seed_onnx'] = {}
    for r in runs:
      sn = Net(ARCH, NORM)
      sn.load_state_dict(torch.load(out/'runs'/r['checkpoint'], map_location='cpu'))
      cfg['seed_onnx'][r['seed']] = {'onnx': f"seed{r['seed']}.onnx", 'parity': export(sn, out/'onnx'/f"seed{r['seed']}.onnx")}
  (out/'selection.json').write_text(json.dumps({'config': f'{ARCH}/{AUG}', 'norm': NORM, **cfg}, indent=1), encoding='utf-8')
  test_rows, xt = load_set(a.test)
  if a.limit:
    keep = sorted(np.random.default_rng(0).choice(len(test_rows), min(len(test_rows), max(1, a.limit//4)), replace=False).tolist())
    test_rows, xt = [test_rows[i] for i in keep], xt[keep]
  xtt = tensors(test_rows, xt, device)[0]
  for r in runs:
    net = Net(ARCH, NORM).to(device)
    net.load_state_dict(torch.load(out/'runs'/r['checkpoint'], map_location=device))
    r['test_human'] = evaluate_probs(*predict(net, xtt), test_rows, r['operating_threshold'])
  (out/'runs.json').write_text(json.dumps({'runs': runs, 'configs': {f'{ARCH}/{AUG}': cfg}}, indent=1), encoding='utf-8')
  print(json.dumps(cfg, indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--data', required=True)
  p.add_argument('--test', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--epochs', type=int, default=40)
  p.add_argument('--arch', default=ARCH, help='train_strip.Net arch (stripnet for the stage-1 retrain)')
  p.add_argument('--aug', default=AUG, choices=tuple(AUGMENTS))
  p.add_argument('--norm', default=NORM, choices=('fixed', 'road', 'road-y'), help='train_strip.Normalize mode (inside the ONNX)')
  p.add_argument('--lr-scale', type=float, default=1., help='OneCycle max lr factor (stage-3 e-long: 0.5)')
  p.add_argument('--export-seeds', action='store_true', help='also export every seed to onnx/seed<k>.onnx')
  p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
  p.add_argument('--device', default='cuda', help='cuda (TRAIN.ps1) or cpu (VM smoke test)')
  p.add_argument('--limit', type=int, default=0, help='smoke test: first N train / N validation / N test frames only')
  p.add_argument('--pretrained', default='.analysis/lane-model-comparison-20261018/pretrained/PPLCNet_x0_5_pretrained.pdparams')
  main(p.parse_args())
