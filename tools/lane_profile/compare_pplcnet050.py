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
from tools.lane_profile.train_strip import Export, Net, predict, tensors, train  # noqa: TID251

ARCH, AUG, SEEDS = 'pplcnet050', 'plus', (0, 1, 2)  # --arch/--aug override (default: the PP-LCNet-0.5 run)


def main(a):
  global ARCH, AUG
  ARCH, AUG = a.arch, a.aug
  device = 'cuda'
  out = Path(a.output)
  (out/'runs').mkdir(parents=True, exist_ok=True)
  rows, x = load_set(a.data)
  tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
  va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
  train_rows, val_rows = [rows[i] for i in tr], [rows[i] for i in va]
  xv = tensors(val_rows, x[va], device)[0]
  runs = []
  for seed in SEEDS:
    net, info = train(ARCH, seed, train_rows, x[tr], val_rows, x[va], device, a.epochs, {ARCH: Path(a.pretrained)}, AUG)
    pt, pc = predict(net, xv)
    th, op, _ = operating_point(pt, pc, val_rows)
    path = out/'runs'/f'{ARCH}-{AUG}-{seed}.pt'
    torch.save(net.state_dict(), path)
    np.savez(out/'runs'/f'{ARCH}-{AUG}-{seed}-val.npz', pt=pt, pc=pc)
    runs.append({'arch': ARCH, 'aug': AUG, 'seed': seed, 'best_epoch': info['best_epoch'], 'params': info['params'],
                 'gpu_seconds': info['gpu_seconds'], 'val_macro_score': info['best_val_score'], 'operating_threshold': th,
                 'operating_score': op, 'validation': evaluate_probs(pt, pc, val_rows, th), 'checkpoint': path.name,
                 'history': info['history']})
    print(ARCH, AUG, seed, 'epoch', info['best_epoch'], 'val', round(info['best_val_score'], 4), 'op th', th, flush=True)
  cfg = {'operating_score_mean': float(np.mean([r['operating_score'] for r in runs])),
         'operating_score_std': float(np.std([r['operating_score'] for r in runs])),
         'val_macro_mean': float(np.mean([r['val_macro_score'] for r in runs])),
         'val_macro_std': float(np.std([r['val_macro_score'] for r in runs])),
         'best_seed': max(runs, key=lambda r: (r['operating_score'], r['val_macro_score']))['seed']}
  net = Net(ARCH)
  net.load_state_dict(torch.load(out/'runs'/f"{ARCH}-{AUG}-{cfg['best_seed']}.pt", map_location='cpu'))
  wrapper = Export(net).eval()
  onnx = out/'onnx'/f'{ARCH}-{AUG}.onnx'
  onnx.parent.mkdir(exist_ok=True)
  torch.onnx.export(wrapper, torch.full((2, 3, 128, 96), 100.), str(onnx), input_names=['strip_ycbcr'], output_names=['line_type', 'color'],
                    dynamic_axes={'strip_ycbcr': {0: 'n'}, 'line_type': {0: 'n'}, 'color': {0: 'n'}}, opset_version=14)
  import onnxruntime as ort
  probe = torch.rand(8, 3, 128, 96)*255
  ref = [o.detach().numpy() for o in wrapper(probe)]
  got = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None, {'strip_ycbcr': probe.numpy()})
  cfg.update(onnx=onnx.name, onnx_bytes=onnx.stat().st_size, onnx_parity=float(max(np.abs(p - q).max() for p, q in zip(ref, got, strict=True))))
  (out/'selection.json').write_text(json.dumps({'config': f'{ARCH}/{AUG}', **cfg}, indent=1), encoding='utf-8')
  test_rows, xt = load_set(a.test)
  xtt = tensors(test_rows, xt, device)[0]
  for r in runs:
    net = Net(ARCH).to(device)
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
  p.add_argument('--aug', default=AUG, choices=('base', 'plus'))
  p.add_argument('--pretrained', default='.analysis/lane-model-comparison-20261018/pretrained/PPLCNet_x0_5_pretrained.pdparams')
  main(p.parse_args())
