"""Render frozen diagnostic measurements; no inference, tuning or training."""
import argparse
import json
from pathlib import Path

NAMES = ('pplcnet_025', 'pplcnet_050', 'mobilenet_v3_small', 'shufflenet_v2_050')


def percentage(value):
  return '측정 불가' if value is None else f'{100*value:.2f}'


def run(root):
  comparison = json.loads((root/'comparison.json').read_text(encoding='utf-8'))
  lines = ['# 보존된 구 ROI 진단 비교표', '',
    '**채택 근거로 사용 불가. ROI 픽셀 감사 HOLD 전 완료된 실험이며 신규 lossless 표현의 성능이 아니다.**', '',
    '원시 정확도·F1은 test 173개 전체 기준이다. 입력 미확보 13개는 UNKNOWN이며 오답/미검출에 포함한다. ' +
    '실제 두 head 출력이 있는 160개만의 정확도는 comparison.json의 confirmed_accuracy로 구분한다.', '',
    '| 모델 | 파라미터 | ONNX MiB | RTX4090 학습 s | GPU 최대 allocated MiB | PC 평균/p95/p99 ms | PC RSS MiB |',
    '|---|---:|---:|---:|---:|---|---:|']
  for name in NAMES:
    m = json.loads((root/f'native-expanded/{name}/manifest.json').read_text(encoding='utf-8'))
    pc = json.loads((root/f'pc-{name}.json').read_text(encoding='utf-8'))
    latency = pc['latency_ms']
    lines.append(f"| {name} | {m['parameters']:,} | {m['onnx_bytes']/2**20:.3f} | {m['training_seconds']:.2f} | " +
      f"{m['max_allocated_bytes']/2**20:.2f} | {latency['mean']:.3f}/{latency['p95']:.3f}/{latency['p99']:.3f} | {pc['rss_sample_max_mib']:.2f} |")
  lines.extend(['', 'GPU 시간은 작은 검수 데이터의 40 epoch 학습/검증 루프이며 데이터 추출·검수·export 시간을 포함하지 않는다. ' +
    'PC 시간은 양측 두 head 신경망만이며 ROI 추출/정규화/디코딩/IPC를 제외한다. RSS는 프로세스 표본 최댓값이다.', '',
    '| 모델 | 종류 정확도 % | 종류 macro F1 % | 색상 정확도 % | 색상 macro F1 % | 노란색 Recall % | Raw False Dashed/실선59 | 노란→흰/노란21 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|'])
  for name in NAMES:
    a = comparison['models'][f'expanded/{name}']['single']['all']
    t, c = a['raw']['type'], a['raw']['color']
    lines.append(f"| {name} | {percentage(t['accuracy'])} | {percentage(t['macro_f1'])} | {percentage(c['accuracy'])} | " +
      f"{percentage(c['macro_f1'])} | {percentage(c['classes']['yellow']['recall'])} | {a['false_dashed_raw']}/59 | {a['yellow_as_white_raw']}/21 |")
  lines.extend(['', 'False Dashed의 raw 분모는 모든 실제 실선 59개이다. 최종 표시 False Dashed 분모는 실제로 점선 표시한 횟수와 별도 표기한다.', '',
    '| 모델 | 클래스 | Precision % | Recall % | F1 % | 표본 |', '|---|---|---:|---:|---:|---:|'])
  for name in NAMES:
    a = comparison['models'][f'expanded/{name}']['single']['all']['raw']
    for head in ('type', 'color'):
      for label, m in a[head]['classes'].items():
        lines.append(f"| {name} | {label} | {percentage(m['precision'])} | {percentage(m['recall'])} | {percentage(m['f1'])} | {m['support']} |")
  lines.extend(['', '| 모델 | 기존자료 종류 정확도 % | 확장자료 종류 정확도 % | 변화 pp | 기존 노란 Recall % | 확장 노란 Recall % |',
                '|---|---:|---:|---:|---:|---:|'])
  for name in NAMES:
    a = comparison['models'][f'baseline/{name}']['single']['all']['raw']
    b = comparison['models'][f'expanded/{name}']['single']['all']['raw']
    lines.append(f"| {name} | {percentage(a['type']['accuracy'])} | {percentage(b['type']['accuracy'])} | " +
      f"{100*(b['type']['accuracy']-a['type']['accuracy']):+.2f} | {percentage(a['color']['classes']['yellow']['recall'])} | " +
      f"{percentage(b['color']['classes']['yellow']['recall'])} |")
  lines.extend(['', '기존자료 train50 vs 확장 train170; validation44/test173 동일. 양쪽40 epoch이지만 update80 vs240으로 ' +
    '데이터 효과와 update 수가 교란된다. 별도120 epoch/240 update 대조군은 equal-update-control.json에 보존했다. ' +
    '추가 대조군은 최초 테스트 열람 후의 탐색 실험이므로 신규 모델 선택 근거로 사용하지 않았다.', '',
    '| 모델 | 최종 표시 확정 % (173) | 확정 내 정확도 % | False Dashed/점선 표시 | 주황 오표시/주황 표시 | dense 단일 확정 % (142) | dense 시간누적 확정 % |',
    '|---|---:|---:|---|---|---:|---:|'])
  for name in NAMES:
    m = comparison['models'][f'expanded/{name}']
    a = m['single']['all']['display']
    lines.append(f"| {name} | {percentage(a['confirmed_rate'])} | {percentage(a['conditional_accuracy'])} | " +
      f"{a['false_dashed']}/{a['dashed_shown']} | {a['false_orange']}/{a['orange_shown']} | " +
      f"{percentage(m['dense_single']['display']['confirmed_rate'])} | {percentage(m['dense_temporal']['display']['confirmed_rate'])} |")
  lines.extend(['', 'dense 시간누적에서 PP.25는 표시0개, 나머지는 주황8개/142이고 점선 출력은 모두0개다. ' +
    'False Dashed 0건의 분모가0이므로 오류율은 측정 불가다. 숨김 증가를 안전성 개선으로 해석하지 않는다.', '',
    '| 모델 | 구분 | N | 종류 정확도 % | 색상 정확도 % | 종류 macro F1 % | 색상 macro F1 % |',
    '|---|---|---:|---:|---:|---:|---:|'])
  for name in NAMES:
    for scenario in ('fcamera', 'qcamera', 'night', 'curve', 'low_speed', 'stopped', 'lane_change', 'unobservable'):
      a = comparison['models'][f'expanded/{name}']['single'][scenario]
      t, c = a['raw']['type'], a['raw']['color']
      lines.append(f"| {name} | {scenario} | {a['samples']} | {percentage(t['accuracy'])} | {percentage(c['accuracy'])} | " +
        f"{percentage(t['macro_f1'])} | {percentage(c['macro_f1'])} |")
  lines.extend(['', 'fcamera76/qcamera97는 장면 분포가 다르고 같은 장면 쌍으로 제한한 화질 인과 비교가 아니다. ' +
    '혼합 학습 효과를 분리한 f-only/q-only 실험은 미실시다. night8, 저속1, 정차1, 차선변경0으로 상황별 일반화 판단은 불가하다.', ''])
  (root/'comparison-tables.md').write_text('\n'.join(lines), encoding='utf-8')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
  for ax, name in zip(axes.flat, NAMES, strict=True):
    m = json.loads((root/f'native-expanded/{name}/manifest.json').read_text(encoding='utf-8'))
    history = m['history']
    for key in ('train_loss', 'validation_loss'):
      ax.plot([h['epoch'] for h in history], [h[key] for h in history], label=key)
    ax.axvline(m['best_epoch'], linestyle='--', color='gray', label='validation checkpoint')
    ax.set(title=name, xlabel='Epoch', ylabel='Weighted CE sum')
    ax.legend(fontsize=7)
  fig.savefig(root/'training-loss.png', dpi=140)
  plt.close(fig)
  print('frozen tables and training-loss.png written; all experiments remain quarantined')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--root', type=Path, default=Path('.analysis/lane-model-comparison-20261018'))
  run(p.parse_args().root)
