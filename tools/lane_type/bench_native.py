"""PC-only two-boundary native RGB feature/temporal benchmark, no device claims."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from tools.lane_type.prepare_dual_data import json_write  # noqa: TID251 - offline repository tooling
from openpilot.selfdrive.carrot.lane_type.native import native_observation
from openpilot.selfdrive.carrot.lane_type.temporal import DistanceEvidence


def benchmark(output, seconds=32):
  import cv2
  import psutil
  cv2.setNumThreads(1)
  source = Path('.analysis/archive/2026-10-18/lane-roi-quality')
  rows = json.loads((source/'audit.json').read_text(encoding='utf-8'))
  rows = [r for r in rows if r['split']=='train' and r['video_source']=='fcamera'][:2]
  images = [np.asarray(Image.open(source/r['frame']).convert('RGB')) for r in rows]
  accumulators = [DistanceEvidence(), DistanceEvidence()]
  extract, temporal = [], []
  process = psutil.Process()
  peak = process.memory_info().rss
  for _ in range(3):
    for r, image in zip(rows, images, strict=True):
      native_observation(image, r['line'], 1., r['rpy'], r['intrinsics'], r['model_leads'])
  start, cpu_start = time.perf_counter(), time.process_time()
  for cycle in range(round(seconds*5)):
    t = time.perf_counter()
    profiles = [native_observation(image, r['line'], 1., r['rpy'], r['intrinsics'], r['model_leads'])
                for r, image in zip(rows, images, strict=True)]
    extract.append((time.perf_counter()-t)*1000)
    t = time.perf_counter()
    for r, a, p in zip(rows, accumulators, profiles, strict=True):
      boundary = [float(np.interp(d,r['line']['x'],r['line']['y'])) for d in (10,25)]
      a.update(p['profile'], boundary, r['speed_mps'], cycle*200_000_000)
    temporal.append((time.perf_counter()-t)*1000)
    peak = max(peak, process.memory_info().rss)
    time.sleep(max(0, start+(cycle+1)/5-time.perf_counter()))
  elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu_start
  def summary(values):
    return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99))}
  result = {'environment':'Windows PC, NOT Snapdragon/comma','calls':len(extract),'boundaries_per_call':2,
    'seconds':elapsed,'cpu_one_core_percent':cpu/elapsed*100,'cpu_counter_seconds':cpu,'rss_sample_max_mib':peak/2**20,
    'native_RGB_feature_ms':summary(extract),'temporal_update_ms':summary(temporal),
    'scope':'two prerecorded HD boundary observations; full RGB/HSV processing included, image decoding and CNN/display excluded',
    'performance_input_repetition_only':'Clock/travel repetition here benchmarks operation cost, never accuracy or real odometry.',
    'NV12_device_cost':'unmeasured; native RGB research path is not wired into existing daemon'}
  json_write(output,result)


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--seconds',type=float,default=32.)
  args=parser.parse_args()
  benchmark(args.output,args.seconds)
