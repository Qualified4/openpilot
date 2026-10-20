"""Parallel driver for replay_full.py.

Logs are grouped into chains of consecutive segments per route (display state carries across a chain). Each chain
is cut into chunks of --chunk segments; a chunk that starts mid-chain replays --warm preceding segments first
without writing them, so its state is rebuilt (60 s per warm segment, far beyond any display memory: rear MAX_TIME
10 s, filters < 5 s). Chunks run as separate processes, --jobs at a time.

  python tools/ccnc/replay_par.py --out DIR [--jobs N] [--chunk 3] [--warm 1] [replay_full options] LOG_OR_DIR...

A directory argument means every *--rlog.zst in it. Extra options (--module, --trace, --map, --rear, --params) pass
through to replay_full.py. The decoded-event cache is shared (.analysis/cache/rlog_events).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay_full import segkey  # noqa: E402


def chunks(logs, size, warm):
  by_route = {}
  for p in logs:
    by_route.setdefault(segkey(p)[0], []).append(p)
  for paths in by_route.values():
    paths.sort(key=segkey)
    chain = [paths[0]]
    chains = [chain]
    for p in paths[1:]:
      if segkey(p)[1] == segkey(chain[-1])[1]+1:
        chain.append(p)
      else:
        chain = [p]
        chains.append(chain)
    for chain in chains:
      for i in range(0, len(chain), size):
        lead = chain[max(0, i-warm):i]
        yield lead, chain[i:i+size], chain[0]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logs', nargs='+')
  ap.add_argument('--out', required=True)
  ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2)-2))
  ap.add_argument('--chunk', type=int, default=3)
  ap.add_argument('--warm', type=int, default=1)
  ap.add_argument('--module')
  ap.add_argument('--trace', action='store_true')
  ap.add_argument('--map')
  ap.add_argument('--rear')
  ap.add_argument('--params')
  a = ap.parse_args()
  extra = [*(['--module', str(Path(a.module).resolve())] if a.module else []), *(['--trace'] if a.trace else []),
           *(['--map', a.map] if a.map is not None else []), *(['--rear', a.rear] if a.rear is not None else []),
           *(['--params', a.params] if a.params else [])]
  logs = []
  for item in a.logs:
    p = Path(item)
    logs += sorted(map(str, p.glob('*--rlog.zst'))) if p.is_dir() else [str(p)]
  work = sorted(chunks(logs, a.chunk, a.warm), key=lambda w: -len(w[0]+w[1]))
  start = time.time()

  def run(item):
    lead, body, start = item
    cmd = [sys.executable, str(HERE/'replay_full.py'), '--out', a.out, '--warm', str(len(lead)), '--chain-start', start,
           *extra, *lead, *body]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return body, r

  total, failed = {}, []
  with ThreadPoolExecutor(a.jobs) as pool:
    for body, r in pool.map(run, work):
      lines = r.stdout.strip().splitlines()
      summary = next((l for l in reversed(lines) if l.startswith('total ')), None)
      if r.returncode or summary is None:
        failed.append((body, r.stderr[-2000:]))
        continue
      for k, v in json.loads(summary[6:]).items():
        total[k] = total.get(k, 0)+v
      for l in lines:
        if 'failed True' in l or "('exception'" in l or "('error'" in l:
          print(l)
  print(f'{len(logs)} logs, {len(work)} chunks, jobs {a.jobs}: {time.time()-start:.0f} s', json.dumps(total))
  for body, err in failed:
    print('FAILED', [Path(b).name for b in body], err)
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
