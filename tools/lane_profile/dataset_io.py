"""Torch-free dataset access shared by training (GPU venv) and evaluation (device OpenCV)."""
import json
from pathlib import Path

import numpy as np

TYPES, COLORS = ('dashed', 'solid'), ('white', 'yellow')


def load_set(folder):
  """manifest.jsonl rows and their row-aligned YCbCr strips (N x 128 x 96 x 3 uint8)."""
  folder = Path(folder)
  rows = [json.loads(line) for line in open(folder/'manifest.jsonl', encoding='utf-8')]
  shards, x = {}, np.zeros((len(rows), 128, 96, 3), np.uint8)
  for i, r in enumerate(rows):
    if r['shard'] not in shards:
      shards[r['shard']] = np.load(folder/'shards'/r['shard'])['row_strip_ycbcr']
    x[i] = shards[r['shard']][r['shard_row']]
  return rows, x
