"""Torch-free dataset access shared by training (GPU venv) and evaluation (device OpenCV)."""
import json
from pathlib import Path

import numpy as np

# Label vocabulary. The first two entries of each are the StripNet v1-v3 heads (TYPES_V1 / COLORS_V1, index = class id);
# later entries are additive (2026-10-21 wide strip round: double lines and blue bus-lane paint) and never reorder them.
TYPES_V1, COLORS_V1 = ('dashed', 'solid'), ('white', 'yellow')
TYPES = TYPES_V1 + ('dbl_near', 'dbl_far', 'unknown')
COLORS = COLORS_V1 + ('blue',)


def check_labels(rows, types=TYPES, colors=COLORS):
  """Raise ValueError for a row whose type / color is outside the vocabulary."""
  for i, r in enumerate(rows):
    if r['type'] not in types or r['color'] not in colors:
      raise ValueError(f"row {i}: label {r['type']}/{r['color']} not in {types} x {colors}")


def load_set(folder):
  """manifest.jsonl rows and their row-aligned YCbCr strips (N x 128 x W x 3 uint8; W = 96 for the +-0.75 m strip,
  128 for +-1.0 m: taken from the shards, which must all agree). Labels are checked against TYPES / COLORS."""
  folder = Path(folder)
  rows = [json.loads(line) for line in open(folder/'manifest.jsonl', encoding='utf-8')]
  check_labels(rows)
  shards, x = {}, None
  for i, r in enumerate(rows):
    if r['shard'] not in shards:
      shards[r['shard']] = np.load(folder/'shards'/r['shard'])['row_strip_ycbcr']
      if x is None:
        x = np.zeros((len(rows),) + shards[r['shard']].shape[1:], np.uint8)
      elif shards[r['shard']].shape[1:] != x.shape[1:]:
        raise ValueError(f"shard {r['shard']} strip shape {shards[r['shard']].shape[1:]} != {x.shape[1:]}")
    x[i] = shards[r['shard']][r['shard_row']]
  return rows, (x if x is not None else np.zeros((0, 128, 96, 3), np.uint8))
