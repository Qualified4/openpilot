"""Torch-free dataset access shared by training (GPU venv) and evaluation (device OpenCV)."""
import json
from pathlib import Path

import numpy as np

# Label vocabulary. The first two entries of each are the StripNet v1-v3 heads (TYPES_V1 / COLORS_V1, index = class id);
# later entries are additive (2026-10-21 wide strip round: double lines and blue bus-lane paint) and never reorder them.
TYPES_V1, COLORS_V1 = ('dashed', 'solid'), ('white', 'yellow')
TYPES = TYPES_V1 + ('dbl_near', 'dbl_far', 'unknown')
COLORS = COLORS_V1 + ('blue',)
# 4-type head (2026-10-26 fourclass1026, user decision: 'unknown is not a class'): the model learns TYPES4 only; 'unknown' in a
# manifest is an abstention label (not judged / not confidently a lane marking) -> type id -100 (masked from the losses, kept to
# measure the abstention rate). Display family: dbl_near (ego-side line dashed) -> dashed, dbl_far (ego-side line solid) -> solid.
# Two solid lines are 'solid' (not a double class). Colour: blue is displayed white.
TYPES4 = TYPES[:4]
HOLD = 'unknown'
DISPLAY_TYPE = {'dashed': 'dashed', 'dbl_near': 'dashed', 'solid': 'solid', 'dbl_far': 'solid'}
DISPLAY_COLOR = {'white': 'white', 'blue': 'white', 'yellow': 'yellow'}


def type4_id(type_):
  """TYPES4 index of a manifest type, -100 for the 'unknown' abstention label (ValueError for anything else)."""
  return -100 if type_ == HOLD else TYPES4.index(type_)


def check_labels(rows, types=TYPES, colors=COLORS):
  """Raise ValueError for a row whose type / color is outside the vocabulary."""
  for i, r in enumerate(rows):
    if r['type'] not in types or r['color'] not in colors:
      raise ValueError(f"row {i}: label {r['type']}/{r['color']} not in {types} x {colors}")


def load_set(folder, mirror_right=False):
  """manifest.jsonl rows and their row-aligned YCbCr strips (N x 128 x W x 3 uint8; W = 96 for the +-0.75 m strip,
  128 for +-1.0 m: taken from the shards, which must all agree). Labels are checked against TYPES / COLORS.
  mirror_right (StripNet2 / manifest v4 'mirror_right'): RIGHT-side strips are flipped left/right so the ego side (near) is
  always the HIGH column index, as on the unflipped left strip (profile_scanner.strip_geometry: column 0 = line y - half_m,
  i.e. image-left; the ego car is image-right of the left line and image-left of the right line).
  A shard name may be a relative path ('../../other-set/shards/x.npz') so a merged set can reuse another set's shards."""
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
    if mirror_right and r['side'] == 'right':
      x[i] = x[i][:, ::-1]
  return rows, (x if x is not None else np.zeros((0, 128, 96, 3), np.uint8))
