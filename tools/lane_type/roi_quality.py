"""PC-only ROI projection audit. Invalid pixels are never inference evidence.

Requested ranges are diagnostic: a partially visible 3m crop is not silently
presented as a complete 3m observation. Paint masks require image review.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.lane_type.retrain_data import metadata  # noqa: TID251 - offline tools
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.roi import rgb_strip, nv12_strip

RANGES = ((3., 15.), (3., 25.), (3., 35.), (6., 46.))


def projection_maps(line, rpy, intrinsics, near, far, rows=192, columns=64):
  if not 0 < near < far or line['x'][0] > near or line['x'][-1] < far:
    raise ValueError('requested distance outside model geometry')
  distances = np.linspace(far, near, rows)
  points = np.stack((np.broadcast_to(distances[:, None], (rows, columns)),
                     np.interp(distances, line['x'], line['y'])[:, None]+np.linspace(-.45, .45, columns),
                     np.broadcast_to(np.interp(distances, line['x'], line['z'])[:, None], (rows, columns))), -1)
  pixels = points @ (intrinsics @ view_frame_from_device_frame @ rot_from_euler(rpy)).T
  return ((pixels[..., 0]/pixels[..., 2]).astype(np.float32),
          (pixels[..., 1]/pixels[..., 2]).astype(np.float32)), distances


def in_frame(maps, width, height):
  mx, my = maps
  return np.isfinite(mx+my) & (mx >= 0) & (mx < width-1) & (my >= 0) & (my < height-1)


def geometry_quality(maps, distances, width, height):
  valid = in_frame(maps, width, height)
  complete_rows = valid.all(1)
  mx, my = maps
  longitudinal_step = np.hypot(np.diff(mx[:, 32]), np.diff(my[:, 32]))
  lateral_step = np.hypot(mx[:, -1]-mx[:, 0], my[:, -1]-my[:, 0])/63
  result = {'requested_range_m': [float(distances[-1]), float(distances[0])],
    'in_frame_pixel_fraction': float(valid.mean()), 'complete_row_fraction': float(complete_rows.mean()),
    'complete_requested_roi': bool(valid.all()),
    'first_full_width_visible_m': float(distances[complete_rows].min()) if complete_rows.any() else None,
    'visible_longitudinal_span_m': float(np.ptp(distances[complete_rows])) if complete_rows.sum() > 1 else 0.,
    'bands': {}}
  for name, lo, hi in [('near', 3, 15), ('middle', 15, 25), ('far', 25, 46.01)]:
    use = (distances >= lo) & (distances < hi) & complete_rows
    steps = longitudinal_step[use[:-1] & use[1:]]
    result['bands'][name] = {'rows': int(use.sum()),
      'median_source_px_per_output_row': float(np.median(steps)) if len(steps) else None,
      'median_source_px_per_output_column': float(np.median(lateral_step[use])) if use.any() else None,
      'upsampled_longitudinal_fraction': float(np.mean(steps < 1)) if len(steps) else None}
  return result


def paint_quality(mask, maps, distances):
  """Measure reviewed painted pixels, never an automatic colour mask as truth.

  Distance bands use model projection, not surveyed road distances. Annotation
  recall covers only the explicitly traced paint, not an entire frame's truth.
  """
  import cv2
  valid = in_frame(maps, mask.shape[1], mask.shape[0])
  sampled = cv2.remap(mask, *maps, cv2.INTER_NEAREST) > 0
  painted = sampled & valid
  rows = painted.sum(1) >= 2
  centers = np.divide((painted*np.arange(64)[None]).sum(1), painted.sum(1),
                      out=np.full(192, np.nan), where=painted.sum(1) > 0)
  outline = np.stack((np.r_[maps[0][:, 0], maps[0][::-1, -1]], np.r_[maps[1][:, 0], maps[1][::-1, -1]]), 1)
  footprint = np.zeros_like(mask)
  cv2.fillPoly(footprint, [np.rint(outline).astype(np.int32)], 1)
  # Pixel-level nearest model centre gives a reproducible approximate depth
  # band even when the actual stripe is displaced from the predicted centre.
  from scipy.spatial import cKDTree
  y, x = np.nonzero(mask)
  centerline = np.stack((maps[0][:, 32], maps[1][:, 32]), -1)
  _, nearest = cKDTree(centerline).query(np.stack((x, y), 1))
  # Exclude annotations extending beyond the requested end planes.
  inside_depth = (nearest > 0) & (nearest < len(distances)-1)
  denominator = int(inside_depth.sum())
  recall = int(((footprint[y, x] > 0) & inside_depth).sum())
  return {'annotated_paint_pixels_in_depth_band': denominator,
    'annotated_paint_capture_recall': recall/denominator if denominator else None,
    'paint_fraction_of_visible_roi': float(painted.sum()/valid.sum()) if valid.any() else None,
    'rows_with_visible_annotated_paint_fraction': float(rows.sum()/valid.all(1).sum()) if valid.all(1).any() else None,
    'paint_center_absolute_offset_median_m': float(np.nanmedian(np.abs(centers-31.5))*.9/63) if rows.any() else None,
    'paint_captured_by_center_third_fraction': float(painted[:, 21:43].sum()/painted.sum()) if painted.any() else None}


def resampling_quality(rgb, line, rpy, intrinsics, near, far):
  """A reproducible warp/resize round-trip probe, not optical ground truth.

  Compare upscaled 64x192 against a direct 256x768 warp of the same decoded
  image. Both share compression and camera blur, so this cannot measure detail
  already lost before decoding or prove a model's recognition accuracy.
  """
  import cv2
  small_maps, _ = projection_maps(line, rpy, intrinsics, near, far)
  dense_maps, distances = projection_maps(line, rpy, intrinsics, near, far, 768, 256)
  small = rgb_strip(rgb, small_maps)
  direct = rgb_strip(rgb, dense_maps)
  reconstructed = cv2.resize(small, (256, 768), interpolation=cv2.INTER_LINEAR)
  valid = in_frame(dense_maps, rgb.shape[1], rgb.shape[0])
  # Avoid the unverified bottom/hood region in this diagnostic; this does NOT
  # certify visibility of the remaining rows for runtime inference.
  valid &= dense_maps[1] < rgb.shape[0]*.86
  error = np.abs(direct.astype(np.float32)-reconstructed.astype(np.float32)).mean(2)
  result = {}
  for name, lo, hi in [('near', 3, 15), ('middle', 15, 25), ('far', 25, 46.01), ('all', near, far+.01)]:
    use = valid & ((distances >= lo) & (distances < hi))[:, None]
    result[name] = {'samples': int(use.sum()), 'mae_8bit': float(error[use].mean()) if use.any() else None}
  return result


def audit(manifest, inventory, output):
  import av
  import cv2
  cv2.setNumThreads(1)
  rows = json.loads(manifest.read_text(encoding='utf-8'))
  sources = json.loads(inventory.read_text(encoding='utf-8'))['sources']
  output.mkdir(parents=True, exist_ok=True)
  for name in ('frames', 'strips', 'previews'):
    (output/name).mkdir(exist_ok=True)
  grouped = defaultdict(list)
  for row in rows:
    grouped[row['route'], row['segment']].append(row)
  records = []
  for source in sources:
    selected = grouped[source['route'], source['segment']]
    if not selected:
      continue
    identity, models, calibrations, _, cameras, encodes = metadata(source)
    times = [c[0] for c in calibrations]
    for quality, video in source['videos'].items():
      targets = defaultdict(list)
      for row in selected:
        if row['video_source'] == quality:
          targets[row['frame_id'], row['camera_eof_ns']].append(row)
      needed = {index: key for index, key in encodes[quality].items() if key in targets}
      if not needed:
        continue
      with av.open(video['path']) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
          if index not in needed:
            continue
          key = needed[index]
          model = models[key]
          rpy = calibrations[bisect_right(times, key[1])-1][1]
          camera = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam
          k = camera.intrinsics.copy()
          k[0] *= frame.width/camera.width
          k[1] *= frame.height/camera.height
          rgb = frame.to_ndarray(format='rgb24')
          # Encoded YUV reformatted as NV12 is only a domain probe, not live IPC.
          nv = frame.reformat(format='nv12')
          raw_nv = nv.to_ndarray().tobytes()
          frame_name = f"frames/{source['id']}-{index}-{quality}.png"
          Image.fromarray(rgb).save(output/frame_name)
          for row in targets[key]:
            line = model['lines'][1 if row['side'] == 'left' else 2]
            result = {'id': row['id'], 'scene_id': row['scene_id'], 'split': row['split'], 'video_source': quality,
              'side': row['side'], 'verified': row['verified'], 'type': row['type'], 'color': row['color'],
              'lighting': row['lighting'], 'speed_mps': row['speed_mps'], 'curve': row['curve'],
              'frame': frame_name, 'frame_size': [frame.width, frame.height], 'rpy': rpy, 'intrinsics': k.tolist(),
              'line': line, 'model_leads': model['leads'], 'lane_change': row['lane_change'], 'ranges': {}}
            page = Image.new('RGB', (1500, 780), '#202020')
            context = Image.fromarray(rgb)
            context.thumbnail((1008, 570))
            drawing = ImageDraw.Draw(context)
            sx, sy = context.width/frame.width, context.height/frame.height
            for j, (near, far) in enumerate(RANGES):
              tag = f'{near:g}-{far:g}'
              maps, ds = projection_maps(line, rpy, k, near, far)
              valid = in_frame(maps, frame.width, frame.height)
              strip = rgb_strip(rgb, maps)
              strip[~valid] = 0
              path = 'strips/'+row['id']+'-'+tag+'.png'
              Image.fromarray(strip).save(output/path)
              q = geometry_quality(maps, ds, frame.width, frame.height)
              q['image'] = path
              # Round-trip through the SAME encoded YUV planes isolates the
              # decode/conversion path, not optical detail lost by compression.
              nv_strip = nv12_strip(raw_nv, frame.width, frame.height, frame.width, frame.width*frame.height, maps)
              q['decoded_nv12_vs_rgb_mae_8bit'] = float(np.abs(nv_strip.astype(float)-strip)[valid].mean())
              q['bands_sharpness_proxy'] = {}
              gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
              lap = cv2.Laplacian(gray, cv2.CV_32F)
              for band, lo, hi in [('near', 3, 15), ('middle', 15, 25), ('far', 25, 46.01)]:
                use = (ds >= lo) & (ds < hi) & valid.all(1)
                q['bands_sharpness_proxy'][band] = float(lap[use].var()) if use.any() else None
              result['ranges'][tag] = q
              color = ('#ff8040', '#40ff80', '#40a0ff', '#ffffff')[j]
              for edge in (0, -1):
                points = [(float(x*sx), float(y*sy)) for x, y in zip(maps[0][:, edge], maps[1][:, edge], strict=True)]
                drawing.line(points, fill=color, width=2)
              page.paste(Image.fromarray(strip).resize((96, 576)), (1040+j*114, 95))
              ImageDraw.Draw(page).text((1040+j*114, 55), tag+'m', fill=color)
              ImageDraw.Draw(page).text((1040+j*114, 73), f"FOV {q['complete_row_fraction']:.0%}", fill=color)
            page.paste(context, (0, 65))
            draw = ImageDraw.Draw(page)
            draw.text((10, 10), row['id']+' '+row['split'], fill='white')
            draw.text((10, 30), 'BLACK = unavailable, never paint gap. Out-of-frame 3m is rejected for inference.', fill='#ff8040')
            page.save(output/'previews'/f"{row['id']}.jpg", quality=96)
            records.append(result)
    print(source['id'], 'audited', len(records), flush=True)
  (output/'audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
  return records


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--inventory', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  audit(args.manifest, args.inventory, args.output)
