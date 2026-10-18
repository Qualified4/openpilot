"""Offline source-pixel/odometry audit of the frozen 80 reviewed frames.

SIFT is an independent, expensive audit tool, never an inference dependency.
Feature matches are image evidence, not manually approved physical ground truth.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import av
import numpy as np
from PIL import Image

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.carrot.lane_type.pixel_ribbon import collect_pixels
from openpilot.selfdrive.carrot.lane_type.pixel_tracker import candidates, road_points, project_road, path_y, to_world, to_local
from tools.lane_type.dataset import read_events  # noqa: TID251
from tools.lane_type.retrain_data import metadata  # noqa: TID251
from tools.lane_type.validate_pixel_tracking import write_json, real, score, gallery  # noqa: TID251


def integrate_pose(stamps, speeds, rates, query):
  """Calibrated x forward/y right/z down. Positive yaw turns towards +y.

  Trapezoidal inputs and midpoint SE(2); integrate at every recorded sample,
  inserting exact camera EOF times. Input timestamps are measurement times.
  """
  stamps, query = np.asarray(stamps, np.int64), np.asarray(query, np.int64)
  times = np.unique(np.r_[stamps[(stamps >= query[0]) & (stamps <= query[-1])], query])
  relative = (stamps-query[0])/1e9
  t = (times-query[0])/1e9
  v, omega = np.interp(t, relative, speeds), np.interp(t, relative, rates)
  result, pose = {}, np.zeros(3)
  for i, stamp in enumerate(times):
    if i:
      dt = t[i]-t[i-1]
      dyaw = .5*(omega[i]+omega[i-1])*dt
      distance = .5*(v[i]+v[i-1])*dt
      heading = pose[2]+dyaw/2
      pose[:2] += distance*np.array([np.cos(heading), np.sin(heading)])
      pose[2] += dyaw
    result[int(stamp)] = pose.copy()
  return np.array([result[int(stamp)] for stamp in query])


def odometry(output):
  rows = json.loads((output/'clips.json').read_text())
  inventory = json.loads(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json').read_text())
  reports = []
  for route, segment in sorted({(r['route'], r['segment']) for r in rows}):
    cache = output/'telemetry'/f'{route}-{segment}.json'
    if cache.exists():
      data = json.loads(cache.read_text())
    else:
      source = next(s for s in inventory['sources'] if (s['route'], s['segment']) == (route, segment))
      _, _, calibrations, cars, _, _ = metadata(source)
      ct, gyro = [c[0] for c in calibrations], []
      for e in read_events(Path(source['log']['path'])):
        if e.which() != 'livePose':
          continue
        p = e.livePose
        ci = bisect_right(ct, int(e.logMonoTime))-1
        if ci < 0:
          continue
        omega = np.array([p.angularVelocityDevice.x, p.angularVelocityDevice.y, p.angularVelocityDevice.z])
        rate = float((rot_from_euler(calibrations[ci][1]).T @ omega)[2])
        gyro.append([int(p.timestamp), int(e.logMonoTime), rate,
          bool(e.valid and p.inputsOK and p.sensorsOK and p.angularVelocityDevice.valid)])
      data = {'cars': cars, 'gyro': gyro}
      write_json(cache, data)
    cars, gyro = np.array(data['cars'], float), np.array(data['gyro'], float)
    gyro = gyro[gyro[:, 3] == 1]
    gyro = gyro[np.argsort(gyro[:, 0])]
    for clip in sorted({r['clip'] for r in rows if (r['route'], r['segment']) == (route, segment)}):
      subset = [r for r in rows if r['clip'] == clip]
      stamps = np.array([r['stamp'] for r in subset], np.int64)
      times = np.unique(np.r_[cars[:, 0].astype(np.int64), gyro[:, 0].astype(np.int64)])
      # Rebase timestamps before interpolation to avoid unnecessary float precision loss.
      t = (times-stamps[0])/1e9
      speeds = np.interp(t, (cars[:, 0]-stamps[0])/1e9, cars[:, 1])
      rates = np.interp(t, (gyro[:, 0]-stamps[0])/1e9, gyro[:, 2])
      poses = integrate_pose(times, speeds, rates, stamps)
      old = np.array([r.get('pose_before', r['pose']) for r in subset])
      for row, pose in zip(subset, poses, strict=True):
        row.setdefault('pose_before', row['pose'])
        row['pose'] = pose.tolist()
        row['odometry'] = 'all_samples_measurement_timestamp_offline_interpolation'
      nearby = gyro[(gyro[:, 0] >= stamps[0]-150_000_000) & (gyro[:, 0] <= stamps[-1]+150_000_000)]
      reports.append({'clip': clip, 'end_translation_change_m': float(np.linalg.norm(poses[-1, :2]-old[-1, :2])),
        'end_yaw_change_rad': float(poses[-1, 2]-old[-1, 2]),
        'speed_range_mps': [float(np.interp(stamps[0], cars[:, 0], cars[:, 1])), float(np.interp(stamps[-1], cars[:, 0], cars[:, 1]))],
        'gyro_arrival_delay_ms_median': float(np.median((nearby[:, 1]-nearby[:, 0])/1e6)),
        'gyro_measurement_max_gap_ms': float(np.max(np.diff(nearby[:, 0]))/1e6),
        'causal_runtime_validated': False})
      print('odometry', clip, reports[-1], flush=True)
  write_json(output/'clips.json', rows)
  write_json(output/'odometry.json', reports)


def integrate_road_pose(stamps, speeds, omega, query):
  """Full body rotation for pixel correspondence; does not estimate heave."""
  stamps, query = np.asarray(stamps, np.int64), np.asarray(query, np.int64)
  times = np.unique(np.r_[stamps[(stamps >= query[0]) & (stamps <= query[-1])], query])
  t, relative = (times-query[0])/1e9, (stamps-query[0])/1e9
  v = np.interp(t, relative, speeds)
  rates = np.column_stack([np.interp(t, relative, omega[:, j]) for j in range(3)])
  result, transform = {}, np.eye(4)
  for i, stamp in enumerate(times):
    if i:
      dt = t[i]-t[i-1]
      step = .5*(rates[i]+rates[i-1])*dt
      midpoint = transform[:3, :3] @ cv2.Rodrigues(step/2)[0]
      transform[:3, 3] += midpoint @ np.array([.5*(v[i]+v[i-1])*dt, 0., 0.])
      transform[:3, :3] = transform[:3, :3] @ cv2.Rodrigues(step)[0]
    result[int(stamp)] = transform.copy()
  return np.array([result[int(stamp)] for stamp in query])


def attitude(output):
  """Keep SE(2) baseline; add measured roll/pitch integration as a control."""
  inventory = json.loads(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json').read_text())
  documents = {name: json.loads((output/name).read_text()) for name in ('clips.json', 'pairs.json')}
  rows = documents['clips.json']
  for route, segment in sorted({(r['route'], r['segment']) for r in rows}):
    path = output/'telemetry-attitude'/f'{route}-{segment}.json'
    if path.exists():
      data = json.loads(path.read_text())
    else:
      source = next(s for s in inventory['sources'] if (s['route'], s['segment']) == (route, segment))
      _, _, calibrations, cars, _, _ = metadata(source)
      ct, gyro, timing = [c[0] for c in calibrations], [], {}
      for e in read_events(Path(source['log']['path'])):
        if e.which() == 'roadCameraState':
          c = e.roadCameraState
          timing[str(c.frameId)] = [int(c.timestampSof), int(c.timestampEof), int(c.integLines)]
        if e.which() != 'livePose':
          continue
        p = e.livePose
        ci = bisect_right(ct, int(e.logMonoTime))-1
        if ci < 0 or not (e.valid and p.inputsOK and p.sensorsOK and p.angularVelocityDevice.valid):
          continue
        w = rot_from_euler(calibrations[ci][1]).T @ np.array([p.angularVelocityDevice.x, p.angularVelocityDevice.y, p.angularVelocityDevice.z])
        gyro.append([int(p.timestamp), *w.tolist()])
      data = {'cars': cars, 'gyro': gyro, 'camera_timing': timing}
      write_json(path, data)
    cars, gyro = np.array(data['cars'], float), np.array(data['gyro'], float)
    gyro = gyro[np.argsort(gyro[:, 0])]
    times = np.unique(np.r_[cars[:, 0].astype(np.int64), gyro[:, 0].astype(np.int64)])
    speeds = np.interp((times-times[0])/1e9, (cars[:, 0]-times[0])/1e9, cars[:, 1])
    rates = np.column_stack([np.interp((times-times[0])/1e9, (gyro[:, 0]-times[0])/1e9, gyro[:, j]) for j in (1, 2, 3)])
    for document in documents.values():
      subset = [r for r in document if (r['route'], r['segment']) == (route, segment)]
      for clip in sorted({r['clip'] for r in subset}):
        group = [r for r in subset if r['clip'] == clip]
        stamps = [r['stamp'] for r in group]
        poses = integrate_road_pose(times, speeds, rates, stamps)
        for row, pose in zip(group, poses, strict=True):
          row['road_pose'] = pose.tolist()
          row['camera_timing_sof_eof_integ'] = data['camera_timing'][str(row['frame_id'])]
    print('attitude', route, segment, flush=True)
  for name, document in documents.items():
    write_json(output/name, document)


def matrix(row):
  return np.array(row['intrinsics']) @ view_frame_from_device_frame @ rot_from_euler(row['rpy'])


def prepare_pairs(output):
  """Decode only immediate neighbours of the same frozen 80 source frames."""
  rows = json.loads((output/'clips.json').read_text())
  inventory = json.loads(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json').read_text())
  combined = []
  for route, segment in sorted({(r['route'], r['segment']) for r in rows}):
    subset = [r for r in rows if (r['route'], r['segment']) == (route, segment)]
    source = next(s for s in inventory['sources'] if (s['route'], s['segment']) == (route, segment))
    identity, models, calibrations, cars, cameras, encodes = metadata(source)
    ct, st = [c[0] for c in calibrations], [c[0] for c in cars]
    needed = {r['video_index']+1: r for r in subset}
    data = json.loads((output/'telemetry'/f'{route}-{segment}.json').read_text())
    gyro = np.array(data['gyro'], float)
    gyro = gyro[gyro[:, 3] == 1]
    gyro = gyro[np.argsort(gyro[:, 0])]
    car = np.array(data['cars'], float)
    samples = np.unique(np.r_[gyro[:, 0].astype(np.int64), car[:, 0].astype(np.int64)])
    speeds = np.interp((samples-samples[0])/1e9, (car[:, 0]-samples[0])/1e9, car[:, 1])
    rates = np.interp((samples-samples[0])/1e9, (gyro[:, 0]-samples[0])/1e9, gyro[:, 2])
    with av.open(source['videos']['fcamera']['path']) as container:
      container.streams.video[0].thread_count = 1
      for i, frame in enumerate(container.decode(video=0)):
        if i not in needed:
          continue
        old = needed[i]
        key = encodes['fcamera'][i]
        if key not in models or key not in cameras:
          continue
        ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
        k = DEVICE_CAMERAS[identity['device'], cameras[key]].fcam.intrinsics.copy()
        path = output/'neighbours'/f'{route}-{segment}-{key[0]}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = frame.to_ndarray(format='rgb24')
        Image.fromarray(rgb).save(path)
        # Each pair has its own zero pose, avoiding any implicit clip timestamp.
        poses = integrate_pose(samples, speeds, rates, [old['stamp'], key[1]])
        pair = f"{old['clip']}-pair-{old['frame_id']}"
        combined.append(dict(old, clip=pair, pose=[0., 0., 0.], pose_before=[0., 0., 0.]))
        combined.append(dict(old, clip=pair, frame_id=key[0], stamp=key[1], video_index=i,
          frame=str(path.resolve()), line=models[key]['lines'][1], probability=models[key]['probs'][1],
          leads=models[key]['leads'], rpy=calibrations[ci][1], intrinsics=k.tolist(),
          lane_change=bool(models[key]['lane_change'] or cars[si][2]), pose=poses[1].tolist(), pose_before=poses[1].tolist(),
          rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    print('prepared exact neighbours', route, segment, flush=True)
  write_json(output/'pairs.json', combined)


def road_feature_mask(row, rgb, path=None):
  mask = np.zeros(rgb.shape[:2], np.uint8)
  ribbon = collect_pixels(rgb, row['line'], 1., row['rpy'], np.array(row['intrinsics']), half_width_m=.9)
  for r in ribbon['rows']:
    if not r['geometry_visible']:
      continue
    d = r['distance_m']
    lateral = path_y(path, row['line'], d) if path else np.interp(d, row['line']['x'], row['line']['y'])
    p = project_road(np.array([[d, lateral]]), row['line'], matrix(row))[0]
    # Static barrier/post corners are NOT on the road plane. Include the left
    # stripe and the road on its ego side, excluding the roadside structures.
    half_paint = max(4, int(row['intrinsics'][0][0]*.18/d))
    left = max(0, int(p[0])-half_paint)
    right = min(rgb.shape[1], int(p[0])+max(40, int(row['intrinsics'][0][0]*.8/d)))
    mask[r['y'], left:right] = 255
  return mask


def features(row, path=None, descriptors=True):
  rgb = np.array(Image.open(row['frame']))
  if not descriptors:
    return rgb, None, None
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  mask = road_feature_mask(row, rgb, path)
  keypoints, desc = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.025, edgeThreshold=5).detectAndCompute(gray, mask)
  return rgb, keypoints, desc


def corner_matches(old_rgb, rgb, mask):
  """Independent short-interval corner tracking, excluding aperture edges."""
  a_gray, b_gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (old_rgb, rgb)]
  points = cv2.goodFeaturesToTrack(a_gray, maxCorners=400, qualityLevel=.02, minDistance=8, mask=mask, blockSize=5)
  if points is None:
    return np.empty((0, 2)), np.empty((0, 2))
  eigen = cv2.cornerEigenValsAndVecs(a_gray, 5, 3)
  xy = np.rint(points[:, 0]).astype(int)
  values = eigen[xy[:, 1], xy[:, 0]]
  # A long line has one vanishing eigenvalue; it is never a physical landmark.
  points = points[values[:, 1] >= .08*values[:, 0]]
  if not len(points):
    return np.empty((0, 2)), np.empty((0, 2))
  forward, status, error = cv2.calcOpticalFlowPyrLK(a_gray, b_gray, points, None, winSize=(21, 21), maxLevel=4)
  backward, back_status, _ = cv2.calcOpticalFlowPyrLK(b_gray, a_gray, forward, None, winSize=(21, 21), maxLevel=4)
  good = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
  good &= np.linalg.norm(backward[:, 0]-points[:, 0], axis=1) <= .5
  good &= error[:, 0] <= 20
  return points[good, 0], forward[good, 0]


def correspondence(output, pairs=False, corners=False):
  rows = json.loads((output/('pairs.json' if pairs else 'clips.json')).read_text())
  records = []
  baseline = json.loads(Path('.analysis/pixel-ribbon-tracking-20261018/real-results.json').read_text())
  for clip in sorted({r['clip'] for r in rows}):
    previous = None
    anchor = int(clip.split('-pair-')[-1]) if pairs else None
    for row in [r for r in rows if r['clip'] == clip]:
      reference = next((r for r in baseline if r['frame_id'] == (anchor or row['frame_id'])), None)
      path = reference['tracks'][0] if reference and reference['tracks'] else None
      rgb, kp, desc = features(row, path, descriptors=not corners)
      if previous is not None and (corners or (desc is not None and previous[3] is not None)):
        old, old_rgb, old_kp, old_desc = previous
        if corners:
          a, b = corner_matches(old_rgb, rgb, road_feature_mask(old, old_rgb, path))
        else:
          matcher = cv2.BFMatcher()
          forward = matcher.knnMatch(old_desc, desc, k=2)
          back = matcher.knnMatch(desc, old_desc, k=2)
          reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in back if len(pair) == 2 and pair[0].distance < .7*pair[1].distance}
          matches = [pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < .7*pair[1].distance and
                     reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
          a = np.array([old_kp[m.queryIdx].pt for m in matches], float).reshape(-1, 2)
          b = np.array([kp[m.trainIdx].pt for m in matches], float).reshape(-1, 2)
        # Image-only planar consensus: independent of ego prediction/errors.
        inliers = np.zeros(len(a), bool)
        if len(a) >= 4:
          _, flags = cv2.findHomography(a, b, cv2.RANSAC, 2., maxIters=3000, confidence=.999)
          if flags is not None:
            inliers = flags[:, 0].astype(bool)
        if len(a):
          local = road_points(a, old['line'], matrix(old))[:, :2]
          now = road_points(b, row['line'], matrix(row))[:, :2]
          world = to_world(local, old['pose'])
          predicted = to_local(world, row['pose'])
          px = project_road(predicted, row['line'], matrix(row))
          before = to_local(to_world(local, old.get('pose_before', old['pose'])), row.get('pose_before', row['pose']))
          old_px = project_road(before, row['line'], matrix(row))
          source3d = road_points(a, old['line'], matrix(old))
          full_pose = np.linalg.inv(np.array(row.get('road_pose', np.eye(4)))) @ np.array(old.get('road_pose', np.eye(4)))
          corrected3d = source3d @ full_pose[:3, :3].T+full_pose[:3, 3]
          full_px = corrected3d @ matrix(row).T
          full_px = full_px[:, :2]/full_px[:, 2, None]
          valid = np.isfinite(predicted).all(1) & np.isfinite(now).all(1)
          valid &= (local[:, 0] >= 4.5) & (local[:, 0] <= 20) & (now[:, 0] >= 4.5) & (now[:, 0] <= 20)
          errors = np.linalg.norm(px-b, axis=1)
          metric = np.linalg.norm(predicted-now, axis=1)
          selected = valid & inliers
          # Predictions with reversed signs are controls, never fit to the matches.
          reverse_pose = np.array(row['pose'])*-1
          controls = project_road(to_local(world, reverse_pose), row['line'], matrix(row))
          records.append({'clip': clip, 'from': old['frame_id'], 'to': row['frame_id'], 'mutual_unique_matches': len(a),
            'valid_road_matches': int(valid.sum()), 'image_planar_inliers': int(selected.sum()),
            'all_valid_error_px': errors[valid].tolist(), 'error_px': errors[selected].tolist(),
            'error_before_px': np.linalg.norm(old_px-b, axis=1)[selected].tolist(),
            'full_attitude_error_px': np.linalg.norm(full_px-b, axis=1)[selected].tolist(),
            'full_attitude_predicted_xy': full_px[selected].tolist(),
            'error_m': metric[selected].tolist(), 'reverse_motion_error_px': np.linalg.norm(controls-b, axis=1)[selected].tolist(),
            'source_xy': a[selected].tolist(), 'matched_xy': b[selected].tolist(), 'predicted_xy': px[selected].tolist(),
            'depth_before_m': local[selected, 0].tolist(), 'depth_after_m': now[selected, 0].tolist()})
          if selected.any():
            panel = np.concatenate([old_rgb, rgb], axis=1).copy()
            for i, (p, q, pred) in enumerate(zip(a[selected], b[selected], px[selected], strict=True)):
              p, q, pred = np.rint(p).astype(int), np.rint(q).astype(int), np.rint(pred).astype(int)
              cv2.circle(panel, tuple(p), 4, (255, 0, 100), 1)
              cv2.circle(panel, (q[0]+rgb.shape[1], q[1]), 4, (0, 255, 50), 1)
              cv2.drawMarker(panel, (pred[0]+rgb.shape[1], pred[1]), (255, 80, 0), cv2.MARKER_CROSS, 8)
              cv2.putText(panel, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 0), 1)
              cv2.putText(panel, str(i), (q[0]+rgb.shape[1], q[1]), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 0), 1)
            folder = 'corner-correspondence' if corners else 'pair-correspondence' if pairs else 'correspondence'
            target = output/folder/f"{clip}-{old['frame_id']}-{row['frame_id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(target)
      previous = row, rgb, kp, desc
    print('matched', clip, flush=True)
  write_json(output/('corner-correspondence.json' if corners else 'pair-correspondence.json' if pairs else 'correspondence.json'), records)


def trace_audit(output):
  rows = json.loads((output/'clips.json').read_text())
  annotations = json.loads((output/'annotation/paint-revised.json').read_text())
  baseline = json.loads(Path('.analysis/pixel-ribbon-tracking-20261018/real-results.json').read_text())
  current = json.loads((output/'real-results.json').read_text()) if (output/'real-results.json').exists() else []
  records = []
  for a in annotations[:2]:
    row = next(r for r in rows if r['frame_id'] == a['frame_id'])
    rgb = np.array(Image.open(row['frame']))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ribbon = collect_pixels(rgb, row['line'], 1., row['rpy'], np.array(row['intrinsics']), half_width_m=.9)
    gray, rr, found = candidates(rgb, ribbon, row['line'], matrix(row))
    result = next(r for r in baseline if r['frame_id'] == a['frame_id'])
    after = next((r for r in current if r['frame_id'] == a['frame_id']), None)
    truth = np.zeros(rgb.shape[:2], np.uint8)
    for polygon in a['paint_polygons']:
      cv2.fillPoly(truth, [np.rint(np.array(polygon)*[rgb.shape[1], rgb.shape[0]]/a['coordinate_size']).astype(np.int32)], 1)
    info = []
    for r in rr:
      y = r['y']
      xx = np.flatnonzero(truth[y])
      if not len(xx) or not r['geometry_visible']:
        continue
      road = float(np.percentile(gray[y, r['left_x']:r['right_x']+1], 30))
      record = {'y': y, 'trace_x': [int(xx.min()), int(xx.max())], 'hsv_median': np.median(hsv[y, xx], axis=0).tolist(),
        'gray_road': road, 'trace_gray_median': float(np.median(gray[y, xx])),
        'stage_reason_before': next((p['reason'] for p in result['profile'] if p['y'] == y), 'outside'),
        'stage_reason_after': next((p['reason'] for p in after['profile'] if p['y'] == y), 'outside') if after else None,
        'candidates': [found[i] for i in r['candidates']]}
      info.append(record)
    records.append({'frame_id': row['frame_id'], 'trace_rows': info,
      'before_reasons': dict(Counter(r['stage_reason_before'] for r in info)),
      'after_reasons': dict(Counter(r['stage_reason_after'] for r in info))})
    panels = []
    for name, folder in [('SOURCE', None), ('LEGACY PARTIAL TRACE', None), ('BEFORE SELECTED', Path('.analysis/pixel-ribbon-tracking-20261018')),
                         ('AFTER SELECTED', output)]:
      panel = rgb.copy()
      if name == 'LEGACY PARTIAL TRACE':
        panel[truth != 0] = (255, 100, 0)
      elif folder:
        xy = np.load(folder/'real'/f"{row['clip']}-{row['frame_id']}.npz")['selected_xy']
        panel[xy[:, 1], xy[:, 0]] = (0, 255, 100)
      cv2.putText(panel, name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 0), 2)
      panels.append(panel)
    image = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0)
    Image.fromarray(image).save(output/f"white-layers-{row['frame_id']}.png")
  write_json(output/'white-trace-audit.json', records)


def endpoint_audit(output):
  rows = json.loads((output/'pairs.json').read_text())
  annotation = json.loads((output/'annotation/endpoints.json').read_text())
  records = []
  for a in annotation['tracks']:
    old = next(r for r in rows if r['frame_id'] == a['from'])
    row = next(r for r in rows if r['clip'] == old['clip'] and r['frame_id'] == a['to'])
    points = []
    for r, window in zip((old, row), a['source_windows'], strict=True):
      rgb = np.array(Image.open(r['frame']))
      gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
      x0, y0, x1, y1 = window
      binary = (gray[y0:y1, x0:x1] > a['gray_threshold']).astype(np.uint8)
      count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
      if count < 2:
        raise ValueError('reviewed endpoint has no source component')
      selected = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])
      yy, xx = np.nonzero(labels == selected)
      edge = yy.max()  # this audit uses only identifiable near endpoints
      points.append([float(xx[yy == edge].mean()+x0), float(edge+y0)])
    source, observed = np.array([points[0]]), np.array([points[1]])
    xyz = road_points(source, old['line'], matrix(old))
    predicted = project_road(to_local(to_world(xyz[:, :2], old['pose']), row['pose']), row['line'], matrix(row))
    transform = np.linalg.inv(np.array(row['road_pose'])) @ np.array(old['road_pose'])
    new = xyz @ transform[:3, :3].T+transform[:3, 3]
    px = new @ matrix(row).T
    full = px[:, :2]/px[:, 2, None]
    backprojected = road_points(observed, row['line'], matrix(row))
    records.append(dict(a, source_xy=source[0].tolist(), matched_xy=observed[0].tolist(),
      se2_error_px=float(np.linalg.norm(predicted-observed)),
      full_attitude_error_px=float(np.linalg.norm(full-observed)),
      se2_error_m=float(np.linalg.norm(to_local(to_world(xyz[:, :2], old['pose']), row['pose'])-backprojected[:, :2])),
      predicted_se2_xy=predicted[0].tolist(), predicted_attitude_xy=full[0].tolist(),
      dt_ms=(row['stamp']-old['stamp'])/1e6,
      source_road_height_m=float(xyz[0, 2]), matched_model_road_height_m=float(backprojected[0, 2])))
    rgb = np.array(Image.open(row['frame']))
    for point, color in [(observed[0], (0, 255, 0)), (predicted[0], (255, 0, 0)), (full[0], (0, 180, 255))]:
      cv2.drawMarker(rgb, tuple(np.rint(point).astype(int)), color, cv2.MARKER_CROSS, 11)
    Image.fromarray(rgb).save(output/f"endpoint-prediction-{a['from']}.png")
  write_json(output/'endpoint-results.json', {'annotation_approval': annotation['approval'], 'records': records})


def slow_audit(output):
  from tools.lane_type.validate_pixel_tracking import synthetic_frame  # noqa: TID251
  from openpilot.selfdrive.carrot.lane_type.pixel_tracker import PaintTracker
  records = []
  for speed in (0., 1., 2., 3., 5.):
    tracker, observations = PaintTracker('left'), []
    for i in range(16):
      pose = np.array([speed*.2*i, 0., 0.])
      rgb, line, k, rpy, _, _, _ = synthetic_frame(pose)
      result = tracker.update(rgb, line, 1., rpy, k, i*200_000_000, pose=pose, pose_error_m=.02)
      observations.append({'time_s': i*.2, 'type': result['type'], 'distinct_m': result['distinct_m']})
    first = next((r['time_s'] for r in observations if r['type'] == 'solid'), None)
    records.append({'speed_mps': speed, 'first_solid_s': first, 'observations': observations,
      'unknown_fraction': sum(r['type'] == 'unknown' for r in observations)/16, 'physical_ground_truth': 'synthetic solid, exact ego pose'})
    print('solid22m', speed, first, flush=True)
  write_json(output/'slow-solid22m.json', records)


def dense_control(output):
  """Cadence ablation on the same five intervals; never a runtime-rate change."""
  rows = json.loads((output/'clips.json').read_text())
  inventory = json.loads(Path('.analysis/archive/2026-10-18/lane-type-retrain/data/inventory.json').read_text())
  destination = output/'dense-control'
  write_json(destination/'ROI_REVIEW_HOLD.json', {'training_resumed': False, 'model_promotion_allowed': False,
    'reason': 'Cadence ablation only; physical alignment and device CPU cost unqualified'})
  result = []
  for route, segment in sorted({(r['route'], r['segment']) for r in rows}):
    subset = [r for r in rows if (r['route'], r['segment']) == (route, segment)]
    source = next(s for s in inventory['sources'] if (s['route'], s['segment']) == (route, segment))
    _, models, calibrations, cars, cameras, encodes = metadata(source)
    ct, st = [c[0] for c in calibrations], [c[0] for c in cars]
    needed = {}
    for clip in sorted({r['clip'] for r in subset}):
      group = [r for r in subset if r['clip'] == clip]
      for i in range(group[0]['video_index'], group[-1]['video_index']+1):
        needed[i] = group[0]
    with av.open(source['videos']['fcamera']['path']) as container:
      container.streams.video[0].thread_count = 1
      for i, frame in enumerate(container.decode(video=0)):
        if i not in needed:
          continue
        template, key = needed[i], encodes['fcamera'].get(i)
        if key not in models or key not in cameras:
          continue
        ci, si = bisect_right(ct, key[1])-1, bisect_right(st, key[1])-1
        path = destination/'frames'/f'{route}-{segment}-{key[0]}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = frame.to_ndarray(format='rgb24')
        Image.fromarray(rgb).save(path)
        result.append(dict(template, frame_id=key[0], stamp=key[1], video_index=i, frame=str(path.resolve()),
          rpy=calibrations[ci][1], line=models[key]['lines'][1], probability=models[key]['probs'][1],
          leads=models[key]['leads'], lane_change=bool(models[key]['lane_change'] or cars[si][2]),
          rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
  for clip in sorted({r['clip'] for r in result}):
    originals = [r for r in rows if r['clip'] == clip]
    times = np.array([r['stamp'] for r in originals], np.int64)
    before = np.array([r.get('pose_before', r['pose']) for r in originals])
    for row in [r for r in result if r['clip'] == clip]:
      reference = [float(np.interp((row['stamp']-times[0])/1e9, (times-times[0])/1e9, before[:, j])) for j in range(3)]
      row['pose'] = row['pose_before'] = reference
      row['pose_error_m'] = max(r['pose_error_m'] for r in originals)
      row['pose_error_kind'] = 'conservative_original_clip_proxy_not_measured_registration'
  write_json(destination/'clips.json', result)
  # Recompute every pose from measured inputs rather than inheriting an anchor.
  odometry(destination)
  write_json(destination/'pairs.json', [])
  attitude(destination)
  print('dense exact frames', len(result), flush=True)
  if not (destination/'annotation').exists():
    shutil.copytree(output/'annotation', destination/'annotation')
  real(destination)
  score(destination)
  sources = json.loads((destination/'clips.json').read_text())
  results = json.loads((destination/'real-results.json').read_text())
  ids = {r['frame_id'] for r in rows}
  gallery(destination, [r for r in sources if r['frame_id'] in ids], [r for r in results if r['frame_id'] in ids])
  path = destination/'PREVIEW.html'
  html = path.read_text(encoding='utf-8').replace('원본 도색과 선택 결과 — 학습·모델 승격 보류',
    '20Hz 관측 누적 — 동일 80프레임을 5Hz로 미리보기 — 채택 보류')
  path.write_text(html, encoding='utf-8')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, default=Path('.analysis/pixel-ribbon-alignment-20261018'))
  p.add_argument('--odometry', action='store_true')
  p.add_argument('--matches', action='store_true')
  p.add_argument('--trace', action='store_true')
  p.add_argument('--prepare-pairs', action='store_true')
  p.add_argument('--pairs', action='store_true')
  p.add_argument('--attitude', action='store_true')
  p.add_argument('--corners', action='store_true')
  p.add_argument('--endpoints', action='store_true')
  p.add_argument('--slow', action='store_true')
  p.add_argument('--dense-control', action='store_true')
  args = p.parse_args()
  if args.odometry:
    odometry(args.output)
  if args.matches:
    correspondence(args.output)
  if args.prepare_pairs:
    prepare_pairs(args.output)
  if args.attitude:
    attitude(args.output)
  if args.pairs:
    correspondence(args.output, pairs=True)
  if args.corners:
    correspondence(args.output, pairs=True, corners=True)
  if args.trace:
    trace_audit(args.output)
  if args.endpoints:
    endpoint_audit(args.output)
  if args.slow:
    slow_audit(args.output)
  if args.dense_control:
    dense_control(args.output)
