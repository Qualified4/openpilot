"""fixtures/events.{bin,json}: a few real rlog events (original bytes) and their pycapnp values, for messages.rs tests.

  python openpilot/selfdrive/carrot/lane_type/rust/tools/make_event_fixture.py <rlog.zst> <fixtures dir>
"""
import json
from pathlib import Path
import sys

from tools.lane_type.stripnet_rust_parity import events  # noqa: TID251

WANT = {'modelV2': 2, 'carState': 2, 'liveCalibration': 1, 'deviceState': 1, 'roadCameraState': 1, 'radarState': 2}


def values(which, e):
  out = {'which': which, 'logMonoTime': e.logMonoTime, 'valid': e.valid}
  m = getattr(e, which)
  if which == 'modelV2':
    out.update(frameId=m.frameId, timestampEof=m.timestampEof, laneLineProbs=list(m.laneLineProbs),
               laneChangeState=str(m.meta.laneChangeState),
               laneLines=[{'x': list(l.x), 'y': list(l.y), 'z': list(l.z)} for l in m.laneLines])
  elif which == 'carState':
    out.update(vEgo=m.vEgo, leftBlinker=m.leftBlinker, rightBlinker=m.rightBlinker)
  elif which == 'liveCalibration':
    out.update(calStatus=str(m.calStatus), rpyCalib=list(m.rpyCalib))
  elif which == 'deviceState':
    out.update(started=m.started, deviceType=str(m.deviceType))
  elif which == 'roadCameraState':
    out.update(sensor=str(m.sensor))
  elif which == 'radarState':
    out.update(leadStatus=m.leadOne.status, leadDRel=m.leadOne.dRel)
  return out


def main(rlog, directory):
  left, raws, rows = dict(WANT), [], []
  for which, raw, e in events(rlog):
    if left.get(which, 0) > 0 and (which != 'radarState' or e.radarState.leadOne.status or left[which] == 1):
      left[which] -= 1
      raws.append(raw)
      rows.append(values(which, e))
    if not any(left.values()):
      break
  d = Path(directory)
  (d/'events.bin').write_bytes(b''.join(raws))
  (d/'events.json').write_text(json.dumps(rows, indent=0))
  print(len(rows), 'events', sum(map(len, raws)), 'bytes')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
