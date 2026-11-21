"""Dump the cereal schema positions the Rust message reader (src/capnp.rs, src/messages.rs) uses.

Run from the repository root: python openpilot/selfdrive/carrot/lane_type/rust/tools/schema_offsets.py > \
openpilot/selfdrive/carrot/lane_type/rust/fixtures/schema_offsets.json. `cargo test` compares the Rust constants with it,
so a schema change shows up as a failing test (fix the constants, regenerate this file).
"""
import json
import sys

from openpilot.cereal import log, car  # noqa: F401

FIELDS = {
  'Event': ['logMonoTime', 'valid', 'modelV2', 'carState', 'liveCalibration', 'deviceState', 'roadCameraState', 'radarState',
            'customReservedRawData2'],
  'ModelDataV2': ['frameId', 'timestampEof', 'laneLines', 'laneLineProbs', 'meta'],
  'ModelDataV2.MetaData': ['laneChangeState'],
  'XYZTData': ['x', 'y', 'z'],
  'LiveCalibrationData': ['calStatus', 'rpyCalib'],
  'DeviceState': ['started', 'deviceType'],
  'FrameData': ['sensor'],
  'RadarState': ['leadOne'],
  'RadarState.LeadData': ['dRel', 'status'],
}
CAR_FIELDS = {'CarState': ['vEgo', 'leftBlinker', 'rightBlinker']}
ENUMS = {'LaneChangeState': log.LaneChangeState, 'CalStatus': log.LiveCalibrationData.Status,
         'DeviceType': log.InitData.DeviceType, 'ImageSensor': log.FrameData.ImageSensor}


def struct(module, name):
  s = module
  for part in name.split('.'):
    s = getattr(s, part)
  return s.schema


def field_info(schema, name):
  f = schema.fields[name]
  p = f.proto
  out = {}
  if p.discriminantValue != 65535:
    out['discriminant'] = p.discriminantValue
  if p.which() == 'slot':
    t = p.slot.type.which()
    out.update(kind=t, offset=p.slot.offset)
    d = p.slot.defaultValue
    if t in ('bool', 'uint32', 'uint64', 'float32', 'enum') and d.which() == t:
      v = getattr(d, t)
      out['default'] = v if t != 'float32' else float(v)
  else:
    out['kind'] = 'group'
  return out


def main():
  out = {'structs': {}, 'enums': {}}
  for module, table in ((log, FIELDS), (car, CAR_FIELDS)):
    for name, fields in table.items():
      schema = struct(module, name)
      entry = {'fields': {f: field_info(schema, f) for f in fields}}
      node = schema.node.struct
      entry['dataWords'], entry['pointers'] = node.dataWordCount, node.pointerCount
      if node.discriminantCount:
        entry['discriminantOffset'] = node.discriminantOffset
      out['structs'][name] = entry
  for name, enum in ENUMS.items():
    out['enums'][name] = [e.name for e in enum.schema.node.enum.enumerants]
  json.dump(out, sys.stdout, indent=1, sort_keys=True)
  print()


if __name__ == '__main__':
  main()
