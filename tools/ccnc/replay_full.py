"""Full live-entry CCNC replay with an event cache and optional warm-up segments.

Every recorded sendcan carrying 0x162 calls hyundaicanfd.create_ccnc_messages with the rlog's capnp readers, as
card does. One gzip JSON line per send: replayed 0x162 (det/dist), the car's own 0x162 (actual) and, with --trace,
the RearHandover memory and nearby radar points.

Cache: the events the replay reads (carState, modelV2, liveTracks, sendcan, radarState, carControl, can), sorted by
logMonoTime, are stored once per segment as plain capnp bytes in .analysis/cache/rlog_events/ and reused while the
rlog's size and mtime are unchanged. Decoding is ~10-15 % of a segment's replay time; the rest is the display code.

  python tools/ccnc/replay_full.py --out DIR [--warm N] [--module rear_variant.py] [--trace] LOG...

--warm N: the first N logs only build state (same route, consecutive segments) and are not written. Used by
replay_par.py to start a chunk mid-route.
"""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback
from types import ModuleType
from types import SimpleNamespace as N

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'.analysis/cache/rlog_events'
RX_ADDRESSES = (0x161, 0x162, 0x200, 0x1ea, 0x1e0)
WANTED = ('carState', 'modelV2', 'liveTracks', 'sendcan', 'radarState', 'carControl', 'can')
sys.path[:0] = [str(ROOT/'opendbc_repo'), str(ROOT)]
PARAMS = dict(CcncLaneColor=True, CcncModelLanes=2, CcncRadarVehicles=True, CcncVehiclePositionCorrection=True,
              CcncRadarMap=True, CcncLaneTypeDetection=0, LaneLineCheck=0)


def _params_stub():
  m = ModuleType('openpilot.common.params')

  class Params:
    def get_bool(self, k, *a, **kw): return bool(PARAMS.get(k, False))
    def get_int(self, k, *a, **kw): return int(PARAMS.get(k, 0))
    def get(self, k, *a, **kw): return PARAMS.get(k)
    def get_float(self, k, *a, **kw): return float(PARAMS.get(k, 0.))
    def put(self, *a, **kw): pass
    put_bool = put_nonblocking = put_bool_nonblocking = put

  class ParamKeyType: pass
  m.Params, m.ParamKeyType, m.ParamKeyFlag, m.UnknownKeyName = Params, ParamKeyType, ParamKeyType, KeyError
  sys.modules[m.__name__] = m


def _ascii_schemas():
  """pycapnp cannot open schema files under a non-ASCII path (Windows: 프로그래밍\\파이썬). Stage them."""
  if str(ROOT).isascii():
    return
  import capnp
  staging = Path(tempfile.mkdtemp(prefix='ccnc-schema-'))
  for base in ('openpilot/cereal', 'opendbc_repo/opendbc/car'):
    for source in (ROOT/base).rglob('*.capnp'):
      target = staging/source.relative_to(ROOT)
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(source, target)
  original = capnp.SchemaParser

  def mapped(value):
    try:
      return (staging/Path(value).resolve().relative_to(ROOT)).as_posix()
    except ValueError:
      return value

  class Parser:
    def __init__(self): self.parser = original()
    def load(self, filename, *args, **kwargs):
      kwargs['imports'] = [mapped(v) for v in kwargs.get('imports', [])]
      return self.parser.load(mapped(filename), *args, **kwargs)
    def __getattr__(self, name): return getattr(self.parser, name)
  capnp.SchemaParser = Parser


def segkey(path):
  m = re.search(r'([0-9a-f]+--[0-9a-f]+)--(\d+)--rlog', str(path))
  return m.group(1), int(m.group(2))


def events(log, path):
  """Sorted wanted events of one rlog, through the byte cache."""
  path = Path(path)
  route, seg = segkey(path)
  stat = path.stat()
  stamp = f'{stat.st_size}:{int(stat.st_mtime)}:{",".join(WANTED)}:can{RX_ADDRESSES}'
  data, meta = CACHE/f'{route}--{seg}.capnp', CACHE/f'{route}--{seg}.json'
  try:
    if json.loads(meta.read_text())['stamp'] == stamp:
      return list(log.Event.read_multiple_bytes(data.read_bytes(), traversal_limit_in_words=2**50))
  except (OSError, ValueError, KeyError):
    pass
  import zstandard
  with open(path, 'rb') as f:
    raw = zstandard.ZstdDecompressor().stream_reader(f).read()
  evs = sorted((e for e in log.Event.read_multiple_bytes(raw, traversal_limit_in_words=2**50) if e.which() in WANTED),
               key=lambda e: e.logMonoTime)
  blobs = []
  for e in evs:
    if e.which() != 'can':
      blobs.append(e.as_builder().to_bytes())
      continue
    frames = [p for p in e.can if p.src == 2 and p.address in RX_ADDRESSES]  # all the replay reads from can
    if not frames:
      continue
    small = log.Event.new_message(logMonoTime=e.logMonoTime, valid=e.valid)
    for dst, src in zip(small.init('can', len(frames)), frames, strict=True):
      dst.address, dst.src, dst.dat = src.address, src.src, bytes(src.dat)
    blobs.append(small.to_bytes())
  CACHE.mkdir(parents=True, exist_ok=True)
  tmp = data.with_suffix(f'.{os.getpid()}.tmp')
  tmp.write_bytes(b''.join(blobs))
  os.replace(tmp, data)
  meta.write_text(json.dumps({'stamp': stamp, 'source': str(path)}))
  return list(log.Event.read_multiple_bytes(data.read_bytes(), traversal_limit_in_words=2**50))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logs', nargs='+')
  ap.add_argument('--out', required=True)
  ap.add_argument('--warm', type=int, default=0)
  ap.add_argument('--map', type=int, default=1)
  ap.add_argument('--rear', type=int, default=1)
  ap.add_argument('--module', help='RearHandover variant file (replaces the class before configure)')
  ap.add_argument('--trace', action='store_true', help='RearHandover memory + radar points per send')
  ap.add_argument('--params', default='', help='JSON overrides for Params')
  ap.add_argument('--chain-start', help='first log of the consecutive chain: frame numbers count from it, as in a '
                  'sequential replay (frame%%N schedules depend on it)')
  a = ap.parse_args()
  PARAMS.update(CcncRadarMap=bool(a.map), **(json.loads(a.params) if a.params else {}))
  _params_stub()
  _ascii_schemas()

  from openpilot.cereal import log
  from opendbc.can import CANPacker
  from opendbc.can.parser import get_raw_value
  from opendbc.car.carlog import carlog
  from opendbc.car.hyundai import hyundaicanfd as main_mod
  from opendbc.car.hyundai import hyundaicanfd_ccnc_extension as ext
  from opendbc.car.hyundai import hyundaicanfd_ccnc_radar_map as rm
  from opendbc.car.hyundai.values import HyundaiFlags

  logged, first = collections.Counter(), {}

  def hook(kind):
    def f(msg, *x, **k):
      tb = traceback.format_exc() if kind == 'exception' else ''
      key = (kind, str(msg)[:120], tb.strip().splitlines()[-1][:200] if tb.strip() else '')
      logged[key] += 1
      first.setdefault(key, tb)
    return f
  for kind in ('exception', 'error'):
    setattr(carlog, kind, hook(kind))
  clock = [0.]
  ext.time = N(monotonic=lambda: clock[0])
  if a.module:
    import importlib.util
    spec = importlib.util.spec_from_file_location('rear_variant', a.module)
    variant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variant)
    ext.RearHandover = variant.RearHandover

  trace = []
  if a.trace:
    original = ext.RearHandover.update

    def traced(self, values, CS, md, frame, selected, filters, context, *, proven=(True, True)):
      keys = [p+s for p in ('LF', 'LR', 'RF', 'RR') for s in ('_DETECT', '_DETECT_DISTANCE')]
      before = {k: values.get(k) for k in keys}
      r = original(self, values, CS, md, frame, selected, filters, context, proven=proven)

      def mem(m):
        return None if m is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()
                                       if isinstance(v, (int, float, str, bool, list, tuple)) or v is None}
      pts = getattr(CS.live_tracks, 'points', ()) if CS.live_tracks is not None else None
      trace.append(dict(frame=frame, sel=[None if k is None else list(k) for k in selected], proven=list(proven),
                        before=before, after={k: values.get(k) for k in keys},
                        front=[mem(f) for f in self.fronts], rear=[mem(f) for f in self.rears],
                        ev=list(self.events), fresh=self.fresh,
                        pts=None if pts is None else [
                          [int(p.trackId), round(p.dRel, 2), round(p.yRel, 2), round(p.vRel, 2), int(p.motionState),
                           int(p.trackState)] for p in pts
                          if str(p.radarSource) == 'frontRadar' and 1.2 < abs(p.yRel) < 5.5 and p.dRel < 20]))
      return r
    ext.RearHandover.update = traced

  packer = CANPacker('hyundai_canfd_generated')
  dbc = packer.dbc
  rx_map = {0x161: ('ADRV_0x161', 'adrv_0x161'), 0x162: ('CCNC_0x162', 'ccnc_0x162'), 0x200: ('ADRV_0x200', 'adrv_0x200'),
            0x1ea: ('ADRV_0x1ea', 'adrv_0x1ea'), 0x1e0: ('LFAHDA_CLUSTER', 'lfahda_cluster')}

  def decode(name, dat):
    return {k: get_raw_value(dat, s)*s.factor+s.offset for k, s in dbc.name_to_msg[name].sigs.items()}
  K = ['FF_DETECT', 'LF_DETECT', 'RF_DETECT', 'LR_DETECT', 'RR_DETECT']
  D = ['FF_DISTANCE', 'LF_DETECT_DISTANCE', 'RF_DETECT_DISTANCE', 'LR_DETECT_DISTANCE', 'RR_DETECT_DISTANCE']
  os.makedirs(a.out, exist_ok=True)
  CP, CAN = N(flags=HyundaiFlags.CAMERA_SCC.value), N(ECAN=0, CAM=2)
  prev, total = None, collections.Counter()

  def chain_base(path):
    # The first send a sequential replay of this chain would number 0.
    seen = set()
    for e in events(log, path):
      kind = e.which()
      if kind == 'sendcan':
        if seen >= {'carState', 'modelV2', 'liveTracks', 'carControl'} and any(
            p.src == 0 and p.address == 0x162 for p in e.sendcan):
          return e.logMonoTime
      elif kind != 'can':
        seen.add(kind)
    return None
  start_base = chain_base(a.chain_start) if a.chain_start else None
  for n, path in enumerate(sorted(a.logs, key=segkey)):
    route, seg = segkey(path)
    if prev != (route, seg-1):
      base, latest, rx, lastframe, readers = None, {}, {}, None, {}
      if start_base is not None and n == 0:
        base = start_base
      ext.reset()
      rm.reset_live()
      for name in ('_display_options', '_radar_map', '_lane_line_check'):
        if hasattr(main_mod.create_ccnc_messages, name):
          delattr(main_mod.create_ccnc_messages, name)
    prev = (route, seg)
    write = n >= a.warm
    evs = events(log, path)
    zero = next(e.logMonoTime for e in evs if e.which() == 'sendcan')
    out = gzip.open(f'{a.out}/{route[:8]}-{seg}.jsonl.gz', 'wt') if write else None
    c = collections.Counter()
    before = sum(logged.values())
    for e in evs:
      now, kind = e.logMonoTime, e.which()
      clock[0] = now/1e9
      if kind == 'can':
        for p in e.can:
          if p.src == 2 and p.address in rx_map:
            rx[rx_map[p.address][1]] = decode(rx_map[p.address][0], bytes(p.dat))
        continue
      if kind != 'sendcan':
        latest[kind] = e
        readers[kind] = getattr(e, kind)  # SubMaster keeps one reader per message
        continue
      actual = next((p for p in e.sendcan if p.src == 0 and p.address == 0x162), None)
      if actual is None or not all(k in latest for k in ('carState', 'modelV2', 'liveTracks', 'carControl')):
        continue
      if base is None:
        base = now
      frame = 5*round((now-base)/5e7)
      if lastframe is not None and frame <= lastframe:
        frame = lastframe+5
      lastframe = frame

      def fresh(key, budget):
        ev = latest.get(key)
        return readers[key] if ev is not None and ev.valid and 0 <= now-ev.logMonoTime < budget else None
      cs, cc = readers['carState'], readers['carControl']
      CS = N(out=cs, modelV2=fresh('modelV2', 250000000), radarState=fresh('radarState', 10**9),
             live_tracks=fresh('liveTracks', 150000000), lfahda_cluster=rx.get('lfahda_cluster'),
             adrv_0x161=rx.get('adrv_0x161'), adrv_0x200=rx.get('adrv_0x200'), adrv_0x1ea=rx.get('adrv_0x1ea'),
             ccnc_0x162=dict(rx['ccnc_0x162']) if 'ccnc_0x162' in rx else None, cruise_buttons_msg=None,
             cruise_buttons_alt2=None, scc_control=None, MainMode_ACC=True, ACCMode=1, cruise_btns_msg_canfd='CRUISE_BUTTONS',
             paddle_button_prev=0, softHoldActive=0, trailer_connected=False, is_metric=True)
      msgs = main_mod.create_ccnc_messages(CP, packer, CAN, frame, cc, CS, cc.hudControl, cs.steeringAngleDeg, False, False,
                                           False, False, 0, 0, hud_lateral=None, rear_handover=bool(a.rear))
      tr = trace[-1] if trace else None
      trace.clear()
      if not write:
        continue
      c['sends'] += 1
      row = next((decode('CCNC_0x162', dat) for addr, dat, bus in msgs if addr == 0x162), {})
      act = decode('CCNC_0x162', bytes(actual.dat))
      fb = [row.get(k) for k in K] == [7, 4, 9, 6, 11]
      c['fallback'] += fb
      rh = ext.state.rear_handover
      out.write(json.dumps(dict(
        t=round((now-zero)/1e9, 3), v=round(cs.vEgo, 2), det=[row.get(k) for k in K], dist=[round(row.get(k, 0), 2) for k in D],
        actual=[act.get(k) for k in K]+[round(act.get(k, 0), 2) for k in D], fb=fb,
        bsd=[bool(cs.leftBlindspot), bool(cs.rightBlindspot)], blink=[bool(cs.leftBlinker), bool(cs.rightBlinker)],
        ev=[(x['side'], x['event']) for x in (rh.events if rh else [])], tr=tr))+'\n')
    if out:
      out.close()
      c['logged'] = sum(logged.values())-before
      total += c
      print(f'{route[:8]}/{seg}:', dict(c), 'map_failed', rm._live.failed,
            'rear_failed', bool(getattr(ext.state.rear_handover, 'failed', False)), flush=True)
  print('total', json.dumps(dict(total)), flush=True)
  for k, n in logged.most_common():
    print(n, k)
  for k, tb in list(first.items())[:3]:
    print('-----', k)
    print(tb[-2500:])


if __name__ == '__main__':
  main()
