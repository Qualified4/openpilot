"""Offline comparison of the independent map with both existing correction settings.

Manifest: {"entries": [{"name": route--segment, "path": rlog.zst, "sha256": ...}]}.
Consecutive available segments share state. Logs and replay output stay private.
Windows can inject the local cereal bootstrap before invoking main().
"""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace as N


def load_map(path, name='ccnc_map_replay'):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--baseline', type=Path, default=Path('opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py'))
  parser.add_argument('--candidate', type=Path, default=Path('opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_radar_map.py'))
  parser.add_argument('--maps', action='store_true')
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--summary-only', action='store_true')
  parser.add_argument('--stop-memory', action='store_true', help='Give the standalone map ego acceleration and blind-spot state')
  parser.add_argument('--explain', action='store_true', help='Record why legacy-selected points are not the standalone map display')
  parser.add_argument('--write-cache', type=Path, help='Save decoded inputs and the unchanged extension outputs for replay_radar_map_fast.py')
  parser.add_argument('--evidence-cache', action='store_true',
                      help='Comparison only: the earlier id+distance class cache without time, bus, validity or identity linking')
  parser.add_argument('--radar-evidence', action='store_true',
                      help="Give the candidate map the radar's own motion class read from raw 0x3A5.. track messages")
  parser.add_argument('--reference-map', type=Path, help='Also run the previous map source on identical inputs')
  selection = parser.add_mutually_exclusive_group()
  selection.add_argument('--hybrid', action='store_true', help='Refine existing correction-enabled candidates with map wall evidence')
  selection.add_argument('--integrated', action='store_true', help='Replace near structure processing before existing candidate selection')
  args = parser.parse_args()
  from openpilot.cereal import log
  from opendbc.can import CANPacker
  args.output.mkdir(parents=True, exist_ok=True)
  entries = json.loads(args.manifest.read_text(encoding='utf8'))['entries']
  for entry in entries:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['name']
  (args.output / 'identity.json').write_text(json.dumps(dict(
    baseline=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
    candidate=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
    manifest=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    reference_map=hashlib.sha256(args.reference_map.read_bytes()).hexdigest() if args.reference_map else None,
    hybrid=args.hybrid,
    integrated=args.integrated,
    profiles=['existing_correction_disabled', 'existing_correction_enabled', 'integrated' if args.integrated else 'map_candidate']
             + (['previous_hybrid' if args.integrated else 'previous_map'] if args.reference_map else [])), indent=2))
  modules = []
  clock = [0.]
  for module_index, correction in enumerate((False, True, True) if args.integrated else (False, True)):
    m = ModuleType(f'ccnc_baseline_{module_index}')
    source = args.baseline.read_text(encoding='utf8').replace(
      '  except:\n    values = CS.ccnc_0x162.copy()', '  except:\n    raise\n    values = CS.ccnc_0x162.copy()')
    exec(compile(source, str(args.baseline), 'exec'), m.__dict__)
    m.time = N(monotonic=lambda: clock[0])
    modules.append(m)
  candidate = load_map(args.candidate)
  packer = CANPacker('hyundai_canfd_generated')
  vehicle_fields = packer.dbc.name_to_msg['CCNC_0x162'].sigs
  lane_fields = packer.dbc.name_to_msg['ADRV_0x161'].sigs
  summaries = []
  latest, base = {}, None
  tracker = candidate.CcncRadarMap()
  reference = load_map(args.reference_map, 'ccnc_map_reference').CcncRadarMap() if args.reference_map else None
  profile_count = 4 if reference else 3
  for repeat in range(args.repeat):
    previous = None
    for entry in entries:
      name = entry['name']
      route, segment = name.rsplit('--', 1)
      segment = int(segment)
      if previous != (route, segment - 1):
        latest, base = {}, None
        tracker.reset()
        if reference:
          reference.reset()
        for correction, m in enumerate(modules):
          m.reset()
          m.configure(True, True, True, bool(correction), model_lane_mode=2)
          if correction == 2:
            m.state.radar_display_tracker = candidate.integrated_tracker(m.state.radar_display_tracker, tracker)
      previous = route, segment
      raw = subprocess.check_output(['zstd', '-d', '-c', entry['path']])
      events = sorted([e for e in log.Event.read_multiple_bytes(raw, traversal_limit_in_words=2**50)
                       if e.which() in ('carState', 'modelV2', 'liveTracks', 'sendcan', 'roadEncodeIdx', 'qRoadEncodeIdx')
                       or (args.radar_evidence and e.which() == 'can')],
                      key=lambda e: e.logMonoTime)
      evidence, raw_tracks, radar_bus, link_stats = {}, {}, None, Counter()
      cache = dict(name=name, lives=[], models=[], sends=[], live_index={}, model_index={}) if args.write_cache else None
      radar_evidence = {'evidence': evidence} if args.radar_evidence else {}
      indices = [e for e in events if e.which() == 'roadEncodeIdx'] or [e for e in events if e.which() == 'qRoadEncodeIdx']
      zero = getattr(indices[0], indices[0].which()).timestampEof if indices else events[0].logMonoTime
      comparison_indices = (0, 1, 3) if reference else (0, 1)
      counts = [Counter() for _ in comparison_indices]
      durations = [[] for _ in range(profile_count)]
      peaks = dict(objects=0, walls=0)
      output = gzip.open(args.output / f'{name}.jsonl.gz', 'wt', encoding='utf8') if not args.summary_only else None
      try:
        for e in events:
          now, kind = e.logMonoTime, e.which()
          clock[0] = now / 1e9
          if kind == 'can':
            # Track id = 32 + message index, as assigned by the radar interface. Byte 3
            # bits 4..3 (DBC VALID2): 1 stationary, 2 moving and latched, 0 other.
            for c in e.can:
              if 0x3a5 <= c.address < 0x3a5 + 32 and len(c.dat) == 24:
                if radar_bus is None:
                  radar_bus = c.src
                if c.src != radar_bus:
                  continue
                dat = bytes(c.dat)
                word = int.from_bytes(dat, 'little')
                lateral = (word >> 76) & 0xfff
                raw_tracks.setdefault(32 + c.address - 0x3a5, deque(maxlen=3)).append(
                  (now, (dat[3] >> 3) & 3, ((word >> 63) & 0x1fff) * .05, (lateral - 0x1000 if lateral & 0x800 else lateral) * .05, dat[5],
                   dat[3] & 3, (word >> 30) & 0x3ff))  # ..., DBC VALID (track state), PROB
                if args.evidence_cache:
                  evidence[32 + c.address - 0x3a5] = ((dat[3] >> 3) & 3, ((word >> 63) & 0x1fff) * .05)
            continue
          if kind == 'liveTracks' and args.radar_evidence and not args.evidence_cache:
            # Attach a class only to the liveTracks observation it was reported with: same bus,
            # a valid track (count > 10 as in the radar interface), sent at most 100ms before this
            # liveTracks message, and carrying exactly the distance and lateral value of the point.
            evidence.clear()
            for p in e.liveTracks.points:
              if str(p.radarSource) != 'frontRadar' or not p.measured:
                continue
              recent = raw_tracks.get(int(p.trackId), ())
              linked = next((m for m in reversed(recent) if 0 <= now - m[0] <= 100000000 and m[4] > 10
                             and abs(m[2] - p.dRel) <= .06 and abs(m[3] - p.yRel) <= .06), None)
              link_stats['points'] += 1
              link_stats['linked'] += linked is not None
              if linked is None and recent and abs(recent[-1][2] - p.dRel) <= 1.5:
                link_stats['unlinked, old cache would attach'] += 1
                link_stats['of those class 2'] += recent[-1][1] == 2
              if linked is not None:
                evidence[int(p.trackId)] = (linked[1], linked[2], linked[5], linked[6])  # class, distance, state, PROB
          if kind != 'sendcan':
            latest[kind] = e
            continue
          if base is None:
            base = now
          if not any(p.address == 0x162 and p.src == 0 for p in e.sendcan):
            continue
          for c in counts:
            c['packets'] += 1
          if not all(k in latest for k in ('carState', 'modelV2', 'liveTracks')):
            for c in counts:
              c['unready'] += 1
            continue
          cs = latest['carState'].carState
          def fresh(key, budget):
            event = latest[key]
            return getattr(event, key) if event.valid and 0 <= now - event.logMonoTime < budget else None
          md, live = fresh('modelV2', 250000000), fresh('liveTracks', 150000000)
          frame = round((now - base) / 1e7)
          values = [None] * profile_count
          for i in ((frame + j) % profile_count for j in range(profile_count)):
            start = time.perf_counter()
            if i < 2 or (i == 2 and args.integrated):
              m = modules[i]
              car = N(out=cs, live_tracks=live, ccnc_0x162=dict.fromkeys(vehicle_fields, 0))
              m.update_lanes(dict.fromkeys(lane_fields, 0), car, md, cs.vEgo * 3.6, cs.aEgo * 3.6,
                             int(md.meta.desire.raw) if md else 0, cs.latEnabled, True, True, frame)
              start = time.perf_counter()
              if i == 2:
                m.state.radar_display_tracker.map_reverse = str(cs.gearShifter) == 'reverse'
                m.state.radar_display_tracker._update_model(md)
                tracker.update(live, md, now / 1e9, cs.vEgo,
                               radar_time=latest['liveTracks'].logMonoTime / 1e9,
                               model_time=latest['modelV2'].logMonoTime / 1e9,
                               reverse=str(cs.gearShifter) == 'reverse', select=False,
                               lane_curves=m.state.radar_display_tracker._curve, **radar_evidence)
              v = m.update_vehicles(dict(car.ccnc_0x162), car, md, frame, cs.vEgo * 3.6, cs.aEgo * 3.6, True)
            else:
              current = tracker if i == 2 else reference
              extra = {'select': False} if (i == 2 and args.hybrid) or (i == 3 and args.integrated) else {}
              if i == 2:
                extra.update(radar_evidence)
              if i == 2 and args.stop_memory:
                extra.update(a_ego=cs.aEgo, blindspot=(cs.leftBlindspot, cs.rightBlindspot))
              current.update(live, md, now / 1e9, cs.vEgo,
                             radar_time=latest['liveTracks'].logMonoTime / 1e9,
                             model_time=latest['modelV2'].logMonoTime / 1e9,
                             reverse=str(cs.gearShifter) == 'reverse', **extra)
              v = current.display_values(dict.fromkeys(vehicle_fields, 0))
            durations[i].append((time.perf_counter() - start) * 1000.)
            packer.make_can_msg('CCNC_0x162', 0, v)
            values[i] = {k: v[k] for fields in tracker.SLOT_FIELDS for k in fields}
          if args.hybrid or (args.integrated and reference):
            start = time.perf_counter()
            old = modules[1].state.radar_display_tracker
            by_id = {p[0].trackId: p[0] for p in old._points}
            points = [by_id.get(key) for key in old.selected]
            index, current = (2, tracker) if args.hybrid else (3, reference)
            values[index] = current.refine_existing(dict(values[1]), points)
            durations[index][-1] += (time.perf_counter() - start) * 1000.
            packer.make_can_msg('CCNC_0x162', 0, values[index])
          if cache is not None:
            # Everything replay_radar_map_fast.py needs to rerun the map candidate without
            # decoding the log or rerunning the unchanged extension.
            le, me, old = latest['liveTracks'], latest['modelV2'], modules[1].state.radar_display_tracker
            if le.logMonoTime not in cache['live_index']:
              cache['live_index'][le.logMonoTime] = len(cache['lives'])
              cache['lives'].append(dict(time=le.logMonoTime / 1e9, evidence=dict(evidence), points=[
                (int(p.trackId), p.dRel, p.yRel, p.vRel, p.vLead, p.yvRel, bool(p.measured), str(p.radarSource)) for p in le.liveTracks.points]))
            if me.logMonoTime not in cache['model_index']:
              cache['model_index'][me.logMonoTime] = len(cache['models'])
              mv = me.modelV2
              cache['models'].append(dict(
                time=me.logMonoTime / 1e9, lines=[(list(line.x), list(line.y)) for line in mv.laneLines], probs=list(mv.laneLineProbs),
                edges=[(list(line.x), list(line.y)) for line in mv.roadEdges], yaw=list(mv.orientationRate.z)[:1],
                leads=[(lead.prob, list(lead.x)[:1], list(lead.y)[:1]) for lead in mv.leadsV3]))
            by_id = {p[0].trackId: p[0] for p in old._points}
            entries = [old.tracks.get(key) for key in old.selected]
            cache['sends'].append(dict(
              now=now / 1e9, t=(now - zero) / 1e9, live=cache['live_index'][le.logMonoTime], live_fresh=live is not None,
              model=cache['model_index'][me.logMonoTime], model_fresh=md is not None, v=cs.vEgo, a=cs.aEgo,
              blind=(bool(cs.leftBlindspot), bool(cs.rightBlindspot)), reverse=str(cs.gearShifter) == 'reverse',
              values=[values[0], values[1]],
              points=[None if (p := by_id.get(key)) is None else (p.trackId, p.dRel, p.yRel) for key in old.selected],
              sources=[None if entry is None else (int(entry[3].ccnc_source_id if old.temporal is not None else key), entry[3].dRel, entry[3].yRel)
                       for key, entry in zip(old.selected, entries)]))
          for i, c in zip(comparison_indices, counts):
            c['rows'] += 1
            for side, fields in zip(('FF', 'LF', 'RF'), tracker.SLOT_FIELDS):
              detect = fields[2]
              c[side + '_changed'] += any(abs(values[i][k] - values[2][k]) > 1e-6 for k in fields)
              c[side + '_added'] += not bool(values[i][detect]) and bool(values[2][detect])
              c[side + '_removed'] += bool(values[i][detect]) and not bool(values[2][detect])
          peaks['objects'] = max(peaks['objects'], len(tracker.objects))
          peaks['walls'] = max(peaks['walls'], len(tracker.walls))
          if output is not None:
            row = dict(t=(now - zero) / 1e9, now=now / 1e9, values=values, selected=tracker.selected)
            # Vehicle evidence of each candidate-selected object, so every display
            # change can be reported together with whether it is a vehicle.
            row['evidence'] = [None if (o := tracker.objects.get(k)) is None else
                               'radar' if getattr(o, 'radar_moving', False) else
                               'doppler' if getattr(o, 'doppler_confirmed', getattr(o, 'doppler', 0) >= getattr(tracker, 'DOPPLER_SAMPLES', 1 << 30)) else
                               'vision' if o.vision else 'moved' if o.motion_confirmed else
                               'waiting' if getattr(o, 'admitted', False) else 'none'
                               for k in tracker.selected]
            if args.explain:
              # Why each legacy-selected point is or is not the map candidate's display, and
              # which radar ids both sides selected (to recognise slot disagreements).
              old = modules[1].state.radar_display_tracker
              why, ids = [], []
              for slot, key in enumerate(old.selected):
                entry = old.tracks.get(key)
                if entry is None:
                  why.append(None)
                  ids.append(None)
                  continue
                p = entry[3]
                source = int(p.ccnc_source_id if old.temporal is not None else key)
                ids.append(source)
                why.append(tracker.explain(source, p.dRel, p.yRel, slot))
              row['why'], row['legacy_ids'] = why, ids
              row['candidate'] = [None if (o := tracker.objects.get(k)) is None else
                                  (o.source, *tracker.local[k], tracker.cos * o.vx + tracker.sin * o.vy) for k in tracker.selected]
            if args.stop_memory:
              row['remembered'] = list(tracker.remembered)
              row['ego'] = cs.vEgo
            if args.maps:
              row['map'] = tracker.snapshot()
              row['radar_time'] = latest['liveTracks'].logMonoTime / 1e9
              row['near_input'] = [(int(p.trackId), float(p.dRel), float(p.yRel)) for p in live.points
                                   if p.measured and str(p.radarSource) == 'frontRadar' and 0. <= p.dRel <= 15.
                                   and abs(p.yRel) <= 12.] if live is not None else []
              if reference:
                row['reference_map'] = reference.snapshot()
              if args.integrated:
                display = modules[2].state.radar_display_tracker
                row['integration'] = dict(selected=display.selected, rejected=list(display.map_blocked),
                                          far_points=len(display.stationary_map),
                                          baseline_selected=modules[1].state.radar_display_tracker.selected,
                                          observations=[dict(key=key, source=(entry[3].ccnc_source_id if display.temporal is not None else key),
                                                             xy=(entry[3].dRel, entry[3].yRel), fresh=(entry[3].ccnc_fresh if display.temporal is not None else True))
                                                        for key in display.selected if (entry := display.tracks.get(key)) is not None],
                                          stop_ids=[saved[0] for saved in display.stop_holds.values()],
                                          approach_ids=[saved[0] for saved in display.approach_holds.values()])
            output.write(json.dumps(row, separators=(',', ':')) + '\n')
      finally:
        if output is not None:
          output.close()
      if cache is not None:
        args.write_cache.mkdir(parents=True, exist_ok=True)
        del cache['live_index'], cache['model_index']
        cache.update(baseline=hashlib.sha256(args.baseline.read_bytes()).hexdigest(), log=entry['sha256'])
        with open(args.write_cache / f'{name}.pickle', 'wb') as f:
          pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
      summary = dict(name=name, repeat=repeat, counts=list(map(dict, counts)), peaks=peaks, evidence_link=dict(link_stats),
                     timing=[dict(mean=statistics.mean(d) if d else 0., maximum=max(d, default=0.),
                                  p99=sorted(d)[min(len(d) - 1, int(len(d) * .99))] if d else 0.) for d in durations])
      summaries.append(summary)
      (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
      print(name, summary['counts'], peaks, flush=True)
      del events, raw


if __name__ == '__main__':
  main()
