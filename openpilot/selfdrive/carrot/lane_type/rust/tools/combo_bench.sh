#!/bin/bash
# StripNet optimisation combo table, M8 or the device, in lockstep (tools/lane_type/stripnet_rust_parity.py): every
# engine is its own process fed the same events / NV12 frames, one engine at a time per modelV2, so engines never
# overlap and video decoding is outside the timings. All Rust engines use --threads 2 (both sides in parallel).
#   base      SIMD off, no map cache, tract N=1 single-threaded           (STRIPNET_SIMD=0)
#   simd      remap SIMD (aarch64 NEON KleidiCV rule; x86 AVX2 OpenCV rule) (STRIPNET_SIMD=1)
#   cache     rows / maps reused while the geometry bits are identical     (STRIPNET_MAP_CACHE=1)
#   batch     both sides in one tract N=2 run                              (--batch)
#   mt2, mt4  tract intra-op threads 2 / 4                                 (STRIPNET_TRACT_THREADS)
#   sc, sc_mt2, sc_b   simd+cache, + mt2, + batch
#   (mt2 / mt4 need --features tract-mt; device binary: --features kleidicv,opencv,tract-mt)
#   device only, remap backend with SIMD on:
#   kc        kleidicv_remap_f32_u8 linked statically                      (STRIPNET_BACKEND=kleidicv)
#   cv        cv::remap, OpenCV 4.13.0 + KleidiCV HAL, static, 1 cv thread (STRIPNET_BACKEND=opencv)
#   py        the Python daemon path (parity reference; only in the parity run)
# Runs: `parity` once with py and recording (strip sha256 / logits), then `time` REPEATS times without py or recording,
# the engine order rotated each repeat. Summary: tools/extract_summary.py OUT.
# Usage: combo_bench.sh m8|device [all|parity|time|summary]
#   m8:     from the worktree root; OUT=.analysis/stripnet-combo (override with OUT=), LIMIT default 1200 per segment
#           (the 6 segments → 7,199 modelV2), waits while the 1-minute load average is above MAX_LOAD (default 1.5).
#   device: everything in /data/tmp/stripnet_bench (binary, harness, assets read from /data/openpilot, segments in
#           seg/); taskset -c 0-3 nice -n 10; stops (exit 3) whenever IsOnroad is not 0; LIMIT default 800 (2
#           segments → 1,600 modelV2); CPUS=4-7 for the big cores when they are online. Remove $B afterwards.
# REPEATS default 3. ENGINES="name ..." to choose a subset.
set -u
HOST=${1:?m8|device}; STEP=${2:-all}
if [ "$HOST" = device ]; then
  B=/data/tmp/stripnet_bench
  export PYTHONPATH=/data/openpilot/pydeps:/data/openpilot:/data/pythonpath
  PY=/usr/local/venv/bin/python3
  A=/data/openpilot/openpilot/selfdrive/carrot/lane_type/assets
  H=$B/stripnet_rust_parity.py; BIN=$B/stripnet_replay; SUM=$B/extract_summary.py
  PIN="taskset -c ${CPUS:-0-3} nice -n 10"
  SEGS="$B/seg/0000045e--375fb2e9ea:8 $B/seg/00000463--ed159f2996:2"
  OUT=${OUT:-$B/out}; LIMIT=${LIMIT:-800}
  DEFAULT="base simd cache batch mt2 mt4 sc sc_b kc cv"
  check() { [ "$(cat /data/params/d/IsOnroad 2>/dev/null)" = "0" ] || { echo "IsOnroad != 0: stop"; exit 3; }; }
else
  R=/mnt/nas-docs/openpilot-logs/roadviewer
  PY=${PY:-/home/orca/projects/openpilot/.venv/bin/python}
  export PYTHONPATH=$PWD
  A=openpilot/selfdrive/carrot/lane_type/assets
  H=tools/lane_type/stripnet_rust_parity.py
  BIN=openpilot/selfdrive/carrot/lane_type/rust/target/release/stripnet_replay
  SUM=openpilot/selfdrive/carrot/lane_type/rust/tools/extract_summary.py
  PIN=""
  SEGS="$R/lane-20261106-07/0000044c--b4fe596043:6 $R/lane-20261112/0000045e--375fb2e9ea:8 $R/lane-20261117/0000046b--f2f704fd72:8 $R/lane-20261112/00000463--ed159f2996:8 $R/lane-20261112/00000463--ed159f2996:2 $R/lane-20261117/00000465--d10f5da786:6"
  OUT=${OUT:-.analysis/stripnet-combo}; LIMIT=${LIMIT:-1200}
  DEFAULT="base simd cache batch mt2 mt4 sc sc_b"
  check() { while [ "$(awk -v m="${MAX_LOAD:-1.5}" '{print ($1 > m)}' /proc/loadavg)" = 1 ]; do echo "load $(cut -d' ' -f1 /proc/loadavg) > ${MAX_LOAD:-1.5}: waiting"; sleep 30; done; }
fi
REPEATS=${REPEATS:-3}
read -r -a ORDER <<< "${ENGINES:-$DEFAULT}"
engine() {  # NAME RECORD(0|1)
  local rec=--no-record env="" extra=""; [ "$2" = 1 ] && rec=
  case $1 in
    py) [ "$2" = 1 ] && rec=--record || rec=""; echo "py=$PIN $PY $H --serve-python $rec --threads 2"; return ;;
    base) env="STRIPNET_SIMD=0" ;;
    simd) env="STRIPNET_SIMD=1" ;;
    cache) env="STRIPNET_SIMD=0 STRIPNET_MAP_CACHE=1" ;;
    batch) env="STRIPNET_SIMD=0"; extra=--batch ;;
    mt2) env="STRIPNET_SIMD=0 STRIPNET_TRACT_THREADS=2" ;;
    mt4) env="STRIPNET_SIMD=0 STRIPNET_TRACT_THREADS=4" ;;
    sc) env="STRIPNET_SIMD=1 STRIPNET_MAP_CACHE=1" ;;
    sc_mt2) env="STRIPNET_SIMD=1 STRIPNET_MAP_CACHE=1 STRIPNET_TRACT_THREADS=2" ;;
    sc_b) env="STRIPNET_SIMD=1 STRIPNET_MAP_CACHE=1"; extra=--batch ;;
    rs) env="STRIPNET_SIMD=1 STRIPNET_BACKEND=rust" ;;
    kc) env="STRIPNET_SIMD=1 STRIPNET_BACKEND=kleidicv" ;;
    cv) env="STRIPNET_SIMD=1 STRIPNET_BACKEND=opencv" ;;
    *) echo "unknown engine $1" >&2; exit 2 ;;
  esac
  case $1 in kc|cv) ;; *) env="STRIPNET_BACKEND=rust $env" ;; esac  # the default backend is kleidicv when compiled in
  echo "$1=env $env $PIN $BIN $A --threads 2 $extra $rec"
}
run() {  # NAME RECORD ENGINES...
  local out=$1 rec=$2; shift 2
  local args=()
  for e in "$@"; do args+=(--engine "$(engine "$e" "$rec")"); done
  check
  echo "== $out $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg) engines: $*" | tee -a "$OUT/runs.log"
  $PY $H --out "$OUT/$out" --limit "$LIMIT" "${args[@]}" $SEGS > "$OUT/$out.log" 2>&1 || { echo "harness failed: $OUT/$out.log"; tail -5 "$OUT/$out.log"; exit 1; }
  echo "   done $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/runs.log"
  check
}
mkdir -p "$OUT"
timed() { local o=("${ORDER[@]}"); for i in $(seq 1 "$REPEATS"); do run "time$i" 0 "${o[@]}"; o=("${o[@]:1}" "${o[0]}"); done; }
case "$STEP" in
  parity) run parity 1 py "${ORDER[@]}" ;;
  time) timed ;;
  summary) $PY "$SUM" "$OUT" ;;
  all) run parity 1 py "${ORDER[@]}"; timed; $PY "$SUM" "$OUT" | tee "$OUT/summary.json" ;;
esac
