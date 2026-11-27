#!/usr/bin/env bash
# manager runs from this directory. State is retained across onroad sessions until reboot.
BIN=${VISIOND_LAUNCH_BIN:-./prebuilt/linux-$(uname -m)/ccnc_visiond}
STATE=${VISIOND_LAUNCH_STATE:-/dev/shm/ccnc_visiond}
FAST_S=${VISIOND_LAUNCH_FAST_S:-60}
CRASHES=3

# The models and aarch64 runtimes (MNN OpenCL FP16 + CPU ORT fallback) are not in the repository.
# They are copied to the device once and checked against assets.sha256 here at every start.
ASSETS=${VISIOND_LAUNCH_ASSETS:-/data/ccnc_visiond}
MANIFEST=${VISIOND_LAUNCH_MANIFEST:-$PWD/assets.sha256}
ORT_LIB=libonnxruntime.so.1.32.0
MNN_CACHE=${CCNC_VISIOND_MNN_CACHE:-/data/cache/ccnc_visiond_mnn}

# Offroad OpenCL tuning cache pre-generation: does not touch crash counters or .failed.
if [[ ${1:-} == "--warmup-cache" ]]; then
  if [[ ! -x "$BIN" ]]; then
    echo "ccnc_visiond: no binary $BIN" >&2
    exit 1
  fi
  export CCNC_VISIOND_ASSETS="$ASSETS"
  export CCNC_VISIOND_MNN_CACHE="$MNN_CACHE"
  export LD_LIBRARY_PATH="$ASSETS:${LD_LIBRARY_PATH:-}"
  exec "$BIN" --warmup-cache
fi

[[ ! -e "$STATE.failed" ]] || exit 0
# A process that could not load its model, onnxruntime or maps stays alive and writes the reason here. This is not a crash.
rm -f "$STATE.halted"
if [[ ! -x "$BIN" ]]; then
  echo "no binary $BIN" > "$STATE.failed"
  exit 0
fi

# Prints why the assets cannot be used, nothing when every file has its listed sha256.
asset_problem() {
  local sum name got
  if [[ ! -r $MANIFEST ]]; then
    echo "no manifest $MANIFEST"
    return
  fi
  while read -r sum name; do
    [[ -n $sum ]] || continue
    if [[ ! -f $ASSETS/$name ]]; then
      echo "missing $ASSETS/$name"
      return
    fi
    got=$(sha256sum "$ASSETS/$name" 2>&1) || { echo "sha256sum $ASSETS/$name: $got"; return; }
    got=${got%% *}
    if [[ $got != "$sum" ]]; then
      echo "sha256 of $ASSETS/$name is $got, expected $sum"
      return
    fi
  done < "$MANIFEST"
}
# The process halts on its own with this reason (status message and .halted), as it does when loading fails.
problem=$(asset_problem)
if [[ -n $problem ]]; then
  export CCNC_VISIOND_HALT="$problem"
else
  unset CCNC_VISIOND_HALT
fi
export CCNC_VISIOND_ASSETS="$ASSETS" ORT_DYLIB_PATH="$ASSETS/$ORT_LIB"
export CCNC_VISIOND_MNN_CACHE="$MNN_CACHE"
export LD_LIBRARY_PATH="$ASSETS:${LD_LIBRARY_PATH:-}"
# Default scheduling: little cores 0-2 under SCHED_IDLE with 2 ORT threads (stays completely off big driving cores
# 4/5/6/7 and core 3 pandad/encoderd, and yields to loggerd/calibrationd/ui/ccnc_lane_type on 0-2).
# Override via env for gate comparisons (e.g. CCNC_VISIOND_CPUS=4 CCNC_VISIOND_THREADS=1).
export CCNC_VISIOND_POLICY=${CCNC_VISIOND_POLICY:-idle}
export CCNC_VISIOND_CPUS=${CCNC_VISIOND_CPUS:-0-2}
export CCNC_VISIOND_THREADS=${CCNC_VISIOND_THREADS:-2}
export CCNC_VISIOND_NICE=${CCNC_VISIOND_NICE:-19}
export CCNC_VISIOND_MIN_PERIOD_MS=${CCNC_VISIOND_MIN_PERIOD_MS:-0}

start=$SECONDS
stopping=0
child=
stop_child() {
  stopping=1
  if [[ -n $child ]]; then
    kill -TERM "$child" 2>/dev/null || true
  fi
}
# Bash background jobs can inherit SIGINT ignored. Forward manager's SIGINT as SIGTERM.
trap stop_child INT TERM
VISIOND_PARENT_PID=$$ VISIOND_STATE="$STATE" "$BIN" &
child=$!
if (( stopping )); then
  stop_child
fi
while :; do
  wait "$child"
  rc=$?
  kill -0 "$child" 2>/dev/null || break
done
# A trapped signal can interrupt wait just as the child exits. Bash keeps the child's status.
if (( stopping )); then
  wait "$child"
  exit $?
fi
(( rc != 0 )) || exit 0

ran=$((SECONDS - start))
count=0
if [[ -r "$STATE.crashes" ]]; then
  read -r count < "$STATE.crashes"
  [[ $count =~ ^[0-9]+$ ]] || count=0
fi
count=$((count + 1))
echo "$count" > "$STATE.crashes"
echo "ccnc_visiond: exited with $rc after ${ran}s, crash $count since boot" >&2
if (( ran < FAST_S || count >= CRASHES )); then
  echo "exit $rc after ${ran}s, crash $count" > "$STATE.failed"
fi
exit "$rc"
