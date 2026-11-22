#!/usr/bin/env bash
# ccnc_lane_type (process_config): run Python StripNet by default. CcncStripNetRust=1 opts in to Rust on startup.
# Usage: stripnet_launch.sh PYTHON   (cwd: this directory; PYTHON is the manager's interpreter)
#
# Failure = the Rust process ends with a non-zero status or a signal we did not forward, either within FAST_S of its
# start or for the CRASHES-th time since boot. The marker lives on tmpfs, so a reboot tries Rust again.
# A stalled publisher falls back immediately. Normal exits while disabled and manager stops are not failures.

PYTHON=${1:-python3}
BIN=${STRIPNET_LAUNCH_BIN:-./prebuilt/linux-$(uname -m)/stripnet_daemon}  # override: tests only
STATE=${STRIPNET_LAUNCH_STATE:-/dev/shm/ccnc_lane_type_rust}
FAST_S=60
CRASHES=3
STALL_S=${STRIPNET_STALL_S:-20}
PARAMS=${PARAMS_DIR:-/data/params/d}

param_enabled() {
  local enabled
  { IFS= read -r -n 1 enabled < "$PARAMS/$1"; } 2>/dev/null && [[ $enabled = 1 ]]
}

python_daemon() {
  echo "ccnc_lane_type: $1, running the Python StripNet daemon" >&2
  # from the checkout root (rust -> lane_type -> carrot -> selfdrive -> openpilot -> root), so `-m` resolves the package
  cd ../../../../.. || exit 1
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
  exec "$PYTHON" -m openpilot.selfdrive.carrot.lane_type.marking_daemon
}

param_enabled CcncStripNetRust || python_daemon "using the default backend"
[ -x "$BIN" ] || python_daemon "no Rust binary $BIN"
[ -e "$STATE.failed" ] && python_daemon "the Rust daemon failed earlier ($(cat "$STATE.failed" 2>/dev/null))"

start=$SECONDS
stopping=0
export STRIPNET_HEARTBEAT="$STATE.progress.$$"
stall_marker="$STRIPNET_HEARTBEAT.stalled"
"$BIN" &
child=$!
# manager stops processes with SIGINT (then SIGKILL; the daemon also dies with its parent through PR_SET_PDEATHSIG).
# Forward it as SIGTERM: a background job of a non-interactive shell starts with SIGINT ignored.
trap 'stopping=1; kill -TERM "$child" 2>/dev/null' INT TERM
(
  last_progress=$SECONDS
  previous=
  while kill -0 "$child" 2>/dev/null; do
    sleep 1
    if { IFS= read -r progress < "$STRIPNET_HEARTBEAT"; } 2>/dev/null && [[ $progress != "$previous" ]]; then
      last_progress=$SECONDS
      previous=$progress
    fi
    if (( SECONDS - last_progress >= STALL_S )) && param_enabled CcncExtendedDisplay && kill -0 "$child" 2>/dev/null; then
      : > "$stall_marker"
      kill -KILL "$child" 2>/dev/null
      break
    fi
  done
) &
watchdog=$!
while :; do
  wait "$child"
  rc=$?
  kill -0 "$child" 2>/dev/null || break  # wait returned because of a trapped signal: keep waiting for the child
done
kill "$watchdog" 2>/dev/null
wait "$watchdog" 2>/dev/null
stalled=0
[ ! -e "$stall_marker" ] || stalled=1
rm -f -- "$STRIPNET_HEARTBEAT" "$stall_marker"
[ "$stopping" = 0 ] || exit "$rc"
if [ "$rc" = 0 ]; then
  param_enabled CcncExtendedDisplay || exit 0
  rc=1
fi

ran=$((SECONDS - start))
count=$(( $(cat "$STATE.crashes" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$STATE.crashes"
echo "ccnc_lane_type: Rust StripNet daemon exited with $rc after ${ran}s (crash $count since boot)" >&2
if [ "$stalled" = 1 ]; then
  echo "no publication after ${STALL_S}s" > "$STATE.failed"
  python_daemon "the Rust daemon stopped publishing"
elif [ "$ran" -lt "$FAST_S" ] || [ "$count" -ge "$CRASHES" ]; then
  echo "exit $rc after ${ran}s, crash $count" > "$STATE.failed"
  python_daemon "the Rust daemon exited with $rc"
fi
exit "$rc"
