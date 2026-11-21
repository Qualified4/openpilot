#!/usr/bin/env bash
# ccnc_lane_type (process_config): run the Rust StripNet daemon, or the Python one (marking_daemon.py) when the Rust
# binary is missing for this machine or has failed since boot.
# Usage: stripnet_launch.sh PYTHON   (cwd: this directory; PYTHON is the manager's interpreter)
#
# Failure = the Rust process ends with a non-zero status or a signal we did not forward, either within FAST_S of its
# start or for the CRASHES-th time since boot. The marker lives on tmpfs, so a reboot tries Rust again.
# Normal exits (status 0: CcncExtendedDisplay turned off) are not failures.

PYTHON=${1:-python3}
BIN=${STRIPNET_LAUNCH_BIN:-./prebuilt/linux-$(uname -m)/stripnet_daemon}  # override: tests only
STATE=${STRIPNET_LAUNCH_STATE:-/dev/shm/ccnc_lane_type_rust}
FAST_S=60
CRASHES=3

python_daemon() {
  echo "ccnc_lane_type: $1, running the Python StripNet daemon" >&2
  # from the checkout root (rust -> lane_type -> carrot -> selfdrive -> openpilot -> root), so `-m` resolves the package
  cd ../../../../.. || exit 1
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
  exec "$PYTHON" -m openpilot.selfdrive.carrot.lane_type.marking_daemon
}

[ -x "$BIN" ] || python_daemon "no Rust binary $BIN"
[ -e "$STATE.failed" ] && python_daemon "the Rust daemon failed earlier ($(cat "$STATE.failed" 2>/dev/null))"

start=$SECONDS
stopping=0
"$BIN" &
child=$!
# manager stops processes with SIGINT (then SIGKILL; the daemon also dies with its parent through PR_SET_PDEATHSIG).
# Forward it as SIGTERM: a background job of a non-interactive shell starts with SIGINT ignored.
trap 'stopping=1; kill -TERM "$child" 2>/dev/null' INT TERM
while :; do
  wait "$child"
  rc=$?
  kill -0 "$child" 2>/dev/null || break  # wait returned because of a trapped signal: keep waiting for the child
done
[ "$stopping" = 1 ] || [ "$rc" = 0 ] && exit "$rc"

ran=$((SECONDS - start))
count=$(( $(cat "$STATE.crashes" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$STATE.crashes"
echo "ccnc_lane_type: Rust StripNet daemon exited with $rc after ${ran}s (crash $count since boot)" >&2
if [ "$ran" -lt "$FAST_S" ] || [ "$count" -ge "$CRASHES" ]; then
  echo "exit $rc after ${ran}s, crash $count" > "$STATE.failed"
  python_daemon "the Rust daemon exited with $rc"
fi
exit "$rc"
