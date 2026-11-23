#!/usr/bin/env bash
# manager runs from this directory. State is retained across onroad sessions until reboot.
BIN=${VISIOND_LAUNCH_BIN:-./prebuilt/linux-$(uname -m)/ccnc_visiond}
STATE=${VISIOND_LAUNCH_STATE:-/dev/shm/ccnc_visiond}
FAST_S=${VISIOND_LAUNCH_FAST_S:-60}
CRASHES=3

[[ ! -e "$STATE.failed" ]] || exit 0
# A process that could not load its model, onnxruntime or maps stays alive and writes the reason here. This is not a crash.
rm -f "$STATE.halted"
if [[ ! -x "$BIN" ]]; then
  echo "no binary $BIN" > "$STATE.failed"
  exit 0
fi

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
