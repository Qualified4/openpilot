#!/usr/bin/env bash
RUST_LAUNCH_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
if [[ $(uname -m) == aarch64 ]]; then
  # The Stage 1 bundle contains representative binaries for host verification.
  # Admit a complete, validated AGNOS package in a later stage.
  echo 'ccnc-hda1-rust is a development branch. Device startup requires a validated AGNOS runtime package.' >&2
  exit 1
fi
export LD_LIBRARY_PATH="$RUST_LAUNCH_ROOT/rust/bin/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUST_LAUNCH_ROOT/rust/bin/openpilot-launch" --root "$RUST_LAUNCH_ROOT" "$@"
