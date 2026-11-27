#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MNN_INC="${MNN_INCLUDE_DIR:-/tmp/mnn-build/MNN/include}"

TARGET="${1:-all}"

build_host() {
    local lib_dir="${MNN_HOST_LIB_DIR:-/tmp/mnn-build/build-host}"
    local out_dir="$ROOT_DIR/target/mnn-host"
    mkdir -p "$out_dir"
    g++ -O2 -std=c++17 -fPIC -shared \
        -I"$MNN_INC" -I"$SCRIPT_DIR" \
        "$SCRIPT_DIR/mnn_capi.cpp" \
        -L"$lib_dir" -lMNN \
        -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$lib_dir" \
        -o "$out_dir/libmnn_capi.so"
    echo "Built host: $out_dir/libmnn_capi.so"
}

build_aarch64() {
    local lib_dir="${MNN_ARM64_LIB_DIR:-/tmp/mnn-build/arm64_dist_stripped}"
    local out_dir="$ROOT_DIR/prebuilt/linux-aarch64"
    mkdir -p "$out_dir"
    aarch64-linux-gnu-g++ -O2 -std=c++17 -fPIC -shared \
        -I"$MNN_INC" -I"$SCRIPT_DIR" \
        "$SCRIPT_DIR/mnn_capi.cpp" \
        -L"$lib_dir" -lMNN \
        -Wl,-rpath,'$ORIGIN' -Wl,-rpath,/data/ccnc_visiond \
        -o "$out_dir/libmnn_capi.so"
    aarch64-linux-gnu-strip --strip-unneeded "$out_dir/libmnn_capi.so"
    echo "Built aarch64: $out_dir/libmnn_capi.so"
}

case "$TARGET" in
    host)    build_host ;;
    aarch64) build_aarch64 ;;
    all)     build_host; build_aarch64 ;;
    *)       echo "Usage: $0 [host|aarch64|all]" >&2; exit 2 ;;
esac
