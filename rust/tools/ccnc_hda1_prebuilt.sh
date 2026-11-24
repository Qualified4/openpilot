#!/usr/bin/env bash
# Build and verify the Stage 1 offline ARM64 candidate. Always run on a build host.
set -euo pipefail
export LC_ALL=C

target=aarch64-unknown-linux-gnu
binaries=(openpilot-launch openpilot-manager openpilot-process-child openpilot-card
          openpilot-controlsd openpilot-radarcan openpilot-radard)
libraries=(libavcodec.so.60 libavformat.so.60 libavutil.so.58 libswresample.so.4 libvpx.so.9)
elf_files=()
for binary in "${binaries[@]}"; do elf_files+=("bin/$binary"); done
for library in "${libraries[@]}"; do elf_files+=("lib/$library"); done
files=(build.txt elf.txt model-inputs.sha256)
files+=("${elf_files[@]}")

die() { printf '%s\n' "$*" >&2; exit 1; }
usage() {
  die "Usage: $0 build OUTPUT SDK CARGO_TARGET_DIR | verify DIR SOURCE_SHA MANIFEST_SHA256"
}

case ${1:-} in
  build)
    [[ $# == 4 && $(uname -m) == x86_64 && ! -e /AGNOS ]] || usage
    output=$(realpath -m -- "$2")
    sdk=$(realpath -e -- "$3")
    export CARGO_TARGET_DIR
    CARGO_TARGET_DIR=$(realpath -m -- "$4")
    [[ -d $sdk/vendor && -d $sdk/ffmpeg/lib && ! -e $output && ! -L $output ]] ||
      die 'Need the extracted SDK and a new output directory'
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
    [[ -z $(git status --porcelain) ]] || die 'Build requires a clean checkout'
    source=$(git rev-parse --verify HEAD)
    export CC_aarch64_unknown_linux_gnu=${CC_aarch64_unknown_linux_gnu:-aarch64-linux-gnu-gcc-13}
    export CXX_aarch64_unknown_linux_gnu=${CXX_aarch64_unknown_linux_gnu:-aarch64-linux-gnu-g++-13}
    export AR_aarch64_unknown_linux_gnu=${AR_aarch64_unknown_linux_gnu:-aarch64-linux-gnu-ar}
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=$CC_aarch64_unknown_linux_gnu
    export FFMPEG_DIR=$sdk/ffmpeg
    export CARGO_ENCODED_RUSTFLAGS="-Clink-arg=-Wl,-rpath-link,$FFMPEG_DIR/lib"
    export CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_LTO=false
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_DEBUG=0
    args=()
    for binary in "${binaries[@]}"; do args+=(--bin "$binary"); done
    cargo +1.94.0 --config 'source.crates-io.replace-with="device-vendor"' \
      --config "source.device-vendor.directory=\"$sdk/vendor\"" \
      build --manifest-path rust/Cargo.toml --locked --offline --release --target "$target" \
      --jobs "${JOBS:-1}" --features openpilot-card/native \
      -p openpilot-runtime-launcher -p openpilot-manager -p openpilot-process-supervision \
      -p openpilot-card -p openpilot-controlsd -p openpilot-radarcan -p openpilot-radard "${args[@]}"
    [[ $(git rev-parse --verify HEAD) == "$source" && -z $(git status --porcelain) ]] ||
      die 'Checkout changed while building'
    mkdir -p -- "$(dirname -- "$output")"
    pending=$(mktemp -d -- "${output}.pending.XXXXXXXX")
    trap 'rm -rf -- "$pending"' EXIT
    mkdir -- "$pending/bin" "$pending/lib"
    {
      printf 'source_commit=%s\ndirty=false\ntarget=%s\nprofile=release (LTO disabled, 16 codegen units)\n' "$source" "$target"
      printf 'agnos_target=19.8-carrot-bt1\nfixture=representative, not bootable\n'
      rustc +1.94.0 -Vv
      cargo +1.94.0 -V
      "$CC_aarch64_unknown_linux_gnu" -dumpfullversion
      capnp --version
    } > "$pending/build.txt"
    sha256sum device-inputs/MANIFEST device-inputs/usbgpu-models/*/manifest.json > "$pending/model-inputs.sha256"
    for binary in "${binaries[@]}"; do
      install -m 0755 -- "$CARGO_TARGET_DIR/$target/release/$binary" "$pending/bin/$binary"
    done
    for library in "${libraries[@]}"; do
      install -m 0644 -- "$FFMPEG_DIR/lib/$library" "$pending/lib/$library"
    done
    for file in "${elf_files[@]}"; do
      readelf -hW "$pending/$file" | grep 'Machine:.*AArch64' >/dev/null ||
        die "Wrong architecture: $file"
      printf '\n[%s]\n' "$file" >> "$pending/elf.txt"
      readelf -lWdV "$pending/$file" >> "$pending/elf.txt"
    done
    (cd -- "$pending" && sha256sum -- "${files[@]}" > SHA256SUMS)
    mv -T --no-clobber -- "$pending" "$output"
    [[ ! -d $pending ]] || die 'Output already exists'
    sha256sum -- "$output/SHA256SUMS"
    ;;
  verify)
    [[ $# == 4 && $3 =~ ^[0-9a-f]{40}$ && $4 =~ ^[0-9a-f]{64}$ ]] || usage
    cd -- "$2"
    [[ -d bin && ! -L bin && -d lib && ! -L lib ]] || die 'Expected regular bin and lib directories'
    for file in SHA256SUMS "${files[@]}"; do
      [[ -f $file && ! -L $file ]] || die "Missing regular file: $file"
    done
    for binary in "${binaries[@]}"; do
      [[ -x bin/$binary ]] || die "Missing execute permission: $binary"
    done
    cmp -s <(printf '%s\0' bin lib SHA256SUMS "${files[@]}" | sort -z) \
      <(find . -mindepth 1 -printf '%P\0' | sort -z) || die 'Unexpected package contents'
    printf '%s  SHA256SUMS\n' "$4" | sha256sum --check --status || die 'Manifest SHA-256 mismatch'
    cmp -s <(printf '%s\n' "${files[@]}") <(cut -c 67- SHA256SUMS) || die 'Unexpected manifest paths'
    sha256sum --check --strict SHA256SUMS
    grep -Fxq "source_commit=$3" build.txt && grep -Fxq 'dirty=false' build.txt &&
      grep -Fxq "target=$target" build.txt && grep -Fxq 'agnos_target=19.8-carrot-bt1' build.txt ||
      die 'Source, target or AGNOS identity mismatch'
    ;;
  *) usage ;;
esac
