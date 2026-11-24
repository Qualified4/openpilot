# CCNC HDA1 Rust 외부 사전 빌드

`ccnc-hda1-rust`는 `ajouatom/openpilot:bin9208/carrot-rust-dev`의 `dbe642fa030dd488bec4b093244cb56fe93460fc`에서 시작합니다. 동작 비교 기준은 `Qualified4/openpilot:ccnc-hda1`의 `5250bdff2394a4aafe252a736f53af095e2692cd`입니다. 이 브랜치의 ARM64 부팅은 전체 런타임 패키지 검증까지 차단합니다.

## 호스트에서 만들기

Ubuntu 24.04 x86_64에 Rust 1.94.0과 `aarch64-unknown-linux-gnu` 표준 라이브러리, AArch64 GCC/G++ 13 및 Binutils, ARM64 Ubuntu sysroot, 호스트 Cap’n Proto compiler와 libclang 18을 준비합니다. FFmpeg의 링크에는 ARM64 sysroot의 `libx264.so.164`, `libz.so.1`, `libssl.so.3`, `libcrypto.so.3`가 필요합니다. 별도로 추출한 도구체인을 쓰면 `CC_aarch64_unknown_linux_gnu`와 `CXX_aarch64_unknown_linux_gnu`에 sysroot를 사용하는 래퍼를 지정하고, bindgen의 `LIBCLANG_PATH` 및 `BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_gnu`에 호스트 libclang과 ARM64 헤더 경로를 지정합니다.

clean checkout의 루트에서 실행합니다.

```bash
set -euo pipefail
work=$(mktemp -d /tmp/ccnc-arm64.XXXXXXXX)
mkdir "$work/sdk"
(
  cd device-inputs
  awk '$2 ~ /^sdk\.tar\.zst\./' MANIFEST | sha256sum -c
  cat sdk.tar.zst.* | zstd -dc | tar -xf - -C "$work/sdk"
)
JOBS=1 rust/tools/ccnc_hda1_prebuilt.sh build "$work/package" "$work/sdk" "$work/target"
```

산출물은 launcher, manager, process child, card, controlsd, radarcan, radard와 SDK의 FFmpeg 관련 .so를 포함합니다. `build.txt`에 컴파일에 사용한 실제 Git SHA와 toolchain이 기록됩니다. `elf.txt`에는 interpreter·NEEDED·심볼 버전이, `model-inputs.sha256`에는 원본 모델 입력 manifest의 식별자가 들어갑니다. 모델, 수치 자산, firmware와 장치별 라이브러리는 이후 전체 런타임 구축에 필요합니다.

## 후보 검증

빌드 출력에서 `SHA256SUMS`의 SHA-256을 기록하고, 소스 SHA와 함께 후보 외부의 신뢰할 수 있는 경로로 전달합니다. 검증은 파일 존재·실행 모드·패키지 목록·해시와 대상 버전을 확인합니다.

```bash
rust/tools/ccnc_hda1_prebuilt.sh verify "$candidate" "$source_sha" "$manifest_sha256"
```

ARM64 실행 검사는 QEMU user mode로 수행합니다. `agnos`는 검증된 19.8-carrot-bt1 이미지에서 로더와 라이브러리를 추출한 경로입니다. 이미지의 원본 해시와 추출 절차는 Rust 원본 이력의 `68188021^:rust/ci/runtime/package.py`에 있습니다. manager의 `--inspect`는 프로세스를 시작하기 전에 실제 native binding과 부족한 자산을 보여 줍니다.

```bash
qemu-aarch64-static -L "$agnos" \
  -E "LD_LIBRARY_PATH=$candidate/lib:$agnos/usr/lib/aarch64-linux-gnu:$agnos/lib" \
  -E LD_BIND_NOW=1 "$candidate/bin/openpilot-manager" \
  --root "$checkout" --bin-dir "$candidate/bin" --inspect
```

향후 전체 패키지에서는 `rust/crates/runtime-installer`의 offline descriptor·archive 및 source transaction을 재사용할 수 있습니다. 후보 checkout과 전체 패키지의 SHA가 일치하고 필수 자산과 AGNOS 호환성을 검증한 뒤 전환합니다. 이전 정상 checkout과 패키지를 유지해 복구에 사용합니다. [AGNOS 19.8-carrot-bt1 시스템 이미지](https://github.com/ajouatom/agnos-builder/releases/tag/agnos-19.8-carrot-bt1)

2026-11-24 첫 오프라인 패키지 소스는 `bb299aa2aafa0259da6e92666db423bd3c189ee9`입니다. AGNOS 이미지 로더로 7개 ELF 의존성을 해석했고 manager의 `--inspect`에서 카드·제어·레이더 네이티브 바인딩을 확인했습니다. 이 문서를 추가한 커밋에서 패키지를 다시 만들면 그 커밋의 SHA가 기록됩니다.
