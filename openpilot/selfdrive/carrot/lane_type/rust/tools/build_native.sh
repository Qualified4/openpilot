#!/bin/bash
# Static aarch64 builds for the extraction-speed candidates (features `kleidicv`, `opencv`):
#   KleidiCV 0.7.0 (libkleidicv.a) and OpenCV 4.13.0 core + imgproc with the KleidiCV 0.7.0 HAL — the versions in the
#   device's opencv-python-headless 4.13.0.92 (cv2 build information: "KleidiCV (ver 0.7.0)").
# The device's cv2.abi3.so exports only PyInit_cv2 (-fvisibility=hidden, stripped), so it cannot be linked or dlopen'd
# for cv::remap; this builds the same sources instead. Output: $NATIVE/{include,lib} for STRIPNET_NATIVE_DIR.
# Needs: zig (pip ziglang, the one cargo-zigbuild uses), cmake + ninja (pip), curl. ~25 min with -j2 on M8.
# Usage: tools/build_native.sh [NATIVE_DIR] (default target/native-aarch64); WORK=<dir> for sources and the build tree.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
NATIVE=$(realpath -m "${1:-$HERE/target/native-aarch64}")
WORK=$(realpath -m "${WORK:-$HERE/target/native-work}")
PY=${ZIG_PYTHON:-python3}   # a python with the ziglang package
JOBS=${JOBS:-2}
mkdir -p "$WORK/tc" "$NATIVE"
for k in cc c++; do
  f="$WORK/tc/z$k"
  cat > "$f" <<W
#!/bin/bash
# zig cc reads -march as a CPU name: map KleidiCV's -march=armv8-a[+ext] to -mcpu=generic[+ext].
a=()
for x in "\$@"; do case "\$x" in -march=armv8-a*) a+=("-mcpu=generic\${x#-march=armv8-a}");; *) a+=("\$x");; esac; done
exec $PY -m ziglang $k -target aarch64-linux-gnu.2.35 "\${a[@]}"
W
  chmod +x "$f"
done
for t in ar ranlib; do printf '#!/bin/sh\nexec %s -m ziglang %s "$@"\n' "$PY" "$t" > "$WORK/tc/z$t"; chmod +x "$WORK/tc/z$t"; done
cat > "$WORK/tc/aarch64.cmake" <<W
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER $WORK/tc/zcc)
set(CMAKE_CXX_COMPILER $WORK/tc/zc++)
set(CMAKE_AR $WORK/tc/zar)
set(CMAKE_RANLIB $WORK/tc/zranlib)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
W
cd "$WORK"
[ -d opencv-4.13.0 ] || curl -sL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz | tar xz
[ -d kleidicv-0.7.0 ] || curl -sL https://gitlab.arm.com/kleidi/kleidicv/-/archive/0.7.0/kleidicv-0.7.0.tar.gz | tar xz
KSRC=${KLEIDICV_SRC:-$WORK/kleidicv-0.7.0}
# -ffp-contract=fast: GCC's default (the wheel is built with GCC 10), so vmlaq_f32 contracts to fmla as on the device.
cmake -S opencv-4.13.0 -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$WORK/tc/aarch64.cmake" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc \
  -DWITH_KLEIDICV=ON -DKLEIDICV_SOURCE_PATH="$KSRC" -DKLEIDICV_ENABLE_SME=OFF \
  -DCMAKE_C_FLAGS=-ffp-contract=fast -DCMAKE_CXX_FLAGS=-ffp-contract=fast \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_OPENMP=OFF \
  -DWITH_CAROTENE=OFF -DBUILD_ZLIB=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DCV_TRACE=OFF > cmake.log
ninja -C build -j"$JOBS" opencv_imgproc > ninja.log
# Only two modules are built, so take the libraries and headers from the build tree (cmake --install wants them all).
mkdir -p "$NATIVE/lib" "$NATIVE/include/opencv4"
cp build/lib/libopencv_core.a build/lib/libopencv_imgproc.a "$NATIVE/lib/"
cp build/3rdparty/lib/libkleidicv.a build/3rdparty/lib/libkleidicv_thread.a build/3rdparty/lib/libkleidicv_hal.a build/3rdparty/lib/libzlib.a "$NATIVE/lib/"
cp -r opencv-4.13.0/modules/core/include/opencv2 "$NATIVE/include/opencv4/"
cp -r opencv-4.13.0/modules/imgproc/include/opencv2/* "$NATIVE/include/opencv4/opencv2/"
cp build/opencv2/* "$NATIVE/include/opencv4/opencv2/"
cp -r "$KSRC"/kleidicv/include/kleidicv "$NATIVE/include/"
cp build/hal/kleidicv/include/kleidicv/config.h "$NATIVE/include/kleidicv/"
ls -la "$NATIVE/lib"
