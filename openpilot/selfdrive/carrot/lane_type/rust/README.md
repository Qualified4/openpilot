# stripnet (Rust) — CCNC lane marking recogniser

The processing path of `../marking_daemon.py` (process `ccnc_lane_type`) in Rust. The device runs it:
process_config starts `stripnet_launch.sh`, which runs `prebuilt/linux-$(uname -m)/stripnet_daemon` and falls back to
`marking_daemon.py` when that binary is missing (PC) or the Rust process fails (non-zero exit within 60 s of its start,
or the 3rd crash since boot; marker `/dev/shm/ccnc_lane_type_rust.failed`, cleared by a reboot).

Device binary (committed, no build on the device; rebuild after any change under `src/`, `ipc/`, `native/`):

```sh
# ION: a dir with linux/{ion.h,msm_ion.h} copied from the device's /usr/include (zig's libc has no ion headers)
CXXFLAGS_aarch64_unknown_linux_gnu=-I$ION STRIPNET_NATIVE_DIR=<tools/build_native.sh output> STRIPNET_VISIONBUF=ion \
  cargo zigbuild -j2 --release --target aarch64-unknown-linux-gnu.2.31 --features ipc,kleidicv --bin stripnet_daemon \
  --config profile.release.strip=true
cp target/aarch64-unknown-linux-gnu/release/stripnet_daemon prebuilt/linux-aarch64/
```

`STRIPNET_VISIONBUF=ion|plain` picks msgq's VisionBuf implementation for a cross build (default: whether the build
machine has `/dev/ion`, as `msgq_repo/SConscript`).

| module | ports |
|---|---|
| `strip` | `strip_extractor.Extractor` / `profile_scanner.strip_geometry`: projection, numpy arange/interp, OpenCV INTER_LINEAR remap (8-bit fixed point), 4:1 box mean, row resize (INTER_AREA / INTER_LINEAR), right-strip mirror — pure Rust, bit-identical to cv2 4.13 on M8 |
| `model` | `strip_model.StripModel` (5-type v4 manifest only) on tract, `decide4` |
| `recogniser` | `Recogniser` (StripNet, lane-change variant C2, gate G1): skip gates, crossing, lifecycle resets |
| `filter` | `strip_filter.StripFilter` product configuration (asym A8/4 + colour 8 m + latch) |
| `display` | `display.DisplayPolicy.confirm` |
| `daemon` | `main`'s per-modelV2 body: SubMaster state, FrameBuffer, odometer, stale / error / drop handling, `packet` |
| `capnp`, `messages` | schema-less reader of the subscribed cereal fields (positions checked against `fixtures/schema_offsets.json`) |
| `ipc` (feature `ipc`) | msgq / VisionIPC through `ipc/shim.cc` over `msgq_repo` — the only FFI in the default build |
| `native` (features `kleidicv`, `opencv`) | extraction-speed candidates: the KleidiCV-rule remap through `kleidicv_remap_f32_u8` or `cv::remap` (OpenCV + KleidiCV HAL), static, aarch64 only |

Binaries: `stripnet_replay` (stdin replay for `tools/lane_type/stripnet_rust_parity.py`), `stripnet_daemon` (feature `ipc`).

```sh
cargo test -j2                     # + --features ipc for the msgq round trip
cargo build --release -j2 --bin stripnet_replay
python tools/lane_type/stripnet_rust_parity.py --rust <target/release/stripnet_replay> --out <dir> ROUTE_DIR:SEG ...
python openpilot/selfdrive/carrot/lane_type/rust/tools/schema_offsets.py > openpilot/selfdrive/carrot/lane_type/rust/fixtures/schema_offsets.json
cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.35   # cross build (zig from pip: ziglang, cargo-zigbuild)
```

## Speed switches (all bit-identical; environment, read once per process)

| switch | effect | default |
|---|---|---|
| `STRIPNET_SIMD=0\|1` | remap SIMD: aarch64 NEON (KleidiCV rule), x86_64 AVX2 (OpenCV rule, when the CPU has it) | on |
| `STRIPNET_MAP_CACHE=1` | reuse a side's rows / maps while line x/y/z, projection (rpy, K), frame height and strip shape are bit-identical; any changed bit is a miss | off |
| `--batch` (replay) | both sides in one tract N=2 run | off |
| `STRIPNET_TRACT_THREADS=n` | tract intra-op rayon pool of n threads (feature `tract-mt`) | 1 |
| `STRIPNET_BACKEND=rust\|kleidicv\|opencv` | who runs the KleidiCV-rule remap on aarch64 (features `kleidicv` / `opencv`) | kleidicv if compiled in, else rust |
| `STRIPNET_CV_THREADS=n` | cv::setNumThreads for the `opencv` backend | 1 |
| `STRIPNET_ENGINE=tract\|ort-cpu\|ort-xnnpack` | inference engine; `ort-*` need feature `ort` (onnxruntime loaded from `ORT_DYLIB_PATH`, `ort` 2.0.0-rc.15 / api-31 as the detection runner). N=1: two 1-thread sessions, one per side; `--batch`: one N=2 session | tract |
| `STRIPNET_ORT_THREADS=n` | intra-op threads of the ort N=2 session | 1 |
| `STRIPNET_ORT_KLEIDIAI=0` | `mlas.disable_kleidiai` (MLAS's KleidiAI kernels off) | on |
| `STRIPNET_REMAP=opencv\|kleidicv\|kleidicv-unfused` | which rounding rule (OpenCV generic x86 vs device KleidiCV) | per arch |
| `STRIPNET_SIDES=spawn\|worker\|seq` | second side on a new scoped thread per phase / one resident worker thread / sequential on the caller (`tail`) | worker |
| `STRIPNET_TRACT_STATE=0\|1` | one reused tract SimpleState per side instead of a new one per run | on |
| `STRIPNET_CPUS=list` | the daemon's CPU affinity (`0-3`, `0-2`, `1-2`, `4`, `0,2,5-6`); invalid → `0-2`; refused by the kernel (all offline) → unchanged. `stripnet_replay` applies it only when set | `0-2` |
| `STRIPNET_MALLOPT=0\|1` | glibc mallopt as `strip_extractor.tune_allocator` (64 MB / 256 MB / 16 MB) | off |

`stripnet_replay` prints the stage times per modelV2 (`stagesMs`: rows, maps, remap, reduce = 4:1 mean + row resize +
float32 output, summed over both sides) and `cacheHits`.

Device build to use: `--features kleidicv` (the fastest remap measured, 0.1 MB larger than the plain binary, no new .so).
Inference stays on tract: onnxruntime 1.32 (2026-11-21, device, parity 3,178/3,178) cut the inference tail with N=2
(p95 13.6 vs 18.1 ms) but not the whole step, for +27 MB .so and +25 MB RSS; the device CPU (SD845) has no
dotprod / i8mm / SME, so none of the wheel's KleidiAI kernels can run there (on/off identical); the wheel has no XNNPACK EP.
Map cache, N=2 batch and tract threads stay off: no hits on real drives, no gain, and slower / more CPU respectively.

### Inference tail (2026-11-21, device, parked, taskset 0-3, nice 10, 1,560 modelV2 x 3 rotated runs, n=4,680)

`stripnet_replay --diag` adds minflt, the caller's wait for the other side (waitMs) and per-side inference (inferSideMs);
`combo_bench.sh` engines s0 (594e1391: spawn, new state, no mallopt), v1 / v2 / v2b / v3 and combinations. ms, median / p95 / p99:

| run (load) | engine | inference | whole step | minflt / modelV2 |
|---|---|---|---|---|
| 1 (4.8-6.2) | py | 6.18 / 15.06 / 22.32 | 29.18 / 53.23 / 60.95 | 0 |
| 1 | s0 | 8.28 / 23.30 / 36.08 | 23.44 / 48.10 / 55.73 | 0 |
| 1 | v1 mallopt | 8.57 / 24.89 / 39.43 | 26.93 / 51.25 / 57.56 | 0 |
| 1 | v2 worker | 7.69 / 20.26 / 32.51 | 21.20 / 43.88 / 51.72 | 0 |
| 1 | v2b seq | 7.36 / 24.24 / 35.18 | 26.66 / 48.11 / 56.33 | 0.3 |
| 1 | v3 state | 8.37 / 18.61 / 26.56 | 22.69 / 42.81 / 51.14 | 0 |
| 1 | v123 | 7.77 / 18.28 / 26.77 | 18.98 / 37.22 / 46.74 | 0 |
| 1 | v12b3 | 7.55 / 18.23 / 28.07 | 24.17 / 44.08 / 53.32 | 0 |
| 2 (4.0-5.4) | py | 6.08 / 10.39 / 13.75 | 24.47 / 34.70 / 39.56 | 0 |
| 2 | s0 | 8.03 / 15.23 / 21.25 | 18.97 / 29.25 / 36.16 | 0 |
| 2 | v123 | 7.37 / 14.10 / 20.42 | 17.71 / 28.30 / 34.56 | 0 |
| 2 | v23 (default) | 6.57 / 12.55 / 17.66 | 16.94 / 26.57 / 33.37 | 0 |

No page faults per modelV2 with or without mallopt, so mallopt stays off. Reusing the tract state and the resident
thread each cut the tail; together the inference p95 / p99 fall ~18 % / ~17 % against s0, but stay above Python's: one
side's tract run (median 4.0 ms) is slower than Python's two sides together on cv2.dnn (6.1 ms), so Rust only wins the
median by running the sides in parallel, and each frame then waits for the slower of two threads on a loaded CPU.
The whole step's tail is below Python's in every run. Parity of every engine: M8 14,386 / 14,386, strips 11,789 / 11,789;
device 3,178 / 3,178, strips 2,395 / 2,395.

### KleidiCV / OpenCV backends (aarch64)

The device's cv2 (`opencv-python-headless` 4.13.0.92) is one `cv2.abi3.so` exporting only `PyInit_cv2`
(`-fvisibility=hidden`, stripped), so it cannot be linked or dlopen'd for `cv::remap`. `tools/build_native.sh` builds the
same sources (KleidiCV 0.7.0, OpenCV 4.13.0 core + imgproc with the KleidiCV HAL) as static libraries with zig; the
binary then needs no new `.so` on the device.

```sh
ZIG_PYTHON=<python with ziglang> tools/build_native.sh target/native-aarch64      # ~10 min with -j2 (cmake, ninja from pip)
STRIPNET_NATIVE_DIR=$PWD/target/native-aarch64 cargo zigbuild -j2 --release --target aarch64-unknown-linux-gnu.2.35 \
    --bin stripnet_replay --features kleidicv,opencv,tract-mt     # the combo-bench binary; the device build: kleidicv
```

### Combo bench (M8 and device)

`tools/combo_bench.sh m8|device [all|parity|time|summary]` runs the table base / simd / cache / batch / mt2 / mt4 / sc
(simd+cache) / sc_b (+batch), and on the device kc / cv (backends), in lockstep with the Python path: one parity run
(py, strips and logits recorded) and 3 timing repeats with the engine order rotated; `tools/extract_summary.py OUT`
gives median / p95 / p99 (first 20 modelV2 of each segment dropped), stage medians, RSS and CPU per modelV2.
Device: copy `stripnet_replay` (features kleidicv,opencv), `tools/lane_type/stripnet_rust_parity.py`,
`tools/extract_summary.py`, `tools/combo_bench.sh` and the two segments (`seg/<route>/<route>--<n>--{rlog.zst,fcamera.mp4}`)
to `/data/tmp/stripnet_bench`, run `bash combo_bench.sh device all` parked (it checks IsOnroad before and after every run;
taskset -c 0-3, nice 10; `CPUS=4-7` for the big cores), then `rm -rf /data/tmp/stripnet_bench`.
