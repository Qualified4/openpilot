# stripnet (Rust) — CCNC lane marking recogniser

The processing path of `../marking_daemon.py` (process `ccnc_lane_type`) in Rust. Stage 1: offline equivalence only;
the device still runs the Python process (no process_config entry yet).

| module | ports |
|---|---|
| `strip` | `strip_extractor.Extractor` / `profile_scanner.strip_geometry`: projection, numpy arange/interp, OpenCV INTER_LINEAR remap (8-bit fixed point), 4:1 box mean, row resize (INTER_AREA / INTER_LINEAR), right-strip mirror — pure Rust, bit-identical to cv2 4.13 on M8 |
| `model` | `strip_model.StripModel` (5-type v4 manifest only) on tract, `decide4` |
| `recogniser` | `Recogniser` (StripNet, lane-change variant C2, gate G1): skip gates, crossing, lifecycle resets |
| `filter` | `strip_filter.StripFilter` product configuration (asym A8/4 + colour 8 m + latch) |
| `display` | `display.DisplayPolicy.confirm` |
| `daemon` | `main`'s per-modelV2 body: SubMaster state, FrameBuffer, odometer, stale / error / drop handling, `packet` |
| `capnp`, `messages` | schema-less reader of the subscribed cereal fields (positions checked against `fixtures/schema_offsets.json`) |
| `ipc` (feature `ipc`) | msgq / VisionIPC through `ipc/shim.cc` over `msgq_repo` — the only FFI |

Binaries: `stripnet_replay` (stdin replay for `tools/lane_type/stripnet_rust_parity.py`), `stripnet_daemon` (feature `ipc`).

```sh
cargo test -j2                     # + --features ipc for the msgq round trip
cargo build --release -j2 --bin stripnet_replay
python tools/lane_type/stripnet_rust_parity.py --rust <target/release/stripnet_replay> --out <dir> ROUTE_DIR:SEG ...
python openpilot/selfdrive/carrot/lane_type/rust/tools/schema_offsets.py > openpilot/selfdrive/carrot/lane_type/rust/fixtures/schema_offsets.json
cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.35   # cross build (zig from pip: ziglang, cargo-zigbuild)
```
