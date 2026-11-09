import assert from "node:assert/strict";
import test from "node:test";

import {
  dashcamUploadArtifactKind,
  dashcamUploadEta,
  dashcamUploadSpeed,
  dashcamUploadStats,
} from "../src/features/logs/upload_summary.js";

test("upload artifacts are classified from server kind or original filename", () => {
  assert.equal(dashcamUploadArtifactKind({ kind: "qcamera", name: "anything" }), "qcamera");
  assert.equal(dashcamUploadArtifactKind({ name: "qcamera.ts" }), "qcamera");
  assert.equal(dashcamUploadArtifactKind({ name: "rlog.zst" }), "rlog");
  assert.equal(dashcamUploadArtifactKind({ name: "qlog.zst" }), "");
});

test("upload stats preserve original segment names and count selected artifacts", () => {
  const stats = dashcamUploadStats([
    {
      segment: "00001da7--8626316d57--0",
      files: [
        { kind: "qcamera", name: "qcamera.ts", size: 100 },
        { kind: "rlog", name: "rlog.zst", size: 40 },
      ],
      totalSize: 140,
    },
    {
      segment: "00001da7--8626316d57--1",
      files: [
        { name: "qcamera.mp4", size: 80 },
        { name: "rlog.bz2", size: 20 },
      ],
      totalSize: 100,
    },
  ]);

  assert.deepEqual(stats.segmentNames, [
    "00001da7--8626316d57--0",
    "00001da7--8626316d57--1",
  ]);
  assert.equal(stats.segments, 2);
  assert.equal(stats.qcamera, 2);
  assert.equal(stats.rlog, 2);
  assert.equal(stats.files, 4);
  assert.equal(stats.bytes, 240);
});

test("upload ETA comes from remaining bytes and the reported speed, rounded to stay stable", () => {
  const eta = (current, total, speed) => dashcamUploadEta({ bytes_current: current, bytes_total: total, bytes_per_second: speed });
  assert.equal(eta(0, 100, 0), null);                  // no speed yet
  assert.equal(eta(100, 100, 50), null);               // nothing left
  assert.equal(eta(0, 1e12, 1), null);                 // absurd estimate hidden
  assert.deepEqual(eta(0, 12, 1), { hours: 0, minutes: 0, seconds: 15 });
  assert.deepEqual(eta(0, 1, 1), { hours: 0, minutes: 0, seconds: 5 });
  assert.deepEqual(eta(0, 154, 1), { hours: 0, minutes: 2, seconds: 30 });
  assert.deepEqual(eta(0, 179, 1), { hours: 0, minutes: 3, seconds: 0 });   // 2:59 carries
  assert.deepEqual(eta(0, 3599, 1), { hours: 1, minutes: 0, seconds: 0 });
  assert.deepEqual(eta(0, 3 * 3600 + 25 * 60 + 40, 1), { hours: 3, minutes: 26, seconds: 0 });
  assert.deepEqual(eta(20e6, 260e6, 600e3), { hours: 0, minutes: 6, seconds: 40 });
});

test("upload ETA uses the whole-job wall-clock average and the speed label the windowed rate", () => {
  // Burst speed 2 MB/s, but pauses bring the real average to 0.5 MB/s: 100 MB left = 200 s, not 50 s.
  const snapshot = { bytes_current: 0, bytes_total: 100e6, bytes_per_second: 2e6, average_bytes_per_second: 0.5e6, window_bytes_per_second: 0.8e6 };
  assert.deepEqual(dashcamUploadEta(snapshot), { hours: 0, minutes: 3, seconds: 20 });
  assert.equal(dashcamUploadSpeed(snapshot), 0.8e6);
  // Before the server has enough history (null), the burst speed is the fallback.
  const early = { bytes_current: 0, bytes_total: 100e6, bytes_per_second: 2e6, average_bytes_per_second: null, window_bytes_per_second: null };
  assert.deepEqual(dashcamUploadEta(early), { hours: 0, minutes: 0, seconds: 50 });
  assert.equal(dashcamUploadSpeed(early), 2e6);
  assert.equal(dashcamUploadSpeed({ window_bytes_per_second: 0, bytes_per_second: 5 }), 0);  // a stalled window shows 0, not the burst
});
