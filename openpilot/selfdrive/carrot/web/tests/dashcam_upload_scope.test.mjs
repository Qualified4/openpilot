import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import { dashcamUploadStats } from "../src/features/logs/upload_summary.js";

const source = await readFile(new URL("../src/features/logs/dashcam.js", import.meta.url), "utf8");
const uploadFunction = source.slice(source.indexOf("async function uploadDashcamSegments("), source.indexOf("async function uploadRecentDashcamSegments("));

async function runUpload(scope, confirm = true, options = {}) {
  const requests = [];
  const summaries = [];
  const context = vm.createContext({
    dashcamUploadActiveJobId: null,
    getRememberedDashcamUploadJob: () => null,
    getUIText: (_key, fallback) => fallback,
    showAppToast: (message) => { throw new Error(message); },
    ensureRoadViewerConnection: async () => true,
    chooseRoadViewerVideos: async () => ({ include_front: true, include_wide: false }),
    openAppDialog: async (dialog) => {
      assert.notEqual(options.destination, "road_viewer");
      assert.equal(dialog.choices[0].value, "default");
      assert.equal(dialog.choices[0].current, true);
      return scope;
    },
    appConfirm: async () => confirm,
    dashcamUploadConfirmHtml: (stats) => { summaries.push(stats); return "summary"; },
    dashcamUploadStats,
    postJson: async (url, body) => {
      requests.push({ url, ...body });
      if (url.endsWith("/summary")) return { summaries: [{
        segment: "route--0",
        files: [
          { name: "qcamera.ts", size: 10 }, { name: "rlog.zst", size: 20 },
          ...(body.includeAllFiles ? [{ name: "ecamera.hevc", size: 100 }, { name: "fcamera.hevc", size: 200 }] : body.include_front ? [{ name: "fcamera.hevc", size: 200 }] : []),
        ],
      }] };
      return { job_id: "job" };
    },
    rememberDashcamUploadJob() {},
    clearRememberedDashcamUploadJob() {},
    pollDashcamUploadJob: async () => ({ ok: true, uploaded: 1, total: 1, results: [] }),
    isDashcamUploadCanceledError: () => false,
  });
  vm.runInContext(uploadFunction, context);
  await context.uploadDashcamSegments(["route--0"], {
    showProgress: false, showResult: false, showSuccessToast: false, ...options,
  });
  return { requests, summaries };
}

for (const scope of ["default", "all"]) {
  test(`upload ${scope} uses the same scope for confirmation and transfer`, async () => {
    const { requests, summaries } = await runUpload(scope);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].includeAllFiles, scope === "all");
    assert.equal(requests[1].includeAllFiles, scope === "all");
    assert.equal(summaries[0].files, scope === "all" ? 4 : 2);
    assert.equal(summaries[0].bytes, scope === "all" ? 330 : 30);
  });
}

test("canceling file selection sends no request", async () => {
  assert.equal((await runUpload(null)).requests.length, 0);
});

test("canceling size confirmation never starts an upload", async () => {
  const { requests } = await runUpload("all", false);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/dashcam/upload/summary");
});


test("Road Viewer keeps its file scope and retry destination", async () => {
  const { requests, summaries } = await runUpload("all", true, { destination: "road_viewer", retryJobId: "prior-job" });
  assert.equal(requests.length, 2);
  assert.equal(requests[0].url, "/api/road-viewer/summary");
  assert.equal(requests[1].url, "/api/road-viewer/start");
  assert.equal(requests[1].retry_job_id, "prior-job");
  assert.ok(requests.every((request) => !("includeAllFiles" in request)));
  assert.equal(summaries[0].files, 3);
  assert.equal(summaries[0].bytes, 230);
  assert.ok(requests.every(request => request.include_front === true && request.include_wide === false));
});

test("Road Viewer leaves out segments the device reports as unsendable", async () => {
  const requests = [];
  const confirmed = [];
  const context = vm.createContext({
    dashcamUploadActiveJobId: null,
    getRememberedDashcamUploadJob: () => null,
    getUIText: (_key, fallback) => fallback,
    showAppToast: (message) => { throw new Error(message); },
    ensureRoadViewerConnection: async () => true,
    chooseRoadViewerVideos: async () => ({ include_front: false, include_wide: false }),
    appConfirm: async (_message, dialog) => { confirmed.push(dialog.messageHtml); return true; },
    dashcamUploadConfirmHtml: (stats) => `skipped:${(stats.skipped || []).map((item) => item.segment).join(",")}|summary:${stats.segments}`,
    dashcamUploadStats,
    postJson: async (url, body) => {
      requests.push({ url, ...body });
      if (url.endsWith("/summary")) return {
        summaries: [{ segment: "route--1", files: [{ name: "rlog.zst", size: 20 }] }],
        skipped: [{ segment: "route--0", error: "segment_incomplete" }, { segment: "route--2", error: "segment_incomplete" }],
      };
      return { job_id: "job", segments: ["route--1"], skipped: [] };
    },
    rememberDashcamUploadJob() {},
    clearRememberedDashcamUploadJob() {},
    pollDashcamUploadJob: async () => ({ ok: true, uploaded: 1, total: 1, results: [] }),
    isDashcamUploadCanceledError: () => false,
  });
  vm.runInContext(uploadFunction, context);
  await context.uploadDashcamSegments(["route--0", "route--1", "route--2"], {
    destination: "road_viewer", showProgress: false, showResult: false, showSuccessToast: false,
  });
  assert.deepEqual(Array.from(requests[0].segments), ["route--0", "route--1", "route--2"]);
  assert.equal(requests[1].url, "/api/road-viewer/start");
  assert.deepEqual(Array.from(requests[1].segments), ["route--1"]);
  assert.equal(confirmed[0], "skipped:route--0,route--2|summary:1");
});
