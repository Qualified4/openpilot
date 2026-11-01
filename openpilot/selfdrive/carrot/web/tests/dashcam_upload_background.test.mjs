import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createDialogController } from "../src/ui/components/dialog/dialog.js";
import {
  isUploadJobMissingError,
  isUploadPollRetryableError,
  pollUploadJobSnapshots,
  uploadPollRetryDelay,
  waitUntilVisibleOrTimeout,
} from "../src/features/logs/upload_poll.js";

const dashcam = await readFile(new URL("../src/features/logs/dashcam.js", import.meta.url), "utf8");

function dialogHarness() {
  const timers = new Map();
  let serial = 0;
  const keyListeners = [];
  const nodes = new Map();
  const makeNode = () => ({
    hidden: true, value: "", style: { removeProperty() {}, setProperty() {} },
    classList: { add() {}, remove() {}, toggle() {} },
    setAttribute() {}, removeAttribute() {}, focus() {}, select() {},
    querySelector() { return null; },
  });
  for (const id of ["appDialog", "appDialogBackdrop", "appDialogTitle", "appDialogBody",
    "appDialogConfirm", "appDialogCancel"]) {
    nodes.set(id, makeNode());
  }
  const document = {
    getElementById: (id) => nodes.get(id),
    activeElement: makeNode(), body: makeNode(),
    addEventListener(type, callback) { if (type === "keydown") keyListeners.push(callback); },
  };
  const dialog = createDialogController({
    target: { addEventListener() {} },
    document,
    setTimeout: (callback) => { timers.set(++serial, callback); return serial; },
    requestAnimationFrame: (callback) => { timers.set(++serial, callback); return serial; },
    cancelAnimationFrame: (id) => timers.delete(id),
    createFocusTrap: () => ({ activate() {}, deactivate() {} }),
  });
  return {
    dialog,
    nodes,
    escape() { keyListeners.forEach((listener) => listener({ key: "Escape", preventDefault() {} })); },
    finishAnimation() { for (const [id, callback] of timers) { timers.delete(id); callback(); } },
  };
}

for (const action of ["backdrop", "escape"]) {
  test(`${action} hides a dismissToBackground progress dialog without canceling`, async () => {
    const h = dialogHarness();
    let canceled = 0;
    const progress = h.dialog.openAppProgressDialog({
      title: "Uploading",
      onCancel: () => { canceled += 1; },
      dismissToBackground: true,
    });
    if (action === "backdrop") h.nodes.get("appDialogBackdrop").onclick();
    else h.escape();
    h.finishAnimation();
    assert.deepEqual(await progress.completion, { dismissed: true });
    assert.equal(canceled, 0);
    assert.equal(progress.isOpen(), false);
    // Updates after hiding are ignored instead of throwing.
    progress.setMessage("still running");
    progress.setSummary("1 MB");
    progress.setProgressState({ value: 50 });
  });
}

test("the Cancel button of a dismissToBackground dialog still cancels", async () => {
  const h = dialogHarness();
  let canceled = 0;
  h.dialog.openAppProgressDialog({ onCancel: () => { canceled += 1; }, dismissToBackground: true });
  await h.nodes.get("appDialogCancel").onclick();
  assert.equal(canceled, 1);
});

test("progress dialogs without the option keep cancel-on-backdrop", async () => {
  const h = dialogHarness();
  let canceled = 0;
  h.dialog.openAppProgressDialog({ onCancel: () => { canceled += 1; } });
  await h.nodes.get("appDialogBackdrop").onclick();
  assert.equal(canceled, 1);
});

function networkError() {
  return new TypeError("Failed to fetch");
}

function httpError(status) {
  const error = new Error(`HTTP ${status}`);
  error.status = status;
  return error;
}

test("polling survives two network failures and returns the finished job", async () => {
  const answers = [
    () => { throw networkError(); },
    () => { throw httpError(502); },
    () => ({ status: "running", done: false, step_current: 1 }),
    () => ({ status: "done", done: true, result: { ok: true, results: [] } }),
  ];
  const waits = [];
  const retries = [];
  let recovered = 0;
  const seen = [];
  const snapshot = await pollUploadJobSnapshots(async () => answers.shift()(), {
    wait: async (ms) => { waits.push(["poll", ms]); },
    waitRetry: async (ms) => { waits.push(["retry", ms]); },
    intervalMs: 160,
    onRetry: (error, attempt, delay) => retries.push([attempt, delay]),
    onRecovered: () => { recovered += 1; },
    onSnapshot: (next) => seen.push(next.status),
  });
  assert.equal(snapshot.done, true);
  assert.deepEqual(snapshot.result, { ok: true, results: [] });
  assert.deepEqual(retries, [[1, 1000], [2, 2000]]);
  assert.equal(recovered, 1);
  assert.deepEqual(seen, ["running", "done"]);
  assert.deepEqual(waits, [["retry", 1000], ["retry", 2000], ["poll", 160]]);
  // Network failures are not a reason to forget the job.
  assert.equal(isUploadPollRetryableError(networkError()), true);
  assert.equal(isUploadPollRetryableError(httpError(502)), true);
});

test("an unknown job (404) ends polling at once and is forgotten", async () => {
  let calls = 0;
  let retries = 0;
  await assert.rejects(
    pollUploadJobSnapshots(async () => { calls += 1; throw httpError(404); }, {
      wait: async () => {},
      waitRetry: async () => {},
      onRetry: () => { retries += 1; },
    }),
    (error) => isUploadJobMissingError(error) && !isUploadPollRetryableError(error),
  );
  assert.equal(calls, 1);
  assert.equal(retries, 0);
});

test("retry delays back off 1, 2, 4, 8 then stay at 15 seconds", () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 20].map(uploadPollRetryDelay), [1000, 2000, 4000, 8000, 15000, 15000, 15000]);
});

test("a retry wait ends early when the page becomes visible", async () => {
  const listeners = new Set();
  const doc = {
    visibilityState: "hidden",
    addEventListener(type, callback) { if (type === "visibilitychange") listeners.add(callback); },
    removeEventListener(type, callback) { listeners.delete(callback); },
  };
  let cleared = 0;
  const waiting = waitUntilVisibleOrTimeout(15000, {
    document: doc,
    setTimeout: () => 1,
    clearTimeout: () => { cleared += 1; },
  });
  doc.visibilityState = "visible";
  listeners.forEach((listener) => listener());
  await waiting;
  assert.equal(cleared, 1);
  assert.equal(listeners.size, 0);
});

test("upload error paths keep the remembered job unless the job definitely ended", () => {
  assert.match(dashcam, /function shouldForgetDashcamUploadJob\(error\)/);
  assert.match(dashcam, /if \(jobId && shouldForgetDashcamUploadJob\(e\)\) \{\s*clearRememberedDashcamUploadJob\(jobId\);/);
  assert.match(dashcam, /if \(shouldForgetDashcamUploadJob\(e\)\) clearRememberedDashcamUploadJob\(jobId\);/);
  assert.doesNotMatch(dashcam, /progress\.close\(\);\s*clearRememberedDashcamUploadJob\(jobId\);/);
  assert.doesNotMatch(dashcam, /if \(jobId\) clearRememberedDashcamUploadJob\(jobId\);/);
  assert.match(dashcam, /dismissToBackground: true/);
});
