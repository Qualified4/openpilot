import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { ensureRoadViewerConnection, openRoadViewerSettings, roadViewerError } from "../src/features/road_viewer/index.js";

function setup(t) {
  const previous = new Map();
  const set = (name, value) => {
    if (!previous.has(name)) previous.set(name, globalThis[name]);
    globalThis[name] = value;
  };
  t.after(() => { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } });
  set("getUIText", (_key, fallback) => fallback);
  set("showAppToast", () => {});
  return set;
}

test("connected Road Viewer needs no destination dialog", async (t) => {
  const set = setup(t);
  set("openAppDialog", () => assert.fail("Connected uploads must not open a destination dialog"));
  set("getJson", async (url) => {
    assert.equal(url, "/api/road-viewer/status");
    return { state: "connected" };
  });
  assert.equal(await ensureRoadViewerConnection(), true);
});

test("cancelled pairing and status errors stop Road Viewer uploads", async (t) => {
  const set = setup(t);
  set("getJson", async () => ({ state: "disconnected" }));
  set("openAppDialog", async () => null);
  assert.equal(await ensureRoadViewerConnection(), false);
  set("getJson", async () => { throw new Error("unreachable"); });
  assert.equal(await ensureRoadViewerConnection(), false);
});

test("direct upload goes straight to its confirmation and contacts only its destination", async () => {
  const source = readFileSync(new URL("../src/features/logs/dashcam.js", import.meta.url), "utf8");
  const fn = source.slice(source.indexOf("async function uploadDashcamSegments("), source.indexOf("async function uploadRecentDashcamSegments("));
  for (const destination of [undefined, "road_viewer"]) {
    const calls = [];
    const context = {
      dashcamUploadActiveJobId: null,
      getRememberedDashcamUploadJob: () => null,
      ensureRoadViewerConnection: async () => { calls.push("connection"); return true; },
      postJson: async (url) => { calls.push(url); return { summaries: [{}] }; },
      dashcamUploadStats: () => ({}),
      dashcamUploadConfirmHtml: () => "summary",
      getUIText: (_key, fallback) => fallback,
      appConfirm: async () => { calls.push("confirmation"); return false; },
      openAppDialog: () => assert.fail("No destination picker before confirmation"),
      showAppToast: () => assert.fail("Unexpected upload error"),
    };
    const upload = runInNewContext(`${fn}; uploadDashcamSegments`, context);
    await upload(["route--0"], destination ? { destination } : {});
    assert.deepEqual(calls, destination
      ? ["connection", "/api/road-viewer/summary", "confirmation"]
      : ["/api/dashcam/upload/summary", "confirmation"]);
  }
});

test("pairing uses only the Carrot backend and credentials never enter browser storage", async (t) => {
  const set = setup(t);
  const calls = [];
  set("localStorage", { setItem() { assert.fail("Connection data must not be saved in the browser"); } });
  set("getJson", async () => ({ state: "disconnected", url: "", error: "" }));
  set("openAppDialog", async () => "pair");
  set("postJson", async (...args) => { calls.push(args); return { ok: true, state: "connected" }; });
  let count = 0;
  set("appForm", async (_message, options) => {
    const value = count++ === 0 ? "https://example.test:18443/" : "PAIR-CODE";
    if (count === 2) assert.equal(options.inputType, "password");
    await options.onSubmit(value);
    return value;
  });
  assert.equal(await openRoadViewerSettings(), true);
  assert.deepEqual(calls, [["/api/road-viewer/pair", { url: "https://example.test:18443/", code: "PAIR-CODE" }]]);
});

test("offline disconnect still uses the local backend", async (t) => {
  const set = setup(t);
  set("getJson", async () => ({ state: "unreachable", url: "https://example.test:18443", error: "unreachable" }));
  set("openAppDialog", async () => "disconnect");
  set("appConfirm", async () => true);
  set("postJson", async (url, data) => { assert.equal(url, "/api/road-viewer/disconnect"); assert.deepEqual(data, {}); });
  assert.equal(await openRoadViewerSettings(), false);
});

test("untrusted error bodies do not appear in the UI", (t) => {
  setup(t);
  assert.equal(roadViewerError({ payload: { error: "secret from a remote proxy" } }),
    "Road Viewer request failed. Check the connection and retry.");
});


test("recent and individual log menus expose both destinations at the same level", async () => {
  const runtime = readFileSync(new URL("../src/features/logs/runtime.js", import.meta.url), "utf8");
  const calls = [];
  const context = {
    getUIText: (_key, fallback) => fallback,
    LOGS_MENU_SORT: "sort", LOGS_MENU_UPLOAD: "upload_recent", LOGS_RECENT_UPLOAD_LIMITS: [2, 5, 10],
    uploadRecentDashcamSegments: async (...args) => calls.push(args),
  };
  const code = runtime.slice(runtime.indexOf("function logsMenuChoices()"), runtime.indexOf("async function openLogsMenu()"));
  const menu = runInNewContext(`${code}; ({ choices: logsMenuChoices(), run: runLogsMenuAction })`, context);
  for (const count of [2, 5, 10]) {
    for (const prefix of ["upload_recent", "upload_recent_road_viewer"]) {
      const action = `${prefix}:${count}`;
      assert.ok(menu.choices.some((choice) => choice.value === action));
      await menu.run(action);
    }
  }
  assert.deepEqual(calls, [[2], [2, "road_viewer"], [5], [5, "road_viewer"], [10], [10, "road_viewer"]]);
  const source = readFileSync(new URL("../src/features/logs/dashcam.js", import.meta.url), "utf8");
  const start = source.indexOf("async function showDashcamSegmentMenu(");
  const fn = source.slice(start, source.indexOf("\n}\n", start) + 3);
  for (const action of ["upload", "upload_road_viewer"]) {
    const uploads = [];
    const show = runInNewContext(`${fn}; showDashcamSegmentMenu`, {
      getUIText: context.getUIText, dashcamState: { routes: [] },
      formatDashcamSegmentTimeLabel: () => "", formatDashcamSegmentFullTime: () => "",
      openAppDialog: async ({ choices }) => {
        assert.ok(choices.some((choice) => choice.value === "upload"));
        assert.ok(choices.some((choice) => choice.value === "upload_road_viewer"));
        return action;
      },
      uploadDashcamSegments: async (segments, options) => uploads.push({ segments: Array.from(segments), destination: options?.destination || "web" }),
    });
    await show("route", "route--1");
    assert.deepEqual(uploads, [{ segments: ["route--1"], destination: action === "upload" ? "web" : "road_viewer" }]);
  }
});
