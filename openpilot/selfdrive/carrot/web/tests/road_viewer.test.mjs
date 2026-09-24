import assert from "node:assert/strict";
import test from "node:test";
import { chooseUploadDestination, openRoadViewerSettings, roadViewerError } from "../src/features/road_viewer/index.js";

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

test("existing destination does not contact Road Viewer", async (t) => {
  const set = setup(t);
  set("openAppDialog", async () => "web");
  set("getJson", () => assert.fail("Existing destination must stay independent"));
  assert.equal(await chooseUploadDestination(), "web");
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
