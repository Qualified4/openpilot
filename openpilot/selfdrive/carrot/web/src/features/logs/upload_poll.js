"use strict";

// Upload jobs run on the device in the background. A phone that turns its
// screen off (or drops Wi-Fi for a moment) only loses the poll requests, so
// connection failures are retried until the server says the job is gone; the
// server's own 30 minute inactivity expiry bounds the wait.

export const UPLOAD_POLL_RETRY_DELAYS_MS = Object.freeze([1000, 2000, 4000, 8000, 15000]);

export function uploadPollRetryDelay(attempt) {
  const delays = UPLOAD_POLL_RETRY_DELAYS_MS;
  return delays[Math.min(Math.max(0, Number(attempt) || 0), delays.length - 1)];
}

// The job endpoint answers an unknown id with HTTP 404 {ok:false,error:"job not found"}.
export function isUploadJobMissingError(error) {
  return Number(error?.status) === 404;
}

// Fetch failures, aborted/timed-out requests and server errors are transient.
// Any other HTTP answer (404 unknown job, 400 bad id) is final.
export function isUploadPollRetryableError(error) {
  const status = Number(error?.status);
  if (!status) return true;
  return status >= 500 || status === 408 || status === 429;
}

export async function fetchUploadSnapshotWithRetry(fetchSnapshot, options = {}) {
  const wait = options.waitRetry || options.wait;
  let failures = 0;
  while (true) {
    try {
      const snapshot = await fetchSnapshot();
      if (failures) options.onRecovered?.();
      return snapshot;
    } catch (error) {
      if (!isUploadPollRetryableError(error)) throw error;
      const delay = uploadPollRetryDelay(failures);
      failures += 1;
      options.onRetry?.(error, failures, delay);
      await wait(delay);
    }
  }
}

export function isFinalUploadSnapshot(snapshot) {
  return Boolean(snapshot?.done || snapshot?.status === "canceled" || snapshot?.result?.canceled);
}

// Polls until the job reports done/canceled. Returns the last snapshot.
export async function pollUploadJobSnapshots(fetchSnapshot, options = {}) {
  while (true) {
    const snapshot = await fetchUploadSnapshotWithRetry(fetchSnapshot, options);
    options.onSnapshot?.(snapshot);
    if (isFinalUploadSnapshot(snapshot)) return snapshot;
    await options.wait(options.intervalMs || 0);
  }
}

// A retry wait that ends early when the page becomes visible again, so a
// phone that wakes up polls immediately instead of finishing a 15 s backoff.
export function waitUntilVisibleOrTimeout(ms, environment = {}) {
  const doc = environment.document ?? globalThis.document;
  const setTimer = environment.setTimeout ?? globalThis.setTimeout;
  const clearTimer = environment.clearTimeout ?? globalThis.clearTimeout;
  return new Promise((resolve) => {
    let timer = null;
    const finish = () => {
      clearTimer(timer);
      doc?.removeEventListener?.("visibilitychange", onVisibility);
      resolve();
    };
    const onVisibility = () => {
      if (doc?.visibilityState === "visible") finish();
    };
    timer = setTimer(finish, ms);
    doc?.addEventListener?.("visibilitychange", onVisibility);
  });
}
