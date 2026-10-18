"use strict";

const UPLOAD_ARTIFACT_KINDS = new Set(["qcamera", "rlog"]);

export function dashcamUploadArtifactKind(file) {
  const declared = String(file?.kind || "").trim().toLowerCase();
  if (UPLOAD_ARTIFACT_KINDS.has(declared)) return declared;
  const name = String(file?.name || "").trim().toLowerCase();
  if (name.startsWith("qcamera.")) return "qcamera";
  if (name === "rlog" || name.startsWith("rlog.")) return "rlog";
  return "";
}

export function dashcamUploadStats(items) {
  const list = Array.isArray(items) ? items : [];
  const segmentNames = [];
  const seenSegments = new Set();
  const stats = {
    segments: 0,
    segmentNames,
    files: 0,
    qcamera: 0,
    rlog: 0,
    bytes: 0,
  };

  list.forEach((item) => {
    const segment = String(item?.segment || "").trim();
    if (segment && !seenSegments.has(segment)) {
      seenSegments.add(segment);
      segmentNames.push(segment);
    }
    const files = Array.isArray(item?.files) ? item.files : [];
    const totalSize = Number(item?.totalSize)
      || files.reduce((sum, file) => sum + (Number(file?.size) || 0), 0);
    stats.segments += 1;
    stats.files += files.length;
    stats.bytes += totalSize;
    files.forEach((file) => {
      const kind = dashcamUploadArtifactKind(file);
      if (kind) stats[kind] += 1;
    });
  });

  return stats;
}

// Remaining upload time from the job's own byte counters and average speed.
// Rounded coarser as it grows (5 s under a minute, 10 s under an hour, then
// minutes) so the label does not jump on every progress poll.
export function dashcamUploadEta(snapshot) {
  const total = Number(snapshot?.bytes_total) || 0;
  const current = Number(snapshot?.bytes_current) || 0;
  const speed = Number(snapshot?.bytes_per_second) || 0;
  const remaining = total - current;
  if (!(speed > 0) || !(remaining > 0)) return null;
  const seconds = remaining / speed;
  if (!Number.isFinite(seconds) || seconds > 24 * 3600) return null;
  if (seconds < 60) return { hours: 0, minutes: 0, seconds: Math.max(5, Math.ceil(seconds / 5) * 5) };
  if (seconds < 3600) {
    let minutes = Math.floor(seconds / 60);
    let rest = Math.round((seconds - minutes * 60) / 10) * 10;
    if (rest === 60) { minutes += 1; rest = 0; }
    return minutes >= 60 ? { hours: 1, minutes: 0, seconds: 0 } : { hours: 0, minutes, seconds: rest };
  }
  let hours = Math.floor(seconds / 3600);
  let minutes = Math.round((seconds - hours * 3600) / 60);
  if (minutes === 60) { hours += 1; minutes = 0; }
  return { hours, minutes, seconds: 0 };
}
