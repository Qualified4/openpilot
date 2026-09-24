// Road Viewer credentials and upload sessions are owned by the device backend.
export function roadViewerError(error) {
  const code = error?.payload?.error || error?.message || error;
  return getUIText(`rv_error_${code}`, getUIText("rv_error_upload_failed", "Road Viewer request failed. Check the connection and retry."));
}

export async function openRoadViewerSettings() {
  try {
    const status = await getJson("/api/road-viewer/status");
    const selected = await openAppDialog({
      mode: "choice", title: getUIText("rv_settings", "Road Viewer connection"), choiceLayout: "list",
      message: [getUIText(`rv_state_${status.state}`, status.state), status.url,
        status.error ? roadViewerError(status.error) : "",
        getUIText("rv_status_hint", "Connection status is updated when sending. Disconnect removes local credentials; revoke the device in Road Viewer to remove server access.")].filter(Boolean).join("\n\n"),
      choices: [
        { label: getUIText(status.url ? "rv_reconnect" : "rv_connect", status.url ? "Pair again / change URL" : "Connect"), value: "pair" },
        ...(status.state !== "disconnected" ? [{ label: getUIText("rv_disconnect", "Disconnect"), value: "disconnect", danger: true }] : []),
      ],
    });
    if (selected === "disconnect") {
      const yes = await appConfirm(getUIText("rv_disconnect_hint", "Stop Road Viewer uploads and remove saved credentials on this device?"));
      if (!yes) return false;
      await postJson("/api/road-viewer/disconnect", {});
      return false;
    }
    if (selected !== "pair") return status.state === "connected";
    const url = await appForm(getUIText("rv_url_hint", "Enter the HTTPS address, including the port. Do not include an API path."), {
      title: "Road Viewer URL", defaultValue: status.url, placeholder: "https://example.com:18443",
      onSubmit: (value) => {
        let parsed;
        try { parsed = new URL(value.trim()); } catch { throw new Error(roadViewerError("invalid_url")); }
        if (parsed.protocol !== "https:" || parsed.username || parsed.password || !["", "/"].includes(parsed.pathname) || parsed.search || parsed.hash) {
          throw new Error(roadViewerError("invalid_url"));
        }
      },
    });
    if (!url) return false;
    let connected = false;
    await appForm(getUIText("rv_pair_hint", "Generate a new pairing code in Road Viewer → External devices and enter it here."), {
      title: getUIText("rv_pair_code", "Pairing code"), inputType: "password", autocomplete: "off",
      confirmLabel: getUIText("rv_connect", "Connect"),
      onSubmit: async (code) => {
        try {
          await postJson("/api/road-viewer/pair", { url: url.trim(), code: code.trim() });
          connected = true;
        } catch (error) { throw new Error(roadViewerError(error)); }
      },
    });
    if (connected) showAppToast(getUIText("rv_state_connected", "Connected"));
    return connected;
  } catch (error) {
    showAppToast(roadViewerError(error), { tone: "error", duration: 5000 });
    return false;
  }
}

export async function chooseUploadDestination() {
  const target = await openAppDialog({
    mode: "choice", title: getUIText("rv_destination", "Upload destination"), choiceLayout: "list",
    choices: [
      { label: getUIText("rv_existing_upload", "Existing log upload"), value: "web" },
      { label: getUIText("rv_send", "Send to Road Viewer"), value: "road_viewer" },
    ],
  });
  if (target !== "road_viewer") return target;
  try {
    const status = await getJson("/api/road-viewer/status");
    if (["disconnected", "re_pair_required", "revoked_or_unknown"].includes(status.state)) {
      return await openRoadViewerSettings() ? target : null;
    }
    return target;
  } catch (error) {
    showAppToast(roadViewerError(error), { tone: "error" });
    return null;
  }
}
