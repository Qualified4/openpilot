# Road Viewer upload verification

Run from the openpilot repository root. Production uses the existing `aiohttp`
dependency. The integration checks additionally need `pytest`, `flask`,
`werkzeug` and the `openssl` executable in the development environment.

```sh
ROAD_VIEWER_SOURCE=/path/to/roadviewer-ha \
python openpilot/selfdrive/carrot/server/road_viewer/tests/run_desktop.py -q
```

This starts the **actual Road Viewer application from that checkout**, with
isolated temporary storage and HTTPS on loopback port 8098. The fixture supplies
a trusted temporary CA certificate; certificate and hostname verification stay
on. Port 8098 must be free. No deployed Home Assistant data or credentials are
used. `ROAD_VIEWER_TEST_URL` can specify the local test server's reachable HTTPS
address if the environment requires it; its certificate must match that address.
The desktop runner skips unrelated feature registration and substitutes hardware
serial lookup/replay report parsing only. It runs the real catalog, jobs, HTTP
routes, HMAC and upload protocol. In a fully built openpilot environment, the test
file can also be run with the usual pytest setup.

For an actual completed recording, also set:

```sh
ROAD_VIEWER_TEST_SEGMENT=/path/to/00000395--0d0eda17c5--7 \
ROAD_VIEWER_SOURCE=/path/to/roadviewer-ha \
python openpilot/selfdrive/carrot/server/road_viewer/tests/run_desktop.py -q
```

The source directory must contain `rlog.zst` and `qcamera.ts`. Tests copy those
files into disposable storage and compare both destination hashes. Source files
are never modified. Without this variable only this recording check is skipped;
without `ROAD_VIEWER_SOURCE`, real-server checks are explicitly skipped.

Existing upload regression checks:

```sh
python openpilot/selfdrive/carrot/server/road_viewer/tests/run_desktop.py \
  openpilot/selfdrive/carrot/server/tests/test_web_upload.py -q
cd openpilot/selfdrive/carrot/web
npm test
npm run build
```

For a real device, open the existing Carrot Web log menu → **Road Viewer
connection**, enter the configured HTTPS URL and a fresh pairing code generated
by Road Viewer. Select completed segments, choose **Upload selected → Road
Viewer**, and verify them in the Road Viewer list. Test network interruption and
**Resume upload**, cancel, device revocation, reconnect, and restart. Never paste
credentials or pairing codes into shared logs.

Credentials persist in `/data/carrot/state/road_viewer_secret.json` (0600), or the
corresponding `CARROT_DATA_DIR/state` directory. Job/checkpoint data is memory-only,
matching existing upload jobs (12 finished jobs retained). A restart clears retry
history. Completed received chunks are reused while the server session exists;
idle expiry is 15 minutes and absolute expiry is 2 hours. An expired session may
require retransmission. A batch is registered only after all files are received.

Local disconnect removes credentials and retry checkpoints, cancels active
uploads, and attempts to release the upload session. An unreachable server clears
its remaining temporary session by expiry. The server has no device-facing
revoke endpoint: remove server access from Road Viewer's device management UI.

Desktop tests do not validate C3/C4 timing under driving load. This adapter uses
one file reader/request at a time onroad and offroad, bounded 256 KiB blocks,
threaded disk reads, and no idle polling or driving-process scheduling changes.
