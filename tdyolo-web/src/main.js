// TDYolo_v2 — browser entry point.
//
// 1. Init ONNX RT Web session(s) for the configured model(s).
// 2. Open a WebSocket connection back to the TouchDesigner-hosted
//    webserverDAT on the URL-supplied port.
// 3. Announce ourselves with {loaded: true} and (in webcam mode) the
//    enumerated device list.
// 4. Pick frame ingestion mode:
//      - binary mode (TD pushes frames)  → handleBinaryMessage on incoming
//      - webcam mode (browser captures)  → start RAF loop
//
// License: MIT.

import { WS_PORT, USE_BINARY } from "./config.js";
import { initSessions } from "./inference/onnx.js";
import {
    handleBinaryMessage,
    setWebSocketSender as setBinarySender,
} from "./modes/binary.js";
import {
    startWebcam,
    listWebcamDevices,
    setWebSocketSender as setWebcamSender,
} from "./modes/webcam.js";

const statusEl = document.getElementById("status");
const setStatus = (msg) => {
    if (statusEl) {
        statusEl.textContent = msg ?? "";
        statusEl.style.display = msg ? "" : "none";
    }
};

(async function main() {
    setStatus("Loading ONNX session…");
    const baseURL = new URL(".", location.href).href;
    try {
        await initSessions(baseURL);
    } catch (e) {
        setStatus(`Model load failed: ${e?.message || e}`);
        console.error("[TDYolo_v2] initSessions failed", e);
        return;
    }

    setStatus(`Connecting WS ${WS_PORT}…`);
    const ws = new WebSocket(`ws://localhost:${WS_PORT}`);
    ws.binaryType = "arraybuffer";

    const sender = (msg) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        if (msg instanceof ArrayBuffer || ArrayBuffer.isView(msg)) {
            ws.send(msg);
        } else {
            ws.send(msg);
        }
    };
    setBinarySender(sender);
    setWebcamSender(sender);

    ws.onopen = async () => {
        ws.send(JSON.stringify({ type: "loaded" }));
        // Announce available webcams so TD can populate its Webcam menu.
        const devices = await listWebcamDevices();
        ws.send(
            JSON.stringify({
                type: "webcamDevices",
                devices: devices.map((d) => d.label),
            }),
        );
        setStatus(USE_BINARY ? "Ready (binary)" : "Ready (webcam)");
    };

    ws.onerror = () => setStatus(`WebSocket error on port ${WS_PORT}`);
    ws.onclose = (ev) =>
        setStatus(`Disconnected (code ${ev.code}) on port ${WS_PORT}`);

    if (USE_BINARY) {
        ws.onmessage = (ev) => {
            if (ev.data instanceof ArrayBuffer) {
                handleBinaryMessage(ev.data);
            }
            // (Text messages from TD in binary mode are currently ignored.)
        };
    } else {
        await startWebcam();
    }
})();
