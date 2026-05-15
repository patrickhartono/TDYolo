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

    // Reconnect loop — survives the init_port race window on .toe reopen,
    // when webrender1 can hit ERR_CONNECTION_REFUSED before webserver1 is
    // listening. Without this, a single failed first attempt strands the
    // page forever.
    let currentWs = null;
    let backoffMs = 200;
    const BACKOFF_MAX = 3000;
    const BACKOFF_FACTOR = 1.5;
    let webcamStarted = false;

    const sender = (msg) => {
        const ws = currentWs;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(msg);
    };
    setBinarySender(sender);
    setWebcamSender(sender);

    function scheduleReconnect() {
        const delay = backoffMs;
        backoffMs = Math.min(Math.round(backoffMs * BACKOFF_FACTOR), BACKOFF_MAX);
        setStatus(`Reconnecting in ${delay}ms (port ${WS_PORT})…`);
        setTimeout(connect, delay);
    }

    function connect() {
        setStatus(`Connecting WS ${WS_PORT}…`);
        let ws;
        try {
            ws = new WebSocket(`ws://localhost:${WS_PORT}`);
        } catch (e) {
            console.error("[TDYolo_v2] WebSocket ctor threw", e);
            scheduleReconnect();
            return;
        }
        ws.binaryType = "arraybuffer";
        currentWs = ws;

        ws.onopen = async () => {
            backoffMs = 200; // reset for next disconnect
            ws.send(JSON.stringify({ type: "loaded" }));
            // Announce available webcams so TD can populate its Webcam menu.
            try {
                const devices = await listWebcamDevices();
                ws.send(
                    JSON.stringify({
                        type: "webcamDevices",
                        devices: devices.map((d) => d.label),
                    }),
                );
            } catch (e) {
                console.warn("[TDYolo_v2] listWebcamDevices failed", e);
            }
            setStatus(USE_BINARY ? "Ready (binary)" : "Ready (webcam)");

            if (!USE_BINARY && !webcamStarted) {
                webcamStarted = true;
                try {
                    await startWebcam();
                } catch (e) {
                    console.error("[TDYolo_v2] startWebcam failed", e);
                }
            }
        };

        ws.onerror = () => {
            // onclose will fire next; let it handle the reconnect.
            setStatus(`WebSocket error on port ${WS_PORT}`);
        };

        ws.onclose = (ev) => {
            setStatus(`Disconnected (code ${ev.code}) on port ${WS_PORT}`);
            if (currentWs === ws) currentWs = null;
            scheduleReconnect();
        };

        if (USE_BINARY) {
            ws.onmessage = (ev) => {
                if (ev.data instanceof ArrayBuffer) {
                    handleBinaryMessage(ev.data);
                }
                // (Text messages from TD in binary mode are currently ignored.)
            };
        }
    }

    connect();
})();
