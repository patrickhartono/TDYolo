// TDYolo_v2 — browser webcam ingestion mode (fallback when TD does not push
// binary frames).
//
// Capture: getUserMedia → off-screen 640×640 letterboxed canvas → ImageData
// → input tensor → pipeline.
//
// License: MIT.

import { INPUT_W, INPUT_H, FLIP_HORIZONTAL, WEBCAM_LABEL } from "../config.js";
import { fromImageData } from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";

let _sender = null;
let _videoEl = null;
let rafProcessing = false;

// Off-screen 640×640 buffer canvas.
const procCanvas = document.createElement("canvas");
procCanvas.width = INPUT_W;
procCanvas.height = INPUT_H;
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

export function setWebSocketSender(senderFn) {
    _sender = senderFn;
}

/** List available video input devices (for TD's Webcam menu). */
export async function listWebcamDevices() {
    try {
        const all = await navigator.mediaDevices.enumerateDevices();
        return all
            .filter((d) => d.kind === "videoinput")
            .map((d) => ({ label: d.label || "", deviceId: d.deviceId || "" }));
    } catch (e) {
        console.warn("[TDYolo_v2 webcam] enumerateDevices failed:", e);
        return [];
    }
}

function findDeviceIdByLabel(devices, wantedLabel) {
    if (!wantedLabel) return null;
    const exact = devices.find((d) => d.label === wantedLabel);
    if (exact) return exact.deviceId;
    const loose = devices.find(
        (d) =>
            d.label && d.label.toLowerCase().includes(wantedLabel.toLowerCase()),
    );
    return loose ? loose.deviceId : null;
}

/**
 * Draw the current video frame letterboxed into the off-screen 640×640
 * canvas and return its ImageData.
 */
function drawLetterboxed(video) {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scl = Math.min(INPUT_W / vw, INPUT_H / vh);
    const dw = Math.round(vw * scl);
    const dh = Math.round(vh * scl);
    const ox = ((INPUT_W - dw) / 2) | 0;
    const oy = ((INPUT_H - dh) / 2) | 0;

    procCtx.save();
    procCtx.fillStyle = "#000";
    procCtx.fillRect(0, 0, INPUT_W, INPUT_H);
    if (FLIP_HORIZONTAL) {
        procCtx.translate(INPUT_W, 0);
        procCtx.scale(-1, 1);
        procCtx.drawImage(video, ox, oy, dw, dh);
    } else {
        procCtx.drawImage(video, ox, oy, dw, dh);
    }
    procCtx.restore();
    return procCtx.getImageData(0, 0, INPUT_W, INPUT_H);
}

async function rafLoop() {
    if (!_videoEl || _videoEl.readyState < 2 || rafProcessing) {
        requestAnimationFrame(rafLoop);
        return;
    }
    rafProcessing = true;
    try {
        const img = drawLetterboxed(_videoEl);
        // Already flipped in canvas if needed, so flipH=false here.
        const input = fromImageData(img, false);
        await runInferencePipeline(input, _sender);
    } catch (e) {
        console.error("[TDYolo_v2 webcam]", e);
    } finally {
        rafProcessing = false;
        requestAnimationFrame(rafLoop);
    }
}

export async function startWebcam() {
    _videoEl = document.getElementById("video");
    if (!_videoEl) {
        _videoEl = document.createElement("video");
        _videoEl.id = "video";
        _videoEl.autoplay = true;
        _videoEl.muted = true;
        _videoEl.playsInline = true;
        document.body.appendChild(_videoEl);
    }

    const devices = await listWebcamDevices();
    const exactId = findDeviceIdByLabel(devices, WEBCAM_LABEL);

    const constraints = exactId
        ? { video: { deviceId: { exact: exactId } } }
        : { video: true };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        _videoEl.srcObject = stream;
        _videoEl.onloadedmetadata = () => requestAnimationFrame(rafLoop);
    } catch (e) {
        console.error("[TDYolo_v2 webcam] getUserMedia failed:", e);
    }

    return devices;
}
