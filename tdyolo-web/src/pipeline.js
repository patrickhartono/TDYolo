// TDYolo_v2 — per-frame pipeline orchestrator.
//
// Called by binary.js and webcam.js once per frame. Owns the read of TDYolo
// detection parameters (class filter, confidence, detection limit) from the
// URL each frame so live param changes propagate cleanly.
//
// License: MIT.

import {
    DET_CLASS_FILTER,
    DET_SCORE_T,
    DET_LIMIT,
    ENABLE_SEG,
} from "./config.js";
import {
    detSession,
    segSession,
    runDetect,
    runSeg, // SEG-HOOK — currently returns null
} from "./inference/onnx.js";
import { formatDetections, formatSegmentation } from "./utils/protocol.js";

/**
 * Run a single frame through the inference pipeline.
 *
 * @param {ort.Tensor} inputTensor
 * @param {(msg: string | ArrayBuffer | Uint8Array) => void} sender — WS send fn
 */
export async function runInferencePipeline(inputTensor, sender) {
    // 1. Detection (gated by session presence — empty if model failed to load).
    let detections = [];
    if (detSession) {
        detections = await runDetect(
            inputTensor,
            DET_CLASS_FILTER,
            DET_SCORE_T,
            DET_LIMIT,
        );
    }

    // 2. SEG-HOOK: only runs in v2.1 when ENABLE_SEG=true AND segSession loaded.
    let segResult = null;
    if (ENABLE_SEG && segSession) {
        segResult = await runSeg(inputTensor);
    }

    // 3. Send detection JSON over WS.
    if (sender) {
        const msg = formatDetections(detections);
        sender(JSON.stringify(msg));

        // SEG-HOOK: future binary seg send.
        if (segResult) {
            const binary = formatSegmentation(segResult);
            if (binary) sender(binary);
        }
    }
}
