// TDYolo_v2 — ONNX Runtime Web session manager.
//
// Loads the YOLO detection ONNX model and exposes runDetect() / runSeg()
// helpers. runSeg() is a stub for v2.1 (SEG-HOOK).
//
// References used while writing this file:
//   - https://github.com/microsoft/onnxruntime-inference-examples
//     (specifically the `web/yolo` family of samples)
//   - https://onnxruntime.ai/docs/tutorials/web/
//   - https://docs.ultralytics.com/integrations/onnx/
//
// No code copied from torinmb/yolo-touchdesigner.
//
// License: MIT.

import * as ort from "onnxruntime-web";
import {
    USE_CPU,
    ENABLE_DET,
    ENABLE_SEG,
    MODEL_DETECT_KEY,
    MODEL_SEG_KEY,
    DET_SCORE_T,
    DET_TOPK,
    SEG_SCORE_T,
    SEG_TOPK,
} from "../config.js";
import {
    decodeYOLO,
    nmsPerClass,
    applyClassFilter,
    applyDetectionLimit,
    buildClassFilter,
    decodeYOLOSeg, // SEG-HOOK — currently returns null
} from "./postprocess.js";
import { COCO_NAMES } from "../config.js";

// ONNX RT Web environment.
// `wasmPaths = './'` makes the runtime look for its WASM artefacts next to
// the served HTML (which is what we want — the TouchDesigner-hosted webserverDAT
// flattens path lookups to basename, so all .wasm files appear at /).
ort.env.wasm.wasmPaths = "./";
// Disable proxy worker and threading: both require SharedArrayBuffer workers,
// which hang inside TouchDesigner's webrenderTOP CEF context regardless of
// COOP/COEP headers. Single-threaded main-thread ONNX works reliably.
ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;
ort.env.allowLocalModels = true;
ort.env.allowRemoteModels = false;
ort.env.useBrowserCache = false;

// ----------------------------------------------------------------------------
// Session state
// ----------------------------------------------------------------------------

export let detSession = null;

// SEG-HOOK: seg session reserved for v2.1.
export let segSession = null;

const NMS_IOU = 0.45; // standard YOLO default; not currently user-tunable

/**
 * Initialise ONNX RT Web sessions. Called once from main.js.
 *
 * @param {string} baseURL — origin + path where the ONNX models live (served
 *                           from the TD-hosted webserverDAT via VFS)
 */
export async function initSessions(baseURL) {
    console.log("[TDYolo_v2] initSessions start — USE_CPU =", USE_CPU,
                "| wasmPaths =", ort.env.wasm.wasmPaths);

    // WebGPU availability check (informational — fallback to WASM on failure).
    if (!USE_CPU && typeof navigator !== "undefined" && navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            console.log("[TDYolo_v2] WebGPU adapter:", adapter ? "OK" : "null (will use WASM)");
        } catch (gpuErr) {
            console.warn("[TDYolo_v2] WebGPU requestAdapter threw:", gpuErr);
        }
    } else if (!USE_CPU) {
        console.warn("[TDYolo_v2] navigator.gpu unavailable — will use WASM");
    }

    const providers = USE_CPU ? ["wasm"] : ["webgpu", "wasm"];

    if (ENABLE_DET) {
        const modelURL = `${baseURL}models/${MODEL_DETECT_KEY}.onnx`;
        console.log("[TDYolo_v2] loading model:", modelURL, "providers:", providers);
        detSession = await ort.InferenceSession.create(modelURL, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
        });
        console.log("[TDYolo_v2] detSession ready — inputs:", detSession.inputNames,
                    "outputs:", detSession.outputNames);
    }

    // SEG-HOOK: load seg model only when ENABLE_SEG=true. In MVP this branch
    // is always skipped.
    if (ENABLE_SEG) {
        try {
            const modelURL = `${baseURL}models/${MODEL_SEG_KEY}.onnx`;
            segSession = await ort.InferenceSession.create(modelURL, {
                executionProviders: providers,
                graphOptimizationLevel: "all",
            });
        } catch (e) {
            // TODO: SEG-HOOK — once seg models are bundled, drop the try/catch.
            console.warn("[TDYolo_v2] Segmentation model not loaded:", e);
            segSession = null;
        }
    }
}

/**
 * Run detection inference on the given input tensor.
 *
 * Applies class filter (TDYolo `Detectionlables` parameter), score threshold,
 * NMS, and detection limit. Returns a sorted array of detections in
 * [x_topleft, y_topleft, w, h] pixel-space boxes.
 *
 * @param {ort.Tensor} inputTensor
 * @param {string} classFilterStr — comma-separated class names ("" = all)
 * @param {number} confThreshold
 * @param {number} detLimit — 0 = unlimited
 */
export async function runDetect(
    inputTensor,
    classFilterStr,
    confThreshold,
    detLimit,
) {
    if (!detSession) return [];
    const outs = await detSession.run({
        [detSession.inputNames[0]]: inputTensor,
    });
    const head = outs[detSession.outputNames[0]];

    // Decode raw tensor → list of detections.
    const decoded = decodeYOLO(head, confThreshold ?? DET_SCORE_T, DET_TOPK);
    // NMS.
    const afterNMS = nmsPerClass(decoded, NMS_IOU, DET_TOPK);
    // Class filter (resolved from TDYolo's `Detectionlables`).
    const filterSet = buildClassFilter(classFilterStr, COCO_NAMES);
    const filtered = applyClassFilter(afterNMS, filterSet);
    // Detection limit (`Detectionlimit` parameter).
    const limited = applyDetectionLimit(filtered, detLimit ?? 0);
    return limited;
}

// ----------------------------------------------------------------------------
// SEG-HOOK: runSeg stub
// ----------------------------------------------------------------------------
/**
 * Stub segmentation runner — returns null in MVP. Implement in v2.1.
 *
 * @param {ort.Tensor} inputTensor
 */
export async function runSeg(inputTensor) {
    if (!segSession) return null;
    // TODO: SEG-HOOK
    //   1. const outs = await segSession.run({ [segSession.inputNames[0]]: inputTensor });
    //   2. Decode the two-tensor seg output via decodeYOLOSeg().
    //   3. Return { width, height, data: Float32Array } for binary WS send.
    return decodeYOLOSeg(null, SEG_SCORE_T, SEG_TOPK);
}
