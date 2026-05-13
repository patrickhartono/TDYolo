// TDYolo_v2 — ONNX output decoding + NMS.
//
// Implements anchor-free YOLO11 detection decode. Layout assumption:
//   Output tensor shape [1, C, N] (channels first) or [1, N, C] (channels last),
//   auto-detected. C = 4 (box) + num_classes for detection-only models exported
//   from Ultralytics. yolo11n has 80 COCO classes so C = 84.
//
// References used while writing this file:
//   - https://docs.ultralytics.com/integrations/onnx/  (export schema)
//   - https://github.com/microsoft/onnxruntime-inference-examples
//     (specifically the YOLO web samples for sanity checks on the NMS algorithm)
//
// No code copied from torinmb/yolo-touchdesigner.
//
// License: MIT.

import { INPUT_W, INPUT_H } from "../config.js";
import { iou } from "../utils/math.js";

/**
 * Decode a YOLO detection output tensor into a sorted, optionally clamped
 * list of detections.
 *
 * @param {{data: Float32Array | number[], dims: number[]}} tensor
 * @param {number} scoreThreshold — minimum confidence
 * @param {number} topk — keep at most this many after sort-by-score
 * @returns {Array<{box: [number, number, number, number], label: number, score: number}>}
 */
export function decodeYOLO(tensor, scoreThreshold, topk) {
    const d = tensor.data;
    const sh = tensor.dims;
    if (sh.length !== 3) return [];

    // Auto-detect channels-first vs channels-last by comparing magnitude.
    const dim0 = sh[1];
    const dim1 = sh[2];
    let C, N, channelsFirst;
    if (dim0 < dim1) {
        // Typical Ultralytics export: [1, C=84, N=8400]
        C = dim0;
        N = dim1;
        channelsFirst = true;
    } else {
        // Some exporters emit [1, N, C] instead.
        C = dim1;
        N = dim0;
        channelsFirst = false;
    }

    const getCell = channelsFirst
        ? (c, i) => d[c * N + i]
        : (c, i) => d[i * C + c];

    const W = INPUT_W;
    const H = INPUT_H;
    const out = [];

    // 0..3 = cx, cy, w, h (in pixels)
    // 4..C-1 = per-class scores (sigmoid-applied at export time)
    for (let i = 0; i < N; i++) {
        // Find best class.
        let bestC = -1;
        let bestS = -1;
        for (let c = 4; c < C; c++) {
            const s = getCell(c, i);
            if (s > bestS) {
                bestS = s;
                bestC = c - 4;
            }
        }
        if (bestS < scoreThreshold) continue;

        const cx = getCell(0, i);
        const cy = getCell(1, i);
        const w = getCell(2, i);
        const h = getCell(3, i);

        // Convert (cx, cy, w, h) → top-left (bx, by).
        let bx = cx - w / 2;
        let by = cy - h / 2;
        let bw = w;
        let bh = h;

        // Clamp to image bounds.
        bx = Math.max(0, Math.min(W - 1, bx));
        by = Math.max(0, Math.min(H - 1, by));
        bw = Math.max(1, Math.min(W - bx, bw));
        bh = Math.max(1, Math.min(H - by, bh));

        out.push({ box: [bx, by, bw, bh], label: bestC, score: bestS });
    }

    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

/**
 * Per-class non-maximum suppression.
 *
 * @param {Array<{box, label, score}>} dets
 * @param {number} iouThreshold
 * @param {number} topk
 */
export function nmsPerClass(dets, iouThreshold, topk) {
    if (dets.length === 0) return dets;

    // Group by class label.
    const byClass = new Map();
    for (const d of dets) {
        if (!byClass.has(d.label)) byClass.set(d.label, []);
        byClass.get(d.label).push(d);
    }

    const keepAll = [];
    for (const arr of byClass.values()) {
        arr.sort((a, b) => b.score - a.score);
        const keep = [];
        for (const candidate of arr) {
            let suppress = false;
            for (const k of keep) {
                if (iou(candidate.box, k.box) > iouThreshold) {
                    suppress = true;
                    break;
                }
            }
            if (!suppress) {
                keep.push(candidate);
                if (keep.length >= topk) break;
            }
        }
        keepAll.push(...keep);
    }

    keepAll.sort((a, b) => b.score - a.score);
    if (keepAll.length > topk) keepAll.length = topk;
    return keepAll;
}

/**
 * Resolve class-name filter to a set of class indices.
 *
 * @param {string} commaList — e.g. "person, car" or "" for all classes
 * @param {string[]} cocoNames — COCO_NAMES from config
 * @returns {Set<number> | null} — null means "no filter"
 */
export function buildClassFilter(commaList, cocoNames) {
    if (!commaList || !commaList.trim()) return null;
    const wanted = commaList
        .split(",")
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean);
    if (wanted.length === 0) return null;

    const indices = new Set();
    for (const w of wanted) {
        const idx = cocoNames.findIndex((n) => n.toLowerCase() === w);
        if (idx >= 0) indices.add(idx);
    }
    return indices.size > 0 ? indices : null;
}

/**
 * Apply a class filter (set of indices) to a detection list.
 */
export function applyClassFilter(dets, filterSet) {
    if (filterSet == null) return dets;
    return dets.filter((d) => filterSet.has(d.label));
}

/**
 * Apply a hard cap on number of detections (sort by confidence, keep top N).
 * Used for the TDYolo `Detectionlimit` parameter.
 *
 * @param {Array} dets — sorted descending by score on input
 * @param {number} limit — 0 means "no limit"
 */
export function applyDetectionLimit(dets, limit) {
    if (!limit || limit <= 0 || dets.length <= limit) return dets;
    return dets.slice(0, limit);
}

// ----------------------------------------------------------------------------
// TODO: SEG-HOOK
// ----------------------------------------------------------------------------
// Stub segmentation decoder. Implement in v2.1 when seg models are added.
// Reference for implementation:
//   - https://github.com/microsoft/onnxruntime-inference-examples (yolo-seg samples)
// Expected ONNX output shape: 2 tensors,
//   [1, N, 4+nc+nm] (detection head with mask coefficients), and
//   [1, nm, mh, mw] (prototype masks).
// Convert: for each surviving detection (after NMS), compute
//   mask = sigmoid(sum_c(coeffs[c] * proto[c, y, x]))
// Pack pixel = floor(class_id + 1) + sigmoid(mask) * 0.99.
// ----------------------------------------------------------------------------
export function decodeYOLOSeg(_tensors, _scoreThr, _topk) {
    // TODO: SEG-HOOK — implement when segmentation is enabled.
    return null;
}
