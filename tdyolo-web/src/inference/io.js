// TDYolo_v2 — input tensor preparation.
//
// Maintains a single shared Float32Array(1*3*640*640) reused across frames,
// wrapped in an onnxruntime-web Tensor. Both ingest modes (binary frame
// pushed from TD, browser webcam) write into the same buffer in-place.
//
// License: MIT.

import * as ort from "onnxruntime-web";
import { INPUT_W, INPUT_H } from "../config.js";

const NUM = 3 * INPUT_H * INPUT_W;
export const f32InputBuffer = new Float32Array(NUM);
export const inputTensor = new ort.Tensor("float32", f32InputBuffer, [
    1, 3, INPUT_H, INPUT_W,
]);

/**
 * Fill the shared input tensor from a CHW uint8 payload (e.g. binary
 * WebSocket frame pushed by TouchDesigner). Divides by 255 into [0, 1].
 *
 * @param {Uint8Array} payload — length must equal 3*H*W
 */
export function fromU8CHW(payload) {
    // payload layout: [R...][G...][B...]
    for (let i = 0; i < NUM; i++) {
        f32InputBuffer[i] = payload[i] * (1 / 255);
    }
    return inputTensor;
}

/**
 * Fill the shared input tensor from a 2D canvas ImageData (RGBA uint8).
 * Converts to planar CHW float32 in [0, 1].
 *
 * @param {ImageData} imgData — must be INPUT_W × INPUT_H
 * @param {boolean} flipH — mirror horizontally during pack
 */
export function fromImageData(imgData, flipH = false) {
    const src = imgData.data; // RGBA
    const W = imgData.width;
    const H = imgData.height;
    const plane = W * H;
    if (!flipH) {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const p = y * W + x;
                const s = p * 4;
                f32InputBuffer[0 * plane + p] = src[s] / 255;
                f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
                f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
            }
        }
    } else {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const x2 = W - 1 - x;
                const p = y * W + x;
                const s = (y * W + x2) * 4;
                f32InputBuffer[0 * plane + p] = src[s] / 255;
                f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
                f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
            }
        }
    }
    return inputTensor;
}
