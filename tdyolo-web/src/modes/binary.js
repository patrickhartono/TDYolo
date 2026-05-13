// TDYolo_v2 — binary frame ingestion mode.
//
// Receives frames pushed from TouchDesigner over WebSocket as binary
// ArrayBuffer messages. Each message has a small fixed-length header
// followed by a CHW uint8 payload.
//
// Binary frame format (TDYolo_v2's own — distinct from torinmb's 16-byte
// header by intentional design):
//
//   Offset Size Field            Meaning
//   ------ ---- ---------------- ------------------------------------------
//   0      1    TYPE             Always 0x10 (= frame payload).
//                                Reserved values: 0x20 = future SEG mask.
//   1      1    DTYPE            Always 0x01 (= uint8).
//   2      1    LAYOUT           Always 0x01 (= CHW planar).
//   3      1    reserved         Must be 0.
//   4-5    2    HEIGHT (u16 LE)
//   6-7    2    WIDTH (u16 LE)
//   8-11   4    FRAME (u32 LE)   Frame counter from TD.
//   12...  N    payload          H * W * 3 bytes (CHW order: R then G then B planes)
//
// Total header size: 12 bytes (4 bytes less than the yolo container's 16-byte
// header, which carries an extra seq field we don't need).
//
// Under load, only the latest received frame is kept — older frames are
// discarded so latency stays bounded.
//
// License: MIT.

import { INPUT_W, INPUT_H } from "../config.js";
import { fromU8CHW } from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";

const HEADER_BYTES = 12;
const TYPE_FRAME = 0x10;
// SEG-HOOK: reserved binary type for future seg mask messages from TD or
// browser-to-browser relay. Not yet used.
// const TYPE_SEG_MAP = 0x20;
const DTYPE_U8 = 0x01;
const LAYOUT_CHW = 0x01;

let latestJob = null;
let isProcessing = false;
let _sender = null;

export function setWebSocketSender(senderFn) {
    _sender = senderFn;
}

function parseHeader(buf) {
    if (buf.byteLength < HEADER_BYTES) return null;
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    const dtype = dv.getUint8(1);
    const layout = dv.getUint8(2);
    if (type !== TYPE_FRAME || dtype !== DTYPE_U8 || layout !== LAYOUT_CHW) {
        return null;
    }
    const height = dv.getUint16(4, true);
    const width = dv.getUint16(6, true);
    const frame = dv.getUint32(8, true);
    const payload = new Uint8Array(buf, HEADER_BYTES);
    return { height, width, frame, payload };
}

/**
 * Receive a binary frame from WebSocket.onmessage and queue it for inference.
 * Only the most recent message is kept — under load, older frames are
 * discarded. The actual inference runs asynchronously in `pumpBinary()`.
 *
 * @param {ArrayBuffer} data
 */
export function handleBinaryMessage(data) {
    if (data.byteLength < HEADER_BYTES) return;
    const job = parseHeader(data);
    if (!job) return;
    if (job.height !== INPUT_H || job.width !== INPUT_W) return; // strict size
    latestJob = job;
    pumpBinary();
}

async function pumpBinary() {
    if (isProcessing) return;
    isProcessing = true;
    try {
        const job = latestJob;
        latestJob = null;
        if (!job) return;
        const input = fromU8CHW(job.payload);
        await runInferencePipeline(input, _sender);
    } catch (e) {
        console.error("[TDYolo_v2 binary]", e);
    } finally {
        isProcessing = false;
        // If a new frame arrived while we were busy, drain it on the next tick.
        if (latestJob) queueMicrotask(pumpBinary);
    }
}
