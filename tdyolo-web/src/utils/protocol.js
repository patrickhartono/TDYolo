// TDYolo_v2 — output protocol formatter.
//
// Produces JSON in the legacy TDYolo `report` table schema so the existing
// top-level XY_from_Row1/2 dattoCHOPs continue to work without rewiring.
//
//   {
//     "type": "yolo_detect",
//     "predictions": [
//       {object_type, confidence, x_center, y_center, width, height, id}
//     ]
//   }
//
// Coordinates are emitted in **pixel space** of the source image (typically
// 640×640 since that is what TD's `res1` produces), matching the existing
// `report` table's units. The per-object `id` is a per-class running counter
// (first person = 1, second person = 2, first car = 1, …), also matching
// TDYolo's existing semantics. The id is NOT a persistent tracker id across
// frames — TDYolo never had one.
//
// License: MIT.

import { COCO_NAMES, INPUT_W, INPUT_H } from "../config.js";

/**
 * Format detection results as TDYolo `report`-schema JSON for the WS send.
 *
 * @param {Array<{box: [number, number, number, number], label: number, score: number}>} dets
 *        Output of decodeYOLO / nmsPerClass — boxes in [x_topleft, y_topleft,
 *        w, h] in INPUT_W × INPUT_H pixel space, label is the class index.
 * @returns {object} the JSON-serialisable message
 */
export function formatDetections(dets) {
    // Per-class running counter for `id` (matches TDYolo semantics).
    const counters = new Map();

    const predictions = dets.map((d) => {
        const [x, y, w, h] = d.box;
        const className = COCO_NAMES[d.label] ?? `class_${d.label}`;
        const id = (counters.get(className) ?? 0) + 1;
        counters.set(className, id);
        return {
            object_type: className,
            confidence: +d.score.toFixed(3),
            x_center: +(x + w / 2).toFixed(1),
            y_center: +(y + h / 2).toFixed(1),
            width: +w.toFixed(1),
            height: +h.toFixed(1),
            id,
        };
    });

    return {
        type: "yolo_detect",
        width: INPUT_W,
        height: INPUT_H,
        predictions,
    };
}

/**
 * SEG-HOOK: future segmentation-output formatter.
 * Reserved binary message shape (header layout for future v2.1):
 *   [type:u8=0x20, reserved:3, width:u16, height:u16, frame:u32] + Float32 pixels
 * The pixel pack format matches the per-pixel `class_id + alpha` convention
 * documented in ARCHITECTURE-yolo.md §C.4.10 (or any equivalent your seg
 * implementation chooses). Implement when segmentation is added.
 */
export function formatSegmentation(_segResult) {
    // TODO: SEG-HOOK — format binary segmentation map for WS send.
    return null;
}
