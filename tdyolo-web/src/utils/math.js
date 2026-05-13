// TDYolo_v2 — small math helpers.
// License: MIT.

/**
 * Intersection-over-Union for two AABB boxes in [x, y, w, h] format.
 * Returns 0 when boxes don't overlap.
 */
export function iou(a, b) {
    const [ax, ay, aw, ah] = a;
    const [bx, by, bw, bh] = b;
    const ax2 = ax + aw, ay2 = ay + ah;
    const bx2 = bx + bw, by2 = by + bh;
    const ix = Math.max(ax, bx);
    const iy = Math.max(ay, by);
    const ix2 = Math.min(ax2, bx2);
    const iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix);
    const ih = Math.max(0, iy2 - iy);
    const inter = iw * ih;
    const uni = aw * ah + bw * bh - inter + 1e-9;
    return inter / uni;
}
