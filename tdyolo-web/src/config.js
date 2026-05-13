// TDYolo_v2 — URL parameter parsing.
//
// All configuration is read from query-string params on the page URL.
// TouchDesigner builds that URL inside `webrender1.par.url` from custom
// parameters on the TDYolo_v2 baseCOMP.
//
// License: MIT.

const qs = new URLSearchParams(location.search);

const boolish = (v, def = false) =>
    v == null ? def : /^(1|true|on|yes)$/i.test(String(v));

const pickStr = (name, def) => (qs.has(name) ? qs.get(name) : def);

const pickNum = (name, def) => {
    if (!qs.has(name)) return def;
    const x = parseFloat(qs.get(name));
    return Number.isNaN(x) ? def : x;
};

const pickInt = (name, def) => {
    if (!qs.has(name)) return def;
    const x = parseInt(qs.get(name), 10);
    return Number.isNaN(x) ? def : x;
};

// ----------------------------------------------------------------------------
// Network
// ----------------------------------------------------------------------------

export const WS_PORT = pickStr("wsPort", "62309");
export const USE_BINARY = boolish(qs.get("binary"), false);
export const USE_CPU = boolish(qs.get("cpu"), false); // force WASM backend

// ----------------------------------------------------------------------------
// Detection (active in MVP)
// ----------------------------------------------------------------------------

export const ENABLE_DET = boolish(qs.get("det"), true);
export const MODEL_DETECT_KEY = pickStr("model", "yolo11n");
export const DET_SCORE_T = pickNum("conf", 0.25);
export const DET_TOPK = pickInt("topk", 100);

// Detection class filter — comma-separated COCO class names from TD's
// `Detectionlables` custom parameter.  Empty string means "all classes".
export const DET_CLASS_FILTER = pickStr("classes", "");

// Per-frame frame-skip count (matches the existing TDYolo `Frameskip`
// parameter). 0 means inference every frame.
export const FRAME_SKIP = pickInt("skip", 0);

// Hard cap on returned detections per frame (TDYolo `Detectionlimit`).
// 0 means unlimited (use TOPK).
export const DET_LIMIT = pickInt("limit", 0);

// ----------------------------------------------------------------------------
// Webcam
// ----------------------------------------------------------------------------

export const WEBCAM_LABEL = pickStr("webcamLabel", null);
export const FLIP_HORIZONTAL = boolish(qs.get("flipHorizontal"), false);

// ----------------------------------------------------------------------------
// SEG-HOOK: Segmentation parameters reserved for future implementation.
// All currently default to disabled. The JS pipeline gates on ENABLE_SEG and
// the runSeg() body is a stub. To enable segmentation in v2.1:
//   1. Export yolo11n-seg.onnx and place into VFS.
//   2. Implement runSeg() and decodeYOLOSeg().
//   3. Flip ENABLE_SEG=true via TD's Segmentationenabled custom param.
// ----------------------------------------------------------------------------

export const ENABLE_SEG = boolish(qs.get("seg"), false);
export const MODEL_SEG_KEY = pickStr("segModel", "yolo11n-seg");
export const SEG_SCORE_T = pickNum("segConf", 0.2);
export const SEG_TOPK = pickInt("segTopk", 100);

// ----------------------------------------------------------------------------
// Fixed
// ----------------------------------------------------------------------------

export const INPUT_W = 640;
export const INPUT_H = 640;

// COCO 80-class label list (canonical Ultralytics ordering).
// Used by postprocess.js to map class indices → string names without
// having to embed the full names dict from the .pt model.
export const COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
];
