# TDYolo_v2 — Drop-in Browser-Based YOLO Detection for TouchDesigner

A self-contained TouchDesigner component that runs YOLO11 object detection
inside a headless browser bundled with TouchDesigner — **no Python environment,
no conda, no pip install required on the user's machine**. Open the `.toe`,
press play.

- License: **MIT** (see `tdyolo-web/LICENSE`)
- Architecture inspired by [torinmb/yolo-touchdesigner](https://github.com/torinmb/yolo-touchdesigner).
  Implementation is independent — no source code was copied. See "Attribution"
  below.

---

## Quickstart

1. Open `TDYolo.toe` in TouchDesigner 2023.12000 or newer.
2. Press play.
3. Read detections from `/project1/TDYolo_v2.out1` (table DAT with 7 columns:
   `Object_Type, Confidence, X_Center, Y_Center, Width, Height, ID`).
4. Read per-class counts from `/project1/TDYolo_v2.out2` (table DAT with 2
   columns: `Object_Type, Count`).

That is the entire setup. There is no Python environment to install and no
external runtime dependency.

---

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| `in1` | TOP | External video feed. If unconnected, `moviefilein1` (the bundled `video/example.mp4`) is used. |

## Outputs

| Output | Type | Schema |
| --- | --- | --- |
| `out1` | DAT (table) | **Report**: 7 columns `Object_Type, Confidence, X_Center, Y_Center, Width, Height, ID`. Coordinates are **top-left origin, pixel-space** in the 640×640 inference canvas. |
| `out2` | DAT (table) | **Summary**: 2 columns `Object_Type, Count`. |
| `out3` | TOP | Synched 640×640 frame that the browser actually saw (use this if you want to overlay boxes). |
| `out4` | DAT | Reserved for future segmentation mask (see Roadmap). Empty in MVP. |

Schema is **drop-in compatible** with the legacy `TDYolo` v1 component — any
downstream nodes pointing at v1's `report` / `summary` can be re-pointed at
`TDYolo_v2.out1` / `TDYolo_v2.out2` without rewiring or column-mapping changes.

---

## Custom parameters

### Detection page

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `Detectionlables` | Str | `person, car` | Comma-separated whitelist of COCO class names. Empty = all classes. |
| `Confidence` | Float | `0.25` | Minimum detection confidence (0–1). |
| `Frameskip` | Float | `0` | Frames to skip between inference runs (0 = every frame). |
| `Detectionlimit` | Int | `0` | Max detections per frame (0 = unlimited). |
| `Model` | Menu | `yolo11n` | Detection model. Only `yolo11n` bundled in MVP. |

Changing any of these parameters live updates the URL passed to the browser
and pulses a reload so the new value takes effect on the next page load
(~3 second handover).

### Segmentation page (placeholder — not active in MVP)

All segmentation params are disabled and labelled `[Future]`. The page exists
so that v2.1 segmentation work is a parameter-flip rather than a network
modification. See Roadmap.

### Server page

| Param | Type | Description |
| --- | --- | --- |
| `Port` | Int (display) | The TCP port that the embedded webserverDAT is listening on. Assigned automatically by `init_port` at project load. |
| `Findnewport` | Pulse | Pick a fresh free port and rebind the server. |
| `Reset` | Pulse | Force-reload the embedded browser. |

### About page

Version, license, and attribution strings.

---

## How it works (high-level)

```
   in1 (or moviefilein1)
        │
        ▼
   flip1 → res1 (640×640) → source ───┐
                                      │
                                      ▼
                              glsl2 (RGB → 3 mono planes, 1920×640 mono8fixed)
                                      │
                                      ▼
                              chopexec3 (trigger=absTime.frame)
                                      │
                                      ▼  WebSocket (binary) → 127.0.0.1:<port>
                              ┌───────┴────────────────────────────┐
                              ▼                                    │
                       webserver_v2 (webserverDAT + VFS-backed     │
                              HTTP server)                         │
                              │                                    │
                              │  serves index.html + JS bundle +   │
                              │  ONNX model from VFS to:           │
                              ▼                                    │
                       webrender1 (headless Chromium)              │
                              │                                    │
                              │  ONNX Runtime Web runs yolo11n     │
                              │  on each binary frame              │
                              │                                    │
                              │  JSON predictions over WS ─────────┘
                              ▼
                       webserver_v2/predictions (textDAT, raw JSON)
                              │
                              ▼
                       datexec1 (parse → tables)
                              │
                              ▼
                       report (7-col) / summary (2-col)
```

Frames travel TouchDesigner → browser as compact 12-byte-header binary
messages followed by H×W×3 CHW uint8 payload. Detections travel
browser → TouchDesigner as JSON text messages on the same WebSocket.

---

## Files

```
TDYolo/
├── TDYolo.toe                       # contains /project1/TDYolo_v2
├── README-TDYolo_v2.md              # this file
├── ARCHITECTURE-TDYolo_v2.md        # node-level architecture reference
├── COMPARISON.md                    # TDYolo v1 vs TDYolo_v2 vs torinmb's yolo container
├── tdyolo-web/                      # browser app source (MIT-licensed)
│   ├── LICENSE
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/models/yolo11n.onnx   # bundled detection model
│   ├── src/
│   │   ├── main.js                  # entry — WS lifecycle + mode select
│   │   ├── config.js                # URL param parsing
│   │   ├── pipeline.js              # per-frame orchestrator
│   │   ├── inference/
│   │   │   ├── onnx.js              # ORT session manager (det + SEG-HOOK stub)
│   │   │   ├── postprocess.js       # YOLO decode + NMS (+ decodeYOLOSeg stub)
│   │   │   └── io.js                # ImageData/CHW → input tensor
│   │   ├── modes/
│   │   │   ├── binary.js            # TD → browser binary frame protocol
│   │   │   └── webcam.js            # browser webcam fallback
│   │   └── utils/
│   │       ├── protocol.js          # JSON output schema
│   │       └── math.js              # IoU
│   └── dist/                        # built output, embedded into TD VFS
└── video/example.mp4                # test footage
```

---

## Roadmap

### v2.1 — Add segmentation

Hooks are already in place. To enable segmentation in v2.1:

1. Export `yolo11n-seg.onnx` and place in `tdyolo-web/public/models/`.
2. In `tdyolo-web/src/inference/postprocess.js`, implement the body of
   `decodeYOLOSeg()` (currently returns `null`). The function signature and
   call-site are already wired.
3. In `tdyolo-web/src/inference/onnx.js`, fill in the body of `runSeg()`
   (currently returns `null`).
4. In `tdyolo-web/src/utils/protocol.js`, implement
   `formatSegmentation()` for binary WebSocket send.
5. In the TD network, fill out `datexec1`'s SEG-HOOK branch (currently
   `pass # TODO`) to write decoded masks into the `segmentation_map` DAT.
6. Flip `TDYolo_v2.par.Segmentationenabled` from `False` to `True`.

Every place a future change is required is tagged with the comment
`SEG-HOOK` in the JS source, and `TODO: SEG-HOOK` in the TD callbacks.
Find them with:

```bash
grep -rn 'SEG-HOOK' tdyolo-web/src/
```

### v3 — Pose, OBB, multi-stream

Not currently planned. Open an issue if you want to discuss.

---

## Performance notes

- Default backend is **WebGPU** with WASM fallback. Pass `?cpu=true` in the
  URL to force WASM (slower but compatible with older Chromium builds).
- A 640×640 frame of `yolo11n` runs at roughly 30–60 FPS on an Apple Silicon
  MacBook Pro depending on backend choice and webrenderTOP's overhead.
- WebSocket frame transport adds ≈ 1 frame of latency vs the in-process
  TDYolo v1.

---

## Attribution

The architecture pattern (browser-hosted ONNX inference, frames pushed
over WebSocket from TD, JSON predictions returned and parsed by a
datexecuteDAT) is inspired by Torin Blankensmith's
[yolo-touchdesigner](https://github.com/torinmb/yolo-touchdesigner) project.

No source code from `yolo-touchdesigner` has been copied. The browser app,
the TouchDesigner-side glue, the binary frame protocol, and the output
schema were all written independently, with the official
[ONNX Runtime Web examples](https://github.com/microsoft/onnxruntime-inference-examples)
and the [Ultralytics ONNX export docs](https://docs.ultralytics.com/integrations/onnx/)
as references.

This project is **MIT licensed** to make it easy to use in classrooms and
share publicly. See `tdyolo-web/LICENSE` for the full text.

---

## License

```
MIT License

Copyright (c) 2026 Patrick Hartono

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
