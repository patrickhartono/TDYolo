# TDYolo - TouchDesigner YOLO Integration

This project was created and dedicated to all the students, colleagues, and friends during my time teaching at the Department of Computational Art at Goldsmiths, University of London 

Farewell for now! Thank you all for everything! 

A self-contained TouchDesigner component that runs YOLO11 object detection inside a headless browser bundled with TouchDesigner — **no Python environment, no conda, no pip install required on the user's machine**. Drop the `.tox` into your project and you're done.

> ⚠️ **Requires an internet connection.** The embedded headless browser needs network access while running (to bootstrap the inference runtime). Make sure your machine is online.

> ✅ **Works on TouchDesigner Commercial and Non-Commercial.** The binary frame pipeline stays within the Non-Commercial 1280×720 TOP cap (frames are 640×640).

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-blue)
![TouchDesigner](https://img.shields.io/badge/TouchDesigner-2023.12000%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

- License: **MIT** (see `tdyolo-web/LICENSE`)
- Architecture inspired by [torinmb/yolo-touchdesigner](https://github.com/torinmb/yolo-touchdesigner). Implementation is independent — no source code was copied. See "Attribution" below.

---

## Quickstart

Two ways to use the component:

### 🟢 Option A — Just the component (recommended for most users)

1. Download `TDYolo_v2.tox` from the [Releases](https://github.com/patrickhartono/TDYolo/releases) page.
2. In TouchDesigner 2023.12000 or newer, drag `TDYolo_v2.tox` into your `.toe` project.

### 🔧 Option B — Full project (developers, modifications, sample videos)

1. Clone the repo:
   ```bash
   git clone https://github.com/patrickhartono/TDYolo.git
   cd TDYolo
   ```
2. Open `TDYolo_v2.toe` in TouchDesigner 2023.12000 or newer.

### Reading detections (both options)

- `report` (table DAT, 8 columns): `Object_Type, Confidence, X_Center, Y_Center, Width, Height, ID, ClassID`
- `summary` (table DAT, 2 columns): `Object_Type, Count`
- Visual overlays via `null2` (bbox), `null5` (labels), `null3` (source frame) — see Outputs below

That is the entire setup. There is no Python environment to install and no external runtime dependency.

---

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| `in1` | TOP | External video feed. If unconnected, the bundled `moviefilein1` (sample video in `video/`) is used. |

## Outputs

The component exposes 6 outputConnectors at its boundary:

| Output | Type | Description |
| --- | --- | --- |
| `report` | DAT (table) | 8 columns: `Object_Type, Confidence, X_Center, Y_Center, Width, Height, ID, ClassID`. Coordinates are **top-left origin, pixel-space** in the 640×640 inference canvas. |
| `summary` | DAT (table) | 2 columns: `Object_Type, Count`. Aggregated per-class counts for the current frame. |
| `null2` | TOP | Bounding box overlay (transparent canvas, only edges drawn, per-class colors). |
| `null5` | TOP | Detection labels overlay (transparent canvas, "person 1: 0.93" badges above each bbox, black text on class-colored background). |
| `null3` | TOP | Synched 640×640 source frame the inference engine actually processed. Use this as the underlying image when compositing overlays. |

Compose at the parent `/project1` level using compositeTOPs to combine source + bboxes + labels into your final view.

---

## Custom parameters

### Detection page

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `Detectionlables` | Str | empty | Comma-separated whitelist of COCO class names (e.g. `person, car, dog`). Empty = all 80 COCO classes. |
| `Confidence` | Float | `0.2` | Minimum detection confidence (0–1). Lower = more detections; higher = stricter. |
| `Frameskip` | Int | `0` | Frames to skip between inference runs (0 = every frame, 2 = run every 3rd frame, etc.). Performance lever. |
| `Detectionlimit` | Int | `1` | Max detections per frame (`0` = unlimited). Highest-confidence kept first. |
| `Model` | Str (read-only) | `yolo11n` | Detection model. Only `yolo11n` is bundled — displayed as text, not configurable. |

Changing any of these parameters auto-reloads the embedded browser inference page (~1–2 second handover). **No manual reload needed.**

### Server page

| Param | Type | Description |
| --- | --- | --- |
| `Port` | Int (display) | The TCP port the embedded webserverDAT is bound to. Assigned automatically at project load. |
| `Reset` | Pulse | Manual browser reload. Same effect as a Detection-page param change. Useful if the visual feels stuck but the server is still alive. |
| `Hardreconnect` | Pulse | **Destroy + recreate** the embedded Chromium browser on the same port. Use when the browser is frozen (`ERR_CONNECTION_REFUSED` or no detections coming through). |
| `Findnewport` | Pulse | Pick a fresh free port and rebind the server. Use only when the port itself is locked by another process. |

### About page

| Param | Type | Description |
| --- | --- | --- |
| `Version` | Str (display) | Component version (`0.1.0`). |
| `License` | Str (display) | `MIT`. |
| `Attribution` | Str (display) | Architecture inspiration credit. |
| `Help` | Pulse | Opens https://github.com/patrickhartono/TDYolo in your default browser. |

---

## How it works (high-level)

```
   in1 (or moviefilein1)
        │
        ▼
   flip1 → letterbox (640×640) → source ───┐
                                           │
                                           ▼
                                   chopexec3 (reads source RGBA directly,
                                              packs to CHW uint8 in Python,
                                              trigger=absTime.frame)
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
                            report (8-col) / summary (2-col)
```

Frames travel TouchDesigner → browser as compact 12-byte-header binary messages followed by H×W×3 CHW uint8 payload. Detections travel browser → TouchDesigner as JSON text messages on the same WebSocket.

---

## Files

```
TDYolo/
├── README.md                          # this file
├── TDYolo_v2.toe                      # bundled TouchDesigner project
├── TDYolo_v2.tox                      # exported drop-in component (drag into your own .toe)
├── ARCHITECTURE-TDYolo_v2.md          # node-level architecture reference
├── COMPARISON.md                      # TDYolo v1 vs TDYolo_v2 vs torinmb's yolo container
├── AGENTS.md                          # collaboration notes
├── tdyolo-web/                        # browser app source (MIT-licensed)
│   ├── LICENSE
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/models/yolo11n.onnx     # bundled detection model
│   ├── src/
│   │   ├── main.js                    # entry — WS lifecycle + mode select
│   │   ├── config.js                  # URL param parsing
│   │   ├── pipeline.js                # per-frame orchestrator
│   │   ├── inference/
│   │   │   ├── onnx.js                # ORT session manager
│   │   │   ├── postprocess.js         # YOLO decode + NMS
│   │   │   └── io.js                  # ImageData/CHW → input tensor
│   │   ├── modes/
│   │   │   ├── binary.js              # TD → browser binary frame protocol
│   │   │   └── webcam.js              # browser webcam fallback
│   │   └── utils/
│   │       ├── protocol.js            # JSON output schema
│   │       └── math.js                # IoU
│   └── dist/                          # built output, embedded into TD VFS
├── video/
│   ├── example-1.mp4                  # short sample
│   └── example-2.mov                  # longer sample
└── python-script/                     # legacy v1 (conda-based) — kept for historical reference
```

---

## Performance notes

- Default backend is **WebGPU** with WASM fallback. Pass `?cpu=true` in the URL to force WASM (slower but compatible with older Chromium builds).
- A 640×640 frame of `yolo11n` runs at roughly **30–60 FPS on an Apple Silicon MacBook Pro** depending on backend choice and webrenderTOP overhead.
- WebSocket frame transport adds ≈ 1 frame of latency vs an in-process inference loop — unnoticeable for most artistic uses.

### Performance levers

| If you need… | Tune |
|---|---|
| Higher FPS on weaker hardware | Increase `Frameskip` (1–3) |
| Fewer overlays (cleaner output) | Lower `Detectionlimit` (3–5) |
| Stricter detections (fewer false positives) | Raise `Confidence` (0.4–0.6) |
| All COCO classes detected | Leave `Detectionlables` empty |

---

## Troubleshooting

**Browser stuck / no detections coming through**
- Pulse `Hardreconnect` (Server page) — destroy + recreate the embedded Chromium on the same port.

**Port conflict (rare — only if another local process grabbed the same port)**
- Pulse `Findnewport` (Server page) — picks a fresh free port and rebinds.

**Detection labels look wrong or off-screen**
- Verify input video is reaching `in1`. The component falls back to bundled sample video if no input is connected.

**Want to verify everything is running**
- Inspect `report` table — should populate within ~3 seconds of project load.

---

## Roadmap

### v3 — Pose, OBB, multi-stream

Not currently planned. Open an issue if interested.

---

## Attribution

The architecture pattern (browser-hosted ONNX inference, frames pushed over WebSocket from TD, JSON predictions returned and parsed by a `datexecuteDAT`) is inspired by Torin Blankensmith's [yolo-touchdesigner](https://github.com/torinmb/yolo-touchdesigner) project.

No source code from `yolo-touchdesigner` has been copied. The browser app, the TouchDesigner-side glue, the binary frame protocol, and the output schema were all written independently, with the official [ONNX Runtime Web examples](https://github.com/microsoft/onnxruntime-inference-examples) and the [Ultralytics ONNX export docs](https://docs.ultralytics.com/integrations/onnx/) as references.

This project is **MIT licensed** to make it easy to use in classrooms and share publicly. See `tdyolo-web/LICENSE` for the full text.

---

## Acknowledgments

- **Ultralytics** for YOLOv11
- **ONNX Runtime Web** team for the in-browser inference runtime
- **Derivative** for TouchDesigner
- **Torin Blankensmith** for the architectural inspiration

---

## License

MIT — Copyright (c) 2026 Patrick Hartono. See `tdyolo-web/LICENSE` for full text.
