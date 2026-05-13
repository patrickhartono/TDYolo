# ARCHITECTURE — TDYolo_v2

Node-by-node reference for the `/project1/TDYolo_v2` baseCOMP inside
`TDYolo.toe`. Companion to
[`ARCHITECTURE-TDYolo.md`](ARCHITECTURE-TDYolo.md) (the legacy v1, in-process
Python) and
[`ARCHITECTURE-yolo.md`](ARCHITECTURE-yolo.md) (Torin Blankensmith's
third-party container that inspired the architectural pattern).

- Format mirrors the existing ARCHITECTURE-*.md files in this repo.
- Every place that v2.1 segmentation work will edit is tagged `SEG-HOOK`.
- Implementation is MIT-licensed and independent from
  `torinmb/yolo-touchdesigner`. See `README-TDYolo_v2.md` for the
  attribution statement.

---

## Top-level shape

```
/project1/TDYolo_v2  (baseCOMP)
│
├── in1 ─────────────────────────► flip1 ─► res1 ─► source ─┬─► glsl2 ─► chopexec3 ─► (WS binary) ─► browser
│                                                          │
│  moviefilein1 ──► in1                                    └─► synched_frame ─► out3
│
├── webserver_v2 (sub-base)
│     ├── webserver1 (webserverDAT)        ◄────── browser WS (JSON predictions)
│     ├── webserver_callback (textDAT)     ◄────── HTTP + WS dispatch
│     ├── init_port (executeDAT)
│     ├── execute1 (executeDAT)            ── starts timer1 at project load
│     ├── timer1 (timerCHOP)               ── 5-second cyclic keepalive ping
│     ├── timer1_callbacks (textDAT)
│     ├── virtualFile (baseCOMP)           ── VFS host (~131 MB of JS + WASM + model)
│     ├── predictions (textDAT)            ── inbound JSON sink
│     ├── tick (textDAT)                   ── heartbeat sink
│     ├── webcam_list (tableDAT)
│     ├── active_client (textDAT)          ── last connected WS client address
│     ├── index (nullDAT) / index_html (textDAT)
│
├── webrender1 (webrenderTOP)              ── headless Chromium, loads served index.html
├── server_status (nullCHOP fed by ws_info infoCHOP)
│
├── datexec1 (datexecuteDAT)               ── parses predictions JSON → report/summary
├── report (tableDAT, 7 cols)              ── DROP-IN COMPATIBLE WITH v1
├── summary (tableDAT, 2 cols)
├── segmentation_map (textDAT)             ── SEG-HOOK placeholder, empty in MVP
│
├── param_exec (parameterexecuteDAT)       ── rebuild URL + pulse reload on custom-par change
│
├── out1 ◄── report
├── out2 ◄── summary
├── out3 ◄── synched_frame
└── out4 ◄── segmentation_map (SEG-HOOK)
```

---

## Custom parameter pages

### `Detection` (active in MVP)

| Param | Type | Default | Used where |
| --- | --- | --- | --- |
| `Detectionlables` | Str | `person, car` | `webrender1.par.url` expression → URL query `classes=` |
| `Confidence` | Float | `0.25` | URL `conf=` |
| `Frameskip` | Float | `0` | URL `skip=` (browser-side throttle) |
| `Detectionlimit` | Int | `0` | URL `limit=` (browser-side cap) |
| `Model` | Menu | `yolo11n` | URL `model=` |

A `parameterexecuteDAT` (`param_exec`) watches these params plus
`Segmentationenabled`. On any value change, the URL is re-built from the
current parameter values and `webrender1.par.reload` is pulsed so the
headless browser loads the new URL.

### `Segmentation` (placeholder — SEG-HOOK)

All four parameters exist on this page but are set `enable = False` and
prefixed `[Future]` in their labels. They will be wired in v2.1.

### `Server`

| Param | Type | Description |
| --- | --- | --- |
| `Port` | Int (display only) | Mirror of `webserver_v2/webserver1.par.port`. Set by `init_port` at project load. |
| `Findnewport` | Pulse | Trigger `init_port._apply()` → pick a fresh free port. |
| `Reset` | Pulse | `webrender1.par.reload.pulse()`. |

### `About`

`Version` (`0.1.0`), `License` (`MIT`), `Attribution` (`Architecture inspired
by torinmb/yolo-touchdesigner`).

---

## Input pipeline

### `in1` — inTOP

External video input. Optional. If unconnected, the bundled
`moviefilein1` provides a fallback feed.

### `moviefilein1` — moviefileinTOP

Plays `video/example.mp4` at speed 0.5. Wired into `in1` as a default
source when no external input is provided.

### `flip1` — flipTOP

`flipy = True`. Matches the existing TDYolo v1 convention so up/down is
oriented as the browser canvas expects.

### `res1` — resolutionTOP

Output `640×640` (driven by `constant1` CHOP). Letterbox is **not**
applied here — the browser app's `drawLetterboxed()` handles aspect
preservation in webcam mode. In binary mode, the source TOP is whatever the
upstream pipeline produced at 640×640.

### `constant1` — constantCHOP

Provides the `chan1 = 640` channel that `res1` reads for both output
width and height.

### `source` — nullTOP

Canonical "what we send to the browser". Inputs to `glsl2` and
`synched_frame`. Keep this node as the read-point for the binary pipeline.

---

## Frame transmitter

### `glsl2` — glslTOP (compute mode)

- Resolution: 1920 × 640
- Format: `mono8fixed` (single 8-bit channel)
- Compute shader: `glsl2_compute` (textDAT)
- Algorithm:
  - The 640×640 RGBA source is unpacked into three side-by-side 640-wide
    mono planes: `[0..640) = R`, `[640..1280) = G`, `[1280..1920) = B`.
  - This layout is what the chopexec3 numpy code expects — three contiguous
    planes that can be sliced into a CHW `(3, 640, 640)` uint8 tensor with
    zero copies between planes.

This is a from-scratch packer. It is intentionally simpler than the
multi-pass packer used in the yolo container — there's only one compute
shader and no helper pixel shader.

### `glsl2_compute` — textDAT (GLSL compute source)

The actual shader. See inline comments in the DAT for the full algorithm.
20 lines including comments.

### `glsl2_pixel` — textDAT (stub)

Empty placeholder. The glslTOP's compute mode does not require a pixel
shader, but TouchDesigner expects the slot to exist on the node, so the
DAT is left empty.

### `glsl2_info` — infoCHOP

Reports `glsl2` cook status for debugging. Not used in the live pipeline.

### `trigger` — constantCHOP

One channel with expression `absTime.frame`. Ticks once per frame.
Drives `chopexec3.par.chop`.

### `chopexec3` — chopexecuteDAT

- `chop = trigger`
- `channel = *`
- `valuechange = True`

On every change of `absTime.frame` (i.e. every frame), `onValueChange`
calls `_send_frame()`:

1. Look up `webserver_v2/active_client` — the address of the connected
   WebSocket client (the headless browser).
2. Read `glsl2.numpyArray(delayed=True)` — the latest unpacked frame as
   `(640, 1920, 4)` float32 (`delayed=True` lets TD give us a copy from a
   later cook frame without blocking).
3. Slice the array into three 640-wide planes (R, G, B), convert to uint8,
   and copy them into the pre-allocated `_payload` buffer (CHW order).
4. Pack a 12-byte little-endian header into `_buf`:
   `[TYPE=0x10, DTYPE=0x01, LAYOUT=0x01, reserved, H:u16, W:u16, FRAME:u32]`.
5. `webserver1.webSocketSendBinary(client, bytes(_buf))`.

All numpy buffers are pre-allocated module-level — zero per-frame
allocations on the hot path.

The 12-byte header is intentionally smaller than the 16-byte header used
by the yolo container. Both work; this layout is documented in
`tdyolo-web/src/modes/binary.js`.

---

## `webserver_v2` sub-base

All HTTP and WebSocket plumbing lives here, kept separate so the rest of
the TDYolo_v2 network can stay focused on inference data flow.

### `webserver1` — webserverDAT

- `par.port` — assigned by `init_port` at project load.
- `par.active = True` (auto-flipped on by `init_port`).
- `par.callbacks = webserver_callback`.

The DAT both serves HTTP and accepts WebSocket connections on the same
port. There is no separate `webrenderDAT` server.

### `webserver_callback` — textDAT

Defines the standard `onHTTPRequest`, `onWebSocketOpen`,
`onWebSocketClose`, `onWebSocketReceiveText`, `onWebSocketReceiveBinary`,
`onWebSocketReceivePing`, `onWebSocketReceivePong`, `onServerStart`,
`onServerStop` callbacks, plus a helper `send_pings(webServerDAT)` called
by the keepalive timer.

HTTP routing:

| URI | Behaviour |
| --- | --- |
| `/` | Return `index.text` (the `index_html` textDAT) with `Cache-Control: no-store`. |
| anything else | Look up the basename of the URI in `virtualFile.vfs[]`. If found, return its `byteArray` with a MIME type guessed via `mimetypes`, with `.wasm`, `.mjs`, `.onnx` special-cased. Else 404. |

WS routing in `onWebSocketReceiveText`: cheap substring check before full
JSON parse (matches the established WS-server pattern in the
TouchDesigner palette):

| substring | Action |
| --- | --- |
| `yolo_detect` | Write to `predictions` textDAT (datexec1 picks it up). |
| `webcamDevices` | Write to `webcam_list` tableDAT. |
| `tick` | Write to `tick` textDAT. |
| `loaded` | Mirror `webserver1.par.port` onto parent's `Port` display param. |
| else | Relay to other connected clients (debug-only). |

**SEG-HOOK** branch reserved at the bottom of
`onWebSocketReceiveBinary` for future segmentation mask uploads from the
browser. Currently empty.

### `init_port` — executeDAT

- `par.start = True`, `par.create = True`, `par.active = True`.

On project load (`onStart`) and on DAT creation (`onCreate`) it runs
`_apply()` with a 10-frame delay (gives TD time to fully initialize).

Two paths:

**Happy path (saved port still free)** — most reloads of an existing .toe:

1. Probe `webserver1.par.port` (the value last saved with the project) by
   trying to `socket().bind(('', port))`. If the OS lets us bind, the port
   is free.
2. Activate `webserver1`. The saved `webrender1.par.url` already points
   at this port, so the browser navigates correctly. No webrender
   recreation needed.

**Conflict path (saved port busy or `Findnewport` pulse)**:

1. Ask the OS for any free port (`bind(('', 0))`).
2. Apply the new port to `webserver1`, restart it.
3. Mirror the port onto the parent component's `Port` custom param.
4. Rebuild the URL via `param_exec._build_url(parent)`.
5. **Destroy and recreate `webrender1`** — this is the only reliable
   way to force the embedded Chromium to navigate to a new URL after a
   prior load failed (e.g. on the first frame of the fresh project, before
   the webserver came up). Plain `par.reload`, `par.resetcount`,
   `par.autorestartpulse`, and active toggles do not navigate the
   stuck instance. Recreation takes ~10–15 seconds for Chromium to
   warm up.

The free-port discovery is ~10 lines and written from scratch.

### `execute1` — executeDAT

On project load, initializes and starts `timer1` (the keepalive timer).

### `timer1` — timerCHOP

5-second cyclic timer. Each cycle, `timer1_callbacks.onCycleStart` calls
`webserver_callback.module.send_pings(webserver1)`, which sends a
WebSocket ping to every connected client to keep the connection alive
through middleboxes.

### `virtualFile` — baseCOMP (VFS host)

A bare baseCOMP whose built-in `.vfs` attribute holds the served bundle:
- `index.html` (small wrapper)
- `assets/index-*.js` (Vite-built browser app)
- ONNX Runtime Web `.mjs` + `.wasm` artefacts (multiple variants — Vite
  copies all of them via `viteStaticCopy`)
- `yolo11n.onnx` (the detection model, ~10 MB)

Total embedded payload is ≈ 131 MB on disk because `vite-plugin-static-copy`
brings in every ONNX RT Web variant. This could be slimmed (only the
`webgpu` + `wasm` variants are actually loaded), but it is harmless: VFS
is mmap-backed and only what the browser requests is read from disk.

The TouchDesigner palette `virtualFile` component is **not** required —
every baseCOMP has a built-in `.vfs` attribute that supports `addFile()`,
`__getitem__`, and iteration.

### `predictions`, `tick`, `webcam_list`, `active_client`, `index`, `index_html`

Sink DATs and small content DATs. `index_html` is the actual served HTML;
`index` is a `nullDAT` aliasing it for clarity in the callback code.

### `param_exec` — parameterexecuteDAT

Lives at `/project1/TDYolo_v2/param_exec`. Watches
`Detectionlables`, `Confidence`, `Frameskip`, `Detectionlimit`,
`Model`, `Segmentationenabled` on the parent (the TDYolo_v2 baseCOMP).
On any value change:

1. Read all current values from the parent.
2. Build the new URL string (`http://localhost:<port>?wsPort=...&binary=...
   &conf=...&classes=...&...&model=...&seg=...`).
3. URL-encode any unsafe characters via the
   `''.join('%{:02X}'.format(ord(c)) if c in ' %<>#{}|\\^~[]\`' else c for c in raw)`
   pattern.
4. Set `webrender1.par.url` to that string (CONSTANT mode).
5. Pulse `webrender1.par.reload` to navigate.

Using CONSTANT mode + parexec instead of an EXPRESSION on
`webrender1.par.url` is intentional. WebrenderTOP only navigates when the
parameter value changes; EXPRESSION mode does not always fire a "value
changed" notification reliably, so the explicit set + reload pulse is the
robust path.

---

## Browser host

### `webrender1` — webrenderTOP

- `par.url = http://localhost:<port>?wsPort=<port>&binary=1&conf=...&...`
  (constant mode, written by `param_exec`).
- `par.options = --force_high_performance_gpu` (and any flags added by
  user).
- `par.active = True`.
- `par.mediastream = True` (allows webcam fallback when binary mode is off).
- `par.resolutionw = par.resolutionh = 640`.

The webrenderTOP is the offline Chromium that actually runs the ONNX
inference. Its texture output is not used downstream (the inference
results travel back via WebSocket, not via the rendered pixels).

### `ws_info` — infoCHOP (feeds `server_status`)

Wired to `webserver1`. Reports `server_running`,
`websocket_connections`, `total_cooks`, etc.

### `server_status` — nullCHOP

Always-cook null fed by `ws_info`. Used by debug tools and by the
`Port` custom parameter mirror.

---

## Output parsers

### `datexec1` — datexecuteDAT

Watches the `predictions` textDAT. On every cell change, parses the JSON
payload, then:

1. Clears the `report` and `summary` tables (preserving header rows).
2. For each prediction in `predictions[]`, append a row to `report` with
   `[object_type, confidence, x_center, y_center, width, height, id]`.
3. Build a per-class count map and append rows to `summary`.

**SEG-HOOK**: a top-level `if seg_data: ...` dispatch branch is
already present, currently `pass # TODO: SEG-HOOK`. When v2.1 enables
segmentation, the body will write a decoded mask into the
`segmentation_map` placeholder DAT.

### `report` — tableDAT (7 columns)

`Object_Type, Confidence, X_Center, Y_Center, Width, Height, ID`. Cleared
+ rewritten each frame. Header is preserved.

### `summary` — tableDAT (2 columns)

`Object_Type, Count`. Cleared + rewritten each frame.

### `segmentation_map` — textDAT (SEG-HOOK)

Empty placeholder. The v2.1 datexec1 SEG-HOOK branch will write decoded
mask data here.

### `synched_frame` — nullTOP

Pass-through of `source` exposed on `out3`. Lets downstream nodes overlay
detection boxes on the exact frame the browser saw, including any
letterboxing applied client-side.

---

## Data flow (one frame)

1. **TD frame** — `glsl2` produces a new `(1920, 640)` mono8 texture
   containing the unpacked R|G|B planes of the source.
2. **trigger ticks** — `trigger.value0` changes (`absTime.frame` is
   monotonic), `chopexec3.onValueChange` fires.
3. **TD → browser** — `_send_frame()` packs the 12-byte header + CHW
   uint8 payload and calls `webSocketSendBinary(client, buf)`.
4. **Browser ingestion** — `tdyolo-web/src/modes/binary.js` parses the
   header, builds a `Float32Array` input tensor from the CHW uint8 payload
   (zero-allocation: the tensor is pre-allocated module-level).
5. **Inference** — `tdyolo-web/src/inference/onnx.js` runs `detSession.run()`
   on the input tensor. Output is decoded by `postprocess.js` (score
   threshold, per-class NMS, class-name filter, detection limit).
6. **Browser → TD** — `tdyolo-web/src/utils/protocol.js`'s
   `formatDetections()` builds the TDYolo-compatible JSON, sent as a
   text WS message.
7. **WS receive** — `webserver_callback.onWebSocketReceiveText` cheaply
   matches `yolo_detect` and writes the raw JSON into the `predictions`
   textDAT.
8. **TD parse** — `datexec1` parses the JSON and rewrites `report` and
   `summary`.
9. **Downstream** — the same frame's `synched_frame` is exposed on `out3`
   so users can overlay boxes onto it.

End-to-end latency is approximately 1 inference frame plus 1 cook frame
on a modern Apple Silicon Mac.

---

## SEG-HOOK index

Every place a future v2.1 segmentation implementation needs to edit:

### Browser app (`tdyolo-web/src/`)

| File | What's already there | What v2.1 adds |
| --- | --- | --- |
| `config.js` | Parses `ENABLE_SEG`, `MODEL_SEG_KEY`, `SEG_SCORE_T`, `SEG_TOPK` from URL. | Nothing — already complete. |
| `inference/onnx.js` | `segSession = null` slot; `initSessions()` gated branch loads `${MODEL_SEG_KEY}.onnx` only if `ENABLE_SEG`. | Implement `runSeg()` body. |
| `inference/postprocess.js` | `decodeYOLOSeg()` stub returning `null`. | Implement decode logic from ONNX RT Web YOLO-seg sample. |
| `pipeline.js` | `if (segResult) sender(binary)` branch already wired. | Nothing — branch fires once `runSeg` returns non-null. |
| `modes/binary.js` | `TYPE_SEG_MAP = 0x20` reserved in protocol comment. | Add browser-side handler if TD ever pushes seg masks. |
| `utils/protocol.js` | `formatSegmentation()` stub returning `null`. | Implement binary mask serialiser. |

### TD-side

| Node | What's already there | What v2.1 adds |
| --- | --- | --- |
| `webserver_callback.onWebSocketReceiveBinary` | TODO: SEG-HOOK comment. | Parse binary mask, write into `segmentation_map`. |
| `datexec1` | `if seg_data: pass # TODO: SEG-HOOK`. | Body of the branch. |
| Custom params `Segmentation` page | All four params exist, `enable=False`. | Flip `enable=True`, set sensible defaults. |
| `segmentation_map` textDAT | Empty placeholder. | Becomes the live sink. |
| `out4` | Wired to `segmentation_map`. | Live data flowing through. |

Find them all with:

```bash
grep -rn 'SEG-HOOK' tdyolo-web/src/
```

---

## Why CONSTANT URL + parexec instead of EXPRESSION

Originally `webrender1.par.url` was an EXPRESSION that read each custom
parameter and rebuilt the URL on every cook. This worked for *evaluation*
(`webrender1.par.url.eval()` returned the right string) but webrenderTOP
only navigates when it sees a `parChange` notification, and EXPRESSION-mode
parameters don't always emit one when their dependencies change.

The fix is to make `url` a CONSTANT and use a parameterexecuteDAT
(`param_exec`) to: rebuild the URL string in Python, assign it
(triggering the navigate), and pulse `par.reload` for good measure. This
gives reliable browser refresh on every Detection-page parameter change.

---

## Differences from TDYolo v1

| Concern | v1 (`/project1/TDYolo`) | v2 (`/project1/TDYolo_v2`) |
| --- | --- | --- |
| Inference runtime | Local conda env (~3 GB) + ultralytics + PyTorch | Headless browser (webrenderTOP) + ONNX Runtime Web |
| Install requirement | User must install conda + env | None — open the .toe |
| Inference happens | In-process Python, blocking the TD cook | Async via WebSocket, off the TD cook thread |
| Output schema (`report`) | 7 cols pixel coords | **7 cols pixel coords — identical** |
| Output schema (`summary`) | 2 cols `Object_Type, Count` | **2 cols — identical** |
| FPS on M-series MBP | ~25–40 FPS | ~30–60 FPS (varies by webrender + WebGPU support) |
| Adds segmentation easily? | No — requires Python plumbing | Yes — SEG-HOOK markers identify every edit |
| License | Project-default | MIT |

Both components can run side-by-side. The drop-in test in Phase 4
confirmed that pointing existing `dattochop` operators at
`TDYolo_v2.out1` works without any column-mapping changes.

---

## Risks observed during build

1. **Menu parameter assignment** — `p.menuNames = 'yolo11n'` iterates the
   string character-by-character (`['y', 'o', 'l', 'o', '1', '1', 'n']`)
   instead of storing the string as a list of length 1. Always assign
   `menuNames` and `menuLabels` as **lists**: `p.menuNames = ['yolo11n']`.
2. **chopexecuteDAT / parameterexecuteDAT auto-stubs** — setting
   `par.callbacks = my_dat` causes TD to silently auto-create an empty
   stub DAT alongside, overriding the assigned one. Always re-check
   `par.callbacks` after the assignment.
3. **webrenderTOP URL navigate** — EXPRESSION mode evaluates correctly
   but doesn't always trigger navigation. Use CONSTANT + parexec.
4. **webrenderTOP stuck on failed load** — if Chromium hits an
   `ERR_CONNECTION_REFUSED` (which happens during the brief window between
   project load and `init_port` firing), subsequent `par.reload`,
   `par.resetcount`, `par.reloadsrc`, and `par.autorestartpulse` calls do
   **not** make it re-attempt the navigation. The only reliable fix is
   `op.destroy()` + recreate via `comp.create(webrenderTOP, ...)`. This is
   why `init_port` keeps the saved port when possible.
5. **VFS payload size** — `vite-plugin-static-copy` brought in every
   ORT Web variant for ~131 MB. Harmless but a slim-down opportunity.

---

## See also

- [`README-TDYolo_v2.md`](README-TDYolo_v2.md) — user-facing quickstart and license.
- [`ARCHITECTURE-TDYolo.md`](ARCHITECTURE-TDYolo.md) — legacy v1 architecture.
- [`ARCHITECTURE-yolo.md`](ARCHITECTURE-yolo.md) — `yolo` container by Torin Blankensmith.
- [`COMPARISON.md`](COMPARISON.md) — feature matrix across v1, v2, and the `yolo` container.
