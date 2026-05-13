import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Build config for the TDYolo_v2 browser app.
// Output goes to ./dist/ which is then embedded into TouchDesigner's
// virtual file system (VFS) inside the TDYolo_v2 baseCOMP.
//
// We copy the onnxruntime-web WASM artefacts and the YOLO ONNX model
// directly into dist/ so the served URLs resolve via simple basename
// lookup (the TD-side webserver_callback strips path prefixes).
export default defineConfig({
    base: "./", // produce relative URLs so the bundle works under any served path
    build: {
        target: "es2022",
        outDir: "dist",
        emptyOutDir: true,
        rollupOptions: {
            output: {
                // Single bundle keeps the VFS embed simple.
                inlineDynamicImports: false,
                manualChunks: undefined,
            },
        },
    },
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
                    dest: ".",
                },
                {
                    src: "public/models/*.onnx",
                    dest: "models",
                },
            ],
        }),
    ],
});
