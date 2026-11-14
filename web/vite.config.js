import { defaultClientConditions, defineConfig } from "vite";

export default defineConfig({
  // Relative base: the built site works from any folder or sub-path
  // (Cloudflare Pages root, GitHub Pages /<repo>/, a USB stick...).
  base: "./",
  resolve: {
    // Use onnxruntime-web's build that loads its wasm glue from env.wasm.wasmPaths
    // (our self-hosted ./ort/ folder) instead of bundling a second copy.
    conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { outDir: "dist", assetsInlineLimit: 0, chunkSizeWarningLimit: 2000 },
  server: { port: 5173, strictPort: true },
});
