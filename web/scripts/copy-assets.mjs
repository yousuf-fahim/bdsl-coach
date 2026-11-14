// Copy the runtime wasm files from node_modules into public/ so the build
// self-hosts them (no CDN). Only the variants we use are copied; all are under
// Cloudflare Pages' 25 MiB per-file limit.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nm = join(root, "node_modules");
const jobs = [
  ["@mediapipe/tasks-vision/wasm", "public/mediapipe-wasm", [
    "vision_wasm_internal.js", "vision_wasm_internal.wasm",
    "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm",
  ]],
  ["onnxruntime-web/dist", "public/ort", ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]],
];
for (const [src, dst, files] of jobs) {
  mkdirSync(join(root, dst), { recursive: true });
  for (const f of files) copyFileSync(join(nm, src, f), join(root, dst, f));
}
for (const f of ["landmark_mlp.onnx", "hand_landmarker.task", "config.json"]) {
  if (!existsSync(join(root, "public/models", f))) {
    console.error(`Missing public/models/${f}. Run: .venv\\Scripts\\python src\\export_onnx.py`);
    process.exit(1);
  }
}
console.log("assets copied");
