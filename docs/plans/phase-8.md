# Phase 8 plan: browser demo (static, self-hosted)

Constraints: no backend, everything runs in the browser, the camera feed never leaves the device. **Hosting change from Fahim:** no Vercel. Build a pure static site with a relative base `./`, output to `web/dist`, self-host every asset (no CDN), and document Cloudflare Pages and GitHub Pages.

1. **Export:** `src/export_onnx.py` produces `models/landmark_mlp.onnx` (opset 17, input `features[batch,131]`, output `logits[batch,49]`). It verifies ONNX Runtime against PyTorch on 20 validation samples (8 with two hands, 8 with one, 4 with none) and writes the parity fixtures plus `web/public/models/{landmark_mlp.onnx, hand_landmarker.task, config.json}`.
2. **Stack:** plain Vite (vanilla JS) with `base: "./"`, `@mediapipe/tasks-vision@1.2.0` (HandLandmarker, CPU delegate, the same 0.3 thresholds as Python), `onnxruntime-web@1.31.1` (wasm, single thread, so no COOP/COEP headers are needed) and `@fontsource/noto-sans-bengali` (self-hosted font). Versions are pinned and at least two weeks old.
3. **Self-hosting:** `scripts/copy-assets.mjs` copies only the needed wasm files from `node_modules` into `public/`. Every file is under Cloudflare's 25 MiB limit. The ONNX Runtime build that loads its glue from our own `./ort/` folder is used, so it isn't bundled twice.
4. **Preprocessing parity:** `web/src/normalize.js` is a line-by-line port of `normalize_hands()`.
   - `tests/normalize.test.mjs` (Node) compares it with the 20 Python feature vectors.
   - `tests/parity.mjs` (headless Edge via Playwright, using the installed browser) checks JS features against Python, ONNX Runtime Web logits against PyTorch, and MediaPipe JS against Python on the same padded images.
5. **UI:** mirrored live video with a landmark overlay, top-3 bars, 5-frame probability smoothing, and Practice mode (green after a 0.5 s hold above 70%). Noto Sans Bengali, responsive down to phone width, a privacy note, and a provisional-label note.
6. **Mirror and handedness:** the raw, un-mirrored frame goes to MediaPipe (like the dataset photos). Only the display is mirrored with CSS, and the overlay canvas is mirrored the same way so it lines up.
7. **Local test:** `npm run build` and `tests/smoke.mjs` serve `dist/` and open it in headless Edge with a fake camera (`src/make_fake_camera.py`, random validation images). They assert that models load, frames are processed, there are no console errors and there are zero non-localhost requests. Screenshots feed `docs/demo.gif` (`src/make_demo_gif.py`).
8. **Deploy docs:** `docs/DEPLOY.md` plus `.github/workflows/deploy-pages.yml`. Fahim deploys.
