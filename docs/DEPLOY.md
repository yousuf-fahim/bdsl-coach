# Deploying the demo (static site)

The demo is a **pure static site**. There's no server code, and everything it needs is inside `web/dist/`:

| File | Size | What it is |
|---|---|---|
| `models/landmark_mlp.onnx` | 0.29 MB | our trained classifier |
| `models/hand_landmarker.task` | 7.5 MB | MediaPipe hand model |
| `models/config.json` | small | labels + detection thresholds |
| `mediapipe-wasm/*` | 26 MB | MediaPipe runtime (SIMD + no-SIMD builds) |
| `ort/*` | 14 MB | ONNX Runtime Web (single-threaded wasm) |
| `assets/*` | about 0.5 MB | app code, CSS, self-hosted Noto Sans Bengali |

Total: about 49 MB. The largest file is 14.2 MB, under Cloudflare Pages' 25 MiB per-file limit. No CDN is used. The automated smoke test confirms the page makes zero requests to other hosts (`results/web_smoke.json`).

All asset paths are **relative** (Vite `base: "./"`), so the same `dist/` works at a domain root, under any sub-path, or opened from a local static server.

> The webcam only works over **HTTPS** (or `http://localhost`). Both hosts below give you HTTPS automatically.

---

## 0. Build and test locally (do this first)

Requires Node 20.19+ or 22.12+ (you have Node 25). From the project folder:

```powershell
# (only if you retrained) refresh the model files in web/public/models
.venv\Scripts\python src\export_onnx.py

cd web
npm ci                     # install exact versions from package-lock.json
npm run test:normalize     # JS normalisation == Python (Node test)
npm run test:parity        # headless Edge: JS + ONNX + MediaPipe vs Python
npm run build              # -> web/dist
npm run preview            # open http://localhost:4173 and allow the camera
```

Optional: `node tests/smoke.mjs` tests the built `dist/` with a fake camera. It needs `logs/fake_camera.y4m`, which you create with `.venv\Scripts\python src\make_fake_camera.py`.

`npm run dev` also works for live editing at http://localhost:5173.

---

## A. Cloudflare Pages

### A1. Quickest: upload the built folder (no Git needed)
1. Run step 0 so `web/dist/` exists.
2. Go to https://dash.cloudflare.com and choose **Workers & Pages → Create → Pages → Upload assets**.
3. Name the project `bdsl-coach` and click **Create project**.
4. Drag the **contents** of `web\dist` (or the `dist` folder itself) into the upload box, then click **Deploy site**.
5. You get `https://bdsl-coach.pages.dev`. Open it on your phone and allow the camera.

To update later, run `npm run build` again, then **Create new deployment** in the same project and upload `web\dist` again.

### A1 (CLI variant) with Wrangler
```powershell
cd web
npm run build
npx wrangler@4 login                                    # opens a browser once
npx wrangler@4 pages deploy dist --project-name bdsl-coach
```

### A2. Automatic builds from GitHub
After the repo is on GitHub (see B1):
1. Choose **Workers & Pages → Create → Pages → Connect to Git** and pick the repo.
2. Build settings:
   - Framework preset: **None**
   - **Root directory:** `web`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
3. Environment variables (Production and Preview):
   - `NODE_VERSION` = `22`
   - `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD` = `1` (skips a browser download the build doesn't need)
4. Choose **Save and Deploy**. The model files are committed in `web/public/models/`, so the build doesn't need Python.

---

## B. GitHub Pages

### Base path: nothing to configure
A GitHub *project* site is served from `https://<user>.github.io/<repo>/`, so it isn't at the domain root. Vite apps normally need `base: "/<repo>/"` for that. **This project uses `base: "./"`** (relative paths, set in `web/vite.config.js`), and the app loads every model and wasm file relative to the page. So it works under `/<repo>/` (or any other path) **without changing anything**.

Only if you ever switch to an absolute base: build with `npx vite build --base=/<repo>/`, replacing `<repo>` with the exact repository name. Pages URLs are case-sensitive.

### B1. Put the repo on GitHub
```powershell
cd C:\Users\FAHIM\OneDrive\Documents\PROJECTS\bdsl-coach
gh repo create bdsl-coach --public --source . --remote origin   # or create it on github.com
git push -u origin main feat/bdsl-pipeline
gh pr create --base main --head feat/bdsl-pipeline --fill        # review, then merge yourself
```
Free GitHub Pages needs a **public** repo (private repos need a paid plan).

### B2. Option 1: GitHub Actions (recommended, already prepared)
The workflow `.github/workflows/deploy-pages.yml` builds `web/` and publishes `web/dist`.
1. On GitHub, open the repo and go to **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Merge the PR into `main`. The workflow runs on every push to `main` that touches `web/`. You can also start it from **Actions → Deploy demo to GitHub Pages → Run workflow**.
3. When it finishes, the URL appears in the workflow run and under Settings → Pages: `https://<user>.github.io/bdsl-coach/`.

### B3. Option 2: publish a pre-built folder (no Actions)
```powershell
cd web
npm run build
npx gh-pages@6 -d dist -b gh-pages --nojekyll
```
Then go to **Settings → Pages → Source: Deploy from a branch → Branch: `gh-pages` / root**. The `--nojekyll` flag stops GitHub from hiding files and folders that start with `_`.

---

## After deploying: checklist
- [ ] Open the URL on a laptop and on your phone, and allow the camera.
- [ ] The status line says "Models loaded", and after **Start camera** you see the green and blue hand skeleton.
- [ ] The preview is mirrored like a selfie (expected). Predictions use the raw, un-mirrored frame, like the training photos.
- [ ] Practice mode: the target letter's box turns green after a 0.5 s hold above 70%.
- [ ] Put the URL in `README.md` (marked TODO) and in the report.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| "Camera unavailable" | The page isn't on HTTPS, or the permission was denied. Use the `https://` URL and allow the camera in site settings. |
| "Could not load the models" + 404s in the browser console | The upload is missing `models/`, `mediapipe-wasm/` or `ort/`. Upload the whole `dist` folder. |
| Works locally, blank on GitHub Pages | An absolute base path is in use. Keep `base: "./"` (see "Base path" above). |
| Slow on an old phone | Expected. MediaPipe runs on the CPU in wasm. Close other tabs. |
| Locally, "both async and sync fetching of the wasm failed" | Don't test with `python -m http.server`: it was too slow serving the 14 MB wasm files in our test, and the browser's fetch failed. Use `npm run preview` (or any normal static host). With a proper static server, the build was verified to load correctly from a sub-path (`/bdsl-coach/`), the same situation as GitHub Pages. |
