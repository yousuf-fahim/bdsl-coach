# STATUS

> **Current phase: 0 (setup), in progress.** Last updated: 2026-11-14 19:15
>
> Dataset ready: 14,714 images extracted to `data/raw/Recognition_Processed_Compressed/`.

## Done
- 19:03: download finished: `Downloads\BDSL 49 A Comprehensive Dataset of Bengali Sign Language.zip` (10.25 GB). It contains two inner zips, Recognition (1.8 GB) and Detection with annotations (8.4 GB).
- Extracted **only the Recognition set** into `data/raw/`, using a guarded extraction that refuses files trying to unpack outside the folder. The detection set is for object detection, which we don't need, and skipping it keeps the OneDrive upload small. The zip in Downloads is untouched.
- Recognition set: 49 class folders (`0` to `48`) with the authors' own `train/` (11,774) and `test/` (2,940) split.

## Next
Phase 0: venv, packages, git init, first commit. Then Phase 1.

## Blockers
None.

## Standing instructions
- **Hosting:** static site only, no Vercel. Output to `web/dist` with a relative `./` base. MediaPipe wasm, the hand landmarker model and the ONNX model are all self-hosted. `docs/DEPLOY.md` covers Cloudflare Pages and GitHub Pages.
- **GPU fallback:** Kaggle, not Colab. The token is in `~/.kaggle/access_token` and the CLI works.
