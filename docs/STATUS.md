# STATUS

> **Current phase: 2 (landmarks), starting.** Last updated: 2026-11-14 20:05

## Done
- **Phase 0 (setup):** `.venv` on Python 3.13 with all packages installed. CPU only (Ryzen 5 7500F, no GPU). Git repo on branch `feat/bdsl-pipeline`.
- **Phase 1 (data):**
  - Dataset: 14,714 images, 49 classes, 0 corrupt.
  - Labels: `labels.json` is **PROVISIONAL** (QUESTIONS.md Q1.1).
  - Split: leakage-aware, 68.7 / 14.8 / 15.0%, with 228 near-duplicate images excluded (Q1.2).
  - Leakage demo: a colour-only 1-NN scores 41.9% on a random split versus 17.4% on ours.

## Next
Phase 2: MediaPipe landmark extraction, run in the background.

## Waiting on you (not blocking)
- Verify the label mapping: open `results/sample_grid.png` (Q1.1).
- Check one suspicious pair, র `29_232` vs ন `31_279` (Q1.3).

## Blockers
None.

## Standing instructions
- **Hosting:** static site only, no Vercel. Output to `web/dist` with a relative `./` base. MediaPipe wasm, the hand landmarker model and the ONNX model are all self-hosted. `docs/DEPLOY.md` covers Cloudflare Pages and GitHub Pages.
- **GPU fallback:** Kaggle, not Colab.
