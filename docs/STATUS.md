# STATUS

> **Current phase: 4 (CNN baselines), training in the background.** Last updated: 2026-11-14 19:37 (CNN training started 19:35)

## Done
- **Phase 0:** environment, CPU only, git branch `feat/bdsl-pipeline`.
- **Phase 1:**
  - 14,714 images, 49 classes.
  - Leakage-aware split, 68.7 / 14.8 / 15.0%.
  - Labels are **PROVISIONAL** (Q1.1).
- **Phase 2:** MediaPipe found a hand in 89.2% of images. Worst classes: "space" 39%, গ 65%.
- **Phase 3:** landmark MLP reaches **79.0%** validation accuracy on detected images (70.4% counting misses as errors). 73,777 parameters, trains in 20 seconds.

## In progress (20:35)
- Phase 4:
  - **MobileNetV3-small:** done. 97.5% validation accuracy, trained in 11.2 min.
  - **Custom CNN, 25 epochs:** 66.1% validation accuracy, but still improving at the last epoch.
  - **Custom CNN, 45 epochs:** retraining now (Q4.2), ETA about 21:20. Log: `logs/train_cnn_custom.log`.
- `src/robustness.py`: fixed a worker-pool hang. It now uses a fresh pool per setting with a timeout and retry, and was verified on 60 validation images.

## Prepared while training runs (uncommitted until their phase)
- `src/evaluate.py` (Phase 5), `src/robustness.py`, `src/capture_own.py` and `src/eval_own.py` (Phase 6), and `src/error_analysis.py` (Phase 7).
- **Phase 8 web demo** is built and tested locally:
  - JS normalisation matches Python exactly.
  - Browser ONNX logits are within 7.6×10⁻⁶ of PyTorch.
  - MediaPipe JS agrees with Python on 16/16 predictions.
  - The smoke test of `web/dist` passes, with zero external requests.
  - `docs/DEPLOY.md` is written for Cloudflare Pages and GitHub Pages.

## Next
Phase 5: the one-time test-set comparison, after MobileNet finishes (about 20:20).

## Waiting on you (not blocking)
- Verify the label mapping: `results/sample_grid.png` (Q1.1).
- Check one suspicious pair (Q1.3).

## Blockers
None.

## Standing instructions
- **Hosting:** static site only, no Vercel. Output to `web/dist` with a relative `./` base, and everything self-hosted. `docs/DEPLOY.md` covers Cloudflare Pages and GitHub Pages.
- **GPU fallback:** Kaggle, not Colab.
