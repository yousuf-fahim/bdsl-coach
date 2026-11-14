# STATUS

> **Current phase: 6 (robustness), running in the background.** Last updated: 2026-11-14 21:15

## Done
- **Phase 0:** environment, CPU only, branch `feat/bdsl-pipeline`.
- **Phase 1:**
  - 14,714 images, 49 classes.
  - Leakage-aware split, 68.7 / 14.8 / 15.0%.
  - Labels are **PROVISIONAL** (Q1.1).
- **Phase 2:** MediaPipe found a hand in 89.2% of images.
- **Phase 3:** landmark MLP reaches 79.0% validation accuracy on detected images.
- **Phase 4:** validation accuracy: MobileNetV3-small 97.5%, custom CNN 73.8% (45 epochs, Q4.2).
- **Phase 5, test set (used once):** `results/comparison.csv`, with 95% CIs in `results/comparison_ci.csv`.
  - **MobileNetV3-small: 97.2%**
  - Custom CNN: 73.2%
  - Landmark MLP: 75.7% on images with a detected hand, 68.6% end-to-end.
  - The MLP classifier is 297 KB and takes 0.05 ms, but needs 7.6 MB of MediaPipe and 15.4 ms end-to-end.

## In progress
- Phase 6: 7 perturbations × 5 severities on the test set (`logs/robustness.log`).
- Own-webcam evaluation: **PENDING** your photos. The scripts are ready and self-tested.

## Next
Phase 7 (error analysis), then committing Phase 8 (demo already built and tested), then Phase 9 (report).

## Waiting on you (not blocking)
- Verify the label mapping: `results/sample_grid.png` (Q1.1).
- Check one suspicious pair (Q1.3).
- Take your own photos: `.venv\Scripts\python src\capture_own.py`, then `.venv\Scripts\python src\eval_own.py`.

## Blockers
None.

## Standing instructions
- **Hosting:** static site only, no Vercel. Output to `web/dist` with a relative `./` base, and everything self-hosted.
- **GPU fallback:** Kaggle, not Colab.
