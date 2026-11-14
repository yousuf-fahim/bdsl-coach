# Phase 2 plan: landmark features

1. Choose detection settings on a 392-image train/val sample (never test): padding 0 / 0.25 / 0.5 x confidence 0.5 / 0.3 (`src/landmark_settings_test.py`).
2. Extract landmarks for every non-excluded image (`src/landmarks.py`, 6 worker processes). Store raw landmarks, handedness, scores and image size in `data/processed/landmarks.npz`.
3. `normalize_hands()` is a standalone function, unit-tested in `tests/test_normalize.py`, and will be ported to JS in Phase 8.
4. `src/landmark_report.py`: detection rates (overall, per split, per class), failure examples, 20 overlays.
5. Undetected images: excluded from training. At test time they're reported both ways (detected-only, and counted as errors).
