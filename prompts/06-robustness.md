# Phase 6: Robustness tests (60 min)

Run after /clear. Paste everything below the line. When Claude Code says the capture script is ready, take about 10 photos each of about 10 signs in your own room, then tell it: Photos are in data/own/, continue.

Checkpoint: why can a model score 98% on the test set and still fail on your own webcam? (Look up the term domain shift.)

---

Phase 6: robustness. Test how each approach holds up when conditions change.

1. Apply perturbations to the test images at 4-5 severity levels each: Gaussian noise, blur, darker and brighter lighting, rotation (up to 15 degrees), random occlusion patches, JPEG compression. For approach B, re-run MediaPipe on each perturbed image.
2. Plot accuracy against severity for all three models in one chart per perturbation.
3. Create src/capture_own.py, a small webcam script that lets me take 10 photos each of about 10 signs in my own room and saves them to data/own/. I will capture them, then evaluate all models on this tiny out-of-distribution set. Tell me when the script is ready and wait for me.
4. Write src/robustness.py so everything is reproducible, then append the Phase 6 entry to docs/learning-log.md and commit.
