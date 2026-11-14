# Phase 5: Final comparison (45 min)

Run after /clear. Paste everything below the line.

You should see: a comparison table you can show immediately.
Checkpoint: can you explain why you must not tune anything on the test set?

---

Phase 5: the head-to-head comparison. This is the one time we use the test set.

1. Evaluate custom CNN, MobileNetV3-small and landmark MLP on the test set: accuracy, macro F1, and per-class F1. For approach B, report both on detected images only and with detection failures counted as errors.
2. Measure model size (parameters and file size in KB), and CPU inference latency per sample (batch size 1, median of 200 runs after warm-up). For approach B also report the end-to-end time including MediaPipe, and include the MediaPipe model size honestly.
3. Save a single results/comparison.csv and make clear charts: accuracy, size, latency.
4. Write a short plain-language interpretation: which approach wins on which metric and why. Do not overstate.
5. Write all of it as src/evaluate.py so the numbers are reproducible. Append the Phase 5 entry to docs/learning-log.md and commit.
