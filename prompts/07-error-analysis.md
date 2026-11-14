# Phase 7: Error analysis (40 min)

Run after /clear. Paste everything below the line.

Checkpoint: pick one confusion pair and explain, with evidence, why the model mixes it up.

---

Phase 7: error analysis.

1. Confusion matrices for the best model of each approach. Identify the top 10 most-confused sign pairs.
2. For the top 5 confused pairs, show example images side by side and give a plausible reason (similar finger configuration, for instance). For approach B, back it up with a number, such as the distance between the average landmark vectors of the two classes.
3. Run Grad-CAM on the CNN for about 10 correct and 10 wrong predictions to show what the network looks at.
4. If signer IDs are available, report per-signer accuracy.
5. Write a short interpretation in results/error_analysis.md, then append the Phase 7 entry to docs/learning-log.md and commit.
