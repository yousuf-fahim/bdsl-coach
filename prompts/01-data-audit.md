# Phase 1: Data audit, leakage check and splits (60-75 min) PLAN

Download the dataset into data/raw/ first. Run in plan mode (Shift+Tab), read the plan, then approve. Paste everything below the line.

You should see: splits.csv, labels.json, charts in results/, and a plain-language explanation of leakage.
Checkpoint: what is data leakage, and why would a random split give a falsely high accuracy on this dataset?

---

Phase 1: data. I've put BDSL 49 in data/raw/. If it isn't there, tell me exactly what to download and stop. Plan first, then do the following:

1. Inspect the folder structure: number of classes, images per class, image sizes and formats. Flag corrupt files. Save a class-count bar chart and a grid of sample images per class into results/.
2. Class labels: the folders may be named with numbers or Roman letters. Build data/processed/labels.json mapping each class to its Bangla letter, using the dataset paper's label table as the source. Show me the mapping and ask me to confirm it before continuing. A wrong mapping would silently ruin the demo.
3. Leakage check: compute perceptual hashes (imagehash) and find clusters of near-duplicate images. Check whether signer IDs can be recovered from file names or folder structure. Explain in simple terms why random splitting leaks information here.
4. Build train/val/test splits (70/15/15), stratified by class and grouped by signer if recoverable, otherwise by near-duplicate cluster. Save as data/processed/splits.csv. Write a script src/make_splits.py so it is reproducible.
5. Report class imbalance and anything odd (very similar classes, suspicious images).
6. Append the Phase 1 entry to docs/learning-log.md and commit.

Do not start Phase 2.
