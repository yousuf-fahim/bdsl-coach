# Phase 1 plan: data audit, leakage check and splits

## What we know before starting
- `data/raw/Recognition_Processed_Compressed/{train,test}/{0..48}/N_M.jpg` holds 14,714 cropped hand images.
- `N` (filename prefix) = folder + 1 = the paper's label number (Table 1 of Hasib et al., 2023). The per-class counts match exactly half of Table 1's totals, because the paper adds the Detection section.
- `M` is an image index within the class. The authors' test set is one contiguous block of about 60 indices per class, and the block differs per class. This suggests files are in capture order (signer by signer or session by session).
- **No signer ID** in filenames or folders. The paper says 14 signers but gives no file-to-signer mapping.

## Steps
1. **`src/audit_data.py`** scans every image: size, mode, format, file size, whether it decodes (corrupt check), perceptual hash (pHash) and difference hash (dHash). It writes `data/processed/inventory.csv` and the following outputs:
   - `results/class_counts.png`: bar chart of images per class.
   - `results/sample_grid.png`: 4 samples per class with the Bangla label.
   - `results/image_sizes.png`: histogram of image sizes.
   - `results/audit_summary.json`: counts, corrupt files, size statistics, imbalance ratio.
2. **`src/make_labels.py`** writes `data/processed/labels.json` from the paper's Table 1, transcribed in code. It maps folder index to paper label number, Bangla character, romanisation and type (vowel / consonant / digit / special). Marked PROVISIONAL in QUESTIONS.md.
3. **Leakage check** (inside `make_splits.py`):
   - all-pairs Hamming distance between 64-bit pHashes, computed in chunks with numpy
   - near-duplicate pairs within a class, and across classes (suspicious)
   - consecutive-index similarity, to test the "capture order" idea
4. **Grouping.** Union-find over two kinds of links: (a) near-duplicate pairs, and (b) consecutive indices `M`, `M+1` in the same class whose hashes are close (same burst or session). If the resulting groups are too large to split, fall back to fixed contiguous index blocks.
5. **Split.** Per class, assign whole groups to train/val/test until about 70/15/15 of the images (seed 42). Write `data/processed/splits.csv` and `results/split_summary.json`. Verify that no group crosses splits and that no test image has a near-duplicate in train.
6. **Report.** Class imbalance, odd or suspicious images, and visually similar classes, in `results/leakage_report.json` and `results/near_duplicate_examples.png`.
7. Learning-log entry and commit.

## Decision note
We deliberately do **not** reuse the authors' 80/20 split. The project rules ask for 70/15/15 with a validation set and leakage-aware grouping. It is also unclear whether the authors' split was grouped. The authors' original split is kept as a column (`orig_split`) in the inventory for reference.
