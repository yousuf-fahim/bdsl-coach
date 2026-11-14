# Learning log

One entry per phase, written by Claude Code: what we did, why, the key concept in plain language, and three likely faculty questions with short answers. Read it top to bottom before your viva.

## Phase 0: Setup (2026-11-14)

**What we did:** created a Python 3.13 virtual environment (`.venv`), installed the 14 libraries in `requirements.txt` (exact versions are pinned in `requirements-lock.txt`), initialised git, and recorded the machine's specs in `results/environment.json` via `src/check_env.py`. The machine has a 6-core AMD Ryzen 5 7500F and no CUDA GPU, so all training runs on the CPU. Kaggle is the fallback if a model is too slow to train.

**Why:** a virtual environment keeps this project's library versions separate from the rest of the computer, so someone else (or a faculty member) can rebuild the exact same setup from `requirements-lock.txt`.

**Key concept: reproducibility.** An ML result is only trustworthy if someone else can get the same number again. That needs three things fixed: the code (git), the library versions (lock file) and the randomness (seed 42 everywhere). If a library version changes, results can shift slightly, which is why we pin versions and record the environment the numbers came from.

**Likely faculty questions**
1. *Why a virtual environment?* It isolates the project's dependencies, so upgrading a library for another project can't silently change our results.
2. *Why fix the random seed?* Weight initialisation, data shuffling and augmentation are random. A fixed seed makes runs repeatable, so differences between models come from the models and not from luck.
3. *Does training without a GPU matter?* Only for speed. The maths is identical. A GPU just runs the matrix operations in parallel. Our models are small enough for a CPU.

## Phase 1: Data audit, leakage check and splits (2026-11-14)

**What we did:**
- Audited all 14,714 images in the BDSL 49 recognition set: 49 classes, 296–319 images each (max/min ratio 1.08, so nearly balanced), all valid RGB JPGs, sizes from 42 px to 5,570 px because they are hand crops.
- Built `labels.json` from the paper's table. It's PROVISIONAL until Fahim checks it.
- Measured near-duplicates with perceptual hashes and made a leakage-aware 70/15/15 split (`src/audit_data.py`, `src/make_labels.py`, `src/make_splits.py`).

**Why:** the test set must contain images the model has never effectively seen. If a near-copy of a test image is in training, the test score measures memory, not recognition.

**Key concept: data leakage.** Leakage is when information about the test set sneaks into training. The images here are stored in capture order, so neighbours are the same person, background and pose a moment apart (`results/leakage_gap_curve.png`). A random split puts these siblings on both sides. We proved it: a nearest-neighbour model that only compares colour histograms, and knows nothing about hand shape, scores 41.9% on a random split but 17.4% on our grouped split (`results/leakage_knn_demo.json`). The extra 24 points come purely from recognising backgrounds. Our split keeps each stretch of capture together, so the test set mostly contains unseen sessions.

**Likely faculty questions**
1. *Why not just split randomly?* Neighbouring images are near-copies (same signer and background). A random split would inflate accuracy. Our colour-only baseline gains 24 points from a random split, without looking at the hand at all.
2. *You don't have signer IDs, so how did you group?* We showed that image similarity fades with index distance and flattens at about 20–25 images, which suggests capture order. So we grouped 15-image blocks and gave each split contiguous runs. We also excluded 228 images (1.5%) so that no near-duplicate pair crosses splits.
3. *Is the leakage fully removed?* No, and we say so. A signer can straddle a block boundary: 10.6% of test images have a training neighbour within 5 indices, versus 100% with a random split. True signer IDs would be needed to remove it completely.
