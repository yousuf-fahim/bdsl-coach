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

## Phase 2: Hand landmarks with MediaPipe (2026-11-14)

**What we did:** ran Google's MediaPipe Hand Landmarker on all 14,486 kept images (1.5 min on 6 CPU cores). For each hand it gives 21 3D points (wrist, knuckles, fingertips). We turned these into a 131-number feature vector per image (`src/landmarks.py`). MediaPipe found at least one hand in 89.2% of images (`results/landmark_detection.json`). The hardest classes are "space" (39%), গ (65%) and ঝ (71%): hands seen edge-on, overlapping or interlocked (`results/landmark_failures.png`).

**Why:** approach B never shows the classifier pixels, only hand geometry. Background, skin tone and lighting are thrown away by design, which should make it smaller and more robust.

**Key concept: normalisation makes features invariant.** Raw coordinates depend on where the hand is in the frame and how close it is to the camera. We subtract the wrist position (translation invariance) and divide by the wrist-to-middle-knuckle distance (scale invariance). We also mirror a lone left hand so that left-handed signers look like right-handed ones. The classifier then only has to learn the hand shape, not every position and size it might appear at. We padded the tight dataset crops with a black border because MediaPipe's palm detector expects context around the hand. That alone raised detection from 75% to 83%, and lowering the confidence threshold to 0.3 took it to about 90%.

**Likely faculty questions**
1. *What is MediaPipe doing?* Two networks: a palm detector finds the hand's bounding box, then a landmark model predicts 21 3D keypoints inside it. We use it as a frozen feature extractor. Only our MLP is trained.
2. *Why not mirror both hands in two-handed signs?* Mirroring each hand separately would break how the hands relate spatially. We instead order the hands left-to-right and add the offset between their wrists.
3. *What about the 10.8% of images with no detection?* That's a real weakness of approach B. In Phase 5 we report accuracy both on detected images and with failures counted as errors, so it isn't hidden.

## Phase 3: Landmark MLP, approach B (2026-11-14)

**What we did:** trained classifiers on the 131-number landmark vectors (`src/train_landmark_mlp.py`), validation set only (`results/landmark_mlp_val.json`):

| Model | Val acc (detected images) | Counting undetected as errors |
|---|---|---|
| Logistic regression | 73.5% | 65.5% |
| Random forest | 72.1% | 64.3% |
| MLP, no augmentation | 77.3% | 68.9% |
| **MLP + augmentation** | **79.0%** | **70.4%** |

The selected MLP has 73,777 parameters and trains in about 20 seconds on a CPU. Class weights weren't needed, since class sizes differ by at most 8%.

**Why:** the classical baselines show how much the neural network adds over simple models: about 5.5 points over logistic regression. Augmentation adds about 1.7 more.

**Key concept: overfitting vs. underfitting.** Overfitting means training accuracy keeps rising while validation accuracy stalls or falls: the network memorises. Our curves (`results/landmark_mlp_curves.png`) show train and validation tracking closely. Making the network bigger didn't help either: 512→256 hidden units scored 77.6% on validation (`results/landmark_mlp_capacity.json`). So the ceiling is the information in the features, not model size. Some signs differ in ways 21 noisy keypoints can't capture well (overlapping hands, edge-on views). Augmentation adds small random rotations, stretches and noise to the landmark coordinates, which teaches the network that those changes don't alter the sign.

**Likely faculty questions**
1. *Why AdamW and early stopping?* AdamW adapts the step size per weight and applies weight decay correctly. Early stopping keeps the weights from the epoch with the lowest validation loss, before overfitting starts.
2. *Why is the MLP better than logistic regression?* Logistic regression can only draw straight boundaries between classes. The MLP's hidden layers with ReLU can learn curved boundaries, which helps with combinations like "this finger bent AND that one straight".
3. *Why report two accuracies?* The MLP can only classify images where MediaPipe found a hand. "End-to-end" accuracy counts the roughly 11% detection failures as errors, which is what a user would actually experience.
