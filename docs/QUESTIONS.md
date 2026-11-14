# Decisions made while you were away

Each entry: what came up, the options, what I chose, and how to change it.

## Q0.1 Python version (Phase 0)
- **What came up:** PLAN.md asks for Python 3.10-3.12, but only Python 3.13.3 is installed.
- **Options:** (a) install Python 3.12 alongside it, or (b) use 3.13 if MediaPipe supports it.
- **Chosen:** (b). MediaPipe 1.2.3 publishes a `py3-none-win_amd64` wheel, which works on any Python 3, including 3.13. No extra install needed.
- **To change:** install Python 3.12, delete `.venv`, run `py -3.12 -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`.

## Q0.2 Which part of the download to use (Phase 0)
- **What came up:** the Mendeley zip holds a Recognition set (1.8 GB, one image per sign, classification) and a Detection set (8.4 GB, scenes with bounding boxes).
- **Chosen:** Recognition only. It is the classification task this project is about.
- **To change:** extract `Detection Dataset with Annotation.zip` from the outer zip yourself. Nothing in the pipeline uses it.

## Q0.3 Git workflow (Phase 0)
- **What came up:** your global rules say: never commit to `main`, use one branch per task, open a PR. The project rules say: commit at the end of each phase. There's no GitHub remote yet, so I can't push or open a PR.
- **Chosen:** the scaffold you already wrote (CLAUDE.md, PLAN.md, prompts/) is the first commit on `main`, as a baseline. All phase work goes on branch `feat/bdsl-pipeline`, with one commit per phase. I didn't create a GitHub repo, push or open a PR.
- **To change:** create the GitHub repo, then run `git remote add origin <url>`, `git push -u origin main feat/bdsl-pipeline`, and `gh pr create` from the branch. If you'd rather have a branch per phase, I can split the history.

## Q0.4 OneDrive sync of the dataset (Phase 0)
- **What came up:** the project folder is inside OneDrive, so `data/raw/` (1.8 GB) gets uploaded.
- **Chosen:** left as is. 1.8 GB is manageable. git ignores `data/` either way.
- **To change:** in File Explorer, right-click `data` and choose "Free up space", or pause OneDrive.

## Q1.1 Label mapping: PROVISIONAL, please verify (Phase 1)
- **What came up:** `labels.json` must map folders `0`–`48` to Bangla characters, and a wrong mapping would silently break the demo.
- **What I did:** transcribed Table 1 of the BDSL 49 paper (Hasib et al., 2023) into `src/make_labels.py`. Folder `k` holds files named `(k+1)_*.jpg`, and `k+1` is the paper's label number. The audit checks this for all 14,714 files: 0 mismatches.
- **Evidence it's right:**
  - The per-class counts are exactly half of the paper's Table 1 totals. For example, label 2 আ has 638 in the paper and 319 here; label 3 ই has 592 and 296. The paper counts the Recognition and Detection sets together.
  - In `results/sample_grid.png`, digits ০–৫ (folders 36–41) show 0–5 fingers as expected.
- **Odd entries from the paper:** 35 = ং (anusvara), 36 = ঃ (visarga), 47 = ্ (hasanta, shown as ◌্), 48 = "space", 49 = ঞ. The order is the paper's, including ঞ last.
- **To verify:** open `results/sample_grid.png` and check a few letters you know. If something's wrong, edit `TABLE` in `src/make_labels.py` and re-run it. Everything downstream reads `labels.json`.

## Q1.2 Split method: signer IDs aren't recoverable (Phase 1)
- **What came up:** filenames carry no signer ID, so a signer-grouped split isn't possible.
- **Evidence for capture order:** images a few numbers apart look much more alike than random pairs, and the similarity flattens out at a gap of about 20–25 (`results/leakage_gap_curve.png`).
- **Chosen:**
  - Group each class into blocks of 15 consecutive image numbers.
  - Per class, give contiguous runs of blocks to test, then val, then train, starting at a random rotation (seed 42).
  - Exclude 228 images (1.5%) so that no near-duplicate pair (pHash distance ≤ 8) crosses splits.
  - Result: 68.7 / 14.8 / 15.0%.
- **Rejected:**
  - (a) Merging blocks through near-duplicate links. The links chain together, and one class ended up 88% in val.
  - (b) Moving leaking val/test images into train. That put train images inside test blocks: 41% of test images then had a same-class train image within 5 numbers, versus 10.6% with the exclusion rule.
- **Residual risk:** a signer can straddle a block boundary. 10.6% of test images have a train image of the same class within 5 numbers, versus 100% for a random split.
- **To change:** edit `BLOCK` / `NEAR_DUP` in `src/make_splits.py` and re-run it.

## Q1.3 Suspicious image pair (Phase 1)
- `train/28/29_232.jpg` (র) and `train/30/31_279.jpg` (ন) are near-identical (pHash distance 2) but labelled differently. Either one is mislabelled, or the two signs differ only subtly. Both stay in train, so they don't affect evaluation. The other 23 cross-class pairs are visually similar signs from the same signer and background (উ/এ, র/ল/ন). See `results/leakage_report.json`.

## Q1.4 Scope: all 49 classes (Phase 1)
- The project says "alphabet", but BDSL 49 also includes 10 digits and 3 special signs (ং, ঃ, ্) plus "space". I kept all 49 classes, as the dataset defines them.
- **To change:** filter labels in `make_splits.py`.

## Q2.1 MediaPipe detection settings (Phase 2)
- **What came up:** with default settings, MediaPipe found a hand in only 75% of a 392-image train/val sample. The images are tight crops, and the palm detector expects some context around the hand.
- **Chosen:** pad each image with 25% black border on every side and use confidence thresholds of 0.3 (detection and presence). That reached 90.6% on the sample and 89.2% on the full set. 50% padding detected slightly more (91.8%) but found both hands less often (27% vs 30%). The web demo uses the same 0.3 thresholds.
- **To change:** edit `PAD_FRAC` / `MIN_DET_CONF` in `src/landmarks.py`, re-run `src/landmarks.py` (1.5 min), then retrain.

## Q2.2 Feature design for two-handed signs (Phase 2)
- **What came up:** the prompt says to mirror left hands to look like right hands. About 29% of images have two hands detected, and mirroring each hand separately would destroy how the two hands relate to each other.
- **Chosen:**
  - **One hand detected:** mirror it if MediaPipe calls it Left.
  - **Two hands detected:** no mirroring. Order them left-to-right by wrist position, normalise each, and add hand 2's wrist offset relative to hand 1.
  - Result: a 131-number vector (63 + 63 + 2 presence flags + 3 offset).
- **To change:** `normalize_hands()` in `src/landmarks.py`. The JS port in Phase 8 must match it.

## Q2.3 Undetected images (Phase 2)
- **Chosen:** 1,558 images (10.8%) have no detected hand. They're excluded from MLP training and validation selection. In Phase 5 the test accuracy is reported both on detected images only and with every undetected image counted as an error.

## Q4.1 Custom CNN too slow on CPU (Phase 4)
- **What came up:** the first custom CNN design, with two 3×3 convolutions per block (1.19M parameters), measured 2.9 min per epoch, about 73 minutes for 25 epochs (`results/cnn_time_estimates_two_conv_custom.json`). That's over the 60-minute budget. MobileNetV3-small takes 0.5 min per epoch, about 13 minutes.
- **Options:** fewer epochs, smaller images, Kaggle GPU, or a lighter block.
- **Chosen:** one convolution per block (4 blocks, 32/64/128/256 channels, 401k parameters). That's 1.2 min per epoch, about 30 minutes for 25 epochs. It's still a "small custom CNN with 3–4 conv blocks", and it keeps the 128 px input and the full epoch budget. Kaggle wasn't needed.
- **To change:** add the second `Conv2d` back in `SmallCNN` (`src/train_cnn.py`). For speed, run it on Kaggle.

## Q4.2 Custom CNN retrained for 45 epochs (Phase 4)
- **What came up:** after 25 epochs (29.6 min) the custom CNN's best validation epoch was the last one, at 66.1%. It was still improving, so the epoch cap cut it off (`results/cnn_custom_history_25ep.csv`).
- **Chosen:** retrain from scratch with up to 45 epochs and early-stopping patience 8, which is about 52 min at the measured 70 s per epoch. That's inside the 60-minute budget. The cosine learning-rate schedule spans the 45 epochs. MobileNetV3-small was left as is, because its best epoch (23 of 25) came before the cap.
- **To change:** `python src/train_cnn.py --model custom --epochs N`.

## Q4.3 MobileNet's very high validation accuracy (Phase 4), to keep in mind
- MobileNetV3-small reaches 97.5% on validation. Our split keeps capture sessions apart, but the same 14 signers (and probably their rooms) appear in all splits. Some of the accuracy may come from recognising people and backgrounds. The colour-only 1-NN still scores 17% versus 2% chance, so that signal exists.
- **How we check it:** the Phase 6 robustness tests (lighting, occlusion and so on) and your own webcam photos (`data/own/`, PENDING) test exactly this. I haven't changed anything because of it.

## Q6.1 Which model should the demo use? Needs your decision later (Phase 6)
- **What came up:** the plan ships the landmark MLP (approach B) in the browser. But on this dataset MobileNetV3-small is more accurate (97.2% vs 68.6% end-to-end on test) **and** more robust under every perturbation we tried (`results/robustness_summary.csv`). Approach B mostly fails because MediaPipe stops detecting the hand under noise, brightness changes and occlusion.
- **Options:**
  - (a) Keep the landmark MLP, as the phase prompts specify: tiny classifier, background-independent features, and it already ships the hand skeleton.
  - (b) Hybrid: use MediaPipe to find and crop the hand, then run MobileNet (exported to ONNX, about 6 MB) on the crop. MobileNet was trained on hand crops, so it needs a cropper on webcam frames anyway.
  - (c) Run both and show both predictions.
- **Chosen for now:** (a). The Phase 8 prompt asks for the landmark model, and the decisive test, your own webcam photos (`data/own/`, a new person, room and camera), is still PENDING. The report presents this as an open trade-off, not as "B is better".
- **To change:** ask me to add option (b). It needs an ONNX export of MobileNet, a crop step in `web/src/pipeline.js` and a new parity test.

## Q6.2 Robustness method notes (Phase 6)
- Perturbations are applied to each test image after shrinking it to at most 640 px. Many dataset crops are smaller than that, so pixel noise is relatively coarse on them, which may make the noise test harsher than real camera-sensor noise. MediaPipe is re-run on every perturbed image, and approach B is scored end-to-end.
- Own-webcam evaluation: **PENDING.** `src/capture_own.py` and `src/eval_own.py` are ready. `eval_own.py` was self-tested on 6 validation images that were then removed. It writes `results/own_eval.json`, which currently says PENDING.
