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

## Phase 4: CNN baselines, approach A (2026-11-14)

**What we did:** trained two CNNs on 128×128 images with the same splits and seed (`src/train_cnn.py`, `src/cnn_data.py`), validation set only (`results/cnn_val.json`):

| Model | Parameters | Val acc | Macro F1 | Training time (CPU) |
|---|---|---|---|---|
| Custom CNN (4 conv blocks, from scratch) | 401,489 | 73.8% | 0.731 | 54.0 min (45 epochs) |
| MobileNetV3-small (ImageNet-pretrained, fine-tuned) | 1,568,081 | **97.5%** | 0.975 | 11.2 min (25 epochs) |

Two compromises (QUESTIONS.md Q4.1 and Q4.2):
- The planned two-convs-per-block design would have taken about 73 min, so we used one conv per block.
- The custom CNN was still improving after 25 epochs (66.1%), so we retrained it for 45 epochs.

**Why:** the CNNs are the "classic" computer-vision approach. They see every pixel, including the background, which is both their strength and their risk.

**Key concept: transfer learning.** MobileNet already learned edges, textures and shapes from 1.2 million ImageNet photos, so on our 10k images it only has to adapt those features to hand signs. It hit 96% validation accuracy within 3 epochs (`results/cnn_curves.png`). The custom CNN starts from random weights and has to learn everything from our small dataset, so it's slower and weaker. Augmentation:
- **Random crops:** the hand isn't always centred.
- **Small rotations:** the camera may be tilted.
- **Colour jitter:** lighting and skin tone vary.
- **No horizontal flips:** a flipped image swaps the hands and can turn one sign into another.

**Likely faculty questions**
1. *Why is MobileNet so much better than your own CNN?* Pretraining: it starts with general visual features from ImageNet, while our CNN learns from 10k images alone. Same task, very different starting points.
2. *Is 97.5% too good to be true?* Possibly partly. The split keeps capture sessions apart, but the same 14 signers appear in every split, and a CNN can use background and skin cues. The robustness tests and webcam photos of a new person (Phase 6) check this.
3. *Why label smoothing and a cosine learning rate?* Label smoothing (0.05) stops the network from becoming over-confident. The cosine schedule lowers the learning rate smoothly so training settles into a good minimum.

## Phase 5: Head-to-head comparison on the test set (2026-11-14)

**What we did:** evaluated all three models on the 2,200 test images, **once** (`src/evaluate.py`, `results/comparison.csv`). We added 95% bootstrap confidence intervals from the saved predictions (`src/bootstrap_ci.py`, `results/comparison_ci.csv`). Headline results:
- **Accuracy:** MobileNetV3-small 97.2%, custom CNN 73.2%, landmark MLP 75.7% on images with a detected hand (68.6% counting the 9.3% detection failures as errors).
- **Classifier size:** MLP 297 KB, custom CNN 1.6 MB, MobileNet 6.2 MB. Approach B also needs the 7.6 MB MediaPipe model.
- **Latency:** MLP 0.05 ms alone, 15.4 ms end-to-end with MediaPipe. MobileNet 2.6 ms, custom CNN 4.2 ms.

Interpretation: `results/comparison_interpretation.md`.

**Why:** a fair comparison needs the same test images, a test set touched only once (so we can't tune toward it), and every cost counted: accuracy, size and speed.

**Key concept: confidence intervals and "is the difference real?"** One accuracy number on 2,200 images has sampling noise. The bootstrap resamples the test images thousands of times to see how much the score would wobble. MobileNet's +24-point lead over the custom CNN has an interval of 22–26 points, so it's clearly real. The MLP's +2.4-point edge over the custom CNN on detected images has an interval from −0.1 to +5.0, which crosses zero, so we call it a tie. Validation and test scores were close for all models (for example 97.5% vs 97.2%), so choosing models on the validation set didn't overfit it.

**Likely faculty questions**
1. *Why touch the test set only once?* Each look at test results tempts you to tweak the model toward them. Then the test score is no longer an unbiased estimate of performance on new data.
2. *Your landmark approach lost. Is the project a failure?* No. It matches a from-scratch CNN with 1/20th the parameters when the hand is detected. Its weak point (detection) is measured and reported. A pretrained CNN winning on in-distribution accuracy is a valid, explainable result.
3. *Is the latency comparison fair?* Partly. The CNNs were timed on images already cropped to the hand. In a webcam app they'd also need a hand detector, which approach B gets from MediaPipe. We report classifier-only and end-to-end times separately for that reason.

## Phase 6: Robustness (2026-11-14)

**What we did:** applied 7 perturbations at 5 severities each to every test image, re-running MediaPipe each time for approach B (`src/robustness.py`, `results/robustness_all.png`, `results/robustness_summary.csv`, `results/robustness_examples.png`). Results at the middle severity (3) versus clean accuracy:
- **MobileNet:** loses at most 17 points (occlusion), and 1 point or less for darker, rotation and JPEG.
- **Custom CNN:** collapses under blur (−55 points) and noise (−29).
- **Landmark MLP:** collapses under noise, where MediaPipe detects only 3% of hands at σ = 35 (−68.5 points), and loses about 50 points when much brighter. It stays stable under rotation (−3) and JPEG (−3), and degrades more slowly than the custom CNN under blur.

We also wrote `src/capture_own.py` and `src/eval_own.py` for webcam photos of a new person. That evaluation is **PENDING** until Fahim takes the photos.

**Why:** a model used in a real room will see dim light, blur and cheap cameras. Clean test accuracy doesn't show how gracefully a model fails.

**Key concept: a pipeline is only as strong as its weakest stage.** We expected landmarks to be robust because they ignore background and colour. But approach B is two stages, a hand detector and then a classifier. When the hand *is* found, the MLP's accuracy holds up fairly well: 75.5% clean, 61–76% at severity 3 for blur, darker, rotation and JPEG. But noise and over-exposure make MediaPipe miss the hand entirely, and then the classifier never gets a chance. MobileNet, pretrained on millions of varied photos, has learned features that tolerate these changes. **The plan's hypothesis that B would be more robust was not supported on this data,** and that's a valid, reportable finding.

**Likely faculty questions**
1. *Why did the "background-independent" approach turn out less robust?* Its features are background-independent, but its hand *detector* isn't perturbation-proof. Most of B's losses are detection failures (dotted line in `robustness_all.png`), not wrong classifications.
2. *Are these perturbations realistic?* Darker, brighter, blur, JPEG and small rotations are common with webcams. Strong noise is less common, and our noise is applied to small crops, so that test may be harsh (QUESTIONS.md Q6.2). Your own webcam photos will be the real test.
3. *Does this mean the demo should use MobileNet?* On this dataset the evidence favours it. It would need a hand cropper (MediaPipe again) in front of it, and it's untested on new people. That decision is left open (QUESTIONS.md Q6.1).

## Phase 7: Error analysis (2026-11-14)

**What we did:** analysed the saved test predictions (`src/error_analysis.py`). The written interpretation is in `results/error_analysis.md`. We produced:
- confusion matrices for the best model of each approach: MobileNet (chosen by validation accuracy) and the landmark MLP
- the top-10 confused pairs, with example images
- landmark distances between the class averages of the confused pairs
- Grad-CAM on 20 MobileNet predictions
- accuracy per capture block, as a stand-in for per-signer accuracy (signer IDs don't exist)

**Why:** a single accuracy number hides *which* signs fail and *why*. Error analysis tells you what to fix next and which letters a learner needs most help with.

**Key concept: Grad-CAM and "is the model right for the right reason?"** Grad-CAM asks which regions of the last convolutional feature maps pushed the score of the predicted class up. It colours those regions on the image. MobileNet's heat sits on the hands and fingers, not the background (`results/gradcam.png`), which supports (but doesn't prove) that it learned hand shape. For the landmark model, distances between class-average landmark vectors explain its confusions directly. Its worst pairs (উ-এ, র-ন, র-ল, ল-ন, ঝ-ফ) are all in the closest 1% of the 1,176 class pairs. These are two-hand signs where one hand's finger touches a different spot on the other hand.

**Likely faculty questions**
1. *Which signs are hardest and why?* The র/ল/ন and উ/এ/ও families: a flat or open hand touched by a finger of the other hand at slightly different positions. In landmark space their averages are 5–13× closer than a typical class pair (0.43–1.18 versus a median of 5.48).
2. *Is MobileNet's top error (৭→৮) a model weakness?* Partly a session effect: 6 of its 11 errors come from one 15-image capture block. Most of MobileNet's errors are spread thin.
3. *Why no per-signer accuracy?* BDSL 49 doesn't record who signed each image. Per capture block is the closest stand-in, and it shows large session-to-session variation for the weaker models: some blocks score 0%.
