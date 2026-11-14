# Viva cheat sheet

Read `docs/learning-log.md` once the night before. Every number below comes from `results/`; the file is given in brackets.

## Numbers to know by heart
- **Data:** 14,714 images, 49 classes, 14 signers. Split: 10,109 train / 2,177 val / 2,200 test, with 228 excluded (`split_summary.json`).
- **Leakage demo:** colour-only 1-NN scores 41.9% on a random split vs 17.4% on our split. Chance is 2% (`leakage_knn_demo.json`).
- **MediaPipe detection:** 89.2% of all images, 90.7% of test images (`landmark_detection.json`, `comparison.csv`).
- **Test accuracy:**
  - MobileNet **97.2%**
  - custom CNN **73.2%**
  - landmark MLP **75.7%** on detected images, **68.6%** end-to-end
  
  (`comparison.csv`)
- **Size:** MLP 74k params / 297 KB (+7.6 MB MediaPipe). Custom CNN 401k / 1.6 MB. MobileNet 1.57M / 6.2 MB.
- **Latency:** MLP 0.05 ms alone, 15.4 ms with MediaPipe. MobileNet 2.6 ms. Custom CNN 4.2 ms (CPU, batch 1, 1 thread).

## 20 likely questions

1. **What is the project in one sentence?** We compared CNNs on raw images against a hand-landmark MLP for 49 static BdSL signs, measured accuracy, size, speed and robustness, and shipped the landmark model as a private in-browser webcam coach.

2. **Why two approaches?** They're different philosophies. CNNs learn features from all the pixels. Landmarks reduce the image to 21 hand keypoints, which should be small and background-free. Comparing them on identical splits shows the trade-off.

3. **What is data leakage and how did you prevent it?** Test information leaking into training. Here, neighbouring images are near-copies (same person, background and pose). We grouped images into 15-image capture blocks, gave each split contiguous runs of blocks, and excluded near-duplicates that crossed splits. Proof it matters: a colour-only baseline drops from 41.9% (random split) to 17.4% (ours).

4. **Why not group by signer?** The dataset doesn't record signer IDs. Capture-order blocks are the best available stand-in. Image similarity flattens after about 20–25 indices (`leakage_gap_curve.png`).

5. **How does MediaPipe work?** A palm detector finds the hand, then a landmark network predicts 21 3D keypoints. We use it frozen, as a feature extractor.

6. **What normalisation did you apply and why?** Move the wrist to the origin (position-invariant), divide by the wrist-to-middle-knuckle length (size-invariant), and mirror a lone left hand (left-handers look like right-handers). For two hands we order them left-to-right and add the offset between them, without mirroring.

7. **Why pad the images before MediaPipe?** The dataset images are tight crops, and the palm detector expects context. Padding plus a 0.3 threshold raised detection from 75% to about 90% on a train/val sample (`landmark_detection_settings.json`).

8. **Why is MobileNet so much better than your own CNN?** Transfer learning. It starts from features learned on 1.2M ImageNet photos. Our CNN learns from 10k images from scratch. MobileNet hit 96% validation accuracy after 3 epochs.

9. **Is 97% too good to be true?** It could be partly inflated: the same 14 signers appear in all splits, in different sessions. Grad-CAM shows it looks at hands, not backgrounds, but the real test is new people. The own-webcam evaluation is pending.

10. **Did the landmark approach fail?** It lost on accuracy overall, but it ties the from-scratch CNN on detected images (+2.4 points, CI −0.1 to +5.0) with 1/5 the parameters. Its weakness is detection: 9.3% of test images had no hand found.

11. **Why report two accuracies for approach B?** The MLP can only classify images where MediaPipe finds a hand. "Detected only" measures the classifier. "End-to-end" (misses count as errors) is what users experience.

12. **Is the latency comparison fair?** Not fully, so we report both views. The CNNs were timed on pre-cropped hands. A webcam CNN app would also need a hand detector. Approach B's 15.4 ms includes its detector.

13. **What did the robustness test show?** MobileNet stayed the most accurate at every severity. Approach B collapsed under noise and over-exposure, mostly because MediaPipe stopped detecting hands. When it did detect them, MLP accuracy held up fairly well (`robustness_summary.csv`).

14. **That contradicts your hypothesis. Is that a problem?** No. We stated the hypothesis, tested it properly, and reported that it wasn't supported. That's the scientific method working.

15. **Which signs are hardest and why?** Two-hand "contact" signs (র/ল/ন, উ/এ/ও, ঝ/ফ), where one finger touches different spots on the other hand. Their average landmark vectors are among the closest 1% of all class pairs (উ-এ: 0.43 vs a median of 5.48).

16. **What is Grad-CAM?** It weights the last convolutional feature maps by the gradient of the predicted class score, and shows which image regions drove the decision. For MobileNet: the hands and fingers.

17. **Why AdamW, early stopping, dropout and label smoothing?** AdamW gives adaptive steps with proper weight decay. Early stopping keeps the best-validation epoch. Dropout fights co-adaptation. Label smoothing stops over-confidence.

18. **Why did you use the test set only once?** Repeated looks at the test set bias the estimate toward it. Validation and test scores matched closely (97.5/97.2, 73.8/73.2, 79.0/75.7), so model selection didn't overfit the validation set.

19. **How do you know the browser matches Python?** Automated parity tests: identical features, logits within 7.6×10⁻⁶, and 16/16 same predictions even with MediaPipe JS vs Python on the same images (`web_parity.json`). A smoke test confirms zero external requests.

20. **What would you do next?** (a) Collect signer IDs or new signers for a true unseen-signer test. (b) A hybrid model: MediaPipe crops the hand, a pretrained CNN classifies it. (c) Better two-hand features, such as which finger of hand 1 is touched by hand 2. (d) Motion signs with sequence models.

Bonus, if asked about ethics: it's a learning aid for static signs, not an interpreter. Camera frames never leave the device. The dataset's faces are blurred.

## 5-slide outline (5–7 minutes)
1. **Problem and goal.** BdSL learning aid. Two approaches (CNN on pixels vs landmarks + MLP), same data, measured fairly. Show `docs/demo.gif`.
2. **Data and leakage.** BDSL 49 (49 classes, 14 signers), and why random splits cheat: the 41.9% vs 17.4% demo and the gap curve (`leakage_gap_curve.png`).
3. **Methods.** The pipeline diagram for both approaches. Key choices: padding for MediaPipe, normalisation, transfer learning, augmentations.
4. **Results.** The table from `comparison.csv` plus the robustness figure (`robustness_all.png`). Message: MobileNet wins on accuracy and robustness; landmarks win on classifier size and speed but depend on detection.
5. **Errors, demo and limits.** Confused pairs (র/ল/ন), Grad-CAM, the in-browser demo (private, static), limitations (static signs, 14 signers, pending own-photo test) and next steps.
