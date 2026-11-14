# Phase 5: what the comparison shows (plain language)

All numbers come from `results/comparison.csv` (`src/evaluate.py`, a single run on the 2,200 test images) and `results/comparison_ci.csv` (`src/bootstrap_ci.py`, 95% bootstrap intervals from the saved predictions).

**Accuracy: MobileNetV3-small wins clearly.** It scores 97.2% (95% CI 96.6–97.9%) against 73.2% for the custom CNN and 68.6% for the landmark MLP end-to-end. The gap to the custom CNN, +24.1 points (CI 22.2–26.0), is far larger than the uncertainty. Pretraining on ImageNet is the main reason.

**Landmark MLP vs. custom CNN: a tie on detected images, a loss end-to-end.**
- MediaPipe found a hand in 1,995 of 2,200 test images (90.7%).
- On those images the MLP scores 75.7% and the custom CNN 73.3%. The difference (+2.4 points) has a CI from −0.1 to +5.0, so we can't call it a win.
- Counting the 205 images without a detected hand as errors, the MLP drops to 68.6%, which is 4.6 points below the custom CNN (CI −7.0 to −2.0).
- So with about 1/5 of the parameters (74k vs 401k), the landmark classifier matches a from-scratch CNN when it gets landmarks. Detection failures are what cost it.

**Size: depends on what you count.**
- The MLP file is tiny: 297 KB, versus 1,580 KB (custom CNN) and 6,245 KB (MobileNet).
- But approach B also needs MediaPipe's hand model (7,636 KB). So the full approach B is about 7.9 MB, slightly more than MobileNet.

**Speed: also depends on what you count.**
- The MLP alone takes 0.05 ms per image, versus 2.6 ms (MobileNet) and 4.2 ms (custom CNN), on a CPU with 1 thread.
- End-to-end, approach B takes 15.4 ms, because MediaPipe's detection dominates.
- Caveat in B's favour: the CNNs were measured on pre-cropped hand images. A real webcam app using a CNN would also need to find and crop the hand first, which this comparison doesn't charge them for.

**What we do NOT claim.**
- We don't claim approach B is more accurate. It isn't, on this test set.
- We don't claim MobileNet will hold 97% on new people. The test signers also appear in training (in different capture sessions), and a CNN can use their backgrounds. Phase 6 (robustness) and the webcam photos (PENDING) test this.
- These are single-run results, with one split and seed 42. The CIs cover test-set sampling, not training randomness.

**Why the demo still uses approach B.** Its features ignore background by design, it runs entirely in the browser with a 0.3 MB classifier, and MediaPipe also gives the hand skeleton the UI shows. Whether it's the "better trade-off" depends on the robustness results in Phase 6. The report revisits this choice there.
