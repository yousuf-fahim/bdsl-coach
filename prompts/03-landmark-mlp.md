# Phase 3: Landmark MLP, approach B (60 min)

Run after /clear. Paste everything below the line.

Checkpoint: what does the gap between training and validation curves tell you? What did augmentation change?

---

Phase 3: train approach B. Using landmarks.npz and the existing splits:

1. First, quick classical baselines on the landmark features (logistic regression and random forest) so I have context for the neural network.
2. Build a small PyTorch MLP (for example input -> 256 -> 128 -> classes, with BatchNorm and Dropout). Train with AdamW, early stopping on the validation set, and class weights if imbalance matters.
3. Add landmark-level augmentation: small random rotation, scale jitter and Gaussian noise on coordinates. Compare results with and without augmentation.
4. Plot training and validation loss and accuracy curves. Explain what they show about overfitting.
5. Save the best model to models/landmark_mlp.pt, print the parameter count, and write everything as src/train_landmark_mlp.py with a fixed seed.
6. Evaluate on the validation set only. Do not touch the test set yet.
7. Append the Phase 3 entry to docs/learning-log.md and commit.
