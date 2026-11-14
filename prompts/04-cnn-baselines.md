# Phase 4: CNN baselines, approach A (75-90 min)

Run after /clear. Paste everything below the line.

Checkpoint: why does the pretrained model usually beat the from-scratch one? What is transfer learning in one sentence?

---

Phase 4: train approach A on raw images, same splits and seed as approach B.

1. Build a data loader at 128x128 with augmentation (random crop, small rotation, color jitter). Explain each augmentation choice.
2. Model 1: a small custom CNN from scratch (3-4 conv blocks). Model 2: pretrained MobileNetV3-small fine-tuned for our classes.
3. Detect CPU vs GPU and estimate training time before starting. If it will take longer than about 40 minutes per model, tell me and propose options (fewer epochs, smaller images, Colab) before running.
4. Run training in the background with logs so I can watch progress. Use early stopping on validation.
5. Plot learning curves, save the best checkpoints to models/, and print parameter counts and training time for each.
6. Validation set only. Do not use the test set.
7. Append the Phase 4 entry to docs/learning-log.md and commit.
