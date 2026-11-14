# Phase 2: Landmark extraction (45 min)

Run after /clear. Paste everything below the line.

You should see: a detection-rate chart and overlay images that look right.
Checkpoint: why normalize landmarks before feeding them to a network? What would happen if you did not?

---

Phase 2: landmark features (approach B). Using the splits from Phase 1:

1. Run the MediaPipe Hand Landmarker (Tasks API, image mode, up to 2 hands) on every image. Store 21 3D landmarks per detected hand with its handedness label.
2. Normalize each hand: translate the wrist to the origin, scale by hand size (for example wrist to middle-finger-base distance), and mirror left hands to look like right hands. Explain why each step is needed. For two-hand signs, concatenate both hands into one fixed-length vector, zero-padded with an indicator for missing hands.
3. Save features to data/processed/landmarks.npz with labels and split names. Put the logic in src/landmarks.py and make the normalization a standalone function, because I will need the exact same function in JavaScript later.
4. Compute the detection rate overall and per class. Plot it. List the classes where MediaPipe fails most and show example failures.
5. Draw landmark overlays on 20 random images to confirm the pipeline works.
6. Decide how to treat undetected images and tell me your choice. I want accuracy reported two ways later: on detected images only, and with detection failures counted as errors.
7. Append the Phase 2 entry to docs/learning-log.md and commit.
