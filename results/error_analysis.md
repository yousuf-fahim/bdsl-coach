# Phase 7: error analysis

Sources: `src/error_analysis.py` → `confused_pairs.csv`, `landmark_class_distances.json`, `error_analysis_numbers.json`, `confusion_*.png`, `confused_pairs_examples.png`, `gradcam.png`, `per_block_accuracy.*`. Per-class F1 comes from `per_class_f1.csv` (`src/evaluate.py`). Everything uses the predictions saved in the single test run. The best CNN, MobileNetV3-small, was chosen by **validation** accuracy.

## 1. What gets confused

**Approach B (landmark MLP, images with a detected hand).** The top-5 confused pairs are র-ল (21 errors), ল-ন (20), ঝ-ফ (20), উ-এ (19) and উ-ও (18).

These are the same families of signs the leakage audit had already flagged as near-duplicates across classes:
- **র / ল / ন:** a flat hand plus the index finger of the other hand. The pointing finger touches a different finger of the flat hand for each letter.
- **উ / এ / ও:** an open palm touched by the other hand in slightly different places.
- **ঝ / ফ:** crossed fingers.

The differences are small contact positions between two hands, often with the second hand only partly in the cropped image.

The landmark numbers back this up. Measured as the distance between class-average landmark vectors (training set), these pairs are among the most similar of all 1,176 class pairs:
- উ-এ: 0.43, the most similar pair of all
- র-ন: 0.95 (0.2nd percentile)
- র-ল: 1.04 (0.4th percentile)
- ল-ন: 1.18 (0.6th percentile)
- ঝ-ফ and উ-ও: 1.29 (0.9th percentile)
- median over all pairs: 5.48

The MLP's weakest class is র (F1 0.074). Then come য় (0.338), ঝ (0.351), "space" (0.367) and ও (0.372). "Space" suffers mainly because MediaPipe detects a hand in only 39% of its images.

**Approach A (MobileNetV3-small).** Errors are few and differently distributed:
- **৭ → ৮: 11 errors, all in one direction.** ৭ is thumb plus one finger, ৮ is thumb plus two.
- র-ল (5), ম-ন (5), র-ন (4) and উ-এ (3) are the same two-hand "pointing" family as above.
- MobileNet's weakest classes are ৭ (F1 0.861), র (0.886), ৮ (0.891) and ন (0.894).

## 2. Signer/session effects vs. genuinely similar signs
Per-signer accuracy can't be computed, because the dataset has no signer IDs. As a stand-in we count where each pair's errors fall among the test "capture blocks" (15 consecutive images of one class) (`error_analysis_numbers.json`, `top_pair_error_concentration`).
- **MobileNet's ৭→৮ errors cluster:** 6 of 11 fall in a single block, and only 3 of the 6 relevant blocks have any errors. That points to one signer or session whose ৭ looks like ৮, not a general weakness.
- **The MLP's top pairs are spread out:** 4–6 of 6–7 blocks have errors, and no block holds more than 40% of them. The difficulty lies in the signs themselves, as seen through landmarks.

Accuracy per capture block (`per_block_accuracy.png`, 152 test blocks):
- MobileNet: median 1.00, minimum 0.60
- custom CNN: median 0.80
- landmark MLP: median 0.78

Both weaker models have some blocks at 0%. Performance varies a lot between sessions, which is why per-signer data would matter for a real evaluation.

## 3. Where does the CNN look? (Grad-CAM)
On 10 correct and 10 wrong test predictions, MobileNet's Grad-CAM heat concentrates on the **hands and fingers**, not on backgrounds or clothing (`gradcam.png`). In the wrong cases it usually looks at the right region but at a hand configuration that resembles the predicted class, for example the extended fingers in ৭→৮. This is reassuring, but only qualitative: 20 images don't prove the network never uses background cues.

## 4. Takeaways
1. Approach B's main weaknesses are (a) MediaPipe detection failures, and (b) two-hand "contact" signs whose landmark vectors are nearly identical. A sharper feature for (b) would encode **which finger of hand 1 is closest to the fingertip of hand 2**. That's a promising next step.
2. MobileNet makes few errors, and its biggest confusion looks session-specific.
3. The same sign families (র/ল/ন, উ/এ/ও) are hard for every model, so these are the letters a learner should get the most practice feedback on.
