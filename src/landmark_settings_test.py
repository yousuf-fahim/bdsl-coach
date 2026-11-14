"""Phase 2: choose detection settings on a 400-image TRAIN+VAL sample (never test).

Compares padding (0, 0.25, 0.5) and detection confidence (0.5 vs 0.3) by the share
of images where MediaPipe finds at least one hand.
Output: results/landmark_detection_settings.json
"""
import numpy as np
import pandas as pd

import landmarks as L
from common import PROCESSED, RESULTS, SEED, save_json


def main():
    df = pd.read_csv(PROCESSED / "splits.csv")
    df = df[df.split.isin(["train", "val"])]
    sample = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(8, random_state=SEED)).path.tolist()  # 8 per class = 392
    out = []
    for pad in (0.0, 0.25, 0.5):
        for conf in (0.5, 0.3):
            L.MIN_DET_CONF = L.MIN_PRESENCE_CONF = conf
            L._landmarker = None
            # run in-process to honour the patched confidence (workers would re-import)
            lmk = L._get_landmarker(conf, conf)
            n = [len(L.detect(L.prepare_image(L.RAW / p, L.MAX_SIDE, pad), lmk)) for p in sample]
            n = np.array(n)
            row = {"pad_frac": pad, "confidence": conf, "images": len(n),
                   "detected_any": float((n > 0).mean()), "two_hands": float((n == 2).mean())}
            print(row)
            out.append(row)
    save_json(out, RESULTS / "landmark_detection_settings.json")


if __name__ == "__main__":
    main()
