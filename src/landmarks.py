"""Phase 2: MediaPipe hand landmarks -> fixed-length feature vectors (approach B).

Pipeline per image
  1. load RGB, shrink so the longest side is MAX_SIDE (speed; similar to a webcam frame)
  2. pad PAD_FRAC of the size on every side with black. The dataset images are
     tight hand crops; MediaPipe's palm detector expects some context around the
     hand (as in a webcam frame), and padding raised the detection rate in our
     test (see results/landmark_detection_settings.json).
  3. run the HandLandmarker (Tasks API, IMAGE mode, up to 2 hands)
  4. store raw landmarks (normalised image coords x, y in [0,1] and relative depth z)
     plus handedness, and the padded image size (needed to undo the aspect ratio)
  5. normalize_hands() turns raw landmarks into the 131-number feature vector.

normalize_hands() is deliberately standalone and uses only basic maths, because
web/src/normalize.js re-implements it line by line (Phase 8 parity test).

Run: .venv\\Scripts\\python src\\landmarks.py            (full extraction)
     .venv\\Scripts\\python src\\landmarks.py --overlays  (only redraw figures)
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from common import MODELS, PROCESSED, RAW, RESULTS, SEED, save_json

MODEL_PATH = MODELS / "mediapipe" / "hand_landmarker.task"
MAX_SIDE = 640
PAD_FRAC = 0.25
MIN_DET_CONF = 0.3       # web demo uses the same thresholds
MIN_PRESENCE_CONF = 0.3
N_LM = 21                # landmarks per hand
FEAT_DIM = 2 * N_LM * 3 + 2 + 3  # 131: two hands, two presence flags, inter-hand offset

# Landmark indices (MediaPipe hand model)
WRIST, MIDDLE_MCP = 0, 9


# --------------------------------------------------------------------------
# Normalisation: the function the browser must reproduce exactly.
# --------------------------------------------------------------------------
def normalize_hands(hands, width, height):
    """Turn up to two detected hands into one 131-number feature vector.

    hands  : list of (landmarks, handedness) with landmarks a (21, 3) array of
             MediaPipe normalised coords (x, y in [0, 1], z relative depth) and
             handedness "Left" or "Right" as reported by MediaPipe.
    width, height : size of the image MediaPipe saw (to undo aspect ratio).

    Steps, per hand
      a) to pixel-like units: x*W, y*H, z*W  (so circles stay circles)
      b) translate: wrist -> origin             (position in frame doesn't matter)
      c) scale: divide by |wrist - middle-finger MCP| (hand size/distance doesn't matter)
      d) mirror (x -> -x) only when exactly ONE hand is visible and it is a Left
         hand, so left-handed signers look like right-handed ones. With two hands
         we do NOT mirror: that would destroy how the hands relate to each other.
    Two hands are ordered left-to-right by wrist x in the image (handedness
    labels are unreliable on tight crops), and we add the offset of hand 2's
    wrist from hand 1's wrist, divided by hand 1's scale.

    Layout: [hand0 (63) | hand1 (63) | present0, present1 | offset (3)]
    Missing hands are zeros with present = 0.
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not hands:
        return feat
    pts = []
    for lm, side in hands:
        p = np.asarray(lm, dtype=np.float64).reshape(N_LM, 3).copy()
        p[:, 0] *= width
        p[:, 1] *= height
        p[:, 2] *= width
        pts.append((p, side))
    if len(pts) == 2:
        pts.sort(key=lambda t: t[0][WRIST, 0])  # left-to-right in the image
    wrists, scales = [], []
    for slot, (p, side) in enumerate(pts[:2]):
        wrist = p[WRIST].copy()
        scale = float(np.linalg.norm(p[MIDDLE_MCP, :2] - wrist[:2])) or 1.0
        q = (p - wrist) / scale
        if len(pts) == 1 and side == "Left":
            q[:, 0] = -q[:, 0]
        feat[slot * 63:(slot + 1) * 63] = q.reshape(-1)
        feat[126 + slot] = 1.0
        wrists.append(wrist)
        scales.append(scale)
    if len(pts) == 2:
        feat[128:131] = (wrists[1] - wrists[0]) / scales[0]
    return feat


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
_landmarker = None


def _get_landmarker(min_det=MIN_DET_CONF, min_pres=MIN_PRESENCE_CONF):
    """One HandLandmarker per worker process (created lazily)."""
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_pres,
            min_tracking_confidence=0.5,
        )
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    return _landmarker


def prepare_image(path, max_side=MAX_SIDE, pad_frac=PAD_FRAC):
    """Load, shrink to max_side, pad with black. Returns an RGB uint8 array."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    if pad_frac > 0:
        w, h = im.size
        pw, ph = int(round(w * pad_frac)), int(round(h * pad_frac))
        canvas = Image.new("RGB", (w + 2 * pw, h + 2 * ph))
        canvas.paste(im, (pw, ph))
        im = canvas
    return np.asarray(im)


def detect(rgb, landmarker=None):
    """Run MediaPipe on an RGB array. Returns list of (landmarks(21,3), side, score)."""
    import mediapipe as mp
    lmk = landmarker or _get_landmarker()
    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        out.append((arr, hd[0].category_name, float(hd[0].score)))
    return out


def extract_one(args):
    """Worker: one image path -> raw detection record."""
    rel, max_side, pad_frac = args
    rgb = prepare_image(RAW / rel, max_side, pad_frac)
    hands = detect(rgb)
    raw = np.zeros((2, N_LM, 3), dtype=np.float32)
    sides = ["", ""]
    scores = np.zeros(2, dtype=np.float32)
    for k, (lm, side, sc) in enumerate(hands[:2]):
        raw[k], sides[k], scores[k] = lm, side, sc
    return rel, len(hands), raw, sides, scores, rgb.shape[1], rgb.shape[0]


def run_extraction(paths, max_side=MAX_SIDE, pad_frac=PAD_FRAC, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    jobs = [(p, max_side, pad_frac) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(extract_one, jobs, chunksize=16))


def features_from_record(n_hands, raw, sides, w, h):
    hands = [(raw[k], sides[k]) for k in range(min(n_hands, 2))]
    return normalize_hands(hands, w, h)


# --------------------------------------------------------------------------
def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", action="store_true", help="only redraw figures from landmarks.npz")
    args = ap.parse_args()

    if not args.overlays:
        df = pd.read_csv(PROCESSED / "splits.csv")
        df = df[df.split != "excluded"].reset_index(drop=True)
        t0 = time.time()
        recs = run_extraction(df["path"].tolist())
        secs = time.time() - t0
        n_hands = np.array([r[1] for r in recs])
        raw = np.stack([r[2] for r in recs])
        sides = np.array([r[3] for r in recs])
        scores = np.stack([r[4] for r in recs])
        wh = np.array([[r[5], r[6]] for r in recs])
        X = np.stack([features_from_record(n_hands[i], raw[i], sides[i], wh[i, 0], wh[i, 1])
                      for i in range(len(recs))])
        np.savez_compressed(
            PROCESSED / "landmarks.npz",
            X=X, y=df["label"].to_numpy(), split=df["split"].to_numpy(), path=df["path"].to_numpy(),
            n_hands=n_hands, raw=raw, sides=sides, scores=scores, wh=wh,
            settings=np.array([MAX_SIDE, PAD_FRAC, MIN_DET_CONF, MIN_PRESENCE_CONF]))
        print(f"extracted {len(recs)} images in {secs/60:.1f} min")
        save_json({"images": len(recs), "seconds": round(secs, 1),
                   "max_side": MAX_SIDE, "pad_frac": PAD_FRAC,
                   "min_detection_confidence": MIN_DET_CONF,
                   "min_presence_confidence": MIN_PRESENCE_CONF},
                  RESULTS / "landmark_extraction_run.json")
    from landmark_report import make_report
    make_report()


if __name__ == "__main__":
    main()
