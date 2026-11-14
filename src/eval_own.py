"""Phase 6: evaluate all three models on your own webcam photos in data/own/<class>/*.jpg.

This is a tiny out-of-distribution test: new person, new room, new camera, and
full webcam frames instead of the dataset's tight hand crops. Expect lower
accuracy, especially for the CNNs, which were trained on crops.

Images go through exactly the same path as the robustness test's "clean" level
(shrink to 640 px -> CNN letterbox / pad + MediaPipe + MLP).
If data/own is empty, writes status PENDING and exits.

Outputs: results/own_eval.json, results/own_predictions.csv
Run: .venv\\Scripts\\python src\\eval_own.py
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import ROOT, RESULTS, load_labels, save_json
from evaluate import NAMES
from robustness import predict_all, to_cnn_input, to_landmark_features
from models_io import load_cnn, load_mlp
import landmarks as L


def main():
    labels = load_labels()
    own = ROOT / "data" / "own"
    files = sorted(p for p in own.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        save_json({"status": "PENDING", "note": "No photos in data/own yet. Run src/capture_own.py first."},
                  RESULTS / "own_eval.json")
        print("PENDING: no photos in data/own/. Run src/capture_own.py first.")
        return
    y = np.array([int(p.parent.name) for p in files])
    cnn_in, feats, dets = [], [], []
    for p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((L.MAX_SIDE, L.MAX_SIDE))
        f, d = to_landmark_features(im)
        cnn_in.append(to_cnn_input(im))
        feats.append(f)
        dets.append(d)
    models = {"custom": load_cnn("custom"), "mobilenet": load_cnn("mobilenet"), "landmark_mlp": load_mlp()}
    preds = predict_all(models, np.stack(cnn_in), np.stack(feats).astype(np.float32), np.array(dets))
    rows = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in files], "label": y,
                         "label_bangla": [labels[c]["bangla"] for c in y], "detected": dets})
    for name, pr in preds.items():
        rows[f"pred_{name}"] = pr
    rows.to_csv(RESULTS / "own_predictions.csv", index=False)
    res = {"status": "DONE", "n_images": len(files), "n_classes": int(len(set(y))),
           "detection_rate": float(np.mean(dets)),
           "accuracy": {NAMES[n]: float((p == y).mean()) for n, p in preds.items()},
           "landmark_mlp_accuracy_detected_only":
               float((preds["landmark_mlp"][np.array(dets)] == y[np.array(dets)]).mean()) if any(dets) else None,
           "note": "Own photos are full webcam frames; the CNNs were trained on tight hand crops."}
    save_json(res, RESULTS / "own_eval.json")
    print(res)


if __name__ == "__main__":
    main()
