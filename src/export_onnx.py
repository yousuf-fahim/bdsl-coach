"""Phase 8: export the landmark MLP to ONNX and build parity fixtures for the web demo.

1. models/landmark_mlp.onnx (input "features" [batch, 131] -> output "logits" [batch, 49])
2. Check ONNX Runtime (Python) == PyTorch on 20 saved validation samples.
3. web/tests/fixtures/parity_samples.json: for those 20 samples, the raw MediaPipe
   landmarks + handedness + image size, the Python feature vector, and the PyTorch
   logits. The JS test feeds the same raw landmarks through web/src/normalize.js and
   the ONNX model in a real browser, and compares.
4. web/tests/fixtures/images/*.png: the same 20 images, resized + padded exactly as
   MediaPipe saw them in Python (gitignored: derived from the dataset), so the
   browser test can also compare MediaPipe JS vs. MediaPipe Python end-to-end.
5. Copies the ONNX model, the MediaPipe .task file and labels.json into web/public/models/.

Outputs: models/landmark_mlp.onnx, results/onnx_parity.json, web fixtures, web/public/models/*
Run: .venv\\Scripts\\python src\\export_onnx.py
"""
import json
import shutil

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

import landmarks as L
from common import MODELS, PROCESSED, RAW, RESULTS, ROOT, SEED, save_json
from models_io import load_mlp

WEB = ROOT / "web"


def main():
    mlp = load_mlp()
    onnx_path = MODELS / "landmark_mlp.onnx"
    torch.onnx.export(mlp, torch.zeros(1, L.FEAT_DIM), str(onnx_path), input_names=["features"],
                      output_names=["logits"], dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    d = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    rng = np.random.default_rng(SEED)
    val = np.flatnonzero(d["split"] == "val")
    # 20 samples: 8 with two hands, 8 with one hand, 4 with none (edge cases matter)
    pick = np.concatenate([
        rng.choice(val[d["n_hands"][val] == 2], 8, replace=False),
        rng.choice(val[d["n_hands"][val] == 1], 8, replace=False),
        rng.choice(val[d["n_hands"][val] == 0], 4, replace=False)])
    X = d["X"][pick].astype(np.float32)
    with torch.no_grad():
        torch_logits = mlp(torch.from_numpy(X)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"features": X})[0]
    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    same_argmax = bool((torch_logits.argmax(1) == onnx_logits.argmax(1)).all())
    print(f"ONNX vs PyTorch on 20 samples: max |diff| = {max_diff:.2e}, same predictions: {same_argmax}")

    fx = WEB / "tests" / "fixtures"
    (fx / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for j, i in enumerate(pick):
        n = int(d["n_hands"][i])
        img_name = f"sample_{j:02d}.png"
        Image.fromarray(L.prepare_image(RAW / d["path"][i])).save(fx / "images" / img_name)
        samples.append({
            "id": j, "path": str(d["path"][i]), "label": int(d["y"][i]), "image": img_name,
            "width": int(d["wh"][i][0]), "height": int(d["wh"][i][1]),
            "hands": [{"side": str(d["sides"][i][k]), "landmarks": d["raw"][i][k].astype(float).tolist()}
                      for k in range(n)],
            "py_features": X[j].astype(float).tolist(),
            "py_logits": torch_logits[j].astype(float).tolist(),
        })
    (fx / "parity_samples.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")

    pub = WEB / "public" / "models"
    pub.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, pub / "landmark_mlp.onnx")
    shutil.copy(MODELS / "mediapipe" / "hand_landmarker.task", pub / "hand_landmarker.task")
    labels = json.loads((PROCESSED / "labels.json").read_text(encoding="utf-8"))
    cfg = {"labels": labels["classes"], "label_status": labels["status"],
           "min_detection_confidence": L.MIN_DET_CONF, "min_presence_confidence": L.MIN_PRESENCE_CONF,
           "feature_dim": L.FEAT_DIM}
    (pub / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    save_json({"samples": len(pick), "max_abs_logit_diff": max_diff, "same_predictions": same_argmax,
               "onnx_kb": onnx_path.stat().st_size / 1024, "opset": 17}, RESULTS / "onnx_parity.json")


if __name__ == "__main__":
    main()
