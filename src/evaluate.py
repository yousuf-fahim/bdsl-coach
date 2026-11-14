"""Phase 5: head-to-head comparison. The TEST split is used exactly once.

For each model: accuracy, macro F1, per-class F1.
  Approach B (landmark MLP) is reported two ways:
    - detected only: images where MediaPipe found a hand
    - end-to-end   : every undetected image counted as an error
  For a fair comparison the CNNs are also scored on that same detected subset.
Size: parameters and checkpoint file size (KB). For approach B the MediaPipe
model file (hand_landmarker.task) is reported too, because the app needs it.
Latency: CPU, batch size 1, torch with 1 thread (closer to a phone/browser),
median of 200 runs after 20 warm-up runs. For approach B also the end-to-end
time per image: MediaPipe detection + normalisation + MLP.

Outputs (with --split test):
  results/comparison.csv, results/per_class_f1.csv, results/test_predictions.npz,
  results/comparison_accuracy.png, results/comparison_size.png, results/comparison_latency.png
With --split val the same is written with a "val_" prefix (used to debug the script).
Run: .venv\\Scripts\\python src\\evaluate.py --split test
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import landmarks as L
from cnn_data import center_crop, load_cache, normalize, to_tensor
from common import BENGALI_FONT, MODELS, PROCESSED, RAW, RESULTS, SEED, load_labels
from models_io import MODEL_FILES, load_cnn, load_mlp

NAMES = {"custom": "Custom CNN", "mobilenet": "MobileNetV3-small", "landmark_mlp": "Landmark MLP"}


def cnn_probs(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.softmax(model(normalize(center_crop(to_tensor(X[i:i + 256])))), 1).numpy())
    return np.concatenate(out)


def median_ms(fn, runs=200, warmup=20):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args()
    pre = "" if args.split == "test" else "val_"
    torch.manual_seed(SEED)
    labels = load_labels()

    imgs, df = load_cache()
    lm = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    # landmarks.npz rows follow the same order as the non-excluded splits.csv rows
    assert (lm["path"] == df["path"].to_numpy()).all()
    sel = (df.split == args.split).to_numpy()
    y = df.label.to_numpy()[sel]
    X_img = imgs[sel]
    X_lm = lm["X"][sel].astype(np.float32)
    detected = lm["n_hands"][sel] > 0
    paths = df.path.to_numpy()[sel]
    print(f"{args.split}: {sel.sum()} images, {detected.sum()} with a detected hand")

    probs = {}
    for name in ("custom", "mobilenet"):
        probs[name] = cnn_probs(load_cnn(name), X_img)
    mlp = load_mlp()
    with torch.no_grad():
        p = torch.softmax(mlp(torch.from_numpy(X_lm)), 1).numpy()
    p[~detected] = 0  # no hand -> no prediction
    probs["landmark_mlp"] = p

    rows, f1_rows = [], {}
    for name, pr in probs.items():
        pred = pr.argmax(1)
        valid = detected if name == "landmark_mlp" else np.ones_like(detected)
        pred_e2e = np.where(valid, pred, -1)  # -1 = "no answer" = wrong
        row = {"model": NAMES[name], "key": name,
               "accuracy_all": accuracy_score(y, pred_e2e),
               "macro_f1_all": f1_score(y, pred_e2e, average="macro", labels=list(range(49))),
               "accuracy_detected_subset": accuracy_score(y[detected], pred[detected]),
               "macro_f1_detected_subset": f1_score(y[detected], pred[detected], average="macro",
                                                    labels=list(range(49)))}
        f1_rows[NAMES[name]] = f1_score(y, pred_e2e, average=None, labels=list(range(49)))
        rows.append(row)

    # ---- size ----
    mp_kb = (MODELS / "mediapipe" / "hand_landmarker.task").stat().st_size / 1024
    models = {"custom": load_cnn("custom"), "mobilenet": load_cnn("mobilenet"), "landmark_mlp": mlp}
    for row in rows:
        m = models[row["key"]]
        row["parameters"] = sum(q.numel() for q in m.parameters())
        row["file_kb"] = MODEL_FILES[row["key"]].stat().st_size / 1024
        row["extra_runtime_model_kb"] = mp_kb if row["key"] == "landmark_mlp" else 0.0

    # ---- latency (batch 1, 1 thread) ----
    torch.set_num_threads(1)
    x_img = normalize(center_crop(to_tensor(X_img[:1])))
    x_lm = torch.from_numpy(X_lm[detected][:1])
    with torch.no_grad():
        for row in rows:
            m = models[row["key"]]
            x = x_lm if row["key"] == "landmark_mlp" else x_img
            row["latency_ms_model_only"] = median_ms(lambda: m(x))
    # end-to-end for approach B: MediaPipe on a prepared (resized + padded) frame + normalise + MLP
    lmk = L._get_landmarker()
    rng = np.random.default_rng(SEED)
    frames = [L.prepare_image(RAW / p) for p in rng.choice(paths, 10, replace=False)]
    k = {"i": 0}

    def e2e():
        f = frames[k["i"] % len(frames)]
        k["i"] += 1
        hands = L.detect(f, lmk)
        feat = L.normalize_hands([(h[0], h[1]) for h in hands[:2]], f.shape[1], f.shape[0])
        with torch.no_grad():
            mlp(torch.from_numpy(feat[None]))
    e2e_ms = median_ms(e2e, runs=200, warmup=10)
    for row in rows:
        row["latency_ms_end_to_end"] = e2e_ms if row["key"] == "landmark_mlp" else np.nan
    torch.set_num_threads(6)

    out = pd.DataFrame(rows)
    out["n_images"] = len(y)
    out["n_detected"] = int(detected.sum())
    out.to_csv(RESULTS / f"{pre}comparison.csv", index=False)
    f1_df = pd.DataFrame(f1_rows)
    f1_df.insert(0, "label", [labels[c]["bangla"] for c in range(49)])
    f1_df.to_csv(RESULTS / f"{pre}per_class_f1.csv", index=False)
    np.savez_compressed(RESULTS / f"{pre}test_predictions.npz" if pre == "" else RESULTS / "val_predictions.npz",
                        y=y, detected=detected, path=paths, **{f"probs_{k}": v for k, v in probs.items()})
    print(out.drop(columns=["key"]).round(4).to_string())

    # ---- charts ----
    plt.rcParams["font.family"] = [BENGALI_FONT, "DejaVu Sans"]
    names = out.model.tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, out.accuracy_all, 0.4, label=f"all {args.split} images (undetected = error)")
    ax.bar(xx + 0.2, out.accuracy_detected_subset, 0.4, label="images where MediaPipe found a hand")
    for i, (a, b) in enumerate(zip(out.accuracy_all, out.accuracy_detected_subset)):
        ax.text(i - 0.2, a + 0.01, f"{a:.1%}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.01, f"{b:.1%}", ha="center", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy on the {args.split} set ({len(y)} images)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{pre}comparison_accuracy.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xx, out.file_kb, label="classifier file")
    ax.bar(xx, out.extra_runtime_model_kb, bottom=out.file_kb, label="+ MediaPipe hand model (needed at runtime)")
    for i, (a, b) in enumerate(zip(out.file_kb, out.extra_runtime_model_kb)):
        ax.text(i, a + b, f"{a:,.0f} KB" + (f"\n(+{b:,.0f} KB)" if b else ""), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels(names)
    ax.set_ylabel("size (KB)")
    ax.set_title("Model size on disk")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{pre}comparison_size.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xx, out.latency_ms_model_only, label="classifier only")
    e = out.latency_ms_end_to_end.fillna(0)
    ax.bar(xx, e - out.latency_ms_model_only.where(e > 0, 0), bottom=out.latency_ms_model_only,
           alpha=0.5, label="+ MediaPipe detection (approach B end-to-end)")
    for i, (a, b) in enumerate(zip(out.latency_ms_model_only, e)):
        ax.text(i, max(a, b), f"{a:.2f} ms" + (f"\n(e2e {b:.1f} ms)" if b else ""), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels(names)
    ax.set_ylabel("median ms per image (CPU, batch 1, 1 thread)")
    ax.set_title("Inference latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{pre}comparison_latency.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
