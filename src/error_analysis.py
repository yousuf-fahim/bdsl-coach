"""Phase 7: error analysis on the saved test predictions (results/test_predictions.npz).

No new test-set evaluation happens here: we analyse the predictions that
evaluate.py produced in its single test run (Grad-CAM re-runs the chosen CNN on
20 of those test images only to visualise attention, not to score anything).

  1. Confusion matrices for the best model of each approach. "Best" is chosen
     by VALIDATION accuracy (results/cnn_val.json), not by test accuracy.
  2. Top 10 most-confused sign pairs per approach (A->B and B->A counted together).
  3. Top 5 pairs: example images side by side + the distance between the two
     classes' mean landmark vectors (train set), compared with all class pairs.
  4. Grad-CAM on the best CNN for 10 correct and 10 wrong test predictions.
  5. Per-signer accuracy is impossible (no signer IDs); as a proxy we report
     accuracy per capture block (15 consecutive images of one class).

Outputs: results/confusion_<model>.png, results/confused_pairs.csv,
results/confused_pairs_examples.png, results/landmark_class_distances.json,
results/gradcam.png, results/per_block_accuracy.csv/.png, results/error_analysis_numbers.json
Run: .venv\\Scripts\\python src\\error_analysis.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix

from cnn_data import center_crop, load_cache, normalize, to_tensor
from common import BENGALI_FONT, PROCESSED, RAW, RESULTS, SEED, load_labels, save_json
from evaluate import NAMES
from models_io import load_cnn


def top_pairs(cm, k=10):
    """Most-confused unordered pairs: cm[a,b] + cm[b,a], a != b."""
    pairs = []
    for a in range(cm.shape[0]):
        for b in range(a + 1, cm.shape[0]):
            n = cm[a, b] + cm[b, a]
            if n:
                pairs.append((int(n), a, b, int(cm[a, b]), int(cm[b, a])))
    return sorted(pairs, reverse=True)[:k]


def gradcam(model, layer, x, cls):
    """Grad-CAM: weight the layer's feature maps by the mean gradient of the class score."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    out[0, cls].backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), int(out.argmax(1))


def main():
    labels = load_labels()
    disp = [l["display"] for l in labels]
    plt.rcParams["font.family"] = [BENGALI_FONT, "DejaVu Sans"]
    P = np.load(RESULTS / "test_predictions.npz", allow_pickle=True)
    y, detected, paths = P["y"], P["detected"], P["path"]
    val = json.loads((RESULTS / "cnn_val.json").read_text())
    best_cnn = max(val, key=lambda k: val[k]["val_acc"])
    chosen = {"A": best_cnn, "B": "landmark_mlp"}
    numbers = {"best_cnn_by_val": best_cnn}

    # ---- 1 + 2: confusion matrices and top pairs ----
    rows = []
    for approach, key in chosen.items():
        pred = P[f"probs_{key}"].argmax(1)
        mask = detected if key == "landmark_mlp" else np.ones_like(detected)  # B: detected images only
        cm = confusion_matrix(y[mask], pred[mask], labels=list(range(49)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(13, 12))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(49))
        ax.set_yticks(range(49))
        ax.set_xticklabels(disp, fontsize=8)
        ax.set_yticklabels(disp, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        sub = " (images with a detected hand)" if key == "landmark_mlp" else ""
        ax.set_title(f"Confusion matrix, {NAMES[key]}, test set{sub}; rows normalised")
        fig.colorbar(im, fraction=0.03)
        fig.tight_layout()
        fig.savefig(RESULTS / f"confusion_{key}.png", dpi=110)
        plt.close(fig)
        for rank, (n, a, b, ab, ba) in enumerate(top_pairs(cm), 1):
            rows.append({"approach": approach, "model": NAMES[key], "rank": rank, "class_a": disp[a],
                         "class_b": disp[b], "a_idx": a, "b_idx": b, "a_predicted_as_b": ab,
                         "b_predicted_as_a": ba, "total": n})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(RESULTS / "confused_pairs.csv", index=False)
    print(pairs.to_string())

    # ---- 3: landmark distances between class means (train set, detected) ----
    lm = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    tr = (lm["split"] == "train") & (lm["n_hands"] > 0)
    X, Y = lm["X"][tr], lm["y"][tr]
    means = np.stack([X[Y == c].mean(0) for c in range(49)])
    D = np.linalg.norm(means[:, None] - means[None], axis=2)
    iu = np.triu_indices(49, 1)
    all_d = D[iu]
    dist_rows = []
    for _, r in pairs.drop_duplicates(["a_idx", "b_idx"]).iterrows():
        d = D[r.a_idx, r.b_idx]
        dist_rows.append({"pair": f"{r.class_a}-{r.class_b}", "approach_listing": r.approach,
                          "mean_landmark_distance": float(d),
                          "percentile_among_all_49x48_2_pairs": float((all_d < d).mean() * 100)})
    save_json({"median_distance_all_pairs": float(np.median(all_d)),
               "min_distance_all_pairs": float(all_d.min()), "pairs": dist_rows},
              RESULTS / "landmark_class_distances.json")

    # example images: top 5 pairs of approach B and top 5 of approach A (deduplicated)
    top5 = pd.concat([pairs[pairs.approach == "B"].head(5), pairs[pairs.approach == "A"].head(5)]) \
        .drop_duplicates(["a_idx", "b_idx"]).head(8)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(top5), 6, figsize=(12, 2.3 * len(top5)))
    for r, (_, p) in enumerate(top5.iterrows()):
        for half, c in enumerate((p.a_idx, p.b_idx)):
            cand = paths[y == c]
            for j, path in enumerate(rng.choice(cand, 3, replace=False)):
                ax = axes[r, half * 3 + j]
                ax.imshow(Image.open(RAW / path).convert("RGB"))
                ax.axis("off")
                if j == 0:
                    ax.set_title(disp[c], fontsize=14)
        d = D[p.a_idx, p.b_idx]
        axes[r, 0].text(-0.15, 0.5, f"{p.approach} #{p['rank']}\n{p.total} errors\nd={d:.2f}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle(f"Most-confused pairs (test images). d = distance between class-mean landmark vectors; "
                 f"median over all pairs = {np.median(all_d):.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "confused_pairs_examples.png", dpi=90)
    plt.close(fig)

    # ---- 4: Grad-CAM on the best CNN ----
    model = load_cnn(best_cnn)
    layer = model.features[-2] if best_cnn == "custom" else model.features[-1]  # last conv stage
    imgs, df = load_cache()
    idx_of = {p: i for i, p in enumerate(df.path)}
    pred_cnn = P[f"probs_{best_cnn}"].argmax(1)
    correct = rng.choice(np.flatnonzero(pred_cnn == y), 10, replace=False)
    wrong = rng.choice(np.flatnonzero(pred_cnn != y), 10, replace=False)
    fig, axes = plt.subplots(4, 10, figsize=(20, 9))
    for row0, group, title in ((0, correct, "correct"), (2, wrong, "wrong")):
        for j, t in enumerate(group):
            x8 = imgs[idx_of[paths[t]]][None]
            x = normalize(center_crop(to_tensor(x8)))
            cam, pr = gradcam(model, layer, x, int(pred_cnn[t]))
            show = center_crop(to_tensor(x8))[0].permute(1, 2, 0).numpy()
            axes[row0, j].imshow(show)
            axes[row0, j].set_title(f"true {disp[y[t]]} / pred {disp[pr]}", fontsize=10)
            axes[row0 + 1, j].imshow(show)
            axes[row0 + 1, j].imshow(cam, cmap="jet", alpha=0.45)
            for a in (axes[row0, j], axes[row0 + 1, j]):
                a.axis("off")
        axes[row0, 0].text(-0.2, 0.0, title, transform=axes[row0, 0].transAxes, rotation=90, fontsize=14,
                           va="center", ha="right")
    fig.suptitle(f"Grad-CAM, {NAMES[best_cnn]}: where the network looks (red = most important for the predicted class)")
    fig.tight_layout()
    fig.savefig(RESULTS / "gradcam.png", dpi=80)
    plt.close(fig)

    # ---- 5: per capture block (signer proxy) ----
    splits = pd.read_csv(PROCESSED / "splits.csv").set_index("path")
    blocks = splits.loc[paths, "group"].to_numpy()
    rows = []
    for key in ("custom", "mobilenet", "landmark_mlp"):
        pr = P[f"probs_{key}"].argmax(1)
        ok = (pr == y) & (detected if key == "landmark_mlp" else True)
        s = pd.DataFrame({"block": blocks, "ok": ok}).groupby("block")["ok"].agg(["mean", "size"])
        for b, r in s.iterrows():
            rows.append({"model": NAMES[key], "block": int(b), "accuracy": r["mean"], "images": int(r["size"])})
    blk = pd.DataFrame(rows)
    blk.to_csv(RESULTS / "per_block_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [blk[blk.model == NAMES[k]].accuracy.to_numpy() for k in ("custom", "mobilenet", "landmark_mlp")]
    ax.boxplot(data, tick_labels=[NAMES[k] for k in ("custom", "mobilenet", "landmark_mlp")])
    ax.set_ylabel("accuracy per capture block (end-to-end)")
    ax.set_title("Spread of accuracy across test capture blocks (proxy for signer/session)")
    fig.tight_layout()
    fig.savefig(RESULTS / "per_block_accuracy.png", dpi=120)
    plt.close(fig)
    numbers["per_block_summary"] = blk.groupby("model")["accuracy"].describe().round(3).to_dict(orient="index")

    # Are the top confusions concentrated in a few capture blocks (one signer/session)
    # or spread out (genuinely similar signs)?
    conc = []
    for _, r in pairs[pairs["rank"] <= 5].iterrows():
        key = chosen[r.approach]
        pr = P[f"probs_{key}"].argmax(1)
        m = ((y == r.a_idx) & (pr == r.b_idx)) | ((y == r.b_idx) & (pr == r.a_idx))
        counts = pd.Series(blocks[m]).value_counts()
        conc.append({"approach": r.approach, "pair": f"{r.class_a}-{r.class_b}", "errors": int(m.sum()),
                     "blocks_with_errors": int(len(counts)),
                     "share_in_top_block": round(float(counts.iloc[0] / m.sum()), 3) if m.sum() else None,
                     "test_blocks_of_the_two_classes": int(len(set(blocks[(y == r.a_idx) | (y == r.b_idx)])))})
    numbers["top_pair_error_concentration"] = conc
    numbers["n_test_blocks"] = int(blk.block.nunique())
    save_json(numbers, RESULTS / "error_analysis_numbers.json")
    print(json.dumps(numbers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
