"""Phase 2 report: detection rates and visual checks from data/processed/landmarks.npz.

Outputs
  results/landmark_detection.json            overall / per split / per class rates
  results/landmark_detection_per_class.png   per-class detection rate (and 2-hand share)
  results/landmark_failures.png              undetected examples from the worst classes
  results/landmark_overlays.png              20 random images with landmarks drawn
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BENGALI_FONT, PROCESSED, RAW, RESULTS, SEED, load_labels, save_json

# MediaPipe hand skeleton (pairs of landmark indices)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def make_report():
    import landmarks as L  # local import avoids a circular import at module load
    labels = load_labels()
    d = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    df = pd.DataFrame({"label": d["y"], "split": d["split"], "path": d["path"], "n_hands": d["n_hands"]})
    df["detected"] = df.n_hands > 0

    per_class = df.groupby("label").agg(rate=("detected", "mean"), two=("n_hands", lambda s: (s == 2).mean()),
                                        n=("detected", "size"))
    worst = per_class.sort_values("rate").head(8)
    save_json({
        "overall_detection_rate": float(df.detected.mean()),
        "two_hand_share": float((df.n_hands == 2).mean()),
        "per_split": df.groupby("split")["detected"].mean().round(4).to_dict(),
        "per_class": {labels[c]["bangla"]: round(float(r), 4) for c, r in per_class.rate.items()},
        "worst_classes": {labels[c]["bangla"]: round(float(r), 4) for c, r in worst.rate.items()},
        "n_undetected": int((~df.detected).sum()),
        "policy_for_undetected": ("Excluded from MLP training/validation; at test time reported two ways: "
                                  "accuracy on detected images only, and with every undetected image counted as an error."),
    }, RESULTS / "landmark_detection.json")
    print("overall detection:", round(df.detected.mean(), 4), "| worst:",
          {labels[c]["bangla"]: round(r, 3) for c, r in worst.rate.items()})

    plt.rcParams["font.family"] = [BENGALI_FONT, "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(49)
    ax.bar(x, per_class.rate, color="#4C72B0", label="any hand detected")
    ax.bar(x, per_class.two, color="#DD8452", alpha=0.8, label="two hands detected")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c]["display"] for c in per_class.index], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.axhline(df.detected.mean(), color="gray", ls="--", lw=1)
    ax.set_ylabel("share of images")
    ax.set_title(f"MediaPipe detection rate per class (overall {df.detected.mean():.1%})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "landmark_detection_per_class.png", dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(SEED)

    # Failures from the 4 worst classes (4 each)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for r, c in enumerate(worst.index[:4]):
        miss = df[(df.label == c) & ~df.detected]
        picks = miss.sample(min(4, len(miss)), random_state=SEED) if len(miss) else miss
        for j in range(4):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(picks):
                ax.imshow(L.prepare_image(RAW / picks.path.iloc[j]))
                ax.set_title(f"{labels[c]['display']} (rate {worst.rate[c]:.0%})", fontsize=9)
    fig.suptitle("Undetected examples from the worst classes (padded input as MediaPipe saw it)")
    fig.tight_layout()
    fig.savefig(RESULTS / "landmark_failures.png", dpi=100)
    plt.close(fig)

    # 20 random detected overlays
    det_idx = np.flatnonzero(df.detected.to_numpy())
    pick = rng.choice(det_idx, 20, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(13, 11))
    for ax, i in zip(axes.flat, pick):
        img = L.prepare_image(RAW / df.path[i])
        h, w = img.shape[:2]
        ax.imshow(img)
        for k in range(int(d["n_hands"][i])):
            p = d["raw"][i][k]
            col = "lime" if k == 0 else "cyan"
            for a, b in CONNECTIONS:
                ax.plot([p[a, 0] * w, p[b, 0] * w], [p[a, 1] * h, p[b, 1] * h], color=col, lw=1)
            ax.scatter(p[:, 0] * w, p[:, 1] * h, s=6, color="red", zorder=3)
        ax.set_title(f"{labels[int(d['y'][i])]['display']}  hands={int(d['n_hands'][i])}", fontsize=9)
        ax.axis("off")
    fig.suptitle("20 random images with MediaPipe landmarks")
    fig.tight_layout()
    fig.savefig(RESULTS / "landmark_overlays.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    make_report()
