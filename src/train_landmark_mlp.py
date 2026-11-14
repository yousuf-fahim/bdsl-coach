"""Phase 3: approach B, landmark features -> classifier. VALIDATION SET ONLY.

1. Classical baselines on the 131-d landmark vectors: logistic regression, random forest.
2. A small PyTorch MLP (131 -> 256 -> 128 -> 49, BatchNorm + ReLU + Dropout),
   AdamW, early stopping on validation loss.
3. Landmark-level augmentation (rotation, scale jitter, Gaussian noise), compared
   with no augmentation. The better of the two (by val accuracy) is saved.

Images where MediaPipe found no hand have an all-zero feature vector. They are
left out of training. On validation we report accuracy two ways:
detected-only, and "end-to-end" where each undetected image counts as an error.

Outputs: models/landmark_mlp.pt, results/landmark_mlp_val.json,
results/landmark_mlp_curves.png, results/landmark_mlp_history.csv
Run: .venv\\Scripts\\python src\\train_landmark_mlp.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, PROCESSED, RESULTS, SEED, save_json, set_seed

N_CLASSES = 49
EPOCHS, PATIENCE, BATCH, LR, WD = 300, 30, 128, 1e-3, 1e-4


class LandmarkMLP(nn.Module):
    """input -> 256 -> 128 -> classes. BatchNorm steadies training; Dropout fights overfitting."""

    # Module-level defaults so capacity_check() can try other sizes.
    SIZES = {"h1": 256, "h2": 128, "p": 0.3}

    def __init__(self, d_in=131, n_classes=N_CLASSES, h1=None, h2=None, p=None):
        h1 = h1 or self.SIZES["h1"]
        h2 = h2 or self.SIZES["h2"]
        p = self.SIZES["p"] if p is None else p
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def augment(x, rng):
    """Landmark augmentation on a batch of normalised feature vectors (numpy, float32).

    - rotation: the whole sign rotated by up to +-15 degrees in the image plane
      (a slightly tilted camera or hand)
    - scale jitter: x and y stretched independently by 0.9-1.1
      (different camera angles / aspect ratios)
    - Gaussian noise: sigma 0.02 on every coordinate (MediaPipe's own jitter)
    Only the coordinates of present hands are changed; the presence flags are not.
    """
    x = x.copy()
    n = len(x)
    theta = np.deg2rad(rng.uniform(-15, 15, n))
    c, s = np.cos(theta), np.sin(theta)
    sx, sy = rng.uniform(0.9, 1.1, n), rng.uniform(0.9, 1.1, n)
    for slot in (0, 1):
        pts = x[:, slot * 63:(slot + 1) * 63].reshape(n, 21, 3)
        px, py = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = (c[:, None] * px - s[:, None] * py) * sx[:, None]
        pts[:, :, 1] = (s[:, None] * px + c[:, None] * py) * sy[:, None]
        pts += rng.normal(0, 0.02, pts.shape).astype(np.float32)
        pts *= x[:, 126 + slot][:, None, None]  # keep missing hands at exactly zero
        x[:, slot * 63:(slot + 1) * 63] = pts.reshape(n, 63)
    ox, oy = x[:, 128].copy(), x[:, 129].copy()
    x[:, 128] = (c * ox - s * oy) * sx
    x[:, 129] = (s * ox + c * oy) * sy
    return x


def train_mlp(Xtr, ytr, Xva, yva, use_aug, log_name):
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = LandmarkMLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    best = {"val_loss": np.inf}
    hist, bad = [], 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        tl, tc = 0.0, 0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue  # BatchNorm needs at least 2 samples
            xb = augment(Xtr[b], rng) if use_aug else Xtr[b]
            xb, yb = torch.from_numpy(xb), torch.from_numpy(ytr[b])
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == yb).sum().item()
        model.eval()
        with torch.no_grad():
            vo = model(Xva_t)
            vl = loss_fn(vo, yva_t).item()
            va = (vo.argmax(1) == yva_t).float().mean().item()
        hist.append({"run": log_name, "epoch": ep, "train_loss": tl / len(Xtr), "train_acc": tc / len(Xtr),
                     "val_loss": vl, "val_acc": va})
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "val_acc": va, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best["state"])
    best["seconds"] = time.time() - t0
    best["epochs_run"] = ep
    print(f"{log_name}: best epoch {best['epoch']} val_acc {best['val_acc']:.4f} ({best['seconds']:.0f}s)")
    return model, best, hist


def capacity_check(Xtr, ytr, Xva, yva):
    """Is the MLP too small? Try wider nets / less dropout (validation only)."""
    out = []
    for h1, h2, p in ((256, 128, 0.3), (256, 128, 0.1), (512, 256, 0.2), (512, 256, 0.1)):
        LandmarkMLP.SIZES = {"h1": h1, "h2": h2, "p": p}
        m, best, _ = train_mlp(Xtr, ytr, Xva, yva, True, f"capacity_{h1}_{h2}_p{p}")
        train_acc = float((predict(m, Xtr) == ytr).mean())
        out.append({"h1": h1, "h2": h2, "dropout": p, "val_acc": best["val_acc"],
                    "train_acc_eval_mode": train_acc, "parameters": sum(q.numel() for q in m.parameters())})
    LandmarkMLP.SIZES = {"h1": 256, "h2": 128, "p": 0.3}
    save_json(out, RESULTS / "landmark_mlp_capacity.json")


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).argmax(1).numpy()


def main():
    set_seed(SEED)
    d = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    X, y, split, nh = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"], d["n_hands"]
    det = nh > 0
    tr, va = (split == "train") & det, (split == "val") & det
    va_all = split == "val"
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train {tr.sum()} | val {va.sum()} detected of {va_all.sum()}")

    def scores(pred, name):
        acc_det = float((pred == yva).mean())
        # end-to-end: undetected val images are counted as wrong
        acc_e2e = float((pred == yva).sum() / va_all.sum())
        return {"model": name, "val_acc_detected": acc_det, "val_acc_end_to_end": acc_e2e,
                "val_macro_f1_detected": float(f1_score(yva, pred, average="macro"))}

    rows = []
    t = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr, ytr)
    rows.append(scores(lr.predict(Xva), "logistic_regression") | {"train_seconds": time.time() - t})
    t = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    rows.append(scores(rf.predict(Xva), "random_forest") | {"train_seconds": time.time() - t})

    runs = {}
    hist_all = []
    for use_aug, name in ((False, "mlp_no_aug"), (True, "mlp_aug")):
        model, best, hist = train_mlp(Xtr, ytr, Xva, yva, use_aug, name)
        runs[name] = (model, best)
        hist_all += hist
        rows.append(scores(predict(model, Xva), name) |
                    {"train_seconds": best["seconds"], "best_epoch": best["epoch"], "epochs_run": best["epochs_run"]})
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    best_name = max(("mlp_no_aug", "mlp_aug"), key=lambda n: runs[n][1]["val_acc"])
    model = runs[best_name][0]
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"state_dict": model.state_dict(), "d_in": X.shape[1], "n_classes": N_CLASSES,
                "h1": 256, "h2": 128, "p": 0.3, "variant": best_name}, MODELS / "landmark_mlp.pt")
    hist_df = pd.DataFrame(hist_all)
    hist_df.to_csv(RESULTS / "landmark_mlp_history.csv", index=False)
    save_json({"results": rows, "selected": best_name, "mlp_parameters": int(n_params),
               "train_images_detected": int(tr.sum()), "val_images_detected": int(va.sum()),
               "val_images_total": int(va_all.sum()),
               "hyperparameters": {"epochs_max": EPOCHS, "patience": PATIENCE, "batch": BATCH,
                                   "lr": LR, "weight_decay": WD, "dropout": 0.3},
               "class_weights": "not used: class sizes differ by at most 8% (results/audit_summary.json)"},
              RESULTS / "landmark_mlp_val.json")
    print("selected:", best_name, "| parameters:", n_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, ls in (("mlp_no_aug", "-"), ("mlp_aug", "--")):
        h = hist_df[hist_df.run == name]
        axes[0].plot(h.epoch, h.train_loss, ls, color="tab:blue", label=f"{name} train")
        axes[0].plot(h.epoch, h.val_loss, ls, color="tab:orange", label=f"{name} val")
        axes[1].plot(h.epoch, h.train_acc, ls, color="tab:blue", label=f"{name} train")
        axes[1].plot(h.epoch, h.val_acc, ls, color="tab:orange", label=f"{name} val")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_ylabel("accuracy (detected images)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.suptitle("Landmark MLP learning curves (solid: no augmentation, dashed: augmentation)")
    fig.tight_layout()
    fig.savefig(RESULTS / "landmark_mlp_curves.png", dpi=130)
    plt.close(fig)

    capacity_check(Xtr, ytr, Xva, yva)


if __name__ == "__main__":
    main()
