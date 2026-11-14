"""Phase 4: approach A, CNNs on raw images (128x128). VALIDATION SET ONLY.

Models
  custom     : small CNN from scratch, 4 conv blocks (conv-BN-ReLU x2 + maxpool)
  mobilenet  : torchvision MobileNetV3-small pretrained on ImageNet, new 49-class head,
               all layers fine-tuned

Same splits (data/processed/splits.csv) and seed as approach B. AdamW, cosine
learning-rate schedule, early stopping on validation loss. Before the full
run, --estimate times a few batches and prints the expected minutes.

Outputs: models/cnn_<name>.pt, results/cnn_<name>_history.csv,
results/cnn_val.json (merged across models), results/cnn_curves.png
Run: .venv\\Scripts\\python src\\train_cnn.py --model custom [--epochs 25] [--estimate]
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from cnn_data import augment, center_crop, load_cache, normalize, to_tensor
from common import MODELS, RESULTS, SEED, save_json, set_seed

N_CLASSES = 49
BATCH = 64


class SmallCNN(nn.Module):
    """4 blocks of (conv3x3-BN-ReLU) x2 + maxpool: 128 -> 64 -> 32 -> 16 -> 8, then global average pool."""

    def __init__(self, n_classes=N_CLASSES, widths=(32, 64, 128, 256), p=0.3):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(name):
    if name == "custom":
        return SmallCNN(), 1e-3
    if name == "mobilenet":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, N_CLASSES)
        return m, 5e-4  # smaller LR: don't wreck the pretrained features
    raise ValueError(name)


def evaluate(model, X, y):
    model.eval()
    preds, loss = [], 0.0
    lf = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = normalize(center_crop(to_tensor(X[i:i + 256])))
            out = model(xb)
            loss += lf(out, torch.from_numpy(y[i:i + 256])).item()
            preds.append(out.argmax(1).numpy())
    p = np.concatenate(preds)
    return loss / len(X), float((p == y).mean()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["custom", "mobilenet"], required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--estimate", action="store_true", help="time 20 batches and exit")
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(6)
    imgs, df = load_cache()
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
    Xtr, ytr = imgs[tr], df.label.to_numpy()[tr].astype(np.int64)
    Xva, yva = imgs[va], df.label.to_numpy()[va].astype(np.int64)
    model, lr = build(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    steps = int(np.ceil(len(Xtr) / BATCH))
    print(f"{args.model}: {n_params:,} params | train {len(Xtr)} | val {len(Xva)} | {steps} steps/epoch", flush=True)

    if args.estimate:
        model.train()
        t0 = time.time()
        for i in range(20):
            b = rng.choice(len(Xtr), BATCH, replace=False)
            out = model(normalize(augment(to_tensor(Xtr[b]), gen)))
            loss = loss_fn(out, torch.from_numpy(ytr[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        per_step = (time.time() - t0) / 20
        t0 = time.time()
        evaluate(model, Xva[:512], yva[:512])
        val_s = (time.time() - t0) * len(Xva) / 512
        epoch_min = (per_step * steps + val_s) / 60
        est = {"model": args.model, "sec_per_step": round(per_step, 3), "min_per_epoch": round(epoch_min, 2),
               "min_for_max_epochs": round(epoch_min * args.epochs, 1), "max_epochs": args.epochs}
        print(json.dumps(est), flush=True)
        f = RESULTS / "cnn_time_estimates.json"
        allest = json.loads(f.read_text()) if f.exists() else {}
        allest[args.model] = est
        save_json(allest, f)
        return

    hist, best, bad = [], {"val_loss": np.inf}, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(Xtr))
        tl, tc = 0.0, 0
        for i in range(0, len(order), BATCH):
            b = np.sort(order[i:i + BATCH])
            xb = normalize(augment(to_tensor(Xtr[b]), gen))
            yb = torch.from_numpy(ytr[b])
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == yb).sum().item()
        sched.step()
        vl, va_acc, _ = evaluate(model, Xva, yva)
        row = {"epoch": ep, "train_loss": tl / len(Xtr), "train_acc": tc / len(Xtr), "val_loss": vl,
               "val_acc": va_acc, "epoch_seconds": time.time() - t0}
        hist.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        pd.DataFrame(hist).to_csv(RESULTS / f"cnn_{args.model}_history.csv", index=False)
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "val_acc": va_acc, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
            torch.save({"state_dict": best["state"], "model": args.model, "epoch": ep}, MODELS / f"cnn_{args.model}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping", flush=True)
                break
    train_minutes = (time.time() - t_start) / 60
    model.load_state_dict(best["state"])
    _, acc, pred = evaluate(model, Xva, yva)
    f = RESULTS / "cnn_val.json"
    allres = json.loads(f.read_text()) if f.exists() else {}
    allres[args.model] = {"val_acc": acc, "val_macro_f1": float(f1_score(yva, pred, average="macro")),
                          "best_epoch": best["epoch"], "epochs_run": len(hist), "max_epochs": args.epochs,
                          "parameters": int(n_params), "train_minutes": round(train_minutes, 1),
                          "image_size": 128, "batch": BATCH, "lr": lr}
    save_json(allres, f)
    print("done", allres[args.model], flush=True)
    plot_curves()


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, col in (("custom", "tab:blue"), ("mobilenet", "tab:green")):
        f = RESULTS / f"cnn_{name}_history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        axes[0].plot(h.epoch, h.train_loss, "-", color=col, label=f"{name} train")
        axes[0].plot(h.epoch, h.val_loss, "--", color=col, label=f"{name} val")
        axes[1].plot(h.epoch, h.train_acc, "-", color=col, label=f"{name} train")
        axes[1].plot(h.epoch, h.val_acc, "--", color=col, label=f"{name} val")
    axes[0].set_ylabel("loss (train includes label smoothing)")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.suptitle("CNN learning curves (solid: train, dashed: validation)")
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
