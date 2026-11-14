"""Phase 6: robustness of the three models under image perturbations (test split).

Both approaches see the same perturbed picture:
  original -> shrink to 640 px longest side -> PERTURB -> (A) letterbox 144 + centre crop 128 -> CNN
                                                        -> (B) pad 25% -> MediaPipe -> normalise -> MLP
Severity 0 ("clean") goes through the same path, so every curve starts from a
comparable baseline. Approach B is scored end-to-end (no hand found = error).

Perturbations, 5 severity levels each:
  gaussian_noise  pixel noise sigma 10, 20, 35, 50, 70 (0-255 scale)
  blur            Gaussian blur sigma 0.5, 1, 1.5, 2, 3 % of the image's longest side
  darker          brightness x0.8, 0.6, 0.45, 0.3, 0.2
  brighter        brightness x1.3, 1.6, 1.9, 2.2, 2.6
  rotation        3, 6, 9, 12, 15 degrees (random direction)
  occlusion       grey square covering 5, 10, 15, 20, 30 % of the image, random position
  jpeg            JPEG quality 50, 30, 20, 10, 5
Randomness (noise, rotation direction, occlusion position) is seeded per image.

Outputs: results/robustness.csv, results/robustness_<perturbation>.png, results/robustness_all.png
Run: .venv\\Scripts\\python src\\robustness.py
"""
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

import landmarks as L
from cnn_data import CACHE, center_crop, normalize, to_tensor
from common import PROCESSED, RAW, RESULTS, SEED
from evaluate import NAMES
from models_io import load_cnn, load_mlp

LEVELS = {
    "gaussian_noise": [10, 20, 35, 50, 70],
    "blur": [0.5, 1, 1.5, 2, 3],
    "darker": [0.8, 0.6, 0.45, 0.3, 0.2],
    "brighter": [1.3, 1.6, 1.9, 2.2, 2.6],
    "rotation": [3, 6, 9, 12, 15],
    "occlusion": [5, 10, 15, 20, 30],
    "jpeg": [50, 30, 20, 10, 5],
}


def load_640(path):
    im = Image.open(RAW / path)
    im.draft("RGB", (L.MAX_SIDE, L.MAX_SIDE))  # fast JPEG downscale while decoding
    im = im.convert("RGB")
    im.thumbnail((L.MAX_SIDE, L.MAX_SIDE))
    return im


def perturb(im, kind, level, seed):
    """Apply one perturbation to a PIL RGB image. level=None means clean."""
    if level is None:
        return im
    rng = np.random.default_rng(seed)
    if kind == "gaussian_noise":
        a = np.asarray(im).astype(np.float32) + rng.normal(0, level, (im.height, im.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(level / 100 * max(im.size)))
    if kind in ("darker", "brighter"):
        return ImageEnhance.Brightness(im).enhance(level)
    if kind == "rotation":
        return im.rotate(level * rng.choice([-1, 1]), resample=Image.BILINEAR, expand=False)
    if kind == "occlusion":
        side = int(np.sqrt(level / 100 * im.width * im.height))
        side = min(side, im.width, im.height)
        x0 = int(rng.integers(0, im.width - side + 1))
        y0 = int(rng.integers(0, im.height - side + 1))
        a = np.asarray(im).copy()
        a[y0:y0 + side, x0:x0 + side] = 128
        return Image.fromarray(a)
    if kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(kind)


def to_cnn_input(im):
    """Letterbox to CACHE x CACHE (same as cnn_data.letterbox, from a PIL image)."""
    im = im.copy()
    im.thumbnail((CACHE, CACHE), Image.BILINEAR)
    canvas = Image.new("RGB", (CACHE, CACHE))
    canvas.paste(im, ((CACHE - im.width) // 2, (CACHE - im.height) // 2))
    return np.asarray(canvas)


def to_landmark_features(im):
    """Pad 25% black border, run MediaPipe, return (131-d features, detected?)."""
    w, h = im.size
    pw, ph = int(round(w * L.PAD_FRAC)), int(round(h * L.PAD_FRAC))
    canvas = Image.new("RGB", (w + 2 * pw, h + 2 * ph))
    canvas.paste(im, (pw, ph))
    rgb = np.asarray(canvas)
    hands = L.detect(rgb)
    feat = L.normalize_hands([(x[0], x[1]) for x in hands[:2]], rgb.shape[1], rgb.shape[0])
    return feat, len(hands) > 0


def process(args):
    """Worker: one image, one (kind, level) -> CNN input + landmark features."""
    path, i, kind, level = args
    im = perturb(load_640(path), kind, level, seed=SEED * 100000 + i)
    feat, det = to_landmark_features(im)
    return to_cnn_input(im), feat, det


def run_combo(jobs, workers=6, retries=3):
    """Process all images of one (perturbation, level) in a fresh worker pool.

    A long-lived pool hung once during development (idle workers, no error), so
    each combination gets its own pool with a timeout; on a hang the pool is
    terminated (workers killed) and retried, then we fall back to one process.
    """
    import multiprocessing as mp
    timeout = 60 + 0.5 * len(jobs)  # generous: normally ~0.01 s per image per worker
    for attempt in range(retries):
        pool = mp.get_context("spawn").Pool(workers)
        try:
            res = pool.map_async(process, jobs, chunksize=16).get(timeout=timeout)
            pool.close()
            pool.join()
            return res
        except mp.TimeoutError:
            print(f"  pool timed out (attempt {attempt + 1}); restarting", flush=True)
            pool.terminate()
            pool.join()
    print("  falling back to a single process", flush=True)
    return [process(j) for j in jobs]


def predict_all(models, cnn_in, feats, dets):
    out = {}
    with torch.no_grad():
        for name in ("custom", "mobilenet"):
            preds = []
            for s in range(0, len(cnn_in), 256):
                preds.append(models[name](normalize(center_crop(to_tensor(cnn_in[s:s + 256])))).argmax(1).numpy())
            out[name] = np.concatenate(preds)
        p = models["landmark_mlp"](torch.from_numpy(feats)).argmax(1).numpy()
        out["landmark_mlp"] = np.where(dets, p, -1)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"], help="val = quick debugging only")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (debugging)")
    args = ap.parse_args()
    df = pd.read_csv(PROCESSED / "splits.csv")
    test = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit]
    out_csv = RESULTS / ("robustness.csv" if args.split == "test" and not args.limit else "debug_robustness.csv")
    y = test.label.to_numpy()
    models = {"custom": load_cnn("custom"), "mobilenet": load_cnn("mobilenet"), "landmark_mlp": load_mlp()}
    combos = [("clean", 0, None)] + [(k, i + 1, lv) for k, lvs in LEVELS.items() for i, lv in enumerate(lvs)]
    rows = []
    t0 = time.time()
    for kind, sev, level in combos:
        res = run_combo([(p, i, kind, level) for i, p in enumerate(test.path)])
        cnn_in = np.stack([r[0] for r in res])
        feats = np.stack([r[1] for r in res]).astype(np.float32)
        dets = np.array([r[2] for r in res])
        preds = predict_all(models, cnn_in, feats, dets)
        for name, pr in preds.items():
            rows.append({"perturbation": kind, "severity": sev, "level": level, "model": NAMES[name],
                         "accuracy": float((pr == y).mean()),
                         "detection_rate": float(dets.mean()) if name == "landmark_mlp" else np.nan})
        print(f"{kind:15s} sev {sev} ({level}): " +
              ", ".join(f"{NAMES[n]} {(p == y).mean():.3f}" for n, p in preds.items()) +
              f" | det {dets.mean():.3f} | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    if out_csv.name == "robustness.csv":
        plot()


def plot_examples():
    """One test image under every perturbation and severity, so readers can see the strength."""
    df = pd.read_csv(PROCESSED / "splits.csv")
    path = df[df.split == "test"].sample(1, random_state=SEED).path.iloc[0]
    base = load_640(path)
    fig, axes = plt.subplots(len(LEVELS), 6, figsize=(12, 2.2 * len(LEVELS)))
    for r, (kind, lvs) in enumerate(LEVELS.items()):
        for c, lv in enumerate([None] + lvs):
            ax = axes[r, c]
            ax.imshow(perturb(base, kind, lv, seed=SEED))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("clean" if lv is None else f"{kind} {lv}", fontsize=8)
    fig.suptitle("Perturbation severities used in the robustness test (one test image)")
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness_examples.png", dpi=80)
    plt.close(fig)


def summarize(r):
    """Accuracy at the mildest/middle/strongest severity, plus the MLP's accuracy among
    images where MediaPipe still found a hand (undetected images are scored as wrong,
    so accuracy_given_detected = accuracy / detection_rate)."""
    rows = []
    clean = r[r.perturbation == "clean"].set_index("model").accuracy
    for kind in LEVELS:
        for model in clean.index:
            s = r[(r.perturbation == kind) & (r.model == model)].set_index("severity")
            row = {"perturbation": kind, "model": model, "clean": clean[model],
                   "sev1": s.accuracy[1], "sev3": s.accuracy[3], "sev5": s.accuracy[5],
                   "drop_sev3_points": 100 * (clean[model] - s.accuracy[3])}
            if model == "Landmark MLP":
                row["detection_sev3"] = s.detection_rate[3]
                row["mlp_acc_given_detected_sev3"] = s.accuracy[3] / s.detection_rate[3] if s.detection_rate[3] else np.nan
            rows.append(row)
    c = r[(r.perturbation == "clean") & (r.model == "Landmark MLP")].iloc[0]
    rows.append({"perturbation": "clean", "model": "Landmark MLP", "clean": c.accuracy,
                 "detection_sev3": c.detection_rate,  # clean detection rate (no severity)
                 "mlp_acc_given_detected_sev3": c.accuracy / c.detection_rate})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "robustness_summary.csv", index=False)
    return out


def plot():
    plot_examples()
    r = pd.read_csv(RESULTS / "robustness.csv")
    summarize(r)
    clean = r[r.perturbation == "clean"]
    colors = {"Custom CNN": "tab:blue", "MobileNetV3-small": "tab:green", "Landmark MLP": "tab:red"}
    kinds = list(LEVELS)
    fig_all, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax_all, kind in zip(axes.flat, kinds):
        fig, ax = plt.subplots(figsize=(6, 4))
        for a in (ax, ax_all):
            for model, col in colors.items():
                sub = r[(r.perturbation == kind) & (r.model == model)].sort_values("severity")
                c = clean[clean.model == model]
                xs = [0] + sub.severity.tolist()
                ys = c.accuracy.tolist() + sub.accuracy.tolist()
                a.plot(xs, ys, "o-", color=col, label=model)
            det = r[(r.perturbation == kind) & (r.model == "Landmark MLP")].sort_values("severity")
            a.plot([0] + det.severity.tolist(),
                   clean[clean.model == "Landmark MLP"].detection_rate.tolist() + det.detection_rate.tolist(),
                   ":", color="tab:red", alpha=0.6, label="MediaPipe detection rate")
            a.set_xticks(range(6))
            a.set_xticklabels(["clean"] + [str(v) for v in LEVELS[kind]], fontsize=8)
            a.set_ylim(0, 1)
            a.set_title(kind.replace("_", " "))
            a.set_ylabel("test accuracy")
        ax.legend(fontsize=8)
        ax.set_xlabel("severity")
        fig.tight_layout()
        fig.savefig(RESULTS / f"robustness_{kind}.png", dpi=120)
        plt.close(fig)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center", fontsize=11)
    fig_all.suptitle("Robustness: test accuracy vs. perturbation severity (approach B scored end-to-end)")
    fig_all.tight_layout()
    fig_all.savefig(RESULTS / "robustness_all.png", dpi=110)
    plt.close(fig_all)


if __name__ == "__main__":
    main()
