"""Phase 1: inventory and audit every raw image.

Outputs
  data/processed/inventory.csv   one row per image (path, class, index, size, hashes)
  results/audit_summary.json     counts, corrupt files, size stats, imbalance
  results/class_counts.png       images per class
  results/image_sizes.png        distribution of image width/height
  results/sample_grid.png        4 random samples per class with the Bangla label

Run: .venv\\Scripts\\python src\\audit_data.py
"""
from concurrent.futures import ProcessPoolExecutor

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import BENGALI_FONT, PROCESSED, RAW, RESULTS, SEED, load_labels, save_json


def inspect(path):
    """Open one image, check it decodes, and compute two perceptual hashes."""
    row = {"path": path.relative_to(RAW).as_posix(), "bytes": path.stat().st_size}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im = im.convert("RGB")  # forces a full decode, so truncated files fail here
            row["phash"] = str(imagehash.phash(im))   # 64-bit DCT-based hash
            row["dhash"] = str(imagehash.dhash(im))   # 64-bit gradient hash
            row["mean_brightness"] = float(np.asarray(im.convert("L")).mean())
        row["corrupt"] = False
    except Exception as e:  # noqa: BLE001 - we want to record any decode error
        row.update(corrupt=True, error=repr(e))
    return row


def main():
    files = sorted(RAW.glob("*/*/*"))
    print("files found:", len(files))
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(inspect, files, chunksize=64))
    df = pd.DataFrame(rows)

    # Parse structure: <orig_split>/<folder>/<prefix>_<index>.<ext>
    parts = df["path"].str.split("/", expand=True)
    df["orig_split"], df["folder"], fname = parts[0], parts[1].astype(int), parts[2]
    stem = fname.str.rsplit(".", n=1).str[0]
    df["prefix"] = stem.str.split("_").str[0].astype(int)
    df["img_idx"] = stem.str.split("_").str[1].astype(int)
    df["ext"] = fname.str.rsplit(".", n=1).str[1].str.lower()
    df["label"] = df["folder"]  # class index 0..48

    # Every file's name prefix must equal folder + 1 (the paper's label number).
    mismatch = df[df["prefix"] != df["folder"] + 1]
    dup_names = df.duplicated(["folder", "img_idx"]).sum()

    labels = load_labels()
    df.to_csv(PROCESSED / "inventory.csv", index=False)

    counts = df.groupby("label").size()
    summary = {
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "orig_split_counts": df["orig_split"].value_counts().to_dict(),
        "extensions": df["ext"].value_counts().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "n_corrupt": int(df["corrupt"].sum()),
        "corrupt_files": df.loc[df["corrupt"], "path"].tolist(),
        "prefix_folder_mismatches": mismatch["path"].tolist(),
        "duplicate_class_index_pairs": int(dup_names),
        "width": df["width"].describe().round(1).to_dict(),
        "height": df["height"].describe().round(1).to_dict(),
        "n_distinct_sizes": int(df.groupby(["width", "height"]).ngroups),
        "most_common_sizes": {f"{w}x{h}": int(n) for (w, h), n in
                              df.groupby(["width", "height"]).size().nlargest(5).items()},
        "images_per_class": {"min": int(counts.min()), "max": int(counts.max()),
                             "mean": round(float(counts.mean()), 1),
                             "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 3)},
        "smallest_classes": {labels[i]["bangla"]: int(n) for i, n in counts.nsmallest(3).items()},
        "largest_classes": {labels[i]["bangla"]: int(n) for i, n in counts.nlargest(3).items()},
    }
    save_json(summary, RESULTS / "audit_summary.json")
    print({k: summary[k] for k in ("n_images", "n_classes", "n_corrupt", "images_per_class")})

    plt.rcParams["font.family"] = [BENGALI_FONT, "DejaVu Sans"]

    # Class-count bar chart
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts.values, color="#4C72B0")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([labels[i]["display"] for i in counts.index], fontsize=9)
    ax.axhline(counts.mean(), color="gray", ls="--", lw=1, label=f"mean {counts.mean():.0f}")
    ax.set_ylabel("images")
    ax.set_title("BDSL 49 recognition set: images per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "class_counts.png", dpi=130)
    plt.close(fig)

    # Image size distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["width"], df["height"], s=3, alpha=0.3)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image sizes")
    fig.tight_layout()
    fig.savefig(RESULTS / "image_sizes.png", dpi=130)
    plt.close(fig)

    # Sample grid: 4 classes per row, 4 random images per class (13 rows x 16 columns)
    rng = np.random.default_rng(SEED)
    k = 4
    fig, axes = plt.subplots(13, 16, figsize=(16, 14))
    for ax in axes.flat:
        ax.axis("off")
    for c in range(49):
        sub = df[(df["label"] == c) & ~df["corrupt"]]
        picks = sub.iloc[rng.choice(len(sub), size=k, replace=False)]
        r, c0 = divmod(c, 4)  # 4 classes per row, 4 images each
        for j, (_, row) in enumerate(picks.iterrows()):
            ax = axes[r, c0 * 4 + j]
            ax.imshow(Image.open(RAW / row["path"]).convert("RGB"))
            if j == 0:
                ax.set_title(f"{c}: {labels[c]['display']}", fontsize=10, loc="left")
    fig.suptitle("4 random samples per class (folder index: Bangla label)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_grid.png", dpi=90)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
