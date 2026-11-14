"""Phase 1: leakage analysis and leakage-aware train/val/test split (70/15/15).

Why not a random split?
  Within each class the images are stored in capture order: images a few
  index numbers apart come from the same signer/session (same hand, same
  background, near-identical pose). A random split puts these "siblings" in
  both train and test, so the model can score well by recognising the
  background or the person instead of the sign.

What this script does
  1. Hash distances: find near-duplicate pairs (pHash Hamming distance <= NEAR_DUP).
  2. Capture-order evidence: similarity vs. index gap (results/leakage_gap_curve.png).
  3. Groups: contiguous blocks of BLOCK consecutive indices per class.
  4. Split: per class, walk the blocks in index order starting at a random
     rotation (seed 42) and give contiguous runs to test, then val, then train.
     Any near-duplicate pair still crossing splits has its member in the larger
     set marked "excluded", so no evaluated image has a near-duplicate in train.
  5. Checks + a demonstration: 1-nearest-neighbour accuracy on a random split
     vs. our grouped split, to show how much a random split inflates scores.

Outputs: data/processed/splits.csv, results/split_summary.json,
results/leakage_report.json, results/leakage_gap_curve.png,
results/near_duplicate_examples.png, results/leakage_knn_demo.json
Run: .venv\\Scripts\\python src\\make_splits.py
"""
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import BENGALI_FONT, PROCESSED, RAW, RESULTS, SEED, load_labels, save_json

BLOCK = 15        # consecutive indices per group (similarity plateaus at a gap of ~20-25)
NEAR_DUP = 8      # pHash Hamming distance <= 8 of 64 bits = near-duplicate
FRACTIONS = {"test": 0.15, "val": 0.15}  # remainder is train (0.70)


# ---------- helpers ----------
def hex_to_u64(hexes):
    return np.array([int(x, 16) for x in hexes], dtype=np.uint64)


def hamming_rows(a, b):
    """Hamming distance between a[i] and every b[j]: returns (len(a), len(b))."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(2)


def colour_hist(path):
    """8x4x4 HSV histogram of a 64x64 thumbnail: a crude 'background + skin' descriptor."""
    im = Image.open(RAW / path).convert("HSV").resize((64, 64))
    h, _ = np.histogramdd(np.asarray(im).reshape(-1, 3), bins=(8, 4, 4), range=((0, 256),) * 3)
    h = h.ravel()
    return h / h.sum()


# ---------- main ----------
def main():
    labels = load_labels()
    df = pd.read_csv(PROCESSED / "inventory.csv")
    df = df.sort_values(["label", "img_idx"]).reset_index(drop=True)
    n = len(df)
    ph = hex_to_u64(df["phash"])
    lab = df["label"].to_numpy()
    idx = df["img_idx"].to_numpy()

    # 1. All near-duplicate pairs (chunked all-pairs Hamming distance).
    pairs = []
    for s in range(0, n, 1000):
        d = hamming_rows(ph[s:s + 1000], ph)
        ii, jj = np.nonzero(d <= NEAR_DUP)
        ii = ii + s
        keep = jj > ii
        pairs += list(zip(ii[keep], jj[keep], d[ii[keep] - s, jj[keep]]))
    pairs = pd.DataFrame(pairs, columns=["i", "j", "dist"])
    pairs["same_class"] = lab[pairs.i] == lab[pairs.j]
    pairs["index_gap"] = np.where(pairs.same_class, np.abs(idx[pairs.i] - idx[pairs.j]), -1)
    print(f"near-duplicate pairs (pHash <= {NEAR_DUP}):", len(pairs),
          "cross-class:", int((~pairs.same_class).sum()))

    # 2. Similarity vs. index gap (evidence for capture order).
    with ProcessPoolExecutor() as ex:
        H = np.array(list(ex.map(colour_hist, df["path"], chunksize=64)))
    key = {(l, i): k for k, (l, i) in enumerate(zip(lab, idx))}
    gaps = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60, 100, 150]
    curve = []
    for g in gaps:
        A = [k for (l, i), k in key.items() if (l, i + g) in key]
        B = [key[(lab[k], idx[k] + g)] for k in A]
        A, B = np.array(A), np.array(B)
        x = np.bitwise_xor(ph[A], ph[B])
        hd = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        chi = 0.5 * ((H[A] - H[B]) ** 2 / (H[A] + H[B] + 1e-9)).sum(1)
        curve.append({"gap": g, "pairs": len(A), "phash_mean": float(hd.mean()),
                      "colour_chi2_mean": float(chi.mean())})
    curve = pd.DataFrame(curve)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(curve.gap, curve.phash_mean, "o-", label="pHash distance (structure)")
    ax1.set_xscale("log")
    ax1.set_xlabel("gap between image indices within a class (log scale)")
    ax1.set_ylabel("mean pHash distance (0-64)")
    ax2 = ax1.twinx()
    ax2.plot(curve.gap, curve.colour_chi2_mean, "s--", color="tab:orange", label="colour distance (background)")
    ax2.set_ylabel("mean colour chi-square distance")
    ax1.axvline(BLOCK, color="gray", lw=1, ls=":")
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.18))
    ax1.set_title("Nearby indices look alike: images are stored in capture order")
    fig.tight_layout()
    fig.savefig(RESULTS / "leakage_gap_curve.png", dpi=130)
    plt.close(fig)

    # 3. Groups = contiguous index blocks within a class (capture-order proxy for signer).
    #    We do NOT union blocks through near-duplicate links: links are transitive, and at
    #    this threshold they chained whole classes into one group (tried; see learning log).
    df["group"] = df["label"] * 1000 + (df["img_idx"] - 1) // BLOCK
    print("groups:", df.group.nunique(), "| median images/group:", int(df.groupby("group").size().median()))

    # 4. Per-class contiguous assignment from a random rotation (seed 42):
    #    test gets the first ~15% of blocks, val the next ~15%, train the rest.
    rng = np.random.default_rng(SEED)
    df["split"] = "train"
    for c in range(49):
        sub = df[df.label == c]
        order = sorted(sub["group"].unique())
        sizes = sub.groupby("group").size()
        start = int(rng.integers(len(order)))
        order = order[start:] + order[:start]
        total, pos = len(sub), 0
        for split_name, frac in FRACTIONS.items():
            taken = 0
            # add whole blocks while that brings us closer to the target share
            while pos < len(order) and abs(taken + sizes[order[pos]] - frac * total) < abs(taken - frac * total):
                df.loc[df.group == order[pos], "split"] = split_name
                taken += sizes[order[pos]]
                pos += 1

    # 4b. Residual leakage: a same-class near-duplicate pair split across two sets.
    #     Exclude the member in the LARGER set (train > val > test), so the
    #     evaluation blocks stay intact. (Moving the val/test member into train
    #     was tried first: it put train images inside test blocks, and 41% of test
    #     images then had a same-class train neighbour within 5 indices vs 10.6% now.)
    rank = {"train": 0, "val": 1, "test": 2, "excluded": 3}
    same = pairs[pairs.same_class]
    moved = []  # excluded images
    for p in same.itertuples():
        a, b = df.at[p.i, "split"], df.at[p.j, "split"]
        if a != b and "excluded" not in (a, b):
            k = p.i if rank[a] < rank[b] else p.j
            df.at[k, "split"] = "excluded"
            moved.append(df.path[k])
    print("images excluded to break near-duplicate leaks:", len(moved))
    df[["path", "label", "img_idx", "orig_split", "group", "split"]].to_csv(PROCESSED / "splits.csv", index=False)

    # 5a. Checks: groups crossing splits (only via moved images); near-duplicates across splits.
    kept = df[df.split != "excluded"]
    groups_crossing = int((kept.groupby("group")["split"].nunique() > 1).sum())
    s = df["split"].to_numpy()
    cross_split = pairs[(s[pairs.i] != s[pairs.j]) & (s[pairs.i] != "excluded") & (s[pairs.j] != "excluded")]
    per_split = df.groupby("split").size()
    per_class = df.groupby(["label", "split"]).size().unstack(fill_value=0)

    # 5b. Demonstration: 1-NN on colour histograms (no learning of hand shape at all).
    def knn_acc(train_mask, test_mask):
        knn = KNeighborsClassifier(n_neighbors=1).fit(H[train_mask], lab[train_mask])
        return float((knn.predict(H[test_mask]) == lab[test_mask]).mean())

    tr_r, te_r = train_test_split(np.arange(n), test_size=0.15, stratify=lab, random_state=SEED)
    m_tr_r = np.isin(np.arange(n), tr_r)
    demo = {
        "description": "1-nearest-neighbour on colour histograms only (background/skin colour, no hand shape).",
        "random_split_test_accuracy": knn_acc(m_tr_r, ~m_tr_r),
        "grouped_split_test_accuracy": knn_acc(s == "train", s == "test"),
        "chance_accuracy": 1 / 49,
    }
    # How many test images have a train image of the same class within 5 indices?
    tr_keys = {(l, i) for l, i, sp in zip(lab, idx, s) if sp == "train"}
    def frac_with_neighbor(mask):
        ks = [(l, i) for l, i, m in zip(lab, idx, mask) if m]
        return float(np.mean([any((l, i + g) in tr_keys for g in range(-5, 6) if g) for l, i in ks]))
    tr_keys_r = {(lab[k], idx[k]) for k in tr_r}
    demo["test_images_with_train_neighbor_within_5_idx"] = {
        "random_split": float(np.mean([any((lab[k], idx[k] + g) in tr_keys_r for g in range(-5, 6) if g) for k in te_r])),
        "grouped_split": frac_with_neighbor(s == "test"),
    }
    save_json(demo, RESULTS / "leakage_knn_demo.json")
    print("kNN demo:", demo)

    save_json({
        "n_images": n, "block_size": BLOCK, "near_dup_threshold_phash": NEAR_DUP,
        "n_groups": int(df.group.nunique()),
        "split_counts": per_split.to_dict(),
        "split_fractions": (per_split / n).round(4).to_dict(),
        "per_class_min": per_class.min().to_dict(), "per_class_max": per_class.max().to_dict(),
        "groups_crossing_splits_excluding_excluded": groups_crossing,
        "images_excluded_near_dup": len(moved),
        "excluded_images": moved,
        "near_dup_pairs_across_splits_same_class": int(cross_split.same_class.sum()),
        "near_dup_pairs_across_splits_cross_class": int((~cross_split.same_class).sum()),
    }, RESULTS / "split_summary.json")

    # Leakage report + cross-class near-duplicates (suspicious: same image, two labels?)
    xc = pairs[~pairs.same_class].sort_values("dist")
    save_json({
        "near_dup_pairs_total": int(len(pairs)),
        "near_dup_pairs_same_class": int(pairs.same_class.sum()),
        "near_dup_pairs_cross_class": int(len(xc)),
        "exact_hash_duplicates": int((pairs.dist == 0).sum()),
        "same_class_pairs_index_gap_le_5": int((pairs.same_class & (pairs.index_gap <= 5)).sum()),
        "same_class_pairs_index_gap_gt_25": int((pairs.same_class & (pairs.index_gap > 25)).sum()),
        "signer_ids_recoverable": False,
        "signer_note": "File names are <paper_label>_<index>.jpg; no signer field. Grouping uses capture-order blocks.",
        "gap_curve": curve.round(3).to_dict(orient="records"),
        "cross_class_pairs": [
            {"a": df.path[r.i], "b": df.path[r.j], "dist": int(r.dist),
             "label_a": labels[lab[r.i]]["bangla"], "label_b": labels[lab[r.j]]["bangla"]}
            for r in xc.itertuples()],
    }, RESULTS / "leakage_report.json")

    # Example figure: 6 closest same-class pairs far apart in index + up to 6 cross-class pairs.
    plt.rcParams["font.family"] = [BENGALI_FONT, "DejaVu Sans"]
    show = pd.concat([pairs[pairs.same_class].sort_values("dist").head(6), xc.head(6)])
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2.2 * len(show)))
    for r, p in enumerate(show.itertuples()):
        for c, k in enumerate((p.i, p.j)):
            ax = axes[r, c]
            ax.imshow(Image.open(RAW / df.path[k]).convert("RGB"))
            ax.set_title(f"{labels[lab[k]]['display']}  {df.path[k].split('/')[-1]} ({s[k]})", fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.1, 0.5, f"d={p.dist}", transform=axes[r, 0].transAxes, ha="right", fontsize=9)
    fig.suptitle("Near-duplicate pairs (top: same class, bottom: different class)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(RESULTS / "near_duplicate_examples.png", dpi=100)
    plt.close(fig)
    print("split counts:", per_split.to_dict(), "| groups crossing splits:", groups_crossing,
          "| same-class near-dups across splits:", int(cross_split.same_class.sum()))


if __name__ == "__main__":
    main()
