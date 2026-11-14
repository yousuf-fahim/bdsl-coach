"""Phase 5: 95% bootstrap confidence intervals for test accuracy.

Uses the SAVED test predictions (results/test_predictions.npz); it does not run
any model on the test set again. Resamples the 2,200 test images with
replacement 2,000 times (seed 42) and reports the 2.5th and 97.5th percentiles.
Also gives a paired bootstrap of the accuracy difference between models.

Output: results/comparison_ci.csv
Run: .venv\\Scripts\\python src\\bootstrap_ci.py
"""
import numpy as np
import pandas as pd

from common import RESULTS, SEED
from evaluate import NAMES

B = 2000


def main():
    P = np.load(RESULTS / "test_predictions.npz", allow_pickle=True)
    y, det = P["y"], P["detected"]
    correct = {}
    for k in ("custom", "mobilenet", "landmark_mlp"):
        pred = P[f"probs_{k}"].argmax(1)
        ok = pred == y
        if k == "landmark_mlp":
            correct["Landmark MLP (end-to-end)"] = ok & det
            correct["Landmark MLP (detected only)"] = np.where(det, ok, np.nan)
        else:
            correct[NAMES[k]] = ok
            correct[f"{NAMES[k]} (detected subset)"] = np.where(det, ok, np.nan)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (B, len(y)))
    rows = []
    for name, c in correct.items():
        c = c.astype(float)
        boots = np.nanmean(c[idx], axis=1)
        rows.append({"model": name, "accuracy": float(np.nanmean(c)),
                     "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)),
                     "n": int((~np.isnan(c)).sum())})
    # paired differences (same resampled images for both models)
    pairs = [("MobileNetV3-small", "Custom CNN"), ("Landmark MLP (end-to-end)", "Custom CNN"),
             ("Landmark MLP (detected only)", "Custom CNN (detected subset)")]
    for a, b in pairs:
        ca, cb = correct[a].astype(float), correct[b].astype(float)
        diff = np.nanmean(ca[idx], 1) - np.nanmean(cb[idx], 1)
        rows.append({"model": f"difference: {a} minus {b}", "accuracy": float(np.nanmean(ca) - np.nanmean(cb)),
                     "ci95_low": float(np.percentile(diff, 2.5)), "ci95_high": float(np.percentile(diff, 97.5)),
                     "n": int(len(y))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_ci.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
