r"""Section 2: discrimination of the qualifier, and the operating points derived from it.

Runs on `annotations/*.csv` alone -- no model, no inference. The `cos_rr_ln` column is the
cosine of the RR+LN ensemble, pooled over its six checkpoints, precomputed and released so that
this section is reproducible without the weights.

/!\ The AUC is computed on the COSINE, not on the probability, and that is exact:
    prob = sigmoid(2*s*cos) is monotone increasing, so AUC(cos) == AUC(prob). It is also what
    licenses averaging the members' cosines -- the mean does not commute with the sigmoid, so
    averaging probabilities would not be valid.
/!\ Thresholds are on THIS ensemble's cosine scale and transfer to no other model. Two encoders
    with the same ranking have different cosine distributions; only the RANK is comparable.

The calibration scanned a grid of step 0.0125, which is why the published reliabilizer threshold
(0.5375 = 43 * 0.0125) sits slightly above the raw empirical optimum.

Usage:  python analysis/section2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

ANN = Path(__file__).resolve().parent.parent / "annotations"
GRID = np.arange(0.0, 1.0 + 1e-9, 0.0125)
SETS = [("REAL-DEV400  400 real annotated spans", "real-dev400.csv"),
        ("SYN-CLIN500  500 generated clinical spans", "syn-clin500.csv"),
        ("SYN-CAL300   300 generated spans, stratified by extraction score", "syn-cal300.csv")]


def main() -> None:
    frames = {}
    for title, name in SETS:
        d = pd.read_csv(ANN / name)
        frames[name] = d
        y, s = d["label"].to_numpy(int), d["cos_rr_ln"].to_numpy(float)
        print(f"===== {title}   n={len(y)}  positives={int(y.sum())} "
              f"({y.mean():.0%})")
        print(f"  ROC-AUC (RR+LN ensemble, 6 checkpoints): {roc_auc_score(y, s):.4f}")
        # The three votes are three passes, of which two share a model -- see annotations/README.
        agree = (d[["vote_opus", "vote_fable_v2", "vote_fable_v3"]].nunique(axis=1) == 1).mean()
        print(f"  unanimous annotation passes: {agree:.1%}\n")

    # Operating points, calibrated on SYN-CAL300 and on nothing else.
    d = frames["syn-cal300.csv"]
    y, s = d["label"].to_numpy(int), d["cos_rr_ln"].to_numpy(float)
    print("===== Operating points on SYN-CAL300 (grid step 0.0125)")
    rows = []
    for t in GRID:
        keep = s >= t
        if keep.sum() == 0:
            continue
        rows.append((t, y[keep].mean(), keep.sum() and y[keep].sum() / y.sum()))
    reliabilizer = next((t, p, r) for t, p, r in rows if p >= 0.90)
    verifier = max((r for r in rows if r[2] >= 0.90), key=lambda r: r[0])
    below = max(t for t, _, _ in rows if t < reliabilizer[0])
    p_below = next(p for t, p, _ in rows if t == below)
    print(f"  reliabilizer (first threshold reaching precision >= 0.90):"
          f" {reliabilizer[0]:.4f}   precision {reliabilizer[1]:.4f}, recall {reliabilizer[2]:.4f}")
    print(f"    one grid step below ({below:.4f}): precision {p_below:.4f} -- "
          "which is what makes 0.5375 the answer rather than a nearby round number")
    print(f"  verifier (highest threshold holding recall >= 0.90):"
          f" {verifier[0]:.4f}   precision {verifier[1]:.4f}, recall {verifier[2]:.4f}")

    prec, rec, _ = precision_recall_curve(y, s)
    reachable = prec >= 0.90
    print(f"\n  recall attainable at 90% precision: {rec[reachable].max():.4f}")
    print("  (unstable across seeds by construction -- never compare two variants on it "
          "with one run each)")


if __name__ == "__main__":
    main()
