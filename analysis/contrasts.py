r"""Per-patient paired contrast between two arms: the part of the uncertainty that seeds reduce,
and the part that nothing will.

Two uncertainties behave differently and must not be pooled blindly:

  * TRAINING variance -- from one seed to the next, with patients held fixed. Its standard error
    is sd_between_seeds / sqrt(n_seeds): it shrinks if you add seeds.
  * PATIENT sampling -- what if PARHAF had drawn 288 other patients. It is measured on the
    per-patient differences averaged over seeds, and it does NOT shrink with the number of
    seeds. It is a floor, common to every seed.

Hence the question this script answers: is a null contrast "no effect" or "no power" -- and if
it is power, is it fixable? An effect smaller than the patient floor will never be established
on this test set, however many seeds are trained. More patients, or give up.

/!\ The 288 patients are THE SAME at every seed. The inter-seed spread is therefore already pure
    training variance -- there is nothing to subtract from it. An earlier version of this
    analysis wrongly removed patient sampling from it, as if each seed drew new patients.
/!\ Pairing is done on the patient id, never on row order.

Usage:  python analysis/contrasts.py <arm_A> <arm_B> [seeds...]
Example: python analysis/contrasts.py filt_v4_veto rnd_v4_veto     # the paper's +13.0
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PREDS = Path(__file__).resolve().parent.parent / "predictions"
SEEDS_DEFAULT = [42, 43, 44, 45, 46, 47]
COLUMNS = {"juste": "exact code (micro-F1)", "juste_cat3": "three-character category"}


def load(arm: str, seeds: list[int]) -> dict[int, pd.DataFrame]:
    out = {}
    for s in seeds:
        f = PREDS / f"{arm}-s{s}.parquet"
        if f.exists():
            out[s] = pd.read_parquet(f).set_index("id").sort_index()
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test, on the disagreements between the two models only."""
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return n10, n01, 1.0
    return n10, n01, float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    arm_a, arm_b = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3:]] or SEEDS_DEFAULT

    A, B = load(arm_a, seeds), load(arm_b, seeds)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(f"no paired seed between {arm_a} and {arm_b} in {PREDS}")
    unpaired = sorted((set(A) | set(B)) - set(common))
    print(f"\n{arm_a}  vs  {arm_b}")
    print(f"paired seeds: {common}" + (f"   (unpaired, ignored: {unpaired})" if unpaired else ""))

    for col, label in COLUMNS.items():
        print(f"\n-- {label} " + "-" * (70 - len(label)))
        print(f"{'seed':>6}{'A':>9}{'B':>9}{'delta':>10}{'A+B-':>7}{'A-B+':>7}{'McNemar p':>12}")

        # Patients present in every run of both arms: the pairing has to be total, otherwise we
        # would be comparing means over different populations.
        idx = None
        for s in common:
            for df in (A[s], B[s]):
                idx = df.index if idx is None else idx.intersection(df.index)
        n_pat = len(idx)

        diffs, per_patient, p_values = [], [], []
        for s in common:
            av = A[s].loc[idx, col].to_numpy(bool)
            bv = B[s].loc[idx, col].to_numpy(bool)
            n10, n01, p = mcnemar(av, bv)
            p_values.append(p)
            diffs.append((av.mean() - bv.mean()) * 100)
            per_patient.append(av.astype(float) - bv.astype(float))
            print(f"{s:>6}{av.mean() * 100:>9.2f}{bv.mean() * 100:>9.2f}"
                  f"{diffs[-1]:>+10.2f}{n10:>7}{n01:>7}{p:>12.4f}")

        diffs, n = np.array(diffs), len(diffs)
        d_pat = np.mean(per_patient, axis=0)
        se_pat = float(np.std(d_pat, ddof=1) / np.sqrt(n_pat)) * 100
        sd_seed = float(np.std(diffs, ddof=1))
        se_seed = sd_seed / np.sqrt(n)
        # Model delta(s,i) = mu + a_s + b_i + e_si, so
        #   se_seed^2 = var_S/nS + var_E/(nS*nP)      (seeds, patients fixed)
        #   se_pat^2  = var_P/nP + var_E/(nS*nP)      (patients, seeds averaged)
        # and the true Var(mean delta) = var_S/nS + var_P/nP + var_E/(nS*nP). Adding the two
        # therefore counts the residual term TWICE: se_tot is CONSERVATIVE by var_E/(nS*nP),
        # negligible at nP = 288. That excess is preferable to the opposite error.
        se_tot = float(np.hypot(se_seed, se_pat))

        print(f"\n  mean difference {diffs.mean():+.2f} pp   "
              f"({(diffs > 0).sum()}/{n} seeds, {n_pat} paired patients)")
        print(f"  standard error from RETRAINING  {se_seed:>6.2f} pp"
              f"   (inter-seed sd {sd_seed:.2f}; /sqrt(seeds), so reducible)")
        print(f"  standard error from PATIENTS    {se_pat:>6.2f} pp"
              f"   (floor: adding seeds changes NOTHING)")
        print(f"  combined                        {se_tot:>6.2f} pp"
              f"   => 95% CI ~ [{diffs.mean() - 1.96 * se_tot:+.2f}; "
              f"{diffs.mean() + 1.96 * se_tot:+.2f}]")

        floor = 1.96 * se_pat
        print(f"\n  DETECTABILITY FLOOR on this test set: +/-{floor:.2f} pp. A true effect smaller")
        print(f"  than that will never be established on the {n_pat} PARHAF cases, at any number")
        print("  of seeds. It would take more patients.")
        mde = stats.t.ppf(0.975, n - 1) * sd_seed / np.sqrt(n)
        print(f"  At {n} seeds the reducible part adds +/-{mde:.2f} pp -- "
              + ("it dominates, so more seeds would help."
                 if mde > floor else "the floor dominates, so more seeds would not help."))

        signif = [p for p in p_values if p < 0.05]
        signs = {np.sign(d) for d, p in zip(diffs, p_values) if p < 0.05 and d != 0}
        if abs(diffs.mean()) > 1.96 * se_tot and (diffs > 0).sum() in (0, n):
            print("  => EFFECT: consistent in sign across every seed and larger than the "
                  "combined uncertainty.")
        elif not signif:
            fixable = "REPAIRABLE by more seeds" if abs(diffs.mean()) > floor else "IRREPARABLE"
            print(f"  => NO POWER, and {fixable}: no seed separates the two arms on these "
                  "patients.")
        elif len(signs) > 1:
            print("  => TRAINING INSTABILITY: significant seeds of OPPOSITE signs.")
        else:
            print(f"  => plausible but not established: significant on {len(signif)}/{n} seeds "
                  "only.")


if __name__ == "__main__":
    main()
