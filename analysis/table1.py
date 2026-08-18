r"""Regenerate Table 1 of the paper from the released per-patient predictions.

Micro-F1 on the 288 PARHAF cases, mean over six seeds. The diagnosis is mono-label, so
`juste` is binary per patient and `mean(juste)` *is* the micro-F1 (P = R = F1). That is what
makes the per-patient analysis in `contrasts.py` legitimate; on a macro metric it would not be.

/!\ The +/- is an inter-seed STANDARD DEVIATION, not a standard error and not a contrast
    interval. With six seeds, SE = sd/sqrt(6) ~ sd/2.45, so overlapping bars can hide an
    established difference. Read a contrast with `contrasts.py`, never off this table.

Training-set sizes (N) are printed from the published corpus counts: they are a property of the
filtered corpora, which are not part of this repository.

Usage:  python analysis/table1.py [--latex]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = Path(__file__).resolve().parent.parent / "predictions"
SEEDS = range(42, 48)

# (label, native arm, corrupted arm, native N, corrupted N, is_control)
ROWS = [
    ("No filter", "clean", "noisy", 11605, 11605, False),
    ("Verifier filter", "clean_v4_verif", "filt_v4_verif", 11573, 10617, False),
    ("same-size random drop", "cln_rnd_v4_verif", "rnd_v4_verif", None, None, True),
    ("Reliabilizer filter", "clean_v4_fiab", "filt_v4_fiab", 9771, 7488, False),
    ("same-size random drop", "cln_rnd_v4_fiab", "rnd_v4_fiab", None, None, True),
    ("Reliabilizer filter + margin", "clean_v4_veto", "filt_v4_veto", 7977, 6670, False),
    ("same-size random drop", "cln_rnd_v4_veto", "rnd_v4_veto", None, None, True),
]
BOLD = "Reliabilizer filter + margin"   # the row the paper emphasises, in both corpora


def stat(arm: str) -> tuple[float, float, int] | None:
    """Mean, inter-seed sd and seed count for one arm. Refuses a cell under two seeds."""
    v = [pd.read_parquet(f)["juste"].mean() * 100
         for f in (PREDS / f"{arm}-s{s}.parquet" for s in SEEDS) if f.exists()]
    return (float(np.mean(v)), float(np.std(v, ddof=1)), len(v)) if len(v) >= 2 else None


def main() -> None:
    latex, missing, seeds_seen = "--latex" in sys.argv, [], set()
    body = []
    for label, native, corrupted, n_nat, n_cor, control in ROWS:
        cells = []
        for arm in (native, corrupted):
            r = stat(arm)
            if r is None:
                missing.append(arm)
            else:
                seeds_seen.add(r[2])
            cells.append(r)
        body.append({"label": label, "cells": cells, "n": (n_nat, n_cor), "control": control})

    # Delta against the size-matched random drop on the line directly below. A control row has
    # no delta of its own; the unfiltered row has no control, and the paper prints "---" there.
    for i, row in enumerate(body):
        below = body[i + 1] if i + 1 < len(body) else None
        row["delta"] = [
            None if row["control"] or not below or not below["control"]
            or not row["cells"][j] or not below["cells"][j]
            else row["cells"][j][0] - below["cells"][j][0]
            for j in (0, 1)]

    if not latex:
        print(f"{'':<30}{'N':>9}{'micro-F1':>13}{'Δ ctrl.':>10}"
              f"{'N':>11}{'micro-F1':>13}{'Δ ctrl.':>10}")
        print("  " + "-" * 94)

    for row in body:
        bold = latex and row["label"] == BOLD
        out = []
        for j in (0, 1):
            c, d = row["cells"][j], row["delta"][j]
            if c is None:
                score = "---"
            elif latex:
                score = f"{c[0]:.1f}\\,\\sd{{{c[1]:.1f}}}"
            else:
                score = f"{c[0]:5.1f} ±{c[1]:.1f}"
            if d is None:
                delta = "" if row["control"] else "---"
            else:
                delta = f"{abs(d):.1f}"
                delta = (("$-$" if d < 0 else "+") + delta) if latex else f"{d:+.1f}"
            if bold:
                score, delta = f"\\textbf{{{score}}}", f"\\textbf{{{delta}}}"
            n = row["n"][j]
            n = "" if n is None else (f"{n:,}".replace(",", "{,}") if latex else f"{n:,}")
            out += [n, score, delta]
        if latex:
            label = ("\\ \\ " if row["control"] else "") + row["label"]
            print(f"    {label:<32}" + " & " + " & ".join(out[:3] + [""] + out[3:]) + r" \\")
        else:
            label = ("  " if row["control"] else "") + row["label"]
            print(f"  {label:<28}" + "".join(f"{x:>10}" if k % 3 != 1 else f"{x:>13}"
                                             for k, x in enumerate(out)))
    if not latex:
        print(f"\nseeds per cell: {sorted(seeds_seen)}   (± is the inter-seed sd)")
    if missing:
        print(f"\nMissing dumps for {len(missing)} arm(s) — do not regenerate the table: "
              f"{', '.join(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
