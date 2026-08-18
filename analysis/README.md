# Analysis

Three scripts, each reproducing a published number from the material in this repository and
nothing else. No model is loaded and no inference is run — this is level 2.

```bash
pip install pandas numpy scipy scikit-learn pyarrow

python analysis/table1.py                                # Table 1, all 14 cells
python analysis/table1.py --latex                        # the same, as the paper's tabular
python analysis/contrasts.py filt_v4_veto rnd_v4_veto    # the headline +13.0, with its CI
python analysis/section2.py                              # the ROC-AUCs and the thresholds
```

| Script | Reproduces | Reads |
|---|---|---|
| `table1.py` | every cell of Table 1 and both Δ columns | `predictions/` |
| `contrasts.py` | any paired contrast, its 95% CI, its McNemar tests | `predictions/` |
| `section2.py` | ROC-AUC 0.95 and 0.82, and the 0.5375 threshold | `annotations/` |

`table1.py` prints the training-set sizes from the published corpus counts rather than measuring
them: `N` is a property of the filtered corpora, which are not part of this repository.

## Read a contrast with `contrasts.py`, never off Table 1

The `±` in Table 1 is an inter-seed standard deviation, and a Δ column is a difference of two
means. Neither says whether a contrast is established. `contrasts.py` separates the two
uncertainties that Table 1 cannot:

- **training variance**, which shrinks as `1/√seeds`;
- **patient sampling**, which does not shrink at all and sets a floor of ±1.6 to ±3.5 pp
  depending on the pair of arms compared.

The consequence is worth stating, because it cuts against the intuitive reading. On the native
corpus the margin filter gains **+1.85 pp, in the same direction on all six seeds, with an
inter-seed sd of only 1.05** — and it is still not established: the patient floor is ±3.45 pp,
so the 95% interval is `[-1.70; +5.41]`. More seeds would not change that. It would take more
patients. Conversely the corrupted-corpus contrast of the same filter, `+12.96 pp`, clears the
floor comfortably and is established.

That is why the script prints the floor beside every contrast, and says whether the missing
power is repairable. On these 288 cases a null result is more often a power limit than an
absence of effect.

## What these scripts do not reach

Several numbers in the paper depend on the corpora rather than on the predictions, and cannot be
recomputed from this repository:

- the attestation and rejection rates of §3 (99.7 / 72.2 %, 96.2 / 92.0 / 78.5 %) and the
  residual noise (22 / 7 / 3 %) — they are measured on the filtered training corpora;
- the emptied-class analysis (102 codes against 75, and the subgroup of 30 patients) — it needs
  to know which codes survive each filter, which is corpus composition;
- the scale figures of §2 (494k pairs over 42,377 codes, the extraction precision curve) — they
  belong to the thesaurus and the extraction stage.

Reproducing those is level 3, and needs the corpora and the model repositories listed in the
[top-level README](../README.md).
