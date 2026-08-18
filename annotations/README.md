# Annotated evaluation pairs

The three sets behind §2 of the paper: 1,200 (code, passage) pairs, each judged for whether the
passage attests the code.

| File | Pairs | Origin | Used for |
|---|---|---|---|
| `real-dev400.csv` | 400 | passages extracted from **physician-written** PARHAF reports | ROC-AUC on real text |
| `syn-clin500.csv` | 500 | passages from generated reports | ROC-AUC on generated text |
| `syn-cal300.csv` | 300 | generated, stratified by extraction score | threshold calibration |

## Schema

| Column | Meaning |
|---|---|
| `span_id` | index within the set |
| `code` | ICD-10 code queried |
| `definition` | official label of that code — the qualifier's left input |
| `libelle` | the candidate passage — the qualifier's right input |
| `cos_rr_ln` | pooled cosine of the `RR+LN` ensemble (mean over its 6 checkpoints) |
| `gliner_score` | extractor confidence, used for stratification only |
| `bin` | extraction-score band the pair was sampled from (empty for `syn-clin500`) |
| `vote_opus`, `vote_fable_v2`, `vote_fable_v3` | the three annotation passes, 0/1 |
| `n_yes` | number of passes answering yes |
| `label` | the ground truth used in the paper — the majority, i.e. `n_yes >= 2` |

`definition` and `libelle` keep their original names: they are the column names of the data
contract the models are trained against.

## ⚠️ Three passes, but two models

The passes are Claude Opus, and two prompt variants (`v2`, `v3-strict`) of a second frontier
model. `vote_fable_v2` and `vote_fable_v3` therefore come from the **same** model, so agreement
between those two columns measures prompt stability, not inter-annotator agreement. Read the
reported κ range with that in mind, and prefer Opus-vs-either as the independent comparison.

No human adjudication was performed on these three sets.

## What this reproduces

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
d = pd.read_csv("annotations/real-dev400.csv")
roc_auc_score(d.label, d.cos_rr_ln)      # 0.9462, printed as 0.95 in §2
```

Same on `syn-clin500.csv` gives 0.8209 (printed 0.82). On `syn-cal300.csv`, scanning thresholds
on the 0.0125 grid the calibration used, the lowest one reaching precision ≥ 0.90 is **0.5375** —
the reliabilizer threshold the paper operates at. Precision is 0.9186 there and 0.8989 one step
below.

`python analysis/section2.py` prints all of it, both operating points included.

## Provenance and license

`real-dev400.csv` quotes short passages from **PARHAF** (Tannier et al., 2026), a corpus of
reports written by physicians for **fictitious** patients, released under CC BY 4.0 and Etalab
2.0. No real patient data is involved, here or anywhere in this pipeline. The passages are
reproduced under that license; cite PARHAF if you use them.

## Not here

Sets that exist in the study but back no number in the paper: `SYN-GLI200`, `DIAG39`,
`HARDNEG544`, `REFVAL300`.
