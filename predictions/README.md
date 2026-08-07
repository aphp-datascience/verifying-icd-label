# Per-patient predictions

84 files, one per (arm, seed): the 14 arms of Table 1 at seeds 42–47. Each holds the 288 PARHAF
cases scored by one trained classifier.

| Column | Meaning |
|---|---|
| `id` | PARHAF patient identifier |
| `dp_gold` | principal diagnosis recorded in the corpus |
| `dp` | code predicted by that run |
| `juste` | exact match |
| `juste_cat3` | match at the three-character category |

Micro-F1 in the paper is `juste.mean()` over the 288 cases, averaged across the six seeds; the
`±` is the standard deviation across seeds, not a standard error. Arm names map to Table 1 rows
in the [top-level README](../README.md).

```python
import pandas as pd, numpy as np
f = lambda a: np.mean([pd.read_parquet(f"predictions/{a}-s{s}.parquet").juste.mean()*100
                       for s in range(42, 48)])
f("filt_v4_veto") - f("rnd_v4_veto")     # +13.0, the paper's headline contrast
```

The contrast that matters is always **against the arm directly below it in the table** — the
size-matched random drop — never against the whole corpus, which would confound selection with
mere shrinkage.

## One absence worth stating

Predictions on the 307-case PARHAF extension are not here. The replication on that set was
measured but cut from the paper for length, so no number here rests on it.
