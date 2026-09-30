# Verifying ICD-10 Labels in Clinical Corpora

Code, models, data and annotated evaluation sets for the paper *Verifying ICD-10 Labels in
Clinical Corpora*, plus the material to check every number in it. The label-verification pipeline
is one component of the PARTAGES / CU2 project, not the whole of it.

The pipeline scores any (text, code) pair for whether the text attests the code. Filtering a
training corpus with it produces a better downstream coder.

![The pipeline: evidence extractor, evidence qualifier, and the three decision modes](docs/pipeline.png)

## Code: three repositories

| Stage | Repository | Role |
|---|---|---|
| 1. extract | [partages-cu2-icd-evidence-extractor](https://github.com/aphp-datascience/partages-cu2-icd-evidence-extractor) | GLiNER, conditioned on the code's label. Proposes evidence spans. No manual annotation, cross-fitted over folds |
| 2. qualify | [partages-cu2-icd-evidence-qualifier](https://github.com/aphp-datascience/partages-cu2-icd-evidence-qualifier) | Contrastive bi-encoder on code–synonym pairs. Scores each span; a document's score is its best span |
| downstream | [partages-cu2-encoder-baseline](https://github.com/aphp-datascience/partages-cu2-encoder-baseline) | Document-level ICD-10 classifier. Measures what filtering is worth |

Three decision modes turn a document score into a verdict, all calibrated on
`annotations/syn-cal300.csv` and never on a corpus this work evaluates:

| Mode | Rule | Value |
|---|---|---|
| verifier | score ≥ threshold | 0.190097, 90% recall |
| reliabilizer | score ≥ threshold | 0.537453, 90% precision |
| reliabilizer + margin | and the labelled code leads its best sibling | margin +0.06 |

[`pipeline/modes.py`](pipeline/modes.py) is their reference implementation. Five choices change
the numbers and the paper's prose pins down none of them: what counts as a sibling, when the
conjunction is evaluated, what happens to a document with no evidence. Specification, not demo:
the released annotations carry no sibling scores, so nothing here exercises the margin.
`python pipeline/modes.py` runs a self-test on a fixture.

## Models

| Model | Where | Status |
|---|---|---|
| Evidence qualifier | Hugging Face Hub, with the code that pools it | **release in progress** |
| Evidence extractor, `full` model | extractor repo | not released as weights |
| Downstream coder, CamemBERTa-v2 + attention pooling | recipe in `provenance/configs/` | not released as weights |

⚠️ **The qualifier is an ensemble and cannot be collapsed.** It is the mean of six checkpoints'
**cosine similarities**. Averaging weights or probabilities gives a different model, since the
mean does not commute with the sigmoid. Pool on the cosines, before any threshold. A single
checkpoint reproduces no number in the paper.

⚠️ **Do not transfer the thresholds.** They are calibrated for this ensemble on SYN-CAL300.
Cosine scales differ between models, and the sibling margin has been observed to flip sign;
recalibrate on your own labelled pairs.

⚠️ **The extractor's per-fold models are not published, and should not be used.** Fold *k*'s model
is trained on every other fold so that it never scores a document it has seen. Out of that
context they invite exactly the mistake the folds exist to prevent.

The qualifier's weights are published rather than only its recipe because several of its training
settings were read from environment variables that no run manifest captured, so the recipe alone
does not reconstruct it.

## Data

| Corpus | Role | Where |
|---|---|---|
| **PARHAF** (Tannier et al., 2026) | test set, 288 physician-written reports, principal diagnosis in outpatient surgery | [`HealthDataHub/PARHAF`](https://huggingface.co/datasets/HealthDataHub/PARHAF), CC BY 4.0 + Etalab 2.0 |
| **Generated corpus** | training corpus, 11,605 LLM-generated surgery reports over 886 codes | **release in progress** |

PARHAF describes **fictitious** patients; no real patient data is involved anywhere in this
pipeline. Only its public part is read, and no token is needed.

⚠️ The pipeline cannot be assessed on the generated corpus: the LLM writes *from* the target code,
so the text echoes the expected phrasing and the corpus is **self-attesting**. Sensitivity is
measured by injecting known label errors into PARHAF.

## Annotated evaluation sets

1,200 (code, passage) pairs, each judged for whether the passage attests the code. They evaluate
any model that scores a code against a passage of French clinical text.

| File | Pairs | Origin | Used for |
|---|---|---|---|
| `annotations/real-dev400.csv` | 400 | passages from **physician-written** PARHAF reports | ROC-AUC on real text: **0.95** |
| `annotations/syn-clin500.csv` | 500 | passages from generated reports | ROC-AUC on generated text: **0.82** |
| `annotations/syn-cal300.csv` | 300 | generated, stratified by extraction score | threshold calibration |

Each pair carries its three annotation passes, the pooled cosine of the released ensemble, and the
majority label. ⚠️ Two of the three passes come from the **same** model under two prompts, so
their agreement measures prompt stability, not inter-annotator agreement. Schema and caveats:
[annotations/README.md](annotations/README.md).

## Reproducing the paper

| Level | What you re-do | Needs | Cost |
|---|---|---|---|
| **1. verify** | Re-render every table and figure from the released numbers | this repo | seconds |
| **2. re-analyse** | Recompute every contrast, interval and subgroup from the per-patient predictions | this repo | minutes, CPU |
| **3. retrain** | Re-run the pipeline end to end | the corpora and the three repositories | GPU-weeks |

```bash
python analysis/table1.py                                # the paper's Table 1, below
python analysis/contrasts.py filt_v4_veto rnd_v4_veto    # the headline +13.0, with its CI
python analysis/section2.py                              # ROC-AUC 0.95 / 0.82, threshold 0.5375
python analysis/parhaf_testset.py                        # rebuild the 288 cases, audit dp_gold
```

Levels 1 and 2 run on `annotations/` and `predictions/` alone: no model, no inference, four
dependencies. `table1.py` prints **Table 1 of the paper**, micro-F1 on the 288 PARHAF cases over
six seeds:

```
                                       native corpus                    30% corrupted
                                      N     micro-F1   Δ ctrl.          N     micro-F1   Δ ctrl.
  ----------------------------------------------------------------------------------------------
  No filter                       11,605    41.1 ±1.5       ---    11,605    28.6 ±1.0       ---
  Verifier filter                 11,573    39.7 ±3.1      -1.5    10,617    30.5 ±1.4      +2.5
    same-size random drop                   41.2 ±1.9                        28.0 ±2.5
  Reliabilizer filter              9,771    42.2 ±1.2      +0.8     7,488    37.6 ±2.2      +8.6
    same-size random drop                   41.4 ±1.2                        28.9 ±2.0
  Reliabilizer filter + margin     7,977    43.3 ±0.8      +1.9     6,670    41.8 ±1.5     +13.0
    same-size random drop                   41.4 ±1.2                        28.9 ±2.9
```

Each filter is followed by its own size-matched random drop. The comparison that means something
is the vertical one, against the line directly below, never against the unfiltered corpus.

⚠️ **Level 2 does not re-run inference.** It recomputes statistics from predictions already
written. It verifies the analysis, not that those predictions came from the models; that is
level 3.

⚠️ **Level 3 is reproduction, not bit-identical replay.** Same recipe, same data, same seed gives
a model within the inter-seed standard deviation, never the same weights. GPU kernels, mixed
precision and dataloader ordering are not deterministic.

⚠️ **Read a contrast with `contrasts.py`, never off the table.** The `±` is an inter-seed standard
deviation and hides the sampling of the 288 patients, a floor of ±1.6 to ±3.5 pp that no number of
seeds reduces. A gain consistent on all six seeds can still fail to clear it: the native `+1.9`
does. [analysis/README.md](analysis/README.md) also lists the paper's numbers this repository
cannot reach.

## What is here

```
docs/pipeline.png                       the figure above
pipeline/modes.py                       the three decision modes, with the thresholds
analysis/table1.py                      the paper's Table 1, from the predictions
analysis/contrasts.py                   paired per-patient contrast, CI and McNemar
analysis/section2.py                    ROC-AUCs and the calibrated thresholds
analysis/parhaf_testset.py              rebuilds the test set from PARHAF, audits dp_gold
annotations/*.csv                       1,200 annotated (code, passage) pairs, §2
predictions/<arm>-s<seed>.parquet       84 runs × 288 patients, §3 and the table above
provenance/
  configs/<family>/<run>/config.yml     90 resolved training configs
  runs_summary.csv                      84 runs: final loss, micro/macro P/R/F, steps, seed
  checkpoints_sha256.txt                6 qualifier checkpoints
archive/                                full training histories (gitignored)
```

**Scope: only the runs behind a number in the paper.** The study explored 149 encoder arms; 135
answer questions the paper does not report, and shipping them would invite reconstructing results
we chose not to claim. Here: the 14 arms of the table at 6 seeds, plus the 6 qualifier
checkpoints.

| Row of the table | native corpus | 30% corrupted |
|---|---|---|
| No filter | `clean` | `noisy` |
| Verifier filter | `clean_v4_verif` | `filt_v4_verif` |
| ⟶ same-size random drop | `cln_rnd_v4_verif` | `rnd_v4_verif` |
| Reliabilizer filter | `clean_v4_fiab` | `filt_v4_fiab` |
| ⟶ same-size random drop | `cln_rnd_v4_fiab` | `rnd_v4_fiab` |
| Reliabilizer filter + margin | `clean_v4_veto` | `filt_v4_veto` |
| ⟶ same-size random drop | `cln_rnd_v4_veto` | `rnd_v4_veto` |

Each name takes the suffix `-s42` … `-s47`. The six qualifier checkpoints are pinned by
`provenance/checkpoints_sha256.txt`.

`provenance/configs/` holds *resolved* configs, not templates: corpus paths, backbone, `max_step`,
`loss_scale` and referential are substituted in. With the seed, encoded in the run name, each file
is the complete recipe for one run.

`runs_summary.csv` carries one row per run: family, arm, seed, steps, final loss, micro and macro
precision/recall/F1. ⚠️ These are each run's own validation figures on the generated corpus; the
paper's numbers come from re-scoring the saved predictions on the 288 PARHAF cases, which is what
level 2 does. The per-step histories, 860 MB raw, are archived outside git.

## Known gap in the provenance

Training runs wrote a manifest with the commit, the resolved config and the data fingerprints. Of
the 74 that survive, **not one captured the environment variables**, and none documents a run
reported in the paper, which is why no manifest is shipped.

For the encoder this is harmless: its configuration lives entirely in the config file, so
`provenance/configs/` is complete. For the **qualifier** it is not: several training settings come
from the environment and are recorded nowhere, so its runs cannot be reconstructed exactly from
this repository. The same gap explains why it contributes no row to `runs_summary.csv`. That is
why its weights are published rather than only its recipe.

## License

Apache 2.0, see [LICENSE](LICENSE). PARHAF passages quoted in `annotations/real-dev400.csv` are
reproduced under CC BY 4.0 / Etalab 2.0; cite PARHAF if you use them.
