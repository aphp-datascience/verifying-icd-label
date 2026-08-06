# Verifying ICD-10 Labels in Clinical Corpora — artifacts

Reproduction material for *Verifying ICD-10 Labels in Clinical Corpora* (PARTAGES / CU2).

This repository does not hold a model or a corpus. It holds **what a reader needs to check the
paper**: the exact recipe of every training run behind the reported numbers, the per-run
metrics, and the analysis that turns per-patient predictions into the tables of the paper.

## Three levels of reproduction

Each costs an order of magnitude more than the previous one. Pick the one that matches what you
want to establish.

| Level | What you re-do | Needs | Cost |
|---|---|---|---|
| **1 — verify** | Re-render every table and figure from the released numbers | this repo | seconds |
| **2 — re-analyse** | Recompute every contrast, confidence interval and subgroup from the per-patient predictions | this repo | minutes, CPU |
| **3 — retrain** | Re-run the pipeline end to end | the corpora and the model repos below | GPU-weeks |

⚠️ **What level 2 does not do.** It recomputes the statistics from predictions that were
*already written*; it does not re-run inference. It therefore verifies the whole analysis —
which is where the paper's claims live — but not that those predictions came from the models.
Checking that is level 3.

⚠️ **Level 3 is reproduction, not bit-identical replay.** Same recipe, same data, same seed gives
a model within the inter-seed standard deviation the paper prints — never the same weights.
GPU kernels, mixed precision and dataloader ordering are not deterministic.

## What is here

```
provenance/
  configs/<family>/<run>/config.yml    90 resolved training configs
  runs_summary.csv                     84 runs: final loss, micro/macro P/R/F, steps, seed
  checkpoints_sha256.txt                6 qualifier checkpoints
archive/                              full training histories (gitignored, see below)
```

**Scope: only the runs behind a number in the paper.** The study explored 149 encoder arms; 135
of them answer questions the paper does not report, and shipping them would invite a reader to
reconstruct results we chose not to claim. What is here is the 14 arms of Table 1 at 6 seeds,
plus the 6 checkpoints of the qualifier.

| Table 1 row | native corpus | 30% corrupted |
|---|---|---|
| No filter | `clean` | `noisy` |
| Verifier filter | `clean_v4_verif` | `filt_v4_verif` |
| ⟶ same-size random drop | `cln_rnd_v4_verif` | `rnd_v4_verif` |
| Reliabilizer filter | `clean_v4_fiab` | `filt_v4_fiab` |
| ⟶ same-size random drop | `cln_rnd_v4_fiab` | `rnd_v4_fiab` |
| Reliabilizer filter + margin | `clean_v4_veto` | `filt_v4_veto` |
| ⟶ same-size random drop | `cln_rnd_v4_veto` | `rnd_v4_veto` |

Each name takes the suffix `-s42` … `-s47`. The qualifier is
`stepNOISEA-xE-contr{RR,LN}-s4{2,3,4}`.

**`provenance/configs/` is the important part.** These are *resolved* configs, not templates:
corpus paths, backbone, `max_step`, `loss_scale` and referential are substituted in. With the
seed — encoded in the run name — each file is the complete recipe for one run.

`runs_summary.csv` carries one row per run: family, arm, seed, steps, final loss, micro and
macro precision/recall/F1. Note that these are each run's own validation figures on the
generated corpus; the paper's numbers come from re-scoring the saved predictions on the 288
PARHAF cases, which is what level 2 does. The full per-step histories, including the per-code
breakdown over all 886 codes, are 860 MB raw and are kept as a compressed archive outside git.

## Known gap in the provenance

Training runs wrote a manifest recording the commit, the resolved config and the data
fingerprints. Of the 74 that survive, **not one captured the environment variables** — and none
of the 74 documents a run reported in the paper, which is why no manifest is shipped here.

For the encoder the gap is harmless: its configuration lives entirely in the config file, so
`provenance/configs/` is complete. For the **qualifier** it is not — several of its training
settings are read from the environment and are recorded nowhere, so its runs cannot be
reconstructed exactly from this repository. The same gap explains why the qualifier contributes
no row to `runs_summary.csv`: no per-run metrics survive under an identifiable arm name for its
six checkpoints.

That is why the qualifier's weights are published rather than only its recipe: for that model,
the weights are the only faithful record. See [MODELS.md](MODELS.md).

## Code

The pipeline lives in three repositories, each public:

| Repository | Role |
|---|---|
| [partages-cu2-icd-evidence-extractor](https://github.com/aphp-datascience/partages-cu2-icd-evidence-extractor) | Extracts candidate evidence spans for a code |
| [partages-cu2-icd-evidence-qualifier](https://github.com/aphp-datascience/partages-cu2-icd-evidence-qualifier) | Decides whether a span is a valid denomination of the code |
| [partages-cu2-encoder-baseline](https://github.com/aphp-datascience/partages-cu2-encoder-baseline) | Downstream document-level ICD-10 classifier |

## License

Apache 2.0 — see [LICENSE](LICENSE).
