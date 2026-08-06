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
  configs/<family>/<run>/config.yml   752 resolved training configs
  manifests/<family>/<run>.json        74 run manifests (commit, config, data fingerprints)
  runs_summary.csv                    725 runs: final loss, micro/macro P/R/F, steps, seed
archive/                              full training histories (gitignored, see below)
```

**`provenance/configs/` is the important part.** These are *resolved* configs, not templates:
corpus paths, backbone, `max_step`, `loss_scale` and referential are substituted in. With the
seed — encoded in the run name as `-s42` … `-s47` — each file is the complete recipe for one
run. 654 encoder runs across **149 distinct arms** and 6 seeds, plus 71 qualifier runs.

`runs_summary.csv` carries one row per run: family, arm, seed, steps, final loss, micro and
macro precision/recall/F1. The full per-step histories, including the per-code breakdown over
all 886 codes, are 860 MB raw and are kept as a compressed archive outside git.

## Known gap in the provenance

The 74 run manifests record the commit, the resolved config and the data fingerprints, but
**none of them captured the environment variables**. For the encoder that is harmless: its
configuration lives entirely in the config file. For the **qualifier** it is not — several of
its training settings are read from the environment and are recorded nowhere, so its runs cannot
be reconstructed exactly from this repository.

That is why the qualifier's weights are published rather than only its recipe: for that model,
the weights are the only faithful record. See [MODELS.md](MODELS.md).

## Code

The pipeline lives in four repositories, each public:

| Repository | Role |
|---|---|
| [partages-cu2-icd-evidence-extractor](https://github.com/aphp-datascience/partages-cu2-icd-evidence-extractor) | Extracts candidate evidence spans for a code |
| [partages-cu2-icd-evidence-qualifier](https://github.com/aphp-datascience/partages-cu2-icd-evidence-qualifier) | Decides whether a span is a valid denomination of the code |
| [partages-cu2-encoder-baseline](https://github.com/aphp-datascience/partages-cu2-encoder-baseline) | Downstream document-level ICD-10 classifier |
| [partages-cu2-plm-icd](https://github.com/aphp-datascience/partages-cu2-plm-icd) | PLM-ICD-style comparison baseline |

## License

Apache 2.0 — see [LICENSE](LICENSE).
