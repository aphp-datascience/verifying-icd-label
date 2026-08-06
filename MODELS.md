# Models

Weights are not kept in this repository. Two models are worth publishing, for two different
reasons; the rest deliberately are not.

## To publish

### The evidence qualifier — because its recipe is lost

The contrastive qualifier decides whether a candidate span is a valid denomination of an ICD-10
code. It exists in two variants, `RR` and `LN`, each trained at seeds 42/43/44 — six
checkpoints, `best_model_stepNOISEA.pt`, 444 MB each.

**Publishing these weights is not a convenience, it is the only faithful record.** Several of
the qualifier's training settings are read from environment variables, and no run manifest
captured them (see the provenance gap in [README.md](README.md)). Nobody — including us — can
retrain this model exactly from what survives. If the weights go, the artifact goes.

#### ⚠️ The published model is an ensemble, and it cannot be collapsed

`RR+LN` is **the mean of the six checkpoints' cosine similarities**, not a single set of
weights. Averaging the weights would not give the same model, and neither would averaging
probabilities: **the mean does not commute with the sigmoid**, so the pooling has to happen on
the cosines, before any calibrated threshold is applied.

The artifact is therefore **one model repository holding the six checkpoints plus the inference
code that pools them**. One artifact, six files — publishing a single checkpoint as "the" model
would not reproduce any number in the paper.

Per-checkpoint SHA-256 are recorded in `provenance/checkpoints_sha256.txt`, and the local copies
were verified byte-identical to the cluster originals.

### The evidence extractor — because it is reusable

The GLiNER extractor finds candidate evidence spans for a code in a note. Publish the **`full`
model only**.

⛔ **Never publish the per-fold models.** They are cross-fitting artifacts: fold *k*'s model is
trained on every other fold precisely so that it never scores a document it has seen. Released
without that context they invite exactly the mistake the folds exist to prevent.

## Not to publish

The downstream classifier checkpoints — 149 arms × up to 6 seeds, 278 GB. Their recipe *is*
recoverable (`provenance/configs/`), so the weights add nothing that cannot be regenerated, and
an arm-specific checkpoint has no standalone meaning. Level 2 already reproduces every number
they back.

## Checks

**1. Checkpoint identity — settled.** The paper's model is the six `stepNOISEA-xE-contr{RR,LN}-s4*`
checkpoints pooled as above. A `BACKUP_avant_ft_thesam_aac` directory also holds `xE-contr`
weights: those are a *pre-fine-tuning* snapshot and are **not** the paper's model.

**2. Tokenizer sanity — checked, no anomaly found.** The backbone declares a tokenizer class
that contradicts its own tokenizer file, so a naive load falls back to one token per character.
The guard catches it: cluster logs show `ratio 0.99 < 2.5` on the naive load, then `ratio 3.50`
after reloading from `tokenizer.json`.

`TOKENIZER_LEGACY`, the switch that reproduces the *old* broken tokenisation for checkpoints
that need it, **was never enabled** for these runs — it appears in the logs only inside the
warning text that mentions it, never as a setting. Those checkpoints were scored on the sane
tokenizer and produced the paper's figures.

*Residual uncertainty, stated rather than hidden:* the training-time logs for these six runs are
no longer on the cluster, so the ratio printed **during training** cannot be shown directly. The
argument is indirect — the guard's own message says legacy mode is required only for checkpoints
trained before the fix, and these were scored without it.

## Where

The Hugging Face Hub, not GitHub: 444 MB per checkpoint is above GitHub's per-file limit, and
the Hub gives model cards, versioning and a resolvable identifier. This repository will link to
them.
