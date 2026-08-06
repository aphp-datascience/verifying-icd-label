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

## ⚠️ Checks required before any upload

Neither has been done yet.

1. **Confirm the checkpoint identity.** `RR` and `LN` are two arms; the paper's model is their
   combination. A `BACKUP_avant_ft_thesam_aac` directory also holds `xE-contr` weights — those
   are a *pre-fine-tuning* snapshot and are **not** the paper's model. Do not confuse them.
2. **Verify tokenizer sanity.** A checkpoint trained under a tokenizer that had silently fallen
   back to one token per character must not be published: those results were retracted and will
   not be presented. The characters-per-token guard shipped in the model repositories tests this
   in under a minute — a healthy ratio is ≈ 3.5–4, a broken one ≈ 1.0.

## Where

The Hugging Face Hub, not GitHub: 444 MB per checkpoint is above GitHub's per-file limit, and
the Hub gives model cards, versioning and a resolvable identifier. This repository will link to
them.
