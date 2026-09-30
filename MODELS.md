# Models

| Model | Published | Why |
|---|---|---|
| Evidence qualifier | yes, 6 checkpoints | its recipe is not recoverable |
| Evidence extractor | yes, `full` model only | reusable on its own |
| Downstream classifier | no | recipe recoverable from `provenance/configs/` |

## Using the qualifier

Contrastive bi-encoder, decides whether a candidate span is a valid denomination of an ICD-10
code. Two training variants at seeds 42/43/44, six checkpoints, 444 MB each.

| Variant | What it changes |
|---|---|
| `RR` | round-robin over a code's synonyms: training sees all of them, not a fixed sample |
| `LN` | lexical negatives: the model is trained to reject near-miss wordings |

⚠️ **It is an ensemble and cannot be collapsed.** `RR+LN` is the mean of the six checkpoints'
**cosine similarities**. Averaging the weights gives a different model; so does averaging
probabilities, since the mean does not commute with the sigmoid. Pool on the cosines, before any
threshold. A single checkpoint reproduces no number in the paper.

⚠️ **Do not transfer the thresholds.** 0.190097, 0.537453 and the +0.06 margin are calibrated for
this ensemble on SYN-CAL300. Cosine scales differ between models; recalibrate on your own labelled
pairs. The sibling margin in particular has been observed to flip sign between models.

Inference code that pools the six checkpoints ships with the weights.

## Using the extractor

GLiNER, finds candidate evidence spans for a code in a note. The **`full` model only** is
published.

⛔ **The per-fold models are not, and should not be used.** Fold *k*'s model is trained on every
other fold so that it never scores a document it has seen. Out of that context they invite exactly
the mistake the folds exist to prevent.

## Where

The Hugging Face Hub: 444 MB per checkpoint is above GitHub's per-file limit, and the Hub gives
model cards, versioning and a resolvable identifier.

⏳ **Release in progress.** The link lands here and in [README.md](README.md) once the checkpoints
are up.

---

# Release record

Not needed to use the models. Kept so the choices are auditable.

**Why the qualifier's weights are published.** Several of its training settings were read from
environment variables that no run manifest captured (see the provenance gap in
[README.md](README.md)). It cannot be retrained exactly from what survives, so for this model the
weights are the only faithful record.

**Why the downstream classifier's are not.** 278 GB across the study, of which the paper rests on
the 14 arms of Table 1 at 6 seeds. Their recipe is in `provenance/configs/`, an arm-specific
checkpoint has no standalone meaning, and level 2 reproduces every number they back.

**Checkpoint identity.** The paper's model is the six `stepNOISEA-xE-contr{RR,LN}-s4*` checkpoints.
A `BACKUP_avant_ft_thesam_aac` directory also holds `xE-contr` weights: a *pre-fine-tuning*
snapshot, not the paper's model. Per-checkpoint SHA-256 in `provenance/checkpoints_sha256.txt`;
the local copies were verified byte-identical to the cluster originals.

**Tokenizer.** The backbone declares a tokenizer class that contradicts its own tokenizer file, so
a naive load falls back to one token per character. Cluster logs show the guard firing: `ratio
0.99 < 2.5` on the naive load, then `ratio 3.50` after reloading from `tokenizer.json`.
`TOKENIZER_LEGACY`, which reproduces the old broken tokenisation, was never enabled for these
runs: it appears in the logs only inside the warning text, never as a setting.

⚠️ Residual uncertainty: the training-time logs for these six runs are gone from the cluster, so
the ratio printed *during training* cannot be shown. The argument is indirect, from the guard's
own message.
