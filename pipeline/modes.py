r"""The three decision modes, as executable specification.

A document score is one number per (document, code); the paper turns it into a keep/drop verdict
in three ways, and Table 1 of the paper reports all three:

  * VERIFIER            the permissive threshold, tuned for 90% recall on valid codes
  * RELIABILIZER        the strict threshold, tuned for 90% precision on corrupted ones
  * RELIABILIZER+MARGIN the strict threshold, plus the sibling test of section 2

This file exists because the rule is easy to restate and hard to reimplement. Five decisions
below change the numbers and none of them is guessable from the paper's prose. They are marked
/!\ in the code.

/!\ NOT A RUNNABLE DEMO. The released annotations carry (code, passage) pairs without their
    siblings, so nothing in this repository exercises `margin`. Point these functions at your
    own scored pairs. The self-test at the bottom checks the logic on a fixture, not on data.

The thresholds are calibrated for THIS ensemble on SYN-CAL300. Cosine scales differ between
models: recalibrate before reusing one, never transfer an absolute threshold.

Usage:
    from pipeline.modes import attest, VERIFIER, RELIABILIZER, MARGIN
    kept = attest(spans, siblings, mode="margin")
"""

from __future__ import annotations

import re

import pandas as pd

VERIFIER = 0.190097      # 90% recall on valid codes, SYN-CAL300
RELIABILIZER = 0.537453  # 90% precision on corrupted codes, SYN-CAL300
MARGIN = 0.06            # calibrated on injected noise, never on a corpus the paper evaluates


def stem(code: str) -> str:
    r"""The parent a code's siblings share.

    /!\ DECISION 1. Siblings are codes of the same PARENT, which is the code minus its last
    character, not codes of the same three-character category. The two coincide for a
    four-character code and diverge beyond it: the siblings of M6534 are the other M653*, not
    every M65*. Evaluation in section 4 groups by three characters, which is a different
    partition and a wider one; do not reuse this function there.
    """
    base = re.split(r"[+\-]", str(code))[0]
    return base[:-1] if len(base) > 3 else base


def siblings_of(code: str, vocabulary: list[str]) -> list[str]:
    """Every other code sharing the parent of `code`."""
    parent = stem(code)
    return [c for c in vocabulary if c != code and stem(c) == parent]


def attest(
    spans: pd.DataFrame,
    siblings: pd.DataFrame | None = None,
    mode: str = "reliabilizer",
    documents: list[str] | None = None,
) -> pd.Series:
    """Verdict per document, indexed by `doc_id`, True when the code is attested.

    spans     one row per (doc_id, code, span): columns doc_id, code, span, cosine.
              `code` is the LABEL UNDER TEST, never the true one.
    siblings  one row per (doc_id, code, span, sibling_code): columns doc_id, code, span,
              cosine. Required for mode="margin", ignored otherwise.
    documents the corpus. Documents absent from `spans` are verdicted False, not dropped.
    """
    if mode not in ("verifier", "reliabilizer", "margin"):
        raise ValueError(f"unknown mode {mode!r}")

    sp = spans.copy()
    threshold = VERIFIER if mode == "verifier" else RELIABILIZER
    passes = sp["cosine"].to_numpy() >= threshold

    if mode == "margin":
        if siblings is None:
            raise ValueError("mode='margin' needs the sibling scores")
        # /!\ DECISION 2. The margin opposes the labelled code to its BEST sibling on the SAME
        # span, so the worst case is a max over siblings, computed span by span.
        worst = (siblings.groupby(["doc_id", "code", "span"])["cosine"].max()
                 .rename("cos_sibling"))
        sp = sp.merge(worst, on=["doc_id", "code", "span"], how="left")
        # /!\ DECISION 3. A span with no scored sibling gets margin = its own cosine, so it
        # passes. The rule is silent where it has no comparison, it does not reject by default.
        # Consequence to state when reporting: over a corpus whose sibling coverage is partial,
        # the filter removes fewer documents than the rule would at full coverage, and any gain
        # measured that way is a FLOOR.
        margin = sp["cosine"].to_numpy() - sp["cos_sibling"].fillna(0.0).to_numpy()
        passes = passes & (margin >= MARGIN)

    # /!\ DECISION 4. The conjunction is evaluated SPAN BY SPAN, then aggregated with any().
    # Never take the document's best span and test the rule on it: one span can clear the
    # threshold while another clears the margin, and the two are not the same verdict.
    verdict = pd.Series(passes).groupby(sp["doc_id"].to_numpy()).max()

    if documents is not None:
        # /!\ DECISION 5. A document for which the extractor proposed no span is NOT attested,
        # so it leaves the corpus. Reindexing here is what makes that explicit; dropping those
        # documents silently would inflate every acceptance rate.
        verdict = verdict.reindex(documents).fillna(False)
    return verdict.astype(bool)


def keep(
    corpus: pd.DataFrame,
    spans: pd.DataFrame,
    siblings: pd.DataFrame | None = None,
    mode: str = "reliabilizer",
) -> pd.DataFrame:
    r"""Filter a training corpus. `corpus` needs a doc_id column; every other column survives.

    /!\ The label under test is the one the corpus CARRIES. Scoring the true code instead leaks
    the ground truth and the filter stops detecting anything.
    """
    verdict = attest(spans, siblings, mode=mode, documents=corpus["doc_id"].tolist())
    return corpus[corpus["doc_id"].map(verdict).fillna(False).to_numpy()]


def _self_test() -> None:
    """Fixture, not data: each case pins one of the five decisions above."""
    sp = pd.DataFrame({
        "doc_id": ["d1", "d1", "d2", "d3"],
        "code":   ["K810", "K810", "K810", "K810"],
        "span":   ["s1", "s2", "s1", "s1"],
        "cosine": [0.30, 0.70, 0.70, 0.10],
    })
    sib = pd.DataFrame({
        "doc_id":        ["d1", "d1", "d2"],
        "code":          ["K810", "K810", "K810"],
        "span":          ["s2", "s2", "s1"],
        "sibling_code":  ["K819", "K811", "K819"],
        "cosine":        [0.89, 0.70, 0.20],
    })
    docs = ["d1", "d2", "d3", "d4"]

    v = attest(sp, mode="verifier", documents=docs)
    assert v.tolist() == [True, True, False, False], v.tolist()

    r = attest(sp, mode="reliabilizer", documents=docs)
    assert r.tolist() == [True, True, False, False], r.tolist()

    m = attest(sp, sib, mode="margin", documents=docs)
    # d1: span at 0.70 against a sibling at 0.89, margin -0.19 -> rejected
    # d2: 0.70 against 0.20, margin +0.50 -> kept
    # d3: below both thresholds. d4: no span at all -> not attested (decision 5)
    assert m.tolist() == [False, True, False, False], m.tolist()

    # decision 3: drop d1's sibling rows and its margin becomes its own cosine, so it passes
    m_nosib = attest(sp, sib[sib.doc_id != "d1"], mode="margin", documents=docs)
    assert m_nosib["d1"], "a span with no scored sibling must pass, not be rejected"

    assert stem("M6534") == "M653" and stem("K810") == "K81" and stem("A00") == "A00"
    assert siblings_of("K810", ["K810", "K811", "K819", "K802"]) == ["K811", "K819"]

    print("modes.py: 5 decisions pinned, self-test OK")


if __name__ == "__main__":
    _self_test()
