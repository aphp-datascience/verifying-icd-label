r"""Rebuild the test set from PARHAF, and check the released predictions against it.

PARHAF is public on the Hugging Face Hub, so this is the one part of the pipeline a reader can
re-derive from the source rather than take on trust. The script rebuilds the 288 cases of the
coding use case and verifies that every `dp_gold` in `predictions/` is the code PARHAF records
for that patient. Until now `dp_gold` had to be believed; here it is checked.

PARHAF is the only human-authored, peer-reviewed corpus available, so it is the test set and
nothing else: never training, never tuning, never threshold calibration.

The diagnosis is a patient-level label and each of the patient's documents inherits it. Codes are
normalised the way the training side normalises them -- uppercase, no dot. When the structured
abstract and the suggested scenario disagree, the abstract wins: it is what the author actually
wrote up, the scenario is only what they were asked to write about.

/!\ Only the public part of PARHAF is read, and no token is used. The embargoed patients are not
    part of this test set and cannot be reached from here.

Usage:  python analysis/parhaf_testset.py [--out-dir DIR]
        pip install datasets   # in addition to the analysis requirements
"""

import argparse
import collections
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PREDS = Path(__file__).resolve().parent.parent / "predictions"
POOL = "CU 2 - ICD-10 coding"


def normalise(code: str) -> str:
    return (code or "").replace(".", "").strip().upper()


def primary_diagnosis(patient: dict, source: str) -> str:
    block = patient.get(source) or {}
    diagnosis = block.get("primary_diagnosis") if isinstance(block, dict) else None
    if not diagnosis:
        return ""
    code = diagnosis.get("code")
    if isinstance(code, list):
        code = code[0] if code else None
    return normalise(code) if code else ""


def build(dataset_name: str) -> tuple[pd.DataFrame, dict]:
    dataset = load_dataset(dataset_name)["train"]        # public part, no token
    patients = [p for p in dataset if p["pool"] == POOL]

    rows, disagreements, missing = [], 0, 0
    for patient in patients:
        declared = primary_diagnosis(patient, "structured_abstract")
        suggested = primary_diagnosis(patient, "suggested_scenario")
        if declared and suggested and declared != suggested:
            disagreements += 1
        dp = declared or suggested
        if not dp:
            missing += 1
            continue
        documents = patient.get("documents") or {}
        texts = documents.get("text") or []
        types = documents.get("type") or [""] * len(texts)
        for i, text in enumerate(texts):
            if text and text.strip():
                rows.append({"doc_id": f"{patient['id']}_{i}", "patient_id": patient["id"],
                             "clinical_note": text, "icd_primary_code": dp, "weight": 1.0,
                             "specialty": patient["specialty"],
                             "doc_type": types[i] if i < len(types) else ""})

    test = pd.DataFrame(rows)
    stats = {"patients": len(patients), "documents": len(test),
             "distinct_dp": int(test["icd_primary_code"].nunique()),
             "specialties": dict(collections.Counter(p["specialty"] for p in patients)),
             "patients_without_dp": missing,
             "abstract_vs_scenario_disagreements": disagreements}
    return test, stats


def check(test: pd.DataFrame) -> int:
    """Compare the rebuilt labels with `dp_gold` in every released prediction file."""
    gold = test.drop_duplicates("patient_id").set_index("patient_id")["icd_primary_code"]
    files = sorted(PREDS.glob("*.parquet"))
    if not files:
        print("\nno prediction file found - nothing to check against.")
        return 0

    mismatches = 0
    for f in files:
        d = pd.read_parquet(f).set_index("id")
        unknown = d.index.difference(gold.index)
        if len(unknown):
            print(f"  {f.name}: {len(unknown)} patient id(s) absent from PARHAF, "
                  f"e.g. {list(unknown[:3])}")
            mismatches += len(unknown)
        common = d.index.intersection(gold.index)
        bad = d.loc[common, "dp_gold"] != gold.loc[common]
        if bad.any():
            print(f"  {f.name}: {int(bad.sum())} label(s) differ from PARHAF, "
                  f"e.g. {list(bad[bad].index[:3])}")
            mismatches += int(bad.sum())

    n = len(files)
    if mismatches:
        print(f"\n{mismatches} mismatch(es) across {n} files - the released dp_gold does NOT "
              "agree with PARHAF. Do not use these predictions until this is explained.")
    else:
        print(f"\nOK: all {n} prediction files carry exactly the diagnosis PARHAF records, "
              f"for all {len(gold)} patients.")
    return mismatches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write test.csv and test_meta.csv there (for level 3)")
    ap.add_argument("--dataset", default="HealthDataHub/PARHAF")
    args = ap.parse_args()

    test, stats = build(args.dataset)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # The evaluation file carries exactly the training columns and nothing else, because
        # load_dataset("csv", ...) requires train, validation and test to share one schema. The
        # metadata needed to break results down lives alongside, keyed by doc_id.
        test[["doc_id", "clinical_note", "icd_primary_code", "weight"]].to_csv(
            args.out_dir / "test.csv", index=False)
        test[["doc_id", "patient_id", "specialty", "doc_type", "icd_primary_code"]].to_csv(
            args.out_dir / "test_meta.csv", index=False)
        print(f"\nwrote test.csv and test_meta.csv to {args.out_dir}")

    raise SystemExit(1 if check(test) else 0)


if __name__ == "__main__":
    main()
