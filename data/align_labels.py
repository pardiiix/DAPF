#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# Map any known label variant -> {healthy, dementia}
LABEL_MAP = {
    # CCC / ADReSS common
    "ct": "healthy",
    "control": "healthy",
    "hc": "healthy",
    "healthy": "healthy",
    "0": "healthy",

    "ad": "dementia",
    "alz": "dementia",
    "alzheimers": "dementia",
    "alzheimer's": "dementia",
    "alzheimer’s": "dementia",
    "dementia": "dementia",
    "1": "dementia",
}


def normalize_label(x: str) -> str:
    if x is None:
        return x
    s = str(x).strip()
    if s == "":
        return s
    key = s.lower().strip()

    # handle variants like "AD " or " Control"
    key = key.replace("\u2019", "'")  # normalize curly apostrophe
    return LABEL_MAP.get(key, s)      # if unknown, keep original


def process_file(path: Path, label_col: str = "label", dry_run: bool = False) -> dict:
    df = pd.read_csv(path, dtype=str)

    if label_col not in df.columns:
        return {"file": str(path), "status": "skipped (no label column)", "changed": 0}

    before = df[label_col].fillna("").astype(str).str.strip()
    df[label_col] = df[label_col].apply(normalize_label)

    after = df[label_col].fillna("").astype(str).str.strip()
    changed = int((before != after).sum())

    # sanity: warn if unexpected labels remain
    remaining = sorted(set(after.unique()) - {"healthy", "dementia", ""})
    info = {
        "file": str(path),
        "status": "ok",
        "changed": changed,
        "remaining_other_labels": remaining[:20],  # cap
    }

    if not dry_run:
        df.to_csv(path, index=False)

    return info


def main():
    ap = argparse.ArgumentParser(description="Align dataset labels to {healthy, dementia}.")
    ap.add_argument("--data_dir", default=".", help="Directory containing the CSVs (default: current dir)")
    ap.add_argument(
        "--files",
        nargs="*",
        default=["ccc_train_all.csv", "adress-train_all.csv", "adress-test_all.csv"],
        help="CSV filenames to process (relative to data_dir).",
    )
    ap.add_argument("--label_col", default="label", help="Label column name (default: label)")
    ap.add_argument("--dry_run", action="store_true", help="Do not overwrite files; just report changes.")
    args = ap.parse_args()

    base = Path(args.data_dir).resolve()
    results = []

    for fname in args.files:
        p = base / fname
        if not p.exists():
            results.append({"file": str(p), "status": "missing", "changed": 0})
            continue
        results.append(process_file(p, label_col=args.label_col, dry_run=args.dry_run))

    # Print summary
    print("\nLabel alignment summary (target labels: healthy/dementia):")
    for r in results:
        print(f"- {r['file']}: {r['status']}, changed={r['changed']}", end="")
        if r.get("remaining_other_labels"):
            print(f", remaining_other_labels={r['remaining_other_labels']}")
        else:
            print()

    if args.dry_run:
        print("\n(dry run) No files were modified.")


if __name__ == "__main__":
    main()