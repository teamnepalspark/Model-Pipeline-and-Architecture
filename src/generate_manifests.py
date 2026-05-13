from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def discover_subject_dirs(data_root: Path) -> List[Path]:
    subjects = []
    # Walk one or two levels deep to find folders that contain T1.nii.gz or b0.nii.gz
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        # check p itself
        if (p / "T1.nii.gz").exists() or (p / "b0.nii.gz").exists():
            subjects.append(p)
            continue
        # check immediate children
        for c in sorted(p.iterdir()):
            if not c.is_dir():
                continue
            if (c / "T1.nii.gz").exists() or (c / "b0.nii.gz").exists():
                subjects.append(c)
    return subjects


def build_record(subject_dir: Path) -> dict:
    # Expect files T1.nii.gz, b0.nii.gz, optional b0_u.nii.gz
    sid = subject_dir.name
    rec = {"id": sid}
    t1 = subject_dir / "T1.nii.gz"
    b0 = subject_dir / "b0.nii.gz"
    b0u = subject_dir / "b0_u.nii.gz"
    if t1.exists():
        rec["t1"] = str(t1)
    if b0.exists():
        rec["b0_distorted"] = str(b0)
    if b0u.exists():
        rec["b0_corrected"] = str(b0u)
    return rec


def split_records(records: List[dict], val_frac: float = 0.2, seed: int = 42):
    records = list(records)
    random.Random(seed).shuffle(records)

    if len(records) == 0:
        return [], []
    if len(records) == 1:
        # For smoke runs with one subject, keep it in train and mirror to val.
        return [records[0]], [records[0]]

    n_val = max(1, int(len(records) * val_frac))
    if n_val >= len(records):
        n_val = len(records) - 1
    return records[n_val:], records[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/local manifests from data directory structure.")
    parser.add_argument("--data_root", default="data", help="Root data folder to scan")
    parser.add_argument("--out_dir", default="data/manifests", help="Output manifests folder")
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subject_dirs(data_root)
    labeled = []
    unlabeled = []
    for s in subjects:
        rec = build_record(s)
        # require at least t1 and b0
        if "t1" in rec and "b0_distorted" in rec and "b0_corrected" in rec:
            labeled.append(rec)
        elif "t1" in rec and "b0_distorted" in rec:
            unlabeled.append(rec)

    train, val = split_records(labeled, val_frac=args.val_frac)

    # write manifests
    (out_dir / "train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (out_dir / "val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    (out_dir / "local.json").write_text(json.dumps(unlabeled, indent=2), encoding="utf-8")

    print(f"Discovered {len(subjects)} subjects: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
    print(f"Wrote manifests to: {out_dir}")


if __name__ == "__main__":
    main()
