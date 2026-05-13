from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_vol(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).mean() * (b * b).mean()) + 1e-6
    return float((a * b).mean() / den)


def evaluate_labeled(pred_dir: Path, gt_manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_manifest_csv)
    rows = []
    for _, r in df.iterrows():
        case_id = str(r["id"])
        gt = load_vol(r["gt_path"])
        pred_path = pred_dir / f"{case_id}_b0_corrected_pred.nii.gz"
        pred = load_vol(pred_path)

        # Use dynamic range from GT for PSNR/SSIM stability.
        data_range = float(gt.max() - gt.min()) + 1e-6
        rows.append(
            {
                "id": case_id,
                "ssim": float(structural_similarity(gt, pred, data_range=data_range)),
                "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=data_range)),
                "mae": float(np.mean(np.abs(gt - pred))),
                "ncc": ncc(gt, pred),
            }
        )
    out = pd.DataFrame(rows)
    return out


def evaluate_clinician(clinician_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(clinician_csv)
    criteria = [
        "distortion_skull_base",
        "distortion_temporal_lobe",
        "alignment_t1",
        "lesion_confidence",
        "overall_acceptability",
    ]
    for c in criteria:
        if c not in df.columns:
            raise ValueError(f"Missing required column in clinician CSV: {c}")

    df["clinical_mean"] = df[criteria].mean(axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quantitative and clinician metrics.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--labeled_gt_csv", default=None)
    parser.add_argument("--clinician_csv", default=None)
    parser.add_argument("--output_report", default="outputs/reports/evaluation_summary.csv")
    args = parser.parse_args()

    reports = []

    if args.labeled_gt_csv:
        qdf = evaluate_labeled(Path(args.pred_dir), Path(args.labeled_gt_csv))
        q_summary = qdf[["ssim", "psnr", "mae", "ncc"]].mean().to_dict()
        q_summary["section"] = "quantitative"
        reports.append(q_summary)
        qdf.to_csv(Path(args.output_report).with_name("quantitative_per_case.csv"), index=False)

    if args.clinician_csv:
        cdf = evaluate_clinician(Path(args.clinician_csv))
        c_summary = {"clinical_mean": float(cdf["clinical_mean"].mean()), "section": "qualitative"}
        reports.append(c_summary)
        cdf.to_csv(Path(args.output_report).with_name("clinician_scored_cases.csv"), index=False)

    if not reports:
        raise ValueError("Provide at least one of --labeled_gt_csv or --clinician_csv")

    report_df = pd.DataFrame(reports)
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    print(f"Saved evaluation summary: {out_path}")


if __name__ == "__main__":
    main()
