from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def robust_zscore(img: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img, [1.0, 99.0])
    img = np.clip(img, p1, p99)
    mu = float(img.mean())
    std = float(img.std()) + 1e-6
    return (img - mu) / std


def minmax(img: np.ndarray) -> np.ndarray:
    p0, p99 = np.percentile(img, [0.5, 99.5])
    img = np.clip(img, p0, p99)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def pad_to_min_shape(img: np.ndarray, min_shape: tuple[int, int, int]) -> np.ndarray:
    pads = []
    for dim, min_dim in zip(img.shape[:3], min_shape):
        total = max(0, min_dim - dim)
        before = total // 2
        after = total - before
        pads.append((before, after))
    return np.pad(img, pad_width=tuple(pads), mode="constant", constant_values=0)


def resample_to_shape(img: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(img.shape[:3]) == tuple(target_shape):
        return img
    factors = [t / s for s, t in zip(img.shape[:3], target_shape)]
    return zoom(img, zoom=factors, order=1)


def save_nifti_like(reference_nii: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> None:
    out = nib.Nifti1Image(data.astype(np.float32), affine=reference_nii.affine, header=reference_nii.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))


def process_case(case: dict, out_root: Path, has_target: bool = True, min_shape: tuple[int, int, int] = (64, 64, 64)) -> dict:
    case_id = case["id"]
    t1_nii = nib.load(case["t1"])
    b0_nii = nib.load(case["b0_distorted"])

    t1 = t1_nii.get_fdata(dtype=np.float32)
    b0 = b0_nii.get_fdata(dtype=np.float32)

    # Ensure channel alignment by resampling T1 to b0 grid first.
    t1 = resample_to_shape(t1, tuple(b0.shape[:3]))

    t1 = pad_to_min_shape(t1, min_shape)
    b0 = pad_to_min_shape(b0, min_shape)

    # Lightweight normalization only; registration should be cached upstream if available.
    t1_n = robust_zscore(t1)
    b0_n = minmax(b0)

    out_case = out_root / case_id
    t1_out = out_case / "t1_norm.nii.gz"
    b0_out = out_case / "b0_distorted_norm.nii.gz"
    save_nifti_like(t1_nii, t1_n, t1_out)
    save_nifti_like(b0_nii, b0_n, b0_out)

    out = {
        "id": case_id,
        "t1": str(t1_out),
        "b0_distorted": str(b0_out),
    }

    if has_target and "b0_corrected" in case and case["b0_corrected"]:
        tgt_nii = nib.load(case["b0_corrected"])
        tgt = tgt_nii.get_fdata(dtype=np.float32)
        tgt = resample_to_shape(tgt, tuple(b0_nii.shape[:3]))
        tgt = pad_to_min_shape(tgt, min_shape)
        tgt = minmax(tgt)
        tgt_out = out_case / "b0_corrected_norm.nii.gz"
        save_nifti_like(tgt_nii, tgt, tgt_out)
        out["b0_corrected"] = str(tgt_out)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MRI cases and write normalized cache manifest.")
    parser.add_argument("--input_manifest", required=True, help="Path to source JSON manifest.")
    parser.add_argument("--output_manifest", required=True, help="Path to output JSON manifest.")
    parser.add_argument("--output_cache_dir", required=True, help="Directory for cached normalized NIfTI files.")
    parser.add_argument("--has_target", action="store_true", help="Set for labeled training/validation/test data.")
    parser.add_argument("--min_shape", nargs=3, type=int, default=(64, 64, 64), help="Minimum XYZ shape after padding.")
    args = parser.parse_args()

    in_manifest = Path(args.input_manifest)
    out_manifest = Path(args.output_manifest)
    out_root = Path(args.output_cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with in_manifest.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    processed = []
    for case in tqdm(cases, desc="Preprocessing"):
        processed.append(process_case(case, out_root, has_target=args.has_target, min_shape=tuple(args.min_shape)))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    print(f"Wrote {len(processed)} cached records to: {out_manifest}")


if __name__ == "__main__":
    main()
