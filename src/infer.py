from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference

from src.config import load_config
from src.dataset import infer_transforms, load_manifest
from src.model import build_swinunetr
from src.utils import ensure_dir


def save_prediction(reference_path: str, pred: np.ndarray, out_path: Path) -> None:
    ref = nib.load(reference_path)
    pred_img = nib.Nifti1Image(pred.astype(np.float32), affine=ref.affine, header=ref.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pred_img, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SwinUNETR inference on local MRI dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_dir", default="outputs/predictions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_swinunetr(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = load_manifest(args.input_manifest)
    ds = CacheDataset(data=data, transform=infer_transforms(), cache_rate=1.0, num_workers=2)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    out_dir = ensure_dir(args.output_dir)
    roi_size = tuple(cfg["data"]["patch_size"])
    sw_batch_size = int(cfg["inference"]["sw_batch_size"])
    overlap = float(cfg["inference"]["overlap"])

    with torch.no_grad():
        for batch in loader:
            case_id = batch["id"][0]
            x = torch.cat([batch["t1"].to(device), batch["b0_distorted"].to(device)], dim=1)
            pred = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode=cfg["inference"].get("mode", "gaussian"),
            )
            pred = torch.clamp(pred, 0.0, 1.0)
            pred_np = pred[0, 0].detach().cpu().numpy()

            out_path = Path(out_dir) / f"{case_id}_b0_corrected_pred.nii.gz"
            save_prediction(batch["b0_distorted_meta_dict"]["filename_or_obj"][0], pred_np, out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
