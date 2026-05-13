from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import MAEMetric, PSNRMetric, SSIMMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import load_config
from src.dataset import build_loader, train_transforms, val_transforms
from src.losses import CompositeCorrectionLoss
from src.model import build_swinunetr
from src.utils import ensure_dir, seed_everything


def concat_inputs(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t1 = batch["t1"].to(device)
    b0 = batch["b0_distorted"].to(device)
    target = batch["b0_corrected"].to(device)
    x = torch.cat([t1, b0], dim=1)
    return x, target


def validate(model: torch.nn.Module, loader, device: torch.device, roi_size: tuple[int, int, int]) -> dict[str, float]:
    model.eval()
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()

    with torch.no_grad():
        for batch in loader:
            x, y = concat_inputs(batch, device)
            pred = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )
            pred = torch.clamp(pred, 0.0, 1.0)
            ssim_metric(pred, y)
            psnr_metric(pred, y)
            mae_metric(pred, y)

    return {
        "ssim": float(ssim_metric.aggregate().item()),
        "psnr": float(psnr_metric.aggregate().item()),
        "mae": float(mae_metric.aggregate().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SwinUNETR for b0 distortion correction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", default="outputs", help="Output root for checkpoints and logs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"], cfg["project"].get("deterministic", True))

    output_dir = Path(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "reports")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = tuple(cfg["data"]["patch_size"])
    train_loader = build_loader(
        cfg["data"]["train_manifest"],
        train_transforms(patch_size),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        cache_rate=cfg["data"]["cache_rate"],
        shuffle=True,
    )
    val_loader = build_loader(
        cfg["data"]["val_manifest"],
        val_transforms(),
        batch_size=1,
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
        cache_rate=cfg["data"]["cache_rate"],
        shuffle=False,
    )

    model = build_swinunetr(cfg).to(device)
    loss_fn = CompositeCorrectionLoss(
        l1_weight=cfg["loss"]["l1_weight"],
        ssim_weight=cfg["loss"]["ssim_weight"],
        grad_weight=cfg["loss"]["grad_weight"],
        ncc_weight=cfg["loss"]["ncc_weight"],
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["scheduler"]["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    max_epochs = int(cfg["scheduler"]["max_epochs"])
    grad_accum_steps = int(cfg["train"]["grad_accum_steps"])
    clip_norm = float(cfg["train"]["grad_clip_norm"])
    patience = int(cfg["train"]["early_stopping_patience"])

    best_score = float("-inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}"), start=1):
            x, y = concat_inputs(batch, device)
            with torch.autocast(device_type=device.type, enabled=bool(cfg["train"]["amp"])):
                pred = model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                loss, _ = loss_fn(pred, y)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * grad_accum_steps

        scheduler.step()
        val_metrics = validate(model, val_loader, device, patch_size)

        # Higher is better, with MAE penalty.
        val_score = (0.5 * val_metrics["ssim"]) + (0.3 * (val_metrics["psnr"] / 50.0)) - (0.2 * val_metrics["mae"])
        epoch_log = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            **val_metrics,
            "val_score": val_score,
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch}: loss={epoch_log['train_loss']:.4f}, "
            f"ssim={val_metrics['ssim']:.4f}, psnr={val_metrics['psnr']:.3f}, "
            f"mae={val_metrics['mae']:.4f}, score={val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            stale_epochs = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "best_score": best_score}, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print("Early stopping triggered.")
            break

    with (log_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
