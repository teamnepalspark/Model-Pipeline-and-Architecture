from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_step(cmd: list[str], step_name: str) -> None:
    print(f"\n=== {step_name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential runner for SwinUNETR MRI pipeline.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--config", default="configs/swinunetr_baseline.yaml")
    parser.add_argument("--train_source_manifest", required=False)
    parser.add_argument("--val_source_manifest", required=False)
    parser.add_argument("--local_source_manifest", required=False)
    parser.add_argument("--output_root", default="outputs")
    args = parser.parse_args()

    py = args.python_bin
    output_root = Path(args.output_root)
    data_cache = output_root / "cache"
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    train_cached = manifests / "train_cached.json"
    val_cached = manifests / "val_cached.json"
    local_cached = manifests / "local_cached.json"

    # If manifests not provided, generate them from data/ folder
    default_manifests_dir = Path("data") / "manifests"
    if not args.train_source_manifest or not args.val_source_manifest or not args.local_source_manifest:
        print("One or more source manifests not provided — auto-generating using src.generate_manifests")
        run_step([py, "-m", "src.generate_manifests", "--data_root", "data", "--out_dir", str(default_manifests_dir)], "Generate Manifests")
        args.train_source_manifest = args.train_source_manifest or str(default_manifests_dir / "train.json")
        args.val_source_manifest = args.val_source_manifest or str(default_manifests_dir / "val.json")
        args.local_source_manifest = args.local_source_manifest or str(default_manifests_dir / "local.json")

    run_step(
        [
            py,
            "-m",
            "src.preprocess",
            "--input_manifest",
            args.train_source_manifest,
            "--output_manifest",
            str(train_cached),
            "--output_cache_dir",
            str(data_cache / "train"),
            "--has_target",
        ],
        "Preprocess Train",
    )

    run_step(
        [
            py,
            "-m",
            "src.preprocess",
            "--input_manifest",
            args.val_source_manifest,
            "--output_manifest",
            str(val_cached),
            "--output_cache_dir",
            str(data_cache / "val"),
            "--has_target",
        ],
        "Preprocess Validation",
    )

    run_step(
        [
            py,
            "-m",
            "src.preprocess",
            "--input_manifest",
            args.local_source_manifest,
            "--output_manifest",
            str(local_cached),
            "--output_cache_dir",
            str(data_cache / "local"),
        ],
        "Preprocess Local",
    )

    with Path(args.config).open("r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    runtime_cfg["data"]["train_manifest"] = str(train_cached)
    runtime_cfg["data"]["val_manifest"] = str(val_cached)
    runtime_cfg_path = manifests / "runtime_config.yaml"
    with runtime_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)

    run_step(
        [py, "-m", "src.train", "--config", str(runtime_cfg_path), "--output_dir", str(output_root)],
        "Train",
    )

    run_step(
        [
            py,
            "-m",
            "src.infer",
            "--config",
            str(runtime_cfg_path),
            "--checkpoint",
            str(output_root / "checkpoints" / "best_model.pt"),
            "--input_manifest",
            str(local_cached),
            "--output_dir",
            str(output_root / "predictions"),
        ],
        "Infer Local",
    )

    print("\nPipeline completed successfully.")
    print("Run evaluation separately once clinician CSV is available.")


if __name__ == "__main__":
    main()
