# SwinUNETR MRI Distortion Correction

This project implements a hackathon-ready MRI distortion-correction pipeline using SwinUNETR with a sequential workflow:

1. Manifest generation from local MRI folders.
2. Preprocessing (T1-to-b0 resampling, normalization, padding).
3. Training (SwinUNETR, composite loss).
4. Inference (sliding-window prediction).
5. Evaluation (quantitative and clinician rubric).

## Quick Start (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.generate_manifests --data_root data --out_dir data/manifests
.\.venv\Scripts\python.exe run_pipeline.py --python_bin .\.venv\Scripts\python.exe --config configs/swinunetr_smoke.yaml --train_source_manifest data/manifests/train.json --val_source_manifest data/manifests/val.json --local_source_manifest data/manifests/local.json --output_root outputs/smoke
```

## Full Configuration

- Primary config: `configs/swinunetr_baseline.yaml`
- Smoke config: `configs/swinunetr_smoke.yaml`

### Colab

Use notebook: `notebooks/colab_swinunetr_pipeline.ipynb`
- When a notebook is useful:

- Quick demos, exploratory analysis, visualization, or interactive Colab demos for judges.
- Sharing runnable examples with non-technical reviewers (Colab).
  
### Recommendation:

Keep run_pipeline.py and src/* as the canonical, reproducible interface.
Also keep a lightweight Colab notebook for demos and presentation (add to notebooks).
