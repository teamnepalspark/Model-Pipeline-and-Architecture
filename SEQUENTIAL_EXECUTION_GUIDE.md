# Sequential Execution Guide

## 1) Install dependencies

```powershell
c:/python314/python.exe -m pip install -r requirements.txt
```

## 2) Prepare manifests

Create your real manifests by copying these templates and replacing paths:
- data/manifests/train.example.json
- data/manifests/local.example.json

For validation in the runner, create:
- data/manifests/val.json

## 3) Run full pipeline in sequence

```powershell
c:/python314/python.exe run_pipeline.py `
  --train_source_manifest data/manifests/train.json `
  --val_source_manifest data/manifests/val.json `
  --local_source_manifest data/manifests/local.json `
  --config configs/swinunetr_baseline.yaml `
  --output_root outputs
```

This executes:
1. Preprocess train data
2. Preprocess val data
3. Preprocess local data
4. Train SwinUNETR
5. Infer on local data

## 4) Evaluate quantitative metrics (if GT exists)

```powershell
c:/python314/python.exe -m src.evaluate `
  --pred_dir outputs/predictions `
  --labeled_gt_csv your_labeled_gt.csv `
  --output_report outputs/reports/evaluation_summary.csv
```

`your_labeled_gt.csv` format:
- id
- gt_path

## 5) Evaluate clinician qualitative metrics

Fill:
- templates/clinician_scoring_template.csv

Then run:

```powershell
c:/python314/python.exe -m src.evaluate `
  --pred_dir outputs/predictions `
  --clinician_csv templates/clinician_scoring_template.csv `
  --output_report outputs/reports/evaluation_summary.csv
```
