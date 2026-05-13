# Implementation Runbook: SwinUNETR MRI Distortion Correction

## Recommended Project Structure

.
|-- configs/
|   `-- swinunetr_baseline.yaml
|-- data/
|   |-- manifests/
|   |   |-- train.json
|   |   |-- val.json
|   |   `-- test.json
|   `-- cache/
|-- src/
|   |-- preprocess.py
|   |-- dataset.py
|   |-- model.py
|   |-- losses.py
|   |-- train.py
|   |-- infer.py
|   `-- evaluate.py
|-- outputs/
|   |-- checkpoints/
|   |-- predictions/
|   `-- reports/
`-- README.md

## End-to-End Pipeline

1. Build manifests containing paths for each subject:
   - t1
   - b0_distorted
   - b0_corrected (public data only)

2. Preprocess once and cache:
   - Reorient
   - Resample
   - Register T1 to b0
   - Normalize intensities
   - Save cache outputs

3. Train SwinUNETR:
   - 2-channel input, 1-channel output
   - Composite loss
   - AMP + gradient accumulation
   - Save top checkpoints

4. Validate and select best model:
   - SSIM, PSNR, MAE, NCC
   - Composite ranking score

5. Run local inference:
   - Sliding window inference
   - Optional TOPUP downstream correction
   - Restore to subject native space

6. Clinician scoring workflow:
   - Blinded comparison against baseline outputs
   - 5-point rubric across 5 criteria
   - Agreement statistics

## Minimal Starter Tasks

Task 1: Preprocessing cache
- Implement preprocessing script with deterministic transforms.
- Save transformed T1 and b0 volumes and transform metadata.

Task 2: Model and loss
- Implement SwinUNETR wrapper and composite loss class.
- Add config-driven construction.

Task 3: Training
- Build training loop with AMP and checkpointing.
- Add fold support for 5-fold cross-validation.

Task 4: Inference and export
- Implement sliding-window inference script.
- Save NIfTI outputs plus per-case metadata JSON.

Task 5: Evaluation
- Quantitative script for public data.
- Qualitative rubric template and aggregate script for local data.

## Hackathon Priorities

1. Reproducibility first:
   - Fixed seeds and deterministic preprocessing.

2. Speed second:
   - Full caching of expensive registration steps.

3. Metrics and story third:
   - Show baseline vs SwinUNETR gains on both quantitative and clinician-centered outcomes.

## Risks and Mitigations

Risk: Domain shift from HCP to local clinical data.
Mitigation: Add intensity augmentations and test-time normalization checks.

Risk: Slow preprocessing and inference.
Mitigation: Cache transforms and use mixed precision + patch inference.

Risk: No local ground truth.
Mitigation: Strict blinded clinician protocol with inter-rater agreement.
