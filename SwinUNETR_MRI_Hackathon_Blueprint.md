# SwinUNETR MRI Distortion-Correction Blueprint (Hackathon-Ready)

## 1) Problem Framing and Winning Strategy

Goal: Correct geometric distortion in single-shot EPI DWI (especially b0) using paired T1 + distorted b0 input, then optionally pass corrected synthesis into FSL TOPUP-style downstream correction.

## 2) Data Design

### 2.1 Input/Target Definition
- Inputs (2 channels):
  - Channel 0: T1w (registered to diffusion space or common canonical space)
  - Channel 1: distorted b0
- Target (1 channel): corrected/undistorted b0 (from opposite phase-encoding correction pipeline)

### 2.2 Recommended Dataset Splits
- Public labeled dataset (HCP-derived):
  - 70% train
  - 15% validation
  - 15% test
  - Subject-wise split only (no leakage across sessions/volumes)
- Local unlabeled dataset (289 subjects):
  - 20-30 subjects for clinician rubric pilot and protocol tuning
  - Remaining for blinded final qualitative scoring

### 2.3 Fast, Stable Preprocessing
- NIfTI harmonization:
  - Reorient to LAS or RAS consistently
  - Resample T1 and b0 to common spacing (start with 1.5 mm isotropic; keep 1.25 mm option for final runs)
- Intensity normalization:
  - T1: robust z-score within brain mask
  - b0: percentile clipping (0.5 to 99.5) then min-max or z-score
- Registration:
  - T1 to b0 space via rigid + affine (avoid expensive non-linear for first leaderboard runs)
  - Cache transforms and warped volumes to disk
- Crop strategy:
  - Brain bounding-box crop with fixed margin
  - Then patch-based training (for GPU memory efficiency)

## 3) Model Architecture (SwinUNETR)

### 3.1 Core Network
- Model: SwinUNETR 3D
- in_channels: 2
- out_channels: 1
- img_size (patch input): 96x96x96 (upgrade to 128x128x128 if memory allows)
- feature_size: 48 (or 24 for low VRAM)
- depths: [2, 2, 2, 2]
- num_heads: [3, 6, 12, 24]
- norm: instance
- drop_rate: 0.0, attn_drop_rate: 0.0, dropout_path_rate: 0.1
- use_checkpoint: true (if memory constrained)

### 3.2 Objective Function (Recommended)
Use a composite loss to balance voxel fidelity and structure:

L_total = 0.50 * L1 + 0.25 * SSIM_loss + 0.15 * Gradient_loss + 0.10 * NCC_loss

Notes:
- L1 improves stability vs MSE for robust reconstruction.
- SSIM preserves anatomical structure.
- Gradient loss keeps edges near skull base/temporal lobe sharper.
- NCC helps modality-aligned consistency under intensity shifts.

### 3.3 Optional Distortion-Field Head (Advanced)
If time allows, extend to dual-head multi-task:
- Head A: corrected b0 synthesis
- Head B: displacement field (phase-encoding axis constrained)
- Add smoothness regularizer on field head

This can improve geometric realism, but only attempt after stable single-head results.

## 4) Training Pipeline (Efficient and Reproducible)

### 4.1 Training Hyperparameters (Start Point)
- Optimizer: AdamW
- LR: 1e-4
- Weight decay: 1e-5
- Betas: (0.9, 0.999)
- Scheduler: cosine decay with 5-epoch warmup
- Epochs: 120
- Batch size: 1-2 (3D patches)
- Gradient accumulation: 2-4 steps to emulate larger batch
- AMP mixed precision: enabled
- Gradient clipping: 1.0
- Early stopping: patience 20 on validation composite metric
- Cross-validation: 5-fold (to match baseline reporting)

### 4.2 Augmentation (Distortion-Relevant)
- Random flip (exclude incompatible orientation flips if PE axis constraints apply)
- Small rotations (<= 10 degrees)
- Random bias field on T1 channel
- Random gamma/contrast on b0 channel
- Mild elastic or grid distortion (small magnitude) to improve robustness
- Random Gaussian noise (low sigma)

### 4.3 Validation Metrics (Labeled Public Data)
- PSNR
- SSIM
- MAE
- NCC
- Optional: Jacobian regularity if deformation field is predicted

Leaderboard-style composite:

Score = 0.35 * SSIM + 0.25 * NCC + 0.20 * PSNR_norm - 0.20 * MAE_norm

Normalize PSNR and MAE per validation distribution.

## 5) Inference and Post-Processing Pipeline

### 5.1 Inference Sequence
1. Load local subject (T1 + DWI/b0).
2. Apply cached preprocessing and T1-to-b0 alignment.
3. Run sliding-window SwinUNETR inference (overlap 0.5, Gaussian blending).
4. Generate synthesized undistorted b0.
5. Option A: direct corrected output for qualitative review.
6. Option B: combine synthesized + distorted b0 and run TOPUP-based correction chain.
7. Transform result back to native subject space for radiologist-friendly viewing.

### 5.2 Runtime Optimizations
- Precompute and cache all registration outputs once.
- Use persistent dataset caching.
- Use pinned memory + prefetch workers.
- Save model as TorchScript/ONNX for faster inference if stable.
- Batch inference by patches even for single subject.

## 6) Clinical Qualitative Evaluation Protocol (No Local GT)

Use 5-point Likert scoring by 2-3 clinicians for each criterion:
- Distortion reduction near skull base
- Distortion reduction in temporal lobes
- Anatomical alignment with T1
- Lesion confidence preservation (stroke/tumor visibility)
- Overall clinical acceptability

Protocol:
- Blind raters to method (baseline vs SwinUNETR).
- Randomize case order.
- Show paired viewer panels with standardized windowing.
- Compute:
  - Mean score per criterion
  - Overall mean
  - Inter-rater agreement (Cohen/Fleiss kappa)

Hackathon reporting metric:

Clinical_Score = Mean(5 criteria) + 0.1 * Kappa

## 7) System Architecture (End-to-End)

### 7.1 Modular Architecture
- Module A: Data Ingestion
- Module B: Preprocessing + Registration Cache
- Module C: Training Engine (SwinUNETR + experiments)
- Module D: Inference Engine (sliding window + export)
- Module E: Post-processing (TOPUP chain / native-space restore)
- Module F: Evaluation (quant + qualitative dashboards)

### 7.2 Logical Flow
1. Public data -> preprocess -> train/val/test.
2. Best checkpoint selection by composite validation score.
3. Local data -> preprocess -> infer -> post-process.
4. Clinician scoring app/sheets -> aggregated analytics.
5. Final report: baseline vs SwinUNETR + ablations.

## 8) Ablation Plan (What Judges Love)

Run these minimum ablations:
- A1: 3D U-Net baseline (reproduced)
- A2: SwinUNETR + L1 only
- A3: SwinUNETR + composite loss
- A4: SwinUNETR + composite loss + optimized preprocessing cache
- A5: (optional) dual-head field + image output

Report each on:
- Public quant metrics
- Local clinician score
- Inference time per subject

## 9) Hackathon Execution Plan

### 1
- Reproduce baseline 3D U-Net pipeline and metrics.
- Freeze preprocessing conventions and data split.

### 2
- Train SwinUNETR v1 (L1 or L1+SSIM).
- Validate and compare with baseline.

### 3
- Add full composite loss and augmentation tuning.
- Launch 5-fold or reduced-fold (if compute-limited) runs.

### 4
- Integrate efficient inference + cache + optional TOPUP chain.
- Prepare clinician scoring package.

### 5
- Clinician blinded evaluation.
- Aggregate quantitative + qualitative results.

### 6
- Final ablation table, visuals, and demo story.
- Build a concise presentation emphasizing clinical impact in low-resource settings.

## 10) Practical Defaults for Limited GPU

- Use patch size 96^3 and feature_size 24.
- Use AMP and gradient accumulation.
- Reduce depths if OOM persists.
- Keep registration affine-only for speed.
- Cache everything after first pass.

## 11) Submission-Ready Deliverables Checklist

- Reproducible training config (seed, splits, versions).
- Baseline vs SwinUNETR comparison table.
- 3-5 qualitative case panels (before/after/T1 overlay).
- Clinician rubric and agreement stats.
- Runtime table (preprocess time, inference time).
- Failure-case analysis and mitigation notes.
- One-slide impact statement for LMIC clinical workflow.

---

## Quick Start Recommendation

If we need one highest-probability path:
1. Implement SwinUNETR single-head (2->1 channels) with composite loss.
2. Focus on preprocessing cache + fast affine registration.
3. Run strong baseline comparison and clinician-blinded rubric.
4. Prioritize robust reporting over very complex model tricks.

This combination usually scores best in hackathons because it balances innovation, reliability, and clinical relevance.
