

## Repository Structure

```
Model and Pipeline/
├── configs/                          # Training configs
│   ├── swinunetr_baseline.yaml      # Full training config
│   └── swinunetr_smoke.yaml         # Quick smoke test config
├── src/                              # Source code
│   ├── preprocess.py                # Data preprocessing & alignment
│   ├── dataset.py                   # MONAI datasets & transforms
│   ├── model.py                     # SwinUNETR model builder
│   ├── losses.py                    # Composite loss functions
│   ├── train.py                     # Training loop
│   ├── infer.py                     # Inference pipeline
│   ├── evaluate.py                  # Evaluation metrics
│   └── generate_manifests.py        # Auto-manifest generation
├── data/                             # MRI data (not synced, git-ignored)
│   └── manifests/                   # Data indices (synced)
├── outputs/                          # Trained models & results (git-ignored)
├── notebooks/                        # Jupyter notebooks
│   └── colab_swinunetr_pipeline.ipynb
├── templates/                        # Clinician scoring sheets
├── run_pipeline.py                  # Sequential orchestrator
├── requirements.txt                 # Python dependencies
└── README.md                        # Quick start guide
```

## Branch Strategy

- **master**: Stable baseline (do not commit directly here).
- **hackathon/swinunetr-distortion-correction**: Main development branch for hackathon.
- **feature/\***: Individual feature branches (e.g., `feature/baseline-comparison`, `feature/clinician-ui`).

## Setup Instructions for Team Members

### 1. Clone the repository
```powershell
git clone <REPO_URL> "Model and Pipeline"
cd "Model and Pipeline"
git checkout hackathon/swinunetr-distortion-correction
```

### 2. Set up Python virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure git user (one-time)
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Workflow for Team Collaboration

### For small changes (bug fixes, config tweaks):
```powershell
git pull origin hackathon/swinunetr-distortion-correction
# Make edits
git add .
git commit -m "Brief description of change"
git push origin hackathon/swinunetr-distortion-correction
```

### For feature work (new components, improvements):
```powershell
git pull origin hackathon/swinunetr-distortion-correction
git checkout -b feature/your-feature-name
# Make edits
git add .
git commit -m "Descriptive commit message"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub for review
```

## Commit Message Convention

Write clear commit messages describing what and why:

```
Short summary (50 chars max)

More detailed explanation if needed (wrapped at 72 chars).
Reference issue/task if applicable: #123

Examples:
- "Add sliding-window inference for validation"
- "Fix T1-to-b0 resampling for shape mismatch"
- "Increase default patch size to 64x64x64"
```

## Roles and Responsibilities

| Role | Tasks | Branch/Files |
|------|-------|------|
| **Data Eng** | Manifest generation, preprocessing, data validation | `src/generate_manifests.py`, `src/preprocess.py`, `data/manifests/` |
| **Model Eng** | Model tuning, loss functions, training | `src/model.py`, `src/losses.py`, `src/train.py`, `configs/` |
| **Infra/Eval** | Inference, evaluation, reporting | `src/infer.py`, `src/evaluate.py`, `templates/` |
| **Demo/Viz** | Colab notebooks, clinician UI, slides | `notebooks/`, presentation deck |

## Key Development Tasks

### Phase 1: Baseline Reproduction
- [ ] Verify smoke test passes on all machines
- [ ] Add more labeled training subjects to `data/`
- [ ] Regenerate manifests with full dataset
- [ ] Run full baseline training (20 epochs first)
- [ ] Create quantitative results table

### Phase 2: SwinUNETR Optimization
- [ ] Ablation study (loss components)
- [ ] Hyperparameter tuning (LR, batch size, patch size)
- [ ] Compare against baseline on validation set
- [ ] Document improvements

### Phase 3: Clinical Validation 
- [ ] Prepare clinician scoring UI/CSV template
- [ ] Run inference on local unlabeled dataset
- [ ] Collect blinded clinician ratings
- [ ] Aggregate inter-rater agreement metrics

### Phase 4: Presentation & Deployment
- [ ] Create stage presentation slides
- [ ] Generate before/after qualitative panels
- [ ] Prepare demo notebook for Colab
- [ ] Document final results and lessons learned

## Common Commands

### Check current status
```powershell
git status
git log --oneline -5
git branch -v
```

### Sync with team
```powershell
git pull origin hackathon/swinunetr-distortion-correction
```

### View changes before commit
```powershell
git diff
git diff --staged
```

### Undo uncommitted changes
```powershell
git checkout -- <file>    # Undo one file
git reset --hard          # Undo all changes (CAUTION!)
```

### Merge latest from main branch
```powershell
git fetch origin
git merge origin/hackathon/swinunetr-distortion-correction
```

## Data Management

- **DO NOT commit large data files** (`.nii.gz` volumes are git-ignored).
- **DO commit manifests** (`data/manifests/*.json`) so team stays synced on data indices.
- **Share data externally**:
  - Google Drive folder (shared link for all team members).
  - Kaggle dataset (for public reproducibility).
  - Local NAS/server if available.

### Instructions to add new training data:
1. Add subject folders to `data/` locally (not committed).
2. Run: `python -m src.generate_manifests --data_root data --out_dir data/manifests`
3. Commit only the updated `data/manifests/train.json`, `val.json`, `local.json`.
4. Notify team about new data via Slack/Discord.

## Testing Before Push

Always test locally before pushing:

```powershell
# Quick smoke test (1 epoch)
python run_pipeline.py --python_bin python --config configs/swinunetr_smoke.yaml --train_source_manifest data/manifests/train.json --val_source_manifest data/manifests/val.json --local_source_manifest data/manifests/local.json --output_root outputs/smoke

# Check for Python syntax errors
python -m compileall src run_pipeline.py
```



