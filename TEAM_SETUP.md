# GitHub Setup Instructions for Hackathon Team

## Prerequisites
- GitHub account for each team member
- Git installed on your machine
- Python 3.9+ installed

## Step 1: Create GitHub Repository (Team Lead Only)

1. Go to https://github.com/new
2. **Repository name**: `MRI-SwinUNETR-Hackathon` (or similar)
3. **Description**: "SwinUNETR for b0 MRI distortion correction - Hackathon submission"
4. **Visibility**: Public (for reproducibility) or Private (for confidentiality)
5. **Do NOT initialize with README** (we already have one)
6. Click "Create repository"
7. Share the repository URL with team members: `https://github.com/[USERNAME]/MRI-SwinUNETR-Hackathon`

## Step 2: Push Local Repository to GitHub (Team Lead)

From your local machine in PowerShell:

```powershell
cd "c:\Users\HP\OneDrive\Desktop\Model and Pipeline"
git remote add origin https://github.com/[USERNAME]/MRI-SwinUNETR-Hackathon.git
git branch -M main
git push -u origin main
git push -u origin hackathon/swinunetr-distortion-correction
git branch -v
```

**Expected output:**
```
  hackathon/swinunetr-distortion-correction 8dc4e81 Initial SwinUNETR MRI pipeline
* main                                       8dc4e81 Initial SwinUNETR MRI pipeline
```

## Step 3: Team Members Clone & Setup

Each team member runs:

```powershell
# Clone the repository
git clone https://github.com/[USERNAME]/MRI-SwinUNETR-Hackathon.git "Model and Pipeline"
cd "Model and Pipeline"

# Switch to hackathon branch
git checkout hackathon/swinunetr-distortion-correction

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure git user (one-time)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Verify setup
python -c "import torch; import monai; print('✓ Setup complete')"
```

## Step 4: Add GitHub Collaborators

On GitHub, go to:  
**Settings → Collaborators → Add people**

Add each team member by GitHub username. They should receive an invitation and accept it.

## Step 5: Start Development

**Typical workflow:**
```powershell
# Sync with team's latest work
git pull origin hackathon/swinunetr-distortion-correction

# Create feature branch for your task
git checkout -b feature/your-task-name

# Make changes, test locally
# ... edit files ...

# Commit and push
git add .
git commit -m "Clear description of your change"
git push origin feature/your-task-name

# Open Pull Request on GitHub
# - Go to https://github.com/[USERNAME]/MRI-SwinUNETR-Hackathon/pulls
# - Click "New pull request"
# - Select base: hackathon/swinunetr-distortion-correction
# - Select compare: feature/your-task-name
# - Add title and description
# - Assign reviewers (other team members)
# - Click "Create pull request"
```

## Step 6: Code Review Process

**For reviewers:**
1. Check the Pull Request on GitHub
2. Review code changes in the "Files changed" tab
3. Leave comments or approve
4. Team lead merges PR after approval

**For code authors:**
1. Address review feedback
2. Push new commits to the same branch
3. Comments on GitHub to discuss changes
4. Request re-review once ready

## Step 7: Data Sharing Strategy

Since MRI `.nii.gz` files are large and git-ignored:

**Option 1: Google Drive (Recommended for quick access)**
```
- Create shared Google Drive folder "Hackathon-MRI-Data"
- Each team member adds subjects to data/ folder locally
- Run: python -m src.generate_manifests
- Commit only the updated data/manifests/*.json to GitHub
```

**Option 2: Kaggle Dataset (For reproducibility)**
```
- Upload dataset to Kaggle
- Add download script: notebooks/download_kaggle_data.ipynb
- Team members run the notebook to fetch data
```

**Option 3: GitHub Releases (For official submissions)**
```
- After hackathon, create a GitHub Release
- Attach representative sample data and final model
- Include MD5 checksums for validation
```

## Common Team Workflows

### Share a new training config
```powershell
# Edit configs/swinunetr_baseline.yaml
git add configs/swinunetr_baseline.yaml
git commit -m "Update baseline config: LR from 1e-4 to 5e-4"
git push origin feature/baseline-tuning
# Create PR for team feedback
```

### Add trained model weights to shared storage
```powershell
# Do NOT commit .pt files (git-ignored)
# Instead, upload to:
# 1. GitHub Releases (for official submission)
# 2. Google Drive (for team sharing)
# 3. Kaggle Models (for public access)
# Add a note to outputs/README.md with download links
```

### Merge a completed feature
```powershell
# After PR is approved and tests pass:
# On GitHub: Click "Merge pull request" → "Confirm merge"
# OR locally:
git checkout hackathon/swinunetr-distortion-correction
git pull origin hackathon/swinunetr-distortion-correction
git merge feature/completed-task
git push origin hackathon/swinunetr-distortion-correction
```

## Verification Commands

```powershell
# Check you're on the right branch
git branch -v

# See recent commits
git log --oneline -10

# View all branches (local and remote)
git branch -a

# See which files changed in last commit
git diff HEAD~1 --name-only

# Check remote configuration
git remote -v
```

## Troubleshooting

### Remote URL wrong?
```powershell
git remote set-url origin https://github.com/[CORRECT_USERNAME]/MRI-SwinUNETR-Hackathon.git
git remote -v  # Verify
```

### Accidentally on master/main?
```powershell
git checkout hackathon/swinunetr-distortion-correction
```

### Push rejected (behind remote)?
```powershell
git pull origin hackathon/swinunetr-distortion-correction
git push origin hackathon/swinunetr-distortion-correction
```

### Lost your work?
```powershell
git reflog                    # See all commits (even deleted ones)
git checkout <commit-hash>   # Restore to any point
```

---

**Repository**: https://github.com/[USERNAME]/MRI-SwinUNETR-Hackathon  
**Main Branch**: `hackathon/swinunetr-distortion-correction`  
**For Help**: See CONTRIBUTING.md or README.md
