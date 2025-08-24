# Running WellTestAnalysis_Colab.ipynb in Google Colab

This README explains how to run the enhanced `WellTestAnalysis_Colab.ipynb` (v1.2) in Google Colab for petroleum engineering classroom sessions.

## What's New in Version 1.2

The enhanced notebook now includes:
- **Multi-criteria fit quality assessment** combining RMSE, relative error %, and R²
- **Educational R² vs RMSE analysis** explaining why R² can mislead engineers
- **Enhanced visualizations** with fit quality color coding and error bands
- **Robust optimization** handling edge cases and small data groups
- **Comprehensive classroom exercises** including engineering interpretation scenarios
- **Practical demonstrations** showing when to trust models despite poor R² values

## Getting Started Options

This section explains three easy ways to open the enhanced notebook in Google Colab:

## Option A — Upload the notebook directly (fast, single-session)
1. Open Google Colab: https://colab.research.google.com
2. Click "File → Upload notebook" and choose `WellTestAnalysis_Colab.ipynb` from your machine.
3. Upload the CSV data file `Test-1 welltestdata.csv` using the Files pane in Colab.
4. Run the cells sequentially. The install cell (cell 3) includes packages, but Colab usually has them preinstalled.

**Recommended for**: Single classroom session, quick demos, or when you have files locally.

## Option B — Put the notebook in Google Drive (recommended for classes)
1. Copy both `WellTestAnalysis_Colab.ipynb` and `Test-1 welltestdata.csv` into your Google Drive (e.g., `My Drive/ColabNotebooks/`).
2. In Colab, run the drive mount cell (cell 16) or paste this into a code cell:

```python
from google.colab import drive
drive.mount('/content/drive')
# Copy files from Drive into the session folder (update paths as needed)
!cp "/content/drive/My Drive/ColabNotebooks/WellTestAnalysis_Colab.ipynb" ./
!cp "/content/drive/My Drive/ColabNotebooks/Test-1 welltestdata.csv" ./
!ls -l *.ipynb *.csv
```

3. Open the notebook from the Files pane.

**Benefits**: Students keep personal copies, can resume work later, and access enhanced features across sessions.

## Option C — Fetch from GitHub (if you publish the repo)
If you push this repository to GitHub, students can open the enhanced notebook directly in Colab:

- **Colab UI**: File → Open notebook → GitHub tab → paste `owner/repo` or a notebook path.

- **Direct fetch** in a Colab code cell (replace `<raw-url>`):

```bash
# Fetch both notebook and data file
!wget -O WellTestAnalysis_Colab.ipynb "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/path/WellTestAnalysis_Colab.ipynb"
!wget -O "Test-1 welltestdata.csv" "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/path/Test-1%20welltestdata.csv"
```

**Benefits**: Always get the latest enhanced version with all v1.2 improvements.

## Key Features for Instructors

The enhanced notebook (v1.2) provides rich educational content:

### Technical Learning Objectives
- **IPR Model Implementation**: Three-case composite model with automatic flow regime selection
- **Optimization Techniques**: Multiple methods with robust bounds and fallback strategies  
- **Data Quality Assessment**: Correlation analysis and validation with clear warnings
- **Physical Constraints**: Enforcement of engineering realities in mathematical models

### Engineering Insight Development
- **R² vs RMSE Analysis**: Critical section explaining why statistical metrics can mislead engineers
- **Multi-criteria Assessment**: Teaching students to evaluate models using practical accuracy measures
- **Field Application Focus**: Examples showing when to trust models for field development decisions
- **Edge Case Handling**: Demonstrating robust engineering practices with challenging data

### Classroom Exercise Structure
- **Core Technical Exercises**: Sensitivity analysis, bubble point effects, data quality investigation
- **Advanced Interpretation**: R² vs RMSE comparisons, multi-criteria fit assessment
- **Discussion Questions**: Engineering judgment scenarios for field applications
- **Practical Demonstrations**: Side-by-side comparison of old vs new assessment methods

## Quick Start for Students

**Recommended sequence for 50-90 minute class sessions:**
1. Run cells 1-5 (setup and imports)
2. Run cell 7 (enhanced optimizer class with v1.2 features)  
3. Upload data file and run cells 16-17 (data loading)
4. Run cell 19 (optimization with enhanced metrics)
5. Run cell 21 (enhanced visualizations with fit quality insights)
6. Explore cells 8-10 (R² vs RMSE educational content)
7. Work through classroom exercises (cell 11) based on instructor assignment

## Publishing the notebook and CSV to GitHub (so students can fetch them)

If you want students to fetch the notebook and CSV directly from GitHub (repository: https://github.com/SPE-PFAC01/ALCE), you can push the files from this workspace to that repo. Below are two options: push from this machine, or let me prepare a PR if you grant access.

Important assumptions and notes:
- I assume the repository uses the `main` branch. If the repo uses a different default branch, replace `main` below with the correct branch name.
- You must have push access to `SPE-PFAC01/ALCE`. These commands run locally (or in CI) where your Git credentials are configured.

Option 1 — Push the files yourself (PowerShell commands):

Open a PowerShell terminal in the project folder and run:

```powershell
# 1) Initialize a repo (if not already a git repo) or fetch latest
git init
git remote add origin https://github.com/SPE-PFAC01/ALCE.git
git fetch origin
git checkout -b add-colab-notebook

# 2) Copy the notebook and sample CSV into the repository layout you prefer
# (If you're already in the repo root and files exist, skip the copy step)
# Example: copy files from current workspace into repo working tree
# (Adjust the source paths if needed)
cp "WellTestAnalysis_Colab.ipynb" "data/Test-1 welltestdata.csv" .

# 3) Add, commit and push
git add WellTestAnalysis_Colab.ipynb data/Test-1\ welltestdata.csv README_COLAB.md
git commit -m "Add Colab notebook and example CSV + README updates"
git push -u origin add-colab-notebook
```

Then open GitHub and create a pull request from `add-colab-notebook` → `main`, or merge directly if you have rights.

Option 2 — Students fetch raw files from GitHub (wget example)

Once the files are on GitHub in the repo `SPE-PFAC01/ALCE` (branch `main` and path e.g. `WellTestAnalysis/`), students can fetch them in Colab using the raw URL. Example (replace path/branch as appropriate):

```bash
# Replace <owner>,<repo>,<branch>,<path> accordingly
!wget -O WellTestAnalysis_Colab.ipynb "https://raw.githubusercontent.com/SPE-PFAC01/ALCE/main/WellTestAnalysis/WellTestAnalysis_Colab.ipynb"
!wget -O "Test-1 welltestdata.csv" "https://raw.githubusercontent.com/SPE-PFAC01/ALCE/main/WellTestAnalysis/data/Test-1%20welltestdata.csv"
```

Notes on URL encoding: spaces in filenames must be percent-encoded as `%20` (see the second wget example above).

Option 3 — I can prepare a patch/branch and PR for you

I cannot push to your GitHub repo from here without access to your GitHub credentials or integration. If you want me to prepare a branch/PR automatically, you can either:
- Provide a GitHub repo link with write access via an integration I can use (not recommended to share tokens in chat), or
- Create a new branch locally with the commands above and run them, or
- Ask me to generate a clean patch file (diff) or a zip of files that you or someone with access can apply and push. If you'd like the patch/zip, tell me your preferred format and I'll create it.

---
File: `README_COLAB.md` — Guide for running the enhanced `WellTestAnalysis_Colab.ipynb` (v1.2) in Google Colab.

## Enhanced Version Notes

**v1.2 Improvements over Basic Version:**
- **Multi-criteria fit quality assessment** replacing R²-only evaluation
- **Educational R² vs RMSE content** explaining engineering vs statistical model assessment  
- **Enhanced visualizations** with color-coded fit quality and error analysis
- **Robust optimization** handling edge cases (small groups, identical data, correlation issues)
- **Comprehensive classroom exercises** including advanced engineering interpretation
- **Practical demonstration cells** showing when R² misleads and multi-criteria assessment succeeds

**For Instructors:**
- Use the "Understanding R² vs RMSE" section (cell 9) to teach critical engineering judgment
- Assign exercises from cell 11 based on desired learning depth (50-90 minute sessions)
- Run the practical demonstration (cell 10) to show side-by-side old vs new assessment methods
- Encourage student discussion about engineering vs statistical model evaluation approaches

**System Requirements:**
- Google Colab (free tier sufficient)
- CSV data file: `Test-1 welltestdata.csv` (included with notebook)
- No additional package installations required (uses standard scientific Python stack)

## Troubleshooting Common Issues

**"File not found" errors:**
- Ensure both `.ipynb` and `.csv` files are uploaded to the same Colab session directory
- Check file names match exactly (case-sensitive): `Test-1 welltestdata.csv`
- Use `!ls -l *.csv` in a code cell to verify file presence

**Package import errors:**
- Run cell 3 (install dependencies) if needed: `!pip install pandas numpy matplotlib scipy openpyxl`
- Restart runtime after installing packages: Runtime → Restart runtime

**"No optimization results" warnings:**
- Verify CSV has required columns: `Test Date`, `Pwf`, `Total Rate`
- Check data values are positive (no zeros or negative values)
- Try different `group_size` values (3, 5, or 10) if default fails

**Enhanced features not working:**
- Ensure you're running the correct version (v1.2) with enhanced WellTestOptimizer class
- Run cells sequentially - cell 7 must execute before optimization cells
- Check for any Python syntax errors in the enhanced class definition
