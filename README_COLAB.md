# Running WellTestAnalysis_Colab.ipynb in Google Colab

This short README explains three easy ways to open `WellTestAnalysis_Colab.ipynb` in Google Colab so students can run the notebook in class.

## Option A — Upload the notebook directly (fast, single-session)
1. Open Google Colab: https://colab.research.google.com
2. Click "File → Upload notebook" and choose `WellTestAnalysis_Colab.ipynb` from your machine.
3. Run the cells. If you need packages, run the install cell near the top (Cell 3) or uncomment the `pip install` line.

Use this when you have the notebook file locally and you only need a single Colab session.

## Option B — Put the notebook in Google Drive (recommended for classes)
1. Copy `WellTestAnalysis_Colab.ipynb` into your Google Drive (e.g., `My Drive/ColabNotebooks/`).
2. In Colab, run the following cells (or paste into a code cell) to mount Drive and copy the file into the Colab workspace:

```python
from google.colab import drive
drive.mount('/content/drive')
# then copy from Drive into the session folder (update the path to match where you put it)
!cp "/content/drive/My Drive/ColabNotebooks/WellTestAnalysis_Colab.ipynb" ./
!ls -l WellTestAnalysis_Colab.ipynb
```

3. Open the notebook by double-clicking it in the left Files pane or: File → Open notebook → Upload → choose the file from the session filesystem.

Benefits: students keep a personal copy in Drive and can re-open it later.

## Option C — Fetch from GitHub (if you publish the repo)
If you push this repository to GitHub, students can open the notebook directly in Colab:

- Colab UI: File → Open notebook → GitHub tab → paste `owner/repo` or a notebook path.

- Or in a Colab code cell, fetch the raw file directly (replace `<raw-url>`):

```bash
!wget -O WellTestAnalysis_Colab.ipynb "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/path/WellTestAnalysis_Colab.ipynb"
```

After fetching, open it from the Files pane.

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
File: `README_COLAB.md` — created to help students load `WellTestAnalysis_Colab.ipynb` into Google Colab quickly.
