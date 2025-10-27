Movie Recommender

This script (`Movie recommender.py`) uses the MovieLens 100k dataset (folder `ml-100k/`) to build a hybrid recommender combining SVD (matrix-factorization) and content-based similarity (movie genres + simple user profiles).

## Quick setup (macOS/Linux - zsh/bash)

1) Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the recommender:

```bash
python "Movie recommender.py"
```

Tip: On some systems, you may need to use `python` instead of `python3`.

## Quick setup (Windows)

1) Create a virtual environment (PowerShell or Command Prompt):

```powershell
python -m venv .venv
```

2) Activate the virtual environment:

- PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If you see a policy error, allow script execution for the current user (then re-run Activate.ps1):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

- Command Prompt (cmd.exe)

```bat
.\.venv\Scripts\activate.bat
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

If `pip` is not found or you have multiple Python versions, try:

```powershell
py -m pip install -r requirements.txt
```

4) Run the recommender:

```powershell
python "Movie recommender.py"
```

## Flags
- `--svd-weight`     Weight for SVD predictions (default 0.7)
- `--content-weight` Weight for content similarity (default 1 - svd_weight)
- `--normalize-sim`  Normalize cosine similarity to rating range [1,5]
- `--learn-sim`      Learn linear mapping from similarity->rating using training data
- `--users`          List of user IDs to recommend for (default `2 10 30`)
- `--topn`           Number of recommendations per user (default 5)
- `--no-eval`        Disable evaluation

## Notes
- Dependencies are pinned in `requirements.txt` for reproducibility and binary compatibility (e.g., Surprise with NumPy).
- Dataset files from MovieLens (`u.data`, `u.item`, etc.) must be in the `ml-100k/` folder adjacent to the script.
- Windows only: If package installation fails for compiled extensions, ensure you have recent Microsoft C++ Build Tools installed, or use a supported Python version as pinned here.
