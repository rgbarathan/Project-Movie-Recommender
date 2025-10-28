# Movie Recommender System

A **hybrid movie recommendation system** combining collaborative filtering (SVD), content-based filtering, and demographic-aware category profiles to provide personalized, explainable movie recommendations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a hybrid recommender that combines:
- **Collaborative Filtering:** SVD (Singular Value Decomposition) for rating prediction
- **Content-Based Filtering:** Movie genre features
- **Demographic-Aware Category Profiles:** Age-group, gender, and occupation group profiles influence recommendations
- **Hybrid Scoring:** Configurable weighted combination of SVD, content similarity, and category similarity, with optional top-genre boost

Built on the **MovieLens 100k dataset** with comprehensive evaluation metrics including precision, recall, NDCG, hit rate, coverage, and diversity.

## Key Features

✨ **Hybrid Recommendation Approach**
- Weighted blend of collaborative and content-based signals
- Demographic-aware category blending (age, gender, occupation)
- Optional learned similarity-to-rating mapping
- SVD hyperparameter tuning with GridSearchCV (3-fold)

📊 **Comprehensive Evaluation**
- Rating prediction: RMSE, MAE
- Ranking quality: Precision@K, Recall@K, NDCG@K, Hit Rate@K
- Diversity: Item Coverage, Intra-List Diversity

🧾 **Explainable, Clear Output**
- Tabular recommendations by default with per-item component contributions (SVD, Content, Category, Boost)
- Concise demographic badges in headers, combined top genres per user
- Matched Genres badges per item (on by default)
- Tabular evaluation comparing SVD vs Hybrid with Delta column

🛠️ **Robust Implementation**
- Model persistence (save/load trained models)
- Input validation and error handling
- Cross-platform support (macOS, Linux, Windows)
- Automated testing

## Performance

Sample results on MovieLens 100k (SVD-only base model):

### Rating Prediction

- SVD RMSE: 0.9434
- SVD MAE: 0.7415

### Hybrid Approach Performance (SVD vs Hybrid)

| Metric | SVD | Hybrid | Delta |
|--------|-----|--------|-------|
| Precision@3 | 0.1417 | 0.1533 | +0.0117 |
| Recall@3    | 0.0180 | 0.0202 | +0.0022 |
| NDCG@3      | 0.1408 | 0.1547 | +0.0139 |
| HitRate@3   | 0.3200 | 0.3500 | +0.0300 |
| Coverage    | 0.0916 | 0.0850 | -0.0065 |
| ILD         | 0.7177 | 0.5656 | -0.1522 |

Notes: Hybrid boosts ranking quality with a small trade-off in coverage/diversity depending on weights.

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/)

### Setup (macOS/Linux)

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

### Setup (Windows)

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

**Important:** Download the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) and extract the `ml-100k/` folder into the project root directory before running.

## Usage

### Command-Line Arguments

Core
- `--svd-weight`       Weight for SVD predictions (default: 0.7)
- `--content-weight`   Weight for content similarity (default: 1 - svd_weight - category_weight)
- `--category-weight`  Weight for demographic category similarity (default: 0.2)
- `--category-dims`    Comma-separated list of category dimensions: `age,gender,occupation` (default: all)
- `--normalize-sim`    Normalize cosine similarity to rating scale [1,5]
- `--learn-sim`        Learn linear mapping from similarity→rating using training data
- `--users`            User IDs to recommend for (default: 3 random users if omitted)
- `--topn`             Number of recommendations per user (default: 3)
- `--only-top3`        Force Top-N to 3 regardless of `--topn`
- `--no-eval`          Disable evaluation (enabled by default)
- `--relevance`        Rating threshold for relevant items in evaluation (default: 4.0)

Category Top-Genres and Boost (optional)
- `--cat-genre-boost`  Additive boost for items matching selected top genres (default: 0.0 disabled)
- `--cat-topk-genres`  K for selecting top genres (default: 3)
- `--cat-genre-mode`   How to select top genres: `combined` (avg profile) or `union` (per-dimension union)

Display and Explainability
- `--recs-table` / `--no-recs-table`  Enable/disable tabular recommendations (default: enabled)
- `--recs-show-components`            Include SVD/Content/Category/Boost columns
- `--recs-show-confidence`            Show confidence hint (e.g., `[###--] 3.5`)
- `--badges-demographics`             Show compact demographic badges (default: enabled)
- `--show-per-dim-top-genres`         Also print per-dimension top genres (off by default)
- `--show-matched-genres` / `--no-show-matched-genres`  Show/hide Matched Genres badges (default: show)
- `--recs-max-genres`                 Max number of matched genres to display per item (default: 2)

Model Tuning and Persistence
- `--tune-svd`     Run GridSearchCV over SVD hyperparameters (fixed 3-fold RMSE)
- `--save-model`   Save the trained SVD model to `artifacts/svd_model.dump` (by default)
- `--load-model`   Load a previously saved SVD model (skips training if found)
- `--model-dir`    Directory for model artifacts (default: `artifacts`)
- `--model-file`   Filename for the SVD model artifact (default: `svd_model.dump`)

### Example Commands

**Basic usage with default settings (tabular view, concise demographics, matched genres):**
```bash
python "Movie recommender.py"
```

**Tune SVD hyperparameters (3-fold CV) and save model:**
```bash
python "Movie recommender.py" --tune-svd --save-model
```

**Generate recommendations for specific users:**
```bash
python "Movie recommender.py" --users 1 50 100 --topn 10
```

**Adjust hybrid weights (more collaborative):**
```bash
python "Movie recommender.py" --svd-weight 0.8 --content-weight 0.2
```

**Run without evaluation (faster):**
```bash
python "Movie recommender.py" --no-eval
```

**Show full explainable table (components and confidence):**
```bash
python "Movie recommender.py" --recs-show-components --recs-show-confidence
```

**Enable top-genre boost and union of per-dimension top-K:**
```bash
python "Movie recommender.py" --cat-genre-boost 0.1 --cat-genre-mode union --cat-topk-genres 3
```

**Load saved model and generate recommendations:**
```bash
python "Movie recommender.py" --load-model --users 1 50 100
```

## Architecture

The recommendation pipeline consists of:

1. **Data Loading:** Import ratings, movie metadata (genres), and user demographics
2. **Feature Engineering:** Compute user profile vectors by averaging rated movie genres
3. **Model Training:** Train SVD on the user-item rating matrix (collaborative filtering)
4. **Content Similarity:** Calculate cosine similarity between user profiles and movie features
5. **Category Profiles:** Compute demographic group profiles (age-group, gender, occupation) and similarity to items
6. **Hybrid Scoring:** Combine SVD, content, and category signals with optional top-genre boost
6. **Evaluation:** Assess performance on test set using multiple metrics

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

The test suite includes smoke tests to verify basic functionality.

## Project Structure

```
Project-Movie-Recommender/
├── Movie recommender.py    # Main recommendation system
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
├── .gitignore           # Git ignore rules
├── tests/
│   └── test_smoke.py    # Automated tests
├── artifacts/           # Saved models (created at runtime)
│   └── svd_model.dump  # Trained SVD model (if --save-model used)
└── ml-100k/            # MovieLens dataset (not in repo, download separately)
    ├── u.data         # Ratings data
    ├── u.item         # Movie metadata
    └── u.user         # User demographics
```

## Notes
- Dependencies are pinned in `requirements.txt` for reproducibility and binary compatibility (e.g., Surprise with NumPy).
- Dataset files from MovieLens (`u.data`, `u.item`, `u.user`) must be in the `ml-100k/` folder adjacent to the script.
- The `ml-100k/` dataset folder is excluded from the repository to keep it lightweight. Download separately from [GroupLens](https://grouplens.org/datasets/movielens/100k/).
- Windows only: If package installation fails for compiled extensions, ensure you have Microsoft C++ Build Tools installed.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:
- Bug fixes
- New features (e.g., additional algorithms, evaluation metrics)
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Raja Gopal Barathan**

## Acknowledgments

- [MovieLens dataset](https://grouplens.org/datasets/movielens/) by GroupLens Research
- [Surprise library](http://surpriselib.com/) for collaborative filtering algorithms
- Built as part of a recommender systems course project

---

For questions or issues, please open a GitHub issue or contact via the repository.
