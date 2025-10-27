# Movie Recommender System

A **hybrid movie recommendation system** combining collaborative filtering (SVD) and content-based filtering to provide personalized movie recommendations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a hybrid recommender that combines:
- **Collaborative Filtering:** SVD (Singular Value Decomposition) for rating prediction
- **Content-Based Filtering:** Movie genre features and user demographics
- **Hybrid Scoring:** Configurable weighted combination of both approaches

Built on the **MovieLens 100k dataset** with comprehensive evaluation metrics including precision, recall, NDCG, hit rate, coverage, and diversity.

## Key Features

✨ **Hybrid Recommendation Approach**
- Weighted blend of collaborative and content-based signals
- Optional learned similarity-to-rating mapping
- Hyperparameter tuning with GridSearchCV

📊 **Comprehensive Evaluation**
- Rating prediction: RMSE, MAE
- Ranking quality: Precision@K, Recall@K, NDCG@K, Hit Rate@K
- Diversity: Item Coverage, Intra-List Diversity

🛠️ **Robust Implementation**
- Model persistence (save/load trained models)
- Input validation and error handling
- Cross-platform support (macOS, Linux, Windows)
- Automated testing

## Performance

Sample results on MovieLens 100k:

| Model | Precision@5 | NDCG@5 | Hit Rate@5 | RMSE |
|-------|------------|---------|------------|------|
| SVD | 0.1390 | 0.1437 | 0.4250 | 0.9352 |
| **Hybrid** | **0.1620** ↑16.5% | **0.1643** ↑14.3% | **0.4750** ↑11.8% | — |

*The hybrid approach improves ranking metrics while maintaining catalog coverage.*

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
- `--svd-weight`     Weight for SVD predictions (default 0.7)
- `--content-weight` Weight for content similarity (default 1 - svd_weight)
- `--normalize-sim`  Normalize cosine similarity to rating range [1,5]
- `--learn-sim`      Learn linear mapping from similarity->rating using training data
- `--users`          List of user IDs to recommend for (default `2 10 30`)
- `--topn`           Number of recommendations per user (default 5)
- `--no-eval`        Disable evaluation
 - `--tune-svd`      Run a quick GridSearchCV over SVD hyperparameters (3-fold RMSE)
 - `--save-model`    Save the trained SVD model to `artifacts/svd_model.dump` (by default)
 - `--load-model`    Load a previously saved SVD model (skips training if found)
 - `--model-dir`     Directory for model artifacts (default `artifacts`)
 - `--model-file`    Filename for the SVD model artifact (default `svd_model.dump`)

### Example Commands

**Basic usage with default settings:**
```bash
python "Movie recommender.py"
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

**Tune hyperparameters and save model:**
```bash
python "Movie recommender.py" --tune-svd --save-model
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
5. **Hybrid Scoring:** Combine SVD predictions and content scores with configurable weights
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
